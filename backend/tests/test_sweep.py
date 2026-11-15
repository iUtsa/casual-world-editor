"""Sweep scan: carving geometry and scanned-mesh physics (ground truth from synthetic renders)."""
import numpy as np
import pytest

from backend.app.physics.sim import World, stabilize
from backend.app.schemas import Camera, Scene, SceneObject, Table


def _scene(objs):
    cam = Camera(width=640, height=480, fx=500, fy=500, cx=320, cy=240, hfov_deg=65,
                 R_wc=[[1, 0, 0], [0, 0, 1], [0, -1, 0]], t_wc=[0, -0.6, 0.4])
    return Scene(scene_id="t", source="test", camera=cam, table=Table(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5,
                 height_above_floor=0.75), objects=objs)


def _cube(a):
    h = a / 2
    return [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)]


def test_mesh_object_rests_with_its_weight():
    o = SceneObject(id="o1", idx=1, label="block", shape="mesh", size=[0.06, 0.06, 0.06], pos=[0, 0, 0.032],
                    mass=0.2, hulls=[_cube(0.06)], volume_m3=0.06 ** 3)
    sc, q, warn = stabilize(_scene([o]))
    w = World(sc)
    w.set_qpos(q)
    for _ in range(200):
        w.step()
    z = w.pose("o1")[2]
    assert abs(z - 0.03) < 0.002                      # resting on the table, centre at half its height
    fz = sum(abs(c["fvec"][2]) for c in w.contacts() if "o1" in (c["a"], c["b"]))
    assert fz == pytest.approx(0.2 * 9.81, rel=0.05)  # it presses down with its weight


def test_two_piece_mesh_centre_of_mass_follows_shape():
    # an L: a tall post (left) and a low slab (right); mass split by piece volume puts the COM over the slab side
    post = [[x, y, z] for x in (-0.05, -0.03) for y in (-0.01, 0.01) for z in (-0.04, 0.04)]
    slab = [[x, y, z] for x in (-0.03, 0.05) for y in (-0.01, 0.01) for z in (-0.04, -0.03)]
    o = SceneObject(id="L", idx=1, label="bracket", shape="mesh", size=[0.10, 0.02, 0.08], pos=[0, 0, 0.041],
                    mass=0.1, hulls=[post, slab], volume_m3=0.02 * 0.02 * 0.08 + 0.08 * 0.02 * 0.01)
    w = World(_scene([o]))
    assert len(w.geoms_of["L"]) == 2                   # both pieces belong to the one object
    import mujoco
    mujoco.mj_forward(w.m, w.d)
    com_x = float(w.d.subtree_com[w.body["L"]][0])
    assert -0.05 < com_x < 0.05


def _flat_hand(cx, cy, cz):
    """21 joints of a flat right hand, palm down, fingers pointing +y (metres, world)."""
    J = np.zeros((21, 3))
    J[0] = [cx, cy - 0.06, cz]
    fingers = {1: -0.045, 5: -0.025, 9: -0.005, 13: 0.015, 17: 0.035}   # base joint -> x offset
    for base, dx in fingers.items():
        for k in range(4):
            J[base + k] = [cx + dx * (1.0 if base != 1 else 1.0), cy - 0.02 + 0.025 * k + (0 if base != 1 else -0.02), cz]
    return J


def test_hand_sweep_pushes_block_physically():
    from backend.app.physics.hand import HandSession
    o = SceneObject(id="o1", idx=1, label="block", shape="box", size=[0.06, 0.06, 0.06], pos=[0, 0, 0.03], mass=0.15)
    sc, q, _ = stabilize(_scene([o]))
    J0 = _flat_hand(-0.2, 0.0, 0.02)                    # hand resting low on the table, left of the block
    hs = HandSession(sc, J0, base_qpos=q)
    t, v = 0.0, 0.4                                      # sweep right at 0.4 m/s for 1.0 s
    out = None
    max_speed = 0.0
    while t < 1.6:
        x = -0.2 + v * min(t, 1.0)
        out = hs.update(_flat_hand(x, 0.0, 0.02).tolist(), wall=t)
        if len(hs.record) >= 2:
            a, b = hs.record[-2]["poses"]["o1"], hs.record[-1]["poses"]["o1"]
            max_speed = max(max_speed, np.linalg.norm(np.array(b[:3]) - np.array(a[:3])) * 60)
        t += 1 / 30
    p = np.array(out["poses"]["o1"][:3])
    assert p[0] > 0.12                                   # pushed to the right, past where the hand stopped
    assert abs(p[1]) < 0.05                              # not flung sideways
    assert max_speed < 1.0                               # no explosive contact (hand moved at 0.4 m/s)
    zs = [min(j[2] for j in r["joints"]) for r in hs.record if r["joints"]]
    assert min(zs) > -0.005                              # targets stay above the table


def test_hand_appearing_inside_nothing_is_calm():
    from backend.app.physics.hand import HandSession
    o = SceneObject(id="o1", idx=1, label="block", shape="box", size=[0.06, 0.06, 0.06], pos=[0, 0, 0.03], mass=0.15)
    sc, q, _ = stabilize(_scene([o]))
    hs = HandSession(sc, _flat_hand(0.25, 0.2, 0.1), base_qpos=q)
    for k in range(30):                                  # hand hovering far away, then leaving the view
        hs.update(_flat_hand(0.25, 0.2, 0.1).tolist() if k < 20 else None, wall=k / 30)
    p = np.array(hs.poses()["o1"][:3])
    assert np.linalg.norm(p[:2]) < 0.002 and abs(p[2] - 0.03) < 0.002


def test_live_hand_session_replays_identically():
    """A live session, replayed as a 'hand' intervention by the normal simulator, ends in the same place."""
    from backend.app.physics.hand import HandSession
    from backend.app.physics.sim import simulate
    from backend.app.schemas import Intervention
    o = SceneObject(id="o1", idx=1, label="block", shape="box", size=[0.06, 0.06, 0.06], pos=[0, 0, 0.03], mass=0.15)
    sc, q, _ = stabilize(_scene([o]))
    hs = HandSession(sc, _flat_hand(-0.2, 0.0, 0.02), base_qpos=q)
    t = 0.0
    while t < 1.2:
        hs.update(_flat_hand(-0.2 + 0.4 * min(t, 0.9), 0.0, 0.02).tolist(), wall=t)
        t += 1 / 30
    hs.update(None, wall=t)
    for k in range(60):                                   # let it come to rest live
        hs.update(None, wall=t + (k + 1) / 30)
    live = np.array(hs.poses()["o1"][:3])
    sim = simulate(sc, q, [Intervention(type="hand", target=None, params=hs.intervention_params())])
    i = sim["object_ids"].index("o1")
    replay = np.array(sim["frames"][-1][i][:3])
    assert live[0] > 0.1                                  # it really was pushed
    assert np.linalg.norm(live - replay) < 0.01           # replay lands within 1 cm of the live result


def test_ws_hand_live_session_commits_replayable_intervention():
    """End to end over the WebSocket: a hand sweeps through a scanned object live, the session ends, and the
    official simulation (a replay of the recorded 'hand' intervention) puts it where the live session did."""
    import cv2
    from fastapi.testclient import TestClient
    from backend.app import synth
    from backend.app.main import SESSION, app
    from backend.app.perception import pipeline
    info = synth.render("bridge")
    rgb = cv2.cvtColor(cv2.imread(info["rgb"]), cv2.COLOR_BGR2RGB)
    depth = np.load(str(info["rgb"]).replace("rgb.png", "depth_gt.npy"))
    st = pipeline.scan(rgb, info["hfov_deg"], "synthetic:bridge", use_claude=False, depth_override=depth, label=False)
    with SESSION.lock:
        SESSION.perception = st
        SESSION.interventions.clear(); SESSION.redo_stack.clear(); SESSION.invalidate(); SESSION.sim = None
    o = min(st.scene.objects, key=lambda x: x.pos[0])           # the left-most object (the lone block)
    x0, y0 = o.pos[0] - max(o.size[:2]) / 2 - 0.08, o.pos[1]
    c = TestClient(app)
    live_end = None
    with c.websocket_connect("/ws/hand") as ws:
        t = 0.0
        while t < 1.3:
            x = x0 + 0.35 * min(t, 0.9)
            ws.send_json({"joints": _flat_hand(x, y0, 0.015).tolist(), "t": t})
            out = ws.receive_json()
            t += 1 / 30
        for _ in range(45):
            ws.send_json({"joints": None, "t": t})
            out = ws.receive_json()
            t += 1 / 30
        live_end = np.array(out["poses"][o.id][:3])
        assert o.id in out["moved"]
        ws.send_json({"end": True})
        done = ws.receive_json()["done"]
    assert SESSION.interventions[-1].type == "hand"
    i = done["object_ids"].index(o.id)
    replay_end = np.array(done["frames"][-1][i][:3])
    assert live_end[0] - o.pos[0] > 0.03                      # pushed to the right
    assert np.linalg.norm(replay_end - live_end) < 0.02        # the official replay agrees with what was seen live
