"""Physics + causal attribution on ground-truth synthetic scenes (no perception)."""
import json

from backend.app import synth
from backend.app.explain import explain
from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Camera, Intervention, Scene, SceneObject, Table


def gt_scene(name: str) -> Scene:
    sc = synth.SCENES[name]
    objs = []
    for i, o in enumerate(sc.objects):
        size = [o.size[0], o.size[0], o.size[1]] if o.shape == "cylinder" else list(o.size)
        objs.append(SceneObject(id=o.name, idx=i + 1, label=o.name.replace("_", " "), shape=o.shape, size=size,
                                pos=list(o.pos), yaw=o.yaw, mass=o.mass, friction=0.5))
    import numpy as np
    pos, tgt = np.array(sc.cam_pos), np.array(sc.cam_target)
    fwd = (tgt - pos) / np.linalg.norm(tgt - pos)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    fy = (sc.height / 2) / np.tan(np.radians(sc.fovy) / 2)
    cam = Camera(width=sc.width, height=sc.height, fx=fy, fy=fy, cx=sc.width / 2, cy=sc.height / 2,
                 hfov_deg=float(np.degrees(2 * np.arctan(sc.width / 2 / fy))),
                 R_wc=np.stack([right, down, fwd], axis=1).tolist(), t_wc=pos.tolist())
    table = Table(x_min=-0.7, x_max=0.7, y_min=-0.4, y_max=0.6, height_above_floor=0.75)
    return Scene(scene_id="gt", source="synthetic-gt", camera=cam, table=table, objects=objs)


def run(name, interventions):
    scene, q, warnings = stabilize(gt_scene(name))
    sim = simulate(scene, q, interventions)
    return scene, sim, warnings


def test_bridge_remove_mug():
    scene, sim, warnings = run("bridge", [Intervention(type="remove", target="mug_left")])
    assert not warnings, warnings
    seg = sim["segments"][0]
    book = scene.objects[[o.id for o in scene.objects].index("book")]
    assert set(book.supported_by) == {"mug_left", "mug_right"}
    assert seg["summary"]["book"]["moved"]
    assert seg["summary"]["tea_box"]["moved"]
    assert not seg["summary"]["bottle"]["moved"] and not seg["summary"]["block"]["moved"]
    effects = {c["effect"]: c for c in seg["causal_edges"]}
    assert effects["book"]["kind"] == "support_removed"
    assert "mug_left" not in sim["moved_ids"]


def test_tower_remove_middle():
    scene, sim, _ = run("tower", [Intervention(type="remove", target="middle_box")])
    s = sim["segments"][0]["summary"]
    assert s["top_box"]["drop_m"] > 0.05      # falls onto the base
    assert not s["base_box"]["moved"]


def test_push_and_chain():
    scene, sim, _ = run("tower", [Intervention(type="push", target="base_box", params={"direction": [1, 0, 0], "speed": 1.2}),
                                  Intervention(type="set_gravity", params={"g": 1.62})])
    s0 = sim["segments"][0]
    assert s0["summary"]["base_box"]["moved"]
    assert {c["effect"] for c in s0["causal_edges"]} >= {"base_box"}
    assert len(sim["segments"]) == 2


if __name__ == "__main__":
    import time
    t = time.time()
    scene, sim, warnings = run("bridge", [Intervention(type="remove", target="mug_left")])
    print(f"sim {time.time() - t:.2f}s duration={sim['duration']}s frames={len(sim['frames'])} moved={sim['moved_ids']} warnings={warnings}")
    seg = sim["segments"][0]
    print("support_before:", json.dumps(seg["support_before"]))
    for e in seg["events"]:
        print("  ", e)
    print("causal:", seg["causal_edges"])
    for k, v in seg["summary"].items():
        print("  ", k, v)
    ex = explain(scene, sim, 0, use_claude=False)
    ex.pop("evidence_packet")
    print(json.dumps(ex, indent=1))
