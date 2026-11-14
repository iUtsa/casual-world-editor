"""Live AR hands: the exact interventions the hand layer sends (HandLayer.tsx) simulate correctly.

  throw  = place(lifted, hold, group) + launch(velocity, same group)   -> one segment, ballistic until it lands
  drop   = place(lifted, hold=False)                                    -> falls and settles where the hand was
  push   = push(direction, speed, point)                                -> reaches the fingertip's speed
"""
import numpy as np

from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Intervention

from backend.tests.test_physics import gt_scene


def _setup():
    scene, q, _ = stabilize(gt_scene("bridge"))
    return scene, q


def _track(sim, oid):
    """Positions of `oid` from the last segment's start, with the true sample times (a frame is taken on the
    first physics step at or after each 1/60 s mark)."""
    from backend.app import config
    i = sim["object_ids"].index(oid)
    seg = sim["segments"][-1]
    f0 = int(round(seg["t_start"] * sim["fps"]))
    xs = np.array([f[i][:3] for f in sim["frames"][f0:]])
    dt = config.SIM_TIMESTEP
    ts = np.array([np.ceil(k / sim["fps"] / dt - 1e-6) * dt for k in range(len(xs))])
    return xs, ts


def test_throw_is_one_segment_and_ballistic():
    scene, q = _setup()
    lift = [-0.24, -0.02, 0.09]
    v = np.array([-1.2, -0.3, 0.3])                    # away from the book/mugs: free flight, then the table
    sim = simulate(scene, q, [Intervention(type="place", target="block", params={"pos": lift, "hold": True, "clear": True, "group": "hand-1"}),
                              Intervention(type="launch", target="block", params={"velocity": v.tolist(), "group": "hand-1"})])
    assert len(sim["segments"]) == 1, "place + launch must happen at the same instant"
    xs, ts = _track(sim, "block")
    g = sim["gravity"]
    from backend.app import config
    dt = config.SIM_TIMESTEP
    worst = 0.0
    for k in range(1, len(xs)):
        t = ts[k]
        # MuJoCo integrates v first, then x, with a 2 ms step: z = z0 + vz t - g t (t + dt) / 2 exactly
        # (vs the continuous parabola that is a dt*g*t/2 offset: < 1 mm over this flight)
        want = np.array(lift) + v * t + np.array([0, 0, -0.5 * g * t * (t + dt)])
        if want[2] < 0.031 + 0.005:                    # about to touch the table: flight is over
            break
        worst = max(worst, float(np.linalg.norm(xs[k] - want)))
    assert k > 3, "the throw should fly for a few frames"
    assert worst < 2e-4, f"trajectory off ballistic by {worst * 1000:.2f} mm"
    assert xs[-1][0] < lift[0] - 0.1                   # it went where it was thrown


def test_drop_settles_where_the_hand_let_go():
    scene, q = _setup()
    target = [-0.30, -0.10, 0.08]                      # 5 cm above clear table
    sim = simulate(scene, q, [Intervention(type="place", target="block", params={"pos": target, "hold": False, "clear": True})])
    xs, _ = _track(sim, "block")
    end = xs[-1]
    assert abs(end[2] - 0.03) < 3e-3, end               # resting on the table (half-height 3 cm)
    assert np.hypot(end[0] - target[0], end[1] - target[1]) < 0.01, end


def test_drop_into_another_object_lands_on_top_of_it():
    """A hand can't put the block inside the book: it is lifted clear and set down on the book."""
    scene, q = _setup()
    book = next(o for o in scene.objects if o.id == "book")
    target = [-0.08, 0.0, 0.11]                         # inside the book's overhang
    sim = simulate(scene, q, [Intervention(type="place", target="block", params={"pos": target, "hold": False, "clear": True})])
    xs, _ = _track(sim, "block")
    book_top = book.pos[2] + book.size[2] / 2
    assert xs[1][2] - 0.03 >= book_top - 2e-3, (xs[1], book_top)   # released clear of the book, not inside it
    bi = sim["object_ids"].index("book")
    b_end = np.array(sim["frames"][-1][bi][:3])
    assert np.linalg.norm(b_end - np.array(book.pos)) < 0.02, "the book must not be blasted out of the way"


def test_push_reaches_the_fingertip_speed():
    scene, q = _setup()
    blk = next(o for o in scene.objects if o.id == "block")
    sim = simulate(scene, q, [Intervention(type="push", target="block",
                                           params={"direction": [0.0, -1.0, 0.0], "speed": 0.8, "point": list(blk.pos)})])
    xs, ts = _track(sim, "block")
    sp = np.linalg.norm(np.diff(xs[:, :2], axis=0), axis=1) / np.diff(ts)
    assert 0.65 < sp.max() < 0.95, sp.max()
    assert xs[-1][1] < blk.pos[1] - 0.02                 # moved the way the finger swept


def test_api_accepts_hand_interventions():
    from backend.app.main import ApplyPlan
    plan = ApplyPlan(interventions=[{"type": "place", "target": "block", "params": {"pos": [0, 0, 0.1], "hold": True, "group": "g"}},
                                    {"type": "launch", "target": "block", "params": {"velocity": [1, 0, 0.2], "group": "g"}},
                                    {"type": "push", "target": "block", "params": {"direction": [1, 0, 0], "speed": 0.5, "point": [0, 0, 0.03]}}])
    assert [i.type for i in plan.interventions] == ["place", "launch", "push"]


def test_checkpoint_restart_is_exact():
    """Live AR continues from the current end state: rebuilding the world from (final qpos + persistent interventions)
    must leave every body exactly where the full-history run ended - nothing jumps, nothing pops upright."""
    scene, q = _setup()
    hist = [Intervention(type="spawn", target="s1", params={"asset": "baseball_01", "pos": [-0.3, -0.1, 0.001], "yaw": 0.0, "group": "a"}),
            Intervention(type="push", target="block", params={"direction": [0.0, -1.0, 0.0], "speed": 0.8, "point": [-0.24, -0.02, 0.03]}),
            Intervention(type="remove", target="mug_left", params={}),
            Intervention(type="push", target="bottle", params={"direction": [1.0, 0.0, 0.0], "speed": 1.2, "point": [0.23, 0.16, 0.18]})]
    full = simulate(scene, q, hist)
    restore = [Intervention(type="spawn", target="s1", params={**hist[0].params, "keep": True, "group": "restore"}),
               Intervention(type="remove", target="mug_left", params={"group": "restore"})]
    again = simulate(scene, np.array(full["final_qpos"]), restore)
    ids = full["object_ids"]
    worst = 0.0
    for k, oid in enumerate(ids):
        if oid == "mug_left":
            continue
        a, b = np.array(full["frames"][-1][k][:3]), np.array(again["frames"][-1][again["object_ids"].index(oid)][:3])
        worst = max(worst, float(np.linalg.norm(a - b)))
    assert worst < 2e-3, f"restored world drifted {worst * 1000:.2f} mm"
    assert len(again["segments"]) == 1 and again["duration"] < full["duration"] / 2
