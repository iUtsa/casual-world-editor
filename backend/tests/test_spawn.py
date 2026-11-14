"""Summoned objects: spawn, a real wheeled car driving into objects, launches."""
import numpy as np

from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Intervention
from backend.tests.test_physics import gt_scene


def run(ivs, name="bridge"):
    scene, q, _ = stabilize(gt_scene(name))
    return scene, simulate(scene, q, ivs)


def test_car_drives_into_bottle():
    # bottle at (0.23, 0.16); start the car 35 cm to its left, facing it
    ivs = [Intervention(type="spawn", target="s1", params={"asset": "toy_car", "pos": [-0.12, 0.16, 0.0005], "yaw": 0.0}),
           Intervention(type="drive", target="s1", params={"speed": 0.8, "goal": [0.23, 0.16], "duration": 1.2})]
    scene, sim = run(ivs)
    spawn_seg, drive_seg = sim["segments"]
    assert not spawn_seg["summary"]["s1"]["toppled"]
    assert spawn_seg["summary"]["s1"]["displacement_m"] < 0.01          # lands and stays
    d = drive_seg["summary"]
    assert d["s1"]["displacement_m"] > 0.2                               # actually drove
    assert d["bottle"]["moved"]                                          # and hit the bottle
    edges = {(c["cause"], c["effect"]): c["kind"] for c in drive_seg["causal_edges"]}
    assert edges.get(("intervention", "s1")) == "driven"
    assert any(c["effect"] == "bottle" and c["cause"] == "s1" for c in drive_seg["causal_edges"])
    w = np.array(sim["wheels"]["s1"])
    assert abs(w[-1, 4]) > 10                                            # rear wheel turned > 10 rad


def test_ball_launch_and_spawned_not_in_moved_ids():
    ivs = [Intervention(type="spawn", target="s1", params={"asset": "baseball_01", "pos": [-0.35, -0.25, 0.0005]}),
           Intervention(type="launch", target="s1", params={"velocity": [0, 0.6, 0]})]   # clear path
    scene, sim = run(ivs)
    assert "s1" not in sim["moved_ids"]
    assert sim["segments"][1]["summary"]["s1"]["displacement_m"] > 0.15   # rolls, not slides to a stop
