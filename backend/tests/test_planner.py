"""Rule-based planner + geometric grounding + simulation, end to end (no Claude)."""
import pytest

from backend.app import planner
from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Intervention
from backend.tests.test_physics import gt_scene


@pytest.fixture(scope="module")
def world():
    scene, q, _ = stabilize(gt_scene("bridge"))
    return scene, q


def run(world, text, selected=None):
    scene, q = world
    ents = planner.snapshot(scene, None, [])
    plan = planner.parse_rules(text, ents, selected)
    ivs, human, notes = planner.resolve(plan, scene, ents, 0)
    sim = simulate(scene, q, [Intervention(**i) for i in ivs])
    return ivs, human, notes, sim


def test_drive_car_into_bottle(world):
    ivs, human, notes, sim = run(world, "run this experiment with a car, drive it into the bottle fast")
    assert [i["type"] for i in ivs] == ["spawn", "drive"]
    assert ivs[0]["params"]["asset"] == "toy_car" and ivs[1]["params"]["speed"] == 1.2
    assert sim["segments"][1]["summary"]["bottle"]["moved"], human


def test_drop_ball_on_book(world):
    ivs, human, notes, sim = run(world, "drop a baseball on the book")
    assert ivs[0]["params"]["pos"][2] > 0.3
    seg = sim["segments"][0]
    assert seg["summary"]["s1"]["drop_m"] > 0.15
    hits = {e["other"] for e in seg["events"] if e["type"] == "impact" and e["subject"] == "s1"}
    assert hits & {"book", "tea_box"}, hits                  # lands on the book (or the tea box on it)
    assert all(c["cause"] != "unknown" for c in seg["causal_edges"]), seg["causal_edges"]


def test_domino_row_topples(world):
    ivs, human, notes, sim = run(world, "line up 8 dominoes and knock them over")
    assert sum(i["type"] == "spawn" for i in ivs) == 8
    assert len(sim["segments"]) == 2                       # one spawn segment + the tip
    toppled = [k for k, v in sim["segments"][1]["summary"].items() if k.startswith("s") and v["toppled"]]
    assert len(toppled) >= 6, (human, toppled)


def test_unknown_object_is_reported(world):
    ivs, human, notes, sim = run(world, "make a dragon breathe fire")
    assert ivs == [] and notes
