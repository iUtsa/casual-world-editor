"""General experiment engine: words -> world -> simulation -> measurements checked against physics.

Each test states the physical law it checks, so a failure says which part of reality broke.
Runs rules-only (CWE_DISABLE_LOCAL_AI=1 from conftest) so it is deterministic and offline.
"""
import math

import pytest

from backend.app import experiment


def rows(r):
    return {x["label"]: x["value"] for x in r["results"]}


def find(d, *words):
    for k, v in d.items():
        if all(w.lower() in k.lower() for w in words):
            return v
    raise KeyError(f"no row with {words}: {list(d)}")


# ---------- understanding ----------

@pytest.mark.parametrize("text, names, verb", [
    ("two baseballs clash into each other", ["baseball"], "collide"),
    ("a dragon crashes into a castle", ["dragon", "castle"], "collide"),
    ("two blue whales fight underwater", ["blue whale"], "fight"),
    ("kick a soccer ball at a chair really hard", ["soccer ball", "chair"], "throw"),
    ("a dog chases a cat", ["dog", "cat"], "chase"),
    ("two carts connected by a spring", ["cart"], "spring"),
])
def test_parse(text, names, verb):
    e = experiment.parse_rules(text, {})
    got = [o.name for o in e.objects]
    for n in names:
        assert n in got, (text, got)
    assert verb in [a.do for a in e.actions], (text, [a.do for a in e.actions])


def test_connectors_are_not_objects():
    e = experiment.parse_rules("a ball on a spring", {})
    assert "spring" not in [o.name for o in e.objects]
    assert e.actions[0].do == "spring" and e.actions[0].a == experiment.ANCHOR


# ---------- physics checks ----------

def test_equal_balls_collision_momentum_and_restitution():
    d = rows(experiment.run("two baseballs collide"))
    e = find(d, "restitution")
    assert 0.4 < e < 0.7                                    # baseball-baseball ~0.55
    assert find(d, "momentum along the line of impact conserved") > 99                 # head-on: the total is ~0, so compare against the momentum in play
    assert find(d, "energy lost") == pytest.approx(100 * (1 - e * e), abs=6)   # equal masses head-on: 1 - e^2


def test_drop_speed_matches_free_fall():
    d = rows(experiment.run("drop an egg from 2 m"))
    v = find(d, "impact speed")
    assert v == pytest.approx(math.sqrt(2 * 9.81 * 2.0), rel=0.08)


def test_hanging_spring_period():
    d = rows(experiment.run("a ball on a spring"))
    assert find(d, "period (measured)") == pytest.approx(find(d, "period (theory"), rel=0.03)


def test_pendulum_period_and_speed():
    d = rows(experiment.run("a pendulum with a 2 m string pulled back 20 degrees"))
    assert find(d, "swing period (measured)") == pytest.approx(find(d, "swing period (theory"), rel=0.03)
    assert find(d, "bottom (measured)") == pytest.approx(find(d, "bottom (theory"), rel=0.08)


def test_two_body_spring_period():
    d = rows(experiment.run("two carts connected by a spring"))
    assert find(d, "period (measured)") == pytest.approx(find(d, "period (theory"), rel=0.05)


def test_rolling_balls_spring_includes_rotational_inertia():
    d = rows(experiment.run("connect two steel balls with a spring"))
    assert find(d, "period (measured)") == pytest.approx(find(d, "period (theory"), rel=0.05)


def test_pull_force_is_traction_limited_and_explained():
    r = experiment.run("a man pulls a truck with a rope")
    d = rows(r)
    T = find(d, "tension (average")
    assert T <= find(d, "most the man can pull") * 1.05      # can't out-pull your grip
    assert any("rolling resistance" in s for s in r["story"])


def test_whales_stay_afloat_and_meet():
    r = experiment.run("two blue whales fight underwater")
    sim = r["sim"]
    sid = r["objects"][0]["sid"]
    f0 = int(sim["segments"][-1]["t_start"] * sim["fps"])
    z0, z1 = sim["kinematics"][sid]["c"][f0][2], sim["kinematics"][sid]["c"][-1][2]
    assert abs(z1 - z0) < 0.5 * 3.5                           # neutrally buoyant (shoving moves them a little)
    assert find(rows(r), "peak collision force") > 1e5


def test_structure_stays_put():
    r = experiment.run("a dragon crashes into a castle")
    sid = next(o["sid"] for o in r["objects"] if o["name"] == "castle")
    c = [x for x in r["sim"]["kinematics"][sid]["c"] if x]
    assert max(abs(a - b) for a, b in zip(c[0], c[-1])) < 0.01          # the castle never moves
    assert any("anchored" in s for s in r["story"])


def test_kick_reaches_target_from_ground():
    d = rows(experiment.run("kick a soccer ball at a chair really hard"))
    assert find(d, "soccer ball speed just before") > 8
    assert find(d, "chair: distance") > 0.05


def test_energy_is_not_created_by_contacts():
    r = experiment.run("kick a soccer ball at a chair really hard")
    assert experiment._energy_gain(r["sim"], r["_compiled"]) < 0.05
