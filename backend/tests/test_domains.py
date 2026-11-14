"""Motion verbs, circuits, gravity and the domain router — each test names the law it checks."""
import math

import pytest

from backend.app import circuits, experiment, lab, orbits


def rows(r):
    return {x["label"]: x["value"] for x in r["results"]}


def find(d, *words):
    for k, v in d.items():
        if all(w.lower() in k.lower() for w in words):
            return v
    raise KeyError(f"{words} not in {list(d)}")


# ---------------- routing: every sentence goes to an engine that can really do it
@pytest.mark.parametrize("text, dom", [
    ("circuit", "circuit"), ("a 9 V battery with two bulbs in parallel", "circuit"), ("throw a battery at a wall", "mechanics"),
    ("jupiter collides with moon", "gravity"), ("the moon orbits earth", "gravity"), ("drop a hammer on the moon", "mechanics"),
    ("an asteroid hits a car", "mechanics"), ("methane + oxygen -> carbon dioxide + water", "chem"),
    ("light refracts through a prism", "unsupported:light and optics (refraction, lenses, mirrors)"), ("a human runs on earth", "mechanics"),
])
def test_route(text, dom):
    assert lab.route(text) == dom


# ---------------- motion
def test_human_runs_at_running_speed_and_two_x_doubles_it():
    a = rows(experiment.run("a human runs on earth"))
    b = rows(experiment.run("a human runs on earth at 2x speed"))
    assert 3.5 < find(a, "human top speed") < 5.5           # jogging/running pace, not a drop
    assert find(b, "human top speed") == pytest.approx(2 * find(a, "human top speed"), rel=0.12)


def test_jump_height_scales_with_one_over_g():
    e = rows(experiment.run("a man jumps"))
    m = rows(experiment.run("a man jumps on the moon"))
    assert find(m, "man jump height") / find(e, "man jump height") == pytest.approx(9.81 / 1.62, rel=0.1)
    assert find(m, "jump height") == pytest.approx(find(m, "theory v²/2g"), rel=0.05)


def test_eagle_holds_altitude_human_cannot_fly():
    r = experiment.run("an eagle flies")
    sid = r["objects"][0]["sid"]
    f0 = int(r["sim"]["segments"][-1]["t_start"] * 60)
    zs = [c[2] for c in r["sim"]["kinematics"][sid]["c"][f0:] if c]
    assert max(zs) - min(zs) < 0.3
    assert any("can't fly" in n for n in experiment.run("a man flies")["notes"])


def test_swimmer_floats_and_is_drag_limited():
    d = rows(experiment.run("a man swims"))
    assert 0.6 < find(d, "man top speed") < 2.0


# ---------------- circuits (Ohm's and Kirchhoff's laws)
def test_series_and_parallel_bulbs():
    s = rows(circuits.run("a 9 V battery with two 27 ohm bulbs in series"))
    p = rows(circuits.run("a 9 V battery with two 27 ohm bulbs in parallel"))
    assert find(s, "current from the battery") == pytest.approx(9 / (54 + 1.5), rel=0.02)
    assert find(p, "current from the battery") == pytest.approx(9 / (13.5 + 1.5), rel=0.02)
    assert find(p, "bulb 1: brightness") > 3 * find(s, "bulb 1: brightness")


def test_rc_time_constant_and_open_switch():
    r = circuits.run("RC circuit with a 100 µF capacitor and a 1 kΩ resistor")
    assert find(rows(r), "time constant") == pytest.approx(0.1, rel=0.01)
    assert find(rows(circuits.run("a battery, an open switch and a bulb")), "current from the battery") == 0


def test_led_needs_its_forward_voltage():
    d = rows(circuits.run("a red LED and a 330 ohm resistor on a 9 V battery"))
    assert find(d, "red led: current") == pytest.approx((9 - 2.0) / (330 + 10 + 1.5), rel=0.03)


# ---------------- gravity (Newton + Kepler)
def test_moon_period_matches_kepler():
    d = rows(orbits.run("the moon orbits earth"))
    assert find(d, "period (measured)") == pytest.approx(find(d, "period (kepler"), rel=0.01)
    assert find(d, "period (measured)") / 24 == pytest.approx(27.3, rel=0.02)


def test_impact_speed_near_escape_speed():
    d = rows(orbits.run("an asteroid hits earth"))
    assert find(d, "impact speed") == pytest.approx(find(d, "escape speed"), rel=0.12)
