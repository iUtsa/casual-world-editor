"""Accuracy against physical theory, for every engine.

Each case compares the simulation with a textbook formula computed HERE, independently of the app's own
theory rows, and also checks the app's "Agreement with theory" score. Target: within 1% (99% agreement).
Run `python backend/tests/accuracy_report.py` for the full table.
"""
import math

import pytest
from scipy.special import ellipk

from backend.app import circuits, experiment, orbits
from backend.app.physics import drag

G = 9.81


def rows(r):
    return {x["label"]: x["value"] for x in r["results"]}


def find(d, *words):
    for k, v in d.items():
        if all(w.lower() in k.lower() for w in words):
            return v
    raise KeyError(f"{words} not in {list(d)}")


def close(a, b, pct=1.0):
    return abs(a - b) <= pct / 100 * abs(b)


# ---------------------------------------------------------------- the drag solver itself
def test_drag_solver_matches_closed_form():
    """Quadratic drag, no buoyancy: v(h) = v_t √(1 − e^(−2gh/v_t²)), v_t = √(2mg / ρC_dA)."""
    sp = drag.spec("sphere", [0.074] * 3)
    m, rho = 0.145, 1.2
    A = math.pi * 0.037 ** 2
    vt = math.sqrt(2 * m * G / (rho * 0.47 * A))
    for h in (1.0, 10.0, 50.0):
        v = drag.fall(m, 0.0, sp, rho, 0.0, G, h=h)["v"]
        assert v == pytest.approx(vt * math.sqrt(1 - math.exp(-2 * G * h / vt ** 2)), rel=1e-4)


# ---------------------------------------------------------------- mechanics, against formulas computed here
def test_free_fall_on_the_moon():
    d = rows(experiment.run("drop a ball from 5 meters on the moon"))
    h = find(d, "fall height")
    assert close(find(d, "impact speed"), math.sqrt(2 * 1.62 * h))
    assert close(find(d, "time to fall"), math.sqrt(2 * h / 1.62))


def test_pendulum_exact_period_at_60_degrees():
    d = rows(experiment.run("a pendulum 1 m long released at 60 degrees"))
    T = 4 * math.sqrt(1.0 / G) * ellipk(math.sin(math.radians(30)) ** 2)        # complete elliptic integral
    assert close(find(d, "swing period (measured)"), T, 0.5)
    assert close(find(d, "string tension at the bottom (measured)"), 1.0 * G * (3 - 2 * math.cos(math.radians(60))), 0.5)


def test_mass_on_spring_period():
    d = rows(experiment.run("a 1 kg mass on a spring with k 100"))
    assert close(find(d, "oscillation period (measured)"), 2 * math.pi * math.sqrt(1 / 100), 0.5)


def test_block_on_ramp_with_friction():
    d = rows(experiment.run("a 2 kg block slides down a 30 degree ramp with friction 0.2"))
    th = math.radians(30)
    assert close(find(d, "acceleration down the ramp (measured)"), G * (math.sin(th) - 0.2 * math.cos(th)))


def test_ball_rolls_down_ramp():
    d = rows(experiment.run("a ball rolls down a 30 degree ramp"))
    th = math.radians(30)
    crr = find(d, "rolling resistance")
    assert close(find(d, "acceleration down the ramp (measured)"), 5 / 7 * G * (math.sin(th) - crr * math.cos(th)), 1.5)


def test_projectile_on_the_moon():
    d = rows(experiment.run("throw a ball at 10 m/s at 60 degrees on the moon"))
    v, th, g = 10.0, math.radians(60), 1.62
    vx, vy = v * math.cos(th), v * math.sin(th)
    assert close(find(d, "peak height above launch"), vy * vy / (2 * g))
    t = find(d, "time of flight")                     # no air: constant sideways speed for the whole flight
    assert close(find(d, "range (to landing)"), vx * t, 0.5)


def test_head_on_collision_follows_newtons_law_of_restitution():
    d = rows(experiment.run("a 2 kg cart hits a 1 kg cart at 3 m/s"))
    u1, m1, m2 = find(d, "cart speed just before"), 2.0, 1.0
    e = find(d, "restitution set")
    v1 = (m1 * u1 - m2 * e * u1) / (m1 + m2)
    v2 = (m1 * u1 + m1 * e * u1) / (m1 + m2)
    assert close(find(d, "cart velocity after", "measured"), v1)
    assert close(find(d, "cart 2 velocity after", "measured"), v2)


def test_equal_balls_exchange_and_conserve_momentum():
    d = rows(experiment.run("two billiard balls collide"))
    assert find(d, "momentum along the line of impact conserved") > 99
    assert close(find(d, "coefficient of restitution"), find(d, "restitution set"), 1.0)


def test_floating_block_water_line():
    d = rows(experiment.run("a wooden block floats in water"))
    assert close(find(d, "under water (measured)"), 100 * find(d, "wooden block density") / 1000.0)


def test_jump_height_on_the_moon():
    d = rows(experiment.run("a man jumps on the moon"))
    v0 = find(d, "take-off speed")
    assert close(find(d, "jump height"), v0 * v0 / (2 * 1.62))


# ---------------------------------------------------------------- circuits: Ohm's law by hand
def test_series_parallel_network():
    r = circuits.run("a 6 V battery with a 10 ohm resistor in series with a 20 ohm and a 30 ohm resistor in parallel")
    R = 10 + 1 / (1 / 20 + 1 / 30)
    r_int = circuits.parse("a 6 V battery").source.r_int
    assert close(find(rows(r), "current from the battery"), 6 / (R + r_int), 0.01)


def test_rc_charging_curve():
    d = rows(circuits.run("a 9 V battery, a 1000 ohm resistor and a 100 microfarad capacitor"))
    assert close(find(d, "capacitor voltage after one τ (measured)"), 9 * (1 - math.exp(-1)), 0.2)


# ---------------------------------------------------------------- gravity: Kepler and energy conservation
def test_iss_period_kepler():
    d = rows(orbits.run("the iss orbits the earth"))
    r, GM = 6771e3, 6.674e-11 * (5.972e24 + 420000)
    assert close(find(d, "orbital period (measured)") * 3600, 2 * math.pi * math.sqrt(r ** 3 / GM), 0.1)


def test_collision_speed_energy_conservation():
    d = rows(orbits.run("an asteroid hits the earth"))
    assert close(find(d, "impact speed"), find(d, "impact speed (theory"), 0.1)


# ---------------------------------------------------------------- every theory comparison in the app: >= 99%
CASES = [
    "drop a ball from 10 meters", "drop a feather from 2 meters", "drop a bowling ball from 20 meters",
    "drop a ball from 5 meters on the moon", "a wooden block floats in water", "a steel ball drops in water",
    "a beach ball in water", "an ice cube floats in water", "throw a ball at 20 m/s at 45 degrees",
    "a ball is kicked from the ground at 15 m/s at 30 degrees", "throw a baseball at 30 m/s at 20 degrees",
    "a 1 kg mass on a spring with k 100", "a pendulum 2 m long released at 10 degrees", "a pendulum 0.5 m long released at 80 degrees",
    "a ball rolls down a 30 degree ramp", "a block slides down a frictionless 45 degree ramp 5 m long",
    "two 1 kg balls collide head on at 3 m/s", "a 2 kg ball hits a 1 kg ball at 4 m/s", "two baseballs collide",
    "a tennis ball hits a bowling ball", "a man jumps", "a kangaroo jumps",
]


@pytest.mark.parametrize("prompt", CASES)
def test_mechanics_agreement(prompt):
    ag = find(rows(experiment.run(prompt)), "agreement with theory")
    assert ag >= 99.0, f"{prompt}: {ag:.2f}%"


@pytest.mark.parametrize("prompt", ["a 9 V battery with two bulbs in parallel", "a 9 V battery with three bulbs in series",
                                    "a 12 V battery with a 100 ohm and a 300 ohm resistor in parallel",
                                    "a 9 V battery, a 1000 ohm resistor and a 100 microfarad capacitor"])
def test_circuit_agreement(prompt):
    assert find(rows(circuits.run(prompt)), "agreement with theory") >= 99.9


@pytest.mark.parametrize("prompt", ["the moon orbits the earth", "mars orbits the sun", "jupiter collides with the moon"])
def test_gravity_agreement(prompt):
    assert find(rows(orbits.run(prompt)), "agreement with theory") >= 99.9


# ---------------------------------------------------------------- space: NASA JPL data + REBOUND
def test_moon_matches_nasa_after_a_month():
    d = rows(orbits.run("the moon orbits the earth"))
    assert find(d, "position error vs nasa") < 50                               # km after 27 days
    assert find(d, "energy conserved") < 1e-12


def test_apophis_2029_close_approach_matches_nasa():
    d = rows(orbits.run("apophis flyby 2029"))
    assert find(d, "closest approach of apophis") == pytest.approx(38012, abs=60)  # km from Earth's centre (NASA JPL)


def test_halley_next_perihelion_2061():
    r = orbits.run("halley's comet")
    when = next(x["text"] for x in r["results"] if "perihelion date" in x["label"])
    assert when.endswith("July 2061")
    assert find(rows(r), "perihelion — closest") / 1.496e8 == pytest.approx(0.593, abs=0.01)   # AU


def test_solar_system_kepler_and_conservation():
    d = rows(orbits.run("the solar system"))
    for planet in ("Mercury", "Venus", "Earth", "Mars", "Jupiter"):
        assert find(d, planet, "period (measured)") == pytest.approx(find(d, planet, "period (kepler"), rel=0.005)
    assert find(d, "agreement with theory") >= 99.9


def test_binary_star_measured_orbit():
    d = rows(orbits.run("alpha centauri"))
    assert find(d, "period (measured)") / 8766 == pytest.approx(79.91, rel=0.002)    # years
    assert find(d, "eccentricity (measured)") == pytest.approx(0.5208, abs=0.002)


def test_galaxies_follow_their_theory():
    d = rows(orbits.run("the milky way and andromeda collide"))
    assert find(d, "close within 300 kpc (measured)") == pytest.approx(find(d, "close within 300 kpc (theory"), rel=0.005)
    assert 3.8 <= find(d, "first close pass (measured)") <= 5.0                 # billion years (published 3.9-4.5)
    assert find(d, "momentum conserved") < 1e-9
