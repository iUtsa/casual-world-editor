"""Measurements are physics, not estimates: conservation and consistency checks."""
import math

import numpy as np
import pytest

from backend.app import measure, planner
from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Intervention
from backend.tests.test_physics import gt_scene


@pytest.fixture(scope="module")
def crash():
    scene, q, _ = stabilize(gt_scene("bridge"))
    ents = planner.snapshot(scene, None, [])
    ivs = [Intervention(**i) for i in planner.resolve(planner.parse_rules("drive a toy car into the bottle fast", ents, None), scene, ents, 0)[0]]
    return scene, ivs, simulate(scene, q, ivs)


def test_resting_contact_force_equals_weight(crash):
    scene, ivs, sim = crash
    for oid in ("mug_left", "book", "s1"):
        F = sim["kinematics"][oid]["F_contact"][-1]
        assert F == pytest.approx(sim["masses"][oid] * 9.81, rel=0.03), oid


def test_kinetic_energy_matches_half_m_v_squared_for_translation(crash):
    scene, ivs, sim = crash
    k = sim["kinematics"]["s1"]
    m = sim["masses"]["s1"]
    for f in range(len(k["v"])):
        if k["v"][f] is not None and math.hypot(*k["v"][f]) > 0.3:
            v2 = sum(c * c for c in k["v"][f])
            assert k["KE"][f] >= 0.5 * m * v2 * 0.999 and k["KE"][f] < 0.5 * m * v2 * 1.2


def test_energy_bookkeeping_balances_for_passive_object(crash):
    scene, ivs, sim = crash
    mt = measure.metrics(sim, "bottle", sim["segments"][1]["t_start"], sim["segments"][1]["t_end"])
    assert mt["energy_received_J"] == pytest.approx(mt["energy_given_up_J"], rel=0.1)   # starts and ends at rest


def test_answers(crash):
    scene, ivs, sim = crash
    a = measure.answer("how fast was the car when it hit the bottle?", scene, sim, ivs, None)
    assert 0.5 < a["value"] < 1.5 and a["unit"] == "m/s"
    a = measure.answer("what was the car's maximum acceleration?", scene, sim, ivs, None)
    assert a["unit"] == "m/s²"                                 # measured, not the capability test


def test_vehicle_capability_is_emergent():
    cap = measure.vehicle_capability("toy_car")
    assert 2.5 < cap["top_speed_mps"] < 3.2                     # below the 3.2 m/s no-load speed: losses are real
    assert cap["time_0_to_1mps_s"] < 1.0 and cap["braking_distance_from_top_m"] > 0.2
