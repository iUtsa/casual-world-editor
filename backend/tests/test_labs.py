"""Inclined planes, the transformer lab, and diagram-reading sentences (what the vision model returns)."""
import math

import pytest

from backend.app import circuits, experiment, transformer_lab


def rows(r):
    return {x["label"]: x["value"] for x in r["results"]}


def find(d, *words):
    for k, v in d.items():
        if all(w.lower() in k.lower() for w in words):
            return v
    raise KeyError(f"{words} not in {list(d)}")


# ---------------- inclined plane: a = g(sinθ − μcosθ), rolling (5/7)g·sinθ, stays if tanθ ≤ μ
def test_block_slides_with_friction():
    d = rows(experiment.run("a 2 kg block slides down a 30 degree ramp with friction 0.2"))
    theory = 9.81 * (math.sin(math.radians(30)) - 0.2 * math.cos(math.radians(30)))
    assert find(d, "acceleration down the ramp (measured)") == pytest.approx(theory, rel=0.05)


def test_block_stays_when_friction_wins():
    d = rows(experiment.run("a block on a 10 degree ramp with friction 0.4"))
    assert abs(find(d, "acceleration down the ramp (measured)")) < 0.05


def test_mass_and_units_not_objects():
    e = experiment.parse_rules("A 2 kg block is placed on a inclined plane with a 30-degree angle and a coefficient of friction of 0.2", {})
    assert [(o.name, o.count) for o in e.objects] == [("block", 1)]
    assert e.actions[0].do == "incline" and e.actions[0].angle_deg == 30 and e.actions[0].mu == 0.2


# ---------------- what the diagram reader produces must run correctly
def test_diagram_circuit_sentence():
    r = circuits.run("The circuit diagram shows a series circuit with a 9V battery and two resistors, one with 100 ohms and the other with 220 ohms")
    assert sorted(e["value"] for e in r["elements"]) == [100.0, 220.0]
    assert find(rows(r), "current from the battery") == pytest.approx(9 / (320 + 1.5), rel=0.02)


def test_diagram_pendulum_sentence():
    d = rows(experiment.run("A simple pendulum with a length of 2 meters and an angle of 20 degrees is shown"))
    assert find(d, "string length") == pytest.approx(2.0, rel=0.01)
    assert find(d, "swing period (measured)") == pytest.approx(2 * math.pi * math.sqrt(2 / 9.81), rel=0.03)


# ---------------- transformer lab: real model behaviour
@pytest.fixture(scope="module")
def eiffel():
    return transformer_lab.run("The Eiffel Tower is located in the city of")


def test_gpt2_predicts_paris(eiffel):
    assert eiffel["next"][0]["token"].strip() == "Paris"
    assert len(eiffel["attention"]) == 12 and len(eiffel["attention"][0]) == 12
    assert eiffel["lens"][-1]["top"][0]["token"].strip() == "Paris"


def test_hiding_eiffel_loses_paris(eiffel):
    r = transformer_lab.run(eiffel["text"], mask=[1, 2, 3])
    assert "Paris" not in [x["token"].strip() for x in r["next"][:3]]


def test_head_ablation_changes_output(eiffel):
    r = transformer_lab.run(eiffel["text"], ablate=[[l, h] for l in range(6, 12) for h in range(12)])
    assert r["next"][0]["token"].strip() != "Paris"
