"""How bodies really meet: tidal disruption, gravitational-wave inspiral, swallowed whole, or an impact — each checked."""
import pytest

from backend.app import encounters, orbits


def rows(r):
    return {x["label"]: x for x in r["results"]}


@pytest.mark.parametrize("q,kind", [
    ("the sun collides with a black hole", "tidal disruption"),
    ("the sun falls into sagittarius a*", "tidal disruption"),
    ("a neutron star hits the earth", "tidal disruption"),
    ("two black holes collide", "gravitational-wave inspiral"),
    ("two neutron stars collide", "gravitational-wave inspiral"),
    ("the moon crashes into earth", None),
    ("an asteroid hits the earth", None),
    ("two stars collide", None),
])
def test_encounter_type(q, kind):
    r = orbits.run(q)
    assert r.get("encounter") == kind
    assert r["mode"] == "collide"


def test_tde_matches_textbook_for_a_supermassive_black_hole():
    r = orbits.run("the sun falls into sagittarius a*")
    d = rows(r)
    assert 45 <= d["Debris left bound (measured)"]["value"] <= 55              # frozen-in: half bound
    assert d["Agreement with theory"]["value"] >= 95                           # energy spread + fall-back time
    assert r["events"][0]["kind"] == "disrupt"


def test_tde_not_scored_outside_its_validity():
    r = orbits.run("the sun collides with a black hole")                       # 10 Msun: textbook TDE estimates don't hold
    assert "Agreement with theory" not in rows(r)
    assert any("only rough" in n for n in r["notes"])


def test_swallowed_whole_by_a_huge_black_hole():
    sun = orbits._body("sun"); bh = orbits._body("black hole", {"*": 1e9})
    assert encounters.classify(bh, sun)["type"] == "swallowed"                 # r_t inside the horizon for ~1e9 Msun


def test_black_hole_merger_energy_matches_ligo():
    d = rows(orbits.run("two black holes collide"))
    e = d["Energy radiated as gravitational waves"]["value"]
    assert 0.040 * 20 <= e <= 0.050 * 20                                      # equal 10+10: 4.5–4.8 % of 20 Msun
    m1, m2 = 36.0, 29.0
    nu = m1 * m2 / (m1 + m2) ** 2
    e150914 = (encounters.ETA_SCHW * nu + 0.498 * nu ** 2) * (m1 + m2)
    assert 2.5 <= e150914 <= 3.5                                               # LIGO: 3.0 ± 0.5 Msun c²


@pytest.mark.parametrize("q,fate", [("the moon crashes into earth", "The Moon is destroyed"), ("two stars collide", "merge into one star")])
def test_impact_fate(q, fate):
    r = orbits.run(q)
    assert any(fate in s for s in r["story"]), r["story"]
    assert rows(r)["Agreement with theory"]["value"] >= 99.5                    # impact speed vs energy conservation
