"""Regressions from a live session: 'through a ball at 45 degree' on a scan with one huge object.

1. The free-spot search only looked within 0.6 m of the table centre, so one big object over the centre made the
   planner give up ('no free, visible spot') although the table had room near its edges.
2. 'through' (a common misspelling of 'throw') wasn't an action word, so the ball was dropped instead of thrown."""
import numpy as np
import pytest

from backend.app.experiment import parse_rules
from backend.app.planner import Entity, place
from backend.tests.test_physics import gt_scene


def test_free_spot_found_around_a_huge_central_object():
    scene = gt_scene("bridge")
    t = scene.table
    centre = np.array([(t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2, 0.3])
    huge = Entity("huge", "huge thing", centre, 0.65, 0.6)
    xy, _, _ = place(scene, [huge], 0.037, "none", None)
    assert t.x_min < xy[0] < t.x_max and t.y_min < xy[1] < t.y_max
    assert np.linalg.norm(xy - centre[:2]) > huge.radius + 0.037          # not inside the big object


def test_no_room_at_all_still_says_why():
    scene = gt_scene("bridge")
    t = scene.table
    centre = np.array([(t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2, 0.3])
    with pytest.raises(ValueError, match="rescanning"):
        place(scene, [Entity("all", "everything", centre, 5.0, 0.6)], 0.037, "none", None)


@pytest.mark.parametrize("text,angle", [("through a ball at 45 degree", 45.0), ("thru a ball at 30 degrees", 30.0),
                                        ("throw a ball at 45 degrees", 45.0)])
def test_throw_misspellings(text, angle):
    e = parse_rules(text, {})
    assert [(a.do, a.angle_deg) for a in e.actions] == [("throw", angle)]


def test_through_keeps_its_meaning_mid_sentence():
    assert [a.do for a in parse_rules("roll a ball through the tunnel", {}).actions] == ["roll"]
