"""Perception regression tests on synthetic scenes with sensor (renderer) depth.

These isolate segmentation / lifting / support logic from monocular-depth
error. They load the real SAM 2 + run the full geometric pipeline (~5 s each)."""
import cv2
import numpy as np
import pytest

from backend.app import synth
from backend.app.perception import pipeline


def _scan(name):
    meta = synth.render(name)
    rgb = cv2.cvtColor(cv2.imread(meta["rgb"]), cv2.COLOR_BGR2RGB)
    depth = np.load(f"data/synthetic/{name}/depth_gt.npy")
    gt_ids = cv2.imread(f"data/synthetic/{name}/ids_gt.png", 0)
    st = pipeline.scan(rgb, meta["hfov_deg"], "synthetic", use_claude=False, depth_override=depth)
    match = {}
    for r in st.records:
        ious = [((gt_ids == k + 1) & r.mask).sum() / max(1, ((gt_ids == k + 1) | r.mask).sum()) for k in range(len(meta["labels"]))]
        k = int(np.argmax(ious))
        if ious[k] > 0.5:
            match[r.id] = meta["labels"][k]
    return st, {v: k for k, v in match.items()}


@pytest.fixture(scope="module")
def bridge():
    return _scan("bridge")


def test_tower_top_on_middle():
    st, by_name = _scan("tower")
    objs = {o.id: o for o in st.scene.objects}
    assert {"can", "mug", "middle_box", "top_box"} <= set(by_name)
    assert objs[by_name["top_box"]].supported_by == [by_name["middle_box"]]


def test_bridge_finds_every_object(bridge):
    st, by_name = bridge
    assert set(by_name) == {"mug_left", "mug_right", "book", "tea_box", "bottle", "block"}
    assert not st.scene.warnings


def test_bridge_shapes_from_curvature(bridge):
    st, by_name = bridge
    objs = {o.id: o for o in st.scene.objects}
    for n in ("mug_left", "mug_right", "bottle"):
        assert objs[by_name[n]].shape == "cylinder", n
    for n in ("book", "block", "tea_box"):
        assert objs[by_name[n]].shape == "box", n


def test_bridge_support_graph(bridge):
    st, by_name = bridge
    objs = {o.id: o for o in st.scene.objects}
    assert set(objs[by_name["book"]].supported_by) == {by_name["mug_left"], by_name["mug_right"]}
    assert objs[by_name["tea_box"]].supported_by == [by_name["book"]]


@pytest.mark.xfail(reason="Known limitation: stacked boxes of similar footprint meet at concave contact lines with "
                          "little occlusion, so the base box is under-segmented automatically. Fix in the UI with Add/refine.",
                   strict=False)
def test_tower_stack_order():
    st, by_name = _scan("tower")
    objs = {o.id: o for o in st.scene.objects}
    assert objs[by_name["top_box"]].supported_by == [by_name["middle_box"]]
    assert by_name["middle_box"] in objs[by_name["base_box"]].supporting or \
        objs[by_name["middle_box"]].supported_by == [by_name["base_box"]]
