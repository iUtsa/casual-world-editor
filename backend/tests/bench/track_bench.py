"""Geometry-guided object masks through a synthetic sweep, scored against ground-truth object ids.

Round 1 starts from rough 3D boxes (ground truth with the hidden back grown 50 %, like a one-photo scan's guess);
round 2 uses the voxels carved from round-1 masks. Camera poses are ground truth here, so this measures the masks.
Usage: python backend/tests/bench/track_bench.py [n_frames]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402

from backend.app import synth  # noqa: E402
from backend.app.perception import carve as C  # noqa: E402
from backend.app.perception import track  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
o = synth.render_sweep("bridge", n=n, az_path=(0, -85, -40, 0, 40, 85, 0), el_path=(-35, -22, -60, -70, -60, -22, -35))
gt, K, poses = o["ids"], o["K"], o["poses"]
k_obj = len(o["objects"])
ref = n - 1
names = [x["name"] for x in o["objects"]]


MIN_AREA = int(0.002 * gt[0].size)            # clearly visible: >= 0.2 % of the frame (slivers counted apart)


def score(ids, ok, label):
    ious, flagged_bad, flagged = [], 0, 0
    per = np.zeros(k_obj)
    slivers = sum(1 for k in range(n) for i in range(1, k_obj + 1) if 30 <= (gt[k] == i).sum() < MIN_AREA)
    for k in range(n):
        for i in range(1, k_obj + 1):
            a, b = ids[k] == i, gt[k] == i
            if b.sum() < MIN_AREA:
                continue
            iou = (a & b).sum() / max(1, (a | b).sum())
            ious.append(iou)
            per[i - 1] += iou
            if not ok[k][i]:
                flagged += 1
                flagged_bad += iou < 0.7
    cnt = np.array([sum((gt[k] == i).sum() >= MIN_AREA for k in range(n)) for i in range(1, k_obj + 1)])
    print(f"{label}: mean IoU {np.mean(ious):.3f} | IoU<0.7 in {np.mean(np.array(ious) < 0.7) * 100:.0f}% of masks | "
          f"flagged {flagged} ({flagged_bad} of them really bad) | {slivers} sliver views not scored")
    print("   per object: " + "  ".join(f"{s}={v:.2f}" for s, v in zip(names, per / np.maximum(cnt, 1))))


objs = []
for x in o["objects"]:
    if x["shape"] == "cylinder":
        size = np.array([x["size"][0], x["size"][0], x["size"][1]])
    else:
        size = np.array(x["size"], float)
    half = size / 2 * np.array([1.0, 1.5, 1.0])          # back half grown: a single view can't see it
    c = np.array(x["pos"], float) + np.array([0, size[1] * 0.25, 0])
    objs.append({"box": (c, half, x.get("yaw", 0.0)), "points": None})
ref_masks = [gt[ref] == i for i in range(1, k_obj + 1)]

t = time.time()
unk1: list = []
ids1, ok1 = track.masks(o["frames"], poses, K, objs, ref, ref_masks, unknown_out=unk1)
print(f"round 1 (boxes): {time.time() - t:.1f}s")
score(ids1, ok1, "round 1")

t = time.time()
grid, occ = C.union_hull([m > 0 for m in ids1], poses, K, (np.array([-0.6, -0.4, 0.0]), np.array([0.6, 0.6, 0.4])),
                         voxel=0.004, tol_frac=0.05, unknown=unk1)
occ = C.consistent_carve(grid, occ, ids1, poses, K, k_obj, view_ok=ok1)
L, _ = C.label_voxels(grid, occ, ids1, poses, K, k_obj, view_ok=ok1)
for i in range(1, k_obj + 1):
    P = grid.centres(np.argwhere(L == i))
    objs[i - 1]["points"] = P if len(P) >= 20 else None
print(f"carve: {time.time() - t:.1f}s")
t = time.time()
ids2, ok2 = track.masks(o["frames"], poses, K, objs, ref, ref_masks)
print(f"round 2 (carved voxels): {time.time() - t:.1f}s")
score(ids2, ok2, "round 2")

# diagnostics: the worst frames per object, prediction | truth | image
import cv2  # noqa: E402
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if out:
    out.mkdir(parents=True, exist_ok=True)
    pal = np.array([[0, 0, 0], [230, 230, 220], [50, 90, 200], [200, 40, 40], [240, 190, 50], [60, 170, 100], [150, 90, 170]], np.uint8)
    for i in range(1, k_obj + 1):
        sc = []
        for k in range(n):
            b = gt[k] == i
            if b.sum() >= 30:
                a = ids2[k] == i
                sc.append(((a & b).sum() / max(1, (a | b).sum()), k))
        for iou, k in sorted(sc)[:2]:
            row = np.concatenate([pal[ids2[k]], pal[gt[k]], o["frames"][k]], 1)
            cv2.imwrite(str(out / f"{names[i - 1]}_f{k:02d}_iou{iou:.2f}.jpg"), cv2.cvtColor(cv2.resize(row, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    np.savez_compressed(out / "masks.npz", ids1=np.stack(ids1), ids2=np.stack(ids2))
