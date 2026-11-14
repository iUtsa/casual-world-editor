"""End-to-end perception check against renderer ground truth.

Usage: python -m backend.tests.eval_synth [scene] [--calibrate obj_name]"""
import json, sys, time
import numpy as np, cv2
from backend.app.perception import pipeline
from backend.app import synth

name = sys.argv[1] if len(sys.argv) > 1 else "bridge"
meta = synth.render(name)
rgb = cv2.cvtColor(cv2.imread(meta["rgb"]), cv2.COLOR_BGR2RGB)
gt_ids = cv2.imread(f"data/synthetic/{name}/ids_gt.png", 0)
t = time.time()
gtd = np.load(f"data/synthetic/{name}/depth_gt.npy") if "--gt-depth" in sys.argv else None
st = pipeline.scan(rgb, meta["hfov_deg"], "synthetic", use_claude=False, depth_override=gtd)
print(f"scan {time.time()-t:.2f}s timings={st.scene.timings_ms}")
# match predicted masks to GT objects by IoU
gt_names = meta["labels"]
matches = {}
for r in st.records:
    best, bi = 0, None
    for k, gname in enumerate(gt_names):
        g = gt_ids == k + 1
        iou = (g & r.mask).sum() / max(1, (g | r.mask).sum())
        if iou > best: best, bi = iou, gname
    matches[r.id] = (bi, best)
gt = {o["name"]: o for o in meta["objects"]}
def report(scene):
    for o in scene.objects:
        g, iou = matches.get(o.id, (None, 0))
        line = f"  {o.id:4s} -> {str(g):11s} IoU={iou:.2f} shape={o.shape:8s} size={np.round(o.size,3)} pos={np.round(o.pos,3)} on={o.supported_by} stable={o.stability.stable}"
        if g:
            gs = gt[g]["size"]; gs = [gs[0], gs[0], gs[1]] if gt[g]["shape"] == "cylinder" else gs
            line += f"\n        GT size={np.round(gs,3)} pos={np.round(gt[g]['pos'],3)}"
        print(line)
    print("  warnings:", scene.warnings)
print(f"detected {len(st.records)} objects vs {len(gt_names)} GT; unmatched GT:", set(gt_names) - {m[0] for m in matches.values() if m[1] > 0.5})
report(st.scene)
if "--calibrate" in sys.argv:
    gname = sys.argv[sys.argv.index("--calibrate") + 1]
    oid = max((k for k, (g, iou) in matches.items() if g == gname), key=lambda k: matches[k][1])
    gs = gt[gname]["size"]; h = gs[-1] if gt[gname]["shape"] == "cylinder" else gs[2]
    pipeline.calibrate(st, oid, "height", h)
    print(f"--- after calibrating {gname} height = {h} m (scale {st.depth_scale:.3f})")
    report(st.scene)
cv2.imwrite(f"data/synthetic/{name}/pred_ids.png", (pipeline.id_map(st) * 35).astype(np.uint8))
