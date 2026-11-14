"""Table plane + objects from multi-view geometry on a synthetic sweep, scored against ground truth.
Caches keyframes/depth/cameras in data/bench/<scene>_sfm.pkl. Usage: .../discover_bench.py [desk|bridge] [n]"""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app import synth  # noqa: E402
from backend.app.perception import discover, models, sfm  # noqa: E402

scene = sys.argv[1] if len(sys.argv) > 1 else "desk"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 56
o = synth.render_sweep(scene, n=n, az_path=(0, -85, -40, 0, 40, 85, 0), el_path=(-35, -22, -60, -70, -60, -22, -35))
K, ref = o["K"], n - 1
cache = Path(f"data/bench/{scene}_sfm_{n}.pkl")
if cache.exists():
    r = pickle.loads(cache.read_bytes())
else:
    dep = [models.estimate_depth(Image.fromarray(f)).astype(np.float32) for f in o["frames"]]
    r = sfm.reconstruct(o["frames"], dep, K, ref=ref)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(r))
t = time.time()
X = r.points[np.isfinite(r.points[:, 0])]
pl = discover.fit_plane(X)
# truth: the table plane in the reference camera's frame, scaled like the sweep
Rg, cg = o["poses"][ref]
n_true = Rg.T @ np.array([0, 0, 1.0])
ang = np.degrees(np.arccos(min(1.0, abs(float(n_true @ pl.n)))))
s_true = cg[2] / pl.d                                     # true camera height / estimated height = scale error
print(f"plane: {pl.inliers.sum()}/{len(X)} inliers, rms {pl.rms * 1000:.1f} mm, tilt error {ang:.2f} deg, "
      f"camera height {pl.d:.3f} (true {cg[2]:.3f}; scale x{s_true:.3f})")
par = discover.parallax_map(o["frames"], r.poses, K, ref, pl)
table = discover.table_region(pl, X, K, par.shape)
sess = models.SamSession(Image.fromarray(o["frames"][ref]))
masks = discover.find_objects(o["frames"][ref], par, table, sess)
print(f"discovery {time.time() - t:.1f}s: {len(masks)} objects found, {len(o['objects'])} true")
gt = o["ids"][ref]
for i, ob in enumerate(o["objects"], 1):
    b = gt == i
    best = max(((m & b).sum() / max(1, (m | b).sum()) for m in masks), default=0.0)
    print(f"   {ob['name']:17s} best IoU {best:.2f}")
vis = np.clip(par * 4, 0, 255).astype(np.uint8)
over = o["frames"][ref].copy()
pal = np.array([[230, 60, 60], [60, 160, 230], [240, 200, 40], [80, 200, 120], [170, 90, 200], [240, 130, 40], [40, 220, 220], [250, 250, 250]])
for j, m in enumerate(masks):
    over[m] = (0.5 * over[m] + 0.5 * pal[j % len(pal)]).astype(np.uint8)
cv2.imwrite(f"data/bench/discover_{scene}.jpg", cv2.cvtColor(np.concatenate([cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB), over], 1), cv2.COLOR_RGB2BGR))
