"""Why does the reference view of a sweep find few objects? One-frame pipeline on the desk's reference view with
true depth, raw Depth Anything depth, and depth scaled to the true median. Usage: python .../ref_scan_bench.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from backend.app import synth  # noqa: E402
from backend.app.perception import models, pipeline  # noqa: E402

o = synth.render_sweep("desk", n=3, az_path=(0, -85, -40, 0, 40, 85, 0), el_path=(-35, -22, -60, -70, -60, -22, -35))
rgb, gt_d = o["frames"][-1], o["depth"][-1]
hfov = float(np.degrees(2 * np.arctan(o["K"][0, 2] / o["K"][0, 0])))
da = models.estimate_depth(Image.fromarray(rgb)).astype(np.float32)
ok = gt_d < 3
print(f"depth model / true median ratio {np.median(da[ok] / gt_d[ok]):.2f}")
Rg, cg = o["poses"][-1]
print(f"true camera height above table {cg[2]:.3f} m; true objects {len(o['objects'])}")
for name, d in [("true depth", np.where(ok, gt_d, np.nan)), ("depth model", da), ("depth model, median-scaled", da * np.median(gt_d[ok] / da[ok]))]:
    st = pipeline.scan(rgb, hfov, "bench", use_claude=False, depth_override=d, label=False)
    sc = st.scene
    print(f"{name:28s}: {len(sc.objects)} objects | camera height {sc.camera.t_wc[2]:.3f} m | plane rms {sc.table.plane_rms_m * 100:.1f} cm "
          f"inliers {sc.table.plane_inlier_ratio:.2f} | sizes " + ", ".join(f"{max(x.size[:2]) * 100:.0f}x{x.size[2] * 100:.0f}" for x in sc.objects))
    if sc.warnings:
        print("   warnings:", " | ".join(w[:90] for w in sc.warnings))
