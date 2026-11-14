"""End-to-end sweep scan on a synthetic sweep, everything estimated from the RGB frames (like a real video), scored
against the true meshes. Usage: python backend/tests/bench/sweep_bench.py [scene=desk] [n_frames=56]"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402

from backend.app import synth  # noqa: E402
from backend.app.perception import sweep  # noqa: E402

scene = sys.argv[1] if len(sys.argv) > 1 else "desk"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 56
o = synth.render_sweep(scene, n=n, az_path=(0, -85, -40, 0, 40, 85, 0), el_path=(-35, -22, -60, -70, -60, -22, -35))
hfov = float(np.degrees(2 * np.arctan(o["K"][0, 2] / o["K"][0, 0])))
t = time.time()
res = sweep.run(o["frames"], hfov, "sweep:bench", progress=lambda s, f, m="": m and print(f"  [{s}] {m}", flush=True),
                keyframe_count=n, label=False)
print(f"sweep {time.time() - t:.0f}s  timings {json.dumps({k: round(v) for k, v in res.timings_s.items()})}  {res.pose_note}")
sc = res.st.scene

# the estimated world is the reference camera's table frame; GT is MuJoCo's. Align with the reference camera pose:
# world_gt <- cam_ref <- world_est  (both put the table at z = 0; scale differs by the depth model's error)
Rg, cg = o["poses"][-1]
Re, ce = np.array(sc.camera.R_wc), np.array(sc.camera.t_wc)
A = Rg @ Re.T                                            # rotation est-world -> gt-world


def to_gt(p, s=1.0):
    return (A @ ((np.asarray(p) - ce) * s).T).T + cg


def chamfer(Va, Fa, Vb, Fb, k=4000):
    import trimesh
    from scipy.spatial import cKDTree
    pa = trimesh.Trimesh(Va, Fa).sample(k)
    pb = trimesh.Trimesh(Vb, Fb).sample(k)
    return 0.5 * (cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())


import trimesh  # noqa: E402
# metric scale error: the reference camera's height above the table, estimated vs true
s_fix = cg[2] / max(1e-6, ce[2])
print(f"scale: estimated camera height {ce[2]:.3f} m vs true {cg[2]:.3f} m  (x{s_fix:.3f} to correct)")
gts = o["objects"]
rows = []
for ob in sc.objects:
    c_gt = to_gt(ob.pos, s_fix)
    j = int(np.argmin([np.linalg.norm(np.array(g["pos"][:2]) - c_gt[:2]) for g in gts]))
    g = gts[j]
    d = np.linalg.norm(np.array(g["pos"][:2]) - c_gt[:2])
    line = f"{g['name']:17s} <- {ob.id:4s} {ob.shape:5s} centre off {d * 100:4.1f} cm"
    if ob.shape == "mesh":
        import trimesh as _t
        glb = _t.load(Path("data/meshes") / sc.scene_id / f"{ob.id}.glb", force="mesh")
        c, s_ = np.cos(ob.yaw), np.sin(ob.yaw)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
        Vw = np.asarray(glb.vertices) @ Rz.T + np.array(ob.pos)                 # body -> est world
        Vg = to_gt(Vw, s_fix)
        ext = Vg.max(0) - Vg.min(0)
        vol = abs(_t.Trimesh(np.asarray(glb.vertices), glb.faces).convex_hull.volume) * s_fix ** 3
        gvol = abs(_t.Trimesh(g["V"], g["F"]).convex_hull.volume)
        ch = chamfer(Vg, np.asarray(glb.faces), g["V"], g["F"])
        line += (f" | height {ext[2] * 100:5.1f} (gt {g['size'][2] * 100:5.1f}) cm | hull vol {vol * 1e6:6.0f} (gt {gvol * 1e6:6.0f}) cm3"
                 f" | chamfer {ch * 1000:5.1f} mm")
        rows.append(ch)
    print(line)
if rows:
    print(f"mean chamfer {np.mean(rows) * 1000:.1f} mm over {len(rows)} meshes")
