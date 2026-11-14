"""Sweep scan: a hand-held video around the table -> every object as a real, textured, closed 3D mesh.

    video / phone burst --keyframes--> sharp, evenly spread frames (the LAST one is the reference view: where the
                                       phone ends up, so it is the live camera afterwards)
        --depth-->      Depth Anything V2 metric depth per keyframe
        --sfm-->        camera pose of every keyframe from the pixels (sfm.py), metric scale from the depth maps,
                        each depth map calibrated to the triangulated points
        --scan-->       the one-frame pipeline on the reference view (with its calibrated depth) finds the table,
                        the objects, their names and materials
        --track/carve-> masks in every keyframe from projected geometry (track.py), objects carved from all views
                        (carve.py); twice, the second round prompted by the first round's voxels
        --mesh-->       per object: watertight mesh, texture atlas from the original pixels, convex pieces for
                        physics (CoACD), volume -> mass

Everything that is measured vs. assumed is written to the object's notes.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .. import config

log = logging.getLogger(__name__)
MESH_DIR = config.DATA_DIR / "meshes"


@dataclass
class SweepResult:
    st: object                                   # PerceptionState of the reference view (scene objects = meshes)
    keyframes: int
    sfm_rms_px: float
    pose_note: str
    timings_s: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def sharpness(img: np.ndarray) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def keyframes(frames: list[np.ndarray], n: int = 56) -> list[int]:
    """Indices of up to n frames: the sharpest in each of n equal time windows; always ends with the last steady
    frame (the reference view)."""
    if len(frames) <= n:
        return list(range(len(frames)))
    sh = np.array([sharpness(f) for f in frames])
    edges = np.linspace(0, len(frames) - 1, n).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        out.append(int(a + np.argmax(sh[a:max(a + 1, b)])))
    tail = range(max(0, len(frames) - 8), len(frames))           # the reference: sharpest of the last few frames
    out.append(int(max(tail, key=lambda i: sh[i])))
    return sorted(set(out))


def read_video(path: str | Path, max_width: int = config.MAX_PROCESS_WIDTH, max_frames: int = 900) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("could not open the video (MP4 or MOV, H.264 or HEVC)")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, total // max_frames) if total else 1
    out, k = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if k % step == 0:
            ok, bgr = cap.retrieve()
            if ok:
                h, w = bgr.shape[:2]
                if w > max_width:
                    s = max_width / w
                    bgr = cv2.resize(bgr, (max_width, int(round(h * s / 2) * 2)), interpolation=cv2.INTER_AREA)
                out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        k += 1
    cap.release()
    if len(out) < 8:
        raise ValueError(f"the video has only {len(out)} usable frames; record 10-20 s moving slowly around the table")
    return out


def intrinsics(w: int, h: int, hfov_deg: float) -> np.ndarray:
    f = w / 2 / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def run(frames: list[np.ndarray], hfov_deg: float, source: str = "sweep", progress=lambda stage, frac, msg="": None,
        keyframe_count: int = 56, label: bool = True) -> SweepResult:
    from PIL import Image

    from . import carve as C
    from . import models, pipeline, sfm, track
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    idx = keyframes(frames, keyframe_count)
    kf = [frames[i] for i in idx]
    if len({f.shape for f in kf}) != 1:
        raise ValueError("frames have different sizes")
    h, w = kf[0].shape[:2]
    K = intrinsics(w, h, hfov_deg)
    ref = len(kf) - 1
    progress("depth", 0.0, f"{len(kf)} keyframes")
    depths = []
    for k, f in enumerate(kf):
        depths.append(models.estimate_depth(Image.fromarray(f)).astype(np.float32))
        progress("depth", (k + 1) / len(kf))
    T["depth"] = time.perf_counter() - t0

    t = time.perf_counter()
    progress("cameras", 0.0, "matching features")
    r = sfm.reconstruct(kf, depths, K, ref=ref, progress=lambda m: progress("cameras", 0.5, m))
    n_reg = sum(r.registered)
    if n_reg < max(8, 0.6 * len(kf)):
        raise ValueError(f"only {n_reg} of {len(kf)} views could be placed: move slower, keep the table in view, "
                         "and avoid plain, textureless surfaces")
    T["cameras"] = time.perf_counter() - t

    # the reference view through the one-frame pipeline (table plane, objects, labels), with calibrated depth
    t = time.perf_counter()
    progress("objects", 0.0, "finding objects in the reference view")
    st = pipeline.scan(kf[ref], hfov_deg, source, use_claude=True, depth_override=r.depths[ref], label=label)
    T["objects"] = time.perf_counter() - t
    sc = st.scene
    R_wr, t_wr = np.array(sc.camera.R_wc), np.array(sc.camera.t_wc)       # world <- reference camera
    poses, ok_view = [], []
    for k, (Rk, ck) in enumerate(r.poses):                               # sfm world = the reference camera's frame
        poses.append((R_wr @ Rk, R_wr @ ck + t_wr))
        ok_view.append(r.registered[k])
    keep = [k for k in range(len(kf)) if ok_view[k]]
    kf2 = [kf[k] for k in keep]
    poses2 = [poses[k] for k in keep]
    dep2 = [r.depths[k] if r.depth_err[k] < 0.05 else None for k in keep]
    ref2 = keep.index(ref)
    objs = sc.objects
    n_obj = len(objs)
    if not n_obj:
        return SweepResult(st, len(kf), r.rms_px, "no objects found in the reference view", T)
    rec_by_id = {rec.id: rec for rec in st.records}
    ref_masks = [rec_by_id[o.id].mask for o in objs]
    gobjs = [{"box": (np.array(o.pos), np.array(o.size) / 2 * [1.0, 1.0, 1.0] + 0.01, o.yaw), "points": None} for o in objs]
    tb = sc.table
    zmax = max(o.pos[2] + o.size[2] / 2 for o in objs)
    bounds = (np.array([tb.x_min, tb.y_min, 0.0]), np.array([tb.x_max, tb.y_max, max(0.1, 1.6 * zmax + 0.05)]))

    t = time.perf_counter()
    for rnd in (1, 2):
        progress("carving", (rnd - 1) / 2, f"round {rnd}: object outlines in every view")
        unknown: list = []
        ids, flags = track.masks(kf2, poses2, K, gobjs, ref2, ref_masks, unknown_out=unknown,
                                 progress=lambda a, b: progress("carving", (rnd - 1 + 0.6 * a / b) / 2))
        grid, occ = C.union_hull([m > 0 for m in ids], poses2, K, bounds, voxel=0.004 if rnd == 1 else 0.003,
                                 tol_frac=0.05, depths=dep2, margin=(0.015, 0.06), unknown=unknown)
        occ = C.consistent_carve(grid, occ, ids, poses2, K, n_obj, view_ok=flags)
        L, zbs = C.label_voxels(grid, occ, ids, poses2, K, n_obj, view_ok=flags)
        for i in range(n_obj):
            P = grid.centres(np.argwhere(L == i + 1))
            gobjs[i]["points"] = P if len(P) >= 20 else None
    T["carving"] = time.perf_counter() - t

    t = time.perf_counter()
    progress("meshes", 0.0, "meshes, textures and physics shapes")
    out_dir = MESH_DIR / sc.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = []
    for i, o in enumerate(objs):
        res = C.object_mesh(L == i + 1, grid)
        if res is None:
            o.notes.append("sweep: not enough views to carve it; kept the one-photo shape")
            continue
        V, F = res
        apply_mesh(o, V, F, i + 1, kf2, ids, zbs, poses2, K, grid.voxel, flags, out_dir, len(keep))
        progress("meshes", (i + 1) / n_obj)
    for o in objs:
        if o.shape == "mesh":
            st.meshes[o.id] = {"depth_scale": st.depth_scale, "size": list(o.size), "pos": list(o.pos), "yaw": o.yaw,
                               "mesh_url": o.mesh_url, "hulls": o.hulls, "volume_m3": o.volume_m3, "mass": o.mass,
                               "mass_source": o.mass_source, "notes": [n for n in o.notes if n.startswith("sweep")]}
    pipeline.rebuild(st)                                  # re-settle the world with the real shapes
    sc = st.scene
    T["meshes"] = time.perf_counter() - t
    sc.timings_ms.update({f"sweep_{k}": round(v * 1000) for k, v in T.items()})
    sc.source = source
    pose_note = f"{n_reg}/{len(kf)} views placed, reprojection error {r.rms_px:.2f} px, metric scale from depth x{r.metric_scale:.2f}"
    sc.warnings.append(f"Sweep scan: {pose_note}.")
    return SweepResult(st, len(kf), r.rms_px, pose_note, T, notes)


def apply_mesh(o, V: np.ndarray, F: np.ndarray, oid: int, frames, ids, zbs, poses, K, voxel: float, flags,
               out_dir: Path, n_views: int) -> None:
    """Turn a carved world-space mesh into the object's shape: body frame, texture, physics pieces, mass."""
    import trimesh

    from .materials import density_for
    mesh = trimesh.Trimesh(V, F, process=True)
    trimesh.smoothing.filter_taubin(mesh, iterations=6)
    if len(mesh.faces) > 9000:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=9000)
        except Exception:                                    # optional dependency; keep the full mesh
            pass
    V, F = np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32)
    # body frame: footprint's min-area rectangle (yaw), centre of the bounding box, z up
    (cx, cy), (bw, bd), ang = cv2.minAreaRect((V[:, :2] * 1000).astype(np.float32))
    yaw = float(np.radians(ang))
    zlo, zhi = float(V[:, 2].min()), float(V[:, 2].max())
    centre = np.array([cx / 1000, cy / 1000, (zlo + zhi) / 2])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Vb = (V - centre) @ Rz                                   # world -> body
    V2, F2, UV, atlas = __import__("backend.app.perception.carve", fromlist=["x"]).texture_atlas(
        V, F, oid, frames, ids, zbs, poses, K, voxel=voxel, view_ok=flags)
    V2b = (V2 - centre) @ Rz
    tm = trimesh.Trimesh(V2b, F2, process=False)
    from PIL import Image
    tm.visual = trimesh.visual.TextureVisuals(uv=UV, image=Image.fromarray(atlas))
    glb = out_dir / f"{o.id}.glb"
    tm.export(glb)
    # physics: convex pieces (CoACD) when the shape is clearly not convex, else its convex hull
    body = trimesh.Trimesh(Vb, F, process=True)
    vol = float(abs(body.volume)) if body.is_watertight else float(body.convex_hull.volume)
    hull = body.convex_hull
    pieces = [np.asarray(hull.vertices)]
    if hull.volume > 1.15 * max(vol, 1e-9):
        try:
            import coacd
            parts = coacd.run_coacd(coacd.Mesh(body.vertices, body.faces), threshold=0.06, max_convex_hull=8,
                                    preprocess_mode="off", resolution=1500)
            if parts:
                pieces = [np.asarray(v) for v, _ in parts]
        except Exception as e:                               # never fail a scan on decomposition
            log.warning("coacd failed for %s: %s", o.id, e)
    dens, dens_note = density_for(o.label, o.material)
    o.shape = "mesh"
    o.size = [float(bw / 1000), float(bd / 1000), float(zhi - zlo)]
    o.pos = [float(centre[0]), float(centre[1]), float(centre[2])]
    o.yaw = yaw
    o.profile = None
    o.mesh_url = f"/api/mesh/{out_dir.name}/{o.id}.glb"
    o.hulls = [[[round(float(x), 5) for x in p] for p in pc] for pc in pieces]
    o.volume_m3 = vol
    if o.mass_source != "user":
        o.mass = round(vol * dens, 4)
        o.mass_source = f"scanned volume {vol * 1e6:.0f} cm³ × {dens_note}"
    o.notes.append(f"sweep: carved from {n_views} views, {len(F)} faces, {len(pieces)} convex piece(s) for physics")
