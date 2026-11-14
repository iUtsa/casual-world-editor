"""Asset library: real 3D models (CC0) normalised for physics and AR rendering.

Each asset gets:
  * a transform from glTF space (Y-up, metres-ish) to the physics body frame
    (Z-up, origin at the bottom-centre of the bounding box, vehicles face +X),
    including a scale to a realistic tabletop size;
  * a convex-hull collision mesh in body coordinates (MuJoCo mesh geom);
  * physical properties (mass, friction, bounciness);
  * for vehicles, wheel geometry found in the mesh (centres, radius, axle) so
    the physics can drive real wheels and the renderer can spin them.
The manifest is cached in data/assets_manifest.json and rebuilt when files change.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)
LIB = config.ROOT / "assets" / "library"
CACHE = config.DATA_DIR / "assets_manifest.json"

# glTF (x, y, z) -> body (z_gltf, x_gltf, y_gltf): Y-up becomes Z-up, glTF +Z (model front) becomes +X.
R_GLTF_TO_BODY = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)

CURATED: dict[str, dict] = {
    "toy_car": {"name": "toy car", "aliases": ["car", "toy car", "vehicle", "racecar", "race car", "auto"],
                "size": 0.15, "mass": 0.25, "friction": 0.9, "kind": "vehicle", "nodes": ["ToyCar", "Glass"]},
    "rubber_duck_toy": {"name": "rubber duck", "aliases": ["duck", "rubber duck", "ducky"], "size": 0.10, "mass": 0.04, "friction": 0.8},
    "baseball_01": {"name": "baseball", "aliases": ["ball", "baseball"], "size": 0.074, "mass": 0.145, "friction": 0.6, "bouncy": True, "collision": "sphere"},
    "lemon": {"name": "lemon", "aliases": ["lemon", "fruit"], "size": 0.085, "mass": 0.12, "friction": 0.6},
    "yellow_onion": {"name": "onion", "aliases": ["onion"], "size": 0.08, "mass": 0.15, "friction": 0.6},
    "can_rusted": {"name": "tin can", "aliases": ["tin can", "tin", "soda can", "food can"], "size": 0.12, "mass": 0.35, "friction": 0.4},
    "cardboard_box_01": {"name": "cardboard box", "aliases": ["box", "cardboard box", "package", "parcel"], "size": 0.15, "mass": 0.08, "friction": 0.6},
    "wooden_crate_01": {"name": "wooden crate", "aliases": ["crate", "wooden crate"], "size": 0.2, "mass": 0.3, "friction": 0.6},
    "garden_gnome": {"name": "garden gnome", "aliases": ["gnome", "garden gnome", "figurine"], "size": 0.15, "mass": 0.2, "friction": 0.6},
    "alarm_clock_01": {"name": "alarm clock", "aliases": ["clock", "alarm clock"], "size": 0.13, "mass": 0.25, "friction": 0.6},
    "croissant": {"name": "croissant", "aliases": ["croissant", "pastry"], "size": 0.13, "mass": 0.06, "friction": 0.7},
    "wooden_bowl_01": {"name": "wooden bowl", "aliases": ["bowl", "wooden bowl"], "size": 0.16, "mass": 0.2, "friction": 0.6},
}

# Procedural assets: simple real-world things whose true shape IS a primitive.
PROCEDURAL: dict[str, dict] = {
    "domino": {"name": "domino", "aliases": ["domino", "dominoes", "dominos"], "shape": "box",
               "dims": [0.024, 0.0075, 0.048], "mass": 0.012, "friction": 0.35, "material": "domino"},
    "marble": {"name": "glass marble", "aliases": ["marble", "glass marble"], "shape": "sphere",
               "dims": [0.025, 0.025, 0.025], "mass": 0.02, "friction": 0.3, "material": "glass", "bouncy": True},
    "ramp": {"name": "wooden ramp", "aliases": ["ramp", "slope", "incline", "wedge"], "shape": "wedge",
             "dims": [0.30, 0.10, 0.07], "mass": 0.4, "friction": 0.5, "material": "wood", "static": True},
    "steel_ball": {"name": "steel ball", "aliases": ["steel ball", "ball bearing", "metal ball"], "shape": "sphere",
                   "dims": [0.04, 0.04, 0.04], "mass": 0.26, "friction": 0.3, "material": "steel"},
}


def _model_file(d: Path) -> Path | None:
    for n in ("model.glb", "model.gltf"):
        if (d / n).exists():
            return d / n
    return None


def _load_mesh(path: Path, nodes: list[str] | None):
    import trimesh
    sc = trimesh.load(path, force="scene")
    parts = []
    for name in sc.graph.nodes_geometry:
        if nodes and name not in nodes:
            continue
        T, g = sc.graph.get(name)
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        parts.append(m)
    return trimesh.util.concatenate(parts)


def _wheels(mesh_body: "np.ndarray", verts_gltf: np.ndarray, faces: np.ndarray, M: np.ndarray) -> dict | None:
    """Find the four wheels: small disc-shaped components (hubcaps / tyres) low
    on the body at the four corners. Returns centres (body frame), radius, width."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    import trimesh
    m = trimesh.Trimesh(verts_gltf, faces, process=True)
    adj = m.face_adjacency
    n = len(m.faces)
    A = sp.coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    vb = (np.c_[m.vertices, np.ones(len(m.vertices))] @ M.T)[:, :3]    # to body frame
    zmin, zmax = vb[:, 2].min(), vb[:, 2].max()
    cands = []
    for c in np.unique(lab):
        f = m.faces[lab == c]
        v = vb[np.unique(f)]
        e = v.max(0) - v.min(0)
        ctr = (v.max(0) + v.min(0)) / 2
        # disc facing sideways (thin along body y), low on the car
        if e[1] < 0.35 * min(e[0], e[2]) and abs(e[0] - e[2]) < 0.25 * max(e[0], e[2]) and ctr[2] < zmin + 0.45 * (zmax - zmin):
            cands.append((ctr, e))
    if len(cands) < 4:
        return None
    # cluster by corner (sign of x and y)
    corners: dict[tuple, list] = {}
    for ctr, e in cands:
        corners.setdefault((np.sign(ctr[0]), np.sign(ctr[1])), []).append((ctr, e))
    if len(corners) != 4:
        return None
    centres, radii, ys = [], [], []
    for key in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:          # FL, FR, RL, RR (front = +x, left = +y)
        group = corners.get(key)
        if not group:
            return None
        cz = float(np.median([g[0][2] for g in group]))
        cx = float(np.median([g[0][0] for g in group]))
        outer = max(abs(g[0][1]) for g in group)
        inner = min(abs(g[0][1]) for g in group)
        centres.append([cx, float(key[1] * (outer + inner) / 2), cz])
        radii.append(cz - zmin)
        ys.append(outer - inner)
    r = float(np.median(radii))
    width = max(0.25 * r, float(np.median(ys)) + 0.15 * r)
    return {"centres": centres, "radius": r, "width": width, "order": ["fl", "fr", "rl", "rr"]}


def _fps(pts: np.ndarray, k: int) -> np.ndarray:
    idx = [int(np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return pts[idx]


def _build_one(asset_id: str, cfg: dict, base: Path | None = None, url_prefix: str = "/assets/library") -> dict | None:
    import trimesh
    d = (base or LIB) / asset_id
    f = _model_file(d)
    if f is None:
        return None
    mesh = _load_mesh(f, cfg.get("nodes"))
    v = mesh.vertices @ R_GLTF_TO_BODY.T
    ext = v.max(0) - v.min(0)
    s = cfg["size"] / float(ext.max())
    lo, hi = v.min(0) * s, v.max(0) * s
    offset = np.array([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    M = np.eye(4)
    M[:3, :3] = R_GLTF_TO_BODY * s
    M[:3, 3] = offset
    vb = (np.c_[mesh.vertices, np.ones(len(mesh.vertices))] @ M.T)[:, :3]
    hull = trimesh.Trimesh(vb, mesh.faces, process=False).convex_hull
    if len(hull.vertices) > 64:       # keep MuJoCo hulls light: farthest-point subsample, re-hull
        hull = trimesh.Trimesh(_fps(np.asarray(hull.vertices), 64)).convex_hull
    lic = json.loads((d / "LICENSE.json").read_text()) if (d / "LICENSE.json").exists() else {}
    entry = {
        "id": asset_id, "name": cfg["name"], "aliases": cfg["aliases"], "kind": cfg.get("kind", "rigid"),
        "url": f"{url_prefix}/{asset_id}/{f.name}", "nodes": cfg.get("nodes"),
        "gltf_to_body": M.tolist(), "dims": [round(float(x), 4) for x in (hi - lo)],
        "mass": cfg["mass"], "friction": cfg["friction"], "bouncy": cfg.get("bouncy", False),
        "static": cfg.get("static", False), "collision": cfg.get("collision", "hull"),
        "hull": [[round(float(x), 5) for x in p] for p in hull.vertices],
        "thumb": f"{url_prefix}/{asset_id}/thumb.png" if (d / "thumb.png").exists() else None,
        "license": lic.get("license"), "source": lic.get("source"), "authors": lic.get("authors"),
        "generated": bool(cfg.get("generated")), "prompt": cfg.get("prompt"),
    }
    if entry["kind"] == "vehicle":
        w = _wheels(None, mesh.vertices, mesh.faces, M)
        if w is None:
            log.warning("vehicle %s: wheels not found, treated as rigid", asset_id)
            entry["kind"] = "rigid"
        else:
            entry["wheels"] = w
            # wheel cut regions in glTF space for the renderer (inverse transform of centres)
            Minv = np.linalg.inv(M)
            entry["wheels"]["centres_gltf"] = [list((Minv @ np.r_[c, 1])[:3]) for c in w["centres"]]
            entry["wheels"]["radius_gltf"] = w["radius"] / s
            entry["wheels"]["width_gltf"] = w["width"] / s
    return entry


def _procedural_entry(pid: str, cfg: dict) -> dict:
    dx, dy, dz = cfg["dims"]
    if cfg["shape"] == "wedge":
        hull = [[-dx / 2, -dy / 2, 0], [dx / 2, -dy / 2, 0], [-dx / 2, dy / 2, 0], [dx / 2, dy / 2, 0],
                [-dx / 2, -dy / 2, dz], [-dx / 2, dy / 2, dz]]
    else:
        hull = None
    return {"id": pid, "name": cfg["name"], "aliases": cfg["aliases"], "kind": "procedural", "shape": cfg["shape"],
            "dims": cfg["dims"], "mass": cfg["mass"], "friction": cfg["friction"], "bouncy": cfg.get("bouncy", False),
            "static": cfg.get("static", False), "material": cfg["material"], "hull": hull,
            "license": "generated", "source": None, "authors": None, "thumb": None}


_manifest: dict | None = None


def manifest() -> dict[str, dict]:
    global _manifest
    if _manifest is not None:
        return _manifest
    stamp = sorted((p.name, p.stat().st_mtime) for p in LIB.glob("*/*") if p.is_file()) if LIB.exists() else []
    key = json.dumps([CURATED, PROCEDURAL, stamp], sort_keys=True, default=str)
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        if cached.get("key") == key:
            _manifest = cached["assets"]
            return _manifest
    out = {}
    for aid, cfg in CURATED.items():
        try:
            e = _build_one(aid, cfg)
            if e:
                out[aid] = e
        except Exception as ex:   # a broken download must not take the app down
            log.warning("asset %s skipped: %s", aid, ex)
    for pid, cfg in PROCEDURAL.items():
        out[pid] = _procedural_entry(pid, cfg)
    for e in GENERATED_DIR.glob("*/entry.json") if GENERATED_DIR.exists() else []:
        g = json.loads(e.read_text())
        out[g["id"]] = g
    CACHE.write_text(json.dumps({"key": key, "assets": out}))
    _manifest = out
    return out


def invalidate() -> None:
    global _manifest
    _manifest = None


GENERATED_DIR = config.DATA_DIR / "generated_assets"


def find(word: str) -> dict | None:
    """Resolve a user word ('car', 'duck', 'ball') to an asset."""
    w = word.lower().strip()
    best = None
    for a in manifest().values():
        for alias in [a["name"], *a["aliases"]]:
            if w == alias:
                return a
            if alias in w or (len(w) > 3 and w in alias):
                best = best or a
    return best
