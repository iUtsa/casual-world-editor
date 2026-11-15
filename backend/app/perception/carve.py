"""Objects in 3D from many views: shape by carving, surfaces textured with the real pixels.

Given camera poses and, per view, a map of which object every pixel shows:

  1. union visual hull: a voxel survives only if every camera that sees it sees *some* object there. Using the
     union of all object masks means objects hiding each other never carve each other away.
  2. labelling: each camera's first visible voxel along every pixel ray gets a vote for the object that pixel
     shows; the votes label the hull's surface, and inner voxels take the label of the nearest surface voxel.
  3. per object: marching cubes on its voxels (closed at the table, which is z = 0) gives a watertight mesh.
  4. texture: every triangle is painted from the view that sees it most squarely, with the original pixels
     (a texture atlas), so labels and print stay sharp instead of being averaged into a smear.

World frame: z up, table top at z = 0. Poses are (R_wc, t_wc): world-from-camera rotation and camera centre, with
OpenCV camera axes (x right, y down, z forward).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Grid:
    origin: np.ndarray       # world position of voxel (0,0,0) centre
    voxel: float
    shape: tuple[int, int, int]

    def centres(self, idx: np.ndarray | None = None) -> np.ndarray:
        if idx is None:
            idx = np.indices(self.shape).reshape(3, -1).T
        return self.origin + idx * self.voxel


def project(P: np.ndarray, pose, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World points -> pixel (u, v) and camera depth z."""
    R, t = pose
    Xc = (P - t) @ R
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    return u, v, z


def _carve(grid: Grid, fg: list[np.ndarray], poses, K, tol_frac: float, min_seen: int,
           depths: list[np.ndarray] | None = None, margin: tuple[float, float] = (0.01, 0.03),
           free_frac: float = 0.3) -> np.ndarray:
    """A voxel gets an 'empty' vote from a camera when its pixel shows no object, or when the camera measured a
    surface FURTHER away along that pixel (the voxel is in the free space in front of it). margin = (absolute m,
    fraction of depth) of slack for depth noise. Silhouettes are exact, so tol_frac of them may disagree; depth
    from a monocular model can be wrong on a whole object in one view, so a voxel is only freed when free_frac of
    the cameras that see it agree."""
    P = grid.centres()
    out = np.zeros(len(P), np.int16)
    free = np.zeros(len(P), np.int16)
    seen = np.zeros(len(P), np.int16)
    for k, (m, pose) in enumerate(zip(fg, poses)):
        h, w = m.shape
        u, v, z = project(P, pose, K)
        ok = (z > 0.05) & (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        ui, vi = u[ok].astype(np.int32), v[ok].astype(np.int32)
        inside = m[vi, ui]
        if depths is not None and depths[k] is not None:
            D = depths[k]
            dh, dw = D.shape
            d = D[(vi * dh // h).clip(0, dh - 1), (ui * dw // w).clip(0, dw - 1)]
            zz = z[ok]
            with np.errstate(invalid="ignore"):
                fr = inside & np.isfinite(d) & (zz < d - np.maximum(margin[0], margin[1] * d))
            free[np.flatnonzero(ok)[fr]] += 1
        seen[ok] += 1
        out[np.flatnonzero(ok)[~inside]] += 1
    keep = (seen >= min_seen) & (out <= np.maximum(0, np.floor(tol_frac * seen))) & (free < np.maximum(2, free_frac * seen))
    return keep.reshape(grid.shape)


def union_hull(fg_masks: list[np.ndarray], poses, K, bounds: tuple[np.ndarray, np.ndarray],
               voxel: float = 0.003, coarse: float = 0.01, tol_frac: float = 0.0, dilate_px: int = 2,
               min_seen: int = 3, depths: list[np.ndarray] | None = None,
               margin: tuple[float, float] = (0.01, 0.03), unknown: list[np.ndarray] | None = None) -> tuple[Grid, np.ndarray]:
    """Two passes: 1 cm over the whole table, then `voxel` inside the dilated coarse hull's bounding box."""
    if dilate_px:
        k = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        fg_masks = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in fg_masks]
    if unknown is not None:                      # "an object should be here but its mask failed": not evidence of space
        fg_masks = [m | u for m, u in zip(fg_masks, unknown)]
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    g0 = Grid(lo, coarse, tuple(np.maximum(1, np.ceil((hi - lo) / coarse).astype(int))))
    occ0 = _carve(g0, fg_masks, poses, K, tol_frac, min_seen, depths, (margin[0] + coarse, margin[1]))
    if not occ0.any():
        return g0, occ0
    occ0 = ndimage.binary_dilation(occ0, iterations=2)
    idx = np.argwhere(occ0)
    a = g0.origin + (idx.min(0) - 0.5) * coarse
    b = g0.origin + (idx.max(0) + 0.5) * coarse
    a[2] = max(a[2], lo[2])
    g = Grid(a + voxel / 2, voxel, tuple(np.maximum(1, np.ceil((b - a) / voxel).astype(int))))
    return g, _carve(g, fg_masks, poses, K, tol_frac, min_seen, depths, margin)


def zbuffer(P: np.ndarray, pose, K: np.ndarray, hw: tuple[int, int], voxel: float, down: int = 2) -> tuple[np.ndarray, tuple]:
    """Nearest occupied-voxel depth per (downsampled) pixel; each voxel is splatted to its projected size."""
    h, w = hw[0] // down, hw[1] // down
    u, v, z = project(P, pose, K)
    u, v = u / down, v / down
    ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zb = np.full((h, w), np.inf, np.float32)
    if not ok.any():
        return zb, (u, v, z, ok)
    r = np.clip(np.ceil(0.5 * K[0, 0] / down * voxel / z[ok]).astype(int), 0, 4)
    ui, vi, zi = u[ok].astype(int), v[ok].astype(int), z[ok].astype(np.float32)
    for rr in np.unique(r):
        s = r == rr
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                uu, vv = np.clip(ui[s] + dx, 0, w - 1), np.clip(vi[s] + dy, 0, h - 1)
                np.minimum.at(zb, (vv, uu), zi[s])
    return zb, (u, v, z, ok)


def _surface_votes(P: np.ndarray, id_maps, poses, K, voxel: float, n_obj: int, view_ok=None):
    """For each voxel: per-label counts over the views where it is the first thing along its pixel ray."""
    votes = np.zeros((len(P), n_obj + 1), np.float32)
    zbs = []
    for k, (ids, pose) in enumerate(zip(id_maps, poses)):
        zb, (u, v, z, ok) = zbuffer(P, pose, K, ids.shape, voxel)
        zbs.append(zb)
        sel = np.flatnonzero(ok)
        ui, vi = u[sel].astype(int), v[sel].astype(int)
        vis = z[sel] <= zb[vi, ui] + 1.5 * voxel
        sel, ui, vi = sel[vis], ui[vis], vi[vis]
        lab = ids[np.clip(vi * 2, 0, ids.shape[0] - 1), np.clip(ui * 2, 0, ids.shape[1] - 1)].astype(int)
        if view_ok is not None:
            good = view_ok[k][lab] | (lab == 0)
            sel, lab = sel[good], lab[good]
        np.add.at(votes, (sel, lab), 1.0)
    return votes, zbs


def consistent_carve(grid: Grid, occ: np.ndarray, id_maps, poses, K, n_obj: int, view_ok=None,
                     min_agree: float = 0.6, max_iters: int = 40) -> np.ndarray:
    """Space carving with object labels (Kutulakos & Seitz 2000, labels in place of colours): a voxel on the
    current surface must show the same object to the cameras that see it. Voxels that look like different things
    from different sides are empty space between objects (\"ghosts\" of the union hull), so they are removed,
    which exposes the next layer; repeat until nothing changes."""
    occ = occ.copy()
    for _ in range(max_iters):
        idx = np.argwhere(occ)
        P = grid.centres(idx)
        votes, _ = _surface_votes(P, id_maps, poses, K, grid.voxel, n_obj, view_ok)
        tot = votes.sum(1)
        best = votes[:, 1:].max(1) if n_obj else np.zeros(len(P))
        bad = (tot >= 2) & ((best / np.maximum(tot, 1) < min_agree) | (votes[:, 0] >= 0.5 * tot))
        if not bad.any() or bad.sum() < 0.002 * len(P):      # converged: what's left is boundary noise
            break
        occ[tuple(idx[bad].T)] = False
    return occ


def label_voxels(grid: Grid, occ: np.ndarray, id_maps: list[np.ndarray], poses, K, n_obj: int,
                 view_ok: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Object id per occupied voxel (0 = none). Also returns each view's z-buffer of the hull (for texturing).

    view_ok[k][i] False excludes object i's votes from view k (its mask in that view was judged unreliable)."""
    idx = np.argwhere(occ)
    P = grid.centres(idx)
    votes, zbs = _surface_votes(P, id_maps, poses, K, grid.voxel, n_obj, view_ok)
    votes[:, 0] = 0
    surf = votes.max(1) > 0
    lab = np.zeros(len(P), np.int16)
    lab[surf] = votes[surf].argmax(1)
    L = np.zeros(grid.shape, np.int16)
    L[tuple(idx.T)] = lab
    # inner voxels: the label of the nearest labelled surface voxel
    unl = occ & (L == 0)
    if unl.any() and (L > 0).any():
        _, nearest = ndimage.distance_transform_edt(L == 0, return_indices=True)
        fill = L[tuple(nearest)]
        L[unl] = fill[unl]
    # Space no camera ever saw (under a book, behind two mugs) can't be carved; without a limit it would be given
    # to the nearest object. Limit each object to the convex hull of the surface the cameras SAW belonging to it:
    # concavities that were seen empty stay carved, unseen space stops inflating the object.
    tot = votes.sum(1)
    seen = (votes[:, 1:].max(1) >= 2) & (votes[:, 1:].max(1) >= 0.75 * np.maximum(tot, 1))
    for i in range(1, n_obj + 1):
        obs = idx[seen & (lab == i)]
        mine = np.argwhere(L == i)
        if len(obs) < 12 or not len(mine):
            continue
        # stray votes at the border between two objects would stretch the hull: keep the main connected piece
        g = np.zeros(grid.shape, bool)
        g[tuple(obs.T)] = True
        comp, nc = ndimage.label(ndimage.binary_dilation(g, iterations=1), structure=np.ones((3, 3, 3)))
        if nc > 1:
            sizes = ndimage.sum(g, comp, range(1, nc + 1))
            obs = obs[comp[tuple(obs.T)] == 1 + int(np.argmax(sizes))]
        inside = in_hull(grid.centres(mine), grid.centres(obs), pad=1.0 * grid.voxel)
        drop = mine[~inside]
        L[tuple(drop.T)] = 0
    return L, zbs


def in_hull(Q: np.ndarray, pts: np.ndarray, pad: float = 0.0) -> np.ndarray:
    from scipy.spatial import ConvexHull, QhullError
    try:
        h = ConvexHull(pts)
    except (QhullError, ValueError):
        return np.ones(len(Q), bool)
    A, b = h.equations[:, :3], h.equations[:, 3]
    return ((Q @ A.T + b) <= pad).all(1)


def object_mesh(mask: np.ndarray, grid: Grid, smooth: float = 1.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Watertight mesh (world coordinates) of one object's voxels, closed flat where it meets the table."""
    from skimage import measure
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    if n > 1:                                              # keep the main piece; specks are carving noise
        sizes = ndimage.sum(mask, lab, range(1, n + 1))
        mask = lab == (1 + int(np.argmax(sizes)))
    vol = np.pad(mask.astype(np.float32), 2)
    vol[:, :, :2] = 0
    vol = ndimage.gaussian_filter(vol, smooth) if smooth else vol
    try:
        verts, faces, _, _ = measure.marching_cubes(vol, 0.5)
    except (ValueError, RuntimeError):
        return None
    verts = grid.origin + (verts - 2) * grid.voxel
    verts[:, 2] = np.maximum(verts[:, 2], 0.0)
    return verts.astype(np.float32), faces.astype(np.int32)


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n = np.zeros_like(V)
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def texture_atlas(V: np.ndarray, F: np.ndarray, oid: int, frames: list[np.ndarray], id_maps: list[np.ndarray],
                  zbs: list[np.ndarray], poses, K, cell: int = 12, voxel: float = 0.003,
                  view_ok: list[np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-triangle best view -> texture atlas. Returns (V', F', UV, atlas RGB); vertices are split per triangle
    so each triangle owns its UVs (the seams between views are where the source view changes)."""
    nT = len(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    C = V[F].mean(1)
    best = np.full(nT, -1)
    score = np.zeros(nT)
    uvs_all = []
    for k, (img, ids, zb, pose) in enumerate(zip(frames, id_maps, zbs, poses)):
        if view_ok is not None and not view_ok[k][oid]:
            uvs_all.append(None)
            continue
        R, t = pose
        h, w = ids.shape
        u, v, z = project(V[F].reshape(-1, 3), pose, K)
        u, v, z = u.reshape(nT, 3), v.reshape(nT, 3), z.reshape(nT, 3)
        uc, vc, zc = project(C, pose, K)
        ok = (z > 0.05).all(1) & (u >= 1).all(1) & (u < w - 2).all(1) & (v >= 1).all(1) & (v < h - 2).all(1)
        d = t - C
        dist = np.linalg.norm(d, axis=1)
        cos = (fn * d).sum(1) / (dist + 1e-9)
        ok &= cos > 0.15
        ui = np.clip((uc / 2).astype(int), 0, zb.shape[1] - 1)
        vi = np.clip((vc / 2).astype(int), 0, zb.shape[0] - 1)
        ok &= zc <= zb[vi, ui] + 2.5 * voxel                    # not hidden behind another object
        ok &= ids[np.clip(vc.astype(int), 0, h - 1), np.clip(uc.astype(int), 0, w - 1)] == oid
        sharp_px = K[0, 0] / np.maximum(zc, 0.05)              # pixels per metre: closer views are sharper
        s = np.where(ok, cos ** 2 * sharp_px, 0)
        better = s > score
        score[better], best[better] = s[better], k
        uvs_all.append(np.stack([u, v], -1))
    # atlas: two triangles per square cell (upper-left / lower-right halves), 1 px padding
    pairs = (nT + 1) // 2
    cols = int(np.ceil(np.sqrt(pairs)))
    rows = int(np.ceil(pairs / cols))
    S = cell
    atlas = np.zeros((rows * S, cols * S, 3), np.uint8)
    UV = np.zeros((nT, 3, 2), np.float32)
    mean_col = np.median(np.concatenate([f.reshape(-1, 3)[::97] for f in frames]), 0)
    for i in range(nT):
        c, half = divmod(i, 2)
        r_, c_ = divmod(c, cols)
        x0, y0 = c_ * S, r_ * S
        if half == 0:
            dst = np.float32([[x0 + 1, y0 + 1], [x0 + S - 2, y0 + 1], [x0 + 1, y0 + S - 2]])
        else:
            dst = np.float32([[x0 + S - 1, y0 + S - 1], [x0 + 2, y0 + S - 1], [x0 + S - 1, y0 + 2]])
        UV[i] = dst
        k = best[i]
        if k < 0:
            cv2.fillConvexPoly(atlas, dst.astype(np.int32), tuple(int(x) for x in mean_col))
            continue
        src = uvs_all[k][i].astype(np.float32)
        M = cv2.getAffineTransform(src, dst)
        bx0, by0 = int(max(0, np.floor(dst[:, 0].min()) - 1)), int(max(0, np.floor(dst[:, 1].min()) - 1))
        # warp only the cell: shift the transform into cell coordinates
        M2 = M.copy()
        M2[:, 2] -= [bx0, by0]
        w_, h_ = min(S + 2, atlas.shape[1] - bx0), min(S + 2, atlas.shape[0] - by0)
        patch = cv2.warpAffine(frames[k], M2, (w_, h_), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m = np.zeros((h_, w_), np.uint8)
        cv2.fillConvexPoly(m, (dst - [bx0, by0]).round().astype(np.int32), 1)
        m = cv2.dilate(m, np.ones((3, 3), np.uint8)) > 0
        region = atlas[by0:by0 + h_, bx0:bx0 + w_]
        region[m] = patch[m]
    # each triangle gets its own three vertices
    V2 = V[F].reshape(-1, 3)
    F2 = np.arange(nT * 3, dtype=np.int32).reshape(nT, 3)
    UVn = UV.reshape(-1, 2) / [atlas.shape[1], atlas.shape[0]]
    return V2, F2, UVn.astype(np.float32), atlas
