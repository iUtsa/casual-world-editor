"""Find the table and the objects in a sweep's reference view from multi-view geometry, not from a depth model.

  * table plane: RANSAC over the triangulated feature points (thousands of accurate 3D points, mostly on the table);
  * what sticks up: plane + parallax (Irani & Anandan). Every neighbouring view is warped onto the reference view
    through the table plane's homography; whatever lies ON the table lines up, whatever rises above it doesn't.
    The per-pixel median difference over several views (robust to occlusion in any single view) marks the objects;
  * outlines: SAM 2.1 prompted with points inside each region, de-duplicated; a mask is kept when it sits over the
    table and the region's pixels really rise above the plane;
  * starting boxes: the triangulated points that fall inside each outline.
Coordinates: the reference camera's frame (OpenCV), metres in the sweep's (depth-model) scale.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class TablePlane:
    n: np.ndarray        # unit normal, pointing to the camera's side ("up")
    d: float             # n . X + d = 0
    inliers: np.ndarray  # bool over the input points
    rms: float


def fit_plane(X: np.ndarray, iters: int = 600, seed: int = 0) -> TablePlane:
    """The table: the plane with the most triangulated points, refined on its inliers."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    depth = np.linalg.norm(X, axis=1)
    tau = max(0.002, 0.006 * float(np.median(depth)))       # relative: the sweep's scale is not yet trusted
    best, best_n = None, -1
    for _ in range(iters):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ p[0]
        cnt = int((np.abs(X @ n + d) < tau).sum())
        if cnt > best_n:
            best, best_n = (n, d), cnt
    n, d = best
    inl = np.abs(X @ n + d) < tau
    for _ in range(3):                                       # least-squares refit on the inliers
        c = X[inl].mean(0)
        n = np.linalg.svd(X[inl] - c)[2][-1]
        d = -n @ c
        inl = np.abs(X @ n + d) < tau
    if d < 0:                                                # the camera (origin) is on the positive ("up") side
        n, d = -n, -d
    rms = float(np.sqrt(np.mean((X[inl] @ n + d) ** 2)))
    return TablePlane(n, float(d), inl, rms)


def plane_homography(K: np.ndarray, R_kr: np.ndarray, t_kr: np.ndarray, pl: TablePlane) -> np.ndarray:
    """Pixel in the reference view -> pixel in view k, for points on the plane (X_k = R_kr X_r + t_kr)."""
    H = K @ (R_kr - np.outer(t_kr, pl.n) / pl.d) @ np.linalg.inv(K)
    return H / H[2, 2]


def parallax_map(frames: list[np.ndarray], poses, K: np.ndarray, ref: int, pl: TablePlane,
                 n_views: int = 8, max_angle_deg: float = 35.0) -> np.ndarray:
    """Per-pixel median photometric difference after warping neighbours onto the reference through the plane.
    poses: (R_wc, c) of every frame in the reference camera's frame."""
    h, w = frames[ref].shape[:2]
    gref = cv2.GaussianBlur(cv2.cvtColor(frames[ref], cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 1.2)
    fwd = poses[ref][0][:, 2]
    cand = []
    for k, (R, c) in enumerate(poses):
        if k == ref:
            continue
        ang = np.degrees(np.arccos(np.clip(R[:, 2] @ fwd, -1, 1)))
        base = np.linalg.norm(c - poses[ref][1])
        if 6.0 < ang < max_angle_deg and base > 1e-6:
            cand.append((abs(ang - 18.0), k))                # prefer ~18 deg: clear parallax, little occlusion
    picks = [k for _, k in sorted(cand)[:n_views]]
    res = []
    for k in picks:
        R_wk, c_k = poses[k]
        R_kr = R_wk.T                                        # ref -> k (ref frame is the world here)
        t_kr = -R_wk.T @ c_k
        H = plane_homography(K, R_kr, t_kr, pl)
        g = cv2.GaussianBlur(cv2.cvtColor(frames[k], cv2.COLOR_RGB2GRAY).astype(np.float32), (5, 5), 1.2)
        warped = cv2.warpPerspective(g, H, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        # exposure can differ between frames: match the warped view's brightness to the reference locally
        valid = np.isfinite(warped)
        wv = np.where(valid, warped, 0)
        mu_r = cv2.blur(gref, (41, 41))
        mu_w = cv2.blur(wv, (41, 41)) / np.maximum(cv2.blur(valid.astype(np.float32), (41, 41)), 1e-3)
        d = np.abs((wv - mu_w) - (gref - mu_r))
        d[~valid] = np.nan
        res.append(d)
    if not res:
        return np.zeros((h, w), np.float32)
    with np.errstate(all="ignore"):
        return np.nan_to_num(np.nanmedian(np.stack(res), 0), nan=0.0).astype(np.float32)


def table_region(pl: TablePlane, X: np.ndarray, K: np.ndarray, hw) -> np.ndarray:
    """Reference-view pixels over the table: the convex hull of the plane's inlier points, projected."""
    h, w = hw
    P = X[pl.inliers]
    P = P[P[:, 2] > 0.05]
    m = np.zeros((h, w), np.uint8)
    if len(P) < 10:
        m[:] = 1
        return m > 0
    u = K[0, 0] * P[:, 0] / P[:, 2] + K[0, 2]
    v = K[1, 1] * P[:, 1] / P[:, 2] + K[1, 2]
    pts = np.stack([u, v], 1).astype(np.float32)
    hull = cv2.convexHull(pts)
    cv2.fillConvexPoly(m, hull.astype(np.int32), 1)
    return cv2.dilate(m, np.ones((25, 25), np.uint8)) > 0


def find_objects(rgb: np.ndarray, par: np.ndarray, table: np.ndarray, sam_session, thresh: float | None = None,
                 min_area_frac: float = 0.0015) -> list[np.ndarray]:
    """Object masks in the reference view from the parallax map: regions that rise above the plane, outlined by SAM.
    Prompts sit at the centres of the blobs (peaks of the distance transform), so objects that touch are prompted
    separately; for each prompt the best of SAM's three candidate masks is used."""
    from scipy import ndimage
    h, w = par.shape
    p = par.copy()
    p[~table] = 0
    if thresh is None:
        vals = p[table & (p > 0)]
        thresh = max(10.0, float(np.percentile(vals, 85))) if len(vals) else 12.0
    mask = (p > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = ndimage.binary_fill_holes(mask)
    min_area = min_area_frac * h * w
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1)) if n else np.zeros(0)
    big = 1 + np.flatnonzero(sizes >= 0.3 * min_area)
    keep = np.isin(lab, big)
    dist = ndimage.distance_transform_edt(keep)
    peaks = (dist == ndimage.maximum_filter(dist, size=31)) & (dist > 5)
    cand = []

    point_masks: list[np.ndarray] = []

    def consider(m_, score, from_box=False):
        area = m_.sum()
        if area < min_area or area > 0.25 * h * w or score < 0.6:
            return
        if (m_ & keep).sum() / area < 0.4 or (m_ & table).sum() / area < 0.6:
            return                                       # mostly flat table or background: not an object
        cand.append((score, m_, from_box))
        if not from_box:
            point_masks.append(m_)

    # 1. one box per raised region (+ its most central points): SAM sees the whole object's extent
    boxes, pos, neg = [], [], []
    for i in big:
        comp = lab == i
        ys, xs = np.nonzero(comp)
        mx, my = 0.04 * (xs.max() - xs.min()) + 3, 0.04 * (ys.max() - ys.min()) + 3
        boxes.append([max(0, xs.min() - mx), max(0, ys.min() - my), min(w - 1, xs.max() + mx), min(h - 1, ys.max() + my)])
        cy, cx = np.nonzero(peaks & comp)
        o_ = np.argsort(-dist[cy, cx])[:2]
        pos.append([(float(cx[j]), float(cy[j])) for j in o_])
        neg.append([])
    if boxes:
        lg, sc = sam_session.predict_prompts(boxes, pos, neg)
        ms = sam_session.upsample(lg)
        for m_, s_ in zip(ms, sc.tolist()):
            consider(m_, float(s_) + 0.5, from_box=True)  # whole-region prompts are preferred over single points
    # 2. single points at blob centres: separates objects whose regions touch
    py, px = np.nonzero(peaks)
    order = np.argsort(-dist[py, px])
    pts = np.stack([px[order], py[order]], 1).astype(float)[:40]
    if len(pts):
        logits, iou = sam_session.predict_points(pts)
        for j in range(len(pts)):
            k = int(np.argmax(iou[j]))
            consider(sam_session.upsample(logits[j:j + 1, k])[0],
                     float(iou[j, k]) * min(1.0, dist[int(pts[j, 1]), int(pts[j, 0])] / 15))
    def distinct_inside(b):
        """How many separate single-point objects a mask swallows (2+ = several objects merged into one box)."""
        inner = [p for p in point_masks if (p & b).sum() / p.sum() > 0.7 and p.sum() < 0.85 * b.sum()]
        picked: list[np.ndarray] = []
        for p in sorted(inner, key=lambda x: -x.sum()):
            if all((p & q).sum() / min(p.sum(), q.sum()) < 0.3 for q in picked):
                picked.append(p)
        return len(picked)

    out: list[np.ndarray] = []
    for _, m_, from_box in sorted(cand, key=lambda x: -x[0]):
        if from_box and distinct_inside(m_) >= 2:
            continue                                     # a box that spans several objects
        if any((m_ & o).sum() / min(m_.sum(), o.sum()) > 0.5 for o in out):
            continue                                     # the same object again
        out.append(m_)
    return out


def boxes_from_points(masks: list[np.ndarray], X_world: np.ndarray, uv: np.ndarray, frame_R: np.ndarray) -> list:
    """(centre, half extents, yaw) per mask from the triangulated points inside it (world frame, z up)."""
    out = []
    for m in masks:
        h, w = m.shape
        u, v = uv[:, 0].astype(int), uv[:, 1].astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        sel = np.zeros(len(uv), bool)
        sel[ok] = m[v[ok], u[ok]]
        P = X_world[sel]
        P = P[P[:, 2] > 0.004]
        if len(P) < 4:
            out.append(None)
            continue
        (cx, cy), (bw, bd), ang = cv2.minAreaRect((P[:, :2] * 1000).astype(np.float32))
        top = float(np.percentile(P[:, 2], 95))
        half = np.array([max(bw, 20) / 2000 + 0.01, max(bd, 20) / 2000 + 0.01, top / 2 + 0.005])
        out.append((np.array([cx / 1000, cy / 1000, top / 2]), half, float(np.radians(ang))))
    return out
