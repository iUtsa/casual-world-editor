"""Camera intrinsics, back-projection, table-plane estimation and the world frame.

World frame convention: z points up out of the table, table top is z=0, the
x axis follows the camera's horizontal axis and y points away from the camera.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        fx = (width / 2.0) / np.tan(np.radians(hfov_deg) / 2.0)
        return cls(width, height, fx, fx, width / 2.0, height / 2.0, hfov_deg)


def backproject(depth: np.ndarray, K: Intrinsics) -> np.ndarray:
    """depth (H,W) z-depth in metres -> (H,W,3) camera-frame points."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x = (u - K.cx) * depth / K.fx
    y = (v - K.cy) * depth / K.fy
    return np.stack([x, y, depth], axis=-1)


@dataclass
class Plane:
    normal: np.ndarray   # unit, camera frame, pointing toward the camera side ("up")
    d: float             # n.p + d = 0
    inlier_mask: np.ndarray
    rms: float
    floor_offset: float | None = None   # measured distance down to a parallel floor plane
    room: bool = False                   # no work surface seen from above: whole-room mode (plane = floor/ground)
    structure: np.ndarray | None = None  # room mode: pixels on walls / floor / ceiling (not objects)

    def height(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


def normals_of(points: np.ndarray, span: int = 4) -> np.ndarray:
    P = points.astype(np.float32)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, span:-span] = P[:, 2 * span:] - P[:, :-2 * span]
    dy[span:-span] = P[2 * span:] - P[:-2 * span]
    n = np.cross(dy, dx)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n[n[..., 1] > 0] *= -1                       # orient toward camera-up
    return n


def surface_agreement(plane: "Plane", normals: np.ndarray) -> float:
    """Fraction of the plane's pixels whose own surface normal agrees with it. A real tabletop agrees;
    a band of wall pixels that happens to line up in noisy depth (normals pointing sideways) does not."""
    m = plane.inlier_mask
    if m.sum() < 50:
        return 0.0
    return float(((normals[m] @ plane.normal) > 0.8).mean())


def fit_table_plane(points: np.ndarray, valid: np.ndarray, rng_seed: int = 0) -> Plane:
    """The work surface, validated; when there is no surface seen from above (an eye-level view of a
    room), fall back to room mode instead of inventing a table out of wall pixels."""
    normals = normals_of(points)
    upish = valid & (-normals[..., 1] > 0.75)
    pl = None
    try:
        pl = _fit_surface(points, valid, rng_seed, prefer=upish if upish.sum() > 3000 else None)
    except ValueError:
        pl = None
    if pl is not None:
        agree = surface_agreement(pl, normals)
        if agree >= 0.45 and pl.inlier_mask.mean() >= 0.04:
            return pl
    return fit_room(points, valid, normals)


def fit_room(points: np.ndarray, valid: np.ndarray, normals: np.ndarray) -> Plane:
    """Room mode: 'up' from a level camera, corrected to be parallel to the biggest wall; walls/floor/
    ceiling are found as large planes and kept out of the object search; the ground is the floor if it is
    visible, else the lowest visible point."""
    h, w, _ = points.shape
    up = np.array([0.0, -1.0, 0.0])
    structure = np.zeros((h, w), bool)
    structure_had_wall = [False]
    remaining = valid.copy()
    rng = np.random.default_rng(0)
    tried = np.zeros((h, w), bool)
    allP = points[valid]
    for k in range(7):
        ys, xs = np.nonzero(remaining & ~tried)
        if len(ys) < 0.04 * h * w:
            break
        sel = rng.choice(len(ys), size=min(len(ys), 15000), replace=False)
        P = points[ys[sel], xs[sel]]
        thresh = max(0.02, 0.025 * float(np.median(P[:, 2])))
        best = None
        for _ in range(300):
            a, b, c = P[rng.choice(len(P), 3, replace=False)]
            n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            d = -float(n @ a)
            cnt = int((np.abs(P @ n + d) < thresh).sum())
            if best is None or cnt > best[2]:
                best = (n, d, cnt)
        if best is None:
            break
        n, d, _ = best
        if d < 0:                                   # orient so the camera (origin) is on the positive side
            n, d = -n, -d
        full = np.abs(points.reshape(-1, 3) @ n + d).reshape(h, w) < thresh
        mask = full & remaining
        tried |= mask
        big = mask.sum() > 0.04 * h * w
        agrees = ((normals[mask] @ n) ** 2 > 0.6).mean() > 0.4 if mask.any() else False
        # walls, floor and ceiling bound the room: (almost) nothing visible lies BEHIND them.
        # A desk front or cupboard has the wall behind it, so it stays an object.
        behind = float(((allP @ n + d) < -3 * thresh).mean())
        if big and agrees and behind < 0.03:
            structure |= mask
            remaining &= ~mask
            if abs(n[1]) < 0.4 and np.linalg.norm(up - (up @ n) * n) > 0.5 and not structure_had_wall[0]:
                u = up - (up @ n) * n                  # the first wall makes 'up' exactly vertical
                up = u / np.linalg.norm(u)
                structure_had_wall[0] = True
    up = up / np.linalg.norm(up)
    structure |= valid & ((normals @ up) < -0.75)        # ceiling: faces straight down
    hts = points[valid] @ up
    floorish = valid & ((normals @ up) > 0.85)
    if floorish.sum() > 0.02 * h * w:
        ground = float(np.percentile(points[floorish] @ up, 10))
    else:
        ground = float(np.percentile(hts, 1.0))
    inl = valid & (np.abs(points @ up - ground) < 0.03)
    return Plane(normal=up.astype(np.float64), d=-ground, inlier_mask=inl, rms=0.0, room=True, structure=structure)


def _fit_surface(points: np.ndarray, valid: np.ndarray, rng_seed: int = 0, prefer: np.ndarray | None = None) -> Plane:
    """Highest of up to three parallel upward planes (tabletop above the floor)."""
    h, w, _ = points.shape
    planes = []
    remaining = valid.copy()
    for k in range(3):
        try:
            pl = _fit_one_plane(points, remaining, rng_seed + k, prefer=None if prefer is None else prefer & remaining)
        except ValueError:
            break
        if planes and abs(float(pl.normal @ planes[0].normal)) < 0.985:
            remaining &= ~pl.inlier_mask
            continue
        if pl.inlier_mask.sum() < 0.05 * h * w:
            break
        planes.append(pl)
        remaining &= ~pl.inlier_mask
    if not planes:
        raise ValueError("no upward-facing plane found")
    n0 = planes[0].normal
    heights = [-(p.d if float(p.normal @ n0) > 0 else -p.d) for p in planes]
    best = int(np.argmax(heights))
    chosen = planes[best]
    lowest = min(heights)
    if heights[best] - lowest > 0.15:
        chosen.floor_offset = float(heights[best] - lowest)
    return chosen


def _fit_one_plane(points: np.ndarray, valid: np.ndarray, rng_seed: int = 0, prefer: np.ndarray | None = None) -> Plane:
    """RANSAC for the dominant upward-facing plane (the tabletop).

    Upward-facing means the normal has a strong component along camera -y,
    which rejects walls and the background even when they cover more pixels."""
    h, w, _ = points.shape
    rng = np.random.default_rng(rng_seed)
    if prefer is not None and prefer.sum() > 2000:
        ys, xs = np.nonzero(prefer)              # sample where the surface actually faces up
    else:
        ys, xs = np.nonzero(valid)
        # Bias sampling to the lower 70% of the frame where the tabletop lives.
        keep = ys > h * 0.3
        ys, xs = ys[keep], xs[keep]
    if len(ys) < 500:
        raise ValueError("not enough valid depth to find a table plane")
    sel = rng.choice(len(ys), size=min(len(ys), 20000), replace=False)
    P = points[ys[sel], xs[sel]]
    med_z = float(np.median(P[:, 2]))
    thresh = max(0.006, 0.018 * med_z)

    best_n, best_d, best_count = None, 0.0, -1
    for _ in range(400):
        i = rng.choice(len(P), 3, replace=False)
        a, b, c = P[i]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n[1] > 0:          # orient toward camera-up (-y)
            n = -n
        if -n[1] < 0.35:      # must be an upward-facing surface
            continue
        d = -float(n @ a)
        count = int((np.abs(P @ n + d) < thresh).sum())
        if count > best_count:
            best_n, best_d, best_count = n, d, count
    if best_n is None:
        raise ValueError("no upward-facing plane found; is the camera looking down at a table?")

    # Least-squares refinement on inliers (two rounds).
    n, d = best_n, best_d
    for _ in range(2):
        inl = P[np.abs(P @ n + d) < thresh]
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[2]
        if n[1] > 0:
            n = -n
        d = -float(n @ c)

    full_h = points.reshape(-1, 3) @ n + d
    inlier_mask = (np.abs(full_h).reshape(h, w) < thresh) & valid
    # keep the largest connected tabletop region
    num, lab, stats, _ = cv2.connectedComponentsWithStats(inlier_mask.astype(np.uint8), 8)
    if num > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inlier_mask = lab == biggest
    rms = float(np.sqrt(np.mean(full_h.reshape(h, w)[inlier_mask] ** 2)))
    return Plane(normal=n.astype(np.float64), d=float(d), inlier_mask=inlier_mask, rms=rms)


@dataclass
class WorldFrame:
    R_wc: np.ndarray   # world_from_camera rotation
    t_wc: np.ndarray   # camera position in world == translation of world_from_camera

    def to_world(self, pts_c: np.ndarray) -> np.ndarray:
        return pts_c @ self.R_wc.T + self.t_wc


def make_world_frame(plane: Plane, K: Intrinsics, points: np.ndarray) -> WorldFrame:
    n = plane.normal
    # Origin: where the optical axis meets the table (fallback: inlier centroid).
    ray = np.array([0.0, 0.0, 1.0])
    denom = n @ ray
    origin = None
    if abs(denom) > 1e-6:
        s = -plane.d / denom
        if 0.05 < s < 10:
            origin = ray * s
    if origin is None or plane.room:
        src = points[plane.inlier_mask] if plane.inlier_mask.sum() > 50 else points.reshape(-1, 3)[np.isfinite(points.reshape(-1, 3)).all(1)]
        origin = np.median(src, axis=0) if plane.room else src.mean(axis=0)
        origin = origin - (origin @ n + plane.d) * n
    x_c = np.array([1.0, 0.0, 0.0])
    x_w = x_c - (x_c @ n) * n
    x_w /= np.linalg.norm(x_w)
    y_w = np.cross(n, x_w)
    if y_w @ np.array([0.0, 0.0, 1.0]) < 0:   # y must point away from the camera
        y_w, x_w = -y_w, -x_w
    B = np.stack([x_w, y_w, n], axis=1)       # columns = world axes in camera coords
    R_wc = B.T
    t_wc = -R_wc @ origin
    return WorldFrame(R_wc=R_wc, t_wc=t_wc)


def table_extent(plane: Plane, frame: WorldFrame, points: np.ndarray, extend: bool = True) -> tuple[float, float, float, float]:
    """Visible table rectangle in world xy. Sides whose inliers touch the image
    border are extended: the table continues beyond what the camera sees."""
    m = plane.inlier_mask
    if m.sum() < 50:                              # room mode with no visible floor: start small, objects extend it
        return -0.5, 0.5, 0.5, 1.5
    pw = frame.to_world(points[m])
    x0, x1 = np.percentile(pw[:, 0], [1.5, 98.5])
    y0, y1 = np.percentile(pw[:, 1], [1.5, 98.5])
    h, w = m.shape
    band = 6
    ext = 0.6 if extend else 0.0
    if m[:, :band].mean() > 0.15:
        x0 -= ext
    if m[:, -band:].mean() > 0.15:
        x1 += ext
    if m[-band:, :].mean() > 0.15:
        y0 -= ext
    if m[:band, :].mean() > 0.15:
        y1 += ext
    return float(x0), float(x1), float(y0), float(y1)
