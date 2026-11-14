"""Lift 2D masks to physics primitives in the table frame and infer support.

A camera only sees the front and top of each object, so raw points under-
estimate depth and miss occluded bottoms. Rules applied (each recorded in the
object's `notes` so the UI can show what was assumed):
  * footprint = min-area rectangle of the object's points projected on the table
  * cylinders are made round; thin boxes are thickened away from the camera
  * bottom = top of the supporting object(s) under it, else the table; an object
    whose visible bottom floats with nothing under it was occluded -> extend to table
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Lifted:
    shape: str
    center_xy: np.ndarray
    dims_xy: np.ndarray          # full extents along local x/y
    yaw: float
    z_min_vis: float
    z_top: float
    bottom: float = 0.0
    supported_by: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    slice_r: list[tuple[float, float, int]] = field(default_factory=list)   # (z_world, radius, n_points)
    top_hidden: bool = False      # something rests on it and hides its top

    def profile(self, n: int = 14) -> list[list[float]] | None:
        """Radius profile [[z_from_bottom, r], ...] for revolved shapes, gaps filled,
        clamped so the visual mesh stays consistent with the physics footprint."""
        if self.shape != "cylinder" or len(self.slice_r) < 3:
            return None
        R = float(self.dims_xy[0]) / 2
        counts = np.array([c for _, _, c in self.slice_r])
        keep = counts >= 0.5 * np.median(counts)          # partially occluded slices under-measure width
        if self.top_hidden:                                # slices right under an occluder are cut by it
            zz = np.array([z for z, _, _ in self.slice_r])
            keep &= zz < self.z_min_vis + 0.55 * (zz.max() - self.z_min_vis)
        if keep.sum() < 2:
            return None
        zs = np.array([z for z, _, _ in self.slice_r])[keep] - self.bottom
        rs = np.array([r for _, r, _ in self.slice_r])[keep]
        rs = rs * (R / max(1e-6, rs.max()))              # edge erosion shrinks every slice alike: rescale to footprint
        grid = np.linspace(0, self.height, n)
        prof = np.interp(grid, zs, rs, left=rs[0], right=rs[-1])
        prof = np.convolve(np.pad(prof, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        prof = np.clip(prof, 0.3 * R, R)
        prof[0] = max(prof[0], 0.6 * prof[1])            # flat stable base
        return [[round(float(z), 4), round(float(r), 4)] for z, r in zip(grid, prof)]

    @property
    def height(self) -> float:
        return max(0.01, self.z_top - self.bottom)

    def footprint_poly(self) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        hx, hy = self.dims_xy / 2
        corners = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        R = np.array([[c, -s], [s, c]])
        return corners @ R.T + self.center_xy


def lift_mask(mask: np.ndarray, pts_world: np.ndarray, shape_hint: str | None,
              view_right: np.ndarray | None = None) -> Lifted | None:
    m = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if m.sum() < 30:
        m = mask
    P = pts_world[m]
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 20:
        return None
    # robust outlier rejection (flying pixels at depth edges)
    med = np.median(P, axis=0)
    mad = np.median(np.abs(P - med), axis=0) + 1e-4
    P = P[(np.abs(P - med) < 4.5 * mad * 1.4826 + 0.01).all(axis=1)]
    if len(P) < 20:
        return None
    # pixels at table height inside the mask are table/shadow bleeding into it
    hi = P[P[:, 2] > 0.008]
    if len(hi) > 0.3 * len(P) and len(hi) >= 20:
        P = hi
    z_top = float(np.percentile(P[:, 2], 97))
    z_min = float(np.percentile(P[:, 2], 3))
    xy = P[:, :2].astype(np.float32)
    (cx, cy), (w, d), ang = cv2.minAreaRect(xy * 1000.0)   # mm for numeric stability
    cx, cy, w, d = cx / 1000, cy / 1000, w / 1000, d / 1000
    yaw = np.radians(ang)
    notes = []
    shape = shape_hint
    if shape is None:
        shape, why = classify_shape(P, w, d, yaw)
        if why:
            notes.append(why)
    if shape == "cylinder":
        diam = max(w, d)
        # visible points are the front arc: true centre is further from the camera
        front_y = float(np.percentile(xy[:, 1], 3))
        cy = max(cy, front_y + diam / 2)
        w = d = diam
        yaw = 0.0
        notes.append("cylinder completed from visible front arc")
    else:
        # thicken along whichever local axis is closest to the view direction (+y)
        axis_y_local = 1 if abs(np.cos(yaw)) >= abs(np.sin(yaw)) else 0
        dims = [w, d]
        other = dims[1 - axis_y_local]
        vis_h = z_top - z_min
        # top barely visible (upright object seen from the side): assume roughly as deep as wide
        min_depth = 0.85 * other if vis_h > 1.2 * dims[axis_y_local] else 0.45 * other
        if dims[axis_y_local] < min_depth:
            grow = min_depth - dims[axis_y_local]
            dims[axis_y_local] = min_depth
            u = np.array([-np.sin(yaw), np.cos(yaw)]) if axis_y_local == 1 else np.array([np.cos(yaw), np.sin(yaw)])
            if u[1] < 0:
                u = -u
            cx, cy = cx + u[0] * grow / 2, cy + u[1] * grow / 2
            notes.append("hidden depth extended (back face not visible)")
        w, d = dims
    w, d = max(w, 0.012), max(d, 0.012)
    L = Lifted(shape, np.array([cx, cy]), np.array([w, d]), float(yaw), z_min, z_top, notes=notes)
    if shape == "cylinder":
        # silhouette half-width per height slice, measured across the camera's horizontal axis
        u = view_right[:2] / (np.linalg.norm(view_right[:2]) + 1e-9) if view_right is not None else np.array([1.0, 0.0])
        across = P[:, :2] @ u
        edges = np.linspace(z_min, z_top, 17)
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (P[:, 2] >= a) & (P[:, 2] < b)
            if sel.sum() >= 12:
                lo, hi = np.percentile(across[sel], [2, 98])
                L.slice_r.append(((a + b) / 2, float(hi - lo) / 2, int(sel.sum())))
        if L.slice_r:
            notes.append(f"revolved profile from {len(L.slice_r)} height slices")
    return L


def classify_shape(P: np.ndarray, w: float, d: float, yaw: float) -> tuple[str, str]:
    """Box or cylinder from the horizontal cross-section of the visible surface:
    a cylinder's front is a circular arc (depth varies quadratically across it,
    with radius ~ half its width); a box's visible faces are planar."""
    z = P[:, 2]
    lo, hi = np.percentile(z, [30, 70])
    band = P[(z > lo) & (z < hi)]
    width = max(w, d)
    if len(band) < 60 or width < 0.02:
        return "box", ""
    x, y = band[:, 0], band[:, 1]
    xc = x - x.mean()
    A1 = np.stack([xc, np.ones_like(xc)], 1)
    A2 = np.stack([xc ** 2, xc, np.ones_like(xc)], 1)
    r1 = np.linalg.lstsq(A1, y, rcond=None)[1]
    c2, r2 = np.linalg.lstsq(A2, y, rcond=None)[:2]
    if len(r1) == 0 or len(r2) == 0:
        return "box", ""
    a = c2[0]
    gain = 1 - float(r2[0]) / max(1e-12, float(r1[0]))
    if a <= 0:
        return "box", ""
    R = 1 / (2 * a)
    half = (x.max() - x.min()) / 2
    strong = gain > 0.35 and 0.3 * width / 2 < R < 2.5 * width / 2
    # noisy monocular depth: weaker curvature still counts if the arc radius matches the visible half-width
    consistent = gain > 0.15 and 0.6 * half < R < 1.4 * half
    if strong or consistent:
        return "cylinder", f"curved front (radius {R * 100:.1f} cm, fit gain {gain:.2f}) -> cylinder"
    return "box", ""


def _overlap_frac(a: Lifted, b: Lifted) -> float:
    pa = a.footprint_poly().astype(np.float32)
    pb = b.footprint_poly().astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    smaller = min(cv2.contourArea(pa), cv2.contourArea(pb))
    return float(inter) / max(1e-9, smaller)


def infer_support(objs: list[Lifted], table_tol: float = 0.025) -> None:
    """Bottom-up: rest each object on the table or on the highest object(s) under it.

    If an object floats above whatever is directly beneath it, that supporter's
    top was hidden behind the object (a mug under a book): extend it upward."""
    order = sorted(range(len(objs)), key=lambda i: objs[i].z_min_vis)
    placed: list[int] = []
    for i in order:
        o = objs[i]
        if o.z_min_vis < table_tol:
            o.bottom = 0.0
            placed.append(i)
            continue
        tol = max(0.03, 0.4 * (o.z_top - o.z_min_vis), 0.15 * o.z_min_vis)
        below = [j for j in placed if objs[j].z_top < o.z_min_vis + tol and objs[j].z_top < o.z_top - 0.01
                 and _overlap_frac(o, objs[j]) > 0.12]
        if below:
            top = max(objs[j].z_top for j in below)
            sup = [j for j in below if objs[j].z_top > top - 0.015]
            if o.z_min_vis - top > 0.006:
                for j in sup:
                    if objs[j].z_top < o.z_min_vis:
                        objs[j].notes.append(f"top hidden by object above; extended {100 * (o.z_min_vis - objs[j].z_top):.1f} cm")
                        objs[j].z_top = o.z_min_vis
                        objs[j].top_hidden = True
                o.bottom = o.z_min_vis
            else:
                o.bottom = top
            o.supported_by = sup
        else:
            o.bottom = 0.0
            o.notes.append("bottom occluded; extended down to the table")
        placed.append(i)
