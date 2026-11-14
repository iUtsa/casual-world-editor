"""Automatic object proposals restricted to things standing on the table.

Instead of SAM's dense "segment everything" grid (hundreds of decoder calls,
too slow on an 8 GB M1), prompts come from geometry:
  * "geometric blobs": above-table pixels split at depth discontinuities. Each
    blob is prompted with several positive points (whole-object hypothesis).
  * "appearance pieces": blobs further split at colour edges, one point each
    (catches objects that touch without a depth jump, e.g. a box on a book).
Cleaning then decides, for every mask that contains other masks, whether it is
one object with painted parts (keep the whole) or a stack of objects (keep the
parts). The test is physical: a stack has occlusion depth jumps inside it,
paint and labels do not."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .. import config
from .models import SamSession


DEBUG = False


@dataclass
class Proposal:
    mask: np.ndarray
    score: float
    seed: tuple[float, float]


def depth_jumps(pts_c: np.ndarray, thresh: float = 0.15, span: int = 2) -> np.ndarray:
    """Occlusion edges. Monocular depth smooths discontinuities into surfaces
    nearly parallel to the viewing ray ("veil points"); flag those. Unlike a raw
    depth-difference test this ignores steep-but-real surfaces (front faces,
    the far end of the table)."""
    P = pts_c.astype(np.float32)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, span:-span] = P[:, 2 * span:] - P[:, :-2 * span]
    dy[span:-span] = P[2 * span:] - P[:-2 * span]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    g = np.abs((n * ray).sum(-1))
    out = g < thresh
    out[:span + 1] = out[-span - 1:] = False
    out[:, :span + 1] = out[:, -span - 1:] = False
    return out


def above_table_mask(height: np.ndarray, pts_world: np.ndarray, table_rect, valid: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = table_rect
    m = 0.02
    on_table = (pts_world[..., 0] > x0 - m) & (pts_world[..., 0] < x1 + m) & \
               (pts_world[..., 1] > y0 - m) & (pts_world[..., 1] < y1 + m)
    above = (height > config.MIN_OBJECT_HEIGHT) & (height < 0.8) & valid & on_table
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    h, w = height.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(above, 8)
    keep = np.zeros(above.shape, bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= config.MIN_MASK_FRAC * h * w:
            keep |= lab == i
    return keep


def room_object_mask(height: np.ndarray, structure: np.ndarray | None, valid: np.ndarray) -> np.ndarray:
    """Room mode: everything that isn't wall, floor or ceiling and stands above the ground is a candidate."""
    st = cv2.dilate((structure if structure is not None else np.zeros(height.shape, bool)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    above = valid & ~st & (height > 0.03) & (height < 3.0)
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    h, w = height.shape
    num, lab, stats, _ = cv2.connectedComponentsWithStats(above, 8)
    keep = np.zeros(above.shape, bool)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= 3 * config.MIN_MASK_FRAC * h * w:
            keep |= lab == i
    return keep


def _spread_points(comp: np.ndarray, k: int) -> list[tuple[int, int]]:
    dist = cv2.distanceTransform(comp.astype(np.uint8), cv2.DIST_L2, 3)
    pts = []
    d = dist.copy()
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        if d[y, x] < 3:
            break
        pts.append((int(x), int(y)))
        r = max(6, int(dist[y, x] * 1.6))
        cv2.circle(d, (int(x), int(y)), r, 0, -1)
    return pts


def _prompts(pts_c, above, rgb, max_single: int = 36):
    h, w = above.shape
    jumps = cv2.dilate(depth_jumps(pts_c).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    min_area = max(60, int(0.0005 * h * w))
    blobs = (above & ~jumps).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(blobs, 4)
    multi = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            pts = _spread_points(lab == i, 3)
            if pts:
                multi.append(pts)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cedge = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8)) > 0
    pieces = (blobs.astype(bool) & ~cedge).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(pieces, 4)
    single = []
    for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[i, cv2.CC_STAT_AREA] < min_area or len(single) >= max_single:
            break
        single.extend(_spread_points(lab == i, 1))
    return multi, single, jumps


def propose(sam: SamSession, pts_c: np.ndarray, height: np.ndarray, above: np.ndarray, rgb: np.ndarray,
            pts_w: np.ndarray | None = None) -> list[Proposal]:
    h, w = above.shape
    multi, single, jumps = _prompts(pts_c, above, rgb)
    cands: list[Proposal] = []
    loose = cv2.dilate(above.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    # whole-object hypotheses: several positive points per geometric blob
    for pts in multi:
        logits, iou = sam.predict_multi(np.array(pts, np.float32))
        m = sam.upsample(logits[None])[0]
        cands.append(Proposal(m, float(iou), tuple(pts[0])))
    # part hypotheses: largest on-table mask among SAM's three per point
    if single:
        logits, iou = sam.predict_points(np.array(single, np.float32), chunk=16)
        small = cv2.resize(loose.astype(np.uint8), (logits.shape[-1], logits.shape[-2]), interpolation=cv2.INTER_NEAREST) > 0
        sel_l, sel_s, sel_p = [], [], []
        for i in range(len(single)):
            for k in np.argsort(-(logits[i] > 0).sum(dim=(1, 2)).numpy()):
                m = (logits[i, k] > 0).numpy()
                if m.sum() < 8 or iou[i, k] < 0.5:
                    continue
                if (m & small).sum() / m.sum() > 0.85:
                    sel_l.append(logits[i, k]); sel_s.append(float(iou[i, k])); sel_p.append(single[i])
                    break
        if sel_l:
            ms = sam.upsample(torch.stack(sel_l))
            cands += [Proposal(ms[i], sel_s[i], sel_p[i]) for i in range(len(ms))]
    props = clean_proposals(cands, above, height, jumps, pts_c[..., 2])
    return merge_convex_faces(props, pts_w, pts_c[..., 2]) if pts_w is not None else props


def _iou(a, b):
    return np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum())


def _relative_height(mask: np.ndarray, height: np.ndarray, above: np.ndarray) -> float:
    """Object top relative to the table ring right around it (robust to warped depth)."""
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0) & ~mask & ~above
    hv = height[mask]
    hv = hv[np.isfinite(hv)]
    if len(hv) == 0:
        return 0.0
    rv = height[ring]
    rv = rv[np.isfinite(rv)]
    ground = float(np.median(rv)) if len(rv) > 20 else 0.0
    return float(np.percentile(hv, 90)) - ground


def _occlusion_fraction(part: np.ndarray, container: np.ndarray, depth: np.ndarray) -> float:
    """Along the part's seam inside the container, how often is the container
    surface just outside the part clearly *behind* the part? That is what an
    object standing in front of / on top of another looks like; paint, labels
    and faces of the same object have no depth step across the seam."""
    inner_core = cv2.erode(part.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if inner_core.sum() < 20:
        return 0.0
    ring = (cv2.dilate(part.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~part
    ring &= cv2.erode(container.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if ring.sum() < 15:
        return 0.0
    src = (~inner_core).astype(np.uint8)          # zeros at inner-core pixels
    _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    core_yx = np.argwhere(inner_core)             # row-major == label order
    ys, xs = np.nonzero(ring)
    nearest = core_yx[lab[ys, xs] - 1]
    d_in = depth[nearest[:, 0], nearest[:, 1]]
    d_out = depth[ys, xs]
    occl = (d_out - d_in) > np.maximum(0.012, 0.02 * d_in)
    return float(occl.mean())


def clean_proposals(props, above, height, jumps, depth) -> list[Proposal]:
    h, w = above.shape
    min_area = config.MIN_MASK_FRAC * h * w
    jumps_d = cv2.dilate(jumps.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    good = []
    for p in props:
        a = p.mask.sum()
        if a < min_area or (p.mask & above).sum() / a < 0.6:
            continue
        if _relative_height(p.mask, height, above) < 1.5 * config.MIN_OBJECT_HEIGHT:
            continue          # a texture patch on a warped table, not an object
        good.append(p)
    # 1. dedupe near-identical masks
    good.sort(key=lambda p: -p.score)
    kept: list[Proposal] = []
    for p in good:
        if all(_iou(p.mask, k.mask) < 0.7 for k in kept):
            kept.append(p)
    # 2. containers: one object with painted parts, or several objects?
    kept.sort(key=lambda p: -p.mask.sum())
    alive = [True] * len(kept)
    for i, P in enumerate(kept):
        if not alive[i]:
            continue
        inside = [j for j, Q in enumerate(kept) if j != i and alive[j] and Q.mask.sum() < P.mask.sum()
                  and (Q.mask & P.mask).sum() / Q.mask.sum() > 0.8]
        if not inside:
            continue
        fr = {j: _occlusion_fraction(kept[j].mask, P.mask, depth) for j in inside}
        real = [j for j in inside if fr[j] > 0.10]
        union_all = np.any([kept[j].mask for j in inside], axis=0)
        rem = P.mask & ~union_all
        rem_h = height[rem]
        rem_h = rem_h[np.isfinite(rem_h)]
        # parts separated by bare table (two mugs + the gap between them) = union of objects
        # parts that tile the container, or are joined only by bare table, are separate objects
        # (two mugs + the gap between them). Known limitation: an object fully covered by
        # several label bands also tiles; it is rescued only if a part's seam shows occlusion.
        bridged_by_table = len(inside) >= 2 and (
            rem.sum() < 0.12 * P.mask.sum() or (len(rem_h) > 0 and np.median(rem_h) < 1.5 * config.MIN_OBJECT_HEIGHT))
        if DEBUG:
            print(f"container {i} area={P.mask.sum()} rem={rem.sum()} remh={np.median(rem_h) if len(rem_h) else -1:.3f} bridged={bridged_by_table} parts=" + ", ".join(f"{j}:{kept[j].mask.sum()}@{fr[j]:.2f}" for j in inside))
        if bridged_by_table and not real:
            alive[i] = False
            continue
        if not real:
            for j in inside:            # paint, labels, faces of one object
                alive[j] = False
            continue
        # P is several objects: keep the occluding parts, P becomes the remainder
        union = np.any([kept[j].mask for j in real], axis=0)
        rest = P.mask & ~cv2.dilate(union.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rest_h = height[rest]
        rest_h = rest_h[np.isfinite(rest_h)]
        if rest.sum() > max(min_area, 0.15 * P.mask.sum()) and len(rest_h) and np.median(rest_h) > 1.5 * config.MIN_OBJECT_HEIGHT:
            kept[i] = Proposal(rest, P.score, P.seed)
            for j in inside:
                if j not in real and (kept[j].mask & rest).sum() / kept[j].mask.sum() > 0.8:
                    alive[j] = False    # paint on the remainder
        else:
            alive[i] = False
    final = [p for i, p in enumerate(kept) if alive[i]]
    # 3. resolve overlaps: smaller masks win contested pixels
    final.sort(key=lambda p: p.mask.sum())
    taken = np.zeros_like(above)
    out = []
    for p in final:
        orig = p.mask.sum()
        m = p.mask & ~taken
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        num, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        if num > 2:
            m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        if m.sum() < 0.5 * orig or m.sum() < min_area:
            continue
        taken |= m
        out.append(Proposal(m, p.score, p.seed))
    return out


def surface_normals(pts: np.ndarray, span: int = 3) -> np.ndarray:
    P = pts.astype(np.float32)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, span:-span] = P[:, 2 * span:] - P[:, :-2 * span]
    dy[span:-span] = P[2 * span:] - P[:-2 * span]
    n = np.cross(dy, dx)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def seam_convexity(a: np.ndarray, b: np.ndarray, pts_w: np.ndarray, normals: np.ndarray,
                   depth: np.ndarray) -> tuple[int, float, float]:
    """LCCP-style test on the seam between two masks.
    Returns (seam length, fraction occluding, fraction convex). Faces of one
    object meet at convex creases; an object resting on another meets it at a
    concave crease; objects in front of others meet at occlusion steps."""
    k = np.ones((5, 5), np.uint8)
    near_b = cv2.dilate(b.astype(np.uint8), k) > 0
    near_a = cv2.dilate(a.astype(np.uint8), k) > 0
    sa = a & near_b
    sb = b & near_a
    n_seam = int(min(sa.sum(), sb.sum()))
    if n_seam < 12:
        return n_seam, 0.0, 0.0
    band_a = a & (cv2.dilate(b.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    band_b = b & (cv2.dilate(a.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
    def med(mask, arr):
        v = arr[mask]
        v = v[np.isfinite(v).all(axis=1)] if v.ndim == 2 else v[np.isfinite(v)]
        return np.median(v, axis=0) if len(v) else None
    pa, pb = med(band_a, pts_w), med(band_b, pts_w)
    na, nb = med(band_a, normals), med(band_b, normals)
    if pa is None or pb is None or na is None or nb is None:
        return n_seam, 0.0, 0.0
    # orient normals toward the camera side (upward / outward visible surfaces)
    dvec = pb - pa
    convex = float(np.dot(na, dvec) < 0.002 and np.dot(nb, -dvec) < 0.002)
    da, db = med(sa, depth), med(sb, depth)
    occl = float(abs(da - db) > max(0.012, 0.02 * min(da, db))) if da is not None and db is not None else 0.0
    return n_seam, occl, convex


def merge_convex_faces(props: list[Proposal], pts_w: np.ndarray, depth: np.ndarray) -> list[Proposal]:
    """Merge adjacent masks that are faces of one object (convex seam, no step)."""
    if len(props) < 2:
        return props
    normals = surface_normals(pts_w)
    # world normals point "out" of surfaces seen by the camera if they face it
    merged = True
    while merged:
        merged = False
        for i in range(len(props)):
            for j in range(i + 1, len(props)):
                n, occl, convex = seam_convexity(props[i].mask, props[j].mask, pts_w, normals, depth)
                if DEBUG and n >= 12:
                    print(f"seam {i}-{j}: len={n} occl={occl} convex={convex}")
                if n >= 25 and not occl and convex:
                    m = props[i].mask | props[j].mask
                    props[i] = Proposal(m, max(props[i].score, props[j].score), props[i].seed)
                    del props[j]
                    merged = True
                    break
            if merged:
                break
    return props


def mask_from_click(sam: SamSession, x: float, y: float):
    logits, iou = sam.predict_points(np.array([[x, y]], np.float32))
    k = int(np.argmax(iou[0].numpy()))
    m = sam.upsample(logits[0, k:k + 1])[0]
    if m.sum() < 30:
        return None
    return m, float(iou[0, k])


def local_ground(mask: np.ndarray, height: np.ndarray, others: np.ndarray) -> float:
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~mask & ~others
    rv = height[ring]
    rv = rv[np.isfinite(rv) & (np.abs(rv) < 0.05)]
    return float(np.median(rv)) if len(rv) > 30 else 0.0
