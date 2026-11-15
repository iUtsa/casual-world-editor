"""Every scanned object's mask in every frame of a sweep, guided by geometry instead of video memory.

The cameras are already solved, so each object's 3D extent can be projected into any frame: that gives SAM 2.1
(single-image mode, one encoder pass per frame, constant memory) a box and points for exactly that object. Two
rounds:
  1. rough: the object's 3D box from the reference-frame scan -> box + centre point;
  2. tight: after a first carve, the object's own carved voxels that are visible in that frame -> positive
     points, nearby objects' visible voxels -> negative points.
Overlaps are resolved by depth (a pixel belongs to the nearer object). Each (frame, object) mask is checked
against the geometric prediction; one that disagrees badly is marked unreliable so the carver skips it."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .carve import project, zbuffer
from .models import SamSession


def box_corners(center, half, yaw) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    g = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    return (g * half) @ R.T + center


def _pred_from_points(P: np.ndarray, pose, K, hw, voxel: float) -> tuple[np.ndarray, float] | None:
    """Predicted silhouette (bool, H/4 x W/4) and mean depth of a 3D point set seen from `pose`."""
    h, w = hw
    u, v, z = project(P, pose, K)
    ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if ok.sum() < 3:
        return None
    m = np.zeros((h // 4 + 1, w // 4 + 1), np.uint8)
    r = max(1, int(np.ceil(0.5 * K[0, 0] * voxel / np.median(z[ok]) / 4)) + 1)
    m[(v[ok] / 4).astype(int), (u[ok] / 4).astype(int)] = 1          # mark, then grow every mark at once
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return m > 0, float(np.median(z[ok]))


def masks(frames: list[np.ndarray], poses, K, objects: list[dict], ref: int, ref_masks: list[np.ndarray],
          progress=lambda k, n: None, unknown_out: list | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """objects[i]: {"points": (M,3) world points of object i (round 2) or None, "box": (center, half, yaw)}.
    Returns per-frame id maps (pixel = object number 1..n) and per-frame reliability flags (index 0 unused).
    If unknown_out is a list, it receives per frame a bool map of pixels where geometry says an object should be
    but its mask is missing or unreliable: the carver must not read those as empty space."""
    n = len(objects)
    h, w = frames[0].shape[:2]
    ids, ok = [], []
    # visibility of each object's points in each frame comes from one z-buffer over all objects
    allP = [o["points"] if o.get("points") is not None else box_corners(*o["box"]) for o in objects]
    for k, img in enumerate(frames):
        if k == ref:
            idm = np.zeros((h, w), np.uint8)
            for i, m in enumerate(ref_masks, 1):
                idm[m.astype(bool)] = i
            ids.append(idm)
            ok.append(np.ones(n + 1, bool))
            if unknown_out is not None:
                unknown_out.append(np.zeros((h, w), bool))
            progress(k + 1, len(frames))
            continue
        pose = poses[k]
        boxes, pos, neg, order, preds, depth = [], [], [], [], [], []
        if any(o.get("points") is not None for o in objects):
            cat = np.concatenate([p for p in allP])
            owner = np.concatenate([np.full(len(p), i) for i, p in enumerate(allP)])
            zb, (uh, vh, z, inb) = zbuffer(cat, pose, K, (h, w), 0.004)       # uh, vh: half-resolution
            u, v = uh * 2, vh * 2                                             # full-resolution pixels
            vis = np.zeros(len(cat), bool)
            sel = np.flatnonzero(inb)
            vis[sel] = z[sel] <= zb[vh[sel].astype(int), uh[sel].astype(int)] + 0.008
        for i, o in enumerate(objects):
            P = allP[i]
            u_, v_, z_ = project(P, pose, K)
            front = z_ > 0.05
            if front.sum() < 3:
                continue
            x0, x1 = np.clip(u_[front].min(), 0, w - 1), np.clip(u_[front].max(), 0, w - 1)
            y0, y1 = np.clip(v_[front].min(), 0, h - 1), np.clip(v_[front].max(), 0, h - 1)
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            p_pts, n_pts = [], []
            if o.get("points") is not None:
                mine = np.flatnonzero((owner == i) & vis)
                if len(mine) < 5:                     # fully hidden behind other objects in this view
                    continue
                pick = mine[np.linspace(0, len(mine) - 1, min(4, len(mine))).astype(int)]
                p_pts = [(u[j], v[j]) for j in pick]
                vx0, vx1 = u[mine].min(), u[mine].max()
                vy0, vy1 = v[mine].min(), v[mine].max()
                x0, x1, y0, y1 = max(0, vx0 - 6), min(w - 1, vx1 + 6), max(0, vy0 - 6), min(h - 1, vy1 + 6)
                near = np.flatnonzero((owner != i) & vis & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1))
                if len(near):
                    pk = near[np.linspace(0, len(near) - 1, min(3, len(near))).astype(int)]
                    n_pts = [(u[j], v[j]) for j in pk]
            else:
                cu, cv_, cz = project(np.asarray(o["box"][0])[None], pose, K)
                if cz[0] > 0.05 and 0 <= cu[0] < w and 0 <= cv_[0] < h:
                    p_pts = [(cu[0], cv_[0])]
            pr = _pred_from_points(P, pose, K, (h, w), 0.006)
            boxes.append([x0, y0, x1, y1]); pos.append(p_pts); neg.append(n_pts); order.append(i)
            preds.append(pr[0] if pr else None); depth.append(pr[1] if pr else 9.0)
        idm = np.zeros((h, w), np.uint8)
        flags = np.zeros(n + 1, bool)
        if boxes:
            sess = SamSession(Image.fromarray(img))
            logits, iou = sess.predict_prompts(boxes, pos, neg)
            mk = sess.upsample(logits)
            del sess
            # paint far to near so the nearer object wins overlapping pixels
            for j in np.argsort(depth)[::-1]:
                i = order[j]
                m = _clean(mk[j])                    # printed labels on an object must not become holes in it
                idm[m] = i + 1
            for j, i in enumerate(order):
                m = idm == (i + 1)
                pr = preds[j]
                if pr is None or m.sum() < 30:
                    continue
                small = m[::4, ::4][:pr.shape[0], :pr.shape[1]]
                pr = pr[:small.shape[0], :small.shape[1]]
                inter = (small & pr).sum()
                # the mask must sit inside the predicted silhouette and cover a fair part of what is visible
                flags[i + 1] = inter / max(1, small.sum()) > 0.6 and float(iou[j]) > 0.5
        ids.append(idm)
        ok.append(flags)
        if unknown_out is not None:
            unk = np.zeros((h, w), bool)
            for j, i in enumerate(order):
                if not flags[i + 1] and preds[j] is not None:
                    big = cv2.resize(preds[j].astype(np.uint8), (w // 4 * 4 + 4, h // 4 * 4 + 4),
                                     interpolation=cv2.INTER_NEAREST)[:h, :w] > 0
                    unk |= big & (idm == 0)
            unknown_out.append(unk)
        progress(k + 1, len(frames))
    return ids, ok


def _clean(m: np.ndarray) -> np.ndarray:
    """Fill holes (labels, highlights) and close hairline gaps; drop specks."""
    from scipy import ndimage
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    m = ndimage.binary_fill_holes(m)
    lab, nl = ndimage.label(m)
    if nl > 1:
        sizes = ndimage.sum(m, lab, range(1, nl + 1))
        m = np.isin(lab, 1 + np.flatnonzero(sizes >= max(40, 0.05 * sizes.max())))
    return m
