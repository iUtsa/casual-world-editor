"""Clean-plate generation for objects that leave their original pixels."""
from __future__ import annotations

import cv2
import numpy as np


def clean_plate_patch(rgb: np.ndarray, masks: list[np.ndarray], clean_plate: np.ndarray | None = None) -> np.ndarray:
    """RGBA image: inpainted (or captured empty-table) pixels where the given
    objects were, transparent elsewhere. Composited over the live/frozen frame."""
    h, w = rgb.shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    if not masks:
        return out
    union = np.any(masks, axis=0).astype(np.uint8)
    hole = cv2.dilate(union, np.ones((9, 9), np.uint8))           # include soft edges / contact shadow
    if clean_plate is not None and clean_plate.shape == rgb.shape:
        fill = clean_plate
    else:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # inpaint at half resolution (faster, smoother for large holes), then upsample
        small = cv2.resize(bgr, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        hs = cv2.resize(hole, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
        hs = cv2.dilate(hs, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, hs, 6, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC)
        fill = cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)
    alpha = cv2.GaussianBlur(hole.astype(np.float32) * 255, (7, 7), 0)
    out[..., :3] = fill
    out[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return out
