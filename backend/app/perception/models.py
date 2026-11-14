"""Lazy singletons for the two vision foundation models.

Depth Anything V2 (metric, indoor) gives z-depth in metres; SAM 2.1 gives
promptable masks. Both are small enough for an 8 GB M1 with MPS."""
from __future__ import annotations

import threading
import time

import numpy as np
import torch
from PIL import Image

from .. import config

_lock = threading.Lock()
_depth = None
_sam = None


def _load_depth():
    global _depth
    with _lock:
        if _depth is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            proc = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL_ID)
            model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL_ID).to(config.DEVICE).eval()
            _depth = (proc, model)
    return _depth


def _load_sam():
    global _sam
    with _lock:
        if _sam is None:
            from transformers import Sam2Model, Sam2Processor
            proc = Sam2Processor.from_pretrained(config.SAM_MODEL_ID)
            model = Sam2Model.from_pretrained(config.SAM_MODEL_ID).to(config.DEVICE).eval()
            _sam = (proc, model)
    return _sam


def warmup() -> dict[str, float]:
    t = {}
    s = time.perf_counter(); _load_depth(); t["depth_load"] = (time.perf_counter() - s) * 1000
    s = time.perf_counter(); _load_sam(); t["sam_load"] = (time.perf_counter() - s) * 1000
    img = Image.new("RGB", (320, 240), (128, 128, 128))
    estimate_depth(img)
    SamSession(img).predict_points(np.array([[160.0, 120.0]]))
    return t


@torch.inference_mode()
def estimate_depth(img: Image.Image) -> np.ndarray:
    proc, model = _load_depth()
    inputs = proc(images=img, return_tensors="pt").to(config.DEVICE)
    out = model(**inputs)
    post = proc.post_process_depth_estimation(out, target_sizes=[(img.height, img.width)])
    return post[0]["predicted_depth"].float().cpu().numpy()


class SamSession:
    """Holds one image's embeddings so many prompts (auto proposals, user
    clicks) reuse a single encoder pass."""

    def __init__(self, img: Image.Image):
        self.proc, self.model = _load_sam()
        self.img = img
        with torch.inference_mode():
            inputs = self.proc(images=img, return_tensors="pt")
            self.original_sizes = inputs["original_sizes"]
            pv = inputs["pixel_values"].to(config.DEVICE)
            self.embeddings = self.model.get_image_embeddings(pv)

    @torch.inference_mode()
    def predict_points(self, pts: np.ndarray, chunk: int = 48, labels: np.ndarray | None = None):
        """One positive point per prompt. Returns (logits_lowres [N,3,h,w], iou [N,3])."""
        all_logits, all_iou = [], []
        for s in range(0, len(pts), chunk):
            p = pts[s:s + chunk]
            ip = [[[[float(x), float(y)]] for x, y in p]]
            il = [[[1] for _ in p]]
            enc = self.proc(images=None, input_points=ip, input_labels=il,
                            original_sizes=self.original_sizes, return_tensors="pt")
            out = self.model(input_points=enc["input_points"].to(config.DEVICE),
                             input_labels=enc["input_labels"].to(config.DEVICE),
                             image_embeddings=self.embeddings, multimask_output=True)
            all_logits.append(out.pred_masks[0].float().cpu())
            all_iou.append(out.iou_scores[0].float().cpu())
        return torch.cat(all_logits), torch.cat(all_iou)

    @torch.inference_mode()
    def predict_multi(self, pts: np.ndarray):
        """Several positive points describing ONE object. Returns (logits [h,w], iou)."""
        ip = [[[[float(x), float(y)] for x, y in pts]]]
        il = [[[1] * len(pts)]]
        enc = self.proc(images=None, input_points=ip, input_labels=il,
                        original_sizes=self.original_sizes, return_tensors="pt")
        out = self.model(input_points=enc["input_points"].to(config.DEVICE),
                         input_labels=enc["input_labels"].to(config.DEVICE),
                         image_embeddings=self.embeddings, multimask_output=False)
        return out.pred_masks[0, 0, 0].float().cpu(), float(out.iou_scores[0, 0, 0])

    @torch.inference_mode()
    def predict_prompts(self, boxes: list, pos: list, neg: list, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
        """One object per prompt: a box [x0,y0,x1,y1] plus positive / negative points (each a list of (x, y)).
        Returns (logits_lowres [N,h,w], iou [N]) with a single mask per object."""
        out_l, out_i = [], []
        for s in range(0, len(boxes), chunk):
            bb, pp, nn = boxes[s:s + chunk], pos[s:s + chunk], neg[s:s + chunk]
            npt = max(1, max(len(a) + len(b) for a, b in zip(pp, nn)))
            pts, lab = [], []
            for a, b in zip(pp, nn):
                q = [list(map(float, x)) for x in a] + [list(map(float, x)) for x in b]
                l_ = [1] * len(a) + [0] * len(b)
                if not q:                                   # box only: pad with an ignored point
                    q, l_ = [[0.0, 0.0]], [-10]
                while len(q) < npt:                         # pad to the same count (label -10 = ignored)
                    q.append([0.0, 0.0]); l_.append(-10)
                pts.append(q); lab.append(l_)
            enc = self.proc(images=None, input_points=[pts], input_labels=[lab], input_boxes=[[list(map(float, b)) for b in bb]],
                            original_sizes=self.original_sizes, return_tensors="pt")
            out = self.model(input_points=enc["input_points"].to(config.DEVICE),
                             input_labels=enc["input_labels"].to(config.DEVICE),
                             input_boxes=enc["input_boxes"].to(config.DEVICE),
                             image_embeddings=self.embeddings, multimask_output=False)
            out_l.append(out.pred_masks[0, :, 0].float().cpu())
            out_i.append(out.iou_scores[0, :, 0].float().cpu())
        return torch.cat(out_l), torch.cat(out_i)

    def upsample(self, logits: torch.Tensor) -> np.ndarray:
        """[N,h,w] low-res logits -> [N,H,W] bool masks at original resolution."""
        if len(logits) == 0:
            return np.zeros((0, self.img.height, self.img.width), bool)
        masks = self.proc.post_process_masks(logits[None, :, None], self.original_sizes)[0]
        return masks[:, 0].numpy().astype(bool)
