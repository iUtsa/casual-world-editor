"""Second opinion on object names with CLIP (free, local, ~0.6 GB).

The small on-device vision-language model proposes a name for each object crop; CLIP then scores that
crop against the proposal AND a vocabulary of everyday things. The proposal wins unless CLIP is clearly
more confident in something else — this fixes the VLM's typical slips ("moon" for a person in a dark
shirt, "laptop" for the back of an office chair) without overruling it on things it gets right.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from .. import config

log = logging.getLogger(__name__)
_lock = threading.Lock()
_model = {"m": None, "p": None, "text": None, "vocab": None}
PATH = config.DATA_DIR / "models" / "clip-vit-b32"

def _vocab() -> list[str]:
    from ..knowledge import _T
    words = set(_T)
    multi = ["office chair", "desk chair", "fax machine", "computer monitor", "computer mouse", "desk phone", "remote control", "desk lamp",
             "ceiling light", "light bulb", "security camera", "cctv camera", "wall clock", "picture frame", "filing cabinet", "stack of paper",
             "file tray", "document tray", "water bottle", "potted plant", "cardboard box", "tissue box", "trash can", "power strip",
             "air conditioner", "printer paper", "teddy bear", "rubber duck", "toy car", "snack bag", "coffee maker", "cutting board",
             "soap bottle", "dish rack"]
    singles = "person man woman child chair stool table desk printer scanner copier monitor laptop keyboard phone smartphone tablet television " \
              "speaker headphones lamp clock painting poster whiteboard window blinds curtain door cabinet drawer bookshelf shelf book notebook " \
              "paper envelope folder binder pen pencil marker stapler scissors tape mug cup glass bottle can bowl plate jar vase plant flower " \
              "backpack bag handbag purse wallet keys umbrella shoe box calculator charger cable fan heater router pillow cushion sofa bed blanket " \
              "towel toy ball apple banana orange fruit bread sandwich pizza cake cookie jacket coat hat glasses sunglasses watch candle mirror " \
              "sink faucet microwave toaster kettle refrigerator oven stove pan pot spoon fork knife"
    words |= set(multi) | set(singles.split())
    words -= {"support", "mass", "weight"}
    return sorted(words)


def _load():
    with _lock:
        if _model["m"] is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = CLIPModel.from_pretrained(PATH, local_files_only=True).to(dev).eval()
        p = CLIPProcessor.from_pretrained(PATH, local_files_only=True)
        vocab = _vocab()
        with torch.no_grad():
            tok = p(text=[f"a photo of a {w}" for w in vocab], return_tensors="pt", padding=True).to(dev)
            te = _vec(m.get_text_features(**tok))
            te = te / te.norm(dim=-1, keepdim=True)
        _model.update(m=m, p=p, text=te, vocab=vocab, dev=dev)
        log.info("CLIP ready (%d vocabulary words)", len(vocab))


def _vec(x):
    """transformers 5 returns an output object (projected embedding in pooler_output); older versions a tensor."""
    return x if hasattr(x, "norm") else x.pooler_output


def available() -> bool:
    # opt-in (CWE_CLIP=1): ~0.6 GB of memory for a modest gain on dim camera crops; an 8 GB Mac needs the room
    import os
    return os.environ.get("CWE_CLIP") == "1" and (PATH / "model.safetensors").exists()


def rerank(crop_rgb: np.ndarray, proposal: str | None) -> tuple[str | None, float]:
    """(best name, confidence). Keeps the VLM's proposal unless CLIP strongly prefers another word."""
    if not available():
        return proposal, 0.0
    try:
        _load()
    except Exception as e:      # never let a second opinion break a scan
        log.warning("CLIP unavailable: %s", e)
        return proposal, 0.0
    import torch
    from PIL import Image
    m, p, te, vocab, dev = _model["m"], _model["p"], _model["text"], _model["vocab"], _model["dev"]
    with torch.no_grad():
        im = p(images=Image.fromarray(crop_rgb), return_tensors="pt").to(dev)
        ie = _vec(m.get_image_features(**im))
        ie = ie / ie.norm(dim=-1, keepdim=True)
        logits = (100.0 * ie @ te.T)[0]
        if proposal:
            tp = p(text=[f"a photo of a {proposal}"], return_tensors="pt", padding=True).to(dev)
            pe = _vec(m.get_text_features(**tp))
            pe = pe / pe.norm(dim=-1, keepdim=True)
            lp = float((100.0 * ie @ pe.T)[0, 0])
        else:
            lp = -1e9
    probs = torch.softmax(torch.cat([logits, torch.tensor([lp], device=logits.device)]), 0).cpu().numpy()
    k = int(np.argmax(probs[:-1]))
    p_best, p_prop = float(probs[k]), float(probs[-1])
    if proposal is None or (p_best > 0.30 and p_best > 3.0 * p_prop):
        return vocab[k], p_best
    return proposal, p_prop
