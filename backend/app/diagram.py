"""Read a photo of a diagram (whiteboard, textbook, slide) and turn it into something the labs can run.

The free on-device vision model (Qwen2.5-VL-3B) is asked plain questions — no example answers to copy:
first what kind of diagram it is, then a targeted reading for that kind (the parts and values of a circuit,
the set-up and numbers of a physics drawing, the reaction in a chemistry diagram, the labels of a
neural-network diagram). The result is shown to the student as an editable sentence BEFORE anything runs,
because a small model can misread messy handwriting; the student confirms or fixes it.
"""
from __future__ import annotations

import logging
import re

import cv2
import numpy as np

log = logging.getLogger(__name__)

KINDS = {
    "circuit": r"circuit|electric|resistor|battery|voltage",
    "neural-network": r"neural|network|transformer|attention|layer|machine learning|deep learning|encoder|decoder|embedding",
    "chemistry": r"chem|reaction|molecule|atom|bond|compound",
    "astronomy": r"astronom|planet|orbit|solar|space|moon|star",
    "physics": r"physic|force|motion|ramp|incline|pendulum|spring|projectile|pulley|block|mechanic|collision",
}

QUESTIONS = {
    "circuit": "This is a circuit diagram. List every component you can see with its value (for example volts, ohms, farads) and say whether they "
               "are connected in series or in parallel. Answer in one sentence that starts with 'a ' and the battery.",
    "physics": "This is a physics diagram. Describe the situation it shows as ONE short sentence saying what object does what, including every "
               "number and unit written on it (angles, masses, lengths, speeds, heights, friction).",
    "chemistry": "This is a chemistry diagram. Write the chemical reaction it shows as 'reactants -> products' using names or formulas.",
    "astronomy": "This is an astronomy diagram. Describe what it shows in one sentence, naming the bodies and what they do (orbit, collide).",
    "neural-network": "This is a machine-learning diagram. List the names of the blocks or layers written in it, in order, separated by commas.",
}


def _jpeg(img_bytes: bytes, max_side: int = 896) -> bytes:
    arr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("not an image")
    h, w = arr.shape[:2]
    k = min(1.0, max_side / max(h, w))
    if k < 1:
        arr = cv2.resize(arr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # Qwen2.5-VL needs sides that are multiples of 28
    h, w = arr.shape[:2]
    arr = cv2.resize(arr, (max(28, w // 28 * 28), max(28, h // 28 * 28)))
    ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


def read(img_bytes: bytes) -> dict:
    from . import local_ai
    if not local_ai.available():
        return {"kind": "unknown", "reading": "", "error": "The on-device vision model isn't running, so I can't read the photo. Describe the diagram in words instead."}
    jpg = _jpeg(img_bytes)
    kind_txt = (local_ai.generate("What kind of diagram is this? Answer with one or two words: an electric circuit, a physics diagram, "
                                  "a chemistry diagram, an astronomy diagram, or a neural network / machine learning diagram.", jpg, max_tokens=12) or "").lower()
    kind = next((k for k, pat in KINDS.items() if re.search(pat, kind_txt)), "physics")
    reading = (local_ai.generate(QUESTIONS[kind], jpg, max_tokens=90) or "").strip().split("\n")[0].strip(" .\"'")
    out = {"kind": kind, "reading": reading, "kind_raw": kind_txt.strip()}
    if kind == "neural-network":
        out["lab"] = "transformer"
        out["labels"] = [x.strip() for x in re.split(r"[,;→>]", reading) if x.strip()][:16]
        out["reading"] = "Show me how a transformer processes: The Eiffel Tower is located in the city of"
    elif kind == "chemistry":
        out["lab"] = "chemistry"
    else:
        out["lab"] = "experiment"
        if kind == "circuit" and not re.search(r"\bbattery|\bcell|\bvolt|\d\s*v\b", reading.lower()):
            out["reading"] = "a 9 V battery with " + reading
    log.info("diagram read as %s: %s", kind, out["reading"])
    return out
