"""Optional Claude vision labelling: names, materials, shapes and mass estimates.

Without credentials the objects keep neutral names ("object 3") and the
"unknown" material prior; the UI lets the user fix both."""
from __future__ import annotations

import base64
import json
import re
import logging

import cv2
import numpy as np

from .. import config
from .materials import MATERIALS

log = logging.getLogger(__name__)

SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "label": {"type": "string"},
                    "material": {"type": "string", "enum": sorted(MATERIALS.keys())},
                    "shape": {"type": "string", "enum": ["box", "cylinder"]},
                    "estimated_mass_kg": {"type": "number"},
                    "confidence": {"type": "number"},
                },
                "required": ["number", "label", "material", "shape", "estimated_mass_kg", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["objects"],
    "additionalProperties": False,
}


def annotate(rgb: np.ndarray, masks: list[np.ndarray]) -> bytes:
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, m in enumerate(masks):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 255), 2)
        ys, xs = np.nonzero(m)
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.circle(img, (cx, cy), 14, (0, 0, 0), -1)
        cv2.putText(img, str(i + 1), (cx - 8 if i < 9 else cx - 13, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def label_with_claude(rgb: np.ndarray, masks: list[np.ndarray]) -> list[dict] | None:
    if not config.CLAUDE_ENABLED or not masks:
        return None
    import anthropic
    from .. import llm
    client = llm.client()
    if client is None:
        return None
    img_b64 = base64.standard_b64encode(annotate(rgb, masks)).decode()
    prompt = (
        f"The photo shows a tabletop. {len(masks)} objects are outlined in yellow and numbered 1..{len(masks)}. "
        "For each number, give a short everyday label (e.g. 'ceramic mug', 'hardcover book'), the dominant material "
        "from the allowed list, the collision shape that best approximates it (box or cylinder), your best estimate "
        "of its mass in kg, and a 0-1 confidence. If an outline covers background or two objects, still answer and "
        "use low confidence."
    )
    try:
        resp = client.beta.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
    except anthropic.AuthenticationError as e:
        log.warning("Claude labelling skipped (auth): %s", e)
        return None
    except anthropic.APIConnectionError as e:
        log.warning("Claude labelling skipped (network): %s", e)
        return None
    except TypeError as e:          # credentials vanished between check and call
        log.warning("Claude call skipped (no credentials): %s", e)
        return None
    except anthropic.APIStatusError as e:
        log.warning("Claude labelling failed (%s): %s", e.status_code, e)
        return None
    if resp.stop_reason == "refusal":
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    try:
        return json.loads(text)["objects"]
    except (json.JSONDecodeError, KeyError):
        return None


LOCAL_PROMPT = (
    "The photo shows objects outlined in yellow and numbered 1..{n}. For EACH number give a short everyday "
    "name, its main material and its rough shape. Materials: {mats}. Shapes: box, cylinder. "
    'Answer ONLY with JSON in this form: {{"objects":[{{"number":<n>,"label":"<what you see>","material":"<material>","shape":"<box|cylinder>"}}]}}'
)


def label_with_local_ai(rgb: np.ndarray, masks: list[np.ndarray]) -> list[dict] | None:
    """Free on-device labelling (Qwen2.5-VL via MLX), one cropped object per question so a
    small model can't skip or mix up numbers. Every field is validated."""
    from .. import local_ai
    if not masks or not local_ai.available():
        return None
    out = []
    h, w = rgb.shape[:2]
    for n, m in enumerate(masks, start=1):
        # tight, unmodified crop + "main object" prompt: best of three variants measured on real photos
        # (outlines biased colour names; blacked-out backgrounds made a book look like "a rock")
        ys, xs = np.nonzero(m)
        pad = int(0.08 * max(np.ptp(xs), np.ptp(ys))) + 6
        x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
        y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad)
        # dim (don't black out) everything that isn't this object, so a neighbour in the crop doesn't win
        sub = rgb[y0:y1, x0:x1].astype(np.float32)
        mm = m[y0:y1, x0:x1].astype(np.float32)
        mm = cv2.GaussianBlur(cv2.dilate(mm, np.ones((5, 5), np.uint8)), (0, 0), 3)[..., None]
        sub = sub * (0.45 + 0.55 * mm)
        crop = cv2.cvtColor(sub.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        k = max(1.0, 336 / max(crop.shape[:2]))
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        # Plain questions, no example answers: small vision models copy examples ("white mug" for a printer)
        # and do worse when forced into JSON. Name and material are asked separately.
        name = _clean_name(local_ai.generate("What is the brightest, outlined object in the middle of this photo? "
                                             "Answer with only its common name, 1 to 3 words.", buf.tobytes(), max_tokens=12))
        # second opinion from CLIP on the plain (undimmed) crop: fixes the small model's typical slips
        from .clip_rank import rerank
        plain = rgb[y0:y1, x0:x1]
        name2, conf = rerank(plain, name)
        if name2 and name2 != name:
            log.info("label %d: '%s' -> '%s' (CLIP %.2f)", n, name, name2, conf)
        name = name2
        if not name:
            log.warning("local labeller: no usable name for object %d", n)
            continue
        mat = _known_material(name)
        if mat is None:
            mat_txt = (local_ai.generate(f"This is a {name}. What material is it mostly made of? Answer with exactly one word from this list: "
                                         + ", ".join(sorted(MATERIALS)) + ".", buf.tobytes(), max_tokens=6) or "").strip().lower()
            mat = next((x for x in MATERIALS if x in mat_txt), "unknown")
        out.append({"number": n, "label": name, "material": mat, "shape": None, "estimated_mass_kg": 0, "confidence": 0.6})
    return out or None


_ECHO = {"object", "objects", "photo", "image", "picture", "middle", "outlined", "brightest", "thing", "item", "unknown", "none"}


def _clean_name(text: str | None) -> str | None:
    """'A printer.' -> 'printer'; reject empty answers and ones that just echo the question."""
    if not text:
        return None
    t = re.sub(r"[^a-z0-9 \-]", " ", text.strip().lower().split("\n")[0])
    t = re.sub(r"^(it is|it's|this is|that is|a|an|the)\s+", "", t.strip())
    t = re.sub(r"^(a|an|the)\s+", "", t.strip())
    words = t.split()[:3]
    if not words or all(w in _ECHO for w in words):
        return None
    return " ".join(words)[:40]


def _known_material(name: str) -> str | None:
    """Material from the knowledge base when the thing is known (a chair is wood, a phone is glass) —
    saves a second question to the small model and is more reliable."""
    from ..knowledge import _T, _singular
    n = _singular(name.lower().strip())
    for key in (n, " ".join(n.split()[-2:]), n.split()[-1]):      # exact name, then its head noun
        if key in _T and _T[key][3] in MATERIALS:
            return _T[key][3]
    return None
