"""Physical knowledge for anything a user can name.

Resolution order for a name ("two blue whales", "a duck", "my phone"):
  1. a real object in the scanned scene with that label,
  2. a library model (real 3D asset) - fuzzy + semantic match,
  3. this table of real-world sizes/masses (stand-in shape, clearly labelled),
  4. the on-device language model's estimate, accepted only if its density is physically sane.
Sizes are full extents in metres (length, width, height), masses in kg.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from . import assets as asset_lib

# name: (shape, (L, W, H) m, mass kg, material, colour, bounciness 0..1, habitat, self_propelled, typical speed m/s)
# habitat: table | ground | water | air | space
_T = {
    # balls & sport
    "baseball": ("sphere", (0.074,) * 3, 0.145, "leather", "#f5f5f0", 0.55, "table", False, 35),
    "tennis ball": ("sphere", (0.067,) * 3, 0.058, "rubber", "#d4f542", 0.75, "table", False, 50),
    "golf ball": ("sphere", (0.043,) * 3, 0.046, "plastic", "#ffffff", 0.78, "table", False, 70),
    "ping pong ball": ("sphere", (0.04,) * 3, 0.0027, "plastic", "#ffffff", 0.89, "table", False, 20),
    "basketball": ("sphere", (0.24,) * 3, 0.62, "rubber", "#e2711d", 0.8, "ground", False, 10),
    "soccer ball": ("sphere", (0.22,) * 3, 0.43, "rubber", "#f8fafc", 0.75, "ground", False, 30),
    "football": ("sphere", (0.22,) * 3, 0.43, "rubber", "#f8fafc", 0.75, "ground", False, 30),
    "volleyball": ("sphere", (0.21,) * 3, 0.27, "rubber", "#fde68a", 0.75, "ground", False, 20),
    "bowling ball": ("sphere", (0.218,) * 3, 6.8, "plastic", "#1e3a8a", 0.1, "ground", False, 8),
    "beach ball": ("sphere", (0.5,) * 3, 0.15, "plastic", "#ef4444", 0.7, "ground", False, 5),
    "billiard ball": ("sphere", (0.057,) * 3, 0.17, "plastic", "#dc2626", 0.92, "table", False, 5),
    "pool ball": ("sphere", (0.057,) * 3, 0.17, "plastic", "#dc2626", 0.92, "table", False, 5),
    "cricket ball": ("sphere", (0.072,) * 3, 0.16, "leather", "#991b1b", 0.5, "ground", False, 40),
    "marble": ("sphere", (0.016,) * 3, 0.005, "glass", "#93c5fd", 0.8, "table", False, 2),
    "bouncy ball": ("sphere", (0.045,) * 3, 0.03, "rubber", "#a855f7", 0.9, "table", False, 5),
    "steel ball": ("sphere", (0.04,) * 3, 0.26, "metal", "#cbd5e1", 0.6, "table", False, 3),
    "puck": ("cylinder", (0.076, 0.076, 0.025), 0.17, "rubber", "#111111", 0.4, "ground", False, 40),
    "frisbee": ("cylinder", (0.27, 0.27, 0.03), 0.175, "plastic", "#f97316", 0.3, "air", False, 15),
    # everyday
    "egg": ("ellipsoid", (0.045, 0.045, 0.06), 0.06, "shell", "#fef3c7", 0.05, "table", False, 1),
    "apple": ("sphere", (0.08,) * 3, 0.18, "food", "#dc2626", 0.2, "table", False, 2),
    "orange": ("sphere", (0.08,) * 3, 0.15, "food", "#f97316", 0.25, "table", False, 2),
    "watermelon": ("ellipsoid", (0.35, 0.25, 0.25), 7.0, "food", "#15803d", 0.1, "ground", False, 1),
    "banana": ("capsule", (0.2, 0.04, 0.04), 0.12, "food", "#facc15", 0.1, "table", False, 1),
    "phone": ("box", (0.15, 0.072, 0.008), 0.18, "glass", "#111827", 0.2, "table", False, 2),
    "smartphone": ("box", (0.15, 0.072, 0.008), 0.18, "glass", "#111827", 0.2, "table", False, 2),
    "laptop": ("box", (0.31, 0.22, 0.016), 1.4, "metal", "#9ca3af", 0.1, "table", False, 1),
    "book": ("box", (0.23, 0.155, 0.03), 0.5, "paper", "#1d4ed8", 0.05, "table", False, 1),
    "brick": ("box", (0.215, 0.1025, 0.065), 2.3, "ceramic", "#b45309", 0.05, "ground", False, 1),
    "mug": ("cylinder", (0.085, 0.085, 0.1), 0.33, "ceramic", "#f8fafc", 0.1, "table", False, 1),
    "cup": ("cylinder", (0.08, 0.08, 0.095), 0.25, "ceramic", "#f8fafc", 0.1, "table", False, 1),
    "glass": ("cylinder", (0.075, 0.075, 0.12), 0.25, "glass", "#e0f2fe", 0.1, "table", False, 1),
    "bottle": ("cylinder", (0.07, 0.07, 0.24), 0.55, "plastic", "#22c55e", 0.2, "table", False, 1),
    "can": ("cylinder", (0.066, 0.066, 0.122), 0.37, "aluminum", "#ef4444", 0.2, "table", False, 1),
    "pencil": ("cylinder", (0.0075, 0.0075, 0.19), 0.006, "wood", "#facc15", 0.2, "table", False, 1),
    "pen": ("cylinder", (0.01, 0.01, 0.14), 0.012, "plastic", "#1e3a8a", 0.2, "table", False, 1),
    "coin": ("cylinder", (0.024, 0.024, 0.00175), 0.0057, "metal", "#d1d5db", 0.4, "table", False, 1),
    "dice": ("box", (0.016,) * 3, 0.004, "plastic", "#f8fafc", 0.4, "table", False, 2),
    "die": ("box", (0.016,) * 3, 0.004, "plastic", "#f8fafc", 0.4, "table", False, 2),
    "block": ("box", (0.05,) * 3, 0.08, "wood", "#d97706", 0.3, "table", False, 1),
    "wooden block": ("box", (0.05,) * 3, 0.08, "wood", "#d97706", 0.3, "table", False, 1),
    "lego brick": ("box", (0.032, 0.016, 0.0096), 0.0025, "plastic", "#dc2626", 0.4, "table", False, 1),
    "chair": ("box", (0.45, 0.45, 0.9), 6.0, "wood", "#92400e", 0.1, "ground", False, 1),
    "table": ("box", (1.2, 0.75, 0.75), 30.0, "wood", "#a16207", 0.1, "ground", False, 1),
    "box": ("box", (0.3, 0.2, 0.2), 1.0, "cardboard", "#c08a4b", 0.2, "ground", False, 1),
    "fridge": ("box", (0.7, 0.7, 1.8), 80.0, "metal", "#e5e7eb", 0.05, "ground", False, 1),
    "pillow": ("box", (0.5, 0.35, 0.12), 0.6, "fabric", "#e0e7ff", 0.3, "ground", False, 1),
    "feather": ("box", (0.15, 0.02, 0.002), 0.0005, "fabric", "#f8fafc", 0.0, "air", False, 1),
    "rock": ("sphere", (0.15,) * 3, 4.5, "stone", "#78716c", 0.2, "ground", False, 1),
    "boulder": ("sphere", (1.5,) * 3, 4800.0, "stone", "#57534e", 0.15, "ground", False, 1),
    "anvil": ("box", (0.4, 0.15, 0.25), 50.0, "metal", "#374151", 0.2, "ground", False, 1),
    "piano": ("box", (1.5, 0.6, 1.2), 230.0, "wood", "#111827", 0.05, "ground", False, 1),
    # people & animals (stand-in body shapes; self-propelled)
    "person": ("capsule", (0.4, 0.3, 1.75), 75.0, "flesh", "#2563eb", 0.2, "ground", True, 4),
    "human": ("capsule", (0.4, 0.3, 1.75), 75.0, "flesh", "#2563eb", 0.2, "ground", True, 4),
    "man": ("capsule", (0.42, 0.3, 1.78), 80.0, "flesh", "#2563eb", 0.2, "ground", True, 4),
    "woman": ("capsule", (0.38, 0.28, 1.65), 65.0, "flesh", "#be185d", 0.2, "ground", True, 4),
    "child": ("capsule", (0.3, 0.2, 1.2), 25.0, "flesh", "#16a34a", 0.2, "ground", True, 3),
    "dog": ("ellipsoid", (0.9, 0.3, 0.6), 30.0, "flesh", "#a16207", 0.2, "ground", True, 9),
    "cat": ("ellipsoid", (0.46, 0.2, 0.3), 4.5, "flesh", "#78716c", 0.2, "ground", True, 8),
    "horse": ("ellipsoid", (2.4, 0.7, 1.7), 500.0, "flesh", "#78350f", 0.15, "ground", True, 15),
    "cow": ("ellipsoid", (2.5, 0.8, 1.5), 700.0, "flesh", "#f5f5f4", 0.1, "ground", True, 6),
    "elephant": ("ellipsoid", (6.0, 2.5, 3.3), 6000.0, "flesh", "#6b7280", 0.1, "ground", True, 7),
    "rhino": ("ellipsoid", (3.8, 1.5, 1.8), 2300.0, "flesh", "#78716c", 0.1, "ground", True, 13),
    "bull": ("ellipsoid", (2.6, 0.9, 1.6), 900.0, "flesh", "#3f3f46", 0.1, "ground", True, 10),
    "lion": ("ellipsoid", (2.0, 0.6, 1.1), 190.0, "flesh", "#ca8a04", 0.15, "ground", True, 22),
    "tiger": ("ellipsoid", (2.5, 0.6, 1.0), 220.0, "flesh", "#ea580c", 0.15, "ground", True, 18),
    "bear": ("ellipsoid", (2.2, 0.9, 1.3), 400.0, "flesh", "#451a03", 0.15, "ground", True, 13),
    "kangaroo": ("ellipsoid", (1.2, 0.5, 1.5), 66.0, "flesh", "#b45309", 0.3, "ground", True, 15),
    "ant": ("ellipsoid", (0.005, 0.002, 0.002), 3e-6, "flesh", "#111111", 0.2, "table", True, 0.05),
    "chicken": ("ellipsoid", (0.4, 0.25, 0.4), 2.5, "flesh", "#f8fafc", 0.2, "ground", True, 4),
    "blue whale": ("ellipsoid", (25.0, 4.0, 4.0), 150000.0, "flesh", "#1e3a8a", 0.1, "water", True, 8),
    "whale": ("ellipsoid", (15.0, 3.0, 3.0), 40000.0, "flesh", "#334155", 0.1, "water", True, 8),
    "orca": ("ellipsoid", (7.0, 1.6, 1.6), 5400.0, "flesh", "#111111", 0.1, "water", True, 15),
    "killer whale": ("ellipsoid", (7.0, 1.6, 1.6), 5400.0, "flesh", "#111111", 0.1, "water", True, 15),
    "dolphin": ("ellipsoid", (2.5, 0.5, 0.5), 200.0, "flesh", "#64748b", 0.1, "water", True, 11),
    "shark": ("ellipsoid", (4.5, 0.9, 0.9), 900.0, "flesh", "#475569", 0.1, "water", True, 10),
    "great white shark": ("ellipsoid", (5.0, 1.0, 1.0), 1100.0, "flesh", "#475569", 0.1, "water", True, 11),
    "fish": ("ellipsoid", (0.3, 0.08, 0.12), 0.4, "flesh", "#f59e0b", 0.1, "water", True, 2),
    "goldfish": ("ellipsoid", (0.08, 0.02, 0.04), 0.03, "flesh", "#f97316", 0.1, "water", True, 1),
    "octopus": ("sphere", (0.6,) * 3, 15.0, "flesh", "#be123c", 0.1, "water", True, 2),
    "submarine": ("capsule", (70.0, 10.0, 10.0), 6000000.0, "metal", "#1f2937", 0.05, "water", True, 12),
    "boat": ("box", (5.0, 2.0, 1.0), 800.0, "plastic", "#f8fafc", 0.1, "water", True, 10),
    "eagle": ("ellipsoid", (0.9, 2.0, 0.3), 5.0, "flesh", "#78350f", 0.2, "air", True, 30),
    "bird": ("ellipsoid", (0.2, 0.3, 0.08), 0.1, "flesh", "#38bdf8", 0.2, "air", True, 10),
    # vehicles
    "car": ("box", (4.5, 1.8, 1.45), 1400.0, "metal", "#dc2626", 0.1, "ground", True, 30),
    "sports car": ("box", (4.4, 1.9, 1.2), 1500.0, "metal", "#facc15", 0.1, "ground", True, 60),
    "truck": ("box", (7.0, 2.5, 3.0), 8000.0, "metal", "#f97316", 0.05, "ground", True, 25),
    "bus": ("box", (12.0, 2.55, 3.2), 12000.0, "metal", "#facc15", 0.05, "ground", True, 20),
    "motorcycle": ("box", (2.1, 0.8, 1.1), 200.0, "metal", "#111827", 0.1, "ground", True, 40),
    "bicycle": ("box", (1.75, 0.6, 1.05), 12.0, "metal", "#0ea5e9", 0.1, "ground", True, 8),
    "tank": ("box", (9.8, 3.7, 2.4), 60000.0, "metal", "#4d7c0f", 0.05, "ground", True, 15),
    "train": ("box", (25.0, 3.0, 4.0), 60000.0, "metal", "#1d4ed8", 0.05, "ground", True, 40),
    "skateboard": ("box", (0.8, 0.2, 0.1), 3.5, "wood", "#7c3aed", 0.2, "ground", False, 5),
    "shopping cart": ("box", (0.9, 0.55, 1.0), 15.0, "metal", "#9ca3af", 0.1, "ground", False, 2),
    "airplane": ("capsule", (40.0, 4.0, 4.0), 70000.0, "metal", "#f8fafc", 0.05, "air", True, 230),
    "rocket": ("capsule", (70.0, 9.0, 9.0), 500000.0, "metal", "#f8fafc", 0.05, "ground", True, 100),
    "drone": ("box", (0.35, 0.35, 0.1), 0.9, "plastic", "#111827", 0.2, "air", True, 15),
    "toy car": ("box", (0.15, 0.07, 0.045), 0.25, "metal", "#16a34a", 0.2, "table", True, 2),
    # astronomy (space)
    "earth": ("sphere", (12.742e6,) * 3, 5.972e24, "stone", "#2563eb", 0.0, "space", False, 0),
    "moon": ("sphere", (3.474e6,) * 3, 7.35e22, "stone", "#9ca3af", 0.0, "space", False, 0),
    "asteroid": ("sphere", (100.0,) * 3, 1.3e9, "stone", "#57534e", 0.1, "space", False, 20000),
    "meteor": ("sphere", (1.0,) * 3, 1500.0, "stone", "#57534e", 0.1, "space", False, 20000),
    "satellite": ("box", (3.0, 2.0, 2.0), 1000.0, "metal", "#e5e7eb", 0.1, "space", True, 7600),
}

SETTING_WORDS = {
    "water": r"\b(ocean|sea|underwater|under water|water|lake|pool|swim|swimming|river)\b",
    "space": r"\b(space|orbit|zero gravity|weightless|planet|asteroid)\b",
    "air": r"\b(sky|air|flying|fly|mid-?air)\b",
    "ground": r"\b(road|street|field|ground|floor|park|highway|arena|outside)\b",
    "table": r"\b(table|desk|tabletop)\b",
}


@dataclass
class Thing:
    """A resolved physical thing."""
    name: str                      # what the user called it
    source: str                    # scene | library | knowledge | estimate
    shape: str                     # sphere | box | cylinder | capsule | ellipsoid | asset | scene
    dims: list[float]              # L, W, H (m)
    mass: float
    material: str = "unknown"
    color: str = "#94a3b8"
    bounciness: float = 0.3
    habitat: str = "table"
    self_propelled: bool = False
    typical_speed: float = 1.0
    asset_id: str | None = None    # library asset
    scene_id: str | None = None    # real object in the scan
    note: str = ""
    extra: dict = field(default_factory=dict)


_EMB = {"model": None, "keys": None, "vecs": None}


def _embed(texts):
    from sentence_transformers import SentenceTransformer
    if _EMB["model"] is None:
        _EMB["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _EMB["model"].encode(texts, normalize_embeddings=True)


def _semantic_best(name: str, keys: list[str]) -> tuple[str | None, float]:
    if not keys:
        return None, 0.0
    v = _embed([name] + keys)
    sims = v[1:] @ v[0]
    i = int(sims.argmax())
    return keys[i], float(sims[i])


def _singular(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r"^(a|an|the|some|my|two|three|four|five|\d+)\s+", "", w)
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        return w[:-1]
    return w


# More everyday, school, nature, fantasy and engineering things (real typical sizes L,W,H in m; mass kg).
# Fantasy creatures use the sizes they're usually drawn at, with animal-tissue density (~1000 kg/m^3 x fill).
_T.update({
    # structures (anchored to the ground: see FIXED)
    "house": ("box", (10, 8, 7), 150000, "wood", "#d6c7a1", 0.1, "ground", False, 0),
    "castle": ("box", (30, 30, 20), 5.0e6, "stone", "#a8a29e", 0.1, "ground", False, 0),
    "wall": ("box", (4, 0.3, 2.5), 6000, "stone", "#b0a99f", 0.15, "ground", False, 0),
    "brick wall": ("box", (4, 0.25, 2), 3600, "stone", "#a0522d", 0.15, "ground", False, 0),
    "tree": ("cylinder", (0.5, 0.5, 8), 1500, "wood", "#6b4f2a", 0.2, "ground", False, 0),
    "building": ("box", (20, 20, 40), 1.2e7, "stone", "#9ca3af", 0.1, "ground", False, 0),
    "skyscraper": ("box", (40, 40, 250), 2.0e8, "metal", "#94a3b8", 0.1, "ground", False, 0),
    "bridge": ("box", (50, 8, 2), 2.0e6, "metal", "#7c8590", 0.1, "ground", False, 0),
    "lighthouse": ("cylinder", (6, 6, 30), 900000, "stone", "#f1f5f9", 0.1, "ground", False, 0),
    "pyramid": ("box", (230, 230, 139), 6.0e9, "stone", "#d6b98c", 0.1, "ground", False, 0),
    "lamp post": ("cylinder", (0.2, 0.2, 5), 120, "metal", "#4b5563", 0.3, "ground", False, 0),
    "fence": ("box", (3, 0.1, 1.2), 60, "wood", "#a16207", 0.2, "ground", False, 0),
    "door": ("box", (0.9, 0.05, 2.0), 30, "wood", "#92400e", 0.25, "ground", False, 0),
    "window": ("box", (1.2, 0.02, 1.2), 25, "glass", "#bfdbfe", 0.3, "ground", False, 0),
    "goal": ("box", (7.3, 2, 2.4), 150, "metal", "#f8fafc", 0.3, "ground", False, 0),
    "mountain": ("box", (2000, 2000, 1000), 1.0e13, "stone", "#78716c", 0.1, "ground", False, 0),
    "iceberg": ("ellipsoid", (100, 60, 40), 1.1e8, "ice", "#e0f2fe", 0.1, "water", False, 0.3),
    # fantasy & prehistoric
    "dragon": ("ellipsoid", (12, 4, 4), 6000, "flesh", "#3f6212", 0.2, "air", True, 25),
    "dinosaur": ("ellipsoid", (12, 3, 4), 8000, "flesh", "#4d7c0f", 0.2, "ground", True, 8),
    "t-rex": ("ellipsoid", (12, 3, 4), 8000, "flesh", "#4d7c0f", 0.2, "ground", True, 8),
    "tyrannosaurus": ("ellipsoid", (12, 3, 4), 8000, "flesh", "#4d7c0f", 0.2, "ground", True, 8),
    "mammoth": ("ellipsoid", (5, 2.5, 3.5), 6000, "flesh", "#78350f", 0.2, "ground", True, 6),
    "unicorn": ("ellipsoid", (2.4, 0.6, 1.6), 500, "flesh", "#f5f3ff", 0.2, "ground", True, 13),
    "giant": ("capsule", (1.5, 1, 6), 3000, "flesh", "#d6a77a", 0.2, "ground", True, 4),
    "monster": ("ellipsoid", (3, 2, 3), 2000, "flesh", "#7c3aed", 0.2, "ground", True, 5),
    "godzilla": ("ellipsoid", (100, 30, 100), 6.0e7, "flesh", "#365314", 0.15, "ground", True, 10),
    "robot": ("box", (0.6, 0.4, 1.7), 80, "metal", "#cbd5e1", 0.3, "ground", True, 1.5),
    "zombie": ("capsule", (0.5, 0.3, 1.75), 70, "flesh", "#84cc16", 0.2, "ground", True, 1.0),
    "alien": ("capsule", (0.5, 0.3, 1.5), 50, "flesh", "#86efac", 0.2, "ground", True, 2.0),
    "ufo": ("ellipsoid", (15, 15, 4), 20000, "metal", "#cbd5e1", 0.3, "air", True, 100),
    "spaceship": ("capsule", (30, 8, 8), 100000, "metal", "#e5e7eb", 0.3, "space", True, 50),
    "spacecraft": ("capsule", (10, 4, 4), 10000, "metal", "#e5e7eb", 0.3, "space", True, 50),
    # more animals
    "giraffe": ("ellipsoid", (2.5, 1, 5), 1200, "flesh", "#e5a84b", 0.2, "ground", True, 14),
    "hippo": ("ellipsoid", (4, 1.6, 1.5), 1800, "flesh", "#7f6f74", 0.2, "ground", True, 8),
    "hippopotamus": ("ellipsoid", (4, 1.6, 1.5), 1800, "flesh", "#7f6f74", 0.2, "ground", True, 8),
    "gorilla": ("ellipsoid", (1, 0.8, 1.7), 160, "flesh", "#262626", 0.2, "ground", True, 11),
    "monkey": ("ellipsoid", (0.5, 0.3, 0.6), 8, "flesh", "#8b5a2b", 0.25, "ground", True, 9),
    "wolf": ("ellipsoid", (1.4, 0.4, 0.8), 45, "flesh", "#6b7280", 0.2, "ground", True, 16),
    "fox": ("ellipsoid", (0.9, 0.3, 0.45), 7, "flesh", "#ea580c", 0.2, "ground", True, 13),
    "deer": ("ellipsoid", (1.8, 0.5, 1.2), 90, "flesh", "#a16207", 0.2, "ground", True, 15),
    "zebra": ("ellipsoid", (2.4, 0.6, 1.4), 350, "flesh", "#e5e7eb", 0.2, "ground", True, 15),
    "camel": ("ellipsoid", (3, 0.8, 2), 500, "flesh", "#c2a26b", 0.2, "ground", True, 10),
    "pig": ("ellipsoid", (1.5, 0.6, 0.9), 150, "flesh", "#f9a8d4", 0.2, "ground", True, 5),
    "sheep": ("ellipsoid", (1.3, 0.5, 0.9), 70, "flesh", "#f5f5f4", 0.25, "ground", True, 8),
    "goat": ("ellipsoid", (1.2, 0.4, 0.8), 60, "flesh", "#e7e5e4", 0.25, "ground", True, 8),
    "rabbit": ("ellipsoid", (0.4, 0.2, 0.25), 2, "flesh", "#d6d3d1", 0.3, "ground", True, 12),
    "squirrel": ("ellipsoid", (0.25, 0.1, 0.15), 0.5, "flesh", "#9a3412", 0.3, "ground", True, 6),
    "rat": ("ellipsoid", (0.25, 0.08, 0.08), 0.3, "flesh", "#78716c", 0.3, "ground", True, 3),
    "snake": ("capsule", (2, 0.08, 0.08), 5, "flesh", "#65a30d", 0.2, "ground", True, 2),
    "crocodile": ("ellipsoid", (4.5, 0.8, 0.6), 500, "flesh", "#3f6212", 0.2, "ground", True, 4),
    "alligator": ("ellipsoid", (4, 0.8, 0.6), 400, "flesh", "#3f6212", 0.2, "ground", True, 4),
    "turtle": ("ellipsoid", (0.4, 0.3, 0.15), 10, "shell", "#4d7c0f", 0.3, "ground", True, 0.3),
    "tortoise": ("ellipsoid", (0.8, 0.6, 0.4), 150, "shell", "#78716c", 0.3, "ground", True, 0.3),
    "frog": ("ellipsoid", (0.1, 0.06, 0.05), 0.1, "flesh", "#16a34a", 0.3, "ground", True, 2),
    "penguin": ("ellipsoid", (0.4, 0.4, 1), 30, "flesh", "#111827", 0.2, "ground", True, 2),
    "polar bear": ("ellipsoid", (2.4, 1, 1.4), 450, "flesh", "#f8fafc", 0.2, "ground", True, 11),
    "panda": ("ellipsoid", (1.6, 0.8, 0.9), 100, "flesh", "#f8fafc", 0.2, "ground", True, 9),
    "cheetah": ("ellipsoid", (1.3, 0.35, 0.8), 55, "flesh", "#eab308", 0.2, "ground", True, 30),
    "leopard": ("ellipsoid", (1.6, 0.4, 0.7), 60, "flesh", "#ca8a04", 0.2, "ground", True, 16),
    "duck": ("ellipsoid", (0.5, 0.25, 0.35), 1.2, "flesh", "#fefce8", 0.25, "ground", True, 1.5),
    "rubber duck": ("ellipsoid", (0.09, 0.07, 0.08), 0.03, "rubber", "#facc15", 0.6, "table", False, 2),
    "swan": ("ellipsoid", (1.4, 0.5, 1.2), 11, "flesh", "#ffffff", 0.2, "water", True, 2),
    "owl": ("ellipsoid", (0.4, 0.3, 0.5), 1.5, "flesh", "#a16207", 0.2, "air", True, 12),
    "parrot": ("ellipsoid", (0.4, 0.15, 0.2), 1, "flesh", "#22c55e", 0.2, "air", True, 15),
    "pigeon": ("ellipsoid", (0.33, 0.15, 0.15), 0.35, "flesh", "#9ca3af", 0.2, "air", True, 20),
    "bat": ("cylinder", (0.07, 0.07, 0.85), 0.9, "wood", "#d6b98c", 0.4, "ground", False, 30),
    "bee": ("ellipsoid", (0.015, 0.007, 0.007), 0.0001, "flesh", "#facc15", 0.2, "air", True, 7),
    "butterfly": ("box", (0.03, 0.08, 0.005), 0.0005, "flesh", "#f97316", 0.2, "air", True, 3),
    "mosquito": ("ellipsoid", (0.005, 0.002, 0.002), 0.0000025, "flesh", "#111827", 0.2, "air", True, 0.5),
    "spider": ("ellipsoid", (0.02, 0.02, 0.01), 0.001, "flesh", "#111827", 0.2, "ground", True, 0.5),
    "beetle": ("ellipsoid", (0.02, 0.01, 0.008), 0.002, "shell", "#14532d", 0.3, "ground", True, 0.1),
    "jellyfish": ("ellipsoid", (0.4, 0.4, 0.3), 5, "flesh", "#f0abfc", 0.1, "water", True, 0.5),
    "seal": ("ellipsoid", (1.8, 0.6, 0.6), 120, "flesh", "#6b7280", 0.2, "water", True, 7),
    "sea turtle": ("ellipsoid", (1.2, 0.9, 0.4), 150, "shell", "#4d7c0f", 0.2, "water", True, 2.5),
    "squid": ("ellipsoid", (1, 0.3, 0.3), 20, "flesh", "#fda4af", 0.1, "water", True, 3),
    "giant squid": ("ellipsoid", (12, 0.6, 0.6), 275, "flesh", "#fb7185", 0.1, "water", True, 3),
    "humpback whale": ("ellipsoid", (15, 3.5, 3.5), 30000, "flesh", "#334155", 0.1, "water", True, 4),
    "sperm whale": ("ellipsoid", (16, 3.5, 3.5), 41000, "flesh", "#475569", 0.1, "water", True, 4),
    "crab": ("ellipsoid", (0.2, 0.15, 0.08), 1, "shell", "#dc2626", 0.3, "ground", True, 0.5),
    "lobster": ("ellipsoid", (0.4, 0.15, 0.1), 1, "shell", "#b91c1c", 0.3, "water", True, 1),
    "salmon": ("ellipsoid", (0.75, 0.15, 0.2), 4.5, "flesh", "#fda4af", 0.2, "water", True, 3),
    "tuna": ("ellipsoid", (2, 0.5, 0.6), 250, "flesh", "#1e3a8a", 0.2, "water", True, 15),
    "baby": ("capsule", (0.3, 0.2, 0.7), 7, "flesh", "#f5d0c5", 0.25, "ground", False, 0.3),
    "boy": ("capsule", (0.4, 0.25, 1.4), 35, "flesh", "#e0b090", 0.25, "ground", True, 4),
    "girl": ("capsule", (0.4, 0.25, 1.4), 35, "flesh", "#e0b090", 0.25, "ground", True, 4),
    "student": ("capsule", (0.45, 0.28, 1.72), 65, "flesh", "#e0b090", 0.25, "ground", True, 4),
    "athlete": ("capsule", (0.5, 0.3, 1.8), 75, "flesh", "#c08860", 0.25, "ground", True, 9),
    "sumo wrestler": ("capsule", (0.8, 0.6, 1.8), 150, "flesh", "#e0b090", 0.25, "ground", True, 4),
    "boxer": ("capsule", (0.5, 0.3, 1.8), 80, "flesh", "#c08860", 0.25, "ground", True, 5),
    "knight": ("capsule", (0.55, 0.35, 1.8), 110, "metal", "#9ca3af", 0.3, "ground", True, 3),
    "ninja": ("capsule", (0.45, 0.28, 1.75), 70, "flesh", "#111827", 0.25, "ground", True, 8),
    "astronaut": ("capsule", (0.7, 0.5, 1.9), 150, "fabric", "#f8fafc", 0.25, "space", True, 1),
    # vehicles
    "van": ("box", (5, 2, 2.2), 2500, "metal", "#e5e7eb", 0.2, "ground", True, 30),
    "pickup truck": ("box", (5.5, 2, 1.9), 2500, "metal", "#b91c1c", 0.2, "ground", True, 30),
    "jeep": ("box", (4.2, 1.9, 1.8), 2000, "metal", "#4d7c0f", 0.2, "ground", True, 30),
    "suv": ("box", (4.8, 1.9, 1.75), 2100, "metal", "#1f2937", 0.2, "ground", True, 30),
    "taxi": ("box", (4.6, 1.8, 1.5), 1500, "metal", "#facc15", 0.2, "ground", True, 25),
    "police car": ("box", (5, 1.9, 1.5), 1800, "metal", "#1e3a8a", 0.2, "ground", True, 35),
    "race car": ("box", (5, 2, 1), 750, "metal", "#dc2626", 0.2, "ground", True, 80),
    "formula 1 car": ("box", (5.6, 2, 0.95), 800, "metal", "#dc2626", 0.2, "ground", True, 90),
    "go kart": ("box", (1.8, 1.2, 0.6), 150, "metal", "#f97316", 0.2, "ground", True, 20),
    "ambulance": ("box", (6, 2.2, 2.8), 4500, "metal", "#f8fafc", 0.2, "ground", True, 30),
    "fire truck": ("box", (10, 2.5, 3.3), 18000, "metal", "#dc2626", 0.2, "ground", True, 25),
    "garbage truck": ("box", (9, 2.5, 3.5), 25000, "metal", "#16a34a", 0.2, "ground", True, 20),
    "tractor": ("box", (4, 2.2, 2.8), 5000, "metal", "#15803d", 0.2, "ground", True, 10),
    "bulldozer": ("box", (6, 3.5, 3.4), 20000, "metal", "#facc15", 0.2, "ground", True, 3),
    "excavator": ("box", (9, 3, 3), 20000, "metal", "#facc15", 0.2, "ground", True, 2),
    "forklift": ("box", (2.5, 1.2, 2.1), 3500, "metal", "#f97316", 0.2, "ground", True, 5),
    "scooter": ("box", (1.8, 0.7, 1.1), 110, "metal", "#0ea5e9", 0.2, "ground", True, 15),
    "kick scooter": ("box", (0.9, 0.4, 0.9), 5, "metal", "#0ea5e9", 0.3, "ground", False, 4),
    "wheelchair": ("box", (1, 0.65, 0.9), 15, "metal", "#6b7280", 0.3, "ground", False, 1.5),
    "stroller": ("box", (0.9, 0.6, 1.05), 10, "metal", "#6b7280", 0.3, "ground", False, 1.5),
    "wagon": ("box", (1, 0.5, 0.4), 12, "metal", "#dc2626", 0.3, "ground", False, 1.5),
    "cart": ("box", (1, 0.6, 0.5), 15, "wood", "#a16207", 0.3, "ground", False, 1.5),
    "trolley": ("box", (1, 0.6, 1), 20, "metal", "#9ca3af", 0.3, "ground", False, 1.5),
    "sled": ("box", (1.2, 0.5, 0.3), 8, "wood", "#b45309", 0.3, "ground", False, 5),
    "helicopter": ("box", (13, 3, 4), 3000, "metal", "#1f2937", 0.2, "air", True, 60),
    "jet": ("capsule", (20, 3, 3), 15000, "metal", "#cbd5e1", 0.2, "air", True, 250),
    "fighter jet": ("capsule", (19, 3, 3), 15000, "metal", "#6b7280", 0.2, "air", True, 300),
    "glider": ("box", (7, 15, 1.2), 300, "metal", "#f8fafc", 0.2, "air", True, 30),
    "hot air balloon": ("sphere", (18, 18, 18), 400, "fabric", "#ef4444", 0.1, "air", False, 3),
    "balloon": ("sphere", (0.3, 0.3, 0.3), 0.003, "rubber", "#ef4444", 0.6, "air", False, 1),
    "paper plane": ("box", (0.25, 0.12, 0.03), 0.005, "paper", "#f8fafc", 0.2, "air", False, 5),
    "kite": ("box", (0.8, 0.8, 0.05), 0.2, "fabric", "#f43f5e", 0.2, "air", False, 5),
    "ship": ("box", (150, 20, 25), 2.0e7, "metal", "#1e3a8a", 0.1, "water", True, 10),
    "cruise ship": ("box", (300, 40, 60), 1.0e8, "metal", "#f8fafc", 0.1, "water", True, 10),
    "titanic": ("box", (269, 28, 53), 5.2e7, "metal", "#111827", 0.1, "water", True, 11),
    "canoe": ("box", (5, 0.9, 0.4), 30, "wood", "#b45309", 0.2, "water", False, 2),
    "kayak": ("box", (4, 0.6, 0.35), 25, "plastic", "#f97316", 0.2, "water", False, 2),
    "raft": ("box", (3, 1.5, 0.4), 50, "rubber", "#facc15", 0.2, "water", False, 1),
    "jet ski": ("box", (3, 1.2, 1.1), 350, "plastic", "#0ea5e9", 0.2, "water", True, 25),
    "torpedo": ("capsule", (6, 0.53, 0.53), 1600, "metal", "#4b5563", 0.2, "water", True, 25),
    "space shuttle": ("capsule", (37, 8, 8), 78000, "metal", "#f8fafc", 0.2, "space", True, 7800),
    "space station": ("box", (109, 73, 20), 420000, "metal", "#cbd5e1", 0.2, "space", False, 0),
    "cannonball": ("sphere", (0.15, 0.15, 0.15), 14, "metal", "#1f2937", 0.2, "ground", False, 150),
    "cannon": ("cylinder", (2.5, 0.6, 0.6), 1500, "metal", "#374151", 0.2, "ground", False, 0),
    "bullet": ("capsule", (0.03, 0.009, 0.009), 0.008, "metal", "#b45309", 0.2, "air", False, 370),
    "arrow": ("capsule", (0.75, 0.01, 0.01), 0.025, "wood", "#a16207", 0.2, "air", False, 90),
    "spear": ("capsule", (2.2, 0.03, 0.03), 0.8, "wood", "#a16207", 0.2, "air", False, 30),
    "javelin": ("capsule", (2.6, 0.03, 0.03), 0.8, "metal", "#e5e7eb", 0.2, "air", False, 30),
    "shot put": ("sphere", (0.12, 0.12, 0.12), 7.26, "metal", "#4b5563", 0.15, "ground", False, 14),
    "discus": ("cylinder", (0.22, 0.22, 0.045), 2, "metal", "#9ca3af", 0.2, "ground", False, 25),
    "hammer": ("box", (0.33, 0.1, 0.03), 0.6, "metal", "#6b7280", 0.25, "table", False, 8),
    "sledgehammer": ("box", (0.9, 0.15, 0.1), 5, "metal", "#6b7280", 0.25, "ground", False, 8),
    "axe": ("box", (0.7, 0.2, 0.04), 2, "metal", "#6b7280", 0.25, "ground", False, 8),
    "sword": ("box", (1, 0.15, 0.03), 1.4, "metal", "#cbd5e1", 0.3, "ground", False, 10),
    "shield": ("cylinder", (0.7, 0.7, 0.05), 5, "metal", "#9ca3af", 0.3, "ground", False, 3),
    "wrecking ball": ("sphere", (1.2, 1.2, 1.2), 5000, "metal", "#1f2937", 0.15, "ground", False, 5),
    # sports gear
    "rugby ball": ("ellipsoid", (0.29, 0.19, 0.19), 0.44, "leather", "#92400e", 0.6, "ground", False, 25),
    "american football": ("ellipsoid", (0.28, 0.17, 0.17), 0.42, "leather", "#92400e", 0.6, "ground", False, 25),
    "hockey puck": ("cylinder", (0.076, 0.076, 0.025), 0.17, "rubber", "#111827", 0.4, "table", False, 40),
    "shuttlecock": ("box", (0.065, 0.065, 0.085), 0.005, "fabric", "#f8fafc", 0.2, "air", False, 50),
    "badminton shuttlecock": ("box", (0.065, 0.065, 0.085), 0.005, "fabric", "#f8fafc", 0.2, "air", False, 50),
    "softball": ("sphere", (0.097,) * 3, 0.19, "leather", "#fef08a", 0.5, "table", False, 30),
    "dodgeball": ("sphere", (0.21,) * 3, 0.3, "rubber", "#dc2626", 0.7, "ground", False, 15),
    "medicine ball": ("sphere", (0.35,) * 3, 5, "rubber", "#1f2937", 0.3, "ground", False, 5),
    "yoga ball": ("sphere", (0.65,) * 3, 1.2, "rubber", "#60a5fa", 0.75, "ground", False, 4),
    "exercise ball": ("sphere", (0.65,) * 3, 1.2, "rubber", "#60a5fa", 0.75, "ground", False, 4),
    "handball": ("sphere", (0.19,) * 3, 0.45, "leather", "#f8fafc", 0.65, "ground", False, 20),
    "water polo ball": ("sphere", (0.22,) * 3, 0.43, "rubber", "#facc15", 0.65, "water", False, 15),
    "baseball bat": ("cylinder", (0.07, 0.07, 0.85), 0.9, "wood", "#d6b98c", 0.4, "ground", False, 30),
    "cricket bat": ("box", (0.86, 0.11, 0.06), 1.2, "wood", "#e7d3a6", 0.4, "ground", False, 30),
    "tennis racket": ("box", (0.69, 0.27, 0.03), 0.3, "metal", "#1f2937", 0.4, "ground", False, 30),
    "golf club": ("cylinder", (0.03, 0.03, 1.1), 0.35, "metal", "#9ca3af", 0.4, "ground", False, 45),
    "hockey stick": ("cylinder", (0.03, 0.03, 1.5), 0.5, "wood", "#a16207", 0.4, "ground", False, 30),
    "skis": ("box", (1.7, 0.1, 0.05), 3, "wood", "#ef4444", 0.3, "ground", False, 15),
    "surfboard": ("box", (2.1, 0.55, 0.07), 3.5, "plastic", "#22d3ee", 0.3, "water", False, 5),
    "boomerang": ("box", (0.4, 0.4, 0.01), 0.1, "wood", "#d97706", 0.3, "air", False, 25),
    "yo-yo": ("cylinder", (0.06, 0.06, 0.04), 0.06, "plastic", "#ef4444", 0.4, "table", False, 3),
    "spinning top": ("cylinder", (0.05, 0.05, 0.06), 0.03, "wood", "#f59e0b", 0.4, "table", False, 1),
    "domino": ("box", (0.048, 0.024, 0.0075), 0.012, "plastic", "#f8fafc", 0.3, "table", False, 1),
    "jenga block": ("box", (0.075, 0.025, 0.015), 0.014, "wood", "#e7d3a6", 0.3, "table", False, 1),
    "toy train": ("box", (0.2, 0.06, 0.08), 0.3, "plastic", "#dc2626", 0.3, "table", True, 0.5),
    "teddy bear": ("ellipsoid", (0.3, 0.2, 0.35), 0.4, "fabric", "#a16207", 0.1, "table", False, 3),
    "doll": ("capsule", (0.1, 0.06, 0.3), 0.2, "plastic", "#fda4af", 0.3, "table", False, 3),
    "action figure": ("capsule", (0.06, 0.04, 0.15), 0.06, "plastic", "#2563eb", 0.4, "table", False, 3),
    # everyday
    "keyboard": ("box", (0.44, 0.13, 0.03), 0.9, "plastic", "#1f2937", 0.3, "table", False, 2),
    "mouse": ("ellipsoid", (0.09, 0.03, 0.035), 0.02, "flesh", "#a8a29e", 0.3, "ground", True, 3),
    "computer mouse": ("ellipsoid", (0.11, 0.06, 0.04), 0.09, "plastic", "#1f2937", 0.3, "table", False, 2),
    "monitor": ("box", (0.6, 0.2, 0.45), 5, "plastic", "#111827", 0.3, "table", False, 1),
    "tv": ("box", (1.2, 0.08, 0.7), 15, "plastic", "#111827", 0.3, "ground", False, 1),
    "television": ("box", (1.2, 0.08, 0.7), 15, "plastic", "#111827", 0.3, "ground", False, 1),
    "tablet": ("box", (0.25, 0.17, 0.007), 0.47, "glass", "#111827", 0.3, "table", False, 2),
    "headphones": ("box", (0.18, 0.08, 0.2), 0.25, "plastic", "#111827", 0.3, "table", False, 2),
    "wallet": ("box", (0.11, 0.09, 0.02), 0.1, "leather", "#78350f", 0.2, "table", False, 2),
    "backpack": ("box", (0.32, 0.18, 0.45), 1.2, "fabric", "#1d4ed8", 0.2, "ground", False, 2),
    "suitcase": ("box", (0.45, 0.25, 0.7), 4, "plastic", "#1f2937", 0.2, "ground", False, 2),
    "umbrella": ("cylinder", (0.06, 0.06, 0.9), 0.45, "fabric", "#111827", 0.3, "ground", False, 2),
    "bucket": ("cylinder", (0.3, 0.3, 0.3), 1, "plastic", "#2563eb", 0.3, "ground", False, 1),
    "barrel": ("cylinder", (0.6, 0.6, 0.9), 15, "wood", "#92400e", 0.25, "ground", False, 2),
    "oil drum": ("cylinder", (0.57, 0.57, 0.88), 20, "metal", "#1d4ed8", 0.25, "ground", False, 2),
    "tire": ("cylinder", (0.65, 0.65, 0.2), 10, "rubber", "#111827", 0.6, "ground", False, 5),
    "wheel": ("cylinder", (0.65, 0.65, 0.2), 15, "rubber", "#111827", 0.5, "ground", False, 5),
    "plank": ("box", (2, 0.2, 0.04), 6, "wood", "#d6b98c", 0.3, "ground", False, 1),
    "ladder": ("box", (0.45, 0.08, 3), 10, "metal", "#9ca3af", 0.3, "ground", False, 1),
    "stool": ("cylinder", (0.35, 0.35, 0.45), 3, "wood", "#a16207", 0.3, "ground", False, 1),
    "sofa": ("box", (2, 0.9, 0.85), 60, "fabric", "#64748b", 0.15, "ground", False, 1),
    "couch": ("box", (2, 0.9, 0.85), 60, "fabric", "#64748b", 0.15, "ground", False, 1),
    "bed": ("box", (2, 1.5, 0.5), 50, "wood", "#e7e5e4", 0.2, "ground", False, 1),
    "desk": ("box", (1.2, 0.6, 0.75), 30, "wood", "#a16207", 0.25, "ground", False, 1),
    "bookshelf": ("box", (0.9, 0.3, 1.8), 40, "wood", "#a16207", 0.25, "ground", False, 1),
    "cabinet": ("box", (0.9, 0.5, 1.8), 50, "wood", "#a16207", 0.25, "ground", False, 1),
    "washing machine": ("box", (0.6, 0.6, 0.85), 70, "metal", "#f8fafc", 0.2, "ground", False, 1),
    "microwave": ("box", (0.5, 0.4, 0.3), 12, "metal", "#e5e7eb", 0.25, "table", False, 1),
    "toaster": ("box", (0.3, 0.18, 0.2), 2, "metal", "#cbd5e1", 0.3, "table", False, 1),
    "kettle": ("cylinder", (0.2, 0.2, 0.25), 1.2, "metal", "#cbd5e1", 0.3, "table", False, 1),
    "pan": ("cylinder", (0.28, 0.28, 0.05), 1.2, "metal", "#374151", 0.3, "table", False, 2),
    "frying pan": ("cylinder", (0.28, 0.28, 0.05), 1.2, "metal", "#374151", 0.3, "table", False, 2),
    "plate": ("cylinder", (0.26, 0.26, 0.02), 0.5, "ceramic", "#f8fafc", 0.3, "table", False, 2),
    "bowl": ("cylinder", (0.16, 0.16, 0.07), 0.3, "ceramic", "#f8fafc", 0.3, "table", False, 2),
    "spoon": ("box", (0.17, 0.035, 0.015), 0.03, "metal", "#cbd5e1", 0.3, "table", False, 2),
    "fork": ("box", (0.19, 0.025, 0.015), 0.04, "metal", "#cbd5e1", 0.3, "table", False, 2),
    "knife": ("box", (0.22, 0.025, 0.01), 0.07, "metal", "#cbd5e1", 0.3, "table", False, 2),
    "candle": ("cylinder", (0.05, 0.05, 0.15), 0.25, "wax", "#fef3c7", 0.2, "table", False, 1),
    "vase": ("cylinder", (0.12, 0.12, 0.3), 1, "ceramic", "#60a5fa", 0.3, "table", False, 1),
    "flower pot": ("cylinder", (0.2, 0.2, 0.2), 2.5, "ceramic", "#c2410c", 0.25, "table", False, 1),
    "lamp": ("cylinder", (0.25, 0.25, 0.5), 2, "metal", "#fde68a", 0.3, "table", False, 1),
    "clock": ("cylinder", (0.3, 0.3, 0.05), 1, "plastic", "#f8fafc", 0.3, "table", False, 1),
    "paper": ("box", (0.297, 0.21, 0.0001), 0.005, "paper", "#f8fafc", 0.1, "table", False, 1),
    "paperclip": ("box", (0.03, 0.008, 0.001), 0.0005, "metal", "#cbd5e1", 0.3, "table", False, 1),
    "eraser": ("box", (0.05, 0.02, 0.01), 0.02, "rubber", "#fda4af", 0.5, "table", False, 1),
    "ruler": ("box", (0.3, 0.03, 0.003), 0.03, "plastic", "#e0f2fe", 0.3, "table", False, 1),
    "notebook": ("box", (0.25, 0.18, 0.01), 0.3, "paper", "#2563eb", 0.2, "table", False, 1),
    "stapler": ("box", (0.18, 0.04, 0.06), 0.3, "metal", "#111827", 0.3, "table", False, 1),
    "scissors": ("box", (0.2, 0.08, 0.01), 0.1, "metal", "#9ca3af", 0.3, "table", False, 1),
    "battery": ("cylinder", (0.0145, 0.0145, 0.05), 0.023, "metal", "#111827", 0.3, "table", False, 1),
    "magnet": ("box", (0.05, 0.02, 0.01), 0.07, "metal", "#dc2626", 0.3, "table", False, 1),
    "lightbulb": ("ellipsoid", (0.06, 0.06, 0.11), 0.03, "glass", "#fef9c3", 0.3, "table", False, 1),
    "light bulb": ("ellipsoid", (0.06, 0.06, 0.11), 0.03, "glass", "#fef9c3", 0.3, "table", False, 1),
    "ice cube": ("box", (0.025, 0.025, 0.025), 0.014, "ice", "#e0f2fe", 0.3, "table", False, 1),
    "snowball": ("sphere", (0.08,) * 3, 0.15, "ice", "#f8fafc", 0.1, "ground", False, 15),
    "water balloon": ("sphere", (0.1,) * 3, 0.5, "rubber", "#38bdf8", 0.2, "ground", False, 12),
    "pillow fort": ("box", (1.5, 1.5, 1), 5, "fabric", "#e5e7eb", 0.1, "ground", False, 0),
    # food
    "pumpkin": ("ellipsoid", (0.35, 0.35, 0.28), 5, "food", "#ea580c", 0.2, "ground", False, 1),
    "coconut": ("sphere", (0.14,) * 3, 0.7, "shell", "#78350f", 0.35, "table", False, 2),
    "pineapple": ("ellipsoid", (0.14, 0.14, 0.25), 1.5, "food", "#ca8a04", 0.25, "table", False, 2),
    "lemon": ("ellipsoid", (0.075, 0.06, 0.06), 0.1, "food", "#facc15", 0.35, "table", False, 2),
    "grape": ("sphere", (0.02,) * 3, 0.005, "food", "#7c3aed", 0.4, "table", False, 2),
    "strawberry": ("ellipsoid", (0.03, 0.03, 0.035), 0.012, "food", "#dc2626", 0.25, "table", False, 2),
    "potato": ("ellipsoid", (0.1, 0.06, 0.05), 0.2, "food", "#d6b98c", 0.3, "table", False, 2),
    "tomato": ("sphere", (0.065,) * 3, 0.12, "food", "#dc2626", 0.2, "table", False, 2),
    "carrot": ("capsule", (0.18, 0.03, 0.03), 0.07, "food", "#f97316", 0.3, "table", False, 2),
    "cake": ("cylinder", (0.25, 0.25, 0.1), 1.5, "food", "#fde68a", 0.05, "table", False, 1),
    "pizza": ("cylinder", (0.3, 0.3, 0.02), 0.8, "food", "#f59e0b", 0.05, "table", False, 2),
    "donut": ("cylinder", (0.09, 0.09, 0.03), 0.06, "food", "#f9a8d4", 0.2, "table", False, 2),
    "burger": ("cylinder", (0.11, 0.11, 0.07), 0.25, "food", "#b45309", 0.1, "table", False, 2),
    "bread": ("box", (0.25, 0.12, 0.1), 0.5, "food", "#d6b98c", 0.15, "table", False, 2),
    "cookie": ("cylinder", (0.07, 0.07, 0.01), 0.02, "food", "#b45309", 0.2, "table", False, 2),
    "cheese": ("box", (0.1, 0.07, 0.04), 0.25, "food", "#facc15", 0.2, "table", False, 2),
    "jelly": ("box", (0.1, 0.1, 0.08), 0.6, "food", "#ef4444", 0.4, "table", False, 1),
    # lab & physics class
    "pendulum bob": ("sphere", (0.05,) * 3, 1.0, "metal", "#9ca3af", 0.3, "table", False, 1),
    "kilogram weight": ("cylinder", (0.05, 0.05, 0.065), 1.0, "metal", "#9ca3af", 0.3, "table", False, 1),
    "beaker": ("cylinder", (0.07, 0.07, 0.09), 0.1, "glass", "#e0f2fe", 0.3, "table", False, 1),
    "flask": ("cylinder", (0.08, 0.08, 0.15), 0.15, "glass", "#e0f2fe", 0.3, "table", False, 1),
    "test tube": ("cylinder", (0.016, 0.016, 0.15), 0.01, "glass", "#e0f2fe", 0.3, "table", False, 1),
    "trolley cart": ("box", (0.17, 0.1, 0.07), 0.5, "plastic", "#2563eb", 0.4, "table", False, 1),
    "dynamics cart": ("box", (0.17, 0.1, 0.07), 0.5, "plastic", "#2563eb", 0.4, "table", False, 1),
    "glider block": ("box", (0.2, 0.08, 0.05), 0.2, "metal", "#9ca3af", 0.4, "table", False, 1),
    "newton's cradle ball": ("sphere", (0.025,) * 3, 0.065, "metal", "#cbd5e1", 0.95, "table", False, 1),
    "ball bearing": ("sphere", (0.01,) * 3, 0.004, "metal", "#cbd5e1", 0.8, "table", False, 1),
    "ramp": ("box", (1, 0.3, 0.3), 3, "wood", "#d6b98c", 0.3, "table", False, 0),
    # space
    "sun": ("sphere", (1.39e9,) * 3, 1.989e30, "plasma", "#fbbf24", 0.0, "space", False, 0),
    "mars": ("sphere", (6.78e6,) * 3, 6.42e23, "stone", "#c2410c", 0.1, "space", False, 0),
    "jupiter": ("sphere", (1.4e8,) * 3, 1.9e27, "gas", "#d6a77a", 0.0, "space", False, 0),
    "comet": ("ellipsoid", (5000, 3000, 3000), 1.0e13, "ice", "#e0f2fe", 0.1, "space", False, 30000),
    "space rock": ("ellipsoid", (2, 1.5, 1.2), 5000, "stone", "#57534e", 0.2, "space", False, 20000),
    "black hole": ("sphere", (60000,) * 3, 2.0e31, "unknown", "#000000", 0.0, "space", False, 0),
})

# things that are built into the ground: they don't get launched or wander; they stand and take the hit
FIXED = {"house", "castle", "wall", "brick wall", "tree", "building", "skyscraper", "bridge", "lighthouse",
         "pyramid", "lamp post", "fence", "door", "window", "goal", "mountain", "cannon", "space station"}


def from_table(key: str, name: str, note="") -> Thing:
    shape, dims, mass, mat, color, bounce, hab, prop, speed = _T[key]
    t = Thing(name, "knowledge", shape, list(dims), mass, mat, color, bounce, hab, prop, speed, note=note or f"real-world typical {key}")
    if key in FIXED:
        t.extra["fixed"] = True
    return t


def resolve(name: str, scene=None, scene_labels: dict | None = None, allow_estimate: bool = True) -> Thing:
    raw = name.strip()
    n = _singular(raw)
    # 1. a real scanned object
    for oid, lab in (scene_labels or {}).items():
        if n and (n == lab.lower() or n in lab.lower().split() or lab.lower() in n):
            o = next((x for x in scene.objects if x.id == oid), None) if scene is not None else None
            if o is not None:
                return Thing(raw, "scene", "scene", list(o.size), o.mass, o.material, "#ffffff", float(getattr(o, "bounciness", 0.4)), "table", False, 1.0,
                             scene_id=oid, note="the real object from your camera")
    # 2. library model - only when it is the same kind of thing at the same scale
    #    ("car" is a real car, not the 15 cm toy car; "baseball" uses the real baseball model)
    lib = asset_lib.manifest()
    for a in lib.values():
        if n in [a["name"].lower(), *[x.lower() for x in a["aliases"]]]:
            kb = _T.get(n)
            if kb is None or 0.5 < max(a["dims"]) / max(kb[1]) < 2.0:
                t = _from_asset(a, raw)
                if kb is not None:            # keep the real-world physical properties
                    t.mass, t.bounciness, t.material, t.typical_speed, t.habitat = kb[2], kb[5], kb[3], kb[8], kb[6]
                return t
    # 3. knowledge table (exact, then semantic)
    if n in _T:
        return from_table(n, raw)
    for k in sorted(_T, key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", n):
            return from_table(k, raw)
    lib_names = {a["name"]: a for a in lib.values()}
    best_lib, s_lib = _semantic_best(n, list(lib_names))
    best_kb, s_kb = _semantic_best(n, list(_T))
    if s_lib >= 0.72 and s_lib >= s_kb:
        return _from_asset(lib_names[best_lib], raw)
    if s_kb >= 0.62:
        return from_table(best_kb, raw, note=f"closest known: {best_kb}")
    # 4. on-device model estimate, checked for physical sanity
    if allow_estimate:
        t = _estimate(raw)
        if t is not None:
            return t
    # last resort: an honest generic stand-in
    return Thing(raw, "estimate", "box", [0.2, 0.2, 0.2], 1.0, "unknown", "#94a3b8", 0.3, "ground", False, 1.0,
                 note="unknown object: generic 20 cm, 1 kg stand-in (edit size/mass)")


def _from_asset(a: dict, raw: str) -> Thing:
    known = _T.get(a["name"], None)
    return Thing(raw, "library", "asset", list(a["dims"]), a["mass"], a.get("material") or "unknown", "#ffffff",
                 0.55 if a.get("bouncy") else 0.25, "table", a["kind"] == "vehicle", known[8] if known else 1.5,
                 asset_id=a["id"], note=f"3D model: {a['name']}")


def _estimate(name: str) -> Thing | None:
    from . import local_ai
    if not local_ai.available():
        return None
    prompt = (f"Give typical real-world physical properties of: {name}. Answer ONLY JSON like "
              '{"length_m":0.3,"width_m":0.2,"height_m":0.1,"mass_kg":1.2,"shape":"box|sphere|cylinder|capsule|ellipsoid",'
              '"habitat":"table|ground|water|air|space","moves_by_itself":false,"color":"#808080"}')
    d = local_ai.extract_json(local_ai.generate(prompt, max_tokens=120))
    try:
        dims = [float(d["length_m"]), float(d["width_m"]), float(d["height_m"])]
        mass = float(d["mass_kg"])
    except (TypeError, KeyError, ValueError):
        return None
    if not all(1e-4 < x < 1e5 for x in dims) or not (1e-7 < mass < 1e12):
        return None
    shape = d.get("shape") if d.get("shape") in ("box", "sphere", "cylinder", "capsule", "ellipsoid") else "box"
    vol = dims[0] * dims[1] * dims[2] * (0.52 if shape in ("sphere", "ellipsoid") else 1.0)
    rho = mass / max(vol, 1e-12)
    if not (5 <= rho <= 25000):          # physically implausible density: reject the estimate
        return None
    color = d.get("color") if isinstance(d.get("color"), str) and re.fullmatch(r"#[0-9a-fA-F]{6}", d.get("color", "")) else "#94a3b8"
    hab = d.get("habitat") if d.get("habitat") in ("table", "ground", "water", "air", "space") else "ground"
    return Thing(name, "estimate", shape, dims, mass, "unknown", color, 0.3, hab, bool(d.get("moves_by_itself")), 2.0,
                 note="size and mass estimated by the on-device AI (check and edit)")


def setting_for(text: str, things: list[Thing], has_scene: bool) -> str:
    t = text.lower()
    for k, pat in SETTING_WORDS.items():
        if re.search(pat, t):
            return k
    habs = {x.habitat for x in things}
    if "space" in habs:
        return "space"
    if "water" in habs:
        return "water"
    biggest = max((max(x.dims) for x in things), default=0.1)
    if has_scene and biggest <= 0.6 and all(x.habitat in ("table",) or x.source in ("scene", "library") for x in things):
        return "scene"
    if biggest <= 0.6:
        return "table"
    return "ground"


# ---------------------------------------------------------------------------
# archetypes: how a thing should LOOK and move when there's no 3D model for it
# ---------------------------------------------------------------------------
_ARCH_WORDS = [
    ("castle", r"castle|fortress|palace"),
    ("building", r"house|building|skyscraper|tower|lighthouse|pyramid|hut|barn|church|temple|shop|school|hospital"),
    ("wall", r"wall|fence|door|window|goal|barrier"),
    ("tree", r"tree|palm|pine|oak|bush"),
    ("ball", r"ball|marble|puck|orange|apple|grape|tomato|coconut|melon|planet|earth|moon|mars|jupiter|sun|globe|snowball|bead|bearing|bob\b|cannonball|shot put|egg"),
    ("fish", r"whale|orca|dolphin|shark|fish|salmon|tuna|seal|squid|submarine|torpedo|jellyfish"),
    ("winged", r"dragon|eagle|bird|owl|parrot|pigeon|bat\b|bee|butterfly|fly|mosquito|duck|swan|chicken|penguin"),
    ("aircraft", r"plane|jet|airplane|glider|helicopter|drone|ufo|spaceship|spacecraft|shuttle|rocket|satellite|space station"),
    ("boat", r"boat|ship|canoe|kayak|raft|jet ski|titanic|yacht"),
    ("vehicle", r"car|truck|bus|van|jeep|suv|taxi|tractor|bulldozer|excavator|forklift|tank|train|kart|motorcycle|scooter|bicycle|bike|ambulance|cart|trolley|wagon|stroller|wheelchair|skateboard|sled"),
    ("biped", r"man|woman|person|human|child|boy|girl|baby|student|athlete|wrestler|boxer|knight|ninja|astronaut|robot|zombie|alien|giant|gorilla|monkey|kangaroo"),
    ("snake", r"snake|worm|eel"),
    ("quadruped", r"dog|cat|horse|cow|elephant|rhino|bull|lion|tiger|bear|mouse|rat|giraffe|hippo|wolf|fox|deer|zebra|camel|pig|sheep|goat|rabbit|squirrel|crocodile|alligator|turtle|tortoise|frog|panda|cheetah|leopard|dinosaur|t-rex|tyrannosaurus|mammoth|unicorn|monster|godzilla|lizard|spider|beetle|ant|crab|lobster"),
    ("rock", r"rock|boulder|stone|asteroid|meteor|comet|iceberg|space rock"),
    ("anchor", r"^support$"),
]


def archetype(name: str, t: "Thing | None" = None) -> str:
    n = name.lower()
    for arch, pat in _ARCH_WORDS:
        if re.search(rf"\b(?:{pat})", n):
            return arch
    if t is not None:
        if t.shape == "sphere":
            return "ball"
        if t.self_propelled:
            return {"water": "fish", "air": "winged"}.get(t.habitat, "quadruped")
    return "generic"
