"""Material priors: density (kg/m^3), fill factor (hollow objects), friction.

Mass = volume(primitive) * density * fill. These are priors, surfaced in the
UI with their source so the user can override them."""
from __future__ import annotations

MATERIALS: dict[str, dict[str, float]] = {
    "ceramic":   {"density": 2300, "fill": 0.16, "friction": 0.55, "restitution": 0.5},
    "glass":     {"density": 2500, "fill": 0.12, "friction": 0.45, "restitution": 0.6},
    "metal":     {"density": 7800, "fill": 0.05, "friction": 0.40, "restitution": 0.6},
    "aluminum":  {"density": 2700, "fill": 0.05, "friction": 0.40, "restitution": 0.55},
    "plastic":   {"density": 950,  "fill": 0.25, "friction": 0.45, "restitution": 0.6},
    "wood":      {"density": 650,  "fill": 1.00, "friction": 0.55, "restitution": 0.5},
    "paper":     {"density": 750,  "fill": 1.00, "friction": 0.60, "restitution": 0.3},   # books, notebooks
    "cardboard": {"density": 250,  "fill": 0.40, "friction": 0.60, "restitution": 0.3},
    "fabric":    {"density": 300,  "fill": 0.80, "friction": 0.75, "restitution": 0.15},
    "rubber":    {"density": 1100, "fill": 1.00, "friction": 0.90, "restitution": 0.8},
    "food":      {"density": 950,  "fill": 0.90, "friction": 0.60, "restitution": 0.2},
    "foam":      {"density": 60,   "fill": 1.00, "friction": 0.70, "restitution": 0.25},
    "unknown":   {"density": 500,  "fill": 0.50, "friction": 0.50, "restitution": 0.4},
}

CYLINDER_WORDS = ("mug", "cup", "bottle", "can", "jar", "glass", "candle", "tube", "roll", "vase", "tumbler", "flask")


def guess_shape(label: str) -> str | None:
    l = label.lower()
    return "cylinder" if any(w in l for w in CYLINDER_WORDS) else None


def mass_from_volume(shape: str, size: list[float], material: str) -> float:
    m = MATERIALS.get(material, MATERIALS["unknown"])
    if shape == "cylinder":
        import math
        vol = math.pi * (size[0] / 2) ** 2 * size[2]
    else:
        vol = size[0] * size[1] * size[2]
    return max(0.01, vol * m["density"] * m["fill"])


def friction_for(material: str) -> float:
    return MATERIALS.get(material, MATERIALS["unknown"])["friction"]


def restitution_for(material: str) -> float:
    """Coefficient of restitution of the material against a hard surface (rebound speed / impact speed)."""
    return MATERIALS.get(material, MATERIALS["unknown"])["restitution"]


def density_for(label: str, material: str) -> tuple[float, str]:
    """Effective density of an object's OUTER volume (what a scan measures): material density x fill factor
    (a mug or bottle is mostly air inside). Returns (kg/m^3, how it was chosen) for the object's notes."""
    m = MATERIALS.get(material, MATERIALS["unknown"])
    name = material if material in MATERIALS else "unknown material"
    if m["fill"] < 1.0:
        return m["density"] * m["fill"], f"{name} {m['density']:.0f} kg/m³ × {m['fill']:.0%} solid"
    return m["density"], f"{name} {m['density']:.0f} kg/m³"
