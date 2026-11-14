"""Live AR: what you say to the mapped world acts ON that world.

  "push the mug", "push the bottle into the book", "slide the box left", "knock the bottle over",
  "throw the block at the mug", "drop the box from 30 cm", "lift the cup", "remove the book",
  "what if the mug wasn't there", "make the mug twice as heavy", "make the book icy", "spin it", "... on the Moon"

Deterministic grounding on the CURRENT state (end of the last simulation, so it continues what you see): objects are
found by their names (or "it" = the selected / last one), directions are camera-relative, throws are solved
ballistically to land on the target, pushes toward something are just hard enough to get there (v = sqrt(2 mu g d)).
Returns None when the sentence isn't a command on an existing object (then the experiment engine takes it).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from . import assets as asset_lib

G = 9.81
LAST = {"id": None}            # "it" after a command = the thing it acted on


@dataclass
class Obj:
    id: str
    label: str
    base: np.ndarray           # body origin (scanned objects: centre; added objects: bottom centre)
    centre: np.ndarray
    half: np.ndarray
    mass: float
    friction: float
    above_bottom: float        # body origin height above the object's bottom


def objects(scene, sim: dict | None, interventions) -> list[Obj]:
    removed = {iv.target for iv in interventions if iv.type == "remove"}
    last = {oid: sim["frames"][-1][i] for i, oid in enumerate(sim["object_ids"])} if sim else {}
    masses = (sim or {}).get("masses", {})
    out = []
    for o in scene.objects:
        if o.id in removed or (sim and o.id in sim.get("removed_at", {})):
            continue
        b = np.array(last[o.id][:3]) if o.id in last else np.array(o.pos, float)
        h = np.array(o.size, float) / 2
        out.append(Obj(o.id, o.label, b, b.copy(), h, float(masses.get(o.id, o.mass)), float(o.friction), float(h[2])))
    lib = None
    for sid, sp in ((sim or {}).get("spawned") or {}).items():
        if sid in removed or sid in sim.get("removed_at", {}) or sid not in last:
            continue
        dims = (sp.get("proxy") or {}).get("dims")
        if dims is None:
            lib = lib or asset_lib.manifest()
            dims = lib.get(sp["asset"], {}).get("dims")
        if dims is None:
            continue
        b = np.array(last[sid][:3])
        h = np.array(dims, float) / 2
        out.append(Obj(sid, sp.get("label") or sim.get("labels", {}).get(sid, sid), b, b + np.array([0, 0, h[2]]), h,
                       float(masses.get(sid, 0.2)), 0.5, 0.0))
    return out


_STOP = {"the", "a", "an", "my", "that", "this", "real", "one", "object", "thing"}


def _words(s: str) -> set[str]:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z]+", s.lower()) if w not in _STOP}


def find(fragment: str, objs: list[Obj], selected: str | None) -> Obj | None:
    f = fragment.strip().lower()
    if re.fullmatch(r"(it|that|this|this one|that one|the object|the thing|them)", f):
        pick = selected or LAST["id"]
        return next((o for o in objs if o.id == pick), None)
    if re.match(r"(a|an|some|another|two|three|four|five|\d+)\b", f):
        return None                                         # "a baseball": something new, not one of your objects
    fw = _words(f)
    if not fw:
        return None
    best, score = None, 0.0
    for o in objs:
        lw = _words(o.label)
        common = len(fw & lw)
        if not common:
            continue
        sc = common / len(lw) + 0.5 * common / len(fw) + (0.1 if o.id == selected else 0)
        if sc > score:
            best, score = o, sc
    return best


def _cam_axes(scene):
    R = np.array(scene.camera.R_wc, float)
    right = R[:, 0].copy(); right[2] = 0
    fwd = R[:, 2].copy(); fwd[2] = 0
    return right / max(1e-9, np.linalg.norm(right)), fwd / max(1e-9, np.linalg.norm(fwd))


def _direction(t: str, scene) -> tuple[np.ndarray | None, str]:
    right, fwd = _cam_axes(scene)
    for pat, v, name in ((r"\b(to the )?left\b", -right, "to the left"), (r"\b(to the )?right\b", right, "to the right"),
                         (r"\b(toward|towards|at) (me|the camera|you)\b|\bcloser\b|\bforward\b", -fwd, "toward the camera"),
                         (r"\baway\b|\bback(wards?)?\b|\bfurther\b", fwd, "away from the camera")):
        if re.search(pat, t):
            return v, name
    return None, ""


def _speed(t: str, default: float) -> tuple[float, bool]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m/s|meters? per second|metres? per second|km/h|kph|mph)", t)
    if m:
        v = float(m.group(1)) * {"km/h": 1 / 3.6, "kph": 1 / 3.6, "mph": 0.44704}.get(m.group(2), 1.0)
        return v, True
    for w, v in (("very gently", 0.2), ("gently", 0.3), ("slowly", 0.3), ("softly", 0.3), ("lightly", 0.35), ("very hard", 2.5),
                 ("really hard", 2.5), ("as hard as", 3.0), ("hard", 1.6), ("fast", 1.6), ("quickly", 1.4), ("strongly", 1.6)):
        if re.search(rf"\b{w}\b", t):
            return v, True
    return default, False


def _length(t: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(cm|centimet\w*|mm|millimet\w*|met(?:er|re)s?|m)\b(?!/)", t)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * 0.01 if u.startswith("c") else v * 0.001 if u.startswith("m") and u not in ("m",) and not u.startswith("met") else v


GRAVITY = {"moon": 1.62, "mars": 3.71, "jupiter": 24.79, "venus": 8.87, "mercury": 3.7, "pluto": 0.62, "earth": 9.81}


def parse(text: str, scene, sim: dict | None, interventions, selected: str | None = None) -> dict | None:
    """-> {"ivs": [...], "say": str, "target": id, "start": [x,y,z], "label": str} or None."""
    t = " " + re.sub(r"[?!.,]+", " ", text.lower()) + " "
    t = re.sub(r"^\s*(what if|what would happen if|now|then|ok|okay|please|can you|could you|let'?s|i want to|try to)\s+", " ", t).strip() + " "
    objs = objects(scene, sim, interventions)
    if not objs:
        return None
    g_planet = next((p for p in GRAVITY if re.search(rf"\b(on|at|with|in) (the )?{p}\b|\b{p}('s)? gravity\b", t)), None)
    if re.search(r"\b(zero|no) gravity\b|\bweightless\b|\bin space\b", t):
        g_planet = "zero"
    gval = 0.0 if g_planet == "zero" else GRAVITY.get(g_planet or "earth", G)
    import uuid
    group = f"say-{uuid.uuid4().hex[:8]}"                    # unique: the same sentence twice is two actions
    pre: list[dict] = []
    if g_planet and g_planet != "earth":
        pre.append({"type": "set_gravity", "target": None, "params": {"g": gval, "group": group}})
    g_txt = "" if not g_planet or g_planet == "earth" else (" in zero gravity" if g_planet == "zero" else f" with {g_planet.title()}'s gravity ({gval:g} m/s²)")
    t = re.sub(r"\b(on|at|with|in) (the )?(moon|mars|jupiter|venus|mercury|pluto|earth)\b|\b(moon|mars|jupiter)('s)? gravity\b|\b(zero|no) gravity\b|\bin space\b", " ", t)

    def out(target: Obj, ivs: list[dict], say: str) -> dict:
        LAST["id"] = target.id
        for iv in ivs:
            iv["params"]["group"] = group
        return {"ivs": pre + ivs, "say": say + g_txt, "target": target.id, "start": target.base.round(4).tolist(), "label": target.label}

    DIRS = r"(?:the\s+)?(?:left|right|me|you|camera|the camera|front|back|side|other side|edge|floor|ground|table|desk)"

    def grab(pattern: str):
        m = re.search(pattern, t)
        if not m:
            return None
        d = m.groupdict()
        if d.get("x"):
            d["x"] = re.split(r"\s+(?:on|onto|into|at|from|to|toward|towards|off|in|over|under|with|across|by)\s+", " " + d["x"].strip() + " ")[0].strip()
        if d.get("y"):
            d["y"] = re.sub(r"\b(very|really|gently|softly|slowly|lightly|quickly|fast|hard|at \d.*)\b", " ", d["y"]).strip()
        if d.get("y") is not None and (not d["y"] or re.fullmatch(DIRS, d["y"])):
            d["y"] = None                                   # "push it to the left": a direction, not an object
        return d

    # ---- remove / "what if X wasn't there"
    m = (grab(r"\b(?:remove|delete|take away|take out|get rid of|lose)\s+(?P<x>.+?)\s*$")
         or grab(r"\b(?:the\s+)?(?P<x>[\w ]+?)\s+(?:was|were|is)(?:n't|n’t| not)\s+there\b")
         or grab(r"\b(?:the\s+)?(?P<x>[\w ]+?)\s+(?:was|were|is)\s+(?:gone|missing|removed)\b")
         or grab(r"\bwithout\s+(?P<x>.+?)\s*$") or grab(r"\bmake\s+(?P<x>.+?)\s+(?:disappear|vanish)\b"))
    if m:
        o = find(m["x"], objs, selected)
        if o:
            return out(o, [{"type": "remove", "target": o.id, "params": {}}], f"Removed the {o.label} — everything it was holding up is now on its own")

    # ---- knock over / topple
    m = grab(r"\b(?:knock|tip|push|topple|tilt)\s+(?P<x>.+?)\s+over\b") or grab(r"\b(?:knock over|tip over|topple|overturn)\s+(?P<x>.+?)\s*$")
    if m:
        o = find(m["x"], objs, selected)
        if o:
            d, dname = _direction(t, scene)
            if d is None:
                d = _cam_axes(scene)[1]; dname = "away from the camera"
            v, _ = _speed(t, 1.2)
            point = o.centre + np.array([0, 0, 0.8 * o.half[2]])
            return out(o, [{"type": "push", "target": o.id, "params": {"direction": d.round(4).tolist(), "speed": v, "point": point.round(4).tolist()}}],
                       f"Pushed the {o.label} near its top, {dname}, at {v:.2f} m/s")

    # ---- throw X at Y / throw X <direction>
    m = grab(r"\b(?:throw|toss|lob|hurl|fling|chuck)\s+(?P<x>.+?)\s+(?:at|to|toward|towards|onto|on|into)\s+(?P<y>.+?)\s*$") or \
        grab(r"\b(?:throw|toss|lob|hurl|fling|chuck)\s+(?P<x>.+?)\s*$")
    if m:
        x = re.sub(r"\b(to the left|to the right|left|right|away|forward|toward me|hard|gently|fast|slowly|really|very)\b", " ", m["x"])
        o = find(x, objs, selected)
        if o:
            start = o.base + np.array([0, 0, max(0.12, 2 * o.half[2])])
            y = find(m.get("y") or "", [p for p in objs if p.id != o.id], None) if m.get("y") else None
            if y is not None:
                # aim at its upper half and come down onto it (a flat throw clips whatever it stands on)
                aim = y.centre + np.array([0, 0, y.half[2] * 0.6])
                delta = aim - (start - o.base + o.centre)
                dist = float(np.linalg.norm(delta[:2]))
                T = min(0.9, max(0.35, 0.25 + dist / 1.2)) if gval > 0 else max(0.3, dist / 1.5)
                vel = delta / T + np.array([0, 0, 0.5 * gval * T])
                spd = float(np.linalg.norm(vel))
                ang = math.degrees(math.atan2(vel[2], math.hypot(vel[0], vel[1])))
                say = f"Threw the {o.label} at the {y.label}: {spd:.2f} m/s, {ang:.0f}° up — solved so it lands on it after {T:.2f} s"
            else:
                d, dname = _direction(t, scene)
                if d is None:
                    d = _cam_axes(scene)[1]; dname = "away from the camera"
                v, _ = _speed(t, 1.5)
                vel = d * v + np.array([0, 0, 0.6 * v])
                spd = float(np.linalg.norm(vel)); say = f"Threw the {o.label} {dname} at {spd:.2f} m/s"
            return out(o, [{"type": "place", "target": o.id, "params": {"pos": start.round(4).tolist(), "hold": True, "clear": True}},
                           {"type": "launch", "target": o.id, "params": {"velocity": vel.round(4).tolist()}}], say)

    # ---- drop X (from h) / lift X (by h)
    m = grab(r"\b(?:drop|let go of|release)\s+(?P<x>.+?)\s*$")
    if m:
        h = _length(t) or 0.2
        x = re.sub(r"\bfrom\b.*$", "", m["x"])
        o = find(x, objs, selected)
        if o:
            pos = o.base + np.array([0, 0, h])
            return out(o, [{"type": "place", "target": o.id, "params": {"pos": pos.round(4).tolist(), "hold": False, "clear": True}}],
                       f"Lifted the {o.label} {h * 100:.0f} cm and let it fall")
    m = grab(r"\b(?:lift|raise|pick up|hold up)\s+(?P<x>.+?)\s*$")
    if m:
        h = _length(t) or 0.1
        o = find(re.sub(r"\b(by|up)\b.*$", "", m["x"]), objs, selected)
        if o:
            pos = o.base + np.array([0, 0, h])
            return out(o, [{"type": "place", "target": o.id, "params": {"pos": pos.round(4).tolist(), "hold": True, "clear": True}}],
                       f"Lifted the {o.label} {h * 100:.0f} cm and holding it there — say “drop it” to let go")

    # ---- mass / friction
    m = grab(r"\b(?:make|let)\s+(?P<x>.+?)\s+(?P<how>(?:\d+(?:\.\d+)?|twice|double|triple|half|ten|hundred)\s*(?:x|times)?\s*(?:as\s+)?(?:heavy|heavier|light|lighter)|heavier|lighter)\b") \
        or grab(r"\b(?:if\s+)?(?P<x>.+?)\s+(?:was|were|is)\s+(?P<how>(?:\d+(?:\.\d+)?|twice|double|triple|half|ten|hundred)\s*(?:x|times)?\s*(?:as\s+)?(?:heavy|heavier|light|lighter))\b")
    if m:
        o = find(m["x"], objs, selected)
        if o:
            how = m["how"]
            k = {"twice": 2, "double": 2, "triple": 3, "half": 0.5, "ten": 10, "hundred": 100}.get(how.split()[0], None)
            if k is None:
                n = re.match(r"(\d+(?:\.\d+)?)", how)
                k = float(n.group(1)) if n else 2.0
            if "light" in how and k > 1:
                k = 1 / k
            new = o.mass * k
            return out(o, [{"type": "set_mass", "target": o.id, "params": {"mass": round(new, 5)}}],
                       f"The {o.label} is now {new:.3g} kg ({k:g}× its {o.mass:.3g} kg) — push it or throw something at it to feel the difference")
    m = grab(r"\b(?:make|let)\s+(?P<x>.+?)\s+(?P<how>icy|slippery|frictionless|slick|rough|sticky|grippy|rubbery)\b")
    if m:
        o = find(m["x"], objs, selected)
        if o:
            mu = 0.03 if m["how"] in ("icy", "slippery", "frictionless", "slick") else 1.0
            return out(o, [{"type": "set_friction", "target": o.id, "params": {"friction": mu}}],
                       f"The {o.label} is now {m['how']} (friction μ {o.friction:.2f} → {mu:g}) — push it to see it slide")

    # ---- spin
    m = grab(r"\b(?:spin|twirl|rotate|twist)\s+(?P<x>.+?)\s*$")
    if m:
        o = find(re.sub(r"\b(fast|quickly|slowly|gently|hard|around|round)\b", " ", m["x"]), objs, selected)
        if o:
            w = 25.0 if re.search(r"\b(fast|quickly|hard)\b", t) else 8.0 if re.search(r"\b(slowly|gently)\b", t) else 14.0
            return out(o, [{"type": "spin", "target": o.id, "params": {"omega": [0.0, 0.0, w]}}], f"Spun the {o.label} at {w:g} rad/s ({w / (2 * math.pi):.1f} turns a second)")

    # ---- push / slide / shove (X [into|toward|at Y] [direction] [speed])
    m = grab(r"\b(?:push|shove|nudge|slide|kick|flick|bump|move|hit|poke)\s+(?P<x>.+?)\s+(?:into|toward|towards|at|against|onto|to)\s+(?P<y>.+?)\s*$") or \
        grab(r"\b(?:push|shove|nudge|slide|kick|flick|bump|move|hit|poke)\s+(?P<x>.+?)\s*$")
    if m:
        x = re.sub(r"\b(to the left|to the right|left|right|away|forward|backwards?|toward me|towards me|closer|hard|gently|softly|fast|slowly|quickly|lightly|really|very|at \d.*)\b", " ", m["x"])
        o = find(x, objs, selected)
        if o:
            y = find(m.get("y") or "", [p for p in objs if p.id != o.id], None) if m.get("y") else None
            v, said = _speed(t, 0.8)
            if y is not None:
                d = y.centre - o.centre; d[2] = 0
                dist = max(0.0, float(np.linalg.norm(d)) - (max(o.half[:2]) + max(y.half[:2])) * 0.7)
                if not said:          # just hard enough to get there against friction, with a margin
                    v = float(np.clip(1.35 * math.sqrt(2 * max(0.2, o.friction) * max(gval, 0.5) * max(dist, 0.02)), 0.3, 3.0))
                d /= max(1e-9, np.linalg.norm(d))
                say = f"Pushed the {o.label} toward the {y.label} at {v:.2f} m/s" + ("" if said else " (just enough to reach it: v = 1.35·√(2μgd))")
            else:
                d, dname = _direction(t, scene)
                if d is None:
                    d = _cam_axes(scene)[1]; dname = "away from the camera"
                say = f"Pushed the {o.label} {dname} at {v:.2f} m/s"
            return out(o, [{"type": "push", "target": o.id, "params": {"direction": d.round(4).tolist(), "speed": round(v, 3), "point": o.centre.round(4).tolist()}}], say)

    # ---- just a gravity change ("what if we were on the Moon")
    if pre and re.fullmatch(r"\s*(we were|it was|everything was|this was|we are|it is)?\s*", t):
        o = objs[0]
        return {"ivs": pre, "say": f"Switched to {'zero gravity' if g_planet == 'zero' else g_planet.title() + chr(39) + 's gravity'} ({gval:g} m/s²) — now push or throw something",
                "target": o.id, "start": o.base.round(4).tolist(), "label": o.label}
    return None
