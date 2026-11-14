"""Natural-language experiments -> concrete, checkable interventions.

  "run this experiment with a car: drive it into the mug"
      -> spawn toy_car at a free, visible spot with a clear path to the mug, facing it
      -> drive it at 0.6 m/s toward the mug

Two stages, kept separate on purpose:
  1. understanding (what to do): Claude with a strict JSON schema when credentials
     exist, otherwise a deterministic rule parser. Output is an abstract plan.
  2. grounding (where / how exactly): deterministic geometry over the current
     world state: table bounds, free space, a collision-free straight path to the
     target, and visibility in the camera. The planner never invents coordinates.
The resolved interventions are shown to the user before anything runs.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass

import numpy as np

from . import assets as asset_lib
from . import config

log = logging.getLogger(__name__)

SPEED_WORDS = {"very slowly": 0.2, "slowly": 0.3, "slow": 0.3, "gently": 0.3, "quickly": 1.1, "quick": 1.1, "fast": 1.2,
               "very fast": 1.8, "hard": 1.5, "full speed": 2.0}
GRAVITY_WORDS = {"moon": 1.62, "mars": 3.71, "jupiter": 24.79, "earth": 9.81, "zero gravity": 0.0,
                 "no gravity": 0.0, "weightless": 0.0}


@dataclass
class Entity:
    id: str
    label: str
    pos: np.ndarray        # centre (world)
    radius: float          # footprint radius
    top: float             # z of top surface
    spawned: bool = False


# ---------------------------------------------------------------------------
# world snapshot (state at the end of the current simulation)
# ---------------------------------------------------------------------------

def snapshot(scene, sim: dict | None, interventions) -> list[Entity]:
    lib = asset_lib.manifest()
    removed = {iv.target for iv in interventions if iv.type == "remove"}
    last = None
    if sim:
        last = {oid: sim["frames"][-1][i] for i, oid in enumerate(sim["object_ids"])}
    out = []
    for o in scene.objects:
        if o.id in removed:
            continue
        p = np.array(last[o.id][:3]) if last and o.id in last else np.array(o.pos)
        r = 0.5 * math.hypot(o.size[0], o.size[1])
        out.append(Entity(o.id, o.label, p, r, float(p[2] + o.size[2] / 2)))
    for iv in interventions:
        if iv.type == "spawn" and iv.target not in removed:
            px = iv.params.get("proxy") or {}
            a = lib.get(iv.params["asset"]) or {"name": iv.params.get("label", "object"), "dims": px.get("dims", [0.1, 0.1, 0.1])}
            p = np.array(last[iv.target][:3]) if last and iv.target in last else np.array(iv.params["pos"])
            out.append(Entity(iv.target, iv.params.get("label") or a["name"], p,
                              0.5 * math.hypot(a["dims"][0], a["dims"][1]), float(p[2] + a["dims"][2]), spawned=True))
    return out


# ---------------------------------------------------------------------------
# stage 1a: rule-based understanding
# ---------------------------------------------------------------------------

SPATIAL = {"left": (0, -1), "right": (0, 1), "front": (1, -1), "nearest": (1, -1), "closest": (1, -1),
           "back": (1, 1), "far": (1, 1), "farthest": (1, 1), "middle": (0, 0), "center": (0, 0), "centre": (0, 0)}
_CAMERA = None      # set by the API per request so "left/right" means left/right in the camera view


def _image_xy(e: Entity):
    if _CAMERA is None:
        return float(e.pos[0]), float(e.pos[1])
    uv = _project(_CAMERA, e.pos)
    return (uv[0], uv[1]) if uv else (float(e.pos[0]), float(e.pos[1]))


def _find_entity(text: str, ents: list[Entity], exclude: set[str] = frozenset()) -> Entity | None:
    t = text.lower()
    found = _find_entity_plain(t, ents, exclude)
    if found is None:
        return None
    # several objects share the name ("the left mug"): use the spatial word, in camera terms
    same = [e for e in ents if e.id not in exclude and e.label.lower() == found.label.lower()]
    if len(same) > 1:
        for word, (axis, sign) in SPATIAL.items():
            if re.search(rf"\b{word}(most)?\b", t):
                if sign == 0:
                    xs = sorted(same, key=lambda e: _image_xy(e)[0])
                    return xs[len(xs) // 2]
                # image u grows to the right; image v grows toward the camera (lower in the frame)
                key = (lambda e: _image_xy(e)[0]) if axis == 0 else (lambda e: -_image_xy(e)[1])
                return (max if sign > 0 else min)(same, key=key)
    return found


def _find_entity_plain(text: str, ents: list[Entity], exclude: set[str] = frozenset()) -> Entity | None:
    t = text.lower()
    m = re.search(r"object\s*(\d+)", t)
    if m:
        lab = f"object {m.group(1)}"
        for e in ents:
            if e.label.lower() == lab and e.id not in exclude:
                return e
    best, best_len = None, 0
    for e in ents:
        if e.id in exclude:
            continue
        for word in re.findall(r"[a-z]+", e.label.lower()):
            if len(word) > 2 and re.search(rf"\b{word}s?\b", t) and len(word) > best_len and word != "object":
                best, best_len = e, len(word)
    return best


def _find_asset(text: str) -> dict | None:
    t = text.lower()
    best, best_len = None, 0
    for a in asset_lib.manifest().values():
        for alias in [a["name"], *a["aliases"]]:
            if re.search(rf"\b{re.escape(alias)}\b", t) and len(alias) > best_len:
                best, best_len = a, len(alias)
    return best


def _speed(text: str, default: float):
    t = text.lower()
    if re.search(r"\b(full speed|full throttle|top speed|max(imum)? speed|as fast as (it|possible|you can))\b", t):
        return "max"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m/s|meters? per second|cm/s)", t)
    if m:
        v = float(m.group(1))
        return v / 100 if m.group(2) == "cm/s" else v
    for w in sorted(SPEED_WORDS, key=len, reverse=True):
        if w in t:
            return SPEED_WORDS[w]
    return default


def parse_rules(text: str, ents: list[Entity], selected: str | None) -> dict:
    t = text.lower()
    steps, notes = [], []
    for w, g in GRAVITY_WORDS.items():
        if re.search(rf"\b{w}\b", t) and ("gravity" in t or w in ("moon", "mars", "jupiter")):
            steps.append({"type": "set_gravity", "g": g})
            break
    asset = _find_asset(t)
    sel = next((e for e in ents if e.id == selected), None)
    # "drive THE car" reuses a car that's already on the table; "drive A car" summons a new one
    reuse = None
    if asset:
        existing = [e for e in ents if e.spawned and e.label == asset["name"]]
        names = "|".join(re.escape(x) for x in [asset["name"], *asset["aliases"]])
        if existing and re.search(rf"\b(the|that|this|my)\s+({names})\b", t):
            reuse = existing[-1]
    actor_ids = {reuse.id} if reuse else set()
    target = _find_entity(t, [e for e in ents if not (asset and e.spawned and e.label == asset["name"])], exclude=actor_ids) \
        or (sel if (sel and sel.id not in actor_ids and (re.search(r"\b(it|this|that|selected)\b", t) or not asset)) else None)
    if asset and target is None and sel is not None:
        target = sel
    verb_drive = re.search(r"\b(drive|drives|driving|crash|ram|race|ride|zoom|speed)\b", t)
    verb_launch = re.search(r"\b(roll|throw|launch|shoot|fire|toss|hit|smash|knock|chuck|hurl|fling|lob|bowl|slam|pelt)\b", t)
    verb_drop = re.search(r"\b(drop|fall|falls|land)\b", t)
    verb_push = re.search(r"\b(push|shove|nudge)\b", t)
    verb_remove = re.search(r"\b(remove|delete|take away|pull out)\b", t)
    m_count = re.search(r"(\d+)\s+(dominoes|dominos|domino)", t)

    if asset and asset["id"] == "domino" and (m_count or "row" in t or "line" in t):
        n = int(m_count.group(1)) if m_count else 8
        steps.append({"type": "domino_row", "count": max(2, min(20, n)), "target": target.id if target else None})
        steps.append({"type": "topple_first"})
    elif asset and reuse is not None:
        if verb_drive and asset["kind"] == "vehicle":
            steps.append({"type": "drive", "actor": reuse.id, "toward": target.id if target else None, "speed": _speed(t, 0.6)})
        elif (verb_launch or verb_push) and target:
            steps.append({"type": "launch", "actor": reuse.id, "toward": target.id, "speed": _speed(t, 1.0)})
        elif verb_remove:
            steps.append({"type": "remove", "target": reuse.id})
        else:
            notes.append(f"The {reuse.label} is already on the table; say what it should do (drive, roll, remove...).")
    elif asset:
        actor = "$new"
        if verb_drop and target:
            steps.append({"type": "spawn", "asset": asset["id"], "relation": "above", "ref": target.id, "height": 0.25})
        elif verb_drive and asset["kind"] == "vehicle":
            steps.append({"type": "spawn", "asset": asset["id"], "relation": "approach", "ref": target.id if target else None})
            steps.append({"type": "drive", "actor": actor, "toward": target.id if target else None, "speed": _speed(t, 0.6)})
        elif (verb_launch or verb_drive) and target:
            steps.append({"type": "spawn", "asset": asset["id"], "relation": "approach", "ref": target.id})
            if asset["kind"] == "vehicle":
                steps.append({"type": "drive", "actor": actor, "toward": target.id, "speed": _speed(t, 0.6)})
            else:
                steps.append({"type": "launch", "actor": actor, "toward": target.id, "speed": _speed(t, 1.0)})
        elif re.search(r"\bon (top of )?(the )?\w+", t) and target:
            steps.append({"type": "spawn", "asset": asset["id"], "relation": "on", "ref": target.id})
        else:
            steps.append({"type": "spawn", "asset": asset["id"], "relation": "near" if target else "free", "ref": target.id if target else None})
            if asset["kind"] == "vehicle" and verb_drive:
                steps.append({"type": "drive", "actor": actor, "toward": None, "speed": _speed(t, 0.6)})
    elif verb_remove and target:
        steps.append({"type": "remove", "target": target.id})
    elif verb_push and target:
        steps.append({"type": "push", "target": target.id, "speed": _speed(t, 0.8)})
    generate = None
    if not steps:
        notes.append("Couldn't map this to the object library or a known action. Library: "
                     + ", ".join(sorted({a['name'] for a in asset_lib.manifest().values()})) + ".")
        m = re.search(r"\b(?:a|an|some)\s+((?:[a-z]+\s){0,2}?[a-z]+?)(?=\s+(?:into|on|onto|at|and|to|that|which|from|near|next)\b|[.,!?]|$)", t)
        if m and not asset:
            generate = m.group(1)
    return {"steps": steps, "notes": notes, "planner": "rules", "generate": generate}


# ---------------------------------------------------------------------------
# stage 1b: Claude understanding (same plan format)
# ---------------------------------------------------------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["spawn", "drive", "launch", "push", "remove", "set_gravity",
                                                    "set_mass", "set_friction", "domino_row", "topple_first"]},
                "asset": {"type": ["string", "null"]},
                "relation": {"type": ["string", "null"], "enum": ["approach", "near", "on", "above", "free", None]},
                "ref": {"type": ["string", "null"]},
                "actor": {"type": ["string", "null"]},
                "toward": {"type": ["string", "null"]},
                "target": {"type": ["string", "null"]},
                "speed": {"type": ["number", "null"]},
                "height": {"type": ["number", "null"]},
                "g": {"type": ["number", "null"]},
                "mass": {"type": ["number", "null"]},
                "friction": {"type": ["number", "null"]},
                "count": {"type": ["integer", "null"]},
            },
            "required": ["type", "asset", "relation", "ref", "actor", "toward", "target", "speed", "height", "g",
                         "mass", "friction", "count"],
            "additionalProperties": False,
        }},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["steps", "notes"],
    "additionalProperties": False,
}

PLAN_SYSTEM = (
    "You turn a spoken request about a tabletop physics experiment into steps for a simulator. The scene objects "
    "(ids, labels, sizes, positions in metres; z is up, the table top is z=0) and the library of objects that can "
    "be summoned are given. Rules: use only object ids and asset ids that appear in the input. A newly spawned object "
    "is referred to as \"$new\" (the most recent spawn). spawn.relation: approach = a start spot with a clear path to "
    "ref (for driving/rolling into it), near = beside ref, on = resting on top of ref, above = held `height` metres "
    "above ref and released, free = any visible free spot. drive is only for vehicles (speed m/s, toward = object id). "
    "launch gives an object an initial speed toward an object. domino_row lines up `count` dominoes ending at target, "
    "and topple_first tips the first one. Speeds for tabletop toys are 0.2-2 m/s. If part of the request cannot be "
    "expressed with these steps or this library, leave it out and say so in notes. Unused fields are null."
)


def plan_with_claude(text: str, ents: list[Entity], selected: str | None) -> dict | None:
    from . import llm
    client = llm.client()
    if client is None:
        return None
    import anthropic
    lib = asset_lib.manifest()
    payload = {
        "request": text,
        "selected_object": selected,
        "scene_objects": [{"id": e.id, "label": e.label, "position_m": [round(float(v), 3) for v in e.pos],
                           "footprint_radius_m": round(e.radius, 3), "top_z_m": round(e.top, 3)} for e in ents],
        "library": [{"asset_id": a["id"], "name": a["name"], "kind": a["kind"],
                     "size_cm": [round(x * 100, 1) for x in a["dims"]]} for a in lib.values()],
    }
    try:
        resp = client.beta.messages.create(
            model=config.CLAUDE_MODEL, max_tokens=4000, system=PLAN_SYSTEM,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": PLAN_SCHEMA}},
            messages=[{"role": "user", "content": json.dumps(payload)}],
        )
    except (anthropic.APIConnectionError, anthropic.AuthenticationError, TypeError) as e:
        log.warning("Claude planner unavailable: %s", e)
        return None
    except anthropic.APIStatusError as e:
        log.warning("Claude planner failed (%s): %s", e.status_code, e)
        return None
    if resp.stop_reason == "refusal":
        return None
    txt = next((b.text for b in resp.content if b.type == "text"), None)
    try:
        plan = json.loads(txt) if txt else None
    except json.JSONDecodeError:
        return None
    if plan is None:
        return None
    ids = {e.id for e in ents} | {"$new"}
    clean = []
    for s in plan["steps"]:      # never trust ids blindly: drop steps that reference unknown things
        refs = [s.get(k) for k in ("ref", "toward", "target", "actor")]
        if any(r and r not in ids for r in refs) or (s["type"] == "spawn" and s.get("asset") not in lib):
            plan["notes"].append(f"Dropped a step that referenced something unknown: {s['type']}")
            continue
        clean.append({k: v for k, v in s.items() if v is not None})
    return {"steps": clean, "notes": plan["notes"], "planner": f"claude ({config.CLAUDE_MODEL})"}


# ---------------------------------------------------------------------------
# stage 2: grounding (deterministic geometry)
# ---------------------------------------------------------------------------

def _project(cam, p) -> tuple[float, float] | None:
    R = np.array(cam.R_wc)
    pc = R.T @ (np.asarray(p, float) - np.array(cam.t_wc))
    if pc[2] < 0.05:
        return None
    return cam.fx * pc[0] / pc[2] + cam.cx, cam.fy * pc[1] / pc[2] + cam.cy


def _visible(cam, p, margin=0.06) -> bool:
    uv = _project(cam, p)
    return uv is not None and margin * cam.width < uv[0] < (1 - margin) * cam.width and \
        margin * cam.height < uv[1] < (1 - margin) * cam.height


def _seg_point_dist(a, b, p) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab)), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


def place(scene, ents: list[Entity], radius: float, relation: str, ref: Entity | None, gap: float = 0.16) -> tuple[np.ndarray, float, list[str]]:
    """Choose an (x, y) and a yaw for a new object. Returns (xy, yaw, notes)."""
    t = scene.table
    cam = scene.camera
    notes = []
    view = np.array(cam.R_wc)[:, 2][:2]
    view = view / (np.linalg.norm(view) + 1e-9)
    others = [e for e in ents if ref is None or e.id != ref.id]

    def free(p, extra=0.01):
        if not (t.x_min + radius < p[0] < t.x_max - radius and t.y_min + radius < p[1] < t.y_max - radius):
            return False
        return all(np.linalg.norm(p - e.pos[:2]) > e.radius + radius + extra for e in ents)

    if relation in ("on", "above") and ref is not None:
        return ref.pos[:2].copy(), 0.0, notes

    best, best_score = None, -1e9
    if ref is not None:
        dists = [ref.radius + radius + gap] if relation == "approach" else [ref.radius + radius + 0.015, ref.radius + radius + 0.05]
        if relation == "approach":
            dists += [ref.radius + radius + gap * 0.6, ref.radius + radius + gap * 1.5]
        for D in dists:
            for k in range(24):
                ang = 2 * math.pi * k / 24
                u = np.array([math.cos(ang), math.sin(ang)])
                p = ref.pos[:2] + u * D
                if not free(p):
                    continue
                score = 0.0
                if not _visible(cam, [p[0], p[1], 0.02]):
                    score -= 5
                if relation == "approach":
                    # clear straight path: nothing else between start and target
                    blocked = any(_seg_point_dist(p, ref.pos[:2], e.pos[:2]) < e.radius + radius * 0.8 for e in others)
                    if blocked:
                        continue
                    score += 2.0 * (1 - abs(float(u @ view)))     # prefer side-on views of the motion
                score -= 0.5 * abs(D - dists[0])
                if score > best_score:
                    best, best_score = p, score
        if best is None:
            notes.append(f"No free spot with a clear path next to {ref.label}; placed at the nearest free spot.")
    if best is None:
        cx, cy = (t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2
        centre = ref.pos[:2] if ref is not None else np.array([cx, cy])
        for rr in np.linspace(0.05, 0.6, 24):
            for k in range(24):
                p = centre + rr * np.array([math.cos(2 * math.pi * k / 24), math.sin(2 * math.pi * k / 24)])
                if free(p) and _visible(cam, [p[0], p[1], 0.02]):
                    best = p
                    break
            if best is not None:
                break
    if best is None:
        raise ValueError("no free, visible spot on the table for that object")
    yaw = 0.0
    if ref is not None:
        d = ref.pos[:2] - best
        yaw = math.atan2(d[1], d[0])
    return best, yaw, notes


def _free_run(scene, ents: list[Entity], actor: Entity, u: np.ndarray, margin: float = 0.12) -> float:
    """Distance an object can travel along u before the table edge or another object."""
    t = scene.table
    p = actor.pos[:2]
    best = 2.0
    for k, (lo, hi) in enumerate(((t.x_min, t.x_max), (t.y_min, t.y_max))):
        if abs(u[k]) > 1e-6:
            edge = hi if u[k] > 0 else lo
            best = min(best, (edge - p[k]) / u[k])
    for e in ents:
        if e.id == actor.id:
            continue
        d = e.pos[:2] - p
        along = float(d @ u)
        if along > 0 and abs(float(d[0] * u[1] - d[1] * u[0])) < e.radius + actor.radius:
            best = min(best, along - e.radius - actor.radius)
    return max(0.0, best - margin)


def resolve(plan: dict, scene, ents: list[Entity], n_spawned: int, prior: list[dict] | None = None,
            actor_yaw: dict[str, float] | None = None) -> tuple[list[dict], list[str], list[str]]:
    """Abstract plan -> concrete intervention dicts + human-readable steps + notes.
    prior: interventions already applied (to find existing spawned objects); actor_yaw: their current headings."""
    prior = prior or []
    actor_yaw = actor_yaw or {}
    lib = asset_lib.manifest()
    by_id = {e.id: e for e in ents}
    ivs, human, notes = [], [], list(plan.get("notes", []))
    last_new = None
    k = n_spawned

    def ent(x):
        if x == "$new":
            return by_id.get(last_new) if last_new else None
        return by_id.get(x) if x else None

    for s in plan["steps"]:
        typ = s["type"]
        if typ == "spawn":
            a = lib[s["asset"]]
            radius = 0.5 * math.hypot(a["dims"][0], a["dims"][1])
            ref = ent(s.get("ref"))
            rel = s.get("relation") or ("near" if ref else "free")
            xy, yaw, n = place(scene, ents, radius, rel, ref)
            notes += n
            z = 0.0008
            if rel == "on" and ref is not None:
                z = ref.top + 0.001
            elif rel == "above" and ref is not None:
                z = ref.top + float(s.get("height") or 0.25)
            k += 1
            sid = f"s{k}"
            ivs.append({"type": "spawn", "target": sid, "params": {"asset": a["id"], "pos": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(z, 4)],
                                                                    "yaw": round(yaw, 4), "label": a["name"], "static": bool(a.get("static"))}})
            e = Entity(sid, a["name"], np.array([xy[0], xy[1], z]), radius, z + a["dims"][2], spawned=True)
            ents.append(e)
            by_id[sid] = e
            last_new = sid
            where = {"approach": f"with a clear run at {ref.label}" if ref else "on the table",
                     "near": f"next to {ref.label}" if ref else "on the table", "on": f"on top of {ref.label}" if ref else "",
                     "above": f"{(z - (ref.top if ref else 0)) * 100:.0f} cm above {ref.label}, then released" if ref else "",
                     "free": "in a free spot on the table"}[rel]
            human.append(f"Place a {a['name']} ({' x '.join(f'{d * 100:.0f}' for d in a['dims'])} cm, {a['mass'] * 1000:.0f} g) {where}")
        elif typ == "drive":
            actor = ent(s.get("actor") or "$new")
            spawn_iv = next((iv for iv in ivs + prior if iv["target"] == actor.id and iv["type"] == "spawn"), None) if actor else None
            if actor is None or not actor.spawned or spawn_iv is None or lib[spawn_iv["params"]["asset"]]["kind"] != "vehicle":
                notes.append("Drive needs a vehicle; skipped.")
                continue
            tgt = ent(s.get("toward"))
            raw = s.get("speed")
            full = raw == "max"
            speed = None if full else float(raw or 0.6)
            v_est = 1.6 if full else speed
            stop = False
            if tgt is not None:
                dist = float(np.linalg.norm(tgt.pos[:2] - actor.pos[:2]))
                goal = [round(float(tgt.pos[0]), 4), round(float(tgt.pos[1]), 4)]
                dur = min(6.0, dist / max(0.1, v_est) + 0.8)
            else:
                # straight ahead to the end of the free run, then brake on the table
                yaw = actor_yaw.get(actor.id, spawn_iv["params"]["yaw"])
                # straight ahead if there is room, otherwise the longest free run within reach of the
                # steering (it turns while driving; heading changes beyond ~100 deg are not attempted)
                best_u, run = None, -1.0
                for dyaw in [0.0] + [s_ * k * math.pi / 12 for k in range(1, 8) for s_ in (1, -1)]:
                    u_ = np.array([math.cos(yaw + dyaw), math.sin(yaw + dyaw)])
                    r_ = _free_run(scene, ents, actor, u_) - (0.1 * abs(dyaw) if dyaw else 0.0)   # prefer small turns
                    if r_ > run + 0.05:
                        best_u, run = u_, r_
                u = best_u
                if run < 0.15:
                    notes.append(f"Not enough free table in front of the {actor.label} to drive; move it or remove obstacles.")
                    continue
                goal = [round(float(actor.pos[0] + u[0] * run), 4), round(float(actor.pos[1] + u[1] * run), 4)]
                dur, stop = 6.0, True
            ivs.append({"type": "drive", "target": actor.id, "params": {"speed": "max" if full else speed, "goal": goal,
                                                                         "duration": round(dur, 2), "stop_at_goal": stop}})
            human.append(f"Drive the {actor.label} " + ("at full throttle" if full else f"at {speed:.2f} m/s")
                         + (f" toward {tgt.label}" if tgt else f" along the clearest path for {np.linalg.norm(np.array(goal) - actor.pos[:2]):.2f} m, then brake"))
        elif typ == "launch":
            actor = ent(s.get("actor") or "$new")
            tgt = ent(s.get("toward"))
            if actor is None or tgt is None:
                notes.append("Launch needs an object and a target; skipped.")
                continue
            d = tgt.pos[:2] - actor.pos[:2]
            d = d / (np.linalg.norm(d) + 1e-9)
            speed = float(s.get("speed") or 1.0)
            ivs.append({"type": "launch", "target": actor.id, "params": {"velocity": [round(float(d[0] * speed), 4), round(float(d[1] * speed), 4), 0.0]}})
            human.append(f"Roll the {actor.label} toward {tgt.label} at {speed:.2f} m/s")
        elif typ == "push":
            tgt = ent(s.get("target"))
            if tgt is None:
                continue
            away = tgt.pos[:2] - np.array(scene.camera.t_wc)[:2]
            ivs.append({"type": "push", "target": tgt.id, "params": {"direction": [float(away[0]), float(away[1]), 0.0], "speed": float(s.get("speed") or 0.8)}})
            human.append(f"Push {tgt.label} away from the camera")
        elif typ == "remove":
            tgt = ent(s.get("target"))
            if tgt is not None:
                ivs.append({"type": "remove", "target": tgt.id, "params": {}})
                human.append(f"Remove {tgt.label}")
        elif typ == "set_gravity":
            ivs.append({"type": "set_gravity", "target": None, "params": {"g": float(s.get("g", 9.81))}})
            human.append(f"Set gravity to {float(s.get('g', 9.81)):.2f} m/s²")
        elif typ in ("set_mass", "set_friction"):
            tgt = ent(s.get("target"))
            key = "mass" if typ == "set_mass" else "friction"
            if tgt is not None and s.get(key) is not None:
                ivs.append({"type": typ, "target": tgt.id, "params": {key: float(s[key])}})
                human.append(f"Set {key} of {tgt.label} to {s[key]}")
        elif typ == "domino_row":
            a = lib["domino"]
            tgt = ent(s.get("target"))
            n = int(s.get("count") or 8)
            spacing = a["dims"][2] * 0.6
            length = spacing * (n - 1)
            if tgt is not None:
                xy, yaw, nn = place(scene, ents, length / 2 + 0.02, "approach", tgt, gap=0.03)
                notes += nn
                u = np.array([math.cos(yaw), math.sin(yaw)])
                start = xy - u * length / 2
            else:
                xy, _, nn = place(scene, ents, length / 2 + 0.02, "free", None)
                notes += nn
                view = np.array(scene.camera.R_wc)[:, 0][:2]
                u = view / (np.linalg.norm(view) + 1e-9)        # line runs left-to-right in the image
                yaw = math.atan2(u[1], u[0])
                start = xy - u * length / 2
            first = None
            for i in range(n):
                p = start + u * spacing * i
                k += 1
                sid = f"s{k}"
                first = first or sid
                ivs.append({"type": "spawn", "target": sid, "params": {"asset": "domino", "pos": [round(float(p[0]), 4), round(float(p[1]), 4), 0.0008],
                                                                        "yaw": round(yaw + math.pi / 2, 4), "label": f"domino {i + 1}", "group": first}})
                e = Entity(sid, f"domino {i + 1}", np.array([p[0], p[1], 0.0]), 0.013, a["dims"][2], spawned=True)
                ents.append(e)
                by_id[sid] = e
            last_new = first
            human.append(f"Line up {n} dominoes {spacing * 100:.1f} cm apart" + (f" ending at {tgt.label}" if tgt else ""))
        elif typ == "topple_first":
            if last_new:
                e = by_id[last_new]
                sp = next(iv for iv in ivs if iv["target"] == last_new)
                yaw = sp["params"]["yaw"] - math.pi / 2
                d = [math.cos(yaw), math.sin(yaw), 0.0]
                pt = [float(e.pos[0]), float(e.pos[1]), 0.04]
                ivs.append({"type": "push", "target": last_new, "params": {"direction": d, "speed": 0.35, "point": pt}})
                human.append("Tip the first domino")
    return ivs, human, notes


LOCAL_PLAN_PROMPT = """You convert a request into simulator steps. Reply with JSON only.
Step types: spawn (asset, relation: approach|near|on|above|free, ref), drive (actor, toward, speed),
launch (actor, toward, speed), push (target, speed), remove (target), set_gravity (g),
domino_row (count, target), topple_first. The newest spawned object is "$new".
Speeds are m/s (0.2 slow .. 2 fast); "max" means full throttle. Use ONLY these ids.

Example. Objects: o1=mug, o2=book. Library: toy_car(vehicle), baseball_01.
Request: send a car flying into the mug
{"steps":[{"type":"spawn","asset":"toy_car","relation":"approach","ref":"o1"},{"type":"drive","actor":"$new","toward":"o1","speed":1.5}]}

Example. Objects: o1=cup (left), o2=cup (right). Library: baseball_01, steel_ball.
Request: throw a ball at the right cup
{"steps":[{"type":"spawn","asset":"baseball_01","relation":"approach","ref":"o2"},{"type":"launch","actor":"$new","toward":"o2","speed":1.5}]}

Example. Objects: o1=bottle. Library: baseball_01, domino.
Request: make it rain baseballs on the bottle
{"steps":[{"type":"spawn","asset":"baseball_01","relation":"above","ref":"o1","height":0.3}]}

Objects: {objects}
Library: {library}
Selected object: {selected}
Request: {request}
"""


def plan_with_local_ai(text: str, ents: list[Entity], selected: str | None, scene=None) -> dict | None:
    """Free on-device understanding of free-form requests. The output is validated like Claude's."""
    from . import local_ai
    if not local_ai.available():
        return None
    lib = asset_lib.manifest()
    # where each object is in the camera view, so "the left mug" can be resolved
    where = {}
    if scene is not None:
        xs = {e.id: (_project(scene.camera, e.pos) or (scene.camera.cx, 0))[0] for e in ents}
        if xs:
            lo, hi = min(xs.values()), max(xs.values())
            for i, x in xs.items():
                f = (x - lo) / max(1e-6, hi - lo)
                where[i] = "left" if f < 0.34 else "right" if f > 0.66 else "middle"
    fill = {"{objects}": ", ".join(f"{e.id}={e.label} ({where.get(e.id, 'middle')})" for e in ents) or "none",
            "{library}": ", ".join(f"{a['id']}({a['kind']})" if a["kind"] == "vehicle" else a["id"] for a in lib.values()),
            "{selected}": selected or "none", "{request}": text}
    prompt = LOCAL_PLAN_PROMPT
    for k, v in fill.items():          # plain replace: the template contains literal JSON braces
        prompt = prompt.replace(k, v)
    data = local_ai.extract_json(local_ai.generate(prompt, max_tokens=350))
    if not data or not isinstance(data.get("steps"), list):
        return None
    ids = {e.id for e in ents} | {"$new"}
    names = {a["name"]: a["id"] for a in lib.values()}
    allowed = {"spawn", "drive", "launch", "push", "remove", "set_gravity", "set_mass", "set_friction", "domino_row", "topple_first"}
    clean, notes = [], []
    for s in data["steps"]:
        if not isinstance(s, dict) or s.get("type") not in allowed:
            continue
        s = {k: v for k, v in s.items() if v is not None}
        if s["type"] == "spawn":
            rel = str(s.get("relation") or "").lower()
            s["relation"] = {"toward": "approach", "towards": "approach", "at": "approach", "into": "approach",
                             "beside": "near", "next to": "near", "onto": "on", "over": "above", "top": "on"}.get(rel, rel)
            if s["relation"] not in ("approach", "near", "on", "above", "free"):
                s["relation"] = "near" if s.get("ref") else "free"
            if "height" in s:
                try:
                    s["height"] = float(min(1.0, max(0.05, float(s["height"]))))
                except (TypeError, ValueError):
                    s.pop("height")
            s["asset"] = names.get(s.get("asset"), s.get("asset"))
            if s.get("asset") not in lib:
                notes.append(f"'{s.get('asset')}' isn't in the library")
                continue
        if any(s.get(k) and s[k] not in ids for k in ("ref", "toward", "target", "actor")):
            notes.append(f"dropped a '{s['type']}' step that referred to something not in the scene")
            continue
        if "speed" in s and s["speed"] != "max":
            try:
                s["speed"] = float(min(3.0, max(0.05, float(s["speed"]))))
            except (TypeError, ValueError):
                s.pop("speed")
        clean.append(s)
    if not clean:
        return None
    return {"steps": clean, "notes": notes, "planner": "local AI (Qwen2.5-VL-3B, on this Mac)"}


def strong(plan: dict) -> bool:
    """A rule-parser result we trust over the language model: it resolved a definite action."""
    st = plan.get("steps", [])
    return any(s["type"] in ("drive", "launch", "push", "remove", "domino_row", "set_gravity") or
               (s["type"] == "spawn" and s.get("relation") in ("above", "on", "approach")) or
               (s["type"] == "spawn" and s.get("relation") == "near" and s.get("ref")) for s in st)
