"""Measurements and quantitative answers, computed from the simulator's state.

Every number here comes from MuJoCo state recorded at 60 Hz (centre-of-mass
velocity, contact-force vectors, kinetic/potential energy). Answers always say
how the number was obtained and what it depends on (estimated masses, the
reconstructed scale). Nothing is estimated by a language model.
"""
from __future__ import annotations

import math
import re

import mujoco
import numpy as np

from . import assets as asset_lib
from .physics.sim import TABLE, FLOOR, World

G = 9.81


# ---------------------------------------------------------------------------
# per-object metrics over a time window
# ---------------------------------------------------------------------------

def series(sim: dict, oid: str, t0: float = 0.0, t1: float | None = None) -> dict:
    """Time series for one object: t, speed, horizontal speed, acceleration, forces, energy."""
    k = sim["kinematics"][oid]
    fps = sim["fps"]
    a = int(round(t0 * fps))
    b = len(k["v"]) if t1 is None else min(len(k["v"]), int(round(t1 * fps)) + 1)
    t, v, F, Fo, KE, PE = [], [], [], [], [], []
    for f in range(a, b):
        if k["v"][f] is None:
            continue
        t.append(f / fps)
        v.append(k["v"][f])
        F.append(k["F_contact"][f])
        Fo.append(k["F_objects"][f])
        KE.append(k["KE"][f])
        PE.append(k["PE"][f])
    v = np.array(v) if v else np.zeros((0, 3))
    t = np.array(t)
    speed = np.linalg.norm(v, axis=1) if len(v) else np.zeros(0)
    hspeed = np.linalg.norm(v[:, :2], axis=1) if len(v) else np.zeros(0)
    # acceleration from the 60 Hz velocity, lightly smoothed (3-frame) so contact
    # chatter does not dominate; impacts are reported separately via contact force
    if len(v) >= 3:
        vs = np.copy(v)
        vs[1:-1] = (v[:-2] + v[1:-1] + v[2:]) / 3
        acc = np.linalg.norm(np.gradient(vs, t, axis=0), axis=1)
    else:
        acc = np.zeros(len(v))
    return {"t": t, "v": v, "speed": speed, "hspeed": hspeed, "acc": acc, "F": np.array(F), "F_objects": np.array(Fo),
            "KE": np.array(KE), "PE": np.array(PE)}


def metrics(sim: dict, oid: str, t0: float = 0.0, t1: float | None = None) -> dict:
    s = series(sim, oid, t0, t1)
    if len(s["t"]) < 2:
        return {}
    i = sim["object_ids"].index(oid)
    fps = sim["fps"]
    pos = np.array([sim["frames"][int(round(tt * fps))][i][:3] for tt in s["t"]])
    path = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    m = sim["masses"][oid]
    im = int(np.argmax(s["speed"]))
    ih = int(np.argmax(s["hspeed"]))
    ia = int(np.argmax(s["acc"]))
    io = int(np.argmax(s["F_objects"])) if len(s["F_objects"]) else 0
    E = s["KE"] + s["PE"]
    if len(E) >= 5:                                   # contact micro-jitter must not accumulate into "energy"
        E = np.convolve(np.pad(E, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    dE = np.diff(E)
    dE[np.abs(dE) < 1e-5] = 0.0
    return {
        "mass_kg": m,
        "max_speed_mps": round(float(s["speed"][im]), 3), "t_max_speed": round(float(s["t"][im]), 3),
        "max_horizontal_speed_mps": round(float(s["hspeed"][ih]), 3), "t_max_horizontal_speed": round(float(s["t"][ih]), 3),
        "peak_acceleration_mps2": round(float(s["acc"][ia]), 2), "t_peak_acceleration": round(float(s["t"][ia]), 3),
        "peak_object_contact_force_N": round(float(s["F_objects"][io]), 3), "t_peak_object_force": round(float(s["t"][io]), 3),
        "path_length_m": round(path, 4),
        "displacement_m": round(float(np.linalg.norm(pos[-1] - pos[0])), 4),
        "drop_m": round(float(pos[0][2] - pos[-1][2]), 4),
        "max_kinetic_energy_J": round(float(s["KE"].max()), 5),
        "momentum_at_max_speed_kgmps": round(m * float(s["speed"][im]), 4),
        # energy the object gave up over time (to friction, collisions, other objects): sum of decreases
        "energy_given_up_J": round(float(-dE[dE < 0].sum()), 5),
        "energy_received_J": round(float(dE[dE > 0].sum()), 5),
        "weight_N": round(m * G, 3),
    }


# ---------------------------------------------------------------------------
# capability tests: what an object can do, independent of this table's size
# ---------------------------------------------------------------------------

def vehicle_capability(asset_id: str, seconds: float = 5.0) -> dict:
    """Full-throttle run on a long flat virtual track, then an emergency stop.
    Same vehicle model, motors, tyres and integrator as in the scene."""
    from .schemas import Camera, Scene, Table
    lib = asset_lib.manifest()
    a = lib[asset_id]
    cam = Camera(width=10, height=10, fx=10, fy=10, cx=5, cy=5, hfov_deg=60, R_wc=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], t_wc=[0, 0, 1])
    scene = Scene(scene_id="captest", source="capability", camera=cam,
                  table=Table(x_min=-1, x_max=60, y_min=-2, y_max=2, height_above_floor=0.75), objects=[])
    w = World(scene, G, [{"sid": "v", "asset": a, "k": 0, "mass": a["mass"]}])
    w.spawn("v", [0, 0, 0.0008], 0.0)
    for _ in range(int(0.3 / 0.002)):
        w.step()
    w.drive["v"] = {"speed": None, "goal": None, "t_end": w.d.time + seconds}
    t, v = [], []
    n = int(seconds / 0.002)
    for k_ in range(n):
        w.step()
        if k_ % 10 == 0:
            vel = w.d.qvel[w.vadr["v"]:w.vadr["v"] + 3]
            t.append(k_ * 0.002)
            v.append(float(np.linalg.norm(vel[:2])))
    v = np.array(v)
    t = np.array(t)
    top = float(v.max())
    t_1 = float(t[np.argmax(v >= 1.0)]) if (v >= 1.0).any() else None
    t_90 = float(t[np.argmax(v >= 0.9 * top)])
    dist_90 = float(np.trapezoid(v[t <= t_90], t[t <= t_90])) if hasattr(np, "trapezoid") else float(np.trapz(v[t <= t_90], t[t <= t_90]))
    # emergency stop from top speed
    x0 = float(w.d.xpos[w.body["v"]][0])
    w.drive["v"] = {"speed": None, "goal": [1e6, 0.0], "t_end": w.d.time + 5, "stop_at_goal": True, "braking": True}
    for _ in range(int(3.0 / 0.002)):
        w.step()
        if np.linalg.norm(w.d.qvel[w.vadr["v"]:w.vadr["v"] + 2]) < 0.005:
            break
    brake = float(w.d.xpos[w.body["v"]][0]) - x0
    return {"top_speed_mps": round(top, 3), "top_speed_kmh": round(top * 3.6, 2),
            "time_0_to_1mps_s": round(t_1, 3) if t_1 else None, "time_to_90pct_top_s": round(t_90, 3),
            "distance_to_90pct_top_m": round(dist_90, 3), "braking_distance_from_top_m": round(brake, 3),
            "mass_kg": a["mass"], "track": "flat virtual track, same tyre/table friction as the scene",
            "speed_trace": {"t": [round(x, 3) for x in t[::3]], "v": [round(x, 4) for x in v[::3]]}}


# ---------------------------------------------------------------------------
# questions -> numbers
# ---------------------------------------------------------------------------

QUANTITIES = [
    ("speed", r"\b(speed|velocity|valocity|fast|quick|mph|km/?h|m/s)\b"),
    ("acceleration", r"\baccelerat"),
    ("force", r"\b(force|impact|hit|hard|newton)"),
    ("energy", r"\b(energy|joule|work)\b"),
    ("distance", r"\b(far|distance|travel|how long a way|displac)"),
    ("time", r"\b(how long|time|seconds?|when)\b"),
    ("momentum", r"\bmomentum\b"),
    ("height", r"\b(height|high|drop|fall)\b"),
]
CAPABILITY = r"\b(can|could|capable|able to|top speed|achieve|earn|get up to)\b"


def _subject(question: str, scene, sim: dict | None, interventions, selected: str | None):
    """Return (kind, id_or_asset, label): kind in {'object', 'asset'}."""
    q = question.lower()
    labels = {}
    if sim:
        labels.update(sim.get("labels", {}))
    for o in scene.objects:
        labels.setdefault(o.id, o.label)
    # the grammatical subject is (almost always) the FIRST thing named: "how fast was the car when it hit object 2"
    hits: list[tuple[int, str, str, str]] = []           # (position, kind, id, label)
    for oid, lab in labels.items():
        for word in re.findall(r"[a-z]+", lab.lower()):
            if len(word) > 2 and word != "object":
                m_ = re.search(rf"\b{word}s?\b", q)
                if m_:
                    hits.append((m_.start(), "object", oid, lab))
        m_ = re.search(rf"\b{re.escape(lab.lower())}\b", q)
        if m_:
            hits.append((m_.start(), "object", oid, lab))
    for a in asset_lib.manifest().values():
        for alias in [a["name"], *a["aliases"]]:
            m_ = re.search(rf"\b{re.escape(alias)}\b", q)
            if m_ and not any(h[2] in (sim or {}).get("spawned", {}) and (sim or {})["spawned"][h[2]]["asset"] == a["id"] for h in hits):
                hits.append((m_.start() + 0.5, "asset", a["id"], a["name"]))
    if hits:
        pos, kind, ref, lab = min(hits)
        return kind, ref, lab
    if re.search(r"\b(it|this|that)\b", q) and selected:
        return "object", selected, labels.get(selected, selected)
    if sim and sim["segments"]:
        tgt = sim["segments"][-1]["intervention"].get("target")
        if tgt:
            return "object", tgt, labels.get(tgt, tgt)
    return None, None, None


def answer(question: str, scene, sim: dict | None, interventions, selected: str | None) -> dict:
    q = question.lower()
    quantity = next((name for name, pat in QUANTITIES if re.search(pat, q)), "speed")
    kind, ref, label = _subject(question, scene, sim, interventions, selected)
    capability = bool(re.search(CAPABILITY, q))
    assumptions = []
    if kind is None:
        return {"answer": "Tell me which object (name it, or select it and say 'it').", "value": None}

    lib = asset_lib.manifest()
    asset_id = None
    if kind == "asset":
        asset_id = ref
    elif sim and ref in sim.get("spawned", {}):
        asset_id = sim["spawned"][ref]["asset"]
    # "how fast CAN the car go" -> capability test, not limited by this table
    if capability and quantity in ("speed", "acceleration", "distance", "time") and asset_id and lib[asset_id]["kind"] == "vehicle":
        cap = vehicle_capability(asset_id)
        txt = (f"Top speed of the {label}: {cap['top_speed_mps']:.2f} m/s ({cap['top_speed_kmh']:.1f} km/h). "
               f"0→1 m/s in {cap['time_0_to_1mps_s']:.2f} s; reaches 90% of top speed after {cap['time_to_90pct_top_s']:.2f} s "
               f"and {cap['distance_to_90pct_top_m']:.2f} m; emergency stop from top speed takes {cap['braking_distance_from_top_m']:.2f} m.")
        return {"answer": txt, "value": cap["top_speed_mps"], "unit": "m/s", "quantity": "top speed", "subject": label,
                "method": "Full-throttle run of the same MuJoCo vehicle (4 wheel motors with a DC torque-speed curve, "
                          "back-EMF, bearing friction, tyre friction 0.9) on a long flat virtual track, sampled at 50 Hz.",
                "assumptions": ["motor: 0.34 N stall thrust per wheel, 3.2 m/s no-load speed (toy gear motor)",
                                f"mass {cap['mass_kg']} kg (library value)"],
                "details": cap}
    if kind == "asset" or sim is None or ref not in sim.get("kinematics", {}):
        return {"answer": f"I need an experiment with the {label} first. Run one (e.g. 'drive a toy car into the mug'), then ask again.",
                "value": None, "needs_experiment": True, "subject": label}

    # which experiment: the latest one if asked ("last", "now", "just"), otherwise the one in which
    # this object moved the most
    segs = [sg for sg in sim["segments"] if ref in sg["summary"]]
    if not segs:
        segs = sim["segments"]
    if re.search(r"\b(last|latest|just|now|this time)\b", q):
        seg = segs[-1]
    else:
        seg = max(segs, key=lambda sg: (sg["summary"].get(ref, {}).get("displacement_m", 0), sg["index"]))
    mt = metrics(sim, ref, seg["t_start"], seg["t_end"])
    if not mt:
        return {"answer": f"The {label} doesn't exist during the last experiment.", "value": None}
    o = next((x for x in scene.objects if x.id == ref), None)
    if o is not None:
        assumptions.append(f"mass {o.mass:.3f} kg ({o.mass_source})")
        assumptions.append("distances use the reconstructed scale (calibrate with one known height for best accuracy)")
    else:
        assumptions.append(f"mass {mt['mass_kg']} kg (library value)")
    impact = None
    for e in seg["events"]:
        if e["type"] == "impact" and ref in (e["subject"], e["other"]) and e.get("other") not in (TABLE, FLOOR):
            impact = e
            break
    if quantity == "speed":
        if impact and re.search(r"\b(hit|impact|collid|crash)", q):
            # speed just before contact (the contact-detection instant already includes some soft-contact slowdown)
            other = sim["labels"].get(impact["other"] if impact["subject"] == ref else impact["subject"], "it")
            before = series(sim, ref, max(seg["t_start"], impact["t"] - 0.15), impact["t"] - 0.02)
            v_before = float(before["speed"].mean()) if len(before["speed"]) else 0.0
            if v_before > 0.05:
                sr = series(sim, ref, max(seg["t_start"], impact["t"] - 0.12), impact["t"])
                val, unit = round(float(sr["speed"].max()), 3), "m/s"
                txt = f"The {label} was moving at {val:.2f} m/s ({val * 3.6:.1f} km/h) when it hit the {other} (t = {impact['t']:.2f} s)."
            else:      # it was standing still: report how fast the impact knocked it
                after = series(sim, ref, impact["t"], impact["t"] + 0.2)
                val, unit = round(float(after["speed"].max()), 3), "m/s"
                txt = (f"The {label} was at rest; the {other} knocked it to {val:.2f} m/s ({val * 3.6:.1f} km/h) "
                       f"within 0.2 s of the impact at t = {impact['t']:.2f} s.")
        else:
            val, unit = mt["max_speed_mps"], "m/s"
            txt = (f"Maximum speed of the {label} in experiment {seg['index'] + 1} ({seg['intervention']['type']}): {val:.2f} m/s ({val * 3.6:.1f} km/h) at t = {mt['t_max_speed']:.2f} s"
                   + (f"; maximum horizontal speed {mt['max_horizontal_speed_mps']:.2f} m/s." if abs(mt["max_horizontal_speed_mps"] - val) > 0.05 else "."))
    elif quantity == "acceleration":
        val, unit = mt["peak_acceleration_mps2"], "m/s²"
        txt = f"Peak acceleration of the {label}: {val:.1f} m/s² ({val / G:.2f} g) at t = {mt['t_peak_acceleration']:.2f} s (averaged over 1/60 s)."
    elif quantity == "force":
        val, unit = mt["peak_object_contact_force_N"], "N"
        txt = (f"Peak contact force between the {label} and other objects: {val:.2f} N at t = {mt['t_peak_object_force']:.2f} s "
               f"(its weight is {mt['weight_N']:.2f} N). Impact peaks depend on contact stiffness, so treat them as ±30%.")
    elif quantity == "energy":
        val, unit = mt["max_kinetic_energy_J"], "J"
        if val < 1.0:
            val, unit = round(val * 1000, 2), "mJ"
        driven = any(e["type"] == "intervention" and e["subject"] == ref and e.get("action") in ("drive", "launch", "push") for e in seg["events"])
        txt = f"Maximum kinetic energy of the {label}: {mt['max_kinetic_energy_J'] * 1000:.1f} mJ (½mv² + ½Iω²)."
        if not driven:
            txt += (f" It received {abs(mt['energy_received_J']) * 1000:.1f} mJ from collisions and gave up "
                    f"{abs(mt['energy_given_up_J']) * 1000:.1f} mJ to friction, impacts and other objects.")
    elif quantity == "distance":
        val, unit = mt["path_length_m"], "m"
        txt = f"The {label} travelled {val * 100:.1f} cm along its path; net displacement {mt['displacement_m'] * 100:.1f} cm."
    elif quantity == "momentum":
        val, unit = mt["momentum_at_max_speed_kgmps"], "kg·m/s"
        txt = f"Momentum of the {label} at its top speed: {val:.3f} kg·m/s (p = m·v = {mt['mass_kg']} kg × {mt['max_speed_mps']:.2f} m/s)."
    elif quantity == "height":
        val, unit = mt["drop_m"], "m"
        txt = f"The {label} ended {val * 100:.1f} cm lower than it started."
    else:   # time
        moving = [e["t"] for e in seg["events"] if e["subject"] == ref and e["type"] in ("started_moving", "came_to_rest")]
        val = round(moving[-1] - moving[0], 3) if len(moving) >= 2 else None
        unit = "s"
        txt = f"The {label} was in motion for {val:.2f} s." if val else f"The {label} didn't start and stop within the last experiment."
    return {"answer": txt, "value": val, "unit": unit, "quantity": quantity, "subject": label, "subject_id": ref,
            "method": "MuJoCo state recorded at 60 Hz: centre-of-mass velocity, contact-force vectors, kinetic + potential energy.",
            "assumptions": assumptions, "details": mt}
