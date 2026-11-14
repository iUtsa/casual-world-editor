"""Measurements and explanations for general experiments, from recorded simulator state only."""
from __future__ import annotations

import math

from .experiment import ANCHOR

import numpy as np

G = 9.81


def _series(sim, sid):
    k = sim["kinematics"][sid]
    i = sim["object_ids"].index(sid)
    fps = sim["fps"]
    rows = [(f / fps, k["v"][f], (k["c"][f] if k.get("c") and k["c"][f] is not None else sim["frames"][f][i][:3]), k["KE"][f])
            for f in range(len(k["v"])) if k["v"][f] is not None]
    if not rows:
        return None
    t = np.array([r[0] for r in rows]); v = np.array([r[1] for r in rows]); p = np.array([r[2] for r in rows]); ke = np.array([r[3] for r in rows])
    return {"t": t, "v": v, "p": p, "ke": ke, "speed": np.linalg.norm(v, axis=1)}


def _medium(compiled, t, m):
    """Everything the fluid laws need for one object, exactly as the simulator used it."""
    from .physics import drag
    env = compiled.get("env", {})
    rho, mu = float(env.get("density", 0.0)), float(env.get("viscosity", 0.0))
    sp = (t.extra.get("drag") if t is not None else None) or drag.spec(t.shape if t else "box", t.dims if t else [0.1] * 3)
    V = float(t.extra.get("volume", 0.0)) if (t is not None and rho > 0) else 0.0
    return {"m": float(m), "V": V, "sp": sp, "rho": rho, "mu": mu}


def agreement(measured, theory) -> float | None:
    """How closely a measurement matches theory, in percent (100 = identical)."""
    if measured is None or theory is None or not math.isfinite(theory) or abs(theory) < 1e-12:
        return None
    return max(0.0, 100.0 * (1 - abs(measured - theory) / abs(theory)))


def _contacts(sim, a, b):
    """[(t, mean |F| over the frame, peak |F|, impulse)] for frames in which a and b touched."""
    key = "|".join(sorted((a, b)))
    return [(f / sim["fps"], F, pk, J) for f, F, pk, J in sim["pair_forces"].get(key, [])]


def _at(s, t):
    i = int(np.clip(np.searchsorted(s["t"], t), 0, len(s["t"]) - 1))
    return i


def _fmt(v, unit, digits=2):
    """Readable numbers with SI prefixes: 1.76 MJ, 5.66 MN, 12.4 kN·s, 3.1 mJ."""
    if v is None:
        return "—"
    av = abs(v)
    if unit == "kg·m/s":
        return f"{v:.3g} {unit}" if av >= 1e5 else f"{v:.0f} {unit}" if av >= 100 else f"{v:.{max(digits, 3) if av < 1 else digits}f} {unit}"
    if unit in ("J", "N", "N·s", "W", "N/m"):
        for k, pre in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
            if av >= k:
                x = v / k
                return f"{x:.3g} {pre}{unit}" if abs(x) < 100 else f"{x:.0f} {pre}{unit}"
        if 0 < av < 0.1 and unit in ("J", "N", "N·s"):
            return f"{v * 1000:.3g} m{unit}"
    if unit == "kg" and av >= 1000:
        return f"{v / 1000:.3g} t"
    if unit == "kg" and 0 < av < 0.1:
        return f"{v * 1000:.3g} g"
    if av >= 1e5:
        return f"{v:.3g} {unit}"
    if av >= 100:
        return f"{v:.0f} {unit}".strip()
    if 0 < av < 0.01:
        return f"{v:.2g} {unit}".strip()
    return f"{v:.{digits}f} {unit}".strip()


def analyse(compiled: dict, sim: dict, extra: list[str] | None = None) -> dict:
    things, sid_of = compiled["things"], compiled["sid_of"]
    name_of = {v: k for k, v in sid_of.items()}
    acts = compiled.get("actions", [])
    seg = sim["segments"][-1]                         # the action segment
    t0, t1 = seg["t_start"], seg["t_end"]
    g = sim.get("gravity", G)
    rows, story = [], []

    seen_labels = set()

    def add(label, value, unit, who="", digits=2, why=None):
        if label in seen_labels:
            return
        seen_labels.add(label)
        rows.append({"label": label, "value": None if value is None else float(value), "unit": unit, "about": who,
                     "text": _fmt(value, unit, digits), "why": why})

    S = {sid: _series(sim, sid) for sid in name_of if sid in sim["kinematics"]}
    mass = {sid: sim["masses"].get(sid) for sid in name_of}

    drops: list[dict] = []
    for a in acts:
        A, B = sid_of.get(a["a"]), sid_of.get(a.get("b")) if a.get("b") else None
        na, nb = a["a"], a.get("b")
        sa, sb = S.get(A), S.get(B) if B else None
        if a["do"] in ("collide", "hit", "roll", "push", "throw", "fight", "chase") and B and sa is not None and sb is not None:
            cont = [c for c in _contacts(sim, A, B) if c[0] >= t0]
            if not cont:
                story.append(f"The {na} and the {nb} never touched" + (" — the throw missed or fell short." if a["do"] == "throw" else "."))
                if a["do"] == "throw":
                    _throw(add, story, na, sa, t0, g)
                continue
            # contact episodes (a fight has many)
            eps, cur = [], [cont[0]]
            for c in cont[1:]:
                if c[0] - cur[-1][0] > 2.5 / sim["fps"]:
                    eps.append(cur); cur = [c]
                else:
                    cur.append(c)
            eps.append(cur)
            first = eps[0]
            tc, te = first[0][0], first[-1][0]
            dt_f = 1 / sim["fps"]
            ib, ia = _at(sa, tc - 1.5 * dt_f), _at(sa, te + 1.0 * dt_f)
            jb, ja = _at(sb, tc - 1.5 * dt_f), _at(sb, te + 1.0 * dt_f)
            propelled = {iv_["target"] for iv_ in compiled["interventions"] if iv_["type"] == "propel" and not iv_["params"].get("hand")}
            active = A in propelled or B in propelled
            ma, mb = mass[A], mass[B]
            va0, vb0, va1, vb1 = sa["v"][ib], sb["v"][jb], sa["v"][ia], sb["v"][ja]
            p0 = ma * va0 + mb * vb0
            p1 = ma * va1 + mb * vb1
            ke0 = 0.5 * ma * va0 @ va0 + 0.5 * mb * vb0 @ vb0
            ke1 = 0.5 * ma * va1 @ va1 + 0.5 * mb * vb1 @ vb1
            # contact normal = direction of the momentum the impact gave b (robust, unlike positions)
            dvb = vb1 - vb0
            n = dvb if np.linalg.norm(dvb) > 1e-4 else (sb["p"][jb] - sa["p"][ib])
            n = n / max(1e-9, np.linalg.norm(n))
            vrel0, vrel1 = float((vb0 - va0) @ n), float((vb1 - va1) @ n)
            J = sum(c[3] for c in first)
            fixed_target = bool(things.get(nb) is not None and things[nb].extra.get("fixed"))
            # measured: the frames just before and just after the impact (Newton's law of restitution:
            # e = separation speed / approach speed along the line of impact)
            vrel1 = float((vb1 - va1) @ n)
            e = float(np.clip(-vrel1 / vrel0, 0.0, 1.0)) if vrel0 < -1e-4 else None
            if fixed_target:
                p1 = ma * va1 + mb * vb0
            # off-centre hits turn part of the energy into spin
            spun = []
            for sid_, nm_ in ((A, na), (B, nb)):
                w_ = sim["kinematics"][sid_]["w"]
                fb, fa = int(tc * sim["fps"]) - 2, min(len(w_) - 1, int(te * sim["fps"]) + 2)
                if fb >= 0 and w_[fb] is not None and w_[fa] is not None:
                    w0_, w1_ = float(np.linalg.norm(w_[fb])), float(np.linalg.norm(w_[fa]))
                    if w1_ - w0_ > 2.0 and things.get(nm_) is not None and things[nm_].shape != "sphere":
                        spun.append((nm_, w1_))
            Fpk = max(c[2] for c in first)
            # ---- theory: momentum conservation and Newton's law of restitution along the line of impact
            from .physics.sim import pair_restitution
            ta_, tb_ = things.get(na), things.get(nb)
            both_new = ta_ is not None and tb_ is not None and not fixed_target          # new or real scanned: both obey the impact law
            va1m, vb1m = sa["v"][ia], sb["v"][ja]                       # measured, frames after contact
            ev = next((x for x in sim.get("impacts", []) if {x["a"], x["b"]} == {A, B} and x["t1"] >= t0 - 1e-6), None)
            if ev is not None:
                # the impact as a high-speed camera sees it: at first touch and at the instant they separate
                sw = ev["a"] != A
                va0, vb0 = np.array(ev["vb0" if sw else "va0"]), np.array(ev["va0" if sw else "vb0"])
                va1m, vb1m = np.array(ev["vb1" if sw else "va1"]), np.array(ev["va1" if sw else "vb1"])
                n = np.array(ev["n"]) * (-1.0 if sw else 1.0)
                va1, vb1 = va1m, vb1m
                vrel0 = float((vb0 - va0) @ n)
                vrel1 = float((vb1 - va1) @ n)
                e = float(np.clip(-vrel1 / vrel0, 0.0, 1.0)) if vrel0 < -1e-4 else e
                p0, p1 = ma * va0 + mb * vb0, ma * va1 + mb * vb1
                ke0 = 0.5 * ma * va0 @ va0 + 0.5 * mb * vb0 @ vb0
                ke1 = 0.5 * ma * va1 @ va1 + 0.5 * mb * vb1 @ vb1
            on_floor = compiled["setting"] not in ("water", "space", "air")
            slanted = abs(float(n[2])) > 0.2 and on_floor          # different-sized things on a table: the floor joins in
            p0m, p1m = ma * va0 + mb * vb0, ma * va1m + mb * vb1m
            p_scale_m = max(ma * float(np.linalg.norm(va0)) + mb * float(np.linalg.norm(vb0)), 1e-9)
            ag_c = []
            # a third body touching either one during the hit (a book resting on a mug, a wall behind it) takes part in
            # the collision: then it is not a free two-body impact and that law doesn't apply
            third = []
            for key_, fr_ in sim.get("pair_forces", {}).items():
                ids_ = key_.split("|")
                if (A in ids_ or B in ids_) and set(ids_) != {A, B} and any(abs(f_ / sim["fps"] - tc) < 0.06 and Fm_ > 0.05 for f_, Fm_, _, _ in fr_):
                    other = next(x for x in ids_ if x not in (A, B))
                    third.append(name_of.get(other, other))
            if both_new and not active and third:
                story.append(f"Not a free two-body collision: the {third[0]} was touching {'them' if len(third) > 1 else 'one of them'} during the hit "
                             f"(resting on it or leaning against it), so it took part — its weight and friction also act, and momentum is shared with it. "
                             f"The two-body formulas don't apply here, so no theory score is shown.")
            elif both_new and not active and slanted:
                hz = lambda v_: np.array([v_[0], v_[1], 0.0])
                ph0, ph1 = hz(p0m if ev is None else p0), hz(p1m if ev is None else p1)
                add("Momentum along the table conserved (after ÷ before)", 100 * (1 - float(np.linalg.norm(ph1 - ph0)) / p_scale_m), "%", "system", 1,
                    "the line of impact tilts (different sizes), so the floor pushes too: only the momentum along the table is conserved")
                ag_c = [max(0.0, 100 * (1 - float(np.linalg.norm(ph1 - ph0)) / p_scale_m))]
            elif both_new and not active:
                e_set = pair_restitution(ta_.bounciness, tb_.bounciness)
                ua, ub = float(va0 @ n), float(vb0 @ n)
                M = ma + mb
                ua_th = (ma * ua + mb * ub + mb * e_set * (ub - ua)) / M
                ub_th = (ma * ua + mb * ub + ma * e_set * (ua - ub)) / M
                ua_m, ub_m = float(va1m @ n), float(vb1m @ n)
                scale = max(abs(ua - ub), 1e-6)
                if ev is not None:
                    p0m, p1m = p0, p1
                p_scale_m = max(ma * float(np.linalg.norm(va0)) + mb * float(np.linalg.norm(vb0)), 1e-9)
                dpn = abs(float((p1m - p0m) @ n))
                # standing on a surface, friction acts during the contact too: at most mu*m*g per body for its duration. For a
                # quick hit that's negligible; for a long, soft contact (toys, flesh) it isn't - then the theory is an interval.
                dt_c = float(ev["t1"] - ev["t0"]) if ev is not None and "t0" in ev else 0.0
                mu_ = compiled.get("mu", {})
                J_f = (float(compiled.get("gravity", G)) * dt_c * (ma * mu_.get(na, 0.5) + mb * mu_.get(nb, 0.5))) if on_floor else 0.0
                if J_f > 0.08 * p_scale_m:
                    add(f"Table friction during the {dt_c * 1000:.0f} ms contact (at most μ·m·g·Δt)", J_f, "kg·m/s", "system", 3,
                        "the floor's friction acts while they're in contact, so momentum can drop by up to this much")
                    add("Momentum lost in the hit (must be ≤ that friction)", dpn, "kg·m/s", "system", 3)
                else:
                    J_f = 0.0
                    add("Momentum along the line of impact conserved (after ÷ before)", 100 * (1 - dpn / p_scale_m), "%", "system", 2,
                        "along the line of impact nothing outside acts during the hit (the floor only pushes up/down), so momentum there must stay the same")
                add("Restitution set for these materials √(e₁e₂)", e_set, "", "pair", 2)
                add(f"{na} velocity after, along the impact (theory)", ua_th, "m/s", na, 2, "momentum conservation + restitution")
                add(f"{na} velocity after, along the impact (measured)", ua_m, "m/s", na, 2)
                add(f"{nb} velocity after, along the impact (theory)", ub_th, "m/s", nb, 2)
                add(f"{nb} velocity after, along the impact (measured)", ub_m, "m/s", nb, 2)
                # with friction during the contact, each velocity may fall short of the free-collision value by up to J_f/m
                ag_c = [100 * (1 - max(0.0, abs(ua_m - ua_th) - J_f / ma) / scale), 100 * (1 - max(0.0, abs(ub_m - ub_th) - J_f / mb) / scale)]
                if e is not None:
                    # e is a fraction of the impact speed: for soft, low-bounce pairs compare it absolutely (the rebound-speed
                    # error as a share of the impact speed); a relative error on 0.09 vs 0.10 would overstate a 1 % difference
                    ag_c.append(agreement(e, e_set) if e_set > 0.3 else 100 * (1 - abs(e - e_set)))
                ag_c.append(100 * (1 - max(0.0, dpn - J_f) / p_scale_m))
                ag_c = [max(0.0, x) for x in ag_c if x is not None]
            add(f"{na} speed just before impact", float(np.linalg.norm(va0)), "m/s", na)
            add(f"{nb} speed just before impact", float(np.linalg.norm(vb0)), "m/s", nb)
            add(f"{na} speed just after", float(np.linalg.norm(va1)), "m/s", na)
            add(f"{nb} speed just after", float(np.linalg.norm(vb1)), "m/s", nb)
            add("Total momentum before", float(np.linalg.norm(p0)), "kg·m/s", "system", 3)
            add("Total momentum after", float(np.linalg.norm(p1)), "kg·m/s", "system", 3)
            add("Kinetic energy before", ke0, "J", "system", 3)
            add("Kinetic energy after", ke1, "J", "system", 3)
            if ke0 > 1e-9 and not active:
                add("Energy lost in the collision", 100 * (ke0 - ke1) / ke0, "%", "system", 0)
            if e is not None and not active:
                add("Coefficient of restitution", e, "", "pair", 2, "how bouncy the impact was: 1 = perfectly elastic, 0 = they stick")
            add("Collision impulse", J, "N·s", "pair", 3)
            add("Peak collision force", Fpk, "N", "pair", 1)
            add("Contact duration", max(dt_f, J / max(1e-9, Fpk) * 2) * 1000, "ms", "pair", 0)
            if ag_c:
                add("Agreement with theory", min(ag_c), "%", "pair", 1)
            if a["do"] == "fight":
                add("Hits", len(eps), "", "pair", 0)
                add("Hardest hit", max(max(c[2] for c in ep) for ep in eps), "N", "pair", 1)
            # why
            p_scale = max(float(np.linalg.norm(p0)), ma * float(np.linalg.norm(va0)) + mb * float(np.linalg.norm(vb0)), 1e-9)
            dp = float(np.linalg.norm(p1 - p0)) / p_scale
            ext = "friction with the ground" if compiled["setting"] not in ("water", "space", "air") else ("water drag" if compiled["setting"] == "water" else "nothing")
            story.append(
                f"At t = {tc:.2f} s the {na} ({_fmt(ma, 'kg')}) at {np.linalg.norm(va0):.2f} m/s met the {nb} ({_fmt(mb, 'kg')}) at {np.linalg.norm(vb0):.2f} m/s. "
                f"For the {max(dt_f, J / max(1e-9, Fpk) * 2) * 1000:.0f} ms they were in contact they pushed on each other with equal and opposite forces "
                f"(Newton's third law), peaking at {_fmt(Fpk, 'N', 0)} and adding up to an impulse of {J:.3g} N·s.")
            fixed_b = bool(things.get(nb) is not None and things[nb].extra.get("fixed"))
            if fixed_b:
                add(f"Force the {nb} took", Fpk, "N", nb, 1)
                story.append(
                    f"The {nb} is anchored to the ground, so it didn't move: the ground (and the whole Earth) took the "
                    f"{_fmt(float(np.linalg.norm(p0 - p1)), 'kg·m/s')} of momentum the {na} lost. Its foundation pushed back with the same "
                    f"{_fmt(Fpk, 'N', 0)} — that reaction force is what stopped the {na}.")
            else:
                story.append(
                    f"Head-on and equal, their momenta cancel: the total is about zero before ({_fmt(float(np.linalg.norm(p0)), 'kg·m/s')}) "
                    f"and stays about zero after ({_fmt(float(np.linalg.norm(p1)), 'kg·m/s')}) — each one's momentum is undone by the other's equal push."
                    if float(np.linalg.norm(p0)) < 0.1 * p_scale else
                    f"Momentum before {_fmt(float(np.linalg.norm(p0)), 'kg·m/s')}, after {_fmt(float(np.linalg.norm(p1)), 'kg·m/s')}: "
                    + (f"conserved within {dp * 100:.0f}% of the momentum in play (only external forces like {ext} can change it)." if dp < 0.15 else
                       f"a {dp * 100:.0f}% change because {ext}{' and their own propulsion' if active else ''} acted during the impact."))
            if active:
                story.append("Muscles/engines kept pushing through the impact, so energy was being added: momentum and impulse tell the story here, not energy loss.")
            elif ke0 > 1e-9:
                story.append(
                    f"Kinetic energy fell from {_fmt(ke0, 'J')} to {_fmt(ke1, 'J')}: {100 * (ke0 - ke1) / ke0:.0f}% went into deformation, heat and sound"
                    + (f" (restitution {e:.2f}: {'a bouncy' if e > 0.7 else 'a fairly soft' if e > 0.3 else 'an almost sticky'} impact)." if e is not None else "."))
            if ma > 3 * mb or mb > 3 * ma:
                heavy, light = (na, nb) if ma > mb else (nb, na)
                story.append(f"The {heavy} is {max(ma, mb) / min(ma, mb):.0f}× heavier, so the same impulse changed the {light}'s velocity far more (Δv = J/m).")
        elif a["do"] in ("drop", "bounce") and sa is not None:
            if not _fluid(add, story, na, sa, t0, g, things.get(na), compiled, mass[A]):
                dr = _drop(add, story, na, sa, t0, g, _medium(compiled, things.get(na), mass[A]))
                if dr:
                    drops.append(dr)
        elif a["do"] == "throw" and sa is not None:
            v_cmd = next((iv_["params"]["velocity"] for iv_ in compiled["interventions"] if iv_["type"] == "launch" and iv_["target"] == A), None)
            _throw(add, story, na, sa, t0, g, _medium(compiled, things.get(na), mass[A]), v_cmd)
        elif a["do"] in ("rope", "pull") and B:
            for lk in sim.get("links", []):
                if {lk["a"], lk["b"]} == {A, B}:
                    T = np.array([x[1] for x in sim["tensions"][lk["id"]]][int(t0 * sim["fps"]):])
                    if len(T):
                        add("Rope tension (average while taut)", float(T[T > 1e-3].mean()) if (T > 1e-3).any() else 0.0, "N", "rope", 1)
                        add("Rope tension (peak)", float(T.max()), "N", "rope", 1)
            if a["do"] == "pull" and sb is not None:
                i0 = _at(sb, t0)
                moved = float(np.linalg.norm(sb["p"][-1][:2] - sb["p"][i0][:2]))
                add(f"{nb} moved", moved, "m", nb)
                add(f"{nb} top speed", float(sb["speed"][i0:].max()), "m/s", nb)
                resist = things[nb].mass * g * compiled.get("mu", {}).get(nb, 0.5)
                lnk = next((l for l in sim.get("links", []) if {l["a"], l["b"]} == {A, B}), None)
                Tser = [x[1] for x in sim["tensions"][lnk["id"]]][int(t0 * sim["fps"]):] if lnk else []
                Tm = float(np.mean([x for x in Tser if x > 1e-3])) if any(x > 1e-3 for x in Tser) else 0.0
                add(f"{nb} resistance to motion", resist, "N", nb, 0)
                add(f"Most the {na} can pull (grip μ·m·g)", compiled.get("pull_max", {}).get(na, 0.0), "N", na, 0)
                if moved < 0.01:
                    story.append(f"The {nb} didn't move: the {na} can pull at most {_fmt(compiled.get('pull_max', {}).get(na, 0.0), 'N', 0)} before their feet slip, "
                                 f"but the {nb} needs {_fmt(resist, 'N', 0)} just to start rolling ({things[nb].mass:.0f} kg × g × rolling coefficient). "
                                 f"A {nb} under {compiled.get('pull_max', {}).get(na, 0.0) / (g * max(1e-6, compiled.get('mu', {}).get(nb, 0.5))):.0f} kg would move.")
                else:
                    acc = max(0.0, (Tm - resist) / things[nb].mass)
                    story.append(f"The rope pulls with about {Tm:.0f} N; the {nb}'s rolling resistance is {resist:.0f} N, so the net {max(0.0, Tm - resist):.0f} N "
                                 f"accelerates {things[nb].mass:.0f} kg at only ~{acc:.3f} m/s² (a = F/m). That's why it creeps: {moved:.2f} m, top speed {float(sb['speed'][i0:].max()):.2f} m/s. "
                                 f"The {na} can't pull harder than their grip on the ground allows (μ·m·g).")
        elif a["do"] == "spring" and B and sb is not None:
            for lk in sim.get("links", []):
                if {lk["a"], lk["b"]} == {A, B}:
                    L = np.array([x[0] for x in sim["tensions"][lk["id"]]][int(t0 * sim["fps"]):])
                    period = _period(L)
                    hung = na == ANCHOR
                    rolling = not hung and compiled["setting"] in ("table", "ground", "scene") and all(
                        things[x].shape == "sphere" or (things[x].source == "library" and "ball" in x) for x in (na, nb))
                    if period:
                        m_eff = mass[B] if hung else mass[A] * mass[B] / (mass[A] + mass[B])
                        if rolling:
                            m_eff *= 1.4          # a rolling solid ball also stores rotational energy: m_eff = 7/5 m
                        theory = 2 * math.pi * math.sqrt(m_eff / lk["stiffness"])
                        add("Oscillation period (measured)", period, "s", "spring", 3)
                        add("Oscillation period (theory 2π√(m/k))" if hung else ("Oscillation period (theory 2π√(1.4μ/k), rolling)" if rolling else "Oscillation period (theory 2π√(μ/k))"), theory, "s", "spring", 3)
                        add("Spring stiffness k", lk["stiffness"], "N/m", "spring", 1)
                        add("Agreement with theory", agreement(period, theory), "%", "spring", 1)
                        add("Stretch range", float(L.max() - L.min()), "m", "spring", 3)
                        if hung:
                            add("Equilibrium stretch (mg/k)", mass[B] * g / lk["stiffness"], "m", "spring", 3)
                            story.append(f"Gravity pulls the {nb} down and the spring ({lk['stiffness']:.1f} N/m) pulls back in proportion to its stretch (Hooke's law, F = −kx). "
                                         f"It overshoots the balance point (stretch mg/k = {mass[B] * g / lk['stiffness']:.3f} m) and bobs every {period:.2f} s; theory 2π√(m/k) = {theory:.2f} s, "
                                         f"independent of how far you pull it.")
                        else:
                            story.append(f"The spring ({lk['stiffness']:.0f} N/m) pulls back proportionally to its stretch (Hooke's law), so the pair oscillates every {period:.2f} s; theory predicts {theory:.2f} s.")
                            if rolling:
                                story.append("The balls roll rather than slide, so part of the energy goes into spinning them: they behave as if 7/5 heavier, "
                                             "which stretches the period by √1.4 ≈ 1.18× compared with sliding blocks.")
        elif a["do"] in ("swing", "rope") and na == ANCHOR and sb is not None:
            lk = next((l for l in sim.get("links", []) if {l["a"], l["b"]} == {A, B}), None)
            if lk is not None:
                i0 = _at(sb, t0)
                anc = S[A]["p"][i0] if A in S else sb["p"][i0]
                x = sb["p"][i0:, 0] - anc[0]
                period = _period(x)
                Lr = lk["length"]
                theta0 = math.degrees(math.asin(min(1.0, abs(float(x[0])) / max(1e-6, Lr))))
                # exact period at any amplitude: T = 2π√(L/g) / AGM(1, cos(θ₀/2))  (complete elliptic integral)
                theory = 2 * math.pi * math.sqrt(Lr / g) / _agm(1.0, math.cos(math.radians(theta0) / 2))
                T_small = 2 * math.pi * math.sqrt(Lr / g)
                T = np.array([v[1] for v in sim["tensions"][lk["id"]]][int(t0 * sim["fps"]):])
                vmax = float(sb["speed"][i0:].max())
                h = Lr * (1 - math.cos(math.radians(theta0)))
                ag = []
                if period:
                    add("Swing period (measured)", period, "s", nb, 3)
                    add("Swing period (theory, exact for this amplitude)", theory, "s", nb, 3,
                        f"2π√(L/g) = {T_small:.3f} s for small swings, longer at {theta0:.0f}°: 2π√(L/g)/AGM(1, cos(θ₀/2))")
                    ag.append(agreement(period, theory))
                add("String length", Lr, "m", nb)
                add("Release angle", theta0, "°", nb, 0)
                add("Speed at the bottom (measured)", vmax, "m/s", nb)
                add("Speed at the bottom (theory √(2gh))", math.sqrt(2 * g * h), "m/s", nb)
                ag.append(agreement(vmax, math.sqrt(2 * g * h)))
                if len(T):
                    T_th = mass[B] * g * (3 - 2 * math.cos(math.radians(theta0)))
                    # tension each time the bob passes the bottom (its fastest moments), first swings only
                    spd = sb["speed"][i0:]
                    n_ = min(len(spd), len(T))
                    bot = [k for k in range(1, n_ - 1) if spd[k] >= spd[k - 1] and spd[k] >= spd[k + 1] and spd[k] > 0.8 * spd[:n_].max()][:4]
                    T_bot = float(np.mean([T[k] for k in bot])) if bot else float(T.max())
                    add("String tension at the bottom (measured)", T_bot, "N", "string", 2)
                    add("String tension at the bottom (theory mg(3 − 2cosθ₀))", T_th, "N", "string", 2, "weight + centripetal force mv²/L")
                    add("Weight of the bob (mg)", mass[B] * g, "N", nb, 1)
                    ag.append(agreement(T_bot, T_th))
                ag = [x for x in ag if x is not None]
                if ag:
                    add("Agreement with theory", min(ag), "%", nb, 1)
                story.append(f"Released at {theta0:.0f}° on a {Lr:.2f} m string, the {nb} falls {h * 100:.1f} cm and trades that height for speed: "
                             f"at the bottom it moves at {vmax:.2f} m/s (energy conservation predicts √(2gh) = {math.sqrt(2 * g * h):.2f} m/s).")
                if period:
                    story.append(f"It swings back and forth every {period:.3f} s (theory {theory:.3f} s). The period depends on the string length and gravity — "
                                 f"2π√(L/g) = {T_small:.3f} s for small swings, a little longer for wide ones — not on the mass of the {nb}.")
                if len(T):
                    story.append(f"The string pulls hardest at the bottom ({T_bot:.2f} N vs its weight {mass[B] * g:.2f} N; theory mg(3 − 2cosθ₀) = {T_th:.2f} N) because it must also supply the centripetal force mv²/L.")
        elif a["do"] == "incline" and sa is not None:
            _incline(add, story, na, sa, t0, g, things.get(na), compiled, a)
        elif a["do"] == "move" and sa is not None:
            _motion(add, story, na, sa, t0, things.get(na), compiled, a.get("mode"), g, mass.get(A) or 1.0)
        elif a["do"] == "jump" and sa is not None:
            _jump(add, story, na, sa, t0, g)
        elif a["do"] == "race":
            for nm_r, sid_r in sid_of.items():
                if sid_r in S and things.get(nm_r) is not None and things[nm_r].self_propelled:
                    _motion(add, story, nm_r, S[sid_r], t0, things[nm_r], compiled, a.get("mode"), g, mass.get(sid_r) or 1.0, brief=True)
            dist = {}
            for nm, sid in sid_of.items():
                s_ = S.get(sid)
                if s_ is not None:
                    i0 = _at(s_, t0)
                    dist[nm] = float(s_["p"][-1][0] - s_["p"][i0][0])
            if dist:
                win = max(dist, key=dist.get)
                for nm, d_ in sorted(dist.items(), key=lambda kv: -kv[1]):
                    add(f"{nm} distance", d_, "m", nm)
                story.append(f"The {win} wins: with traction-limited acceleration, the one with more push per kilogram and less resistance pulls ahead.")
        elif a["do"] == "spin" and sa is not None:
            w = np.array(sim["kinematics"][A]["w"][int(t0 * sim["fps"]):])
            w = w[[x is not None for x in w]] if len(w) else w
            if len(w):
                ws = np.linalg.norm(np.array(list(w), float), axis=1)
                add("Spin rate at start", ws[0] * 60 / (2 * math.pi), "rpm", na, 0)
                add("Spin rate at end", ws[-1] * 60 / (2 * math.pi), "rpm", na, 0)
                story.append("Friction torque at the contact slowly drains the spin's rotational energy.")

    if len(drops) >= 2:
        first, last = min(drops, key=lambda d: d["t"]), max(drops, key=lambda d: d["t"])
        add(f"{first['name']} landed earlier by", last["t"] - first["t"], "s", "comparison")
        if last["t"] - first["t"] < 0.03:
            story.insert(0, f"They landed together ({first['t']:.2f} s). With no air to push back, nothing slows the lighter one: "
                            f"gravity gives every object the same acceleration ({g:.2f} m/s² here) regardless of its mass — "
                            f"exactly what Apollo 15's hammer-and-feather drop on the Moon showed.")
        else:
            story.insert(0, f"The {first['name']} landed first, {last['t'] - first['t']:.2f} s before the {last['name']}. Gravity pulls every kilogram equally "
                        f"({g:.2f} m/s² for all of them): in a vacuum they would land together, as Galileo argued and Apollo 15 showed with a hammer and a "
                        f"feather on the Moon. The difference here is air drag, which matters far more for light, broad things.")

    # per-object summary (always)
    for sid, nm in name_of.items():
        s_ = S.get(sid)
        if s_ is None or nm == ANCHOR or (things.get(nm) is not None and things[nm].extra.get("fixed")):
            continue
        i0 = _at(s_, t0)
        add(f"{nm}: top speed", float(s_["speed"][i0:].max()), "m/s", nm)
        add(f"{nm}: distance travelled", float(np.sum(np.linalg.norm(np.diff(s_["p"][i0:], axis=0), axis=1))), "m", nm)
    picks = set(x.lower() for x in (extra or []))
    for r in rows:
        l = r["label"].lower()
        r["picked"] = any(p in l for p in picks) or (("momentum" in picks) and "momentum" in l) or \
            (("energy" in picks) and "energy" in l) or (("force" in picks) and ("force" in l or "impulse" in l))
    if not story:
        story.append("Nothing collided or changed much: the objects followed their initial motion until friction or drag stopped them.")
    return {"rows": rows, "story": story}


def _motion(add, story, na, sa, t0, t, compiled, mode, g, m, brief=False):
    """Self-propelled motion: how fast, how quickly, how far — and what limits it."""
    from .experiment import knowledge
    i0 = _at(sa, t0)
    sp = sa["speed"][i0:]
    tt = sa["t"][i0:] - sa["t"][i0]
    if len(sp) < 3:
        return
    vmax = float(sp.max())
    k90 = next((k for k in range(len(sp)) if sp[k] >= 0.9 * vmax), len(sp) - 1)
    t90 = float(tt[k90])
    dist = float(np.sum(np.linalg.norm(np.diff(sa["p"][i0:], axis=0), axis=1)))
    acc = np.diff(sp) / np.maximum(1e-6, np.diff(tt))
    if len(acc) > 5:
        acc = np.convolve(acc, np.ones(5) / 5, mode="same")
    a_avg = 0.9 * vmax / max(t90, 1e-3)
    P = float(np.max(m * np.clip(acc, 0, None) * sp[1:])) if len(acc) else 0.0
    arch = knowledge.archetype(na, t) if t is not None else "generic"
    setting = compiled["setting"]
    add(f"{na} top speed", vmax, "m/s", na, 2, f"{vmax * 3.6:.1f} km/h")
    add(f"{na} top speed (km/h)", vmax * 3.6, "km/h", na, 1)
    add(f"{na} time to reach top speed (90%)", t90, "s", na, 2)
    add(f"{na} average acceleration", a_avg, "m/s²", na, 2)
    add(f"{na} distance covered", dist, "m", na, 1)
    if P > 0:
        add(f"{na} peak power (mass × acceleration × speed)", P, "W", na, 0)
    if brief:
        return
    W = m * g
    if setting == "water":
        rho = float(compiled.get("env", {}).get("density", 1000.0))
        A = math.pi / 4 * t.dims[1] * t.dims[2] if t else 0.1
        Fd = 0.5 * rho * 0.6 * A * vmax ** 2
        add(f"Water drag on the {na} at top speed (½ρC_dAv²)", Fd, "N", na, 1)
        story.append(f"The {na} pushes water backwards and the water pushes it forwards (Newton's third law). Water is ~850× denser than air, "
                     f"so drag rises steeply with speed (∝ v²): at {vmax:.2f} m/s it is about {Fd:.0f} N, matching the push the {na} can make, "
                     f"so it can't go faster. It covered {dist:.1f} m in {float(tt[-1]):.1f} s.")
        if arch == "biped":
            story.append(f"A person's body (about 985 kg/m³) is slightly less dense than the water ({rho:.0f} kg/m³), so buoyancy holds them near the surface while they swim.")
    elif mode == "fly" or (t is not None and t.habitat == "air" and arch in ("winged", "aircraft")):
        add(f"Lift needed to stay up (= weight m·g)", W, "N", na, 1)
        story.append(f"To stay at the same height the {na}'s {'wings' if arch == 'winged' else 'wings and engines'} must push air down hard enough to make "
                     f"a lift force equal to its weight, {W:,.1f} N. Its thrust beats air drag until {vmax:.1f} m/s ({vmax * 3.6:.0f} km/h), where they balance.")
    else:
        grip = 0.7 * W
        what = "tyres" if arch in ("vehicle",) else "feet"
        story.append(f"Each time its {what} push backwards on the ground, friction pushes the {na} forwards (Newton's third law). That push can't exceed the grip "
                     f"μ·m·g ≈ {grip:,.0f} N, so it speeds up at about {a_avg:.1f} m/s² and reaches {0.9 * vmax:.1f} m/s in {t90:.1f} s. Then "
                     + ("air drag and the engine's power limit" if arch == "vehicle" else "how fast its legs can swing")
                     + f" cap it at {vmax:.1f} m/s ({vmax * 3.6:.0f} km/h). It covered {dist:.1f} m.")
        if g < 5:
            story.append(f"With gravity only {g:.2f} m/s², the {na} weighs less, so the ground grips it less (friction ∝ weight): it can't accelerate as hard as on Earth.")


def _incline(add, story, na, sa, t0, g, t, compiled, a):
    """Block/ball on a ramp: measured acceleration along the slope vs g(sinθ − μcosθ) or (5/7)g·sinθ."""
    th = math.radians(a.get("angle_deg") or 30.0)
    mu = compiled.get("mu", {}).get(na, 0.5)
    ramp = compiled["things"].get("ramp")
    L = ramp.dims[0] if ramp else 2.0
    roll = t is not None and t.shape in ("sphere",) or (t is not None and t.source == "library" and "ball" in na)
    i0 = _at(sa, t0)
    sp, tt, p = sa["speed"][i0:], sa["t"][i0:] - sa["t"][i0], sa["p"][i0:]
    # on the ramp until it has travelled the ramp length (or stops)
    travelled = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    on = travelled < 0.92 * (L - (t.dims[0] if t else 0.1) * 0.6 - 0.05)
    k_end = int(np.argmin(on)) if (~on).any() else len(on) - 1
    sel = slice(1, max(3, int(k_end * 0.85)))
    if k_end > 4:
        a_meas = float(np.polyfit(tt[sel], sp[sel], 1)[0])
    else:
        a_meas = 0.0
    if roll:
        from .physics.sim import C_RR
        crr = C_RR.get(t.material if t is not None else "", 0.005)
        a_th = 5.0 / 7.0 * g * (math.sin(th) - crr * math.cos(th))
        how = "(5/7)·g·(sinθ − c_rr·cosθ) (a rolling ball spends some energy spinning; c_rr = rolling resistance)"
    else:
        a_th = max(0.0, g * (math.sin(th) - mu * math.cos(th)))
        how = "g(sinθ − μcosθ)"
    stays = not roll and math.tan(th) <= mu
    # the same law with air drag along the slope, solved over the same moments the acceleration was measured
    a_cmp = a_th
    med = _medium(compiled, t, t.mass if t is not None else 1.0)
    if med["rho"] > 0 and k_end > 4 and a_th > 0.05:
        from .physics import drag
        k_in = 5.0 / 7.0 if roll else 1.0
        u_ = (1.0, 0.0, 0.0) if not roll else (0.0, 0.0, 1.0)
        A_ = drag.area(med["sp"], u_)
        kq = 0.5 * med["rho"] * med["sp"]["cd"] * A_
        ts_, vs_, v_, dt_ = [0.0], [0.0], 0.0, 1e-4
        for _ in range(int(float(tt[k_end]) / dt_) + 2):
            v_ += dt_ * (a_th - k_in * kq * v_ * v_ / med["m"])
            ts_.append(ts_[-1] + dt_); vs_.append(v_)
        v_th_at = np.interp(tt[sel], ts_, vs_)
        a_cmp = float(np.polyfit(tt[sel], v_th_at, 1)[0])
    add("Ramp angle", math.degrees(th), "°", "ramp", 0)
    if not roll:
        add("Friction coefficient μ", mu, "", "ramp", 2)
    add(f"{na} acceleration down the ramp (measured)", a_meas, "m/s²", na, 2)
    add(f"Acceleration (theory {how.split(' (')[0]})", a_th, "m/s²", na, 2, how[how.find('(') + 1:-1] if '(' in how else None)
    if roll:
        add("Rolling resistance coefficient c_rr", crr, "", na, 3)
    if abs(a_cmp - a_th) > 0.001 * a_th:
        add("Acceleration (theory with air drag)", a_cmp, "m/s²", na, 2, "the same law plus air drag ½ρC_dAv², solved over the same stretch")
    if a_th > 0.05:
        add("Agreement with theory", agreement(a_meas, a_cmp), "%", na, 1)
    if k_end > 2 and not stays:
        add(f"{na} speed at the bottom (measured)", float(sp[k_end]), "m/s", na)
        add(f"{na} time to reach the bottom", float(tt[k_end]), "s", na)
        add("Speed at the bottom (theory √(2aL))", math.sqrt(2 * a_th * max(0.0, travelled[k_end])), "m/s", na)
    W = (t.mass if t else 1.0) * g
    add(f"{na} weight component along the ramp (m·g·sinθ)", W * math.sin(th), "N", na, 2)
    add(f"{na} normal force from the ramp (m·g·cosθ)", W * math.cos(th), "N", na, 2)
    if not roll:
        add(f"Friction force (μ·m·g·cosθ)", mu * W * math.cos(th), "N", na, 2)
    story.append(f"Gravity pulls the {na} straight down with its weight {W:.2f} N. On a {math.degrees(th):.0f}° slope that splits into two parts: "
                 f"{W * math.sin(th):.2f} N along the slope (m·g·sinθ), pulling it down the ramp, and {W * math.cos(th):.2f} N into the ramp (m·g·cosθ), "
                 f"which the ramp pushes back against.")
    if stays:
        story.append(f"Friction can resist up to μ × {W * math.cos(th):.2f} N = {mu * W * math.cos(th):.2f} N — more than the {W * math.sin(th):.2f} N pulling it "
                     f"down — so the {na} stays put. It only slides once tanθ > μ, i.e. above {math.degrees(math.atan(mu)):.0f}° for μ = {mu:.2f}.")
    elif roll:
        story.append(f"The ball rolls rather than slides, so part of gravity's work goes into spinning it: its acceleration is (5/7)·g·(sinθ − c_rr·cosθ) = {a_th:.2f} m/s² "
                     f"(measured {a_meas:.2f} m/s²) instead of g·sinθ = {g * math.sin(th):.2f} m/s² for something sliding without friction."
                     + f" A real ball also squashes slightly where it touches, giving a little rolling resistance (c_rr = {crr:.3f}), which the formula includes.")
    else:
        story.append(f"Friction pushes back with μ·m·g·cosθ = {mu * W * math.cos(th):.2f} N, so the net force along the slope gives an acceleration of "
                     f"g(sinθ − μcosθ) = {a_th:.2f} m/s²; the simulation measured {a_meas:.2f} m/s². Notice the mass cancels out: a heavier block "
                     f"slides down exactly as fast.")


def _jump(add, story, na, sa, t0, g):
    i0 = _at(sa, t0)
    z = sa["p"][i0:, 2]
    vz = sa["v"][i0:, 2]
    tt = sa["t"][i0:]
    if len(z) < 3:
        return
    v0 = float(vz[:4].max())
    h = float(z.max() - z[0])
    land = next((k for k in range(2, len(z)) if z[k] <= z[0] + 0.01 and vz[k - 1] < 0), None)
    add(f"{na} take-off speed", v0, "m/s", na)
    add(f"{na} jump height", h, "m", na, 2)
    add("Jump height (theory v²/2g)", v0 * v0 / (2 * g), "m", na, 2)
    ag = [agreement(h, v0 * v0 / (2 * g))]
    if land:
        add(f"{na} time in the air", float(tt[land] - tt[0]), "s", na, 2)
        add("Time in the air (theory 2v/g)", 2 * v0 / g, "s", na, 2)
    ag = [x for x in ag if x is not None]
    if ag:
        add("Agreement with theory", min(ag), "%", na, 1)
    story.append(f"The {na}'s legs launch it upward at {v0:.2f} m/s. Gravity ({g:.2f} m/s²) slows it by {g:.2f} m/s every second, so it stops rising after "
                 f"v/g = {v0 / g:.2f} s at a height of v²/2g ≈ {v0 * v0 / (2 * g):.2f} m, then falls back the same way.")
    if abs(g - 9.81) > 0.5:
        story.append(f"The same muscles give the same take-off speed anywhere, so the height scales with 1/g: on Earth this jump would reach only "
                     f"{v0 * v0 / (2 * 9.81):.2f} m — here it's {9.81 / g:.1f}× higher.")


def _agm(a: float, b: float) -> float:
    """Arithmetic-geometric mean (gives the complete elliptic integral K exactly: K(k) = π / (2·AGM(1, √(1−k²))))."""
    for _ in range(40):
        a, b = (a + b) / 2, math.sqrt(a * b)
    return a


def _crossings(x, fps: float = 60.0) -> np.ndarray:
    """Times at which x crosses its mid-level, interpolated between samples (sub-frame precision)."""
    x = np.asarray(x, float)
    c = 0.5 * (x.max() + x.min())
    y = x - c
    i = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    return (i + y[i] / (y[i] - y[i + 1])) / fps


def _period(x, fps: float = 60.0) -> float | None:
    """Oscillation period from interpolated mid-level crossings (needs at least 1.5 cycles).
    Uses the first few cycles, where damping has changed the amplitude least."""
    x = np.asarray(x, float)
    if len(x) < 10 or np.ptp(x) < 1e-4:
        return None
    tc = _crossings(x, fps)
    if len(tc) < 3:
        return None
    tc = tc[:9]
    return 2 * float(tc[-1] - tc[0]) / (len(tc) - 1)


def volume_of(t) -> float:
    L, W, H = t.dims
    k = {"sphere": math.pi / 6, "ellipsoid": math.pi / 6, "cylinder": math.pi / 4, "capsule": 0.72}.get(t.shape, 0.6 if t.shape == "asset" else 1.0)
    return k * L * W * H


def _fluid(add, story, na, sa, t0, g, t, compiled, m=None) -> bool:
    """Sinking/floating in water, or rising/falling through air where buoyancy and drag matter.
    Theory: Newton's 2nd law with weight, buoyancy ρVg and drag ½ρC_dAv², solved exactly, plus the
    floating equilibrium (weight = weight of displaced water). Returns False when ordinary fall analysis applies."""
    from .physics import drag
    from .physics.sim import submerged_fraction
    rho_f = float(compiled.get("env", {}).get("density", 0.0))
    if t is None or rho_f <= 0:
        return False
    med = _medium(compiled, t, m if m else t.mass)
    m, V, sp = med["m"], med["V"], med["sp"]
    if V <= 0:
        return False
    rho = m / V
    W, Fb = m * g, rho_f * V * g
    if rho_f < 10 and Fb < 0.05 * W:
        return False                                  # air barely matters for dense things: ordinary fall
    i0 = _at(sa, t0)
    z, vz = sa["p"][i0:, 2], sa["v"][i0:, 2]
    rises = Fb > W
    fluid = "water" if rho_f > 100 else "air"
    add(f"{na} density", rho, "kg/m³", na, 0)
    add(f"{fluid.title()} density", rho_f, "kg/m³", "fluid", 1)
    add(f"{na} weight (mg)", W, "N", na, 2)
    add(f"Buoyant force on the {na} (ρ·V·g, fully under)", Fb, "N", na, 2)
    add(f"Net force ({'up' if rises else 'down'})", abs(Fb - W), "N", na, 2)
    dz = float(z[-1] - z[0])
    add(f"{na} {'rose' if dz > 0 else 'sank' if rho_f > 100 else 'fell'}", abs(dz), "m", na)
    # fastest moment of the rise/sink, and theory at the same distance travelled
    sgn = 1.0 if rises else -1.0
    surf0 = compiled.get("env", {}).get("surface_z")
    under = np.ones(len(z), bool) if surf0 is None or fluid != "water" else (z + t.dims[2] / 2 < surf0)
    # compare while fully under water and before reaching the bottom (contact) — the law being tested
    if (~under).any() and under.any():
        under[int(np.argmin(under)):] = False
    k = int(np.argmax(np.where(under, sgn * vz, -np.inf))) if under.any() else (int(np.argmax(sgn * vz)) if len(vz) else 0)
    v_meas = float(abs(vz[k])) if len(vz) else 0.0
    d_k = float(abs(z[k] - z[0]))
    g_net = abs(m - rho_f * V) * g / m
    ma = drag.added_mass(sp, rho_f, V)
    th = drag.fall(m, 0.0, sp, rho_f, med["mu"], g_net, h=d_k, ma=ma) if d_k > 1e-4 else {"v": 0.0, "terminal": 0.0}
    vt = abs(drag.fall(m, 0.0, sp, rho_f, med["mu"], g_net, t_end=0.0)["terminal"])
    add(f"{'Rising' if rises else 'Sinking'} speed at its fastest (measured)", v_meas, "m/s", na)
    add(f"{'Rising' if rises else 'Sinking'} speed at its fastest (theory, Newton's 2nd law with drag)", abs(th["v"]), "m/s", na, 2,
        f"(m + m_a)·dv/dt = |ρVg − mg| − ½ρC_dAv², solved exactly (m_a = {ma:.3g} kg of water it pushes along)")
    add(f"{na} terminal speed (theory)", vt, "m/s", na, 2, f"drag ½ρC_dAv² = net force; C_d = {sp['cd']:.2f} for a {sp['shape']}")
    ag = [agreement(v_meas, abs(th["v"]))]
    surf = compiled.get("env", {}).get("surface_z")
    floats_txt = ""
    if rises and fluid == "water" and surf is not None:
        H = t.dims[2]
        n_avg = min(len(z), 60)                           # the water line averaged over the last second (it bobs)
        bottom = float(np.mean(z[-n_avg:])) - H / 2
        x_meas = float(np.clip((surf - bottom) / H, 0, 1))
        settled = len(vz) > 30 and float(np.abs(vz[-30:]).max()) < 0.05
        target = rho / rho_f
        xs = np.linspace(0, 1, 20001)
        frac = np.array([submerged_fraction({"drag": sp}, x) for x in xs[::100]])
        x_th = float(np.interp(target, frac, xs[::100]))
        if settled:
            add(f"{na} floats with this much of its height under water (measured)", 100 * x_meas, "%", na, 1)
            add("…(theory: weight = weight of the water it pushes aside)", 100 * x_th, "%", na, 1,
                f"submerged volume fraction = ρ_object/ρ_water = {target:.3f}")
            ag.append(agreement(x_meas, x_th))
            floats_txt = (f" At the surface it settles with {100 * x_meas:.0f}% of its height under water — where the water it displaces weighs exactly "
                          f"as much as it does: {100 * target:.0f}% of its volume submerged = ρ_object/ρ_water (theory {100 * x_th:.0f}% of its height).")
    ag = [a for a in ag if a is not None]
    if ag:
        add("Agreement with theory", min(ag), "%", na, 1)
    if rises:
        story.append(f"The {na} ({rho:.0f} kg/m³) is less dense than {fluid} ({rho_f:.0f} kg/m³), so the {fluid} it pushes aside weighs more than it does: "
                     f"buoyancy {Fb:.2f} N beats its weight {W:.2f} N and it rises (Archimedes' principle). Drag ½ρC_dAv² grows until it balances the net "
                     f"{abs(Fb - W):.2f} N, limiting it to {vt:.2f} m/s (measured top {v_meas:.2f} m/s)." + floats_txt)
    else:
        story.append(f"The {na} ({rho:.0f} kg/m³) is denser than {fluid} ({rho_f:.0f} kg/m³): its weight {W:.2f} N beats the buoyancy {Fb:.2f} N, "
                     f"so it sinks — but {fluid} drag (½ρC_dAv², C_d {sp['cd']:.2f}) grows with speed until it balances the net {abs(W - Fb):.2f} N, "
                     f"at a terminal speed of {vt:.2f} m/s. Measured: {v_meas:.2f} m/s; Newton's 2nd law at the same depth predicts {abs(th['v']):.2f} m/s. "
                     f"In a vacuum it would keep accelerating at {g:.2f} m/s².")
    return True


def _drop(add, story, na, sa, t0, g, med=None):
    from .physics import drag
    i0 = _at(sa, t0)
    z = sa["p"][i0:, 2]
    vz = sa["v"][i0:, 2]
    tt = sa["t"][i0:]
    # touch-down: falling, then suddenly slowed (gravity and drag alone can never make a faller slow down this way)
    imp = next((k for k in range(1, len(vz)) if vz[k - 1] < -0.05 and vz[k] > vz[k - 1] + 0.03 + 0.02 * abs(vz[k - 1])), None)
    h = float(z[0] - z.min())
    if imp is None:
        add(f"{na} fall height", h, "m", na)
        return None
    k = imp - 1                                   # last recorded instant before touching down
    d = max(0.0, float(z[0] - z[k]))
    vi, ti = float(-vz[k]), float(tt[k] - t0)          # from the moment of release
    v_free = math.sqrt(2 * g * d)
    th = drag.fall(med["m"], med["V"], med["sp"], med["rho"], med["mu"], g, h=d) if med and (med["rho"] > 0 or med["mu"] > 0) else {"t": math.sqrt(2 * d / g) if g > 0 else 0.0, "v": v_free, "terminal": math.inf}
    air = bool(med and med["rho"] > 0)
    add(f"{na} fall height", d, "m", na)
    add(f"{na} impact speed", vi, "m/s", na)
    add("Impact speed (theory" + (", Newton's 2nd law with air drag)" if air else " √(2gh))"), th["v"], "m/s", na, 2,
        "m·dv/dt = mg − ½ρC_dAv² (− buoyancy), solved exactly" if air else "free fall: no air")
    add(f"{na} time to fall", ti, "s", na)
    add("Time to fall (theory" + (", with air drag)" if air else " √(2h/g))"), th["t"], "s", na, 3)
    if air:
        add("Impact speed with no air at all (√(2gh))", v_free, "m/s", na, 2, "for comparison: a vacuum")
        if math.isfinite(th["terminal"]):
            add(f"{na} terminal speed (theory)", th["terminal"], "m/s", na, 2, "where air drag would balance its weight")
    add("Agreement with theory", min(agreement(vi, th["v"]) or 0, agreement(ti, th["t"]) or 0), "%", na, 1)
    after = z[imp:]
    peak = after.max() - z[imp]
    if peak > 0.002:
        add(f"{na} bounce height", float(peak), "m", na, 3)
        e = math.sqrt(peak / max(1e-9, float(z[0] - z[imp])))
        add("Restitution from bounce √(h₂/h₁)", e, "", na)
    if not air:
        story.append(f"With no air, nothing but gravity acts: the {na} gains {g:.2f} m/s every second and hits at √(2gh) = {v_free:.2f} m/s "
                     f"after √(2h/g) = {th['t']:.2f} s — the simulation measured {vi:.2f} m/s after {ti:.2f} s.")
    elif vi > 0.9 * v_free:
        story.append(f"The {na} fell almost freely, gaining speed at {g:.2f} m/s² until it hit at {vi:.2f} m/s; "
                     f"air drag (½ρC_dAv², growing with speed²) shaved it a little below the vacuum value √(2gh) = {v_free:.2f} m/s. "
                     f"Solving Newton's 2nd law with that drag gives {th['v']:.2f} m/s — what the simulation measured.")
    else:
        story.append(f"Air drag dominated the {na}'s fall: drag grows with speed², and for something this light and broad it soon balances the weight "
                     f"(terminal speed {th['terminal']:.2f} m/s), so it came down at {vi:.2f} m/s instead of the {v_free:.2f} m/s it would reach in a vacuum. "
                     f"Newton's 2nd law with air drag predicts {th['v']:.2f} m/s.")
    return {"name": na, "t": ti, "v": vi, "free": v_free}


def _throw(add, story, na, sa, t0, g, med=None, v_cmd=None):
    """Projectile: measured launch, peak, range and flight time vs Newton's laws (gravity + air drag, solved exactly)."""
    from .physics import drag
    i0 = _at(sa, t0)
    z, vz, tt, P = sa["p"][i0:, 2], sa["v"][i0:, 2], sa["t"][i0:], sa["p"][i0:]
    v0 = np.array(v_cmd, float) if v_cmd is not None else sa["v"][min(i0 + 1, len(sa["v"]) - 1)]
    speed = float(np.linalg.norm(v0))
    ang = math.degrees(math.atan2(v0[2], math.hypot(v0[0], v0[1])))
    p0 = P[0] - v0 * max(0.0, float(tt[0] - t0))          # where it was at the instant of launch
    # landing: falling, then suddenly stopped by the ground
    land = next((k for k in range(2, len(vz)) if vz[k - 1] < -0.05 and vz[k] > vz[k - 1] + 0.03 + 0.02 * abs(vz[k - 1])), None)
    k = (land - 1) if land else len(z) - 1
    rng = float(np.linalg.norm(P[k][:2] - p0[:2]))
    peak = float(z[:k + 1].max() - p0[2])
    tfl = float(tt[k] - t0)
    add(f"{na} launch speed", speed, "m/s", na)
    add(f"{na} launch angle", ang, "°", na, 0)
    add(f"{na} launch height", float(p0[2]), "m", na)
    add(f"{na} peak height above launch", peak, "m", na)
    add(f"{na} range (to landing)", rng, "m", na)
    add(f"{na} time of flight", tfl, "s", na)
    vac_peak = (v0[2] ** 2) / (2 * g) if v0[2] > 0 and g > 0 else 0.0
    vh, vv = float(math.hypot(v0[0], v0[1])), float(v0[2])
    dz = float(p0[2] - z[k])
    vac_t = (vv + math.sqrt(max(0.0, vv * vv + 2 * g * dz))) / g if g > 0 else math.inf
    vac_rng = vh * vac_t
    ag = []
    if med is not None and land:
        air = med["rho"] > 0
        if air:
            th = drag.flight(med["m"], med["V"], med["sp"], med["rho"], med["mu"], g, p0, v0, float(z[k]))
            t_peak, t_rng, t_t = th["peak"] - float(p0[2]), th["range"], th["t"]
        else:
            t_peak, t_rng, t_t = vac_peak, vac_rng, vac_t
        lab = " (theory, Newton's laws with air drag)" if air else " (theory, no air)"
        add("Peak height" + lab, t_peak, "m", na, 2, "v_y²/2g in a vacuum" if not air else "gravity + drag ½ρC_dAv², solved exactly")
        add("Range" + lab, t_rng, "m", na, 2, "v_x · t_flight in a vacuum" if not air else None)
        add("Time of flight" + lab, t_t, "s", na, 3)
        if air:
            add("Range with no air at all", vac_rng, "m", na, 2, "for comparison: a vacuum (the textbook parabola)")
        ag = [agreement(rng, t_rng), agreement(tfl, t_t)] + ([agreement(peak, t_peak)] if t_peak > 0.05 else [])
        ag = [a for a in ag if a is not None]
        if ag:
            add("Agreement with theory", min(ag), "%", na, 1)
        story.append(f"Launched at {speed:.2f} m/s and {ang:.0f}°, the {na} keeps its sideways speed while gravity pulls it down at {g:.2f} m/s²: "
                     f"it rose {peak:.2f} m, flew {tfl:.2f} s and landed {rng:.2f} m away. "
                     + (f"Air drag (½ρC_dAv²) steals speed all along, so it falls short of the vacuum parabola's {vac_rng:.2f} m; "
                        f"Newton's laws with drag predict {t_rng:.2f} m." if air else
                        f"With no air the path is a perfect parabola: range {vac_rng:.2f} m, peak v_y²/2g = {vac_peak:.2f} m."))
    else:
        story.append(f"Launched at {speed:.2f} m/s and {ang:.0f}°, the {na} follows a parabola: constant horizontal speed (minus drag) while gravity pulls it down at {g:.2f} m/s².")


# ---------------------------------------------------------------------------
# measurement catalogue: every quantity for every object, for the "measure anything" picker
# ---------------------------------------------------------------------------
CATALOG = [
    ("top speed", "m/s"), ("final speed", "m/s"), ("distance travelled", "m"), ("displacement", "m"),
    ("max height", "m"), ("lowest point", "m"), ("max acceleration", "m/s²"), ("peak momentum", "kg·m/s"),
    ("peak kinetic energy", "J"), ("energy change (KE+PE)", "J"), ("peak contact force", "N"),
    ("peak force from other objects", "N"), ("time moving", "s"), ("max spin", "rpm"), ("mass", "kg"),
]


def catalog(compiled: dict, sim: dict) -> list[dict]:
    from .experiment import ANCHOR
    things, sid_of = compiled["things"], compiled["sid_of"]
    seg = sim["segments"][-1]
    t0 = seg["t_start"]
    fps = sim["fps"]
    f0 = int(t0 * fps)
    out = []
    for nm, sid in sid_of.items():
        if nm == ANCHOR or sid not in sim["kinematics"]:
            continue
        k = sim["kinematics"][sid]
        m = float(sim["masses"].get(sid) or things[nm].mass)
        idx = [f for f in range(f0, len(k["v"])) if k["v"][f] is not None and k["c"][f] is not None]
        if len(idx) < 2:
            continue
        v = np.array([k["v"][f] for f in idx], float)
        c = np.array([k["c"][f] for f in idx], float)
        sp = np.linalg.norm(v, axis=1)
        acc = np.linalg.norm(np.diff(v, axis=0), axis=1) * fps if len(v) > 1 else np.zeros(1)
        if len(acc) > 4:
            acc = np.convolve(acc, np.ones(3) / 3, mode="same")
        KE = np.array([k["KE"][f] or 0.0 for f in idx])
        PE = np.array([k["PE"][f] or 0.0 for f in idx])
        Fc = np.array([k["F_contact"][f] or 0.0 for f in idx])
        Fo = np.array([k["F_objects"][f] or 0.0 for f in idx])
        w = [k["w"][f] for f in idx if k.get("w") and k["w"][f] is not None]
        moving = sp > 0.02
        vals = {
            "top speed": float(sp.max()), "final speed": float(sp[-1]),
            "distance travelled": float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1))),
            "displacement": float(np.linalg.norm(c[-1] - c[0])),
            "max height": float(c[:, 2].max()), "lowest point": float(c[:, 2].min()),
            "max acceleration": float(acc.max()) if len(acc) else 0.0,
            "peak momentum": float(m * sp.max()), "peak kinetic energy": float(KE.max()),
            "energy change (KE+PE)": float((KE[-1] + PE[-1]) - (KE[0] + PE[0])),
            "peak contact force": float(Fc.max()), "peak force from other objects": float(Fo.max()),
            "time moving": float(moving.sum() / fps), "max spin": float(max((np.linalg.norm(x) for x in w), default=0.0) * 60 / (2 * math.pi)),
            "mass": m,
        }
        for q, unit in CATALOG:
            out.append({"label": f"{nm}: {q}", "value": vals[q], "unit": unit, "about": nm, "quantity": q, "text": _fmt(vals[q], unit)})
    return out
