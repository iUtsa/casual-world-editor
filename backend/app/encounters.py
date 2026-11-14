"""How two bodies really meet. orbits.run hands a two-body collision here when a rigid "touch and merge" would be wrong.

  * tidal disruption - the victim is torn apart before it touches: when its tidal radius r_t = R (M/m)^(1/3) lies well
    outside contact. A star near a black hole, a planet near a black hole, ... (a "TDE"). Approach on a parabola with
    pericentre r_t/beta; at r_t the victim becomes debris that keeps the velocity it had (the standard frozen-in model,
    Rees 1988; Stone et al. 2013) and each piece then follows its own orbit around the black hole (REBOUND, IAS15).
    Half the debris is bound and falls back, the most bound first after t_fb = 2 pi G M / (2 de)^(3/2), de = G M R / r_t^2.
  * swallowed whole - if r_t lies inside the event horizon (very massive black holes), nothing is torn: it just vanishes.
  * compact binary - two black holes / neutron stars spiral together by emitting gravitational waves (Peters 1964
    quasi-circular inspiral) until the innermost stable orbit, then merge; radiated energy and final spin from
    numerical-relativity fits (Barausse et al. 2012; Rezzolla et al. 2008).
Plain impacts (planets, moons, asteroids, stars hitting stars) stay in orbits.run, now on a realistic approach path.
"""
from __future__ import annotations

import base64
import math

import numpy as np

from .ephemeris import G, MSUN

C = 2.99792458e8
COMPACT = ("black hole", "neutron star")
BETA = 2.0                       # penetration factor r_t / pericentre
L_EDD_PER_MSUN = 1.26e31         # W
ETA_SCHW = 1 - math.sqrt(8 / 9)  # radiative efficiency down to the ISCO of a non-spinning black hole (5.7 %)


def compact(b: dict) -> bool:
    return b.get("kind") in COMPACT


def tidal_radius(victim: dict, other: dict) -> float:
    return victim["radius"] * (other["mass"] / victim["mass"]) ** (1 / 3)


def classify(a: dict, b: dict) -> dict:
    """{"type": "inspiral" | "disrupt" | "swallowed" | "impact", ...}"""
    if compact(a) and compact(b):
        return {"type": "inspiral"}
    best = None
    for v, o in ((a, b), (b, a)):
        if compact(v) or v["radius"] < 2e5:          # small bodies are held by material strength, not just gravity
            continue
        rt = tidal_radius(v, o)
        if o.get("kind") == "black hole" and rt <= o["radius"]:
            return {"type": "swallowed", "victim": v, "other": o, "rt": rt}
        ratio = rt / (v["radius"] + o["radius"])
        if ratio > 1.5 and (best is None or ratio > best[0]):
            best = (ratio, v, o, rt)
    if best:
        return {"type": "disrupt", "victim": best[1], "other": best[2], "rt": best[3]}
    return {"type": "impact"}


def parabolic(M: float, q: float, D: float, speed: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Relative state at distance D on an orbit with pericentre q: parabolic (falling in from far away), or hyperbolic at
    the given speed. Approaching, moving anticlockwise."""
    v = speed if speed else math.sqrt(2 * G * M / D)
    vq = math.sqrt(v * v + 2 * G * M * (1 / q - 1 / D))            # vis-viva at pericentre
    vt = min(v, q * vq / D)                                         # angular momentum h = q * vq = D * vt
    vr = math.sqrt(max(v * v - vt * vt, 0.0))
    return np.array([D, 0.0, 0.0]), np.array([-vr, vt, 0.0])


def _nm(b: dict) -> str:
    from .orbits import nm
    if b["name"] in ("black hole", "neutron star", "star", "white dwarf"):
        return f"the {b['name']}"                      # a generic one, not a named object
    return nm(b["name"])


def _fmt(v, unit):
    from .orbits import _fmt as f
    return f(v, unit)


def _dur(s):
    from .orbits import _dur as d
    return d(s)


def _body_out(b: dict, mass: float | None = None) -> dict:
    from .orbits import nm
    return {"name": b["name"], "label": b.get("label") or nm(b["name"], True), "kind": b["kind"], "mass": mass if mass is not None else b["mass"],
            "radius": b["radius"], "color": b["color"], "rings": b.get("rings"), "temp": b.get("temp"), "parent": None, "pole": None}


def _row(rows, label, v, unit, about="system", text=None, why=None):
    rows.append({"label": label, "value": float(v), "unit": unit, "about": about, "text": text or _fmt(v, unit), "why": why})


# ------------------------------------------------------------------------------------------------ tidal disruption
def tde(text: str, star: dict, bh: dict, rt: float, notes: list[str], n_debris: int = 1400) -> dict:
    import rebound
    M, m, R = bh["mass"], star["mass"], star["radius"]
    q = rt / BETA
    D = 4.0 * rt
    r0, v0 = parabolic(M + m, q, D)
    # 1) the approach: exact two-body motion until the star reaches r_t
    s = rebound.Simulation(); s.G = G; s.integrator = "ias15"
    s.add(m=M); s.add(m=m, x=r0[0], y=r0[1], z=r0[2], vx=v0[0], vy=v0[1], vz=v0[2])
    s.move_to_com()
    t_in = 2 * math.sqrt(D ** 3 / (2 * G * (M + m)))                    # comfortably longer than the fall to r_t
    approach_t, approach_pos = [], []
    n_pre = 120
    dt = t_in / n_pre
    t_d, rel_d, vrel_d = None, None, None
    for k in range(n_pre * 4):
        p = s.particles
        rel = np.array([p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z])
        approach_t.append(s.t); approach_pos.append([[0.0, 0.0, 0.0], rel.tolist()])
        if np.linalg.norm(rel) <= rt:
            t_d, rel_d = s.t, rel
            vrel_d = np.array([p[1].vx - p[0].vx, p[1].vy - p[0].vy, p[1].vz - p[0].vz])
            break
        s.integrate(s.t + dt)
    if t_d is None:
        raise ValueError("the approach never reached the tidal radius")
    # 2) torn apart: debris keeps the star's velocity; each piece now orbits the black hole alone (frozen-in model)
    rng = np.random.default_rng(7)
    half = n_debris // 2                                 # mirrored pairs through the centre: no sampling bias in "half bound"
    u = rng.normal(size=(half, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = np.concatenate([u, -u])
    rad = R * np.tile(rng.random(half) ** (1 / 3), 2)
    n_debris = 2 * half
    pos = rel_d + u * rad[:, None]
    eps = 0.5 * float(np.dot(vrel_d, vrel_d)) - G * M / np.linalg.norm(pos, axis=1)     # specific orbital energy of each piece
    de_th = G * M * R / rt ** 2
    bound = eps < 0
    t_fb_th = 2 * math.pi * G * M / (2 * de_th) ** 1.5
    eps_min = float(eps.min())
    t_fb_meas = 2 * math.pi * G * M / (2 * abs(eps_min)) ** 1.5 if eps_min < 0 else float("nan")   # Kepler period of the most bound piece
    T_post = 2.6 * t_fb_th
    d = rebound.Simulation(); d.G = G; d.integrator = "ias15"
    d.add(m=M)
    for x, vv in zip(pos, np.repeat(vrel_d[None, :], n_debris, axis=0)):
        d.add(m=0.0, x=x[0], y=x[1], z=x[2], vx=vv[0], vy=vv[1], vz=vv[2])
    d.N_active = 1
    n_post = 240
    post_t, frames = [], []
    # the most bound piece: it swings out to its farthest point in half an orbit and is back after a full one -> 2 x (time to apocentre)
    kmb = int(np.argmin(eps))
    r_mb = []
    for k in range(n_post + 1):
        d.integrate(k * T_post / n_post)
        P = np.array([[pp.x, pp.y, pp.z] for pp in d.particles[1:]])
        post_t.append(t_d + d.t); frames.append(P)
        r_mb.append(float(np.linalg.norm(P[kmb])))
    r_mb = np.array(r_mb)
    k_apo = 0
    if eps_min < 0:                                          # its FIRST farthest point (it goes round several times)
        for k in range(2, len(r_mb) - 1):
            if r_mb[k] > 2 * rt and r_mb[k] >= r_mb[k - 1] and r_mb[k] > r_mb[k + 1]:
                k_apo = k
                break
    k_peri = int(np.argmin(r_mb[:max(1, k_apo)])) if k_apo > 0 else 0
    returned_first = 2 * (post_t[k_apo] - post_t[k_peri]) if 0 < k_apo < len(r_mb) - 1 else None
    # frames for the whole run: before disruption the debris sits inside the star
    star_rel_pre = [np.array(p[1]) for p in approach_pos]
    pre_frames = [sp + u * rad[:, None] for sp in star_rel_pre]
    allP = np.stack(pre_frames + frames) / rt                                 # in units of r_t
    ts = [float(t) for t in approach_t] + [float(t) for t in post_t]
    ps = [[p[0], p[1]] for p in approach_pos] + [[[0.0, 0.0, 0.0], rel_d.tolist()] for _ in frames]     # the star is gone: debris shows it
    col = np.array([int(star["color"][i:i + 2], 16) for i in (1, 3, 5)]) / 255 if str(star.get("color", "")).startswith("#") else np.array([1, .85, .5])
    cols = np.where(bound[:, None], np.clip(col * 0.6 + np.array([0.5, 0.25, 0.05]), 0, 1), np.clip(col * 0.5 + 0.5, 0, 1))
    data = allP.astype(np.float16)
    # ---- results
    rows: list[dict] = []
    checks = []
    Rs = 2 * G * M / C ** 2
    _row(rows, "Tidal radius r_t = R★(M/m)^⅓", rt / 1000, "km", text=f"{rt / 1000:,.0f} km ({rt / R:.2f}× the star's radius)",
         why="where the black hole's pull on the star's near side exceeds its pull on the far side by more than the star's own gravity")
    if bh.get("kind") == "black hole":
        _row(rows, "Event horizon (Schwarzschild radius 2GM/c²)", Rs / 1000, "km", text=f"{Rs / 1000:,.1f} km")
    _row(rows, "Closest approach of the star's centre", q / 1000, "km", text=f"{q / 1000:,.0f} km (β = r_t/q = {BETA:g})")
    valid = M / m >= 100              # the textbook TDE estimates assume the black hole is far heavier than the star
    _row(rows, "Debris left bound (measured)", 100 * bound.mean(), "%", text=f"{100 * bound.mean():.1f} %",
         why="the part of the star nearer the black hole ends up on bound orbits, the far part is flung away")
    if valid:
        _row(rows, "Bound fraction (theory, frozen-in symmetric spread)", 50.0, "%", text="50 %")
        checks.append(100 * (1 - abs(bound.mean() - 0.5) / 0.5))
    de_meas = 0.5 * float(eps.max() - eps.min())
    _row(rows, "Spread in orbital energy of the debris (measured)", de_meas, "J/kg", text=f"{de_meas:.3e} J/kg")
    _row(rows, "Spread (theory Δε = GMR★/r_t²)", de_th, "J/kg", text=f"{de_th:.3e} J/kg")
    if valid:
        checks.append(100 * (1 - abs(de_meas - de_th) / de_th))
    _row(rows, "First debris falls back (theory t_fb = 2πGM/(2Δε)^{3/2})", t_fb_th / 3600, "h", text=_dur(t_fb_th))
    if returned_first is not None:
        _row(rows, "First debris falls back (measured: most bound piece, 2 × time to its farthest point)", returned_first / 3600, "h", text=_dur(returned_first))
        if valid:
            checks.append(100 * (1 - abs(returned_first - t_fb_th) / t_fb_th))
    mdot_peak = (m / 3) / t_fb_th
    E_acc = ETA_SCHW * 0.5 * m * C ** 2
    L_peak = ETA_SCHW * mdot_peak * C ** 2
    L_edd = L_EDD_PER_MSUN * M / MSUN
    _row(rows, "Peak fallback rate ≈ m/(3 t_fb)", mdot_peak / MSUN * 3.156e7, "Msun/yr", text=f"{mdot_peak / MSUN * 3.156e7:.3g} Sun-masses per year")
    _row(rows, "Light if the bound half is swallowed (5.7 % of mc²)", E_acc, "J", text=_fmt(E_acc, "J"))
    _row(rows, "Peak power vs the Eddington limit", L_peak / L_edd, "×", text=f"{L_peak / L_edd:.3g}× Eddington" if L_peak / L_edd < 1e4 else f"{L_peak / L_edd:.2e}× Eddington",
         why="above 1 the light pushes gas away faster than gravity pulls it in, so much of the debris is blown off in winds")
    _row(rows, "Black hole after feeding (at most)", (M + 0.5 * m) / MSUN, "Msun", text=f"≤ {(M + 0.5 * m) / MSUN:.3g} Sun-masses")
    if checks:
        ag = float(np.mean([max(0.0, min(100.0, c)) for c in checks]))
        rows.append({"label": "Agreement with theory", "value": ag, "unit": "%", "about": "system", "text": f"{ag:.1f} %", "why": None})
    if not valid:
        notes = notes + [f"{_nm(bh)[0].upper() + _nm(bh)[1:]} is only {M / m:.0f}× heavier than {_nm(star)}; the textbook tidal-disruption estimates "
                         f"(half bound, fall-back time) assume ≫100×, so here they are only rough and aren't scored — the simulated debris is shown as it is."]
    who = _nm(bh)
    story = [
        f"{_nm(star)[0].upper() + _nm(star)[1:]} never touches {who}. At {rt / 1000:,.0f} km — {rt / R:.1f}× its own radius — {who} pulls "
        f"its near side so much harder than its far side that the difference beats {_nm(star)}'s own gravity: it is torn apart. "
        f"Astronomers see this as a tidal disruption event (a sudden flare in a galaxy's centre).",
        f"Each piece keeps the speed it had at that moment, but the near half sits deeper in {who}'s gravity: it is bound, and loops back "
        f"after {_dur(t_fb_th)} (the most bound first); the far half has more than escape energy and is flung away for good. "
        f"Measured here: {100 * bound.mean():.0f} % bound" + (" — theory says half." if M / m >= 100 else "."),
        f"The returning gas collides with itself, heats up and spirals in through a disk. Swallowing the bound half could release "
        f"{_fmt(E_acc, 'J')} of light — {L_peak / L_edd:,.3g}× the Eddington limit at the peak, so much of it is blown away in winds. "
        f"{who[0].upper() + who[1:]} grows by at most {0.5 * m / MSUN:.3g} Sun-masses.",
    ]
    notes = notes + [f"Shown from {who}; its true size is only {2 * bh['radius'] / 1000:,.1f} km across, drawn larger so you can see it. Debris: "
                     f"{n_debris} pieces following their own orbits (frozen-in model)."]
    events = [{"t": float(t_d), "kind": "disrupt", "a": star["name"], "b": bh["name"], "speed": float(np.linalg.norm(vrel_d)), "energy": 0.0,
               "pos": rel_d.tolist(), "small_radius": float(R), "big_radius": float(bh["radius"])}]
    return {
        "domain": "gravity", "text": text, "mode": "collide", "source": "hypothetical set-up · REBOUND (IAS15) · frozen-in tidal disruption",
        "date": None, "frame": None, "focus": None,
        "bodies": [_body_out(bh), _body_out(star)], "t": ts, "pos": ps, "events": events, "alive": [True, False],
        "sim_seconds": ts[-1], "duration": 16.0, "results": rows, "story": story, "notes": notes, "belts": [], "sun_index": None,
        "encounter": "tidal disruption", "view_extent_m": float(5.0 * rt),
        "particles": {"n": int(n_debris), "frames": int(data.shape[0]), "unit_m": float(rt),
                      "data": base64.b64encode(data.tobytes()).decode(), "colors": base64.b64encode((cols * 255).astype(np.uint8).tobytes()).decode(),
                      "split": int(n_debris)},
    }


# ------------------------------------------------------------------------------------------------ compact binaries
def inspiral(text: str, a: dict, b: dict, notes: list[str]) -> dict:
    m1, m2 = a["mass"], b["mass"]
    M = m1 + m2
    mu = m1 * m2 / M
    nu = mu / M
    GM = G * M
    r_isco = 6 * GM / C ** 2

    def tau(a0):
        return (5 / 256) * C ** 5 * a0 ** 4 / (G ** 3 * m1 * m2 * M)

    def orbits_left(a0):          # orbits from a0 to merger, from the Peters chirp (numerically)
        aa = np.linspace(a0, r_isco, 4000)
        dadt = (64 / 5) * G ** 3 * m1 * m2 * M / (C ** 5 * aa ** 3)
        w = np.sqrt(GM / aa ** 3)
        return float(np.trapezoid(w / dadt, -aa) / (2 * math.pi)) if hasattr(np, "trapezoid") else float(np.trapz(w / dadt, -aa) / (2 * math.pi))
    a0 = r_isco * 1.5
    while orbits_left(a0) < 9:                        # start ~9 orbits before the merger: enough to see the chirp
        a0 *= 1.08
    T_in = tau(a0) - tau(r_isco)
    n = 900
    ts = np.linspace(0, T_in, n)
    aa = (a0 ** 4 - (256 / 5) * G ** 3 * m1 * m2 * M / C ** 5 * ts) ** 0.25
    w = np.sqrt(GM / aa ** 3)
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(ts))])
    r1 = (m2 / M) * aa; r2 = (m1 / M) * aa
    pos = [[[float(-r1[k] * math.cos(phi[k])), float(-r1[k] * math.sin(phi[k])), 0.0],
            [float(r2[k] * math.cos(phi[k])), float(r2[k] * math.sin(phi[k])), 0.0]] for k in range(n)]
    # merger + ringdown: remnant at the centre of mass
    E_rad = (ETA_SCHW * nu + 0.498 * nu ** 2) * M * C ** 2 if nu > 0 else 0.0           # Barausse et al. 2012 (non-spinning)
    Mf = M - E_rad / C ** 2
    af = 2 * math.sqrt(3) * nu - 3.871 * nu ** 2 + 4.028 * nu ** 3                        # Rezzolla et al. 2008
    f_isco_gw = C ** 3 / (6 ** 1.5 * math.pi * GM)
    f_ring = C ** 3 / (2 * math.pi * G * Mf) * (1 - 0.63 * (1 - af) ** 0.3)               # Echeverria 1989, fundamental l=m=2 mode
    t_post = np.linspace(T_in, T_in * 1.18, 160)[1:]
    ts_all = list(ts) + list(t_post)
    pos += [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] for _ in t_post]
    both_bh = a["kind"] == b["kind"] == "black hole"
    rows: list[dict] = []
    _row(rows, "Total mass", M / MSUN, "Msun", text=f"{M / MSUN:.3g} Sun-masses")
    _row(rows, "Innermost stable orbit (6GM/c²)", r_isco / 1000, "km", text=f"{r_isco / 1000:,.0f} km")
    _row(rows, "Separation at the start", a0 / 1000, "km", text=f"{a0 / 1000:,.0f} km")
    _row(rows, "Time to merge from there (Peters: τ = 5c⁵a⁴/256G³m₁m₂M)", T_in, "s", text=f"{T_in:.3g} s")
    _row(rows, "Gravitational-wave frequency at the last stable orbit", f_isco_gw, "Hz", text=f"{f_isco_gw:,.0f} Hz",
         why="twice the orbital frequency; LIGO hears 10 Hz – a few kHz")
    _row(rows, "Energy radiated as gravitational waves", E_rad / (MSUN * C ** 2), "Msun c²", text=f"{E_rad / (MSUN * C ** 2):.3g} Sun-masses × c² ({100 * E_rad / (M * C ** 2):.1f} % of the mass)")
    _row(rows, "Remnant mass", Mf / MSUN, "Msun", text=f"{Mf / MSUN:.3g} Sun-masses")
    if both_bh:
        _row(rows, "Remnant spin a/M", af, "", text=f"{af:.2f}")
        _row(rows, "Ringdown frequency of the new black hole", f_ring, "Hz", text=f"{f_ring:,.0f} Hz")
    t_low = text.lower()
    if "gw150914" in t_low or ("36" in t_low and "29" in t_low):
        _row(rows, "LIGO GW150914 measured energy radiated", 3.0, "Msun c²", text="3.0 ± 0.5 Sun-masses × c²")
    story = [
        f"Two {'black holes' if both_bh else 'compact stars'} can't just fall together and “hit”: long before they touch they circle each other, and every orbit "
        f"shakes space-time — gravitational waves — carrying energy away. So the orbit shrinks and speeds up (the “chirp”): from {a0 / 1000:,.0f} km apart "
        f"they merge in {T_in:.3g} s, the waves rising to {f_isco_gw:,.0f} Hz.",
        f"At the merger {100 * E_rad / (M * C ** 2):.1f} % of their mass — {E_rad / (MSUN * C ** 2):.2g} Suns' worth — leaves as gravitational waves, not heat or "
        f"light: the remnant weighs {Mf / MSUN:.3g} Sun-masses, less than the {M / MSUN:.3g} that went in."
        + (f" It is born spinning (a/M ≈ {af:.2f}) and rings like a bell at about {f_ring:,.0f} Hz before settling." if both_bh else
           " Neutron stars colliding throw out hot neutron-rich debris that forges gold and platinum (a kilonova, seen in 2017 with GW170817)."),
    ]
    notes = notes + ["Newton's gravity fails this close (speeds approach light): the orbit's shrinking follows Peters' general-relativity formula, "
                     "and the merger's radiated energy and spin come from numerical-relativity fits. Compact objects drawn larger than their true size."]
    ev = [{"t": float(T_in), "kind": "gw", "a": b["name"], "b": a["name"], "speed": float(math.sqrt(GM / r_isco)), "energy": float(E_rad),
           "pos": [0.0, 0.0, 0.0], "small_radius": float(r_isco), "big_radius": float(r_isco)}]
    out_a = _body_out(a, Mf); out_a["label"] = f"{out_a['label']} → remnant"
    return {
        "domain": "gravity", "text": text, "mode": "collide", "source": "hypothetical set-up · Peters inspiral + numerical-relativity merger fits",
        "date": None, "frame": None, "focus": None, "bodies": [out_a, _body_out(b)], "t": [float(t) for t in ts_all], "pos": pos,
        "events": ev, "alive": [True, False], "sim_seconds": float(ts_all[-1]), "duration": 14.0, "results": rows, "story": story, "notes": notes,
        "belts": [], "sun_index": None, "encounter": "gravitational-wave inspiral",
    }
