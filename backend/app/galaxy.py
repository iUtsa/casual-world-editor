"""Colliding galaxies: the Milky Way and Andromeda (and other pairs).

Method (Toomre & Toomre 1972, the classic way tidal tails were first understood): each galaxy is a dark-matter
halo with a Hernquist profile carrying almost all the mass, plus thousands of stars set on circular orbits in a
spiral disk and a central bulge. The two halos pull each other (softened Newtonian gravity between extended
bodies); every star feels both halos. When the halos overlap, dynamical friction (Chandrasekhar 1943) drains
their orbital energy so they merge. Momentum is exactly conserved (the friction acts equally and oppositely).

Real numbers: Andromeda is 770 kpc away, approaching at 110 km/s with ~57 km/s sideways motion (HST + Gaia:
van der Marel et al. 2012, 2019). Published simulations put the first close pass ~4 billion years from now and
the merger into one elliptical galaxy ("Milkomeda") at ~6 billion years.

Units inside: kpc, Myr, solar masses.
"""
from __future__ import annotations

import base64
import math
import re

import numpy as np

from .ephemeris import KPC, MSUN

G = 4.49850215e-12                 # kpc^3 / (Msun Myr^2)
KMS = 1.0227121650537077e-3        # 1 km/s in kpc/Myr
R_VIR = 300.0                      # kpc: where a Milky-Way-size dark halo ends
MYR = 3.15576e13                   # s

# name: dark halo mass (Msun) and Hernquist scale (kpc), stellar core (disk+bulge) mass and scale, disk scale length,
# disk radius (kpc), stars to draw, tint. Core masses/scales give the observed rotation speeds (MW ~220 km/s at 8 kpc).
GALAXIES = {
    "milky way": dict(M=1.0e12, a=60.0, Mc=1.3e11, ac=3.0, Rd=3.0, Rmax=16.0, n=2600, tint=(0.75, 0.85, 1.0), label="the Milky Way"),
    "andromeda": dict(M=1.5e12, a=70.0, Mc=1.8e11, ac=4.0, Rd=5.3, Rmax=24.0, n=3000, tint=(1.0, 0.88, 0.75), label="Andromeda (M31)"),
    "triangulum": dict(M=5.0e10, a=20.0, Mc=6.0e9, ac=1.5, Rd=1.8, Rmax=8.0, n=900, tint=(0.8, 0.9, 1.0), label="the Triangulum Galaxy (M33)"),
    "large magellanic cloud": dict(M=1.4e11, a=25.0, Mc=1.0e10, ac=1.5, Rd=1.5, Rmax=6.0, n=900, tint=(0.9, 0.9, 1.0), label="the Large Magellanic Cloud"),
}
ALIASES = {"m31": "andromeda", "andromeda galaxy": "andromeda", "our galaxy": "milky way", "m33": "triangulum", "lmc": "large magellanic cloud",
           "magellanic cloud": "large magellanic cloud"}


def _names(text: str) -> list[str]:
    t = text.lower()
    out = []
    for k in sorted(list(GALAXIES) + list(ALIASES), key=len, reverse=True):
        if re.search(rf"\b{re.escape(k)}\b", t):
            n = ALIASES.get(k, k)
            if n not in out:
                out.append(n)
    if not out:
        out = ["milky way", "andromeda"]
    if len(out) == 1:
        out = ["milky way", out[0]] if out[0] != "milky way" else ["milky way", "andromeda"]
    return out[:2]


def _acc_hern(r: np.ndarray, M: float, a: float) -> np.ndarray:
    d = np.linalg.norm(r, axis=-1, keepdims=True) + 1e-9
    return -G * M * r / (d * (d + a) ** 2)


def _disk(g: dict, rng, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Star positions/velocities (galaxy frame) and colours: two-armed spiral disk + bulge, on circular orbits."""
    n = g["n"]
    nb = n // 7                                                   # bulge
    nd = n - nb
    R = np.clip(rng.gamma(2.0, g["Rd"], nd), 0.4, g["Rmax"])     # exponential disk surface density
    arm = rng.random(nd) < 0.65
    pitch = math.radians(14.0)
    th = np.where(arm, np.log(R / 1.0) / math.tan(pitch) + np.pi * rng.integers(0, 2, nd) + rng.normal(0, 0.28, nd), rng.uniform(0, 2 * np.pi, nd))
    z = rng.normal(0, 0.25, nd)
    pos_d = np.stack([R * np.cos(th), R * np.sin(th), z], axis=1)
    vc = np.sqrt(G * g["M"] * R / (R + g["a"]) ** 2 + G * g["Mc"] * R / (R + g["ac"]) ** 2)    # circular speed: halo + stars
    vel_d = np.stack([-vc * np.sin(th), vc * np.cos(th), np.zeros(nd)], axis=1) + rng.normal(0, 8 * KMS, (nd, 3))
    rb = 0.7 * rng.random(nb) ** 1.5 / (1 - 0.9 * rng.random(nb))
    rb = np.clip(rb, 0.05, 3.0)
    u = rng.normal(size=(nb, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    pos_b = u * rb[:, None] * np.array([1, 1, 0.6])
    sig = np.sqrt(G * g["Mc"] * rb / (rb + g["ac"]) ** 2 * rb + G * g["M"] * rb / (rb + g["a"]) ** 2 * rb) * 0.6
    vel_b = rng.normal(size=(nb, 3)) * sig[:, None] / math.sqrt(3)
    pos, vel = np.vstack([pos_d, pos_b]), np.vstack([vel_d, vel_b])
    # disk colours: young blue stars in the arms, older yellow/white elsewhere, warm bulge
    col = np.zeros((n, 3))
    tint = np.array(g["tint"])
    col[:nd] = np.where(arm[:, None], np.array([0.62, 0.78, 1.0]), np.array([1.0, 0.93, 0.82])) * tint
    col[nd:] = np.array([1.0, 0.82, 0.58])
    # rotate the disk so its spin axis is `normal`
    z0 = np.array([0.0, 0.0, 1.0]); nn = normal / np.linalg.norm(normal)
    v = np.cross(z0, nn); s = np.linalg.norm(v); c = float(z0 @ nn)
    if s > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rm = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
        pos, vel = pos @ Rm.T, vel @ Rm.T
    return pos, vel, col


def run(text: str) -> dict:
    t = text.lower()
    a_name, b_name = _names(t)
    A, B = GALAXIES[a_name], GALAXIES[b_name]
    rng = np.random.default_rng(11)
    # relative orbit: real Andromeda numbers; other pairs a generic bound encounter
    if {a_name, b_name} == {"milky way", "andromeda"}:
        d0, v_rad, v_tan = 770.0, -110.0 * KMS, 57.0 * KMS
        src = "measured distance and motion of Andromeda (HST + Gaia, van der Marel et al. 2012/2019)"
    else:
        d0, v_rad, v_tan = 300.0, -80.0 * KMS, 40.0 * KMS
        src = "a typical bound encounter (no measured orbit for this pair)"
    MA, MB = A["M"] + A["Mc"], B["M"] + B["Mc"]
    Mt = MA + MB
    rel_r = np.array([d0, 0.0, 0.0]); rel_v = np.array([v_rad, v_tan, 0.0])
    cA = -rel_r * MB / Mt; cB = rel_r * MA / Mt
    vA = -rel_v * MB / Mt; vB = rel_v * MA / Mt
    pA, uA, colA = _disk(A, rng, np.array([0.0, 0.0, 1.0]))
    pB, uB, colB = _disk(B, rng, np.array([0.35, -0.55, 0.76]))
    P = np.vstack([pA + cA, pB + cB]); V = np.vstack([uA + vA, uB + vB])
    col = np.vstack([colA, colB])
    nA = len(pA)
    T_end = float(re.search(r"(\d+(?:\.\d+)?)\s*(billion|bn|gyr)", t).group(1)) * 1000 if re.search(r"(\d+(?:\.\d+)?)\s*(billion|bn|gyr)", t) else 9000.0
    dt = 1.0
    frames = 200
    every = max(1, int(round(T_end / dt / frames)))
    lnL = float(re.search(r"lnl=(\d+(?:\.\d+)?)", t).group(1)) if re.search(r"lnl=", t) else 2.5
    a_eff = 0.5 * (A["a"] + B["a"])

    def halo_acc(cA, cB, vA, vB):
        r = cB - cA
        d = float(np.linalg.norm(r)) + 1e-9
        F = G * MA * MB / (d + a_eff) ** 2 * r / d                  # softened pull between the extended halos (on A, toward B)
        # dynamical friction on the relative motion, only where the halos overlap (Chandrasekhar), equal and opposite
        vrel = vB - vA
        vr = float(np.linalg.norm(vrel)) + 1e-12
        # combined halo density at the separation (Hernquist), zero beyond the halos' virial radius (~300 kpc): no friction outside
        rho = Mt * a_eff / (2 * math.pi * d * (d + a_eff) ** 3) if d < R_VIR else 0.0
        sigma = math.sqrt(G * Mt * d) / (d + a_eff) / math.sqrt(2)
        X = vr / (math.sqrt(2) * sigma)
        fX = math.erf(X) - 2 * X / math.sqrt(math.pi) * math.exp(-X * X)
        mu = MA * MB / Mt
        # Chandrasekhar: the relative motion is braked at 4πG²·μ·ρ·lnΛ·f(X)/v²; as a force pair (+F on A along v_rel, −F on B)
        Fdf = 4 * math.pi * G ** 2 * mu * mu * rho * lnL * fX / vr ** 2 * (vrel / vr)
        return (F + Fdf) / MA, (-F - Fdf) / MB

    def star_acc(P, cA, cB):
        return (_acc_hern(P - cA, A["M"], A["a"]) + _acc_hern(P - cA, A["Mc"], A["ac"])
                + _acc_hern(P - cB, B["M"], B["a"]) + _acc_hern(P - cB, B["Mc"], B["ac"]))

    aA, aB = halo_acc(cA, cB, vA, vB)
    aP = star_acc(P, cA, cB)
    out_frames, cores, times, seps = [], [], [], []
    p_tot0 = MA * vA + MB * vB
    t_now = 0.0
    step = 0
    d_hist: list[tuple[float, float]] = []
    peri_t, prev_d, merged_t = None, d0, None
    E0 = 0.5 * MA * (vA @ vA) + 0.5 * MB * (vB @ vB) - G * MA * MB / (d0 + a_eff)
    E_before = None
    while t_now < T_end - 1e-9:
        vA += 0.5 * dt * aA; vB += 0.5 * dt * aB; V += 0.5 * dt * aP
        cA += dt * vA; cB += dt * vB; P += dt * V
        aA, aB = halo_acc(cA, cB, vA, vB)
        aP = star_acc(P, cA, cB)
        vA += 0.5 * dt * aA; vB += 0.5 * dt * aB; V += 0.5 * dt * aP
        t_now += dt; step += 1
        d = float(np.linalg.norm(cB - cA))
        if peri_t is None and d > prev_d and prev_d < 0.5 * d0:
            peri_t = t_now - dt
        if E_before is None and d < 300:
            E_before = (t_now, 0.0)
        d_hist.append((t_now, d))
        prev_d = d
        if step % every == 0:
            out_frames.append(P.astype(np.float16))
            cores.append([cA.copy(), cB.copy()])
            times.append(t_now)
            seps.append(d)
    # merged: from then on the two centres stay within 30 kpc of each other
    merged_t = None
    for k in range(len(d_hist)):
        if all(dd < 30.0 for _, dd in d_hist[k:]):
            merged_t = d_hist[k][0] if k < len(d_hist) - 50 else None
            break
    p_tot1 = MA * vA + MB * vB
    dp = float(np.linalg.norm(p_tot1 - p_tot0) / max(np.linalg.norm(MA * vA) + 1e-30, 1e-30))
    # theory for the approach: conservation of energy and angular momentum for the relative motion in the halos'
    # potential Φ(r) = −G M/(r + a) gives the inbound speed at every r, so t = ∫ dr / v_r (friction is negligible that far out)
    r0n = float(np.linalg.norm(rel_r))
    Eps = float(rel_v @ rel_v) / 2 - G * Mt / (r0n + a_eff)
    h = float(np.linalg.norm(np.cross(rel_r, rel_v)))
    rr = np.linspace(300.0, r0n, 200001)
    vr = np.sqrt(np.maximum(2 * (Eps + G * Mt / (rr + a_eff)) - h * h / rr ** 2, 1e-30))
    t_kep = float(np.trapezoid(1.0 / vr, rr))
    t_sim = E_before[0] if E_before else None
    rows, story = [], []

    def add(label, v, unit, text_=None, why=None):
        rows.append({"label": label, "value": float(v), "unit": unit, "about": "system", "text": text_ or f"{v:,.3g} {unit}", "why": why})
    add("Starting distance", d0, "kpc", f"{d0:,.0f} kpc ({d0 * 3261.56 / 1e6:.2f} million light years)")
    add("Approach speed", -v_rad / KMS, "km/s", f"{-v_rad / KMS:.0f} km/s (sideways {v_tan / KMS:.0f} km/s)")
    if t_sim:
        add("Time to close within 300 kpc (measured)", t_sim / 1000, "Gyr", f"{t_sim / 1000:.3f} billion years")
        add("Time to close within 300 kpc (theory: energy + angular-momentum conservation, t = ∫dr/v_r)", t_kep / 1000, "Gyr", f"{t_kep / 1000:.3f} billion years")
    if peri_t:
        add("First close pass (measured)", peri_t / 1000, "Gyr", f"{peri_t / 1000:.2f} billion years from now")
        if {a_name, b_name} == {"milky way", "andromeda"}:
            add("First close pass in published simulations (van der Marel et al. 2012: 3.87 Gyr; with Gaia's sideways speed ~4.5 Gyr)", 3.87, "Gyr",
                "3.9–4.5 billion years")
    if merged_t:
        add("Merged into one galaxy (measured)", merged_t / 1000, "Gyr", f"{merged_t / 1000:.2f} billion years from now")
        if {a_name, b_name} == {"milky way", "andromeda"}:
            add("Merger in published simulations (van der Marel et al. 2012: 5.86 Gyr; later with Gaia's wider orbit)", 5.86, "Gyr", "≈6–8 billion years")
    add("Total momentum conserved (change)", dp, "", f"{dp:.1e} (relative)", "the pull and the friction between the halos act equally and oppositely")
    add("Stars drawn", len(P), "", f"{len(P):,} ({nA:,} + {len(P) - nA:,})")
    checks = [100 * (1 - dp)]
    if t_sim:
        checks.append(100 * (1 - abs(t_sim - t_kep) / t_kep))
    add("Agreement with theory", max(0.0, min(checks)), "%")
    LA, LB = A["label"], B["label"]
    story.append(f"{LA[0].upper() + LA[1:]} and {LB} are {d0:,.0f} kpc ({d0 * 3261.56 / 1e6:.1f} million light years) apart and falling toward each other at "
                 f"{-v_rad / KMS:.0f} km/s — gravity from their {MA:.1e} and {MB:.1e} solar masses (mostly invisible dark matter) wins over the expansion of the universe at this distance.")
    if peri_t:
        story.append(f"They first sweep through each other {peri_t / 1000:.1f} billion years from now. Stars almost never hit each other — the gaps between stars are "
                     f"millions of times their size — but each galaxy's gravity flings the other's outer stars into long tidal tails, the shapes Toomre & Toomre "
                     f"first explained in 1972 and that we see in the Antennae and Mice galaxies.")
    if merged_t:
        story.append(f"Dynamical friction — the halos dragging on each other's dark matter and stars — steals their orbital energy, so instead of flying apart they "
                     f"fall back and merge after {merged_t / 1000:.1f} billion years into one big elliptical galaxy, nicknamed Milkomeda.")
    story.append("The Sun will survive: it will most likely be thrown into a wider orbit in the merged galaxy. By then the Sun itself will be nearing the end "
                 "of its life (in ~5 billion years it becomes a red giant).")
    data = np.stack(out_frames)                            # frames x stars x 3 (float16, kpc)
    colors = (np.clip(col, 0, 1) * 255).astype(np.uint8)
    cores_m = [[(c * KPC).tolist() for c in fr] for fr in cores]
    return {
        "domain": "gravity", "text": text, "mode": "galaxies", "source": src, "date": None, "frame": None,
        "bodies": [{"name": a_name, "label": LA, "kind": "galaxy", "mass": MA * MSUN, "radius": A["Rmax"] * KPC, "color": "#c7d2fe"},
                   {"name": b_name, "label": LB, "kind": "galaxy", "mass": MB * MSUN, "radius": B["Rmax"] * KPC, "color": "#fed7aa"}],
        "t": [x * MYR for x in times], "pos": cores_m, "events": [], "alive": [True, True], "sim_seconds": times[-1] * MYR if times else 0.0,
        "duration": 24.0, "results": rows, "story": story, "notes": ["Each galaxy's dark-matter halo is a smooth Hernquist sphere; its stars are tracers "
                                                                    "feeling both halos (the Toomre & Toomre method), so the halos' own shapes don't change."],
        "belts": [], "sun_index": None,
        "particles": {"n": int(len(P)), "frames": int(data.shape[0]), "unit_m": KPC,
                      "data": base64.b64encode(data.tobytes()).decode(), "colors": base64.b64encode(colors.tobytes()).decode(), "split": int(nA)},
    }
