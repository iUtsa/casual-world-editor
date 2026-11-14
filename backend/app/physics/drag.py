"""Fluid forces with textbook laws, shared by the simulator and the theory checks.

Quadratic (pressure) drag   F = ½ ρ C_d A v²   with standard C_d values and the exact projected area A
                                               of the body's shape facing the flow;
linear (Stokes) drag        F = 3 π μ d v      (matters only for tiny or very slow things);
buoyancy                    F = ρ V g          (applied in the simulator's World._buoyancy);
added mass                  m_a = C_a ρ V      (water a body must push along when it speeds up; only in
                                               water, where it matters: half the displaced water for a ball).

The same functions drive the simulation and the "theory" rows, so measured-vs-theory differences come
only from the simulation itself (integration, contacts), never from two different drag models.
"""
from __future__ import annotations

import math

import numpy as np

# Standard drag coefficients (high-Reynolds, subcritical): sphere 0.47, cube 1.05, cylinder 0.82, streamlined ~0.3-0.5
CD = {"sphere": 0.47, "ellipsoid": 0.47, "capsule": 0.5, "cylinder": 0.82, "box": 1.05}
# Added-mass coefficients (potential flow): sphere 0.5; cube ~0.6; cylinder ~0.6
CA = {"sphere": 0.5, "ellipsoid": 0.5, "capsule": 0.5, "cylinder": 0.6, "box": 0.6}


def added_mass(sp: dict, rho: float, V: float) -> float:
    """Mass of water dragged along when the body accelerates (negligible in air, so only for liquids)."""
    return CA.get(sp["shape"], 0.6) * rho * V if rho > 100 else 0.0


def spec(shape: str | None, dims, cd: float | None = None) -> dict:
    """What the drag law needs about a body: its shape, size along its own axes (L, W, H) and C_d."""
    shp = shape if shape in CD else "box"
    return {"shape": shp, "dims": [float(x) for x in dims], "cd": float(cd if cd else CD[shp])}


def area(sp: dict, u) -> float:
    """Exact projected area (m²) of the body seen along the unit direction u (given in the body's own axes)."""
    L, W, H = sp["dims"]
    ux, uy, uz = (abs(float(x)) for x in u)
    s = sp["shape"]
    if s == "sphere":
        return math.pi * (L / 2) ** 2
    if s in ("ellipsoid", "capsule"):
        a, b, c = L / 2, W / 2, H / 2
        return math.pi * math.sqrt((b * c * ux) ** 2 + (a * c * uy) ** 2 + (a * b * uz) ** 2)
    if s == "cylinder":                     # axis along z, diameter L
        return uz * math.pi * L * L / 4 + math.sqrt(max(0.0, 1 - uz * uz)) * L * H
    return ux * W * H + uy * L * H + uz * L * W


def d_eq(sp: dict) -> float:
    L, W, H = sp["dims"]
    return float((L * W * H) ** (1 / 3))


def coef(sp: dict, v_local, rho: float, mu: float) -> float:
    """c in F_drag = −c·v (N per m/s) at this velocity: ½ρC_dA|v| + 3πμd."""
    v = np.asarray(v_local, float)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return 3 * math.pi * mu * d_eq(sp)
    return 0.5 * rho * sp["cd"] * area(sp, v / s) * s + 3 * math.pi * mu * d_eq(sp)


def force(sp: dict, v_local, rho: float, mu: float) -> np.ndarray:
    """Drag force (in the body's axes) for velocity v_local relative to still fluid."""
    return -coef(sp, v_local, rho, mu) * np.asarray(v_local, float)


def step_force(sp: dict, v_local, rho: float, mu: float, m: float, dt: float) -> np.ndarray:
    """Drag force to apply over one time step of dt: the exact decay of dv/dt = −(c/m)v over the step,
    v → v·e^(−c·dt/m). Identical to −c·v when c·dt/m is small, and stable even for very light, broad
    things (a beach ball in water) where the explicit force would overshoot and blow up."""
    v = np.asarray(v_local, float)
    x = coef(sp, v, rho, mu) * dt / max(m, 1e-9)
    return m * (math.expm1(-x) / dt) * v


def torque(sp: dict, w_local, rho: float, I: float | None = None, dt: float | None = None) -> np.ndarray:
    """Rotational drag (inertia-box form): τ_i = −ρ/64 · d_i (d_j⁴ + d_k⁴) |ω_i| ω_i
    (with I and dt: the same exact-decay form as step_force, so it can never reverse the spin)."""
    d = sp["dims"]
    w = np.asarray(w_local, float)
    c = np.array([rho / 64 * d[i] * (d[(i + 1) % 3] ** 4 + d[(i + 2) % 3] ** 4) * abs(w[i]) for i in range(3)])
    if I and dt:
        return I * np.expm1(-c * dt / I) / dt * w
    return -c * w


def fall(m: float, V: float, sp: dict, rho: float, mu: float, g: float, h: float | None = None, t_end: float | None = None,
         v0: float = 0.0, u=(0.0, 0.0, 1.0), ma: float = 0.0) -> dict:
    """Newton's 2nd law for a vertical fall through a fluid, solved to high precision (RK4, 0.1 ms):
    (m + m_a) dv/dt = m g − ρ V g − ½ ρ C_d A v|v| − 3πμ d v   (v positive downward).
    Stops after falling h metres or at t_end. Returns {t, v, terminal}."""
    A = area(sp, u)
    k2, k1 = 0.5 * rho * sp["cd"] * A, 3 * math.pi * mu * d_eq(sp)
    Fg = m * g - rho * V * g

    def acc(v):
        return (Fg - k2 * v * abs(v) - k1 * v) / (m + ma)
    t, x, v, dt = 0.0, 0.0, float(v0), 1e-4
    while True:
        if h is not None and x >= h:
            break
        if t_end is not None and t >= t_end:
            break
        if t > 600:
            break
        a1 = acc(v); a2 = acc(v + 0.5 * dt * a1); a3 = acc(v + 0.5 * dt * a2); a4 = acc(v + dt * a3)
        x1 = x + dt * (v + dt / 6 * (a1 + a2 + a3))
        v1 = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        if h is not None and x1 > h:                 # land exactly on h
            f = (h - x) / max(1e-12, x1 - x)
            t += f * dt; v = v + f * (v1 - v); x = h
            break
        t, x, v = t + dt, x1, v1
    disc = k1 * k1 + 4 * k2 * abs(Fg)
    vt = (-k1 + math.sqrt(disc)) / (2 * k2) if k2 > 0 else (abs(Fg) / k1 if k1 > 0 else math.inf)
    return {"t": t, "v": v, "x": x, "terminal": math.copysign(vt, Fg)}


def flight(m: float, V: float, sp: dict, rho: float, mu: float, g: float, p0, v0, z_land: float) -> dict:
    """Projectile with gravity, buoyancy and drag, solved precisely (RK4, 0.1 ms) until it comes down to z_land.
    Assumes the body keeps its orientation (spheres: exact). Returns {range, peak, t}."""
    A_of = lambda vv: area(sp, np.abs(vv) / max(1e-12, float(np.linalg.norm(vv))))
    k1 = 3 * math.pi * mu * d_eq(sp)
    gz = np.array([0.0, 0.0, -(m - rho * V) * g / m])

    def acc(v):
        s = float(np.linalg.norm(v))
        return gz - (0.5 * rho * sp["cd"] * A_of(v) * s + k1) * v / m if s > 1e-12 else gz
    p, v = np.array(p0, float), np.array(v0, float)
    t, dt, peak = 0.0, 1e-4, float(p[2])
    while t < 300:
        a1 = acc(v); a2 = acc(v + 0.5 * dt * a1); a3 = acc(v + 0.5 * dt * a2); a4 = acc(v + dt * a3)
        p1 = p + dt * (v + dt / 6 * (a1 + a2 + a3))
        v1 = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        peak = max(peak, float(p1[2]))
        if v1[2] < 0 and p1[2] <= z_land:
            f = (p[2] - z_land) / max(1e-12, p[2] - p1[2])
            p = p + f * (p1 - p); t += f * dt
            break
        p, v, t = p1, v1, t + dt
    return {"range": float(np.linalg.norm(p[:2] - np.asarray(p0, float)[:2])), "peak": peak, "t": t}
