"""MuJoCo world built from the explicit scene graph, plus the causal simulator.

Design:
  * One MJCF model per request, compiled from the Scene (objects = free bodies
    with box/cylinder collision primitives, table = static box, floor plane).
  * Interventions are applied as *segments*: each starts from the settled state
    left by the previous one, runs until everything is at rest, and records
    60 fps poses plus a 125 Hz event stream.
  * Static objects are kinematically pinned (pose reset every step), which
    behaves like infinite mass for anything touching them.
  * Support is measured, not guessed: at rest, each near-vertical contact's
    normal force tells how much of an object's weight each supporter carries.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .. import config
from . import drag as _drag
from ..schemas import Intervention, Scene, SceneObject

TABLE = "table"
FLOOR = "floor"
MOTOR_STALL_THRUST = 0.34     # N per wheel at stall (toy gear motor)
MOTOR_NOLOAD_SPEED = 3.2      # m/s ground speed at zero load, full throttle


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


# soft bodies (flesh, food, fabric) deform: longer contact time constant -> realistic, lower peak forces
SOFTNESS = {"flesh": 4.0, "food": 3.0, "fabric": 5.0, "rubber": 1.5, "shell": 1.5, "foam": 6.0}

_E_CACHE: dict[tuple, float] = {}


def _bounce_e(zeta: float, dt: float, tc: float) -> float:
    """Measured coefficient of restitution (separation speed / impact speed) of one contact with these
    parameters: a ball hits a plane at 3 m/s in a tiny model with the same solver settings as the worlds."""
    xml = f"""<mujoco><option timestep="{dt}" gravity="0 0 0" integrator="implicitfast" cone="elliptic" impratio="5"/>
      <default><geom solimp="0.95 0.99 0.001" margin="0.0005"/></default>
      <worldbody><geom type="plane" size="5 5 0.1"/>
      <body pos="0 0 0.0515"><freejoint/><geom type="sphere" size="0.05" mass="1" friction="0.5 0.005 0.0001" condim="6"
        solref="{tc:.6f} {zeta:.6f}" solmix="1000" priority="1"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qvel[2] = -3.0
    touched = False
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(m, d)
        touched = touched or d.ncon > 0
        if touched and d.ncon == 0 and d.qvel[2] > 0:
            break
    return float(max(0.0, d.qvel[2]) / 3.0)


def contact_zeta(e: float, dt: float, soft: float = 1.0) -> float:
    """Damping ratio that makes a contact bounce back with exactly restitution e at this time step and softness
    (found by bisection on the measured restitution, cached)."""
    e = float(np.clip(e, 0.02, 0.95))
    key = (round(e, 3), round(dt, 7), round(soft, 3))
    if key in _E_CACHE:
        return _E_CACHE[key]
    tc = 10 * dt * soft
    lo, hi = 0.005, 6.0                    # restitution falls as damping rises
    for _ in range(26):
        mid = math.sqrt(lo * hi)
        if _bounce_e(mid, dt, tc) > e:
            lo = mid
        else:
            hi = mid
    _E_CACHE[key] = math.sqrt(lo * hi)
    return _E_CACHE[key]


def restitution_solref(e: float, dt: float, soft: float = 1.0) -> str:
    return f"{10 * dt * soft:.6f} {contact_zeta(e, dt, soft):.6f}"


def pair_restitution(e1: float, e2: float) -> float:
    """Restitution of a contact between two materials: the geometric mean √(e₁e₂) (each one's value against a rigid
    surface; equal materials keep their own value, a dead material kills the bounce)."""
    return math.sqrt(max(0.0, e1) * max(0.0, e2))


def _proxy_xml(sp: dict) -> str:
    """Stand-in body for anything without a library model: real size, mass and bounciness."""
    a, sid = sp["asset"], sp["sid"]
    L, W, H = a["dims"]
    mass = float(sp.get("mass", a["mass"]))
    b = float(a.get("bounciness", 0.3))
    solref = restitution_solref(b, sp.get("dt", 0.002), SOFTNESS.get(a.get("material") or "", 1.0))
    fl = ' fluidshape="ellipsoid"' if sp.get("fluid") else ""
    # priority: against the ground/table this object's own friction and bounciness apply
    # (MuJoCo would otherwise take the larger friction of the two surfaces)
    common = f'name="g_{sid}" mass="{mass:.6g}" friction="{friction_str(a["friction"], a["dims"], a.get("material"), a.get("shape") in ("sphere", "ellipsoid", "capsule"))}" solref="{solref}" solmix="1000" priority="{0 if a.get('surface') else 1}"{fl}'
    shape = a["shape"]
    # The body's origin is its centre (where its mass is): a free body whose origin sits off the centre of
    # mass integrates fast spins badly (a launched ball hops and rolls at 1.9 instead of 5/7 x 4 = 2.86 m/s).
    # Positions given to / reported by the World stay bottom-referenced via z_off.
    if shape == "sphere":
        zo = L / 2
        g = f'<geom {common} type="sphere" size="{L / 2:.6g}" condim="6"/>'
    elif shape == "cylinder":
        zo = H / 2
        g = f'<geom {common} type="cylinder" size="{L / 2:.6g} {H / 2:.6g}" condim="6"/>'
    elif shape == "capsule":
        if H >= L:   # upright (people)
            r = min(L, W) / 2
            top = max(H - r, r * 1.01)
            zo = (r + top) / 2
            g = f'<geom {common} type="capsule" size="{r:.6g}" fromto="0 0 {r - zo:.6g} 0 0 {top - zo:.6g}" condim="4"/>'
        else:        # lying along x (rockets, submarines, bananas)
            r = min(W, H) / 2
            half = max(L / 2 - r, r * 0.01)
            zo = r
            g = f'<geom {common} type="capsule" size="{r:.6g}" fromto="{-half:.6g} 0 0 {half:.6g} 0 0" condim="4"/>'
    elif shape == "ellipsoid":
        zo = H / 2
        g = f'<geom {common} type="ellipsoid" size="{L / 2:.6g} {W / 2:.6g} {H / 2:.6g}" condim="4"/>'
    else:
        zo = H / 2
        g = f'<geom {common} type="box" size="{L / 2:.6g} {W / 2:.6g} {H / 2:.6g}" condim="4"/>'
    sp["z_off"] = zo
    park = f'{sp.get("park_x", 60 + 3 * sp["k"]):.3f} 60 -30'
    sp["_contact"] = {"geom": f"g_{sid}", "e": b, "soft": SOFTNESS.get(a.get("material") or "", 1.0),
                      "fr": [float(x) for x in friction_str(a["friction"], a["dims"], a.get("material"), a.get("shape") in ("sphere", "ellipsoid", "capsule")).split()],
                      "condim": 6 if shape in ("sphere", "cylinder") else 4}
    return f'<body name="{sid}" pos="{park}"><freejoint name="j_{sid}"/>{g}<site name="c_{sid}" pos="0 0 {H / 2 - zo:.6g}"/></body>'


def _spawn_xml(sp: dict) -> tuple[str, str, str]:
    """(asset xml, body xml, actuator xml) for one spawned library object."""
    a = sp["asset"]
    sid = sp["sid"]
    if a["kind"] == "proxy":
        return "", _proxy_xml(sp), ""
    mass = float(sp.get("mass", a["mass"]))
    fr = friction_str(a["friction"], a["dims"], a.get("material"),
                      a.get("collision") == "sphere" or (a["kind"] == "procedural" and a.get("shape") == "sphere"))
    if sp.get("bounciness") is not None:
        solref = f' solref="{restitution_solref(float(sp["bounciness"]), sp.get("dt", 0.002))}" solmix="1000" priority="1"'
    else:
        solref = ' solref="0.006 0.12"' if a.get("bouncy") else ""
    park = f'{sp.get("park_x", 60 + 3 * sp["k"]):.3f} 60 -30'
    assets, acts = "", ""
    if sp.get("bounciness") is not None and a["kind"] != "vehicle":
        round_ = a.get("collision") == "sphere" or (a["kind"] == "procedural" and a.get("shape") == "sphere")
        sp["_contact"] = {"geom": f"g_{sid}", "e": float(sp["bounciness"]), "soft": 1.0, "fr": [float(x) for x in fr.split()],
                          "condim": 6 if round_ else 4}
    if a["kind"] == "vehicle":
        w = a["wheels"]
        r, hw = w["radius"], w["width"] / 2
        L, W, H = a["dims"]
        cz = (r + H) / 2 + 0.002
        chassis = (f'<geom name="g_{sid}" type="box" pos="0 0 {cz:.4f}" size="{L / 2 - 0.004:.4f} {abs(w["centres"][0][1]) - hw - 0.002:.4f} '
                   f'{(H - r) / 2:.4f}" mass="{mass * 0.8:.4f}" friction="0.4 0.01 0.002"/>')
        wheels = []
        for name, c in zip(w["order"], w["centres"]):
            spin = (f'<body name="{sid}_{name}_w"><joint name="{sid}_spin_{name}" type="hinge" axis="0 1 0" damping="0.0000005" armature="0.000004" frictionloss="0.00002"/>'
                    f'<geom name="g_{sid}_{name}" type="cylinder" size="{r:.4f} {hw:.4f}" zaxis="0 1 0" '
                    f'mass="{mass * 0.05:.4f}" friction="{a["friction"]:.2f} 0.005 0.0001" condim="3" solref="0.008 1" solimp="0.9 0.95 0.001"/></body>')
            if name.startswith("f"):
                wheels.append(f'<body name="{sid}_{name}" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}">'
                              f'<joint name="{sid}_steer_{name}" type="hinge" axis="0 0 1" range="-0.5 0.5" limited="true" damping="0.0005"/>'
                              f'<inertial pos="0 0 0" mass="0.002" diaginertia="1e-7 1e-7 1e-7"/>{spin}</body>')
                acts += f'<position name="{sid}_steer_{name}" joint="{sid}_steer_{name}" kp="0.08" ctrlrange="-0.5 0.5"/>'
            else:
                wheels.append(f'<body name="{sid}_{name}" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}">{spin}</body>')
            # DC gear motor per wheel: torque = stall * (throttle - w/w0), computed each step
            # in World._drive_ctrl. Stall thrust ~0.34 N/wheel: it can push things but cannot
            # climb a wall; top speed emerges from back-EMF and bearing friction.
            tstall = MOTOR_STALL_THRUST * r
            acts += (f'<motor name="{sid}_drive_{name}" joint="{sid}_spin_{name}" ctrllimited="true" '
                     f'ctrlrange="{-2 * tstall:.6f} {2 * tstall:.6f}"/>')
        body = f'<body name="{sid}" pos="{park}"><freejoint name="j_{sid}"/>{chassis}{"".join(wheels)}<site name="c_{sid}" pos="0 0 {cz:.4f}"/></body>'
        return assets, body, acts
    zo = 0.0
    if a["kind"] == "procedural" and a["shape"] in ("box", "sphere"):
        dx, dy, dz = a["dims"]
        if a["shape"] == "sphere":
            zo = dx / 2
            g = f'type="sphere" size="{dx / 2:.4f}" condim="6"'
        else:
            zo = dz / 2
            g = f'type="box" size="{dx / 2:.4f} {dy / 2:.4f} {dz / 2:.4f}" condim="4"'
        geom = f'<geom name="g_{sid}" {g} mass="{mass:.4f}" friction="{fr}"{solref}/>'
    elif a.get("collision") == "sphere":          # truly round things must roll, not tumble on facets
        r = max(a["dims"]) / 2
        zo = r
        geom = f'<geom name="g_{sid}" type="sphere" size="{r:.4f}" mass="{mass:.4f}" friction="{fr}" condim="6"{solref}/>'
    else:
        verts = " ".join(f"{x:.5f}" for p in a["hull"] for x in p)
        assets = f'<mesh name="hull_{sid}" vertex="{verts}"/>'
        geom = f'<geom name="g_{sid}" type="mesh" mesh="hull_{sid}" mass="{mass:.4f}" friction="{fr}" condim="4"{solref}/>'
    sp["z_off"] = zo
    return assets, f'<body name="{sid}" pos="{park}"><freejoint name="j_{sid}"/>{geom}<site name="c_{sid}" pos="0 0 {a["dims"][2] / 2 - zo:.5f}"/></body>', acts


# object pairs are reported this far apart (inactive until they really touch: margin − gap = 0.5 mm), so an
# impact can be seen coming one step before it happens
PAIR_LOOKAHEAD = 0.08


def _pairs_xml(spawns: list[dict], dt: float, extra: list[dict] | None = None) -> str:
    """Contacts between two new objects use the pair's own restitution √(e₁e₂), calibrated to be exact
    (MuJoCo would otherwise average the two contacts' damping, which gives neither material's bounce)."""
    cs = [sp["_contact"] for sp in spawns if sp.get("_contact") and not sp["asset"].get("surface")] + list(extra or [])   # movable things only
    out = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            a, b = cs[i], cs[j]
            fr = [max(a["fr"][0], b["fr"][0]), max(a["fr"][1], b["fr"][1]), max(a["fr"][2], b["fr"][2])]
            # stiff (2 steps, MuJoCo's minimum) and critically damped: soft object-object contacts let a small
            # body sink into a big one and wedge against the floor, which pumps in energy. The rebound itself is
            # set exactly by the impact law (World._impact_law).
            out.append(f'<pair geom1="{a["geom"]}" geom2="{b["geom"]}" condim="{max(a["condim"], b["condim"])}" '
                       f'friction="{fr[0]:.4f} {fr[0]:.4f} {fr[1]:.5f} {fr[2]:.6f} {fr[2]:.6f}" solref="{2 * dt:.6f} 1" '
                       f'solimp="0.95 0.99 0.001" margin="{PAIR_LOOKAHEAD:.4f}" gap="{PAIR_LOOKAHEAD - 0.0005:.4f}"/>')
    return "".join(out)


def build_mjcf(scene: Scene, gravity: float = 9.81, spawns: list[dict] | None = None,
               env: dict | None = None, links: list[dict] | None = None) -> str:
    env = env or {}
    t = scene.table
    tx, ty = (t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2
    hx, hy = (t.x_max - t.x_min) / 2, (t.y_max - t.y_min) / 2
    th = config.TABLE_THICKNESS
    if scene.source.startswith("sandbox:") and not scene.source.endswith(":table"):
        th = 2.0            # open ground is solid: thin things can't sink through a 4 cm slab
    bodies = []
    mesh_assets: list[str] = []
    for o in scene.objects:
        q = yaw_quat(o.yaw)
        fr = friction_str(o.friction, o.size, getattr(o, "material", None), o.shape == "sphere")
        if o.shape == "mesh" and getattr(o, "hulls", None):
            # sweep-scanned object: its convex pieces (body frame), scaled with any later size edit (calibration,
            # overlap resolution); mass split by piece volume so the centre of mass follows the real shape
            allv = np.concatenate([np.asarray(h_, float) for h_ in o.hulls])
            ext = np.maximum(allv.max(0) - allv.min(0), 1e-4)
            k3 = np.asarray(o.size, float) / ext
            vols = []
            for h_ in o.hulls:
                hv = np.asarray(h_, float) * k3
                vols.append(max(1e-9, float(np.prod(hv.max(0) - hv.min(0)))))
            geoms = []
            for j, h_ in enumerate(o.hulls):
                hv = np.asarray(h_, float) * k3
                mesh_assets.append(f'<mesh name="m_{o.id}_{j}" vertex="{" ".join(f"{v:.5f}" for v in hv.ravel())}"/>')
                geoms.append(f'<geom name="g_{o.id}{"" if j == 0 else f"__{j}"}" type="mesh" mesh="m_{o.id}_{j}" '
                             f'mass="{o.mass * vols[j] / sum(vols):.6f}" friction="{fr}" priority="1" solmix="1000" condim="4"/>')
            geom = "".join(geoms)
        elif o.shape == "cylinder" and o.profile:
            # revolved collision mesh (bottle necks, tapered mugs tip like the real thing)
            h = o.size[2]
            verts = []
            for z, r in o.profile:
                for k in range(16):
                    a_ = 2 * math.pi * k / 16
                    verts += [r * math.cos(a_), r * math.sin(a_), z - h / 2]
            mesh_assets.append(f'<mesh name="m_{o.id}" vertex="{" ".join(f"{v:.5f}" for v in verts)}"/>')
            geom = (f'<geom name="g_{o.id}" type="mesh" mesh="m_{o.id}" mass="{o.mass:.5f}" friction="{fr}" priority="1" solmix="1000" condim="4"/>')
        elif o.shape == "cylinder":
            geom = (f'<geom name="g_{o.id}" type="cylinder" size="{o.size[0] / 2:.5f} {o.size[2] / 2:.5f}" '
                    f'mass="{o.mass:.5f}" friction="{fr}" priority="1" solmix="1000" condim="6"/>')
        else:
            geom = (f'<geom name="g_{o.id}" type="box" size="{o.size[0] / 2:.5f} {o.size[1] / 2:.5f} {o.size[2] / 2:.5f}" '
                    f'mass="{o.mass:.5f}" friction="{fr}" priority="1" solmix="1000" condim="4"/>')
        bodies.append(f'<body name="{o.id}" pos="{o.pos[0]:.5f} {o.pos[1]:.5f} {o.pos[2]:.5f}" '
                      f'quat="{q[0]:.6f} 0 0 {q[3]:.6f}"><freejoint name="j_{o.id}"/>{geom}<site name="c_{o.id}"/></body>')
    assets, acts = list(mesh_assets), []
    # real scanned objects keep MuJoCo's ordinary (soft) contacts with each other: stiff explicit pairs would make
    # things resting on each other (a book on two mugs) jitter. Their bounce is still set by the impact law.
    scene_contacts: list[dict] = []
    x_park = 60.0
    for sp in spawns or []:                      # each parked object gets its own slot, spaced by its size
        sp["park_x"] = x_park
        x_park += max(sp["asset"]["dims"]) + 3.0
    for sp in spawns or []:
        a_xml, b_xml, act_xml = _spawn_xml(sp)
        assets.append(a_xml)
        bodies.append(b_xml)
        acts.append(act_xml)
    return f"""
<mujoco model="casual_world">
  <compiler angle="radian"/>
  <option timestep="{env.get("timestep", config.SIM_TIMESTEP)}" gravity="0 0 {-gravity}" integrator="implicitfast" cone="elliptic" impratio="5"
          density="0" viscosity="0"/>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001" margin="0.0005"/></default>
  <asset>{''.join(assets)}</asset>
  <worldbody>
    <geom name="{FLOOR}" type="plane" pos="0 0 {-t.height_above_floor:.4f}" size="6 6 0.1" friction="0.7 0.01 0.002"/>
    <geom name="{TABLE}" type="box" pos="{tx:.4f} {ty:.4f} {-th / 2:.4f}" size="{hx:.4f} {hy:.4f} {th / 2:.4f}" friction="0.55 0.01 0.002"/>
    {''.join(bodies)}
    {_hand_bodies_xml(env["hand"]) if env.get("hand") else ""}
  </worldbody>
  <contact>{"" if env.get("no_pairs") else _pairs_xml(spawns or [], float(env.get("timestep", config.SIM_TIMESTEP)), scene_contacts)}</contact>
  <tendon>{''.join(_tendon_xml(l) for l in links or [])}</tendon>
  <actuator>{''.join(acts)}</actuator>
</mujoco>"""


# ---------------------------------------------------------------- the user's hand as a physical body
# Each bone of the tracked hand is a kinematic (mocap) capsule. Mocap bodies add nothing to qpos, so a world with a
# hand has exactly the state layout of one without: checkpoints, baselines and replays stay compatible. The hand's
# targets are interpolated on every physics step, so even a 3 m/s flick moves a bone only ~6 mm per 2 ms step: no
# tunnelling, no explosive contacts. contype 2 / conaffinity 1: bones touch objects (contype 1), never each other.
def _hand_bodies_xml(bones: list[dict]) -> str:
    out = []
    for k, b in enumerate(bones):
        L, r = max(0.004, b["length"]), b["radius"]
        out.append(f'<body name="hand_b{k}" mocap="true" pos="0 0 -5">'
                   f'<geom name="hand_g{k}" type="capsule" fromto="0 0 {-L / 2:.5f} 0 0 {L / 2:.5f}" size="{r:.4f}" '
                   f'contype="2" conaffinity="1" condim="4" friction="1.0 0.02 0.002" solref="0.004 1" '
                   f'rgba="0.95 0.8 0.7 0.5"/></body>')
    return "".join(out)


HAND_BONES: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _quat_z_to(d: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) rotating +z onto direction d."""
    d = d / (np.linalg.norm(d) + 1e-12)
    c = float(d[2])
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([1.0 + c, -d[1], d[0], 0.0])               # axis = z x d
    return q / np.linalg.norm(q)


# Rolling resistance coefficient c_rr (resisting force / normal force) by material. MuJoCo's rolling
# friction has units of length (torque = mu_roll * N), so mu_roll = c_rr * radius.
C_RR = {"metal": 0.001, "glass": 0.001, "ceramic": 0.002, "wood": 0.003, "plastic": 0.004, "stone": 0.004,
        "rubber": 0.008, "leather": 0.01, "shell": 0.006, "food": 0.02, "flesh": 0.03, "fabric": 0.03}


def spawn_volume(sp: dict) -> float:
    """Displaced volume (m^3) of a spawned object: given (mass / real density) or from its shape."""
    if sp.get("volume"):
        return float(sp["volume"])
    a = sp["asset"]
    if a.get("kind") == "proxy":
        L, W, H = a["dims"]
        k = {"sphere": math.pi / 6, "ellipsoid": math.pi / 6, "cylinder": math.pi / 4, "capsule": 0.72}.get(a.get("shape"), 1.0)
        return float(k * L * W * H)
    hull = a.get("hull")
    if hull is not None and len(hull) >= 4:
        try:
            from scipy.spatial import ConvexHull
            return float(ConvexHull(np.asarray(hull, float)).volume)
        except Exception:
            pass
    L, W, H = a["dims"]
    return float((math.pi / 6 if a.get("shape") == "sphere" or a.get("collision") == "sphere" else 0.6) * L * W * H)


def submerged_fraction(sp: dict, x: float) -> float:
    """Fraction of the volume under water when a fraction x of the height is: a sphere/ellipsoid x²(3 − 2x),
    an upright prism (box, cylinder) x."""
    x = float(np.clip(x, 0.0, 1.0))
    a = sp.get("asset", {})
    shape = (sp.get("drag") or {}).get("shape") or a.get("shape")
    return x * x * (3 - 2 * x) if shape in ("sphere", "ellipsoid") else x


def scene_drag_spec(o) -> dict:
    return _drag.spec("cylinder" if o.shape == "cylinder" else "sphere" if o.shape == "sphere" else "box", o.size)


def scene_volume(o) -> float:
    if getattr(o, "shape", None) == "mesh" and getattr(o, "volume_m3", None):
        return float(o.volume_m3)
    L, W, H = o.size
    return float((math.pi / 4 if o.shape == "cylinder" else math.pi / 6 if o.shape == "sphere" else 1.0) * L * W * H)


def _drag_spec(sp: dict) -> dict:
    """Drag description of a spawned object: given by the experiment, or from its collision shape."""
    if sp.get("drag"):
        return sp["drag"]
    a = sp["asset"]
    if a.get("kind") == "proxy":
        return _drag.spec(a.get("shape"), a["dims"], a.get("cd"))
    if a.get("collision") == "sphere" or a.get("shape") == "sphere":
        d = max(a["dims"])
        return _drag.spec("sphere", [d, d, d], a.get("cd"))
    if a.get("kind") == "procedural":
        return _drag.spec(a.get("shape"), a["dims"], a.get("cd"))
    return _drag.spec("box", a["dims"], a.get("cd") or 0.8)        # rounded real objects: between a cube and a sphere


def friction_str(slide: float, size, material: str | None = None, round_: bool = False) -> str:
    """'slide torsional rolling'. Round things touch at a point: tiny twist resistance and rolling resistance
    c_rr * radius. Flat-based things resist twisting over their footprint (~2/3 mu a) and don't roll."""
    r = max(1e-4, 0.5 * min(float(x) for x in size))
    if round_:
        c = C_RR.get(material or "", 0.005)
        return f"{slide:.3f} {min(0.005, 0.05 * r):.5f} {c * r:.6f}"
    foot = 0.5 * min(float(x) for x in list(size)[:2])
    return f"{slide:.3f} {min(0.05, max(0.005, 0.67 * slide * foot)):.5f} 0.002"


def _tendon_xml(l: dict) -> str:
    """Rope = tendon with a maximum length (slack until taut); spring = tendon with stiffness.
    Both start inactive (huge range / zero stiffness) and are switched on by their 'connect' step."""
    if l["kind"] == "spring":
        return (f'<spatial name="{l["id"]}" stiffness="0" damping="{l.get("damping", 0.0)}" springlength="{l["length"]:.5f}" width="0.003">'
                f'<site site="c_{l["a"]}"/><site site="c_{l["b"]}"/></spatial>')
    return (f'<spatial name="{l["id"]}" limited="true" range="0 1000" width="0.003" solreflimit="0.003 1">'
            f'<site site="c_{l["a"]}"/><site site="c_{l["b"]}"/></spatial>')


class World:
    """Thin wrapper giving object-id level access to the MuJoCo model."""

    def __init__(self, scene: Scene, gravity: float = 9.81, spawns: list[dict] | None = None,
                 env: dict | None = None, links: list[dict] | None = None):
        self.scene = scene
        self.env = env or {}
        self.links = {l["id"]: l for l in (links or [])}
        self.spawns = {sp["sid"]: sp for sp in (spawns or [])}
        self.m = mujoco.MjModel.from_xml_string(build_mjcf(scene, gravity, spawns, env, links))
        self.z_off = {sid: float(sp.get("z_off", 0.0)) for sid, sp in self.spawns.items()}
        self.scene_e = {o.id: float(getattr(o, "bounciness", 0.4)) for o in scene.objects if not o.static}
        self.propel: dict[str, dict] = {}
        self.d = mujoco.MjData(self.m)
        self.scene_ids = [o.id for o in scene.objects]
        self.ids = self.scene_ids + list(self.spawns)
        self.body = {oid: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, oid) for oid in self.ids}
        self.geom = {oid: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{oid}") for oid in self.ids}
        self.qadr = {oid: int(self.m.jnt_qposadr[self.m.body_jntadr[b]]) for oid, b in self.body.items()}
        self.vadr = {oid: int(self.m.jnt_dofadr[self.m.body_jntadr[b]]) for oid, b in self.body.items()}
        # every geom belongs to the object whose root body it hangs under (car wheels -> car)
        root_name = {self.body[oid]: oid for oid in self.ids}
        self.geom_owner: dict[int, str] = {}
        for g in range(self.m.ngeom):
            root = int(self.m.body_rootid[self.m.geom_bodyid[g]])
            if root in root_name:
                self.geom_owner[g] = root_name[root]
        self.geom_owner[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, TABLE)] = TABLE
        self.geom_owner[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, FLOOR)] = FLOOR
        self.geoms_of: dict[str, list[int]] = {}
        for g, oid in self.geom_owner.items():
            self.geoms_of.setdefault(oid, []).append(g)
        self._contype = {g: (int(self.m.geom_contype[g]), int(self.m.geom_conaffinity[g])) for g in range(self.m.ngeom)}
        self.pinned: dict[str, np.ndarray] = {}
        self.removed: set[str] = set()
        self.drive: dict[str, dict] = {}
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(self.m.nu)}
        self.jq = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_qposadr[j]) for j in range(self.m.njnt)}
        self.jd = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j): int(self.m.jnt_dofadr[j]) for j in range(self.m.njnt)}
        for sid in self.spawns:          # parked, intangible, until their spawn intervention
            self._set_tangible(sid, False)
            self.removed.add(sid)
        # Archimedes: MuJoCo's fluid model gives drag but no buoyancy, so we add rho * V * g at the centre
        # of mass of every new object (the sandbox worlds are fully immersed in their fluid)
        rho = float(self.env.get("density", 0.0))
        self.buoyant = {sid: rho * spawn_volume(sp) * gravity for sid, sp in self.spawns.items()} if rho > 0 else {}
        self.surface_z = self.env.get("surface_z")          # water level (None: fully immersed)
        # drag: the textbook law (physics/drag.py), the same one the theory checks solve
        self.rho = rho
        self.mu = float(self.env.get("viscosity", 0.0))
        self.drag = {sid: _drag_spec(sp) for sid, sp in self.spawns.items()} if (rho > 0 or self.mu > 0) else {}
        self._scene_air = [o for o in scene.objects if not o.static] if (rho > 0 or self.mu > 0) else []
        self.base_mass = {sid: float(self.m.body_mass[self.body[sid]]) for sid in self.spawns}
        self.added = {sid: _drag.added_mass(self.drag[sid], rho, spawn_volume(sp)) for sid, sp in self.spawns.items()} if rho > 100 else {}
        self.height_of = {sid: float(sp["asset"]["dims"][2]) for sid, sp in self.spawns.items()}
        for o in self._scene_air:            # real scanned objects feel the same air as everything else
            self.drag[o.id] = scene_drag_spec(o)
            if rho > 0:
                self.buoyant[o.id] = rho * scene_volume(o) * gravity
                self.height_of[o.id] = float(o.size[2])
        mujoco.mj_forward(self.m, self.d)
        for o in scene.objects:
            if o.static:
                self.pin(o.id)
        self.hand_bones = self.env.get("hand") or []
        self.hand_mocap = [int(self.m.body_mocapid[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"hand_b{k}")])
                           for k in range(len(self.hand_bones))]
        self.hand_track: list | None = None                   # [(t_world, joints 21x3 or None), ...]
        self.hand_last: np.ndarray | None = None

    # -- the user's hand (kinematic bones following a recorded or live track)
    def set_hand_track(self, updates: list, t0: float) -> None:
        """updates: [[t_rel, joints|None], ...] with t_rel from t0 (world time)."""
        self.hand_track = [(t0 + float(u[0]), None if u[1] is None else np.asarray(u[1], float)) for u in updates]

    def append_hand(self, t_world: float, joints) -> None:
        if self.hand_track is None:
            self.hand_track = []
        self.hand_track.append((t_world, None if joints is None else np.asarray(joints, float)))

    def _hand_joints(self, t: float) -> np.ndarray | None:
        tr = self.hand_track
        if not tr or t < tr[0][0]:
            return None
        i = len(tr) - 1
        while i > 0 and tr[i][0] > t:
            i -= 1
        a = tr[i]
        if a[1] is None:
            return None
        if i + 1 < len(tr) and tr[i + 1][1] is not None and tr[i + 1][0] > a[0]:
            b = tr[i + 1]
            k = min(1.0, (t - a[0]) / (b[0] - a[0]))
            return a[1] + (b[1] - a[1]) * k
        return a[1]

    def _hand_drive(self) -> None:
        J = self._hand_joints(float(self.d.time))
        if J is None:
            for k, mid in enumerate(self.hand_mocap):          # parked well below the floor
                self.d.mocap_pos[mid] = [0.0, 0.0, -5.0 - 0.05 * k]
            self.hand_last = None
            return
        J = J.copy()
        r = np.array([b["radius"] for b in self.hand_bones])
        J[:, 2] = np.maximum(J[:, 2], 0.009)                     # a hand can't be inside the table top
        for k, (a, b) in enumerate(HAND_BONES):
            self.d.mocap_pos[self.hand_mocap[k]] = (J[a] + J[b]) / 2 + [0, 0, max(0.0, r[k] - min(J[a][2], J[b][2]))]
            self.d.mocap_quat[self.hand_mocap[k]] = _quat_z_to(J[b] - J[a])
        self.hand_last = J

    def _set_tangible(self, oid: str, on: bool) -> None:
        for g in self.geoms_of.get(oid, []):
            ct, ca = self._contype[g] if on else (0, 0)
            self.m.geom_contype[g] = ct
            self.m.geom_conaffinity[g] = ca

    def spawn(self, sid: str, pos, yaw: float, static: bool = False, pitch: float = 0.0) -> None:
        a = self.qadr[sid]
        qz = yaw_quat(yaw)
        # yaw about z, then pitch about the body's y axis (ramps, things resting on a slope)
        cy, sy = qz[0], qz[3]
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        q = [cy * cp, -sy * sp, cy * sp, sy * cp]
        c = np.array(pos[:3], float) + self._up(q) * self.z_off.get(sid, 0.0)      # bottom point -> body centre
        self.d.qpos[a:a + 7] = [c[0], c[1], c[2], q[0], q[1], q[2], q[3]]
        self.d.qvel[self.vadr[sid]:self.vadr[sid] + 6] = 0
        self._set_tangible(sid, True)
        self.removed.discard(sid)
        mujoco.mj_forward(self.m, self.d)
        if static:
            self.pin(sid)

    def wheel_state(self, sid: str) -> list[float]:
        """[steer_fl, steer_fr, spin_fl, spin_fr, spin_rl, spin_rr] (rad)."""
        q = self.d.qpos
        return [float(q[self.jq[f"{sid}_steer_fl"]]), float(q[self.jq[f"{sid}_steer_fr"]])] + \
               [float(q[self.jq[f"{sid}_spin_{w}"]]) for w in ("fl", "fr", "rl", "rr")]

    # -- state -------------------------------------------------------------
    def qpos(self) -> np.ndarray:
        return self.d.qpos.copy()

    def set_qpos(self, q: np.ndarray) -> None:
        self.d.qpos[:len(q)] = q          # scene bodies come first; spawned bodies keep their parked pose
        self.d.qvel[:] = 0
        mujoco.mj_forward(self.m, self.d)

    @staticmethod
    def _up(q) -> np.ndarray:
        """The body's own z axis in the world, for quaternion q = (w, x, y, z)."""
        w, x, y, z = (float(v) for v in q)
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def pose(self, oid: str) -> np.ndarray:
        """x y z qw qx qy qz of the object's reference point (the bottom centre of its upright shape)."""
        a = self.qadr[oid]
        q = self.d.qpos[a:a + 7].copy()
        zo = self.z_off.get(oid, 0.0)
        if zo:
            q[:3] -= self._up(q[3:7]) * zo
        return q

    def speed(self, oid: str) -> tuple[float, float]:
        a = self.vadr[oid]
        v = self.d.qvel[a:a + 6]
        return float(np.linalg.norm(v[:3])), float(np.linalg.norm(v[3:]))

    def tilt_deg(self, oid: str) -> float:
        R = self.d.xmat[self.body[oid]].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def mass(self, oid: str) -> float:
        """The object's own mass (without water it drags along)."""
        if oid in self.added:
            return self.base_mass[oid]
        return float(self.m.body_mass[self.body[oid]])

    # -- interventions -------------------------------------------------------
    def pin(self, oid: str) -> None:
        a = self.qadr[oid]
        self.pinned[oid] = self.d.qpos[a:a + 7].copy()          # the raw joint state, restored every step

    def place(self, oid: str, centre, hold: bool = True, clear: bool = False) -> None:
        """Move an object (a real scanned one too) so its centre is at `centre`, keeping its orientation, at rest.
        clear=True raises it until it overlaps nothing (a hand sets a mug down ON the book, never inside it)."""
        a = self.qadr[oid]
        self.unpin(oid)
        self.d.qpos[a:a + 3] = np.asarray(centre, float)
        self.d.qvel[self.vadr[oid]:self.vadr[oid] + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        if clear:
            self.lift_clear(oid)
        if hold:
            self.pin(oid)

    def lift_clear(self, oid: str, max_up: float = 0.6) -> float:
        """Raise an object until no contact on it penetrates more than 2 mm. Returns how far it went up."""
        a = self.qadr[oid]
        z0 = float(self.d.qpos[a + 2])
        for _ in range(300):
            worst = 0.0
            for i in range(self.d.ncon):
                c = self.d.contact[i]
                if oid in (self.geom_owner.get(int(c.geom1)), self.geom_owner.get(int(c.geom2))):
                    worst = min(worst, float(c.dist))
            if worst > -2e-3 or self.d.qpos[a + 2] - z0 > max_up:
                break
            self.d.qpos[a + 2] += max(0.003, -worst)
            mujoco.mj_forward(self.m, self.d)
        return float(self.d.qpos[a + 2]) - z0

    def unpin(self, oid: str) -> None:
        self.pinned.pop(oid, None)

    def remove(self, oid: str) -> None:
        self._set_tangible(oid, False)
        self.unpin(oid)
        a = self.qadr[oid]
        k = self.ids.index(oid)
        px = self.spawns[oid].get("park_x", 50 + 2 * k) if oid in self.spawns else 50 + 2 * k
        self.d.qpos[a:a + 7] = [px, 50, -20, 1, 0, 0, 0]
        self.d.qvel[self.vadr[oid]:self.vadr[oid] + 6] = 0
        self.removed.add(oid)

    def set_mass(self, oid: str, mass: float) -> None:
        b = self.body[oid]
        if oid in self.base_mass:
            self.base_mass[oid] = mass
        ratio = mass / max(1e-9, self.m.body_mass[b])
        self.m.body_mass[b] = mass
        self.m.body_inertia[b] *= ratio
        self._refresh_constants()                  # otherwise MuJoCo keeps simulating the old mass

    def set_friction(self, oid: str, mu: float) -> None:
        for g in self.geoms_of.get(oid, [self.geom[oid]]):
            self.m.geom_friction[g, 0] = mu

    def set_gravity(self, g: float) -> None:
        self.m.opt.gravity[:] = [0, 0, -g]

    # -- stepping ------------------------------------------------------------
    def connect(self, lid: str) -> None:
        l = self.links[lid]
        t = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_TENDON, lid)
        if l["kind"] == "spring":
            self.m.tendon_stiffness[t] = l["stiffness"]
            self.m.tendon_damping[t] = l.get("damping", 0.0)
        else:
            self.m.tendon_range[t] = [0.0, l["length"]]

    def tendon_state(self, lid: str) -> tuple[float, float]:
        """(length m, tension N) of a rope/spring."""
        t = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_TENDON, lid)
        L = float(self.d.ten_length[t])
        if self.links[lid]["kind"] == "spring":
            return L, float(self.m.tendon_stiffness[t] * (L - self.m.tendon_lengthspring[t][0]))
        T = 0.0
        for i in range(self.d.nefc):
            if self.d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON and self.d.efc_id[i] == t:
                T += abs(float(self.d.efc_force[i]))
        return L, T

    def _propel_ctrl(self, sid: str, pr: dict) -> None:
        """Self-propulsion toward a target (animals, swimmers, fighters, chasers).
        Ground: the push comes from grip, so it is capped at mu*m*g. Water/air/space: thrust."""
        b = self.body[sid]
        if self.d.time >= pr["t_end"]:
            self.d.xfrc_applied[b, :3] = 0
            return
        here = self.d.xipos[b]
        tgt = pr.get("target")
        if pr.get("hand") and tgt in self.body and any(
                {self.geom_owner.get(int(self.d.contact[i].geom1)), self.geom_owner.get(int(self.d.contact[i].geom2))} == {sid, tgt}
                for i in range(self.d.ncon) if self.d.contact[i].dist < self.d.contact[i].includemargin):
            pr["t_end"] = self.d.time                     # a hand lets go when what it pushes reaches the target
            self.d.xfrc_applied[b] = 0
            return
        if tgt in self.body:
            goal = self.d.xipos[self.body[tgt]]
        elif pr.get("point") is not None:
            goal = np.asarray(pr["point"], float)
        else:
            goal = here + np.asarray(pr.get("direction", [1, 0, 0]), float)
        d = goal - here
        if pr.get("planar", True):
            d[2] = 0.0
        n = np.linalg.norm(d)
        if n < 1e-6:
            self.d.xfrc_applied[b, :3] = 0
            return
        u = d / n
        v_here = self.d.qvel[self.vadr[sid]:self.vadr[sid] + 3]
        v_top = float(pr.get("top_speed") or 1e9)
        # legs, fins and engines push less as speed rises: zero extra push at the real top speed
        Fm = pr["force"] * float(np.clip((1.0 - float(v_here @ u) / v_top) / 0.15, -0.5, 1.0))
        F = u * Fm
        if pr.get("lift"):
            # wings/rotors/engines carry the weight; a PD holds the cruising altitude (or follows the target's)
            m_ = self.mass(sid)
            g_ = -float(self.m.opt.gravity[2])
            if pr.get("z_hold") is None:
                pr["z_hold"] = float(here[2])
            z_t = float(goal[2]) if tgt in self.body else pr["z_hold"]
            wn = 2.0
            F[2] += m_ * g_ + m_ * (wn * wn * (z_t - float(here[2])) - 2 * wn * float(v_here[2]))
        self.d.xfrc_applied[b, :3] = F
        if pr.get("hand"):
            self.d.xfrc_applied[b, 3:] = 0                # a hand pushes through the centre: no steering
            return
        if pr.get("planar", True):
            # Balance + heading, like any animal or vehicle: stay upright (PD on tilt, sized from the
            # body's own weight, height and inertia) and turn to face where it's going (PD on yaw).
            m_ = self.mass(sid)
            R = self.d.xmat[b].reshape(3, 3)
            zb = R[:, 2]
            w = R @ self.d.qvel[self.vadr[sid] + 3:self.vadr[sid] + 6]          # world angular velocity
            h = max(0.05, float(here[2]))
            I = float(np.max(self.m.body_inertia[b]))
            kp = 6.0 * m_ * 9.81 * h
            kd = 2.0 * math.sqrt(kp * I)
            tilt = np.cross(zb, [0.0, 0.0, 1.0])
            w_tilt = w - (w @ np.array([0.0, 0.0, 1.0])) * np.array([0.0, 0.0, 1.0])
            tau = kp * tilt - kd * w_tilt
            fwd = R[:, 0].copy(); fwd[2] = 0.0
            if np.linalg.norm(fwd) > 1e-6:
                fwd /= np.linalg.norm(fwd)
                want = d / n
                yaw_err = math.atan2(fwd[0] * want[1] - fwd[1] * want[0], float(fwd @ want))
                Iz = float(self.m.body_inertia[b][2])
                kpz = 2.0 * Iz * 4.0
                tau[2] += kpz * yaw_err - 2.0 * math.sqrt(kpz * Iz) * w[2]
            self.d.xfrc_applied[b, 3:] = tau

    def _buoyancy(self) -> None:
        g = -float(self.m.opt.gravity[2])
        for sid, Fb in self.buoyant.items():
            if sid in self.removed or sid in self.pinned or Fb <= 0:
                continue
            b = self.body[sid]
            frac = 1.0
            if self.surface_z is not None:      # only the submerged part displaces water
                h = max(1e-6, self.height_of[sid])
                bottom = float(self.d.xipos[b][2]) - h / 2
                frac = submerged_fraction(self.spawns.get(sid, {"drag": self.drag.get(sid)}), float((self.surface_z - bottom) / h))
                Fb *= frac
            if sid in self.added:
                # added mass: the body's inertia to being pushed grows by m_a (water it must push along) but its
                # weight does not -> armature on its three translation axes (the body origin is its centre of mass)
                ma = round(self.added[sid] * frac / max(1e-9, self.added[sid]) * 50) / 50 * self.added[sid]
                v = self.vadr[sid]
                if abs(float(self.m.dof_armature[v]) - ma) > 1e-12:
                    self.m.dof_armature[v:v + 3] = ma
                    self._consts_dirty = True
            if Fb <= 0:
                continue
            mujoco.mj_applyFT(self.m, self.d, np.array([0.0, 0.0, Fb]), np.zeros(3), self.d.xipos[b].copy(), b, self.d.qfrc_applied)

    def _refresh_constants(self) -> None:
        """MuJoCo 4 caches inertia-derived constants: after changing masses/armature at run time they only take
        effect after mj_setConst, which also resets positions - so the state is kept around it."""
        q = self.d.qpos.copy()
        mujoco.mj_setConst(self.m, self.d)
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        self._consts_dirty = False

    def _drag(self) -> None:
        res = np.zeros(6)
        for sid, dsp in self.drag.items():
            if sid in self.removed or sid in self.pinned:
                continue
            b = self.body[sid]
            rho = self.rho
            if self.surface_z is not None:      # above the water line: air
                if float(self.d.xipos[b][2]) > self.surface_z:
                    rho = 1.2
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
            R = self.d.xmat[b].reshape(3, 3)
            dt = float(self.m.opt.timestep)
            m_eff = float(self.m.body_subtreemass[b]) + (float(self.m.dof_armature[self.vadr[sid]]) if sid in self.added else 0.0)
            f = R @ _drag.step_force(dsp, R.T @ res[3:], rho, self.mu, m_eff, dt)
            tq = R @ _drag.torque(dsp, R.T @ res[:3], rho, float(min(self.m.body_inertia[b])), dt)
            mujoco.mj_applyFT(self.m, self.d, f, tq, self.d.xipos[b].copy(), b, self.d.qfrc_applied)

    # -- impact law ---------------------------------------------------------------
    # MuJoCo's soft contacts make an impact's rebound depend on speed and on where the step grid falls
    # (e = 0.45-0.56 for one setting). Newton's law of restitution is enforced on top: when two bodies
    # separate after an impact, one equal-and-opposite impulse at the contact point sets the separation
    # speed to exactly e x the approach speed. Momentum is conserved exactly; resting/rolling contacts,
    # which never separate, are untouched.
    def _e_of(self, oid: str) -> float | None:
        if oid in self.scene_e:
            return self.scene_e[oid]
        c = self.spawns.get(oid, {}).get("_contact")
        return None if c is None else float(c["e"])

    def _pair_e(self, a: str, b: str) -> float | None:
        ea = None if a in (TABLE, FLOOR) else self._e_of(a)
        eb = None if b in (TABLE, FLOOR) else self._e_of(b)
        if a in (TABLE, FLOOR):
            return eb
        if b in (TABLE, FLOOR):
            return ea
        return None if ea is None or eb is None else pair_restitution(ea, eb)

    def _point_jac(self, oid: str, p: np.ndarray) -> np.ndarray:
        """3 x nv Jacobian of the world point p moving with the object (zero for the ground and pinned things)."""
        J = np.zeros((3, self.m.nv))
        if oid in (TABLE, FLOOR) or oid in self.pinned:
            return J
        mujoco.mj_jac(self.m, self.d, J, None, p, self.body[oid])
        return J

    def _impacts_now(self) -> dict:
        now: dict[tuple, list] = {}
        self._coming: set[tuple] = set()
        dt = float(self.m.opt.timestep)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a, b = self.geom_owner.get(int(c.geom1)), self.geom_owner.get(int(c.geom2))
            if a is None or b is None or a == b or a in self.removed or b in self.removed:
                continue
            if c.dist >= c.includemargin:          # not touching yet (look-ahead band of an object pair)
                movable = lambda o: o in self.spawns or o in self.scene_e
                if movable(a) and movable(b):
                    n = np.array(c.frame[:3])
                    g = (self._point_jac(b, np.array(c.pos)) - self._point_jac(a, np.array(c.pos))).T @ n
                    vn = float(g @ self.d.qvel)
                    if vn < -0.05 and abs(float(n[2])) < 0.5 and float(c.dist) - float(c.includemargin) < -vn * 3 * dt + 0.002:
                        self._coming.add(tuple(sorted((a, b))))
                continue
            ok = lambda o: o in self.spawns or o in self.scene_e or o in (TABLE, FLOOR)
            if not (ok(a) and ok(b)) or (a in (TABLE, FLOOR) and b in (TABLE, FLOOR)):
                continue
            n = np.array(c.frame[:3])                       # from geom1 to geom2
            if (a, b) > (b, a):
                a, b, n = b, a, -n
            now.setdefault((a, b), []).append((np.array(c.pos), n))
        return now

    def _com_vel(self, oid: str, qvel: np.ndarray) -> list[float]:
        if oid in (TABLE, FLOOR) or oid not in self.vadr:
            return [0.0, 0.0, 0.0]
        a, b = self.vadr[oid], self.body[oid]
        w = self.d.xmat[b].reshape(3, 3) @ qvel[a + 3:a + 6]
        v = qvel[a:a + 3] + np.cross(w, self.d.xipos[b] - self.d.xpos[b])
        return [float(x) for x in v]

    def _set_friction_holds(self, objs: set) -> None:
        """A real impact lasts ~1 ms, so the floor's friction impulse during it is negligible (μmg x 1 ms). MuJoCo's
        soft friction would instead fight the sudden velocity change and pop the bodies upward (the floor's normal
        force spikes to 30x their weight). So, for the few steps an impact between two movable things lasts (and
        the step before, when it is seen coming), their friction against everything else is suspended."""
        if not hasattr(self, "_fric_hold"):
            self._fric_hold: dict[str, dict] = {}
        for oid in [o for o in self._fric_hold if o not in objs]:
            for g, f in self._fric_hold.pop(oid).items():
                self.m.geom_friction[g] = f
        for oid in objs:
            if oid not in self._fric_hold:
                gs = self.geoms_of.get(oid, [])
                self._fric_hold[oid] = {g: self.m.geom_friction[g].copy() for g in gs}
                for g in gs:
                    self.m.geom_friction[g] = [1e-6, 0.0, 0.0]

    def _impact_law(self, qvel_prev: np.ndarray) -> None:
        if not hasattr(self, "_imp"):
            self._imp: dict[tuple, dict | None] = {}
            self.impulses: list[tuple[str, str, float]] = []
            self.impact_log: list[dict] = []
        self.impulses = []
        now = self._impacts_now()
        for key, pts in now.items():
            p = np.mean([x[0] for x in pts], axis=0)
            n = np.sum([x[1] for x in pts], axis=0)
            n = n / max(1e-9, float(np.linalg.norm(n)))
            a, b = key
            g = (self._point_jac(b, p) - self._point_jac(a, p)).T @ n
            if key not in self._imp:
                vin = float(g @ qvel_prev)                    # approach speed before this step's contact forces
                e = self._pair_e(a, b)
                # a real impact: two movable things meeting, or something landing/hitting a surface faster than
                # 0.3 m/s (a 5 mm drop); slower touches are micro-contacts of sliding/rolling, left to the solver
                surface = lambda o: o in (TABLE, FLOOR) or o in self.pinned or bool(self.spawns.get(o, {}).get("asset", {}).get("surface"))
                v_min = 0.3 if (surface(a) or surface(b)) else 0.05
                self._imp[key] = {"vin": vin, "e": e, "p": p, "n": n, "t0": float(self.d.time) - float(self.m.opt.timestep),
                                  "va": self._com_vel(a, qvel_prev), "vb": self._com_vel(b, qvel_prev)} if (vin < -v_min and e is not None) else None
                if self._imp[key] is not None and not (surface(a) or surface(b)) and abs(float(n[2])) < 0.5:
                    self._imp[key]["fric"] = True          # sideways impacts only: something landing ON a thing keeps its grip
            elif self._imp[key] is not None:
                self._imp[key]["p"], self._imp[key]["n"] = p, n
        for key in [k for k in self._imp if k not in now]:
            info = self._imp.pop(key)
            if info is None:
                continue
            a, b = key
            if a in self.removed or b in self.removed:
                continue
            g = (self._point_jac(b, info["p"]) - self._point_jac(a, info["p"])).T @ info["n"]
            vout = float(g @ self.d.qvel)
            vdes = -info["e"] * info["vin"]
            if abs(vout - vdes) >= 1e-4:
                x = np.zeros(self.m.nv)
                mujoco.mj_solveM(self.m, self.d, x.reshape(1, -1), g.reshape(1, -1))
                k = float(g @ x)
                if k > 1e-12:
                    P = (vdes - vout) / k
                    self.d.qvel[:] += x * P
                    self.impulses.append((a, b, abs(P)))
            # the impact as a high-speed camera would see it: velocities at first touch and at separation
            self.impact_log.append({"a": a, "b": b, "t0": round(info["t0"], 5), "t1": round(float(self.d.time), 5),
                                    "n": [float(x) for x in info["n"]], "vin": info["vin"], "e": info["e"],
                                    "va0": info["va"], "vb0": info["vb"], "va1": self._com_vel(a, self.d.qvel), "vb1": self._com_vel(b, self.d.qvel)})

    def _update_holds(self) -> None:
        objs = set()
        for key, info in getattr(self, "_imp", {}).items():
            if info is not None and info.get("fric"):
                objs |= set(key)
        for key in getattr(self, "_coming", set()):
            objs |= set(key)
        self._set_friction_holds(objs)

    def step(self) -> None:
        qvel_prev = self.d.qvel.copy()
        self.d.qfrc_applied[:] = 0
        if self.buoyant:
            self._buoyancy()
            if getattr(self, "_consts_dirty", False):
                self._refresh_constants()
        if self.drag:
            self._drag()
        for sid, pr in self.propel.items():
            self._propel_ctrl(sid, pr)
        for sid, dr in self.drive.items():
            self._drive_ctrl(sid, dr)
        for oid, p in self.pinned.items():
            a, v = self.qadr[oid], self.vadr[oid]
            self.d.qpos[a:a + 7] = p
            self.d.qvel[v:v + 6] = 0
        if self.hand_mocap and self.hand_track is not None:
            self._hand_drive()
        mujoco.mj_step(self.m, self.d)
        if not self.env.get("no_impact_law"):
            self._impact_law(qvel_prev)
            self._update_holds()

    def _drive_ctrl(self, sid: str, dr: dict) -> None:
        """Vehicle controller on top of a DC-motor model.

        Each wheel motor: torque = stall * (u - w / w0), u = throttle in [-1, 1].
        speed=None means full throttle (u = 1): top speed is then a physical result.
        Otherwise u = feed-forward (target/no-load) + proportional correction.
        Throttle off (time up, or within the stopping distance of the goal) leaves
        u = 0, i.e. back-EMF braking, like a real toy car's motor."""
        a = self.spawns[sid]["asset"]
        r = a["wheels"]["radius"]
        t_stall = MOTOR_STALL_THRUST * r
        w0 = MOTOR_NOLOAD_SPEED / r
        on = self.d.time < dr["t_end"]
        braking = False
        steer = 0.0
        goal = dr.get("goal")
        if goal is not None:
            pos = self.d.xpos[self.body[sid]][:2]
            to = np.asarray(goal) - pos
            dist = float(np.linalg.norm(to))
            if dr.get("stop_at_goal"):
                v = float(np.linalg.norm(self.d.qvel[self.vadr[sid]:self.vadr[sid] + 2]))
                # stopping distance under active braking (~3 m/s^2 usable) plus a margin
                if dr.get("braking") or dist < v * v / (2 * 3.0) + dr.get("stop_dist", 0.03):
                    dr["braking"] = True
                    on, braking = False, True
            if on and dist > 0.02:
                R = self.d.xmat[self.body[sid]].reshape(3, 3)
                fwd = R[:2, 0]
                err = math.atan2(fwd[0] * to[1] - fwd[1] * to[0], fwd @ to)
                steer = float(np.clip(1.5 * err, -0.45, 0.45))
        for w in ("fl", "fr", "rl", "rr"):
            om = float(self.d.qvel[self.jd[f"{sid}_spin_{w}"]])
            if braking:
                u = float(np.clip(-3.0 * om / w0, -1.0, 1.0))      # reverse voltage: active brake
            elif not on:
                u = 0.0                                            # throttle released: back-EMF only
            elif dr.get("speed") is None:
                u = 1.0
            else:
                wt = dr["speed"] / r
                u = float(np.clip(wt / w0 + 4.0 * (wt - om) / w0, -1.0, 1.0))
            self.d.ctrl[self.act[f"{sid}_drive_{w}"]] = t_stall * (u - om / w0)
        for w in ("fl", "fr"):
            self.d.ctrl[self.act[f"{sid}_steer_{w}"]] = steer

    def kinematics(self, oid: str) -> tuple[np.ndarray, float, float]:
        """(centre-of-mass velocity [world], kinetic energy [J], potential energy above table [J])."""
        b = self.body[oid]
        a = self.vadr[oid]
        v0 = self.d.qvel[a:a + 3]                         # free joint: body-origin velocity, world frame
        R = self.d.xmat[b].reshape(3, 3)
        w = R @ self.d.qvel[a + 3:a + 6]                  # angular velocity: local -> world
        r = self.d.xipos[b] - self.d.xpos[b]
        v = v0 + np.cross(w, r)
        m = float(self.m.body_subtreemass[b])
        Ri = self.d.ximat[b].reshape(3, 3)
        Iw = Ri @ np.diag(self.m.body_inertia[b]) @ Ri.T
        ke = 0.5 * m * float(v @ v) + 0.5 * float(w @ Iw @ w)
        pe = m * float(-self.m.opt.gravity[2]) * float(self.d.subtree_com[b][2])
        return v, ke, pe

    def touching_pairs(self) -> list[tuple[str, str]]:
        """Cheap per-step scan of which objects touch (no forces): catches glancing
        contacts that start and end between event samples."""
        out = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a = self.geom_owner.get(int(c.geom1))
            b = self.geom_owner.get(int(c.geom2))
            if a and b and a != b and a not in (TABLE, FLOOR) and b not in (TABLE, FLOOR):
                out.append((a, b))
        return out

    def contacts(self) -> list[dict]:
        """Object-level contacts with world-frame normal (pointing from a to b) and normal force."""
        out = []
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            a = self.geom_owner.get(int(c.geom1))
            b = self.geom_owner.get(int(c.geom2))
            if a is None or b is None or a in self.removed or b in self.removed:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, f6)
            n = np.array(c.frame[:3])
            fvec = np.array(c.frame).reshape(3, 3).T @ f6[:3]      # world-frame force on b (minus on a)
            out.append({"a": a, "b": b, "normal": n, "force": float(f6[0]), "fvec": fvec, "dist": float(c.dist), "pos": np.array(c.pos)})
        return out

    def support_edges(self) -> dict[tuple[str, str], float]:
        """(upper, lower) -> normal force (N) for near-vertical contacts."""
        edges: dict[tuple[str, str], float] = {}
        for c in self.contacts():
            nz = c["normal"][2]
            if abs(nz) < 0.6:
                continue
            upper, lower = (c["b"], c["a"]) if nz > 0 else (c["a"], c["b"])
            if upper in (TABLE, FLOOR):
                continue
            edges[(upper, lower)] = edges.get((upper, lower), 0.0) + abs(c["force"]) * abs(nz)
        return edges

    def at_rest(self) -> bool:
        for oid in self.ids:
            if oid in self.removed or oid in self.pinned:
                continue
            lin, ang = self.speed(oid)
            if lin > config.REST_LIN_SPEED or ang > config.REST_ANG_SPEED:
                return False
        return True


def support_graph(world: World) -> dict[str, list[dict]]:
    """object -> list of {on, force_N, share} (share of the measured support load)."""
    edges = world.support_edges()
    graph: dict[str, list[dict]] = {oid: [] for oid in world.ids if oid not in world.removed}
    for (upper, lower), f in edges.items():
        if upper in graph:
            graph[upper].append({"on": lower, "force_N": round(f, 3)})
    for oid, lst in graph.items():
        tot = sum(e["force_N"] for e in lst) or 1.0
        for e in lst:
            e["share"] = round(e["force_N"] / tot, 3)
        lst.sort(key=lambda e: -e["force_N"])
    return graph


def dependents(graph: dict[str, list[dict]], root: str) -> list[str]:
    """Everything transitively resting on `root` according to the support graph."""
    out, frontier = [], [root]
    while frontier:
        cur = frontier.pop()
        for oid, lst in graph.items():
            if oid not in out and oid != root and any(e["on"] == cur for e in lst):
                out.append(oid)
                frontier.append(oid)
    return out


# ---------------------------------------------------------------------------
# Scene stabilisation (called right after perception)
# ---------------------------------------------------------------------------

def stabilize(scene: Scene, iters: int = 6) -> tuple[Scene, np.ndarray, list[str]]:
    """Remove reconstruction interpenetrations, then let the scene settle.

    Returns the corrected scene, the settled baseline qpos, and warnings for
    objects the model considers unstable as reconstructed."""
    warnings: list[str] = []
    objs = {o.id: o for o in scene.objects}
    original = {o.id: list(o.size) for o in scene.objects}
    for _ in range(iters):
        w = World(scene)
        changed = False
        for c in w.contacts():
            if c["dist"] > -0.0015:
                continue
            pen = -c["dist"]
            nz = c["normal"][2]
            a, b = c["a"], c["b"]
            if abs(nz) >= 0.6:       # stacked: trim the upper object's bottom
                upper = b if nz > 0 else a
                if upper in objs:
                    o = objs[upper]
                    trim = min(pen + 0.0005, o.size[2] - max(0.01, 0.4 * original[upper][2]))
                    if trim > 0:
                        o.size[2] -= trim
                        o.pos[2] += trim / 2
                        changed = True
            else:                     # side-by-side: shrink footprints along the normal (local axes)
                touches_fixed = a in (TABLE, FLOOR) or b in (TABLE, FLOOR)
                for oid in (a, b):
                    if oid not in objs:
                        continue
                    o = objs[oid]
                    c_, s_ = math.cos(o.yaw), math.sin(o.yaw)
                    nx, ny = c["normal"][0], c["normal"][1]
                    lx, ly = abs(c_ * nx + s_ * ny), abs(-s_ * nx + c_ * ny)
                    k = 1.0 if touches_fixed else 0.5
                    orig = original[oid]
                    o.size[0] = max(0.7 * orig[0], 0.012, o.size[0] - pen * k * lx - 0.0005)
                    o.size[1] = max(0.7 * orig[1], 0.012, o.size[1] - pen * k * ly - 0.0005)
                    if o.shape == "cylinder":
                        o.size[0] = o.size[1] = min(o.size[0], o.size[1])
                    changed = True
        if not changed:
            break
    for o in scene.objects:
        if o.size[0] < original[o.id][0] * 0.95 or o.size[1] < original[o.id][1] * 0.95:
            o.notes.append(f"footprint shrunk to {o.size[0] * 100:.1f} x {o.size[1] * 100:.1f} cm to resolve overlap")
    world = World(scene)
    start = {oid: world.pose(oid) for oid in world.ids}
    for _ in range(int(config.SETTLE_SECONDS / config.SIM_TIMESTEP)):
        world.step()
    # extra settle if still moving, bounded
    for _ in range(int(1.0 / config.SIM_TIMESTEP)):
        if world.at_rest():
            break
        world.step()
    for o in scene.objects:
        p0, p1 = start[o.id], world.pose(o.id)
        moved = float(np.linalg.norm(p1[:3] - p0[:3]))
        rot = _quat_angle_deg(p0[3:], p1[3:])
        o.stability.moved_m = round(moved, 4)
        o.stability.rotated_deg = round(rot, 2)
        o.stability.stable = moved < 0.025 and rot < 12
        if not o.stability.stable:
            warnings.append(f"{o.label}: unstable as reconstructed (settled {moved * 100:.1f} cm / {rot:.0f} deg). "
                            "Interventions start from its settled pose; adjust its size or support if this is wrong.")
    graph = support_graph(world)
    for o in scene.objects:
        o.supported_by = [e["on"] for e in graph.get(o.id, [])]
        o.supporting = [oid for oid, lst in graph.items() if any(e["on"] == o.id for e in lst)]
    world_q = world.qpos()
    world.d.qvel[:] = 0
    return scene, world_q, warnings


def _quat_angle_deg(q0: np.ndarray, q1: np.ndarray) -> float:
    dot = abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1))))
    return float(np.degrees(2 * np.arccos(min(1.0, dot))))


# ---------------------------------------------------------------------------
# Simulation with interventions
# ---------------------------------------------------------------------------

@dataclass
class SegmentLog:
    index: int
    intervention: dict
    t_start: float
    t_end: float = 0.0
    events: list[dict] = field(default_factory=list)
    causal_edges: list[dict] = field(default_factory=list)
    summary: dict[str, dict] = field(default_factory=dict)
    support_before: dict = field(default_factory=dict)
    support_after: dict = field(default_factory=dict)
    predicted_dependents: list[str] = field(default_factory=list)


def _apply(world: World, iv: Intervention, graph_before: dict | None = None) -> dict[str, Any]:
    """Apply an intervention at segment start. Returns push parameters if any."""
    p = iv.params
    if iv.type == "remove":
        world.remove(iv.target)
    elif iv.type == "set_mass":
        world.set_mass(iv.target, float(p["mass"]))
    elif iv.type == "set_friction":
        world.set_friction(iv.target, float(p["friction"]))
    elif iv.type == "set_static":
        (world.pin if p.get("static", True) else world.unpin)(iv.target)
    elif iv.type == "set_gravity":
        world.set_gravity(float(p["g"]))
    elif iv.type == "spawn":
        if p.get("keep"):           # restored from a checkpoint: it is already where the last run left it
            world._set_tangible(iv.target, True)
            world.removed.discard(iv.target)
            mujoco.mj_forward(world.m, world.d)
            if p.get("static"):
                world.pin(iv.target)
        else:
            world.spawn(iv.target, p["pos"], float(p.get("yaw", 0.0)), static=bool(p.get("static")), pitch=float(p.get("pitch", 0.0)))
    elif iv.type == "hand":
        # the user's real hand: kinematic bones follow the recorded joints; contacts do the rest
        ups = list(p.get("updates") or [])
        end = float(ups[-1][0]) if ups else 0.0
        world.set_hand_track(ups + [[end + 1e-3, None]], float(world.d.time))     # then the hand leaves
        return {"drive_until": float(world.d.time) + end + 0.05}
    elif iv.type == "drive":
        goal = p.get("goal")
        dur = float(p.get("duration", 3.0))
        spd = p.get("speed", 0.5)
        world.drive[iv.target] = {"speed": None if spd in (None, "max") else float(np.clip(spd, -3.0, 3.0)),
                                  "goal": goal, "t_end": world.d.time + dur,
                                  "stop_at_goal": bool(p.get("stop_at_goal")), "stop_dist": float(p.get("stop_dist", 0.12))}
        return {"drive_until": world.d.time + dur}
    elif iv.type == "propel":
        world.unpin(iv.target)
        dur = float(p.get("duration", 3.0))
        world.propel[iv.target] = {"target": p.get("toward"), "point": p.get("point"), "direction": p.get("direction"),
                                   "force": float(p["force"]), "t_end": world.d.time + dur, "planar": bool(p.get("planar", True)),
                                   "top_speed": p.get("top_speed"), "lift": bool(p.get("lift")), "hand": bool(p.get("hand"))}
        return {"drive_until": world.d.time + dur}
    elif iv.type == "connect":
        world.connect(p["link"])
    elif iv.type == "hold":
        world.pin(iv.target)
    elif iv.type == "place":
        world.place(iv.target, p["pos"], bool(p.get("hold", True)), clear=bool(p.get("clear")))
    elif iv.type == "release":
        world.unpin(iv.target)
    elif iv.type == "spin":
        a = world.vadr[iv.target]
        world.unpin(iv.target)
        R = world.d.xmat[world.body[iv.target]].reshape(3, 3)
        world.d.qvel[a + 3:a + 6] = R.T @ np.array(p["omega"], float)      # world -> body frame
    elif iv.type == "launch":
        v = np.array(p["velocity"], float)
        a = world.vadr[iv.target]
        world.unpin(iv.target)
        world.d.qvel[a:a + 3] = v
        if "spin" in p:
            world.d.qvel[a + 3:a + 6] = np.array(p["spin"], float)
    elif iv.type == "push":
        world.unpin(iv.target)
        direction = np.array(p["direction"], float)
        direction[2] = 0
        direction /= max(1e-9, np.linalg.norm(direction))
        dv = float(np.clip(p.get("speed", 0.8), 0.05, 5.0))
        point = np.array(p.get("point") or world.pose(iv.target)[:3], float)
        # "speed" means: give the target AND whatever rides on it this speed, so
        # the force covers the carried load and table friction under it.
        load = dependents(graph_before or {}, iv.target)
        m_eff = world.mass(iv.target) + sum(world.mass(o) for o in load if o in world.body)
        mu = float(world.m.geom_friction[world.geom[iv.target], 0])
        g = float(-world.m.opt.gravity[2])
        T = 0.1
        force = m_eff * dv / T + mu * m_eff * g
        return {"direction": direction, "dv": dv, "point": point, "duration": T, "force": force, "m_eff": m_eff}
    return {}


def world_inputs(interventions: list[Intervention], env: dict, lib: dict) -> tuple[list[dict], list[dict]]:
    """The added objects and links an intervention history implies (the same for a replay and a live session)."""
    spawns, links = [], []
    for iv in interventions:
        if iv.type == "spawn":
            if iv.params["asset"] == "proxy":
                px = iv.params["proxy"]
                a = {"id": "proxy", "name": iv.params.get("label", "object"), "kind": "proxy", "shape": px["shape"],
                     "dims": px["dims"], "mass": px["mass"], "friction": px.get("friction", 0.5),
                     "bounciness": px.get("bounciness", 0.3), "color": px.get("color", "#94a3b8"), "material": px.get("material"),
                     "archetype": px.get("archetype", "generic"), "surface": bool(px.get("surface"))}
            else:
                a = lib[iv.params["asset"]]
                if iv.params.get("collision") == "sphere":
                    a = {**a, "collision": "sphere"}       # a ball rolls on a true sphere, not on hull facets
            spawns.append({"sid": iv.target, "asset": a, "k": len(spawns), "mass": iv.params.get("mass", a["mass"]),
                           "fluid": env.get("density", 0) > 10, "dt": env.get("timestep", config.SIM_TIMESTEP),
                           "bounciness": iv.params.get("bounciness"), "volume": iv.params.get("volume"), "drag": iv.params.get("drag")})
        elif iv.type == "connect":
            links.append({"id": iv.params["link"], "a": iv.params["a"], "b": iv.params["b"], "kind": iv.params.get("kind", "rope"),
                          "length": float(iv.params.get("length", 1.0)), "stiffness": float(iv.params.get("stiffness", 50.0)),
                          "damping": float(iv.params.get("damping", 0.0))})
    return spawns, links


def simulate(scene: Scene, baseline_qpos: np.ndarray, interventions: list[Intervention],
             gravity: float = 9.81, env: dict | None = None, max_segment_s: float | None = None) -> dict:
    from .. import assets as asset_lib
    lib = asset_lib.manifest()
    env = env or {}
    spawns, links = world_inputs(interventions, env, lib)
    hand_ivs = [iv for iv in interventions if iv.type == "hand"]
    if hand_ivs and not env.get("hand"):
        env = {**env, "hand": hand_ivs[0].params["bones"]}
    world = World(scene, gravity, spawns, env, links)
    if env.get("max_segment_s"):
        max_segment_s = env["max_segment_s"]
    world.set_qpos(baseline_qpos)
    labels = {o.id: o.label for o in scene.objects}
    for sp in spawns:
        labels[sp["sid"]] = next((iv.params.get("label") for iv in interventions if iv.target == sp["sid"] and iv.type == "spawn"), None) or sp["asset"]["name"]
    vehicles = [sp["sid"] for sp in spawns if sp["asset"]["kind"] == "vehicle"]
    wheel_frames: dict[str, list] = {sid: [] for sid in vehicles}
    kin: dict[str, dict[str, list]] = {oid: {"v": [], "F_contact": [], "F_objects": [], "KE": [], "PE": [], "w": [], "c": []}
                                       for oid in [o.id for o in scene.objects] + [sp["sid"] for sp in spawns]}
    pair_forces: dict[str, list] = {}        # "a|b" -> [[frame, mean |F| over the frame, peak |F|, impulse N*s], ...]
    pair_acc: dict[str, list] = {}           # per-step accumulation between recorded frames: [impulse, peak]
    tensions: dict[str, list] = {l["id"]: [] for l in links}
    spawned_at: dict[str, float] = {}
    dt = float(env.get("timestep", config.SIM_TIMESTEP)) if env else config.SIM_TIMESTEP
    frame_dt = 1.0 / config.RECORD_FPS
    ev_every = max(1, int(round(0.004 / dt)))       # ~250 Hz event sampling whatever the step
    frames: list[list[list[float]]] = []
    t = 0.0
    next_frame = 0.0
    removed_at: dict[str, float] = {}
    segments: list[SegmentLog] = []
    ev_counter = [0]

    def record():
        row = []
        for oid in world.ids:
            q = world.pose(oid)
            row.append([round(float(v), 5) for v in (q[0], q[1], q[2], q[4], q[5], q[6], q[3])])   # xyz + quat xyzw (three.js)
        frames.append(row)
        for sid in vehicles:
            wheel_frames[sid].append([round(x, 4) for x in world.wheel_state(sid)])
        fnet = {oid: np.zeros(3) for oid in world.ids}
        fobj = {oid: 0.0 for oid in world.ids}
        fi = len(frames) - 1
        for lid in tensions:
            L_, T_ = world.tendon_state(lid)
            tensions[lid].append([round(L_, 4), round(T_, 3)])
        for c in world.contacts():
            for oid, sgn in ((c["b"], 1.0), (c["a"], -1.0)):
                if oid in fnet:
                    fnet[oid] += sgn * c["fvec"]
                    other = c["a"] if sgn > 0 else c["b"]
                    if other not in (TABLE, FLOOR):
                        fobj[oid] += float(np.linalg.norm(c["fvec"]))
        for oid in world.ids:
            k = kin[oid]
            if oid in world.removed:
                for key in k:
                    k[key].append(None)
                continue
            v, ke, pe = world.kinematics(oid)
            k["v"].append([round(float(x), 4) for x in v])
            k["F_contact"].append(round(float(np.linalg.norm(fnet[oid])), 4))
            k["F_objects"].append(round(fobj[oid], 4))
            k["KE"].append(round(ke, 6))
            k["PE"].append(round(pe, 5))
            a_ = world.vadr[oid]
            R_ = world.d.xmat[world.body[oid]].reshape(3, 3)
            k["w"].append([round(float(x), 3) for x in R_ @ world.d.qvel[a_ + 3:a_ + 6]])
            k["c"].append([round(float(x), 5) for x in world.d.xipos[world.body[oid]]])
        for key, (imp, peak) in pair_acc.items():
            if imp > 0:
                pair_forces.setdefault(key, []).append([fi, round(imp / frame_dt, 4), round(peak, 4), round(imp, 6)])
        pair_acc.clear()

    record()
    next_frame += frame_dt

    # consecutive interventions sharing a "group" happen at the same instant, in one segment
    # (a row of dominoes appearing; both cars of a collision starting together)
    groups: list[list[Intervention]] = []
    for iv in interventions:
        g = iv.params.get("group")
        if g and groups and groups[-1][0].params.get("group") == g:
            groups[-1].append(iv)
        else:
            groups.append([iv])

    for k, grp in enumerate(groups):
        iv = grp[0]
        dump = iv.model_dump()
        if len(grp) > 1:
            dump["params"] = {**dump["params"], "group_size": len(grp), "label": f'{len(grp)} x {dump["params"].get("label", "")}'}
        seg = SegmentLog(index=k, intervention=dump, t_start=round(t, 4))
        graph_before = support_graph(world)
        seg.support_before = graph_before
        if iv.target and iv.type in ("remove", "push"):
            seg.predicted_dependents = dependents(graph_before, iv.target)
        start_pose = {oid: world.pose(oid) for oid in world.ids}

        def emit(kind: str, subject: str | None, other: str | None = None, **data):
            ev_counter[0] += 1
            e = {"id": f"e{ev_counter[0]}", "t": round(t, 3), "type": kind, "subject": subject, "other": other}
            e.update({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in data.items()})
            seg.events.append(e)
            return e["id"]

        intervention_ev = emit("intervention", iv.target, None, action=iv.type, params=iv.params)
        push = _apply(world, iv, graph_before)
        for extra in grp[1:]:
            r_extra = _apply(world, extra, graph_before) or {}
            if extra.type == "spawn":
                spawned_at[extra.target] = t
                start_pose[extra.target] = world.pose(extra.target)
            if "drive_until" in r_extra:
                push = dict(push or {})
                push["drive_until"] = max(push.get("drive_until", 0), r_extra["drive_until"])
        if iv.type == "remove":
            removed_at[iv.target] = t
        if iv.type == "spawn":
            spawned_at[iv.target] = t
            start_pose[iv.target] = world.pose(iv.target)
        drive_until = push.pop("drive_until", None) if push else None
        if drive_until is not None:
            push = {}
        mujoco.mj_forward(world.m, world.d)

        prev_support = set(world.support_edges().keys()) if iv.type != "remove" else {
            e for e in world.support_edges().keys()}
        # support pairs that existed before the intervention (removal clears them instantly)
        prev_support |= {(u, l) for u, lst in graph_before.items() for l in [e["on"] for e in lst]}
        prev_contacts: set[frozenset] = set()
        moving: dict[str, bool] = {oid: False for oid in world.ids}
        rest_since: dict[str, float] = {}
        cause_of: dict[str, str] = {}
        toppled: set[str] = set()
        off_table: set[str] = set()
        lost_recent: dict[str, tuple[str, float, str]] = {}   # obj -> (supporter, t, event id)
        if iv.type == "remove":
            for (u, l) in list(prev_support):
                if l == iv.target:
                    eid = emit("lost_support", u, l, reason="supporter removed")
                    lost_recent[u] = (l, t, eid)
            prev_support = {(u, l) for (u, l) in prev_support if l != iv.target and u != iv.target}
        absent_count: dict[tuple, int] = {}
        last_lost: dict[tuple, float] = {}
        initial_tilt = {oid: world.tilt_deg(oid) for oid in world.ids if oid not in world.removed}
        for g_iv in grp:
            if g_iv.type == "spawn":
                initial_tilt[g_iv.target] = world.tilt_deg(g_iv.target)
        impact_log: dict[str, tuple[str, float, str]] = {}   # obj -> (striker, t, event id)
        recent_touch: dict[str, tuple[str, float, str]] = {}  # obj -> (moving neighbour, t, kind)
        last_touch_pair: dict[frozenset, float] = {}          # pair -> last time in contact
        pending: dict[str, tuple[float, dict]] = {}           # unexplained starts awaiting evidence
        f6_step = np.zeros(6)
        fine_touch: dict[tuple[str, str], float] = {}         # (me, other) -> last time touching (every step)
        seg_t0 = t
        rest_hold = 0.0
        n = 0
        max_steps = int((max_segment_s or config.MAX_SEGMENT_SECONDS) / dt)
        if any(x.type == "hand" for x in grp) and drive_until is not None:   # a hand session lasts as long as it was recorded
            max_steps = max(max_steps, int((drive_until - t + 3.0) / dt))
        max_speed = {oid: 0.0 for oid in world.ids}

        while n < max_steps:
            if push and t - seg_t0 < push["duration"]:
                b = world.body[iv.target]
                F = push["direction"] * push["force"]
                com = world.d.xipos[b]
                world.d.xfrc_applied[b, :3] = F
                world.d.xfrc_applied[b, 3:] = np.cross(push["point"] - com, F)
            elif push:
                world.d.xfrc_applied[world.body[iv.target]] = 0
            world.step()
            t += dt
            n += 1
            for oa, ob, P_ in getattr(world, "impulses", []):          # restitution impulses count toward the impact
                if oa not in (TABLE, FLOOR) and ob not in (TABLE, FLOOR):
                    pair_acc.setdefault("|".join(sorted((oa, ob))), [0.0, 0.0])[0] += P_
            for a_, b_ in world.touching_pairs():
                fine_touch[(a_, b_)] = t
                fine_touch[(b_, a_)] = t
            # every step: object-object contact force -> impulse and peak (impacts last ~5-20 ms)
            for i_c in range(world.d.ncon):
                c_ = world.d.contact[i_c]
                oa, ob = world.geom_owner.get(int(c_.geom1)), world.geom_owner.get(int(c_.geom2))
                if oa and ob and oa != ob and oa not in (TABLE, FLOOR) and ob not in (TABLE, FLOOR):
                    mujoco.mj_contactForce(world.m, world.d, i_c, f6_step)
                    Fm = float(np.linalg.norm(f6_step[:3]))
                    acc_ = pair_acc.setdefault("|".join(sorted((oa, ob))), [0.0, 0.0])
                    acc_[0] += Fm * dt
                    acc_[1] = max(acc_[1], Fm)
            if t >= next_frame - 1e-9:
                record()
                next_frame += frame_dt
            if n % ev_every:
                continue

            contacts = world.contacts()
            supp = {}
            pairs_now: set[frozenset] = set()
            for c in contacts:
                pairs_now.add(frozenset((c["a"], c["b"])))
                nz = c["normal"][2]
                if abs(nz) >= 0.6:
                    upper, lower = (c["b"], c["a"]) if nz > 0 else (c["a"], c["b"])
                    if upper not in (TABLE, FLOOR):
                        supp[(upper, lower)] = True
            # lost support (debounced: absent >= 40 ms; one report per pair per 300 ms)
            for pair in list(prev_support):
                if pair in supp:
                    absent_count[pair] = 0
                    continue
                absent_count[pair] = absent_count.get(pair, 0) + 1
                if absent_count[pair] * ev_every * dt >= 0.04:
                    u, l = pair
                    if t - last_lost.get(pair, -1.0) > 0.3:
                        eid = emit("lost_support", u, l)
                        lost_recent[u] = (l, t, eid)
                    last_lost[pair] = t
                    prev_support.discard(pair)
            for pair in supp:
                if pair not in prev_support:
                    prev_support.add(pair)
                    absent_count[pair] = 0
                    u, l = pair
                    if l == FLOOR and u not in off_table:
                        off_table.add(u)
                        emit("hit_floor", u, FLOOR)
            # impacts: genuinely new contacts (apart >= 120 ms) with closing speed
            for pair in pairs_now - prev_contacts:
                if t - last_touch_pair.get(pair, -1.0) < 0.12:
                    continue
                a, b = tuple(pair) if len(pair) == 2 else (next(iter(pair)), next(iter(pair)))
                va = world.d.qvel[world.vadr[a]:world.vadr[a] + 3] if a in world.vadr else np.zeros(3)
                vb = world.d.qvel[world.vadr[b]:world.vadr[b] + 3] if b in world.vadr else np.zeros(3)
                rel = float(np.linalg.norm(va - vb))
                if rel > 0.08:
                    # the faster body is the striker
                    striker, struck = (a, b) if np.linalg.norm(va) >= np.linalg.norm(vb) else (b, a)
                    eid = emit("impact", striker, struck, speed=rel)
                    if struck in world.vadr:
                        impact_log[struck] = (striker, t, eid)
            prev_contacts = pairs_now
            for pair in pairs_now:
                last_touch_pair[pair] = t
            for c in contacts:
                for me, other in ((c["a"], c["b"]), (c["b"], c["a"])):
                    if me in moving and other in moving and moving[other] and not moving[me]:
                        nz = c["normal"][2]
                        upper = c["b"] if nz > 0 else c["a"]
                        kind = "carried_by" if abs(nz) >= 0.6 and upper == me else "pushed_by"
                        recent_touch[me] = (other, t, kind)

            for oid, (t0, edge) in list(pending.items()):
                hit = impact_log.get(oid) if oid in impact_log and impact_log[oid][1] >= t0 - 0.02 else None
                touch = recent_touch.get(oid)
                if hit:
                    edge.update(cause=hit[0], kind="struck_by"); edge["evidence"].insert(0, hit[2])
                elif touch is None:
                    for (me, other), tt in fine_touch.items():
                        if me == oid and tt >= t0 - 0.03 and other in moving and (moving[other] or max_speed.get(other, 0) > 0.05):
                            touch = (other, tt, "struck_by")
                            break
                if touch is None and not hit:
                    for c in contacts:
                        other = c["b"] if c["a"] == oid else c["a"] if c["b"] == oid else None
                        if other in moving and moving[other]:
                            touch = (other, t, "pushed_by")
                            break
                if hit or touch:
                    if not hit:
                        edge.update(cause=touch[0], kind=touch[2])
                    cause_of[oid] = edge["cause"]
                    for e in seg.events:
                        if e["type"] == "started_moving" and e["subject"] == oid and e["other"] == "unknown":
                            e["other"], e["cause_kind"] = edge["cause"], edge["kind"]
                    del pending[oid]
                elif t - t0 > 0.08:
                    del pending[oid]
            for oid in world.ids:
                if oid in world.removed or oid in world.pinned:
                    continue
                lin, ang = world.speed(oid)
                max_speed[oid] = max(max_speed[oid], lin)
                is_moving = lin > 0.05 or ang > 0.6
                if is_moving and not moving[oid]:
                    moving[oid] = True
                    rest_since.pop(oid, None)
                    if oid not in cause_of:
                        cause, kind, evid = _attribute(oid, t, seg_t0, iv, push, lost_recent, impact_log,
                                                       contacts, moving, world, intervention_ev, recent_touch)
                        cause_of[oid] = cause
                        eid = emit("started_moving", oid, cause, cause_kind=kind)
                        edge = {"cause": cause, "effect": oid, "kind": kind, "t": round(t, 3),
                                "evidence": [x for x in (evid, eid) if x]}
                        seg.causal_edges.append(edge)
                        if cause == "unknown":
                            pending[oid] = (t, edge)
                elif not is_moving and moving[oid]:
                    rest_since.setdefault(oid, t)
                    if t - rest_since[oid] > 0.25:
                        moving[oid] = False
                        emit("came_to_rest", oid)
                tilt = world.tilt_deg(oid)
                if oid not in toppled and initial_tilt.get(oid, 0) < 25 and tilt > 50:
                    toppled.add(oid)
                    emit("toppled", oid, None, tilt_deg=tilt)
            if world.at_rest() and t - seg_t0 > 0.3 and (drive_until is None or t > drive_until):
                rest_hold += dt * ev_every
                if rest_hold >= config.REST_HOLD_SECONDS:
                    break
            else:
                rest_hold = 0.0

        world.d.xfrc_applied[:] = 0
        seg.t_end = round(t, 4)
        seg.support_after = support_graph(world)
        for oid in world.ids:
            if oid in world.removed:
                continue
            p0, p1 = start_pose[oid], world.pose(oid)
            seg.summary[oid] = {
                "label": labels[oid],
                "displacement_m": round(float(np.linalg.norm(p1[:3] - p0[:3])), 4),
                "drop_m": round(float(p0[2] - p1[2]), 4),
                "rotation_deg": round(_quat_angle_deg(p0[3:], p1[3:]), 1),
                "max_speed_mps": round(max_speed[oid], 3),
                "toppled": oid in toppled,
                "fell_off_table": bool(oid in off_table or p1[2] < -0.05),
                "moved": bool(np.linalg.norm(p1[:3] - p0[:3]) > 0.01 or _quat_angle_deg(p0[3:], p1[3:]) > 5),
                "cause": cause_of.get(oid),
            }
        segments.append(seg)

    # objects that visibly moved anywhere in the trajectory need re-rendering
    F = np.array(frames)            # [T, N, 7]
    moved_ids = []
    for i, oid in enumerate(world.ids):
        if oid in removed_at or oid in world.spawns:
            continue
        disp = np.linalg.norm(F[:, i, :3] - F[0, i, :3], axis=1).max()
        qdots = np.abs(np.einsum("tj,j->t", F[:, i, 3:], F[0, i, 3:]))
        rot = np.degrees(2 * np.arccos(np.clip(qdots.min(), -1, 1)))
        if disp > 0.004 or rot > 2.5:
            moved_ids.append(oid)
    return {
        "final_qpos": [float(x) for x in world.d.qpos],          # exact end state: a later run can start from here
        "final_pinned": sorted(world.pinned),
        "fps": config.RECORD_FPS,
        "duration": round(t, 3),
        "object_ids": world.ids,
        "frames": frames,
        "removed_at": {k: round(v, 3) for k, v in removed_at.items()},
        "moved_ids": moved_ids,
        "segments": [s.__dict__ for s in segments],
        "gravity": float(-world.m.opt.gravity[2]),
        "spawned": {sp["sid"]: {"asset": sp["asset"]["id"], "label": labels[sp["sid"]], "t": round(spawned_at.get(sp["sid"], 0.0), 3),
                                "kind": sp["asset"]["kind"],
                                **({"proxy": {k: sp["asset"].get(k) for k in ("shape", "dims", "color", "material", "archetype")}}
                                   if sp["asset"]["kind"] == "proxy" else {})} for sp in spawns},
        "wheels": wheel_frames,
        "labels": labels,
        "kinematics": kin,
        "impacts": getattr(world, "impact_log", []),
        "masses": {oid: round(float(world.m.body_subtreemass[world.body[oid]]), 6) for oid in world.ids},
        "pair_forces": pair_forces,
        "tensions": tensions,
        "links": [{k: v for k, v in l.items()} for l in links],
        "env": env,
    }


def _attribute(oid, t, seg_t0, iv, push, lost_recent, impact_log, contacts, moving, world, intervention_ev, recent_touch):
    """Why did `oid` start moving? Returns (cause, kind, evidence_event_id)."""
    if push and iv.target == oid and t - seg_t0 < push["duration"] + 0.05:
        return "intervention", "pushed", intervention_ev
    if iv.target == oid and iv.type in ("drive", "launch", "spawn"):
        return "intervention", {"drive": "driven", "launch": "launched", "spawn": "released"}[iv.type], intervention_ev
    if oid in lost_recent and t - lost_recent[oid][1] < 0.4:
        sup, _, eid = lost_recent[oid]
        if iv.type == "remove" and sup == iv.target:
            return "intervention", "support_removed", eid
        return sup, "lost_support", eid
    if oid in impact_log and t - impact_log[oid][1] < 0.15:
        striker, _, eid = impact_log[oid]
        return striker, "struck_by", eid
    for c in contacts:
        other = c["b"] if c["a"] == oid else c["a"] if c["b"] == oid else None
        if other and other in moving and moving[other]:
            nz = c["normal"][2]
            upper = c["b"] if nz > 0 else c["a"]
            kind = "carried_by" if abs(nz) >= 0.6 and upper == oid else "pushed_by"
            return other, kind, None
    if oid in recent_touch and t - recent_touch[oid][1] < 0.15:
        other, _, kind = recent_touch[oid]
        return other, kind, None
    if iv.target == oid and iv.type in ("set_friction", "set_mass", "set_static"):
        return "intervention", f"property_change:{iv.type}", intervention_ev
    if iv.type == "set_gravity":
        return "intervention", "gravity_change", intervention_ev
    return "unknown", "unexplained", None
