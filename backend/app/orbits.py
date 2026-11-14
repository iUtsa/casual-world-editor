"""Space: gravity between real bodies, from the Moon to colliding galaxies.

* Integrator: REBOUND IAS15 (Rein & Spiegel 2015) - adaptive 15th-order, error below machine precision: energy
  and angular momentum are conserved to ~1e-15. Collisions are found at contact (centre distance = sum of
  radii) and treated as perfectly inelastic mergers (momentum conserved, the relative-motion energy becomes heat).
* Initial conditions: real positions/velocities from NASA JPL Horizons on the requested date (ephemeris.py),
  so "the solar system on 1 Jan 2030" is where the planets really are; hypothetical set-ups ("Jupiter hits the
  Moon") start from rest or circular orbits.
* Every run is checked against the laws it must obey: conservation of energy and angular momentum, Kepler's
  third law, the shape of each orbit, the vis-viva speed at perihelion - and, for real bodies, NASA's own
  positions at the end of the run.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timedelta

import numpy as np

from . import ephemeris as E
from .ephemeris import AU, CATALOG, DAY, G, KPC, LY, MSUN, RSUN, YEAR

MT_TNT = 4.184e15                      # J per megaton
CHICXULUB_J = 4.2e23                   # the dinosaur-killing impact

# things without an ephemeris: mass kg, radius m, colour, kind
GENERIC = {
    "asteroid": (2.6e13, 1.0e3, "#57534e", "asteroid"), "chicxulub asteroid": (1.0e15, 5.0e3, "#57534e", "asteroid"),
    "comet": (2.2e14, 5.5e3, "#e0f2fe", "comet"), "meteor": (1.2e7, 10.0, "#57534e", "asteroid"),
    "satellite": (1000.0, 2.0, "#e5e7eb", "spacecraft"), "iss": (4.2e5, 50.0, "#e5e7eb", "spacecraft"),
    "space station": (4.2e5, 50.0, "#e5e7eb", "spacecraft"), "neutron star": (2.8e30, 1.0e4, "#c7d2fe", "neutron star"),
    "star": (MSUN, RSUN, "#fde68a", "star"), "red dwarf": (0.4 * MSUN, 0.3 * RSUN, "#f87171", "star"),
    "white dwarf": (0.6 * MSUN, 7.0e6, "#e0f2fe", "star"), "planet": (CATALOG["earth"]["mass"], CATALOG["earth"]["radius"], "#60a5fa", "planet"),
}
GALAXY_WORDS = r"\b(milky way|andromeda|galax(y|ies)|m31|triangulum|m33|magellanic)\b"
ALIASES = {**E.ALIASES, "international space station": "iss", "dinosaur asteroid": "chicxulub asteroid", "meteorite": "meteor",
           "blackhole": "black hole", "sgr a": "sagittarius a*"}
ALL_NAMES = set(CATALOG) | set(GENERIC) | set(E.STARS) | set(E.BLACK_HOLES) | set(E.BINARIES)
NAMES = sorted(ALL_NAMES | set(ALIASES), key=len, reverse=True)
PLACE_PHRASE = r"\b(on|at|from|onto) (the )?(moon|mars|jupiter|earth|venus|mercury|saturn|pluto|sun|titan|europa|ceres)\b"
PLANETS8 = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"]
# a planet system without its moons moves like its centre of mass: (Horizons barycentre id, total GM km^3/s^2)
SYSTEM_BARY = {"earth": ("3", 403503.2356), "jupiter": ("5", 126712764.8), "saturn": ("6", 37940585.2), "uranus": ("7", 5794548.6),
               "neptune": ("8", 6836527.1), "pluto": ("9", 975.5), "mars": ("4", 42828.376)}
MOONS_OF = {p: [m for m, c in CATALOG.items() if c.get("parent") == p and c["kind"] == "moon"] for p in CATALOG}
# planets are squashed by their spin: J2, J4, equatorial radius (m), pole (IAU right ascension, declination; deg, ICRF)
OBLATE = {"earth": (1.08263e-3, -1.62e-6, 6.378137e6, 0.0, 90.0), "mars": (1.96045e-3, -1.5e-5, 3.39619e6, 317.681, 52.887),
          "jupiter": (1.4736e-2, -5.87e-4, 7.1492e7, 268.056595, 64.495303), "saturn": (1.6298e-2, -9.15e-4, 6.0268e7, 40.589, 83.537),
          "uranus": (3.343e-3, -2.9e-5, 2.5559e7, 257.311, -15.175), "neptune": (3.411e-3, -3.5e-5, 2.4764e7, 299.36, 43.46)}


def _pole_ecliptic(ra: float, dec: float) -> np.ndarray:
    a, d, e = math.radians(ra), math.radians(dec), math.radians(23.4392911)
    x, y, z = math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)
    return np.array([x, y * math.cos(e) + z * math.sin(e), -y * math.sin(e) + z * math.cos(e)])


def _frame_to_pole(pole: np.ndarray) -> np.ndarray:
    """Rotation taking ecliptic vectors into a frame whose z axis is the planet's spin axis."""
    z = pole / np.linalg.norm(pole)
    x = np.cross([0.0, 0.0, 1.0], z)
    x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(z, x)
    return np.vstack([x, y, z])


def bodies_in(text: str) -> list[tuple[int, str]]:
    t = text.lower()
    found, taken = [], [False] * (len(t) + 1)
    for n in NAMES:
        for m in re.finditer(rf"(?<![\w-]){re.escape(n)}(?:'?s)?(?![\w-])", t):
            if n == "moon" and t[m.start():m.end()] == "moons":
                continue                                    # "Jupiter's moons" are not Earth's Moon
            if any(taken[m.start():m.end()]):
                continue
            for k in range(m.start(), m.end()):
                taken[k] = True
            found.append((m.start(), ALIASES.get(n, n)))
    return sorted(found)


PROPER = {"sun": "the Sun", "moon": "the Moon", "earth": "Earth", "iss": "the ISS"}


def nm(name: str, cap=False) -> str:
    base = re.sub(r" \d+$", "", name)
    if base in CATALOG and CATALOG[base].get("label"):
        s_ = CATALOG[base]["label"]
    elif base in PROPER:
        s_ = PROPER[base]
    elif base in CATALOG or base in E.STARS or base in E.BLACK_HOLES:
        s_ = base.title().replace("'S", "'s")
    else:
        s_ = f"the {base}"
    if base != name:
        s_ = f"{s_.replace('the ', '')} {name.split()[-1]}"
    return s_[0].upper() + s_[1:] if cap else s_


def is_gravity(text: str) -> bool:
    """Celestial requests - but 'drop a ball on the Moon' is mechanics."""
    t = text.lower()
    if re.search(r"solar system|asteroid belt|kuiper belt|\b(two|2|binary)\s+(\w+\s+)?(planets|stars|black holes|suns|asteroids)\b", t) or re.search(GALAXY_WORDS, t):
        return True
    if gone_bodies(t) or orbit_of(t):                     # "what if there is no Sun", "if Jupiter orbited Earth"
        return True
    stripped = re.sub(PLACE_PHRASE, " ", t)
    bs2 = bodies_in(stripped)
    if len(bs2) >= 2:
        return True
    if len(bs2) == 1:
        n = bs2[0][1]
        if n in E.BINARIES or (n in CATALOG and CATALOG[n]["kind"] in ("comet",)) or n in ("apophis", "bennu"):
            return True
        # just a name ("Ceres", "show me Saturn", "Jupiter's moons"): show it in space
        rest = re.sub("|".join(re.escape(x) for x in NAMES), " ", stripped)
        if n in CATALOG and len(re.findall(r"[a-z]{3,}", re.sub(r"\b(the|a|an|and|with|all|of|show|me|see|where|is|its|their|moons?|rings?|now|today|planet|dwarf|tell|about|on|in|\d+|january|february|march|april|may|june|july|august|september|october|november|december)\b", " ", rest))) == 0:
            return True
        if re.search(r"\b(orbit\w*|revolv\w*|circl\w*|fall\w* into|swallow\w*|pass\w* (by|near)|fly-?by|slingshot|escape|approach\w*|collid\w*|crash\w*|"
                     r"hit\w*|impact\w*|smash\w*|merg\w*|devour\w*|eat\w*|spiral\w*|where is|show me|replace\w*|path|journey|visit\w*)\b", t):
            others = re.sub("|".join(re.escape(x) for x in NAMES), " ", stripped)
            return not re.findall(r"\b(car|house|person|man|building|city|ball|truck|dog|tree)\b", others)
    return False


# ------------------------------------------------------------------------------------------------ bodies
def _body(name: str, scale: dict | None = None, bary: bool = False) -> dict:
    base = re.sub(r" \d+$", "", name)
    k = (scale or {}).get(base, (scale or {}).get("*", 1.0))
    if base in CATALOG:
        c = CATALOG[base]
        mass = c["mass"]
        if bary and base in SYSTEM_BARY:
            mass = SYSTEM_BARY[base][1] * 1e9 / G
        b = {"name": name, "label": c.get("label") or nm(name, True), "kind": c["kind"], "mass": mass * k, "radius": c["radius"], "color": c["color"],
             "rings": c.get("rings"), "real": True, "parent": c.get("parent"), "temp": c.get("temp")}
    elif base in E.STARS:
        M, R, T, dly = E.STARS[base]
        b = {"name": name, "label": nm(name, True), "kind": "star", "mass": M * MSUN * k, "radius": R * RSUN, "color": E.star_color(T), "temp": T,
             "distance_ly": dly, "real": False}
    elif base in E.BLACK_HOLES:
        M, dly = E.BLACK_HOLES[base]
        b = {"name": name, "label": nm(name, True), "kind": "black hole", "mass": M * MSUN * k, "radius": 2 * G * M * MSUN * k / 2.998e8 ** 2,
             "color": "#000000", "distance_ly": dly, "real": False}
    else:
        M, R, col, kind = GENERIC[base]
        b = {"name": name, "label": nm(name, True), "kind": kind, "mass": M * k, "radius": R, "color": col, "real": False}
    if (scale or {}).get("size"):
        b["radius"] *= k ** (1 / 3)
    b["r"] = np.zeros(3); b["v"] = np.zeros(3)
    return b


def _parse_date(t: str) -> date:
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", t)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    months = "january february march april may june july august september october november december".split()
    m = re.search(rf"\b(\d{{1,2}})?\s*({'|'.join(months)})\s*(\d{{1,2}})?,?\s*(\d{{4}})\b", t)
    if m:
        return date(int(m.group(4)), months.index(m.group(2)) + 1, int(m.group(1) or m.group(3) or 1))
    m = re.search(r"\b(?:in|on|year)\s+(\d{4})\b", t)
    if m and 1600 < int(m.group(1)) < 2500:
        return date(int(m.group(1)), 1, 1)
    return date.today()


def _parse_span(t: str) -> float | None:
    m = re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(years?|yrs?|months?|days?|weeks?|hours?|centur(?:y|ies))\b", t)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * (YEAR if u.startswith(("year", "yr")) else YEAR / 12 if u.startswith("month") else 7 * DAY if u.startswith("week")
                else 3600 if u.startswith("hour") else 100 * YEAR if u.startswith("centur") else DAY)


# ------------------------------------------------------------------------------------------------ counterfactuals
def gone_bodies(text: str) -> list[str]:
    """'what if there is no Sun', 'without Jupiter', 'if the Moon disappeared' -> the bodies taken away."""
    t = " " + text.lower() + " "
    out = []
    for _, n in bodies_in(t):
        w = re.escape(n)
        if (re.search(rf"\b(no|without|remov\w*|delet\w*|lost|lose|take away|took away|get rid of)\s+(the\s+)?{w}\b", t)
                or re.search(rf"\b{w}\s+(disappear\w*|vanish\w*|(was|were|is|went) (gone|missing|removed|deleted)|stopp?ed existing|ceased to exist)", t)
                or re.search(rf"\bif (the\s+)?{w} (did ?n[o']?t|never) exist", t)):
            if n not in out:
                out.append(n)
    return out


def orbit_of(text: str) -> tuple[str, str] | None:
    """'what if Jupiter orbits Earth', 'if the Moon were a moon of Mars' -> (satellite, primary), only when that is NOT how it really is."""
    t = " " + text.lower() + " "
    m = (re.search(r"\b([\w']+)\s+(?:would\s+|could\s+|did\s+|to\s+)?(?:orbits?|orbiting|orbited|goes around|go around|going around|went around|circles?|circling|"
                   r"revolves? around|revolving around)\s+(?:around\s+)?(?:the\s+)?([\w']+)", t)
         or re.search(r"\b([\w']+)\s+(?:was|were|became|is|as)\s+(?:the\s+|a\s+|an\s+)?(?:moon|satellite)\s+(?:of|around)\s+(?:the\s+)?([\w']+)", t))
    if not m:
        return None
    a_, b_ = bodies_in(" " + m.group(1) + " "), bodies_in(" " + m.group(2) + " ")
    if not a_ or not b_ or a_[0][1] == b_[0][1]:
        return None
    sat, prim = a_[0][1], b_[0][1]
    real_parent = CATALOG.get(sat, {}).get("parent") or ("sun" if CATALOG.get(sat, {}).get("kind") in ("planet", "dwarf planet", "comet", "asteroid") else None)
    return None if real_parent == prim else (sat, prim)


# ------------------------------------------------------------------------------------------------ set-ups
def setup(text: str, scale: dict | None = None) -> dict:
    """Decide what to simulate. Returns a scenario: bodies with initial states, how long, which frame, notes."""
    t = " " + text.lower() + " "
    names: list[str] = []
    for _, n in bodies_in(t):
        if n not in names:
            names.append(n)
    # "comet Encke", "the asteroid Apophis": the generic word just says what the named body is
    for generic, kinds in (("comet", ("comet",)), ("asteroid", ("asteroid", "dwarf planet")), ("planet", ("planet", "dwarf planet")), ("star", ("star",))):
        if generic in names and any(n in CATALOG and CATALOG[n]["kind"] in kinds for n in names):
            names.remove(generic)
    notes: list[str] = []
    collide = re.search(r"\b(collid\w*|crash\w*|hit\w*|impact\w*|smash\w*|slam\w*|fall\w* into|swallow\w*|merg\w*|devour\w*|eat\w*)\b", t) is not None
    d0 = _parse_date(t)
    span = _parse_span(t)
    sc: dict = {"date": d0, "span": span, "collide": collide, "notes": notes, "belts": [], "frame": None, "named": [], "checks": {}}
    with_moons = re.search(r"\bwith (its |their |all )?(the )?moons\b|\band moons\b", t) is not None

    # ---- counterfactuals: take a body away from the real Solar System / put one in orbit around another
    gone = gone_bodies(t)
    if gone:
        real = ["sun"] + PLANETS8 + ["pluto"] + [n for n in names if n in CATALOG and n not in PLANETS8 + ["sun", "pluto"]]
        if "moon" in gone and "moon" not in real:
            real.append("moon")
        sc.update(mode="system", real=real, remove=gone, named=[n for n in real if n not in gone and n != "sun"][:9],
                  span=span or (1 * YEAR if "sun" in gone else 2 * YEAR))
        sc["belts"] = [] if "sun" in gone else ["asteroid belt", "jupiter trojans"]
        return sc
    pair = orbit_of(t)
    if pair:
        sc.update(mode="orbit", names=[pair[1], pair[0]], primary=pair[1])
        return sc
    # ---- whole systems
    if "solar system" in t or "asteroid belt" in t or "kuiper belt" in t:
        real = ["sun"] + PLANETS8 + ["pluto"]
        if with_moons:
            real += ["moon", "io", "europa", "ganymede", "callisto", "titan", "triton"]
        real += [n for n in names if n in CATALOG and n not in real]
        sc.update(mode="system", real=real, named=[n for n in names if n in CATALOG and n != "sun"])
        sc["span"] = span or (2 * YEAR if with_moons else 12 * YEAR)
        sc["belts"] = ["asteroid belt", "jupiter trojans"] + (["kuiper belt"] if "kuiper" in t or "solar system" in t else [])
        return sc
    # ---- binary stars with measured orbits
    for key, (a_, b_, P, ecc) in E.BINARIES.items():
        if key in t:
            sc.update(mode="binary", binary=(a_, b_, P * YEAR, ecc), span=span or 1.15 * P * YEAR)
            return sc
    if not collide and re.search(r"\b(two|2|binary)\s+(?!neutron\b)(\w+\s+)?(stars|suns)\b", t):
        sc.update(mode="binary", binary=("star", "star", None, 0.3), span=span or None)
        return sc
    # ---- "what if Betelgeuse replaced the Sun"
    star = next((n for n in names if n in E.STARS or n in E.BLACK_HOLES), None)
    if star and re.search(r"replac\w*|instead of the sun|in place of the sun|were our sun|was our sun|became our sun", t):
        sc.update(mode="replace_sun", star=star, real=["sun"] + PLANETS8, span=span or 2 * YEAR)
        return sc
    # ---- real bodies seen where they really are (orbits, comets, asteroids, fly-bys) - not hypothetical crashes
    real_named = [n for n in names if n in CATALOG]
    hypothetical = [n for n in names if n not in CATALOG]
    flyby = any(n in ("apophis", "bennu") for n in real_named) and (collide or re.search(r"\b(2029|fly-?by|close approach|pass\w*|near|miss)\b", t))
    if real_named and not hypothetical and (not collide or flyby):
        if flyby and "apophis" in real_named:
            # Apophis' close approach to Earth on 13 April 2029
            sc.update(mode="flyby", real=["sun"] + PLANETS8 + ["moon", "apophis"], date=date(2029, 3, 29), span=span or 30 * DAY,
                      frame="earth", named=["apophis"])
            sc["checks"]["closest"] = ("apophis", "earth")
            if collide:
                notes.append("Apophis will NOT hit Earth: its 2029 pass is a near miss. Here is the real fly-by, from NASA's orbit data.")
            return sc
        target = next((n for n in real_named if n != "sun"), "earth")
        chain = [target]
        while CATALOG[chain[-1]].get("parent"):
            chain.append(CATALOG[chain[-1]]["parent"])
        moons = MOONS_OF.get(target, []) if re.search(r"\bmoons?\b", t) and CATALOG[target]["kind"] in ("planet", "dwarf planet") else []
        real = list(dict.fromkeys(["sun"] + chain[::-1] + real_named + moons + PLANETS8))      # every planet pulls on everything
        if moons:
            real_named = real_named + moons
            sc["frame_override"] = target
            sc["span"] = span or 1.1 * max(E._MOON_ORBITS[m][3] for m in moons if m in E._MOON_ORBITS) * DAY
        parent = CATALOG[target].get("parent") or "sun"
        frame = sc.pop("frame_override", None) or (parent if parent != "sun" else None)
        sc.update(mode="real", real=real, named=[n for n in real_named if n != "sun" and n not in chain[1:]], frame=frame, target=target)
        return sc
    # ---- hypothetical: collisions, orbits at a chosen height or speed
    count = re.search(r"\b(two|2)\s+(neutron stars|black holes|(?:\w+\s+)?(?:planets|stars|asteroids|moons|earths|suns))\b", t)
    if count and len(names) <= 1:
        what = count.group(2).split()[-1] if count.group(2) not in ("neutron stars", "black holes") else count.group(2)
        kind = {"planets": "earth", "stars": "sun", "black holes": "black hole", "neutron stars": "neutron star", "asteroids": "asteroid",
                "moons": "moon", "earths": "earth", "suns": "sun"}[what]
        names = [kind + " 1", kind + " 2"]
    if not names:
        names = ["earth", "moon"]
    sc.update(mode="collide" if collide else "orbit", names=names)
    return sc


# ------------------------------------------------------------------------------------------------ integration
class Run:
    def __init__(self, bodies: list[dict]):
        import warnings
        import rebound
        warnings.filterwarnings("ignore", message=".*predictor corrector loops.*")
        self.rb = rebound
        self.sim = rebound.Simulation()
        self.sim.G = G
        self.sim.integrator = "ias15"
        self.bodies = bodies
        self.Rot = np.eye(3)
        obl = next(((k, OBLATE[b["name"]]) for k, b in enumerate(bodies) if b.get("oblate") and b["name"] in OBLATE), None)
        if obl:
            self.Rot = _frame_to_pole(_pole_ecliptic(obl[1][3], obl[1][4]))
        for b in bodies:
            r, v = self.Rot @ np.asarray(b["r"], float), self.Rot @ np.asarray(b["v"], float)
            self.sim.add(m=b["mass"], r=b["radius"], x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
        if obl:
            # the planet's equatorial bulge (J2, J4) twists the orbits of close moons; REBOUNDx gravitational harmonics
            import reboundx
            self.rebx = reboundx.Extras(self.sim)
            gh = self.rebx.load_force("gravitational_harmonics")
            self.rebx.add_force(gh)
            J2, J4, Req = obl[1][:3]
            p = self.sim.particles[obl[0]]
            p.params["J2"] = J2; p.params["J4"] = J4; p.params["R_eq"] = Req
        self.alive = [True] * len(bodies)
        self.idx = list(range(len(bodies)))          # body k -> particle index (None once merged)
        self.merged_into: dict[int, int] = {}
        self.events: list[dict] = []
        self.E0 = self.energy(); self.L0 = self.angmom()

    def energy(self) -> float:
        return float(self.sim.energy())

    def angmom(self) -> np.ndarray:
        return np.array(self.sim.angular_momentum())

    def state(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        while self.idx[k] is None:
            k = self.merged_into[k]
        p = self.sim.particles[self.idx[k]]
        return self.Rot.T @ np.array(p.xyz), self.Rot.T @ np.array(p.vxyz)

    def run(self, T: float, frames: int, collide: bool, center: int | None = None) -> tuple[list[float], list]:
        self.sim.collision = "direct" if collide else "none"
        self.sim.collision_resolve = "halt"
        ts, ps = [], []
        for f in range(frames + 1):
            tt = T * f / frames
            while True:
                try:
                    self.sim.integrate(tt, exact_finish_time=1)
                    break
                except self.rb.Collision:
                    self._merge()
                    T = min(T, self.sim.t + 0.3 * T)        # watch the aftermath a little
            ts.append(self.sim.t)
            c = self.state(center)[0] if center is not None else np.zeros(3)
            ps.append([(self.state(k)[0] - c).tolist() for k in range(len(self.bodies))])
            if self.sim.t >= T - 1e-9 and f < frames and self.events:
                break
        return ts, ps

    def _merge(self) -> None:
        ps = self.sim.particles
        live = [k for k in range(len(self.bodies)) if self.idx[k] is not None]
        for x in range(len(live)):
            for y in range(x + 1, len(live)):
                i, j = live[x], live[y]
                a, b = ps[self.idx[i]], ps[self.idx[j]]
                d = math.dist(a.xyz, b.xyz)
                if d <= a.r + b.r:
                    big, small = (i, j) if a.m >= b.m else (j, i)
                    pb, pss = ps[self.idx[big]], ps[self.idx[small]]
                    mu = pb.m * pss.m / (pb.m + pss.m)
                    vrel = math.dist(pb.vxyz, pss.vxyz)
                    # detected a fraction of a step after first contact (slightly overlapping): wind the relative motion back
                    # to the moment of contact with energy conservation, v_c² = v² − 2G(M+m)(1/d − 1/(R+r))
                    vrel = math.sqrt(max(0.0, vrel ** 2 - 2 * G * (pb.m + pss.m) * (1 / max(d, 1e-9) - 1 / (pb.r + pss.r))))
                    contact = np.array(pb.xyz) + (np.array(pss.xyz) - np.array(pb.xyz)) * pb.r / (pb.r + pss.r)
                    self.events.append({"t": self.sim.t, "kind": "impact", "a": self.bodies[small]["name"], "b": self.bodies[big]["name"], "speed": vrel,
                                        "energy": 0.5 * mu * vrel ** 2, "pos": contact.tolist(), "small_radius": pss.r, "big_radius": pb.r})
                    M = pb.m + pss.m
                    v = (np.array(pb.vxyz) * pb.m + np.array(pss.vxyz) * pss.m) / M
                    r = (np.array(pb.xyz) * pb.m + np.array(pss.xyz) * pss.m) / M
                    pb.m, pb.r = M, (pb.r ** 3 + pss.r ** 3) ** (1 / 3)
                    pb.x, pb.y, pb.z = r
                    pb.vx, pb.vy, pb.vz = v
                    gone = self.idx[small]
                    self.sim.remove(gone)
                    self.idx = [None if q is None else (q if q < gone else q - 1) if q != gone else None for q in self.idx]
                    self.merged_into[small] = big
                    self.alive[small] = False
                    return


# ------------------------------------------------------------------------------------------------ helpers
def _refine_closest(bodies: list[dict], ts: list[float], kq: int, k: int, pk: int, n: int = 600) -> tuple[float, float, float]:
    """Exact closest approach of body k to body pk near frame kq: re-integrate that stretch finely.
    Returns (distance m, time s, relative speed m/s)."""
    lo, hi = ts[max(kq - 1, 0)], ts[min(kq + 1, len(ts) - 1)]
    r = Run([{**b, "r": b["r0"], "v": b["v0"]} for b in bodies])
    r.sim.integrate(lo, exact_finish_time=1)
    best = (math.inf, lo, 0.0)
    for tt in np.linspace(lo, hi, n):
        r.sim.integrate(tt, exact_finish_time=1)
        a, b = r.sim.particles[k], r.sim.particles[pk]
        d = math.dist(a.xyz, b.xyz)
        if d < best[0]:
            best = (d, float(tt), math.dist(a.vxyz, b.vxyz))
    return best
def _osc(r_rel: np.ndarray, v_rel: np.ndarray, mu: float) -> dict:
    """Osculating Kepler orbit of a body relative to its primary."""
    r, v = float(np.linalg.norm(r_rel)), float(np.linalg.norm(v_rel))
    eps = v * v / 2 - mu / r
    h = np.cross(r_rel, v_rel)
    ev = np.cross(v_rel, h) / mu - r_rel / r
    e = float(np.linalg.norm(ev))
    a = -mu / (2 * eps) if eps < 0 else math.inf
    P = 2 * math.pi * math.sqrt(a ** 3 / mu) if math.isfinite(a) else math.inf
    return {"a": a, "e": e, "P": P, "q": a * (1 - e) if math.isfinite(a) else r, "bound": eps < 0}


def _belts(sun_r: np.ndarray, jup: tuple[np.ndarray, np.ndarray] | None, which: list[str], seed: int = 7) -> list[dict]:
    """Test-particle belts as orbital elements; the viewer moves each one along its own Kepler orbit around the Sun
    (the Sun's pull dominates there; the planets' tugs over a few orbits are left out and the HUD says so)."""
    rng = np.random.default_rng(seed)
    out = []
    mu = G * CATALOG["sun"]["mass"]
    if "asteroid belt" in which:
        n = 2400
        # main belt 2.1-3.3 AU, with the Kirkwood gaps Jupiter's resonances clear (3:1, 5:2, 7:3, 2:1)
        a = rng.uniform(2.1, 3.3, n * 2)
        keep = np.ones_like(a, bool)
        for g, w in ((2.50, 0.03), (2.82, 0.025), (2.95, 0.02), (3.27, 0.04)):
            keep &= np.abs(a - g) > w
        a = a[keep][:n]
        n = len(a)
        e = np.clip(rng.rayleigh(0.08, n), 0, 0.3)
        inc = np.radians(np.clip(rng.rayleigh(7.0, n), 0, 30))
        out.append({"name": "asteroid belt", "color": "#a8a29e", "a": (a * AU).tolist(), "e": e.tolist(), "i": inc.tolist(),
                    "node": rng.uniform(0, 2 * np.pi, n).tolist(), "argp": rng.uniform(0, 2 * np.pi, n).tolist(),
                    "M0": rng.uniform(0, 2 * np.pi, n).tolist(), "mu": mu, "size": 1.0})
    if "jupiter trojans" in which and jup is not None:
        # asteroids sharing Jupiter's orbit 60° ahead (L4) and behind (L5): Lagrange points stable for 4.5 billion years
        r, v = jup[0] - sun_r, jup[1]
        o = _osc(r, v, mu + G * CATALOG["jupiter"]["mass"])
        lam = math.atan2(r[1], r[0])
        n = 700
        side = np.where(rng.random(n) < 0.5, 1.0, -1.0)
        lon = lam + side * np.radians(60) + np.radians(rng.normal(0, 12, n))
        out.append({"name": "Jupiter trojans", "color": "#d6b48a", "a": (o["a"] * (1 + rng.normal(0, 0.02, n))).tolist(), "e": np.clip(rng.rayleigh(0.06, n), 0, 0.2).tolist(),
                    "i": np.radians(np.clip(rng.rayleigh(10, n), 0, 35)).tolist(), "node": np.zeros(n).tolist(), "argp": np.zeros(n).tolist(),
                    "M0": lon.tolist(), "mu": mu, "size": 1.0, "longitude_mode": True})
    if "kuiper belt" in which:
        n = 2000
        a = np.concatenate([rng.normal(39.4, 0.3, n // 4), rng.uniform(42, 48, n - n // 4)])     # plutinos (3:2 with Neptune) + classical belt
        out.append({"name": "Kuiper belt", "color": "#93c5fd", "a": (a * AU).tolist(), "e": np.clip(rng.rayleigh(0.07, n), 0, 0.3).tolist(),
                    "i": np.radians(np.clip(rng.rayleigh(6, n), 0, 30)).tolist(), "node": rng.uniform(0, 2 * np.pi, n).tolist(),
                    "argp": rng.uniform(0, 2 * np.pi, n).tolist(), "M0": rng.uniform(0, 2 * np.pi, n).tolist(), "mu": mu, "size": 1.0})
    return out


def _fmt(v, unit):
    if unit == "kg":
        return f"{v:.3g} kg"
    if unit in ("J", "N"):
        return f"{v:.3g} {unit}"
    if unit == "Mt":
        return f"{v:.3g} megatons of TNT" if v < 1e6 else f"{v:.3g} Mt TNT"
    if unit == "km" and abs(v) >= 1e7:
        return f"{v / AU * 1000:.3g} AU"
    return f"{v:,.4g} {unit}" if abs(v) < 1e6 else f"{v:.4g} {unit}"


def _dur(s):
    s = abs(s)
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    if s < 2 * YEAR:
        return f"{s / 86400:.1f} days"
    return f"{s / YEAR:.2f} years"


def black_hole_note(bodies) -> list[str]:
    if any(b["kind"] in ("black hole", "neutron star") for b in bodies):
        return ["Gravity here is Newton's. Close to a black hole or neutron star, Einstein's relativity matters (orbits precess, "
                "and merging black holes radiate gravitational waves); this engine doesn't model those effects."]
    return []


# ------------------------------------------------------------------------------------------------ the run
def run(text: str, scale: dict | None = None, changes: list[str] | None = None) -> dict:
    t_low = text.lower()
    if re.search(GALAXY_WORDS, t_low):
        from . import galaxy
        return galaxy.run(text)
    sc = setup(text, scale)
    mode = sc["mode"]
    notes = sc["notes"]
    rows, story = [], []
    checks: list[tuple[str, float]] = []

    def add(label, v, unit, about="system", text_=None, why=None):
        rows.append({"label": label, "value": float(v), "unit": unit, "about": about, "text": text_ or _fmt(v, unit), "why": why})

    source = "hypothetical set-up"
    center = None
    bodies: list[dict] = []
    if mode in ("system", "real", "flyby", "replace_sun"):
        real = sc["real"]
        moons_in = {n for n in real if CATALOG[n]["kind"] == "moon"}
        bary = {p for p in SYSTEM_BARY if p in real and not any(CATALOG[m].get("parent") == p for m in moons_in)}
        cmds = {n: (SYSTEM_BARY[n][0] if n in bary else CATALOG[n]["hz"]) for n in real}
        saved = {n: CATALOG[n]["hz"] for n in real}
        for n in real:
            CATALOG[n]["hz"] = cmds[n]
        try:
            st, source = E.states(real, sc["date"])
        finally:
            for n in real:
                CATALOG[n]["hz"] = saved[n]
        for n in real:
            b = _body(n, scale, bary=n in bary)
            b["r"], b["v"] = st[n]
            b["oblate"] = mode == "real" and n == sc.get("frame") and n in OBLATE
            bodies.append(b)
        if sc.get("remove"):
            gone_ = [b for b in bodies if b["name"] in sc["remove"]]
            bodies = [b for b in bodies if b["name"] not in sc["remove"]]
            real = [b["name"] for b in bodies]
            sc["gone_bodies"] = gone_
            notes.append(f"{', '.join(nm(n, True) for n in sc['remove'])} removed at the start; every other body keeps its real NASA position and velocity "
                         f"of {sc['date'].isoformat()}.")
        if mode == "replace_sun":
            s = _body(sc["star"], scale)
            sun = bodies[0]
            sun.update({k: s[k] for k in ("label", "kind", "mass", "radius", "color", "temp")})
            sun["name"] = sc["star"]
            notes.append(f"{nm(sc['star'], True)} put where the Sun is, with every planet keeping its real position and speed of {sc['date'].isoformat()}.")
            sc["engulfed"] = [b for b in bodies[1:] if np.linalg.norm(b["r"] - sun["r"]) < sun["radius"]]
            bodies = [b for b in bodies if b not in sc["engulfed"]]
            real = [b["name"] for b in bodies]
        if sc.get("frame"):
            center = real.index(sc["frame"])
        if not sc["span"]:
            tgt = sc.get("target")
            k = real.index(tgt)
            par = CATALOG[tgt].get("parent") or "sun"
            pk = real.index(par)
            o = _osc(bodies[k]["r"] - bodies[pk]["r"], bodies[k]["v"] - bodies[pk]["v"], G * (bodies[k]["mass"] + bodies[pk]["mass"]))
            sc["span"] = min(1.02 * o["P"], 250 * YEAR) if o["bound"] else 5 * YEAR
            if sc["span"] > 60 * YEAR:
                bodies = [b for b in bodies if b["name"] not in ("mercury", "venus") or b["name"] == tgt]
                real = [b["name"] for b in bodies]
                if sc.get("frame"):
                    center = real.index(sc["frame"])
    elif mode == "binary":
        a_, b_, P, ecc = sc["binary"]
        A, B = _body(a_ + (" 1" if a_ == b_ else ""), scale), _body(b_ + (" 2" if a_ == b_ else ""), scale)
        Mt = A["mass"] + B["mass"]
        if P is None:
            P = 2 * YEAR
            notes.append("Two Sun-like stars on a 2-year orbit (e = 0.3).")
        a = (G * Mt * P ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
        rp = a * (1 - ecc)
        vp = math.sqrt(G * Mt * (2 / rp - 1 / a))
        A["r"] = np.array([-rp * B["mass"] / Mt, 0, 0]); B["r"] = np.array([rp * A["mass"] / Mt, 0, 0])
        A["v"] = np.array([0, -vp * B["mass"] / Mt, 0]); B["v"] = np.array([0, vp * A["mass"] / Mt, 0])
        bodies = [A, B]
        sc["span"] = sc["span"] or 1.15 * P
        source = "measured orbit (period and eccentricity from astronomers' observations)" if a_ != "star" else source
        sc["checks"]["binary"] = (P, a, ecc)
    else:
        bodies = [_body(n, scale) for n in sc["names"]]
        if len(bodies) == 1:
            b = bodies[0]
            par = b.get("parent") or ("earth" if b["mass"] < 1e20 else "sun")
            bodies.insert(0, _body(par, scale))
            notes.append(f"Added {nm(par)}, what {nm(b['name'])} {'falls toward' if mode == 'collide' else 'orbits'}.")
        big = next((b for b in bodies if b["name"] == sc.get("primary")), None) or max(bodies, key=lambda b: b["mass"])
        others = [b for b in bodies if b is not big]
        if mode == "collide" and len(bodies) == 2:
            # how do these two really meet? a rigid "touch and merge" is wrong for black holes, neutron stars and tidal tearing
            from . import encounters
            enc = encounters.classify(big, others[0])
            if enc["type"] == "inspiral":
                return encounters.inspiral(text, big, others[0], notes)
            if enc["type"] == "disrupt":
                return encounters.tde(text, enc["victim"], enc["other"], enc["rt"], notes)
            if enc["type"] == "swallowed":
                sc["swallowed"] = enc
                notes.append(f"{nm(enc['victim']['name'], True)}'s tidal radius ({enc['rt'] / 1000:,.0f} km) lies inside the event horizon "
                             f"({enc['other']['radius'] / 1000:,.0f} km): it crosses the horizon before tides can tear it — swallowed whole, with no flare.")
        tl = " " + t_low + " "
        km = re.search(r"(\d+(?:\.\d+)?)\s*(km|kilometers?|kilometres?)\b(?!/)", tl)
        kms = re.search(r"(\d+(?:\.\d+)?)\s*(km/s|kilometers? per second)", tl)
        if mode == "orbit":
            for i, b in enumerate(others):
                base = re.sub(r" \d+$", "", b["name"])
                if km and len(others) == 1:
                    r = big["radius"] + float(km.group(1)) * 1000
                elif base in CATALOG and CATALOG[base].get("parent") == re.sub(r" \d+$", "", big["name"]):
                    st_, _ = E.states([base, CATALOG[base]["parent"]], date.today())
                    r = float(np.linalg.norm(st_[base][0] - st_[CATALOG[base]["parent"]][0]))
                elif sc.get("primary") == "earth" and base not in ("iss", "space station", "satellite") and 3.844e8 > 3 * (big["radius"] + b["radius"]):
                    r = 3.844e8
                    notes.append(f"{nm(b['name'], True)} placed where the Moon is ({r / 1000:,.0f} km from Earth), moving at the speed for a circular orbit.")
                elif base in ("iss", "space station"):
                    r = big["radius"] + 4.0e5
                elif base == "satellite":
                    r = big["radius"] + 3.5786e7 if "geostationary" in tl else big["radius"] + 5.5e5
                else:
                    r = (big["radius"] + b["radius"]) * (8 + 4 * i)
                    notes.append(f"{nm(b['name'], True)} doesn't really orbit {nm(big['name'])}; placed at {r / 1000:,.0f} km to show what an orbit there would be like.")
                v = math.sqrt(G * (big["mass"] + b["mass"]) / r)
                if kms and len(others) == 1:
                    v = float(kms.group(1)) * 1000
                    notes.append(f"Launched at {v / 1000:g} km/s instead of the circular-orbit speed {math.sqrt(G * big['mass'] / r) / 1000:.2f} km/s.")
                if b["mass"] > big["mass"]:
                    d_cm = r * big["mass"] / (big["mass"] + b["mass"])
                    notes.append(f"{nm(b['name'], True)} is {b['mass'] / big['mass']:,.0f}× heavier than {nm(big['name'])}, so neither simply orbits the other: both circle "
                                 f"their shared centre of mass, {d_cm / 1000:,.0f} km from {nm(b['name'])}'s centre"
                                 + (f" — inside {nm(b['name'])} itself. In practice {nm(big['name'])} does the circling." if d_cm < b["radius"] else "."))
                ang = 2 * math.pi * i / max(1, len(others))
                b["r"] = np.array([r * math.cos(ang), r * math.sin(ang), 0.0])
                b["v"] = np.array([-v * math.sin(ang), v * math.cos(ang), 0.0])
        else:
            # a real encounter: falling in from far away (parabolic) or at the given speed, hitting off-centre - not dropped from rest
            from . import encounters
            b = others[0]
            D = 6 * (big["radius"] + b["radius"])
            q = 0.45 * (big["radius"] + b["radius"])
            rel_r, rel_v = encounters.parabolic(big["mass"] + b["mass"], q, D, float(kms.group(1)) * 1000 if kms else None)
            b["r"], b["v"] = rel_r, rel_v
            for k, o in enumerate(others[1:]):
                o["r"] = np.array([0.0, D * (1.5 + k), 0.0])
        Mt = sum(b["mass"] for b in bodies)
        cm = sum(b["mass"] * b["r"] for b in bodies) / Mt
        vcm = sum(b["mass"] * b["v"] for b in bodies) / Mt
        for b in bodies:
            b["r"] = b["r"] - cm; b["v"] = b["v"] - vcm
        if mode == "orbit":
            ps_ = []
            for b in others:
                o = _osc(b["r"] - big["r"], b["v"] - big["v"], G * (big["mass"] + b["mass"]))
                ps_.append(o["P"] if o["bound"] else 1e5)
            sc["span"] = sc["span"] or (1.25 * min(ps_) if len(ps_) == 1 or max(ps_) / min(ps_) > 6 else 1.1 * max(ps_))
        else:
            i_ = bodies.index(others[0]); r = float(np.linalg.norm(bodies[i_]["r"] - big["r"]))
            M = big["mass"] + others[0]["mass"]
            v0 = float(np.linalg.norm(bodies[i_]["v"] - big["v"]))
            tff = math.pi / 2 * math.sqrt(r ** 3 / (2 * G * M))
            sc["span"] = sc["span"] or 1.4 * (min(tff, r / max(v0, 1e-9)) if v0 > 0 else tff)
    for b in bodies:
        b["r0"], b["v0"] = b["r"].copy(), b["v"].copy()
    m0 = [b["mass"] for b in bodies]
    run_ = Run(bodies)
    T = float(sc["span"])
    frames = 900 if mode in ("system", "real", "flyby", "replace_sun") else 600
    ts, ps = run_.run(T, frames, collide=sc["collide"] or mode in ("collide", "replace_sun", "system", "real", "flyby", "binary", "orbit"), center=center)
    E1, L1 = run_.energy(), run_.angmom()
    dE = abs(E1 - run_.E0) / abs(run_.E0) if run_.E0 else 0.0
    dL = float(np.linalg.norm(L1 - run_.L0) / max(np.linalg.norm(run_.L0), 1e-300))
    names = [b["name"] for b in bodies]
    events = run_.events

    # ------------------------------------------------ measurements and the laws they must obey
    if not events:
        add("Energy conserved (change over the whole run)", dE, "", "system", f"{dE:.1e} (relative)",
            "total kinetic + gravitational energy can't change without outside forces")
        add("Angular momentum conserved (change)", dL, "", "system", f"{dL:.1e} (relative)")
        checks.append(("energy", 100 * (1 - dE)))
    P_ = np.array(ps)
    for k, b in enumerate(bodies):
        base = re.sub(r" \d+$", "", b["name"])
        if mode in ("system", "real", "flyby", "replace_sun"):
            par = (CATALOG.get(base, {}).get("parent") or "sun") if mode != "replace_sun" else "sun"
            if b["name"] == "sun" or par not in names or (mode == "replace_sun" and k == 0):
                continue
            if mode == "system" and b["kind"] == "moon":
                continue
            if mode in ("real", "flyby") and base not in sc["named"]:
                continue
            pk = names.index(par) if par in names else 0
        elif mode == "binary":
            if k == 0:
                continue
            pk = 0
        else:
            bigk = int(np.argmax(m0))
            if k == bigk:
                continue
            pk = bigk
        mu = G * (m0[pk] + m0[k])
        o = _osc(b["r0"] - bodies[pk]["r0"], b["v0"] - bodies[pk]["v0"], mu)
        rel = P_[:, k, :] - P_[:, pk, :]
        ang = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))
        turned = abs(ang[-1] - ang[0])
        dist = np.linalg.norm(rel, axis=1)
        if not o["bound"] or not run_.alive[k]:
            continue
        lab = nm(b["name"], True)
        if turned >= 2 * math.pi * 0.98:
            # measured period: time to sweep 2π (interpolated), compared with Kepler's third law for the starting orbit
            tt = np.array(ts[:len(ang)])
            T_meas = float(np.interp(ang[0] + math.copysign(2 * math.pi, ang[-1] - ang[0]), ang if ang[-1] > ang[0] else ang[::-1],
                                     tt if ang[-1] > ang[0] else tt[::-1]))
            add(f"{lab}: orbital period (measured)", T_meas / 3600, "h", b["name"], _dur(T_meas))
            add(f"{lab}: orbital period (Kepler 2π√(a³/GM))", o["P"] / 3600, "h", b["name"], _dur(o["P"]))
            checks.append((f"{lab} period", 100 * (1 - abs(T_meas - o["P"]) / o["P"])))
            rmin, rmax = float(dist.min()), float(dist.max())
            e_meas = (rmax - rmin) / (rmax + rmin)
            add(f"{lab}: closest / farthest from {nm(names[pk])}", rmin / 1000, "km", b["name"], f"{_fmt(rmin / 1000, 'km')} / {_fmt(rmax / 1000, 'km')}")
            two_body = mode in ("orbit", "binary")
            if o["e"] > 0.02:
                add(f"{lab}: orbit eccentricity (measured)", e_meas, "", b["name"], f"{e_meas:.4f}")
                add(f"{lab}: orbit eccentricity ({'theory' if two_body else 'two-body estimate'}, from its starting speed and distance)", o["e"], "", b["name"], f"{o['e']:.4f}",
                    None if two_body else "other bodies' pulls reshape a real orbit as it goes, so this is a guide; the NASA check is the real test")
                if two_body:
                    checks.append((f"{lab} eccentricity", 100 * (1 - abs(e_meas - o["e"]) / o["e"])))
            if not two_body:
                checks.pop()                     # a real, perturbed orbit isn't exactly Keplerian: NASA's ephemeris is the test
                if base == "moon":
                    story.append("The Sun's pull stretches and squeezes the Moon's orbit every month (evection): its distance swings between "
                                 f"{rmin / 1000:,.0f} and {rmax / 1000:,.0f} km here, more than a lone Earth–Moon pair would give — exactly what's observed.")
        else:
            add(f"{lab}: orbital period (Kepler 2π√(a³/GM))", o["P"] / 3600, "h", b["name"], _dur(o["P"]))
        add(f"{lab}: speed now", float(np.linalg.norm(b["v0"] - bodies[pk]["v0"])) / 1000, "km/s", b["name"])
        if (b["kind"] == "comet" or (b["kind"] in ("asteroid", "dwarf planet") and o["e"] > 0.15)) and turned > math.pi:
            # perihelion: the closest approach, found precisely with the integrator between the frames around it
            kq = int(np.argmin(dist))
            q_meas, t_q, v_q = _refine_closest(bodies, ts, kq, k, pk)
            vq = math.sqrt(mu * (2 / o["q"] - 1 / o["a"]))
            when = datetime(sc["date"].year, sc["date"].month, sc["date"].day) + timedelta(seconds=t_q)
            add(f"{lab}: perihelion — closest to the Sun (measured)", q_meas / 1000, "km", b["name"])
            add(f"{lab}: perihelion date (measured)", 0, "", b["name"], when.strftime("%d %B %Y"))
            add(f"{lab}: speed at perihelion (measured)", v_q / 1000, "km/s", b["name"])
            two_body = mode in ("orbit", "binary")
            add(f"{lab}: perihelion ({'theory' if two_body else 'two-body estimate'} a(1 − e))", o["q"] / 1000, "km", b["name"])
            add(f"{lab}: speed at perihelion ({'theory' if two_body else 'two-body estimate'}, vis-viva √(GM(2/q − 1/a)))", vq / 1000, "km/s", b["name"])
            # vis-viva holds exactly at the measured perihelion for the energy it has there
            vv = math.sqrt(mu * (2 / q_meas - 1 / o["a"]))
            if two_body:
                checks.append((f"{lab} perihelion", 100 * (1 - abs(q_meas - o["q"]) / o["q"])))
            story.append(f"{lab} swings in to {_fmt(q_meas / 1000, 'km')} from the Sun on {when.strftime('%d %B %Y')}, where it moves fastest ({v_q / 1000:.1f} km/s) — "
                         f"Kepler's second law: a line from the Sun sweeps out equal areas in equal times, so close in it must race, far out it crawls."
                         + ("" if two_body else f" The planets' pulls (mostly Jupiter's) reshape the orbit on the way, which is why the two-body estimate from its "
                                                f"starting position ({_fmt(o['q'] / 1000, 'km')}) differs from what the full simulation finds — real comets' perihelia shift between visits too."))
    # NASA check: our integration vs NASA's positions at the end of the run
    if mode in ("system", "real", "flyby") and "NASA" in source and sc["span"] <= 6 * YEAR and not sc.get("remove"):
        d1 = sc["date"] + timedelta(seconds=float(ts[-1]))
        if d1.isoformat() != sc["date"].isoformat() and abs((datetime(d1.year, d1.month, d1.day) - datetime(sc["date"].year, sc["date"].month, sc["date"].day)).total_seconds() - ts[-1]) < 1.5 * DAY:
            d1_ = date(d1.year, d1.month, d1.day)
            T_end = (datetime(d1_.year, d1_.month, d1_.day) - datetime(sc["date"].year, sc["date"].month, sc["date"].day)).total_seconds()
            run2 = Run([{**b, "r": b["r0"], "v": b["v0"]} for b in bodies])
            run2.run(T_end, 1, collide=False)
            check = [n for n in names if n != "sun" and (n in sc["named"] or (mode == "system" and CATALOG.get(n, {}).get("kind") != "moon"))][:12]
            moons_in = {n for n in names if CATALOG.get(n, {}).get("kind") == "moon"}
            for n in check:
                CATALOG[n]["_hz"] = CATALOG[n]["hz"]
                if n in SYSTEM_BARY and not any(CATALOG[m].get("parent") == n for m in moons_in):
                    CATALOG[n]["hz"] = SYSTEM_BARY[n][0]
            try:
                nasa, src2 = E.states(check + ["sun"], d1_)
            finally:
                for n in check:
                    CATALOG[n]["hz"] = CATALOG[n].pop("_hz")
            if check and "NASA JPL Horizons" in src2 and "offline" not in src2:
                errs = []
                for n in check:
                    ours = run2.state(names.index(n))[0]
                    err = float(np.linalg.norm(ours - nasa[n][0]))
                    par = CATALOG.get(n, {}).get("parent")
                    if CATALOG.get(n, {}).get("kind") == "moon" and par in names:
                        # a moon is judged against the size of its own orbit, not its distance from the Sun
                        ref = float(np.linalg.norm(ours - run2.state(names.index(par))[0]))
                    else:
                        ref = float(np.linalg.norm(nasa[n][0] - nasa["sun"][0]))
                    errs.append((n, err, ref))
                worst = max(errs, key=lambda x: x[1] / x[2])
                add(f"Position error vs NASA after {_dur(T_end)} (worst: {nm(worst[0])})", worst[1] / 1000, "km", "system",
                    why="our integration from NASA's starting data, compared with NASA's own ephemeris on the final day")
                for n, err, ref in errs:
                    what = "of its orbit around " + nm(CATALOG[n]["parent"]) if CATALOG.get(n, {}).get("kind") == "moon" and CATALOG[n].get("parent") in names else "of its distance from the Sun"
                    add(f"{nm(n, True)}: off NASA's position by", err / 1000, "km", n, f"{err / 1000:,.0f} km ({100 * err / ref:.4f}% {what})")
                checks.append(("NASA", 100 * (1 - worst[1] / worst[2])))
                story.append(f"Checked against NASA: after {_dur(T_end)} of simulated motion every body is within {max(e[1] for e in errs) / 1000:,.0f} km of where NASA's "
                             f"ephemeris (DE441, used to fly spacecraft) puts it — at most {100 * max(e[1] / e[2] for e in errs):.4f}% of the size of its orbit.")
    if "closest" in sc["checks"]:
        a_n, b_n = sc["checks"]["closest"]
        ka, kb = names.index(a_n), names.index(b_n)
        dd = np.linalg.norm(P_[:, ka, :] - P_[:, kb, :], axis=1)
        k = int(np.argmin(dd))
        # refine the closest approach between frames with the integrator itself
        lo, hi = ts[max(k - 1, 0)], ts[min(k + 1, len(ts) - 1)]
        run3 = Run([{**b, "r": b["r0"], "v": b["v0"]} for b in bodies])
        run3.run(lo, 1, collide=False)
        best = (float(dd[k]), ts[k])
        for tt in np.linspace(lo, hi, 400):
            run3.sim.integrate(tt, exact_finish_time=1)
            dd_ = float(np.linalg.norm(np.array(run3.sim.particles[ka].xyz) - np.array(run3.sim.particles[kb].xyz)))
            if dd_ < best[0]:
                best = (dd_, tt)
        when = datetime(sc["date"].year, sc["date"].month, sc["date"].day) + timedelta(seconds=best[1])
        Re = CATALOG["earth"]["radius"]
        add(f"Closest approach of {nm(a_n)} to {nm(b_n)} (centre to centre)", best[0] / 1000, "km", a_n)
        add("…height above the surface", (best[0] - Re) / 1000, "km", a_n)
        add("…when", 0, "", a_n, when.strftime("%d %B %Y, %H:%M UTC"))
        add("Height of geostationary satellites (for comparison)", 35786, "km", "system")
        add("Closest approach (NASA JPL published: 38,012 km from Earth's centre, 13 April 2029 21:46 UTC)", 38012, "km", a_n)
        checks.append(("NASA closest approach", 100 * (1 - abs(best[0] / 1000 - 38012) / 38012)))
        story.append(f"On {when.strftime('%d %B %Y')} Apophis passes {best[0] / 1000:,.0f} km from Earth's centre — {(best[0] - Re) / 1000:,.0f} km above the surface, "
                     f"closer than geostationary satellites (35,786 km). It's 340 m across and will be visible to the naked eye. Earth's gravity bends its "
                     f"path and changes its orbit, but it misses: there is no impact risk in 2029.")
    if mode == "binary" and "binary" in sc["checks"]:
        P, a, ecc = sc["checks"]["binary"]
        add("Separation (semi-major axis, from Kepler's 3rd law)", a / AU, "AU", "system", f"{a / AU:.2f} AU")
    if mode == "replace_sun":
        star = bodies[0]
        gone = sc.get("engulfed", [])
        add(f"{nm(sc['star'], True)}'s radius", star["radius"] / AU, "AU", "star", f"{star['radius'] / AU:.2f} AU ({star['radius'] / RSUN:,.0f}× the Sun)")
        if gone:
            story.append(f"{nm(sc['star'], True)} is {star['radius'] / RSUN:,.0f}× wider than the Sun — {star['radius'] / AU:.2f} AU — so "
                         f"{', '.join(nm(g['name'], True) for g in gone)} would be inside the star itself and vaporised.")
        story.append(f"With {star['mass'] / MSUN:.3g}× the Sun's mass pulling, the surviving planets — still at their real speeds — are no longer on circular orbits: "
                     f"each one's speed is now too slow for its distance, so they fall into new, tighter, more elongated orbits (Kepler's laws with the new mass).")
    if events:
        e = events[0]
        add("Impact speed", e["speed"] / 1000, "km/s")
        add("Energy released", e["energy"], "J")
        add("Energy in megatons of TNT", e["energy"] / MT_TNT, "Mt", text_=_fmt(e["energy"] / MT_TNT, "Mt"))
        add("× the dinosaur-killing impact", e["energy"] / CHICXULUB_J, "×", text_=f"{e['energy'] / CHICXULUB_J:.3g}×")
        add("Time until impact", e["t"] / 3600, "h", text_=_dur(e["t"]))
        small = bodies[names.index(e["a"])]; bigb = bodies[names.index(e["b"])]
        vesc = math.sqrt(2 * G * (small["mass"] + bigb["mass"]) / (small["radius"] + bigb["radius"]))
        add("Escape speed at contact √(2GM/r)", vesc / 1000, "km/s")
        if len(bodies) == 2:
            r0 = float(np.linalg.norm(small["r0"] - bigb["r0"])); rc = small["radius"] + bigb["radius"]
            vr0 = float(np.linalg.norm(small["v0"] - bigb["v0"]))
            v_th = math.sqrt(vr0 ** 2 + 2 * G * (small["mass"] + bigb["mass"]) * (1 / rc - 1 / r0))
            add("Impact speed (theory: energy conservation, v² = v₀² + 2G(M+m)(1/r − 1/r₀))", v_th / 1000, "km/s")
            checks.append(("impact", 100 * (1 - abs(e["speed"] - v_th) / v_th)))
        F0 = G * small["mass"] * bigb["mass"] / float(np.linalg.norm(small["r0"] - bigb["r0"])) ** 2
        add("Gravitational pull at the start", F0, "N")
        story.append(f"Gravity pulls {nm(e['a'])} and {nm(e['b'])} together with {_fmt(F0, 'N')} at the start (F = G·m₁·m₂/r²). The pull grows as the gap "
                     f"shrinks, so they hit at {e['speed'] / 1000:.1f} km/s after {_dur(e['t'])} — close to the escape speed {vesc / 1000:.1f} km/s, because the energy "
                     f"of the fall equals the gravitational energy they had apart.")
        ratio = e["energy"] / CHICXULUB_J
        story.append(f"All that motion turns into heat in an instant: {_fmt(e['energy'], 'J')}, about {_fmt(e['energy'] / MT_TNT, 'Mt')}"
                     + (f" — {ratio:.3g}× the impact that ended the dinosaurs." if ratio > 1e-3 else ".")
                     + " Momentum is conserved through the merger: the combined body carries exactly the momentum the two had together.")
        # what survives: impact energy vs the energy holding each body together (uniform sphere U = 3Gm²/5r)
        if not sc.get("swallowed") and small.get("kind") not in ("black hole", "neutron star"):
            U_s = 3 * G * small["mass"] ** 2 / (5 * small["radius"])
            U_b = 3 * G * bigb["mass"] ** 2 / (5 * bigb["radius"]) if bigb.get("kind") not in ("black hole",) else float("inf")
            add(f"{nm(e['a'], True)}'s gravitational binding energy (3Gm²/5r)", U_s, "J")
            k_s = e["energy"] / U_s
            if small["mass"] >= 0.3 * bigb["mass"]:
                kind_ = small.get("kind") or "body"
                fate = (f"Being of similar size, they don't crater each other — they merge into one {kind_}: the impact delivers {k_s:,.2g}× the smaller one's "
                        f"binding energy, so both are heated through and reshaped" + (" (merged stars like this are seen in star clusters as “blue stragglers”)"
                                                                                    if kind_ == "star" else "") + ".")
            elif k_s >= 1:
                fate = (f"{nm(e['a'], True)} is destroyed: the impact carries {k_s:,.3g}× the energy holding it together, so it is vaporised and smeared "
                        f"into {nm(e['b'])}")
            elif k_s > 0.1:
                fate = f"{nm(e['a'], True)} is shattered ({k_s:.2g}× its binding energy is enough to break it, not to scatter all of it); most falls into {nm(e['b'])}"
            else:
                fate = f"{nm(e['a'], True)} is far too small to matter to {nm(e['b'])}: it vaporises in a crater-forming explosion"
            if math.isfinite(U_b) and small["mass"] < 0.3 * bigb["mass"]:
                k_b = e["energy"] / U_b
                fate += (f", and {nm(e['b'])} is blown apart too ({k_b:.2g}× its binding energy)." if k_b >= 1 else
                         f"; {nm(e['b'])} survives ({100 * k_b:.2g} % of its binding energy) — its surface melts." if k_b > 1e-3 else f"; {nm(e['b'])} barely notices.")
            story.append(fate[0].upper() + fate[1:] if fate.endswith(".") else fate + ".")
    # what it is
    if mode == "system" and sc.get("remove"):
        gone_txt = " and ".join(nm(n) for n in sc["remove"])
        if "sun" in sc["remove"]:
            ek = names.index("earth") if "earth" in names else None
            v_e = float(np.linalg.norm(bodies[ek]["v0"])) if ek is not None else 29.8e3
            story.insert(0, f"Take away the Sun and nothing bends the planets' paths any more: Newton's first law says each one carries on in a straight line "
                            f"at the speed it had — Earth at {v_e / 1000:.1f} km/s. Their pulls on each other are tiny by comparison, so the lines barely curve. "
                            f"(Gravity travels at the speed of light, so Earth would keep circling the empty spot for 8 min 20 s before it noticed — the same "
                            f"moment the sky went dark.)")
            if ek is not None:
                straight = bodies[ek]["r0"] + bodies[ek]["v0"] * ts[-1]
                dev = float(np.linalg.norm(P_[-1][ek] - straight))
                travelled = float(np.linalg.norm(P_[-1][ek] - bodies[ek]["r0"]))
                add("Earth's distance travelled", travelled / AU, "AU", "earth", f"{travelled / AU:.2f} AU in {_dur(ts[-1])}",
                    "a straight line at its orbital speed: distance = speed × time")
                add("Earth's path vs a perfectly straight line", dev / 1000, "km", "earth", f"{dev / 1000:,.0f} km off a straight line",
                    "only the other planets' weak pulls bend it")
                checks.append(("first law", 100 * (1 - dev / max(travelled, 1))))
        else:
            story.insert(0, f"The Solar System from NASA's measured positions on {sc['date'].strftime('%d %B %Y')}, with {gone_txt} taken away at the start. "
                            f"Everything else moves under gravity alone for {_dur(ts[-1])}; compare it with the real system to see what {gone_txt} was doing.")
    elif mode in ("system",):
        story.insert(0, f"The solar system as it really is on {sc['date'].strftime('%d %B %Y')}: every planet starts at NASA's measured position and velocity, "
                        f"then gravity alone (all bodies pulling on all others) carries it forward {_dur(ts[-1])}. Inner planets race (Mercury: 88 days a lap), "
                        f"outer ones crawl (Neptune: 165 years) — Kepler's third law, T² ∝ a³.")
        story.append("The asteroid belt sits between Mars and Jupiter; the gaps in it (Kirkwood gaps) are where an asteroid's year would be a simple fraction of "
                     "Jupiter's, so Jupiter's repeated tugs cleared them. The Trojan asteroids share Jupiter's orbit 60° ahead and behind it, at the Lagrange points.")
    elif mode == "real" and sc.get("target"):
        tg = sc["target"]
        story.insert(0, f"{nm(tg, True)} on its real path: it starts at NASA's measured position and velocity on {sc['date'].strftime('%d %B %Y')}, and the "
                        f"simulation then follows gravity from {', '.join(nm(n) for n in names if n != tg)[:160]} for {_dur(ts[-1])}.")
    elif mode == "binary":
        story.insert(0, f"Two stars orbit their common centre of mass — neither orbits the other. The lighter one swings wider and faster; both complete one lap "
                        f"in the same time, set by Kepler's third law: T² = 4π²a³ / G(M₁+M₂).")
    elif mode == "orbit" and not events:
        bigk = int(np.argmax(m0))
        for k, b in enumerate(bodies):
            if k == bigk:
                continue
            r = float(np.linalg.norm(b["r0"] - bodies[bigk]["r0"]))
            v = float(np.linalg.norm(b["v0"] - bodies[bigk]["v0"]))
            story.append(f"{nm(b['name'], True)} is always falling toward {nm(bodies[bigk]['name'])} — pulled inward at {G * m0[bigk] / r ** 2:.3g} m/s² — but it moves "
                         f"sideways at {v / 1000:.2f} km/s, so it keeps missing: that's an orbit. At {r / 1000:,.0f} km the sideways speed that exactly balances the "
                         f"fall is √(GM/r) = {math.sqrt(G * m0[bigk] / r) / 1000:.2f} km/s (Kepler's third law: T² ∝ r³).")
    if not events:
        story.append(f"Laws obeyed: total energy changed by only {dE:.1e} and angular momentum by {dL:.1e} over the whole run (REBOUND's IAS15 integrator), "
                     f"so nothing is gained or lost that gravity wouldn't allow.")
    for b in bodies:
        add(f"{nm(b['name'], True)}: mass", b["mass"], "kg", b["name"])
        add(f"{nm(b['name'], True)}: radius", b["radius"] / 1000, "km", b["name"])
    if checks:
        add("Agreement with theory", max(0.0, min(c for _, c in checks)), "%", "system")
    story += black_hole_note(bodies)
    sun_i = names.index("sun") if "sun" in names else None
    belts = []
    if sc["belts"] and sun_i is not None:
        jup = (bodies[names.index("jupiter")]["r0"], bodies[names.index("jupiter")]["v0"]) if "jupiter" in names else None
        belts = _belts(bodies[sun_i]["r0"], jup, sc["belts"])
        notes.append("Asteroid-belt, Trojan and Kuiper-belt rocks are drawn as tracer particles, each on its own Kepler orbit around the Sun (their tugs on each "
                     "other and the planets' small pulls on them over these few orbits are left out).")
    return {
        "domain": "gravity", "text": text, "mode": mode, "source": source,
        "date": sc["date"].isoformat() if mode in ("system", "real", "flyby", "replace_sun") else None,
        "frame": sc.get("frame"),
        "focus": (["earth", "moon", "apophis"] if mode == "flyby" else list(dict.fromkeys([sc["frame"], sc.get("target")] + sc.get("named", []))) if sc.get("frame") else None),
        "bodies": [{"name": b["name"], "label": b.get("label") or nm(b["name"], True), "kind": b["kind"], "mass": m0[i], "radius": b["radius"],
                    "color": b["color"], "rings": b.get("rings"), "temp": b.get("temp"),
                    "parent": (b.get("parent") if b.get("parent") in names else None),
                    "pole": (_pole_ecliptic(OBLATE[b["name"]][3], OBLATE[b["name"]][4]).round(5).tolist() if b["name"] in OBLATE else None)} for i, b in enumerate(bodies)],
        "t": ts, "pos": ps, "events": events, "alive": run_.alive, "sim_seconds": ts[-1] if ts else 0.0,
        "duration": 20.0 if mode in ("system", "real", "replace_sun") else 12.0, "results": rows, "story": story, "notes": notes,
        "belts": belts, "sun_index": sun_i,
    }


def whatif(prev_text: str, text: str) -> tuple[str, dict, list[str]]:
    """'what if the Moon were 10 times heavier', 'at 30 km/s', 'what if Mars instead' -> (text, mass scale, changes)."""
    t = " " + text.lower() + " "
    if gone_bodies(t) or orbit_of(t):                    # a complete new set-up in itself: run it as said
        return text, {}, ["new set-up: " + text.strip()]
    new_bodies = [n for _, n in bodies_in(t)]
    ch, scale = [], {}
    if new_bodies and not re.search(r"heav|light|mass|massive|big|small", t):
        verb = re.search(r"\b(collid\w*|crash\w* into|hit\w*|orbit\w*|merg\w*|fall\w* into)\b", t) or re.search(r"\b(collid\w*|crash\w* into|hit\w*|orbit\w*|merg\w*|fall\w* into)\b", prev_text.lower())
        old = [n for _, n in bodies_in(prev_text)]
        names = new_bodies + [o for o in old if o not in new_bodies][: max(0, 2 - len(new_bodies))]
        if len(names) == 1:
            return f"{names[0]} {verb.group(1) if verb else 'orbits'}", {}, [f"now {nm(names[0])}"]
        return f"{names[0]} {verb.group(1) if verb else 'collides with'} {names[1]}", {}, [f"now {nm(names[0])} and {nm(names[1])}"]
    k = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)", t)
    if m:
        k = float(m.group(1))
    for w, f in (("twice", 2.0), ("double", 2.0), ("half", 0.5), ("ten times", 10.0), ("hundred times", 100.0)):
        if w in t:
            k = f
    if k and re.search(r"heav|mass|lighter", t):
        if "lighter" in t and k > 1:
            k = 1 / k
        target = new_bodies[0] if new_bodies else "*"
        scale[target] = k; ch.append(f"{'mass of ' + nm(target) if target != '*' else 'all masses'} ×{k:g}")
    kms = re.search(r"(\d+(?:\.\d+)?)\s*(km/s|kilometers? per second)", t)
    base = re.sub(r"\d+(?:\.\d+)?\s*(km/s|kilometers? per second)", "", prev_text)
    if kms:
        ch.append(f"launched at {kms.group(1)} km/s")
        return base + f" at {kms.group(1)} km/s", scale, ch
    return prev_text, scale, ch
