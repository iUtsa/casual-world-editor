"""Where everything in the sky really is, and what it really is.

* A catalogue of real bodies - the Sun, planets, dwarf planets, major moons, asteroids, comets, famous stars,
  black holes and galaxies - with measured masses (from their gravitational parameters GM, which are known far
  better than G·M separately), radii, colours and kinds.
* Real positions and velocities on any date from NASA JPL Horizons (the DE441 ephemeris that spacecraft are
  navigated with), cached on disk so a demo works offline once a scenario has been run.
* Offline fallback: JPL's "Keplerian elements for approximate positions of the major planets" (Standish,
  valid 1800-2050) and published mean orbits for moons/comets.

Units: SI (m, kg, s). Frame: ecliptic J2000, origin at the solar-system barycentre.
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np

from . import config

log = logging.getLogger(__name__)
G = 6.67430e-11
AU = 1.495978707e11
DAY = 86400.0
YEAR = 365.25 * DAY
MSUN = 1.98847e30
RSUN = 6.957e8
LY = 9.4607e15
KPC = 3.0857e19
CACHE = config.DATA_DIR / "horizons"


def _gm(gm_km3s2: float) -> float:
    return gm_km3s2 * 1e9 / G


# key: kind, mass (kg), radius (m), colour, Horizons command, parent, extra
CATALOG: dict[str, dict] = {
    "sun": dict(kind="star", mass=_gm(1.32712440018e11), radius=6.957e8, color="#ffd27a", hz="10", temp=5772),
    "mercury": dict(kind="planet", mass=_gm(22031.868), radius=2.4397e6, color="#a8a29e", hz="199", parent="sun"),
    "venus": dict(kind="planet", mass=_gm(324858.592), radius=6.0518e6, color="#e8c98f", hz="299", parent="sun"),
    "earth": dict(kind="planet", mass=_gm(398600.4354), radius=6.371e6, color="#3b82f6", hz="399", parent="sun"),
    "moon": dict(kind="moon", mass=_gm(4902.800), radius=1.7374e6, color="#cbd5e1", hz="301", parent="earth"),
    "mars": dict(kind="planet", mass=_gm(42828.375), radius=3.3895e6, color="#c2410c", hz="499", parent="sun"),
    "phobos": dict(kind="moon", mass=1.0659e16, radius=1.127e4, color="#78716c", hz="401", parent="mars"),
    "deimos": dict(kind="moon", mass=1.4762e15, radius=6.2e3, color="#a8a29e", hz="402", parent="mars"),
    "jupiter": dict(kind="planet", mass=_gm(126686531.9), radius=6.9911e7, color="#d8b48a", hz="599", parent="sun"),
    "io": dict(kind="moon", mass=_gm(5959.9), radius=1.8216e6, color="#facc15", hz="501", parent="jupiter"),
    "europa": dict(kind="moon", mass=_gm(3202.7), radius=1.5608e6, color="#e7e5e4", hz="502", parent="jupiter"),
    "ganymede": dict(kind="moon", mass=_gm(9887.8), radius=2.6341e6, color="#a8a29e", hz="503", parent="jupiter"),
    "callisto": dict(kind="moon", mass=_gm(7179.3), radius=2.4103e6, color="#78716c", hz="504", parent="jupiter"),
    "saturn": dict(kind="planet", mass=_gm(37931187.0), radius=5.8232e7, color="#e9d8a6", hz="699", parent="sun", rings=[7.4e7, 1.37e8]),
    "titan": dict(kind="moon", mass=_gm(8978.1), radius=2.5747e6, color="#f59e0b", hz="606", parent="saturn"),
    "enceladus": dict(kind="moon", mass=_gm(7.21), radius=2.521e5, color="#f8fafc", hz="602", parent="saturn"),
    "rhea": dict(kind="moon", mass=_gm(153.9), radius=7.638e5, color="#d6d3d1", hz="605", parent="saturn"),
    "uranus": dict(kind="planet", mass=_gm(5793939.0), radius=2.5362e7, color="#9be3ec", hz="799", parent="sun", rings=[3.8e7, 5.1e7]),
    "titania": dict(kind="moon", mass=_gm(228.2), radius=7.884e5, color="#d6d3d1", hz="703", parent="uranus"),
    "neptune": dict(kind="planet", mass=_gm(6836529.0), radius=2.4622e7, color="#3b6cf6", hz="899", parent="sun"),
    "triton": dict(kind="moon", mass=_gm(1427.6), radius=1.3534e6, color="#e2e8f0", hz="801", parent="neptune"),
    "pluto": dict(kind="dwarf planet", mass=_gm(869.6), radius=1.1883e6, color="#d6c7a1", hz="999", parent="sun"),
    "charon": dict(kind="moon", mass=_gm(106.1), radius=6.06e5, color="#a8a29e", hz="901", parent="pluto"),
    "ceres": dict(kind="dwarf planet", mass=_gm(62.63), radius=4.697e5, color="#9ca3af", hz="1;", parent="sun"),
    "eris": dict(kind="dwarf planet", mass=_gm(1108.0), radius=1.163e6, color="#e5e7eb", hz="136199;", parent="sun"),
    "haumea": dict(kind="dwarf planet", mass=_gm(267.4), radius=8.16e5, color="#f1f5f9", hz="136108;", parent="sun"),
    "vesta": dict(kind="asteroid", mass=_gm(17.29), radius=2.627e5, color="#a8a29e", hz="4;", parent="sun"),
    "pallas": dict(kind="asteroid", mass=_gm(13.6), radius=2.56e5, color="#9ca3af", hz="2;", parent="sun"),
    "apophis": dict(kind="asteroid", mass=6.1e10, radius=170.0, color="#78716c", hz="99942;", parent="sun"),
    "bennu": dict(kind="asteroid", mass=7.329e10, radius=245.0, color="#57534e", hz="101955;", parent="sun"),
    "halley": dict(kind="comet", mass=2.2e14, radius=5.5e3, color="#e0f2fe", hz="DES=1P;CAP;NOFRAG", parent="sun", label="Halley's Comet"),
    "hale-bopp": dict(kind="comet", mass=1.3e16, radius=3.0e4, color="#e0f2fe", hz="DES=C/1995 O1;CAP;NOFRAG", parent="sun", label="Comet Hale-Bopp"),
    "encke": dict(kind="comet", mass=9.2e13, radius=2.4e3, color="#e0f2fe", hz="DES=2P;CAP;NOFRAG", parent="sun", label="Comet Encke"),
    "67p": dict(kind="comet", mass=1.0e13, radius=2.0e3, color="#e0f2fe", hz="DES=67P;CAP;NOFRAG", parent="sun", label="Comet 67P/Churyumov-Gerasimenko"),
    "swift-tuttle": dict(kind="comet", mass=3.0e15, radius=1.3e4, color="#e0f2fe", hz="DES=109P;CAP;NOFRAG", parent="sun", label="Comet Swift-Tuttle"),
}
ALIASES = {
    "the sun": "sun", "sol": "sun", "the moon": "moon", "luna": "moon", "planet earth": "earth", "halley's comet": "halley", "halleys comet": "halley",
    "comet halley": "halley", "halley comet": "halley", "hale bopp": "hale-bopp", "comet hale-bopp": "hale-bopp", "swift tuttle": "swift-tuttle",
    "churyumov": "67p", "comet 67p": "67p", "99942 apophis": "apophis", "101955 bennu": "bennu",
}

# famous stars: mass (Msun), radius (Rsun), surface temperature (K), distance (light years)
STARS = {
    "proxima centauri": (0.1221, 0.1542, 3042, 4.24), "alpha centauri a": (1.079, 1.2175, 5790, 4.37), "alpha centauri b": (0.909, 0.8591, 5260, 4.37),
    "sirius a": (2.063, 1.711, 9940, 8.6), "sirius b": (1.018, 0.0084, 25000, 8.6), "sirius": (2.063, 1.711, 9940, 8.6),
    "betelgeuse": (16.5, 764.0, 3600, 548), "rigel": (21.0, 78.9, 12100, 860), "vega": (2.135, 2.36, 9602, 25.0),
    "polaris": (5.4, 37.5, 6015, 433), "antares": (12.0, 680.0, 3660, 550), "aldebaran": (1.16, 45.1, 3910, 65.3),
    "arcturus": (1.08, 25.4, 4286, 36.7), "uy scuti": (7.0, 909.0, 3365, 9500), "canopus": (8.0, 71.0, 7350, 310),
    "barnard's star": (0.144, 0.196, 3134, 5.96), "tau ceti": (0.783, 0.793, 5344, 11.9),
}
# black holes: mass (Msun), distance (ly)
BLACK_HOLES = {"sagittarius a*": (4.297e6, 26670), "sgr a*": (4.297e6, 26670), "m87*": (6.5e9, 5.35e7), "ton 618": (6.6e10, 1.04e10),
               "cygnus x-1": (21.2, 7240), "gaia bh1": (9.62, 1560), "black hole": (10.0, None)}
# binary stars with measured orbits: (star A, star B, period years, eccentricity)
BINARIES = {"alpha centauri": ("alpha centauri a", "alpha centauri b", 79.91, 0.5208), "sirius": ("sirius a", "sirius b", 50.13, 0.5914)}

# JPL approximate elements, J2000 ecliptic (Standish): a [AU], e, I, L, long.peri, long.node [deg] and rates per century
_ELEM = {
    "mercury": ([0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
    "venus": ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
              [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
    "earth": ([1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
              [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]),
    "mars": ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
             [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
    "jupiter": ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
    "saturn": ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
               [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
    "uranus": ([19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
               [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]),
    "neptune": ([30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
                [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]),
    "pluto": ([39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684],
              [-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482]),
}
# mean orbits about the parent for offline use: a [m], e, i [deg], period [days]
_MOON_ORBITS = {"moon": (3.844e8, 0.0549, 5.145, 27.3217), "phobos": (9.376e6, 0.0151, 1.08, 0.31891), "deimos": (2.3463e7, 0.00033, 1.79, 1.263),
                "io": (4.217e8, 0.0041, 0.05, 1.769), "europa": (6.709e8, 0.009, 0.47, 3.551), "ganymede": (1.0704e9, 0.0013, 0.2, 7.155),
                "callisto": (1.8827e9, 0.0074, 0.19, 16.689), "titan": (1.22187e9, 0.0288, 0.35, 15.945), "enceladus": (2.3802e8, 0.0047, 0.02, 1.37),
                "rhea": (5.2704e8, 0.001, 0.35, 4.518), "titania": (4.359e8, 0.0011, 0.34, 8.706), "triton": (3.547e8, 0.000016, 156.9, 5.877),
                "charon": (1.9591e7, 0.0002, 0.08, 6.387)}
# small bodies: a [AU], e, i, node, arg.peri [deg], time of perihelion (JD TDB)
_SMALL = {"halley": (17.834, 0.96714, 162.26, 58.42, 111.33, 2446470.5), "ceres": (2.7675, 0.0785, 10.59, 80.30, 73.6, 2458238.8),
          "vesta": (2.3615, 0.0887, 7.14, 103.85, 151.2, 2458243.4), "pallas": (2.773, 0.2302, 34.84, 173.0, 310.2, 2458320.0),
          "encke": (2.215, 0.8483, 11.78, 334.57, 186.54, 2460229.0), "hale-bopp": (186.0, 0.99508, 89.43, 282.47, 130.59, 2450539.6),
          "apophis": (0.9224, 0.1914, 3.339, 203.96, 126.6, 2459584.6), "bennu": (1.126, 0.2037, 6.035, 2.06, 66.22, 2459600.0),
          "eris": (67.86, 0.4361, 44.04, 35.95, 151.64, 2545575.0), "haumea": (43.12, 0.1955, 28.2, 121.8, 240.9, 2500000.0),
          "67p": (3.4628, 0.6405, 7.04, 50.14, 12.78, 2459518.5), "swift-tuttle": (26.09, 0.9632, 113.45, 139.38, 152.98, 2448968.5)}


def jd(d: date | datetime) -> float:
    d0 = datetime(2000, 1, 1, 12)
    dt = (datetime(d.year, d.month, d.day) if not isinstance(d, datetime) else d) - d0
    return 2451545.0 + dt.total_seconds() / DAY


def resolve(name: str) -> str | None:
    n = name.lower().strip()
    n = ALIASES.get(n, n)
    return n if n in CATALOG else None


# ------------------------------------------------------------------------------------------- JPL Horizons
def _horizons(cmd: str, d: date) -> tuple[np.ndarray, np.ndarray] | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9]+", "_", cmd) + f"_{d.isoformat()}.json"
    f = CACHE / key
    if f.exists():
        v = json.loads(f.read_text())
        return np.array(v["r"]), np.array(v["v"])
    q = {"format": "json", "COMMAND": f"'{cmd}'", "OBJ_DATA": "'NO'", "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'",
         "CENTER": "'500@0'", "START_TIME": f"'{d.isoformat()}'", "STOP_TIME": f"'{(d + timedelta(days=1)).isoformat()}'",
         "STEP_SIZE": "'1d'", "VEC_TABLE": "'2'", "OUT_UNITS": "'KM-S'", "REF_PLANE": "'ECLIPTIC'", "REF_SYSTEM": "'ICRF'"}
    url = "https://ssd.jpl.nasa.gov/api/horizons.api?" + "&".join(f"{k}={urllib.parse.quote(v)}" for k, v in q.items())
    res = ""
    for attempt in range(4):                     # Horizons throttles bursts: retry with a pause
        try:
            res = json.loads(urllib.request.urlopen(url, timeout=25).read()).get("result", "")
        except Exception as e:                   # offline: the caller falls back to orbital elements
            log.info("Horizons unavailable for %s: %s", cmd, e)
            if "urlopen error" in str(e):
                return None
        if "$$SOE" in res:
            break
        import time as _t
        _t.sleep(0.8 * (attempt + 1))
    blk = res[res.find("$$SOE"):res.find("$$EOE")]
    num = r"([-+]?\d+\.\d+E[-+]\d+)"
    m = re.search(rf"X\s*=\s*{num}\s*Y\s*=\s*{num}\s*Z\s*=\s*{num}\s*VX\s*=\s*{num}\s*VY\s*=\s*{num}\s*VZ\s*=\s*{num}", blk)
    if not m:
        log.info("Horizons gave no vectors for %s", cmd)
        return None
    vals = [float(x) * 1000 for x in m.groups()]
    f.write_text(json.dumps({"r": vals[:3], "v": vals[3:], "cmd": cmd, "date": d.isoformat()}))
    return np.array(vals[:3]), np.array(vals[3:])


def states(names: list[str], d: date) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str]:
    """Barycentric (r, v) for every name on date d: NASA JPL Horizons, else the offline orbital elements."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(3) as ex:
        res = dict(zip(names, ex.map(lambda n: _horizons(CATALOG[n]["hz"], d), names)))
    src = "NASA JPL Horizons (DE441)"
    for n in names:
        if res[n] is not None:
            out[n] = res[n]
    missing = [n for n in names if n not in out]
    if missing:
        snap = _from_snapshot(missing, d)
        if snap is not None:
            out.update(snap[0])
            src = snap[1] if len(missing) == len(names) else src + f" + {snap[1]} for {', '.join(missing)}"
        else:
            fb = _fallback(names, d)
            for n in missing:
                out[n] = fb[n]
            src = "JPL approximate orbital elements (offline)" if len(missing) == len(names) else src + f" + offline elements for {', '.join(missing)}"
    return out, src


SNAPSHOT = CACHE / "snapshot.json"


def bake_snapshot(d: date) -> int:
    """Save NASA's state of every catalogued body on date d, for offline use (see _from_snapshot)."""
    names = list(CATALOG)
    with ThreadPoolExecutor(3) as ex:
        res = dict(zip(names, ex.map(lambda n: _horizons(CATALOG[n]["hz"], d), names)))
    got = {n: {"r": v[0].tolist(), "v": v[1].tolist()} for n, v in res.items() if v is not None}
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps({"date": d.isoformat(), "bodies": got}))
    return len(got)


def _from_snapshot(names: list[str], d: date):
    """Offline: start from NASA's saved snapshot and integrate everything (REBOUND IAS15) to date d."""
    if not SNAPSHOT.exists():
        return None
    snap = json.loads(SNAPSHOT.read_text())
    if not all(n in snap["bodies"] for n in names):
        return None
    import rebound
    d0 = date.fromisoformat(snap["date"])
    perturbers = ["sun", "mercury", "venus", "earth", "moon", "mars", "jupiter", "saturn", "uranus", "neptune"]
    use = [n for n in dict.fromkeys(perturbers + names) if n in snap["bodies"]]
    sim = rebound.Simulation()
    sim.G = G
    sim.integrator = "ias15"
    for n in use:
        b = snap["bodies"][n]
        sim.add(m=CATALOG[n]["mass"], x=b["r"][0], y=b["r"][1], z=b["r"][2], vx=b["v"][0], vy=b["v"][1], vz=b["v"][2])
    sim.integrate((jd(d) - jd(d0)) * DAY, exact_finish_time=1)
    out = {n: (np.array(sim.particles[i].xyz), np.array(sim.particles[i].vxyz)) for i, n in enumerate(use) if n in names}
    return out, f"NASA JPL snapshot of {d0.isoformat()} integrated to {d.isoformat()} (offline)"


# ------------------------------------------------------------------------------------------- offline fallback
def _kepler_state(a, e, i, node, argp, M, mu):
    """Position/velocity (m, m/s) in the ecliptic frame from classical elements (angles in rad)."""
    E = M if e < 0.8 else math.pi
    for _ in range(60):
        E -= (E - e * math.sin(E) - M) / (1 - e * math.cos(E))
    nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
    r = a * (1 - e * math.cos(E))
    p = a * (1 - e * e)
    h = math.sqrt(mu * p)
    rp = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    vp = np.array([-mu / h * math.sin(nu), mu / h * (e + math.cos(nu)), 0.0])
    cO, sO, ci, si, cw, sw = math.cos(node), math.sin(node), math.cos(i), math.sin(i), math.cos(argp), math.sin(argp)
    R = np.array([[cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
                  [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
                  [sw * si, cw * si, ci]])
    return R @ rp, R @ vp


def elements_state(n: str, d: date) -> tuple[np.ndarray, np.ndarray]:
    """Heliocentric (planets, small bodies) or parent-centric (moons) state from published elements."""
    mu_sun = G * CATALOG["sun"]["mass"]
    T = (jd(d) - 2451545.0) / 36525.0
    if n in _ELEM:
        el, rate = _ELEM[n]
        a, e, I, L, wbar, node = [x + r * T for x, r in zip(el, rate)]
        M = math.radians(((L - wbar) + 180) % 360 - 180)
        return _kepler_state(a * AU, e, math.radians(I), math.radians(node), math.radians(wbar - node), M, mu_sun + G * CATALOG[n]["mass"])
    if n in _SMALL:
        a, e, i, node, argp, tp = _SMALL[n]
        aa = a * AU
        nmot = math.sqrt(mu_sun / aa ** 3)
        M = (nmot * (jd(d) - tp) * DAY + math.pi) % (2 * math.pi) - math.pi
        return _kepler_state(aa, e, math.radians(i), math.radians(node), math.radians(argp), M, mu_sun)
    if n in _MOON_ORBITS:
        a, e, i, P = _MOON_ORBITS[n]
        par = CATALOG[n]["parent"]
        mu = G * (CATALOG[par]["mass"] + CATALOG[n]["mass"])
        M = 2 * math.pi * ((jd(d) - 2451545.0) / P % 1.0)
        return _kepler_state(a, e, math.radians(i), 0.0, 0.0, M, mu)
    return np.zeros(3), np.zeros(3)


def _fallback(names: list[str], d: date) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    helio: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for n in names:
        if n == "sun":
            continue
        r, v = elements_state(n, d)
        par = CATALOG[n].get("parent")
        if par and par != "sun":
            pr, pv = elements_state(par, d)
            r, v = r + pr, v + pv
        helio[n] = (r, v)
    # barycentric: shift so the centre of mass is at rest at the origin
    ms = {n: CATALOG[n]["mass"] for n in names}
    helio["sun"] = (np.zeros(3), np.zeros(3))
    Mt = sum(ms.values())
    cr = sum(ms[n] * helio[n][0] for n in names) / Mt
    cv = sum(ms[n] * helio[n][1] for n in names) / Mt
    return {n: (helio[n][0] - cr, helio[n][1] - cv) for n in names}


def star_color(T: float) -> str:
    """Blackbody colour of a star's surface (approximation of the Planck locus)."""
    t = T / 100.0
    r = 255 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = [int(max(0, min(255, x))) for x in (r, g, b)]
    return "#%02x%02x%02x" % tuple(c)
