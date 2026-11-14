"""Electric circuits from words: "a 9 V battery with two bulbs in parallel", "LED and a 220 ohm resistor",
"RC circuit with a 100 µF capacitor and 1 kΩ resistor", "short circuit a battery".

Solved with modified nodal analysis (the method SPICE uses): node voltages + source currents from
Kirchhoff's current law; capacitors and inductors with backward-Euler companion models for transients;
LEDs as a forward-voltage drop with a small on-resistance (re-solved until each diode's state is consistent).
Batteries have real internal resistance, so a short circuit gives a large but finite current.

Every number reported comes from the solve; the layout is a schematic placed on a board for the 3D view.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

KEYWORDS = r"\b(circuit|resistor|resistance|ohm|ohms|Ω|kΩ|volt|volts|voltage|bulb|bulbs|lamp|lamps|led|leds|ammeter|voltmeter|capacitor|inductor|coil|switch|fuse|series|parallel|current|amps?|electric\w*|battery|batteries|cell|motor|buzzer|short circuit|kirchhoff|ohm's law|wire)\b"
STRONG = r"(\d\s*(k)?v\b)|\b(circuit|resistor|ohms?|Ω|volts?|voltage|capacitor|inductor|ammeter|voltmeter|in series|in parallel|short circuit|ohm's law|electric current|led)\b"


def is_circuit(text: str) -> bool:
    t = text.lower()
    strong = len(re.findall(STRONG, t))
    weak = len(re.findall(KEYWORDS, t))
    has_power = re.search(r"\b(batter\w*|cell|volt|supply|mains)\b", t) is not None
    has_load = re.search(r"\b(bulb|lamp|led|resistor|motor|buzzer|capacitor)\b", t) is not None
    throwing = re.search(r"\b(throw|drop|kick|hit|roll|smash)\w*\b", t) is not None
    return strong >= 1 and not throwing or (has_power and has_load and not throwing) or weak >= 3


@dataclass
class Part:
    kind: str                  # battery, resistor, bulb, led, switch, capacitor, inductor, motor, buzzer, fuse
    value: float               # V for battery, Ω for loads, F, H, A (fuse rating)
    label: str
    closed: bool = True        # switches
    color: str = "red"         # LEDs
    vf: float = 2.0
    rated_p: float = 0.0       # bulbs: rated power at rated voltage (for brightness)
    r_int: float = 0.0         # batteries
    capacity_mah: float = 0.0


@dataclass
class Spec:
    source: Part
    series: list = field(default_factory=list)          # items: Part or list[Part] (a parallel group)
    notes: list[str] = field(default_factory=list)


UNITS = {"": 1, "k": 1e3, "m": 1e-3, "µ": 1e-6, "u": 1e-6, "n": 1e-9, "p": 1e-12, "meg": 1e6, "M": 1e6}
WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "several": 3, "another": 1}


def _val(t: str, unit_pat: str):
    out = []
    for m in re.finditer(rf"(\d+(?:\.\d+)?)\s*(k|m|µ|u|n|p|meg|M)?\s*(?:{unit_pat})(?![a-z])", t, re.I):
        out.append((m.start(), float(m.group(1)) * UNITS.get(m.group(2) or "", 1)))
    return out


def _count_before(t: str, pos: int) -> int:
    pre = t[max(0, pos - 18):pos]
    m = re.search(r"(\d+|a|an|one|two|three|four|five|six|several|another)\s+(?:\w+\s+){0,2}$", pre)
    if not m:
        return 1
    w = m.group(1)
    return int(w) if w.isdigit() else WORDNUM.get(w, 1)


def parse(text: str) -> Spec:
    raw = text
    t = " " + text.replace("Ω", " ohm").replace("ω", " ohm") + " "
    tl = t.lower()
    notes: list[str] = []
    volts = _val(t, r"v\b|volts?|volt")
    volts = [(p, v * (1000 if v < 1 and re.search(r"\d\s*k\s*v", tl) else 1)) for p, v in volts]
    ohms = _val(tl, r"ohms?|Ω")
    farads = _val(t, r"f\b|farads?")
    henries = _val(t, r"h\b|henr\w*")
    amps = _val(tl, r"a\b|amps?")
    # source
    if re.search(r"\b(aa|aaa)\b|\bcell\b", tl) and not volts:
        V, rint, cap, lab = 1.5, 0.15, 2500, "1.5 V cell"
    elif re.search(r"mains|wall socket|outlet", tl):
        V, rint, cap, lab = 230.0, 0.5, 0, "230 V mains (AC, shown as its RMS value)"
        notes.append("Mains is AC; the circuit is solved with its RMS value (230 V), which gives the same average power.")
    else:
        V = volts[0][1] if volts else 9.0
        rint = 1.5 if abs(V - 9) < 0.6 else 0.15 * max(1.0, V / 1.5) if V <= 12 else 0.05
        cap = 500 if abs(V - 9) < 0.6 else 2500
        lab = f"{V:g} V battery" if V <= 48 else f"{V:g} V supply"
    ncells = 1
    m = re.search(r"(\d+|two|three|four)\s+(?:aa\s+)?(batter\w*|cells)\b", tl)
    if m:
        ncells = int(m.group(1)) if m.group(1).isdigit() else WORDNUM[m.group(1)]
        if not volts:
            V *= ncells; rint *= ncells
            lab = f"{ncells} × {V / ncells:g} V in series = {V:g} V"
    source = Part("battery", V, lab, r_int=rint, capacity_mah=cap)

    parts: list[tuple[int, Part, int]] = []           # (position, part, count)
    bulb_ohm_pos = {p for p, v in ohms if re.match(r"\s*(ohms?|Ω)?\s*(light ?)?(bulbs?|lamps?)", re.sub(r"^\d+(\.\d+)?\s*(k|m)?", "", tl[p:p + 30]).replace("ohms", "", 1).replace("ohm", "", 1))}
    def add(pos, part, count=None):
        parts.append((pos, part, count if count is not None else _count_before(tl, pos)))
    used_ohm: set[int] = set()
    for m in re.finditer(r"\b(resistors?|resistance)\b", tl):
        n_ = _count_before(tl, m.start())
        after = [(p, v) for p, v in ohms if 0 < p - m.start() < 90 and p not in used_ohm]
        if n_ > 1 and len(after) >= n_:
            # "two resistors, one with 100 ohms and the other with 220 ohms": one value each
            for p, v in after[:n_]:
                used_ohm.add(p)
                add(m.start() + 1 + len(parts), Part("resistor", v, ""), 1)
            continue
        near = sorted([(p, v) for p, v in ohms if abs(p - m.start()) < 30 and p not in used_ohm and p not in bulb_ohm_pos],
                      key=lambda pv: abs(pv[0] - m.start()))
        if near:
            used_ohm.add(near[0][0])
        add(m.start(), Part("resistor", near[0][1] if near else 100.0, ""))
    for p, v in ohms:                                   # "a 100 ohm and a 300 ohm resistor": every value is a part
        if p in bulb_ohm_pos or p in used_ohm:
            continue
        add(p, Part("resistor", v, ""))
    rb = max(1.0, V * V / 3.0)            # a 3 W bulb made for this battery's voltage
    for m in re.finditer(r"\b(bulbs?|lamps?|light ?bulbs?)\b", tl):
        near = [(p, v) for p, v in ohms if 0 < m.start() - p < 14]   # "a 27 ohm bulb"
        if near:
            bulb_ohm_pos.add(near[-1][0])
        rv = near[-1][1] if near else rb
        add(m.start(), Part("bulb", rv, "", rated_p=3.0))       # rated 3 W at the voltage it was made for
    for m in re.finditer(r"\b(red|green|blue|white|yellow)?\s*(leds?|light[- ]emitting diodes?)\b", tl):
        col = m.group(1) or "red"
        add(m.start(), Part("led", 10.0, "", color=col, vf={"red": 2.0, "yellow": 2.1, "green": 2.2, "blue": 3.1, "white": 3.1}[col]))
    for m in re.finditer(r"\b(capacitors?)\b", tl):
        near = [v for p, v in farads if abs(p - m.start()) < 30]
        add(m.start(), Part("capacitor", near[0] if near else 100e-6, ""))
    for m in re.finditer(r"\b(inductors?|coils?)\b", tl):
        near = [v for p, v in henries if abs(p - m.start()) < 30]
        add(m.start(), Part("inductor", near[0] if near else 0.1, ""))
    for m in re.finditer(r"\b(motors?)\b", tl):
        add(m.start(), Part("motor", 6.0, ""))
    for m in re.finditer(r"\b(buzzers?)\b", tl):
        add(m.start(), Part("buzzer", 150.0, ""))
    for m in re.finditer(r"\b(fuses?)\b", tl):
        near = [v for p, v in amps if abs(p - m.start()) < 25]
        add(m.start(), Part("fuse", near[0] if near else 1.0, "", ), 1)
    switch_open = re.search(r"\b(open|opened|off)\s+(the\s+)?switch|switch\s+(is\s+)?(open|off)\b", tl) is not None
    if re.search(r"\bswitch", tl):
        add(tl.index("switch"), Part("switch", 0.0, "", closed=not switch_open), 1)
    if re.search(r"short[- ]?circuit|wire (straight )?across|connect (the )?(two )?terminals", tl):
        parts.append((0, Part("wire", 0.01, "a plain wire (short circuit)"), 1))
        notes.append("A short circuit: almost nothing but the battery's own internal resistance limits the current.")
    if not [p for p in parts if p[1].kind not in ("switch", "fuse")]:
        add(len(tl), Part("bulb", rb, "", rated_p=3.0), 1)
        notes.append("No load was named, so a 3 W bulb is in the circuit (a battery with nothing connected does nothing).")
    if any(p[1].kind == "led" for p in parts) and not any(p[1].kind in ("resistor", "bulb") for p in parts) and V > 3.5:
        notes.append("An LED straight across a battery has nothing to limit its current — real LEDs burn out like this. Add a resistor (e.g. 'with a 330 ohm resistor').")
    parts.sort(key=lambda x: x[0])
    # topology: "N things in parallel" -> that group in parallel; plain "parallel" -> all loads in parallel
    loads = [(pos, p, c) for pos, p, c in parts if p.kind not in ("switch", "fuse")]
    others = [(pos, p, c) for pos, p, c in parts if p.kind in ("switch", "fuse")]
    series: list = []
    par_all = re.search(r"\bparallel\b", tl) and not re.search(r"\bin series with\b", tl)
    # "... with a 20 ohm and a 30 ohm resistor in parallel": "in parallel" groups the parts of the clause it ends
    groups: list[set[int]] = []
    for g in [m.start() for m in re.finditer(r"\b(?:in|are|connected in|wired in) parallel\b", tl)]:
        bounds = [m.end() for m in re.finditer(r"\bin series with\b|\bwith\b|[,;]|\bthen\b|\bfollowed by\b", tl[:g])]
        start = bounds[-1] if bounds else 0
        members = {i for i, (pos, p, c) in enumerate(loads) if start <= pos < g}
        if sum(loads[i][2] for i in members) >= 2:
            groups.append(members)
    copies_of = []
    for pos, p, c in loads:
        copies = []
        for i in range(max(1, min(c, 12))):
            q = Part(**{**p.__dict__})
            q.label = _label(q, i, c)
            copies.append(q)
        copies_of.append(copies)
    done: set[int] = set()
    for i, copies in enumerate(copies_of):
        if i in done:
            continue
        grp = next((g for g in groups if i in g), None)
        if grp is not None:
            series.append([q for j in sorted(grp) for q in copies_of[j]])
            done |= grp
        else:
            series.extend(copies)
    if par_all and not groups and len(loads) > 1:
        series = [[x for x in series]]
    for pos, p, c in others:
        p.label = _label(p, 0, 1)
        series.append(p)
    return Spec(source=source, series=series, notes=notes)


RESISTIVE = ("resistor", "bulb", "motor", "buzzer", "wire")


def _r_eq(series) -> float | None:
    """Equivalent resistance by series/parallel reduction (None if something isn't a plain resistance)."""
    R = 0.0
    for item in series:
        group = item if isinstance(item, list) else [item]
        Rs = []
        for p in group:
            if p.kind in RESISTIVE:
                Rs.append(max(p.value, 1e-6))
            elif p.kind in ("switch", "fuse"):
                Rs.append(1e-3 if (p.kind == "fuse" or p.closed) else math.inf)
            else:
                return None
        g = sum(0.0 if math.isinf(r) else 1.0 / r for r in Rs)
        R += math.inf if g == 0 else 1.0 / g
    return R + 1e-4                                      # + the return wire


def _series_rc(series):
    """(resistance in the loop, C) for one capacitor in series with plain resistances; else None."""
    caps = [p for p in series if not isinstance(p, list) and p.kind == "capacitor"]
    if len(caps) != 1 or any(isinstance(x, list) and any(q.kind == "capacitor" for q in x) for x in series):
        return None
    rest = _r_eq([x for x in series if x is not caps[0]])
    return None if rest is None or not math.isfinite(rest) else (rest, caps[0].value)


def _label(p: Part, i: int, n: int) -> str:
    base = {"resistor": f"{_si(p.value, 'Ω')} resistor", "bulb": "bulb", "led": f"{p.color} LED", "capacitor": f"{_si(p.value, 'F')} capacitor",
            "inductor": f"{_si(p.value, 'H')} inductor", "motor": "motor", "buzzer": "buzzer", "fuse": f"{p.value:g} A fuse",
            "switch": "switch" + ("" if p.closed else " (open)"), "wire": "wire"}[p.kind]
    return f"{base} {i + 1}" if n > 1 else base


def _si(v: float, unit: str) -> str:
    if abs(v) < 1e-9:
        return f"0 {unit}"
    if unit == "h":
        return f"{v:.1f} h" if v < 48 else f"{v / 24:.1f} days"
    if unit == "%":
        return f"{v:.0f}%"
    for k, pre in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")):
        if abs(v) >= k:
            x = v / k
            return f"{x:.3g} {pre}{unit}"
    return f"{v:.3g} {unit}"


# ---------------------------------------------------------------- netlist + MNA
def netlist(spec: Spec):
    """Elements as (id, part, n1, n2). Node 0 = battery negative. Battery: + at node 1 via its internal resistance."""
    els = []
    els.append(("src", spec.source, 2, 0))                # ideal source: node 2 (+) to 0
    els.append(("rint", Part("resistor", max(spec.source.r_int, 1e-3), "internal resistance"), 2, 1))
    node = 1
    nxt = 3
    for item in spec.series:
        a = node
        b = nxt; nxt += 1
        if isinstance(item, list):
            for k, p in enumerate(item):
                els.append((f"e{len(els)}", p, a, b))
        else:
            els.append((f"e{len(els)}", item, a, b))
        node = b
    # close the loop: last node back to 0
    els.append(("ret", Part("wire", 1e-4, "return wire"), node, 0))
    return els, nxt


def solve(spec: Spec, t_end: float | None = None):
    els, n_nodes = netlist(spec)
    has_dyn = any(p.kind in ("capacitor", "inductor") for _, p, _, _ in els)
    taus = []
    R_loop = sum(p.value for _, p, _, _ in els if p.kind in ("resistor", "bulb", "motor", "buzzer", "led", "wire")) + spec.source.r_int
    for _, p, _, _ in els:
        if p.kind == "capacitor":
            taus.append(max(1e-6, R_loop * p.value))
        if p.kind == "inductor":
            taus.append(max(1e-6, p.value / max(R_loop, 1e-3)))
    T = t_end or (5 * max(taus) if taus else 1.0)
    steps = 240 if has_dyn else 1
    sub = 20 if has_dyn else 1                 # 20 internal steps per recorded one: backward-Euler error ~0.05%
    dt = T / steps / sub
    led_on = {eid: True for eid, p, _, _ in els if p.kind == "led"}
    vc = {eid: 0.0 for eid, p, _, _ in els if p.kind == "capacitor"}
    il = {eid: 0.0 for eid, p, _, _ in els if p.kind == "inductor"}
    hist = []
    fuse_blown: set[str] = set()
    for kk in range((steps * sub + 1) if has_dyn else 1):
        k = kk // sub
        for _ in range(6):           # diode states / fuse until consistent
            sol = _mna(els, n_nodes, led_on, vc, il, dt if has_dyn else None, fuse_blown)
            change = False
            for eid, p, a, b in els:
                if p.kind == "led":
                    vd = sol["V"][a] - sol["V"][b]
                    on = vd >= p.vf - 1e-6 if not led_on[eid] else sol["I"][eid] > 0
                    if on != led_on[eid]:
                        led_on[eid] = on; change = True
                if p.kind == "fuse" and eid not in fuse_blown and abs(sol["I"][eid]) > p.value:
                    fuse_blown.add(eid); change = True
            if not change:
                break
        if has_dyn:
            for eid, p, a, b in els:
                if p.kind == "capacitor":
                    vc[eid] = sol["V"][a] - sol["V"][b]
                if p.kind == "inductor":
                    il[eid] = sol["I"][eid]
        if kk % sub == 0:
            hist.append({"t": kk * dt if has_dyn else 0.0, **sol})
    # the instant the switch closes (t = 0+): empty capacitors pass current like a wire, inductors block it
    start = _mna(els, n_nodes, {e: True for e in led_on}, {e: 0.0 for e in vc}, {e: 0.0 for e in il}, 1e-12, set()) if has_dyn else None
    return els, hist, {"tau": max(taus) if taus else None, "T": T, "led_on": led_on, "fuse_blown": sorted(fuse_blown), "start": start}


def _mna(els, n_nodes, led_on, vc, il, dt, blown):
    n = n_nodes - 1                      # unknown node voltages 1..n_nodes-1
    vsrc = [e for e in els if e[1].kind == "battery"]
    m = len(vsrc)
    A = np.zeros((n + m, n + m)); z = np.zeros(n + m)

    def stamp_g(a, b, g):
        if a: A[a - 1, a - 1] += g
        if b: A[b - 1, b - 1] += g
        if a and b: A[a - 1, b - 1] -= g; A[b - 1, a - 1] -= g

    def stamp_i(a, b, i):               # current i flowing from a to b through the element (source injects into b)
        if a: z[a - 1] -= i
        if b: z[b - 1] += i
    G_of = {}
    for eid, p, a, b in els:
        g = 0.0; ieq = 0.0
        if p.kind in ("resistor", "bulb", "motor", "buzzer", "wire"):
            g = 1.0 / max(p.value, 1e-6)
        elif p.kind == "switch":
            g = 1e3 if p.closed else 1e-12
        elif p.kind == "fuse":
            g = 1e-12 if eid in blown else 1e3
        elif p.kind == "led":
            if led_on[eid]:
                g = 1.0 / p.value
                ieq = -p.vf * g          # i = g (v - vf)  ->  conductance g plus a source of g*vf toward a
            else:
                g = 1e-9
        elif p.kind == "capacitor":
            if dt is None:
                g = 1e-12
            else:
                g = p.value / dt; ieq = -g * vc[eid]
        elif p.kind == "inductor":
            if dt is None:
                g = 1e3
            else:
                g = dt / p.value; ieq = il[eid]
        if p.kind != "battery":
            stamp_g(a, b, g)
            if ieq:
                stamp_i(a, b, ieq)
            G_of[eid] = (g, ieq)
    for k, (eid, p, a, b) in enumerate(vsrc):
        r = n + k
        if a: A[a - 1, r] += 1; A[r, a - 1] += 1
        if b: A[b - 1, r] -= 1; A[r, b - 1] -= 1
        z[r] = p.value
    x = np.linalg.solve(A + np.eye(n + m) * 1e-15, z)
    V = np.concatenate([[0.0], x[:n]])
    I = {}
    for eid, p, a, b in els:
        if p.kind == "battery":
            I[eid] = float(-x[n + vsrc.index((eid, p, a, b))])
        else:
            g, ieq = G_of[eid]
            I[eid] = float(g * (V[a] - V[b]) + ieq)
    return {"V": V.tolist(), "I": I}


# ---------------------------------------------------------------- layout for the 3D board
def layout(spec: Spec, els):
    """Schematic on a board (units = 1 grid cell). Battery on the left, loads along the top, return along the bottom."""
    items = spec.series
    W = max(4, len(items) * 2 + 2)
    H = 3
    pos = {}
    wires = []                                 # (points, element id whose current flows here)
    pos["src"] = {"x": 0, "y": H / 2, "dir": "v"}
    x = 1.0
    top = H
    wires.append(([[0, H / 2 + 0.6], [0, top], [x, top]], "src"))
    eid_iter = iter([e for e in els if e[0].startswith("e")])
    for item in items:
        if isinstance(item, list):
            k = len(item)
            xa, xb = x, x + 2
            for j in range(k):
                eid, p, a, b = next(eid_iter)
                yb = top + j * 1.2
                pos[eid] = {"x": (xa + xb) / 2, "y": yb, "dir": "h"}
                if j == 0:
                    wires.append(([[xa, top], [xa + 0.45, top]], eid)); wires.append(([[xb - 0.45, top], [xb, top]], eid))
                else:
                    wires.append(([[xa, top + (j - 1) * 1.2], [xa, yb], [xa + 0.45, yb]], eid))
                    wires.append(([[xb - 0.45, yb], [xb, yb], [xb, top + (j - 1) * 1.2]], eid))
            x = xb
        else:
            eid, p, a, b = next(eid_iter)
            pos[eid] = {"x": x + 1, "y": top, "dir": "h"}
            wires.append(([[x, top], [x + 0.55, top]], eid)); wires.append(([[x + 1.45, top], [x + 2, top]], eid))
            x += 2
    wires.append(([[x, top], [W, top], [W, 0], [0, 0], [0, H / 2 - 0.6]], "src"))
    return pos, wires, {"w": W, "h": H + 1.2 * max([len(i) - 1 for i in items if isinstance(i, list)] + [0])}


# ---------------------------------------------------------------- run + explain
def run(text: str) -> dict:
    spec = parse(text)
    els, hist, info = solve(spec)
    pos, wires, size = layout(spec, els)
    last = hist[-1]
    rows, story = [], list(spec.notes)
    src_I = abs(last["I"]["src"]) if abs(last["I"]["src"]) > 1e-9 else 0.0
    V = spec.source.value
    loads = [e for e in els if e[0].startswith("e")]

    def add(label, v, unit, about="circuit"):
        rows.append({"label": label, "value": float(v), "unit": unit, "about": about, "text": _si(v, unit) if unit else f"{v:.3g}"})
    add("Battery voltage (EMF)", V, "V", "battery")
    add("Current from the battery", src_I, "A", "battery")
    v_term = V - src_I * spec.source.r_int
    add("Voltage at the battery terminals", v_term, "V", "battery")
    add("Power delivered", v_term * src_I, "W", "battery")
    R_ext = v_term / src_I if src_I > 1e-9 else float("inf")
    if hist[0]["t"] != hist[-1]["t"] and info.get("start"):
        add("Current at the start (switch closed)", abs(info["start"]["I"]["src"]), "A", "battery")
    if math.isfinite(R_ext) and hist[0]["t"] == hist[-1]["t"]:
        add("Total resistance of the circuit", R_ext, "Ω")
    if spec.source.capacity_mah and src_I > 1e-6 and hist[0]["t"] == hist[-1]["t"]:
        add("Battery would last about", spec.source.capacity_mah / 1000 / src_I, "h", "battery")
    elements = []
    for eid, p, a, b in els:
        if not eid.startswith("e"):
            continue
        I = last["I"][eid] if abs(last["I"][eid]) > 1e-9 else 0.0
        Vd = last["V"][a] - last["V"][b]
        Vd = Vd if abs(Vd) > 1e-6 else 0.0
        P = abs(I * Vd)
        add(f"{p.label}: current", abs(I), "A", p.label)
        add(f"{p.label}: voltage across it", abs(Vd), "V", p.label)
        if p.kind in ("bulb", "resistor", "led", "motor", "buzzer"):
            add(f"{p.label}: power", P, "W", p.label)
        bright = 0.0
        if p.kind == "bulb":
            bright = min(1.5, P / max(p.rated_p, 1e-6))
            add(f"{p.label}: brightness (vs its rating)", 100 * bright, "%", p.label)
        if p.kind == "led":
            bright = min(1.5, abs(I) / 0.02)
        elements.append({"id": eid, "kind": p.kind, "label": p.label, "value": p.value, "pos": pos[eid], "color": p.color,
                         "closed": p.closed, "blown": eid in info["fuse_blown"], "on": bool(info["led_on"].get(eid, True)),
                         "I": [h["I"][eid] for h in hist], "V": [h["V"][a] - h["V"][b] for h in hist], "brightness": bright})
    # why
    groups = [g for g in spec.series if isinstance(g, list)]
    series_parts = [x for x in spec.series if not isinstance(x, list)]
    story.append(f"The battery pushes charge around the loop with {V:g} V. Ohm's law (I = V / R) sets the current: the loop's total resistance is "
                 f"{_si(R_ext, 'Ω') if math.isfinite(R_ext) else 'infinite (the circuit is open)'}, so {_si(src_I, 'A')} flows out of the battery.")
    if any(p.kind == "switch" and not p.closed for p in series_parts):
        story.append("The switch is open: there is a gap in the loop, so no charge can flow and nothing lights up — the full battery voltage appears across the open switch.")
    if len([p for p in series_parts if p.kind in ("bulb", "resistor", "led", "motor", "buzzer", "capacitor")]) > 1:
        story.append("Parts in series share ONE path: the same current flows through each, and the battery's voltage is divided between them "
                     "(they add up to the battery voltage — Kirchhoff's voltage law). Adding more in series raises the total resistance, so every bulb gets dimmer.")
    for g in groups:
        story.append(f"The {len(g)} parts in parallel each get the full voltage across them, and the currents through the branches add up "
                     f"(Kirchhoff's current law): adding a branch gives the charge another path, LOWERING the total resistance, so the battery has to supply more current.")
        if all(p.kind == "bulb" for p in g):
            story.append("That's why parallel bulbs stay bright (each sees the whole voltage) — but the battery drains faster.")
    if any(p.kind == "led" for _, p, _, _ in els):
        for e in elements:
            if e["kind"] == "led":
                story.append(f"The {e['label']} only conducts once the voltage across it reaches about {next(p.vf for eid, p, a, b in els if eid == e['id']):.1f} V; "
                             + ("it's on, carrying " + _si(abs(e['I'][-1]), 'A') + (" — above the ~20 mA a small LED can take, so it would overheat." if abs(e['I'][-1]) > 0.03 else ".")
                                if e["on"] and abs(e["I"][-1]) > 1e-4 else "the battery isn't strong enough after the other parts take their share, so it stays dark."))
    if info["fuse_blown"]:
        story.append("The current exceeded the fuse's rating, so the fuse melted and opened the circuit — that's its job: protecting the wires from overheating.")
    if any(p.kind == "wire" for p in series_parts) and src_I > 2:
        story.append(f"With almost no resistance, {_si(src_I, 'A')} flows; the battery's internal resistance turns {_si(src_I ** 2 * spec.source.r_int, 'W')} into heat inside it — "
                     "real batteries get hot, can leak or catch fire.")
    # ---- independent theory: series/parallel reduction + Ohm's law, and the RC charging law
    ag = []
    R_eq = _r_eq(spec.series)
    if R_eq is not None and not info["fuse_blown"] and not info["tau"]:
        I_th = V / (R_eq + spec.source.r_int) if math.isfinite(R_eq) else 0.0
        add("Current (theory: Ohm's law, I = V / (R_total + r_battery))", I_th, "A", "battery")
        ag.append(100.0 if I_th == 0 and src_I == 0 else max(0.0, 100 * (1 - abs(src_I - I_th) / max(I_th, 1e-12))))
    rc = _series_rc(spec.series)
    if rc is not None and info.get("start"):
        R_rest, C = rc
        R_th = R_rest + spec.source.r_int
        tau_th = R_th * C
        I0 = V / R_th
        ts = [h["t"] for h in hist]
        cap_id = next(eid for eid, p, a, b in els if p.kind == "capacitor")
        a_, b_ = next((a, b) for eid, p, a, b in els if eid == cap_id)
        vcs = [h["V"][a_] - h["V"][b_] for h in hist]
        vc_tau = float(np.interp(tau_th, ts, vcs))
        add("Current at the start (theory V / R)", I0, "A", "battery")
        add("Capacitor voltage after one τ (measured)", vc_tau, "V", "capacitor")
        add("Capacitor voltage after one τ (theory V(1 − 1/e))", V * (1 - math.exp(-1)), "V", "capacitor")
        ag += [max(0.0, 100 * (1 - abs(abs(info["start"]["I"]["src"]) - I0) / I0)), max(0.0, 100 * (1 - abs(vc_tau - V * (1 - math.exp(-1))) / (V * (1 - math.exp(-1)))))]
    if ag:
        add("Agreement with theory", min(ag), "%")
    dyn = []
    if info["tau"]:
        tau = info["tau"]
        add("Time constant τ", tau, "s")
        caps = [e for e in elements if e["kind"] == "capacitor"]
        if caps:
            c = caps[0]
            story.append(f"The capacitor starts empty, so at first current rushes in; as charge piles up its voltage rises and pushes back, so the current dies away "
                         f"exponentially. After one time constant τ = R·C = {_si(tau, 's')} it's 63% charged, after 5τ ({_si(5 * tau, 's')}) it's practically full "
                         f"at {abs(c['V'][-1]):.2f} V and the current has stopped.")
        inds = [e for e in elements if e["kind"] == "inductor"]
        if inds:
            story.append(f"The inductor fights any change in current (its magnetic field has to build up), so the current rises gradually with τ = L/R = {_si(tau, 's')} "
                         f"instead of jumping straight to its final value.")
        dyn = [h["t"] for h in hist]
    for e in elements:
        if e["kind"] == "bulb":
            e["note"] = f"{e['brightness'] * 100:.0f}% of full brightness"
    over = [e["label"] for e in elements if e["kind"] == "bulb" and e["brightness"] > 1.15]
    if over:
        story.append(f"{', '.join(over)} {'is' if len(over) == 1 else 'are'} running above {'its' if len(over) == 1 else 'their'} rating: brighter and much hotter than designed "
                     f"(power ∝ V²/R), so the filament would burn out far sooner.")
    wires_out = [{"pts": pts, "el": el, "I": ([h["I"][el] for h in hist] if el in hist[-1]["I"] else [h["I"]["src"] for h in hist])} for pts, el in wires]
    for w in wires_out:
        if w["el"] == "src":
            w["I"] = [-x for x in w["I"]]      # charge leaves the + terminal: same direction as the loop
    return {
        "domain": "circuit", "text": text,
        "source": {"label": spec.source.label, "V": V, "r_int": spec.source.r_int, "pos": pos["src"], "I": [abs(h["I"]["src"]) for h in hist]},
        "elements": elements, "wires": wires_out, "board": size,
        "t": dyn or [0.0], "duration": info["T"] if dyn else 4.0, "dynamic": bool(dyn),
        "results": rows, "story": story,
    }


# ---------------------------------------------------------------- follow-ups ("what if it's 12 V", "add another bulb")
KIND_WORDS = {"bulb": r"bulbs?|lamps?", "resistor": r"resistors?", "led": r"leds?", "capacitor": r"capacitors?", "motor": r"motors?",
              "buzzer": r"buzzers?", "switch": r"switch(?:es)?", "inductor": r"inductors?|coils?"}


def describe(spec: Spec) -> dict:
    parts: list[dict] = []
    topo = "series"
    for item in spec.series:
        group = item if isinstance(item, list) else [item]
        if isinstance(item, list):
            topo = "parallel"
        for p in group:
            key = (p.kind, round(p.value, 9), p.color, p.closed)
            hit = next((d for d in parts if (d["kind"], round(d["value"], 9), d["color"], d["closed"]) == key), None)
            if hit:
                hit["count"] += 1
            else:
                parts.append({"kind": p.kind, "value": p.value, "color": p.color, "closed": p.closed, "count": 1})
    return {"V": spec.source.value, "parts": parts, "topology": topo}


def render(d: dict) -> str:
    bits = []
    for p in d["parts"]:
        n = p["count"]
        if n <= 0:
            continue
        if p["kind"] == "bulb":
            bits.append(f"{n} {p['value']:.4g} ohm bulbs" if n > 1 else f"a {p['value']:.4g} ohm bulb")
        elif p["kind"] == "resistor":
            bits.append(f"{n} {p['value']:g} ohm resistors" if n > 1 else f"a {p['value']:g} ohm resistor")
        elif p["kind"] == "led":
            bits.append(f"{n} {p['color']} LEDs" if n > 1 else f"a {p['color']} LED")
        elif p["kind"] == "capacitor":
            bits.append(f"a {p['value'] * 1e6:g} uF capacitor")
        elif p["kind"] == "inductor":
            bits.append(f"a {p['value']:g} H inductor")
        elif p["kind"] == "switch":
            bits.append("an open switch" if not p["closed"] else "a closed switch")
        elif p["kind"] in ("motor", "buzzer"):
            bits.append(f"{n} {p['kind']}s" if n > 1 else f"a {p['kind']}")
        elif p["kind"] == "wire":
            bits.append("a wire across the terminals (short circuit)")
    loads = [p for p in d["parts"] if p["kind"] not in ("switch",) and p["count"] > 0]
    topo = " in parallel" if d["topology"] == "parallel" and sum(p["count"] for p in loads) > 1 else " in series" if sum(p["count"] for p in loads) > 1 else ""
    return f"a {d['V']:g} V battery with " + " and ".join(bits) + topo


def whatif(prev_text: str, text: str) -> tuple[str, list[str]]:
    d = describe(parse(prev_text))
    t = " " + text.lower() + " "
    ch = []
    v = _val(t, r"v\b|volts?|volt")
    if v:
        d["V"] = v[-1][1]; ch.append(f"battery → {d['V']:g} V")
    m = re.search(r"(double|twice|triple|half|halve)\w*\s+(the\s+)?(voltage|battery)", t) or re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\s+(the\s+)?(voltage|battery)", t)
    if m:
        f = {"double": 2, "twice": 2, "triple": 3, "half": 0.5, "halve": 0.5}.get(m.group(1), None) or float(m.group(1))
        d["V"] *= f; ch.append(f"battery voltage ×{f:g} → {d['V']:g} V")
    if re.search(r"\bparallel\b", t):
        d["topology"] = "parallel"; ch.append("loads now in parallel")
    if re.search(r"\bseries\b", t):
        d["topology"] = "series"; ch.append("loads now in series")
    for kind, pat in KIND_WORDS.items():
        for mm in re.finditer(rf"\b(add|put in|connect|plug in|include)\s+(another|one more|a|an|(\d+|two|three|four)(?: more)?)\s+(?:\w+\s+)?({pat})\b", t):
            n = WORDNUM.get(mm.group(2).split()[0], None) or (int(mm.group(3)) if mm.group(3) and mm.group(3).isdigit() else WORDNUM.get(mm.group(3) or "", 1))
            hit = next((p for p in d["parts"] if p["kind"] == kind), None)
            if hit:
                hit["count"] += n
            else:
                d["parts"].append({"kind": kind, "value": {"resistor": 100.0, "capacitor": 100e-6, "inductor": 0.1}.get(kind, 0.0), "color": "red", "closed": True, "count": n})
            ch.append(f"+{n} {kind}")
        for mm in re.finditer(rf"\b(remove|take out|unplug|disconnect)\s+(a|an|one|the|(\d+|two|three))\s+(?:\w+\s+)?({pat})\b", t):
            n = int(mm.group(3)) if mm.group(3) and mm.group(3).isdigit() else WORDNUM.get(mm.group(3) or "", 1)
            hit = next((p for p in d["parts"] if p["kind"] == kind), None)
            if hit:
                hit["count"] = max(0, hit["count"] - n); ch.append(f"−{n} {kind}")
    if re.search(r"\b(open|turn off|switch off)\b", t) and re.search(r"switch|it off|circuit", t):
        sw = next((p for p in d["parts"] if p["kind"] == "switch"), None)
        if sw:
            sw["closed"] = False
        else:
            d["parts"].append({"kind": "switch", "value": 0, "color": "", "closed": False, "count": 1})
        ch.append("switch opened")
    if re.search(r"\b(close|turn on|switch on)\b", t):
        for p in d["parts"]:
            if p["kind"] == "switch":
                p["closed"] = True
        ch.append("switch closed")
    ohm = _val(t.replace("Ω", " ohm"), r"ohms?")
    if ohm:
        for p in d["parts"]:
            if p["kind"] == "resistor":
                p["value"] = ohm[-1][1]
        if not any(p["kind"] == "resistor" for p in d["parts"]):
            d["parts"].append({"kind": "resistor", "value": ohm[-1][1], "color": "", "closed": True, "count": 1})
        ch.append(f"resistor → {_si(ohm[-1][1], 'Ω')}")
    return render(d), ch
