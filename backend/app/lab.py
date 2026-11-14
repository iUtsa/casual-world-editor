"""The experiment lab: runs, follow-ups ("what if…"), questions about results, and the link to the
real scanned scene (an experiment in the real world becomes the AR session's simulation).

Everything a run reports comes from the recorded simulation; nothing is typed in by hand.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from collections import OrderedDict

from . import exp_analysis, experiment
from .experiment import Experiment
from .state import to_jsonable

log = logging.getLogger(__name__)

RUNS: "OrderedDict[str, dict]" = OrderedDict()
CURRENT: dict = {"id": None}
LOCK = threading.RLock()
MAX_RUNS = 24

QUESTION = re.compile(r"^\s*(how|what(?!\s+if)|when|which|why|who|does|did|is|was|were|are|can|could|would|will|do)\b|\?\s*$", re.I)

# quantity words -> substrings of result labels
QUANTITIES = [
    (r"\b(current|amps?|ampere\w*)\b", ["current"]),
    (r"\b(voltage|volts?|potential)\b", ["voltage"]),
    (r"\b(power|watts?)\b", ["power"]),
    (r"\b(resistance|ohms?)\b", ["resistance"]),
    (r"\b(bright\w*|dim\w*|glow\w*)\b", ["brightness"]),
    (r"\b(last|battery life|run out)\b", ["last"]),
    (r"\b(orbit\w*|period|year|lap)\b", ["period", "orbital"]),
    (r"\b(tnt|explosion|energy released|destroy\w*)\b", ["energy released", "megatons", "dinosaur"]),
    (r"\b(speed|velocity|valocity|fast|quick)\b", ["speed", "velocity"]),
    (r"\b(force|hard|impact|newton|push|hit)\b", ["peak collision force", "force", "impulse"]),
    (r"\b(energy|joule|kinetic)\b", ["energy"]),
    (r"\bmomentum\b", ["momentum"]),
    (r"\b(how far|distance|range|travel)\b", ["distance", "range", "moved", "displacement"]),
    (r"\b(how high|height|tall|peak)\b", ["height", "max height"]),
    (r"\b(how long|time|when|duration|seconds?)\b", ["time", "duration", "period", "earlier"]),
    (r"\b(bounc\w*|restitution|elastic)\b", ["restitution", "bounce"]),
    (r"\b(tension|rope|string)\b", ["tension"]),
    (r"\b(period|oscillat\w*|swing|bob)\b", ["period"]),
    (r"\b(accelerat\w*)\b", ["acceleration"]),
    (r"\b(spin|rotat\w*|rpm)\b", ["spin", "rpm"]),
    (r"\b(float|sink|buoyan\w*|density)\b", ["buoyant", "density", "rose", "sank", "terminal"]),
    (r"\b(who wins|winner|first|faster one)\b", ["distance", "earlier", "landed"]),
]


def _public(r: dict, run_id: str, parent: str | None, changes: list[str], text: str) -> dict:
    comp = r["_compiled"]
    out = {k: v for k, v in r.items() if k != "_compiled"}
    out["id"] = run_id
    out["parent"] = parent
    out["changes"] = changes
    out["text"] = text
    out["catalog"] = exp_analysis.catalog(comp, r["sim"])
    out["links"] = r["sim"].get("links", [])
    out["created"] = time.time()
    out["domain"] = "mechanics"
    out["duration"] = r["sim"]["duration"]
    return to_jsonable(out)


def _store(pub: dict, r: dict) -> None:
    with LOCK:
        RUNS[pub["id"]] = {"public": pub, "comp": r["_compiled"], "sim": r["sim"]}
        CURRENT["id"] = pub["id"]
        while len(RUNS) > MAX_RUNS:
            RUNS.popitem(last=False)


def current() -> dict | None:
    with LOCK:
        rid = CURRENT["id"]
        return RUNS.get(rid) if rid else None


def summary() -> list[dict]:
    with LOCK:
        return [{"id": k, "text": v["public"]["text"], "setting": v["public"].get("setting", v["public"].get("domain")), "parent": v["public"]["parent"],
                 "changes": v["public"]["changes"], "created": v["public"]["created"],
                 "headline": [x for x in v["public"]["results"][:3]]} for k, v in RUNS.items()]


CHEM = re.compile(r"\b(react\w*|reaction|molecule\w*|compound|acid|alkali|neutrali[sz]\w*|combust\w*|burn\w* (methane|hydrogen|propane|ethanol|fuel)|"
                  r"electrolys\w*|oxidi[sz]\w*|precipitat\w*|dissolv\w*|bond\w*|atoms?|methane|ethanol|glucose|ammonia|sodium|chlorine|hydrogen|oxygen gas|"
                  r"h2o|nacl|co2|ch4|o2|h2|\+.*->|→)\b", re.I)
UNSUPPORTED = [
    (r"\b(refract\w*|reflect\w*|lens(es)?|prism|laser|optic\w*|mirror|rainbow|light ray|diffract\w*)\b", "light and optics (refraction, lenses, mirrors)"),
    (r"\b(sound wave|echo|frequency|pitch|resonan\w*|wave interference|standing wave|doppler)\b", "sound and waves"),
    (r"\b(melt\w*|boil\w*|freez\w*|evaporat\w*|temperature|heat (flow|transfer)|thermal|conduction|convection)\b", "heat and temperature"),
    (r"\b(magnet\w*|magnetic field|compass)\b", "magnetism"),
    (r"\b(fission|fusion|radioactiv\w*|half-life|nuclear)\b", "nuclear physics"),
    (r"\b(photosynthes\w*|dna|protein|cell divi\w*|mitosis|enzyme)\b", "biology"),
]


def route(text: str) -> str:
    from . import circuits, orbits
    t = text.lower()
    if circuits.is_circuit(t):
        return "circuit"
    if orbits.is_gravity(t):
        return "gravity"
    if re.search(r"\b(transformer|attention|neural net\w*|language model|llm|gpt|token\w*|embedding|large language|next word|machine learning|deep learning)\b", t):
        return "ml"
    if CHEM.search(text) and not re.search(r"\b(throw|drop|collid|hit|roll|kick)\w*", t):
        return "chem"
    for pat, what in UNSUPPORTED:
        if re.search(pat, t):
            return "unsupported:" + what
    return "mechanics"


def run_text(text: str, scene, base, measures: list[str], follow_up: bool, force_scene: bool = False) -> dict:
    """Route the sentence to the right engine; a follow-up ("what if…") modifies the current experiment."""
    from . import circuits, orbits
    prev = current()
    pdom = prev["public"].get("domain", "mechanics") if prev else None
    dom = route(text)
    wf = follow_up and prev is not None and (experiment.is_whatif(text) or re.search(
        r"^\s*(now|then|and|also|instead)\b|\b(add|remove|take out|another|one more|open|close|turn (on|off)|swap|replace|change)\b", text.lower()) is not None)
    if wf and dom != pdom:
        # "drop a hammer on the Moon" after a gravity run is a NEW mechanics experiment: it names everyday things
        found = [o.name for o in experiment.parse_rules(text, {}).objects]
        if pdom == "gravity":
            from .orbits import ALL_NAMES
            wf = all(f in ALL_NAMES for f in found)
        elif pdom == "circuit":
            wf = all(re.search(r"bulb|lamp|resistor|led|battery|switch|capacitor|motor|buzzer|wire|cell", f) for f in found)
        else:
            wf = dom == "mechanics"
    if wf and pdom == "circuit" and dom in ("circuit", "mechanics"):
        new_text, ch = circuits.whatif(prev["public"]["source_text"], text)
        return _store_other(circuits.run(new_text), text, prev["public"]["id"], ch or ["re-ran"], new_text)
    if wf and pdom == "gravity" and dom in ("gravity", "mechanics"):
        new_text, scale, ch = orbits.whatif(prev["public"]["source_text"], text)
        return _store_other(orbits.run(new_text, scale), text, prev["public"]["id"], ch or ["re-ran"], new_text)
    if dom == "circuit":
        return _store_other(circuits.run(text), text, None, [], text)
    if dom == "gravity":
        return _store_other(orbits.run(text), text, None, [], text)
    if dom == "chem":
        return {"domain": "chem", "text": text}
    if dom == "ml":
        return {"domain": "ml", "text": text}
    if dom.startswith("unsupported:"):
        return {"domain": "unsupported", "text": text, "what": dom.split(":", 1)[1]}
    return _run_mechanics(text, scene, base, measures, follow_up, prev, force_scene)


def _store_other(r: dict, text: str, parent: str | None, changes: list[str], source_text: str) -> dict:
    pub = to_jsonable(r | {"id": uuid.uuid4().hex[:8], "parent": parent, "changes": changes, "text": text, "source_text": source_text,
                           "created": time.time(), "catalog": [], "setting": r["domain"], "objects": [], "notes": r.get("notes", [])})
    with LOCK:
        RUNS[pub["id"]] = {"public": pub, "comp": None, "sim": None}
        CURRENT["id"] = pub["id"]
        while len(RUNS) > MAX_RUNS:
            RUNS.popitem(last=False)
    return pub


def _run_mechanics(text: str, scene, base, measures: list[str], follow_up: bool, prev, force_scene: bool = False) -> dict:
    if follow_up and prev is not None and prev["public"].get("domain", "mechanics") == "mechanics" and experiment.is_whatif(text) and not _names_new_things(text, prev):
        exp, changes = experiment.apply_whatif(prev["public"], text)
        if not changes:
            changes = ["re-ran with the same set-up (I couldn't find what to change — try 'twice as heavy', 'on the Moon', '30 km/h')"]
        use_scene = scene if prev["public"]["setting"] == "scene" or force_scene else None
        r = experiment.run_experiment(exp, use_scene, base, measures, force_scene)
        pub = _public(r, uuid.uuid4().hex[:8], prev["public"]["id"], changes, text)
    else:
        r = experiment.run(text, scene, base, measures, force_scene)
        pub = _public(r, uuid.uuid4().hex[:8], None, [], text)
    _store(pub, r)
    return pub


def _names_new_things(text: str, prev: dict) -> bool:
    """'what if a bowling ball hits it instead' introduces a new object -> a fresh experiment."""
    e = experiment.parse_rules(text, {})
    have = {o["name"].rsplit(" ", 1)[0] if o["name"][-1].isdigit() else o["name"] for o in prev["public"]["objects"]}
    planets = set(experiment.PLANETS)
    new = [o.name for o in e.objects if o.name not in have and o.name not in planets and not any(o.name in h or h in o.name for h in have)]
    return bool(new)


def rerun(exp_json: dict, scene, base, measures: list[str], parent: str | None) -> dict:
    exp = Experiment(**exp_json)
    use_scene = scene if exp.setting in ("scene", "auto") else None
    r = experiment.run_experiment(exp, use_scene, base, measures)
    pub = _public(r, uuid.uuid4().hex[:8], parent, ["edited by hand"], exp.title)
    _store(pub, r)
    return pub


def get(run_id: str) -> dict | None:
    with LOCK:
        x = RUNS.get(run_id)
        if x:
            CURRENT["id"] = run_id
        return x["public"] if x else None


def answer(question: str, run_id: str | None = None) -> dict:
    """Answer a question about a run from its measurements (and say why, from its explanation)."""
    with LOCK:
        x = RUNS.get(run_id) if run_id else current()
    if x is None:
        return {"answer": "Run an experiment first, then ask about it.", "rows": []}
    pub = x["public"]
    q = " " + question.lower() + " "
    rows = pub["results"] + pub.get("catalog", [])
    keys = [subs for pat, subs in QUANTITIES if re.search(pat, q)]
    names = [o["name"] for o in pub.get("objects", []) if not o.get("support")] + [b["name"] for b in pub.get("bodies", [])] + [e["label"] for e in pub.get("elements", [])]
    mentioned = [n for n in names if n.split(" ")[0] in q or n in q]
    hits = []
    for subs in keys:
        for sub in subs:                     # earlier words in the list are the better answer
            for r_ in rows:
                lab = r_["label"].lower()
                if sub in lab and (not mentioned or any(n in lab for n in mentioned) or r_.get("about") in ("pair", "system", "spring", "rope", "comparison")):
                    hits.append(r_)
    seen, best = set(), []
    for h in hits:
        if h["label"] not in seen:
            seen.add(h["label"])
            best.append(h)
    best = best[:5]
    why = [s for s in pub["story"] if any(any(sub in s.lower() for sub in subs) for subs in keys)][:2] if keys else []
    if q.strip().startswith("why") or not best:
        why = why or pub["story"][:2]
        txt = _ai_answer(question, pub) or " ".join(why)
        return {"answer": txt, "rows": best, "why": why}
    lead = best[0]
    txt = f"{lead['label']}: {lead['text']}."
    if len(best) > 1:
        txt += " Also: " + "; ".join(f"{b['label']} {b['text']}" for b in best[1:4]) + "."
    return {"answer": txt, "value": lead["value"], "unit": lead["unit"], "text": lead["text"], "rows": best, "why": why}


def _ai_answer(question: str, pub: dict) -> str | None:
    """On-device model, grounded strictly in this run's numbers and explanation."""
    from . import local_ai
    if not local_ai.available():
        return None
    facts = "\n".join(f"- {r['label']}: {r['text']}" for r in pub["results"][:24])
    story = " ".join(pub["story"][:6])
    prompt = (f"You explain a physics simulation to a student. Use ONLY these measured results and this explanation; "
              f"do not invent numbers.\nResults:\n{facts}\nExplanation: {story}\n\nQuestion: {question}\n"
              f"Answer in 2-3 clear sentences:")
    try:
        out = local_ai.generate(prompt, max_tokens=160)
        out = (out or "").strip()
        return out or None
    except Exception:
        log.exception("local AI answer failed")
        return None
