"""General experiments from plain language.

    text --understand--> Experiment (objects, actions, setting, measurements)
         --compile-->    world (scanned scene or a scaled sandbox) + timed interventions
         --simulate-->   MuJoCo (the same core as every other feature)
         --analyse-->    measurements the experiment calls for + the user's picks, and why

Nothing here is specific to one scenario: verbs map to physics primitives (launch with a
computed velocity, self-propulsion, ropes, springs, holds/releases), objects come from the
scene, the 3D library, a physical-knowledge table, or a sanity-checked AI estimate.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from . import knowledge
from .knowledge import Thing

log = logging.getLogger(__name__)
G = 9.81

VERBS = ["incline", "collide", "fight", "hit", "throw", "drop", "push", "pull", "chase", "race", "spin", "roll",
         "rope", "spring", "stack", "place", "remove", "bounce", "orbit", "swing", "move", "jump"]

# how something moves by itself, from the verb used
MODE_WORDS = [("sprint", r"\bsprint\w*|\bdash\w*|\bgallop\w*|full speed|as fast as"), ("run", r"\brun|\bran\b|\brace off|\bzoom"),
              ("jog", r"\bjog\w*|\btrot\w*"), ("walk", r"\bwalk\w*|\bmarch\w*|\bstroll\w*|\bwander\w*"),
              ("swim", r"\bswim\w*|\bswam\b|\bpaddl\w*"), ("fly", r"\bfly|\bflies|\bflying|\bflew|\bsoar\w*|\bglid\w*|takes? off"),
              ("drive", r"\bdrive|\bdriving|\bdrove|\bride|\briding|\brode|\bcruis\w*|\bcycl\w*|\bpedal\w*|\bskat\w*|\bsail\w*"),
              ("crawl", r"\bcrawl\w*|\bslither\w*")]


def mode_of(text: str) -> str | None:
    for m, pat in MODE_WORDS:
        if re.search(pat, text):
            return m
    return None
# things that join objects rather than being objects themselves
CONNECTORS = {"rope": "rope", "string": "rope", "cord": "rope", "cable": "rope", "chain": "rope", "thread": "rope",
              "wire": "rope", "spring": "spring", "bungee": "spring", "elastic": "spring", "rubber band": "spring"}
ANCHOR = "support"


class ExObject(BaseModel):
    name: str
    what: str
    count: int = 1
    size_m: Optional[float] = None
    mass_kg: Optional[float] = None
    bounciness: Optional[float] = None     # 0..1 coefficient of restitution override


class ExAction(BaseModel):
    do: str
    a: str
    b: Optional[str] = None
    speed: Optional[float] = None          # m/s
    force: Optional[float] = None          # N
    height: Optional[float] = None         # m
    distance: Optional[float] = None       # m (separation / rope length)
    angle_deg: Optional[float] = None
    rpm: Optional[float] = None
    stiffness: Optional[float] = None      # N/m
    duration: Optional[float] = None       # s
    from_ground: bool = False              # kicked / rolled off the ground rather than thrown from the hand
    mode: Optional[str] = None             # how it moves by itself: walk, run, sprint, swim, fly, drive, crawl
    mu: Optional[float] = None             # friction coefficient (inclined planes)


class Experiment(BaseModel):
    title: str = ""
    setting: Literal["auto", "scene", "table", "ground", "water", "air", "space"] = "auto"
    objects: list[ExObject] = Field(default_factory=list)
    actions: list[ExAction] = Field(default_factory=list)
    measure: list[str] = Field(default_factory=list)
    seconds: Optional[float] = None
    understood_by: str = "rules"
    notes: list[str] = Field(default_factory=list)
    gravity: Optional[float] = None        # m/s^2 override (Moon 1.62, Mars 3.71, ...)
    surface: Optional[str] = None          # "on/across the table": where it happens, not an object


# ---------------------------------------------------------------------------
# understanding: rules
# ---------------------------------------------------------------------------

VERB_PATTERNS = [   # (canonical, regex) - checked in order; first match per clause wins
    ("fight", r"\b(fight|fighting|battle|battling|attack each other|ram each other|wrestl\w*|duel)\b"),
    ("collide", r"\b(collid\w*|clash\w*|crash(?:es|ing)? into each other|smash(?:es|ing)? into each other|hit each other|bump(?:s|ing)? into each other|head[- ]on|crash(?:es)?|collision)\b"),
    ("race", r"\b(race|racing)\b"),
    ("chase", r"\b(chas\w*|hunt\w*|run(?:s|ning)? after|follow\w*)\b"),
    ("throw", r"\b(throw\w*|threw|toss\w*|launch\w*|shoot\w*|fire[sd]?|hurl\w*|chuck\w*|fling\w*|lob\w*|kick\w*|pitch\w*)\b"),
    ("drop", r"\b(drop\w*|fall\w*|fell|let .{0,30}fall|release[sd]?)\b"),
    ("bounce", r"\b(bounc\w*)\b"),
    ("pull", r"\b(pull\w*|tow\w*|drag\w*|haul\w*|tug\w*)\b"),
    ("swing", r"\b(swing\w*|swung|pendulum|oscillat\w*)\b"),
    ("rope", r"\b(tie[sd]?|tied|rope|tether\w*|string|hang\w*|hung|suspend\w*)\b"),
    ("spring", r"\b(spring)\b"),
    ("hit", r"\b(hit\w*|strik\w*|ram\w*|smash\w*|slam\w*|knock\w*|crash\w* into|run\w* into|charg\w*)\b"),
    ("push", r"\b(push\w*|shov\w*|nudg\w*|slid\w*)\b"),
    ("roll", r"\b(roll\w*)\b"),
    ("spin", r"\b(spin\w*|spun|rotat\w*|twirl\w*)\b"),
    ("orbit", r"\b(orbit\w*)\b"),
    ("jump", r"\b(jump\w*|hop|hops|hopp\w*|leap\w*|leapt)\b"),
    ("move", r"\b(runs?|running|ran|walk\w*|jog\w*|sprint\w*|swim\w*|swam|fly|flies|flying|flew|soar\w*|glid\w*|drives?|driving|drove|rides?|riding|rode|"
             r"moves?|moving|goes|going|went|travel\w*|gallop\w*|trot\w*|crawl\w*|slither\w*|march\w*|dash\w*|accelerat\w*|cruis\w*|sail\w*|"
             r"paddl\w*|skat\w*|cycl\w*|pedal\w*|wander\w*|stroll\w*|race off|zoom\w*|speed\w* along|takes? off|launch\w* into)\b"),
    ("stack", r"\b(stack\w*|on top of|balance\w* on)\b"),
    ("remove", r"\b(remov\w*|delet\w*|take away)\b"),
    ("place", r"\b(put|place\w*|add\w*|spawn\w*|bring|show)\b"),
]
VERB_WORDS = set("""collide collides colliding collided clash clashes clashing clashed crash crashes crashing crashed smash smashes
smashing smashed hit hits hitting strike strikes striking struck ram rams ramming rammed slam slams slamming slammed knock knocks
knocking knocked bump bumps bumping fight fights fighting fought battle battles battling attack attacks attacking wrestle wrestles
duel duels throw throws throwing threw thrown toss tosses tossing tossed launch launches launching launched shoot shoots shooting
shot fire fires firing fired hurl hurls hurling hurled chuck chucks chucking chucked fling flings flinging flung lob lobs lobbing
kick kicks kicking kicked pitch pitches pitching pitched drop drops dropping dropped fall falls falling fell let release releases
released bounce bounces bouncing bounced pull pulls pulling pulled tow tows towing towed drag drags dragging dragged haul hauls
hauling hauled tug tugs tugging tugged tie ties tied tying tether tethers push pushes pushing pushed shove shoves shoving shoved
nudge nudges slide slides sliding slid roll rolls rolling rolled spin spins spinning spun rotate rotates rotating twirl twirls
orbit orbits orbiting jump jumps jumping jumped hop hops leap leaps leapt glide glides gliding glided slide slides sliding walk walks walking walked jog jogs jogging sprint sprints sprinting swim swims swimming swam fly flies flying flew drive drives driving drove ride rides riding rode move moves moving go goes going went gallop gallops crawl crawls crawling swing swings swinging swung hang hangs hanging hung suspend suspended stack stacks stacking stacked balance balances remove removes removing delete take put puts putting place
places placing placed add adds adding spawn bring brings show race races racing chase chases chasing chased hunt hunts hunting
follow follows following run runs running ran charge charges charging charged meet meets meeting met""".split())
COUNTS = {"a": 1, "an": 1, "one": 1, "two": 2, "both": 2, "pair of": 2, "a pair of": 2, "three": 3, "four": 4,
          "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "couple of": 2, "a couple of": 2}
STOP = {"the", "a", "an", "and", "with", "by", "angle", "angles", "coefficient", "friction", "degree", "degrees", "length", "mass", "weight",
        "speed", "velocity", "force", "height", "distance", "value", "values", "number", "inclined", "plane", "incline", "ramp", "slope", "diagram",
        "shown", "shows", "figure", "magnitude", "acceleration", "gravity", "energy", "period", "frequency", "amplitude", "connected", "joined", "attached", "linked", "tied", "hooked", "via",
        "using", "against", "toward", "towards", "onto", "over", "near", "next", "behind", "beside", "between", "as", "into", "each", "other", "at", "on", "from", "to", "of", "then", "it", "them",
        "together", "simulate", "show", "me", "what", "happens", "if", "when", "let", "make", "see", "in", "3d", "experiment",
        "how", "fast", "much", "is", "are", "be", "my", "this", "that", "very", "really", "real", "quick", "quickly", "slowly",
        "high", "up", "down", "ground", "table", "water", "space", "sky", "air", "road", "street", "ocean", "sea", "under",
        "measure", "find", "speed", "force", "energy", "momentum", "who", "wins", "for", "seconds", "second", "times",
        "m", "cm", "mm", "km", "meter", "meters", "metre", "metres", "feet", "ft", "inch", "inches", "s", "sec", "secs",
        "kg", "g", "n", "kn", "mph", "kmh", "rpm", "degrees", "deg", "percent", "other", "another", "same", "time",
        "bit", "lot", "little", "big", "huge", "tiny", "giant", "small", "large", "heavy", "light", "hard", "soft"}
UNIT_RE = r"^(\d+(\.\d+)?)?\s*(m|cm|mm|km|meters?|metres?|feet|ft|inch(es)?|s|secs?|seconds?|kg|g|n|kn|mph|km/?h|rpm|degrees?)$"


def _lexicon(scene_labels: dict[str, str]) -> list[str]:
    from . import assets as asset_lib
    names = set(knowledge._T)
    for a in asset_lib.manifest().values():
        names.add(a["name"].lower())
        names.update(x.lower() for x in a["aliases"])
    names.update(l.lower() for l in scene_labels.values())
    return sorted(names, key=len, reverse=True)


def _numbers(t: str) -> dict:
    out = {}
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|times)\s*(?:the\s+)?(?:normal\s+|usual\s+)?(?:speed|faster|as fast|quicker)", t) or \
        re.search(r"\b(?:at|with)\s+(\d+(?:\.\d+)?)\s*(?:x|×)\b", t)
    if m:
        out["speed_mult"] = float(m.group(1))
    for w, f in (("twice as fast", 2.0), ("double speed", 2.0), ("half speed", 0.5), ("three times as fast", 3.0)):
        if w in t:
            out["speed_mult"] = f
    m = re.search(r"(\d+(?:\.\d+)?)\s*-?\s*(degrees?|deg|°)", t)
    if m:
        out["angle"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(km/?h|kph|kmph|mph|m/s|meters? per second|cm/s)", t)
    if m:
        v = float(m.group(1)); u = m.group(2)
        out["speed"] = v / 3.6 if u.startswith(("km", "kph")) else v * 0.44704 if u == "mph" else v / 100 if u == "cm/s" else v
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kn|n|newtons?)\b", t)
    if m:
        out["force"] = float(m.group(1)) * (1000 if m.group(2) == "kn" else 1)
    m = re.search(r"(?:from|height of|at)\s*(\d+(?:\.\d+)?)\s*(m|meters?|metres?|cm|ft|feet)\b", t)
    if m:
        v = float(m.group(1)); u = m.group(2)
        out["height"] = v / 100 if u == "cm" else v * 0.3048 if u in ("ft", "feet") else v
    m = re.search(r"(\d+(?:\.\d+)?)\s*(m|meters?|metres?|cm)\s*(apart|away|long|rope|of rope|spring|string|of string|chain|cable|pendulum)", t) or \
        re.search(r"(?:length|long|distance)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(m|meters?|metres?|cm)\b()", t)
    if m:
        v = float(m.group(1)); out["distance"] = v / 100 if m.group(2) == "cm" else v
    m = re.search(r"(?:for|over)\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|seconds?)\b", t)
    if m:
        out["duration"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(rpm|revolutions)", t)
    if m:
        out["rpm"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(n/m|newtons? per met(?:er|re))", t) or \
        re.search(r"(?:\bk\b|spring constant|stiffness|force constant)\s*(?:of|is|=|:|equal to)?\s*(\d+(?:\.\d+)?)", t)
    if m:
        out["stiffness"] = float(m.group(1))
    if "speed" not in out:
        if re.search(r"\b(very fast|full speed|as fast as|max(imum)? speed)\b", t):
            out["speed_word"] = 2.0
        elif re.search(r"\b(fast|quick|hard|strong)", t):
            out["speed_word"] = 1.5
        elif re.search(r"\b(slow|gentl|soft)", t):
            out["speed_word"] = 0.5
    if "height" not in out and re.search(r"\b(very high|really high|skyscraper|roof|cliff)\b", t):
        out["height_word"] = 4.0
    return out


def _find_objects(t: str, lex: list[str]) -> list[tuple[int, str, int]]:
    """[(position, name, count)] for every object mention (known names first, then unknown nouns)."""
    found, taken = [], np.zeros(len(t) + 1, bool)
    for name in lex:
        for m in re.finditer(rf"\b{re.escape(name)}(?:es|s)?\b", t):
            if taken[m.start():m.end()].any():
                continue
            taken[m.start():m.end()] = True
            pre = t[max(0, m.start() - 14):m.start()]
            cnt = 1
            for w, c in COUNTS.items():
                if re.search(rf"\b{w}\s+(\w+\s+)?$", pre):
                    cnt = c
            md = re.search(r"\b(\d+)\s+(\w+\s+)?$", pre)
            if md and not (md.group(2) and re.fullmatch(UNIT_RE, md.group(2).strip())):     # "2 kg block" is one block
                cnt = int(md.group(1))
            plural = t[m.start():m.end()].endswith("s") and not name.endswith("s")
            if plural and cnt == 1:
                cnt = 2
            found.append((m.start(), name, min(cnt, 12)))
    # unknown nouns after a determiner/number ("two blue whales" is known; "a dragon" is not)
    toks = [(m.start(), m.group(0)) for m in re.finditer(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", t)]
    verb_word = lambda w: w in VERB_WORDS
    for i, (p0, w) in enumerate(toks):
        if w not in COUNTS and not w.isdigit() and w != "some":
            continue
        cnt = COUNTS.get(w, int(w) if w.isdigit() else 1)
        words, start = [], None
        for p1, w1 in toks[i + 1:i + 4]:
            if w1 in STOP or verb_word(w1) or re.fullmatch(UNIT_RE, w1) or w1 in COUNTS or w1[:1].isdigit():
                break
            if taken[p1]:
                words = []
                break
            words.append(w1)
            start = p1 if start is None else start
        if not words:
            continue
        nm = " ".join(words)
        found.append((start, nm, min(cnt, 12)))
        taken[start:start + len(nm)] = True
    return sorted(found)


MASS_UNITS = {"kg": 1.0, "kgs": 1.0, "kilogram": 1.0, "kilograms": 1.0, "g": 0.001, "gram": 0.001, "grams": 0.001,
              "tonne": 1000.0, "tonnes": 1000.0, "ton": 1000.0, "tons": 1000.0, "t": 1000.0, "lb": 0.45359, "lbs": 0.45359, "pound": 0.45359, "pounds": 0.45359}
_MASS = r"(\d+(?:\.\d+)?)\s*-?\s*(kgs?|kilograms?|grams?|g|tonnes?|tons?|t|lbs?|pounds?)"


SURFACES = ("table", "desk", "counter", "floor", "ground", "road", "grass", "ice")


def _mass_near(t: str, start: int, end: int) -> float | None:
    """The mass written next to an object: 'a 2 kg block', 'a 500 g ball', 'a ball of mass 3 kg', 'a cart weighing 2 kg'."""
    m = re.search(_MASS + r"\s+(?:\w+\s+)?$", t[max(0, start - 24):start])
    if m is None:
        m = re.match(r"\s*(?:\(|,)?\s*(?:of|with|weighing|that weighs|which weighs|whose mass is)?\s*(?:a\s+)?(?:mass\s+(?:of\s+|is\s+|=\s*)?)?" + _MASS + r"\b", t[end:end + 40])
    if m is None:
        return None
    return float(m.group(1)) * MASS_UNITS[m.group(2)]


def parse_rules(text: str, scene_labels: dict[str, str]) -> Experiment:
    t = " " + text.lower().strip() + " "
    lex = _lexicon(scene_labels)
    objs = _find_objects(t, lex)
    nums = _numbers(t)
    if not [o for o in objs if o[1] not in CONNECTORS and o[1].split()[-1] not in CONNECTORS]:
        # "a 1 kg mass on a spring", "a weight hangs from a string": the lab's slotted mass
        mm = re.search(r"\b(?:a|an|the|one|" + _MASS + r")\s+(?:\w+\s+)?(mass|weight|load|bob)\b", t)
        if mm:
            objs = sorted(objs + [(mm.start(mm.lastindex), "kilogram weight", 1)])
    exp = Experiment(title=text.strip()[:120], understood_by="rules")
    for k, pat in knowledge.SETTING_WORDS.items():
        if re.search(pat, t):
            exp.setting = k  # type: ignore[assignment]
            break
    for pl, g in PLANETS.items():
        if pl != "earth" and re.search(rf"\b(on|at) (the )?{pl}\b", t):
            exp.gravity = g
            exp.setting = "ground"  # type: ignore[assignment]
            exp.notes.append(f"Gravity set to {pl.title()}'s {g} m/s².")
    names: list[str] = []
    connector = None
    if re.search(r"\bpendulum", t):
        connector = "rope"
    planet_ctx = {pl for pl in PLANETS if re.search(rf"\b(on|at) (the )?{pl}\b", t)}
    for pos, name, cnt in objs:
        if name in planet_ctx:
            continue
        if name in SURFACES and re.search(rf"\b(on|onto|across|along|over|off|from|at|under|beneath)\s+(?:the |a |my |this |our )?{name}\b", t):
            exp.surface = name                    # the place it happens, not a thing in the experiment
            continue
        if name in CONNECTORS or knowledge._singular(name) in CONNECTORS or name.split()[-1] in CONNECTORS:
            connector = connector or CONNECTORS.get(name, CONNECTORS.get(name.split()[-1], "rope"))
            continue
        if name == "pendulum":
            name = "pendulum bob"
        base = knowledge._singular(name) if name not in lex else name
        mk_ = _mass_near(t, pos, pos + len(name))
        same = [o for o in exp.objects if o.what == base]
        if same:
            if mk_ is not None and all(o.mass_kg != mk_ for o in same):     # "a 2 kg cart hits a 1 kg cart"
                if same[0].mass_kg is None:
                    same[0].mass_kg = mk_
                else:
                    nm2 = f"{base} {len(same) + 1}"
                    exp.objects.append(ExObject(name=nm2, what=base, count=1, mass_kg=mk_))
                    names.append(nm2)
            continue
        exp.objects.append(ExObject(name=base, what=base, count=cnt, mass_kg=mk_))
        names.append(base)
    if not exp.objects:
        return exp
    verb = None
    vpos = len(t)
    for canon, pat in VERB_PATTERNS:
        m = re.search(pat, t)
        if m and m.start() < vpos + 1 and (verb is None or VERB_PATTERNS.index((canon, pat)) < [v for v, _ in VERB_PATTERNS].index(verb)):
            verb, vpos = canon, m.start()
    # instances: "baseball" x2 -> baseball 1, baseball 2
    inst: list[str] = []
    for o in exp.objects:
        inst += [o.name] if o.count == 1 else [f"{o.name} {i + 1}" for i in range(o.count)]
    first = inst[0]
    second = inst[1] if len(inst) > 1 else None
    # who acts on whom: subject before the verb, object after (or "throw X at Y")
    known = {o.name for o in exp.objects} | {o.what for o in exp.objects}
    before = [o for p_, o, _ in objs if p_ < vpos and (o in known or knowledge._singular(o) in known)]
    after = [o for p_, o, _ in objs if p_ > vpos and (o in known or knowledge._singular(o) in known)]

    def inst_of(name):
        name = knowledge._singular(name) if name not in lex else name
        return next((i for i in inst if i == name or i.startswith(name + " ")), None)
    a = inst_of(before[0]) if before else first
    b = None
    if verb in ("throw", "drop", "push", "roll", "spin", "bounce", "remove", "place", "stack", "jump"):
        a = inst_of(after[0]) if after else first
        rest = [inst_of(x) for x in after[1:]] if after else []
        b = rest[0] if rest else (inst_of(before[0]) if before and inst_of(before[0]) != a else None)
    else:
        cands = [i for i in inst if i != a]
        b = inst_of(after[0]) if after and inst_of(after[0]) != a else (cands[0] if cands else None)
    speed = nums.get("speed") or None

    def mk(do, a_, b_):
        act_ = ExAction(do=do, a=a_, b=b_, speed=speed, force=nums.get("force"), height=nums.get("height"),
                        distance=nums.get("distance"), rpm=nums.get("rpm"), stiffness=nums.get("stiffness"),
                        duration=nums.get("duration"), angle_deg=nums.get("angle") if do == "throw" else None)
        if speed is None and "speed_word" in nums:
            act_.speed = -nums["speed_word"]          # negative = relative to the object's typical speed
        if speed is None and "speed_mult" in nums:
            act_.speed = -nums["speed_mult"]
        if act_.height is None and "height_word" in nums:
            act_.height = nums["height_word"]
        return act_
    obj_names = sorted({o.name for o in exp.objects} | {o.name + "s" for o in exp.objects}, key=len, reverse=True)
    has_target = bool(obj_names) and bool(re.search(
        r"\b(on|onto|at|into|toward|towards|against|over|on top of)\s+(?:the |a |an |another |other |each )?(?:\w+\s+)?(?:"
        + "|".join(re.escape(n) for n in obj_names) + r")\b", t[vpos:]))
    if verb == "move" and not has_target and len(inst) > 1:
        verb = "race"                            # several things running/swimming/driving together: a race
    if verb in ("drop", "bounce", "spin", "roll", "push", "throw", "jump") and not has_target and len(inst) > 1:
        for i_ in inst:                          # the same action on each named object (a comparison)
            exp.actions.append(mk(verb, i_, None))
        exp.notes.append(f"Each object gets the same '{verb}' so you can compare them.")
    else:
        if verb is None and second:
            exp.notes.append("No action word found; assuming the two collide.")
            verb = "collide"
        if verb is None and not second:
            t_ = knowledge.resolve(exp.objects[0].what, allow_estimate=False)
            verb = "move" if t_.self_propelled else "drop"
            exp.notes.append(f"No action word found; showing the {exp.objects[0].name} "
                             + ("moving by itself." if verb == "move" else "being dropped — say what should happen (throw, roll, spin, …)."))
        exp.actions.append(mk(verb, a, b))
        if verb in ("move", "race", "chase", "jump"):
            exp.actions[-1].mode = mode_of(t)
    act = exp.actions[-1]
    # a pull is a rope + traction; "tie" alone is a rope
    if verb == "pull" and act.b and act.do == "pull":
        exp.actions.insert(0, ExAction(do="rope", a=act.a, b=act.b, distance=act.distance))
    md = mode_of(t)
    if md == "swim" and exp.setting == "auto":
        exp.setting = "water"  # type: ignore[assignment]
    for x in exp.actions:
        if x.do in ("move", "race", "chase", "collide", "hit", "fight", "jump") and x.mode is None:
            x.mode = md
    if verb == "throw" and re.search(r"\bkick\w*|\bput+\w*|\bgolf|from (the )?(ground|floor)|ground level|level ground", t):
        for x in exp.actions:
            x.from_ground = True
    _attach_connector(exp, connector, verb, nums)
    _attach_incline(exp, t, nums)
    exp.measure = _measures_from_text(t)
    exp.seconds = nums.get("duration")
    return exp


INCLINE_RE = r"\b(ramp|incline\w*|inclined plane|slope|hill|wedge)\b"


def _attach_incline(exp: Experiment, t: str, nums: dict) -> None:
    """'a 2 kg block slides down a 30 degree ramp', 'a ball rolls down a frictionless incline'."""
    if not re.search(INCLINE_RE, t):
        return
    exp.objects = [o for o in exp.objects if not re.search(INCLINE_RE, " " + o.name + " ")]
    if not exp.objects:
        exp.objects.append(ExObject(name="block", what="block"))
    who = exp.objects[0].name
    mu = None
    m = re.search(r"(?:coefficient of )?friction(?: coefficient)?(?: of| is| =|:)?\s*(0?\.\d+|\d(?:\.\d+)?)", t) or re.search(r"(?:μ|mu)\s*=?\s*(0?\.\d+|\d(?:\.\d+)?)", t)
    if m:
        mu = float(m.group(1))
    if re.search(r"frictionless|no friction|smooth", t):
        mu = 0.0
    mass = re.search(r"(\d+(?:\.\d+)?)\s*(kg|g)\b", t)
    if mass:
        exp.objects[0].mass_kg = float(mass.group(1)) * (0.001 if mass.group(2) == "g" else 1.0)
    exp.actions = [ExAction(do="incline", a=who, angle_deg=nums.get("angle") or 30.0, distance=nums.get("distance"), mu=mu, duration=nums.get("duration"))]
    if exp.setting == "auto":
        exp.setting = "ground"  # type: ignore[assignment]


def _attach_connector(exp: Experiment, connector: str | None, verb: str | None, nums: dict) -> None:
    """'a ball on a spring', 'a pendulum', 'a rock hanging from a rope', 'two carts joined by a spring'.
    A connector joins the two named objects, or hangs a single object from a fixed support."""
    if connector is None and verb not in ("swing",):
        return
    connector = connector or "rope"
    joined = {(x.a, x.b) for x in exp.actions if x.do in ("rope", "spring")}
    names = [o.name for o in exp.objects]
    single = len(names) == 1 and exp.objects[0].count == 1
    if single or verb in ("swing", "drop", "bounce") and not any(x.b for x in exp.actions if x.do in ("rope", "spring", "pull")):
        if ANCHOR not in names:
            exp.objects.append(ExObject(name=ANCHOR, what=ANCHOR))
        hung = exp.actions[0].a if exp.actions else names[0]
        kind = "spring" if connector == "spring" else "swing" if (verb == "swing" or connector == "rope") else connector
        exp.actions = [ExAction(do=kind, a=ANCHOR, b=hung, distance=nums.get("distance"), stiffness=nums.get("stiffness"),
                                angle_deg=nums.get("angle"), duration=nums.get("duration"))]
        return
    for x in exp.actions:
        if x.do in ("rope", "spring", "swing"):
            x.do = connector if x.do != "swing" else "swing"
            return
    first = exp.actions[0] if exp.actions else None
    a = first.a if first else names[0]
    b = (first.b if first and first.b else next((n for n in names if n != a), None))
    if b and (a, b) not in joined and not (first and first.do == "pull"):
        exp.actions.insert(0, ExAction(do=connector, a=a, b=b, distance=nums.get("distance"), stiffness=nums.get("stiffness")))


MEASURE_WORDS = {
    "speed": r"\b(speed|velocity|valocity|fast)\b", "momentum": r"\bmomentum\b", "energy": r"\b(energy|joules?)\b",
    "force": r"\b(force|impact|hard|newtons?)\b", "distance": r"\b(how far|distance|range)\b",
    "height": r"\b(how high|height|max height)\b", "time": r"\b(how long|time|when)\b",
    "tension": r"\b(tension)\b", "acceleration": r"\baccelerat", "winner": r"\b(who wins|winner|first)\b",
}


def _measures_from_text(t: str) -> list[str]:
    return [k for k, p in MEASURE_WORDS.items() if re.search(p, t)]


# ---------------------------------------------------------------------------
# understanding: on-device model (same schema), validated
# ---------------------------------------------------------------------------

LLM_PROMPT = """Convert the request into a physics experiment. Reply with JSON only, no prose.
Verbs (field "do"): collide, fight, hit, throw, drop, push, pull, chase, race, spin, roll, rope, spring, stack, place, remove, bounce, move (runs/walks/swims/flies/drives by itself), jump.
"a" is the object doing it, "b" the other object (or null). Name every copy separately when there are several.
Optional numbers: speed (m/s), force (N), height (m), distance (m), rpm, duration (s).
Setting: auto, scene (the real table in the camera), table, ground, water, air, space.

Request: two baseballs clash into each other
{"setting":"auto","objects":[{"name":"baseball 1","what":"baseball"},{"name":"baseball 2","what":"baseball"}],"actions":[{"do":"collide","a":"baseball 1","b":"baseball 2"}],"measure":["momentum","energy"]}

Request: a man pulls a truck with a rope
{"setting":"ground","objects":[{"name":"man","what":"man"},{"name":"truck","what":"truck"}],"actions":[{"do":"rope","a":"man","b":"truck"},{"do":"pull","a":"man","b":"truck"}],"measure":["tension","speed"]}

Request: what if I throw an apple at the mug really hard
{"setting":"scene","objects":[{"name":"apple","what":"apple"},{"name":"mug","what":"mug"}],"actions":[{"do":"throw","a":"apple","b":"mug","speed":8}],"measure":["force","speed"]}

Request: two blue whales fight underwater
{"setting":"water","objects":[{"name":"whale 1","what":"blue whale"},{"name":"whale 2","what":"blue whale"}],"actions":[{"do":"fight","a":"whale 1","b":"whale 2"}],"measure":["force","energy"]}

Objects already in the camera scene: {scene}
Request: {request}
"""


def parse_llm(text: str, scene_labels: dict[str, str]) -> Experiment | None:
    from . import local_ai
    if not local_ai.available():
        return None
    prompt = LLM_PROMPT.replace("{scene}", ", ".join(sorted(set(scene_labels.values()))) or "none").replace("{request}", text)
    d = local_ai.extract_json(local_ai.generate(prompt, max_tokens=400))
    if not d:
        return None
    try:
        exp = Experiment(setting=d.get("setting", "auto") if d.get("setting") in ("auto", "scene", "table", "ground", "water", "air", "space") else "auto",
                         objects=[ExObject(name=str(o["name"]).strip().lower(), what=str(o.get("what") or o["name"]).strip().lower())
                                  for o in d.get("objects", []) if isinstance(o, dict) and o.get("name")],
                         measure=[str(x).lower() for x in d.get("measure", []) if isinstance(x, str)][:8],
                         title=text.strip()[:120], understood_by="on-device AI (Qwen2.5-VL-3B)")
        names = {o.name for o in exp.objects}
        for a in d.get("actions", []):
            if not isinstance(a, dict) or a.get("do") not in VERBS or str(a.get("a", "")).lower() not in names:
                continue
            b = str(a["b"]).lower() if a.get("b") else None
            if b is not None and b not in names:
                continue
            num = lambda k: float(a[k]) if isinstance(a.get(k), (int, float)) and a[k] > 0 else None
            exp.actions.append(ExAction(do=a["do"], a=str(a["a"]).lower(), b=b, speed=num("speed"), force=num("force"),
                                        height=num("height"), distance=num("distance"), rpm=num("rpm"), duration=num("duration")))
    except Exception as e:      # malformed model output is simply not used
        log.info("LLM experiment rejected: %s", e)
        return None
    return exp if exp.objects and exp.actions else None


def understand(text: str, scene_labels: dict[str, str]) -> Experiment:
    """The deterministic reader goes first (units, counts, verbs are parsed exactly). The on-device model is
    consulted only when that reader found no action word or no objects, and its reading is used only if it is
    valid; numbers with units always come from the deterministic reader."""
    rules = parse_rules(text, scene_labels)
    confident = bool(rules.objects) and bool(rules.actions) and not any(n.startswith("No action word") for n in rules.notes)
    if confident:
        return rules
    llm = parse_llm(text, scene_labels)
    if llm is None:
        return rules
    if rules.actions:
        ra = rules.actions[-1]
        for la in llm.actions:
            for f in ("speed", "force", "height", "distance", "rpm", "duration", "angle_deg", "stiffness"):
                if getattr(ra, f) is not None:
                    setattr(la, f, getattr(ra, f))
            la.mode = la.mode or ra.mode or mode_of(" " + text.lower() + " ")
    llm.measure = rules.measure
    if rules.setting != "auto":
        llm.setting = rules.setting
    llm.gravity = rules.gravity
    llm.notes = [n for n in rules.notes if not n.startswith("No action word")] + ["The on-device AI read this sentence."]
    return llm


# ---------------------------------------------------------------------------
# compile: world + interventions
# ---------------------------------------------------------------------------

GRIP = 0.7
PULL_GRIP = 0.85      # rubber soles on dry ground, leaning into a rope: nearly all of mu*m*g becomes pull    # shoe/paw/tyre grip on ground: caps self-propulsion at GRIP * m * g


def _mu(t: Thing) -> float:
    if t.extra.get("mu") is not None:
        return float(t.extra["mu"])
    if t.shape in ("box",) and any(w in t.name.lower() for w in ("car", "truck", "bus", "train", "tank", "cart", "skateboard", "bike", "bicycle", "motorcycle")):
        return 0.007 if t.mass > 3000 else 0.012     # rolls on wheels: rolling resistance (truck tyres ~0.007, car ~0.012)
    if t.self_propelled:
        return 0.05               # walks on legs: its body doesn't drag along the ground
    return {"rubber": 0.9, "metal": 0.35, "glass": 0.4, "plastic": 0.4, "wood": 0.5, "leather": 0.5, "flesh": 0.7,
            "stone": 0.6, "ceramic": 0.45, "paper": 0.5, "food": 0.6, "shell": 0.4, "fabric": 0.7}.get(t.material, 0.5)


def compile_experiment(exp: Experiment, scene, scene_labels: dict[str, str], force_scene: bool = False) -> dict:
    """Returns {things, setting, env, scene, interventions, layout notes} - everything needed to simulate."""
    from .schemas import Camera, Scene, Table
    notes = list(exp.notes)
    things: dict[str, Thing] = {}
    for o in exp.objects:
        copies = [o.name] if o.count == 1 else [f"{o.name} {i + 1}" for i in range(o.count)]
        if o.what == ANCHOR:
            things[ANCHOR] = Thing(name=ANCHOR, source="knowledge", shape="box", dims=[0.5, 0.06, 0.06], mass=20.0,
                                   material="metal", color="#9aa3ad", bounciness=0.2, note="fixed support the object hangs from")
            continue
        # a ramp set-up is self-contained: its block is a new object, not the real "purple block" on the table
        on_ramp = any(a.do == "incline" for a in exp.actions)
        base = knowledge.resolve(o.what, None if on_ramp else scene, {} if on_ramp else scene_labels)
        for k, nm in enumerate(copies):
            t = Thing(**{**base.__dict__})
            t.name = nm
            if o.size_m:
                s_ = o.size_m / max(t.dims)
                t.dims = [d * s_ for d in t.dims]
                t.mass *= s_ ** 3
            if o.mass_kg:
                t.mass = o.mass_kg
                t.extra["user_mass"] = True
            if o.bounciness is not None:
                t.bounciness = float(np.clip(o.bounciness, 0.0, 0.98))
            if any(x.mode == "swim" and (x.a == nm or x.a == o.name) for x in exp.actions) and t.dims[2] > 1.5 * t.dims[0]:
                t.dims = [t.dims[2], t.dims[1], t.dims[0]]          # a swimmer lies flat: long along the direction of travel
                t.extra["prone"] = True
            if t.source == "scene" and k > 0:          # only one real object per name; extra copies are stand-ins
                t.source, t.scene_id, t.shape = "knowledge", None, "cylinder"
            things[nm] = t
    # action names may refer to a base name ("baseball") when copies exist
    def ref(n):
        if n is None:
            return None
        if n in things:
            return n
        n2 = knowledge._singular(n)
        return next((k for k in things if k == n2 or k.startswith(n2 + " ") or n2 in k), None)
    for a in exp.actions:
        a.a = ref(a.a) or next(iter(things))
        a.b = ref(a.b)
        if a.do in ("collide", "fight", "hit", "chase", "pull", "rope", "spring", "race") and a.b is None:
            others = [k for k in things if k != a.a]
            a.b = others[0] if others else None
        if a.do in ("collide", "fight", "hit", "chase", "pull", "rope", "spring", "swing") and a.b is None:
            notes.append(f"'{a.do}' needs two objects; only one was found.")

    has_scene = scene is not None and len(scene.objects) > 0
    if scene is not None and not str(getattr(scene, "source", "")).startswith("sandbox"):
        # "on the table / desk / floor" with a scanned scene means the REAL surface
        surf = [nm for nm, t in things.items() if t.source != "scene" and re.sub(r" \d+$", "", nm) in ("table", "desk", "floor", "ground", "counter", "surface")]
        for nm in surf:
            del things[nm]
            exp.actions = [a for a in exp.actions if a.a != nm]        # "roll a ball across the table": the table doesn't roll
            for a in exp.actions:
                if a.b == nm:
                    a.b = None
        if (surf or exp.surface in ("table", "desk", "counter", "floor")) and exp.setting in ("table", "ground", "auto"):
            exp.setting = "scene"  # type: ignore[assignment]
        has_scene = True
    if exp.setting == "auto" and exp.surface:
        exp.setting = "table" if exp.surface in ("table", "desk", "counter") else "ground"  # type: ignore[assignment]
    setting = exp.setting if exp.setting != "auto" else knowledge.setting_for(exp.title, list(things.values()), has_scene)
    if setting == "table" and any((a.height or 0) > 0.6 or (a.speed or 0) > 4 or (a.distance or 0) > 1.5 for a in exp.actions):
        setting = "ground"            # metres of fall / road speeds don't fit on a tabletop
    if setting == "scene" and not has_scene:
        setting = "table"
    if setting == "scene" and any(a.a == ANCHOR for a in exp.actions) and not any(t.source == "scene" for t in things.values()):
        # a pendulum / mass on a spring needs a support above it, which a tabletop doesn't have: a hologram rig
        setting = "ground"
    if setting != "scene" and any(t.source == "scene" for t in things.values()):
        if has_scene:
            setting = "scene"
    comp_scale = 1.0
    real_scan = scene is not None and not str(getattr(scene, "source", "")).startswith("sandbox") and has_scene
    if real_scan and (setting == "scene" or force_scene):
        # It happens IN your mapped space (Live AR always; Studio when it involves your real objects). Anything too big for
        # the desk becomes a toy-size model: same density (mass ∝ L³), speeds/heights/distances scaled like a scale model
        # (Froude: v ∝ √L), and it is moved the way you'd move a toy — launched, not self-propelled: a toy has no muscles
        # or engine, and scaling those honestly isn't possible.
        if setting in ("water", "air", "space"):
            notes.append({"water": "Your desk has no water, so this runs on the table instead.", "air": "It starts from your table.",
                          "space": "With your desk's gravity, not in space."}[setting])
        T = scene.table
        fit = max(0.06, min(0.22, 0.25 * min(T.x_max - T.x_min, T.y_max - T.y_min)))
        biggest = max((max(t.dims) for t in things.values() if t.source != "scene"), default=0.0)
        k = min(1.0, fit / biggest) if biggest > 0 else 1.0
        if k < 0.999:
            for t in things.values():
                if t.source == "scene":
                    continue
                t.dims = [d * k for d in t.dims]
                t.mass = t.mass * k ** 3
                t.typical_speed = t.typical_speed * math.sqrt(k)
                t.self_propelled, t.habitat = False, "table"
                if t.asset_id:                      # a library mesh has its own fixed size: use the measured shape instead
                    t.asset_id, t.source = None, "knowledge"
                    if t.shape == "asset":
                        t.shape = "box"
            for a in exp.actions:
                a.mode = None
                if a.do in ("fight", "chase", "race", "crash", "attack"):
                    a.do = "collide"
                if a.do in ("pull", "rope", "push") and a.force is None:
                    a.force = None
                for f, p_ in (("speed", 0.5), ("height", 1.0), ("distance", 1.0), ("force", 3.0)):
                    v = getattr(a, f)
                    if v:
                        setattr(a, f, v * k ** p_)
            notes.append(f"Real size ({biggest:.3g} m) doesn't fit on your desk, so {'these are toy-size models' if len(things) > 1 else 'this is a toy-size model'} "
                         f"(1 : {1 / k:,.0f}) in your real scene — same density, speeds scaled by √scale like a scale model, and moved the way you'd move a toy.")
        setting = "scene"
        comp_scale = k
    size = max((max(t.dims) for t in things.values()), default=0.1)
    env = {"density": 1.2, "viscosity": 1.8e-5}
    gravity = G
    if setting == "water":
        salty = bool(re.search(r"\b(sea|ocean|salt\w*|marine|reef|beach)\b", exp.title.lower())) or any(
            t.habitat == "water" and t.material == "flesh" for t in things.values())
        env = {"density": 1025.0 if salty else 1000.0, "viscosity": 0.00108 if salty else 0.001}
    elif setting == "space":
        env, gravity = {"density": 0.0, "viscosity": 0.0}, 0.0
    if exp.gravity is not None:
        gravity = float(exp.gravity)
        if gravity < 5 and setting not in ("water",):
            env = {"density": 0.02 if gravity > 3 else 0.0, "viscosity": 0.0}      # Mars' thin air / Moon: no air
    env["timestep"] = 0.002 if size < 3 else 0.004

    ivs: list[dict] = []
    k_sid = [0]

    def new_sid():
        k_sid[0] += 1
        return f"x{k_sid[0]}"
    sid_of: dict[str, str] = {}
    # ---- the world ----
    if setting == "scene":
        world_scene = scene
    else:
        span = max(6 * size, 2.0) if setting != "table" else max(1.2, 8 * size)
        span = max(span, _needed_span(exp, things))
        gspan = span * 2 if setting == "table" else max(span * 6, 1500.0)    # the ground goes on and on
        cam = Camera(width=1280, height=720, fx=900, fy=900, cx=640, cy=360, hfov_deg=70,
                     R_wc=[[1, 0, 0], [0, 0, -1], [0, 1, 0]], t_wc=[0, -span, span * 0.5])
        world_scene = Scene(scene_id=f"sandbox-{setting}", source=f"sandbox:{setting}", camera=cam,
                            table=Table(x_min=-gspan, x_max=gspan, y_min=-gspan, y_max=gspan,
                                        height_above_floor=100.0 if setting in ("water", "air", "space") else 0.75 if setting == "table" else 50.0),
                            objects=[])
    # ---- placement ----
    incl = next((a for a in exp.actions if a.do == "incline"), None)
    pitch_of: dict[str, float] = {}
    if incl is not None and incl.a in things:
        tb = things[incl.a]
        th = math.radians(float(np.clip(incl.angle_deg or 30.0, 1.0, 80.0)))
        Lr = float(incl.distance or np.clip(14 * max(tb.dims), 1.0, 6.0))
        things["ramp"] = Thing(name="ramp", source="knowledge", shape="box", dims=[Lr, max(0.5, 3 * tb.dims[1]), 0.06], mass=80.0,
                               material="wood", color="#c8a06a", bounciness=0.1, note=f"{math.degrees(th):.0f}° ramp, {Lr:.1f} m long")
        things["ramp"].extra["fixed"] = True
        if incl.mu is not None:
            tb.extra["mu"] = incl.mu
            things["ramp"].extra["mu"] = incl.mu
    pos = _layout(exp, things, setting, world_scene, size)
    if incl is not None and "ramp" in things:
        tb, rp = things[incl.a], things["ramp"]
        Lr, T = rp.dims[0], rp.dims[2]
        h0 = Lr / 2 * math.sin(th) + 0.002
        pos["ramp"] = [0.0, 0.0, h0, 0.0]
        pitch_of["ramp"] = th
        R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
        xb = -Lr / 2 + tb.dims[0] * 0.6 + 0.05
        pb_ = np.array([0.0, 0.0, h0]) + R @ np.array([xb, 0.0, T + 0.001])
        pos[incl.a] = [float(pb_[0]), float(pb_[1]), float(pb_[2]), 0.0]
        pitch_of[incl.a] = th if tb.shape not in ("sphere",) else 0.0
    group0 = "spawn"
    for nm, t in things.items():
        if t.source == "scene":
            sid_of[nm] = t.scene_id
            so = next((o for o in world_scene.objects if o.id == t.scene_id), None)
            if so is not None:
                from .physics.sim import scene_drag_spec, scene_volume
                t.extra["drag"] = scene_drag_spec(so)
                if env.get("density", 0) > 0:
                    t.extra["volume"] = scene_volume(so)
            continue
        sid = new_sid()
        sid_of[nm] = sid
        p = pos[nm]
        flier = flies(t, next((x.mode for x in exp.actions if x.a == nm), None)) and setting not in ("water", "space")
        hold = setting in ("air", "space") or nm == ANCHOR or t.extra.get("fixed") or flier or any(a.do in ("drop", "bounce") and a.a == nm for a in exp.actions) \
            or any(a.do == "throw" and a.a == nm for a in exp.actions) \
            or any(a.do in ("spring", "swing", "rope") and a.a == ANCHOR and a.b == nm for a in exp.actions)
        params = {"pos": [round(float(x), 5) for x in p[:3]], "yaw": round(float(p[3]), 4), "label": nm, "group": group0,
                  "static": hold or (incl is not None and nm == incl.a), "mass": t.mass, "pitch": round(pitch_of.get(nm, 0.0), 5)}
        if env.get("density", 0) > 0:
            params["volume"] = displaced_volume(t)
            t.extra["volume"] = params["volume"]
        params["drag"] = drag_spec(t)
        t.extra["drag"] = params["drag"]
        if t.asset_id:
            params["asset"] = t.asset_id
            params["bounciness"] = t.bounciness
            if knowledge.archetype(t.name, t) == "ball" and max(t.dims) < 1.25 * min(t.dims):
                params["collision"] = "sphere"
        else:
            params["asset"] = "proxy"
            params["proxy"] = {"shape": t.shape, "dims": [round(d, 6) for d in t.dims], "mass": t.mass, "friction": _mu(t),
                               "surface": bool(t.extra.get("fixed")),     # ramps/walls: what moves on them sets the contact
                               "bounciness": t.bounciness, "color": t.color, "material": t.material,
                               "archetype": knowledge.archetype(t.name if nm != ANCHOR else "support", t)}
        ivs.append({"type": "spawn", "target": sid, "params": params})

    # ---- actions (all start together in one segment unless they depend on each other) ----
    group1 = "act"
    T_need = 0.0
    link_k = 0
    for a in exp.actions:
        A, B = sid_of.get(a.a), sid_of.get(a.b) if a.b else None
        ta, tb = things.get(a.a), things.get(a.b) if a.b else None
        if A is None:
            continue
        spd = _speed(a, ta, setting)
        pa = np.array(pos[a.a][:3]) if a.a in pos else _scene_pos(world_scene, A)
        pb = (np.array(pos[a.b][:3]) if a.b in pos else _scene_pos(world_scene, B)) if B else None
        ca = pa + [0, 0, ta.dims[2] / 2] if ta.source != "scene" else pa
        cb = (pb + [0, 0, tb.dims[2] / 2] if tb.source != "scene" else pb) if B else None
        if a.do == "push" and B is not None and not ta.self_propelled and setting not in ("water", "space", "air"):
            # "push X into Y": a hand pushes X steadily along the table until it reaches Y, then lets go
            mu_ = float(ta.extra.get("mu", _mu(ta)))
            load = 0.0
            if ta.source == "scene":
                so = next((o for o in world_scene.objects if o.id == A), None)
                load = sum(o.mass for o in world_scene.objects if so is not None and o.id in so.supporting)
            v_hand = float(a.speed) if a.speed and a.speed > 0 else 0.6 * (-a.speed if a.speed and a.speed < 0 else 1.0)
            F = 1.6 * mu_ * (ta.mass + load) * gravity + 4.0 * ta.mass
            ivs.append({"type": "propel", "target": A, "params": {"toward": B, "force": round(F, 4), "top_speed": round(v_hand, 4), "duration": 6.0,
                                                                    "planar": True, "hand": True, "group": group1}})
            gap = float(np.linalg.norm((cb - ca)[:2]))
            T_need = max(T_need, gap / max(0.05, v_hand) + 2.5)
        elif a.do in ("collide", "hit", "roll", "push") and B is not None or (a.do in ("push", "roll") and B is None):
            d = (cb - ca) if cb is not None else np.array([1.0, 0, 0])
            gap_ = float(np.linalg.norm((cb - ca)[:2])) if cb is not None else 0.5
            if setting not in ("water", "space", "air") and ta.shape not in ("sphere",) and not ta.self_propelled and not (a.speed and a.speed > 0):
                # a sliding object must have enough speed to cover the gap against friction
                share = 0.5 if (a.do == "collide" and tb is not None and tb.source != "scene") else 1.0
                spd = max(spd, 1.3 * math.sqrt(2 * _mu(ta) * gravity * gap_ * share))
            if setting not in ("water", "space", "air"):
                d[2] = 0
            d /= max(1e-9, np.linalg.norm(d))
            ivs.append({"type": "launch", "target": A, "params": {"velocity": (d * spd).round(4).tolist(), "group": group1}})
            if ta.self_propelled and setting not in ("space",):     # vehicles/animals keep driving into it
                fl = flies(ta, a.mode)
                ivs.append({"type": "propel", "target": A, "params": {"toward": B, "force": _thrust(ta, setting, a) * (1.0 if fl else 0.3), "top_speed": spd,
                                                                        "duration": 3.0 if not fl else 60.0, "planar": setting not in ("water", "air") and not fl, "group": group1,
                                                                        **({"lift": True} if fl else {})}})
            if a.do == "collide" and B is not None and tb is not None and tb.source != "scene" and not tb.extra.get("fixed"):
                spb = _speed(a, tb, setting)
                if setting not in ("water", "space", "air") and tb.shape not in ("sphere",) and not tb.self_propelled and not (a.speed and a.speed > 0):
                    spb = max(spb, 1.3 * math.sqrt(2 * _mu(tb) * gravity * gap_ * 0.5))
                ivs.append({"type": "launch", "target": B, "params": {"velocity": (-d * spb).round(4).tolist(), "group": group1}})
                if tb.self_propelled and setting not in ("space",):
                    ivs.append({"type": "propel", "target": B, "params": {"toward": A, "force": _thrust(tb, setting, a) * 0.3, "top_speed": _speed(a, tb, setting),
                                                                            "duration": 3.0, "planar": setting not in ("water", "air"), "group": group1}})
                closing = spd + spb
            else:
                closing = spd
            gap = float(np.linalg.norm((cb - ca)[:2] if cb is not None else [0, 0]))
            T_need = max(T_need, gap / max(0.05, closing) + 2.0)
        elif a.do in ("fight", "chase", "race"):
            if B is None and a.do != "race":
                continue
            if a.do == "fight":
                _propel_pair(ivs, A, B, ta, tb, setting, a, group1)
                T_need = max(T_need, (a.duration or 8.0) + 1.0)
            elif a.do == "chase":
                ivs.append({"type": "propel", "target": A, "params": {"toward": B, "force": _thrust(ta, setting, a), "duration": a.duration or 6.0, "top_speed": ta.typical_speed,
                                                                        "planar": setting not in ("water", "air", "space"), "group": group1}})
                if tb is not None and tb.self_propelled:
                    away = (pb - pa) / max(1e-9, np.linalg.norm(pb - pa))
                    ivs.append({"type": "propel", "target": B, "params": {"direction": away.round(4).tolist(), "force": _thrust(tb, setting, a) * 0.8, "top_speed": tb.typical_speed,
                                                                            "duration": a.duration or 5.0, "planar": setting not in ("water", "air", "space"), "group": group1}})
            else:
                for nm, t in things.items():
                    if t.source == "scene" or nm == ANCHOR or not t.self_propelled:
                        continue
                    _move(ivs, sid_of[nm], t, setting, a, group1, [1.0, 0.0, 0.0], notes)
            T_need = max(T_need, (a.duration or 6.0) + 1.5)
        elif a.do == "move":
            d = (cb - ca) if cb is not None else np.array([1.0, 0, 0])
            if setting not in ("water", "air", "space") and not flies(ta, a.mode):
                d[2] = 0
            d = d / max(1e-9, np.linalg.norm(d))
            v = _move(ivs, A, ta, setting, a, group1, d.round(4).tolist(), notes, toward=B)
            acc = max(0.3, (a.force or _thrust(ta, setting, a, v)) / max(ta.mass, 1e-6))
            T_need = max(T_need, a.duration or float(np.clip(2.2 * v / acc + 3.5, 4.0, 12.0)))
        elif a.do == "jump":
            nm_ = ta.name.lower()
            if any(w in nm_ for w in NO_JUMP):
                notes.append(f"A real {ta.name} can't jump (too heavy for its legs, or no legs) — shown attempting the smallest hop.")
            h_earth = a.height or jump_height(ta)
            v0 = math.sqrt(2 * G * h_earth)          # the same muscles give the same take-off speed anywhere
            if any(w in nm_ for w in NO_JUMP):
                v0 = min(v0, 0.3)
            fwd = gait_speed(ta, "run", ExAction(do="move", a=a.a)) * 0.6 if re.search(r"long jump|forward|over|across", exp.title.lower()) else 0.0
            ivs.append({"type": "launch", "target": A, "params": {"velocity": [round(fwd, 4), 0.0, round(v0, 4)], "group": group1}})
            T_need = max(T_need, 2 * v0 / max(gravity, 0.3) + 2.0)
        elif a.do == "throw":
            if ta.source == "scene":
                # a real object is picked up first (15 cm above where it sits), then thrown from the hand
                ca = np.array([ca[0], ca[1], ca[2] + 0.15])
                ivs.append({"type": "place", "target": A, "params": {"pos": ca.round(5).tolist(), "hold": True, "group": group0}})
                if cb is not None and tb is not None:
                    cb = np.array([cb[0], cb[1], cb[2]])
            if a.angle_deg is not None and cb is None:
                # "thrown at 20 m/s at 45°": exactly that launch velocity
                th_ = math.radians(float(np.clip(a.angle_deg, -89.0, 89.0)))
                v = np.array([spd * math.cos(th_), 0.0, spd * math.sin(th_)])
                vz_, h_ = spd * math.sin(th_), max(0.0, ca[2] - (ta.dims[2] / 2 if ta.source != "scene" else 0.0))
                T_fl = (vz_ + math.sqrt(vz_ * vz_ + 2 * max(gravity, 0.1) * h_)) / max(gravity, 0.1)
                ivs.append({"type": "launch", "target": A, "params": {"velocity": v.round(4).tolist(), "group": group1}})
                T_need = max(T_need, T_fl + 1.5)
            else:
                target = cb if cb is not None else ca + np.array([max(2.0, 8 * max(ta.dims)), 0, -ca[2] + 0.01])
                if cb is not None and tb is not None:
                    # aim the thrown thing's centre at the target's upper half - never below the height its own
                    # centre has when lying on the table, or it lands short
                    top_b = cb[2] + tb.dims[2] / 2
                    target = np.array([cb[0], cb[1], max(cb[2] + tb.dims[2] / 4, min(top_b, ta.dims[2] / 2 + 0.005))])
                v = _ballistic(ca, target, spd, gravity, high=bool(a.angle_deg and a.angle_deg > 45))
                ivs.append({"type": "launch", "target": A, "params": {"velocity": v.round(4).tolist(), "group": group1}})
                T_need = max(T_need, float(np.linalg.norm(target - ca)) / max(0.5, float(np.linalg.norm(v))) + 2.5)
        elif a.do in ("drop", "bounce"):
            if ta.source == "scene":
                # a real object is picked up and held above its target (or above where it sits), then let go
                hh = a.height or 0.3
                if cb is not None and tb is not None:
                    top_b = cb[2] + (tb.dims[2] / 2 if tb.source == "scene" else tb.dims[2] / 2)
                    lift = [float(cb[0]), float(cb[1]), float(top_b + hh + ta.dims[2] / 2)]
                else:
                    lift = [float(ca[0]), float(ca[1]), float(ca[2] + hh)]
                ivs.append({"type": "place", "target": A, "params": {"pos": [round(x, 5) for x in lift], "hold": True, "group": group0}})
                pa = np.array(lift)
            ivs.append({"type": "release", "target": A, "params": {"group": group1}})
            h = pa[2]
            T_need = max(T_need, math.sqrt(2 * max(h, 0.01) / max(gravity, 0.1)) + (3.0 if a.do == "bounce" else 2.0))
            if setting == "water":
                T_need = max(T_need, 9.0)          # long enough to rise/sink and settle at the water line
        elif a.do == "spin":
            w = (a.rpm or 120.0) * 2 * math.pi / 60
            ivs.append({"type": "spin", "target": A, "params": {"omega": [0, 0, w], "group": group1}})
            T_need = max(T_need, 4.0)
        elif a.do in ("rope", "spring", "pull", "swing") and B is not None and a.a == ANCHOR:
            # hanging from the fixed support: a pendulum (rope) or a mass on a spring
            link_k += 1
            L0 = float(pos[a.b][3 + 1]) if len(pos[a.b]) > 4 else float(np.linalg.norm(cb - ca))
            k = a.stiffness or (4 * math.pi ** 2 * tb.mass / 1.0)
            ivs.append({"type": "connect", "target": None, "params": {"link": f"link{link_k}", "a": A, "b": B,
                                                                      "kind": "spring" if a.do == "spring" else "rope",
                                                                      "length": round(L0, 4), "stiffness": round(k, 3),
                                                                      "damping": round(0.02 * math.sqrt(k * tb.mass), 4), "group": group1}})
            ivs.append({"type": "release", "target": B, "params": {"group": group1}})
            T_need = max(T_need, a.duration or 8.0)
        elif a.do in ("rope", "spring", "pull") and B is not None:
            if a.do in ("rope", "spring"):
                link_k += 1
                L0 = float(np.linalg.norm(cb - ca)) if a.distance is None else a.distance
                k = a.stiffness or (4 * math.pi ** 2 * min(ta.mass, tb.mass) / 1.0)     # ~1 s natural period by default
                ivs.append({"type": "connect", "target": None, "params": {"link": f"link{link_k}", "a": A, "b": B, "kind": a.do,
                                                                          "length": round(L0 * (1.0 if a.do == "spring" else 1.02), 4),
                                                                          "stiffness": round(k, 3), "damping": round(0.05 * math.sqrt(k * min(ta.mass, tb.mass)), 4),
                                                                          "group": group1}})
                if a.do == "spring":
                    # stretch it: pull B away a little so it oscillates
                    d = (cb - ca) / max(1e-9, np.linalg.norm(cb - ca))
                    ivs.append({"type": "launch", "target": B, "params": {"velocity": (d * (spd if a.speed else 0.6)).round(4).tolist(), "group": group1}})
                T_need = max(T_need, 6.0)
            else:   # pull: the puller walks/swims away from the pulled thing, with traction-limited force
                away = (pa - pb) / max(1e-9, np.linalg.norm(pa - pb))
                F = a.force or (_thrust(ta, setting, a) if setting in ("water", "air", "space") else PULL_GRIP * ta.mass * gravity)
                ivs.append({"type": "propel", "target": A, "params": {"direction": away.round(4).tolist(), "force": F, "top_speed": round(1.5 * math.sqrt(comp_scale), 3),
                                                                        "duration": a.duration or 6.0, "planar": setting not in ("water", "air", "space"),
                                                                        "group": group1}})
                T_need = max(T_need, (a.duration or 6.0) + 1.0)
        elif a.do == "incline":
            ivs.append({"type": "release", "target": A, "params": {"group": group1}})
            th_ = math.radians(a.angle_deg or 30.0)
            mu_ = ta.extra.get("mu", _mu(ta))
            acc_ = max(0.05, gravity * (math.sin(th_) - mu_ * math.cos(th_)))
            T_need = max(T_need, math.sqrt(2 * things["ramp"].dims[0] / acc_) + 2.0 if acc_ > 0.06 else 4.0)
        elif a.do == "remove":
            ivs.append({"type": "remove", "target": A, "params": {"group": group1}})
            T_need = max(T_need, 3.0)
        elif a.do in ("place", "stack"):
            T_need = max(T_need, 2.0)
        elif a.do == "orbit" and B is not None:
            notes.append("Orbits need gravity between bodies, which this simulator doesn't model yet; showing a spin instead.")
            ivs.append({"type": "spin", "target": A, "params": {"omega": [0, 0, 1.0], "group": group1}})
            T_need = max(T_need, 4.0)
    # fliers carry their own weight once the action starts
    for nm, t in things.items():
        if nm in sid_of and flies(t, next((x.mode for x in exp.actions if x.a == nm), None)) and setting not in ("water", "space"):
            ivs.append({"type": "release", "target": sid_of[nm], "params": {"group": group1}})
            for iv in ivs:
                if iv["type"] == "propel" and iv["target"] == sid_of[nm]:
                    iv["params"]["lift"] = True
                    iv["params"]["planar"] = False
    # held objects that nothing releases (air/space props) are released when the action starts;
    # supports and buildings stay put
    fixed_sids = {sid_of[nm] for nm, t in things.items() if nm == ANCHOR or t.extra.get("fixed")}
    ivs = [iv for iv in ivs if not (iv["target"] in fixed_sids and iv["type"] in ("launch", "propel", "release"))]
    acted = {iv["target"] for iv in ivs if iv["params"].get("group") == group1}
    for iv in [x for x in ivs if x["type"] == "spawn" and x["params"]["static"]]:
        if iv["target"] not in acted and setting not in ("space",) and iv["target"] not in fixed_sids:
            ivs.append({"type": "release", "target": iv["target"], "params": {"group": group1}})
    duration = float(min(20.0, max(2.5, exp.seconds or T_need or 3.0)))
    if setting == "water":
        tops = [p_[2] + things[nm].dims[2] for nm, p_ in pos.items() if nm in things]
        env["surface_z"] = round(max(tops + [0.0]) + max(1.0, 1.5 * size), 3)
    if incl is not None and "ramp" in things:
        notes.append(f"A {math.degrees(th):.0f}° ramp, {things['ramp'].dims[0]:.1f} m long, friction coefficient {_mu(things[incl.a]):.2f}.")
    mu = {nm: _mu(t) for nm, t in things.items()}
    pull_max = {nm: PULL_GRIP * t.mass * gravity for nm, t in things.items() if t.self_propelled}
    return {"things": things, "sid_of": sid_of, "setting": setting, "env": env | {"max_segment_s": duration}, "gravity": gravity, "mu": mu, "pull_max": pull_max,
            "scene": world_scene, "interventions": ivs, "duration": duration, "notes": notes, "positions": pos}


def _needed_span(exp, things) -> float:
    s = 1.0
    for a in exp.actions:
        if a.distance:
            s = max(s, a.distance * 1.5)
    return s


def _speed(a: ExAction, t: Thing | None, setting: str) -> float:
    if a.speed and a.speed > 0:
        return float(a.speed)
    if a.do == "throw" and t is not None and not t.self_propelled and setting != "table":
        v = 0.5 * t.typical_speed * (-a.speed if a.speed and a.speed < 0 else 1.0)
        return float(np.clip(v, 2.0, 45.0))
    base = (t.typical_speed if t else 2.0)
    # for collisions of passive objects, a lively but ordinary speed for their size
    size = max(t.dims) if t else 0.1
    default = min(max(base * 0.35, 1.0), max(1.0, 8 * size)) if not (t and t.self_propelled) else base * 0.5
    if setting == "table":
        default = min(default, 1.5)
    if a.speed and a.speed < 0:
        default *= -a.speed           # "fast" x1.5, "slowly" x0.5
    return float(max(0.2, default))


HUMAN_GAIT = {"walk": 1.4, "jog": 3.0, "run": 4.5, "sprint": 7.5, "swim": 1.2, "crawl": 0.4, "drive": 1.4, "fly": 1.4}
GAIT_FRAC = {"walk": 0.22, "jog": 0.45, "run": 0.8, "sprint": 1.0, "swim": 0.15, "crawl": 0.08, "drive": 0.6, "fly": 0.8}
JUMP_H = {"kangaroo": 1.8, "cat": 1.5, "dog": 0.9, "frog": 1.0, "rabbit": 0.9, "horse": 1.0, "deer": 2.0, "cheetah": 1.5, "leopard": 2.5,
          "lion": 1.5, "tiger": 1.5, "monkey": 1.5, "flea": 0.18, "grasshopper": 0.5, "goat": 1.5, "basketball player": 0.8, "athlete": 0.7}
NO_JUMP = ("elephant", "hippo", "rhino", "whale", "turtle", "tortoise", "snail", "car", "truck", "bus", "train")


def gait_speed(t: Thing, mode: str | None, a: ExAction) -> float:
    """How fast it really goes for that kind of motion (m/s): people walk 1.4, jog 3, sprint ~7.5;
    animals and machines as a fraction of their top speed; fish swim at ~60% of top."""
    if a.speed and a.speed > 0:
        return float(a.speed)
    arch = knowledge.archetype(t.name, t)
    T = max(0.1, t.typical_speed)
    m = mode or ("swim" if arch == "fish" else "fly" if arch in ("winged", "aircraft") and t.habitat == "air" else "drive" if arch in ("vehicle", "boat") else "run")
    if arch == "biped" and not re.search(r"robot", t.name):
        v = HUMAN_GAIT.get(m, 3.0)
    elif arch == "fish":
        v = 0.6 * T
    elif arch in ("winged", "aircraft") and m == "fly":
        v = 0.8 * T
    elif arch in ("vehicle", "boat", "aircraft"):
        v = (1.0 if m in ("sprint", "run") else 0.5) * T
    else:
        v = GAIT_FRAC.get(m, 0.6) * T
    if a.speed and a.speed < 0:
        v *= -a.speed                     # "fast" ×1.5, "slowly" ×0.5, "2x speed" ×2
    return float(v)


def flies(t: Thing, mode: str | None) -> bool:
    arch = knowledge.archetype(t.name, t)
    return t.self_propelled and (t.habitat == "air" or mode == "fly") and arch in ("winged", "aircraft")


def jump_height(t: Thing) -> float:
    n = t.name.lower()
    for k, h in JUMP_H.items():
        if k in n:
            return h
    arch = knowledge.archetype(n, t)
    return 0.5 if arch == "biped" else max(0.05, 0.6 * t.dims[2]) if t.self_propelled else 0.3


def _blew_up(sim: dict, comp: dict) -> bool:
    """MuJoCo resets a diverged state and carries on, so check the recorded velocities: anything non-finite, or a body moving
    faster than anything in the set-up could make it (100x the fastest requested speed, free fall over 100 m, 10 m/s floor)."""
    vmax_req = max([a.get("speed") or 0 for a in comp.get("actions", [])] + [0.0])
    limit = max(10.0, 100 * vmax_req, 3 * math.sqrt(2 * 9.81 * 100))
    for k in sim.get("kinematics", {}).values():
        for v in k.get("v", []):
            if v is None:
                continue
            if not all(math.isfinite(x) for x in v) or math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) > limit:
                return True
    return False


def _thrust(t: Thing, setting: str, a: ExAction, v_target: float | None = None) -> float:
    """Self-propulsion force. On land it comes from grip (<= mu*m*g); in water/air it is the thrust
    that balances drag at the animal's typical speed; in space a gentle 1 m/s^2."""
    if a.force:
        return float(a.force)
    if setting in ("water", "air") or (v_target and setting == "flying"):
        rho = 1025.0 if setting == "water" else 1.2
        area = math.pi / 4 * t.dims[1] * t.dims[2] if setting == "water" else math.pi / 4 * t.dims[2] * min(t.dims[1], t.dims[2])
        v = v_target or t.typical_speed
        if setting == "flying":
            return float(0.5 * rho * 1.2 * area * v ** 2 + 0.25 * t.mass * G)
        # enough to reach its real cruising speed against drag (the taper then caps the speed)
        return float(0.5 * rho * 1.2 * area * v ** 2 + 0.8 * t.mass)
    if setting == "space":
        return float(t.mass * 1.0)
    return float(0.5 * GRIP * t.mass * G)       # traction-limited push (half of maximum grip)


# real densities of solid materials (kg/m^3); hollow things (balls, toys, bottles) use their outer shape
SOLID_DENSITY = {"flesh": 985.0, "metal": 7800.0, "stone": 2600.0, "wood": 600.0, "ice": 917.0, "food": 980.0,
                 "ceramic": 2300.0, "shell": 1500.0, "wax": 900.0, "paper": 700.0}


def drag_spec(t: Thing) -> dict:
    """The drag law's description of a thing (same as the simulator uses): shape, own-axis size, C_d."""
    from .physics import drag, sim as _sim
    if knowledge.archetype(t.name, t) == "ball" and max(t.dims) < 1.25 * min(t.dims):
        d = max(t.dims)
        return drag.spec("sphere", [d, d, d], t.extra.get("cd"))
    if t.asset_id:
        from . import assets as asset_lib
        a = asset_lib.manifest().get(t.asset_id)
        if a is not None:
            return _sim._drag_spec({"asset": a})
    return drag.spec(t.shape, t.dims, t.extra.get("cd"))


def displaced_volume(t: Thing) -> float:
    L, W, H = t.dims
    k = {"sphere": math.pi / 6, "ellipsoid": math.pi / 6, "cylinder": math.pi / 4, "capsule": 0.72}.get(t.shape, 1.0)
    shape_v = k * L * W * H
    rho = SOLID_DENSITY.get(t.material)
    if rho is None:
        return float(shape_v)
    if t.material == "flesh" and t.habitat == "water":
        rho = 1025.0                     # marine animals are neutrally buoyant in seawater
    return float(min(shape_v, t.mass / rho))


def _move(ivs, sid, t: Thing, setting: str, a: ExAction, group: str, direction, notes: list[str], toward=None) -> float:
    """Self-propelled motion (run, walk, swim, fly, drive, crawl) at the real speed for that gait."""
    mode = a.mode
    arch = knowledge.archetype(t.name, t)
    if mode == "fly" and not flies(t, "fly"):
        notes.append(f"A {t.name} can't fly: it would need wings or engines making a lift force equal to its weight "
                     f"({t.mass * G:,.0f} N). Showing it {'swimming' if setting == 'water' else 'running'} instead.")
        mode = "swim" if setting == "water" else "run"
        a.mode = mode
    if mode == "swim" and setting != "water":
        notes.append(f"Swimming needs water: the {t.name} is placed in water.")
    v = gait_speed(t, mode, a)
    natural_max = max(t.typical_speed * (1.15 if arch != "biped" else 1.0), 10.4 if arch == "biped" else 0)
    if v > natural_max * 1.05 and not (a.speed and a.speed > 0 and arch in ("vehicle", "aircraft", "boat")):
        notes.append(f"{v:.1f} m/s is faster than any real {t.name} can go (about {natural_max:.1f} m/s"
                     + (", Usain Bolt's peak was 12.4 m/s" if arch == "biped" else "") + ") — simulated anyway, with the push it would need.")
    fly = flies(t, mode)
    fluid = setting in ("water", "air", "space") or fly
    F = a.force or _thrust(t, "flying" if fly and setting not in ("water", "space") else setting, a, v)
    if not fluid and v > t.typical_speed:
        F *= min(3.0, v / max(0.1, t.typical_speed))          # a faster runner/driver pushes harder (capped by grip below)
    if not fluid:
        F = min(F, GRIP * t.mass * G * (1.0 if arch != "vehicle" else 1.2))
    params = {"force": float(F), "top_speed": float(v), "duration": a.duration or (60.0 if fly else 10.0), "planar": not fluid, "group": group}
    if toward:
        params["toward"] = toward
    else:
        params["direction"] = direction
    if fly:
        params["lift"] = True
    ivs.append({"type": "propel", "target": sid, "params": params})
    return v


def _propel_pair(ivs, A, B, ta, tb, setting, a, group):
    planar = setting not in ("water", "air", "space")
    for X, Y, tx in ((A, B, ta), (B, A, tb)):
        if tx is None or tx.source == "scene":
            continue
        ivs.append({"type": "propel", "target": X, "params": {"toward": Y, "force": _thrust(tx, setting, a), "top_speed": tx.typical_speed * 0.6,
                                                                "duration": a.duration or 8.0, "planar": planar, "group": group}})


def _ballistic(p0, p1, speed, g, high=False) -> np.ndarray:
    """Launch velocity that lands on p1 from p0 at the given speed (low or high arc).
    If the speed can't reach, use the minimum speed that does (45 deg-ish)."""
    d = p1 - p0
    horiz = np.array([d[0], d[1], 0.0])
    x = float(np.linalg.norm(horiz))
    y = float(d[2])
    u = horiz / max(1e-9, x)
    if g <= 1e-6 or x < 1e-6:
        return d / max(1e-9, np.linalg.norm(d)) * speed
    v2 = speed * speed
    disc = v2 * v2 - g * (g * x * x + 2 * y * v2)
    if disc < 0:      # too slow to reach: minimum-speed solution
        v2 = g * (y + math.sqrt(x * x + y * y))
        disc = max(0.0, v2 * v2 - g * (g * x * x + 2 * y * v2))
        speed = math.sqrt(v2)
    root = math.sqrt(disc)
    th = math.atan((v2 + root) / (g * x)) if high else math.atan((v2 - root) / (g * x))
    return u * speed * math.cos(th) + np.array([0, 0, speed * math.sin(th)])


def _scene_pos(scene, oid):
    o = next((x for x in scene.objects if x.id == oid), None)
    return np.array(o.pos) if o is not None else np.zeros(3)


def _layout(exp, things, setting, scene, size) -> dict[str, list[float]]:
    """Positions (x, y, z_bottom, yaw) for every new object, chosen from the actions."""
    pos: dict[str, list[float]] = {}
    real = {nm: t for nm, t in things.items() if t.source == "scene"}
    new = [nm for nm, t in things.items() if t.source != "scene"]
    if setting == "water":
        base_z = max(3.0, 2.5 * size)
    elif setting in ("air",):
        base_z = max(2.0, 6 * size)
    elif setting == "space":
        base_z = 0.0
    else:
        base_z = 0.0008
    if setting == "scene":
        return _layout_scene(exp, things, scene)
    # pairs first, along x; everything else in rows along y
    placed = set()
    row = 0
    for a in exp.actions:
        if a.a in placed and (a.b is None or a.b in placed):
            continue
        ta, tb = things.get(a.a), things.get(a.b) if a.b else None
        ra = max(ta.dims[:2]) / 2 if ta else 0.1
        rb = max(tb.dims[:2]) / 2 if tb else 0.0
        if a.distance and a.do not in ("rope", "spring"):
            gap = a.distance
        elif a.do == "fight":
            gap = 0.3 * (ra + rb) + 0.2                 # face to face, then they charge
        elif a.do == "chase":
            gap = 0.8 * (ra + rb) + 0.2                 # a short head start
        elif a.do in ("collide", "hit", "roll", "push"):
            v = _speed(a, ta, setting) if ta else 1.0
            gap = float(np.clip(v * 0.6, 1.0 * (ra + rb), 25 * (ra + rb) + 0.3))
        elif a.do == "throw":
            gap = max(2.0, 15 * (ra + rb)) if setting != "table" else 0.7
        elif a.do in ("rope", "pull"):
            gap = a.distance or max(1.0, 3 * (ra + rb))
        elif a.do == "spring":
            gap = a.distance or max(0.3, 3 * (ra + rb))
        else:
            gap = 2.5 * (ra + rb) + 0.1
        y = row * max(3 * (ra + rb) + 0.5, 1.0)
        za = base_z
        if a.do in ("drop", "bounce"):
            h = a.height or (1.0 if setting != "table" else 0.3)
            za = base_z + h
            if tb is not None and a.b not in placed:
                pos[a.b] = [0.0, y, base_z, 0.0]
                placed.add(a.b)
                za = base_z + tb.dims[2] + h
            if a.a not in placed:
                pos[a.a] = [0.0, y, za, 0.0]
                placed.add(a.a)
            row += 1
            continue
        if a.a == ANCHOR and a.b is not None and tb is not None:
            # support at the top; the object hangs below (spring: at its unstretched length; pendulum: pulled aside)
            L = a.distance or (0.5 if a.do == "spring" else 1.0)
            hb = tb.dims[2]
            top = base_z + L + hb + 0.4
            pos[ANCHOR] = [0.0, y, top, 0.0]
            ac = top + 0.03                                      # anchor centre
            if a.do == "spring":
                cz = ac - L - hb / 2
                pos[a.b] = [0.0, y, cz - hb / 2, 0.0, L]
            else:
                th = math.radians(a.angle_deg or 30.0)
                pos[a.b] = [L * math.sin(th), y, ac - L * math.cos(th) - hb / 2, 0.0, L]
            placed |= {ANCHOR, a.b}
            row += 1
            continue
        if a.do == "throw":
            za = base_z + (0.01 if a.from_ground else 1.5 if setting not in ("table",) else 0.25)   # kicked: just clear of the turf
        if ta is not None and flies(ta, a.mode) and setting not in ("water", "space", "air"):
            # fliers cruise at an altitude: level with what they're heading for, else a few body-heights up
            za = base_z + (max(0.3, tb.dims[2] * 0.5 - ta.dims[2] * 0.5) if tb is not None else max(2.0, 3 * ta.dims[2]))
        if a.a not in placed:
            pos[a.a] = [-(gap / 2 + ra), y, za, 0.0]
            placed.add(a.a)
        if a.b and a.b not in placed and tb is not None:
            pos[a.b] = [gap / 2 + rb, y, base_z, math.pi]
            placed.add(a.b)
        row += 1
    for nm in new:
        if nm not in placed:
            t = things[nm]
            pos[nm] = [0.0, row * max(1.0, 3 * max(t.dims)), base_z, 0.0]
            row += 1
    return pos


def _layout_scene(exp, things, scene) -> dict[str, list[float]]:
    """New things in the real scanned scene: free, visible spots (same placement logic as commands)."""
    from . import planner
    ents = planner.snapshot(scene, None, [])
    by_name = {}
    for nm, t in things.items():
        if t.source == "scene":
            by_name[nm] = next((e for e in ents if e.id == t.scene_id), None)
    pos = {}
    for a in exp.actions:
        # the thing something is dropped/stacked ON is placed first, so the dropped one can go above it
        order = ((a.b, a.a), (a.a, a.b)) if a.do in ("drop", "bounce", "stack", "place") else ((a.a, a.b), (a.b, a.a))
        for nm, partner in order:
            if nm is None or nm in pos or things[nm].source == "scene":
                continue
            t = things[nm]
            r = 0.5 * math.hypot(t.dims[0], t.dims[1])
            ref = by_name.get(partner) if partner else None
            rel = "approach" if a.do in ("collide", "hit", "throw", "roll", "push", "chase", "fight") else ("above" if nm == a.a else "near") if a.do in ("drop", "bounce") else "near"
            xy, yaw, _ = planner.place(scene, ents, r, rel if ref is not None else "free", ref)
            z = 0.0008
            if rel == "above" and ref is not None and nm == a.a:
                z = ref.top + (a.height or 0.25)
            elif rel == "above" and nm == a.a:
                z = a.height or 0.3                      # dropped onto the table from a height
            elif a.do == "throw":
                z = 0.2
            pos[nm] = [float(xy[0]), float(xy[1]), z, float(yaw)]
            e = planner.Entity(nm, nm, np.array([xy[0], xy[1], z]), r, z + t.dims[2], spawned=True)
            ents.append(e)
            by_name[nm] = e
    for nm, t in things.items():
        if t.source != "scene" and nm not in pos:
            xy, yaw, _ = planner.place(scene, ents, 0.5 * math.hypot(t.dims[0], t.dims[1]), "free", None)
            pos[nm] = [float(xy[0]), float(xy[1]), 0.0008, float(yaw)]
    return pos


def _energy_gain(sim: dict, comp: dict) -> float:
    """Largest rise of mechanical energy (KE + PE) of the unpowered new objects during the action,
    relative to the most kinetic energy they ever had. Objects driven by a motor/legs, or tied to
    one, legitimately gain energy and are left out."""
    powered = {iv["target"] for iv in comp["interventions"] if iv["type"] in ("propel", "drive")}
    for iv in comp["interventions"]:
        if iv["type"] == "connect" and (iv["params"]["a"] in powered or iv["params"]["b"] in powered):
            powered |= {iv["params"]["a"], iv["params"]["b"]}
    ids = [sid for sid in comp["sid_of"].values() if sid in sim["kinematics"] and sid not in powered]
    if not ids or not sim["segments"]:
        return 0.0
    f0 = int(sim["segments"][-1]["t_start"] * sim["fps"]) + 2
    E, K = [], []
    for f in range(f0, len(sim["frames"])):
        e = k_ = 0.0
        for sid in ids:
            k = sim["kinematics"][sid]
            if k["KE"][f] is not None:
                e += k["KE"][f] + k["PE"][f]
                k_ += k["KE"][f]
        E.append(e)
        K.append(k_)
    if len(E) < 3:
        return 0.0
    scale = max(max(K), 1e-9)
    return float(max(0.0, max(E) - E[0]) / scale)


def run(text: str, scene=None, baseline_qpos=None, extra_measures: list[str] | None = None, force_scene: bool = False) -> dict:
    """End to end: words -> experiment -> world -> simulation -> measurements + why."""
    labels = {o.id: o.label for o in scene.objects} if scene is not None else {}
    exp = understand(text, labels)
    if not exp.objects:
        raise ValueError("I couldn't find anything to simulate in that. Name the objects, e.g. "
                         "'two baseballs collide', 'drop an egg from 2 m', 'a man pulls a truck with a rope'.")
    return run_experiment(exp, scene, baseline_qpos, extra_measures, force_scene)


def run_experiment(exp: Experiment, scene=None, baseline_qpos=None, extra_measures: list[str] | None = None, force_scene: bool = False) -> dict:
    """Simulate an already-understood (possibly user-edited) experiment."""
    from . import exp_analysis
    from .physics.sim import simulate
    from .schemas import Intervention
    exp = exp.model_copy(deep=True)
    labels = {o.id: o.label for o in scene.objects} if scene is not None else {}
    comp = compile_experiment(exp, scene, labels, force_scene)
    comp["actions"] = [a.model_dump() for a in exp.actions]
    ivs = [Intervention(**iv) for iv in comp["interventions"]]
    base = baseline_qpos if comp["setting"] == "scene" and baseline_qpos is not None else np.zeros(0)
    sim = simulate(comp["scene"], base, ivs, gravity=comp["gravity"], env=comp["env"])
    # Physics check: with nothing powered, kinetic + potential energy can only go down. If it went
    # up (deep, fast contacts wedging between surfaces), re-simulate with a finer time step.
    refined = 0
    while (_energy_gain(sim, comp) > 0.05 or _blew_up(sim, comp)) and refined < 2:
        comp["env"] = comp["env"] | {"timestep": comp["env"]["timestep"] / 2}
        sim = simulate(comp["scene"], base, ivs, gravity=comp["gravity"], env=comp["env"])
        refined += 1
    if _blew_up(sim, comp):
        raise ValueError("That set-up made the simulation numerically unstable (bodies flew off at impossible speeds), so I won't show it. "
                         "Try it with things closer in size, or in Studio at real size.")
    if refined:
        comp["notes"].append(f"Re-simulated at a {comp['env']['timestep'] * 1000:.1f} ms time step so energy is conserved through the fast impact.")
    analysis = exp_analysis.analyse(comp, sim, exp.measure + list(extra_measures or []))
    objects = []
    for nm, t in comp["things"].items():
        objects.append({"name": nm, "source": t.source, "shape": t.shape, "dims": [float(d) for d in t.dims], "mass": float(t.mass),
                        "material": t.material, "color": t.color, "note": t.note, "sid": comp["sid_of"].get(nm),
                        "bounciness": float(t.bounciness), "self_propelled": bool(t.self_propelled), "habitat": t.habitat,
                        "typical_speed": float(t.typical_speed), "asset_id": t.asset_id, "scene_id": t.scene_id,
                        "fixed": bool(t.extra.get("fixed")) or nm == ANCHOR, "support": nm == ANCHOR,
                        "archetype": knowledge.archetype(nm if nm != ANCHOR else "support", t)})
    return {
        "experiment": exp.model_dump(),
        "setting": comp["setting"],
        "gravity": comp["gravity"],
        "env": {k: v for k, v in comp["env"].items()},
        "objects": objects,
        "notes": comp["notes"],
        "interventions": comp["interventions"],
        "sim": sim,
        "results": analysis["rows"],
        "story": analysis["story"],
        "fx": fx_events(sim, comp),
        "scene_table": comp["scene"].table.model_dump(),
        "_compiled": comp,
    }


def fx_events(sim: dict, comp: dict) -> list[dict]:
    """Moments worth a visual effect, all from recorded physics: object-object impacts (sized by
    peak force), hard landings on the ground, and objects crossing the water surface."""
    fps = sim["fps"]
    out: list[dict] = []
    size = {sid: max(comp["things"][nm].dims) for nm, sid in comp["sid_of"].items() if nm in comp["things"]}

    def com(sid, f):
        k = sim["kinematics"].get(sid)
        if not k:
            return None
        f = max(0, min(len(k["c"]) - 1, f))
        return k["c"][f]
    for key, rows in sim.get("pair_forces", {}).items():
        a, b = key.split("|")
        last = -99
        for f, F, pk, J in rows:
            if f - last > 3 and pk > 0:
                ca, cb = com(a, f), com(b, f)
                if ca is not None and cb is not None:
                    ra, rb = size.get(a, 0.1) / 2, size.get(b, 0.1) / 2
                    w = ra / max(1e-9, ra + rb)
                    p = [ca[i] + (cb[i] - ca[i]) * w for i in range(3)]
                    out.append({"t": round(f / fps, 4), "kind": "hit", "pos": [round(x, 4) for x in p], "force": round(float(pk), 3),
                                "impulse": round(float(J), 5), "a": a, "b": b, "scale": round(float(ra + rb), 4)})
            last = f
    for seg in sim.get("segments", []):
        for e in seg["events"]:
            if e.get("type") == "impact" and e.get("other") in ("table", "floor") and (e.get("speed") or 0) > 0.8:
                c = com(e["subject"], int(e["t"] * fps))
                kv = sim["kinematics"].get(e["subject"], {}).get("v", [])
                fb = max(0, int(e["t"] * fps) - 3)
                vz = (kv[fb] or [0, 0, 0])[2] if fb < len(kv) else 0
                if c is not None and vz < -0.8:
                    out.append({"t": round(e["t"], 4), "kind": "ground", "pos": [c[0], c[1], max(0.0, c[2] - size.get(e["subject"], 0.1) / 2)],
                                "speed": round(float(e["speed"]), 3), "a": e["subject"], "scale": round(size.get(e["subject"], 0.1), 4)})
    zs = comp["env"].get("surface_z")
    if zs is not None:
        for sid in size:
            k = sim["kinematics"].get(sid)
            if not k:
                continue
            prev = None
            for f, c in enumerate(k["c"]):
                if c is None:
                    continue
                side = c[2] > zs
                if prev is not None and side != prev:
                    v = k["v"][f] or [0, 0, 0]
                    out.append({"t": round(f / fps, 4), "kind": "splash", "pos": [c[0], c[1], zs], "speed": round(abs(v[2]), 3), "a": sid,
                                "scale": round(size[sid], 4)})
                prev = side
    out.sort(key=lambda e: e["t"])
    return out[:400]


# ---------------------------------------------------------------------------
# follow-ups: "what if it were twice as heavy?", "on the Moon", "at 30 km/h"
# ---------------------------------------------------------------------------
PLANETS = {"moon": 1.62, "mars": 3.71, "jupiter": 24.79, "venus": 8.87, "mercury": 3.7, "earth": 9.81, "pluto": 0.62,
           "saturn": 10.44, "neptune": 11.15, "uranus": 8.69, "sun": 274.0}
_FACTOR = {"twice": 2.0, "double": 2.0, "doubled": 2.0, "triple": 3.0, "thrice": 3.0, "half": 0.5, "halved": 0.5, "quarter": 0.25,
           "ten times": 10.0, "10x": 10.0, "100 times": 100.0, "a hundred times": 100.0}


def is_whatif(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r"\b(what if|what happens if|instead|now (make|try|do)|again (with|but)|try (it )?(with|on|at|in)|make (it|them|the)|"
                          r"twice|double|half|triple|\d+\s*x\b|times (heavier|lighter|faster|slower|bigger|smaller|higher)|heavier|lighter|faster|slower|"
                          r"bigger|smaller|higher|lower|bouncier|on (the )?(moon|mars|jupiter|venus|mercury|pluto|saturn|sun)|in (space|water|the air))\b", t))


def apply_whatif(prev: dict, text: str) -> tuple[Experiment, list[str]]:
    """Change the previous experiment the way the follow-up says. Each clause ("one is 10 times heavier",
    "twice as fast", "on the Moon") is applied on its own. Returns (new experiment, changes made)."""
    exp = Experiment(**prev["experiment"])
    objs = prev["objects"]
    changes: list[str] = []
    movable = [o for o in objs if not o["support"] and not o["fixed"]]
    # copies ("baseball 1", "baseball 2") must become separate entries to change only one of them
    def split_copies(name: str):
        for i, o in enumerate(list(exp.objects)):
            if o.count > 1 and (name == o.name or name.startswith(o.name + " ")):
                exp.objects[i:i + 1] = [ExObject(name=f"{o.name} {j + 1}", what=o.what, size_m=o.size_m, mass_kg=o.mass_kg,
                                                  bounciness=o.bounciness) for j in range(o.count)]
                return

    def entry(name: str):
        split_copies(name)
        return next((o for o in exp.objects if o.name == name), None) or \
            next((o for o in exp.objects if name.startswith(o.name + " ") or o.name.startswith(name)), None)
    for clause in re.split(r"\s*(?:,|;|\band\b|\bbut\b|\bwhile\b)\s*", " " + text.lower() + " "):
        t = " " + clause + " "
        if not t.strip():
            continue
        k = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|times)\b", t)
        if m:
            k = float(m.group(1))
        for w, f in _FACTOR.items():
            if k is None and re.search(rf"\b{w}\b", t):
                k = f
        named = [o for o in movable if o["name"].split(" ")[0] in t or knowledge._singular(o["name"].rsplit(" ", 1)[0]) in t]
        if re.search(r"\b(one|the first|first one)\b", t):
            targets = movable[:1]
        elif re.search(r"\b(the other|the second|second one)\b", t):
            targets = movable[1:2]
        else:
            targets = named or movable
        tnames = [o["name"] for o in targets]
        nums = _numbers(t)
        mass_m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|kilograms?|g|grams?|tons?|tonnes?)\b", t)
        if mass_m:
            v = float(mass_m.group(1)) * {"g": 1e-3, "gram": 1e-3, "grams": 1e-3, "ton": 1000, "tons": 1000, "tonne": 1000, "tonnes": 1000}.get(mass_m.group(2), 1.0)
            for nm in tnames:
                e = entry(nm)
                if e:
                    e.mass_kg = v
                    changes.append(f"{e.name}: mass → {v:g} kg")
        elif re.search(r"\b(heav\w*|lighter|mass|weigh\w*)\b", t):
            f = k or 2.0
            if "lighter" in t:
                f = 1 / f if f > 1 else f
            for nm in tnames:
                base = next((x["mass"] for x in objs if x["name"] == nm), None)
                e = entry(nm)
                if e and base:
                    e.mass_kg = base * f
                    changes.append(f"{e.name}: mass ×{f:g} → {e.mass_kg:.3g} kg")
        if re.search(r"\b(bigger|smaller|larger|size|giant|tiny)\b", t):
            f = k or (0.5 if re.search(r"smaller|tiny", t) else 2.0)
            for nm in tnames:
                base = next((x for x in objs if x["name"] == nm), None)
                e = entry(nm)
                if e and base:
                    e.size_m = max(base["dims"]) * f
                    if e.mass_kg is None:
                        e.mass_kg = base["mass"] * f ** 3
                    changes.append(f"{e.name}: size ×{f:g}, mass ×{f ** 3:g} (same material)")
        if nums.get("speed") or re.search(r"\b(fast|faster|slow|slower|speed|quick\w*)\b", t):
            f = k or (0.5 if "slower" in t else 2.0)
            for a in exp.actions:
                if nums.get("speed"):
                    a.speed = nums["speed"]
                elif a.speed and a.speed > 0:
                    a.speed *= f
                else:
                    a.speed = -f * (-(a.speed) if a.speed and a.speed < 0 else 1.0)
            changes.append(f"speed → {nums['speed']:.3g} m/s" if nums.get("speed") else f"speed ×{f:g}")
        if nums.get("height") or re.search(r"\b(higher|lower|taller)\b", t):
            for a in exp.actions:
                if a.do in ("drop", "bounce"):
                    a.height = nums.get("height") or ((a.height or 1.0) * (k or (0.5 if "lower" in t else 2.0)))
                    changes.append(f"drop height → {a.height:g} m")
        if re.search(r"\b(bouncier|more bouncy|super bouncy|rubber)\b", t):
            for nm in tnames:
                e = entry(nm)
                if e:
                    e.bounciness = 0.9
                    changes.append(f"{e.name}: bounciness → 0.9")
        if re.search(r"\b(less bouncy|not bouncy|sticky|clay|dead)\b", t):
            for nm in tnames:
                e = entry(nm)
                if e:
                    e.bounciness = 0.05
                    changes.append(f"{e.name}: bounciness → 0.05")
        if nums.get("stiffness") or re.search(r"\b(stiffer|softer)\b", t):
            for a in exp.actions:
                if a.do == "spring":
                    a.stiffness = nums.get("stiffness") or None
            changes.append("spring stiffness changed")
        for pl, g in PLANETS.items():
            if re.search(rf"\b(on|at) (the )?{pl}\b", t):
                exp.gravity = g
                if exp.setting in ("table", "auto", "scene", "space"):
                    exp.setting = "ground"            # type: ignore[assignment]
                changes.append(f"gravity → {g} m/s² ({pl.title()})")
        for word, st in (("in space", "space"), ("underwater", "water"), ("in water", "water"), ("in the ocean", "water"),
                         ("in the air", "air"), ("on a table", "table"), ("on the table", "table"), ("on the ground", "ground")):
            if word in t:
                exp.setting = st                    # type: ignore[assignment]
                if st == "space":
                    exp.gravity = None
                changes.append(f"setting → {st}")
        if nums.get("angle"):
            for a in exp.actions:
                a.angle_deg = nums["angle"]
            changes.append(f"angle → {nums['angle']:g}°")
        if nums.get("distance"):
            for a in exp.actions:
                a.distance = nums["distance"]
            changes.append(f"distance/length → {nums['distance']:g} m")
    # actions refer to copies by name; if copies were split, names already match "x 1"/"x 2"
    exp.measure = list(dict.fromkeys(exp.measure + _measures_from_text(" " + text.lower() + " ")))
    exp.title = (exp.title.split(" — ")[0] + " — " + text.strip())[:200]
    return exp, changes
