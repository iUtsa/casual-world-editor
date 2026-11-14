"""Run a battery of very different experiment requests end to end (sandbox), print what came out."""
import sys, time, traceback
from backend.app import experiment

PROMPTS = sys.argv[1:] or [
    "two baseballs clash into each other",
    "two rubber ducks collide",
    "two blue whales fight underwater",
    "throw a tennis ball at a bottle",
    "drop an egg from 2 m",
    "a man pulls a truck with a rope",
    "two cars crash head on at 50 km/h",
    "a bowling ball hits a watermelon",
    "bounce a basketball",
    "two elephants fight",
    "a dog chases a cat",
    "drop a feather and a bowling ball from 3 meters",
    "connect two steel balls with a spring",
    "kick a soccer ball at a chair really hard",
    "a dragon crashes into a castle",
]
for p in PROMPTS:
    t = time.time()
    try:
        r = experiment.run(p)
        objs = ", ".join(f"{o['name']}[{o['source']}:{o['shape']},{o['mass']:.3g}kg]" for o in r["objects"])
        acts = ", ".join(f"{a['do']}({a['a']}{'→' + a['b'] if a['b'] else ''})" for a in r["experiment"]["actions"])
        key = [x for x in r["results"] if any(k in x["label"] for k in ("before impact", "Energy lost", "restitution", "Peak collision", "impact speed", "tension (average", "range", "Hits", "moved", "bounce height", "period"))][:5]
        print(f"\n▶ {p}  [{time.time()-t:.1f}s, {r['experiment']['understood_by']}, setting={r['setting']}]")
        print(f"   objects: {objs}\n   actions: {acts}")
        for x in key: print(f"   • {x['label']}: {x['text']}")
        print(f"   why: {r['story'][0][:220]}")
        if r["notes"]: print("   notes:", r["notes"])
    except Exception as e:
        print(f"\n✗ {p}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
