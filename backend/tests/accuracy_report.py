"""Accuracy report: every engine against the physical law it should obey.

    python backend/tests/accuracy_report.py            # table in the terminal
    python backend/tests/accuracy_report.py --md FILE  # also write a Markdown table
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.app import circuits, experiment, orbits  # noqa: E402

CASES = [
    # (engine, prompt, law being tested, (measured row, theory row) shown)
    ("mechanics", "drop a ball from 5 meters on the moon", "free fall v = √(2gh)", ("impact speed", "impact speed (theory")),
    ("mechanics", "drop a ball from 10 meters", "fall with air drag (Newton's 2nd law)", ("impact speed", "impact speed (theory")),
    ("mechanics", "drop a feather from 2 meters", "terminal speed (drag = weight)", ("impact speed", "impact speed (theory")),
    ("mechanics", "throw a ball at 10 m/s at 60 degrees on the moon", "projectile parabola", ("range (to landing)", "range (theory")),
    ("mechanics", "throw a ball at 20 m/s at 45 degrees", "projectile with air drag", ("range (to landing)", "range (theory")),
    ("mechanics", "a 2 kg block slides down a 30 degree ramp with friction 0.2", "a = g(sinθ − μcosθ)", ("acceleration down the ramp (measured)", "acceleration (theory")),
    ("mechanics", "a ball rolls down a 30 degree ramp", "rolling a = (5/7)g(sinθ − c_rr cosθ)", ("acceleration down the ramp (measured)", "acceleration (theory")),
    ("mechanics", "a pendulum 1 m long released at 60 degrees", "exact pendulum period (elliptic integral)", ("swing period (measured)", "swing period (theory")),
    ("mechanics", "a 1 kg mass on a spring with k 100", "T = 2π√(m/k)", ("oscillation period (measured)", "oscillation period (theory")),
    ("mechanics", "a 2 kg cart hits a 1 kg cart at 3 m/s", "momentum + Newton's law of restitution", ("cart 2 velocity after, along the impact (measured)", "cart 2 velocity after, along the impact (theory)")),
    ("mechanics", "two billiard balls collide", "momentum + restitution", ("billiard ball 2 velocity after, along the impact (measured)", "billiard ball 2 velocity after, along the impact (theory)")),
    ("mechanics", "a steel ball drops in water", "sinking: buoyancy + drag + added mass", ("sinking speed at its fastest (measured)", "sinking speed at its fastest (theory")),
    ("mechanics", "a wooden block floats in water", "Archimedes: floats at ρ/ρ_water", ("under water (measured)", "theory: weight = weight of the water")),
    ("mechanics", "a man jumps on the moon", "h = v²/2g", ("jump height", "jump height (theory")),
    ("circuits", "a 6 V battery with a 10 ohm resistor in series with a 20 ohm and a 30 ohm resistor in parallel", "Ohm's + Kirchhoff's laws", ("current from the battery", "current (theory")),
    ("circuits", "a 9 V battery, a 1000 ohm resistor and a 100 microfarad capacitor", "RC charging V(1 − e^(−t/RC))", ("capacitor voltage after one τ (measured)", "capacitor voltage after one τ (theory")),
    ("gravity", "the moon orbits the earth", "Kepler's 3rd law", ("orbital period (measured)", "orbital period (kepler")),
    ("gravity", "the iss orbits the earth", "Kepler's 3rd law", ("orbital period (measured)", "orbital period (kepler")),
    ("gravity", "jupiter collides with the moon", "energy conservation", ("impact speed", "impact speed (theory")),
]


def find(rows, label):
    for r in rows:
        if r["label"].lower().find(label.lower()) >= 0:
            return r
    return None


def main():
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    out = []
    t0 = time.time()
    for eng, prompt, law, (mrow, trow) in CASES:
        run = {"mechanics": experiment.run, "circuits": circuits.run, "gravity": orbits.run}[eng]
        res = run(prompt)["results"]
        ag = find(res, "Agreement with theory")
        m = next((r for r in res if r["label"].lower().startswith(mrow.lower()) or mrow.lower() in r["label"].lower()
                  and "theory" not in r["label"].lower()), None)
        t = find(res, trow)
        out.append((eng, prompt, law, m["text"] if m else "—", t["text"] if t else "—", ag["value"] if ag else float("nan")))
        print(f"{out[-1][5]:7.2f}%  {eng:9s} {law:42s} measured {out[-1][3]:>12s}  theory {out[-1][4]:>12s}   «{prompt}»", flush=True)
    worst = min(x[5] for x in out)
    print(f"\n{len(out)} cases in {time.time() - t0:.0f} s — worst agreement {worst:.2f}%, mean {sum(x[5] for x in out) / len(out):.2f}%")
    if md:
        lines = ["| Engine | Law tested | Prompt | Measured | Theory | Agreement |", "|---|---|---|---|---|---|"]
        lines += [f"| {e} | {l} | {p} | {m} | {t} | {a:.2f}% |" for e, p, l, m, t, a in out]
        Path(md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
