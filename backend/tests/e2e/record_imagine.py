"""Record the 'imagination' demo video from the real app: Studio scenes that exist only as a sentence.
Two whales collide -> what if Jupiter orbited Earth? -> methane + oxygen -> GPT-2 thinking, layer by layer.
Writes OUT_DIR/frames/ + frames.json (see screencast.py) and OUT_DIR/marks.json (seconds into the recording where each part starts).
Usage: python backend/tests/e2e/record_imagine.py OUT_DIR   (restart the server afterwards)"""
import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from screencast import Screencast  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
W, H = 1280, 720
BASE = "http://localhost:8000"
NO_CAMERA = "navigator.mediaDevices.getUserMedia = () => Promise.reject(new Error('no camera in this demo'))"

CAPTION = """(t) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div'); el.id = '__cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:99999;max-width:80%;'
      + 'background:#141414;color:#f6f4ee;font:600 17px/1.3 "Instrument Sans",system-ui,sans-serif;padding:10px 18px;'
      + 'border-radius:10px;box-shadow:4px 4px 0 #2f5bff;transition:opacity .3s;pointer-events:none;text-align:center';
    document.body.appendChild(el);
  }
  el.textContent = t || ''; el.style.opacity = t ? 1 : 0;
}"""

marks = {}
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    ctx = b.new_context(viewport={"width": W, "height": H})
    ctx.add_init_script(NO_CAMERA)
    pg = ctx.new_page()
    rec = Screencast(ctx, pg, out, (W, H))
    mark = lambda k: marks.__setitem__(k, time.time())
    cap = lambda t: pg.evaluate(CAPTION, t)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))

    pg.goto(BASE + "/lab/"); pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="Studio", exact=True).click(); pg.wait_for_timeout(1200); mark("studio")

    def ask(text, done_js, timeout=240000):
        box = pg.locator("input.cmd-input").first
        box.click(); box.fill(""); box.type(text, delay=40); pg.wait_for_timeout(300)
        box.press("Enter")
        pg.wait_for_function(done_js, timeout=timeout)

    # 1. physics at a scale no lab has
    cap("Studio: describe a world that doesn't exist. It gets built and simulated.")
    ask("two blue whales collide at 10 m/s",
        "!document.querySelector('.busy-veil') && document.querySelectorAll('.tile').length > 0"); mark("whales")
    cap("Two blue whales, real masses, real physics (MuJoCo), every number measured")
    pg.wait_for_timeout(9000)

    # 2. space
    cap("Ask “what if?” about the solar system")
    ask("what if Jupiter orbited Earth?", "!!document.querySelector('.space-hud')"); mark("space")
    cap("Real planet positions from NASA JPL, integrated with REBOUND, checked against Kepler")
    pg.wait_for_timeout(10000)

    # 3. chemistry
    pg.get_by_role("button", name="🧪 Chemistry").click(); pg.wait_for_timeout(1000); mark("chem")
    cap("Chemistry: name a reaction")
    box = pg.locator(".cmd-input").first
    box.click(); box.fill(""); box.type("methane + oxygen → carbon dioxide + water", delay=30)
    pg.get_by_role("button", name="Simulate").click()
    pg.wait_for_selector(".chem-eq", timeout=120000)
    cap("Balanced, measured energies, molecules moving (GFN2-xTB)")
    try:
        pg.wait_for_selector(".events li", timeout=180000)
    except Exception as e:
        print("no chem events:", e)
    pg.wait_for_timeout(8000)

    # 4. AI Lab
    pg.get_by_role("button", name="🧠 AI Lab").click(); pg.wait_for_timeout(1000); mark("ai")
    cap("Even an imagined mind: a real GPT-2, running on this laptop")
    box = pg.locator(".cmd-input").first
    box.click(); box.fill(""); box.type("the cat sat on the", delay=60)
    pg.get_by_role("button", name="Run GPT-2").click()
    pg.wait_for_function("(document.querySelector('.ai-lab')?.textContent || '').includes(\"' cat'\")", timeout=120000)
    pg.wait_for_timeout(600)
    pg.get_by_role("button", name="▶ Watch it think").click()
    cap("Watch the next word form, layer by layer")
    pg.wait_for_timeout(9000)
    cap("What you imagine, you can simulate."); pg.wait_for_timeout(3000)
    cap(""); pg.wait_for_timeout(600); mark("end")

    t0 = rec.stop()
    ctx.close(); b.close()
marks = {k: round(v - t0, 2) for k, v in marks.items()}
(out / "marks.json").write_text(json.dumps(marks, indent=1))
print(json.dumps(marks), "errors:", errs)
