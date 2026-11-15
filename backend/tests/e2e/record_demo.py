"""Record the landing-page demo video from the real app (Chromium; the camera is the rendered "bridge" test desk).
Landing -> Get started -> mapping -> inspect an object -> "drop a baseball on the red box" -> 3D twin -> 🖐 Touch.
Writes OUT_DIR/frames/ + frames.json (see screencast.py) and OUT_DIR/marks.json (seconds into the recording where each part starts).
Usage: python backend/tests/e2e/record_demo.py OUT_DIR   (restart the server afterwards)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).parent))
from screencast import Screencast  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
W, H = 1280, 720
BASE = "http://localhost:8000"
# The camera: the rendered 'bridge' test desk as a steady 30 fps stream. (Chrome's file-based fake camera stops
# delivering frames while the Mac is busy mapping, which blanks the picture; a canvas stream doesn't.)
FAKE_CAMERA = """(() => {
  const real = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    if (!c || !c.video) return real(c);
    const img = new Image(); img.src = '/api/synthetic/bridge'; await img.decode();
    const cv = document.createElement('canvas'); cv.width = 1280; cv.height = 720;
    // kept in the page (invisible) so Chrome keeps painting it while the screen is being captured
    cv.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0.01;pointer-events:none;z-index:-1';
    document.body.appendChild(cv);
    const ctx = cv.getContext('2d');
    let k = 0;
    const draw = () => { ctx.drawImage(img, 0, 0, 1280, 720); ctx.fillStyle = `rgba(0,0,0,${(k++ % 2) * 0.002})`; ctx.fillRect(0, 0, 1, 1); };
    draw(); setInterval(draw, 33);
    const stream = cv.captureStream(30);
    await new Promise((r) => setTimeout(r, 800));   // real frames flowing before anyone grabs one
    return stream;
  };
})()"""

CAPTION = """(t) => {
  let el = document.getElementById('__cap');
  if (!el) {
    el = document.createElement('div'); el.id = '__cap';
    el.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:99999;max-width:80%;'
      + 'background:#141414;color:#f6f4ee;font:600 17px/1.3 "Instrument Sans",system-ui,sans-serif;padding:10px 18px;'
      + 'border-radius:10px;box-shadow:4px 4px 0 #ff5a1f;transition:opacity .3s;pointer-events:none;text-align:center';
    document.body.appendChild(el);
  }
  el.textContent = t || ''; el.style.opacity = t ? 1 : 0;
}"""


def flat_hand(cx, cy, cz):
    J = np.zeros((21, 3)); J[0] = [cx, cy - 0.06, cz]
    for base, dx in {1: -0.045, 5: -0.025, 9: -0.005, 13: 0.015, 17: 0.035}.items():
        for k in range(4):
            J[base + k] = [cx + dx, cy - 0.02 + 0.025 * k + (0 if base != 1 else -0.02), cz]
    return J


def to_landmarks(Jw, cam):
    R, t = np.array(cam["R_wc"]), np.array(cam["t_wc"])
    Jc = (Jw - t) @ R
    u = cam["fx"] * Jc[:, 0] / Jc[:, 2] + cam["cx"]; v = cam["fy"] * Jc[:, 1] / Jc[:, 2] + cam["cy"]
    c = Jc.mean(0)
    return {"px": [{"x": float(a / cam["width"]), "y": float(b / cam["height"])} for a, b in zip(u, v)],
            "world": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in Jc - c]}


marks = {}
with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    ctx = b.new_context(viewport={"width": W, "height": H}, permissions=["camera"])
    ctx.add_init_script(FAKE_CAMERA)
    pg = ctx.new_page()
    rec = Screencast(ctx, pg, out, (W, H))
    mark = lambda k: marks.__setitem__(k, time.time())
    cap = lambda t: pg.evaluate(CAPTION, t)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))

    # 1. landing
    pg.goto(BASE + "/"); pg.wait_for_selector(".hero h1"); mark("landing")
    pg.wait_for_timeout(3500)
    pg.mouse.move(900, 300); pg.wait_for_timeout(600)
    pg.locator(".hero .hero-actions a[data-start]").click()
    pg.wait_for_url("**/lab/"); mark("lab")

    # 2. mapping
    pg.wait_for_selector(".map-progress", timeout=180000); mark("map")
    cap("Point a camera at a desk: it maps every object (sped up · camera = rendered test desk)")
    pg.wait_for_selector(".map-progress", state="detached", timeout=900000); mark("mapped")
    cap("Size, mass, friction and what rests on what, for every object")
    pg.wait_for_timeout(3000)
    sc = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    cam = sc["camera"]

    # 3. inspect an object (click its centre on the camera image)
    o = max(sc["objects"], key=lambda x: x["size"][0] * x["size"][1])
    stage = pg.locator(".live-stage").bounding_box()
    Rw, tw = np.array(cam["R_wc"]), np.array(cam["t_wc"])
    Pc = (np.array(o["pos"]) - tw) @ Rw
    u = cam["fx"] * Pc[0] / Pc[2] + cam["cx"]; v = cam["fy"] * Pc[1] / Pc[2] + cam["cy"]
    sx = stage["x"] + u / cam["width"] * stage["width"]; sy = stage["y"] + v / cam["height"] * stage["height"]
    pg.mouse.move(sx, sy, steps=12); pg.wait_for_timeout(400)
    pg.mouse.click(sx, sy); mark("inspect")
    cap(f"Click anything to inspect it: the {o.get('label', 'object')}")
    pg.wait_for_timeout(3500)
    pg.locator(".obj-card .x").first.click() if pg.locator(".obj-card .x").count() else None

    # 4. ask what if
    cap("Ask “what if?” in plain words")
    box = pg.locator(".chat-input input")
    box.click(); box.type("drop a baseball on the red box", delay=45); mark("ask")
    pg.keyboard.press("Enter")
    n0 = pg.locator(".msg.ai").count()
    pg.wait_for_function(f"document.querySelectorAll('.msg.ai .msg-tiles').length > 0 || document.querySelectorAll('.msg.ai').length > {n0} + 0 && !document.querySelector('.typing')", timeout=180000)
    pg.wait_for_timeout(500); mark("answer")
    cap("Real physics (MuJoCo) plays on the scene, measured and checked against textbook theory")
    pg.wait_for_timeout(9000)

    # 5. the 3D twin
    pg.get_by_role("button", name="🧊 3D").click(); mark("twin")
    cap("Orbit the 3D digital twin")
    tw_box = pg.locator(".twin3d").bounding_box()
    if tw_box:
        cx, cy = tw_box["x"] + tw_box["width"] / 2, tw_box["y"] + tw_box["height"] / 2
        pg.mouse.move(cx, cy); pg.mouse.down()
        for k in range(40):
            pg.mouse.move(cx + 6 * k, cy - 1.2 * k); pg.wait_for_timeout(40)
        pg.mouse.up()
    pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="📷 Camera").click(); pg.wait_for_timeout(800)

    # 6. 🖐 Touch: a scripted 3D hand (the same landmarks MediaPipe would report) sweeps into the left-most object
    sc = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    tgt = min(sc["objects"], key=lambda x: x["pos"][0])
    pg.get_by_role("button", name="🖐 Touch").click(); mark("touch")
    cap("🖐 Touch: your hand becomes physics: fingers push things by contact (scripted hand here)")
    pg.evaluate("window.__fakeTouch = null"); pg.wait_for_timeout(1200)
    x0, y0 = tgt["pos"][0] - max(tgt["size"][:2]) / 2 - 0.08, tgt["pos"][1]
    for k in range(70):
        J = flat_hand(x0 + 0.35 * min(k / 35, 0.9), y0, 0.03)
        pg.evaluate("h => { window.__fakeTouch = h }", to_landmarks(J, cam))
        pg.wait_for_timeout(33)
    pg.evaluate("window.__fakeTouch = null")
    cap("When the hand leaves, the session is replayed exactly and reported")
    try:
        pg.wait_for_function("[...document.querySelectorAll('.msg.ai .bubble')].some(e => e.textContent.includes('🖐'))", timeout=90000)
    except Exception as e:
        print("no touch report:", e)
    mark("touch_done")
    pg.wait_for_timeout(4000); cap(""); pg.wait_for_timeout(800); mark("end")
    t0 = rec.stop()
    ctx.close(); b.close()
marks = {k: round(v - t0, 2) for k, v in marks.items()}
(out / "marks.json").write_text(json.dumps(marks, indent=1))
print(json.dumps(marks), "errors:", errs)
