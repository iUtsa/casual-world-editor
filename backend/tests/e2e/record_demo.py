"""Record the landing-page demo video from the real app (Chromium, fake camera = the rendered 'bridge' test desk).
Landing -> Get started -> mapping -> inspect an object -> "drop a baseball on the red box" -> 3D twin -> 🖐 Touch.
Writes OUT_DIR/raw.webm and OUT_DIR/marks.json (seconds into the recording where each part starts).
Usage: python backend/tests/e2e/record_demo.py OUT_DIR   (restart the server afterwards)"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
img = cv2.resize(cv2.imread("data/synthetic/bridge/rgb.png"), (1280, 720))
fake = out / "cam.y4m"
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).tobytes()
fake.write_bytes(b"YUV4MPEG2 W1280 H720 F30:1 Ip A1:1 C420jpeg\n" + b"".join(b"FRAME\n" + yuv for _ in range(30)))
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={fake}"]
W, H = 1280, 720
BASE = "http://localhost:8000"

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
    ctx = b.new_context(viewport={"width": W, "height": H}, permissions=["camera"],
                        record_video_dir=str(out), record_video_size={"width": W, "height": H})
    pg = ctx.new_page()
    t0 = time.time()
    mark = lambda k: marks.__setitem__(k, round(time.time() - t0, 2))
    cap = lambda t: pg.evaluate(CAPTION, t)
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))

    # 1. landing
    pg.goto(BASE + "/", wait_until="networkidle"); mark("landing")
    pg.wait_for_timeout(3500)
    pg.mouse.move(900, 300); pg.wait_for_timeout(600)
    pg.locator(".hero .hero-actions a[data-start]").click()
    pg.wait_for_url("**/lab/"); mark("lab")

    # 2. mapping
    pg.wait_for_selector(".map-progress", timeout=60000); mark("map")
    cap("Point a camera at a desk: it maps every object (sped up · camera = rendered test desk)")
    pg.wait_for_selector(".map-progress", state="detached", timeout=300000); mark("mapped")
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
    video = pg.video.path()
    ctx.close(); b.close()
    Path(video).rename(out / "raw.webm")
(out / "marks.json").write_text(json.dumps(marks, indent=1))
print(json.dumps(marks), "errors:", errs)
