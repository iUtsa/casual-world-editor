"""Live AR hands end to end: fake webcam (a tabletop image) + a scripted hand fed into the real gesture pipeline
(MediaPipe still loads and runs on every frame; window.__fakeHand replaces only its landmarks).

  1. pinch the left-most object, flick left, open      -> "Threw the ..." (place + launch, simulated)
  2. pinch another object, move slowly, open           -> "Set the ... down"
  3. sweep a fingertip into the right-most object      -> "Pushed the ..."

Usage: python backend/tests/e2e/ui_hands.py OUT_DIR [image]"""
import json
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
src = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/bridge/rgb.png"
img = cv2.imread(src)
img = cv2.resize(img, (1280, int(1280 * img.shape[0] / img.shape[1])))
ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
fake = out / "fake_cam.mjpeg"
fake.write_bytes(jpg.tobytes() * 60)


def hand(mid, pinched, W, H):
    """21 MediaPipe-style landmarks of an upright right hand whose thumb/index meet at `mid` (pixels)."""
    mx, my = mid
    p4 = (mx - 6, my + 4) if pinched else (mx - 45, my + 30)
    p8 = (mx + 6, my - 4) if pinched else (mx + 10, my - 45)
    wrist, p9 = (mx, my + 140), (mx, my + 60)
    lerp = lambda a, b, t: (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    p5, p13, p17 = (mx - 22, my + 62), (mx + 20, my + 64), (mx + 38, my + 72)
    pts = [wrist, lerp(wrist, p4, 0.35), lerp(wrist, p4, 0.6), lerp(wrist, p4, 0.8), p4,
           p5, lerp(p5, p8, 0.4), lerp(p5, p8, 0.7), p8,
           p9, (mx + 2, my + 20), (mx + 3, my - 5), (mx + 4, my - 25),
           p13, (mx + 24, my + 28), (mx + 26, my + 6), (mx + 27, my - 10),
           p17, (mx + 44, my + 45), (mx + 47, my + 30), (mx + 49, my + 18)]
    return [{"x": x / W, "y": y / H} for x, y in pts]


PLAY = """async ([frames, dt]) => { for (const f of frames) { window.__fakeHand = f; await new Promise(r => setTimeout(r, dt)); } }"""

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                f"--use-file-for-fake-video-capture={fake}"])
    ctx = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera", "microphone"])
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".msg.ai .bubble:not(.typing)", timeout=300000)
    pg.wait_for_timeout(1000)
    pg.get_by_role("button", name="✋ Hands").click()
    pg.wait_for_function("document.querySelector('.hand-pill') && !document.querySelector('.hand-pill').innerText.includes('loading')", timeout=60000)
    print("pill after load:", pg.locator(".hand-pill").inner_text())
    sc = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    W, H = sc["camera"]["width"], sc["camera"]["height"]
    objs = []
    for o in sc["objects"]:
        poly = o["polygon"]
        cx = sum(q[0] for q in poly) / len(poly); cy = sum(q[1] for q in poly) / len(poly)
        objs.append((cx, cy, o["label"], o["id"]))
    objs.sort()
    print("objects (left -> right):", [(round(c[0]), round(c[1]), c[2]) for c in objs])

    def n_msgs():
        return pg.locator(".msg.ai .bubble").count()

    def play(frames, tag, expect):
        before = n_msgs()
        pg.evaluate(PLAY, [frames, 33])
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(out / f"{tag}_gesture.png"))
        pg.wait_for_function(f"document.querySelectorAll('.msg.ai .bubble').length > {before}", timeout=120000)
        pg.wait_for_function("!document.querySelector('.live-busy')", timeout=120000)
        pg.wait_for_timeout(1800)
        pg.screenshot(path=str(out / f"{tag}_result.png"))
        msg = pg.locator(".msg.ai .bubble").last.inner_text().replace("\n", " ")
        print(f"[{tag}] {msg[:240]}")
        assert expect in msg, f"expected '{expect}' in: {msg}"

    # 1. throw: hover, pinch, flick left fast, open while moving
    x, y, lab, _ = objs[0]
    fr = [hand((x, y), False, W, H)] * 8 + [hand((x, y), True, W, H)] * 8
    fr += [hand((x - 40 * k, y + 6 * k), True, W, H) for k in range(1, 6)]
    fr += [hand((x - 40 * (5 + k), y + 6 * (5 + k)), False, W, H) for k in range(1, 4)] + [None] * 3
    play(fr, "1_throw", "Threw the")

    # 2. set down: another object, slow move, open
    pg.wait_for_timeout(1500)
    x, y, lab, _ = objs[len(objs) // 2]
    fr = [hand((x, y), False, W, H)] * 6 + [hand((x, y), True, W, H)] * 8
    fr += [hand((x + 2 * k, y + 1 * k), True, W, H) for k in range(1, 16)]
    fr += [hand((x + 30, y + 15), True, W, H)] * 8 + [hand((x + 30, y + 15), False, W, H)] * 5 + [None] * 2
    play(fr, "2_setdown", "Set the")

    # 3. push: fingertip sweeps right -> left into the right-most object (index tip = mid + (10, -45))
    pg.wait_for_timeout(1500)
    x, y, lab, _ = objs[-1]
    fr = [hand((x + 150 - 18 * k - 10, y + 45), False, W, H) for k in range(0, 14)] + [None] * 3
    play(fr, "3_push", "Pushed the")

    print("page errors:", errs[:5])
    b.close()
