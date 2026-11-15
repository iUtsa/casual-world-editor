"""🖐 Touch end to end in Chromium: scan (fake camera) -> Touch -> a scripted 3D hand sweeps through the left-most object
-> live touch markers -> hand leaves -> the session is replayed and reported. Restart the server afterwards.
Usage: python backend/tests/e2e/ui_touch.py OUT_DIR"""
import sys
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
img = cv2.resize(cv2.imread("data/synthetic/bridge/rgb.png"), (1280, 720))
# Y4M (raw I420 frames): Chrome's most reliable fake-camera format
fake = out / "cam.y4m"
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).tobytes()
fake.write_bytes(b"YUV4MPEG2 W1280 H720 F30:1 Ip A1:1 C420jpeg\n" + b"".join(b"FRAME\n" + yuv for _ in range(30)))
ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-video-capture={fake}"]


def flat_hand(cx, cy, cz):
    J = np.zeros((21, 3)); J[0] = [cx, cy - 0.06, cz]
    for base, dx in {1: -0.045, 5: -0.025, 9: -0.005, 13: 0.015, 17: 0.035}.items():
        for k in range(4):
            J[base + k] = [cx + dx, cy - 0.02 + 0.025 * k + (0 if base != 1 else -0.02), cz]
    return J


def to_landmarks(Jw, cam):
    """World joints -> what MediaPipe would report: normalized pixels + hand-centred metric 'world' landmarks."""
    R, t = np.array(cam["R_wc"]), np.array(cam["t_wc"])
    Jc = (Jw - t) @ R                                         # camera frame (OpenCV)
    u = cam["fx"] * Jc[:, 0] / Jc[:, 2] + cam["cx"]; v = cam["fy"] * Jc[:, 1] / Jc[:, 2] + cam["cy"]
    c = Jc.mean(0)
    return {"px": [{"x": float(a / cam["width"]), "y": float(b / cam["height"])} for a, b in zip(u, v)],
            "world": [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in Jc - c]}


with sync_playwright() as p:
    b = p.chromium.launch(args=ARGS)
    pg = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera"]).new_page()
    errs = []; pg.on("pageerror", lambda e: errs.append(str(e))); pg.on("console", lambda m: m.type == "error" and errs.append(m.text[:300]))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".map-progress", timeout=60000); pg.wait_for_selector(".map-progress", state="detached", timeout=300000)
    print("videos after mapping:", pg.evaluate("[...document.querySelectorAll('.live-stage video')].map(v => ({rs: v.readyState, w: v.videoWidth, paused: v.paused, src: !!v.srcObject}))"))
    sc = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    cam = sc["camera"]; o = min(sc["objects"], key=lambda x: x["pos"][0])
    pg.get_by_role("button", name="🖐 Touch").click()
    pg.evaluate("window.__fakeTouch = null")
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out / "0_after_click.png"))
    print("touch button:", pg.get_by_role("button", name="🖐 Touch").first.get_attribute("class"), "| errors so far:", errs)
    x0, y0 = o["pos"][0] - max(o["size"][:2]) / 2 - 0.08, o["pos"][1]
    saw_touch = False
    for k in range(60):                                      # ~2 s of sweeping, one pose per animation frame batch
        J = flat_hand(x0 + 0.35 * min(k / 30, 0.9), y0, 0.015)
        pg.evaluate("h => { window.__fakeTouch = h }", to_landmarks(J, cam))
        pg.wait_for_timeout(33)
        if not saw_touch and pg.locator(".touch-mark").count():
            saw_touch = True; pg.screenshot(path=str(out / "1_touching.png"))
    print("debug:", pg.evaluate("window.__touchDbg"), "| target", o["id"], [round(v, 3) for v in o["pos"]], "start x0", round(x0, 3), "y0", round(y0, 3))
    print("errors after sweep:", errs[:3])
    pg.screenshot(path=str(out / "1b_after_sweep.png"))
    print("videos:", pg.evaluate("[...document.querySelectorAll('.live-stage video')].map(v => ({rs: v.readyState, w: v.videoWidth, cls: v.className, paused: v.paused, src: !!v.srcObject, track: v.srcObject ? v.srcObject.getVideoTracks().map(t => t.readyState).join() : null}))"))
    saw_touch = saw_touch or bool(pg.evaluate("window.__touchDbg.touchMax"))
    print("live touch markers seen:", saw_touch, "| pill:", pg.locator(".hand-pill").all_inner_texts())
    pg.evaluate("window.__fakeTouch = null")
    pg.wait_for_function("[...document.querySelectorAll('.msg.ai .bubble')].some(e => e.textContent.includes('🖐'))", timeout=90000)
    pg.wait_for_timeout(1500); pg.screenshot(path=str(out / "2_replayed.png"))
    txt = pg.evaluate("[...document.querySelectorAll('.msg.ai .bubble')].map(e => e.textContent).filter(t => t.includes('🖐')).pop()")
    print("chat:", txt[:200])
    st = pg.evaluate("fetch('/api/scene').then(r => r.json())")
    print("last intervention:", st["interventions"][-1]["type"] if st["interventions"] else None)
    print("page errors:", errs)
    b.close()
