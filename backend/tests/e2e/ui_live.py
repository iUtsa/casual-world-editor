"""Live AR end to end with a fake webcam (a tabletop image streamed as the camera).

Usage: python backend/tests/e2e/ui_live.py OUT_DIR [image]"""
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
src = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/bridge/rgb.png"
img = cv2.imread(src)
img = cv2.resize(img, (1280, int(1280 * img.shape[0] / img.shape[1])) if img.shape[1] != 1280 else (img.shape[1], img.shape[0]))
ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
fake = out / "fake_cam.mjpeg"
fake.write_bytes(jpg.tobytes() * 60)          # MJPEG = concatenated JPEG frames

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                f"--use-file-for-fake-video-capture={fake}"])
    ctx = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera", "microphone"])
    pg = ctx.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".live-hud", timeout=20000)
    pg.wait_for_selector(".msg.ai .bubble:not(.typing)", timeout=300000)         # the assistant greets with what it can see once mapping is done
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out / "0_mapped.png"))
    print("hud:", pg.locator(".live-hud").inner_text().replace("\n", " | "))
    print("first message:", pg.locator(".msg.ai .bubble:not(.typing)").last.inner_text()[:200])

    def say(text, tag, wait=3500):
        pg.locator(".chat-input input").fill(text)
        pg.locator(".chat-input input").press("Enter")
        pg.wait_for_function("!document.querySelector('.typing')", timeout=240000)
        pg.wait_for_timeout(wait)
        pg.screenshot(path=str(out / f"{tag}.png"))
        print(f"[{text}] ->", pg.locator(".msg.ai .bubble").last.inner_text().replace("\n", " ")[:260])

    say("drop a baseball on the table", "1_scene", 2500)
    say("two blue whales fight underwater", "2_whales", 5000)
    say("how hard did they hit?", "3_question", 800)
    say("two baseballs collide", "4_balls", 3000)
    say("what if it was twice as heavy?", "5_whatif", 3000)
    say("a 9 V battery with two bulbs in parallel", "6_circuit", 2500)
    say("a dragon crashes into a castle", "7_dragon", 4000)
    print("page errors:", errs[:5])
    b.close()
