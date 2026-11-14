"""The theory check in the UI: Live AR chat badge + Studio badge with its measured-vs-theory table.

Usage: python backend/tests/e2e/ui_theory.py OUT_DIR"""
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
img = cv2.resize(cv2.imread("data/synthetic/bridge/rgb.png"), (1280, 720))
fake = out / "fake_cam.mjpeg"
fake.write_bytes(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes() * 60)

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={fake}"])
    pg = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera", "microphone"]).new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".msg.ai .bubble:not(.typing)", timeout=300000)
    for prompt in ("a pendulum 1 m long released at 60 degrees", "throw a ball at 20 m/s at 45 degrees", "a 9 V battery with two bulbs in parallel"):
        n = pg.locator(".theory-badge").count()
        pg.locator(".chat-input input").fill(prompt)
        pg.locator(".chat-input input").press("Enter")
        pg.wait_for_function(f"document.querySelectorAll('.theory-badge').length > {n}", timeout=240000)
        print(prompt, "->", pg.locator(".theory-badge").last.inner_text().replace("\n", " "))
    pg.locator(".theory-badge").last.click(); pg.wait_for_timeout(500)
    pg.screenshot(path=str(out / "live_theory.png"))
    pg.get_by_role("button", name="Studio", exact=True).click(); pg.wait_for_timeout(2500)
    pg.locator(".lab .theory-badge").first.click(); pg.wait_for_timeout(400)
    print("studio table rows:", pg.locator(".lab .theory-table tbody tr").count())
    pg.screenshot(path=str(out / "studio_theory.png"))
    print("page errors:", errs[:5])
    b.close()
