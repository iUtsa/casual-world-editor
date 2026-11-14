"""Every page, in the shared design: AI Lab (+ experiments), Live AR + diagram snap, Studio, Chemistry.

Usage: python backend/tests/e2e/ui_tour.py OUT_DIR DIAGRAM_DIR"""
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
diag = Path(sys.argv[2])
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

    # --- AI Lab
    pg.get_by_role("button", name="AI Lab").click()
    pg.wait_for_selector(".pred-cloud", timeout=120000)
    pg.wait_for_timeout(7500)                          # the signal rises through all 12 layers
    pg.screenshot(path=str(out / "ai_1_think.png"))
    print("AI story:", pg.locator(".why p").first.inner_text()[:160])
    pg.get_by_role("button", name="Layer by layer").click(); pg.wait_for_timeout(600)
    pg.screenshot(path=str(out / "ai_2_lens.png"))
    pg.get_by_role("button", name="Heads & experiments").click(); pg.wait_for_timeout(400)
    pg.locator(".tok").nth(2).click()                  # hide 'iff' (part of Eiffel)
    pg.wait_for_function("!document.querySelector('.live-busy')", timeout=120000); pg.wait_for_timeout(800)
    pg.screenshot(path=str(out / "ai_3_mask.png"))
    pg.get_by_role("button", name="What's happening").click(); pg.wait_for_timeout(300)
    print("AI experiment:", pg.locator(".why p").first.inner_text()[:200])
    pg.get_by_role("button", name="Next word").click(); pg.wait_for_timeout(400)
    pg.screenshot(path=str(out / "ai_4_next.png"))

    # --- Live AR + diagram
    pg.get_by_role("button", name="Live AR").click(); pg.wait_for_timeout(1500)
    pg.locator(".chat .diagram-btn input[type=file]").set_input_files(str(diag / "ramp.jpg"))
    pg.wait_for_selector(".reading textarea", timeout=240000)
    print("reading:", pg.locator(".reading textarea").last.input_value())
    pg.screenshot(path=str(out / "live_1_reading.png"))
    pg.get_by_role("button", name="▶ Simulate this").last.click()
    pg.wait_for_function("!document.querySelector('.typing')", timeout=240000); pg.wait_for_timeout(3000)
    pg.screenshot(path=str(out / "live_2_ramp.png"))
    print("ramp reply:", pg.locator(".msg.ai .bubble").last.inner_text().replace("\n", " ")[:200])
    pg.locator(".chat .diagram-btn input[type=file]").set_input_files(str(diag / "transformer.jpg"))
    pg.wait_for_selector(".reading textarea >> nth=1", timeout=240000)
    print("transformer reading:", pg.locator(".reading .small").last.inner_text()[:200])
    pg.get_by_role("button", name="▶ Simulate this").last.click(); pg.wait_for_timeout(1500)
    print("mode after transformer diagram:", pg.locator(".modes button.on").inner_text())

    # --- Studio
    pg.get_by_role("button", name="Studio").click(); pg.wait_for_timeout(2500)
    pg.screenshot(path=str(out / "studio.png"))
    # --- Chemistry
    pg.get_by_role("button", name="Chemistry").click(); pg.wait_for_timeout(1500)
    pg.get_by_role("button", name="hydrogen + oxygen -> water").click()
    pg.wait_for_timeout(25000)
    pg.screenshot(path=str(out / "chem.png"))
    print("page errors:", errs[:5])
    b.close()
