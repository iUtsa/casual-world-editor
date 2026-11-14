"""Phone flow over the real LAN HTTPS endpoint (emulated phone, fake camera) + desktop view.

Usage: python backend/tests/e2e/ui_phone.py LAN_IP PHOTO OUT_DIR"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

ip, photo, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                                "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    phone = b.new_context(**p.devices["Pixel 7"], ignore_https_errors=True, permissions=["camera"])
    ph = phone.new_page()
    errs = []
    ph.on("pageerror", lambda e: errs.append("phone: " + str(e)))
    ph.goto(f"https://{ip}:8443/phone")
    ph.wait_for_timeout(2500)
    print("phone secure context:", ph.evaluate("window.isSecureContext"), "| camera started:", ph.locator("#scan").is_enabled())
    ph.locator("#live").click()
    ph.wait_for_timeout(3000)
    print("phone status:", ph.locator("#stat").inner_text())
    ph.screenshot(path=str(out / "phone_live.png"))
    desk = b.new_context(viewport={"width": 1500, "height": 950}).new_page()
    desk.on("pageerror", lambda e: errs.append("desk: " + str(e)))
    desk.goto("http://localhost:8000/")
    desk.wait_for_timeout(1500)
    desk.screenshot(path=str(out / "desk_start.png"))
    desk.get_by_role("button", name="📱 Connect phone").click()
    desk.wait_for_timeout(2500)
    print("desktop sees:", desk.locator(".modal .pill").inner_text())
    desk.screenshot(path=str(out / "desk_qr.png"))
    # phone takes a photo of the real table and scans it
    ph.locator("#live").click()
    ph.locator("#photo").set_input_files(photo)
    ph.wait_for_selector("#done", state="visible", timeout=120000)
    print("phone result:", ph.locator("#doneText").inner_text())
    ph.screenshot(path=str(out / "phone_done.png"))
    desk.locator(".modal-bg").click(position={"x": 5, "y": 5})     # close the dialog
    desk.wait_for_selector(".drawer", timeout=15000)                 # desktop picks up the phone's scan by itself
    desk.wait_for_timeout(2500)
    print("desktop objects:", desk.locator(".tabs").inner_text().replace("\n", " "), "|", desk.locator(".panel h3").first.inner_text())
    desk.screenshot(path=str(out / "desk_scene.png"))
    print("errors:", errs[:4])
    b.close()
