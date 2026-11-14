"""Live AR: what you say acts on the mapped world; big things stay in your scene; Studio runs don't leak into Live AR;
space counterfactuals simulate. Restart the server afterwards (it leaves a scene in the session).

Usage: python backend/tests/e2e/ui_say.py OUT_DIR [image]"""
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
src = sys.argv[2] if len(sys.argv) > 2 else "data/synthetic/bridge/rgb.png"
fake = out / "fake_cam.mjpeg"
fake.write_bytes(cv2.imencode(".jpg", cv2.resize(cv2.imread(src), (1280, 720)))[1].tobytes() * 60)
errs = []

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--use-fake-device-for-media-stream",
                                "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={fake}"])
    pg = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera"]).new_page()
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".map-progress", timeout=30000)
    pg.wait_for_selector(".map-progress", state="detached", timeout=300000)
    pg.wait_for_timeout(1000)
    labels = [o["label"] for o in pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]["objects"]]
    print("mapped:", labels)
    a, c = labels[0], labels[-1]

    def say(text, tag, wait=2500):
        n = pg.locator(".msg.ai .bubble").count()
        pg.locator(".chat-input input").fill(text)
        pg.locator(".chat-input input").press("Enter")
        pg.wait_for_function(f"document.querySelectorAll('.msg.ai .bubble').length > {n} && !document.querySelector('.typing')", timeout=300000)
        pg.wait_for_timeout(wait)
        pg.screenshot(path=str(out / f"{tag}.png"))
        msg = pg.locator(".msg.ai .bubble").last.inner_text().replace("\n", " ")
        extra = " | HOLOGRAM SHOWN" if pg.locator(".holo-panel, .ar-canvas").count() else ""
        print(f"[{text}] -> {msg[:230]}{extra}")
        return msg

    say(f"push the {a} to the left", "1_push")
    say(f"throw the {a} at the {c}", "2_throw")
    say(f"make the {c} twice as heavy", "3_mass")
    say("two blue whales fight underwater", "4_whales", 3500)
    say(f"a car crashes into the {c}", "5_car", 3500)
    n = pg.locator(".msg.ai .bubble").count()
    pg.get_by_role("button", name="🎾 Ball").click()
    pg.wait_for_function(f"document.querySelectorAll('.msg.ai .bubble').length > {n}", timeout=120000)
    pg.wait_for_timeout(1500)
    print("[🎾 Ball] ->", pg.locator(".msg.ai .bubble").last.inner_text()[:120])
    say(f"throw the baseball at the {c}", "6_ball_throw")

    # Studio: space counterfactuals
    pg.get_by_role("button", name="Studio").click()
    pg.wait_for_selector(".cmd-input", timeout=10000)
    for q in ["what if jupiter orbits earth?", "what if there is no sun"]:
        pg.locator(".cmd-input").fill(q); pg.locator(".cmd-input").press("Enter")
        pg.wait_for_timeout(1500)
        pg.wait_for_function("!document.querySelector('.busy, .spinner')", timeout=300000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(out / f"7_studio_{q.split()[2]}.png"))
        txt = pg.locator("body").inner_text()
        hits = [l for l in txt.splitlines() if any(k in l for k in ("Moon is", "centre of mass", "straight line", "removed", "Take away", "orbit"))][:3]
        print(f"[Studio: {q}] ->", [h[:150] for h in hits])
    pg.get_by_role("button", name="Live AR").click()
    pg.wait_for_timeout(2500)
    pg.screenshot(path=str(out / "8_back_to_live.png"))
    print("back in Live AR: hologram/panel leaking from Studio:", pg.locator(".holo-panel, .ar-canvas").count() > 0)
    b.close()
print("page errors:", errs)
