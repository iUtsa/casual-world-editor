"""The Live AR flow end to end, as a user does it. Restart the server afterwards (it leaves a scene in the session).

A. laptop camera: map (progress card) -> outlines + hint -> click an object (card) -> drag it (push, measured) -> 🧊 3D -> pick in 3D
B. phone: laptop has no camera (setup card) -> phone page opens and streams by itself -> laptop switches to the phone and maps

Usage: python backend/tests/e2e/ui_flow.py OUT_DIR LAN_IP [image]"""
import sys
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

out, ip = Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True, exist_ok=True)
src = sys.argv[3] if len(sys.argv) > 3 else "data/synthetic/bridge/rgb.png"
img = cv2.resize(cv2.imread(src), (1280, 720))
fake = out / "fake_cam.mjpeg"
fake.write_bytes(cv2.imencode(".jpg", img)[1].tobytes() * 60)
GL = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
CAM = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", f"--use-file-for-fake-video-capture={fake}"]
errs = []


def shot(pg, name):
    pg.screenshot(path=str(out / f"{name}.png"))


def centroid(poly):
    return sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly)


def to_screen(pg, x, y, W, H):
    r = pg.locator(".live-stage").bounding_box()
    return r["x"] + x / W * r["width"], r["y"] + y / H * r["height"]


with sync_playwright() as p:
    # ---------------- A. laptop camera
    b = p.chromium.launch(args=GL + CAM)
    pg = b.new_context(viewport={"width": 1600, "height": 960}, permissions=["camera"]).new_page()
    pg.on("pageerror", lambda e: errs.append("A: " + str(e)))
    pg.goto("http://localhost:8000/")
    pg.wait_for_selector(".map-progress", timeout=30000)
    pg.wait_for_timeout(4000)
    shot(pg, "A1_mapping")
    print("A1 mapping card:", pg.locator(".map-progress").inner_text().replace("\n", " | ")[:160])
    pg.wait_for_selector(".map-progress", state="detached", timeout=300000)
    pg.wait_for_timeout(1500)
    shot(pg, "A2_mapped")
    print("A2 top:", pg.locator(".live-top").inner_text().replace("\n", " "), "| hint:", pg.locator(".hint-pill").all_inner_texts())
    sc = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    W, H = sc["camera"]["width"], sc["camera"]["height"]
    objs = sorted(sc["objects"], key=lambda o: centroid(o["polygon"])[0])
    o = objs[0]
    cx, cy = centroid(o["polygon"])
    sx, sy = to_screen(pg, cx, cy, W, H)
    pg.mouse.click(sx, sy)
    pg.wait_for_selector(".obj-card", timeout=5000)
    shot(pg, "A3_card")
    print("A3 card:", pg.locator(".obj-card").inner_text().replace("\n", " | ")[:220])
    n0 = pg.locator(".msg.ai .bubble").count()
    pg.mouse.move(sx, sy); pg.mouse.down(); pg.mouse.move(sx - 60, sy + 10, steps=6); pg.mouse.move(sx - 120, sy + 20, steps=6); pg.mouse.up()
    pg.wait_for_function(f"document.querySelectorAll('.msg.ai .bubble').length > {n0}", timeout=120000)
    pg.wait_for_function("!document.querySelector('.live-busy')", timeout=120000)
    pg.wait_for_timeout(2500)
    shot(pg, "A4_pushed")
    print("A4 chat:", pg.locator(".msg.ai .bubble").last.inner_text().replace("\n", " ")[:200])
    pg.mouse.click(sx - 80, sy + 15) if False else None
    pg.locator(".obj-card").wait_for(timeout=5000) if pg.locator(".obj-card").count() else None
    print("A4 card after push:", pg.locator(".obj-card").inner_text().replace("\n", " | ")[:220] if pg.locator(".obj-card").count() else "(closed)")
    pg.get_by_role("button", name="🧊 3D").click()
    pg.wait_for_timeout(3500)
    shot(pg, "A5_3d")
    print("A5 3D canvas:", pg.locator(".twin3d canvas").count())
    pg.get_by_role("button", name="📷 Camera").click()
    pg.get_by_role("button", name="✋ Hands").click()
    pg.wait_for_timeout(2500)
    print("A6 hands pill:", pg.locator(".hand-pill").all_inner_texts(), "| live sync enabled:", pg.evaluate("fetch('/api/live/status').then(r=>r.json()).then(d=>d.enabled)"))
    pg.get_by_role("button", name="✋ Hands on").click()
    pg.wait_for_timeout(500)
    print("A6 live sync after hands off:", pg.evaluate("fetch('/api/live/status').then(r=>r.json()).then(d=>d.enabled)"))
    b.close()

    # ---------------- B. phone
    desk_b = p.chromium.launch(args=GL)                         # no camera on the laptop
    desk = desk_b.new_context(viewport={"width": 1600, "height": 960}).new_page()
    desk.on("pageerror", lambda e: errs.append("B desk: " + str(e)))
    desk.goto("http://localhost:8000/")
    desk.wait_for_selector(".setup-card", timeout=20000)
    shot(desk, "B1_setup")
    print("B1 setup:", desk.locator(".setup-card").inner_text().replace("\n", " | ")[:200])
    ph_b = p.chromium.launch(args=CAM)
    ph = ph_b.new_context(**p.devices["Pixel 7"], ignore_https_errors=True, permissions=["camera"]).new_page()
    ph.on("pageerror", lambda e: errs.append("B phone: " + str(e)))
    ph.goto(f"https://{ip}:8443/phone")
    ph.wait_for_timeout(4000)
    print("B2 phone:", ph.locator("#live").inner_text(), "|", ph.locator("#stat").inner_text())
    desk.wait_for_selector(".map-progress", timeout=30000)
    print("B3 desk switched to:", desk.locator(".source-chip > button").inner_text())
    desk.wait_for_selector(".map-progress", state="detached", timeout=300000)
    desk.wait_for_timeout(2500)
    shot(desk, "B4_phone_mapped")
    print("B4 top:", desk.locator(".live-top").inner_text().replace("\n", " "), "| bg:",
          desk.evaluate("[...document.querySelectorAll('.live-stage img.layer, .live-stage video')].map(e => e.getAttribute('src') || 'video')"))
    desk_b.close(); ph_b.close()
print("page errors:", errs)
