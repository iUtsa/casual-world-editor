"""Real photo through the UI: upload -> scan -> push a mug off the table.

Usage: python backend/tests/e2e/ui_real_photo.py IMAGE HFOV OUT_DIR"""
import json, re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

img, hfov, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.locator("input[type=file]").first.set_input_files(img)
    pg.wait_for_timeout(500)
    pg.locator(".field input[type=range]").fill(str(int(hfov)) if hfov == int(hfov) else str(hfov))
    pg.get_by_role("button", name=re.compile("Scan scene|Rescan frame")).click()
    pg.wait_for_selector(".statusline .busy", timeout=10000)
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    pg.wait_for_selector(".objlist li", timeout=60000)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(out / "1_scanned.png"))
    scene = pg.evaluate("fetch('/api/scene').then(r => r.json())")["scene"]
    W, H = scene["camera"]["width"], scene["camera"]["height"]
    # leftmost mug, push it toward the left table edge
    mug = min((o for o in scene["objects"] if o["size"][2] > 0.05), key=lambda o: o["bbox"][0])
    cx, cy = mug["bbox"][0] + mug["bbox"][2] / 2, mug["bbox"][1] + mug["bbox"][3] / 2
    pg.get_by_role("button", name="Push").click()
    box = pg.locator(".stage").bounding_box()
    to_px = lambda x, y: (box["x"] + x / W * box["width"], box["y"] + y / H * box["height"])
    pg.mouse.move(*to_px(cx, cy)); pg.mouse.down()
    for k in range(1, 11):
        pg.mouse.move(*to_px(cx - 9 * k, cy + 2 * k))
    pg.mouse.up()
    pg.wait_for_selector(".history li", timeout=30000)
    pg.wait_for_timeout(700)
    pg.screenshot(path=str(out / "2_mid.png"))
    pg.wait_for_timeout(4000)
    pg.screenshot(path=str(out / "3_end.png"))
    sim = pg.evaluate("fetch('/api/sim').then(r => r.json())")["sim"]
    seg = sim["segments"][-1]
    print("pushed", mug["id"], "params", {k: v for k, v in seg["intervention"]["params"].items() if k != "point"})
    print("events:", [(e["type"], e["subject"], e["other"]) for e in seg["events"] if e["type"] != "came_to_rest"][:12])
    print("summary:", {k: (v["displacement_m"], v["fell_off_table"]) for k, v in seg["summary"].items()})
    pg.wait_for_selector(".explain .headline", timeout=60000)
    print("headline:", pg.locator(".explain .headline").inner_text())
    print("page errors:", errs)
    b.close()
