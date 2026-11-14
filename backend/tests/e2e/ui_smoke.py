"""Drive the real UI in Chromium: synthetic scene -> scan -> remove -> explanation.

Usage: python backend/tests/e2e/ui_smoke.py OUT_DIR"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
    pg.goto("http://localhost:8000/")
    pg.get_by_role("button", name="Book bridge on two mugs").click()
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Scan scene").click()
    pg.wait_for_selector(".objlist li", timeout=60000)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out / "1_scanned.png"))
    n = pg.locator(".objlist li").count()
    print("objects in list:", n)
    # remove the object under the left mug position via the Remove tool
    pg.get_by_role("button", name="Remove").click()
    box = pg.locator(".stage").bounding_box()
    W, H = 960, 540
    x, y = 405, 290   # image coords of the left mug
    pg.mouse.click(box["x"] + x / W * box["width"], box["y"] + y / H * box["height"])
    pg.wait_for_selector(".history li", timeout=30000)
    pg.wait_for_timeout(400)
    pg.screenshot(path=str(out / "2_mid_sim.png"))
    pg.wait_for_timeout(3500)
    pg.screenshot(path=str(out / "3_after_sim.png"))
    pg.wait_for_selector(".explain .headline", timeout=60000)
    print("headline:", pg.locator(".explain .headline").inner_text())
    pg.locator(".explain").screenshot(path=str(out / "4_explanation.png"))
    pg.get_by_role("button", name="split").click()
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out / "5_split.png"))
    print("status:", pg.locator(".statusline").inner_text())
    print("\n".join(l for l in logs if "error" in l.lower() or "PAGEERROR" in l)[:3000])
    b.close()
