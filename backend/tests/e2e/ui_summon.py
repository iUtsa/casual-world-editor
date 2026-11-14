"""Summon a car with a typed command and drive it into the selected object; place a duck from the library.

Usage: python backend/tests/e2e/ui_summon.py OUT_DIR"""
import re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1050})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
    pg.goto("http://localhost:8000/")
    pg.get_by_role("button", name="Book bridge on two mugs").click()
    pg.wait_for_timeout(800)
    pg.get_by_role("button", name=re.compile("Scan scene|Rescan frame")).click()
    pg.wait_for_selector(".statusline .busy", timeout=10000)
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    box = pg.locator(".stage").bounding_box()
    W, H = 960, 540
    at = lambda x, y: (box["x"] + x / W * box["width"], box["y"] + y / H * box["height"])
    pg.mouse.click(*at(672, 170))                         # the bottle
    pg.wait_for_timeout(300)
    pg.locator(".cmd-input").fill("drive a toy car into it fast")
    pg.get_by_role("button", name="Plan").click()
    pg.wait_for_selector(".plan", timeout=30000)
    print("PLAN:", pg.locator(".plan").inner_text().replace("\n", " | "))
    pg.get_by_role("button", name="Run experiment").click()
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    sim = pg.evaluate("fetch('/api/sim').then(r => r.json())")["sim"]
    drive = sim["segments"][-1]
    print("drive summary:", {k: (v["label"], v["displacement_m"], v["moved"]) for k, v in drive["summary"].items() if v["moved"]})
    t0 = drive["t_start"]
    for k, frac in enumerate([0.25, 0.55, 1.0]):
        tb = pg.locator(".timeline .track").bounding_box()
        x = tb["x"] + (t0 + frac * (drive["t_end"] - t0)) / sim["duration"] * tb["width"]
        pg.mouse.click(min(x, tb["x"] + tb["width"] - 1), tb["y"] + tb["height"] / 2)
        pg.wait_for_timeout(1200)
        pg.locator(".stage").screenshot(path=str(out / f"car_{k}.png"))
    # library placement
    pg.locator(".lib-item", has_text="rubber duck").click()
    pg.mouse.click(*at(250, 430))
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    pg.wait_for_timeout(2500)
    pg.locator(".stage").screenshot(path=str(out / "duck.png"))
    pg.screenshot(path=str(out / "full.png"))
    print("history:", pg.locator(".history").inner_text().replace("\n", " | "))
    print("errors:", errs[:5])
    b.close()
