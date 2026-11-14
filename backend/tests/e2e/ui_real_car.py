"""Real photo: summon a car with a typed command and drive it into a real mug.

Usage: python backend/tests/e2e/ui_real_car.py IMAGE HFOV COMMAND OUT_DIR"""
import re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

img, hfov, command, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1050})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.locator("input[type=file]").first.set_input_files(img)
    pg.wait_for_timeout(500)
    pg.locator(".field input[type=range]").fill(hfov)
    pg.get_by_role("button", name=re.compile("Scan scene|Rescan frame")).click()
    pg.wait_for_selector(".statusline .busy", timeout=10000)
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    pg.locator(".cmd-input").fill(command)
    pg.get_by_role("button", name="Plan").click()
    pg.wait_for_selector(".plan", timeout=30000)
    print("PLAN:", pg.locator(".plan").inner_text().replace("\n", " | "))
    pg.get_by_role("button", name="Run experiment").click()
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    sim = pg.evaluate("fetch('/api/sim').then(r => r.json())")["sim"]
    seg = sim["segments"][-1]
    print("moved:", {v["label"]: (v["displacement_m"], v["rotation_deg"], v["fell_off_table"]) for v in seg["summary"].values() if v["moved"]})
    print("chain:", [(c["cause"], c["kind"], c["effect"]) for c in seg["causal_edges"]])
    tb = pg.locator(".timeline .track").bounding_box()
    for k, frac in enumerate([0.2, 0.4, 0.6, 1.0]):
        tt = seg["t_start"] + frac * (seg["t_end"] - seg["t_start"])
        pg.mouse.click(min(tb["x"] + tt / sim["duration"] * tb["width"], tb["x"] + tb["width"] - 1), tb["y"] + tb["height"] / 2)
        pg.wait_for_timeout(1500)
        pg.locator(".stage").screenshot(path=str(out / f"f{k}.png"))
    print("errors:", errs[:5])
    b.close()
