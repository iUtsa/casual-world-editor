"""Real photo: car experiment, then quantitative questions, measurements panel, velocity vectors.

Usage: python backend/tests/e2e/ui_measure.py IMAGE OUT_DIR"""
import re, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

img, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1300})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.locator("input[type=file]").first.set_input_files(img)
    pg.wait_for_timeout(500)
    pg.locator(".field input[type=range]").fill("50")
    pg.get_by_role("button", name=re.compile("Scan scene|Rescan frame")).click()
    pg.wait_for_selector(".statusline .busy", timeout=10000)
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    def cmd(text):
        pg.locator(".cmd-input").fill(text)
        pg.locator(".cmd-input").press("Enter")
        pg.wait_for_timeout(300)
        pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
        pg.wait_for_selector(".plan", timeout=60000)
    cmd("drive a toy car into object 2 fast")
    pg.get_by_role("button", name="Run experiment").click()
    pg.wait_for_selector(".statusline .busy", state="detached", timeout=60000)
    sim = pg.evaluate("fetch('/api/sim').then(r => r.json())")["sim"]
    seg = sim["segments"][-1]
    tb = pg.locator(".timeline .track").bounding_box()
    tt = seg["t_start"] + 0.45 * (seg["t_end"] - seg["t_start"])
    pg.mouse.click(tb["x"] + tt / sim["duration"] * tb["width"], tb["y"] + tb["height"] / 2)
    pg.wait_for_timeout(1500)
    pg.locator(".stage").screenshot(path=str(out / "vectors.png"))
    for q in ["How fast was the car when it hit object 2?", "How fast can the toy car go?", "How much energy did object 2 have?"]:
        cmd(q)
        print("Q:", q, "\n   A:", pg.locator(".answer").inner_text().replace("\n", " | ")[:400])
    pg.locator(".answer").screenshot(path=str(out / "answer.png"))
    pg.locator(".measure").scroll_into_view_if_needed()
    pg.wait_for_timeout(1200)
    ch = pg.locator(".chart").first.bounding_box()
    pg.mouse.move(ch["x"] + ch["width"] * 0.5, ch["y"] + ch["height"] * 0.5)
    pg.wait_for_timeout(300)
    pg.locator(".measure").screenshot(path=str(out / "measure.png"))
    print("errors:", errs[:5])
    b.close()
