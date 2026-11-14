"""Chemistry mode: type a reaction, wait for quantum dynamics, screenshot mid-reaction.

Usage: python backend/tests/e2e/ui_chem.py "REACTION" OUT_DIR"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

reaction, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1400, "height": 1300})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto("http://localhost:8000/")
    pg.get_by_role("button", name="Chemistry").click()
    pg.locator(".cmd-input").fill(reaction)
    pg.get_by_role("button", name="Simulate").click()
    pg.wait_for_selector(".chem-eq", timeout=60000)
    print("equation:", pg.locator(".chem-eq").inner_text().replace("\n", " | "))
    pg.wait_for_selector(".events li", timeout=180000)
    pg.wait_for_timeout(500)
    first = pg.locator(".events li").first
    print("first event:", first.inner_text())
    first.click()                 # jump to the first bond event
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(out / "chem.png"))
    print("events:", pg.locator(".events").inner_text().replace("\n", " | ")[:300])
    print("errors:", errs[:3])
    b.close()
