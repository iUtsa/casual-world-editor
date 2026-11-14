"""Space scenes in the Studio: solar system, comet, galaxies, Moon, collision, Apophis. Screenshots + page errors.

Usage: python backend/tests/e2e/ui_space.py OUT_DIR [prompt ...]"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
prompts = sys.argv[2:] or ["the solar system", "halley's comet", "the milky way and andromeda collide", "the moon orbits the earth",
                           "jupiter collides with the moon", "apophis flyby 2029"]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_context(viewport={"width": 1600, "height": 960}).new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.on("console", lambda m: errs.append("console: " + m.text) if m.type == "error" else None)
    pg.goto("http://localhost:8000/")
    pg.get_by_role("button", name="Studio", exact=True).click(); pg.wait_for_timeout(1500)
    for i, q in enumerate(prompts):
        box = pg.locator("input.cmd-input").first
        box.fill(q); box.press("Enter")
        pg.wait_for_function("document.querySelector('.space-hud')", timeout=240000)
        pg.wait_for_timeout(9000)
        pg.screenshot(path=str(out / f"space_{i}.png"))
        badge = pg.locator(".theory-badge").first.inner_text().replace("\n", " ") if pg.locator(".theory-badge").count() else "-"
        print(f"{q!r}: hud={pg.locator('.space-hud').inner_text().replace(chr(10), ' | ')}  badge={badge}")
    print("page errors:", errs[:6])
    b.close()
