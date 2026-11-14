"""Drive the experiment lab in Chromium the way a student would.

Usage: python backend/tests/e2e/ui_lab.py OUT_DIR [prompt ...]"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
prompts = sys.argv[2:] or ["two blue whales fight underwater"]
with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    logs = []
    pg.on("console", lambda m: logs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: logs.append(f"PAGEERROR: {e}"))
    pg.goto("http://localhost:8000/")
    pg.wait_for_timeout(1200)
    inp = pg.locator(".cmd-input")

    def run(text, tag, waits=(1500, 3500)):
        inp.fill(text)
        inp.press("Enter")
        pg.wait_for_selector(".busy-veil", state="detached", timeout=180000)
        pg.wait_for_timeout(400)
        for i, w in enumerate(waits):
            pg.wait_for_timeout(w)
            pg.screenshot(path=str(out / f"{tag}_{i}.png"))
        err = pg.locator(".statusline .error")
        if err.count():
            print(f"[{text}] ERROR:", err.inner_text())

    for k, t in enumerate(prompts):
        run(t, f"p{k}")
        tiles = pg.locator(".tile").all_inner_texts()
        print(f"[{t}] tiles:", [x.replace("\n", " = ") for x in tiles[:6]])
    if "--followups" in sys.argv or len(prompts) == 1:
        run("how hard did they hit?", "q", waits=(300,))
        print("answer:", pg.locator(".answer").inner_text()[:300] if pg.locator(".answer").count() else None)
        run("what if they were twice as heavy?", "w")
        print("changes:", pg.locator(".changes").inner_text() if pg.locator(".changes").count() else None)
        pg.get_by_role("button", name="Change & re-run").click()
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(out / "edit.png"))
        pg.get_by_role("button", name="Moon", exact=True).click()
        pg.get_by_role("button", name="Re-run with these values").click()
        pg.wait_for_selector(".busy-veil", state="detached", timeout=180000)
        pg.wait_for_timeout(2500)
        pg.screenshot(path=str(out / "edit_rerun.png"))
        pg.get_by_role("button", name="Runs").click()
        pg.wait_for_timeout(800)
        pg.screenshot(path=str(out / "runs.png"))
    print("console problems:", logs[:12])
    b.close()
