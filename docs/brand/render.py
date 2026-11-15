"""Render docs/brand/brand.html into docs/assets/banner.png (README) and docs/assets/social-preview.png (GitHub).
Usage: .venv/bin/python docs/brand/render.py"""
from pathlib import Path

from playwright.sync_api import sync_playwright

here = Path(__file__).resolve().parent
assets = here.parent / "assets"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1700, "height": 1400}, device_scale_factor=1)
    pg.goto((here / "brand.html").as_uri(), wait_until="networkidle"); pg.wait_for_timeout(800)
    pg.locator("#banner").screenshot(path=str(assets / "banner.png"), omit_background=True)
    pg.locator("#social").screenshot(path=str(assets / "social-preview.png"))
    b.close()
