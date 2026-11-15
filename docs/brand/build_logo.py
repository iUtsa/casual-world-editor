"""Build the Causal World Editor logo kit in logo/ from the master icon (docs/brand/cwe-icon-master.svg).

SVGs are pure vector (the wordmark is outlined, so no font is needed to display them); PNGs are rendered with Chromium.
Usage: .venv/bin/python docs/brand/build_logo.py PATH/TO/BricolageGrotesque[opsz,wdth,wght].ttf"""
import io
import re
import sys
from pathlib import Path

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logo"
INK, CREAM, LIME = "#1E2622", "#F4F1E8", "#C8F135"
NAME = "Causal World Editor"
TAGLINE = ("What you imagine, ", "you can simulate.")

master = (Path(__file__).parent / "cwe-icon-master.svg").read_text()
P = {k: re.search(rf'<path id="{k}" d="([^"]+)"', master).group(1) for k in ("ground", "throw", "ball")}
MARK_T = "translate(0.64 -19.3)"
MARK_BOX = (50, 128, 413, 257)          # x, y, w, h of the mark alone (strokes included), in icon units


def mark(stroke, ball, ball_outline=None, opacity=0.92):
    o = f' stroke="{ball_outline}" stroke-width="9" paint-order="stroke"' if ball_outline else ""
    return (f'<g transform="{MARK_T}">'
            f'<path d="{P["ground"]}" fill="none" stroke="{stroke}" stroke-width="24.7" stroke-linecap="round" stroke-linejoin="round"/>'
            f'<path d="{P["throw"]}" fill="none" stroke="{stroke}" stroke-opacity="{opacity}" stroke-width="20.9" stroke-linecap="round" stroke-dasharray="23.43 26.19"/>'
            f'<path d="{P["ball"]}" fill="{ball}"{o}/></g>')


def tile(bg, stroke, ball, outline=None):
    return f'<rect width="512" height="512" rx="116" fill="{bg}"/>' + mark(stroke, ball, outline)


# ---- outlined text
def load_font(path, wght, opsz):
    f = TTFont(path)
    f = instancer.instantiateVariableFont(f, {"wght": wght, "opsz": opsz, "wdth": 100})
    buf = io.BytesIO(); f.save(buf)
    return f, buf.getvalue()


def text_path(fontpair, text, size, x0=0.0, baseline=0.0, tracking=0.0):
    """Path data for `text` at `size` px with its baseline at y=baseline; returns (d, width)."""
    font, data = fontpair
    face = hb.Face(data); hfont = hb.Font(face)
    buf = hb.Buffer(); buf.add_str(text); buf.guess_segment_properties()
    hb.shape(hfont, buf, {"kern": True, "liga": True})
    upm = font["head"].unitsPerEm; s = size / upm
    gs = font.getGlyphSet(); order = font.getGlyphOrder()
    pen = SVGPathPen(gs); x = 0.0
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
        g = order[info.codepoint]
        gs[g].draw(TransformPen(pen, (s, 0, 0, -s, x0 + (x + pos.x_offset) * s, baseline - pos.y_offset * s)))
        x += pos.x_advance + tracking * upm
    return pen.getCommands(), (x - tracking * upm) * s


def cap_height(fontpair, size):
    font = fontpair[0]
    return font["OS/2"].sCapHeight / font["head"].unitsPerEm * size


def svg(w, h, body, title, bg=None):
    b = f'<rect width="{w}" height="{h}" fill="{bg}"/>' if bg else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:g}" height="{h:g}" viewBox="0 0 {w:g} {h:g}">'
            f'<title>{title}</title>{b}{body}</svg>\n')


def write(name, content):
    (OUT / "svg").mkdir(parents=True, exist_ok=True)
    (OUT / "svg" / name).write_text(content)
    return OUT / "svg" / name


def main(font_path):
    bold = load_font(font_path, 800, 72)
    med = load_font(font_path, 500, 24)
    files = {}

    # 1. app icons (tile)
    files["cwe-icon"] = write("cwe-icon.svg", svg(512, 512, tile(INK, CREAM, LIME), "Causal World Editor app icon"))
    files["cwe-icon-light"] = write("cwe-icon-light.svg", svg(512, 512, tile(CREAM, INK, LIME, INK), "Causal World Editor app icon, light"))
    files["cwe-icon-lime"] = write("cwe-icon-lime.svg", svg(512, 512, tile(LIME, INK, CREAM, INK), "Causal World Editor app icon, lime"))

    # 2. the mark alone, transparent background
    x, y, w, h = MARK_BOX
    def bare(body, title):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{x} {y} {w} {h}"><title>{title}</title>{body}</svg>\n')
    files["cwe-mark"] = write("cwe-mark.svg", bare(mark(INK, LIME, INK), "Causal World Editor mark, for light backgrounds"))
    files["cwe-mark-on-dark"] = write("cwe-mark-on-dark.svg", bare(mark(CREAM, LIME), "Causal World Editor mark, for dark backgrounds"))
    files["cwe-mark-mono-ink"] = write("cwe-mark-mono-ink.svg", bare(mark(INK, INK, opacity=1), "Causal World Editor mark, one colour"))
    files["cwe-mark-mono-white"] = write("cwe-mark-mono-white.svg", bare(mark("#FFFFFF", "#FFFFFF", opacity=1), "Causal World Editor mark, one colour white"))

    # 3. lockups: icon + outlined wordmark
    S = 512; size = 210; gap = 120
    capH = cap_height(bold, size)
    base = S / 2 + capH / 2
    for theme, txt, ico in (("", INK, tile(INK, CREAM, LIME)), ("-on-dark", CREAM, tile(CREAM, INK, LIME, INK))):
        d, tw = text_path(bold, NAME, size, x0=S + gap, baseline=base, tracking=-0.02)
        W = S + gap + tw + 8
        files[f"cwe-logo-horizontal{theme}"] = write(f"cwe-logo-horizontal{theme}.svg", svg(W, S, ico + f'<path d="{d}" fill="{txt}"/>', f"Causal World Editor logo{theme.replace('-', ' ')}"))
        # stacked: icon over two-line wordmark
        ls = 190
        d1, w1 = text_path(bold, "Causal World", ls, tracking=-0.02); d2, w2 = text_path(bold, "Editor", ls, tracking=-0.02)
        W2 = max(w1, w2, S) + 40
        top = S + 110 + cap_height(bold, ls)
        d1, _ = text_path(bold, "Causal World", ls, x0=(W2 - w1) / 2, baseline=top, tracking=-0.02)
        d2, _ = text_path(bold, "Editor", ls, x0=(W2 - w2) / 2, baseline=top + ls * 1.02, tracking=-0.02)
        H2 = top + ls * 1.02 + 60
        files[f"cwe-logo-stacked{theme}"] = write(f"cwe-logo-stacked{theme}.svg", svg(W2, H2, f'<g transform="translate({(W2 - S) / 2:.1f} 0)">{ico}</g><path d="{d1} {d2}" fill="{txt}"/>', f"Causal World Editor logo, stacked{theme.replace('-', ' ')}"))

    # 4. wordmark alone
    for theme, txt in (("", INK), ("-on-dark", CREAM)):
        capH = cap_height(bold, 200)
        d, tw = text_path(bold, NAME, 200, x0=4, baseline=capH + 50, tracking=-0.02)
        files[f"cwe-wordmark{theme}"] = write(f"cwe-wordmark{theme}.svg", svg(tw + 8, capH + 100, f'<path d="{d}" fill="{txt}"/>', "Causal World Editor wordmark"))

    # 5. Devpost thumbnail / cover (3:2) and a 16:9 banner, all outlined
    def cover(W, H, title):
        body = ""
        for gx in range(24, int(W), 32):                  # faint dot grid, like the site
            for gy in range(24, int(H), 32):
                body += f'<circle cx="{gx}" cy="{gy}" r="1.6" fill="{CREAM}" fill-opacity="0.07"/>'
        ms = H * 0.30 / MARK_BOX[3]                       # mark height = 30% of the card
        mw = MARK_BOX[2] * ms
        body += f'<g transform="translate({(W - mw) / 2 - MARK_BOX[0] * ms:.1f} {H * 0.12 - MARK_BOX[1] * ms:.1f}) scale({ms:.4f})">{mark(CREAM, LIME)}</g>'
        ws = H * 0.105
        _, tw = text_path(bold, NAME, ws, tracking=-0.025)
        d, _ = text_path(bold, NAME, ws, x0=(W - tw) / 2, baseline=H * 0.62, tracking=-0.025)
        body += f'<path d="{d}" fill="{CREAM}"/>'
        ts = H * 0.052
        _, a = text_path(med, TAGLINE[0], ts); _, b = text_path(med, TAGLINE[1], ts)
        x0 = (W - a - b) / 2
        da, _ = text_path(med, TAGLINE[0], ts, x0=x0, baseline=H * 0.75)
        db, _ = text_path(med, TAGLINE[1], ts, x0=x0 + a, baseline=H * 0.75)
        body += f'<path d="{da}" fill="{CREAM}" fill-opacity="0.82"/><path d="{db}" fill="{LIME}"/>'
        chips = ["Real world", "Imagination", "Checked against theory", "On-device"]
        cs = H * 0.03; pad = cs * 0.9; widths = [text_path(med, c, cs)[1] + 2 * pad for c in chips]; g = cs * 0.8
        cx = (W - sum(widths) - g * (len(chips) - 1)) / 2; cy = H * 0.84
        for c, cw in zip(chips, widths):
            d, _ = text_path(med, c, cs, x0=cx + pad, baseline=cy + cs * 1.05)
            body += (f'<rect x="{cx:.1f}" y="{cy - cs * 0.2:.1f}" width="{cw:.1f}" height="{cs * 1.7:.1f}" rx="{cs * 0.85:.1f}" fill="none" stroke="{CREAM}" stroke-opacity="0.35" stroke-width="2"/>'
                     f'<path d="{d}" fill="{CREAM}" fill-opacity="0.9"/>')
            cx += cw + g
        return svg(W, H, body, title, bg=INK)
    files["devpost-thumbnail"] = write("devpost-thumbnail.svg", cover(1500, 1000, "Causal World Editor, Devpost thumbnail"))
    files["cover-16x9"] = write("cover-16x9.svg", cover(1920, 1080, "Causal World Editor cover"))

    # 6. PNGs
    png = OUT / "png"; png.mkdir(exist_ok=True)
    jobs = [("cwe-icon", [1024, 512, 256, 180, 128, 64, 32]), ("cwe-icon-light", [1024, 512]), ("cwe-icon-lime", [1024, 512]),
            ("cwe-mark", [1024]), ("cwe-mark-on-dark", [1024]), ("cwe-mark-mono-ink", [1024]), ("cwe-mark-mono-white", [1024]),
            ("cwe-logo-horizontal", [2400]), ("cwe-logo-horizontal-on-dark", [2400]), ("cwe-logo-stacked", [1200]),
            ("cwe-logo-stacked-on-dark", [1200]), ("cwe-wordmark", [2000]), ("cwe-wordmark-on-dark", [2000]),
            ("devpost-thumbnail", [1500]), ("cover-16x9", [1920])]
    with sync_playwright() as p:
        b = p.chromium.launch(); pg = b.new_page()
        for name, widths in jobs:
            src = files[name].read_text()
            m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.]+) ([\d.]+)"', src)
            vw, vh = float(m.group(3)), float(m.group(4))
            for wpx in widths:
                hpx = round(vh * wpx / vw)
                pg.set_viewport_size({"width": wpx, "height": hpx})
                pg.set_content(f'<html><body style="margin:0;background:transparent">{src.replace("<svg ", f"<svg style=\"display:block;width:{wpx}px;height:{hpx}px\" ", 1)}</body></html>')
                suffix = "" if len(widths) == 1 else f"-{wpx}"
                pg.screenshot(path=str(png / f"{name}{suffix}.png"), omit_background=True)
        b.close()
    Image.open(png / "cwe-icon-64.png").save(OUT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print("\n".join(sorted(str(f.relative_to(ROOT)) for f in OUT.rglob("*") if f.is_file())))


if __name__ == "__main__":
    main(sys.argv[1])
