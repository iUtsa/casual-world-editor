"""Cut a recording from record_demo.py / record_imagine.py into the landing-page video, poster and README GIF.

Waiting (mapping, solving) is fast-forwarded with an on-screen "×N" tag; everything else plays at real speed.
Usage: python backend/tests/e2e/cut_demo.py real|imagine REC_DIR OUT_NAME
Writes frontend/public/landing/media/OUT_NAME.mp4 + .jpg and docs/assets/OUT_NAME.gif"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

kind, rec, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
FPS = 30
# (from mark, to mark, how): a number is a speed-up, "fit:S" squeezes the span into S seconds; "mark+S" starts S s later
PLANS = {
    "real": [("landing+2.2", "lab", 1), ("lab", "map", "fit:1.5"), ("map", "mapped", "fit:3"), ("mapped", "inspect", 1),
             ("inspect", "ask", 1), ("ask", "answer", "fit:2.5"), ("answer", "twin", 1), ("twin", "touch", 1),
             ("touch", "touch_done", "fit:9"), ("touch_done", "end", 1)],
    "imagine": [("studio", "whales", "fit:4"), ("whales", "space", "fit:14"), ("space", "chem", "fit:13"),
                ("chem", "ai", "fit:13"), ("ai", "end", "fit:14")],
}
POSTER_AT = {"real": "answer", "imagine": "space"}

times = json.loads((rec / "frames.json").read_text())
marks = json.loads((rec / "marks.json").read_text())
frames = sorted((rec / "frames").glob("*.jpg"))
assert len(frames) == len(times), (len(frames), len(times))


def frame_at(t):
    lo, hi = 0, len(times) - 1
    while lo < hi:                     # last frame shown at or before t
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if times[mid] <= t else (lo, mid - 1)
    return lo


picks = []                             # (frame index, speed-up shown on screen)
for a, b, how in PLANS[kind]:
    at = lambda m: marks[m.split("+")[0]] + (float(m.split("+")[1]) if "+" in m else 0)
    if a.split("+")[0] not in marks or b.split("+")[0] not in marks:
        print("skip", a, b); continue
    ta, tb = at(a), at(b)
    speed = (tb - ta) / float(how[4:]) if isinstance(how, str) else float(how)
    speed = max(speed, 1.0)
    t = ta
    while t < tb:
        picks.append((frame_at(t), speed)); t += speed / FPS

media = Path("frontend/public/landing/media"); media.mkdir(parents=True, exist_ok=True)
ff = imageio_ffmpeg.get_ffmpeg_exe()
W, H = Image.open(frames[0]).size
try:
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
except OSError:
    font = ImageFont.load_default()


def render(i, speed):
    im = Image.open(frames[i]).convert("RGB")
    if speed >= 1.5:                   # say honestly that this part is sped up
        d = ImageDraw.Draw(im); label = f"×{speed:.0f} speed" if speed >= 2 else "sped up"
        x0, y0 = (W - 148) // 2 + 40, 14
        d.rounded_rectangle((x0, y0, x0 + 148, y0 + 38), 9, fill=(255, 210, 63), outline=(20, 20, 20), width=3)
        d.text((x0 + 74, y0 + 19), label, fill=(20, 20, 20), font=font, anchor="mm")
    return im


def encode(path, args, scale=None):
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                          "-i", "-"] + args + [str(path)], stdin=subprocess.PIPE)
    last = None
    for i, sp in picks:
        key = (i, sp >= 1.5)
        if key != last:
            buf = render(i, sp).tobytes(); last = key
        p.stdin.write(buf)
    p.stdin.close(); p.wait()
    assert p.returncode == 0, path


mp4 = media / f"{name}.mp4"
encode(mp4, ["-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"])
render(frame_at(marks[POSTER_AT[kind]] + 2.5), 1).save(media / f"{name}.jpg", quality=82, optimize=True, progressive=True)

# README GIF: same cut at 2× speed, 640 px, 10 fps, one shared palette
gif = Path("docs/assets") / f"{name}.gif"
tmp = rec / "gif.mp4"
subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "setpts=PTS/2,fps=10,scale=640:-1:flags=lanczos", "-an", str(tmp)], check=True)
subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(tmp), "-vf",
                "split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                "-loop", "0", str(gif)], check=True)
for f in (mp4, media / f"{name}.jpg", gif):
    print(f"{f}: {f.stat().st_size / 1e6:.2f} MB")
print(f"length {len(picks) / FPS:.1f} s from {len(frames)} frames")
