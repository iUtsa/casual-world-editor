"""Capture a Playwright page as timestamped JPEG frames through Chrome's screencast.

Playwright's own video recorder (record_video_dir) stalls getUserMedia streams in headless Chromium, which blanks the
camera picture; the raw screencast doesn't. Frames go to OUT/frames/NNNNNN.jpg, their times to OUT/frames.json, and
cut_demo.py turns them into a video."""
import base64
import json
from pathlib import Path


class Screencast:
    def __init__(self, ctx, page, out: Path, size=(1280, 720)):
        self.dir = Path(out) / "frames"; self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.glob("*.jpg"):
            f.unlink()
        self.times: list[float] = []
        self.out = Path(out)
        self.cdp = ctx.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 72, "maxWidth": size[0], "maxHeight": size[1]})

    def _frame(self, f):
        (self.dir / f"{len(self.times):06d}.jpg").write_bytes(base64.b64decode(f["data"]))
        self.times.append(f["metadata"]["timestamp"])
        self.cdp.send("Page.screencastFrameAck", {"sessionId": f["sessionId"]})

    def stop(self):
        try:
            self.cdp.send("Page.stopScreencast")
        except Exception:
            pass
        t0 = self.times[0] if self.times else 0
        (self.out / "frames.json").write_text(json.dumps([round(t - t0, 4) for t in self.times]))
        return t0
