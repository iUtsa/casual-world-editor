"""Live sync between the real world and the 3D world.

The phone does the cheap, always-on work (it is the one holding the camera): motion/steadiness from its
gyro and frame differencing, sharpness, and "did the scene change since the last scan?". It streams
frames tagged with that analysis. When the scene changed and the phone is steady again, it asks for a
precision rescan; the Mac then

  1. takes the sharpest few steady frames of the burst,
  2. estimates metric depth on each and fuses them with a per-pixel median (monocular depth noise is
     largely independent frame to frame, so the median is markedly more accurate than any single map),
  3. rebuilds the 3D world from the sharpest frame + fused depth,
  4. reports which objects appeared / disappeared, and
  5. re-runs the active real-world experiment in the updated world, so what you add to the table is
     in the simulation too.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import cv2
import numpy as np

log = logging.getLogger(__name__)

_buf: deque = deque(maxlen=6)            # (t, jpeg, sharpness) of recent STEADY frames
_lock = threading.Lock()
STATE = {"version": 0, "state": "idle", "last_ms": None, "added": [], "removed": [], "error": None,
         "frames_fused": 0, "enabled": True, "t": 0.0, "pending": False, "rerun": None}


def sharpness(jpeg: bytes) -> float:
    g = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return 0.0
    g = cv2.resize(g, (320, int(320 * g.shape[0] / g.shape[1])))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def offer(jpeg: bytes, steady: bool) -> None:
    """Every phone frame passes here; steady ones are kept for the next precision scan."""
    with _lock:
        if not steady:
            _buf.clear()
            return
        _buf.append((time.time(), jpeg, sharpness(jpeg)))


def status() -> dict:
    with _lock:
        return {k: v for k, v in STATE.items()} | {"buffered": len(_buf)}


def request_rescan(hfov: float, on_done) -> dict:
    """Debounced: one rescan at a time; a request during a rescan queues exactly one more."""
    with _lock:
        if not STATE["enabled"]:
            return status()
        if STATE["state"] == "scanning":
            STATE["pending"] = True
            return status()
        STATE["state"] = "scanning"
    threading.Thread(target=_rescan, args=(hfov, on_done), daemon=True).start()
    return status()


def _rescan(hfov: float, on_done) -> None:
    from . import remote
    from .perception import models, pipeline
    t0 = time.perf_counter()
    try:
        with _lock:
            frames = list(_buf)
        if not frames:
            jpeg, _ = remote.latest()
            if jpeg is None:
                raise RuntimeError("no phone frame")
            frames = [(time.time(), jpeg, sharpness(jpeg))]
        frames.sort(key=lambda f: -f[2])
        use = frames[:3]
        rgb = pipeline.load_frame(use[0][1])
        depth = None
        if len(use) >= 2:
            from PIL import Image
            maps = []
            for _, jp, _ in use:
                im = pipeline.load_frame(jp)
                if im.shape != rgb.shape:
                    continue
                maps.append(models.estimate_depth(Image.fromarray(im)).astype(np.float32))
            if len(maps) >= 2:
                depth = np.median(np.stack(maps), axis=0)
        st = pipeline.scan(rgb, hfov, "phone-live", use_claude=False, depth_override=depth)
        added, removed, rerun = on_done(st)
        with _lock:
            STATE.update(version=STATE["version"] + 1, state="idle", last_ms=round((time.perf_counter() - t0) * 1000),
                         added=added, removed=removed, error=None, frames_fused=len(use) if depth is not None else 1,
                         t=time.time(), rerun=rerun)
    except Exception as e:      # never kill the live loop
        log.exception("live rescan failed")
        with _lock:
            STATE.update(state="idle", error=str(e))
    finally:
        with _lock:
            again = STATE["pending"]
            STATE["pending"] = False
        if again:
            request_rescan(hfov, on_done)
