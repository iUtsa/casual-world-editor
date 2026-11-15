"""Phone as a camera: LAN address, self-signed HTTPS (phone browsers only allow live camera on
HTTPS), QR code, and a latest-frame store with an MJPEG stream for the desktop view.
Frames go phone -> this Mac over the local network only."""
from __future__ import annotations

import io
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

from . import config

CERT_DIR = config.DATA_DIR / "certs"
HTTPS_PORT = 8443
HTTP_PORT = 8000
_latest = {"jpeg": None, "t": 0.0, "n": 0, "t0": 0.0, "meta": {}}
_lock = threading.Lock()


def lan_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))     # no packet is sent; picks the outward interface
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def ensure_cert() -> tuple[Path, Path]:
    """Self-signed certificate for this Mac's LAN IP (regenerated if the IP changes)."""
    ip = lan_ip() or "127.0.0.1"
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    key, crt, stamp = CERT_DIR / "key.pem", CERT_DIR / "cert.pem", CERT_DIR / "ip.txt"
    if not (key.exists() and crt.exists() and stamp.exists() and stamp.read_text() == ip):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                        "-keyout", str(key), "-out", str(crt), "-subj", "/CN=Casual World Editor (local)",
                        "-addext", f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"],
                       check=True, capture_output=True)
        stamp.write_text(ip)
    return key, crt


PUBLIC_ORIGIN = re.compile(r"https://[a-z0-9-]+(\.[a-z0-9-]+)+")   # e.g. https://casual-world-editor.pages.dev


def _qr(url: str) -> str:
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf)
    return buf.getvalue().decode()


def connect_info(origin: str | None = None) -> dict:
    """Where the phone should go. Opened through a public HTTPS address (the hosted demo), the phone uses that
    address too: any network, a real certificate. Opened locally, the phone uses this Mac's LAN address."""
    if origin and PUBLIC_ORIGIN.fullmatch(origin.lower()):
        url = f"{origin.lower()}/phone"
        return {"ok": True, "public": True, "https_url": url, "http_url": url, "qr_svg": _qr(url)}
    ip = lan_ip()
    if ip is None:
        return {"ok": False, "reason": "This Mac isn't on a network."}
    https = f"https://{ip}:{HTTPS_PORT}/phone"
    http = f"http://{ip}:{HTTP_PORT}/phone"
    return {"ok": True, "public": False, "ip": ip, "https_url": https, "http_url": http, "qr_svg": _qr(https)}


def put_frame(jpeg: bytes, meta: dict | None = None) -> None:
    with _lock:
        now = time.time()
        if now - _latest["t"] > 3:
            _latest["n"], _latest["t0"] = 0, now
        _latest.update(jpeg=jpeg, t=now, n=_latest["n"] + 1, meta=meta or {})


def latest() -> tuple[bytes | None, float]:
    with _lock:
        return _latest["jpeg"], _latest["t"]


def status() -> dict:
    with _lock:
        age = time.time() - _latest["t"] if _latest["jpeg"] else None
        span = max(1e-3, _latest["t"] - _latest["t0"])
        return {"connected": age is not None and age < 3, "age_s": round(age, 2) if age is not None else None,
                "fps": round((_latest["n"] - 1) / span, 1) if _latest["n"] > 1 else 0.0, "meta": _latest["meta"]}


SHUTTING_DOWN = threading.Event()


async def mjpeg(request):
    """Live phone video for the desktop. Ends when the viewer disconnects or the server stops
    (an endless stream would otherwise block graceful shutdown)."""
    import asyncio
    last = 0.0
    while not SHUTTING_DOWN.is_set():
        if await request.is_disconnected():
            break
        jpeg, t = latest()
        if jpeg and t != last:
            last = t
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        await asyncio.sleep(0.04)
