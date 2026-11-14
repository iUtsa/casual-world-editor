"""Client for the local VLM worker (free, on this Mac). Every output is parsed and validated;
nothing it says is trusted blindly."""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import time
import urllib.request

from . import config

log = logging.getLogger(__name__)
PORT = 8011
_proc = None
AI_PY = config.ROOT / ".venv-ai" / "bin" / "python"


def start() -> None:
    global _proc
    if not AI_PY.exists() or _proc is not None:
        return
    log_path = config.DATA_DIR / "ai_worker.log"
    _proc = subprocess.Popen([str(AI_PY), str(config.ROOT / "backend" / "ai_worker.py"), str(PORT)],
                             stdout=open(log_path, "a"), stderr=subprocess.STDOUT)


def status() -> dict:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1) as r:
            return json.loads(r.read())
    except Exception:
        return {"ok": False}


def available() -> bool:
    import os
    if os.environ.get("CWE_DISABLE_LOCAL_AI") == "1":      # tests: deterministic paths only
        return False
    return bool(status().get("ok"))


def generate(prompt: str, image_jpeg: bytes | None = None, max_tokens: int = 300, timeout: float = 120) -> str | None:
    body = {"prompt": prompt, "max_tokens": max_tokens}
    if image_jpeg:
        body["image_b64"] = base64.b64encode(image_jpeg).decode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["text"]
    except Exception as e:
        log.warning("local AI unavailable: %s", e)
        return None


def extract_json(text: str | None):
    """First balanced {...} block in the model output, parsed; None if there isn't one."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
