"""Local vision-language model worker (Qwen2.5-VL-3B-Instruct, 4-bit, MLX on the Apple GPU).

Runs in its own virtualenv (.venv-ai) so MLX never interferes with the main app.
Started by the backend; listens on 127.0.0.1 only.
POST /generate {"prompt": str, "image_b64": str|null, "max_tokens": int} -> {"text": str, "seconds": float}
"""
import base64
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = os.environ.get("CWE_VLM", "mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
_lock = threading.Lock()
_state = {"model": None, "processor": None, "error": None}


def _load():
    from mlx_vlm import load
    with _lock:
        if _state["model"] is None:
            _state["model"], _state["processor"] = load(MODEL_ID)


def _generate(prompt: str, image_b64: str | None, max_tokens: int) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    _load()
    m, p = _state["model"], _state["processor"]
    path = None
    if image_b64:
        # Qwen2.5-VL works on a 28 px grid (14 px patches merged 2x2); off-grid sizes produced NaN
        # logits ("!!!!") with this 4-bit build, so snap to the grid and cap the longest side.
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        scale = min(1.0, 896 / max(img.size))
        w = max(28, int(img.width * scale) // 28 * 28)
        h = max(28, int(img.height * scale) // 28 * 28)
        img = img.resize((w, h), Image.LANCZOS)
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(path, quality=92)
    try:
        chat = apply_chat_template(p, m.config, prompt, num_images=1 if path else 0)
        with _lock:   # one generation at a time on the GPU
            out = generate(m, p, chat, image=[path] if path else None, max_tokens=max_tokens)
        return out.text if hasattr(out, "text") else str(out)
    finally:
        if path:
            os.unlink(path)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": True, "model": MODEL_ID, "loaded": _state["model"] is not None})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t = time.time()
            text = _generate(req["prompt"], req.get("image_b64"), int(req.get("max_tokens", 300)))
            self._send(200, {"text": text, "seconds": round(time.time() - t, 2)})
        except Exception as e:
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8011
    threading.Thread(target=_load, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
