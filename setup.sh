#!/bin/bash
# One-time setup: Python venv, model weights (~250 MB), frontend build.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3.12}
[ -d .venv ] || $PY -m venv --system-site-packages .venv
.venv/bin/pip install --no-cache-dir -q -r requirements.txt
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "facebook/sam2.1-hiera-tiny"]:
    snapshot_download(r, allow_patterns=["*.json", "*.safetensors", "*.txt"])
print("models ready")
PY
# local vision-language model worker (Qwen2.5-VL-3B via MLX) in its own environment
[ -d .venv-ai ] || $PY -m venv .venv-ai
.venv-ai/bin/pip install --no-cache-dir -q mlx-vlm
(cd frontend && npm install --no-audit --no-fund && npm run build)
.venv/bin/python -m backend.app.synth
echo "Setup done. Start with ./run.sh"
