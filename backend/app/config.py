"""Central configuration. Everything tunable lives here so hero-scene tuning
doesn't require hunting through modules."""
from __future__ import annotations

import os
from pathlib import Path

# A few ops (anti-aliased resize, used by SAM 2's video tracker) have no Apple-GPU kernel yet; let just those run
# on the CPU instead of failing. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SCENES_DIR = ROOT / "scenes"
DATA_DIR.mkdir(exist_ok=True)

DEPTH_MODEL_ID = os.environ.get("CWE_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
SAM_MODEL_ID = os.environ.get("CWE_SAM_MODEL", "facebook/sam2.1-hiera-tiny")

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Claude (optional). Without credentials the app uses deterministic fallbacks
# that are derived from the same physics data, never canned text.
CLAUDE_MODEL = os.environ.get("CWE_CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_ENABLED = os.environ.get("CWE_DISABLE_CLAUDE", "0") != "1"

# Perception
MAX_PROCESS_WIDTH = 960          # frames are downscaled to this width before perception
DEFAULT_HFOV_DEG = 62.0          # typical laptop webcam; user-adjustable in the UI
MIN_OBJECT_HEIGHT = 0.012        # m above the table plane to count as "object"
MIN_MASK_FRAC = 0.0012           # of image area
TABLE_THICKNESS = 0.04
TABLE_HEIGHT = 0.75              # table top above floor, for fall-off-table events

# Physics
SIM_TIMESTEP = 0.002
RECORD_FPS = 60
MAX_SEGMENT_SECONDS = 6.0
SETTLE_SECONDS = 0.8             # baseline settle before any intervention
REST_LIN_SPEED = 0.012           # m/s
REST_ANG_SPEED = 0.08            # rad/s
REST_HOLD_SECONDS = 0.35
