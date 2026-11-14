"""World State Manager: observed scene + intervention branch + caches."""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field

import numpy as np

from .perception.pipeline import PerceptionState
from .schemas import Intervention


@dataclass
class Session:
    perception: PerceptionState | None = None
    interventions: list[Intervention] = field(default_factory=list)
    redo_stack: list[Intervention] = field(default_factory=list)
    sim: dict | None = None
    explanations: dict[str, dict] = field(default_factory=dict)
    plates: dict[str, bytes] = field(default_factory=dict)
    clean_plate: np.ndarray | None = None
    ckpt: dict | None = None        # Live AR checkpoint: {"qpos": end state, "ivs": the interventions that rebuild that world}
    lock: threading.RLock = field(default_factory=threading.RLock)

    def sim_key(self) -> str:
        scene = self.perception.scene if self.perception else None
        payload = json.dumps({"scene": scene.model_dump() if scene else None,
                              "iv": [i.model_dump() for i in self.interventions]}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def invalidate(self) -> None:
        self.sim = None
        self.explanations.clear()


SESSION = Session()


def to_jsonable(x):
    """Recursively convert numpy types for JSON responses."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x
