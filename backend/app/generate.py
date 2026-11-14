"""AI text-to-3D for objects outside the library (fal.ai, Hunyuan 3D v3.1 Rapid).

Only the text prompt is sent; the camera frame never leaves the machine.
Requires FAL_KEY. Generation runs as a background job: POST starts it, GET polls it.
The endpoint's documented output is `model_urls` (glb/fbx/obj/mtl/texture/usdz), and
its published example only contains OBJ + MTL + texture, so both are handled: a GLB
is used directly, otherwise the OBJ set is downloaded and converted to GLB locally.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from . import assets as asset_lib

log = logging.getLogger(__name__)
ENDPOINT = "fal-ai/hunyuan-3d/v3.1/rapid/text-to-3d"
OUT = asset_lib.GENERATED_DIR
JOBS: dict[str, dict] = {}


def configured() -> bool:
    return bool(os.environ.get("FAL_KEY"))


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "causal-world-editor/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _call_fal(prompt: str) -> dict:
    import fal_client
    return fal_client.subscribe(ENDPOINT, arguments={"prompt": prompt[:200], "enable_pbr": True}, with_logs=False)


def _url(x) -> str | None:
    if isinstance(x, dict):
        return x.get("url")
    return x if isinstance(x, str) else None


def materialize(result: dict, d: Path) -> Path:
    """Write a model.glb into d from a fal result (GLB if offered, else OBJ+MTL+texture -> GLB)."""
    d.mkdir(parents=True, exist_ok=True)
    urls = result.get("model_urls") or {}
    glb = _url(urls.get("glb")) or (_url(result.get("model_glb")) if str(_url(result.get("model_glb")) or "").endswith(".glb") else None)
    if glb:
        (d / "model.glb").write_bytes(_fetch(glb))
        return d / "model.glb"
    obj = _url(urls.get("obj")) or _url(result.get("model_obj"))
    if not obj:
        raise RuntimeError(f"fal returned no usable model: keys={list(result)}")
    obj_text = _fetch(obj).decode("utf-8", "replace")
    mtl = _url(urls.get("mtl")) or _url(result.get("material_mtl"))
    tex = _url(urls.get("texture")) or _url(result.get("texture"))
    mtl_name = None
    m = re.search(r"^mtllib\s+(.+)$", obj_text, re.M)
    if m:
        mtl_name = m.group(1).strip()
    if mtl:
        mtl_text = _fetch(mtl).decode("utf-8", "replace")
        if tex:
            tm = re.search(r"^\s*map_Kd\s+(.+)$", mtl_text, re.M)
            tex_name = tm.group(1).strip() if tm else "texture.png"
            (d / Path(tex_name).name).write_bytes(_fetch(tex))
            mtl_text = re.sub(r"^(\s*map_Kd\s+).+$", lambda mm: mm.group(1) + Path(tex_name).name, mtl_text, flags=re.M)
        (d / (mtl_name or "model.mtl")).write_text(mtl_text)
    (d / "model.obj").write_text(obj_text)
    import trimesh
    scene = trimesh.load(d / "model.obj", force="scene")
    scene.export(d / "model.glb")
    return d / "model.glb"


def register(slug: str, name: str, prompt: str, size_m: float, mass_kg: float | None) -> dict:
    cfg = {"name": name, "aliases": [name, *name.split()[-1:]], "size": size_m,
           "mass": mass_kg or round(max(0.02, 350 * size_m ** 3 * 0.5), 3), "friction": 0.6,
           "generated": True, "prompt": prompt}
    entry = asset_lib._build_one(slug, cfg, base=OUT, url_prefix="/assets/generated")
    if entry is None:
        raise RuntimeError("generated model could not be loaded")
    entry["license"] = "generated (fal.ai Hunyuan3D)"
    (OUT / slug / "entry.json").write_text(json.dumps(entry))
    asset_lib.invalidate()
    return entry


def _run(job_id: str, name: str, prompt: str, size_m: float, call=None):
    job = JOBS[job_id]
    try:
        job["state"] = "generating"
        result = (call or _call_fal)(prompt)
        slug = "gen_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:30] + "_" + job_id[:4]
        job["state"] = "downloading"
        materialize(result, OUT / slug)
        entry = register(slug, name, prompt, size_m, None)
        job.update(state="done", asset=entry["id"], seconds=round(time.time() - job["started"], 1))
    except Exception as e:     # surfaced to the UI verbatim
        log.exception("generation failed")
        job.update(state="error", error=str(e))


def start(name: str, size_m: float = 0.12, call=None) -> str:
    if call is None and not configured():
        raise RuntimeError("FAL_KEY is not set")
    prompt = (f"A single {name}, realistic, physically plausible proportions, standing upright on a flat surface, "
              f"centered, no base or pedestal")
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"id": job_id, "name": name, "prompt": prompt, "state": "queued", "started": time.time()}
    threading.Thread(target=_run, args=(job_id, name, prompt, size_m, call), daemon=True).start()
    return job_id
