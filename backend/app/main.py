"""FastAPI server for the Live Causal World Editor.

Run:  .venv/bin/uvicorn backend.app.main:app --port 8000   (from the repo root)
"""
from __future__ import annotations

import io
import math
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import assets as asset_lib
from . import compositor, config, planner, synth
from .explain import explain
from .perception import models, pipeline
from .physics.sim import simulate
from .schemas import Intervention, PatchObject
from .state import SESSION, to_jsonable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cwe")
app = FastAPI(title="Live Causal World Editor")

_warm = {"state": "cold", "timings": {}}


def _warmup():
    _warm["state"] = "loading"
    try:
        _warm["timings"] = models.warmup()
        _warm["state"] = "ready"
        # voice (Whisper) loads on the first transcription instead: saves ~300 MB until someone speaks
    except Exception as e:   # surfaced in /api/health
        log.exception("warmup failed")
        _warm["state"] = f"error: {e}"


@app.on_event("startup")
def _startup():
    from . import local_ai
    local_ai.start()                     # free on-device language/vision model (separate process)
    threading.Thread(target=_warmup, daemon=True).start()


@app.on_event("shutdown")
def _shutdown():
    from . import local_ai, remote
    remote.SHUTTING_DOWN.set()
    if local_ai._proc is not None:
        local_ai._proc.terminate()


def claude_configured() -> bool:
    from . import llm
    return llm.configured()


def _need_scene():
    st = SESSION.perception
    if st is None or st.scene is None:
        raise HTTPException(409, "no scene yet: scan a frame first")
    return st


def _scene_payload():
    st = SESSION.perception
    return to_jsonable({
        "scene": st.scene.model_dump() if st and st.scene else None,
        "interventions": [i.model_dump() for i in SESSION.interventions],
        "can_redo": bool(SESSION.redo_stack),
        "depth_scale": st.depth_scale if st else 1.0,
        "has_clean_plate": SESSION.clean_plate is not None,
    })


# ---------------------------------------------------------------- health / sources
@app.get("/api/health")
def health():
    from . import generate
    return {"models": _warm["state"], "warmup_ms": _warm["timings"], "device": config.DEVICE,
            "claude": claude_configured(), "claude_model": config.CLAUDE_MODEL, "fal": generate.configured(),
            "local_ai": local_ai_status()}


def local_ai_status() -> str:
    from . import local_ai
    st = local_ai.status()
    return "ready" if st.get("loaded") else ("loading" if st.get("ok") else "off")


@app.get("/api/synthetic")
def list_synthetic():
    return [{"name": k, "title": v.title, "notes": v.notes} for k, v in synth.SCENES.items()]


@app.get("/api/synthetic/{name}")
def synthetic_image(name: str):
    if name not in synth.SCENES:
        raise HTTPException(404, "unknown scene")
    path = config.DATA_DIR / "synthetic" / name / "rgb.png"
    meta = config.DATA_DIR / "synthetic" / name / "meta.json"
    if not path.exists() or not meta.exists():
        synth.render(name)
    import json
    m = json.loads(meta.read_text())
    return FileResponse(path, headers={"X-HFOV": str(m["hfov_deg"]), "Access-Control-Expose-Headers": "X-HFOV"})


@app.get("/api/synthetic/{name}/depth.png")
def synthetic_depth(name: str):
    """Renderer depth buffer as a 16-bit mm PNG: plays the role of an RGB-D sensor."""
    if name not in synth.SCENES:
        raise HTTPException(404, "unknown scene")
    path = config.DATA_DIR / "synthetic" / name / "depth_gt.npy"
    if not path.exists():
        synth.render(name)
    d = np.load(path)
    mm = np.clip(np.nan_to_num(d) * 1000, 0, 65535).astype(np.uint16)
    mm[d > 10] = 0
    ok, buf = cv2.imencode(".png", mm)
    return Response(buf.tobytes(), media_type="image/png")


# ---------------------------------------------------------------- perception
@app.post("/api/scan")
async def scan(image: UploadFile = File(...), hfov: float = Form(config.DEFAULT_HFOV_DEG),
               source: str = Form("camera"), use_claude: bool = Form(True),
               depth: Optional[UploadFile] = File(None)):
    """depth (optional): 16-bit PNG in millimetres, aligned with the image
    (RGB-D cameras, phone LiDAR exports). Without it, monocular depth is used."""
    if _warm["state"] not in ("ready",):
        # blocks until models are loaded on first use
        models._load_depth(); models._load_sam()
    rgb = pipeline.load_frame(await image.read())
    depth_m = None
    if depth is not None:
        raw = cv2.imdecode(np.frombuffer(await depth.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2:
            raise HTTPException(422, "depth must be a single-channel 16-bit PNG in millimetres")
        depth_m = cv2.resize(raw.astype(np.float32) / 1000.0, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_m[depth_m <= 0] = np.nan
        source = f"{source}+depth"
    t = time.perf_counter()
    try:
        st = pipeline.scan(rgb, hfov, source, use_claude=use_claude, depth_override=depth_m)
    except ValueError as e:
        raise HTTPException(422, str(e))
    with SESSION.lock:
        SESSION.perception = st
        SESSION.interventions.clear()
        SESSION.redo_stack.clear()
        SESSION.plates.clear()
        SESSION.invalidate()
        if SESSION.clean_plate is not None and SESSION.clean_plate.shape != rgb.shape:
            SESSION.clean_plate = None
    log.info("scan %.0f ms, %d objects", (time.perf_counter() - t) * 1000, len(st.scene.objects))
    return _scene_payload()


# ---------------------------------------------------------------- sweep scan (multi-view meshes)
SWEEPS: dict[str, dict] = {}
_sweep_lock = threading.Lock()


def _install_scene(st) -> None:
    with SESSION.lock:
        SESSION.perception = st
        SESSION.interventions.clear()
        SESSION.redo_stack.clear()
        SESSION.plates.clear()
        SESSION.invalidate()
        if SESSION.clean_plate is not None and SESSION.clean_plate.shape != st.rgb.shape:
            SESSION.clean_plate = None


def _start_sweep(frames_fn, hfov: float, source: str) -> dict:
    """frames_fn() -> list of RGB frames (decoding a video happens in the job, not the request)."""
    import uuid as _uuid
    from .perception import sweep as _sweep
    if not _sweep_lock.acquire(blocking=False):
        raise HTTPException(409, "a sweep scan is already running")
    jid = _uuid.uuid4().hex[:10]
    job = {"id": jid, "state": "running", "stage": "reading", "frac": 0.0, "message": "reading the video", "log": []}
    SWEEPS[jid] = job

    def prog(stage, frac, msg=""):
        job.update(stage=stage, frac=round(float(frac), 3))
        if msg:
            job["message"] = msg
            job["log"].append(f"{stage}: {msg}")

    def run():
        try:
            if _warm["state"] != "ready":
                models._load_depth(); models._load_sam()
            frames = frames_fn()
            job["message"] = f"{len(frames)} frames"
            res = _sweep.run(frames, hfov, source, progress=prog)
            _install_scene(res.st)
            job.update(state="done", stage="done", frac=1.0, scene_id=res.st.scene_id, keyframes=res.keyframes,
                       pose_note=res.pose_note, timings_s={k: round(v, 1) for k, v in res.timings_s.items()},
                       objects=len(res.st.scene.objects), meshes=sum(o.shape == "mesh" for o in res.st.scene.objects))
        except Exception as e:                      # never leave the page waiting
            log.exception("sweep failed")
            job.update(state="error", error=str(e))
        finally:
            _sweep_lock.release()
    threading.Thread(target=run, daemon=True).start()
    return job


@app.post("/api/sweep")
async def sweep_video(video: UploadFile = File(...), hfov: float = Form(config.DEFAULT_HFOV_DEG)):
    """A 10-20 s video moving slowly around the table, ending where the camera will stay (that is the live view)."""
    import tempfile
    suffix = Path(video.filename or "v.mp4").suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=config.DATA_DIR)
    tmp.write(await video.read())
    tmp.close()

    def frames():
        from .perception import sweep as _sweep
        try:
            return _sweep.read_video(tmp.name)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return _start_sweep(frames, hfov, "sweep:video")


@app.get("/api/sweep_latest")
def sweep_latest():
    """The newest sweep job (the laptop shows progress for scans started from the phone)."""
    if not SWEEPS:
        return {"state": "none"}
    return list(SWEEPS.values())[-1]


@app.get("/api/sweep/{jid}")
def sweep_status(jid: str):
    j = SWEEPS.get(jid)
    if not j:
        raise HTTPException(404, "no such sweep")
    return j


@app.get("/api/mesh/{scene_id}/{name}")
def mesh_file(scene_id: str, name: str):
    from .perception.sweep import MESH_DIR
    if not re.fullmatch(r"[0-9a-f]{6,16}", scene_id) or not re.fullmatch(r"[A-Za-z0-9_]{1,24}\.glb", name):
        raise HTTPException(404, "not found")
    p = MESH_DIR / scene_id / name
    if not p.exists():
        raise HTTPException(404, "not found")
    return Response(p.read_bytes(), media_type="model/gltf-binary", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/scene")
def get_scene():
    return _scene_payload()


@app.get("/api/frame.jpg")
def frame():
    st = _need_scene()
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(st.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/idmap.png")
def idmap():
    st = _need_scene()
    ok, buf = cv2.imencode(".png", pipeline.id_map(st))
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.get("/api/depth.png")
def depth_png():
    st = _need_scene()
    h = st.height.copy()
    h[~np.isfinite(h)] = 0
    vis = cv2.applyColorMap(np.clip(h / max(0.05, np.nanpercentile(h, 99)) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ok, buf = cv2.imencode(".png", vis)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


def _edited():
    SESSION.interventions.clear()
    SESSION.redo_stack.clear()
    SESSION.plates.clear()
    SESSION.invalidate()


class Click(BaseModel):
    x: float
    y: float


@app.post("/api/objects/click")
def click_object(c: Click):
    st = _need_scene()
    with SESSION.lock:
        oid = pipeline.add_by_click(st, c.x, c.y)
        if oid is None:
            raise HTTPException(422, "nothing segmentable at that point")
        _edited()
    return {"id": oid, **_scene_payload()}


class Merge(BaseModel):
    ids: list[str]


@app.post("/api/objects/merge")
def merge_objects(m: Merge):
    st = _need_scene()
    with SESSION.lock:
        recs = [r for r in st.records if r.id in m.ids]
        if len(recs) < 2:
            raise HTTPException(422, "select at least two objects")
        keep = max(recs, key=lambda r: r.mask.sum())
        for r in recs:
            if r is not keep:
                keep.mask = keep.mask | r.mask
                st.records.remove(r)
        keep.shape = None
        pipeline.rebuild(st)
        _edited()
    return {"id": keep.id, **_scene_payload()}


@app.delete("/api/objects/{oid}")
def discard_object(oid: str):
    """Discard a detection (perception edit, not a causal intervention)."""
    st = _need_scene()
    with SESSION.lock:
        st.records = [r for r in st.records if r.id != oid]
        pipeline.rebuild(st)
        _edited()
    return _scene_payload()


@app.patch("/api/objects/{oid}")
def patch_object(oid: str, p: PatchObject):
    st = _need_scene()
    with SESSION.lock:
        r = next((r for r in st.records if r.id == oid), None)
        if r is None:
            raise HTTPException(404, "unknown object")
        if p.label is not None:
            r.label, r.label_source = p.label, "user"
        if p.material is not None:
            r.material = p.material
            r.friction_override = None
            if r.mass_override is None:
                r.mass_claude = None   # recompute from volume x density of the new material
        if p.shape is not None:
            r.shape = p.shape
        if p.mass is not None:
            r.mass_override = max(0.005, p.mass)
        if p.friction is not None:
            r.friction_override = float(np.clip(p.friction, 0.0, 2.0))
        if p.static is not None:
            r.static = p.static
        pipeline.rebuild(st)
        _edited()
    return _scene_payload()


class Calib(BaseModel):
    object_id: str
    dimension: Literal["height", "width", "depth"] = "height"
    value_m: float


@app.post("/api/calibrate")
def calibrate(c: Calib):
    st = _need_scene()
    if not 0.005 < c.value_m < 3:
        raise HTTPException(422, "value out of range")
    with SESSION.lock:
        pipeline.calibrate(st, c.object_id, c.dimension, c.value_m)
        _edited()
    return _scene_payload()


class Fov(BaseModel):
    hfov: float


@app.post("/api/hfov")
def set_hfov(f: Fov):
    st = _need_scene()
    with SESSION.lock:
        st.hfov_deg = float(np.clip(f.hfov, 25, 120))
        pipeline.rebuild(st)
        _edited()
    return _scene_payload()


@app.post("/api/clean_plate")
async def clean_plate(image: UploadFile = File(...)):
    rgb = pipeline.load_frame(await image.read())
    with SESSION.lock:
        SESSION.clean_plate = rgb
        SESSION.plates.clear()
    return {"ok": True, "shape": list(rgb.shape)}


@app.delete("/api/clean_plate")
def drop_clean_plate():
    SESSION.clean_plate = None
    SESSION.plates.clear()
    return {"ok": True}


# ---------------------------------------------------------------- interventions
class NewIntervention(BaseModel):
    type: Literal["remove", "push", "set_mass", "set_friction", "set_static", "set_gravity", "spawn", "drive", "launch", "place"]
    target: Optional[str] = None
    params: dict = {}


def _resolve_push(st, iv: NewIntervention) -> dict:
    """Pixel drag -> 3D application point, horizontal direction and speed."""
    p = dict(iv.params)
    if "pixel_from" not in p:
        return p
    cam = st.scene.camera
    R = np.array(cam.R_wc)
    t = np.array(cam.t_wc)
    u0, v0 = p.pop("pixel_from")
    u1, v1 = p.pop("pixel_to")
    obj = next(o for o in st.scene.objects if o.id == iv.target)
    pw = st.pts_world[int(np.clip(v0, 0, cam.height - 1)), int(np.clip(u0, 0, cam.width - 1))]
    center = np.array(obj.pos)
    half = np.array(obj.size) / 2
    if not np.all(np.isfinite(pw)) or np.any(np.abs(pw - center) > half.max() * 1.5 + 0.02):
        pw = center
    pw = np.clip(pw, center - half, center + half)

    def ray_to_plane(u, v, z0):
        d = R @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
        if abs(d[2]) < 1e-6:
            return None
        s = (z0 - t[2]) / d[2]
        return t + s * d if s > 0 else None

    a = ray_to_plane(u0, v0, pw[2])
    b = ray_to_plane(u1, v1, pw[2])
    if a is None or b is None:
        raise HTTPException(422, "could not project the drag onto the table")
    delta = b - a
    dist = float(np.linalg.norm(delta[:2]))
    if dist < 1e-3:
        raise HTTPException(422, "drag too short")
    p["direction"] = [float(delta[0]), float(delta[1]), 0.0]
    p["point"] = [float(v) for v in pw]
    p.setdefault("speed", float(np.clip(dist * 5.0, 0.15, 3.0)))
    p["drag_m"] = round(dist, 4)
    return p


def _ckpt_valid() -> bool:
    c = SESSION.ckpt
    return bool(c) and len(SESSION.interventions) >= len(c["ivs"]) and all(a is b for a, b in zip(SESSION.interventions, c["ivs"]))


def _checkpoint() -> None:
    """Live AR: make the current end state the new starting point, keeping only what persists (added objects, removals,
    mass/friction/static changes, gravity, links, held objects), so the next action simulates only itself instead of
    replaying the whole history - same physics, a fraction of the time and payload."""
    sim = SESSION.sim
    if not sim or "final_qpos" not in sim or not SESSION.interventions:
        return
    if _ckpt_valid() and len(SESSION.interventions) == len(SESSION.ckpt["ivs"]):
        return                                               # nothing new since the last checkpoint
    if SESSION.ckpt and not _ckpt_valid():
        SESSION.ckpt = None
    g = "restore"
    keep: list[Intervention] = []
    last: dict[tuple[str, str | None], Intervention] = {}
    for iv in SESSION.interventions:
        if iv.type == "spawn":
            keep.append(Intervention(type="spawn", target=iv.target, params={**iv.params, "keep": True, "group": g}))
        elif iv.type == "connect":
            keep.append(Intervention(type="connect", target=iv.target, params={**iv.params, "group": g}))
        elif iv.type in ("set_mass", "set_friction", "set_static", "set_gravity"):
            last[(iv.type, iv.target)] = iv
    for oid in sim.get("removed_at", {}):
        keep.append(Intervention(type="remove", target=oid, params={"group": g}))
    for iv in last.values():
        keep.append(Intervention(type=iv.type, target=iv.target, params={**iv.params, "group": g}))
    for oid in sim.get("final_pinned", []):
        if oid not in sim.get("removed_at", {}):
            keep.append(Intervention(type="hold", target=oid, params={"group": g}))
    if not keep:                                             # nothing persists: an empty restore step keeps the end state
        keep.append(Intervention(type="set_gravity", target=None, params={"g": float(sim.get("gravity", 9.81)), "group": g}))
    SESSION.interventions[:] = keep
    SESSION.redo_stack.clear()
    SESSION.ckpt = {"qpos": list(sim["final_qpos"]), "ivs": list(keep)}


def _run_sim():
    st = _need_scene()
    t = time.perf_counter()
    base = st.baseline_qpos
    if SESSION.ckpt is not None:
        if _ckpt_valid():
            base = np.array(SESSION.ckpt["qpos"])
        else:
            SESSION.ckpt = None
    sim = simulate(st.scene, base, SESSION.interventions)
    sim["key"] = SESSION.sim_key()
    sim["compute_ms"] = round((time.perf_counter() - t) * 1000, 1)
    SESSION.sim = to_jsonable(sim)
    return SESSION.sim


@app.post("/api/interventions")
def add_intervention(iv: NewIntervention):
    st = _need_scene()
    ids = {o.id for o in st.scene.objects} | {i.target for i in SESSION.interventions if i.type == "spawn"}
    if iv.type not in ("set_gravity", "spawn") and iv.target not in ids:
        raise HTTPException(422, "unknown target")
    removed = {i.target for i in SESSION.interventions if i.type == "remove"}
    if iv.target in removed:
        raise HTTPException(422, "that object was already removed")
    with SESSION.lock:
        params = _resolve_push(st, iv) if iv.type == "push" else dict(iv.params)
        if iv.type == "drive" and "pixel_goal" in params:
            g = _pixel_to_table(st, *params.pop("pixel_goal"))
            if g is None:
                raise HTTPException(422, "that point is not on the table")
            params["goal"] = [float(g[0]), float(g[1])]
            ents = {e.id: e for e in planner.snapshot(st.scene, SESSION.sim, SESSION.interventions)}
            here = ents[iv.target].pos[:2] if iv.target in ents else np.zeros(2)
            params.setdefault("speed", 0.6)
            params.setdefault("duration", round(min(6.0, float(np.linalg.norm(np.array(g[:2]) - here)) / params["speed"] + 0.6), 2))
        SESSION.interventions.append(Intervention(type=iv.type, target=iv.target, params=params))
        SESSION.redo_stack.clear()
        SESSION.invalidate()
        sim = _run_sim()
    return {"sim": sim, **_scene_payload()}


def _pixel_to_table(st, u: float, v: float, z0: float = 0.0):
    cam = st.scene.camera
    R, t = np.array(cam.R_wc), np.array(cam.t_wc)
    d = R @ np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    if abs(d[2]) < 1e-6:
        return None
    s_ = (z0 - t[2]) / d[2]
    if s_ <= 0:
        return None
    p = t + s_ * d
    tb = st.scene.table
    if not (tb.x_min < p[0] < tb.x_max and tb.y_min < p[1] < tb.y_max):
        return None
    return p


# ---------------------------------------------------------------- summoned objects & experiments
@app.get("/api/assets")
def list_assets():
    return [{k: v for k, v in a.items() if k != "hull"} for a in asset_lib.manifest().values()]


class SpawnAt(BaseModel):
    asset: str
    pixel: Optional[list[float]] = None        # None: a free, visible spot on the table, near the middle of the view
    checkpoint: bool = False                   # Live AR: continue from the current end state


@app.post("/api/spawn")
def spawn_at(req: SpawnAt):
    """Place a library object where the user clicked (on the table or on top of an object)."""
    st = _need_scene()
    lib = asset_lib.manifest()
    if req.asset not in lib:
        raise HTTPException(404, "unknown asset")
    a = lib[req.asset]
    ents = planner.snapshot(st.scene, SESSION.sim, SESSION.interventions)
    if req.pixel is None:
        W, H = st.scene.camera.width, st.scene.camera.height
        ra = 0.5 * math.hypot(a["dims"][0], a["dims"][1])
        best, score = None, -1e18
        for v_ in np.linspace(0.45 * H, 0.9 * H, 12):
            for u_ in np.linspace(0.15 * W, 0.85 * W, 16):
                pw_ = st.pts_world[int(v_), int(u_)]
                if not np.all(np.isfinite(pw_)) or pw_[2] > 0.012:
                    continue                                  # not bare table
                gap = min((float(np.linalg.norm(e.pos[:2] - pw_[:2])) - e.radius - ra for e in ents), default=1.0)
                if gap < 0.04:
                    continue
                sc_ = -abs(u_ - 0.5 * W) / W - abs(v_ - 0.7 * H) / H + min(gap, 0.15)
                if sc_ > score:
                    best, score = (float(u_), float(v_)), sc_
        if best is None:
            raise HTTPException(422, "no free spot on the table in view — move something or rescan")
        req.pixel = list(best)
    u, v = req.pixel
    p = _pixel_to_table(st, u, v)
    z = 0.0008
    pw = st.pts_world[int(np.clip(v, 0, st.scene.camera.height - 1)), int(np.clip(u, 0, st.scene.camera.width - 1))]
    if np.all(np.isfinite(pw)) and pw[2] > 0.015:          # clicked on an object: rest on top of it
        under = min(ents, key=lambda e: np.linalg.norm(e.pos[:2] - pw[:2]))
        if np.linalg.norm(under.pos[:2] - pw[:2]) < under.radius:
            p, z = np.array([pw[0], pw[1], 0.0]), under.top + 0.001
    if p is None:
        raise HTTPException(422, "click on the table to place an object")
    right = np.array(st.scene.camera.R_wc)[:, 0]
    yaw = float(np.arctan2(right[1], right[0]))            # side-on to the camera
    with SESSION.lock:
        if req.checkpoint:
            _checkpoint()
        taken = {i.target for i in SESSION.interventions if i.type == "spawn"}
        n = 1
        while f"s{n}" in taken:
            n += 1
        SESSION.interventions.append(Intervention(type="spawn", target=f"s{n}", params={
            "asset": a["id"], "pos": [round(float(p[0]), 4), round(float(p[1]), 4), round(float(z), 4)],
            "yaw": round(yaw, 4), "label": a["name"], "static": bool(a.get("static"))}))
        SESSION.redo_stack.clear()
        SESSION.invalidate()
        sim = _run_sim()
    return {"sim": sim, **_scene_payload()}


class PlanReq(BaseModel):
    text: str
    selected: Optional[str] = None
    use_claude: bool = True


@app.post("/api/plan")
def make_plan(r: PlanReq):
    """Understand + ground a request. Nothing runs until /api/plan/apply."""
    st = _need_scene()
    ents = planner.snapshot(st.scene, SESSION.sim, SESSION.interventions)
    plan = planner.plan_with_claude(r.text, ents, r.selected) if r.use_claude else None
    if plan is None:
        # the on-device model understands free-form phrasing; the rule parser is the fallback
        planner._CAMERA = st.scene.camera
        rules = planner.parse_rules(r.text, ents, r.selected)
        # deterministic rules when they found a definite action; the on-device model for everything else
        plan = rules if planner.strong(rules) else (planner.plan_with_local_ai(r.text, ents, r.selected, st.scene) or rules)
    n = sum(1 for i in SESSION.interventions if i.type == "spawn")
    yaws = {}
    if SESSION.sim:
        for sid in SESSION.sim.get("spawned", {}):
            f = SESSION.sim["frames"][-1][SESSION.sim["object_ids"].index(sid)]
            qx, qy, qz, qw = f[3:7]
            yaws[sid] = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    try:
        ivs, human, notes = planner.resolve(plan, st.scene, ents, n, prior=[i.model_dump() for i in SESSION.interventions], actor_yaw=yaws)
    except ValueError as e:
        raise HTTPException(422, str(e))
    from . import generate
    return {"planner": plan["planner"], "steps": human, "notes": notes, "interventions": ivs, "request": r.text,
            "generate": plan.get("generate"), "can_generate": generate.configured()}


class ApplyPlan(BaseModel):
    interventions: list[NewIntervention]
    checkpoint: bool = False      # Live AR: continue from the current end state


@app.post("/api/plan/apply")
def apply_plan(req: ApplyPlan):
    _need_scene()
    if not req.interventions:
        raise HTTPException(422, "empty plan")
    with SESSION.lock:
        if req.checkpoint:
            _checkpoint()
        for iv in req.interventions:
            SESSION.interventions.append(Intervention(type=iv.type, target=iv.target, params=dict(iv.params)))
        SESSION.redo_stack.clear()
        SESSION.invalidate()
        try:
            sim = _run_sim()
        except Exception:
            del SESSION.interventions[-len(req.interventions):]
            raise
    return {"sim": sim, **_scene_payload()}


class GenReq(BaseModel):
    name: str
    size_m: float = 0.12


@app.post("/api/generate")
def generate_asset(r: GenReq):
    """Text-to-3D for an object outside the library. Only the text is sent to fal.ai."""
    from . import generate
    if not generate.configured():
        raise HTTPException(409, "AI generation needs FAL_KEY (fal.ai); set it and restart the backend")
    name = r.name.strip()[:60]
    if not name:
        raise HTTPException(422, "empty name")
    return {"job": generate.start(name, float(np.clip(r.size_m, 0.03, 0.5)))}


@app.get("/api/generate/{job_id}")
def generate_status(job_id: str):
    from . import generate
    job = generate.JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k != "started"} | {"elapsed": round(time.time() - job["started"], 1)}


# ---------------------------------------------------------------- experiments (anything you can say)
class ExpReq(BaseModel):
    text: str
    where: Literal["auto", "scene", "sandbox"] = "auto"
    measures: list[str] = []
    follow_up: bool = True


def _scene_and_base(where: str):
    st = SESSION.perception
    if where == "sandbox" or st is None or st.scene is None:
        return None, None
    return st.scene, st.baseline_qpos


def _adopt(pub: dict) -> dict:
    """A run in the real scanned world becomes the AR session's simulation (overlay, undo, explain)."""
    if pub["setting"] != "scene" or SESSION.perception is None:
        return {}
    with SESSION.lock:
        SESSION.interventions[:] = [Intervention(**iv) for iv in pub["interventions"]]
        SESSION.redo_stack.clear()
        SESSION.invalidate()
        key = SESSION.sim_key()
        pub["sim"]["key"] = key
        SESSION.sim = pub["sim"]
    return _scene_payload()


@app.post("/api/experiment")
def run_experiment(r: ExpReq):
    """Words -> world -> physics -> measurements -> why. Follow-ups ("what if it were twice as heavy",
    "on the Moon") modify the current experiment; questions are answered from its measurements."""
    from . import experiment as ex, lab
    text = r.text.strip()
    if not text:
        raise HTTPException(422, "say what should happen")
    if lab.current() is not None and lab.QUESTION.search(text) and not ex.is_whatif(text):
        return {"kind": "answer", **lab.answer(text)}
    scene, base = _scene_and_base(r.where)
    from . import scene_cmd
    scene_cmd.LAST["id"] = None                          # "it" now means this experiment's thing
    t0 = time.perf_counter()
    try:
        pub = lab.run_text(text, scene, base, r.measures, r.follow_up, force_scene=r.where == "scene" and scene is not None)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if pub.get("domain") == "chem":
        return {"kind": "chem", "text": text}
    if pub.get("domain") == "ml":
        return {"kind": "ml", "text": text}
    if pub.get("domain") == "unsupported":
        return {"kind": "unsupported", "text": text, "what": pub["what"],
                "message": f"I can't simulate {pub['what']} yet — and I won't fake it with a stand-in. What I can run: anything that moves, falls, "
                           f"collides, swims, flies, swings or bounces (mechanics, fluids, buoyancy); electric circuits; gravity, orbits and planet impacts; "
                           f"and chemical reactions (Chemistry tab)."}
    pub["compute_ms"] = round((time.perf_counter() - t0) * 1000)
    payload = _adopt(pub) if pub.get("domain") == "mechanics" else {}
    return {"kind": "run", "run": pub, "runs": lab.summary(), **({"scene_payload": payload} if payload else {})}


class RerunReq(BaseModel):
    experiment: dict
    measures: list[str] = []
    parent: Optional[str] = None
    where: Literal["auto", "scene", "sandbox"] = "auto"


@app.post("/api/experiment/rerun")
def rerun_experiment(r: RerunReq):
    from . import lab
    scene, base = _scene_and_base(r.where)
    t0 = time.perf_counter()
    try:
        pub = lab.rerun(r.experiment, scene, base, r.measures, r.parent)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))
    pub["compute_ms"] = round((time.perf_counter() - t0) * 1000)
    payload = _adopt(pub)
    return {"kind": "run", "run": pub, "runs": lab.summary(), **({"scene_payload": payload} if payload else {})}


@app.get("/api/experiment/runs")
def experiment_runs():
    from . import lab
    cur = lab.current()
    return {"runs": lab.summary(), "current": cur["public"]["id"] if cur else None}


@app.get("/api/experiment/current")
def experiment_current():
    from . import lab
    cur = lab.current()
    return {"run": cur["public"] if cur else None, "live": __import__("backend.app.live", fromlist=["status"]).status()}


@app.get("/api/experiment/{run_id}")
def experiment_get(run_id: str):
    from . import lab
    pub = lab.get(run_id)
    if pub is None:
        raise HTTPException(404, "unknown run")
    return {"run": pub, "scene_payload": (_adopt(pub) or None) if pub.get("domain", "mechanics") == "mechanics" else None}


class AskExp(BaseModel):
    question: str
    run_id: Optional[str] = None


@app.post("/api/experiment/ask")
def experiment_ask(r: AskExp):
    from . import lab
    return lab.answer(r.question, r.run_id)


# ---------------------------------------------------------------- diagrams & the transformer lab
@app.post("/api/diagram")
async def read_diagram(image: UploadFile = File(...)):
    """Photo of a diagram -> what kind it is + an editable sentence the labs can run."""
    from . import diagram
    data = await image.read()
    if len(data) > 15_000_000:
        raise HTTPException(413, "image too large")
    try:
        return diagram.read(data)
    except ValueError as e:
        raise HTTPException(422, str(e))


class TransformerReq(BaseModel):
    text: str = "The Eiffel Tower is located in the city of"
    ablate: list[list[int]] = []
    mask: list[int] = []
    temperature: float = 1.0


@app.post("/api/transformer")
def transformer_run(r: TransformerReq):
    from . import transformer_lab
    t0 = time.perf_counter()
    out = transformer_lab.run(r.text, r.ablate, r.mask, float(min(5.0, max(0.05, r.temperature))))
    out["compute_ms"] = round((time.perf_counter() - t0) * 1000)
    return to_jsonable(out)


# ---------------------------------------------------------------- live sync (phone keeps the 3D world current)
def _live_adopt(st):
    """Called by the live rescan with the new perception state."""
    from . import lab
    old = SESSION.perception.scene if SESSION.perception and SESSION.perception.scene else None
    old_labels = [o.label for o in old.objects] if old else []
    new_labels = [o.label for o in st.scene.objects]
    added = [l for l in new_labels if old_labels.count(l) < new_labels.count(l)]
    removed = [l for l in old_labels if new_labels.count(l) < old_labels.count(l)]
    with SESSION.lock:
        SESSION.perception = st
        SESSION.interventions.clear()
        SESSION.redo_stack.clear()
        SESSION.plates.clear()
        SESSION.invalidate()
    rerun = None
    cur = lab.current()
    if cur is not None and cur["public"]["setting"] == "scene":
        try:
            pub = lab.rerun(cur["public"]["experiment"], st.scene, st.baseline_qpos, [], cur["public"]["id"])
            pub["changes"] = ["the real scene changed: re-ran in the updated world"]
            _adopt(pub)
            rerun = pub["id"]
        except Exception as e:      # the objects the experiment used may be gone
            log.warning("live re-run failed: %s", e)
    return sorted(set(added)), sorted(set(removed)), rerun


class LiveReq(BaseModel):
    hfov: float = 66.0
    enabled: Optional[bool] = None


@app.post("/api/live/rescan")
def live_rescan(r: LiveReq):
    from . import live
    if r.enabled is not None:
        live.STATE["enabled"] = r.enabled
        if not r.enabled:
            return live.status()
    st = SESSION.perception
    if st is not None and st.scene is not None and st.scene.source.startswith("sweep"):
        # a 3D sweep scan is far better than any one-photo rescan: never replace it automatically
        return live.status() | {"kept": "sweep"}
    return live.request_rescan(r.hfov, _live_adopt)


class LiveCmd(BaseModel):
    text: str
    selected: Optional[str] = None


@app.post("/api/live/command")
def live_command(r: LiveCmd):
    """Live AR: a sentence that acts on the mapped objects ("push the mug", "throw the box at the bottle", "remove the book")
    runs on the current world, continuing what is on screen. {"handled": false} when it isn't one (the experiment engine
    takes it then)."""
    from . import scene_cmd
    st = _need_scene()
    with SESSION.lock:
        cmd = scene_cmd.parse(r.text, st.scene, SESSION.sim, SESSION.interventions, r.selected)
        if cmd is not None:
            _checkpoint()
    if cmd is None:
        return {"handled": False}
    with SESSION.lock:
        n0 = len(SESSION.interventions)
        for iv in cmd["ivs"]:
            SESSION.interventions.append(Intervention(type=iv["type"], target=iv["target"], params=dict(iv["params"])))
        SESSION.redo_stack.clear()
        SESSION.invalidate()
        try:
            sim = _run_sim()
        except Exception:
            del SESSION.interventions[n0:]
            raise
    return {"handled": True, "say": cmd["say"], "target": cmd["target"], "start": cmd["start"], "label": cmd["label"], "sim": sim, **_scene_payload()}


class LiveEnabled(BaseModel):
    enabled: bool


@app.post("/api/live/enabled")
def live_enabled(r: LiveEnabled):
    """Pause/resume the phone's automatic rescans (paused while a hand is in view: that's not a changed scene)."""
    from . import live
    with live._lock:
        live.STATE["enabled"] = r.enabled
    return live.status()


@app.get("/api/live/status")
def live_status():
    from . import live, remote
    return live.status() | {"phone": remote.status()}


class ChemReq(BaseModel):
    text: str


@app.post("/api/chem/analyze")
def chem_analyze(r: ChemReq):
    """Balanced equation, xTB-optimised 3D molecules, reaction energies. Local, a few seconds."""
    from . import chem
    try:
        info = chem.analyse(r.text)
    except ValueError as e:
        raise HTTPException(422, str(e))
    info.pop("_objects", None)
    return to_jsonable(info)


@app.post("/api/chem/thermo")
def chem_thermo(r: ChemReq):
    """Quantum-chemistry heat of reaction (M06-2X DFT + vibrations), computed in the background; cached per molecule."""
    from . import thermo_dft
    try:
        return thermo_dft.start(r.text)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/chem/thermo/{jid}")
def chem_thermo_job(jid: str):
    from . import thermo_dft
    j = thermo_dft.JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    return j


@app.post("/api/chem/simulate")
def chem_simulate(r: ChemReq):
    from . import chem
    try:
        job = chem.start_job(r.text)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"job": job["id"]}


@app.get("/api/chem/job/{jid}")
def chem_job(jid: str):
    from . import chem
    j = chem.JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "unknown job")
    return to_jsonable({k: v for k, v in j.items() if k != "started"} | {"elapsed": round(time.time() - j["started"], 1)})


class AskReq(BaseModel):
    question: str
    selected: Optional[str] = None


@app.post("/api/ask")
def ask(r: AskReq):
    """Quantitative questions answered from the simulation state (never estimated by an LLM)."""
    from . import measure
    st = _need_scene()
    return to_jsonable(measure.answer(r.question, st.scene, SESSION.sim, SESSION.interventions, r.selected))


@app.get("/api/measure/{oid}")
def measure_series(oid: str, segment: int = -1):
    from . import measure
    if SESSION.sim is None or oid not in SESSION.sim.get("kinematics", {}):
        raise HTTPException(404, "no measurements for that object")
    segs = SESSION.sim["segments"]
    seg = segs[segment]
    s = measure.series(SESSION.sim, oid, 0.0, None)
    keep = s["t"] <= SESSION.sim["duration"] + 1e-9
    out = {k: (v[keep].tolist() if hasattr(v, "tolist") else v) for k, v in s.items() if k != "v"}
    out["metrics"] = measure.metrics(SESSION.sim, oid, seg["t_start"], seg["t_end"])
    out["label"] = SESSION.sim["labels"].get(oid, oid)
    out["segment"] = {"t_start": seg["t_start"], "t_end": seg["t_end"]}
    return to_jsonable(out)


# ---------------------------------------------------------------- phone camera
@app.get("/api/connect")
def connect():
    from . import remote
    return remote.connect_info() | {"status": remote.status()}


@app.post("/api/remote/frame")
async def remote_frame(image: UploadFile = File(...), steady: bool = Form(False), motion: float = Form(-1.0),
                       change: float = Form(-1.0)):
    """A phone frame plus the phone's own analysis of it (steady? how much motion? how different from
    the last scan?). Steady frames are kept for multi-frame depth fusion."""
    from . import live, remote
    data = await image.read()
    if len(data) > 8_000_000:
        raise HTTPException(413, "frame too large")
    remote.put_frame(data, {"steady": steady, "motion": motion, "change": change})
    live.offer(data, steady)
    return {"ok": True, "live": live.status()}


@app.get("/api/remote/status")
def remote_status():
    from . import remote
    return remote.status()


@app.get("/api/remote/latest.jpg")
def remote_latest():
    from . import remote
    jpeg, _ = remote.latest()
    if jpeg is None:
        raise HTTPException(404, "no phone frame yet")
    return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/remote/stream")
async def remote_stream(request: Request):
    from fastapi.responses import StreamingResponse
    from . import remote
    return StreamingResponse(remote.mjpeg(request), media_type="multipart/x-mixed-replace; boundary=frame",
                             headers={"Cache-Control": "no-store"})


@app.post("/api/scan_remote")
def scan_remote(hfov: float = Form(65.0), use_claude: bool = Form(True)):
    """Scan from the phone: sharpest recent steady frame + depth fused over the steady burst."""
    from PIL import Image
    from . import live, remote
    jpeg, t = remote.latest()
    if jpeg is None or time.time() - t > 30:
        raise HTTPException(409, "no recent phone frame: open the phone page and start the camera")
    with live._lock:
        burst = sorted(live._buf, key=lambda f: -f[2])[:3]
    rgb = pipeline.load_frame(burst[0][1] if burst else jpeg)
    depth = None
    if len(burst) >= 2:
        from .perception import models
        maps = [models.estimate_depth(Image.fromarray(pipeline.load_frame(f[1]))).astype(np.float32) for f in burst]
        maps = [m for m in maps if m.shape == maps[0].shape]
        if len(maps) >= 2:
            depth = np.median(np.stack(maps), axis=0)
    try:
        st = pipeline.scan(rgb, hfov, "phone" + ("+fused" if depth is not None else ""), use_claude=use_claude, depth_override=depth)
    except ValueError as e:
        raise HTTPException(422, str(e))
    with SESSION.lock:
        SESSION.perception = st
        SESSION.interventions.clear()
        SESSION.redo_stack.clear()
        SESSION.plates.clear()
        SESSION.invalidate()
    with live._lock:          # the Mac's page watches this counter: a scan from the phone must show up there
        live.STATE.update(version=live.STATE["version"] + 1, added=[], removed=[], rerun=None, error=None,
                          frames_fused=len(burst) if depth is not None else 1, t=time.time())
    return _scene_payload()


@app.get("/phone")
def phone_page():
    return FileResponse(Path(__file__).parent / "static" / "phone.html", headers={"Cache-Control": "no-store"})


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Local Whisper: 16-bit PCM WAV in, text out."""
    from . import speech
    t = time.perf_counter()
    try:
        text = speech.transcribe_wav(await audio.read())
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"text": text, "ms": round((time.perf_counter() - t) * 1000)}


@app.post("/api/undo")
def undo():
    _need_scene()
    with SESSION.lock:
        floor = len(SESSION.ckpt["ivs"]) if _ckpt_valid() else 0      # Live AR checkpoints can't be undone past
        if len(SESSION.interventions) > floor:
            SESSION.redo_stack.append(SESSION.interventions.pop())
        SESSION.invalidate()
        sim = _run_sim() if SESSION.interventions else None
    return {"sim": sim, **_scene_payload()}


@app.post("/api/redo")
def redo():
    _need_scene()
    with SESSION.lock:
        if SESSION.redo_stack:
            SESSION.interventions.append(SESSION.redo_stack.pop())
        SESSION.invalidate()
        sim = _run_sim() if SESSION.interventions else None
    return {"sim": sim, **_scene_payload()}


@app.post("/api/reset")
def reset():
    with SESSION.lock:
        SESSION.interventions.clear()
        SESSION.redo_stack.clear()
        SESSION.invalidate()
    return {"sim": None, **_scene_payload()}


@app.get("/api/sim")
def get_sim():
    return {"sim": SESSION.sim}


@app.get("/api/plate.png")
def plate(ids: str = ""):
    st = _need_scene()
    key = ",".join(sorted(i for i in ids.split(",") if i)) + ("|cp" if SESSION.clean_plate is not None else "")
    if key not in SESSION.plates:
        want = set(key.split("|")[0].split(",")) - {""}
        masks = [r.mask for r in st.records if r.id in want]
        rgba = compositor.clean_plate_patch(st.rgb, masks, SESSION.clean_plate)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        SESSION.plates[key] = buf.tobytes()
    return Response(SESSION.plates[key], media_type="image/png", headers={"Cache-Control": "no-store"})


class ExplainReq(BaseModel):
    segment: int = -1
    use_claude: bool = True


@app.post("/api/explain")
def explain_segment(r: ExplainReq):
    st = _need_scene()
    if SESSION.sim is None:
        raise HTTPException(409, "nothing simulated yet")
    n = len(SESSION.sim["segments"])
    idx = r.segment if r.segment >= 0 else n + r.segment
    if not 0 <= idx < n:
        raise HTTPException(422, "bad segment")
    key = f"{SESSION.sim['key']}:{idx}:{r.use_claude}"
    if key not in SESSION.explanations:
        t = time.perf_counter()
        ex = explain(st.scene, SESSION.sim, idx, use_claude=r.use_claude)
        ex["compute_ms"] = round((time.perf_counter() - t) * 1000)
        ex["segment"] = idx
        SESSION.explanations[key] = to_jsonable(ex)
    return SESSION.explanations[key]


# ---------------------------------------------------------------- static files
app.mount("/assets/library", StaticFiles(directory=config.ROOT / "assets" / "library"), name="library")
asset_lib.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/assets/generated", StaticFiles(directory=asset_lib.GENERATED_DIR), name="generated")
DIST = config.ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
