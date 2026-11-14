"""AI generation path with a mocked fal response (real OBJ+MTL+texture -> GLB conversion,
normalisation, collision hull, manifest registration, simulation)."""
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from backend.app import assets, generate, planner
from backend.app.physics.sim import simulate, stabilize
from backend.app.schemas import Intervention
from backend.tests.test_physics import gt_scene


def _fake_obj_files(tmp: Path) -> dict:
    """What the documented text-to-3d example returns: OBJ + MTL + texture, no GLB."""
    mesh = trimesh.creation.capsule(height=0.6, radius=0.25, count=[16, 16])
    uv = (mesh.vertices[:, :2] - mesh.vertices[:, :2].min(0)) / np.ptp(mesh.vertices[:, :2], axis=0)
    img = Image.fromarray((np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    mesh.export(tmp / "out.obj")
    files = {p.name: p for p in tmp.iterdir()}
    mtl = next(n for n in files if n.endswith(".mtl"))
    tex = next(n for n in files if n.endswith((".png", ".jpg")))
    return {"model_obj": {"url": f"file://{files['out.obj']}"},
            "model_urls": {"obj": {"url": f"file://{files['out.obj']}"}, "mtl": {"url": f"file://{files[mtl]}"},
                           "texture": {"url": f"file://{files[tex]}"}}}


def test_generated_asset_end_to_end(tmp_path, monkeypatch):
    result = _fake_obj_files(tmp_path)
    monkeypatch.setattr(generate, "_fetch", lambda url: Path(url[len("file://"):]).read_bytes())
    monkeypatch.setattr(generate, "OUT", tmp_path / "gen")
    monkeypatch.setattr(assets, "GENERATED_DIR", tmp_path / "gen")
    monkeypatch.setattr(assets, "CACHE", tmp_path / "manifest.json")
    assets.invalidate()
    job = generate.start("toy rocket", size_m=0.14, call=lambda prompt: result)
    for _ in range(100):
        if generate.JOBS[job]["state"] in ("done", "error"):
            break
        time.sleep(0.1)
    j = generate.JOBS[job]
    assert j["state"] == "done", j
    a = assets.manifest()[j["asset"]]
    assert abs(max(a["dims"]) - 0.14) < 1e-3 and len(a["hull"]) >= 8
    assert a["url"].startswith("/assets/generated/") and Path(tmp_path / "gen" / j["asset"] / "model.glb").exists()
    # the planner now finds it by name, and it simulates like any library object
    scene, q, _ = stabilize(gt_scene("bridge"))
    ents = planner.snapshot(scene, None, [])
    plan = planner.parse_rules("put a toy rocket near the block", ents, None)
    ivs, human, notes = planner.resolve(plan, scene, ents, 0)
    assert ivs and ivs[0]["params"]["asset"] == j["asset"], (plan, notes)
    sim = simulate(scene, q, [Intervention(**i) for i in ivs])
    assert not sim["segments"][0]["summary"][ivs[0]["target"]]["fell_off_table"]
    assets.invalidate()


def test_unknown_object_suggests_generation():
    scene, q, _ = stabilize(gt_scene("bridge"))
    plan = planner.parse_rules("drop a dragon onto the book", planner.snapshot(scene, None, []), None)
    assert plan["generate"] == "dragon" and not plan["steps"]
