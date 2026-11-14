"""Download the CC0 asset library (Poly Haven + Khronos glTF sample assets).

Run: .venv/bin/python assets/fetch_assets.py
Every asset keeps its source, author and licence in assets/library/<id>/LICENSE.json."""
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent / "library"
UA = {"User-Agent": "causal-world-editor/0.1 (local research prototype)"}

POLYHAVEN = ["rubber_duck_toy", "baseball_01", "lemon", "can_rusted", "cardboard_box_01", "wooden_crate_01",
             "garden_gnome", "alarm_clock_01", "croissant", "yellow_onion", "wooden_bowl_01", "chess_set",
             "carved_wooden_elephant", "tea_set_01"]


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def polyhaven(asset_id: str):
    out = ROOT / asset_id
    if (out / "model.gltf").exists():
        return
    info = json.loads(get(f"https://api.polyhaven.com/info/{asset_id}"))
    files = json.loads(get(f"https://api.polyhaven.com/files/{asset_id}"))["gltf"]["1k"]["gltf"]
    out.mkdir(parents=True, exist_ok=True)
    gltf = json.loads(get(files["url"]))
    for rel, f in files.get("include", {}).items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(get(f["url"]))
    (out / "model.gltf").write_text(json.dumps(gltf))
    # the .gltf references its .bin by original name; keep names as-is
    (out / "LICENSE.json").write_text(json.dumps({
        "license": "CC0-1.0", "source": f"https://polyhaven.com/a/{asset_id}",
        "authors": list(info.get("authors", {}).keys()), "name": info.get("name")}, indent=1))
    print("ok", asset_id)


def khronos(name: str, asset_id: str):
    out = ROOT / asset_id
    if (out / "model.glb").exists():
        return
    base = f"https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/{name}"
    meta = json.loads(get(f"{base}/metadata.json"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.glb").write_bytes(get(f"{base}/glTF-Binary/{name}.glb"))
    (out / "LICENSE.json").write_text(json.dumps({
        "license": " / ".join(sorted({l["spdx"] for l in meta["legal"]})),
        "source": f"https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/{name}",
        "authors": [l["artist"] for l in meta["legal"]], "name": meta.get("name")}, indent=1))
    print("ok", asset_id)


if __name__ == "__main__":
    khronos("ToyCar", "toy_car")
    for a in POLYHAVEN:
        try:
            polyhaven(a)
        except Exception as e:
            print("FAILED", a, e)
