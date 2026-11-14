"""Synthetic hero scenes rendered with MuJoCo.

These exist for two honest reasons: (1) automated end-to-end tests of the
perception -> physics pipeline against known ground truth, and (2) a demo
source when no camera is available. They are always labelled "synthetic" in
the UI and API; they are never presented as camera input."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import config

TEX_DIR = config.DATA_DIR / "synth_textures"
TABLE_TOP_Z = 0.75


@dataclass
class SynthObj:
    name: str
    shape: str            # box | cylinder
    size: tuple           # box: full extents (x,y,z); cylinder: (diameter, height)
    pos: tuple            # center (x,y,z) relative to table top at z=0
    rgb: tuple
    pattern: str = "plain"
    yaw: float = 0.0
    mass: float = 0.3


@dataclass
class SynthScene:
    name: str
    title: str
    objects: list[SynthObj]
    cam_pos: tuple = (0.0, -0.62, 0.46)
    cam_target: tuple = (0.0, 0.05, 0.03)
    fovy: float = 42.0
    width: int = 960
    height: int = 540
    notes: str = ""


def _scenes() -> dict[str, SynthScene]:
    s = {}
    s["bridge"] = SynthScene(
        "bridge", "Book bridge on two mugs",
        [
            SynthObj("mug_left", "cylinder", (0.085, 0.10), (-0.085, 0.05, 0.05), (0.85, 0.85, 0.82), "band"),
            SynthObj("mug_right", "cylinder", (0.085, 0.10), (0.085, 0.05, 0.05), (0.20, 0.35, 0.75), "band"),
            SynthObj("book", "box", (0.26, 0.17, 0.035), (0.0, 0.05, 0.1175), (0.75, 0.18, 0.15), "cover"),
            SynthObj("tea_box", "box", (0.09, 0.06, 0.07), (0.05, 0.05, 0.17), (0.95, 0.75, 0.20), "label"),
            SynthObj("bottle", "cylinder", (0.065, 0.20), (0.23, 0.16, 0.10), (0.25, 0.65, 0.40), "label"),
            SynthObj("block", "box", (0.06, 0.06, 0.06), (-0.24, -0.02, 0.03), (0.55, 0.35, 0.65), "plain", yaw=0.4),
        ],
        notes="Remove a mug: the book loses half its support, tilts, and the tea box slides off.",
    )
    s["tower"] = SynthScene(
        "tower", "Three-box tower",
        [
            SynthObj("base_box", "box", (0.16, 0.12, 0.08), (0.0, 0.06, 0.04), (0.65, 0.45, 0.25), "label"),
            SynthObj("middle_box", "box", (0.12, 0.10, 0.07), (0.0, 0.06, 0.115), (0.20, 0.55, 0.70), "label"),
            SynthObj("top_box", "box", (0.08, 0.07, 0.06), (0.01, 0.06, 0.18), (0.90, 0.40, 0.30), "label"),
            SynthObj("can", "cylinder", (0.066, 0.12), (-0.19, 0.02, 0.06), (0.85, 0.15, 0.15), "band"),
            SynthObj("mug", "cylinder", (0.085, 0.10), (0.19, 0.0, 0.05), (0.95, 0.95, 0.90), "band"),
        ],
        notes="Remove the middle box: the top box drops onto the base. Push the base: the tower slides.",
    )
    return s


SCENES = _scenes()


def _texture(name: str, rgb, pattern: str, size=(256, 256)) -> str:
    TEX_DIR.mkdir(parents=True, exist_ok=True)
    path = TEX_DIR / f"{name}.png"
    h, w = size
    base = np.array(rgb[::-1], dtype=np.float32) * 255  # BGR
    img = np.ones((h, w, 3), np.float32) * base
    rng = np.random.default_rng(sum(name.encode()))
    img += rng.normal(0, 6, img.shape)                  # subtle grain
    if pattern == "band":
        img[int(h * 0.35):int(h * 0.55)] = img[int(h * 0.35):int(h * 0.55)] * 0.55 + 255 * 0.3
    elif pattern == "cover":
        img[:, : int(w * 0.08)] *= 0.6
        cv2.rectangle(img, (int(w * 0.25), int(h * 0.3)), (int(w * 0.8), int(h * 0.45)), (235, 235, 235), -1)
    elif pattern == "label":
        cv2.rectangle(img, (int(w * 0.1), int(h * 0.35)), (int(w * 0.9), int(h * 0.65)), (245, 245, 245), -1)
        cv2.line(img, (int(w * 0.2), int(h * 0.5)), (int(w * 0.8), int(h * 0.5)), tuple(float(c) for c in base * 0.5), 6)
    elif pattern == "wood":
        for y in range(h):
            img[y] *= 0.9 + 0.1 * np.sin(y * 0.15 + 3 * np.sin(y * 0.02))
    cv2.imwrite(str(path), np.clip(img, 0, 255).astype(np.uint8))
    return str(path)


def build_xml(scene: SynthScene) -> str:
    tex_assets, mats, bodies = [], [], []
    table_tex = _texture("table_wood", (0.62, 0.47, 0.33), "wood", (512, 512))
    wall_tex = _texture("wall", (0.86, 0.86, 0.84), "plain")
    for o in scene.objects:
        path = _texture(o.name, o.rgb, o.pattern)
        tex_assets.append(f'<texture name="t_{o.name}" type="{"cube" if o.shape == "box" else "2d"}" file="{path}"/>')
        mats.append(f'<material name="m_{o.name}" texture="t_{o.name}" specular="0.15" shininess="0.2"/>')
        x, y, z = o.pos
        if o.shape == "box":
            sx, sy, sz = (v / 2 for v in o.size)
            geom = f'<geom name="g_{o.name}" type="box" size="{sx} {sy} {sz}" material="m_{o.name}" mass="{o.mass}"/>'
        else:
            d, hgt = o.size
            geom = f'<geom name="g_{o.name}" type="cylinder" size="{d / 2} {hgt / 2}" material="m_{o.name}" mass="{o.mass}"/>'
        q = (np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2))
        bodies.append(f'<body name="{o.name}" pos="{x} {y} {z + TABLE_TOP_Z}" quat="{q[0]} {q[1]} {q[2]} {q[3]}"><freejoint/>{geom}</body>')
    cx, cy, cz = scene.cam_pos
    tx, ty, tz = scene.cam_target
    return f"""
<mujoco model="{scene.name}">
  <visual><global offwidth="{scene.width}" offheight="{scene.height}"/><quality shadowsize="4096"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.35 0.35 0.35"/></visual>
  <asset>
    <texture name="t_table" type="2d" file="{table_tex}"/>
    <material name="m_table" texture="t_table" texrepeat="2 2" specular="0.1"/>
    <texture name="t_wall" type="2d" file="{wall_tex}"/>
    <material name="m_wall" texture="t_wall"/>
    {''.join(tex_assets)}
    {''.join(mats)}
  </asset>
  <worldbody>
    <light pos="0.4 -0.5 2.2" dir="-0.2 0.25 -1" diffuse="0.75 0.75 0.72" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.35 0.33 0.32 1"/>
    <geom name="wall" type="box" pos="0 1.2 1.0" size="3 0.02 1.5" material="m_wall"/>
    <geom name="table" type="box" pos="0 0.1 {TABLE_TOP_Z - 0.02}" size="0.7 0.5 0.02" material="m_table"/>
    <body name="camtarget" pos="{tx} {ty} {tz + TABLE_TOP_Z}"/>
    <camera name="cam" pos="{cx} {cy} {cz + TABLE_TOP_Z}" mode="targetbody" target="camtarget" fovy="{scene.fovy}"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def render(name: str, out_dir: Path | None = None) -> dict:
    """Render RGB + ground-truth depth/segmentation. Returns paths + metadata."""
    import mujoco
    scene = SCENES[name]
    out_dir = out_dir or (config.DATA_DIR / "synthetic" / name)
    out_dir.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_string(build_xml(scene))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, scene.height, scene.width)
    r.update_scene(d, camera="cam")
    rgb = r.render().copy()
    r.enable_depth_rendering(); r.update_scene(d, camera="cam"); depth = r.render().copy(); r.disable_depth_rendering()
    r.enable_segmentation_rendering(); r.update_scene(d, camera="cam"); seg = r.render().copy(); r.disable_segmentation_rendering()
    r.close()
    gt_ids = np.zeros(seg.shape[:2], np.uint8)
    names = [o.name for o in scene.objects]
    for i, o in enumerate(scene.objects):
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{o.name}")
        gt_ids[(seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = i + 1
    hfov = float(np.degrees(2 * np.arctan(np.tan(np.radians(scene.fovy) / 2) * scene.width / scene.height)))
    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(out_dir / "depth_gt.npy", depth.astype(np.float32))
    cv2.imwrite(str(out_dir / "ids_gt.png"), gt_ids)
    meta = {
        "name": name, "title": scene.title, "notes": scene.notes, "hfov_deg": hfov,
        "objects": [{"name": o.name, "shape": o.shape, "size": list(o.size), "pos": list(o.pos), "mass": o.mass} for o in scene.objects],
        "labels": names,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"rgb": str(out_dir / "rgb.png"), **meta}


if __name__ == "__main__":
    for n in SCENES:
        print(render(n)["rgb"])


# ---------------------------------------------------------------- photoreal desk (Poly Haven photoscans)
# (asset id, x, y, yaw): real scanned textures, so monocular depth and feature matching behave like on a real desk
PHOTO_SCENES = {
    "desk": [("can_rusted", -0.17, 0.10, 0.3), ("cardboard_box_01", 0.02, 0.14, -0.35), ("alarm_clock_01", 0.20, 0.06, 0.6),
             ("baseball_01", -0.05, -0.04, 0.0), ("wooden_bowl_01", 0.16, -0.10, 0.0), ("lemon", -0.20, -0.08, 1.1)],
}
PHOTO_DIR = config.DATA_DIR / "synth_assets"


def _photo_asset(asset_id: str) -> tuple[str, str, "np.ndarray", "np.ndarray"]:
    """OBJ (body frame: z up, bottom at z=0, real size) + PNG base-colour texture for MuJoCo; and the GT mesh."""
    import trimesh
    from . import assets as A
    e = A.manifest()[asset_id]
    out = PHOTO_DIR / asset_id
    obj, png = out / "m.obj", out / "tex.png"
    sc = trimesh.load(A.LIB / asset_id / Path(e["url"]).name, force="scene")
    meshes = []
    for name in sc.graph.nodes_geometry:
        T, g = sc.graph.get(name)
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        meshes.append(m)
    m = max(meshes, key=lambda x: len(x.faces))              # photoscans are one textured mesh
    m.apply_transform(np.array(e["gltf_to_body"]))
    if not obj.exists():
        out.mkdir(parents=True, exist_ok=True)
        img = m.visual.material.baseColorTexture if hasattr(m.visual.material, "baseColorTexture") else m.visual.material.image
        img.convert("RGB").save(png)
        uv = np.asarray(m.visual.uv)
        lines = [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in m.vertices]
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in m.faces]
        obj.write_text("\n".join(lines) + "\n")
    return str(obj), str(png), np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def photo_xml(name: str, width: int, height: int) -> tuple[str, list[dict]]:
    table_tex = _texture("table_wood", (0.62, 0.47, 0.33), "wood", (512, 512))
    wall_tex = _texture("wall", (0.86, 0.86, 0.84), "plain")
    assets_xml, bodies, gt = [], [], []
    for k, (aid, x, y, yaw) in enumerate(PHOTO_SCENES[name]):
        obj, png, V, F = _photo_asset(aid)
        assets_xml.append(f'<mesh name="m{k}" file="{obj}"/><texture name="t{k}" type="2d" file="{png}"/>'
                          f'<material name="mt{k}" texture="t{k}" specular="0.2" shininess="0.3"/>')
        q = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        bodies.append(f'<body name="{aid}" pos="{x} {y} {TABLE_TOP_Z}" quat="{q[0]} 0 0 {q[3]}">'
                      f'<geom name="g_{aid}" type="mesh" mesh="m{k}" material="mt{k}" contype="0" conaffinity="0"/></body>')
        c, s_ = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]], np.float32)
        Vw = V @ Rz.T + np.array([x, y, 0], np.float32)
        ext = V.max(0) - V.min(0)
        gt.append({"name": aid, "V": Vw, "F": F, "size": [float(v) for v in ext], "pos": [x, y, float(ext[2] / 2)], "yaw": yaw})
    xml = f"""
<mujoco model="{name}">
  <compiler angle="radian"/>
  <visual><global offwidth="{width}" offheight="{height}" fovy="45"/><quality shadowsize="4096"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.35 0.35 0.35"/></visual>
  <asset>
    <texture name="t_table" type="2d" file="{table_tex}"/><material name="m_table" texture="t_table" texrepeat="2 2" specular="0.1"/>
    <texture name="t_wall" type="2d" file="{wall_tex}"/><material name="m_wall" texture="t_wall"/>
    {''.join(assets_xml)}
  </asset>
  <worldbody>
    <light pos="0.4 -0.5 2.2" dir="-0.2 0.25 -1" diffuse="0.8 0.8 0.78" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" rgba="0.35 0.33 0.32 1"/>
    <geom name="wall" type="box" pos="0 1.2 1.0" size="3 0.02 1.5" material="m_wall"/>
    <geom name="table" type="box" pos="0 0.1 {TABLE_TOP_Z - 0.02}" size="0.7 0.5 0.02" material="m_table"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return xml, gt


def render_sweep(name: str, n: int = 36, out_dir: Path | None = None, width: int = 960, height: int = 540,
                 az_path=(0.0, -60.0, 60.0, 0.0), el_path=(-32.0, -26.0, -42.0, -32.0)) -> dict:
    """A phone sweeping around the scene and coming back to where it started (ground truth for the sweep scanner).

    Returns RGB frames, ground-truth depth, object ids, intrinsics and every camera's pose (OpenCV camera frame,
    world z-up with the table top at z=0)."""
    import mujoco
    if name in PHOTO_SCENES:
        xml, gt_objs = photo_xml(name, width, height)
        scene = SynthScene(name, name, [], cam_pos=(0.0, -0.62, 0.46), cam_target=(0.0, 0.03, 0.03))
    else:
        scene = SCENES[name]
        xml = build_xml(scene).replace(f'offwidth="{scene.width}" offheight="{scene.height}"', f'offwidth="{width}" offheight="{height}"')
        gt_objs = None
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height, width)
    target = np.array(scene.cam_target) + [0, 0, TABLE_TOP_Z]
    dist = float(np.linalg.norm(np.array(scene.cam_pos) - np.array(scene.cam_target)))
    knots = np.linspace(0, 1, len(az_path))
    s = np.linspace(0, 1, n)
    az = np.interp(s, knots, az_path) + 90.0          # MuJoCo azimuth: +90 = looking along +y (the front view)
    el = np.interp(s, knots, el_path)
    fovy = float(m.vis.global_.fovy)
    fy = height / 2 / np.tan(np.radians(fovy) / 2)
    K = np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1.0]])
    frames, depths, ids, poses = [], [], [], []
    names = [g["name"] for g in gt_objs] if gt_objs else [o.name for o in scene.objects]
    gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{nm}"): i + 1 for i, nm in enumerate(names)}
    for a, e in zip(az, el):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance, cam.azimuth, cam.elevation = dist, a, e
        r.update_scene(d, camera=cam)
        gl = r.scene.camera[0]
        fwd, up = np.array(gl.forward), np.array(gl.up)
        R_wc = np.stack([np.cross(fwd, up), -up, fwd], axis=1)       # OpenCV: x right, y down, z forward
        # camera[0]/[1] are the two stereo eyes (ipd apart); a mono render looks from their midpoint
        pos = (np.array(gl.pos) + np.array(r.scene.camera[1].pos)) / 2 - [0, 0, TABLE_TOP_Z]
        frames.append(r.render().copy())
        r.enable_depth_rendering(); r.update_scene(d, camera=cam); depths.append(r.render().copy()); r.disable_depth_rendering()
        r.enable_segmentation_rendering(); r.update_scene(d, camera=cam); seg = r.render().copy(); r.disable_segmentation_rendering()
        idm = np.zeros(seg.shape[:2], np.uint8)
        for g, i in gids.items():
            idm[(seg[..., 0] == g) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = i
        ids.append(idm)
        poses.append((R_wc, pos))
    r.close()
    objs = ([{"name": g["name"], "shape": "mesh", "size": g["size"], "pos": g["pos"], "yaw": g["yaw"], "V": g["V"], "F": g["F"]}
             for g in gt_objs] if gt_objs else
            [{"name": o.name, "shape": o.shape, "size": list(o.size), "pos": list(o.pos), "yaw": o.yaw} for o in scene.objects])
    out = {"frames": frames, "depth": depths, "ids": ids, "K": K, "poses": poses, "objects": objs}
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(out_dir / f"f{i:03d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out
