"""Frame -> explicit, simulation-ready scene.

Neural stages (depth, SAM proposals, optional Claude labels) run once per
scan. Everything after them (plane fit, lifting, support, stabilisation) is
cheap and re-runs whenever the user edits the reconstruction: adds/removes a
mask, fixes a shape, calibrates scale or changes the FOV."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .. import config
from ..physics.sim import stabilize
from ..schemas import Camera, Scene, SceneObject, Table
from . import geometry as geo
from .labeler import label_with_claude, label_with_local_ai
from .lifting import infer_support, lift_mask
from .materials import MATERIALS, friction_for, restitution_for, guess_shape, mass_from_volume
from .models import SamSession, estimate_depth
from .segment import above_table_mask, local_ground, mask_from_click, propose


@dataclass
class ObjRecord:
    """Persistent per-object state across rebuilds (mask + user overrides)."""
    id: str
    mask: np.ndarray
    score: float
    label: str
    label_source: str = "none"
    material: str = "unknown"
    shape: str | None = None
    mass_override: float | None = None
    mass_claude: float | None = None
    friction_override: float | None = None
    static: bool = False


@dataclass
class PerceptionState:
    scene_id: str
    source: str
    rgb: np.ndarray                 # H,W,3 uint8 (processed resolution)
    depth_raw: np.ndarray           # model output (m)
    hfov_deg: float
    depth_scale: float = 1.0
    records: list[ObjRecord] = field(default_factory=list)
    sam: SamSession | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    labeler: str = "none"
    # derived
    pts_world: np.ndarray | None = None
    height: np.ndarray | None = None
    plane_inliers: np.ndarray | None = None
    baseline_qpos: np.ndarray | None = None
    scene: Scene | None = None
    next_id: int = 1
    # sweep-scanned shapes by object id (kept across rebuilds; scaled if the scale is calibrated later)
    meshes: dict = field(default_factory=dict)

    def new_id(self) -> str:
        oid = f"o{self.next_id}"
        self.next_id += 1
        return oid


def load_frame(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    h, w = bgr.shape[:2]
    if w > config.MAX_PROCESS_WIDTH:
        s = config.MAX_PROCESS_WIDTH / w
        bgr = cv2.resize(bgr, (config.MAX_PROCESS_WIDTH, int(round(h * s / 2) * 2)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def scan(rgb: np.ndarray, hfov_deg: float, source: str, use_claude: bool = True,
         depth_override: np.ndarray | None = None, label: bool = True) -> PerceptionState:
    """depth_override: sensor depth (RGB-D camera) or renderer ground truth in tests."""
    timings = {}
    img = Image.fromarray(rgb)
    t = time.perf_counter()
    depth = depth_override if depth_override is not None else estimate_depth(img)
    timings["depth"] = (time.perf_counter() - t) * 1000
    st = PerceptionState(scene_id=uuid.uuid4().hex[:10], source=source, rgb=rgb, depth_raw=depth, hfov_deg=hfov_deg)
    t = time.perf_counter()
    _geometry(st)
    timings["plane"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    st.sam = SamSession(img)
    if st._plane.room:
        from .segment import room_object_mask
        above = room_object_mask(st.height, st._plane.structure, np.isfinite(st.height))
    else:
        rect = geo.table_extent(st._plane, st._frame, st._pts_c, extend=False)
        above = above_table_mask(st.height, st.pts_world, rect, np.isfinite(st.height))
    props = propose(st.sam, st._pts_c, st.height, above, rgb, st.pts_world)
    if st._plane.room:
        # room mode: a region at the top edge of the frame more than 2 m up is ceiling, not an object
        cam_z = float(st._frame.t_wc[2])
        props = [p for p in props if not (p.mask[:3].any() and np.nanmedian(st.height[p.mask]) > cam_z + 0.6)]
    timings["segment"] = (time.perf_counter() - t) * 1000
    for p in props:
        st.records.append(ObjRecord(id=st.new_id(), mask=p.mask, score=p.score, label=""))
    _order_left_to_right(st)
    for i, r in enumerate(st.records):
        r.label = f"object {i + 1}"

    _label(st, rgb, use_claude, label, timings)
    st.timings_ms = timings
    rebuild(st)
    return st




def _label(st: PerceptionState, rgb: np.ndarray, use_claude: bool, label: bool, timings: dict) -> None:
    """Names and materials for every record (Claude when available, else the local vision model)."""
    if st.records and label:
        t = time.perf_counter()
        labels = label_with_claude(rgb, [r.mask for r in st.records]) if use_claude else None
        st.labeler = "claude" if labels else "none"
        if not labels:
            try:
                labels = label_with_local_ai(rgb, [r.mask for r in st.records])
            except Exception as e:      # labels are a nicety: never let them break a scan
                import logging
                logging.getLogger(__name__).warning("local labelling failed: %s", e)
                labels = None
            st.labeler = "local AI (Qwen2.5-VL-3B)" if labels else "none"
        timings["label"] = (time.perf_counter() - t) * 1000
        if labels:
            by_num = {int(l["number"]): l for l in labels}
            for i, r in enumerate(st.records):
                l = by_num.get(i + 1)
                if not l:
                    continue
                r.label = l["label"]
                r.material = l["material"] if l["material"] in MATERIALS else "unknown"
                if l.get("shape") and st.labeler.startswith("claude"):
                    r.shape = l["shape"]          # the local model's shape guesses are weaker than geometry
                if 0.005 < float(l.get("estimated_mass_kg") or 0) < 50:
                    r.mass_claude = float(l["estimated_mass_kg"])
                r.label_source = st.labeler


def scan_with_masks(rgb: np.ndarray, hfov_deg: float, source: str, depth: np.ndarray, masks: list[np.ndarray],
                    use_claude: bool = True, label: bool = True) -> PerceptionState:
    """Like scan(), but the objects are given (e.g. found by multi-view geometry in a sweep scan) instead of
    proposed from one photo. depth: metres for every pixel (the sweep supplies an exact table plane)."""
    timings: dict[str, float] = {}
    img = Image.fromarray(rgb)
    st = PerceptionState(scene_id=uuid.uuid4().hex[:10], source=source, rgb=rgb, depth_raw=depth, hfov_deg=hfov_deg)
    _geometry(st)
    st.sam = SamSession(img)
    for m in masks:
        st.records.append(ObjRecord(id=st.new_id(), mask=m.astype(bool), score=1.0, label=""))
    _order_left_to_right(st)
    for i, r in enumerate(st.records):
        r.label = f"object {i + 1}"
    _label(st, rgb, use_claude, label, timings)
    st.timings_ms = timings
    rebuild(st)
    return st


def _order_left_to_right(st: PerceptionState) -> None:
    st.records.sort(key=lambda r: np.nonzero(r.mask)[1].mean())


def _geometry(st: PerceptionState) -> None:
    h, w = st.depth_raw.shape
    K = geo.Intrinsics.from_hfov(w, h, st.hfov_deg)
    depth = st.depth_raw * st.depth_scale
    valid = np.isfinite(depth) & (np.nan_to_num(depth) > 0.05) & (np.nan_to_num(depth) < 8)
    depth = np.where(valid, depth, 1.0)
    pts_c = geo.backproject(depth, K)
    plane = geo.fit_table_plane(pts_c, valid)
    frame = geo.make_world_frame(plane, K, pts_c)
    st.pts_world = frame.to_world(pts_c.reshape(-1, 3)).reshape(h, w, 3)
    st.pts_world[~valid] = np.nan
    st.height = st.pts_world[..., 2]
    st.plane_inliers = plane.inlier_mask
    st._K, st._plane, st._frame, st._pts_c = K, plane, frame, pts_c


def rebuild(st: PerceptionState) -> Scene:
    """Re-run the geometric half of the pipeline from cached neural outputs."""
    t = time.perf_counter()
    _geometry(st)
    K, plane, frame = st._K, st._plane, st._frame
    h, w = st.depth_raw.shape
    lifted, keep = [], []
    all_masks = np.any([r.mask for r in st.records], axis=0) if st.records else np.zeros((h, w), bool)
    for r in st.records:
        shape = r.shape or guess_shape(r.label)
        L = lift_mask(r.mask, st.pts_world, shape, view_right=frame.R_wc[:, 0])
        if L is None:
            continue
        g = local_ground(r.mask, st.height, all_masks)
        if abs(g) > 0.004:     # monocular depth bends the table locally; measure from local ground
            L.z_min_vis -= g
            L.z_top -= g
            L.notes.append(f"local table offset {g * 100:+.1f} cm corrected")
        lifted.append(L)
        keep.append(r)
    infer_support(lifted)
    x0, x1, y0, y1 = geo.table_extent(plane, frame, st._pts_c)
    # the table must at least cover every object's footprint
    for L in lifted:
        poly = L.footprint_poly()
        x0, x1 = min(x0, poly[:, 0].min() - 0.03), max(x1, poly[:, 0].max() + 0.03)
        y0, y1 = min(y0, poly[:, 1].min() - 0.03), max(y1, poly[:, 1].max() + 0.03)
    objects = []
    for i, (r, L) in enumerate(zip(keep, lifted)):
        size = [float(L.dims_xy[0]), float(L.dims_xy[1]), float(L.height)]
        mat = r.material
        if r.mass_override is not None:
            mass, msrc = r.mass_override, "user"
        elif r.mass_claude is not None:
            mass, msrc = r.mass_claude, "claude estimate"
        else:
            mass, msrc = mass_from_volume(L.shape, size, mat), f"volume x density ({mat})"
        fr = r.friction_override if r.friction_override is not None else friction_for(mat)
        cnts, _ = cv2.findContours(r.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
        ys, xs = np.nonzero(r.mask)
        objects.append(SceneObject(
            id=r.id, idx=i + 1, label=r.label, material=mat, shape=L.shape, size=size,
            pos=[float(L.center_xy[0]), float(L.center_xy[1]), float(L.bottom + L.height / 2)],
            yaw=float(L.yaw), profile=L.profile(), mass=round(float(mass), 4), mass_source=msrc, friction=round(float(fr), 3), bounciness=restitution_for(mat),
            static=r.static, polygon=c.astype(float).tolist(),
            bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
            area_px=int(r.mask.sum()), mean_color=(st.rgb[r.mask].mean(axis=0) / 255).round(3).tolist(),
            score=round(r.score, 3), label_source=r.label_source, notes=list(L.notes),
        ))
    st.records = keep
    _apply_meshes(st, objects)
    cam = Camera(width=w, height=h, fx=K.fx, fy=K.fy, cx=K.cx, cy=K.cy, hfov_deg=st.hfov_deg,
                 R_wc=frame.R_wc.tolist(), t_wc=frame.t_wc.tolist())
    floor_h = plane.floor_offset if plane.floor_offset and 0.2 < plane.floor_offset < 1.5 else config.TABLE_HEIGHT
    if plane.room:
        floor_h = 0.05          # the ground IS the floor: things stand on it
    table = Table(x_min=x0, x_max=x1, y_min=y0, y_max=y1, height_above_floor=round(floor_h, 3),
                  plane_inlier_ratio=round(float(plane.inlier_mask.mean()), 3), plane_rms_m=round(plane.rms, 4))
    scene = Scene(scene_id=st.scene_id, source=st.source, camera=cam, table=table, objects=objects,
                  labeler=st.labeler)
    timings = dict(st.timings_ms)
    timings["lift"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    scene, qpos, warnings = stabilize(scene)
    timings["stabilize"] = (time.perf_counter() - t) * 1000
    scene.timings_ms = {k: round(v, 1) for k, v in timings.items()}
    scene.warnings = warnings
    if plane.room:
        scene.warnings.append("Room view: no tabletop seen from above, so heights are measured from the floor (or the lowest visible point). "
                              "For precise tabletop experiments, point the camera down at the table (about 30–60°).")
    if plane.rms > 0.012:
        scene.warnings.append(f"Table plane is noisy (rms {plane.rms * 100:.1f} cm); heights may be off.")
    st.baseline_qpos = qpos
    st.scene = scene
    return scene


def _apply_meshes(st: PerceptionState, objects: list) -> None:
    """Objects that a sweep scan carved keep their scanned shape, position and mass through any rebuild."""
    for o in objects:
        m = st.meshes.get(o.id)
        if not m:
            continue
        k = st.depth_scale / m["depth_scale"]           # scale calibrated after the sweep: everything scales with it
        o.shape = "mesh"
        o.size = [v * k for v in m["size"]]
        o.pos = [v * k for v in m["pos"]]
        o.yaw = m["yaw"]
        o.profile = None
        o.mesh_url, o.hulls = m["mesh_url"], m["hulls"]
        o.volume_m3 = m["volume_m3"] * k ** 3
        if o.mass_source.startswith("volume x density") or o.mass_source.startswith("claude"):
            o.mass = round(m["mass"] * k ** 3, 4)
            o.mass_source = m["mass_source"]
        o.notes = [n for n in o.notes if not n.startswith(("cylinder", "hidden depth", "revolved", "bottom occluded"))]
        o.notes += m["notes"]


def add_by_click(st: PerceptionState, x: float, y: float) -> str | None:
    res = mask_from_click(st.sam, x, y)
    if res is None:
        return None
    m, score = res
    taken = np.any([r.mask for r in st.records], axis=0) if st.records else np.zeros_like(m)
    clicked_existing = [r for r in st.records if r.mask[int(y), int(x)]]
    if clicked_existing:
        # user clicked an existing object: replace its mask (refinement)
        r = clicked_existing[0]
        r.mask = m
        for o in st.records:
            if o is not r:
                o.mask = o.mask & ~m
        rebuild(st)
        return r.id
    m = m & ~taken
    if m.sum() < 50:
        return None
    r = ObjRecord(id=st.new_id(), mask=m, score=score, label=f"object {len(st.records) + 1}", label_source="user click")
    st.records.append(r)
    rebuild(st)
    return r.id


def id_map(st: PerceptionState) -> np.ndarray:
    """uint8 image: pixel = SceneObject.idx (0 = background)."""
    out = np.zeros(st.rgb.shape[:2], np.uint8)
    idx = {o.id: o.idx for o in st.scene.objects}
    for r in st.records:
        if r.id in idx:
            out[r.mask] = idx[r.id]
    return out


def calibrate(st: PerceptionState, object_id: str, dimension: str, value_m: float) -> Scene:
    o = next(o for o in st.scene.objects if o.id == object_id)
    current = {"height": o.size[2], "width": max(o.size[0], o.size[1]), "depth": min(o.size[0], o.size[1])}[dimension]
    st.depth_scale *= value_m / max(1e-4, current)
    return rebuild(st)
