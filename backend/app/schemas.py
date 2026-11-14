from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Shape = Literal["box", "cylinder", "mesh"]


class Camera(BaseModel):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float
    # world_from_camera rotation (3x3, row-major) and camera position in world.
    # Camera frame is OpenCV (x right, y down, z forward); world is z-up, table top at z=0.
    R_wc: list[list[float]]
    t_wc: list[float]


class Table(BaseModel):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height_above_floor: float
    plane_inlier_ratio: float = 0.0
    plane_rms_m: float = 0.0


class Stability(BaseModel):
    moved_m: float = 0.0
    rotated_deg: float = 0.0
    stable: bool = True


class SceneObject(BaseModel):
    id: str
    idx: int                                   # 1-based index used in the id-map PNG
    label: str
    material: str = "unknown"
    shape: Shape = "box"
    size: list[float]                          # full extents [sx, sy, sz] (cylinder: [d, d, h])
    pos: list[float]                           # center, world frame (m)
    yaw: float = 0.0                           # rotation about world z (rad)
    profile: Optional[list[list[float]]] = None  # revolved shape: [[z_from_bottom, radius], ...]
    # sweep-scanned shape (shape == "mesh"): textured GLB for display, convex pieces (body frame) for physics
    mesh_url: Optional[str] = None
    hulls: Optional[list[list[list[float]]]] = None
    volume_m3: Optional[float] = None
    mass: float
    mass_source: str = "volume x density"
    friction: float = 0.5
    bounciness: float = 0.4                    # coefficient of restitution (from its material)
    static: bool = False
    supported_by: list[str] = Field(default_factory=list)
    supporting: list[str] = Field(default_factory=list)
    polygon: list[list[float]] = Field(default_factory=list)   # image-space contour
    bbox: list[int] = Field(default_factory=list)              # x, y, w, h
    area_px: int = 0
    mean_color: list[float] = Field(default_factory=lambda: [0.5, 0.5, 0.5])
    score: float = 0.0
    label_source: str = "none"
    notes: list[str] = Field(default_factory=list)
    stability: Stability = Field(default_factory=Stability)


class Scene(BaseModel):
    scene_id: str
    source: str
    camera: Camera
    table: Table
    objects: list[SceneObject]
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    labeler: str = "none"


class Intervention(BaseModel):
    type: Literal["remove", "push", "set_mass", "set_friction", "set_static", "set_gravity", "place",
                  "spawn", "drive", "launch", "propel", "connect", "hold", "release", "spin", "hand"]
    target: Optional[str] = None
    params: dict[str, Any] = Field(default_factory=dict)


class PatchObject(BaseModel):
    label: Optional[str] = None
    material: Optional[str] = None
    shape: Optional[Shape] = None
    mass: Optional[float] = None
    friction: Optional[float] = None
    static: Optional[bool] = None
