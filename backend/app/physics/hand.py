"""Live physics with the user's real hand: 21 tracked joints -> 21 kinematic bone capsules in MuJoCo, real time.

The browser sends the hand (metres, table frame) about 30 times a second. Each pose is scheduled to be reached by
the time the next one is due (one update later), so the bones glide between samples instead of jumping a centimetre
in one 2 ms physics step; the simulation then catches up to the wall clock. Objects are pushed, flicked, tipped and
carried by the contacts themselves - no gesture rules. The world is built exactly as a replay builds it
(sim.world_inputs), and the track is recorded in the "hand" intervention's format, so the session can be replayed
deterministically, measured and undone like any other action.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from .. import assets as asset_lib
from ..schemas import Scene
from .sim import HAND_BONES, World, world_inputs

BONES = HAND_BONES
PALM = {(0, 5), (5, 9), (9, 13), (13, 17), (0, 17)}
THUMB = {(0, 1), (1, 2), (2, 3), (3, 4)}
MAX_CATCHUP = 0.05          # s of simulation per update at most: a stalled browser never causes a burst
LEAD = 1 / 30               # each pose is reached one update later (smooth motion between samples)


def bone_spec(joints: np.ndarray) -> list[dict]:
    joints = np.asarray(joints, float)
    out = []
    for a, b in BONES:
        L = float(np.linalg.norm(joints[b] - joints[a]))
        r = 0.012 if (a, b) in PALM else 0.0095 if (a, b) in THUMB else 0.0082
        out.append({"length": round(max(0.008, L - 2 * r * 0.6), 4), "radius": r})   # capsule ends overlap the joints
    return out


class HandSession:
    def __init__(self, scene: Scene, joints0, interventions: list | None = None, base_qpos=None,
                 removed: list[str] | None = None, gravity: float = 9.81):
        self.bones = bone_spec(joints0)
        env = {"hand": self.bones}
        spawns, links = world_inputs(interventions or [], env, asset_lib.manifest())
        self.world = World(scene, gravity, spawns, env, links)
        if base_qpos is not None:
            self.world.set_qpos(np.asarray(base_qpos, float))
        for oid in removed or []:
            if oid in self.world.ids:
                self.world.remove(oid)
        mujoco.mj_forward(self.world.m, self.world.d)
        self.t0 = float(self.world.d.time)
        self.world.set_hand_track([], self.t0)
        self.updates: list[list] = []          # [t from session start, joints or None]: the replayable track
        self.wall0: float | None = None
        self.record: list[dict] = []           # 60 Hz: t, joints, object poses (for the live overlay / charts)
        self._next_rec = 0.0
        self.hand_geoms = {mujoco.mj_name2id(self.world.m, mujoco.mjtObj.mjOBJ_GEOM, f"hand_g{k}") for k in range(len(BONES))}

    @property
    def t(self) -> float:
        return float(self.world.d.time) - self.t0

    def update(self, joints, wall: float | None = None) -> dict:
        """joints: 21 x 3 world points, or None when the hand left the view. Steps up to the wall clock."""
        wall = time.perf_counter() if wall is None else wall
        if self.wall0 is None:
            self.wall0 = wall - self.t
        J = None if joints is None else np.asarray(joints, float)
        due = self.t + LEAD
        if self.updates and due <= self.updates[-1][0]:
            due = self.updates[-1][0] + 1e-3
        self.updates.append([round(due, 5), None if J is None else J.round(5).tolist()])
        self.world.append_hand(self.t0 + due, J)
        target = min(wall - self.wall0, self.t + MAX_CATCHUP)
        dt = float(self.world.m.opt.timestep)
        while self.t + dt <= target + 1e-9:
            self.world.step()
            if self.t >= self._next_rec:
                hj = self.world.hand_last
                self.record.append({"t": round(self.t, 4), "joints": None if hj is None else hj.round(4).tolist(),
                                    "poses": self.poses()})
                self._next_rec += 1 / 60
        self.wall0 = wall - self.t                  # if capped, time slows instead of jumping
        return {"t": round(self.t, 4), "poses": self.poses(), "touch": self.touches()}

    def intervention_params(self) -> dict:
        """The session as a replayable 'hand' intervention."""
        return {"bones": self.bones, "updates": self.updates}

    # -- readouts
    def poses(self) -> dict[str, list[float]]:
        return {oid: [round(float(x), 5) for x in self.world.pose(oid)] for oid in self.world.ids}

    def touches(self) -> list[dict]:
        """Hand-object contacts: which object, where, how hard (N)."""
        d, m = self.world.d, self.world.m
        out: dict[str, dict] = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.hand_geoms) == (g2 in self.hand_geoms):
                continue
            oid = self.world.geom_owner.get(g2 if g1 in self.hand_geoms else g1)
            if oid is None or oid in ("table", "floor"):
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            e = out.setdefault(oid, {"id": oid, "force": 0.0, "pos": np.zeros(3), "n": 0})
            e["force"] += float(abs(f6[0]))
            e["pos"] += np.array(c.pos)
            e["n"] += 1
        return [{"id": e["id"], "force": round(e["force"], 3), "pos": (e["pos"] / e["n"]).round(4).tolist()} for e in out.values()]
