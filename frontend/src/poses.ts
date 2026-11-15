import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import type { Camera, SceneObject, Sim } from "./types";
import { scannedNow } from "./scanned";

/** Live physics with the user's hand (/ws/hand): while active, every view shows the server's current poses
 *  instead of the recorded frames. `moved` lists objects the hand has displaced (they render as moving). */
export const LIVE = { active: false, poses: {} as Record<string, number[]>, moved: new Set<string>(), version: 0 };
const liveListeners = new Set<() => void>();
export function bumpLive() { LIVE.version++; liveListeners.forEach((f) => f()); }
export function onLive(f: () => void) { liveListeners.add(f); return () => { liveListeners.delete(f); }; }

/** Interpolated pose of object i at time t (seconds). */
export function poseAt(sim: Sim, i: number, t: number, outPos: THREE.Vector3, outQuat: THREE.Quaternion) {
  if (LIVE.active) {
    const lp = LIVE.poses[sim.object_ids[i]];                 // [x, y, z, qw, qx, qy, qz] (MuJoCo order)
    if (lp) { outPos.set(lp[0], lp[1], lp[2]); outQuat.set(lp[4], lp[5], lp[6], lp[3]); return; }
  }
  const f = Math.max(0, Math.min(sim.frames.length - 1, t * sim.fps));
  const a = Math.floor(f);
  const b = Math.min(sim.frames.length - 1, a + 1);
  const k = f - a;
  const pa = sim.frames[a][i];
  const pb = sim.frames[b][i];
  outPos.set(pa[0] + (pb[0] - pa[0]) * k, pa[1] + (pb[1] - pa[1]) * k, pa[2] + (pb[2] - pa[2]) * k);
  const qa = new THREE.Quaternion(pa[3], pa[4], pa[5], pa[6]);
  const qb = new THREE.Quaternion(pb[3], pb[4], pb[5], pb[6]);
  outQuat.copy(qa).slerp(qb, k);
}

export function observedMatrix(o: SceneObject): THREE.Matrix4 {
  const m = new THREE.Matrix4().makeRotationZ(o.yaw);
  m.setPosition(o.pos[0], o.pos[1], o.pos[2]);
  return m;
}

export function geometryFor(o: SceneObject): THREE.BufferGeometry {
  if (o.shape === "mesh") {
    // the scanned mesh once loaded (callers re-render via useScanned); its box until then
    const s = scannedNow(o.mesh_url);
    if (s) {
      // follow any later size edit (scale calibration, overlap fix): scale the mesh to the object's current box
      const g = s.geometry.clone();
      g.computeBoundingBox();
      const b = g.boundingBox!, e = new THREE.Vector3().subVectors(b.max, b.min);
      g.scale(o.size[0] / Math.max(e.x, 1e-4), o.size[1] / Math.max(e.y, 1e-4), o.size[2] / Math.max(e.z, 1e-4));
      return g;
    }
    return new THREE.BoxGeometry(o.size[0], o.size[1], o.size[2]);
  }
  if (o.shape === "cylinder" && o.profile && o.profile.length >= 3) {
    // revolved mesh from the measured radius profile (closed at both ends)
    const h = o.size[2];
    const pts = [new THREE.Vector2(0, -h / 2), ...o.profile.map(([z, r]) => new THREE.Vector2(r, z - h / 2)),
      new THREE.Vector2(0, h / 2)];
    const g = new THREE.LatheGeometry(pts, 48);
    g.rotateX(Math.PI / 2);                       // lathe axis y -> world z
    g.computeVertexNormals();
    return g;
  }
  if (o.shape === "box") {
    const r = Math.min(o.size[0], o.size[1], o.size[2]) * 0.12;
    return new RoundedBoxGeometry(o.size[0], o.size[1], o.size[2], 3, r);
  }
  if (o.shape === "cylinder") {
    const g = new THREE.CylinderGeometry(o.size[0] / 2, o.size[0] / 2, o.size[2], 40, 1);
    g.rotateX(Math.PI / 2); // three's cylinder axis is y; MuJoCo's is z
    return g;
  }
  return new THREE.BoxGeometry(o.size[0], o.size[1], o.size[2]);
}

/** Configure a three.js camera to exactly match the capture camera. */
export function applyCaptureCamera(c: THREE.PerspectiveCamera, cam: Camera) {
  const R = cam.R_wc;
  const x = new THREE.Vector3(R[0][0], R[1][0], R[2][0]);
  const y = new THREE.Vector3(-R[0][1], -R[1][1], -R[2][1]);   // OpenCV y-down -> GL y-up
  const z = new THREE.Vector3(-R[0][2], -R[1][2], -R[2][2]);   // OpenCV z-forward -> GL z-back
  const m = new THREE.Matrix4().makeBasis(x, y, z);
  c.quaternion.setFromRotationMatrix(m);
  c.position.set(cam.t_wc[0], cam.t_wc[1], cam.t_wc[2]);
  c.fov = THREE.MathUtils.radToDeg(2 * Math.atan(cam.height / 2 / cam.fy));
  c.aspect = cam.width / cam.height;
  c.near = 0.01;
  c.far = 50;
  c.updateProjectionMatrix();
  c.updateMatrixWorld(true);
}

/** Cumulative path length of object i at every frame (for walk cycles, wheel spin). Cached per sim. */
const distCache = new WeakMap<object, Map<number, Float32Array>>();
export function distanceTrack(sim: Sim, i: number): Float32Array {
  let m = distCache.get(sim);
  if (!m) { m = new Map(); distCache.set(sim, m); }
  let d = m.get(i);
  if (!d) {
    d = new Float32Array(sim.frames.length);
    for (let f = 1; f < sim.frames.length; f++) {
      const a = sim.frames[f - 1][i], b = sim.frames[f][i];
      const step = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
      d[f] = d[f - 1] + (step < 5 ? step : 0);
    }
    m.set(i, d);
  }
  return d;
}
export function distanceAt(sim: Sim, i: number, t: number) {
  const d = distanceTrack(sim, i);
  const f = Math.max(0, Math.min(d.length - 1, t * sim.fps));
  const a = Math.floor(f), b = Math.min(d.length - 1, a + 1);
  return d[a] + (d[b] - d[a]) * (f - a);
}
/** Speed (m/s) and vertical velocity of object id at time t from the recorded kinematics. */
export function velocityAt(sim: Sim, id: string, t: number): [number, number] {
  const v = sim.kinematics?.[id]?.v;
  if (!v) return [0, 0];
  const f = Math.max(0, Math.min(v.length - 1, Math.round(t * sim.fps)));
  const x = v[f];
  return x ? [Math.hypot(x[0], x[1], x[2]), x[2]] : [0, 0];
}
