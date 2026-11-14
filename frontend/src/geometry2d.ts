import type { Camera, SceneObject, Sim } from "./types";

export type Pt = [number, number];

export function project(cam: Camera, p: number[]): Pt | null {
  const R = cam.R_wc, t = cam.t_wc;
  const d = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
  // camera = R^T (p - t)
  const x = R[0][0] * d[0] + R[1][0] * d[1] + R[2][0] * d[2];
  const y = R[0][1] * d[0] + R[1][1] * d[1] + R[2][1] * d[2];
  const z = R[0][2] * d[0] + R[1][2] * d[1] + R[2][2] * d[2];
  if (z < 0.02) return null;
  return [cam.fx * x / z + cam.cx, cam.fy * y / z + cam.cy];
}

function rotate(q: number[], v: number[]): number[] {
  const [qx, qy, qz, qw] = q;
  const ix = qw * v[0] + qy * v[2] - qz * v[1];
  const iy = qw * v[1] + qz * v[0] - qx * v[2];
  const iz = qw * v[2] + qx * v[1] - qy * v[0];
  const iw = -qx * v[0] - qy * v[1] - qz * v[2];
  return [ix * qw + iw * -qx + iy * -qz - iz * -qy, iy * qw + iw * -qy + iz * -qx - ix * -qz, iz * qw + iw * -qz + ix * -qy - iy * -qx];
}

export function convexHull(pts: Pt[]): Pt[] {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o: Pt, a: Pt, b: Pt) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: Pt[] = [], upper: Pt[] = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (const q of p.reverse()) { while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** Image-space silhouette (convex hull) of an object's primitive at the end of the simulation. */
export function silhouetteAtEnd(cam: Camera, o: SceneObject, sim: Sim): Pt[] | null {
  const i = sim.object_ids.indexOf(o.id);
  if (i < 0 || sim.removed_at[o.id] !== undefined) return null;
  const f = sim.frames[sim.frames.length - 1][i];
  const q = [f[3], f[4], f[5], f[6]];
  const local: number[][] = [];
  const [sx, sy, sz] = [o.size[0] / 2, o.size[1] / 2, o.size[2] / 2];
  if (o.shape === "cylinder") {
    for (let k = 0; k < 16; k++) {
      const a = (k / 16) * Math.PI * 2;
      for (const z of [-sz, sz]) local.push([Math.cos(a) * sx, Math.sin(a) * sx, z]);
    }
  } else {
    for (const x of [-sx, sx]) for (const y of [-sy, sy]) for (const z of [-sz, sz]) local.push([x, y, z]);
  }
  const pts: Pt[] = [];
  for (const l of local) {
    const r = rotate(q, l);
    const pp = project(cam, [r[0] + f[0], r[1] + f[1], r[2] + f[2]]);
    if (pp) pts.push(pp);
  }
  return pts.length >= 3 ? convexHull(pts) : null;
}

export function pointInPoly(p: Pt, poly: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > p[1]) !== (yj > p[1]) && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Silhouette of a summoned object's bounding box (body origin at bottom-centre) at the end pose. */
export function boxSilhouetteAtEnd(cam: Camera, sid: string, dims: number[], sim: Sim): Pt[] | null {
  const i = sim.object_ids.indexOf(sid);
  if (i < 0 || sim.removed_at[sid] !== undefined) return null;
  const f = sim.frames[sim.frames.length - 1][i];
  const q = [f[3], f[4], f[5], f[6]];
  const pts: Pt[] = [];
  for (const x of [-dims[0] / 2, dims[0] / 2]) for (const y of [-dims[1] / 2, dims[1] / 2]) for (const z of [0, dims[2]]) {
    const r = rotate(q, [x, y, z]);
    const pp = project(cam, [r[0] + f[0], r[1] + f[1], r[2] + f[2]]);
    if (pp) pts.push(pp);
  }
  return pts.length >= 3 ? convexHull(pts) : null;
}
