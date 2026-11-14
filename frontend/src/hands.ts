/* Hand tracking for Live AR: MediaPipe HandLandmarker (free, runs on this Mac's GPU, model served locally)
 * plus the geometry that turns a hand in the camera image into a point / velocity on the real table.
 *
 * A single camera cannot see how high the hand is, so hand motion is read on the horizontal plane at the
 * height of the object being touched. The physics after that (MuJoCo) is exact; the gesture -> velocity
 * mapping is the interpretation, and it is shown on screen (arrow + m/s) before and after every action.
 */
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import type { Camera } from "./types";

let loading: Promise<HandLandmarker> | null = null;

export function loadHands(): Promise<HandLandmarker> {
  if (!loading) {
    loading = (async () => {
      const files = await FilesetResolver.forVisionTasks("/mediapipe");
      const opts = (delegate: "GPU" | "CPU") => ({
        baseOptions: { modelAssetPath: "/mediapipe/hand_landmarker.task", delegate },
        runningMode: "VIDEO" as const, numHands: 1,
        minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
      });
      try { return await HandLandmarker.createFromOptions(files, opts("GPU")); }
      catch { return await HandLandmarker.createFromOptions(files, opts("CPU")); }
    })();
    loading.catch(() => { loading = null; });
  }
  return loading;
}

export const HAND_LINKS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

export type V3 = [number, number, number];

/** World point where the camera ray through pixel (u, v) meets the horizontal plane z = z0 (z-up world). */
export function pixelToPlane(cam: Camera, u: number, v: number, z0: number): V3 | null {
  const R = cam.R_wc, t = cam.t_wc;
  const c = [(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1];
  const d = [0, 1, 2].map((r) => R[r][0] * c[0] + R[r][1] * c[1] + R[r][2] * c[2]);
  if (Math.abs(d[2]) < 1e-6) return null;
  const s = (z0 - t[2]) / d[2];
  if (!(s > 0) || s > 30) return null;
  return [t[0] + s * d[0], t[1] + s * d[1], z0];
}

export interface Sample { t: number; p: number[] }

/** Velocity (units/s) over roughly the last `win` ms of a time-stamped track. */
export function trackVelocity(h: Sample[], now: number, win = 120): number[] | null {
  if (h.length < 2) return null;
  const last = h[h.length - 1];
  if (now - last.t > 160) return null;                     // stale: the hand left a while ago
  let first = h[h.length - 2];
  for (let i = h.length - 2; i >= 0; i--) { first = h[i]; if (last.t - h[i].t >= win) break; }
  const dt = (last.t - first.t) / 1000;
  if (dt < 0.03) return null;
  return last.p.map((x, k) => (x - first.p[k]) / dt);
}

// ---------------------------------------------------------------- the hand in 3D (metres, table frame)
/** MediaPipe's world landmarks are the hand's real shape in metres, centred on the hand and in camera-aligned axes
 *  (x right, y down, z away). Only where the hand IS is unknown: the translation t that makes (world + t) project onto
 *  the 21 detected pixels. That is linear: for each joint, fx(X+tx) = (u-cx)(Z+tz) and fy(Y+ty) = (v-cy)(Z+tz).
 *  Returns the 21 joints in the OpenCV camera frame, or null if the solve is degenerate. */
export function handInCamera(world: { x: number; y: number; z: number }[], px: { x: number; y: number }[], cam: Camera): V3[] | null {
  if (world.length !== 21 || px.length !== 21) return null;
  // normal equations of the 42 x 3 least-squares problem A t = b
  const M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], r = [0, 0, 0];
  const add = (a: number[], b: number) => {
    for (let i = 0; i < 3; i++) { r[i] += a[i] * b; for (let j = 0; j < 3; j++) M[i][j] += a[i] * a[j]; }
  };
  for (let i = 0; i < 21; i++) {
    const du = px[i].x - cam.cx, dv = px[i].y - cam.cy, W = world[i];
    add([cam.fx, 0, -du], du * W.z - cam.fx * W.x);
    add([0, cam.fy, -dv], dv * W.z - cam.fy * W.y);
  }
  const t = solve3(M, r);
  if (!t || !(t[2] > 0.05 && t[2] < 5)) return null;
  return world.map((W) => [W.x + t[0], W.y + t[1], W.z + t[2]] as V3);
}

function solve3(A: number[][], b: number[]): number[] | null {
  const [a, c, d] = A, det = a[0] * (c[1] * d[2] - c[2] * d[1]) - a[1] * (c[0] * d[2] - c[2] * d[0]) + a[2] * (c[0] * d[1] - c[1] * d[0]);
  if (Math.abs(det) < 1e-12) return null;
  const col = (k: number) => A.map((row, i) => row.map((x, j) => (j === k ? b[i] : x)));
  const dt = (m: number[][]) => m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  return [dt(col(0)) / det, dt(col(1)) / det, dt(col(2)) / det];
}

/** Camera-frame point -> table (world) frame, z up. */
export function cameraToWorld(cam: Camera, p: V3): V3 {
  const R = cam.R_wc, t = cam.t_wc;
  return [0, 1, 2].map((i) => R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i]) as V3;
}

/** One Euro filter (Casiez et al., CHI 2012): smooth when the hand is still, little lag when it moves fast. */
export class OneEuro {
  private x: number[] | null = null; private dx: number[] | null = null; private t = 0;
  constructor(private minCutoff = 1.5, private beta = 8, private dCutoff = 1.0) {}
  private static alpha(cutoff: number, dt: number) { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); }
  reset() { this.x = null; this.dx = null; }
  filter(v: number[], tMs: number): number[] {
    if (!this.x || !this.dx) { this.x = v.slice(); this.dx = v.map(() => 0); this.t = tMs; return v.slice(); }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    this.t = tMs;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    const dxNew = v.map((x, i) => (x - this.x![i]) / dt);
    this.dx = this.dx.map((d, i) => d + ad * (dxNew[i] - d));
    const speed = Math.sqrt(this.dx.reduce((s, d) => s + d * d, 0) / this.dx.length);
    const a = OneEuro.alpha(this.minCutoff + this.beta * speed, dt);
    this.x = this.x.map((x, i) => x + a * (v[i] - x));
    return this.x.slice();
  }
}
