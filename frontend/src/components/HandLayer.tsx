/* ✋ Hands in Live AR: your real hand drives the exact physics.
 *
 *   pinch an object  -> grab it (a ghost follows your hand, the arrow shows the throw velocity)
 *   open slowly      -> it is set down where your hand is (dropped from 5 cm)
 *   flick and open   -> it is thrown with your hand's velocity
 *   swipe through it -> it is pushed with your fingertip's velocity
 *
 * Everything becomes ordinary interventions (place / launch / push), so the result is simulated by MuJoCo
 * like any other edit: undo, Studio graphs and measurements all work.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { boxSilhouetteAtEnd, pointInPoly, project, silhouetteAtEnd, type Pt } from "../geometry2d";
import { HAND_LINKS, loadHands, pixelToPlane, trackVelocity, type Sample, type V3 } from "../hands";
import type { Asset, Intervention, Scene, Sim } from "../types";

export interface HandAction { ivs: Intervention[]; note: string; target: string; speed: number; start: number[] }   // start = where it was before the hand

export interface Target {
  id: string; label: string; poly: number[][]; base: V3;  // body origin at the current (end-of-sim) pose
  cz: number; half: V3; originAboveBottom: number;
}

interface Props { scene: Scene; sim: Sim | null; assets: Record<string, Asset>; busy: boolean; onAct: (a: HandAction) => void }

const LIFT = 0.05;              // a grabbed object is lifted 5 cm
const THROW_MIN = 0.35;         // m/s: slower than this when you let go = set it down
const THROW_MAX = 4.0;          // m/s: a desk-sized throw
const PUSH_MIN = 0.2;           // m/s: fingertip speed that counts as a push
const COOLDOWN = 1200;          // ms between hand actions

interface View {
  pts: Pt[] | null; pinch: boolean; hover: Target | null; grab: Target | null;
  ghost: { c: Pt; r: number; from: Pt } | null; arrow: { a: Pt; b: Pt; text: string } | null;
}
const EMPTY: View = { pts: null, pinch: false, hover: null, grab: null, ghost: null, arrow: null };

export function targetsOf(scene: Scene, sim: Sim | null, assets: Record<string, Asset>): Target[] {
  const last = sim?.frames[sim.frames.length - 1];
  const posOf = (id: string): V3 | null => {
    const i = sim ? sim.object_ids.indexOf(id) : -1;
    return last && i >= 0 ? [last[i][0], last[i][1], last[i][2]] : null;
  };
  const out: Target[] = [];
  if (sim?.spawned) {
    for (const [sid, sp] of Object.entries(sim.spawned)) {
      const dims = assets[sp.asset]?.dims ?? sp.proxy?.dims;
      const base = posOf(sid);
      if (!dims || !base || sim.removed_at[sid] !== undefined) continue;
      const poly = boxSilhouetteAtEnd(scene.camera, sid, dims, sim);
      if (poly) out.push({ id: sid, label: sp.label, poly, base, cz: base[2] + dims[2] / 2, half: [dims[0] / 2, dims[1] / 2, dims[2] / 2], originAboveBottom: 0 });
    }
  }
  for (const o of scene.objects) {
    if (sim && sim.removed_at[o.id] !== undefined) continue;
    const poly = sim && sim.moved_ids.includes(o.id) ? silhouetteAtEnd(scene.camera, o, sim) : o.polygon;
    if (!poly || poly.length < 3) continue;
    const base = posOf(o.id) ?? ([o.pos[0], o.pos[1], o.pos[2]] as V3);
    out.push({ id: o.id, label: o.label, poly, base, cz: base[2], half: [o.size[0] / 2, o.size[1] / 2, o.size[2] / 2], originAboveBottom: o.size[2] / 2 });
  }
  return out;
}

/** Mouse drag on an object in the camera view -> a push along the drag on the table plane (longer drag = faster). */
export function dragPush(scene: Scene, t: Target, from: Pt, to: Pt): HandAction | null {
  const a = pixelToPlane(scene.camera, from[0], from[1], t.cz), b = pixelToPlane(scene.camera, to[0], to[1], t.cz);
  if (!a || !b) return null;
  const d = [b[0] - a[0], b[1] - a[1]], L = Math.hypot(d[0], d[1]);
  if (L < 1e-3) return null;
  const speed = Math.min(3, Math.max(0.15, L * 5));
  const off = [a[0] - t.base[0], a[1] - t.base[1]], rmax = 0.8 * Math.min(t.half[0], t.half[1]), r = Math.hypot(off[0], off[1]);
  const k = r > rmax ? rmax / r : 1;
  return { ivs: [{ type: "push", target: t.id, params: { direction: [d[0], d[1], 0], speed, point: [t.base[0] + off[0] * k, t.base[1] + off[1] * k, t.cz] } }],
    note: `Pushed the ${t.label} at ${speed.toFixed(2)} m/s`, target: t.id, speed, start: t.base };
}

const area = (poly: number[][]) => Math.abs(poly.reduce((s, p, i) => { const q = poly[(i + 1) % poly.length]; return s + p[0] * q[1] - q[0] * p[1]; }, 0)) / 2;
function hitTest(ts: Target[], p: Pt, r: number): Target | null {
  const near = (poly: number[][]) => pointInPoly(p, poly) ||
    (r > 0 && [0, 1, 2, 3, 4, 5, 6, 7].some((k) => pointInPoly([p[0] + r * Math.cos(k * Math.PI / 4), p[1] + r * Math.sin(k * Math.PI / 4)], poly)));
  const hits = ts.filter((t) => near(t.poly));
  return hits.length ? hits.reduce((a, b) => (area(b.poly) < area(a.poly) ? b : a)) : null;    // the smallest = the one in front
}
const centroid = (poly: number[][]): Pt => [poly.reduce((s, p) => s + p[0], 0) / poly.length, poly.reduce((s, p) => s + p[1], 0) / poly.length];
const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export default function HandLayer({ scene, sim, assets, busy, onAct }: Props) {
  const W = scene.camera.width, H = scene.camera.height;
  const targets = useMemo(() => targetsOf(scene, sim, assets), [scene, sim, assets]);
  const live = useRef({ targets, scene, busy, onAct });
  live.current = { targets, scene, busy, onAct };
  const [view, setView] = useState<View>(EMPTY);
  const [status, setStatus] = useState<string>("loading");
  const [flash, setFlash] = useState<{ a: Pt; b: Pt; text: string; key: number } | null>(null);

  useEffect(() => {
    let stop = false, raf = 0, lm: HandLandmarker | null = null;
    const g = {
      pinch: false, on: 0, off: 0, lastProc: 0, lastVideoT: -1,
      grab: null as null | { t: Target; off: [number, number]; hist: Sample[]; px0: Pt; moved: number; since: number },
      tip: [] as Sample[], palmH: [] as Sample[], inside: new Set<string>(), cool: 0, closing: 0,
    };
    const act = (a: HandAction, from: Pt, to: Pt) => {
      g.cool = performance.now() + COOLDOWN;
      setFlash({ a: from, b: to, text: a.note, key: Date.now() });
      live.current.onAct(a);
    };

    const release = () => {
      const gr = g.grab!;
      g.grab = null;
      const { scene: sc, targets: ts } = live.current;
      const last = gr.hist[gr.hist.length - 1];
      const v = trackVelocity(gr.hist, last.t) ?? [0, 0];
      const vh = Math.hypot(v[0], v[1]);
      if (gr.moved < 14 && vh < THROW_MIN) return;                 // a pinch that went nowhere: just let go
      const t = gr.t;
      const x = last.p[0] + gr.off[0], y = last.p[1] + gr.off[1];
      // don't put it inside something: if it lands over another object, set it on top of it
      let z = t.base[2] + LIFT;
      for (const o of ts) {
        if (o.id === t.id) continue;
        if (Math.hypot(o.base[0] - x, o.base[1] - y) < Math.max(o.half[0], o.half[1]) + Math.max(t.half[0], t.half[1]) * 0.8)
          z = Math.max(z, o.cz + o.half[2] + t.originAboveBottom + 0.01);
      }
      const pos = [x, y, z];
      const from = project(sc.camera, [t.base[0], t.base[1], t.cz]) ?? centroid(t.poly);
      const to = project(sc.camera, [x, y, z - t.base[2] + t.cz]) ?? from;
      if (vh < THROW_MIN) {
        act({ ivs: [{ type: "place", target: t.id, params: { pos, hold: false, clear: true } }], note: `Set the ${t.label} down (dropped from ${Math.round((z - t.base[2]) * 100)} cm)`, target: t.id, speed: 0, start: t.base }, from, to);
        return;
      }
      const s = Math.min(vh, THROW_MAX) / vh;
      const vel = [v[0] * s, v[1] * s, Math.min(1.5, 0.25 * vh * s)];         // a gentle upward arc: one camera can't see hand height
      const speed = Math.hypot(vel[0], vel[1], vel[2]);
      const group = `hand-${Date.now()}`;
      const end = project(sc.camera, [x + vel[0] * 0.25, y + vel[1] * 0.25, t.cz + LIFT]) ?? to;
      act({ ivs: [{ type: "place", target: t.id, params: { pos, hold: true, clear: true, group } }, { type: "launch", target: t.id, params: { velocity: vel, group } }],
        note: `Threw the ${t.label} at ${speed.toFixed(2)} m/s`, target: t.id, speed, start: t.base }, to, end);
    };

    const tick = () => {
      if (stop) return;
      raf = requestAnimationFrame(tick);
      const now = performance.now();
      if (!lm || now - g.lastProc < 30) return;
      const el = document.querySelector(".live-stage video, .live-stage img[src='/api/remote/stream']") as HTMLVideoElement | HTMLImageElement | null;
      if (!el) return;
      const scripted = (window as unknown as { __fakeHand?: unknown }).__fakeHand !== undefined;
      if (el instanceof HTMLVideoElement) {
        if (el.readyState < 2 || (el.currentTime === g.lastVideoT && !scripted)) return;
        g.lastVideoT = el.currentTime;
      } else if (!el.naturalWidth) return;
      g.lastProc = now;
      // e2e test hook: a scripted hand (21 normalized landmarks) replaces the camera's; the rest of the pipeline is unchanged
      const fake = (window as unknown as { __fakeHand?: { x: number; y: number }[] | null }).__fakeHand;
      let lms: { x: number; y: number }[] | undefined;
      if (scripted) lms = fake ?? undefined;
      else {
        try { lms = lm.detectForVideo(el, now).landmarks[0]; } catch { return; }
        const stats = ((window as unknown as { __handStats?: { frames: number; ms: number } }).__handStats ??= { frames: 0, ms: 0 });
        stats.frames++; stats.ms = performance.now() - now;
      }
      const { scene: sc, targets: ts, busy: b } = live.current;
      if (!lms) {
        if (g.grab) release();
        g.pinch = false; g.on = g.off = 0; g.tip = []; g.palmH = []; g.inside.clear();
        setView(EMPTY);
        return;
      }
      const pts = lms.map((l) => [l.x * W, l.y * H] as Pt);
      const palm = Math.max(1, dist(pts[0], pts[9]));
      const ratio = dist(pts[4], pts[8]) / palm;
      const mid: Pt = [(pts[4][0] + pts[8][0]) / 2, (pts[4][1] + pts[8][1]) / 2];
      // pinch with hysteresis (two frames to switch) so jitter doesn't drop the object
      const wasPinch = g.pinch;
      if (ratio < 0.33) { g.on++; g.off = 0; } else if (ratio > 0.5) { g.off++; g.on = 0; }
      if (ratio < 0.55) g.closing = now;                       // fingers together or closing: not a push
      if (!g.pinch && g.on >= 2) g.pinch = true;
      if (g.pinch && g.off >= 2) g.pinch = false;

      if (g.pinch && !wasPinch && !g.grab && !b && now > g.cool) {
        const t = hitTest(ts, mid, 0.25 * palm);
        const w = t && pixelToPlane(sc.camera, mid[0], mid[1], t.cz);
        if (t && w) g.grab = { t, off: [t.base[0] - w[0], t.base[1] - w[1]], hist: [{ t: now, p: [w[0], w[1]] }], px0: mid, moved: 0, since: now };
      }
      let ghost: View["ghost"] = null, arrow: View["arrow"] = null;
      if (g.grab && g.pinch) {
        const gr = g.grab;
        const w = pixelToPlane(sc.camera, mid[0], mid[1], gr.t.cz);
        if (w && ratio < 0.4) {                               // fingers still together: opening moves the midpoint, not the object
          gr.hist.push({ t: now, p: [w[0], w[1]] });
          while (gr.hist.length > 2 && now - gr.hist[0].t > 400) gr.hist.shift();
        }
        gr.moved = Math.max(gr.moved, dist(mid, gr.px0));
        const l = gr.hist[gr.hist.length - 1].p;
        const c3 = [l[0] + gr.off[0], l[1] + gr.off[1], gr.t.cz + LIFT];
        const c = project(sc.camera, c3), e = project(sc.camera, [c3[0] + Math.max(gr.t.half[0], gr.t.half[1]), c3[1], c3[2]]);
        if (c) ghost = { c, r: e ? Math.max(10, dist(c, e)) : 20, from: project(sc.camera, [gr.t.base[0], gr.t.base[1], gr.t.cz]) ?? centroid(gr.t.poly) };
        const v = trackVelocity(gr.hist, now);
        const vh = v ? Math.hypot(v[0], v[1]) : 0;
        if (c && v && vh >= THROW_MIN) {
          const s = Math.min(vh, THROW_MAX) / vh;
          const b2 = project(sc.camera, [c3[0] + v[0] * s * 0.25, c3[1] + v[1] * s * 0.25, c3[2]]);
          if (b2) {
            const L = dist(c, b2), maxL = Math.min(W, H) * 0.18;     // direction is what matters on screen; the number gives the size
            const k = L > maxL ? maxL / L : 1;
            arrow = { a: c, b: [c[0] + (b2[0] - c[0]) * k, c[1] + (b2[1] - c[1]) * k], text: `${(vh * s).toFixed(2)} m/s` };
          }
        }
        if (now - gr.since > 10000) g.grab = null;           // held too long: give up quietly
      } else if (g.grab && !g.pinch) {
        release();
      }

      // push: an open hand sweeping its index fingertip into an object. The sweep speed is the PALM's
      // (the whole hand moving), so closing the fingers for a pinch never counts as a push.
      const tip = pts[8];
      g.tip.push({ t: now, p: tip });
      g.palmH.push({ t: now, p: pts[9] });
      while (g.tip.length > 2 && now - g.tip[0].t > 300) g.tip.shift();
      while (g.palmH.length > 2 && now - g.palmH[0].t > 300) g.palmH.shift();
      const insideNow = new Set<string>();
      if (!g.pinch && !g.grab && now - g.closing > 250) {
        for (const t of ts) {
          if (!pointInPoly(tip, t.poly)) continue;
          insideNow.add(t.id);
          if (g.inside.has(t.id) || b || now <= g.cool) continue;
          const pv = trackVelocity(g.palmH, now, 100);
          if (!pv) continue;
          const pa = pixelToPlane(sc.camera, tip[0] - pv[0] * 0.1, tip[1] - pv[1] * 0.1, t.cz), pb = pixelToPlane(sc.camera, tip[0], tip[1], t.cz);
          if (!pa || !pb) continue;
          const v = [(pb[0] - pa[0]) / 0.1, (pb[1] - pa[1]) / 0.1];
          const spd = Math.hypot(v[0], v[1]);
          if (spd < PUSH_MIN) continue;
          // contact point: where the fingertip meets it, kept inside the body so the push can spin it but not miss it
          const off = [pb[0] - t.base[0], pb[1] - t.base[1]], rmax = 0.8 * Math.min(t.half[0], t.half[1]), r = Math.hypot(off[0], off[1]);
          const k = r > rmax ? rmax / r : 1;
          const speed = Math.min(3, Math.max(0.15, spd));
          const to = project(sc.camera, [t.base[0] + v[0] / spd * 0.12, t.base[1] + v[1] / spd * 0.12, t.cz]) ?? tip;
          act({ ivs: [{ type: "push", target: t.id, params: { direction: [v[0], v[1], 0], speed, point: [t.base[0] + off[0] * k, t.base[1] + off[1] * k, t.cz] } }],
            note: `Pushed the ${t.label} at ${speed.toFixed(2)} m/s`, target: t.id, speed, start: t.base }, tip, to);
          break;
        }
      }
      g.inside = insideNow;
      setView({ pts, pinch: g.pinch, hover: g.grab ? null : hitTest(ts, mid, 0.25 * palm), grab: g.grab?.t ?? null, ghost, arrow });
    };

    loadHands().then((x) => { if (stop) return; lm = x; setStatus("on"); raf = requestAnimationFrame(tick); })
      .catch((e) => setStatus(`Hand tracking unavailable: ${(e as Error).message}`));
    return () => { stop = true; cancelAnimationFrame(raf); };
  }, [W, H]);

  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash((f) => (f?.key === flash.key ? null : f)), 2200);
    return () => clearTimeout(t);
  }, [flash]);

  const lit = view.grab ?? view.hover;
  const fs = Math.max(12, W / 70);
  return (
    <>
      <svg className="hand-layer" viewBox={`0 0 ${W} ${H}`}>
        <defs>
          <marker id="hand-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#fbbf24" />
          </marker>
        </defs>
        {lit && <polygon className={`hand-target ${view.grab ? "held" : ""}`} points={lit.poly.map((q) => q.join(",")).join(" ")} />}
        {view.ghost && (
          <g className="hand-ghost">
            <line x1={view.ghost.from[0]} y1={view.ghost.from[1]} x2={view.ghost.c[0]} y2={view.ghost.c[1]} />
            <circle cx={view.ghost.c[0]} cy={view.ghost.c[1]} r={view.ghost.r} />
          </g>
        )}
        {view.arrow && (
          <g className="hand-vel">
            <line x1={view.arrow.a[0]} y1={view.arrow.a[1]} x2={view.arrow.b[0]} y2={view.arrow.b[1]} markerEnd="url(#hand-arrow)" />
            <text x={Math.min(W - fs * 6, Math.max(fs * 0.5, view.arrow.b[0] + 8))} y={Math.max(fs * 1.5, view.arrow.b[1] - 8)} fontSize={fs}>{view.arrow.text}</text>
          </g>
        )}
        {flash && (
          <g className="hand-vel flash" key={flash.key}>
            <line x1={flash.a[0]} y1={flash.a[1]} x2={flash.b[0]} y2={flash.b[1]} markerEnd="url(#hand-arrow)" />
          </g>
        )}
        {view.pts && (
          <g className={`hand-skel ${view.pinch ? "pinch" : ""}`}>
            {HAND_LINKS.map(([a, b]) => <line key={`${a}-${b}`} x1={view.pts![a][0]} y1={view.pts![a][1]} x2={view.pts![b][0]} y2={view.pts![b][1]} />)}
            {view.pts.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={i === 4 || i === 8 ? 5 : 3} className={i === 4 || i === 8 ? "tip" : ""} />)}
            <circle className="pinch-ring" cx={(view.pts[4][0] + view.pts[8][0]) / 2} cy={(view.pts[4][1] + view.pts[8][1]) / 2} r={view.pinch ? 9 : 14} />
          </g>
        )}
      </svg>
      <div className="hand-pill">
        {status === "loading" ? "✋ loading hand tracking…"
          : status !== "on" ? status
          : flash ? `✋ ${flash.text}`
          : view.grab ? `✋ holding the ${view.grab.label} · flick and open to throw, open slowly to set down`
          : view.hover ? `✋ pinch to grab the ${view.hover.label} · or swipe through it to push`
          : view.pts ? "✋ pinch an object to grab it · flick to throw · swipe to push"
          : "✋ show your hand to the camera"}
      </div>
    </>
  );
}
