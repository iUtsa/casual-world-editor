/* 🖐 Touch: your real hand as physics, live.
 *
 * Each video frame: MediaPipe's 21 joints (pixels + the hand's real shape in metres) -> the hand in 3D, in the table's
 * frame (hands.ts handInCamera, smoothed with a One Euro filter) -> /ws/hand, stamped with the camera's time. The
 * server moves 21 bone capsules to match and steps the physics; every object's pose comes back and every view shows it
 * (poses.ts LIVE). Nothing is a gesture rule: fingers push, tip and flick things by contact. When the hand leaves for
 * 1.5 s the session ends and becomes an ordinary action (undo, replay, measurements), replayed exactly.
 */
import { useEffect, useRef, useState } from "react";
import type { HandLandmarker } from "@mediapipe/tasks-vision";
import { project, type Pt } from "../geometry2d";
import { HAND_LINKS, OneEuro, cameraToWorld, handInCamera, loadHands, type V3 } from "../hands";
import { LIVE, bumpLive } from "../poses";
import type { Scene, Sim } from "../types";

interface Props { scene: Scene; onSim: (s: Sim) => void; onDone: (s: Sim) => void }
interface Touch { id: string; force: number; pos: number[] }

const IDLE_END_MS = 1500;      // hand gone this long after touching something: the session ends

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/hand`;
}

export default function LiveTouch({ scene, onSim, onDone }: Props) {
  const W = scene.camera.width, H = scene.camera.height;
  const [pts, setPts] = useState<Pt[] | null>(null);
  const [touch, setTouch] = useState<Touch[]>([]);
  const [status, setStatus] = useState("loading hand tracking…");
  const live = useRef({ scene, onSim, onDone });
  live.current = { scene, onSim, onDone };

  useEffect(() => {
    let stop = false, raf = 0, lm: HandLandmarker | null = null, lastVideoT = -1;
    let ws: WebSocket | null = null, inFlight = false, ending = false, touched = false, lastSeen = 0;
    const filter = new OneEuro(1.2, 6);
    const open = () => {
      ws = new WebSocket(wsUrl());
      inFlight = false; ending = false; touched = false;
      ws.onmessage = (ev) => {
        inFlight = false;
        dbg.recv = (dbg.recv as number) + 1;
        const m = JSON.parse(ev.data);
        if (m.error) { setStatus(`🖐 ${m.error}`); return; }
        if (m.sim) live.current.onSim(m.sim as Sim);
        if (m.done) {
          LIVE.active = false; LIVE.poses = {}; LIVE.moved = new Set(); bumpLive();
          setTouch([]); live.current.onDone(m.done as Sim);
          setStatus("🖐 Touch: put your hand in view and push things");
          ws?.close(); if (!stop) setTimeout(open, 50);
          return;
        }
        if (m.poses) {
          LIVE.poses = m.poses; LIVE.active = true;
          const before = LIVE.moved.size;
          (m.moved as string[] | undefined)?.forEach((id) => LIVE.moved.add(id));
          if (LIVE.moved.size !== before) bumpLive();
          const t = (m.touch ?? []) as Touch[];
          dbg.lastMoved = ((m.moved ?? []) as string[]).join(","); dbg.touchN = t.length;
          dbg.touchMax = Math.max(dbg.touchMax as number || 0, t.length);
          if (t.length) touched = true;
          setTouch(t);
        }
      };
      ws.onclose = () => { inFlight = false; };
    };
    const dbg: Record<string, number | string> = { sent: 0, recv: 0, frames: 0, solved: 0, ticks: 0, noEl: 0 };
    (window as unknown as { __touchDbg?: unknown }).__touchDbg = dbg;   // e2e diagnostics
    const send = (joints: V3[] | null, tMs: number) => {
      if (!ws || ws.readyState !== WebSocket.OPEN || inFlight || ending) return;
      inFlight = true;                                   // one message in flight: no backlog, always fresh
      ws.send(JSON.stringify({ joints, t: tMs / 1000 }));
      dbg.sent = (dbg.sent as number) + 1;
    };

    const tick = () => {
      if (stop) return;
      raf = requestAnimationFrame(tick);
      dbg.ticks = (dbg.ticks as number) + 1;
      if (!lm) return;
      const el = document.querySelector(".live-stage video, .live-stage img[src='/api/remote/stream']") as HTMLVideoElement | HTMLImageElement | null;
      if (!el) { dbg.noEl = (dbg.noEl as number) + 1; return; }
      dbg.rs = el instanceof HTMLVideoElement ? el.readyState : "img";
      if (el instanceof HTMLVideoElement) {
        const scripted = (window as unknown as { __fakeTouch?: unknown }).__fakeTouch !== undefined;
        // a scripted hand (e2e) replaces the camera frame entirely, so it doesn't need a decoded frame
        if (!scripted && (el.readyState < 2 || el.currentTime === lastVideoT)) return;
        lastVideoT = el.currentTime;
      } else if (!el.naturalWidth && (window as unknown as { __fakeTouch?: unknown }).__fakeTouch === undefined) return;
      const now = performance.now();
      // e2e hook: a scripted hand (normalized pixels + metric world landmarks) replaces detection; the rest is real
      const fake = (window as unknown as { __fakeTouch?: { px: { x: number; y: number }[]; world: { x: number; y: number; z: number }[] } | null }).__fakeTouch;
      let px, wl;
      if (fake !== undefined) { px = fake?.px; wl = fake?.world; }
      else {
        let res;
        try { res = lm.detectForVideo(el, now); } catch { return; }
        px = res.landmarks?.[0]; wl = res.worldLandmarks?.[0];
      }
      const cam = live.current.scene.camera;
      if (!px || !wl) {
        setPts(null);
        filter.reset();
        if (LIVE.active) send(null, now);
        // end once the hand has gone after touching anything, or after anything moved (a fast pass can push an object
        // between two replies without a reply ever landing mid-contact)
        if ((touched || LIVE.moved.size > 0) && !ending && now - lastSeen > IDLE_END_MS && ws?.readyState === WebSocket.OPEN) {
          ending = true; ws.send(JSON.stringify({ end: true })); setStatus("🖐 replaying what your hand did…");
        }
        return;
      }
      lastSeen = now;
      dbg.frames = (dbg.frames as number) + 1;
      const pix = px.map((p) => ({ x: p.x * cam.width, y: p.y * cam.height }));
      const Jc = handInCamera(wl, pix, cam);
      if (!Jc) return;
      dbg.solved = (dbg.solved as number) + 1;
      const flat = Jc.map((p) => cameraToWorld(cam, p)).flat();
      const sm = filter.filter(flat, now);
      const joints: V3[] = [];
      for (let k = 0; k < 21; k++) joints.push([sm[3 * k], sm[3 * k + 1], sm[3 * k + 2]]);
      dbg.palm = joints[9].map((x) => x.toFixed(3)).join(",");
      setPts(pix.map((p) => [p.x, p.y] as Pt));
      send(joints, now);
    };

    open();
    loadHands().then((x) => { if (stop) return; lm = x; setStatus("🖐 Touch: put your hand in view and push things"); raf = requestAnimationFrame(tick); })
      .catch((e) => setStatus(`Hand tracking unavailable: ${(e as Error).message}`));
    return () => {
      stop = true; cancelAnimationFrame(raf);
      if (ws?.readyState === WebSocket.OPEN && LIVE.active) ws.send(JSON.stringify({ end: true }));
      ws?.close();
      LIVE.active = false; LIVE.poses = {}; LIVE.moved = new Set(); bumpLive();
    };
  }, []);

  const fs = Math.max(12, W / 70);
  return (
    <>
      <svg className="hand-layer" viewBox={`0 0 ${W} ${H}`}>
        {touch.map((t) => {
          const p = project(scene.camera, t.pos);
          if (!p) return null;
          const r = Math.min(40, 10 + 6 * Math.sqrt(t.force));
          return (
            <g key={t.id} className="touch-mark">
              <circle cx={p[0]} cy={p[1]} r={r} />
              <text x={p[0] + r + 4} y={p[1] - 4} fontSize={fs}>{t.force.toFixed(1)} N</text>
            </g>
          );
        })}
        {pts && (
          <g className="hand-skel live">
            {HAND_LINKS.map(([a, b]) => <line key={`${a}-${b}`} x1={pts[a][0]} y1={pts[a][1]} x2={pts[b][0]} y2={pts[b][1]} />)}
            {pts.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={i % 4 === 0 ? 5 : 3} className={i % 4 === 0 && i > 0 ? "tip" : ""} />)}
          </g>
        )}
      </svg>
      <div className="hand-pill">{touch.length ? `🖐 touching ${touch.map((t) => scene.objects.find((o) => o.id === t.id)?.label ?? t.id).join(", ")}` : status}</div>
    </>
  );
}
