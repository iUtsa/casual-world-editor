/* Live AR: the camera stays live, a chat sits beside it, and experiments appear IN the camera view.
 *
 *  - The space is mapped automatically when the camera starts (depth + segmentation + labels), so real
 *    objects are known in 3D and tagged in AR.
 *  - Things that fit in your space run IN the real scene (they collide with your real mug, fall off your
 *    real table): rendered by the AR compositor (Stage) with the real camera's intrinsics and pose.
 *  - Anything too big or imaginary (whales, elephants, a castle) runs at its true size in its own world and
 *    is shown as a hologram diorama standing on your table — the physics is real-scale, the display is a
 *    scale model, and the scale is shown.
 *  - Circuits and planets appear as hologram panels over the camera.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { clock } from "../clock";
import { applyCaptureCamera, geometryFor, observedMatrix } from "../poses";
import { useScanned } from "../scanned";
import Stage, { type SourceView } from "./Stage";
import { Effects, Labels, Links, Thing, Trails, experimentBounds } from "./ExperimentView";
import CircuitView from "./CircuitView";
import SpaceView from "./SpaceView";
import VoiceButton from "./VoiceButton";
import DiagramButton from "./DiagramButton";
import type { AnyRun, Asset, DiagramReading, ExpRun, ResultRow, Scene, ScenePayload, Sim } from "../types";
import TheoryCheck from "./TheoryCheck";
import HandLayer, { dragPush, targetsOf, type HandAction } from "./HandLayer";
import Twin3D from "./Twin3D";
import ObjectCard from "./ObjectCard";
import type { Pt } from "../geometry2d";
import { api, type SweepJob } from "../api";

/** Which camera this page last mapped from (module scope: survives tab switches, resets on reload). */
const MAPPED = { kind: null as SourceView["kind"] | null };

export interface ChatMsg { id: string; role: "user" | "ai"; text: string; tiles?: ResultRow[]; results?: ResultRow[]; runId?: string; why?: string[]; changes?: string[]; error?: boolean;
  reading?: DiagramReading }

const STARTERS_SCENE = ["drop a baseball on the table", "two baseballs collide", "roll a steel ball across the table", "throw a tennis ball at it"];
const STARTERS_ANY = ["two blue whales fight underwater", "a dragon crashes into a castle", "a man pulls a truck with a rope", "a pendulum with a 1 m string",
  "a 9 V battery with two bulbs in parallel", "jupiter collides with moon"];
const FOLLOW = ["what if it was twice as heavy?", "now on the Moon", "make it 3 times faster", "why did that happen?", "how hard did they hit?"];

interface Props {
  scene: Scene | null; payload: ScenePayload | null; sim: Sim | null; run: AnyRun | null; assets: Record<string, Asset>;
  source: SourceView; setSource: (s: SourceView, hfov?: number) => void; frameKey: string; busy: string | null;
  staleSceneId: string | null;   // the scene already on the server when this page opened: not what this camera sees
  phoneOn: boolean; phoneFps: number;
  selected: string | null; onSelect: (id: string | null) => void;
  messages: ChatMsg[]; onSend: (text: string) => void; onScan: (blob: Blob | null) => void; onOpenStudio: () => void; onPhone: () => void;
  onReplay: () => void; onReading: (r: DiagramReading) => void; onRunReading: (r: DiagramReading, text: string) => void;
  onHand: (a: HandAction) => void;
  onBall: () => void;
  onSweepDone: (job: SweepJob) => Promise<void>; hfov: number;
}

const NO_PICTURE = "There's no camera picture yet. Click the camera icon in the browser's address bar and allow it for this page "
  + "(macOS may also need System Settings › Privacy & Security › Camera for this browser), then try again — or use your phone.";

export default function LiveAR(p: Props) {
  const { run, source } = p;
  // a laptop-camera scene left from before this page opened is hidden until this page maps its own;
  // a phone scan is always shown (the phone may have scanned before this page was opened or reloaded)
  const stale = !!p.scene && p.scene.scene_id === p.staleSceneId && !p.scene.source.startsWith("phone");
  const scene = p.scene && !stale ? p.scene : null;
  const video = useRef<HTMLVideoElement>(null);
  const [view, setView] = useState<"camera" | "3d">("camera");
  const [hands, setHands] = useState(false);
  const [camErr, setCamErr] = useState<string | null>(null);
  const [camTry, setCamTry] = useState(0);
  const [aspect, setAspect] = useState(16 / 9);
  const [slowCam, setSlowCam] = useState(false);
  const [menu, setMenu] = useState(false);
  const live = source.kind === "phone" ? p.phoneOn : source.kind === "camera" && !!source.stream;   // a moving picture to track hands in

  const switchTo = (kind: "camera" | "phone") => {
    setMenu(false);
    if (kind === "phone") {
      if (!p.phoneOn && !(scene && scene.source.startsWith("phone"))) { p.onPhone(); return; }   // not connected yet: show the QR code
      source.stream?.getTracks().forEach((t) => t.stop());
      setCamErr(null);
      p.setSource({ kind: "phone" }, 66);
    } else {
      setCamErr(null); setSlowCam(false); setCamTry((n) => n + 1);
      if (source.kind !== "camera") p.setSource({ kind: "camera" });
    }
  };
  // a scan made on the phone, or the phone starting to stream: the phone is the camera now
  const phoneScene = !!scene && scene.source.startsWith("phone");
  useEffect(() => {
    if (!phoneScene || source.kind === "phone") return;
    MAPPED.kind = "phone";
    source.stream?.getTracks().forEach((t) => t.stop());
    p.setSource({ kind: "phone" }, 66);
  }, [phoneScene, source.kind]);   // eslint-disable-line react-hooks/exhaustive-deps
  const wasPhone = useRef(p.phoneOn);
  useEffect(() => {
    if (p.phoneOn && !wasPhone.current && source.kind !== "phone") switchTo("phone");
    wasPhone.current = p.phoneOn;
  }, [p.phoneOn]);   // eslint-disable-line react-hooks/exhaustive-deps

  // the laptop camera: start it (again, on "try again"), and say why if there's no picture
  useEffect(() => {
    if (source.kind === "phone" || source.stream) return;     // anything but the phone: the laptop camera is the default
    let cancelled = false;
    navigator.mediaDevices?.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: "environment" } }, audio: false })
      .then((stream) => { if (!cancelled) p.setSource({ kind: "camera", stream }); })
      .catch((e) => { if (!cancelled) setCamErr(`The laptop camera isn't available (${(e as Error).message}).`); });
    return () => { cancelled = true; };
  }, [source.kind, camTry]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (source.kind === "phone" || source.stream) { setSlowCam(false); return; }
    const t = setTimeout(() => setSlowCam(true), 5000);
    return () => clearTimeout(t);
  }, [source.kind, source.stream, camTry]);
  useEffect(() => { if (live) setCamErr(null); }, [live]);
  useEffect(() => {
    const v = video.current;
    if (v && source.kind === "camera" && source.stream && v.srcObject !== source.stream) { v.srcObject = source.stream; v.play().catch(() => {}); }
  });

  // map automatically whenever a camera starts giving pictures that haven't been mapped yet
  useEffect(() => {
    if (p.busy || MAPPED.kind === source.kind || !live) return;
    const t = setTimeout(async () => {
      if (source.kind === "phone") { MAPPED.kind = "phone"; p.onScan(null); return; }
      const v = liveVideoEl();
      if (!v || !v.videoWidth) return;
      MAPPED.kind = "camera";
      const blob = await grab(v);
      if (blob) p.onScan(blob);
    }, 1500);
    return () => clearTimeout(t);
  });

  // 3D scan from a video moving around the table (iPhone recording or any MP4/MOV): real meshes, all sides
  const sweepInput = useRef<HTMLInputElement>(null);
  const [sweep, setSweep] = useState<SweepJob | null>(null);
  const startSweep = async (file: File) => {
    try {
      setSweep({ id: "", state: "running", stage: "uploading", frac: 0, message: `uploading ${(file.size / 1e6).toFixed(0)} MB`, log: [] });
      let job = await api.sweep(file, file.name, p.hfov);
      setSweep(job);
      while (job.state === "running") {
        await new Promise((r) => setTimeout(r, 800));
        job = await api.sweepStatus(job.id);
        setSweep(job);
      }
      if (job.state === "done") {
        MAPPED.kind = source.kind;                 // the camera now shows a 3D-scanned world: don't auto-map over it
        await p.onSweepDone(job);
        setSweep(null);
      }
    } catch (e) {
      setSweep({ id: "", state: "error", stage: "", frac: 0, message: "", log: [], error: (e as Error).message });
    }
  };

  const capture = async () => {
    if (source.kind === "phone") {
      if (!p.phoneOn) { setCamErr("Your phone isn't streaming. Open the phone page (scan the QR code) and keep it open."); p.onPhone(); return; }
      MAPPED.kind = "phone"; p.onScan(null); return;
    }
    const b = await grab(liveVideoEl());
    if (b) { MAPPED.kind = "camera"; setCamErr(null); p.onScan(b); } else setCamErr(NO_PICTURE);
  };

  // hands: only in the camera view, and the phone's auto-rescan pauses (a hand in view is not a changed scene)
  const toggleHands = () => {
    const on = !hands;
    setHands(on);
    if (on) setView("camera");
    fetch("/api/live/enabled", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: !on }) }).catch(() => {});
  };
  useEffect(() => () => { fetch("/api/live/enabled", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: true }) }).catch(() => {}); }, []);

  const mech = run && (!run.domain || run.domain === "mechanics") ? (run as ExpRun) : null;
  const diorama = mech && mech.setting !== "scene" ? mech : null;
  const panel = run && (run.domain === "circuit" || run.domain === "gravity") ? run : null;
  const sceneRun = mech && mech.setting === "scene";
  // the AR session's own simulation shows in the camera too (hand/mouse pushes and throws), not only chat runs
  const shownSim = sceneRun || !run ? p.sim : null;
  const stageSource: SourceView = source.kind === "phone" && !p.phoneOn ? { kind: "none" } : source;
  const camAspect = scene ? scene.camera.width / scene.camera.height : aspect;
  const mapping = !!p.busy?.startsWith("Mapping");
  const targets = useMemo(() => (scene ? targetsOf(scene, shownSim, p.assets) : []), [scene, shownSim, p.assets]);
  const onPush = (id: string, from: Pt, to: Pt) => {
    const t = targets.find((x) => x.id === id);
    const a = t && scene ? dragPush(scene, t, from, to) : null;
    if (a) p.onHand(a);
  };
  const sourceLabel = source.kind === "phone" ? `📱 Phone · ${p.phoneOn ? `live${p.phoneFps ? ` ${Math.round(p.phoneFps)} fps` : ""}` : "not streaming"}`
    : `💻 Laptop camera${source.stream ? "" : " · off"}`;
  const needCamera = !scene && !p.busy && ((source.kind !== "phone" && (camErr || slowCam)) || (source.kind === "phone" && !p.phoneOn));

  return (
    <div className="live">
      <section className="live-stage-wrap">
        <div className="live-stage" style={{ aspectRatio: `${camAspect}` }}>
          {scene ? (
            <div className={view === "3d" ? "hidden-layer" : ""} style={{ position: "absolute", inset: 0 }}>
              <Stage scene={scene} sim={shownSim} source={stageSource} liveBackground={true} tool="push" selected={p.selected}
                mergeSet={[]} showFit={false} showGhosts={false} showOutlines={!run} frameKey={p.frameKey}
                onPick={p.onSelect} onPush={onPush} onAdd={() => {}} sourceVideoRef={() => {}} assets={p.assets} onPlace={() => {}} showVectors={true} />
            </div>
          ) : (
            <div className="stage">
              {source.kind === "camera" && <video ref={video} className="layer" muted playsInline autoPlay onLoadedMetadata={(e) => setAspect(e.currentTarget.videoWidth / e.currentTarget.videoHeight)} />}
              {source.kind === "phone" && <img className="layer" src="/api/remote/stream" alt="phone camera" />}
            </div>
          )}
          {scene && view === "3d" && (
            <Twin3D scene={scene} sim={shownSim} frameUrl={`/api/frame.jpg?k=${p.frameKey}`} idsUrl={`/api/idmap.png?k=${p.frameKey}`}
              assets={p.assets} selected={p.selected} onPick={p.onSelect} />
          )}
          {scene && view === "camera" && hands && live && <HandLayer scene={scene} sim={shownSim} assets={p.assets} busy={!!p.busy} onAct={p.onHand} />}
          {scene && view === "camera" && diorama && <ARDiorama run={diorama} scene={scene} assets={p.assets} />}
          {panel && view === "camera" && (
            <div className="holo-panel">
              <div className="holo-title">{panel.domain === "circuit" ? "⚡ Circuit hologram" : "🪐 Gravity hologram"}</div>
              {panel.domain === "circuit" ? <CircuitView run={panel} resetKey={0} /> : <SpaceView run={panel} resetKey={0} compact />}
            </div>
          )}

          <div className="live-top">
            {scene && (
              <div className="view-toggle">
                <button className={view === "camera" ? "on" : ""} onClick={() => setView("camera")}>📷 Camera</button>
                <button className={view === "3d" ? "on" : ""} onClick={() => { setView("3d"); if (hands) toggleHands(); }}>🧊 3D</button>
              </div>
            )}
            <span className={`pill dark ${scene && !mapping ? "" : "pulse"}`}>
              {mapping ? "◌ Mapping your space…" : scene ? `◉ ${scene.objects.length} real object${scene.objects.length === 1 ? "" : "s"} mapped` : live ? "◌ camera on" : "◌ no camera yet"}
            </span>
            {diorama && view === "camera" && <span className="pill dark warmth">hologram · {scaleText(diorama, scene)}</span>}
            <div className="grow" />
            <div className="source-chip">
              <button onClick={() => setMenu((m) => !m)} title="Choose the camera">{sourceLabel} ▾</button>
              {menu && (
                <div className="source-menu" onMouseLeave={() => setMenu(false)}>
                  <button onClick={() => switchTo("camera")}>💻 This laptop's camera</button>
                  <button onClick={() => switchTo("phone")}>📱 My phone {p.phoneOn ? "(streaming)" : "(show QR code)"}</button>
                </div>
              )}
            </div>
          </div>

          {mapping && <MappingProgress />}
          {sweep && <SweepProgress job={sweep} onClose={() => setSweep(null)} />}
          {p.busy && !mapping && <div className="live-busy"><div className="spinner" /><span>{p.busy}</span></div>}

          {needCamera && (
            <div className="setup-card">
              <h3>{source.kind === "phone" ? "Waiting for your phone…" : "Choose a camera"}</h3>
              <p className="muted">{source.kind === "phone" ? "Open the phone page from the QR code and keep it open — it streams to this Mac by itself."
                : camErr ?? NO_PICTURE}</p>
              <div className="setup-choices">
                <button onClick={() => switchTo("phone")}><span>📱</span><b>Use my phone</b><small>best: point it down at your desk</small></button>
                <button onClick={() => switchTo("camera")}><span>💻</span><b>{source.kind !== "phone" ? "Try the laptop camera again" : "Use this laptop's camera"}</b><small>tilt the screen toward the desk</small></button>
              </div>
            </div>
          )}
          {!scene && !p.busy && live && <div className="hint-pill">Point the camera at a table with 3–5 objects — mapping starts by itself</div>}
          {scene && camErr && <div className="hint-pill warn">{camErr}</div>}
          {scene && view === "camera" && !hands && !p.busy && !camErr && (
            <div className="hint-pill">🖱 Drag an object to push it · click it to inspect · ✋ Hands to grab & throw · or ask in chat</div>
          )}
          {scene && view === "camera" && hands && !live && <div className="hint-pill warn">✋ Hands need live video — {source.kind === "phone" ? "open the phone page and keep it streaming" : "start the camera"}</div>}

          <div className="live-tools">
            <button onClick={capture} disabled={!!p.busy} title="Map the space again (after you move the camera or the objects)">⟳ Rescan</button>
            <button onClick={() => sweepInput.current?.click()} disabled={!!p.busy || sweep?.state === "running"}
              title="Best quality: a 10–20 s video moving slowly around the table, ending where the camera will stay. Every object becomes a real 3D model.">🎥 3D scan</button>
            <input ref={sweepInput} type="file" accept="video/*" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) startSweep(f); }} />
            {scene && <button onClick={p.onBall} disabled={!!p.busy} title="Put a real-size virtual baseball on your desk — grab it with ✋ Hands and throw it at your things">🎾 Ball</button>}
            {scene && <button className={hands ? "on" : ""} onClick={toggleHands} title="Use your real hand: pinch to grab, flick to throw, swipe to push">✋ Hands{hands ? " on" : ""}</button>}
            {(run || shownSim) && <button onClick={p.onReplay} title="Replay">▶ Replay</button>}
            {(run || shownSim) && <button onClick={p.onOpenStudio} title="Every measurement as graphs, edit & compare">⧉ Graphs</button>}
          </div>
        </div>
        <MiniPlayer run={run} sim={p.sim} />
      </section>
      <div className="live-side">
        {scene && p.selected && targets.some((t) => t.id === p.selected) && <ObjectCard id={p.selected} scene={scene} sim={shownSim} assets={p.assets} onClose={() => p.onSelect(null)} onAsk={p.onSend} />}
        <Chat messages={p.messages} busy={!!p.busy} onSend={p.onSend} hasScene={!!scene} hasRun={!!run} onOpenStudio={p.onOpenStudio}
          onReading={p.onReading} onRunReading={p.onRunReading} grabFrame={async () => (source.kind === "phone" ? (await fetch("/api/remote/latest.jpg")).blob() : grab(liveVideoEl()))} />
      </div>
    </div>
  );
}

const MAP_STEPS = ["Finding the table surface", "Outlining every object", "Measuring depth, size and mass", "Building the physics world"];
/** Mapping takes ~40–60 s on this Mac: show where it is instead of a bare spinner. */
function MappingProgress() {
  const [t0] = useState(() => performance.now());
  const [, tick] = useState(0);
  useEffect(() => { const iv = setInterval(() => tick((x) => x + 1), 250); return () => clearInterval(iv); }, []);
  const s = (performance.now() - t0) / 1000;
  const frac = 1 - Math.exp(-s / 30);                    // honest-ish: slows down rather than lying about 100 %
  const step = Math.min(MAP_STEPS.length - 1, Math.floor(frac * MAP_STEPS.length * 1.05));
  return (
    <div className="map-progress">
      <div className="map-title">Mapping your space <span className="muted">· {Math.round(s)} s</span></div>
      <div className="map-bar"><div style={{ width: `${Math.round(frac * 100)}%` }} /></div>
      <ol>{MAP_STEPS.map((x, i) => <li key={x} className={i < step ? "done" : i === step ? "now" : ""}>{x}</li>)}</ol>
      <div className="small muted">usually under a minute · keep the camera still</div>
    </div>
  );
}

const SWEEP_STAGES: [string, string][] = [["uploading", "Uploading the video"], ["reading", "Reading the frames"],
  ["depth", "Measuring depth in every view"], ["cameras", "Working out where the camera was"], ["objects", "Finding and naming the objects"],
  ["carving", "Carving each object from every angle"], ["meshes", "Textures, physics shapes and masses"]];
/** Real progress from the server (stage + fraction), not a timer. */
function SweepProgress({ job, onClose }: { job: SweepJob; onClose: () => void }) {
  const i = Math.max(0, SWEEP_STAGES.findIndex(([k]) => k === job.stage));
  const frac = job.state === "done" ? 1 : (i + Math.min(1, job.frac)) / SWEEP_STAGES.length;
  return (
    <div className="map-progress sweep">
      <div className="map-title">{job.state === "error" ? "3D scan failed" : "3D scan"} {job.state === "error" && <button className="x" onClick={onClose}>×</button>}</div>
      {job.state === "error" ? <p className="small">{job.error}</p> : (
        <>
          <div className="map-bar"><div style={{ width: `${Math.round(frac * 100)}%` }} /></div>
          <ol>{SWEEP_STAGES.map(([k, label], j) => <li key={k} className={j < i ? "done" : j === i ? "now" : ""}>{label}</li>)}</ol>
          <div className="small muted">{job.message}</div>
        </>
      )}
    </div>
  );
}

function liveVideoEl(): HTMLVideoElement | null {
  return document.querySelector(".live-stage video") as HTMLVideoElement | null;
}
async function grab(v: HTMLVideoElement | null): Promise<Blob | null> {
  if (!v || !v.videoWidth) return null;
  const c = document.createElement("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d")!.drawImage(v, 0, 0);
  return new Promise((res) => c.toBlob((b) => res(b), "image/jpeg", 0.93));
}

// ---------------------------------------------------------------- hologram diorama on the real table
function placement(run: ExpRun, scene: Scene) {
  const b = experimentBounds(run);
  const size = b.getSize(new THREE.Vector3()), center = b.getCenter(new THREE.Vector3());
  const T = scene.table;
  const room = T.height_above_floor < 0.1;
  const tw = Math.max(0.2, Math.min(T.x_max - T.x_min, T.y_max - T.y_min));
  const target = room ? Math.min(1.4, tw * 0.6) : Math.min(0.45, tw * 0.5);
  const s = Math.min(1, target / Math.max(size.x, size.y, size.z * 0.9, 1e-3));
  const R = 0.5 * Math.hypot(size.x, size.y) * s * (run.setting === "water" ? 1.3 : 1.0);
  // search the visible surface for a clear spot: away from real objects, inside the camera view, near the middle
  const cam = new THREE.PerspectiveCamera();
  applyCaptureCamera(cam, scene.camera);
  const camPos = new THREE.Vector3(...scene.camera.t_wc);
  let best: [number, number] = [0, 0], bestScore = -1e9;
  const step = Math.max(0.03, tw / 24);
  for (let x = T.x_min + R; x <= T.x_max - R; x += step) {
    for (let y = T.y_min + R; y <= T.y_max - R; y += step) {
      const clear = Math.min(9, ...scene.objects.map((o) => Math.hypot(o.pos[0] - x, o.pos[1] - y) - 0.5 * Math.hypot(o.size[0], o.size[1])));
      const v = new THREE.Vector3(x, y, size.z * s * 0.5).project(cam);
      if (Math.abs(v.x) > 0.8 || Math.abs(v.y) > 0.8 || v.z > 1) continue;
      // would a nearer real object hide it? compare screen rectangles and distances to the camera
      const W = scene.camera.width, H = scene.camera.height;
      const corners = [[-R, -R, 0], [R, -R, 0], [-R, R, 0], [R, R, 0], [-R, -R, size.z * s], [R, R, size.z * s]].map(([dx, dy, dz]) =>
        new THREE.Vector3(x + dx, y + dy, dz).project(cam)).map((q) => [(q.x + 1) / 2 * W, (1 - q.y) / 2 * H]);
      const rx0 = Math.min(...corners.map((c) => c[0])), rx1 = Math.max(...corners.map((c) => c[0]));
      const ry0 = Math.min(...corners.map((c) => c[1])), ry1 = Math.max(...corners.map((c) => c[1]));
      if (rx0 < 0.04 * W || rx1 > 0.96 * W || ry0 < 0.1 * H || ry1 > 0.86 * H) continue;      // whole hologram in view, clear of the HUD/tools
      const myDist = Math.hypot(x - camPos.x, y - camPos.y);
      let hidden = 0;
      for (const o of scene.objects) {
        const [bx, by, bw, bh] = o.bbox;
        const ox = Math.max(0, Math.min(rx1, bx + bw) - Math.max(rx0, bx)), oy = Math.max(0, Math.min(ry1, by + bh) - Math.max(ry0, by));
        if (ox * oy > 0 && Math.hypot(o.pos[0] - camPos.x, o.pos[1] - camPos.y) < myDist) hidden += (ox * oy) / Math.max(1, (rx1 - rx0) * (ry1 - ry0));
      }
      const score = Math.min(clear - R, 0.25) * 4 - Math.hypot(v.x, v.y * 1.2) * 0.4 - hidden * 3 - v.y * 0.15;
      if (score > bestScore) { bestScore = score; best = [x, y]; }
    }
  }
  const m = new THREE.Matrix4().compose(
    new THREE.Vector3(best[0] - center.x * s, best[1] - center.y * s, 0.002 - b.min.z * s),
    new THREE.Quaternion(), new THREE.Vector3(s, s, s));
  return { m, s, anchor: best, footprint: Math.max(size.x, size.y) * s, height: size.z * s, room };
}
function scaleText(run: ExpRun, scene: Scene | null) {
  if (!scene) return "";
  const { s } = placement(run, scene);
  return s > 0.98 ? "real size" : `1 : ${Math.round(1 / s).toLocaleString()} scale model (physics at full size)`;
}

function CaptureCam({ scene }: { scene: Scene }) {
  const { camera, size } = useThree();
  useEffect(() => { applyCaptureCamera(camera as THREE.PerspectiveCamera, scene.camera); }, [camera, scene.camera, size]);
  return null;
}
function Env() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pm = new THREE.PMREMGenerator(gl);
    scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.7;
    return () => pm.dispose();
  }, [gl, scene]);
  return null;
}

function ARDiorama({ run, scene, assets }: { run: ExpRun; scene: Scene; assets: Record<string, Asset> }) {
  const host = useRef<HTMLDivElement>(null);
  const pl = useMemo(() => placement(run, scene), [run, scene]);
  const bounds = useMemo(() => experimentBounds(run), [run]);
  const scale = useMemo(() => Math.max(0.2, bounds.getSize(new THREE.Vector3()).length()), [bounds]);
  return (
    <>
      <Canvas className="ar-canvas" shadows gl={{ alpha: true, antialias: true }} onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.setClearColor(0x000000, 0); }}
        style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
        <CaptureCam scene={scene} />
        <Env />
        <hemisphereLight args={["#e0f2fe", "#3b2a1d", 0.6]} />
        <directionalLight position={[pl.anchor[0] + 0.4, pl.anchor[1] - 0.5, 1.6]} intensity={2.2} castShadow shadow-mapSize={[1024, 1024]}
          shadow-camera-left={-1} shadow-camera-right={1} shadow-camera-top={1} shadow-camera-bottom={-1} shadow-camera-near={0.1} shadow-camera-far={5} />
        <HoloBase pl={pl} run={run} />
        {scene.objects.map((o) => <Occluder key={o.id} o={o} />)}
        <group matrixAutoUpdate={false} matrix={pl.m}>
          {run.objects.filter((o) => o.sid).map((o) => <Thing key={`${run.id}-${o.sid}`} o={o} run={run} assets={assets} />)}
          <Links run={run} />
          <Trails run={run} />
          <Effects run={run} scale={scale} sizeScale={pl.s} shakeCamera={false} />
          {run.setting === "water" && <Aquarium run={run} />}
          <Labels run={run} show={true} host={host} xf={pl.m} />
        </group>
      </Canvas>
      <div className="xlabels" ref={host} />
    </>
  );
}

/** Invisible stand-in for a real object: writes depth only, so it hides hologram parts behind it. */
function Occluder({ o }: { o: Scene["objects"][number] }) {
  const scanned = useScanned(o);
  const g = useMemo(() => geometryFor(o), [o, scanned]);
  const m = useMemo(() => observedMatrix(o), [o]);
  return <mesh geometry={g} matrixAutoUpdate={false} matrix={m} renderOrder={-1}><meshBasicMaterial colorWrite={false} /></mesh>;
}

/** A soft shadow on the real table plus a glowing projector ring: the diorama is clearly a hologram. */
function HoloBase({ pl, run }: { pl: ReturnType<typeof placement>; run: ExpRun }) {
  const ring = useRef<THREE.Mesh>(null);
  const R = Math.max(0.08, pl.footprint * 0.62);
  useFrame(({ clock: c }) => { if (ring.current) (ring.current.material as THREE.MeshBasicMaterial).opacity = 0.45 + 0.2 * Math.sin(c.elapsedTime * 2); });
  const air = run.setting === "water" || run.setting === "space" || run.setting === "air";
  return (
    <group position={[pl.anchor[0], pl.anchor[1], 0.001]}>
      <mesh receiveShadow><circleGeometry args={[R * 1.6, 64]} /><shadowMaterial opacity={0.35} transparent /></mesh>
      <mesh ref={ring}><ringGeometry args={[R * 0.96, R, 96]} /><meshBasicMaterial color="#38bdf8" transparent opacity={0.6} blending={THREE.AdditiveBlending} depthWrite={false} /></mesh>
      <mesh><circleGeometry args={[R * 0.96, 64]} /><meshBasicMaterial color="#38bdf8" transparent opacity={air ? 0.05 : 0.08} blending={THREE.AdditiveBlending} depthWrite={false} /></mesh>
    </group>
  );
}

function Aquarium({ run }: { run: ExpRun }) {
  const b = useMemo(() => experimentBounds(run), [run]);
  const size = b.getSize(new THREE.Vector3()), c = b.getCenter(new THREE.Vector3());
  const top = run.env.surface_z ?? b.max.z;
  const h = Math.max(size.z, top - b.min.z) * 1.05;
  return (
    <group position={[c.x, c.y, b.min.z + h / 2]}>
      <mesh><boxGeometry args={[size.x * 1.15, size.y * 1.3, h]} /><meshPhysicalMaterial color="#38bdf8" transparent opacity={0.16} roughness={0.05} transmission={0.4} depthWrite={false} /></mesh>
      <lineSegments><edgesGeometry args={[new THREE.BoxGeometry(size.x * 1.15, size.y * 1.3, h)]} /><lineBasicMaterial color="#7dd3fc" transparent opacity={0.7} /></lineSegments>
    </group>
  );
}

// ---------------------------------------------------------------- chat
function Chat({ messages, busy, onSend, hasScene, hasRun, onOpenStudio, onReading, onRunReading, grabFrame }: { messages: ChatMsg[]; busy: boolean; onSend: (t: string) => void;
  hasScene: boolean; hasRun: boolean; onOpenStudio: () => void; onReading: (r: DiagramReading) => void; onRunReading: (r: DiagramReading, text: string) => void;
  grabFrame: () => Promise<Blob | null> }) {
  const [text, setText] = useState("");
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => { list.current?.scrollTo({ top: list.current.scrollHeight, behavior: "smooth" }); }, [messages.length, busy]);
  const send = (t: string) => { if (!t.trim() || busy) return; onSend(t.trim()); setText(""); };
  const chips = hasRun ? FOLLOW : hasScene ? [...STARTERS_SCENE, ...STARTERS_ANY.slice(0, 3)] : STARTERS_ANY;
  return (
    <aside className="chat">
      <header className="chat-head">
        <div className="chat-avatar">◎</div>
        <div><b>Lab assistant</b><div className="small muted">Describe any experiment — it appears in your camera</div></div>
      </header>
      <div className="chat-list" ref={list}>
        {messages.length === 0 && (
          <div className="chat-hello">
            <p><b>Hi!</b> I'm looking through your camera. Tell me what should happen — I'll simulate it with real physics right here in your space, measure everything, and tell you why.</p>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`msg ${m.role} ${m.error ? "err" : ""}`}>
            <div className="bubble">
              {m.text}
              {m.changes && m.changes.length > 0 && <div className="msg-changes">changed: {m.changes.join(" · ")}</div>}
              {m.tiles && m.tiles.length > 0 && (
                <div className="msg-tiles">{m.tiles.map((t) => <div key={t.label} className="msg-tile"><b>{t.text}</b><span>{t.label}</span></div>)}</div>
              )}
              {m.results && <TheoryCheck rows={m.results} compact />}
              {m.why && m.why.length > 0 && <details className="msg-why"><summary>Why?</summary>{m.why.map((w, i) => <p key={i}>{w}</p>)}</details>}
              {m.reading && !m.reading.error && <ReadingCard r={m.reading} onRun={onRunReading} />}
              {m.runId && <button className="link small" onClick={onOpenStudio}>Graphs, every measurement & edit →</button>}
            </div>
          </div>
        ))}
        {busy && <div className="msg ai"><div className="bubble typing"><span /><span /><span /></div></div>}
      </div>
      <div className="chat-chips">
        <DiagramButton className="chip-btn" onRead={onReading} grabFrame={grabFrame} label="📷 Snap a diagram" />
        {chips.map((c) => <button key={c} className="chip" disabled={busy} onClick={() => send(c)}>{c}</button>)}
      </div>
      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(text); }}>
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder={hasRun ? "Ask why, say “what if…”, or try something new" : "e.g. drop a baseball on the table"} />
        <VoiceButton disabled={busy} onText={(t) => send(t)} />
        <button className="send" disabled={busy || !text.trim()} aria-label="send">➤</button>
      </form>
    </aside>
  );
}

export function ReadingCard({ r, onRun }: { r: DiagramReading; onRun: (r: DiagramReading, text: string) => void }) {
  const [txt, setTxt] = useState(r.reading);
  const lab = r.lab === "transformer" ? "AI Lab (transformer)" : r.lab === "chemistry" ? "Chemistry lab" : r.kind === "circuit" ? "circuit simulator" : "physics simulator";
  return (
    <div className="reading">
      <div className="small muted">I read a <b>{r.kind.replace("-", " ")}</b> diagram{r.labels?.length ? `: ${r.labels.join(" → ")}` : ""}. Check it, fix anything I misread, then run it in the {lab}:</div>
      <textarea value={txt} onChange={(e) => setTxt(e.target.value)} rows={3} />
      <button className="btn-glow" onClick={() => onRun(r, txt)}>▶ Simulate this</button>
    </div>
  );
}

function MiniPlayer({ run, sim }: { run: AnyRun | null; sim: Sim | null }) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  if (!run && !sim) return null;
  const dur = clock.duration || 1;
  return (
    <div className="mini-player">
      <button onClick={() => (clock.playing ? clock.pause() : clock.play(clock.t >= dur - 1e-3 ? 0 : undefined))}>{clock.playing ? "❚❚" : "▶"}</button>
      <input type="range" min={0} max={dur} step={0.01} value={clock.t} onChange={(e) => { clock.pause(); clock.set(+e.target.value); }} />
      <span className="mono small">{clock.t.toFixed(2)} s</span>
      <select value={clock.rate} onChange={(e) => (clock.rate = +e.target.value)} className="small">
        {[0.1, 0.25, 0.5, 1, 2].map((r) => <option key={r} value={r}>{r}×</option>)}
      </select>
    </div>
  );
}
