import { useEffect, useMemo, useRef, useState } from "react";
import SimLayer from "./SimLayer";
import { api } from "../api";
import { clock } from "../clock";
import { boxSilhouetteAtEnd, project, silhouetteAtEnd, type Pt } from "../geometry2d";
import type { Asset, Scene, Sim, Tool } from "../types";

export interface SourceView {
  kind: "camera" | "video" | "image" | "phone" | "none";
  stream?: MediaStream | null;
  videoUrl?: string | null;
  imageUrl?: string | null;
  depth?: Blob | null;          // aligned sensor depth (16-bit mm PNG), if the source has one
  depthLabel?: string;
}

interface Props {
  scene: Scene | null; sim: Sim | null; source: SourceView; liveBackground: boolean;
  tool: Tool; selected: string | null; mergeSet: string[]; showFit: boolean; showGhosts: boolean;
  showOutlines: boolean; frameKey: string;
  assets: Record<string, Asset>;
  onPick: (id: string | null) => void;
  onPlace: (p: Pt) => void;
  showVectors: boolean;
  onPush: (id: string, from: Pt, to: Pt) => void;
  onAdd: (p: Pt) => void;
  sourceVideoRef: (v: HTMLVideoElement | null) => void;
}

const COLORS = ["#38bdf8", "#f472b6", "#a3e635", "#fbbf24", "#c084fc", "#34d399", "#fb7185", "#60a5fa", "#facc15", "#2dd4bf"];

export default function Stage(p: Props) {
  const { scene, sim, source } = p;
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ id: string; from: Pt; to: Pt } | null>(null);
  const [atEnd, setAtEnd] = useState(true);
  const [playing, setPlaying] = useState(false);
  const [drift, setDrift] = useState(0);
  const W = scene?.camera.width ?? 960;
  const H = scene?.camera.height ?? 540;

  useEffect(() => clock.subscribe(() => { setAtEnd(!sim || clock.t >= clock.duration - 1e-3); setPlaying(clock.playing); }), [sim]);

  // hit-test shapes: photo outlines before any intervention, simulated end-pose silhouettes after
  const shapes = useMemo(() => {
    if (!scene) return [];
    return scene.objects.map((o, i) => {
      let poly: number[][] | null = o.polygon;
      let moved = false;
      if (sim) {
        if (sim.removed_at[o.id] !== undefined) return null;
        if (sim.moved_ids.includes(o.id)) { poly = silhouetteAtEnd(scene.camera, o, sim); moved = true; }
      }
      return poly ? { o, poly, moved, color: COLORS[i % COLORS.length] } : null;
    }).filter(Boolean) as { o: Scene["objects"][number]; poly: number[][]; moved: boolean; color: string }[];
  }, [scene, sim]);
  const spawnShapes = useMemo(() => {
    if (!scene || !sim?.spawned) return [];
    return Object.entries(sim.spawned).map(([sid, sp]) => {
      const a = p.assets[sp.asset];
      if (!a) return null;
      const poly = boxSilhouetteAtEnd(scene.camera, sid, a.dims, sim);
      return poly ? { sid, label: sp.label, poly } : null;
    }).filter(Boolean) as { sid: string; label: string; poly: number[][] }[];
  }, [scene, sim, p.assets]);

  const dynamicIds = useMemo(() => (sim ? [...new Set([...sim.moved_ids, ...Object.keys(sim.removed_at)])] : []), [sim]);

  const toImg = (e: React.PointerEvent): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * W, ((e.clientY - r.top) / r.height) * H];
  };

  // live-camera drift check: compare live frames with the scanned frame
  const liveVideo = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (!scene || source.kind !== "camera" || !p.liveBackground) { setDrift(0); return; }
    const ref = new Image();
    ref.src = `/api/frame.jpg?k=${p.frameKey}`;
    const c = document.createElement("canvas");
    c.width = 64; c.height = 36;
    const ctx = c.getContext("2d", { willReadFrequently: true })!;
    let refData: Uint8ClampedArray | null = null;
    ref.onload = () => { ctx.drawImage(ref, 0, 0, 64, 36); refData = ctx.getImageData(0, 0, 64, 36).data.slice(); };
    const iv = setInterval(() => {
      const v = liveVideo.current;
      if (!v || !refData || v.readyState < 2) return;
      ctx.drawImage(v, 0, 0, 64, 36);
      const d = ctx.getImageData(0, 0, 64, 36).data;
      let s = 0;
      for (let i = 0; i < d.length; i += 4) s += Math.abs(d[i] - refData[i]) + Math.abs(d[i + 1] - refData[i + 1]) + Math.abs(d[i + 2] - refData[i + 2]);
      setDrift(s / (64 * 36 * 3));
    }, 700);
    return () => clearInterval(iv);
  }, [scene, source.kind, p.liveBackground, p.frameKey]);

  const bindStream = (v: HTMLVideoElement | null) => {
    if (v && source.stream && v.srcObject !== source.stream) { v.srcObject = source.stream; v.play().catch(() => {}); }
  };

  const showLive = !scene || p.liveBackground;
  return (
    <div className="stage" style={{ aspectRatio: `${W} / ${H}` }}>
      {/* layer 1: background */}
      {showLive && source.kind === "camera" && (
        <video ref={(v) => { bindStream(v); (liveVideo as React.MutableRefObject<HTMLVideoElement | null>).current = v; if (!scene) p.sourceVideoRef(v); }}
          className="layer" muted playsInline autoPlay />
      )}
      {showLive && source.kind === "video" && source.videoUrl && (
        <video ref={(v) => { if (!scene) p.sourceVideoRef(v); }} className="layer" src={source.videoUrl} muted playsInline controls={!scene} loop />
      )}
      {showLive && source.kind === "image" && source.imageUrl && !scene && <img className="layer" src={source.imageUrl} alt="" />}
      {showLive && source.kind === "phone" && <img className="layer" src="/api/remote/stream" alt="phone camera" />}
      {scene && (!showLive || source.kind === "image" || source.kind === "none") && <img className="layer" src={`/api/frame.jpg?k=${p.frameKey}`} alt="scanned frame" />}
      {/* layer 2: clean plate where moved/removed objects used to be */}
      {scene && sim && dynamicIds.length > 0 && <img className="layer" src={api.plateUrl(dynamicIds, sim.key)} alt="" />}
      {/* layer 3: simulated objects */}
      {scene && (
        <SimLayer scene={scene} sim={sim} frameUrl={`/api/frame.jpg?k=${p.frameKey}`} idsUrl={`/api/idmap.png?k=${p.frameKey}`}
          selected={p.selected} showFit={p.showFit} showGhosts={p.showGhosts} assets={p.assets} />
      )}
      {/* layer 4: interaction overlay */}
      {scene && (
        <svg ref={svgRef} className={`overlay tool-${p.tool}`} viewBox={`0 0 ${W} ${H}`}
          onPointerDown={(e) => {
            if (p.tool === "add") { p.onAdd(toImg(e)); return; }
            if (p.tool === "place" || p.tool === "driveto") { p.onPlace(toImg(e)); return; }
            if (e.target === svgRef.current) p.onPick(null);
          }}
          onPointerMove={(e) => drag && setDrag({ ...drag, to: toImg(e) })}
          onPointerUp={() => {
            if (drag) {
              const dx = drag.to[0] - drag.from[0], dy = drag.to[1] - drag.from[1];
              if (Math.hypot(dx, dy) > 8) p.onPush(drag.id, drag.from, drag.to);
              else p.onPick(drag.id);                      // a click without a drag selects it
              setDrag(null);
            }
          }}>
          {shapes.map(({ o, poly, moved, color }) => {
            const sel = o.id === p.selected || p.mergeSet.includes(o.id);
            const show = p.showOutlines || sel || p.tool !== "select";
            return (
              <g key={o.id}>
                <polygon points={poly.map((q) => q.join(",")).join(" ")}
                  className={`obj ${sel ? "sel" : ""} ${moved ? "moved" : ""}`}
                  style={{ stroke: show ? color : "transparent", fill: sel ? color + "40" : "transparent", pointerEvents: playing ? "none" : "all" }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    if (p.tool === "push") { const pt = toImg(e); setDrag({ id: o.id, from: pt, to: pt }); (e.target as Element).setPointerCapture?.(e.pointerId); }
                    else if (p.tool === "add") p.onAdd(toImg(e));
                    else p.onPick(o.id);
                  }} />
                {show && !sim && (
                  <text x={o.bbox[0] + 4} y={o.bbox[1] - 6} className="tag" style={{ fill: color }}>{o.label}</text>
                )}
              </g>
            );
          })}
          {spawnShapes.map(({ sid, label, poly }) => {
            const sel = sid === p.selected;
            return (
              <g key={sid}>
                <polygon points={poly.map((q) => q.join(",")).join(" ")} className={`obj spawned ${sel ? "sel" : ""}`}
                  style={{ stroke: sel || p.tool === "remove" || p.tool === "push" ? "#facc15" : "transparent", fill: sel ? "#facc1530" : "transparent", pointerEvents: !playing && p.tool !== "place" && p.tool !== "driveto" ? "all" : "none" }}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    if (p.tool === "push") { const pt = toImg(e); setDrag({ id: sid, from: pt, to: pt }); (e.target as Element).setPointerCapture?.(e.pointerId); }
                    else p.onPick(sid);
                  }} />
                {sel && <text x={poly[0][0]} y={Math.min(...poly.map((q) => q[1])) - 6} className="tag" style={{ fill: "#facc15" }}>{label}</text>}
              </g>
            );
          })}
          {p.showVectors && sim && scene && <VelocityVectors sim={sim} scene={scene} />}
          {drag && (
            <g className="push-arrow">
              <line x1={drag.from[0]} y1={drag.from[1]} x2={drag.to[0]} y2={drag.to[1]} />
              <circle cx={drag.to[0]} cy={drag.to[1]} r={6} />
            </g>
          )}
        </svg>
      )}

      {drift > 18 && <div className="banner warn">Camera or scene changed since the scan (Δ{drift.toFixed(0)}). Rescan for an accurate model.</div>}
      {sim && playing && <div className="banner info">Playing simulation — interact when it finishes</div>}
      {sim && !playing && !atEnd && <div className="banner info">Scrubbing — new interventions apply to the end state</div>}
    </div>
  );
}

/** Live velocity arrows + speed labels for moving objects, from the recorded centre-of-mass velocity. */
function VelocityVectors({ sim, scene }: { sim: Sim; scene: Scene }) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const f = Math.min(sim.frames.length - 1, Math.max(0, Math.round(clock.t * sim.fps)));
  const out: JSX.Element[] = [];
  sim.object_ids.forEach((id, i) => {
    const v = sim.kinematics?.[id]?.v[f];
    if (!v) return;
    const sp = Math.hypot(v[0], v[1], v[2]);
    if (sp < 0.06) return;
    const pz = sim.frames[f][i];
    const lift = sim.spawned?.[id] ? 0.03 : 0;
    const a = project(scene.camera, [pz[0], pz[1], pz[2] + lift]);
    const b = project(scene.camera, [pz[0] + v[0] * 0.2, pz[1] + v[1] * 0.2, pz[2] + lift + v[2] * 0.2]);
    if (!a || !b) return;
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const h = 9;
    out.push(
      <g key={id} className="vel">
        <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />
        <path d={`M${b[0]},${b[1]} L${b[0] - h * Math.cos(ang - 0.45)},${b[1] - h * Math.sin(ang - 0.45)} L${b[0] - h * Math.cos(ang + 0.45)},${b[1] - h * Math.sin(ang + 0.45)} Z`} />
        <text x={b[0] + 6} y={b[1] - 6}>{sp.toFixed(2)} m/s</text>
      </g>,
    );
  });
  return <g pointerEvents="none">{out}</g>;
}
