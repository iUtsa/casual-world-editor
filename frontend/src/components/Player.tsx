import { useEffect, useRef, useState } from "react";
import { clock } from "../clock";
import type { FxEvent } from "../types";
import type { ViewOptions } from "./ExperimentView";

/** Play / scrub / speed, impact markers on the timeline, and view toggles. */
export default function Player({ duration, fx, opts, setOpts, onResetCamera, start }: {
  duration: number; fx: FxEvent[]; opts: ViewOptions | null; setOpts: (o: ViewOptions) => void; onResetCamera?: () => void; start: number;
}) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const bar = useRef<HTMLDivElement>(null);
  const scrub = (e: React.PointerEvent) => {
    const r = bar.current!.getBoundingClientRect();
    clock.pause();
    clock.set(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * duration);
  };
  const hits = fx.filter((e) => e.kind === "hit" || e.kind === "splash");
  const top = Math.max(1, ...hits.map((e) => e.force ?? 0));
  const dedup: FxEvent[] = [];
  for (const e of hits) if (!dedup.length || e.t - dedup[dedup.length - 1].t > 0.12) dedup.push(e);
  return (
    <div className="player">
      <button className="play" onClick={() => (clock.playing ? clock.pause() : clock.play(clock.t >= duration - 1e-3 ? start : undefined))} aria-label={clock.playing ? "pause" : "play"}>
        {clock.playing ? "❚❚" : "▶"}
      </button>
      <button onClick={() => { clock.rate = 1; clock.play(start); }} title="replay the experiment">↺</button>
      <div className="scrub" ref={bar} onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); scrub(e); }}
        onPointerMove={(e) => e.buttons && scrub(e)}>
        <div className="scrub-fill" style={{ width: `${(clock.t / Math.max(1e-6, duration)) * 100}%` }} />
        {dedup.map((e, i) => (
          <span key={i} className={`mark ${e.kind}`} style={{ left: `${(e.t / duration) * 100}%`, height: `${40 + 60 * Math.min(1, (e.force ?? 0) / top)}%` }}
            title={e.kind === "hit" ? `impact at ${e.t.toFixed(2)} s: ${(e.force ?? 0) >= 1000 ? ((e.force ?? 0) / 1000).toFixed(1) + " kN" : (e.force ?? 0).toFixed(0) + " N"}` : `splash at ${e.t.toFixed(2)} s`} />
        ))}
      </div>
      <span className="small mono">{clock.t.toFixed(2)} / {duration.toFixed(1)} s</span>
      <select className="small" value={opts?.slowmo ? "auto" : String(clock.rate)} onChange={(e) => {
        if (e.target.value === "auto") { opts && setOpts({ ...opts, slowmo: true }); clock.rate = 1; }
        else { opts && setOpts({ ...opts, slowmo: false }); clock.rate = +e.target.value; }
      }} title="playback speed">
        {opts && <option value="auto">slow-mo at impacts</option>}
        {[0.1, 0.25, 0.5, 1, 2].map((r) => <option key={r} value={r}>{r}×</option>)}
      </select>
      {opts && (
        <details className="viewmenu">
          <summary>View</summary>
          <div className="menu">
            {([["trails", "motion trails"], ["arrows", "velocity arrows"], ["labels", "names & speeds"], ["follow", "camera follows the action"]] as const).map(([k, l]) => (
              <label key={k} className="check"><input type="checkbox" checked={opts[k]} onChange={(e) => setOpts({ ...opts, [k]: e.target.checked })} />{l}</label>
            ))}
            {onResetCamera && <button onClick={onResetCamera}>Reset camera</button>}
            <p className="muted small">Drag to orbit · scroll to zoom · right-drag to pan</p>
          </div>
        </details>
      )}
    </div>
  );
}
