import { useEffect, useState } from "react";
import { clock } from "../clock";
import type { Sim } from "../types";

const EV_COLOR: Record<string, string> = {
  intervention: "#f87171", lost_support: "#fb923c", impact: "#facc15", started_moving: "#38bdf8",
  toppled: "#c084fc", hit_floor: "#ef4444", came_to_rest: "#64748b",
};

export default function Timeline({ sim, highlight }: { sim: Sim | null; highlight: string[] }) {
  const [, force] = useState(0);
  useEffect(() => clock.subscribe(() => force((x) => x + 1)), []);
  if (!sim) return <div className="timeline muted small">No intervention yet — pick an object and intervene.</div>;
  const D = sim.duration || 1;
  const events = sim.segments.flatMap((s) => s.events);
  return (
    <div className="timeline">
      <button onClick={() => (clock.playing ? clock.pause() : clock.play())}>{clock.playing ? "❚❚" : "▶"}</button>
      <button onClick={() => clock.play(sim.segments[sim.segments.length - 1].t_start)} title="replay the latest intervention">↺ last</button>
      <div className="track"
        onPointerDown={(e) => {
          const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
          const set = (x: number) => clock.set(((x - r.left) / r.width) * D);
          clock.pause(); set(e.clientX);
          const mv = (ev: PointerEvent) => set(ev.clientX);
          const up = () => { window.removeEventListener("pointermove", mv); window.removeEventListener("pointerup", up); };
          window.addEventListener("pointermove", mv); window.addEventListener("pointerup", up);
        }}>
        {sim.segments.map((s) => (
          <div key={s.index} className="seg" style={{ left: `${(s.t_start / D) * 100}%`, width: `${((s.t_end - s.t_start) / D) * 100}%` }}>
            <span>{s.intervention.type.replace("set_", "")}</span>
          </div>
        ))}
        {events.filter((e) => e.type !== "came_to_rest").map((e) => (
          <div key={e.id} className={`tick ${highlight.includes(e.id) ? "hl" : ""}`} title={`${e.id} ${e.type} ${e.subject ?? ""}${e.other ? " / " + e.other : ""} @ ${e.t.toFixed(2)}s`}
            style={{ left: `${(e.t / D) * 100}%`, background: EV_COLOR[e.type] ?? "#94a3b8" }} />
        ))}
        <div className="head" style={{ left: `${(clock.t / D) * 100}%` }} />
      </div>
      <span className="mono small">{clock.t.toFixed(2)} / {D.toFixed(2)} s</span>
      <select value={clock.rate} onChange={(e) => { clock.rate = +e.target.value; clock.emit(); }}>
        <option value={0.25}>0.25×</option><option value={0.5}>0.5×</option><option value={1}>1×</option>
      </select>
    </div>
  );
}
