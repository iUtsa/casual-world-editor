import { useEffect, useRef, useState } from "react";
import { clock } from "../clock";

/** Single-series line chart (title names the measure; no legend), 2px line with a
 *  10% area wash, recessive hairline axes, crosshair + tooltip, playhead. */
export default function LineChart({ t, y, title, unit, duration, digits = 2, now, xUnit = "s" }: {
  t: number[]; y: number[]; title: string; unit: string; duration: number; digits?: number;
  now?: number; xUnit?: string;
}) {
  const W = 320, H = 120, L = 40, R = 8, T = 8, B = 20;
  const [hover, setHover] = useState<number | null>(null);
  const [, tick] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const ymaxRaw = Math.max(1e-9, ...y);
  const yminRaw = Math.min(0, ...y);
  const step = niceStep(Math.max(1e-9, ymaxRaw - yminRaw) / 3);
  const ymax = Math.ceil(ymaxRaw / step) * step;
  const ymin = Math.floor(yminRaw / step) * step;
  const xs = (v: number) => L + (v / Math.max(1e-9, duration)) * (W - L - R);
  const ys = (v: number) => H - B - ((v - ymin) / Math.max(1e-9, ymax - ymin)) * (H - T - B);
  const d = t.map((tt, i) => `${i ? "L" : "M"}${xs(tt).toFixed(1)},${ys(y[i]).toFixed(1)}`).join("");
  const area = t.length && ymin >= 0 ? `${d}L${xs(t[t.length - 1]).toFixed(1)},${ys(0)}L${xs(t[0]).toFixed(1)},${ys(0)}Z` : "";
  const ticks: number[] = [];
  for (let v = ymin; v <= ymax + 1e-9; v += step) ticks.push(Math.round(v / step) * step);
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(digits));
  const onMove = (e: React.PointerEvent) => {
    const r = box.current!.getBoundingClientRect();
    const tx = (((e.clientX - r.left) / r.width) * W - L) / (W - L - R) * duration;
    let best = 0;
    for (let i = 1; i < t.length; i++) if (Math.abs(t[i] - tx) < Math.abs(t[best] - tx)) best = i;
    setHover(t.length ? best : null);
  };
  const peak = y.length ? y.indexOf(Math.max(...y)) : -1;
  return (
    <div className="chart" ref={box} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
      <div className="chart-title">{title} <span className="muted">({unit})</span></div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} over time`}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={ys(v)} y2={ys(v)} className="grid" />
            <text x={L - 4} y={ys(v) + 3} className="tick" textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        <text x={L} y={H - 4} className="tick">0 {xUnit}</text>
        <text x={W - R} y={H - 4} className="tick" textAnchor="end">{duration.toFixed(xUnit === "s" ? 1 : 0)} {xUnit}</text>
        {ymin < 0 && <line x1={L} x2={W - R} y1={ys(0)} y2={ys(0)} className="zero" />}
        <path d={area} className="area" />
        <path d={d} className="line" />
        {peak >= 0 && <circle cx={xs(t[peak])} cy={ys(y[peak])} r={4} className="dot" />}
        {(now ?? clock.t) >= 0 && <line x1={xs(now ?? clock.t)} x2={xs(now ?? clock.t)} y1={T} y2={H - B} className="playhead" />}
        {hover !== null && <line x1={xs(t[hover])} x2={xs(t[hover])} y1={T} y2={H - B} className="crosshair" />}
      </svg>
      {hover !== null && (
        <div className="tooltip" style={{ left: `${(xs(t[hover]) / W) * 100}%` }}>
          <b>{fmt(y[hover])} {unit}</b><span className="muted"> at {t[hover].toFixed(xUnit === "s" ? 2 : 0)} {xUnit}</span>
        </div>
      )}
    </div>
  );
}

function niceStep(x: number) {
  const p = Math.pow(10, Math.floor(Math.log10(x)));
  const m = x / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * p;
}
