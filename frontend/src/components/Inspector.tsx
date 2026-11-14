import { useEffect, useState } from "react";
import type { Scene, SceneObject } from "../types";

const MATERIALS = ["unknown", "ceramic", "glass", "metal", "aluminum", "plastic", "wood", "paper", "cardboard", "fabric", "rubber", "food", "foam"];

interface Props {
  scene: Scene; selected: string | null; onSelect: (id: string) => void;
  onPatch: (id: string, body: Record<string, unknown>) => void;
  onDiscard: (id: string) => void;
  onCalibrate: (id: string, dim: string, meters: number) => void;
  onIntervene: (type: string, target: string | null, params?: Record<string, unknown>) => void;
  removed: Set<string>; busy: boolean; depthScale: number;
}

export default function Inspector(p: Props) {
  const o = p.scene.objects.find((x) => x.id === p.selected) ?? null;
  return (
    <section className="panel">
      <h3>2 · World model <span className="muted">({p.scene.objects.length} objects)</span></h3>
      <ul className="objlist">
        {p.scene.objects.map((x) => (
          <li key={x.id} className={`${x.id === p.selected ? "sel" : ""} ${p.removed.has(x.id) ? "gone" : ""}`} onClick={() => p.onSelect(x.id)}>
            <span className="swatch" style={{ background: `rgb(${x.mean_color.map((c) => c * 255).join(",")})` }} />
            <span className="grow">{x.label}</span>
            {!x.stability.stable && <span className="pill warn" title="unstable as reconstructed">!</span>}
            {x.static && <span className="pill">pinned</span>}
            <span className="muted">{x.mass.toFixed(2)} kg</span>
          </li>
        ))}
      </ul>
      {o ? <ObjectEditor key={o.id + JSON.stringify(o.size) + o.mass} {...p} o={o} /> : <p className="muted small">Click an object in the view or list.</p>}
    </section>
  );
}

function ObjectEditor({ o, scene, onPatch, onDiscard, onCalibrate, onIntervene, removed, busy, depthScale }: Props & { o: SceneObject }) {
  const [label, setLabel] = useState(o.label);
  const [mass, setMass] = useState(o.mass.toFixed(3));
  const [calib, setCalib] = useState("");
  useEffect(() => setLabel(o.label), [o.label]);
  const name = (id: string) => scene.objects.find((x) => x.id === id)?.label ?? id;
  const isRemoved = removed.has(o.id);
  return (
    <div className="editor">
      <div className="kv">
        <span>size</span><b>{o.size.map((v) => (v * 100).toFixed(1)).join(" × ")} cm</b>
        <span>rests on</span><b>{o.supported_by.map(name).join(", ") || "—"}</b>
        <span>supports</span><b>{o.supporting.map(name).join(", ") || "—"}</b>
        <span>mass</span><b>{o.mass.toFixed(3)} kg <i className="muted">({o.mass_source})</i></b>
        <span>label</span><b className="muted">{o.label_source}</b>
      </div>
      {o.notes.length > 0 && <ul className="notes">{o.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
      {!o.stability.stable && (
        <div className="error small">Model says this is unstable as reconstructed: it settled {(o.stability.moved_m * 100).toFixed(1)} cm / {o.stability.rotated_deg.toFixed(0)}°. Check its shape/size.</div>
      )}

      <h4>Intervene <span className="muted small">(adds a causal step)</span></h4>
      <div className="row wrap">
        <button className="danger" disabled={busy || isRemoved} onClick={() => onIntervene("remove", o.id)}>Remove</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_mass", o.id, { mass: o.mass * 10 })}>Mass ×10</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_mass", o.id, { mass: o.mass / 10 })}>Mass ÷10</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_friction", o.id, { friction: 0.03 })}>Make icy</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_friction", o.id, { friction: 1.2 })}>Make grippy</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_static", o.id, { static: true })}>Pin in place</button>
        <button disabled={busy || isRemoved} onClick={() => onIntervene("set_static", o.id, { static: false })}>Unpin</button>
      </div>
      <p className="muted small">Push: pick the Push tool and drag on the object (longer drag = harder push).</p>

      <h4>Fix reconstruction <span className="muted small">(resets interventions)</span></h4>
      <div className="grid2">
        <label>Label<input value={label} onChange={(e) => setLabel(e.target.value)} onBlur={() => label !== o.label && onPatch(o.id, { label })} /></label>
        <label>Material
          <select value={o.material} onChange={(e) => onPatch(o.id, { material: e.target.value })}>
            {MATERIALS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label>Shape
          <select value={o.shape} onChange={(e) => onPatch(o.id, { shape: e.target.value })}>
            <option value="box">box</option><option value="cylinder">cylinder</option>
          </select>
        </label>
        <label>Mass (kg)<input type="number" step="0.01" value={mass} onChange={(e) => setMass(e.target.value)}
          onBlur={() => +mass > 0 && Math.abs(+mass - o.mass) > 1e-4 && onPatch(o.id, { mass: +mass })} /></label>
        <label>Friction {o.friction.toFixed(2)}
          <input type="range" min={0} max={1.5} step={0.05} defaultValue={o.friction} onMouseUp={(e) => onPatch(o.id, { friction: +(e.target as HTMLInputElement).value })} />
        </label>
        <label className="check"><input type="checkbox" checked={o.static} onChange={(e) => onPatch(o.id, { static: e.target.checked })} /> fixed to table</label>
      </div>
      <div className="row">
        <span className="small">Real height</span>
        <input className="narrow" placeholder="cm" value={calib} onChange={(e) => setCalib(e.target.value)} />
        <button disabled={!(+calib > 0)} onClick={() => onCalibrate(o.id, "height", +calib / 100)}
          title="Rescales the whole scene so this object has its true height (fixes monocular depth scale)">Calibrate scale</button>
        <span className="muted small">×{depthScale.toFixed(2)}</span>
      </div>
      <button className="link danger" onClick={() => onDiscard(o.id)}>Discard this detection</button>
    </div>
  );
}
