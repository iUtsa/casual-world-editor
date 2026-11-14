/* Results, measurements, edit-and-re-run, and run comparison for an experiment.
 * Every number shown comes from the server's analysis of the recorded simulation. */
import { useEffect, useMemo, useState } from "react";
import LineChart from "./LineChart";
import type { CircuitRun, ExperimentSpec, ExpRun, GravityRun, ResultRow, RunSummary } from "../types";
import TheoryCheck from "./TheoryCheck";

// ---------------------------------------------------------------- results
const SERIES = [
  { id: "speed", label: "Speed", unit: "m/s" },
  { id: "height", label: "Height", unit: "m" },
  { id: "ke", label: "Kinetic energy", unit: "J" },
  { id: "momentum", label: "Momentum", unit: "kg·m/s" },
  { id: "force", label: "Contact force", unit: "N" },
  { id: "acc", label: "Acceleration", unit: "m/s²" },
] as const;
type SeriesId = typeof SERIES[number]["id"];

function series(run: ExpRun, sid: string, kind: SeriesId) {
  const sim = run.sim, k = sim.kinematics[sid];
  const m = sim.masses?.[sid] ?? 1;
  const t: number[] = [], y: number[] = [];
  const t0 = sim.spawned?.[sid]?.t ?? 0;
  if (!k) return { t, y };
  const f0 = Math.round(t0 * sim.fps) + 2;
  let prev: number[] | null = null;
  for (let f = f0; f < k.v.length; f += 1) {
    const v = k.v[f], c = k.c?.[f];
    if (!v) continue;
    const sp = Math.hypot(v[0], v[1], v[2]);
    let val = 0;
    switch (kind) {
      case "speed": val = sp; break;
      case "height": val = c ? c[2] : 0; break;
      case "ke": val = k.KE[f] ?? 0; break;
      case "momentum": val = m * sp; break;
      case "force": val = k.F_contact[f] ?? 0; break;
      case "acc": val = prev ? Math.hypot(v[0] - prev[0], v[1] - prev[1], v[2] - prev[2]) * sim.fps : 0; break;
    }
    prev = v;
    t.push(f / sim.fps); y.push(val);
  }
  if (kind === "acc" && y.length > 4) for (let i = 1; i < y.length - 1; i++) y[i] = (y[i - 1] + y[i] + y[i + 1]) / 3;
  return { t, y };
}

function fmtVal(r: ResultRow) { return r.text; }

export function ResultsTab({ run, pinned, setPinned }: { run: ExpRun; pinned: string[]; setPinned: (p: string[]) => void }) {
  const movers = run.objects.filter((o) => o.sid && !o.support);
  const [obj, setObj] = useState<string>(movers.find((o) => !o.fixed)?.sid ?? movers[0]?.sid ?? "");
  const [ser, setSer] = useState<SeriesId>("speed");
  const [pickObj, setPickObj] = useState<string>(movers[0]?.name ?? "");
  const [showAll, setShowAll] = useState(false);
  useEffect(() => { setObj(movers.find((o) => !o.fixed)?.sid ?? movers[0]?.sid ?? ""); setPickObj(movers[0]?.name ?? ""); }, [run.id]);   // eslint-disable-line react-hooks/exhaustive-deps
  const all = useMemo(() => [...run.results, ...run.catalog.filter((c) => !run.results.some((r) => r.label === c.label))], [run]);
  const PRIORITY = [/acceleration down the ramp/i, /^Acceleration \(theory/i, /peak collision force/i, /speed just before impact/i, /impact speed/i, /period \(measured\)/i, /landed earlier/i, /tension \(average/i,
    /moved$/i, /terminal speed \(measured/i, /energy lost/i, /restitution/i, /launch speed/i, /range/i, /rose|sank/i, /bounce height/i, /hits/i];
  const rank = (r: ResultRow) => { const i = PRIORITY.findIndex((p) => p.test(r.label)); return i < 0 ? 99 : i; };
  const key = run.results.filter((r) => !r.label.includes(": top speed") && !r.label.includes(": distance travelled") && r.label !== "Agreement with theory")
    .map((r, i) => ({ r, i })).sort((a, b) => rank(a.r) - rank(b.r) || a.i - b.i).map((x) => x.r);
  const pinnedRows = all.filter((r) => pinned.includes(r.label) || r.picked);
  const hero = [...pinnedRows, ...key.filter((r) => !pinnedRows.includes(r))].slice(0, Math.max(4, pinnedRows.length));
  const toggle = (l: string) => setPinned(pinned.includes(l) ? pinned.filter((x) => x !== l) : [...pinned, l]);
  const s = useMemo(() => (obj ? series(run, obj, ser) : { t: [], y: [] }), [run, obj, ser]);
  const serMeta = SERIES.find((x) => x.id === ser)!;
  const objName = movers.find((o) => o.sid === obj)?.name ?? "";
  const quantities = run.catalog.filter((c) => c.about === pickObj);
  return (
    <div className="lab">
      <section className="panel">
        <div className="lab-title">
          <h3>{run.text}</h3>
          <div className="row wrap small">
            <span className="pill">{run.setting === "scene" ? "in your real scene" : run.setting}</span>
            <span className="pill">g = {run.gravity.toFixed(2)} m/s²</span>
            <span className="pill">{run.experiment.understood_by === "rules" ? "understood on-device (rules)" : `understood by ${run.experiment.understood_by}`}</span>
            {run.compute_ms !== undefined && <span className="pill">{(run.compute_ms / 1000).toFixed(1)} s to simulate</span>}
          </div>
          {run.changes.length > 0 && <p className="small changes">Changed: {run.changes.join(" · ")}</p>}
        </div>
        <TheoryCheck rows={run.results} />
        <div className="tiles">
          {hero.map((r) => (
            <button key={r.label} className={`tile ${pinned.includes(r.label) ? "pinned" : ""}`} title={r.why ?? "click to pin / unpin"} onClick={() => toggle(r.label)}>
              <span className="tile-v">{fmtVal(r)}</span>
              <span className="tile-l">{r.label}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="panel why">
        <h3>Why it happened</h3>
        {run.story.map((p, i) => <p key={i}>{p}</p>)}
        {run.notes.length > 0 && <ul className="notes">{run.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
      </section>

      <section className="panel">
        <h3>Graphs</h3>
        <div className="row wrap">
          <select value={obj} onChange={(e) => setObj(e.target.value)}>{movers.map((o) => <option key={o.sid} value={o.sid!}>{o.name}</option>)}</select>
          <div className="seg-toggle small">{SERIES.map((x) => <button key={x.id} className={ser === x.id ? "on" : ""} onClick={() => setSer(x.id)}>{x.label}</button>)}</div>
        </div>
        {s.t.length > 1 ? <LineChart t={s.t} y={s.y} title={`${objName}: ${serMeta.label.toLowerCase()}`} unit={serMeta.unit} duration={run.duration} /> : <p className="muted small">No motion recorded for this object.</p>}
      </section>

      <section className="panel">
        <h3>Measure anything</h3>
        <p className="muted small">Pin a measurement to keep it at the top; it stays pinned when you change things and re-run.</p>
        <div className="row wrap">
          <select value={pickObj} onChange={(e) => setPickObj(e.target.value)}>{movers.map((o) => <option key={o.name}>{o.name}</option>)}</select>
          <select value="" onChange={(e) => { if (e.target.value) toggle(e.target.value); }}>
            <option value="">+ add a measurement…</option>
            {quantities.map((q) => <option key={q.label} value={q.label}>{q.quantity}: {q.text}</option>)}
          </select>
        </div>
        <table className="mtable full">
          <tbody>
            {(showAll ? all : run.results).map((r) => (
              <tr key={r.label} className={pinned.includes(r.label) ? "on" : ""}>
                <td><input type="checkbox" checked={pinned.includes(r.label)} onChange={() => toggle(r.label)} aria-label={`pin ${r.label}`} /></td>
                <td title={r.why ?? ""}>{r.label}</td><td className="num">{r.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <button className="link" onClick={() => setShowAll(!showAll)}>{showAll ? "Show key results only" : `Show every measurement (${all.length})`}</button>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------- edit & re-run
const GRAV: [string, number | null][] = [["Earth", 9.81], ["Moon", 1.62], ["Mars", 3.71], ["Jupiter", 24.79], ["Zero-g", 0]];
const SETTINGS = ["auto", "table", "ground", "water", "air", "space", "scene"];

export function EditTab({ run, busy, onRerun, hasScene }: { run: ExpRun; busy: boolean; hasScene: boolean; onRerun: (spec: ExperimentSpec) => void }) {
  const [spec, setSpec] = useState<ExperimentSpec>(() => structuredClone(run.experiment));
  useEffect(() => setSpec(structuredClone(run.experiment)), [run.id]);   // eslint-disable-line react-hooks/exhaustive-deps
  const resolved = (name: string) => run.objects.find((o) => o.name === name || o.name.startsWith(name + " "));
  const setObj = (i: number, k: "mass_kg" | "size_m" | "bounciness", v: number | null) => setSpec((s) => { const c = structuredClone(s); c.objects[i][k] = v; return c; });
  const setAct = (i: number, k: "speed" | "height" | "distance" | "angle_deg" | "stiffness" | "force", v: number | null) => setSpec((s) => { const c = structuredClone(s); c.actions[i][k] = v; return c; });
  const num = (v: string) => (v.trim() === "" ? null : +v);
  return (
    <div className="lab">
      <section className="panel">
        <h3>World</h3>
        <label className="field">Where
          <select value={spec.setting} onChange={(e) => setSpec({ ...spec, setting: e.target.value })}>
            {SETTINGS.filter((x) => x !== "scene" || hasScene).map((x) => <option key={x} value={x}>{x === "scene" ? "my real scene" : x}</option>)}
          </select></label>
        <div className="field">Gravity
          <div className="row wrap">
            {GRAV.map(([n, g]) => <button key={n} className={(spec.gravity ?? run.gravity) === g ? "on" : ""} onClick={() => setSpec({ ...spec, gravity: g })}>{n}</button>)}
            <input type="number" step="0.01" style={{ width: 80 }} value={spec.gravity ?? ""} placeholder={run.gravity.toFixed(2)} onChange={(e) => setSpec({ ...spec, gravity: num(e.target.value) })} /> m/s²
          </div>
        </div>
        <label className="field">Duration (s)
          <input type="number" step="0.5" min="1" max="20" value={spec.seconds ?? ""} placeholder={run.duration.toFixed(1)} onChange={(e) => setSpec({ ...spec, seconds: num(e.target.value) })} /></label>
      </section>
      <section className="panel">
        <h3>Objects</h3>
        {spec.objects.map((o, i) => {
          const r = resolved(o.name);
          return (
            <div key={i} className="edit-obj">
              <b>{o.count > 1 ? `${o.count} × ` : ""}{o.name}</b>
              <div className="grid3">
                <label className="field">Mass (kg)<input type="number" step="any" value={o.mass_kg ?? ""} placeholder={r ? fmt(r.mass) : ""} onChange={(e) => setObj(i, "mass_kg", num(e.target.value))} /></label>
                <label className="field">Size (m)<input type="number" step="any" value={o.size_m ?? ""} placeholder={r ? fmt(Math.max(...r.dims)) : ""} onChange={(e) => setObj(i, "size_m", num(e.target.value))} /></label>
                <label className="field">Bounciness<input type="range" min={0} max={0.98} step={0.01} value={o.bounciness ?? r?.bounciness ?? 0.3} onChange={(e) => setObj(i, "bounciness", +e.target.value)} /><span className="small">{(o.bounciness ?? r?.bounciness ?? 0.3).toFixed(2)}</span></label>
              </div>
              {r && <p className="muted small">{r.note}{r.source === "estimate" ? " — estimated: set the real values if you know them" : ""}</p>}
            </div>
          );
        })}
      </section>
      <section className="panel">
        <h3>Actions</h3>
        {spec.actions.map((a, i) => (
          <div key={i} className="edit-obj">
            <b>{a.do}</b> <span className="muted">{a.a}{a.b ? ` → ${a.b}` : ""}</span>
            <div className="grid3">
              {["collide", "hit", "throw", "push", "roll", "fight", "chase", "race"].includes(a.do) && <label className="field">Speed (m/s)<input type="number" step="any" value={a.speed && a.speed > 0 ? a.speed : ""} placeholder="auto" onChange={(e) => setAct(i, "speed", num(e.target.value))} /></label>}
              {["drop", "bounce"].includes(a.do) && <label className="field">Height (m)<input type="number" step="any" value={a.height ?? ""} placeholder="1" onChange={(e) => setAct(i, "height", num(e.target.value))} /></label>}
              {["rope", "spring", "swing", "collide", "hit", "pull"].includes(a.do) && <label className="field">{["rope", "spring", "swing"].includes(a.do) ? "Length (m)" : "Distance (m)"}<input type="number" step="any" value={a.distance ?? ""} placeholder="auto" onChange={(e) => setAct(i, "distance", num(e.target.value))} /></label>}
              {["swing", "throw"].includes(a.do) && <label className="field">Angle (°)<input type="number" step="any" value={a.angle_deg ?? ""} placeholder="auto" onChange={(e) => setAct(i, "angle_deg", num(e.target.value))} /></label>}
              {a.do === "spring" && <label className="field">Stiffness (N/m)<input type="number" step="any" value={a.stiffness ?? ""} placeholder="auto" onChange={(e) => setAct(i, "stiffness", num(e.target.value))} /></label>}
              {a.do === "pull" && <label className="field">Pull force (N)<input type="number" step="any" value={a.force ?? ""} placeholder="grip-limited" onChange={(e) => setAct(i, "force", num(e.target.value))} /></label>}
            </div>
          </div>
        ))}
      </section>
      <div className="row">
        <button className="primary" disabled={busy} onClick={() => onRerun(spec)}>{busy ? "Simulating…" : "Re-run with these values"}</button>
        <button onClick={() => setSpec(structuredClone(run.experiment))}>Reset</button>
      </div>
    </div>
  );
}
function fmt(v: number) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2); }

// ---------------------------------------------------------------- runs & comparison
export function RunsTab({ runs, current, onOpen, getRun }: { runs: RunSummary[]; current: { id: string; parent: string | null; results: ResultRow[]; catalog?: ResultRow[] }; onOpen: (id: string) => void; getRun: (id: string) => Promise<{ results: ResultRow[]; catalog?: ResultRow[] }> }) {
  const [other, setOther] = useState<{ results: ResultRow[]; catalog?: ResultRow[] } | null>(null);
  const [otherId, setOtherId] = useState<string>("");
  useEffect(() => {
    const def = current.parent ?? runs.filter((r) => r.id !== current.id).slice(-1)[0]?.id ?? "";
    setOtherId(def);
  }, [current.id]);   // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (otherId) getRun(otherId).then(setOther).catch(() => setOther(null)); else setOther(null); }, [otherId, getRun]);
  const rows = useMemo(() => {
    if (!other) return [];
    const A = new Map([...other.results, ...(other.catalog ?? [])].map((r) => [r.label, r]));
    return [...current.results, ...(current.catalog ?? []).filter((c) => /top speed|max height|peak momentum|peak kinetic|peak contact/.test(c.label))]
      .filter((r) => A.has(r.label) && r.value !== null && A.get(r.label)!.value !== null)
      .map((r) => ({ label: r.label, a: A.get(r.label)!, b: r }));
  }, [current, other]);
  return (
    <div className="lab">
      <section className="panel">
        <h3>Your experiments</h3>
        <ol className="runs">
          {[...runs].reverse().map((r) => (
            <li key={r.id} className={r.id === current.id ? "on" : ""}>
              <button className="link" onClick={() => onOpen(r.id)}>{r.text}</button>
              <span className="muted small"> · {r.setting}{r.parent ? " · follow-up" : ""}</span>
              {r.changes.length > 0 && <div className="muted small">{r.changes.join(" · ")}</div>}
            </li>
          ))}
        </ol>
      </section>
      <section className="panel">
        <h3>Compare</h3>
        <label className="field">Compare this run with
          <select value={otherId} onChange={(e) => setOtherId(e.target.value)}>
            <option value="">—</option>
            {runs.filter((r) => r.id !== current.id).map((r) => <option key={r.id} value={r.id}>{r.text}</option>)}
          </select></label>
        {other && rows.length > 0 && (
          <table className="mtable full">
            <thead><tr><th>Measurement</th><th className="num">before</th><th className="num">now</th><th className="num">change</th></tr></thead>
            <tbody>
              {rows.map(({ label, a, b }) => {
                const d = a.value ? ((b.value! - a.value) / Math.abs(a.value)) * 100 : 0;
                return <tr key={label}><td>{label}</td><td className="num">{a.text}</td><td className="num">{b.text}</td>
                  <td className="num">{Math.abs(d) < 0.5 ? "≈" : `${d > 0 ? "▲" : "▼"} ${Math.abs(d) < 10 ? Math.abs(d).toFixed(1) : Math.abs(d).toFixed(0)}%`}</td></tr>;
              })}
            </tbody>
          </table>
        )}
        {other && rows.length === 0 && <p className="muted small">These two runs don't share measurements (different objects or actions).</p>}
      </section>
    </div>
  );
}

// ---------------------------------------------------------------- circuits & gravity

export function DomainResults({ run, pinned, setPinned }: { run: CircuitRun | GravityRun; pinned: string[]; setPinned: (p: string[]) => void }) {
  const toggle = (l: string) => setPinned(pinned.includes(l) ? pinned.filter((x) => x !== l) : [...pinned, l]);
  const PRI = run.domain === "circuit"
    ? [/current from the battery/i, /brightness/i, /total resistance/i, /time constant/i, /last about/i, /voltage at the battery/i]
    : [/impact speed/i, /megatons/i, /dinosaur/i, /period \(measured/i, /period \(kepler/i, /orbital speed/i, /time until impact/i, /energy released/i];
  const rank = (r: ResultRow) => { const i = PRI.findIndex((p) => p.test(r.label)); return i < 0 ? 99 : i; };
  const ordered = [...run.results].filter((r) => r.label !== "Agreement with theory").map((r, i) => ({ r, i })).sort((a, b) => rank(a.r) - rank(b.r) || a.i - b.i).map((x) => x.r);
  const hero = [...ordered.filter((r) => pinned.includes(r.label)), ...ordered.filter((r) => !pinned.includes(r.label))].slice(0, 4);
  // charts
  const charts: { title: string; unit: string; t: number[]; y: number[]; xUnit: string }[] = [];
  if (run.domain === "circuit" && run.dynamic) {
    charts.push({ title: "Current from the battery", unit: "mA", t: run.t.map((x) => x * 1000), y: run.source.I.map((x) => x * 1000), xUnit: "ms" });
    const c = run.elements.find((e) => e.kind === "capacitor" || e.kind === "inductor");
    if (c) charts.push({ title: c.kind === "capacitor" ? `${c.label}: voltage` : `${c.label}: current`, unit: c.kind === "capacitor" ? "V" : "mA",
      t: run.t.map((x) => x * 1000), y: c.kind === "capacitor" ? c.V.map(Math.abs) : c.I.map((x) => Math.abs(x) * 1000), xUnit: "ms" });
  }
  if (run.domain === "gravity" && run.bodies.length >= 2) {
    const big = run.bodies.reduce((m, b, i) => (b.mass > run.bodies[m].mass ? i : m), 0);
    const other = big === 0 ? 1 : 0;
    const days = run.sim_seconds > 3 * 86400;
    const tt = run.t.map((x) => (days ? x / 86400 : x / 3600));
    const dist = run.pos.map((fr) => Math.hypot(fr[other][0] - fr[big][0], fr[other][1] - fr[big][1], fr[other][2] - fr[big][2]) / 1000);
    charts.push({ title: `Distance between ${run.bodies[other].name} and ${run.bodies[big].name}`, unit: "km", t: tt, y: dist, xUnit: days ? "days" : "h" });
    const sp = run.pos.map((fr, k) => { const k2 = Math.min(run.pos.length - 1, k + 1), k1 = k2 - 1; const d = (run.t[k2] - run.t[k1]) || 1;
      const a = run.pos[k1][other], b = run.pos[k2][other]; return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / d / 1000; });
    charts.push({ title: `${run.bodies[other].name}: speed`, unit: "km/s", t: tt, y: sp, xUnit: days ? "days" : "h" });
  }
  return (
    <div className="lab">
      <section className="panel">
        <div className="lab-title">
          <h3>{run.text}</h3>
          <div className="row wrap small">
            <span className="pill">{run.domain === "circuit" ? "electric circuit · solved with nodal analysis" : "gravity · N-body simulation"}</span>
          </div>
          {run.changes.length > 0 && <p className="small changes">Changed: {run.changes.join(" · ")}</p>}
        </div>
        <TheoryCheck rows={run.results} />
        <div className="tiles">
          {hero.map((r) => (
            <button key={r.label} className={`tile ${pinned.includes(r.label) ? "pinned" : ""}`} onClick={() => toggle(r.label)}>
              <span className="tile-v">{r.text}</span><span className="tile-l">{r.label}</span>
            </button>
          ))}
        </div>
      </section>
      <section className="panel why"><h3>Why it happened</h3>{run.story.map((p, i) => <p key={i}>{p}</p>)}</section>
      {charts.length > 0 && <section className="panel"><h3>Graphs</h3>{charts.map((c) => <LineChart key={c.title} t={c.t} y={c.y} title={c.title} unit={c.unit}
        duration={c.t[c.t.length - 1] || 1} xUnit={c.xUnit} now={-1} />)}</section>}
      <section className="panel">
        <h3>Every measurement</h3>
        <table className="mtable full"><tbody>
          {run.results.map((r) => <tr key={r.label} className={pinned.includes(r.label) ? "on" : ""}>
            <td><input type="checkbox" checked={pinned.includes(r.label)} onChange={() => toggle(r.label)} aria-label={`pin ${r.label}`} /></td>
            <td>{r.label}</td><td className="num">{r.text}</td></tr>)}
        </tbody></table>
      </section>
    </div>
  );
}

const CIRCUIT_CHANGES = ["what if it's 12 V", "double the voltage", "add another bulb", "add a 100 ohm resistor", "put them in parallel", "put them in series",
  "open the switch", "add a red LED", "remove a bulb"];
const GRAVITY_CHANGES = ["what if it was 10 times heavier", "what if it was half as heavy", "launch it at 30 km/s", "what if Mars instead", "what if Earth instead"];
export function DomainEdit({ run, busy, onSay }: { run: CircuitRun | GravityRun; busy: boolean; onSay: (t: string) => void }) {
  const [txt, setTxt] = useState("");
  const opts = run.domain === "circuit" ? CIRCUIT_CHANGES : GRAVITY_CHANGES;
  return (
    <div className="lab">
      <section className="panel">
        <h3>Change something and re-run</h3>
        <p className="muted small">Click a change, or say your own — the experiment re-runs with exactly that change and you can compare the runs.</p>
        <div className="row wrap">{opts.map((o) => <button key={o} className="chip" disabled={busy} onClick={() => onSay(o)}>{o}</button>)}</div>
        <form className="row" onSubmit={(e) => { e.preventDefault(); if (txt.trim()) onSay(txt.trim()); }}>
          <input className="grow" value={txt} onChange={(e) => setTxt(e.target.value)} placeholder={run.domain === "circuit" ? "e.g. use a 9 V battery and 3 bulbs in parallel" : "e.g. what if the Moon was 100 times heavier"} />
          <button disabled={busy || !txt.trim()}>Re-run</button>
        </form>
      </section>
    </div>
  );
}
