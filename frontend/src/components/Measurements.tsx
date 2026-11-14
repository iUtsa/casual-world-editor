import { useEffect, useState } from "react";
import { api } from "../api";
import LineChart from "./LineChart";
import type { Measure, Scene, Sim } from "../types";

/** Measured kinematics for one object: headline tiles, four single-measure charts, table view. */
export default function Measurements({ sim, scene, focus, setFocus }: {
  sim: Sim | null; scene: Scene | null; focus: string | null; setFocus: (id: string) => void;
}) {
  const [m, setM] = useState<Measure | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [table, setTable] = useState(false);
  const candidates = sim ? sim.object_ids.filter((id) => sim.kinematics?.[id]?.v.some((v) => v && Math.hypot(v[0], v[1], v[2]) > 0.03)) : [];
  const oid = focus && candidates.includes(focus) ? focus : candidates[0] ?? null;
  useEffect(() => {
    setM(null); setErr(null);
    if (sim && oid) api.measure(oid).then(setM).catch((e) => setErr(e.message));
  }, [sim?.key, oid]);   // eslint-disable-line react-hooks/exhaustive-deps
  if (!sim || !scene) return null;
  if (!candidates.length) return <section className="panel measure"><h3>Measurements</h3><p className="muted small">Nothing moved yet.</p></section>;
  const label = (id: string) => sim.labels?.[id] ?? scene.objects.find((o) => o.id === id)?.label ?? id;
  const mt = m?.metrics ?? {};
  const tiles: [string, string][] = m ? [
    ["Top speed", `${mt.max_speed_mps?.toFixed(2)} m/s`],
    ["Peak acceleration", `${mt.peak_acceleration_mps2?.toFixed(1)} m/s²`],
    ["Path length", `${((mt.path_length_m ?? 0) * 100).toFixed(1)} cm`],
    ["Peak impact force", `${mt.peak_object_contact_force_N?.toFixed(2)} N`],
    ["Max kinetic energy", `${((mt.max_kinetic_energy_J ?? 0) * 1000).toFixed(1)} mJ`],
  ] : [];
  return (
    <section className="panel measure">
      <h3>Measurements
        <select className="right" value={oid ?? ""} onChange={(e) => setFocus(e.target.value)}>
          {candidates.map((id) => <option key={id} value={id}>{label(id)}</option>)}
        </select>
      </h3>
      {err && <p className="error small">{err}</p>}
      {m && (
        <>
          <div className="tiles">{tiles.map(([k, v]) => <div key={k} className="tile"><span className="muted small">{k}</span><b>{v}</b></div>)}</div>
          <p className="muted small">Latest experiment with {m.label}; mass {mt.mass_kg} kg. Measured from the MuJoCo state at 60 Hz.</p>
          <div className="charts">
            <LineChart t={m.t} y={m.speed} title="Speed" unit="m/s" duration={sim.duration} />
            <LineChart t={m.t} y={m.acc} title="Acceleration" unit="m/s²" duration={sim.duration} digits={1} />
            <LineChart t={m.t} y={m.F_objects} title="Impact force from other objects" unit="N" duration={sim.duration} />
            <LineChart t={m.t} y={m.KE.map((x) => x * 1000)} title="Kinetic energy" unit="mJ" duration={sim.duration} digits={1} />
          </div>
          <button className="link" onClick={() => setTable(!table)}>{table ? "Hide" : "Show"} table</button>
          {table && (
            <table className="mtable"><tbody>
              {Object.entries(mt).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{String(v)}</td></tr>)}
            </tbody></table>
          )}
        </>
      )}
    </section>
  );
}
