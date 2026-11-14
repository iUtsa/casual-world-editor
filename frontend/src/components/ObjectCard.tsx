/* The object you clicked (in the camera view or the 3D twin): what the scan knows about it, and what the physics
 * measured the last time it moved. Everything here comes from the scan or the MuJoCo run, nothing typed in. */
import { useEffect, useState } from "react";
import { api } from "../api";
import LineChart from "./LineChart";
import type { Asset, Measure, Scene, Sim } from "../types";

interface Props { id: string; scene: Scene; sim: Sim | null; assets: Record<string, Asset>; onClose: () => void; onAsk: (text: string) => void }

const cm = (m: number) => `${(m * 100).toFixed(m < 0.1 ? 1 : 0)}`;

export default function ObjectCard({ id, scene, sim, assets, onClose, onAsk }: Props) {
  const o = scene.objects.find((x) => x.id === id);
  const sp = sim?.spawned?.[id];
  const label = o?.label ?? sp?.label ?? sim?.labels?.[id] ?? id;
  const moved = !!sim?.kinematics?.[id]?.v.some((v) => v && Math.hypot(v[0], v[1], v[2]) > 0.03);
  const [m, setM] = useState<Measure | null>(null);
  useEffect(() => {
    setM(null);
    if (moved) api.measure(id).then(setM).catch(() => setM(null));
  }, [id, sim?.key, moved]);
  const dims = o ? o.size : sp ? (assets[sp.asset]?.dims ?? sp.proxy?.dims) : null;
  const mass = sim?.masses?.[id] ?? o?.mass;
  const mt = m?.metrics ?? {};
  return (
    <section className="obj-card">
      <header>
        <div>
          <b>{label}</b>
          <div className="small muted">{o ? "real object from your camera" : "added object"}{o?.material && o.material !== "unknown" ? ` · ${o.material}` : ""}</div>
        </div>
        <button className="x" onClick={onClose} aria-label="close">×</button>
      </header>
      <div className="obj-facts">
        {dims && <div><span>Size</span><b>{o?.shape === "cylinder" ? `⌀${cm(dims[0])} × ${cm(dims[2])} cm` : `${cm(dims[0])} × ${cm(dims[1])} × ${cm(dims[2])} cm`}</b></div>}
        {mass !== undefined && <div><span>Mass</span><b>{mass < 1 ? `${Math.round(mass * 1000)} g` : `${mass.toFixed(2)} kg`}</b></div>}
        {o && <div><span>Friction</span><b>μ {o.friction.toFixed(2)}</b></div>}
        {o && <div><span>Resting on</span><b>{o.supported_by.map((s) => scene.objects.find((x) => x.id === s)?.label ?? s).join(", ") || "—"}</b></div>}
      </div>
      {moved && m ? (
        <>
          <div className="obj-facts hot">
            <div><span>Top speed</span><b>{mt.max_speed_mps?.toFixed(2)} m/s</b></div>
            <div><span>Travelled</span><b>{cm(mt.path_length_m ?? 0)} cm</b></div>
            <div><span>Hardest hit</span><b>{(mt.peak_object_contact_force_N ?? 0) > 0.05 ? `${mt.peak_object_contact_force_N.toFixed(1)} N` : "none"}</b></div>
            <div><span>Max energy</span><b>{(mt.max_kinetic_energy_J ?? 0) >= 1 ? `${mt.max_kinetic_energy_J.toFixed(2)} J` : `${((mt.max_kinetic_energy_J ?? 0) * 1000).toFixed(1)} mJ`}</b></div>
          </div>
          <LineChart t={m.t} y={m.speed} title="Speed" unit="m/s" duration={sim!.duration} />
        </>
      ) : (
        <p className="small muted obj-tip">Drag it to push it, pinch it with ✋ Hands, or ask below — its motion is measured here.</p>
      )}
      <div className="obj-asks">
        {[`drop a baseball on the ${label}`, `what if the ${label} was twice as heavy?`, `how heavy is the ${label}?`].map((q) =>
          <button key={q} className="chip" onClick={() => onAsk(q)}>{q}</button>)}
      </div>
    </section>
  );
}
