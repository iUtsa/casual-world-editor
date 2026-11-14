import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import LineChart from "./LineChart";

interface Species { name: string; formula: string; side: string; coefficient: number; symbols: string[]; positions: number[][]; bonds: number[][]; xtb_energy_hartree: number; dHf_kJmol: number | null }
interface Info { equation: string; species: Species[]; dE_xtb_kJmol: number; dH_exp_kJmol: number | null; exothermic: boolean | null; atoms: number;
  dH_std_kJmol: number | null; dH_gas_kJmol: number | null; per_mol_first: { formula: string; kJmol: number } | null; thermo_source: string; xtb_note: string;
  balance: { element: string; left: number; right: number }[]; balanced: boolean }
interface QC { state: string; done: number; total: number; dH_gas_kJmol?: number; measured_gas_kJmol?: number | null; error_kJmol?: number | null;
  uncertainty_kJmol?: number; method?: string; error?: string; imaginary?: string[] }
interface Traj { symbols: string[]; frames: number[][][]; bonds: number[][][]; times_fs: number[]; energy_kj: number[]; events: { t_fs: number; type: string; pair: string; atoms: number[] }[];
  product_bonds_formed: number; product_bonds_expected: number; seconds: number; temperature_K: number; dt_fs: number; steps: number }

// CPK-style colours and covalent radii (Å)
const COLOR: Record<string, string> = { H: "#f2f2f2", C: "#3b3b3b", N: "#3050f8", O: "#ff2a1a", F: "#90e050", Cl: "#1ff01f", Na: "#ab5cf2", S: "#ffd123", P: "#ff8000" };
const RADIUS: Record<string, number> = { H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, Cl: 1.02, Na: 1.66, S: 1.05, P: 1.07 };
const EXAMPLES = ["hydrogen + oxygen -> water", "methane + oxygen -> carbon dioxide + water", "HCl + NaOH -> NaCl + H2O", "ethene + hydrogen -> ethane", "ammonia + HCl -> ... "];

function Env() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pm = new THREE.PMREMGenerator(gl);
    scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    return () => pm.dispose();
  }, [gl, scene]);
  return null;
}

function Controls({ target }: { target: THREE.Vector3 }) {
  const { camera, gl } = useThree();
  useEffect(() => {
    const c = new OrbitControls(camera, gl.domElement);
    c.target.copy(target); c.enableDamping = true; c.autoRotate = true; c.autoRotateSpeed = 0.6;
    let raf = 0;
    const loop = () => { c.update(); raf = requestAnimationFrame(loop); };
    loop();
    return () => { cancelAnimationFrame(raf); c.dispose(); };
  }, [camera, gl, target]);
  return null;
}

/** Atoms + bonds for one frame. Bond radius and opacity follow the computed bond order. */
function Molecule({ symbols, frame, bonds, flash }: { symbols: string[]; frame: number[][]; bonds: number[][]; flash: Set<string> }) {
  const atoms = useRef<THREE.InstancedMesh>(null);
  const mats = useMemo(() => {
    const m: Record<string, THREE.MeshPhysicalMaterial> = {};
    for (const s of new Set(symbols)) m[s] = new THREE.MeshPhysicalMaterial({ color: COLOR[s] ?? "#ff69b4", roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.15, metalness: 0.05 });
    return m;
  }, [symbols]);
  const bondMat = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#c9ced6", roughness: 0.35, clearcoat: 0.6, transparent: true }), []);
  const glowMat = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#facc15", emissive: "#f59e0b", emissiveIntensity: 1.2, roughness: 0.3, transparent: true }), []);
  const up = new THREE.Vector3(0, 1, 0);
  return (
    <group>
      {symbols.map((s, i) => (
        <mesh key={i} position={frame[i] as [number, number, number]} material={mats[s]} castShadow>
          <sphereGeometry args={[(RADIUS[s] ?? 0.8) * 0.48, 48, 32]} />
        </mesh>
      ))}
      {bonds.map(([i, j, order]) => {
        const a = new THREE.Vector3(...(frame[i] as [number, number, number]));
        const b = new THREE.Vector3(...(frame[j] as [number, number, number]));
        const len = a.distanceTo(b);
        const mid = a.clone().add(b).multiplyScalar(0.5);
        const q = new THREE.Quaternion().setFromUnitVectors(up, b.clone().sub(a).normalize());
        const lit = flash.has(`${Math.min(i, j)}-${Math.max(i, j)}`);
        const n = order > 1.6 ? (order > 2.5 ? 3 : 2) : 1;
        const r = 0.09 * Math.min(1, Math.max(0.35, order));
        const side = new THREE.Vector3(1, 0, 0).applyQuaternion(q).multiplyScalar(0.12);
        return Array.from({ length: n }, (_, k) => {
          const off = side.clone().multiplyScalar(k - (n - 1) / 2);
          return (
            <mesh key={`${i}-${j}-${k}`} position={mid.clone().add(off)} quaternion={q} material={lit ? glowMat : bondMat}
              onUpdate={(m) => ((m.material as THREE.MeshPhysicalMaterial).opacity = Math.min(1, 0.25 + order))}>
              <cylinderGeometry args={[n > 1 ? r * 0.7 : r, n > 1 ? r * 0.7 : r, len, 20]} />
            </mesh>
          );
        });
      })}
      <instancedMesh ref={atoms} args={[undefined, undefined, 0]} />
    </group>
  );
}

function Player({ traj, t, setT, playing, setPlaying }: { traj: Traj; t: number; setT: (v: number) => void; playing: boolean; setPlaying: (v: boolean) => void }) {
  useFrame((_, dt) => {
    if (!playing) return;
    const T = traj.times_fs[traj.times_fs.length - 1];
    const nt = t + dt * 120;          // 120 fs of simulated time per real second
    if (nt >= T) { setT(T); setPlaying(false); } else setT(nt);
  });
  return null;
}

export default function ChemView({ initial }: { initial?: string | null } = {}) {
  const [text, setText] = useState(initial || "methane + oxygen -> carbon dioxide + water");
  const [info, setInfo] = useState<Info | null>(null);
  const [traj, setTraj] = useState<Traj | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [qc, setQc] = useState<QC | null>(null);
  const runId = useRef(0);

  useEffect(() => { if (initial) { setText(initial); run(initial); } }, [initial]);   // eslint-disable-line react-hooks/exhaustive-deps
  /** Quantum chemistry heat of reaction (M06-2X DFT + vibrations) in the background; independent of the animation. */
  async function quantum(txt: string, id: number) {
    try {
      const r = await fetch("/api/chem/thermo", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: txt }) });
      let j = await r.json();
      if (!r.ok) throw new Error(j.detail);
      while (j.state === "running") {
        if (runId.current !== id) return;
        setQc({ ...j });
        await new Promise((res) => setTimeout(res, 1000));
        j = await (await fetch(`/api/chem/thermo/${j.id}`)).json();
      }
      if (runId.current === id) setQc(j);
    } catch (e) { if (runId.current === id) setQc({ state: "error", error: (e as Error).message, done: 0, total: 0 }); }
  }

  async function run(txt: string) {
    const id = ++runId.current;
    setErr(null); setTraj(null); setInfo(null); setT(0); setQc(null);
    quantum(txt, id);
    setBusy("Balancing and optimising molecules with GFN2-xTB…");
    try {
      const r = await fetch("/api/chem/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: txt }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail);
      if (runId.current !== id) return;
      setInfo(j);
      const s = await fetch("/api/chem/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: txt }) });
      const sj = await s.json();
      if (!s.ok) throw new Error(sj.detail);
      for (;;) {
        await new Promise((res) => setTimeout(res, 700));
        if (runId.current !== id) return;                 // a newer reaction was asked for: stop following this one
        const st = await (await fetch(`/api/chem/job/${sj.job}`)).json();
        setBusy(`Quantum reaction dynamics (xTB forces on every atom)… ${Math.round(st.progress * 100)}%`);
        if (st.state === "done") { setTraj(st.trajectory); setPlaying(true); break; }
        if (st.state === "error") throw new Error(st.error);
      }
    } catch (e) { if (runId.current === id) setErr((e as Error).message); } finally { if (runId.current === id) setBusy(null); }
  }

  const fi = traj ? Math.min(traj.frames.length - 1, Math.max(0, traj.times_fs.findIndex((x) => x >= t))) : 0;
  const center = useMemo(() => {
    if (!traj) return new THREE.Vector3();
    const f = traj.frames[0];
    return new THREE.Vector3(...[0, 1, 2].map((k) => f.reduce((a, p) => a + p[k], 0) / f.length) as [number, number, number]);
  }, [traj]);
  const flash = useMemo(() => new Set((traj?.events ?? []).filter((e) => Math.abs(e.t_fs - t) < 25).map((e) => `${Math.min(...e.atoms)}-${Math.max(...e.atoms)}`)), [traj, t]);
  const T = traj ? traj.times_fs[traj.times_fs.length - 1] : 1;

  return (
    <div className="chem">
      <form className="row" onSubmit={(e) => { e.preventDefault(); run(text); }}>
        <input className="grow cmd-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a reaction, e.g. hydrogen + oxygen -> water" />
        <button className="primary narrow-btn">Simulate</button>
      </form>
      <div className="row wrap small">{EXAMPLES.slice(0, 4).map((x) => <button key={x} className="chip" onClick={() => { setText(x); run(x); }}>{x}</button>)}</div>
      {busy && <p className="busy small">{busy}</p>}
      {err && <p className="error small">{err}</p>}
      {info && (
        <div className="chem-eq">
          <div className="eq">{info.equation}</div>
          <div className="chem-balance small">
            {info.balanced ? "✓ balanced:" : "✗ not balanced:"} {info.balance.map((b) => <span key={b.element}><b>{b.element}</b> {b.left} = {b.right}</span>)}
          </div>
          {info.dH_std_kJmol !== null ? (
            <div className="chem-thermo">
              <div className="tile-big">
                <span className="small muted">Heat of reaction ΔH° (25 °C, standard states)</span>
                <b>{info.dH_std_kJmol.toFixed(1)} <small>kJ/mol</small></b>
                <span className={`pill ${info.exothermic ? "warn" : ""}`}>{info.exothermic ? "exothermic: releases heat" : "endothermic: absorbs heat"}</span>
              </div>
              {info.per_mol_first && info.per_mol_first.kJmol !== info.dH_std_kJmol && <div className="tile-big"><span className="small muted">per mole of {info.per_mol_first.formula}</span><b>{info.per_mol_first.kJmol.toFixed(1)} <small>kJ/mol</small></b></div>}
              {info.dH_gas_kJmol !== null && info.dH_gas_kJmol !== info.dH_std_kJmol && (
                <div className="tile-big"><span className="small muted">if everything were a gas</span><b>{info.dH_gas_kJmol.toFixed(1)} <small>kJ/mol</small></b></div>)}
              <p className="small muted">From {info.thermo_source}: ΔH° = Σ ΔHf°(products) − Σ ΔHf°(reactants). Phases as written in the equation.</p>
            </div>
          ) : null}
          <QuantumTile qc={qc} measured={info.dH_std_kJmol !== null} />
          <details className="small muted"><summary>Fast quantum estimate (GFN2-xTB): {info.dE_xtb_kJmol} kJ/mol</summary>{info.xtb_note}</details>
        </div>
      )}
      <div className="chem-stage">
        {traj ? (
          <Canvas shadows camera={{ position: [center.x + 9, center.y + 4, center.z + 9], fov: 40 }} style={{ background: "radial-gradient(circle at 50% 35%, #1e293b, #020617)" }}
            onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1.1; }}>
            <Env />
            <Controls target={center} />
            <directionalLight position={[8, 12, 6]} intensity={1.4} castShadow />
            <Player traj={traj} t={t} setT={setT} playing={playing} setPlaying={setPlaying} />
            <Molecule symbols={traj.symbols} frame={traj.frames[fi]} bonds={traj.bonds[fi]} flash={flash} />
          </Canvas>
        ) : info ? (
          <div className="mol-grid">
            {info.species.map((s, k) => (
              <div key={k} className="mol-card"><b>{s.coefficient > 1 ? `${s.coefficient} ` : ""}{s.formula}</b><span className="muted small">{s.name}</span></div>
            ))}
          </div>
        ) : <div className="empty">Type any reaction between small molecules. The equation is balanced exactly, molecules are built in 3D, and the reaction is simulated with quantum-mechanical forces (GFN2-xTB).</div>}
      </div>
      {traj && (
        <>
          <div className="timeline">
            <button onClick={() => { if (t >= T) setT(0); setPlaying(!playing); }}>{playing ? "❚❚" : "▶"}</button>
            <input type="range" className="grow" min={0} max={T} step={1} value={t} onChange={(e) => { setPlaying(false); setT(+e.target.value); }} />
            <span className="mono small">{t.toFixed(0)} / {T.toFixed(0)} fs</span>
          </div>
          <div className="charts">
            <LineChart t={traj.times_fs} y={traj.energy_kj} title="Potential energy along the reaction" unit="kJ/mol" duration={T} now={t} xUnit="fs" digits={0} />
            <div className="panel">
              <h4>What happened</h4>
              <ol className="events">{traj.events.slice(0, 14).map((e, i) => (
                <li key={i} className={Math.abs(e.t_fs - t) < 25 ? "on" : ""} onClick={() => { setPlaying(false); setT(e.t_fs); }}>
                  <b>{e.pair}</b> bond {e.type} <span className="muted">at {e.t_fs.toFixed(0)} fs</span></li>))}</ol>
              <p className="muted small">Products formed: {traj.product_bonds_formed}/{traj.product_bonds_expected} expected bonds. {traj.steps} steps × {traj.dt_fs} fs at {traj.temperature_K} K in {traj.seconds} s.
                Steered dynamics: every atom moves under GFN2-xTB quantum forces while a gentle bias drives the bonds that must change; bonds shown are the computed Wiberg bond orders.</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

/** The quantum-chemistry answer: a check next to measured data, or THE answer (with its benchmarked uncertainty) when no
 *  measurement exists for every molecule. */
function QuantumTile({ qc, measured }: { qc: QC | null; measured: boolean }) {
  if (!qc) return null;
  if (qc.state === "running") return <p className="small muted">⚛ Quantum chemistry (M06-2X DFT) for each molecule… {qc.done}/{qc.total}</p>;
  if (qc.state === "error") return <p className="small warn-text">Quantum chemistry failed: {qc.error}</p>;
  const v = qc.dH_gas_kJmol!;
  const exo = v < 0;
  return (
    <div className="chem-thermo">
      <div className={`tile-big ${measured ? "" : "main"}`}>
        <span className="small muted">{measured ? "Quantum-chemistry check (all gases)" : "Heat of reaction, all gases — quantum chemistry"}</span>
        <b>{v.toFixed(1)} <small>± {qc.uncertainty_kJmol?.toFixed(0)} kJ/mol</small></b>
        {qc.error_kJmol !== null && qc.error_kJmol !== undefined && qc.measured_gas_kJmol
          ? <span className="small">{qc.error_kJmol > 0 ? "+" : ""}{qc.error_kJmol.toFixed(1)} kJ/mol from the measured {qc.measured_gas_kJmol.toFixed(1)} ({(100 * Math.abs(qc.error_kJmol / qc.measured_gas_kJmol)).toFixed(1)}%)</span>
          : <span className={`pill ${exo ? "warn" : ""}`}>{exo ? "exothermic: releases heat" : "endothermic: absorbs heat"}</span>}
      </div>
      <p className="small muted">{qc.method}. Computed on this Mac from first principles; ± comes from the errors measured on 10 reactions with known heats (≈22 kJ/mol, or ≈4 % for big ones like burning fuels).
        {qc.imaginary && qc.imaginary.length > 0 ? ` Note: ${qc.imaginary.join(", ")} not at an exact minimum (imaginary vibration ignored).` : ""}</p>
    </div>
  );
}
