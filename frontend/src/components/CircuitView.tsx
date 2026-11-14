/* A circuit on a board, driven by the solver: every wire carries its solved current (electrons drift
 * along it, faster for more current, opposite to the conventional-current arrows), bulbs glow with
 * their real power, LEDs light only when conducting, resistors carry their actual colour bands, a fuse
 * that blew shows it, and capacitor/inductor circuits animate their transient over time. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { clock } from "../clock";
import type { CircuitElement, CircuitRun } from "../types";

const U = 0.12;            // metres per grid unit
const Z = 0.012;           // wire height above the board

function sampleAt(arr: number[], run: CircuitRun) {
  if (!run.dynamic || arr.length < 2) return arr[arr.length - 1] ?? 0;
  const tt = (clock.t / run.duration) * run.t[run.t.length - 1];
  let i = 0;
  while (i < run.t.length - 1 && run.t[i + 1] < tt) i++;
  return arr[Math.min(arr.length - 1, i)];
}

const BAND = ["#111111", "#8b4513", "#e11d48", "#f97316", "#facc15", "#16a34a", "#2563eb", "#7c3aed", "#9ca3af", "#f8fafc"];
function bands(ohms: number) {
  const e = Math.max(0, Math.floor(Math.log10(Math.max(ohms, 1))) - 1);
  const d = Math.round(ohms / 10 ** e);
  const s = String(Math.min(99, d)).padStart(2, "0");
  return [BAND[+s[0]], BAND[+s[1]], BAND[Math.min(9, e)], "#d4af37"];
}

function Env() {
  const { gl, scene } = useThree();
  useEffect(() => {
    const pm = new THREE.PMREMGenerator(gl);
    scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.background = new THREE.Color("#0f1420");
    return () => pm.dispose();
  }, [gl, scene]);
  return null;
}

function Part({ e, run }: { e: CircuitElement; run: CircuitRun }) {
  const g = useRef<THREE.Group>(null);
  const glow = useRef<THREE.MeshPhysicalMaterial | null>(null);
  const light = useRef<THREE.PointLight>(null);
  const spin = useRef<THREE.Object3D>(null);
  const x = e.pos.x * U, y = e.pos.y * U;
  const parts = useMemo(() => {
    const out: JSX.Element[] = [];
    if (e.kind === "resistor") {
      out.push(<mesh key="b" rotation={[0, 0, Math.PI / 2]} castShadow><capsuleGeometry args={[0.009, 0.034, 6, 16]} /><meshPhysicalMaterial color="#e8d3a8" roughness={0.5} clearcoat={0.4} /></mesh>);
      bands(e.value).forEach((c, k) => out.push(<mesh key={`band${k}`} position={[-0.012 + k * 0.008 + (k === 3 ? 0.004 : 0), 0, 0]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.0095, 0.0095, 0.003, 20]} /><meshStandardMaterial color={c} /></mesh>));
    } else if (e.kind === "bulb") {
      const m = new THREE.MeshPhysicalMaterial({ color: "#fff7d6", transmission: 0.9, roughness: 0.05, thickness: 0.002, emissive: new THREE.Color("#ffd27a"), emissiveIntensity: 0, transparent: true });
      glow.current = m;
      out.push(<mesh key="g" position={[0, 0, 0.032]} material={m}><sphereGeometry args={[0.022, 32, 24]} /></mesh>);
      out.push(<mesh key="base" position={[0, 0, 0.008]}><cylinderGeometry args={[0.011, 0.011, 0.018, 20]} /><meshStandardMaterial color="#9ca3af" metalness={1} roughness={0.3} /></mesh>);
      out.push(<pointLight key="l" ref={light} position={[0, 0, 0.035]} color="#ffcf7a" intensity={0} distance={0.6} decay={2} />);
    } else if (e.kind === "led") {
      const m = new THREE.MeshPhysicalMaterial({ color: e.color, transmission: 0.5, roughness: 0.1, emissive: new THREE.Color(e.color), emissiveIntensity: 0, transparent: true });
      glow.current = m;
      out.push(<mesh key="d" position={[0, 0, 0.016]} material={m}><capsuleGeometry args={[0.007, 0.01, 6, 16]} /></mesh>);
      out.push(<pointLight key="l" ref={light} position={[0, 0, 0.03]} color={e.color} intensity={0} distance={0.4} decay={2} />);
    } else if (e.kind === "switch") {
      out.push(<mesh key="p" position={[0, 0, 0.004]}><boxGeometry args={[0.05, 0.022, 0.008]} /><meshStandardMaterial color="#1f2937" /></mesh>);
      out.push(<mesh key="lever" position={[-0.018, 0, 0.012]} rotation={[0, e.closed ? -0.05 : -0.6, 0]}><boxGeometry args={[0.045, 0.006, 0.003]} /><meshStandardMaterial color="#cbd5e1" metalness={1} roughness={0.2} /></mesh>);
    } else if (e.kind === "capacitor") {
      out.push(<mesh key="c" position={[0, 0, 0.018]} rotation={[Math.PI / 2, 0, 0]}><cylinderGeometry args={[0.012, 0.012, 0.03, 24]} /><meshPhysicalMaterial color="#1e3a8a" clearcoat={1} /></mesh>);
    } else if (e.kind === "inductor") {
      out.push(<mesh key="i" rotation={[0, Math.PI / 2, 0]} position={[0, 0, 0.012]}><torusKnotGeometry args={[0.01, 0.002, 80, 8, 1, 6]} /><meshStandardMaterial color="#b45309" metalness={1} roughness={0.3} /></mesh>);
    } else if (e.kind === "motor") {
      out.push(<mesh key="m" position={[0, 0, 0.016]} rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.016, 0.016, 0.034, 24]} /><meshStandardMaterial color="#cbd5e1" metalness={0.9} roughness={0.3} /></mesh>);
      out.push(<group key="s" ref={spin as React.RefObject<THREE.Group>} position={[0.022, 0, 0.016]}><mesh rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.002, 0.002, 0.012, 8]} /><meshStandardMaterial color="#9ca3af" /></mesh>
        <mesh position={[0.006, 0, 0]}><boxGeometry args={[0.002, 0.03, 0.006]} /><meshStandardMaterial color="#ef4444" /></mesh></group>);
    } else if (e.kind === "buzzer") {
      out.push(<mesh key="b" position={[0, 0, 0.008]}><cylinderGeometry args={[0.014, 0.014, 0.016, 24]} /><meshStandardMaterial color="#111827" /></mesh>);
    } else if (e.kind === "fuse") {
      out.push(<mesh key="f" rotation={[0, 0, Math.PI / 2]} position={[0, 0, 0.008]}><cylinderGeometry args={[0.006, 0.006, 0.04, 16]} /><meshPhysicalMaterial color={e.blown ? "#57534e" : "#e0f2fe"} transmission={e.blown ? 0 : 0.8} roughness={0.1} /></mesh>);
    } else {
      out.push(<mesh key="w" rotation={[0, 0, Math.PI / 2]}><cylinderGeometry args={[0.002, 0.002, 0.04, 8]} /><meshStandardMaterial color="#b87333" metalness={1} roughness={0.3} /></mesh>);
    }
    return out;
  }, [e]);
  useFrame(() => {
    const I = Math.abs(sampleAt(e.I, run)), V = Math.abs(sampleAt(e.V, run));
    if (e.kind === "bulb" && glow.current) {
      const ratedP = 3.0;
      const k = Math.min(1.6, (I * V) / ratedP);
      glow.current.emissiveIntensity = 6 * k;
      if (light.current) light.current.intensity = 1.2 * k;
    }
    if (e.kind === "led" && glow.current) {
      const k = Math.min(1.5, I / 0.02);
      glow.current.emissiveIntensity = 5 * k;
      if (light.current) light.current.intensity = 0.4 * k;
    }
    if (spin.current) spin.current.rotation.x += I * 2;
  });
  return <group ref={g} position={[x, y, Z]} rotation={[0, 0, e.pos.dir === "v" ? Math.PI / 2 : 0]}>{parts}</group>;
}

function Battery({ run }: { run: CircuitRun }) {
  const x = run.source.pos.x * U, y = run.source.pos.y * U;
  const len = 0.09;
  return (
    <group position={[x, y, Z + 0.014]}>
      <mesh castShadow><cylinderGeometry args={[0.014, 0.014, len, 32]} /><meshPhysicalMaterial color="#111827" clearcoat={1} roughness={0.3} /></mesh>
      <mesh position={[0, len * 0.2, 0]}><cylinderGeometry args={[0.0142, 0.0142, len * 0.6, 32]} /><meshPhysicalMaterial color="#d97706" clearcoat={1} roughness={0.3} /></mesh>
      <mesh position={[0, len / 2 + 0.003, 0]}><cylinderGeometry args={[0.005, 0.005, 0.006, 16]} /><meshStandardMaterial color="#cbd5e1" metalness={1} roughness={0.2} /></mesh>
    </group>
  );
}

/** Wires as copper tubes; electrons as moving dots whose speed follows the solved current. */
function Wires({ run }: { run: CircuitRun }) {
  const curves = useMemo(() => run.wires.map((w) => {
    const pts = w.pts.map(([x, y]) => new THREE.Vector3(x * U, y * U, Z));
    const path = new THREE.CurvePath<THREE.Vector3>();
    for (let i = 0; i < pts.length - 1; i++) path.add(new THREE.LineCurve3(pts[i], pts[i + 1]));
    return { path, len: path.getLength(), w };
  }), [run]);
  const dots = useMemo(() => {
    const total = curves.reduce((s, c) => s + Math.max(2, Math.round(c.len / 0.025)), 0);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(total * 3), 3));
    const m = new THREE.PointsMaterial({ color: "#7dd3fc", size: 0.01, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    const p = new THREE.Points(g, m); p.frustumCulled = false;
    return { p, total };
  }, [curves]);
  const phase = useRef(curves.map(() => 0));
  const Imax = useMemo(() => Math.max(1e-6, ...run.wires.map((w) => Math.max(...w.I.map(Math.abs)))), [run]);
  useFrame((_, dt) => {
    const P = dots.p.geometry.attributes.position as THREE.BufferAttribute;
    let k = 0;
    curves.forEach((c, ci) => {
      const I = sampleAt(c.w.I, run);
      // drift speed shown on a log scale so both mA and A are visible; electrons move against conventional current
      const sp = Math.abs(I) < 1e-7 ? 0 : (0.25 + 0.75 * Math.log10(1 + 9 * Math.abs(I) / Imax)) * 0.06;
      phase.current[ci] -= Math.sign(I) * sp * dt * (run.dynamic ? (clock.playing ? clock.rate : 0.0) : 1.0);     // metres along the wire
      const n = Math.max(2, Math.round(c.len / 0.025));
      const spacing = c.len / n;
      for (let j = 0; j < n && k < dots.total; j++, k++) {
        let sArc = (j * spacing + phase.current[ci]) % c.len;
        if (sArc < 0) sArc += c.len;
        const p = c.path.getPointAt(Math.min(0.9999, sArc / c.len));
        P.setXYZ(k, p.x, p.y, p.z + 0.004);
      }
    });
    P.needsUpdate = true;
    dots.p.geometry.setDrawRange(0, k);
    (dots.p.material as THREE.PointsMaterial).opacity = 0.9;
  });
  return (
    <>
      {curves.map((c, i) => (
        <mesh key={i}><tubeGeometry args={[c.path as unknown as THREE.Curve<THREE.Vector3>, Math.max(8, c.w.pts.length * 12), 0.003, 8, false]} />
          <meshStandardMaterial color="#c2762f" metalness={1} roughness={0.35} /></mesh>
      ))}
      <primitive object={dots.p} />
    </>
  );
}

function Labels({ run, host }: { run: CircuitRun; host: React.RefObject<HTMLDivElement> }) {
  const { camera, size } = useThree();
  const els = useRef<HTMLDivElement[]>([]);
  const items = useMemo(() => [{ id: "src", label: run.source.label, x: run.source.pos.x, y: run.source.pos.y }, ...run.elements.map((e) => ({ id: e.id, label: e.label, x: e.pos.x, y: e.pos.y }))], [run]);
  useEffect(() => {
    const h = host.current; if (!h) return;
    els.current = items.map((it) => { const d = document.createElement("div"); d.className = "x-label"; d.innerHTML = `<b>${it.label}</b><span></span>`; h.appendChild(d); return d; });
    return () => { els.current.forEach((d) => d.remove()); };
  }, [items, host]);
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    items.forEach((it, k) => {
      const d = els.current[k]; if (!d) return;
      v.set(it.x * U, it.y * U + 0.03, Z + 0.06).project(camera);
      d.style.transform = `translate(${((v.x + 1) / 2) * size.width}px, ${((1 - v.y) / 2) * size.height}px) translate(-50%, -100%)`;
      let txt = "";
      if (it.id === "src") txt = `${run.source.V} V · ${fmtA(sampleAt(run.source.I, run))}`;
      else { const e = run.elements.find((x) => x.id === it.id)!; const I = sampleAt(e.I, run), V = sampleAt(e.V, run); txt = e.blown ? "BLOWN" : `${fmtA(I)} · ${Math.abs(V).toFixed(2)} V`; }
      (d.lastChild as HTMLElement).textContent = txt;
    });
  });
  return null;
}
function fmtA(I: number) { const a = Math.abs(I); return a < 1e-9 ? "0 A" : a < 1e-3 ? `${(a * 1e6).toFixed(0)} µA` : a < 1 ? `${(a * 1e3).toFixed(0)} mA` : `${a.toFixed(2)} A`; }

function Rig({ run, resetKey }: { run: CircuitRun; resetKey: number }) {
  const { camera, gl } = useThree();
  useEffect(() => {
    const cx = (run.board.w * U) / 2, cy = (run.board.h * U) / 2;
    const ext = Math.max(run.board.w, run.board.h) * U;
    const cam = camera as THREE.PerspectiveCamera;
    cam.up.set(0, 0, 1); cam.fov = 35; cam.near = 0.005; cam.far = 50;
    cam.position.set(cx - ext * 0.1, cy - ext * 1.25, ext * 1.55);
    cam.updateProjectionMatrix();
    const oc = new OrbitControls(cam, gl.domElement);
    oc.target.set(cx, cy, 0); oc.enableDamping = true; oc.update();
    const loop = () => { oc.update(); raf = requestAnimationFrame(loop); };
    let raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); oc.dispose(); };
  }, [run, camera, gl, resetKey]);
  return null;
}

export default function CircuitView({ run, resetKey }: { run: CircuitRun; resetKey: number }) {
  const host = useRef<HTMLDivElement>(null);
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const W = (run.board.w + 1.4) * U, H = (run.board.h + 1.4) * U;
  return (
    <div className="xview">
      <Canvas shadows dpr={[1, 2]} onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; }}>
        <Env />
        <ambientLight intensity={0.25} />
        <directionalLight position={[0.3, -0.4, 1.2]} intensity={1.6} castShadow />
        <mesh position={[(run.board.w * U) / 2, (run.board.h * U) / 2, -0.004]} receiveShadow>
          <boxGeometry args={[W, H, 0.008]} />
          <meshPhysicalMaterial color="#0f3d2a" roughness={0.55} clearcoat={0.5} />
        </mesh>
        <Battery run={run} />
        {run.elements.map((e) => <Part key={e.id} e={e} run={run} />)}
        <Wires run={run} />
        <Labels run={run} host={host} />
        <Rig run={run} resetKey={resetKey} />
      </Canvas>
      <div className="xlabels" ref={host} />
      <div className="xhud">
        <span className="pill dark">{run.dynamic ? `t = ${((clock.t / run.duration) * run.t[run.t.length - 1] * 1000).toFixed(1)} ms of circuit time` : "steady state (DC)"}</span>
        <span className="pill dark">blue dots = electrons (they drift opposite to conventional current)</span>
      </div>
    </div>
  );
}
