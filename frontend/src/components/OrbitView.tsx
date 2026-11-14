/* Gravity in space, replayed from the N-body integration: real relative positions (scaled to fit), bodies
 * painted like the real ones, orbit trails, and — when bodies meet — a flash, a shock front, molten debris
 * and the merged body glowing red-hot, at the time and place the integrator registered contact. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clock } from "../clock";
import { buildThing } from "../three/things";
import type { GravityRun } from "../types";

function frameAt(run: GravityRun) {
  const f = (clock.t / run.duration) * (run.t.length - 1);
  return Math.max(0, Math.min(run.t.length - 1, f));
}
function posAt(run: GravityRun, i: number, f: number, out: THREE.Vector3) {
  const a = Math.floor(f), b = Math.min(run.pos.length - 1, a + 1), k = f - a;
  const A = run.pos[a][i], B = run.pos[b][i];
  return out.set(A[0] + (B[0] - A[0]) * k, A[1] + (B[1] - A[1]) * k, A[2] + (B[2] - A[2]) * k);
}
function simTimeAt(run: GravityRun, f: number) { const a = Math.floor(f); return run.t[Math.min(run.t.length - 1, a)] ?? 0; }

function Stars() {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry(); const n = 4000; const p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) { const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u); p.set([400 * s * Math.cos(th), 400 * s * Math.sin(th), 400 * u], i * 3); }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3)); return g;
  }, []);
  return <points geometry={geo}><pointsMaterial size={1.2} color="#e2e8f0" sizeAttenuation={false} /></points>;
}

export default function OrbitView({ run, resetKey }: { run: GravityRun; resetKey: number }) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const host = useRef<HTMLDivElement>(null);
  // world scale: fit the whole motion into ~12 units
  const scale = useMemo(() => {
    let ext = 0;
    for (const fr of run.pos) for (const p of fr) ext = Math.max(ext, Math.hypot(p[0], p[1], p[2]));
    run.bodies.forEach((b) => (ext = Math.max(ext, b.radius * 2)));
    return 6 / Math.max(ext, 1);
  }, [run]);
  // bodies too small to see at true scale are drawn bigger, and the HUD says so
  const exaggerate = useMemo(() => {
    const minR = Math.min(...run.bodies.map((b) => b.radius * scale));
    return run.mode === "orbit" && minR < 0.06 ? 0.06 / minR : 1;
  }, [run, scale]);
  const f = frameAt(run);
  return (
    <div className="xview">
      <Canvas dpr={[1, 2]} onCreated={({ gl, scene }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; scene.background = new THREE.Color("#02030a"); }}>
        <Stars />
        <ambientLight intensity={0.08} />
        <Scene run={run} scale={scale} exaggerate={exaggerate} host={host} resetKey={resetKey} />
      </Canvas>
      <div className="xlabels" ref={host} />
      <div className="xhud">
        <span className="pill dark">simulated time: {dur(simTimeAt(run, f))}</span>
        <span className="pill dark">1 s of playback ≈ {dur(run.sim_seconds / run.duration)}</span>
        {exaggerate > 1.5 && <span className="pill dark warmth">bodies drawn {Math.round(exaggerate)}× larger than true scale so you can see them</span>}
      </div>
    </div>
  );
}

function Scene({ run, scale, exaggerate, host, resetKey }: { run: GravityRun; scale: number; exaggerate: number; host: React.RefObject<HTMLDivElement>; resetKey: number }) {
  const { camera, gl, size } = useThree();
  const meshes = useMemo(() => run.bodies.map((b) => {
    const n = b.name.replace(/ \d+$/, "");
    const r = b.radius * scale * (n === "sun" || n === "star" ? Math.min(exaggerate, 3) : exaggerate);
    let obj: THREE.Object3D;
    if (/black hole/.test(n)) {
      const g = new THREE.Group();
      g.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshBasicMaterial({ color: "#000" })));
      const disk = new THREE.Mesh(new THREE.RingGeometry(r * 1.6, r * 4, 96), new THREE.MeshBasicMaterial({ color: "#fb923c", side: THREE.DoubleSide, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending }));
      disk.rotation.x = 1.2; g.add(disk);
      obj = g;
    } else if (/sun|star|dwarf/.test(n)) {
      obj = new THREE.Mesh(new THREE.SphereGeometry(r, 64, 48), new THREE.MeshBasicMaterial({ color: b.color }));
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex(), color: b.color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
      halo.scale.set(r * 6, r * 6, 1); obj.add(halo);
    } else {
      const built = buildThing({ name: n, archetype: /asteroid|meteor|comet|phobos|ceres/.test(n) ? "rock" : "ball", dims: [2 * r, 2 * r, 2 * r], color: b.color, material: "stone", animate: false }, "sphere");
      built.root.position.z = -r;          // builders put the origin at the bottom: centre it
      built.root.traverse((o) => { const mm = o as THREE.Mesh; if (mm.isMesh) mm.material = (mm.material as THREE.Material).clone(); });
      obj = new THREE.Group(); obj.add(built.root);
      if (/earth|venus|titan/.test(n)) {
        const atm = new THREE.Mesh(new THREE.SphereGeometry(r * 1.04, 48, 32), new THREE.MeshBasicMaterial({ color: n === "earth" ? "#7dd3fc" : "#fde68a", transparent: true, opacity: 0.18, blending: THREE.AdditiveBlending, side: THREE.BackSide }));
        obj.add(atm);
      }
      if (/saturn/.test(n)) { const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.3, r * 2.2, 96), new THREE.MeshStandardMaterial({ color: "#e9d8a6", side: THREE.DoubleSide, transparent: true, opacity: 0.7 })); ring.rotation.x = 1.1; obj.add(ring); }
    }
    return { obj, r, name: b.name };
  }), [run, scale, exaggerate]);
  const sunIdx = run.bodies.findIndex((b) => /sun|star/.test(b.name));
  const light = useMemo(() => new THREE.PointLight("#fff6e5", 4, 0, 0), []);
  const dirLight = useMemo(() => { const l = new THREE.DirectionalLight("#fff6e5", 2.2); l.position.set(-5, -3, 4); return l; }, []);
  // trails
  const trails = useMemo(() => run.bodies.map((b) => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(run.pos.length * 3), 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: b.color === "#000000" ? "#fb923c" : b.color, transparent: true, opacity: 0.55 }));
    l.frustumCulled = false; return l;
  }), [run]);
  // impact effects
  const debris = useMemo(() => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.04, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    p.frustumCulled = false; return p;
  }, []);
  const shock = useMemo(() => new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), new THREE.MeshBasicMaterial({ color: "#fde68a", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })), []);
  const flash = useMemo(() => new THREE.PointLight("#ffd6a0", 0, 0, 0), []);
  const labels = useRef<HTMLDivElement[]>([]);
  useEffect(() => {
    const h = host.current; if (!h) return;
    labels.current = run.bodies.map((b) => { const d = document.createElement("div"); d.className = "x-label"; d.innerHTML = `<b>${proper(b.name)}</b><span></span>`; h.appendChild(d); return d; });
    return () => labels.current.forEach((d) => d.remove());
  }, [run, host]);
  const ctl = useRef<OrbitControls | null>(null);
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.up.set(0, 0, 1); cam.fov = 40; cam.near = 0.001; cam.far = 2000;
    cam.position.set(0, -11, run.mode === "orbit" ? 9 : 4); cam.updateProjectionMatrix();
    const oc = new OrbitControls(cam, gl.domElement); oc.enableDamping = true; oc.target.set(0, 0, 0); oc.update();
    ctl.current = oc;
    return () => oc.dispose();
  }, [run, camera, gl, resetKey]);
  const v = useMemo(() => new THREE.Vector3(), []), prev = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    ctl.current?.update();
    const f = frameAt(run);
    const tSim = simTimeAt(run, f);
    meshes.forEach((m, i) => {
      posAt(run, i, f, v).multiplyScalar(scale);
      m.obj.position.copy(v);
      m.obj.rotation.z = tSim * 1e-4 * (i + 1);
      // after a merger the absorbed body is gone
      const absorbed = run.events.find((e) => e.a === m.name && e.t <= tSim);
      m.obj.visible = !absorbed;
      const pos = trails[i].geometry.attributes.position as THREE.BufferAttribute;
      const n = Math.floor(f) + 1;
      for (let k = 0; k < n; k++) { const P = run.pos[k][i]; pos.setXYZ(k, P[0] * scale, P[1] * scale, P[2] * scale); }
      trails[i].geometry.setDrawRange(Math.max(0, n - (run.mode === "orbit" ? 100000 : 200)), n); pos.needsUpdate = true;
      trails[i].visible = !absorbed;
      const d = labels.current[i];
      if (d) {
        const p = v.clone().add(new THREE.Vector3(0, 0, m.r * 1.3)).project(camera);
        d.style.display = absorbed ? "none" : "flex";
        d.style.transform = `translate(${((p.x + 1) / 2) * size.width}px, ${((1 - p.y) / 2) * size.height}px) translate(-50%, -100%)`;
        const a = Math.floor(f), b = Math.min(run.pos.length - 1, a + 1);
        const dt = (run.t[b] - run.t[a]) || 1;
        prev.set(...(run.pos[a][i] as [number, number, number]));
        const speed = new THREE.Vector3(...(run.pos[b][i] as [number, number, number])).sub(prev).length() / dt;
        (d.lastChild as HTMLElement).textContent = `${(speed / 1000).toFixed(speed > 1e4 ? 0 : 2)} km/s`;
      }
    });
    if (sunIdx >= 0) light.position.copy(meshes[sunIdx].obj.position);
    // impact
    const e = run.events[0];
    const D = debris.geometry.attributes.position as THREE.BufferAttribute, Cc = debris.geometry.attributes.color as THREE.BufferAttribute;
    let nd = 0;
    if (e) {
      const age = (tSim - e.t) / Math.max(1, run.sim_seconds) * run.duration;      // seconds of playback since impact
      const c = new THREE.Vector3(...(e.pos as [number, number, number])).multiplyScalar(scale);
      const R0 = e.small_radius * scale;
      if (age >= 0 && age < 6) {
        shock.visible = true; shock.position.copy(c);
        const R = R0 * (1 + age * 6); shock.scale.set(R, R, R);
        (shock.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.7 * (1 - age / 2.5));
        flash.position.copy(c); flash.intensity = Math.max(0, 60 * (1 - age / 1.2));
        for (let k = 0; k < 3000; k++, nd++) {
          const h1 = Math.sin(k * 12.9898) * 43758.5453, r1 = h1 - Math.floor(h1);
          const h2 = Math.sin(k * 78.233) * 12345.678, r2 = h2 - Math.floor(h2);
          const h3 = Math.sin(k * 39.425) * 9876.54, r3 = h3 - Math.floor(h3);
          const th = r1 * Math.PI * 2, ph = Math.acos(2 * r2 - 1), sp = R0 * (0.6 + 3 * r3);
          D.setXYZ(k, c.x + Math.sin(ph) * Math.cos(th) * sp * age, c.y + Math.sin(ph) * Math.sin(th) * sp * age, c.z + Math.cos(ph) * sp * age);
          const a = Math.max(0, 1 - age / 5);
          Cc.setXYZ(k, a, (0.45 + 0.4 * r3) * a, 0.15 * a);
        }
        // the survivor glows red-hot, then cools
        const big = meshes.find((m) => m.name === e.b);
        big?.obj.traverse((o) => {
          const mm = (o as THREE.Mesh).material as THREE.MeshPhysicalMaterial | undefined;
          if (mm && "emissive" in mm) { mm.emissive = new THREE.Color("#ff4d00"); mm.emissiveIntensity = Math.max(0, 1.6 * (1 - age / 6)); }
        });
      } else { shock.visible = false; flash.intensity = 0; }
    }
    debris.geometry.setDrawRange(0, nd); D.needsUpdate = true; Cc.needsUpdate = true;
    (debris.material as THREE.PointsMaterial).size = Math.max(0.01, (e?.small_radius ?? 1) * scale * 0.08);
  });
  return (
    <>
      {meshes.map((m, i) => <primitive key={i} object={m.obj} />)}
      {trails.map((t, i) => <primitive key={`t${i}`} object={t} />)}
      {sunIdx >= 0 ? <primitive object={light} /> : <primitive object={dirLight} />}
      <primitive object={debris} />
      <primitive object={shock} />
      <primitive object={flash} />
    </>
  );
}

let HALO: THREE.Texture | null = null;
function haloTex() {
  if (HALO) return HALO;
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d")!; const r = g.createRadialGradient(64, 64, 10, 64, 64, 64);
  r.addColorStop(0, "rgba(255,240,200,0.9)"); r.addColorStop(0.3, "rgba(255,200,120,0.35)"); r.addColorStop(1, "rgba(255,160,60,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  HALO = new THREE.CanvasTexture(c); return HALO;
}
const PROPER: Record<string, string> = { sun: "Sun", moon: "Moon", earth: "Earth", iss: "ISS" };
function proper(n: string) { const b = n.replace(/ \d+$/, ""); return (PROPER[b] ?? b.replace(/\b\w/g, (c) => c.toUpperCase())) + (n !== b ? n.slice(b.length) : ""); }
function dur(s: number) {
  if (s < 120) return `${s.toFixed(1)} s`;
  if (s < 7200) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  if (s < 6.3e7) return `${(s / 86400).toFixed(1)} days`;
  return `${(s / 3.156e7).toFixed(2)} years`;
}
