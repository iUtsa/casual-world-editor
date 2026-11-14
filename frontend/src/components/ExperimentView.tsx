/* The imagined-world experiment stage.
 *
 * Everything that moves here is a replay of the recorded physics (poses at 60 fps, contact
 * impulses, rope tensions): models from the procedural library or the 3D asset library, ropes and
 * springs drawn between the actual attachment points, trails and velocity arrows from the measured
 * velocities, and visual effects placed where and when the simulator registered impacts, scaled by
 * the measured force. The environment (field, tabletop, ocean with a real surface, space, sky, the
 * Moon or Mars) follows the experiment's setting and gravity.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { clock } from "../clock";
import { distanceAt, poseAt, velocityAt } from "../poses";
import { buildThing, fbm, type BuiltThing } from "../three/things";
import { loadAssetModel, type WheelRig } from "./Spawned";
import type { Asset, ExpObject, ExpRun, FxEvent } from "../types";

export interface ViewOptions { slowmo: boolean; trails: boolean; arrows: boolean; labels: boolean; follow: boolean; forces: boolean }

interface Props { run: ExpRun; assets: Record<string, Asset>; opts: ViewOptions; resetKey: number }

// ---------------------------------------------------------------- helpers
function worldKind(run: ExpRun): "earth" | "moon" | "mars" | "water" | "space" | "air" | "table" | "jupiter" {
  if (run.setting === "water") return "water";
  if (run.setting === "space" || run.gravity === 0) return "space";
  if (run.setting === "table") return "table";
  if (run.setting === "air") return "air";
  const g = run.gravity;
  if (Math.abs(g - 1.62) < 0.2 || g < 1.2) return "moon";
  if (Math.abs(g - 3.71) < 0.3) return "mars";
  if (g > 15) return "jupiter";
  return "earth";
}
function softDot() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d")!;
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.35, "rgba(255,255,255,0.7)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}
const DOT = typeof document !== "undefined" ? softDot() : null;
function groundTexture(kind: string, repeat: number) {
  const c = document.createElement("canvas"); c.width = c.height = 512;
  const g = c.getContext("2d")!;
  const img = g.createImageData(512, 512);
  const pal: Record<string, [string, string]> = {
    earth: ["#3f6b2a", "#6b8f3a"], moon: ["#5b5b5f", "#a3a3a8"], mars: ["#7c2d12", "#c2602b"], jupiter: ["#6b4f3a", "#c08457"],
    table: ["#7a4a24", "#b97a45"], water: ["#b49a6a", "#e2cf9e"], air: ["#3f6b2a", "#8aa04a"], asphalt: ["#2b2d31", "#4b4f56"],
  };
  const [a, b] = pal[kind] ?? pal.earth;
  const ca = new THREE.Color(a), cb = new THREE.Color(b);
  for (let y = 0; y < 512; y++) for (let x = 0; x < 512; x++) {
    let k: number;
    if (kind === "table") k = 0.5 + 0.5 * Math.sin(x * 0.05 + 6 * fbm(x * 0.004, y * 0.03, 0.5, 3));
    else if (kind === "moon") { const cr = fbm(x * 0.02, y * 0.02, 1.3, 5); k = cr + (cr > 0.62 ? -0.35 : 0) + 0.15 * fbm(x * 0.2, y * 0.2, 2, 2); }
    else k = fbm(x * 0.03, y * 0.03, 0.7, 5) * 0.8 + 0.25 * Math.random();
    const col = ca.clone().lerp(cb, Math.max(0, Math.min(1, k)));
    const i = (y * 512 + x) * 4;
    img.data[i] = col.r * 255; img.data[i + 1] = col.g * 255; img.data[i + 2] = col.b * 255; img.data[i + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
  return t;
}

/** Bounds of everything the experiment touches, over the whole recording. */
export function experimentBounds(run: ExpRun) {
  const box = new THREE.Box3();
  const sim = run.sim;
  run.objects.forEach((o) => {
    if (!o.sid) return;
    const i = sim.object_ids.indexOf(o.sid);
    if (i < 0) return;
    const t0 = sim.spawned?.[o.sid]?.t ?? 0;
    const f0 = Math.min(sim.frames.length - 1, Math.round(t0 * sim.fps) + 2);
    const half = Math.max(...o.dims) / 2;
    for (let f = f0; f < sim.frames.length; f += 3) {
      const p = sim.frames[f][i];
      if (Math.abs(p[0]) > 1e4) continue;
      box.expandByPoint(new THREE.Vector3(p[0] - half, p[1] - half, p[2]));
      box.expandByPoint(new THREE.Vector3(p[0] + half, p[1] + half, p[2] + o.dims[2]));
    }
  });
  if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(0, 0, 0.5), new THREE.Vector3(2, 2, 1));
  return box;
}

// ---------------------------------------------------------------- environment
function Sky({ top, bottom, radius }: { top: string; bottom: string; radius: number }) {
  const mat = useMemo(() => new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    uniforms: { top: { value: new THREE.Color(top) }, bottom: { value: new THREE.Color(bottom) } },
    vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix*modelViewMatrix*vec4(position,1.); }",
    fragmentShader: "uniform vec3 top; uniform vec3 bottom; varying vec3 vP; void main(){ float h = clamp(vP.z*1.4+0.25,0.,1.); gl_FragColor = vec4(mix(bottom, top, pow(h,0.7)),1.); }",
  }), [top, bottom]);
  return <mesh material={mat}><sphereGeometry args={[radius, 32, 16]} /></mesh>;
}
function Stars({ radius, count = 3000 }: { radius: number; count?: number }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const p = new Float32Array(count * 3), c = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const u = Math.random() * 2 - 1, th = Math.random() * Math.PI * 2, s = Math.sqrt(1 - u * u);
      p.set([radius * s * Math.cos(th), radius * s * Math.sin(th), radius * u], i * 3);
      const k = 0.6 + Math.random() * 0.4, warm = Math.random();
      c.set([k, k * (0.9 + 0.1 * warm), k * (0.85 + 0.15 * (1 - warm))], i * 3);
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    return g;
  }, [radius, count]);
  return <points geometry={geo}><pointsMaterial size={radius * 0.004} vertexColors sizeAttenuation transparent depthWrite={false} map={DOT ?? undefined} /></points>;
}

/** Ocean: sandy floor with moving caustics, a real surface at the water level, blue fog, drifting particles. */
function Ocean({ size, surfaceZ, floorZ }: { size: number; surfaceZ: number; floorZ: number }) {
  const floorMat = useMemo(() => new THREE.ShaderMaterial({
    uniforms: { uT: { value: 0 }, uFog: { value: new THREE.Color("#0b4f6c") }, uFogD: { value: 0.9 / Math.max(12, size * 0.75) }, uS: { value: 1 / Math.max(1, size * 0.03) } },
    vertexShader: "varying vec3 vW; void main(){ vW = (modelMatrix*vec4(position,1.)).xyz; gl_Position = projectionMatrix*viewMatrix*vec4(vW,1.); }",
    fragmentShader: `uniform float uT; uniform vec3 uFog; uniform float uFogD; uniform float uS; varying vec3 vW;
      float h(vec2 p){ return fract(sin(dot(p, vec2(127.1,311.7)))*43758.5453); }
      float n(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.-2.*f); return mix(mix(h(i),h(i+vec2(1,0)),f.x), mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x), f.y); }
      float fbm(vec2 p){ float s=0., a=.5; for(int i=0;i<5;i++){ s+=a*n(p); p*=2.03; a*=.5; } return s; }
      float caustic(vec2 p){ p*=uS*3.; float v=0.; vec2 q=p; for(int i=0;i<3;i++){ q+=vec2(sin(q.y*1.3+uT*.9), cos(q.x*1.1-uT*.7)); v+=abs(sin(q.x)+sin(q.y)); } return pow(1.-clamp(v/6.,0.,1.),5.); }
      void main(){
        vec2 p = vW.xy*uS;
        float rip = 0.5+0.5*sin(p.x*9.+fbm(p*2.)*6.);
        vec3 sand = mix(vec3(0.55,0.47,0.33), vec3(0.78,0.70,0.52), fbm(p*1.5)*0.8+rip*0.2);
        sand *= 0.85 + 0.15*fbm(p*14.);
        vec3 col = sand*0.55 + vec3(0.55,0.8,0.95)*caustic(vW.xy)*0.7;
        float d = length(cameraPosition - vW);
        float f = 1.-exp(-pow(d*uFogD, 1.6));
        gl_FragColor = vec4(mix(col, uFog, clamp(f,0.,1.)), 1.);
      }`,
  }), [size]);
  const surf = useMemo(() => new THREE.MeshPhysicalMaterial({ color: "#3fa3c9", transmission: 0.6, roughness: 0.08, transparent: true, opacity: 0.55, side: THREE.DoubleSide, ior: 1.33, thickness: 1 }), []);
  const surfMesh = useRef<THREE.Mesh>(null);
  const motes = useMemo(() => {
    const g = new THREE.BufferGeometry(); const n = 1500; const p = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) p.set([(Math.random() - 0.5) * size, (Math.random() - 0.5) * size, floorZ + Math.random() * (surfaceZ - floorZ)], i * 3);
    g.setAttribute("position", new THREE.BufferAttribute(p, 3)); return g;
  }, [size, surfaceZ, floorZ]);
  const geo = useMemo(() => new THREE.PlaneGeometry(size * 3, size * 3, 80, 80), [size]);
  useFrame((_, dt) => {
    floorMat.uniforms.uT.value += dt;
    const p = geo.attributes.position, t = floorMat.uniforms.uT.value;
    for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i); p.setZ(i, Math.sin(x * 0.15 + t) * 0.25 + Math.cos(y * 0.12 + t * 0.8) * 0.2); }
    p.needsUpdate = true; geo.computeVertexNormals();
  });
  return (
    <>
      <mesh position={[0, 0, floorZ - 0.01]} material={floorMat}><planeGeometry args={[size * 6, size * 6]} /></mesh>
      <mesh ref={surfMesh} geometry={geo} material={surf} position={[0, 0, surfaceZ]} />
      <points geometry={motes}><pointsMaterial size={Math.max(0.02, size * 0.002)} color="#cfe9f5" transparent opacity={0.5} depthWrite={false} map={DOT ?? undefined} /></points>
    </>
  );
}

function Environment({ run, bounds }: { run: ExpRun; bounds: THREE.Box3 }) {
  const { scene, gl } = useThree();
  const kind = worldKind(run);
  const size = Math.max(4, bounds.getSize(new THREE.Vector3()).length());
  const hasVehicle = run.objects.some((o) => o.archetype === "vehicle" && o.dims[0] > 1);
  useEffect(() => {
    const pm = new THREE.PMREMGenerator(gl);
    const env = pm.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environment = env;
    scene.environmentIntensity = kind === "space" || kind === "moon" ? 0.35 : kind === "water" ? 0.5 : 0.8;
    const fogCol = { water: "#0b4f6c", earth: "#bcd7ec", mars: "#d9a066", air: "#c7def0", table: "#2a2522", moon: "#000000", space: "#000000", jupiter: "#c69a6d" }[kind];
    scene.fog = kind === "water" ? new THREE.FogExp2(fogCol, 0.9 / Math.max(12, size * 2.2))
      : kind === "moon" || kind === "space" ? null : new THREE.Fog(fogCol, size * 3, size * 18);
    scene.background = new THREE.Color(kind === "water" ? "#0b4f6c" : kind === "table" ? "#1f1b18" : "#000000");
    return () => { pm.dispose(); env.dispose(); scene.fog = null; };
  }, [kind, size, gl, scene]);
  const tex = useMemo(() => (kind === "space" ? null : groundTexture(kind === "earth" && hasVehicle ? "asphalt" : kind === "air" ? "air" : kind, kind === "table" ? 2 : Math.max(4, size / 1.5))), [kind, size, hasVehicle]);
  const T = run.scene_table;
  const sun = useMemo(() => new THREE.Vector3(-0.5, -0.8, 1.2).normalize(), []);
  const c = bounds.getCenter(new THREE.Vector3());
  const shadowR = size * 0.9;
  const lightPos = c.clone().add(sun.clone().multiplyScalar(size * 2.5));
  const lightTarget = useMemo(() => new THREE.Object3D(), []);
  lightTarget.position.copy(c);
  return (
    <>
      <primitive object={lightTarget} />
      <directionalLight position={lightPos} target={lightTarget} intensity={kind === "water" ? 1.4 : kind === "moon" || kind === "space" ? 3.2 : 2.6}
        color={kind === "mars" ? "#ffd9b0" : kind === "water" ? "#bfe9ff" : "#fff6e5"} castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-shadowR} shadow-camera-right={shadowR} shadow-camera-top={shadowR}
        shadow-camera-bottom={-shadowR} shadow-camera-near={0.1} shadow-camera-far={size * 8} shadow-bias={-0.0004} shadow-normalBias={0.02} />
      <hemisphereLight args={[kind === "water" ? "#7dd3fc" : kind === "mars" ? "#f3c79a" : "#cfe6ff", kind === "table" ? "#3b2a1d" : "#4a3f2f", kind === "moon" || kind === "space" ? 0.08 : 0.55]} />
      {kind === "earth" && <Sky top="#3d7fd6" bottom="#d8e9f6" radius={size * 20} />}
      {kind === "air" && <Sky top="#2f6fd0" bottom="#e5f1fb" radius={size * 20} />}
      {kind === "mars" && <Sky top="#b9774a" bottom="#e8bf8f" radius={size * 20} />}
      {kind === "jupiter" && <Sky top="#7a5638" bottom="#d9b48c" radius={size * 20} />}
      {(kind === "moon" || kind === "space") && <Stars radius={size * 18} />}
      {kind === "moon" && <mesh position={[size * 6, size * 9, size * 5]}><sphereGeometry args={[size * 0.9, 48, 32]} /><meshStandardMaterial color="#3b82f6" emissive="#1e3a8a" emissiveIntensity={0.6} /></mesh>}
      {kind === "space" && <mesh position={[size * 8, size * 10, -size * 3]}><sphereGeometry args={[size * 2.2, 64, 32]} /><meshStandardMaterial map={groundTexture("mars", 1)} /></mesh>}
      {tex && kind !== "water" && kind !== "table" && (
        <mesh position={[c.x, c.y, kind === "air" ? 0 : 0]} receiveShadow>
          <planeGeometry args={[size * 40, size * 40]} />
          <meshStandardMaterial map={tex} roughness={0.95} />
        </mesh>
      )}
      {kind === "table" && tex && (
        <>
          <mesh position={[(T.x_min + T.x_max) / 2, (T.y_min + T.y_max) / 2, -0.02]} receiveShadow castShadow>
            <boxGeometry args={[T.x_max - T.x_min, T.y_max - T.y_min, 0.04]} />
            <meshPhysicalMaterial map={tex} roughness={0.45} clearcoat={0.6} clearcoatRoughness={0.3} />
          </mesh>
          <mesh position={[0, 0, -0.76]} receiveShadow><planeGeometry args={[40, 40]} /><meshStandardMaterial color="#3a3029" roughness={0.9} /></mesh>
          <pointLight position={[c.x + 0.6, c.y - 0.4, 1.6]} intensity={4} distance={6} color="#ffe4c4" />
        </>
      )}
      {kind === "water" && <Ocean size={Math.max(size * 3, 40)} surfaceZ={run.env.surface_z ?? bounds.max.z + 2} floorZ={0} />}
      {kind === "air" && <Clouds center={c} size={size} />}
    </>
  );
}
function Clouds({ center, size }: { center: THREE.Vector3; size: number }) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry(); const n = 600; const p = new Float32Array(n * 3);
    for (let k = 0; k < 14; k++) {
      const cx = center.x + (Math.random() - 0.5) * size * 8, cy = center.y + (Math.random() - 0.5) * size * 8, cz = center.z + (Math.random() - 0.3) * size * 2;
      for (let i = 0; i < n / 14; i++) { const j = (k * Math.floor(n / 14) + i) * 3; p.set([cx + (Math.random() - 0.5) * size, cy + (Math.random() - 0.5) * size * 0.5, cz + (Math.random() - 0.5) * size * 0.25], j); }
    }
    g.setAttribute("position", new THREE.BufferAttribute(p, 3)); return g;
  }, [center, size]);
  return <points geometry={geo}><pointsMaterial size={size * 0.35} color="#ffffff" transparent opacity={0.35} depthWrite={false} map={DOT ?? undefined} /></points>;
}

// ---------------------------------------------------------------- objects
export function Thing({ o, run, assets }: { o: ExpObject; run: ExpRun; assets: Record<string, Asset> }) {
  const holder = useRef<THREE.Group>(null);
  const [obj, setObj] = useState<THREE.Object3D | null>(null);
  const built = useRef<BuiltThing | null>(null);
  const rig = useRef<WheelRig | null>(null);
  const sim = run.sim;
  const idx = sim.object_ids.indexOf(o.sid!);
  const spawnT = sim.spawned?.[o.sid!]?.t ?? 0;
  useEffect(() => {
    let alive = true;
    const asset = o.asset_id ? assets[o.asset_id] : null;
    if (asset) {
      loadAssetModel(asset).then(({ obj, rig: r }) => { if (alive) { rig.current = r; setObj(obj); } });
    } else {
      const f0 = Math.min(sim.frames.length - 1, Math.round(spawnT * sim.fps) + 2);
      const b = buildThing({ name: o.name, archetype: o.archetype, dims: o.dims, color: o.color, material: o.material, animate: o.self_propelled,
        postHeight: o.support && idx >= 0 ? sim.frames[f0][idx][2] : undefined }, o.shape);
      built.current = b; setObj(b.root);
    }
    return () => { alive = false; };
  }, [o, assets, sim, spawnT, idx]);
  const pos = useMemo(() => new THREE.Vector3(), []), quat = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    const g = holder.current;
    if (!g || idx < 0) return;
    const t = clock.t;
    g.visible = t >= spawnT - 1e-3;
    poseAt(sim, idx, Math.max(t, spawnT + 1.5 / sim.fps), pos, quat);
    g.position.copy(pos); g.quaternion.copy(quat);
    const [speed, vz] = velocityAt(sim, o.sid!, t);
    const dist = distanceAt(sim, idx, t);
    built.current?.update({ speed, dist, t, vz, propelled: o.self_propelled && speed > 0.15 });
    if (rig.current) for (const p of rig.current.pivots) p.rotation.x = dist / Math.max(0.01, o.dims[2] * 0.25);
  });
  return <group ref={holder} visible={false}>{obj && <primitive object={obj} />}</group>;
}

/** World-space attachment point of object sid (its body's centre site) at time t. */
function siteAt(run: ExpRun, sid: string, t: number, out: THREE.Vector3) {
  const sim = run.sim, i = sim.object_ids.indexOf(sid);
  const o = run.objects.find((x) => x.sid === sid);
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  poseAt(sim, i, t, p, q);
  out.set(0, 0, (o?.dims[2] ?? 0) / 2).applyQuaternion(q).add(p);
  return out;
}

export function Links({ run }: { run: ExpRun }) {
  const links = run.sim.links ?? [];
  return <>{links.map((l) => <LinkView key={l.id} run={run} id={l.id} a={l.a} b={l.b} kind={l.kind} length={l.length} />)}</>;
}
function LinkView({ run, id, a, b, kind, length }: { run: ExpRun; id: string; a: string; b: string; kind: string; length: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const start = useMemo(() => {
    const conn = run.interventions.find((iv) => iv.type === "connect" && (iv.params as Record<string, unknown>).link === id);
    const grp = conn ? (conn.params as Record<string, unknown>).group : null;
    const seg = run.sim.segments.find((s) => (s.intervention.params as Record<string, unknown>)?.group === grp || s.intervention.type === "connect");
    return seg ? seg.t_start : 0;
  }, [run, id]);
  const mat = useMemo(() => new THREE.MeshPhysicalMaterial({ color: kind === "spring" ? "#cbd5e1" : "#d6b98c", metalness: kind === "spring" ? 1 : 0, roughness: kind === "spring" ? 0.25 : 0.85 }), [kind]);
  const pa = useMemo(() => new THREE.Vector3(), []), pb = useMemo(() => new THREE.Vector3(), []);
  const scale = useMemo(() => Math.max(...run.objects.map((o) => Math.max(...o.dims))), [run]);
  useFrame(() => {
    const m = ref.current;
    if (!m) return;
    const t = clock.t;
    m.visible = t >= start;
    if (!m.visible) return;
    siteAt(run, a, t, pa); siteAt(run, b, t, pb);
    const d = pa.distanceTo(pb);
    const f = Math.min(run.sim.frames.length - 1, Math.round(t * run.sim.fps));
    const T = run.sim.tensions?.[id]?.[f]?.[1] ?? 0;
    const pts: THREE.Vector3[] = [];
    const N = kind === "spring" ? 220 : 24;
    const axis = pb.clone().sub(pa).normalize();
    const ref1 = Math.abs(axis.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0);
    const u = new THREE.Vector3().crossVectors(axis, ref1).normalize(), v = new THREE.Vector3().crossVectors(axis, u);
    const R = Math.max(0.006, Math.min(0.05 * scale, length * 0.06));
    for (let i = 0; i <= N; i++) {
      const k = i / N;
      const p = pa.clone().lerp(pb, k);
      if (kind === "spring") {
        const ends = k < 0.06 || k > 0.94;
        if (!ends) { const ang = k * Math.PI * 2 * 14; p.add(u.clone().multiplyScalar(Math.cos(ang) * R)).add(v.clone().multiplyScalar(Math.sin(ang) * R)); }
      } else if (d < length * 0.995) {
        const sag = Math.sqrt(Math.max(0, length * length - d * d)) * 0.5;
        p.z -= sag * 4 * k * (1 - k);
      }
      pts.push(p);
    }
    const curve = new THREE.CatmullRomCurve3(pts);
    const r = kind === "spring" ? R * 0.12 : Math.max(0.002, scale * 0.006);
    m.geometry.dispose();
    m.geometry = new THREE.TubeGeometry(curve, kind === "spring" ? 600 : 48, r, 6);
    const strain = Math.min(1, T / Math.max(1, run.objects.reduce((s, o) => s + o.mass, 0) * 9.81));
    mat.emissive.set(kind === "spring" ? "#000000" : "#ef4444"); mat.emissiveIntensity = kind === "spring" ? 0 : strain * 0.8;
  });
  return <mesh ref={ref} material={mat} castShadow><bufferGeometry /></mesh>;
}

// ---------------------------------------------------------------- trails, arrows, labels
export function Trails({ run }: { run: ExpRun }) {
  const movers = useMemo(() => run.objects.filter((o) => o.sid && !o.fixed && !o.support), [run]);
  const lines = useMemo(() => movers.map((o) => {
    const g = new THREE.BufferGeometry();
    const n = 150;
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9, depthWrite: false }));
    l.frustumCulled = false;
    return { o, l, n, col: new THREE.Color(o.color === "#94a3b8" ? "#38bdf8" : o.color).lerp(new THREE.Color("#ffffff"), 0.35) };
  }), [movers]);
  useFrame(() => {
    const t = clock.t, sim = run.sim;
    const f = Math.round(t * sim.fps);
    for (const { o, l, n, col } of lines) {
      const c = sim.kinematics[o.sid!]?.c as (number[] | null)[] | undefined;
      const pos = l.geometry.attributes.position as THREE.BufferAttribute, cl = l.geometry.attributes.color as THREE.BufferAttribute;
      const t0 = sim.spawned?.[o.sid!]?.t ?? 0;
      let k = 0;
      for (let j = Math.max(Math.round(t0 * sim.fps) + 3, f - n + 1); j <= f && k < n; j++) {
        const p = c?.[Math.min(c.length - 1, j)];
        if (!p) continue;
        pos.setXYZ(k, p[0], p[1], p[2]);
        const a = (k + 1) / n;
        cl.setXYZ(k, col.r * a, col.g * a, col.b * a);
        k++;
      }
      l.geometry.setDrawRange(0, k);
      pos.needsUpdate = true; cl.needsUpdate = true;
    }
  });
  return <>{lines.map(({ o, l }) => <primitive key={o.sid} object={l} />)}</>;
}

export function Arrows({ run, scale }: { run: ExpRun; scale: number }) {
  const movers = useMemo(() => run.objects.filter((o) => o.sid && !o.fixed && !o.support), [run]);
  const arrows = useMemo(() => movers.map(() => new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1, 0x22d3ee, 0.2, 0.1)), [movers]);
  const vmax = useMemo(() => {
    let m = 0.1;
    for (const o of movers) for (const v of run.sim.kinematics[o.sid!]?.v ?? []) if (v) m = Math.max(m, Math.hypot(v[0], v[1], v[2]));
    return m;
  }, [movers, run]);
  useFrame(() => {
    const t = clock.t, sim = run.sim, f = Math.min(sim.frames.length - 1, Math.round(t * sim.fps));
    movers.forEach((o, i) => {
      const a = arrows[i];
      const v = sim.kinematics[o.sid!]?.v[f], c = (sim.kinematics[o.sid!] as { c?: (number[] | null)[] })?.c?.[f];
      const sp = v ? Math.hypot(v[0], v[1], v[2]) : 0;
      a.visible = !!v && !!c && sp > vmax * 0.03 && t >= (sim.spawned?.[o.sid!]?.t ?? 0);
      if (!a.visible) return;
      const len = (sp / vmax) * scale * 0.35 + Math.max(...o.dims) * 0.3;
      a.position.set(c![0], c![1], c![2] + o.dims[2] * 0.5);
      a.setDirection(new THREE.Vector3(v![0], v![1], v![2]).normalize());
      a.setLength(len, len * 0.22, len * 0.12);
    });
  });
  return <>{arrows.map((a, i) => <primitive key={i} object={a} />)}</>;
}

export function Labels({ run, show, host, xf }: { run: ExpRun; show: boolean; host: React.RefObject<HTMLDivElement>; xf?: THREE.Matrix4 | null }) {
  const { camera, size } = useThree();
  const items = useMemo(() => run.objects.filter((o) => o.sid && !o.support), [run]);
  const els = useRef<HTMLDivElement[]>([]);
  useEffect(() => {
    const h = host.current;
    if (!h) return;
    els.current = items.map((o) => {
      const d = document.createElement("div");
      d.className = "x-label";
      d.innerHTML = `<b>${o.name}</b><span></span>`;
      h.appendChild(d);
      return d;
    });
    return () => { els.current.forEach((d) => d.remove()); els.current = []; };
  }, [items, host]);
  const p = useMemo(() => new THREE.Vector3(), []), q = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    const t = clock.t, sim = run.sim;
    items.forEach((o, k) => {
      const d = els.current[k];
      if (!d) return;
      const i = sim.object_ids.indexOf(o.sid!);
      if (!show || i < 0 || t < (sim.spawned?.[o.sid!]?.t ?? 0)) { d.style.display = "none"; return; }
      poseAt(sim, i, t, p, q);
      p.z += o.dims[2] * 1.08 + Math.max(...o.dims) * 0.05;
      if (xf) p.applyMatrix4(xf);
      p.project(camera);
      if (p.z > 1) { d.style.display = "none"; return; }
      d.style.display = "flex";
      d.style.transform = `translate(${((p.x + 1) / 2) * size.width}px, ${((1 - p.y) / 2) * size.height}px) translate(-50%, -100%)`;
      const [sp] = velocityAt(sim, o.sid!, t);
      (d.lastChild as HTMLElement).textContent = o.fixed ? "" : `${sp < 10 ? sp.toFixed(2) : sp.toFixed(1)} m/s`;
    });
  });
  return null;
}

// ---------------------------------------------------------------- impact & splash effects
const MAXP = 4000;
export function Effects({ run, scale, sizeScale = 1, shakeCamera = true }: { run: ExpRun; scale: number; sizeScale?: number; shakeCamera?: boolean }) {
  const fx = run.fx;
  const kind = worldKind(run);
  const weightRef = useMemo(() => Math.max(1, run.objects.filter((o) => !o.fixed).reduce((s, o) => s + o.mass, 0) * Math.max(1, run.gravity)), [run]);
  const pts = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAXP * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(MAXP * 3), 3));
    const m = new THREE.PointsMaterial({ size: 1, vertexColors: true, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, map: DOT ?? undefined, sizeAttenuation: true });
    const p = new THREE.Points(g, m); p.frustumCulled = false;
    return p;
  }, []);
  const dust = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAXP * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(MAXP * 3), 3));
    const m = new THREE.PointsMaterial({ size: 1, vertexColors: true, transparent: true, opacity: 0.55, depthWrite: false, map: DOT ?? undefined });
    const p = new THREE.Points(g, m); p.frustumCulled = false;
    return p;
  }, []);
  const rings = useMemo(() => Array.from({ length: 10 }, () => {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.85, 1, 64), new THREE.MeshBasicMaterial({ color: "#fde68a", transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending }));
    m.visible = false; return m;
  }), []);
  const flash = useRef<THREE.PointLight>(null);
  const { camera } = useThree();
  const shake = useRef(new THREE.Vector3());
  const strength = (e: FxEvent) => e.kind === "hit" ? Math.min(1, Math.log10(1 + (e.force ?? 0) / (weightRef * 0.05)) / 2.2) : Math.min(1, (e.speed ?? 0) / 8);
  useFrame(() => {
    const t = clock.t;
    const P = pts.geometry.attributes.position as THREE.BufferAttribute, Cc = pts.geometry.attributes.color as THREE.BufferAttribute;
    const D = dust.geometry.attributes.position as THREE.BufferAttribute, Dc = dust.geometry.attributes.color as THREE.BufferAttribute;
    let n = 0, nd = 0, ri = 0, fl = 0;
    const g = Math.max(0.5, run.gravity);
    camera.position.sub(shake.current); shake.current.set(0, 0, 0);
    fx.forEach((e, ei) => {
      const age = t - e.t;
      if (age < 0 || age > 2.2) return;
      const s = strength(e);
      const sz = Math.max(0.02, e.scale);
      if (e.kind === "hit") {
        const life = 0.35 + 0.5 * s;
        if (age < life) {
          const count = Math.round(40 + 260 * s);
          for (let k = 0; k < count && n < MAXP; k++, n++) {
            const h1 = Math.sin((ei * 977 + k) * 12.9898) * 43758.5453, r1 = h1 - Math.floor(h1);
            const h2 = Math.sin((ei * 577 + k) * 78.233) * 12345.678, r2 = h2 - Math.floor(h2);
            const h3 = Math.sin((ei * 131 + k) * 39.425) * 9876.54, r3 = h3 - Math.floor(h3);
            const th = r1 * Math.PI * 2, ph = Math.acos(2 * r2 - 1);
            const sp = sz * 0.5 * (1.5 + 5 * s) * (0.3 + r3);
            const vx = Math.sin(ph) * Math.cos(th) * sp, vy = Math.sin(ph) * Math.sin(th) * sp, vz = Math.abs(Math.cos(ph)) * sp * 0.8;
            P.setXYZ(n, e.pos[0] + vx * age, e.pos[1] + vy * age, e.pos[2] + vz * age - 0.5 * g * age * age);
            const a = Math.max(0, 1 - age / life) ** 1.5;
            Cc.setXYZ(n, 1.0 * a, (0.75 + 0.25 * r3) * a, (0.35 + 0.4 * r3) * a);
          }
        }
        if (age < 0.6 && ri < rings.length) {
          const r = rings[ri++]; r.visible = true;
          r.position.set(e.pos[0], e.pos[1], e.pos[2]);
          const R = sz * 0.12 * (0.3 + age * (3 + 6 * s));
          r.scale.set(R, R, R);
          const ia = run.sim.object_ids.indexOf(e.a), ib = e.b ? run.sim.object_ids.indexOf(e.b) : -1;
          if (ia >= 0 && ib >= 0) {
            const f = Math.min(run.sim.frames.length - 1, Math.round(e.t * run.sim.fps));
            const A = run.sim.frames[f][ia], B = run.sim.frames[f][ib];
            const nrm = new THREE.Vector3(B[0] - A[0], B[1] - A[1], B[2] - A[2]).normalize();
            r.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), nrm);
          }
          (r.material as THREE.MeshBasicMaterial).opacity = (1 - age / 0.6) ** 2 * (0.25 + 0.45 * s);
          (r.material as THREE.MeshBasicMaterial).color.set("#fde68a");
        }
        if (age < 0.12) {
          fl = Math.max(fl, (1 - age / 0.12) * (2 + 30 * s));
          if (flash.current) flash.current.position.set(e.pos[0], e.pos[1], e.pos[2] + sz * 0.3);
          if (s > 0.35 && shakeCamera) shake.current.set((Math.random() - 0.5), (Math.random() - 0.5), (Math.random() - 0.5) * 0.5).multiplyScalar(scale * 0.012 * s * (1 - age / 0.12));
        }
      } else if (e.kind === "ground") {
        const life = 1.4;
        if (age < life && kind !== "space") {
          const count = Math.round(25 + 140 * s);
          const dustCol = kind === "moon" ? [0.62, 0.62, 0.64] : kind === "mars" ? [0.72, 0.42, 0.25] : kind === "water" ? [0.85, 0.78, 0.6] : kind === "table" ? [0.6, 0.5, 0.4] : [0.55, 0.48, 0.36];
          for (let k = 0; k < count && nd < MAXP; k++, nd++) {
            const h1 = Math.sin((ei * 311 + k) * 12.9898) * 43758.5453, r1 = h1 - Math.floor(h1);
            const h2 = Math.sin((ei * 719 + k) * 78.233) * 12345.678, r2 = h2 - Math.floor(h2);
            const th = r1 * Math.PI * 2, sp = sz * (0.6 + 2.5 * s) * (0.4 + r2);
            const drag = kind === "moon" ? age : (1 - Math.exp(-age * 3)) / 3;          // no air on the Moon: ballistic dust
            const rise = kind === "moon" ? sp * 0.8 * age - 0.5 * g * age * age : sz * 0.4 * (0.3 + r2) * (1 - Math.exp(-age * 2));
            D.setXYZ(nd, e.pos[0] + Math.cos(th) * sp * drag, e.pos[1] + Math.sin(th) * sp * drag, Math.max(0.005, e.pos[2] + rise));
            const a = Math.max(0, 1 - age / life);
            Dc.setXYZ(nd, dustCol[0] * a, dustCol[1] * a, dustCol[2] * a);
          }
        }
        if (age < 0.8 && ri < rings.length && kind !== "space") {
          const r = rings[ri++]; r.visible = true;
          r.position.set(e.pos[0], e.pos[1], e.pos[2] + 0.003);
          const R = sz * 0.25 * (0.4 + age * 3 * (0.5 + s)); r.scale.set(R, R, R);
          r.quaternion.identity();
          (r.material as THREE.MeshBasicMaterial).opacity = (1 - age / 0.8) * 0.35 * (0.3 + s);
          (r.material as THREE.MeshBasicMaterial).color.set("#e7e5e4");
        }
      } else if (e.kind === "splash") {
        const life = 1.6;
        if (age < life) {
          const count = Math.round(60 + 300 * s);
          for (let k = 0; k < count && n < MAXP; k++, n++) {
            const h1 = Math.sin((ei * 211 + k) * 12.9898) * 43758.5453, r1 = h1 - Math.floor(h1);
            const h2 = Math.sin((ei * 419 + k) * 78.233) * 12345.678, r2 = h2 - Math.floor(h2);
            const th = r1 * Math.PI * 2, sp = sz * (0.5 + 2 * s) * (0.3 + r2);
            const vz = sz * (2 + 6 * s) * (0.4 + r2 * 0.8);
            const z = e.pos[2] + vz * age - 0.5 * 9.81 * age * age;
            if (z < e.pos[2] - 0.05) continue;
            P.setXYZ(n, e.pos[0] + Math.cos(th) * sp * age, e.pos[1] + Math.sin(th) * sp * age, z);
            const a = Math.max(0, 1 - age / life) * 0.7;
            Cc.setXYZ(n, 0.75 * a, 0.9 * a, 1.0 * a);
          }
        }
        for (let w = 0; w < 3 && ri < rings.length; w++) {
          const aw = age - w * 0.25;
          if (aw < 0 || aw > 1.5) continue;
          const r = rings[ri++]; r.visible = true;
          r.position.set(e.pos[0], e.pos[1], e.pos[2] + 0.01);
          const R = sz * 0.3 * (0.6 + aw * 4); r.scale.set(R, R, R); r.quaternion.identity();
          (r.material as THREE.MeshBasicMaterial).opacity = (1 - aw / 1.5) * 0.5;
          (r.material as THREE.MeshBasicMaterial).color.set("#e0f2fe");
        }
      }
    });
    // bubbles behind fast swimmers
    if (kind === "water") {
      for (const o of run.objects) {
        if (!o.sid || o.fixed) continue;
        const c = (run.sim.kinematics[o.sid] as { c?: (number[] | null)[] })?.c;
        if (!c) continue;
        const f = Math.round(t * run.sim.fps);
        for (let k = 0; k < 40 && n < MAXP; k++) {
          const ff = f - k * 2;
          const p = c[Math.max(0, Math.min(c.length - 1, ff))];
          const [sp] = velocityAt(run.sim, o.sid, ff / run.sim.fps);
          if (!p || sp < 0.4) continue;
          const age = k * 2 / run.sim.fps;
          const h = Math.sin((k + 7) * 91.7) * 1000, r = h - Math.floor(h);
          P.setXYZ(n, p[0] + (r - 0.5) * o.dims[1] * 0.5, p[1] + (r - 0.3) * o.dims[1] * 0.5, p[2] + age * 0.6 * Math.max(...o.dims) * 0.3);
          const a = Math.max(0, 1 - k / 40) * 0.5;
          Cc.setXYZ(n, 0.8 * a, 0.95 * a, 1.0 * a);
          n++;
        }
      }
    }
    for (let k = ri; k < rings.length; k++) rings[k].visible = false;
    P.needsUpdate = true; Cc.needsUpdate = true; D.needsUpdate = true; Dc.needsUpdate = true;
    pts.geometry.setDrawRange(0, n); dust.geometry.setDrawRange(0, nd);
    (pts.material as THREE.PointsMaterial).size = Math.max(0.01, scale * 0.012) * sizeScale;
    (dust.material as THREE.PointsMaterial).size = Math.max(0.02, scale * 0.05) * sizeScale;
    if (flash.current) flash.current.intensity = fl * Math.max(1, scale * scale * 0.4);
    if (!shakeCamera) shake.current.set(0, 0, 0);
    camera.position.add(shake.current);
  });
  return (
    <>
      <primitive object={pts} />
      <primitive object={dust} />
      {rings.map((r, i) => <primitive key={i} object={r} />)}
      <pointLight ref={flash} color="#ffd38a" intensity={0} distance={scale * 3} decay={2} />
    </>
  );
}

// ---------------------------------------------------------------- camera director
function Director({ run, bounds, opts, resetKey }: { run: ExpRun; bounds: THREE.Box3; opts: ViewOptions; resetKey: number }) {
  const { camera, gl } = useThree();
  const ctl = useRef<OrbitControls | null>(null);
  const follow = useRef(opts.follow);
  const autoFollow = run.experiment.actions.some((a) => ["move", "race", "chase"].includes(a.do));
  follow.current = opts.follow || autoFollow;
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.up.set(0, 0, 1);
    let size = bounds.getSize(new THREE.Vector3());
    let c = bounds.getCenter(new THREE.Vector3());
    const movers = run.objects.filter((o) => o.sid && o.self_propelled && !o.fixed);
    const travelling = run.experiment.actions.some((a) => ["move", "race", "chase"].includes(a.do));
    if (travelling && movers.length) {
      // frame the runner/swimmer/flier at its own scale and follow it, not the whole 100 m track
      const big = Math.max(...movers.map((o) => Math.max(...o.dims)));
      const sim = run.sim, f0 = Math.round(sim.segments[sim.segments.length - 1].t_start * sim.fps) + 1;
      const pts = movers.map((o) => sim.frames[Math.min(sim.frames.length - 1, f0)][sim.object_ids.indexOf(o.sid!)]);
      c = new THREE.Vector3(pts.reduce((s_, p) => s_ + p[0], 0) / pts.length, pts.reduce((s_, p) => s_ + p[1], 0) / pts.length, pts.reduce((s_, p) => s_ + p[2], 0) / pts.length + big * 0.4);
      size = new THREE.Vector3(big * 3 + (movers.length > 1 ? 4 * big : 0), big * 2, big * 1.5);
      follow.current = true;
    }
    const ext = Math.max(size.x, size.y * 0.8, size.z * 1.4, 0.25);
    cam.fov = 40;
    const dist = (ext * 0.6) / Math.tan(THREE.MathUtils.degToRad(cam.fov / 2));
    const dir = new THREE.Vector3(-0.32, -1, run.setting === "water" ? 0.18 : 0.42).normalize();
    cam.position.copy(c).add(dir.multiplyScalar(dist));
    if (run.setting !== "water" && run.setting !== "space" && cam.position.z < 0.05) cam.position.z = 0.05;
    cam.near = Math.max(0.002, dist / 2000); cam.far = dist * 400;
    cam.updateProjectionMatrix();
    const oc = new OrbitControls(cam, gl.domElement);
    oc.target.copy(c); oc.enableDamping = true; oc.dampingFactor = 0.08; oc.maxDistance = dist * 20; oc.minDistance = ext * 0.05;
    oc.update();
    ctl.current = oc;
    return () => { oc.dispose(); ctl.current = null; };
  }, [run, bounds, camera, gl, resetKey]);
  const tgt = useMemo(() => new THREE.Vector3(), []);
  useFrame((_, delta) => {
    const oc = ctl.current;
    if (!oc) return;
    if (follow.current) {
      // centroid of the moving things, weighted toward the fastest
      const sim = run.sim, t = clock.t;
      let w = 0; tgt.set(0, 0, 0);
      for (const o of run.objects) {
        if (!o.sid || o.fixed || o.support) continue;
        const i = sim.object_ids.indexOf(o.sid);
        const f = Math.min(sim.frames.length - 1, Math.round(t * sim.fps));
        const p = sim.frames[f][i];
        const [sp] = velocityAt(sim, o.sid, t);
        const k = 0.3 + sp;
        tgt.add(new THREE.Vector3(p[0], p[1], p[2] + o.dims[2] / 2).multiplyScalar(k)); w += k;
      }
      if (w > 0) {
        tgt.divideScalar(w);
        const d = tgt.clone().sub(oc.target).multiplyScalar(1 - Math.exp(-Math.min(0.25, delta) * 9));
        oc.target.add(d); camera.position.add(d);
      }
    }
    oc.update();
  });
  return null;
}

/** Slow motion around big impacts (the clock rate dips, then recovers). */
export function SlowMo({ run, on }: { run: ExpRun; on: boolean }) {
  const big = useMemo(() => {
    const hits = run.fx.filter((e) => e.kind === "hit" || e.kind === "splash");
    if (!hits.length) return [] as number[];
    const top = Math.max(...hits.map((e) => e.force ?? (e.speed ?? 0) * 100));
    return hits.filter((e) => (e.force ?? (e.speed ?? 0) * 100) > top * 0.25).map((e) => e.t);
  }, [run]);
  useFrame(() => {
    if (!on || !clock.playing) { if (clock.rate < 1 && on === false) clock.rate = 1; return; }
    const t = clock.t;
    const near = big.some((tb) => t > tb - 0.12 && t < tb + 0.3);
    const want = near ? 0.18 : 1;
    clock.rate += (want - clock.rate) * 0.25;
  });
  return null;
}

// ---------------------------------------------------------------- assembly
export default function ExperimentView({ run, assets, opts, resetKey }: Props) {
  const bounds = useMemo(() => experimentBounds(run), [run]);
  const scale = useMemo(() => Math.max(0.2, bounds.getSize(new THREE.Vector3()).length()), [bounds]);
  const host = useRef<HTMLDivElement>(null);
  const kind = worldKind(run);
  return (
    <div className="xview">
      <Canvas shadows="soft" dpr={[1, 2]} gl={{ antialias: true, preserveDrawingBuffer: true }}
        onCreated={({ gl }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = kind === "water" ? 1.15 : 1.0; }}
        camera={{ position: [0, -3, 2], fov: 40 }}>
        <Environment run={run} bounds={bounds} />
        {run.objects.filter((o) => o.sid).map((o) => <Thing key={`${run.id}-${o.sid}`} o={o} run={run} assets={assets} />)}
        <Links run={run} />
        {opts.trails && <Trails run={run} />}
        {opts.arrows && <Arrows run={run} scale={scale} />}
        <Effects run={run} scale={scale} />
        <Labels run={run} show={opts.labels} host={host} />
        <Director run={run} bounds={bounds} opts={opts} resetKey={resetKey} />
        <SlowMo run={run} on={opts.slowmo} />
      </Canvas>
      <div className="xlabels" ref={host} />
      <Hud run={run} />
    </div>
  );
}

function Hud({ run }: { run: ExpRun }) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const kind = worldKind(run);
  const where = { earth: "Earth", moon: "the Moon", mars: "Mars", jupiter: "Jupiter", water: "underwater", space: "deep space", air: "in the air", table: "tabletop" }[kind];
  return (
    <div className="xhud">
      <span className="pill dark">t = {clock.t.toFixed(2)} s</span>
      {clock.rate < 0.9 && clock.playing && <span className="pill dark warmth">slow motion ×{clock.rate.toFixed(2)}</span>}
      <span className="pill dark">{where} · g = {run.gravity.toFixed(2)} m/s²{run.env.density ? ` · ${run.env.density > 100 ? "water" : "air"} ${run.env.density.toFixed(run.env.density > 100 ? 0 : 2)} kg/m³` : " · vacuum"}</span>
    </div>
  );
}
