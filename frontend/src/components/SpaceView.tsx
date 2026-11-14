/* Space, replayed from the N-body integration (REBOUND IAS15, real NASA JPL starting data where it exists).
 * Planets are procedural shaders lit by their star; stars boil; rings, comet tails pointing away from the Sun,
 * black-hole accretion disks, colliding galaxies as thousands of stars, asteroid/Kuiper belts on their own Kepler
 * orbits, a Milky Way sky. Distances can be shown true-to-scale or radially compressed so everything fits; sizes are
 * enlarged so bodies can be seen - the HUD always says which. Click a name to ride along with that body. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clock } from "../clock";
import type { GravityRun, SpaceBody } from "../types";

const DAY = 86400, YEAR = 365.25 * DAY;

// ------------------------------------------------------------------ timing / interpolation
function frameAt(run: GravityRun) {
  const f = (clock.t / run.duration) * (run.t.length - 1);
  return Math.max(0, Math.min(run.t.length - 1, f));
}
function simTimeAt(run: GravityRun, f: number) {
  const a = Math.floor(f), b = Math.min(run.t.length - 1, a + 1), k = f - a;
  return (run.t[a] ?? 0) + ((run.t[b] ?? 0) - (run.t[a] ?? 0)) * k;
}
/** Catmull-Rom between frames: smooth orbits even when a fast inner planet moves far between frames. */
function rawPos(run: GravityRun, i: number, f: number, out: THREE.Vector3) {
  const n = run.pos.length - 1;
  const a = Math.floor(f), k = f - a;
  const p0 = run.pos[Math.max(0, a - 1)][i], p1 = run.pos[a][i], p2 = run.pos[Math.min(n, a + 1)][i], p3 = run.pos[Math.min(n, a + 2)][i];
  const cr = (x0: number, x1: number, x2: number, x3: number) =>
    0.5 * (2 * x1 + (-x0 + x2) * k + (2 * x0 - 5 * x1 + 4 * x2 - x3) * k * k + (-x0 + 3 * x1 - 3 * x2 + x3) * k * k * k);
  return out.set(cr(p0[0], p1[0], p2[0], p3[0]), cr(p0[1], p1[1], p2[1], p3[1]), cr(p0[2], p1[2], p2[2], p3[2]));
}

// ------------------------------------------------------------------ helpers
function f16decode(b64: string): Float32Array {
  const bin = atob(b64); const u = new Uint16Array(bin.length / 2);
  for (let i = 0; i < u.length; i++) u[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * 5.960464477539063e-8 * m : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}
function u8decode(b64: string): Uint8Array { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
function dur(s: number) {
  s = Math.abs(s);
  if (s < 120) return `${s.toFixed(1)} s`;
  if (s < 7200) return `${(s / 60).toFixed(1)} min`;
  if (s < 172800) return `${(s / 3600).toFixed(1)} h`;
  if (s < 2 * YEAR) return `${(s / DAY).toFixed(1)} days`;
  if (s < 2e6 * YEAR) return `${(s / YEAR).toFixed(s < 200 * YEAR ? 1 : 0)} years`;
  return `${(s / YEAR / 1e9).toFixed(2)} billion years`;
}
function addSeconds(iso: string, s: number) {
  const d = new Date(iso + "T00:00:00Z"); d.setTime(d.getTime() + s * 1000);
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
}
function kepler(M: number, e: number) {
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < 8; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  return E;
}

// ------------------------------------------------------------------ shaders
const NOISE = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;} vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);} vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){const vec2 C=vec2(1.0/6.0,1.0/3.0);const vec4 D=vec4(0.0,0.5,1.0,2.0);vec3 i=floor(v+dot(v,C.yyy));vec3 x0=v-i+dot(i,C.xxx);
vec3 g=step(x0.yzx,x0.xyz);vec3 l=1.0-g;vec3 i1=min(g.xyz,l.zxy);vec3 i2=max(g.xyz,l.zxy);vec3 x1=x0-i1+C.xxx;vec3 x2=x0-i2+C.yyy;vec3 x3=x0-D.yyy;
i=mod289(i);vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
float n_=0.142857142857;vec3 ns=n_*D.wyz-D.xzx;vec4 j=p-49.0*floor(p*ns.z*ns.z);vec4 x_=floor(j*ns.z);vec4 y_=floor(j-7.0*x_);vec4 x=x_*ns.x+ns.yyyy;
vec4 y=y_*ns.x+ns.yyyy;vec4 h=1.0-abs(x)-abs(y);vec4 b0=vec4(x.xy,y.xy);vec4 b1=vec4(x.zw,y.zw);vec4 s0=floor(b0)*2.0+1.0;vec4 s1=floor(b1)*2.0+1.0;
vec4 sh=-step(h,vec4(0.0));vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy;vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;vec3 p0=vec3(a0.xy,h.x);vec3 p1=vec3(a0.zw,h.y);
vec3 p2=vec3(a1.xy,h.z);vec3 p3=vec3(a1.zw,h.w);vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));p0*=norm.x;p1*=norm.y;p2*=norm.z;p3*=norm.w;
vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0);m=m*m;return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));}
float fbm(vec3 p){float a=0.5,s=0.0;for(int i=0;i<5;i++){s+=a*snoise(p);p*=2.03;a*=0.5;}return s;}
`;
const PLANET_VS = /* glsl */ `
varying vec3 vObj; varying vec3 vNormalW; varying vec3 vPosW;
void main(){ vObj = normalize(position); vec4 w = modelMatrix * vec4(position,1.0); vPosW = w.xyz; vNormalW = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * w; }`;
const PLANET_FS = /* glsl */ `
uniform vec3 uA; uniform vec3 uB; uniform vec3 uC; uniform int uType; uniform vec3 uSun; uniform float uTime; uniform float uSeed; uniform float uHeat;
varying vec3 vObj; varying vec3 vNormalW; varying vec3 vPosW;
${NOISE}
void main(){
  vec3 p = vObj * 2.0 + uSeed;
  vec3 col;
  if (uType == 3) {                                             // star: boiling granulation + limb darkening
    float g = fbm(p * 3.0 + vec3(0.0, 0.0, uTime * 0.15)); float s = fbm(p * 9.0 - uTime * 0.2);
    col = mix(uA, uB, 0.5 + 0.5 * g) * (0.85 + 0.3 * s);
    vec3 V = normalize(cameraPosition - vPosW); float mu = max(dot(normalize(vNormalW), V), 0.0);
    col *= 0.45 + 0.55 * pow(mu, 0.5); gl_FragColor = vec4(col * 1.6, 1.0); return;
  }
  float lat = vObj.z;
  if (uType == 1) {                                             // gas giant: latitude bands + turbulence + a storm
    float t = lat * 9.0 + fbm(p * vec3(1.0, 1.0, 4.0)) * 1.4;
    col = mix(uA, uB, 0.5 + 0.5 * sin(t * 3.14159));
    col = mix(col, uC, smoothstep(0.55, 0.9, fbm(p * 4.0)) * 0.35);
    float storm = smoothstep(0.16, 0.0, length(vec2(atan(vObj.y, vObj.x) - 1.0, (lat + 0.35) * 3.0)));
    col = mix(col, vec3(0.75, 0.38, 0.25), storm * 0.8);
  } else if (uType == 2) {                                      // Earth: oceans, continents, ice caps, clouds
    float h = fbm(p * 1.6);
    col = h > 0.08 ? mix(vec3(0.20, 0.42, 0.18), vec3(0.55, 0.47, 0.32), smoothstep(0.1, 0.5, h)) : mix(vec3(0.02, 0.10, 0.32), vec3(0.05, 0.25, 0.55), smoothstep(-0.5, 0.08, h));
    col = mix(col, vec3(0.95), smoothstep(0.78, 0.9, abs(lat)));
    float cl = smoothstep(0.15, 0.6, fbm(p * 2.4 + vec3(uTime * 0.02, 0.0, 0.0)));
    col = mix(col, vec3(1.0), cl * 0.75);
  } else if (uType == 4) {                                      // icy / cloudy smooth (Uranus, Neptune, Venus)
    col = mix(uA, uB, 0.5 + 0.5 * fbm(p * vec3(1.0, 1.0, 3.0)) * 0.6 + lat * 0.15);
  } else {                                                      // rocky: Mars, Mercury, moons, asteroids
    float h = fbm(p * 2.2); float cr = smoothstep(0.55, 0.62, fbm(p * 7.0));
    col = mix(uA, uB, 0.5 + 0.5 * h) * (1.0 - 0.25 * cr);
    col = mix(col, uC, smoothstep(0.82, 0.92, abs(lat)) * step(0.5, uSeed - floor(uSeed)));
  }
  vec3 L = normalize(uSun - vPosW); vec3 N = normalize(vNormalW);
  float d = dot(N, L);
  float lit = smoothstep(-0.08, 0.35, d) * 0.95 + 0.04;
  vec3 V = normalize(cameraPosition - vPosW);
  float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0) * smoothstep(-0.2, 0.4, d);
  col = col * lit + rim * uC * 0.6;
  col = mix(col, vec3(1.0, 0.35, 0.05), uHeat);
  gl_FragColor = vec4(col, 1.0);
}`;
const RING_FS = /* glsl */ `
uniform vec3 uCol; uniform float uIn; varying vec2 vUv; varying vec3 vPosW; uniform vec3 uSun;
${NOISE}
void main(){ float r = vUv.x; float bands = 0.55 + 0.45 * snoise(vec3(r * 60.0, 0.0, 0.0)); float gap = smoothstep(0.6, 0.62, r) * (1.0 - smoothstep(0.66, 0.68, r));
  float a = bands * (1.0 - gap * 0.9) * smoothstep(0.0, 0.05, r) * (1.0 - smoothstep(0.92, 1.0, r));
  gl_FragColor = vec4(uCol * (0.6 + 0.5 * bands), a * 0.8); }`;
const RING_VS = /* glsl */ `varying vec2 vUv; varying vec3 vPosW; uniform float uIn; uniform float uOut;
void main(){ float r = length(position.xy); vUv = vec2((r - uIn) / (uOut - uIn), 0.0); vec4 w = modelMatrix * vec4(position, 1.0); vPosW = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w; }`;
const DISK_FS = /* glsl */ `uniform float uTime; varying vec2 vUv; ${NOISE}
void main(){ float r = vUv.x; vec2 q = vUv.y * vec2(1.0); float sw = snoise(vec3(r * 6.0 - uTime * 0.8, vUv.y * 3.0, uTime * 0.2));
  vec3 hot = mix(vec3(1.0, 0.95, 0.85), vec3(1.0, 0.45, 0.08), r); float a = (1.0 - r) * smoothstep(0.0, 0.08, r) * (0.65 + 0.35 * sw);
  gl_FragColor = vec4(hot * (1.4 - r), a); }`;
const DISK_VS = /* glsl */ `varying vec2 vUv; uniform float uIn; uniform float uOut;
void main(){ float r = length(position.xy); vUv = vec2((r - uIn) / (uOut - uIn), atan(position.y, position.x) / 6.2832 + 0.5); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const GLOW_VS = /* glsl */ `attribute vec3 color; attribute float size; varying vec3 vC; uniform float uScale; uniform float uFixed;
void main(){ vC = color; vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = uFixed > 0.5 ? size * uScale : size * uScale / -mv.z; gl_Position = projectionMatrix * mv; }`;
const GLOW_FS = /* glsl */ `varying vec3 vC; uniform float uAlpha;
void main(){ vec2 d = gl_PointCoord - 0.5; float r = length(d); if (r > 0.5) discard; float a = exp(-r * r * 18.0); gl_FragColor = vec4(vC * a, a * uAlpha); }`;

// how each kind of body looks: [type, colour A, B, C]
function look(b: SpaceBody): [number, string, string, string] {
  const n = b.name.replace(/ \d+$/, "");
  if (b.kind === "star" || n === "sun") { const c = b.color || "#ffd27a"; return [3, c, "#fff1c4", c]; }
  if (n === "earth" || n === "planet") return [2, "#1e3a8a", "#16a34a", "#7dd3fc"];
  if (n === "jupiter") return [1, "#c9a27e", "#efe0c8", "#a0522d"];
  if (n === "saturn") return [1, "#d9c08f", "#f3e6c4", "#c9a66b"];
  if (n === "uranus") return [4, "#9be3ec", "#c8f3f7", "#c8f3f7"];
  if (n === "neptune") return [4, "#2f5bd8", "#5b8cf0", "#93c5fd"];
  if (n === "venus") return [4, "#d9b77a", "#f3dfb0", "#fde68a"];
  if (n === "mars") return [0, "#9a3412", "#d97745", "#f8fafc"];
  if (n === "io") return [0, "#d4a017", "#f5e663", "#fef3c7"];
  if (n === "europa" || n === "enceladus") return [0, "#d6d3d1", "#f5f5f4", "#ffffff"];
  if (n === "titan") return [4, "#c2852b", "#e3a857", "#fbbf24"];
  return [0, shade(b.color, 0.55), b.color || "#a8a29e", "#e2e8f0"];
}
function shade(hex: string, k: number) { const c = new THREE.Color(hex || "#999"); c.multiplyScalar(k); return "#" + c.getHexString(); }

function planetMaterial(b: SpaceBody, seed: number) {
  const [type, a, bb, c] = look(b);
  return new THREE.ShaderMaterial({
    vertexShader: PLANET_VS, fragmentShader: PLANET_FS,
    uniforms: { uA: { value: new THREE.Color(a) }, uB: { value: new THREE.Color(bb) }, uC: { value: new THREE.Color(c) }, uType: { value: type },
      uSun: { value: new THREE.Vector3() }, uTime: { value: 0 }, uSeed: { value: seed }, uHeat: { value: 0 } },
  });
}
let HALO: THREE.Texture | null = null;
function haloTex() {
  if (HALO) return HALO;
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d")!; const r = g.createRadialGradient(128, 128, 8, 128, 128, 128);
  r.addColorStop(0, "rgba(255,245,220,1)"); r.addColorStop(0.18, "rgba(255,210,140,0.55)"); r.addColorStop(0.45, "rgba(255,160,70,0.14)"); r.addColorStop(1, "rgba(255,120,40,0)");
  g.fillStyle = r; g.fillRect(0, 0, 256, 256);
  HALO = new THREE.CanvasTexture(c); return HALO;
}

// ------------------------------------------------------------------ the sky
function Sky({ galaxyMode }: { galaxyMode: boolean }) {
  const pts = useMemo(() => {
    const n = 7000, p = new Float32Array(n * 3), c = new Float32Array(n * 3), s = new Float32Array(n);
    let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    for (let i = 0; i < n; i++) {
      let x, y, z;
      if (!galaxyMode && i < n * 0.55) {                 // the Milky Way: stars crowd a great circle
        const th = rnd() * Math.PI * 2, w = (rnd() - 0.5) * 0.35 * (rnd() < 0.7 ? 0.4 : 1);
        x = Math.cos(th); y = Math.sin(th) * Math.cos(1.05) - w * Math.sin(1.05); z = Math.sin(th) * Math.sin(1.05) + w * Math.cos(1.05);
      } else { const u = rnd() * 2 - 1, th = rnd() * Math.PI * 2, q = Math.sqrt(1 - u * u); x = q * Math.cos(th); y = q * Math.sin(th); z = u; }
      const L = Math.hypot(x, y, z); p.set([(x / L) * 900, (y / L) * 900, (z / L) * 900], i * 3);
      const T = 3000 + rnd() * 9000; const col = new THREE.Color(T < 4500 ? "#ffcf9e" : T < 6500 ? "#fff4e8" : "#cfe0ff");
      const b = 0.35 + rnd() * rnd() * 0.9; c.set([col.r * b, col.g * b, col.b * b], i * 3); s[i] = 1.5 + rnd() * rnd() * 5;
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3)); g.setAttribute("color", new THREE.BufferAttribute(c, 3)); g.setAttribute("size", new THREE.BufferAttribute(s, 1));
    const m = new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, uniforms: { uScale: { value: 900 }, uAlpha: { value: 1 }, uFixed: { value: 0 } }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
    const o = new THREE.Points(g, m); o.frustumCulled = false; return o;
  }, [galaxyMode]);
  useFrame(({ camera }) => pts.position.copy(camera.position));
  return <primitive object={pts} />;
}

// ------------------------------------------------------------------ main component
export default function SpaceView({ run, resetKey, compact = false }: { run: GravityRun; resetKey: number; compact?: boolean }) {
  const [, tick] = useState(0);
  useEffect(() => clock.subscribe(() => tick((x) => (x + 1) % 1e6)), []);
  const host = useRef<HTMLDivElement>(null);
  const sunCentred = run.mode === "system" || run.mode === "real" || run.mode === "replace_sun" || run.mode === "flyby";
  const [compress, setCompress] = useState(sunCentred);
  const [trueSize, setTrueSize] = useState(false);
  const [follow, setFollow] = useState<number | null>(null);
  useEffect(() => { setCompress(sunCentred); setFollow(null); setTrueSize(false); }, [run.id]);   // eslint-disable-line react-hooks/exhaustive-deps
  const f = frameAt(run);
  const tSim = simTimeAt(run, f);
  const galaxy = run.mode === "galaxies";
  return (
    <div className="xview space-view">
      <Canvas dpr={[1, 2]} onCreated={({ gl, scene }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; gl.toneMappingExposure = 1.05; scene.background = new THREE.Color("#010208"); }}>
        <Sky galaxyMode={galaxy} />
        <Scene run={run} host={host} resetKey={resetKey} compress={compress} trueSize={trueSize} follow={follow} setFollow={setFollow} />
      </Canvas>
      <div className="xlabels" ref={host} />
      <div className={`xhud space-hud ${compact ? "compact" : ""}`}>
        {run.date ? <span className="pill dark date">{addSeconds(run.date, tSim)}</span> : <span className="pill dark date">t = {dur(tSim)}</span>}
        <span className="pill dark">1 s ≈ {dur(run.sim_seconds / run.duration)}</span>
        {run.source && <span className="pill dark src" title={run.source}>{run.source.includes("NASA") ? "● NASA JPL data" : run.source.includes("van der Marel") ? "● HST + Gaia measurements" : run.source.includes("measured") ? "● measured orbit" : "hypothetical set-up"}</span>}
        {!compact && !galaxy && (
          <span className="seg-toggle small dark">
            {sunCentred && <button className={compress ? "on" : ""} onClick={() => setCompress(!compress)} title="squeeze distances so the inner and outer planets both fit">{compress ? "distances compressed" : "true distances"}</button>}
            <button className={trueSize ? "on" : ""} onClick={() => setTrueSize(!trueSize)} title="true sizes are tiny compared with the distances between them">{trueSize ? "true sizes" : "sizes enlarged"}</button>
            {follow !== null && <button onClick={() => setFollow(null)}>stop following</button>}
          </span>
        )}
      </div>
    </div>
  );
}

type Vis = { obj: THREE.Object3D; spin: THREE.Object3D; r: number; name: string; mat?: THREE.ShaderMaterial; tail?: THREE.Points; kind: string; idx: number };

function Scene({ run, host, resetKey, compress, trueSize, follow, setFollow }: {
  run: GravityRun; host: React.RefObject<HTMLDivElement>; resetKey: number; compress: boolean; trueSize: boolean; follow: number | null; setFollow: (i: number | null) => void;
}) {
  const { camera, gl, size } = useThree();
  const nB = run.bodies.length;
  const sunI = run.sun_index ?? run.bodies.findIndex((b) => b.kind === "star" || b.name === "sun");
  const parentIdx = useMemo(() => run.bodies.map((b) => (b.parent ? run.bodies.findIndex((x) => x.name === b.parent) : -1)), [run]);
  const centreI = run.frame ? run.bodies.findIndex((b) => b.name === run.frame) : sunI;
  const focusSet = useMemo(() => (run.focus ? new Set(run.focus.filter(Boolean) as string[]) : null), [run]);
  const shown = (i: number) => !focusSet || focusSet.has(run.bodies[i].name) || i === sunI;
  // ---- distance mapping: metres -> scene units
  const ext = useMemo(() => {
    let e = 0;
    const c = centreI;
    for (const fr of run.pos) for (let i = 0; i < nB; i++) {
      if (parentIdx[i] >= 0 && run.bodies[i].kind === "moon" && run.mode === "system") continue;
      if (focusSet && !focusSet.has(run.bodies[i].name)) continue;
      const p = fr[i], o = c >= 0 ? fr[c] : [0, 0, 0];
      e = Math.max(e, Math.hypot(p[0] - o[0], p[1] - o[1], p[2] - o[2]));
    }
    run.bodies.forEach((b) => (e = Math.max(e, b.radius * 2)));
    for (const bl of run.belts ?? []) e = Math.max(e, Math.max(...bl.a) * 1.3);
    if (run.particles && run.mode === "galaxies") e = Math.max(e, 80 * run.particles.unit_m);
    if (run.view_extent_m) e = run.view_extent_m;                  // the encounter says how much space matters
    return Math.max(e, 1);
  }, [run, nB, centreI, parentIdx, focusSet]);
  const R0 = ext;
  const K = compress ? 0.42 : 1;
  const S = 8;                                                       // the scene fits in ~8 units
  const mapRadial = (r: number) => (S * Math.pow(Math.max(r, 0) / R0, K));
  const tmp = useMemo(() => new THREE.Vector3(), []), tmp2 = useMemo(() => new THREE.Vector3(), []), sunP = useMemo(() => new THREE.Vector3(), []);
  /** metres (relative to the frame centre) -> scene: radially compressed around the Sun when asked */
  const toScene = (m: THREE.Vector3, out: THREE.Vector3, sunM: THREE.Vector3 | null) => {
    if (!compress || !sunM) return out.copy(m).multiplyScalar(S / R0);
    tmp2.copy(m).sub(sunM); const r = tmp2.length();
    if (r < 1) return out.copy(sunM).multiplyScalar(S / R0);
    return out.copy(tmp2).multiplyScalar(mapRadial(r) / r).add(tmp.copy(sunM).multiplyScalar(S / R0));
  };
  // ---- visual sizes: true size, or enlarged (ordered by real size) so they can be seen
  const visR = useMemo(() => run.bodies.map((b, i) => {
    const trueR = (b.radius / R0) * S;
    if (compress && i === centreI && run.frame) return S * Math.pow(b.radius / R0, K);
    // compact objects are tens of km across - invisible at these distances, so they get a small visible size (the notes say so)
    if (trueSize || run.mode === "collide") return Math.max(trueR, b.kind === "black hole" ? 0.07 : b.kind === "neutron star" ? 0.045 : 1e-5);
    const kind = b.kind ?? "";
    const base = kind === "star" ? 0.42 : kind === "black hole" ? 0.12 : kind === "galaxy" ? 0.0 : kind === "moon" ? 0.045 : kind === "comet" || kind === "asteroid" || kind === "spacecraft" ? 0.035 : 0.06;
    const k = Math.pow(b.radius / 6.371e6, 0.32);
    return Math.max(trueR, Math.min(kind === "star" ? 0.7 : 0.32, base * (kind === "star" ? 1 : Math.max(0.6, k))));
  }), [run, R0, trueSize, compress, centreI, K]);
  const exaggerated = visR.some((r, i) => r > (run.bodies[i].radius / R0) * S * 1.5);
  // ---- bodies
  const vis: Vis[] = useMemo(() => run.bodies.map((b, i) => {
    const r = visR[i];
    const kind = b.kind ?? "";
    const g = new THREE.Group();
    // the planet's real spin axis (IAU pole): its bands, spin and rings all line up with it - and with its moons' orbits
    if (b.pole) g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(b.pole[0], b.pole[1], b.pole[2]).normalize());
    const spin = new THREE.Group(); g.add(spin);
    let mat: THREE.ShaderMaterial | undefined, tail: THREE.Points | undefined;
    if (kind === "galaxy") {
      // the galaxy's stars are drawn by the particle cloud; its core gets a soft glow
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex(), color: b.color, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }));
      (halo.material as THREE.SpriteMaterial).opacity = 0.12; halo.scale.setScalar((b.radius / R0) * S * 0.6); g.add(halo);
    } else if (kind === "black hole" && run.encounter === "gravitational-wave inspiral") {
      // merging black holes have no gas around them: a dark sphere with a thin lensing rim
      g.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshBasicMaterial({ color: "#000" })));
      const rim = new THREE.Mesh(new THREE.RingGeometry(r * 1.01, r * 1.08, 96), new THREE.MeshBasicMaterial({ color: "#c7d2fe", transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      rim.lookAt(0, -1, 0.3); g.add(rim);
    } else if (kind === "black hole") {
      g.add(new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), new THREE.MeshBasicMaterial({ color: "#000" })));
      const dm = new THREE.ShaderMaterial({ vertexShader: DISK_VS, fragmentShader: DISK_FS, uniforms: { uTime: { value: 0 }, uIn: { value: r * 1.5 }, uOut: { value: r * 5 } },
        transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending });
      const disk = new THREE.Mesh(new THREE.RingGeometry(r * 1.5, r * 5, 128, 4), dm); disk.rotation.x = 1.25; g.add(disk);
      const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.02, r * 1.25, 96), new THREE.MeshBasicMaterial({ color: "#ffb36b", transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }));
      ring.lookAt(0, -1, 0.3); g.add(ring);
      mat = dm;
    } else {
      mat = planetMaterial(b, i * 3.17 + 0.6);
      spin.add(new THREE.Mesh(new THREE.SphereGeometry(r, 96, 64), mat));
      if (kind === "star") {
        const halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: haloTex(), color: b.color, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
        halo.scale.set(r * 7, r * 7, 1); g.add(halo);
      }
      if (/^(earth|venus|titan|planet)$/.test(b.name.replace(/ \d+$/, ""))) {
        g.add(new THREE.Mesh(new THREE.SphereGeometry(r * 1.05, 48, 32), new THREE.MeshBasicMaterial({ color: b.name === "earth" ? "#60a5fa" : "#fde68a", transparent: true, opacity: 0.16, blending: THREE.AdditiveBlending, side: THREE.BackSide })));
      }
      if (b.rings && b.rings.length === 2) {
        const ri = r * (b.rings[0] / b.radius), ro = r * (b.rings[1] / b.radius);
        const rm = new THREE.ShaderMaterial({ vertexShader: RING_VS, fragmentShader: RING_FS, uniforms: { uCol: { value: new THREE.Color(b.name === "saturn" ? "#e9d8a6" : "#9fb6c4") }, uIn: { value: ri }, uOut: { value: ro }, uSun: { value: new THREE.Vector3() } },
          transparent: true, side: THREE.DoubleSide, depthWrite: false });
        const ring = new THREE.Mesh(new THREE.RingGeometry(ri, ro, 160, 1), rm);
        if (!b.pole) ring.rotation.x = 0.47;               // rings lie in the equator
        g.add(ring);
      }
      if (kind === "comet") {
        const n = 900, geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
        geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
        geo.setAttribute("size", new THREE.BufferAttribute(new Float32Array(n), 1));
        tail = new THREE.Points(geo, new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, uniforms: { uScale: { value: 300 }, uAlpha: { value: 0.9 }, uFixed: { value: 0 } },
          transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
        tail.frustumCulled = false;
      }
    }
    return { obj: g, spin, r, name: b.name, mat, tail, kind, idx: i };
  }), [run, visR, R0]);
  // ---- orbit paths (the whole path, faint) + where each has been (bright)
  const paths = useMemo(() => run.bodies.map((b, i) => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(run.pos.length * 3), 3));
    const col = new THREE.Color(b.kind === "black hole" ? "#fb923c" : b.color); col.lerp(new THREE.Color("#ffffff"), 0.25);
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: b.kind === "galaxy" ? 0.25 : 0.35 }));
    l.frustumCulled = false; l.visible = i !== sunI || run.mode === "binary" || run.mode === "collide" || run.mode === "orbit";
    return l;
  }), [run, sunI]);
  // ---- belts
  const belts = useMemo(() => (run.belts ?? []).map((bl) => {
    const n = bl.a.length, geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const c = new Float32Array(n * 3), s = new Float32Array(n); const base = new THREE.Color(bl.color);
    for (let k = 0; k < n; k++) { const v = 0.55 + 0.45 * ((k * 0.618) % 1); c.set([base.r * v, base.g * v, base.b * v], k * 3); s[k] = 2.2 + 2.0 * ((k * 0.377) % 1); }
    geo.setAttribute("color", new THREE.BufferAttribute(c, 3)); geo.setAttribute("size", new THREE.BufferAttribute(s, 1));
    const pts = new THREE.Points(geo, new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, uniforms: { uScale: { value: 60 }, uAlpha: { value: 0.85 }, uFixed: { value: 0 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    pts.frustumCulled = false; return { bl, pts };
  }), [run]);
  // ---- galaxy stars
  const galaxy = useMemo(() => {
    if (!run.particles) return null;
    const P = run.particles; const data = f16decode(P.data); const cols = u8decode(P.colors);
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(P.n * 3), 3));
    const c = new Float32Array(P.n * 3), s = new Float32Array(P.n);
    const big = run.encounter === "tidal disruption";                // debris: fewer, bigger, brighter points than a galaxy's stars
    for (let k = 0; k < P.n; k++) { c.set([cols[3 * k] / 255, cols[3 * k + 1] / 255, cols[3 * k + 2] / 255], 3 * k); s[k] = (big ? 4.5 : 1.9) + (big ? 3.5 : 1.8) * ((k * 0.618) % 1); }
    geo.setAttribute("color", new THREE.BufferAttribute(c, 3)); geo.setAttribute("size", new THREE.BufferAttribute(s, 1));
    const pts = new THREE.Points(geo, new THREE.ShaderMaterial({ vertexShader: GLOW_VS, fragmentShader: GLOW_FS, uniforms: { uScale: { value: window.devicePixelRatio || 1 }, uAlpha: { value: run.encounter === "tidal disruption" ? 1.0 : 0.8 }, uFixed: { value: 1 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
    pts.frustumCulled = false;
    return { pts, data, n: P.n, frames: P.frames, unit: P.unit_m };
  }, [run]);
  // ---- impact effects
  const debris = useMemo(() => {
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(3000 * 3), 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.04, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    p.frustumCulled = false; return p;
  }, []);
  const waves = useMemo(() => [0, 1, 2, 3].map(() => {
    const m = new THREE.Mesh(new THREE.RingGeometry(0.97, 1, 160), new THREE.MeshBasicMaterial({ color: "#a5b4fc", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    m.visible = false; return m;
  }), []);
  const shock = useMemo(() => new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), new THREE.MeshBasicMaterial({ color: "#fde68a", transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false })), []);
  // ---- labels
  const labels = useRef<HTMLDivElement[]>([]);
  useEffect(() => {
    const h = host.current; if (!h) return;
    labels.current = run.bodies.map((b, i) => {
      const d = document.createElement("div"); d.className = `x-label space-label k-${(b.kind ?? "").replace(/\s+/g, "-")}`;
      d.innerHTML = `<b>${b.label ?? b.name}</b><span></span>`; d.title = "click to ride along"; d.style.pointerEvents = "auto"; d.style.cursor = "pointer";
      d.onclick = () => setFollow(i);
      h.appendChild(d); return d;
    });
    return () => labels.current.forEach((d) => d.remove());
  }, [run, host, setFollow]);
  // ---- camera
  const ctl = useRef<OrbitControls | null>(null);
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    cam.up.set(0, 0, 1); cam.fov = 42; cam.near = 0.0005; cam.far = 5000;
    const flat = run.mode === "galaxies" ? [0, -16, 9] : run.mode === "collide" ? [0, -10, 3] : [0, -12, 7.5];
    cam.position.set(flat[0], flat[1], flat[2]); cam.updateProjectionMatrix();
    const oc = new OrbitControls(cam, gl.domElement); oc.enableDamping = true; oc.dampingFactor = 0.08; oc.target.set(0, 0, 0); oc.update();
    ctl.current = oc;
    userMoved.current = false;
    oc.addEventListener("start", () => { userMoved.current = true; });
    return () => oc.dispose();
  }, [run, camera, gl, resetKey]);
  const followPrev = useRef<THREE.Vector3 | null>(null);
  const userMoved = useRef(false);
  // ---- per frame
  const scenePos = useMemo(() => run.bodies.map(() => new THREE.Vector3()), [run]);
  const mPos = useMemo(() => run.bodies.map(() => new THREE.Vector3()), [run]);
  useFrame((st) => {
    ctl.current?.update();
    const f = frameAt(run); const tSim = simTimeAt(run, f); const time = st.clock.elapsedTime;
    for (let i = 0; i < nB; i++) rawPos(run, i, f, mPos[i]);
    const sunM = centreI >= 0 ? mPos[centreI] : null;            // centre of the (compressed) view
    const solM = sunI >= 0 ? mPos[sunI] : null;
    // positions: moons sit at an enlarged distance from their (compressed) planet so you can see them go round
    for (let i = 0; i < nB; i++) {
      const pi = parentIdx[i];
      if (pi >= 0 && run.bodies[i].kind === "moon" && (compress || !trueSize) && sunM && run.mode === "system") {
        toScene(mPos[pi], scenePos[pi], sunM);
        const rel = tmp.copy(mPos[i]).sub(mPos[pi]);
        const d = rel.length() || 1;
        const want = visR[pi] * 2.6 + visR[i] * 2 + 0.05 * Math.log10(1 + d / 1e8);
        scenePos[i].copy(scenePos[pi]).add(rel.multiplyScalar(want / d));
      } else toScene(mPos[i], scenePos[i], sunM);
    }
    const sunS = sunI >= 0 ? scenePos[sunI] : sunP.set(0, 0, 0);
    vis.forEach((v, i) => {
      v.obj.position.copy(scenePos[i]);
      v.spin.rotation.z = time * 0.15 + i;
      const absorbed = run.events.find((e) => e.a === v.name && e.t <= tSim) || !shown(i);
      v.obj.visible = !absorbed;
      if (v.mat?.uniforms.uSun) { v.mat.uniforms.uSun.value.copy(sunI >= 0 ? sunS : tmp.set(-50, -30, 40)); v.mat.uniforms.uTime.value = time; }
      if (v.mat?.uniforms.uTime && !v.mat.uniforms.uSun) v.mat.uniforms.uTime.value = time;
      // comet tails: gas and dust blown straight away from the Sun, longer the closer it gets (sunlight ∝ 1/r²)
      if (v.tail && solM) {
        const rs = tmp.copy(mPos[i]).sub(solM).length() / 1.496e11;
        const L = Math.min(2.5, 0.9 / Math.max(rs, 0.25) ** 1.6) * (rs < 5 ? 1 : Math.max(0, 1 - (rs - 5) / 3));
        const dir = tmp2.copy(scenePos[i]).sub(sunS).normalize();
        const P = v.tail.geometry.attributes.position as THREE.BufferAttribute, C = v.tail.geometry.attributes.color as THREE.BufferAttribute, Z = v.tail.geometry.attributes.size as THREE.BufferAttribute;
        const side = new THREE.Vector3(-dir.y, dir.x, 0).normalize(), up = new THREE.Vector3().crossVectors(dir, side);
        for (let k = 0; k < P.count; k++) {
          const u = ((k * 0.6180339) % 1), w = ((k * 0.7548776) % 1) - 0.5, q = ((k * 0.5698403) % 1) - 0.5;
          const dust = k % 3 === 0, s = u * L;
          const spread = (dust ? 0.12 : 0.05) * s + 0.004;
          const curve = dust ? s * s * 0.25 : 0;
          P.setXYZ(k, scenePos[i].x + dir.x * s + side.x * (w * spread + curve) + up.x * q * spread, scenePos[i].y + dir.y * s + side.y * (w * spread + curve) + up.y * q * spread,
            scenePos[i].z + dir.z * s + side.z * (w * spread + curve) + up.z * q * spread);
          const a = (1 - u) * (L > 0.02 ? 1 : 0);
          if (dust) C.setXYZ(k, 1.0 * a, 0.85 * a, 0.6 * a); else C.setXYZ(k, 0.45 * a, 0.75 * a, 1.0 * a);
          Z.setX(k, 2 + 5 * (1 - u));
        }
        P.needsUpdate = C.needsUpdate = Z.needsUpdate = true;
      }
      // path: whole path faint; grows as it goes
      const pos = paths[i].geometry.attributes.position as THREE.BufferAttribute;
      const n = Math.floor(f) + 1;
      if (paths[i].userData.compress !== compress || paths[i].userData.trueSize !== trueSize) {
        for (let k = 0; k < run.pos.length; k++) {
          const P = run.pos[k][i]; const sm = sunM ? tmp2.set(...(run.pos[k][centreI] as [number, number, number])) : null;
          const pi = parentIdx[i];
          if (pi >= 0 && run.bodies[i].kind === "moon" && (compress || !trueSize) && sunM && run.mode === "system") { pos.setXYZ(k, 0, 0, 0); continue; }
          toScene(tmp.set(P[0], P[1], P[2]), sunP, sm ? new THREE.Vector3().copy(sm) : null);
          pos.setXYZ(k, sunP.x, sunP.y, sunP.z);
        }
        pos.needsUpdate = true; paths[i].userData.compress = compress; paths[i].userData.trueSize = trueSize;
      }
      paths[i].geometry.setDrawRange(0, run.mode === "system" || run.mode === "real" || run.mode === "binary" ? run.pos.length : n);
      (paths[i].material as THREE.LineBasicMaterial).opacity = run.mode === "system" || run.mode === "real" ? 0.28 : 0.5;
      paths[i].visible = !absorbed && run.bodies[i].kind !== "galaxy" && !(i === sunI && (run.mode === "system" || run.mode === "real" || run.mode === "flyby" || run.mode === "replace_sun"))
        && !(parentIdx[i] >= 0 && run.bodies[i].kind === "moon" && run.mode === "system");
      // label
      const d = labels.current[i];
      if (d) {
        const p = tmp.copy(scenePos[i]).add(tmp2.set(0, 0, v.r * 1.25 + 0.02)).project(camera);
        const behind = p.z > 1;
        d.style.display = absorbed || behind ? "none" : "flex";
        d.style.transform = `translate(${((p.x + 1) / 2) * size.width}px, ${((1 - p.y) / 2) * size.height}px) translate(-50%, -100%)`;
        d.classList.toggle("following", follow === i);
        const a = Math.floor(f), b = Math.min(run.pos.length - 1, a + 1); const dt = (run.t[b] - run.t[a]) || 1;
        const pa = run.pos[a][i], pb = run.pos[b][i];
        const ref = parentIdx[i] >= 0 ? parentIdx[i] : sunI >= 0 && i !== sunI ? sunI : -1;
        let sp = Math.hypot(pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]) / dt;
        if (ref >= 0) { const ra = run.pos[a][ref], rb = run.pos[b][ref]; sp = Math.hypot(pb[0] - pa[0] - (rb[0] - ra[0]), pb[1] - pa[1] - (rb[1] - ra[1]), pb[2] - pa[2] - (rb[2] - ra[2])) / dt; }
        (d.lastChild as HTMLElement).textContent = run.mode === "galaxies" ? "" : `${(sp / 1000).toFixed(sp > 1e5 ? 0 : 1)} km/s`;
      }
    });
    // belts: each rock on its own Kepler orbit around the Sun
    if (solM && sunM) belts.forEach(({ bl, pts }) => {
      const P = pts.geometry.attributes.position as THREE.BufferAttribute;
      for (let k = 0; k < bl.a.length; k++) {
        const a = bl.a[k], e = bl.e[k], inc = bl.i[k], node = bl.node[k], w = bl.argp[k];
        const nmot = Math.sqrt(bl.mu / (a * a * a)); const M = bl.M0[k] + nmot * tSim;
        const E = kepler(((M % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI), e);
        const xo = a * (Math.cos(E) - e), yo = a * Math.sqrt(1 - e * e) * Math.sin(E);
        const cO = Math.cos(node), sO = Math.sin(node), ci = Math.cos(inc), si = Math.sin(inc), cw = Math.cos(w), sw = Math.sin(w);
        const x = (cO * cw - sO * sw * ci) * xo + (-cO * sw - sO * cw * ci) * yo, y = (sO * cw + cO * sw * ci) * xo + (-sO * sw + cO * cw * ci) * yo, z = sw * si * xo + cw * si * yo;
        toScene(tmp.set(solM.x + x, solM.y + y, solM.z + z), sunP, sunM);
        P.setXYZ(k, sunP.x, sunP.y, sunP.z);
      }
      P.needsUpdate = true;
    });
    // galaxy stars
    if (galaxy) {
      const P = galaxy.pts.geometry.attributes.position as THREE.BufferAttribute;
      const gf = Math.min(galaxy.frames - 1, (f / Math.max(1, run.t.length - 1)) * (galaxy.frames - 1));
      const a = Math.floor(gf), b = Math.min(galaxy.frames - 1, a + 1), k = gf - a, s = (galaxy.unit / R0) * S;
      const A = a * galaxy.n * 3, B = b * galaxy.n * 3;
      for (let q = 0; q < galaxy.n * 3; q++) (P.array as Float32Array)[q] = (galaxy.data[A + q] + (galaxy.data[B + q] - galaxy.data[A + q]) * k) * s;
      P.needsUpdate = true;
    }
    // galaxies: a cinematic camera keeps both in frame as they close in (until you take over)
    if (run.mode === "galaxies" && !userMoved.current && follow === null && ctl.current && nB >= 2) {
      const mid = tmp.copy(scenePos[0]).add(scenePos[1]).multiplyScalar(0.5);
      const sep = scenePos[0].distanceTo(scenePos[1]);
      const want = Math.max(sep * 1.15, 1.2) + 0.9;
      const off = tmp2.copy(camera.position).sub(ctl.current.target);
      const len = off.length() || 1;
      off.multiplyScalar(1 + (want / len - 1) * 0.06);
      ctl.current.target.lerp(mid, 0.08);
      camera.position.copy(ctl.current.target).add(off);
    }
    // ride along with a body
    if (follow !== null && ctl.current) {
      const target = scenePos[follow];
      if (followPrev.current) { const dv = tmp.copy(target).sub(followPrev.current); camera.position.add(dv); ctl.current.target.add(dv); }
      else { ctl.current.target.copy(target); camera.position.copy(target).add(tmp2.set(0, -Math.max(0.6, vis[follow].r * 8), Math.max(0.35, vis[follow].r * 4))); }
      followPrev.current = (followPrev.current ?? new THREE.Vector3()).copy(target);
    } else followPrev.current = null;
    // impact
    const e0 = run.events[0];
    const e = e0 && e0.kind === "impact" ? e0 : undefined;          // only a real impact explodes; a tidal disruption just tears
    const D = debris.geometry.attributes.position as THREE.BufferAttribute, Cc = debris.geometry.attributes.color as THREE.BufferAttribute;
    let nd = 0;
    // gravitational waves: ripples of space-time spreading in the orbital plane - while they spiral in, then a burst at the merger
    const gw = e0 && e0.kind === "gw" ? e0 : undefined;
    waves.forEach((w, k) => {
      if (!gw) { w.visible = false; return; }
      const span = Math.max(1, run.sim_seconds);
      const ph = ((tSim / span) * 6 + k / waves.length) % 1;          // a steady train of rings...
      const after = (tSim - gw.t) / span * run.duration;              // ...then the merger burst
      const R = after > 0 ? 0.3 + after * 2.2 + k * 0.5 : 0.25 + ph * 7.5;
      const op = after > 0 ? Math.max(0, 0.75 * (1 - after / 3.5)) : 0.35 * (1 - ph) * Math.min(1, tSim / gw.t * 2);
      w.visible = op > 0.01; w.scale.setScalar(R); (w.material as THREE.MeshBasicMaterial).opacity = op;
    });
    if (e) {
      const age = (tSim - e.t) / Math.max(1, run.sim_seconds) * run.duration;
      const c = toScene(new THREE.Vector3(...(e.pos as [number, number, number])), new THREE.Vector3(), sunM);
      const R0i = Math.max(e.small_radius / R0 * S, 0.02);
      if (age >= 0 && age < 6) {
        shock.visible = true; shock.position.copy(c);
        // a short flash hugging the impact site, never bigger than the planet itself (ejecta stay near it; most falls back)
        const Rbig = Math.max(R0i, (e.big_radius / R0) * S);
        const Rr = Math.min(R0i * (1 + age * 2), Rbig * 1.25); shock.scale.set(Rr, Rr, Rr);
        (shock.material as THREE.MeshBasicMaterial).opacity = Math.max(0, 0.45 * (1 - age / 1.2));
        for (let k = 0; k < 3000; k++, nd++) {
          const h1 = Math.sin(k * 12.9898) * 43758.5453, r1 = h1 - Math.floor(h1);
          const h2 = Math.sin(k * 78.233) * 12345.678, r2 = h2 - Math.floor(h2);
          const h3 = Math.sin(k * 39.425) * 9876.54, r3 = h3 - Math.floor(h3);
          const th = r1 * Math.PI * 2, ph = Math.acos(2 * r2 - 1), sp = Math.min(R0i * (0.6 + 3 * r3), Rbig * 0.5);
          D.setXYZ(k, c.x + Math.sin(ph) * Math.cos(th) * sp * age, c.y + Math.sin(ph) * Math.sin(th) * sp * age, c.z + Math.cos(ph) * sp * age);
          const al = Math.max(0, 1 - age / 5); Cc.setXYZ(k, al, (0.45 + 0.4 * r3) * al, 0.15 * al);
        }
        const big = vis.find((m) => m.name === e.b);
        if (big?.mat?.uniforms.uHeat) big.mat.uniforms.uHeat.value = Math.max(0, 0.85 * (1 - age / 6));
      } else shock.visible = false;
    }
    debris.geometry.setDrawRange(0, nd); D.needsUpdate = true; Cc.needsUpdate = true;
    (debris.material as THREE.PointsMaterial).size = Math.max(0.01, (e?.small_radius ?? 1) / R0 * S * 0.08);
  });
  return (
    <>
      {vis.map((m, i) => <primitive key={`b${i}`} object={m.obj} />)}
      {vis.map((m, i) => m.tail ? <primitive key={`t${i}`} object={m.tail} /> : null)}
      {paths.map((p, i) => <primitive key={`p${i}`} object={p} />)}
      {belts.map((b, i) => <primitive key={`belt${i}`} object={b.pts} />)}
      {galaxy && <primitive object={galaxy.pts} />}
      <primitive object={debris} />
      <primitive object={shock} />
      {waves.map((w, i) => <primitive key={`gw${i}`} object={w} />)}
      {exaggerated ? null : null}
    </>
  );
}
