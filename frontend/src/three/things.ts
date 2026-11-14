/* Procedural 3D models for anything without a scanned or library model.
 *
 * Every model is built from the SAME real dimensions the physics uses (L along x = forward,
 * W along y, H along z = up, origin at the bottom centre), so what you see is exactly what
 * collides. An archetype (quadruped, fish, winged, vehicle, building, ball, …) sets the anatomy;
 * species/name details (an elephant's trunk, a soccer ball's pentagons, a castle's towers) are
 * layered on top. Animated parts (legs, flukes, wings, wheels, rotors, flames) are driven by the
 * recorded motion: distance travelled, speed and time.
 */
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";

export interface ThingSpec {
  name: string; archetype: string; dims: number[]; color: string; material: string; animate: boolean;
  postHeight?: number;              // for a support: how far down the ground is
}
export interface MotionState { speed: number; dist: number; t: number; vz: number; propelled: boolean; tension?: number; swim?: boolean }
export interface BuiltThing { root: THREE.Group; update(m: MotionState): void }

// ---------------------------------------------------------------- utilities
const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
function hash(n: number) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
function hashStr(s: string) { let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return Math.abs(h); }
function vnoise(x: number, y: number, z: number) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const xf = x - xi, yf = y - yi, zf = z - zi;
  const s = (t: number) => t * t * (3 - 2 * t);
  const h = (a: number, b: number, c: number) => hash(a * 157 + b * 113 + c * 271);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const u = s(xf), v = s(yf), w = s(zf);
  return lerp(lerp(lerp(h(xi, yi, zi), h(xi + 1, yi, zi), u), lerp(h(xi, yi + 1, zi), h(xi + 1, yi + 1, zi), u), v),
    lerp(lerp(h(xi, yi, zi + 1), h(xi + 1, yi, zi + 1), u), lerp(h(xi, yi + 1, zi + 1), h(xi + 1, yi + 1, zi + 1), u), v), w);
}
export function fbm(x: number, y: number, z: number, oct = 4) {
  let a = 0.5, f = 1, s = 0;
  for (let i = 0; i < oct; i++) { s += a * vnoise(x * f, y * f, z * f); f *= 2.03; a *= 0.5; }
  return s;
}
const has = (name: string, re: RegExp) => re.test(name.toLowerCase());

const MAT_CACHE = new Map<string, THREE.Material>();
export function material(color: string, kind: string, opts: Partial<THREE.MeshPhysicalMaterialParameters> = {}): THREE.MeshPhysicalMaterial {
  const key = `${color}|${kind}|${JSON.stringify(opts)}`;
  if (MAT_CACHE.has(key)) return MAT_CACHE.get(key) as THREE.MeshPhysicalMaterial;
  const base: THREE.MeshPhysicalMaterialParameters = { color, roughness: 0.6, metalness: 0 };
  const k: Record<string, THREE.MeshPhysicalMaterialParameters> = {
    metal: { metalness: 1, roughness: 0.28, clearcoat: 0.3 },
    steel: { metalness: 1, roughness: 0.2 },
    glass: { transmission: 0.95, thickness: 0.02, roughness: 0.04, ior: 1.5, clearcoat: 1, transparent: true },
    ice: { transmission: 0.7, thickness: 0.1, roughness: 0.12, ior: 1.31, transparent: true },
    flesh: { roughness: 0.75, sheen: 0.4, sheenRoughness: 0.6, sheenColor: new THREE.Color("#ffffff") },
    rubber: { roughness: 0.45, clearcoat: 0.2 },
    leather: { roughness: 0.55, sheen: 0.2 },
    wood: { roughness: 0.68 },
    stone: { roughness: 0.92 },
    plastic: { roughness: 0.32, clearcoat: 0.5, clearcoatRoughness: 0.25 },
    fabric: { roughness: 0.95, sheen: 1, sheenRoughness: 0.8, sheenColor: new THREE.Color(color).multiplyScalar(1.3) },
    shell: { roughness: 0.4, clearcoat: 0.6 },
    food: { roughness: 0.5, clearcoat: 0.25 },
    ceramic: { roughness: 0.2, clearcoat: 1 },
    paper: { roughness: 0.9 },
    plasma: { emissive: new THREE.Color(color), emissiveIntensity: 2.5, roughness: 1 },
    gas: { roughness: 0.6 },
    wax: { roughness: 0.4, transmission: 0.2, thickness: 0.05 },
  };
  const m = new THREE.MeshPhysicalMaterial({ ...base, ...(k[kind] ?? {}), ...opts });
  MAT_CACHE.set(key, m);
  return m;
}
const vc = (kind: string, opts: Partial<THREE.MeshPhysicalMaterialParameters> = {}) => material("#ffffff", kind, { vertexColors: true, ...opts });

function mesh(g: THREE.BufferGeometry, m: THREE.Material, p: THREE.Vector3 = V(), r?: THREE.Euler): THREE.Mesh {
  // a vertex-coloured material on geometry without colours renders black: give it a neutral coat
  if ((m as THREE.MeshPhysicalMaterial).vertexColors && !g.attributes.color) paint(g, () => C("#8a8f98"));
  const x = new THREE.Mesh(g, m);
  x.position.copy(p);
  if (r) x.rotation.copy(r);
  x.castShadow = true; x.receiveShadow = true;
  return x;
}
/** Paint vertex colours from a function of the unit direction (for spheres/ellipsoids). */
function paint(g: THREE.BufferGeometry, fn: (n: THREE.Vector3, p: THREE.Vector3) => THREE.Color) {
  const pos = g.attributes.position;
  const col = new Float32Array(pos.count * 3);
  const p = V(), n = V();
  for (let i = 0; i < pos.count; i++) {
    p.fromBufferAttribute(pos, i);
    n.copy(p).normalize();
    const c = fn(n, p);
    col[i * 3] = c.r; col[i * 3 + 1] = c.g; col[i * 3 + 2] = c.b;
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return g;
}
const C = (c: string) => new THREE.Color(c);
function ellipsoid(rx: number, ry: number, rz: number, seg = 40) {
  const g = new THREE.SphereGeometry(1, seg, Math.round(seg * 0.7));
  g.scale(rx, ry, rz);
  return g;
}
/** Cylinder along z with its base at z=0. */
function cylZ(r1: number, r2: number, h: number, seg = 24) {
  const g = new THREE.CylinderGeometry(r2, r1, h, seg);
  g.rotateX(Math.PI / 2); g.translate(0, 0, h / 2);
  return g;
}
/** Cylinder along x centred at 0. */
function cylX(r: number, len: number, seg = 24) {
  const g = new THREE.CylinderGeometry(r, r, len, seg);
  g.rotateZ(Math.PI / 2);
  return g;
}
function canvasTex(w: number, h: number, draw: (c: CanvasRenderingContext2D) => void, repeat?: [number, number]) {
  const cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  draw(cv.getContext("2d")!);
  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}

// ---------------------------------------------------------------- ball & round things
const ICO = (() => {
  const g = new THREE.IcosahedronGeometry(1, 0);
  const p = g.attributes.position, out: THREE.Vector3[] = [];
  for (let i = 0; i < p.count; i++) {
    const v = V().fromBufferAttribute(p, i).normalize();
    if (!out.some((o) => o.distanceTo(v) < 1e-3)) out.push(v);
  }
  return out;
})();

function ballColor(name: string, base: string): ((n: THREE.Vector3) => THREE.Color) | null {
  const b = C(base);
  if (has(name, /soccer|football(?! american)|futbol/) && !has(name, /american|rugby/)) {
    return (n) => (ICO.some((d) => n.angleTo(d) < 0.36) ? C("#111317") : C("#f4f4f2"));
  }
  if (has(name, /basketball/)) {
    return (n) => {
      const line = Math.abs(n.x) < 0.022 || Math.abs(n.y) < 0.022 || Math.abs(Math.hypot(n.x, n.z) - 0.72) < 0.022;
      return line ? C("#1a1a1a") : C("#d4621f").multiplyScalar(0.92 + 0.12 * fbm(n.x * 40, n.y * 40, n.z * 40));
    };
  }
  if (has(name, /baseball|softball|cricket/)) {
    const seam = has(name, /cricket/) ? "#f5f5f5" : "#c8102e";
    const body = has(name, /cricket/) ? "#8b1a1a" : has(name, /softball/) ? "#f2f08a" : "#f3f1ea";
    return (n) => {
      const f = n.z - 0.62 * (n.x * n.x - n.y * n.y);
      if (has(name, /cricket/)) return Math.abs(n.z) < 0.03 ? C(seam) : C(body);
      return Math.abs(f) < 0.035 ? C(seam) : C(body);
    };
  }
  if (has(name, /tennis/)) return (n) => (Math.abs(n.z - 0.62 * (n.x * n.x - n.y * n.y)) < 0.03 ? C("#f8fafc") : C("#cde23a").multiplyScalar(0.95 + 0.1 * fbm(n.x * 60, n.y * 60, n.z * 60)));
  if (has(name, /bowling/)) return (n) => {
    const holes = [V(0.18, 0.1, 0.97).normalize(), V(0.18, -0.1, 0.97).normalize(), V(-0.15, 0, 0.98).normalize()];
    if (holes.some((h) => n.angleTo(h) < 0.07)) return C("#05050a");
    const m = fbm(n.x * 3 + 5, n.y * 3, n.z * 3);
    return C("#1e1b4b").lerp(C("#7c3aed"), Math.max(0, m - 0.45) * 1.6);
  };
  if (has(name, /beach/)) {
    const cols = ["#ef4444", "#facc15", "#3b82f6", "#f8fafc", "#22c55e", "#f97316"];
    return (n) => (Math.abs(n.z) > 0.93 ? C("#f8fafc") : C(cols[Math.floor(((Math.atan2(n.y, n.x) + Math.PI) / (2 * Math.PI)) * 6) % 6]));
  }
  if (has(name, /volleyball/)) return (n) => { const s = Math.floor((Math.atan2(n.y, n.x) + Math.PI) / (Math.PI / 3)); return C(["#f8fafc", "#facc15", "#2563eb"][s % 3]); };
  if (has(name, /earth|globe/)) return (n) => {
    if (Math.abs(n.z) > 0.9) return C("#f1f5f9");
    const land = fbm(n.x * 2.2 + 3, n.y * 2.2, n.z * 2.2, 5);
    if (land > 0.53) return C(land > 0.62 ? "#8b7d5a" : "#3f7d3a");
    return C("#1d4ed8").lerp(C("#0c2a6b"), 0.4 * fbm(n.x * 6, n.y * 6, n.z * 6));
  };
  if (has(name, /moon|asteroid|meteor|mercury|pluto/)) return (n) => {
    const cr = fbm(n.x * 8, n.y * 8, n.z * 8, 5);
    return C("#9ca3af").multiplyScalar(0.55 + 0.6 * cr - (cr > 0.62 ? 0.25 : 0));
  };
  if (has(name, /mars/)) return (n) => C("#c2410c").lerp(C("#7c2d12"), fbm(n.x * 4, n.y * 4, n.z * 4, 5)).lerp(C("#f8fafc"), Math.abs(n.z) > 0.92 ? 1 : 0);
  if (has(name, /jupiter|saturn/)) return (n) => C("#e7c9a0").lerp(C("#a0522d"), 0.5 + 0.5 * Math.sin(n.z * 22 + 3 * fbm(n.x * 3, n.y * 3, n.z * 3)));
  if (has(name, /watermelon/)) return (n) => C((Math.floor(((Math.atan2(n.y, n.x) + Math.PI) / (2 * Math.PI)) * 14 + 0.6 * fbm(n.x * 3, n.y * 3, n.z * 3)) % 2) ? "#14532d" : "#4d7c0f");
  if (has(name, /golf|ping ?pong|cue/)) return (n) => C("#fafafa").multiplyScalar(0.93 + 0.07 * vnoise(n.x * 90, n.y * 90, n.z * 90));
  if (has(name, /orange|lemon|grape|tomato|apple|melon|coconut|pumpkin|egg|potato/))
    return (n) => b.clone().multiplyScalar(0.85 + 0.25 * fbm(n.x * 25, n.y * 25, n.z * 25) - (has(name, /apple/) && n.z > 0.85 ? 0.3 : 0));
  return null;
}

function buildBall(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  const g = ellipsoid(L / 2, W / 2, H / 2, 64);
  const fn = ballColor(s.name, s.color);
  let m: THREE.Material;
  if (fn) { paint(g, fn); m = vc(has(s.name, /bowling|billiard|pool|cue/) ? "ceramic" : s.material === "unknown" ? "plastic" : s.material); }
  else if (has(s.name, /sun|star\b/)) m = material(s.color, "plasma");
  else if (has(s.name, /billiard|pool ball/)) m = material(s.color, "ceramic");
  else if (has(s.name, /marble|glass/)) m = material("#cfefff", "glass", { attenuationColor: new THREE.Color("#60a5fa"), attenuationDistance: 0.05 });
  else if (has(s.name, /steel|bearing|cannon|shot put|bob|metal/)) m = material(s.color === "#94a3b8" ? "#c9ccd1" : s.color, "steel");
  else { paint(g, (n) => C(s.color).multiplyScalar(0.9 + 0.15 * fbm(n.x * 12, n.y * 12, n.z * 12))); m = vc(s.material === "unknown" ? "plastic" : s.material); }
  const b = mesh(g, m, V(0, 0, H / 2));
  root.add(b);
  if (has(s.name, /balloon/)) {
    root.add(mesh(new THREE.ConeGeometry(L * 0.06, L * 0.08, 10).rotateX(Math.PI), material(s.color, "rubber"), V(0, 0, -L * 0.02)));
    const str = new THREE.Mesh(cylZ(0.0015, 0.0015, H * 1.2, 4), material("#e5e7eb", "fabric"));
    str.position.set(0, 0, -H * 1.22); root.add(str);
  }
  if (has(s.name, /pumpkin|apple|tomato/)) root.add(mesh(cylZ(L * 0.04, L * 0.025, H * 0.15, 8), material("#3f6212", "wood"), V(0, 0, H * 0.95)));
  return { root, update: () => {} };
}

// ---------------------------------------------------------------- rock / iceberg
function buildRock(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  const g = new THREE.IcosahedronGeometry(1, 4);
  const p = g.attributes.position, v = V();
  const seed = hashStr(s.name) % 97;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const d = 0.78 + 0.45 * fbm(v.x * 1.7 + seed, v.y * 1.7, v.z * 1.7, 5);
    p.setXYZ(i, v.x * d * L / 2, v.y * d * W / 2, v.z * d * H / 2);
  }
  g.computeVertexNormals();
  const ice = has(s.name, /ice/);
  paint(g, (n) => (ice ? C("#e0f2fe").lerp(C("#7dd3fc"), fbm(n.x * 5, n.y * 5, n.z * 5)) : C(s.color === "#94a3b8" ? "#78716c" : s.color).multiplyScalar(0.6 + 0.6 * fbm(n.x * 9, n.y * 9, n.z * 9))));
  root.add(mesh(g, vc(ice ? "ice" : "stone", { flatShading: !ice }), V(0, 0, H / 2)));
  return { root, update: () => {} };
}

// ---------------------------------------------------------------- four-legged (and many-legged) animals
interface Species { legFrac: number; neck?: number; tail?: number; ears?: "big" | "small" | "round" | "pointy"; trunk?: boolean; tusks?: boolean;
  horn?: boolean; horns?: boolean; mane?: boolean; snout?: number; shell?: boolean; legs?: number; legThick?: number; pattern?: string; biped?: boolean; head?: number }
function species(name: string): Species {
  const n = name.toLowerCase();
  if (/elephant|mammoth/.test(n)) return { legFrac: 0.45, ears: "big", trunk: true, tusks: true, tail: 0.25, legThick: 0.075, head: 1.2 };
  if (/giraffe/.test(n)) return { legFrac: 0.42, neck: 0.45, ears: "pointy", horns: true, tail: 0.3, pattern: "giraffe", legThick: 0.028, head: 0.6 };
  if (/zebra/.test(n)) return { legFrac: 0.45, neck: 0.15, mane: true, tail: 0.35, pattern: "zebra", ears: "pointy" };
  if (/horse|unicorn|donkey/.test(n)) return { legFrac: 0.45, neck: 0.15, mane: true, tail: 0.4, horn: /unicorn/.test(n), ears: "pointy" };
  if (/camel/.test(n)) return { legFrac: 0.48, neck: 0.18, tail: 0.25, ears: "small" };
  if (/tiger/.test(n)) return { legFrac: 0.35, tail: 0.8, ears: "round", pattern: "tiger", snout: 0.3 };
  if (/cheetah|leopard|jaguar/.test(n)) return { legFrac: 0.42, tail: 0.8, ears: "round", pattern: "spots", snout: 0.25 };
  if (/lion/.test(n)) return { legFrac: 0.38, tail: 0.7, ears: "round", mane: true, snout: 0.3 };
  if (/panda/.test(n)) return { legFrac: 0.3, ears: "round", pattern: "panda", tail: 0.05, head: 1.2 };
  if (/bear/.test(n)) return { legFrac: 0.32, ears: "round", tail: 0.05, snout: 0.35, head: 1.1 };
  if (/cow|bull|ox|buffalo/.test(n)) return { legFrac: 0.4, horns: true, tail: 0.5, ears: "small", pattern: /cow/.test(n) ? "cow" : undefined };
  if (/rhino/.test(n)) return { legFrac: 0.33, horn: true, tail: 0.2, ears: "pointy", legThick: 0.07, head: 1.1 };
  if (/hippo/.test(n)) return { legFrac: 0.25, tail: 0.1, ears: "small", legThick: 0.085, head: 1.3, snout: 0.5 };
  if (/crocodile|alligator|lizard/.test(n)) return { legFrac: 0.18, tail: 1.0, snout: 1.1, legThick: 0.09 };
  if (/turtle|tortoise/.test(n)) return { legFrac: 0.2, shell: true, tail: 0.1, legThick: 0.09 };
  if (/dinosaur|t-rex|tyrannosaurus|godzilla/.test(n)) return { legFrac: 0.4, tail: 1.1, biped: true, snout: 0.6, legThick: 0.06, head: 1.4 };
  if (/spider/.test(n)) return { legFrac: 0.5, legs: 8, tail: 0, legThick: 0.03 };
  if (/ant|beetle/.test(n)) return { legFrac: 0.4, legs: 6, tail: 0, legThick: 0.04 };
  if (/crab|lobster/.test(n)) return { legFrac: 0.35, legs: 8, tail: /lobster/.test(n) ? 0.6 : 0, legThick: 0.05 };
  if (/mouse|rat/.test(n)) return { legFrac: 0.2, ears: "round", tail: 1.2, snout: 0.3 };
  if (/rabbit/.test(n)) return { legFrac: 0.3, ears: "big", tail: 0.05 };
  if (/frog/.test(n)) return { legFrac: 0.25, tail: 0, head: 1.2 };
  if (/pig/.test(n)) return { legFrac: 0.28, ears: "pointy", tail: 0.15, snout: 0.2 };
  if (/dog|wolf|fox/.test(n)) return { legFrac: 0.45, ears: "pointy", tail: 0.5, snout: 0.35 };
  if (/cat/.test(n)) return { legFrac: 0.42, ears: "pointy", tail: 0.8, snout: 0.15 };
  return { legFrac: 0.42, ears: "pointy", tail: 0.4, snout: 0.25 };
}
function coatColor(sp: Species, base: THREE.Color, p: THREE.Vector3, scale: number) {
  const q = p.clone().divideScalar(scale);
  switch (sp.pattern) {
    case "zebra": return Math.sin(q.x * 38 + 4 * fbm(q.x * 3, q.y * 3, q.z * 3)) > 0.1 ? C("#111") : C("#f5f5f4");
    case "tiger": return Math.sin(q.x * 30 + 5 * fbm(q.x * 3, q.y * 3, q.z * 3)) > 0.65 ? C("#1c1917") : (q.z < -0.15 ? C("#fef3c7") : C("#ea7a12"));
    case "giraffe": return fbm(q.x * 9, q.y * 9, q.z * 9) > 0.5 ? C("#9a4a12") : C("#f2d29b");
    case "spots": return fbm(q.x * 16, q.y * 16, q.z * 16) > 0.62 ? C("#1c1917") : C("#e9b44c");
    case "cow": return fbm(q.x * 4 + 1, q.y * 4, q.z * 4) > 0.55 ? C("#111") : C("#f8fafc");
    case "panda": return Math.abs(q.x) < 0.12 || q.z < -0.1 ? C("#111") : C("#f8fafc");
  }
  return base.clone().multiplyScalar(0.85 + 0.25 * fbm(q.x * 14, q.y * 14, q.z * 14) + (q.z < -0.2 ? 0.12 : 0));
}

function buildQuadruped(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const sp = species(s.name);
  const root = new THREE.Group();
  const base = C(s.color);
  const legH = H * sp.legFrac;
  const neckH = H * (sp.neck ?? 0);
  const bodyH = Math.max(0.05, (H - legH - neckH) * 0.9);
  const tailFrac = sp.tail ?? 0.3;
  const headR = Math.min(bodyH * 0.42, W * 0.42) * (sp.head ?? 1);
  const bodyL = Math.max(L * 0.25, L - (sp.biped ? 0.2 : 0) * L - headR * 1.6 - (sp.snout ?? 0) * headR - tailFrac * L * 0.25) * 0.95;
  const bodyW = W * 0.82;
  const bodyZ = legH + bodyH / 2;
  const bx = (sp.biped ? -0.08 : -0.05) * L;
  const bodyG = ellipsoid(bodyL / 2, bodyW / 2, bodyH / 2, 48);
  paint(bodyG, (_, p) => coatColor(sp, base, p, Math.max(bodyL, H)));
  const coat = vc("flesh");
  const plain = material(`#${base.clone().multiplyScalar(0.9).getHexString()}`, "flesh", { side: THREE.DoubleSide });
  root.add(mesh(bodyG, coat, V(bx, 0, bodyZ), sp.biped ? new THREE.Euler(0, -0.25, 0) : undefined));
  if (sp.shell) {
    const sh = ellipsoid(bodyL * 0.6, bodyW * 0.62, bodyH * 0.75, 32);
    paint(sh, (n) => C("#4d5d24").lerp(C("#a16207"), Math.abs(Math.sin(n.x * 9) * Math.sin(n.y * 9))));
    root.add(mesh(sh, vc("shell"), V(bx, 0, bodyZ + bodyH * 0.15)));
  }
  // neck + head
  const front = bx + bodyL / 2;
  let hx = front + headR * 0.55, hz = bodyZ + bodyH * 0.3;
  if (neckH > 0) {
    const nl = Math.hypot(neckH, bodyL * 0.15);
    const neck = mesh(cylZ(headR * 0.6, headR * 0.45, nl, 16), coat, V(front - bodyL * 0.12, 0, bodyZ + bodyH * 0.2), new THREE.Euler(0, 0.35, 0));
    paint(neck.geometry, (_, p) => coatColor(sp, base, p.clone().add(V(0, 0, 3)), Math.max(bodyL, H)));
    root.add(neck);
    hx = front + Math.sin(0.35) * nl - bodyL * 0.05; hz = bodyZ + bodyH * 0.2 + Math.cos(0.35) * nl;
  } else if (sp.biped) { hx = front + headR * 0.4; hz = bodyZ + bodyH * 0.55; }
  const headG = ellipsoid(headR * 1.15, headR * 0.9, headR * 0.9, 32);
  paint(headG, (_, p) => coatColor(sp, base, p.clone().add(V(5, 0, 0)), Math.max(bodyL, H)));
  const head = new THREE.Group();
  head.position.set(hx, 0, hz);
  root.add(head);
  head.add(mesh(headG, coat));
  const snout = (sp.snout ?? 0.2) * headR;
  if (snout > 0.01) {
    const sg = ellipsoid(snout * 1.2 + headR * 0.3, headR * 0.5, headR * 0.45, 20);
    paint(sg, () => (sp.pattern === "panda" ? C("#f8fafc") : base.clone().multiplyScalar(0.92)));
    head.add(mesh(sg, coat, V(headR * 0.85 + snout * 0.5, 0, -headR * 0.2)));
    head.add(mesh(new THREE.SphereGeometry(headR * 0.12, 10, 8), material("#111", "plastic"), V(headR * 0.95 + snout * 1.6, 0, -headR * 0.1)));
  }
  const eyeM = material("#0b0b0b", "ceramic");
  for (const y of [-1, 1]) head.add(mesh(new THREE.SphereGeometry(headR * 0.11, 12, 8), eyeM, V(headR * 0.75, y * headR * 0.48, headR * 0.3)));
  if (sp.ears) for (const y of [-1, 1]) {
    const big = sp.ears === "big";
    const e = big ? mesh(new THREE.CircleGeometry(headR * (s.name.includes("rabbit") ? 0.4 : 0.8), 24), plain, V(-headR * 0.1, y * headR * 0.9, headR * 0.2), new THREE.Euler(Math.PI / 2, y * 0.3, 0))
      : mesh(new THREE.ConeGeometry(headR * (sp.ears === "round" ? 0.28 : 0.25), headR * (sp.ears === "round" ? 0.3 : 0.55), 10).rotateX(Math.PI / 2), plain,
        V(-headR * 0.15, y * headR * 0.55, headR * 0.85), new THREE.Euler(0, 0, 0));
    if (big && s.name.includes("rabbit")) { e.scale.set(0.4, 1.8, 1); e.position.z = headR * 1.2; }
    head.add(e);
  }
  if (sp.trunk) {
    let z = -headR * 0.2, x = headR * 0.9;
    for (let i = 0; i < 4; i++) {
      const seg = headR * 0.5;
      head.add(mesh(cylZ(headR * (0.28 - i * 0.04), headR * (0.24 - i * 0.04), seg, 12).rotateX(Math.PI), plain, V(x, 0, z), new THREE.Euler(0, -0.25 * i, 0)));
      x += seg * Math.sin(0.25 * i) * 0.6; z -= seg * 0.95;
    }
  }
  if (sp.tusks) for (const y of [-1, 1]) head.add(mesh(new THREE.ConeGeometry(headR * 0.08, headR * 1.1, 10).rotateZ(-Math.PI / 2.4), material("#f8f5e9", "ceramic"), V(headR * 1.2, y * headR * 0.35, -headR * 0.55)));
  if (sp.horn) head.add(mesh(new THREE.ConeGeometry(headR * 0.15, headR * 0.9, 12).rotateZ(-0.6), material(s.name.includes("unicorn") ? "#fde68a" : "#d6d3d1", "ceramic"), V(headR * (s.name.includes("unicorn") ? 0.6 : 1.3), 0, headR * (s.name.includes("unicorn") ? 1.0 : 0.35))));
  if (sp.horns) for (const y of [-1, 1]) head.add(mesh(new THREE.ConeGeometry(headR * 0.1, headR * 0.6, 8).rotateX(y * -0.6), material("#e7e5e4", "ceramic"), V(-headR * 0.1, y * headR * 0.6, headR * 0.85)));
  if (sp.mane) {
    const mg = s.name.includes("lion") ? ellipsoid(headR * 1.0, headR * 1.45, headR * 1.45, 24) : ellipsoid(bodyL * 0.2, headR * 0.25, headR * 0.5, 16);
    paint(mg, (n) => C(s.name.includes("lion") ? "#7c3f12" : "#3f2a1d").multiplyScalar(0.8 + 0.4 * fbm(n.x * 20, n.y * 20, n.z * 20)));
    head.add(mesh(mg, vc("fabric"), s.name.includes("lion") ? V(-headR * 0.45, 0, 0) : V(-headR * 1.2, 0, -headR * 0.1)));
  }
  // tail
  let tail: THREE.Object3D | null = null;
  if (tailFrac > 0.02) {
    const tl = tailFrac * L * 0.45;
    const thick = sp.tail! > 0.8 ? bodyH * 0.18 : bodyH * 0.06;
    tail = new THREE.Group();
    tail.position.set(bx - bodyL / 2 + thick, 0, bodyZ + bodyH * 0.15);
    const tg = cylZ(thick, thick * 0.3, tl, 12); tg.rotateY(-Math.PI / 2 - (sp.tail! > 0.8 ? 0.15 : 0.6));
    paint(tg, (_, p) => coatColor(sp, base, p.clone().add(V(-4, 0, 0)), Math.max(bodyL, H)));
    tail.add(mesh(tg, coat));
    root.add(tail);
  }
  // legs on hip pivots
  const nLegs = sp.biped ? 2 : sp.legs ?? 4;
  const legs: { pivot: THREE.Object3D; phase: number }[] = [];
  const lr = Math.min(W * 0.14, H * (sp.legThick ?? 0.045));
  const legM = vc("flesh");
  for (let i = 0; i < nLegs; i++) {
    const pair = Math.floor(i / 2), side = i % 2 ? 1 : -1;
    const pairs = Math.max(1, nLegs / 2);
    const fx = sp.biped ? bx - bodyL * 0.05 : bx + bodyL * (pairs === 1 ? 0 : 0.36 - (0.72 * pair) / (pairs - 1));
    const pivot = new THREE.Group();
    pivot.position.set(fx, side * bodyW * (nLegs > 4 ? 0.45 : 0.3), legH + bodyH * 0.1);
    const len = legH + bodyH * 0.1;
    const lg = cylZ(lr * 0.8, lr, len, 12); lg.translate(0, 0, -len);
    paint(lg, (_, p) => coatColor(sp, base, p.clone().add(V(fx, side, 0)), Math.max(bodyL, H)).multiplyScalar(0.92));
    const leg = mesh(lg, legM);
    if (nLegs > 4) { leg.rotation.x = side * 0.7; }
    pivot.add(leg);
    if (!(nLegs > 4)) pivot.add(mesh(new THREE.SphereGeometry(lr * 1.05, 10, 8), material(sp.trunk ? "#9ca3af" : "#2a211b", "flesh"), V(0, 0, -len)));
    root.add(pivot);
    legs.push({ pivot, phase: (pair % 2 ? Math.PI : 0) + (side > 0 ? Math.PI : 0) });
  }
  if (sp.biped) for (const y of [-1, 1]) root.add(mesh(cylZ(lr * 0.3, lr * 0.25, bodyH * 0.45).rotateY(2.2), plain, V(front - bodyL * 0.15, y * bodyW * 0.3, bodyZ + bodyH * 0.1)));
  const stride = Math.max(0.05, legH * 1.6);
  return {
    root,
    update: ({ dist, speed, t }) => {
      const ph = (dist / stride) * Math.PI * 2;
      const amp = Math.min(0.55, speed * 2 / Math.max(0.2, Math.sqrt(legH * 9.81)));
      for (const l of legs) l.pivot.rotation.y = Math.sin(ph + l.phase) * amp;
      if (tail) tail.rotation.z = Math.sin(t * 3 + dist) * 0.25 * (0.3 + amp);
      head.rotation.y = Math.sin(ph * 2) * amp * 0.06;
    },
  };
}

// ---------------------------------------------------------------- people & robots
function buildBiped(s: ThingSpec): BuiltThing {
  if (s.dims[0] > s.dims[2] * 1.5) {
    // lying down (swimming / crawling): build the person standing, then lay them along +x, head first
    const inner = buildBiped({ ...s, dims: [s.dims[2], s.dims[1], s.dims[0]] });
    const root = new THREE.Group();
    inner.root.rotation.y = Math.PI / 2;
    inner.root.position.set(-s.dims[0] / 2, 0, s.dims[2] / 2);
    root.add(inner.root);
    return { root, update: (m) => inner.update({ ...m, swim: true } as MotionState) };
  }
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const robot = /robot/.test(n), astro = /astronaut/.test(n), knight = /knight/.test(n), ape = /gorilla|monkey/.test(n), kanga = /kangaroo/.test(n);
  const skin = C(ape ? "#2b2b2b" : robot ? "#cbd5e1" : astro ? "#f8fafc" : s.color === "#94a3b8" ? "#d9a679" : s.color);
  const shirtCols = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#f59e0b"];
  const shirt = C(robot ? "#94a3b8" : astro ? "#f1f5f9" : knight ? "#9ca3af" : ape ? "#2b2b2b" : /zombie/.test(n) ? "#4d7c0f" : /ninja/.test(n) ? "#111827" : shirtCols[hashStr(s.name) % shirtCols.length]);
  const pants = C(robot ? "#64748b" : astro ? "#e2e8f0" : knight ? "#6b7280" : ape ? "#2b2b2b" : "#1f2937");
  const kind = robot || knight ? "metal" : astro ? "fabric" : "fabric";
  const legL = H * (ape ? 0.35 : 0.47), torsoL = H * (ape ? 0.38 : 0.31), headR = H * 0.065;
  const hipW = W * 0.45, shoulder = W * 0.55;
  const torso = robot ? new RoundedBoxGeometry(L * 0.6, shoulder * 1.6, torsoL, 3, 0.02) : ellipsoid(L * 0.32, shoulder * 0.9, torsoL / 2, 32);
  const tm = mesh(torso, material(`#${shirt.getHexString()}`, kind), V(0, 0, legL + torsoL / 2));
  root.add(tm);
  const head = mesh(robot ? new RoundedBoxGeometry(headR * 2.2, headR * 2, headR * 2.1, 3, 0.01) : new THREE.SphereGeometry(headR, 28, 20),
    material(`#${skin.getHexString()}`, robot ? "metal" : "flesh"), V(0, 0, legL + torsoL + headR * 1.25));
  root.add(head);
  if (astro) root.add(mesh(new THREE.SphereGeometry(headR * 1.5, 28, 20), material("#f8fafc", "plastic"), V(0, 0, legL + torsoL + headR * 1.25)),
    mesh(new THREE.SphereGeometry(headR * 1.3, 24, 16, -0.9, 1.8, 0.8, 1.2), material("#d4a017", "metal", { roughness: 0.05 }), V(headR * 0.3, 0, legL + torsoL + headR * 1.25)));
  if (robot) for (const y of [-1, 1]) root.add(mesh(new THREE.SphereGeometry(headR * 0.22, 10, 8), material("#38bdf8", "plasma"), V(headR * 1.1, y * headR * 0.45, legL + torsoL + headR * 1.4)));
  else if (!astro) for (const y of [-1, 1]) root.add(mesh(new THREE.SphereGeometry(headR * 0.13, 10, 8), material("#111", "ceramic"), V(headR * 0.88, y * headR * 0.35, legL + torsoL + headR * 1.35)));
  if (knight) root.add(mesh(new THREE.CylinderGeometry(headR * 1.15, headR * 1.15, headR * 2.2, 20).rotateX(Math.PI / 2), material("#9ca3af", "metal"), V(0, 0, legL + torsoL + headR * 1.3)));
  const limbs: { p: THREE.Object3D; ph: number; arm: boolean }[] = [];
  const legR = W * 0.11;
  for (const y of [-1, 1]) {
    const hip = new THREE.Group(); hip.position.set(0, y * hipW * 0.45, legL);
    const lg = cylZ(legR * 0.8, legR, legL, 12); lg.translate(0, 0, -legL);
    hip.add(mesh(lg, material(`#${pants.getHexString()}`, kind)));
    hip.add(mesh(new RoundedBoxGeometry(L * 0.5, legR * 2, legR * 1.2, 2, legR * 0.4), material(robot ? "#475569" : "#111827", "leather"), V(L * 0.12, 0, -legL + legR * 0.6)));
    root.add(hip); limbs.push({ p: hip, ph: y > 0 ? 0 : Math.PI, arm: false });
    const sh = new THREE.Group(); sh.position.set(0, y * shoulder * 0.95, legL + torsoL * 0.88);
    const armL = H * (ape ? 0.42 : 0.33);
    const ag = cylZ(legR * 0.55, legR * 0.7, armL, 10); ag.translate(0, 0, -armL);
    sh.add(mesh(ag, material(`#${shirt.getHexString()}`, kind)));
    sh.add(mesh(new THREE.SphereGeometry(legR * 0.62, 10, 8), material(`#${skin.getHexString()}`, robot ? "metal" : "flesh"), V(0, 0, -armL)));
    root.add(sh); limbs.push({ p: sh, ph: y > 0 ? Math.PI : 0, arm: true });
  }
  if (kanga) root.add(mesh(cylZ(W * 0.12, W * 0.03, H * 0.5).rotateY(-2), material(s.color, "flesh"), V(-L * 0.2, 0, legL * 0.4)));
  const stride = legL * 1.4;
  return {
    root,
    update: ({ dist, speed, t, swim }) => {
      if (swim) {
        // front crawl: arms windmill over the head, legs flutter
        const stroke = t * (1.2 + speed * 0.8) * Math.PI;
        for (const l of limbs) l.p.rotation.y = l.arm ? stroke + l.ph : Math.sin(t * 9 + l.ph) * 0.25;
        return;
      }
      const ph = (dist / stride) * Math.PI * 2;
      const amp = Math.min(0.75, speed * 0.45);
      for (const l of limbs) l.p.rotation.y = Math.sin(ph + l.ph) * amp * (l.arm ? 0.8 : 1);
      tm.rotation.y = amp * 0.12;
    },
  };
}

// ---------------------------------------------------------------- fish, whales, subs
function buildFish(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const cetacean = /whale|orca|dolphin|seal/.test(n);
  const machine = /submarine|torpedo/.test(n);
  const jelly = /jellyfish/.test(n), squid = /squid|octopus/.test(n);
  const bodyL = L * (machine ? 0.98 : 0.8);
  const g = ellipsoid(bodyL / 2, W / 2, H / 2, 56);
  const p = g.attributes.position, v = V();
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i);
    const u = v.x / (bodyL / 2);                // -1 tail .. 1 head
    const taper = machine ? (u < -0.85 ? 0.6 + (u + 1) * 2.6 : 1) : u < 0.1 ? Math.max(0.12, 1 - Math.pow((0.1 - u) / 1.1, 1.6)) : 1 - 0.15 * u * u;
    p.setXYZ(i, v.x, v.y * taper, v.z * taper * (cetacean && v.z < 0 ? 0.92 : 1));
  }
  g.computeVertexNormals();
  const base = C(s.color);
  paint(g, (nn, pp) => {
    const belly = pp.z < -H * 0.08;
    if (/orca|killer/.test(n)) {
      const patch = pp.x > bodyL * 0.25 && pp.x < bodyL * 0.36 && pp.z > 0 && Math.abs(pp.y) > W * 0.3;
      return belly || patch ? C("#f8fafc") : C("#0b0b0f");
    }
    if (machine) return base.clone().multiplyScalar(0.9 + 0.1 * fbm(pp.x * 3, pp.y * 3, pp.z * 3));
    const mott = 0.85 + 0.3 * fbm(pp.x / L * 30, pp.y / L * 30, pp.z / L * 30);
    if (/blue whale/.test(n)) return (belly ? C("#94a3b8") : C("#46607e")).multiplyScalar(mott);
    if (/shark/.test(n)) return (belly ? C("#f1f5f9") : C("#64748b")).multiplyScalar(mott);
    if (/dolphin/.test(n)) return (belly ? C("#e2e8f0") : C("#64748b")).multiplyScalar(mott);
    if (/goldfish/.test(n)) return C("#f97316").multiplyScalar(mott);
    if (/salmon|tuna|fish/.test(n)) return (belly ? C("#e5e7eb") : base).multiplyScalar(0.9 + 0.2 * Math.sin(nn.x * 60));
    return (belly ? base.clone().lerp(C("#ffffff"), 0.5) : base).multiplyScalar(mott);
  });
  if (jelly) {
    const bell = mesh(ellipsoid(L / 2, W / 2, H * 0.35, 40), material(s.color, "glass", { transmission: 0.6, emissive: new THREE.Color(s.color), emissiveIntensity: 0.3 }), V(0, 0, H * 0.7));
    root.add(bell);
    for (let i = 0; i < 10; i++) { const a = (i / 10) * Math.PI * 2; root.add(mesh(cylZ(0.004 * L, 0.002 * L, H * 0.6, 4), material(s.color, "glass"), V(Math.cos(a) * L * 0.3, Math.sin(a) * W * 0.3, H * 0.05))); }
    return { root, update: ({ t }) => { bell.scale.set(1 + 0.1 * Math.sin(t * 4), 1 + 0.1 * Math.sin(t * 4), 1 - 0.1 * Math.sin(t * 4)); } };
  }
  const body = mesh(g, vc(machine ? "metal" : "flesh", machine ? {} : { clearcoat: 0.6, clearcoatRoughness: 0.35 }), V(L * 0.08, 0, H / 2));
  root.add(body);
  // tail with fluke (whales: horizontal flukes; fish: vertical fin)
  const tail = new THREE.Group();
  tail.position.set(L * 0.08 - bodyL / 2 + L * 0.02, 0, H / 2);
  root.add(tail);
  const finM = material(cetacean ? (/orca/.test(n) ? "#0b0b0f" : /blue/.test(n) ? "#3c5370" : "#5b6b7e") : s.color, machine ? "metal" : "flesh", { side: THREE.DoubleSide });
  if (machine) {
    const prop = new THREE.Group(); prop.position.x = -L * 0.01; tail.add(prop);
    for (let i = 0; i < 4; i++) { const b = mesh(new THREE.BoxGeometry(0.01 * L, W * 0.06, H * 0.35), material("#b45309", "metal")); b.rotation.x = (i * Math.PI) / 2; prop.add(b); }
    if (/submarine/.test(n)) root.add(mesh(new RoundedBoxGeometry(L * 0.14, W * 0.25, H * 0.45, 3, 0.02), material(s.color, "metal"), V(L * 0.12, 0, H * 1.05)));
    return { root, update: ({ dist, t }) => { prop.rotation.x = dist * 6 + t; } };
  }
  const fl = Math.max(W, H) * (cetacean ? 0.95 : 0.8);
  const fs = new THREE.Shape();
  fs.moveTo(0, 0); fs.quadraticCurveTo(-fl * 0.25, fl * 0.55, -fl * 0.5, fl * 0.62); fs.quadraticCurveTo(-fl * 0.25, fl * 0.15, -fl * 0.35, 0);
  fs.quadraticCurveTo(-fl * 0.25, -fl * 0.15, -fl * 0.5, -fl * 0.62); fs.quadraticCurveTo(-fl * 0.25, -fl * 0.55, 0, 0);
  const fg = new THREE.ShapeGeometry(fs, 16);
  if (!cetacean) fg.rotateX(Math.PI / 2);          // fish: vertical (in x-z)
  tail.add(mesh(fg, finM));
  // fins
  for (const y of [-1, 1]) root.add(mesh(ellipsoid(L * 0.07, W * 0.32, H * 0.03, 16), finM, V(L * 0.2, y * W * 0.5, H * 0.32), new THREE.Euler(y * 0.4, 0.2, 0)));
  if (/shark|orca|dolphin|tuna|salmon|fish/.test(n)) {
    const ds = new THREE.Shape(); ds.moveTo(0, 0); ds.lineTo(-L * 0.06, H * (/orca/.test(n) ? 0.75 : 0.45)); ds.lineTo(-L * 0.12, 0);
    const dg = new THREE.ShapeGeometry(ds); dg.rotateX(Math.PI / 2);
    root.add(mesh(dg, finM, V(L * 0.12, 0, H * 0.92)));
  }
  for (const y of [-1, 1]) root.add(mesh(new THREE.SphereGeometry(Math.max(W, H) * 0.035, 10, 8), material("#050505", "ceramic"), V(L * 0.08 + bodyL * 0.36, y * W * 0.36, H * 0.5)));
  if (squid) for (let i = 0; i < 8; i++) { const a = (i / 8) * Math.PI * 2; tail.add(mesh(cylX(Math.min(W, H) * 0.04, L * 0.35).translate(-L * 0.17, 0, 0), material(s.color, "flesh"), V(0, Math.cos(a) * W * 0.2, Math.sin(a) * H * 0.2))); }
  return {
    root,
    update: ({ t, speed, dist }) => {
      const f = 0.6 + speed / Math.max(0.3, L) * 2.2;
      const a = 0.18 + Math.min(0.35, speed / Math.max(0.5, L) * 0.6);
      const ph = t * f * 2 + dist * 0.5;
      if (cetacean) tail.rotation.y = Math.sin(ph) * a; else tail.rotation.z = Math.sin(ph) * a;
      body.rotation[cetacean ? "y" : "z"] = -Math.sin(ph) * a * 0.12;
    },
  };
}

// ---------------------------------------------------------------- birds, dragons, insects
function buildWinged(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const dragon = /dragon/.test(n), duck = /duck|swan|chicken|penguin/.test(n), insect = /bee|fly|mosquito|butterfly|bat\b/.test(n);
  const base = C(s.color);
  const bodyL = L * (dragon ? 0.42 : duck ? 0.62 : 0.55), bodyH = H * (dragon ? 0.38 : duck ? 0.5 : 0.45), bodyW = W * (dragon ? 0.3 : duck ? 0.75 : 0.35);
  const bodyZ = H * (dragon ? 0.45 : duck ? 0.36 : 0.5);
  const bg = ellipsoid(bodyL / 2, bodyW / 2, bodyH / 2, 40);
  paint(bg, (_, p) => {
    const scale = Math.max(L, H);
    if (dragon) { const sc = Math.abs(Math.sin(p.x / scale * 90) * Math.sin(p.y / scale * 90)); return (p.z < -bodyH * 0.15 ? C("#b5a642") : base).clone().multiplyScalar(0.7 + 0.45 * sc); }
    if (/bee/.test(n)) return Math.sin(p.x / bodyL * 18) > 0 ? C("#111") : C("#facc15");
    return (p.z < -bodyH * 0.2 ? base.clone().lerp(C("#ffffff"), 0.35) : base).clone().multiplyScalar(0.88 + 0.2 * fbm(p.x / scale * 20, p.y / scale * 20, p.z / scale * 20));
  });
  const skin = vc(dragon ? "leather" : duck && !/rubber/.test(n) ? "fabric" : duck ? "rubber" : "fabric");
  root.add(mesh(bg, skin, V(0, 0, bodyZ)));
  // neck + head
  const headR = Math.min(bodyH, bodyW) * (dragon ? 0.38 : duck ? 0.5 : 0.42);
  const head = new THREE.Group();
  const neckLen = dragon ? L * 0.16 : duck ? H * 0.12 : 0;
  head.position.set(bodyL * 0.45 + neckLen * 0.8, 0, bodyZ + bodyH * (dragon ? 0.6 : duck ? 0.62 : 0.2) + neckLen * 0.5);
  if (neckLen) root.add(mesh(cylZ(headR * 0.55, headR * 0.45, neckLen * 1.4, 12).rotateY(0.9), skin, V(bodyL * 0.4, 0, bodyZ + bodyH * 0.25)));
  const hg = ellipsoid(headR * (dragon ? 1.6 : 1.05), headR * 0.9, headR * 0.9, 24);
  paint(hg, () => (/rubber/.test(n) ? C("#facc15") : base.clone()));
  head.add(mesh(hg, skin));
  const beakM = material(dragon ? "#e7e5e4" : duck ? "#f97316" : "#f59e0b", dragon ? "ceramic" : "plastic");
  head.add(mesh(duck ? ellipsoid(headR * 0.7, headR * 0.45, headR * 0.15, 16) : new THREE.ConeGeometry(headR * 0.3, headR * (dragon ? 0.4 : 0.9), 10).rotateZ(-Math.PI / 2), beakM, V(headR * (dragon ? 1.6 : 1.1), 0, -headR * 0.15)));
  for (const y of [-1, 1]) head.add(mesh(new THREE.SphereGeometry(headR * 0.14, 10, 8), material(dragon ? "#facc15" : "#050505", dragon ? "plasma" : "ceramic"), V(headR * 0.6, y * headR * 0.55, headR * 0.35)));
  if (dragon) for (const y of [-1, 1]) head.add(mesh(new THREE.ConeGeometry(headR * 0.15, headR * 1.2, 8).rotateY(-1.9), material("#e7e5e4", "ceramic"), V(-headR * 0.9, y * headR * 0.45, headR * 0.7)));
  root.add(head);
  // wings on shoulder pivots
  const span = (dragon ? 1.0 : insect ? 0.55 : duck ? 0.45 : 0.62) * W;
  const chord = bodyL * (dragon ? 0.9 : 0.55);
  const wings: THREE.Object3D[] = [];
  const wingM = dragon ? material("#2f4f1a", "leather", { side: THREE.DoubleSide, transmission: 0.25, thickness: 0.02, transparent: true })
    : /butterfly/.test(n) ? material("#f97316", "fabric", { side: THREE.DoubleSide }) : insect ? material("#e0f2fe", "glass", { side: THREE.DoubleSide, opacity: 0.5, transparent: true })
      : material(`#${base.clone().multiplyScalar(0.85).getHexString()}`, duck && /rubber/.test(n) ? "rubber" : "fabric", { side: THREE.DoubleSide });
  for (const y of [-1, 1]) {
    const sh = new THREE.Shape();
    if (dragon) { sh.moveTo(0, 0); sh.lineTo(chord * 0.35, span); sh.lineTo(-chord * 0.1, span * 0.78); sh.lineTo(-chord * 0.35, span * 0.82); sh.lineTo(-chord * 0.45, span * 0.5); sh.lineTo(-chord * 0.6, span * 0.45); sh.lineTo(-chord * 0.55, 0); }
    else { sh.moveTo(chord * 0.3, 0); sh.quadraticCurveTo(chord * 0.2, span, -chord * 0.4, span * 0.95); sh.quadraticCurveTo(-chord * 0.3, span * 0.4, -chord * 0.45, 0); }
    const wg = new THREE.ShapeGeometry(sh, 12);
    if (y < 0) wg.scale(1, -1, 1);
    const piv = new THREE.Group();
    piv.position.set(bodyL * 0.05, y * bodyW * 0.35, bodyZ + bodyH * 0.3);
    piv.add(mesh(wg, wingM));
    root.add(piv); wings.push(piv);
  }
  // tail
  const tail = new THREE.Group(); tail.position.set(-bodyL * 0.48, 0, bodyZ); root.add(tail);
  if (dragon) {
    const tl = L * 0.45;
    const tg = cylZ(bodyH * 0.25, bodyH * 0.03, tl, 12); tg.rotateY(-Math.PI / 2 - 0.1);
    paint(tg, () => base.clone().multiplyScalar(0.8));
    tail.add(mesh(tg, vc("leather")));
    tail.add(mesh(new THREE.ConeGeometry(bodyH * 0.12, bodyH * 0.25, 4).rotateZ(Math.PI / 2), material("#7f1d1d", "leather"), V(-tl, 0, -tl * 0.1)));
  } else if (!insect) tail.add(mesh(new THREE.ConeGeometry(bodyH * 0.3, bodyL * 0.3, 8).rotateZ(Math.PI / 2), skin, V(-bodyL * 0.08, 0, bodyH * 0.1)));
  // legs
  const legH = bodyZ - bodyH / 2;
  if (legH > H * 0.05 && !insect) for (const y of [-1, 1]) root.add(mesh(cylZ(bodyW * (dragon ? 0.12 : 0.04), bodyW * (dragon ? 0.1 : 0.03), legH + bodyH * 0.1, 8), material(dragon ? "#365314" : "#f59e0b", dragon ? "leather" : "plastic"), V(0, y * bodyW * 0.22, 0)));
  const freq = dragon ? 1.1 : insect ? 18 : Math.min(8, 2.5 / Math.sqrt(Math.max(0.05, L)));
  return {
    root,
    update: ({ t, speed, propelled }) => {
      const flying = s.animate && (propelled || speed > 0.3);
      const a = flying ? 0.75 : 0.06;
      const r = Math.sin(t * freq * Math.PI * 2) * a + (flying ? 0.15 : -0.9 * (duck ? 1 : 0.6));
      wings[0].rotation.x = -r; wings[1].rotation.x = r;
      tail.rotation.z = Math.sin(t * 1.3) * 0.15;
    },
  };
}

// ---------------------------------------------------------------- vehicles, boats, aircraft
function wheelGeom(r: number, w: number) { const g = new THREE.CylinderGeometry(r, r, w, 28); return g; }
function buildVehicle(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const body = material(s.color, "metal", { metalness: 0.6, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.08 });
  const glass = material("#0f172a", "glass", { transmission: 0.25, roughness: 0.05, metalness: 0.2 });
  const two = /bicycle|bike|motorcycle|scooter/.test(n) && !/kick/.test(n);
  const tank = /tank/.test(n) && !/oil/.test(n);
  const r = two ? H * 0.32 : Math.min(H * 0.24, L * 0.13);
  const wheels: THREE.Object3D[] = [];
  const tyre = material("#111214", "rubber", { roughness: 0.85 });
  const rim = material("#cbd5e1", "steel");
  const addWheel = (x: number, y: number, rr = r, ww = W * 0.16) => {
    const piv = new THREE.Group(); piv.position.set(x, y, rr);
    piv.add(mesh(wheelGeom(rr, ww), tyre));
    piv.add(mesh(wheelGeom(rr * 0.6, ww * 1.02), rim));
    for (let i = 0; i < 5; i++) { const sp = mesh(new THREE.BoxGeometry(rr * 0.12, ww * 1.04, rr * 1.1), rim); sp.rotation.y = (i / 5) * Math.PI; piv.add(sp); }
    root.add(piv); wheels.push(piv);
  };
  if (two) {
    addWheel(L * 0.36, 0, r, W * 0.12); addWheel(-L * 0.36, 0, r, W * 0.12);
    const frame = material(s.color, "metal");
    root.add(mesh(cylX(r * 0.08, L * 0.7), frame, V(0, 0, r * 1.5)));
    root.add(mesh(cylZ(r * 0.07, r * 0.07, r * 1.2).rotateY(0.35), frame, V(L * 0.32, 0, r)));
    root.add(mesh(new RoundedBoxGeometry(L * 0.22, W * 0.25, r * 0.15, 2, 0.01), material("#111", "leather"), V(-L * 0.12, 0, r * 1.95)));
    if (/motor/.test(n)) root.add(mesh(new RoundedBoxGeometry(L * 0.35, W * 0.45, r * 0.8, 3, 0.03), body, V(0, 0, r * 1.15)));
    root.add(mesh(cylX(r * 0.05, W * 0.6).rotateZ(Math.PI / 2), frame, V(L * 0.36, 0, r * 2.2)));
  } else if (tank) {
    for (const y of [-1, 1]) root.add(mesh(new RoundedBoxGeometry(L, W * 0.25, H * 0.35, 3, H * 0.1), material("#27272a", "rubber"), V(0, y * W * 0.38, H * 0.18)));
    root.add(mesh(new RoundedBoxGeometry(L * 0.8, W * 0.55, H * 0.3, 3, 0.05), body, V(0, 0, H * 0.45)));
    root.add(mesh(new THREE.CylinderGeometry(W * 0.25, W * 0.3, H * 0.25, 20).rotateX(Math.PI / 2), body, V(-L * 0.05, 0, H * 0.72)));
    root.add(mesh(cylX(H * 0.05, L * 0.6), body, V(L * 0.35, 0, H * 0.75)));
  } else {
    const small = /cart|wagon|stroller|trolley|wheelchair|skateboard|sled|kart/.test(n);
    const truck = /truck|lorry|fire|garbage|ambulance/.test(n) && !/pickup/.test(n);
    const pickup = /pickup|jeep|tractor/.test(n);
    const bus = /bus|train|tram/.test(n);
    const chassisH = small ? H * 0.35 : H * 0.42;
    const z0 = r * 0.75;
    if (/sled|skateboard/.test(n)) {
      root.add(mesh(new RoundedBoxGeometry(L, W * 0.8, H * 0.25, 3, H * 0.1), material(s.color, "wood"), V(0, 0, H * 0.55)));
      if (/skateboard/.test(n)) for (const x of [-1, 1]) for (const y of [-1, 1]) addWheel(x * L * 0.33, y * W * 0.32, H * 0.22, W * 0.18);
      else for (const y of [-1, 1]) root.add(mesh(cylX(H * 0.05, L), material("#9ca3af", "steel"), V(0, y * W * 0.35, H * 0.08)));
    } else {
      if (truck) {
        root.add(mesh(new RoundedBoxGeometry(L * 0.28, W, H - z0, 4, 0.08), body, V(L * 0.36, 0, z0 + (H - z0) / 2)));
        root.add(mesh(new THREE.BoxGeometry(L * 0.02, W * 0.86, (H - z0) * 0.35), glass, V(L * 0.5, 0, z0 + (H - z0) * 0.7)));
        const cargoC = /fire/.test(n) ? "#dc2626" : /garbage/.test(n) ? "#15803d" : /ambulance/.test(n) ? "#f8fafc" : "#e5e7eb";
        root.add(mesh(new RoundedBoxGeometry(L * 0.7, W, H - z0 * 0.9, 3, 0.04), material(cargoC, "metal", { metalness: 0.3 }), V(-L * 0.14, 0, z0 + (H - z0 * 0.9) / 2)));
        if (/fire/.test(n)) root.add(mesh(cylX(H * 0.04, L * 0.7), material("#cbd5e1", "steel"), V(-L * 0.1, 0, H + 0.05)));
      } else if (bus) {
        root.add(mesh(new RoundedBoxGeometry(L, W, H - z0, 4, 0.15), body, V(0, 0, z0 + (H - z0) / 2)));
        for (const y of [-1, 1]) root.add(mesh(new THREE.BoxGeometry(L * 0.85, 0.01, (H - z0) * 0.3), glass, V(-L * 0.02, y * W * 0.501, z0 + (H - z0) * 0.68)));
        root.add(mesh(new THREE.BoxGeometry(0.01, W * 0.85, (H - z0) * 0.4), glass, V(L * 0.501, 0, z0 + (H - z0) * 0.62)));
      } else {
        root.add(mesh(new RoundedBoxGeometry(L, W, chassisH, 4, Math.min(L, W, chassisH) * 0.3), body, V(0, 0, z0 + chassisH / 2)));
        if (!small) {
          const cabL = pickup ? L * 0.38 : L * 0.52, cabX = pickup ? L * 0.05 : -L * 0.04;
          const cabH = H - z0 - chassisH;
          const cab = new RoundedBoxGeometry(cabL, W * 0.86, cabH * 1.05, 4, Math.min(cabL, cabH) * 0.3);
          root.add(mesh(cab, body, V(cabX, 0, z0 + chassisH + cabH * 0.5 - 0.01)));
          root.add(mesh(new RoundedBoxGeometry(cabL * 0.98, W * 0.875, cabH * 0.6, 3, cabH * 0.2), glass, V(cabX, 0, z0 + chassisH + cabH * 0.5)));
          if (/police/.test(n)) for (const y of [-1, 1]) root.add(mesh(new THREE.BoxGeometry(cabL * 0.1, W * 0.2, cabH * 0.12), material(y > 0 ? "#ef4444" : "#3b82f6", "plasma"), V(cabX, y * W * 0.12, z0 + chassisH + cabH * 1.05)));
          if (/taxi/.test(n)) root.add(mesh(new THREE.BoxGeometry(cabL * 0.2, W * 0.3, cabH * 0.15), material("#fde047", "plasma"), V(cabX, 0, z0 + chassisH + cabH * 1.08)));
        }
      }
      if (!small || /kart/.test(n)) {
        for (const y of [-1, 1]) {
          root.add(mesh(new THREE.SphereGeometry(Math.min(W, H) * 0.06, 12, 8), material("#fefce8", "plasma"), V(L * 0.5, y * W * 0.34, z0 + chassisH * 0.62)));
          root.add(mesh(new THREE.BoxGeometry(0.01, W * 0.15, chassisH * 0.15), material("#ef4444", "plasma"), V(-L * 0.5, y * W * 0.36, z0 + chassisH * 0.62)));
        }
      }
      if (!/sled|skateboard/.test(n)) {
        const axles = bus || truck ? [L * 0.36, -L * 0.3] : [L * 0.32, -L * 0.32];
        for (const x of axles) for (const y of [-1, 1]) addWheel(x, y * (W / 2 - W * 0.08), small ? Math.min(r, H * 0.2) : r);
        if (truck && L > 7) for (const y of [-1, 1]) addWheel(-L * 0.42, y * (W / 2 - W * 0.08), r);
      }
    }
  }
  return {
    root,
    update: ({ dist }) => { for (const w of wheels) w.rotation.y = dist / Math.max(0.01, w.position.z); },
  };
}

function buildBoat(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  const sh = new THREE.Shape();
  sh.moveTo(-L / 2, -W / 2); sh.lineTo(L * 0.3, -W / 2); sh.quadraticCurveTo(L / 2, -W * 0.2, L / 2, 0); sh.quadraticCurveTo(L / 2, W * 0.2, L * 0.3, W / 2); sh.lineTo(-L / 2, W / 2); sh.lineTo(-L / 2, -W / 2);
  const hullH = H * (/canoe|kayak|raft/.test(s.name) ? 0.9 : 0.45);
  const g = new THREE.ExtrudeGeometry(sh, { depth: hullH, bevelEnabled: true, bevelThickness: hullH * 0.15, bevelSize: Math.min(W, hullH) * 0.15, bevelSegments: 3 });
  root.add(mesh(g, material(s.color, /canoe|wood/.test(s.name) ? "wood" : "metal", { metalness: 0.3 })));
  if (!/canoe|kayak|raft/.test(s.name)) {
    root.add(mesh(new RoundedBoxGeometry(L * 0.35, W * 0.7, H * 0.35, 3, 0.05), material("#f8fafc", "metal", { metalness: 0.2 }), V(-L * 0.05, 0, hullH + H * 0.18)));
    root.add(mesh(new THREE.BoxGeometry(L * 0.36, W * 0.71, H * 0.08), material("#0f172a", "glass"), V(-L * 0.05, 0, hullH + H * 0.25)));
    if (/titanic|cruise|ship/.test(s.name)) for (let i = 0; i < 4; i++) root.add(mesh(cylZ(W * 0.07, W * 0.07, H * 0.45), material(i % 2 ? "#111" : "#f59e0b", "metal"), V(-L * 0.25 + i * L * 0.15, 0, hullH + H * 0.3)));
  }
  return { root, update: () => {} };
}

function buildAircraft(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const skin = material(s.color, "metal", { metalness: 0.7, roughness: 0.25, clearcoat: 1 });
  let rotor: THREE.Object3D | null = null, flame: THREE.Mesh | null = null;
  if (/ufo/.test(n)) {
    root.add(mesh(ellipsoid(L / 2, W / 2, H * 0.22, 48), skin, V(0, 0, H * 0.35)));
    root.add(mesh(new THREE.SphereGeometry(L * 0.18, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2).rotateX(Math.PI / 2), material("#a5f3fc", "glass"), V(0, 0, H * 0.48)));
    rotor = new THREE.Group(); root.add(rotor);
    for (let i = 0; i < 12; i++) { const a = (i / 12) * Math.PI * 2; rotor.add(mesh(new THREE.SphereGeometry(L * 0.025, 8, 6), material(i % 2 ? "#22d3ee" : "#a3e635", "plasma"), V(Math.cos(a) * L * 0.44, Math.sin(a) * W * 0.44, H * 0.33))); }
    return { root, update: ({ t }) => { rotor!.rotation.z = t * 2; } };
  }
  const vertical = /rocket|shuttle/.test(n) && H > L;
  const len = vertical ? H : L, rad = (vertical ? Math.min(L, W) : Math.min(W, H)) * (/rocket|torpedo|missile/.test(n) ? 0.5 : 0.18);
  const fus = new THREE.Group();
  fus.add(mesh(new THREE.CapsuleGeometry(rad, len - rad * 2, 8, 24).rotateZ(Math.PI / 2), skin));
  if (/rocket|missile|shuttle/.test(n)) {
    fus.add(mesh(new THREE.ConeGeometry(rad, rad * 3, 24).rotateZ(-Math.PI / 2), material("#dc2626", "metal"), V(len / 2 - rad * 0.2, 0, 0)));
    for (let i = 0; i < 4; i++) { const f = mesh(new THREE.BoxGeometry(len * 0.15, 0.01 + rad * 0.05, rad * 1.6), material("#dc2626", "metal"), V(-len / 2 + len * 0.08, 0, 0)); f.rotation.x = (i * Math.PI) / 2; f.translateZ(rad); fus.add(f); }
    flame = mesh(new THREE.ConeGeometry(rad * 0.8, rad * 5, 20).rotateZ(Math.PI / 2), new THREE.MeshBasicMaterial({ color: "#fb923c", transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }), V(-len / 2 - rad * 2.5, 0, 0));
    fus.add(flame);
  } else {
    const span = W * (/helicopter/.test(n) ? 0 : 1);
    if (span) {
      const wing = new THREE.Shape(); wing.moveTo(L * 0.1, 0); wing.lineTo(-L * 0.05, span / 2); wing.lineTo(-L * 0.15, span / 2); wing.lineTo(-L * 0.12, 0); wing.lineTo(-L * 0.15, -span / 2); wing.lineTo(-L * 0.05, -span / 2);
      fus.add(mesh(new THREE.ExtrudeGeometry(wing, { depth: rad * 0.12, bevelEnabled: false }).translate(0, 0, -rad * 0.06), skin));
      const tail = new THREE.Shape(); tail.moveTo(0, 0); tail.lineTo(-L * 0.12, H * 0.45); tail.lineTo(-L * 0.17, H * 0.45); tail.lineTo(-L * 0.15, 0);
      const tg = new THREE.ExtrudeGeometry(tail, { depth: rad * 0.08, bevelEnabled: false }); tg.rotateX(Math.PI / 2);
      fus.add(mesh(tg, skin, V(-L * 0.32, rad * 0.04, rad * 0.5)));
      fus.add(mesh(new THREE.BoxGeometry(L * 0.08, span * 0.35, rad * 0.08), skin, V(-L * 0.44, 0, rad * 0.4)));
      fus.add(mesh(new THREE.SphereGeometry(rad * 0.9, 20, 12, 0, Math.PI * 2, 0, Math.PI / 2.4).rotateZ(-Math.PI / 2), material("#0f172a", "glass"), V(L * 0.36, 0, rad * 0.35)));
    } else {
      fus.add(mesh(cylX(rad * 0.25, L * 0.5), skin, V(-L * 0.4, 0, rad * 0.3)));
      fus.add(mesh(new THREE.SphereGeometry(rad * 1.05, 24, 16), material("#0f172a", "glass"), V(L * 0.25, 0, rad * 0.1)));
      rotor = new THREE.Group(); rotor.position.set(0, 0, rad * 1.4);
      for (let i = 0; i < 4; i++) { const b = mesh(new THREE.BoxGeometry(L * 0.48, W * 0.04, 0.02), material("#1f2937", "metal")); b.rotation.z = (i * Math.PI) / 2; b.translateX(L * 0.24); rotor.add(b); }
      fus.add(rotor);
    }
  }
  if (vertical) fus.rotation.y = -Math.PI / 2;
  fus.position.z = vertical ? H / 2 : H / 2;
  root.add(fus);
  return {
    root,
    update: ({ t, propelled, speed }) => {
      if (rotor) rotor.rotation.z = t * 40;
      if (flame) { const on = propelled || speed > 2; flame.visible = on; flame.scale.set(1 + 0.25 * Math.sin(t * 60), 1, 1); }
    },
  };
}

// ---------------------------------------------------------------- buildings, castles, walls, trees
function windowsTexture(cols: number, rows: number, glass: string, wall: string, lit = 0.25) {
  return canvasTex(512, 512, (c) => {
    c.fillStyle = wall; c.fillRect(0, 0, 512, 512);
    const cw = 512 / cols, rh = 512 / rows;
    for (let i = 0; i < cols; i++) for (let j = 0; j < rows; j++) {
      c.fillStyle = Math.random() < lit ? "#fde68a" : glass;
      c.fillRect(i * cw + cw * 0.18, j * rh + rh * 0.2, cw * 0.64, rh * 0.55);
    }
  });
}
function stoneTexture(base = "#a8a29e", brick = false) {
  return canvasTex(512, 512, (c) => {
    c.fillStyle = base; c.fillRect(0, 0, 512, 512);
    const bh = brick ? 32 : 48, bw = brick ? 96 : 110;
    for (let y = 0; y < 512; y += bh) for (let x = -bw; x < 512; x += bw) {
      const off = (y / bh) % 2 ? bw / 2 : 0;
      const k = 0.8 + Math.random() * 0.35;
      const col = new THREE.Color(base).multiplyScalar(k);
      c.fillStyle = `#${col.getHexString()}`;
      c.fillRect(x + off + 3, y + 3, bw - 6, bh - 6);
    }
  }, [2, 2]);
}
function buildBuilding(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  if (/house|hut|barn/.test(n)) {
    const wallH = H * 0.6;
    const tex = canvasTex(256, 256, (c) => { c.fillStyle = s.color; c.fillRect(0, 0, 256, 256); c.fillStyle = "#1e3a5f"; for (const x of [40, 170]) c.fillRect(x, 70, 50, 60); c.fillStyle = "#7c2d12"; c.fillRect(105, 140, 46, 116); });
    root.add(mesh(new THREE.BoxGeometry(L, W, wallH).translate(0, 0, wallH / 2), new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.8 })));
    const roof = new THREE.Shape(); roof.moveTo(-W / 2 - 0.3, 0); roof.lineTo(0, H - wallH); roof.lineTo(W / 2 + 0.3, 0);
    const rg2 = new THREE.ExtrudeGeometry(roof, { depth: L + 0.6, bevelEnabled: false }); rg2.rotateX(Math.PI / 2); rg2.rotateZ(Math.PI / 2); rg2.translate(-L / 2 - 0.3, 0, wallH);
    root.add(mesh(rg2, material("#7f1d1d", "ceramic", { roughness: 0.7 })));
    root.add(mesh(new THREE.BoxGeometry(0.6, 0.6, H * 0.35), material("#7c2d12", "stone"), V(-L * 0.25, W * 0.2, wallH + H * 0.2)));
    return { root, update: () => {} };
  }
  const glassy = /skyscraper|tower|building/.test(n);
  const tex = /pyramid/.test(n) ? stoneTexture("#d6b98c") : windowsTexture(Math.max(3, Math.round(L / 3)), Math.max(3, Math.round(H / 3.5)), glassy ? "#1e3a5f" : "#334155", glassy ? "#94a3b8" : s.color, 0.2);
  if (/pyramid/.test(n)) {
    const g = new THREE.ConeGeometry(L / Math.SQRT2, H, 4); g.rotateX(Math.PI / 2); g.rotateZ(Math.PI / 4); g.translate(0, 0, H / 2);
    root.add(mesh(g, new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.95 })));
    return { root, update: () => {} };
  }
  if (/lighthouse/.test(n)) {
    const g = cylZ(L / 2, L * 0.32, H * 0.85, 32);
    paint(g, (_, p) => (Math.floor(p.z / (H * 0.85) * 6) % 2 ? C("#dc2626") : C("#f8fafc")));
    root.add(mesh(g, vc("stone")));
    root.add(mesh(cylZ(L * 0.3, L * 0.3, H * 0.1, 24), material("#fef9c3", "plasma"), V(0, 0, H * 0.85)));
    root.add(mesh(new THREE.ConeGeometry(L * 0.36, H * 0.06, 24).rotateX(Math.PI / 2), material("#111", "metal"), V(0, 0, H * 0.97)));
    return { root, update: () => {} };
  }
  const m = new THREE.MeshPhysicalMaterial({ map: tex, roughness: glassy ? 0.15 : 0.8, metalness: glassy ? 0.5 : 0, clearcoat: glassy ? 1 : 0 });
  root.add(mesh(new THREE.BoxGeometry(L, W, H).translate(0, 0, H / 2), m));
  root.add(mesh(new THREE.BoxGeometry(L * 1.02, W * 1.02, H * 0.015), material("#475569", "metal"), V(0, 0, H)));
  return { root, update: () => {} };
}
function crenellate(root: THREE.Group, x0: number, y0: number, x1: number, y1: number, z: number, t: number, m: THREE.Material) {
  const len = Math.hypot(x1 - x0, y1 - y0), n = Math.max(2, Math.floor(len / (t * 2)));
  const ang = Math.atan2(y1 - y0, x1 - x0);
  for (let i = 0; i <= n; i += 2) {
    const k = i / n;
    const b = mesh(new THREE.BoxGeometry(t, t * 0.9, t), m, V(x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, z + t / 2));
    b.rotation.z = ang; root.add(b);
  }
}
function buildCastle(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  const stone = new THREE.MeshPhysicalMaterial({ map: stoneTexture("#a8a29e"), roughness: 0.95 });
  const tR = Math.min(L, W) * 0.11, wallH = H * 0.5, wallT = Math.min(L, W) * 0.05, t = wallT * 0.9;
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => [a * (L / 2 - tR), b * (W / 2 - tR)]);
  for (let i = 0; i < 4; i++) {
    const [x0, y0] = corners[i], [x1, y1] = corners[(i + 1) % 4];
    const len = Math.hypot(x1 - x0, y1 - y0), ang = Math.atan2(y1 - y0, x1 - x0);
    const w = mesh(new THREE.BoxGeometry(len, wallT, wallH).translate(0, 0, wallH / 2), stone, V((x0 + x1) / 2, (y0 + y1) / 2, 0));
    w.rotation.z = ang; root.add(w);
    crenellate(root, x0, y0, x1, y1, wallH, t, stone);
    root.add(mesh(cylZ(tR, tR * 0.92, H * 0.72, 24), stone, V(x0, y0, 0)));
    root.add(mesh(new THREE.ConeGeometry(tR * 1.25, H * 0.28, 24).rotateX(Math.PI / 2), material("#1e3a8a", "ceramic", { roughness: 0.5 }), V(x0, y0, H * 0.72 + H * 0.14)));
  }
  const kL = L * 0.36, kW = W * 0.36, kH = H * 0.78;
  root.add(mesh(new THREE.BoxGeometry(kL, kW, kH).translate(0, 0, kH / 2), stone));
  crenellate(root, -kL / 2, -kW / 2, kL / 2, -kW / 2, kH, t, stone); crenellate(root, -kL / 2, kW / 2, kL / 2, kW / 2, kH, t, stone);
  root.add(mesh(new THREE.BoxGeometry(L * 0.1, wallT * 1.2, wallH * 0.55).translate(0, 0, wallH * 0.275), material("#3f2a1d", "wood"), V(L / 2 - tR * 0.2, 0, 0)).rotateZ(Math.PI / 2));
  const flag = mesh(new THREE.PlaneGeometry(tR * 1.6, tR), material("#dc2626", "fabric", { side: THREE.DoubleSide }), V(tR * 0.8, 0, kH + H * 0.15));
  flag.rotation.x = Math.PI / 2; root.add(flag);
  root.add(mesh(cylZ(tR * 0.04, tR * 0.04, H * 0.22, 6), material("#111", "metal"), V(0, 0, kH)));
  return { root, update: ({ t: tt }) => { flag.rotation.y = Math.sin(tt * 3) * 0.2; } };
}
function buildWall(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  if (/goal/.test(n)) {
    const post = material("#f8fafc", "metal");
    for (const y of [-1, 1]) root.add(mesh(cylZ(0.06, 0.06, H), post, V(0, y * L / 2, 0)));
    root.add(mesh(cylX(0.06, L).rotateZ(Math.PI / 2), post, V(0, 0, H)));
    const net = mesh(new THREE.PlaneGeometry(L, H, 20, 10), new THREE.MeshBasicMaterial({ color: "#e5e7eb", wireframe: true, transparent: true, opacity: 0.6 }), V(-W * 0.6, 0, H / 2));
    net.rotation.set(Math.PI / 2, Math.PI / 2, 0); root.add(net);
    return { root, update: () => {} };
  }
  if (/fence/.test(n)) {
    const wood = material("#a16207", "wood");
    for (let i = 0; i <= 6; i++) root.add(mesh(new THREE.BoxGeometry(0.08, W, H).translate(0, 0, H / 2), wood, V(-L / 2 + (i / 6) * L, 0, 0)));
    for (const z of [0.3, 0.75]) root.add(mesh(new THREE.BoxGeometry(L, W * 0.6, 0.08), wood, V(0, 0, H * z)));
    return { root, update: () => {} };
  }
  const tex = stoneTexture(/brick/.test(n) ? "#a0522d" : s.color === "#94a3b8" ? "#b0a99f" : s.color, true);
  root.add(mesh(new THREE.BoxGeometry(L, W, H).translate(0, 0, H / 2), new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.9 })));
  return { root, update: () => {} };
}
function buildTree(s: ThingSpec): BuiltThing {
  const [L, , H] = s.dims;
  const root = new THREE.Group();
  const trunkH = H * 0.45;
  const tg = cylZ(L * 0.5, L * 0.32, trunkH + H * 0.15, 14);
  paint(tg, (_, p) => C("#5b3a1e").multiplyScalar(0.75 + 0.4 * fbm(p.x * 20, p.y * 20, p.z * 3)));
  root.add(mesh(tg, vc("wood")));
  const fol = vc("fabric");
  const R = Math.max(L * 3, H * 0.28);
  for (let i = 0; i < 4; i++) {
    const g = new THREE.IcosahedronGeometry(R * (1 - i * 0.18), 2);
    const p = g.attributes.position, v = V();
    for (let k = 0; k < p.count; k++) { v.fromBufferAttribute(p, k); v.multiplyScalar(0.85 + 0.3 * fbm(v.x * 2 + i, v.y * 2, v.z * 2)); p.setXYZ(k, v.x, v.y, v.z); }
    g.computeVertexNormals();
    paint(g, (n) => C("#2f6b2a").lerp(C("#5c9e3a"), Math.max(0, n.z) * 0.6 + 0.3 * fbm(n.x * 5, n.y * 5, n.z * 5)));
    root.add(mesh(g, fol, V((hash(i) - 0.5) * R * 0.4, (hash(i + 9) - 0.5) * R * 0.4, trunkH + R * 0.6 + i * R * 0.45)));
  }
  return { root, update: ({ t }) => { root.rotation.x = Math.sin(t * 0.8) * 0.01; } };
}
function buildSnake(s: ThingSpec): BuiltThing {
  const [L, W] = s.dims;
  const root = new THREE.Group();
  const N = 40, r = W * 0.5;
  const pts = Array.from({ length: N }, (_, i) => V((i / (N - 1) - 0.5) * L * 0.85, 0, r));
  const curve = new THREE.CatmullRomCurve3(pts);
  const m = vc("leather");
  let body = mesh(new THREE.TubeGeometry(curve, 80, r, 12), m);
  root.add(body);
  const head = mesh(ellipsoid(r * 2, r * 1.4, r * 1.1, 16), material(s.color, "leather"), V(L * 0.43, 0, r));
  root.add(head);
  return {
    root,
    update: ({ dist, t }) => {
      for (let i = 0; i < N; i++) pts[i].y = Math.sin(i * 0.45 - dist * 6 - t) * L * 0.06;
      const g = new THREE.TubeGeometry(curve, 80, r, 12);
      paint(g, (_, p) => (Math.sin(p.x / L * 60) > 0.3 ? C("#1a2e05") : C(s.color)));
      body.geometry.dispose(); body.geometry = g;
      head.position.y = pts[N - 1].y;
    },
  };
}

// ---------------------------------------------------------------- support frame (pendulums, springs hang from it)
function buildSupport(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  const steel = material("#9aa3ad", "steel");
  root.add(mesh(new THREE.BoxGeometry(L * 1.6, W, H).translate(0, 0, H / 2), steel));
  const post = s.postHeight ?? 1;
  for (const x of [-1, 1]) {
    root.add(mesh(new THREE.BoxGeometry(W * 0.9, W * 0.9, post + H).translate(0, 0, -post / 2 + H / 2), steel, V(x * L * 0.78, 0, 0)));
    root.add(mesh(new THREE.BoxGeometry(L * 0.5, L * 0.5, 0.03), material("#4b5563", "metal"), V(x * L * 0.78, 0, -post + 0.015)));
  }
  root.add(mesh(new THREE.TorusGeometry(W * 0.35, W * 0.08, 8, 20), steel, V(0, 0, -W * 0.3)).rotateX(Math.PI / 2));
  return { root, update: () => {} };
}

// ---------------------------------------------------------------- everyday objects by shape (+ a few named ones)
function buildGeneric(s: ThingSpec): BuiltThing {
  const [L, W, H] = s.dims;
  const n = s.name.toLowerCase();
  const root = new THREE.Group();
  const kind = s.material === "unknown" ? "plastic" : s.material;
  const m = material(s.color, kind);
  if (/chair|stool/.test(n) && !/wheel/.test(n)) {
    const wood = material(s.color === "#94a3b8" ? "#8b5a2b" : s.color, "wood");
    const seatZ = H * 0.5;
    root.add(mesh(new RoundedBoxGeometry(L, W, H * 0.05, 2, 0.01), wood, V(0, 0, seatZ)));
    for (const x of [-1, 1]) for (const y of [-1, 1]) root.add(mesh(new THREE.BoxGeometry(L * 0.08, W * 0.08, seatZ).translate(0, 0, seatZ / 2), wood, V(x * L * 0.42, y * W * 0.42, 0)));
    if (!/stool/.test(n)) {
      root.add(mesh(new RoundedBoxGeometry(L * 0.08, W, H * 0.45, 2, 0.01), wood, V(-L * 0.46, 0, seatZ + H * 0.25)));
    }
    return { root, update: () => {} };
  }
  if (/table|desk/.test(n)) {
    const wood = material(s.color === "#94a3b8" ? "#a16207" : s.color, "wood");
    root.add(mesh(new RoundedBoxGeometry(L, W, H * 0.06, 2, 0.01), wood, V(0, 0, H * 0.97)));
    for (const x of [-1, 1]) for (const y of [-1, 1]) root.add(mesh(new THREE.BoxGeometry(L * 0.05, W * 0.05, H * 0.94).translate(0, 0, H * 0.47), wood, V(x * L * 0.45, y * W * 0.42, 0)));
    return { root, update: () => {} };
  }
  if (/bottle|vase|flask|beaker|test tube/.test(n)) {
    const r = L / 2, pts = [V(0, 0), V(r * 0.95, 0), V(r, H * 0.05), V(r, H * 0.6), V(r * 0.4, H * 0.78), V(r * 0.33, H * 0.98), V(0, H)].map((p) => new THREE.Vector2(p.x, p.y));
    const g = new THREE.LatheGeometry(pts, 40); g.rotateX(Math.PI / 2);
    root.add(mesh(g, /glass|beaker|flask|tube/.test(n) || s.material === "glass" ? material("#d8f3ff", "glass") : material(s.color, kind)));
    return { root, update: () => {} };
  }
  if (/mug|cup|bucket|bowl|glass\b/.test(n)) {
    const r = L / 2;
    const g = new THREE.LatheGeometry([new THREE.Vector2(0, 0), new THREE.Vector2(r * 0.9, 0), new THREE.Vector2(r, H), new THREE.Vector2(r * 0.9, H), new THREE.Vector2(r * 0.82, H * 0.08), new THREE.Vector2(0, H * 0.08)], 40);
    g.rotateX(Math.PI / 2);
    root.add(mesh(g, /glass/.test(n) ? material("#d8f3ff", "glass") : material(s.color, kind === "plastic" ? "ceramic" : kind)));
    if (/mug/.test(n)) root.add(mesh(new THREE.TorusGeometry(H * 0.25, r * 0.12, 10, 20, Math.PI).rotateZ(-Math.PI / 2).rotateX(Math.PI / 2), material(s.color, "ceramic"), V(r, 0, H * 0.5)));
    return { root, update: () => {} };
  }
  if (/book|notebook/.test(n)) {
    root.add(mesh(new RoundedBoxGeometry(L, W, H, 2, Math.min(H, 0.005)), material(s.color, "paper"), V(0, 0, H / 2)));
    root.add(mesh(new THREE.BoxGeometry(L * 0.96, W * 0.97, H * 0.85), material("#fefce8", "paper"), V(L * 0.02, 0, H / 2)));
    return { root, update: () => {} };
  }
  if (/feather/.test(n)) {
    const sh = new THREE.Shape(); sh.moveTo(-L / 2, 0); sh.quadraticCurveTo(0, W * 0.6, L / 2, 0); sh.quadraticCurveTo(0, -W * 0.6, -L / 2, 0);
    root.add(mesh(new THREE.ShapeGeometry(sh, 16), material("#f8fafc", "fabric", { side: THREE.DoubleSide }), V(0, 0, H / 2)));
    root.add(mesh(cylX(Math.max(0.0008, W * 0.03), L * 1.1), material("#e7e5e4", "plastic"), V(0, 0, H / 2)));
    return { root, update: ({ t }) => { root.rotation.z = Math.sin(t * 2.3) * 0.4; } };
  }
  if (/phone|tablet|laptop|remote|tv|television|monitor/.test(n)) {
    root.add(mesh(new RoundedBoxGeometry(L, W, H, 3, Math.min(L, W, H) * 0.3), material("#1f2937", "metal"), V(0, 0, H / 2)));
    root.add(mesh(new THREE.PlaneGeometry(L * 0.92, W * 0.9), material("#0b1220", "glass", { emissive: new THREE.Color("#1e3a8a"), emissiveIntensity: 0.4 }), V(0, 0, H + 0.0005)));
    return { root, update: () => {} };
  }
  if (/crate|box|cardboard|package/.test(n)) {
    root.add(mesh(new RoundedBoxGeometry(L, W, H, 2, Math.min(L, W, H) * 0.05), material(/cardboard|box|package/.test(n) ? "#c69c6d" : "#a16207", /cardboard|box/.test(n) ? "paper" : "wood"), V(0, 0, H / 2)));
    return { root, update: () => {} };
  }
  if (/brick|block|cube|dice|die|domino|jenga/.test(n)) {
    const tex = /brick/.test(n) ? stoneTexture("#a0522d", true) : null;
    root.add(mesh(new RoundedBoxGeometry(L, W, H, 2, Math.min(L, W, H) * 0.08), tex ? new THREE.MeshPhysicalMaterial({ map: tex, roughness: 0.9 }) : m, V(0, 0, H / 2)));
    if (/dice|die/.test(n)) root.add(mesh(new THREE.SphereGeometry(L * 0.09, 8, 6), material("#111", "plastic"), V(0, 0, H)));
    return { root, update: () => {} };
  }
  if (/coin/.test(n)) { root.add(mesh(cylZ(L / 2, L / 2, H, 40), material("#d4af37", "metal"))); return { root, update: () => {} }; }
  if (/pillow|cushion|bag|backpack|teddy|sofa|couch/.test(n)) {
    root.add(mesh(new RoundedBoxGeometry(L, W, H, 4, Math.min(L, W, H) * 0.45), material(s.color, "fabric"), V(0, 0, H / 2)));
    return { root, update: () => {} };
  }
  // by physical shape
  let g: THREE.BufferGeometry;
  if (s.dims && /sphere|ellipsoid/.test(s.archetype)) g = ellipsoid(L / 2, W / 2, H / 2).translate(0, 0, H / 2);
  else if (/capsule/.test(n)) g = new THREE.CapsuleGeometry(Math.min(W, H) / 2, Math.max(0.001, L - Math.min(W, H)), 8, 20).rotateZ(Math.PI / 2).translate(0, 0, H / 2);
  else g = new RoundedBoxGeometry(L, W, H, 3, Math.min(L, W, H) * 0.12).translate(0, 0, H / 2);
  paint(g, (_, p) => C(s.color).multiplyScalar(0.9 + 0.15 * fbm(p.x * 30, p.y * 30, p.z * 30)));
  root.add(mesh(g, vc(kind)));
  return { root, update: () => {} };
}

/** Shape-only stand-in built from the physics shape (used for "generic" things the
 *  knowledge base calls an ellipsoid/cylinder/capsule rather than a box). */
function byShape(s: ThingSpec, shape: string): BuiltThing {
  const [L, W, H] = s.dims;
  const root = new THREE.Group();
  let g: THREE.BufferGeometry;
  if (shape === "ellipsoid" || shape === "sphere") g = ellipsoid(L / 2, W / 2, H / 2, 48).translate(0, 0, H / 2);
  else if (shape === "cylinder") g = cylZ(L / 2, L / 2, H, 40);
  else if (shape === "capsule") g = new THREE.CapsuleGeometry(Math.min(W, H) / 2, Math.max(0.001, L - Math.min(W, H)), 8, 20).rotateZ(Math.PI / 2).translate(0, 0, H / 2);
  else return buildGeneric(s);
  paint(g, (_, p) => C(s.color).multiplyScalar(0.88 + 0.18 * fbm(p.x * 20, p.y * 20, p.z * 20)));
  root.add(mesh(g, vc(s.material === "unknown" ? "plastic" : s.material)));
  return { root, update: () => {} };
}

export function buildThing(s: ThingSpec, physShape = "box"): BuiltThing {
  const n = s.name.toLowerCase();
  switch (s.archetype) {
    case "ball": return buildBall(s);
    case "rock": return buildRock(s);
    case "quadruped": return buildQuadruped(s);
    case "biped": return buildBiped(s);
    case "fish": return buildFish(s);
    case "winged": return buildWinged(s);
    case "vehicle": return buildVehicle(s);
    case "boat": return buildBoat(s);
    case "aircraft": return buildAircraft(s);
    case "building": return buildBuilding(s);
    case "castle": return buildCastle(s);
    case "wall": return buildWall(s);
    case "tree": return buildTree(s);
    case "snake": return buildSnake(s);
    case "anchor": return buildSupport(s);
  }
  if (/chair|table|desk|bottle|mug|cup|book|feather|phone|laptop|crate|box|brick|block|cube|dice|domino|coin|pillow|bag|teddy|sofa|bucket|bowl|glass|vase|flask|beaker/.test(n)) return buildGeneric(s);
  return byShape(s, physShape);
}
