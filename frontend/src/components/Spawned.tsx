import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { clock } from "../clock";
import { distanceAt, poseAt, velocityAt } from "../poses";
import type { Asset, Sim } from "../types";
import { buildThing, type BuiltThing } from "../three/things";

/* Summoned objects: real glTF models (PBR) driven by the physics poses.
 * Vehicles get their wheels cut out of the body mesh at load time so the
 * simulated wheel spin and steering can be shown. */

const gltfCache = new Map<string, Promise<THREE.Group>>();
function loadGltf(url: string): Promise<THREE.Group> {
  if (!gltfCache.has(url)) {
    gltfCache.set(url, new GLTFLoader().loadAsync(url).then((g) => g.scene));
  }
  return gltfCache.get(url)!;
}

export interface WheelRig { pivots: THREE.Object3D[] }      // order: fl, fr, rl, rr

/** A library asset as a ready-to-place object (body frame), with its wheel rig if it's a vehicle. */
export async function loadAssetModel(asset: Asset): Promise<{ obj: THREE.Object3D; rig: WheelRig | null }> {
  if (asset.kind === "procedural") return { obj: proceduralObject(asset), rig: null };
  const scene = await loadGltf(asset.url!);
  const model = scene.clone(true);
  if (asset.nodes) {
    model.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) {
        let keep = false;
        for (let p: THREE.Object3D | null = o; p; p = p.parent) if (asset.nodes!.includes(p.name)) keep = true;
        o.visible = keep;
      }
    });
  }
  model.traverse((o) => {
    const m = o as THREE.Mesh;
    if (m.isMesh) { m.castShadow = true; m.receiveShadow = true; m.material = (m.material as THREE.Material).clone(); m.geometry = m.geometry.clone(); }
  });
  const root = new THREE.Group();
  root.matrixAutoUpdate = false;
  root.matrix.fromArray(asset.gltf_to_body!.flat()).transpose();
  root.add(model);
  const rig = asset.kind === "vehicle" ? rigWheels(model, asset) : null;
  const holder = new THREE.Group(); holder.add(root);
  return { obj: holder, rig };
}

/** Split wheel triangles out of the car's meshes into pivoting sub-meshes. */
function rigWheels(root: THREE.Object3D, a: Asset): WheelRig | null {
  const w = a.wheels;
  if (!w) return null;
  root.updateMatrixWorld(true);
  const rootInv = new THREE.Matrix4().copy(root.matrixWorld).invert();
  const pivots = w.centres_gltf.map((c) => {
    const p = new THREE.Object3D();
    p.position.set(c[0], c[1], c[2]);
    p.rotation.order = "YXZ";
    root.add(p);
    return p;
  });
  const R = w.radius_gltf * 1.12, H = w.width_gltf * 0.75;
  const meshes: THREE.Mesh[] = [];
  root.traverse((o) => { if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh); });
  for (const mesh of meshes) {
    const g = mesh.geometry as THREE.BufferGeometry;
    const toRoot = new THREE.Matrix4().multiplyMatrices(rootInv, mesh.matrixWorld);
    const pos = g.attributes.position;
    const v = new THREE.Vector3();
    const which = new Int8Array(pos.count).fill(-1);
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(toRoot);
      w.centres_gltf.forEach((c, k) => {
        if (Math.abs(v.x - c[0]) < H && (v.y - c[1]) ** 2 + (v.z - c[2]) ** 2 < R * R) which[i] = k;
      });
    }
    const index = g.index ? Array.from(g.index.array) : [...Array(pos.count).keys()];
    const keep: number[] = [];
    const parts: number[][] = [[], [], [], []];
    for (let t = 0; t < index.length; t += 3) {
      const [i0, i1, i2] = [index[t], index[t + 1], index[t + 2]];
      const k = which[i0];
      if (k >= 0 && which[i1] === k && which[i2] === k) parts[k].push(i0, i1, i2);
      else keep.push(i0, i1, i2);
    }
    if (parts.every((p) => p.length === 0)) continue;
    const body = g.clone();
    body.setIndex(keep);
    mesh.geometry = body;
    parts.forEach((tri, k) => {
      if (!tri.length) return;
      const wg = g.clone();
      wg.setIndex(tri);
      // express wheel geometry relative to its pivot (wheel centre) in root space
      wg.applyMatrix4(toRoot);
      const c = w.centres_gltf[k];
      wg.translate(-c[0], -c[1], -c[2]);
      const wm = new THREE.Mesh(wg, mesh.material);
      wm.castShadow = true;
      pivots[k].add(wm);
    });
  }
  return { pivots };
}

function proceduralObject(a: Asset): THREE.Object3D {
  const [dx, dy, dz] = a.dims;
  let geom: THREE.BufferGeometry;
  if (a.shape === "sphere") {
    geom = new THREE.SphereGeometry(dx / 2, 48, 32);
    geom.translate(0, 0, dx / 2);
  } else if (a.shape === "wedge") {
    const s = new THREE.Shape();
    s.moveTo(-dx / 2, 0); s.lineTo(dx / 2, 0); s.lineTo(-dx / 2, dz); s.lineTo(-dx / 2, 0);
    geom = new THREE.ExtrudeGeometry(s, { depth: dy, bevelEnabled: false });
    geom.rotateX(Math.PI / 2);
    geom.translate(0, dy / 2, 0);
  } else {
    geom = new THREE.BoxGeometry(dx, dy, dz);
    geom.translate(0, 0, dz / 2);
  }
  const mat = {
    glass: new THREE.MeshPhysicalMaterial({ color: "#d8f3ff", transmission: 1, thickness: 0.02, roughness: 0.03, ior: 1.5, clearcoat: 1 }),
    steel: new THREE.MeshPhysicalMaterial({ color: "#c9ccd1", metalness: 1, roughness: 0.18, clearcoat: 0.6 }),
    wood: new THREE.MeshPhysicalMaterial({ color: "#b07a45", roughness: 0.7, sheen: 0.2 }),
    domino: new THREE.MeshPhysicalMaterial({ color: "#f4efe3", roughness: 0.35, clearcoat: 0.8, clearcoatRoughness: 0.2 }),
  }[a.material ?? "wood"] ?? new THREE.MeshStandardMaterial({ color: "#999" });
  const m = new THREE.Mesh(geom, mat);
  m.castShadow = true;
  m.receiveShadow = true;
  return m;
}

function SpawnedOne({ sid, asset, sim }: { sid: string; asset: Asset | null; sim: Sim }) {
  const holder = useRef<THREE.Group>(null);
  const [obj, setObj] = useState<THREE.Object3D | null>(null);
  const rig = useRef<WheelRig | null>(null);
  const built = useRef<BuiltThing | null>(null);
  const idx = sim.object_ids.indexOf(sid);
  const spawnT = sim.spawned[sid].t;

  useEffect(() => {
    let alive = true;
    const sp = sim.spawned[sid] as Sim["spawned"][string] & { proxy?: { shape: string; dims: number[]; color: string; material: string; archetype?: string } };
    if (!asset && sp.proxy) {
      const b = buildThing({ name: sp.label, archetype: sp.proxy.archetype ?? "generic", dims: sp.proxy.dims, color: sp.proxy.color,
        material: sp.proxy.material ?? "plastic", animate: true }, sp.proxy.shape);
      built.current = b; setObj(b.root); return;
    }
    if (!asset) return;
    if (asset.kind === "procedural") { setObj(proceduralObject(asset)); return; }
    loadGltf(asset.url!).then((scene) => {
      if (!alive) return;
      const model = scene.clone(true);
      if (asset.nodes) {
        // keep only the listed nodes (e.g. the car, not the showcase cloth it ships with)
        model.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) {
            let keep = false;
            for (let p: THREE.Object3D | null = o; p; p = p.parent) if (asset.nodes!.includes(p.name)) keep = true;
            o.visible = keep;
          }
        });
      }
      model.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.castShadow = true; m.receiveShadow = true;
          m.material = (m.material as THREE.Material).clone();     // wheel split must not affect the shared cache
          m.geometry = m.geometry.clone();
        }
      });
      const root = new THREE.Group();
      root.matrixAutoUpdate = false;
      root.matrix.fromArray(asset.gltf_to_body!.flat()).transpose();   // row-major -> three column-major
      root.add(model);
      if (asset.kind === "vehicle") rig.current = rigWheels(model, asset);
      setObj(root);
    });
    return () => { alive = false; };
  }, [asset]);

  const pos = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    const g = holder.current;
    if (!g || idx < 0) return;
    const t = clock.t;
    const rem = sim.removed_at[sid];
    g.visible = t >= spawnT - 1e-3 && !(rem !== undefined && t >= rem);
    poseAt(sim, idx, Math.max(t, spawnT + 1.5 / sim.fps), pos, quat);   // frame at spawn time is still the parked pose
    g.position.copy(pos);
    g.quaternion.copy(quat);
    if (built.current) {
      const [speed, vz] = velocityAt(sim, sid, t);
      built.current.update({ speed, dist: distanceAt(sim, idx, t), t, vz, propelled: speed > 0.1 });
    }
    const wf = sim.wheels?.[sid];
    if (rig.current && wf && wf.length) {
      const f = wf[Math.min(wf.length - 1, Math.max(0, Math.round(t * sim.fps)))];
      const [sfl, sfr, wfl, wfr, wrl, wrr] = f;
      const p = rig.current.pivots;
      p[0].rotation.set(wfl, sfl, 0); p[1].rotation.set(wfr, sfr, 0);
      p[2].rotation.set(wrl, 0, 0); p[3].rotation.set(wrr, 0, 0);
    }
  });

  return <group ref={holder} visible={false}>{obj && <primitive object={obj} />}</group>;
}

export default function Spawned({ sim, assets }: { sim: Sim | null; assets: Record<string, Asset> }) {
  if (!sim?.spawned) return null;
  return (
    <>
      {Object.entries(sim.spawned).map(([sid, sp]) => assets[sp.asset] || sp.kind === "proxy"
        ? <SpawnedOne key={`${sim.key}-${sid}`} sid={sid} asset={assets[sp.asset] ?? null} sim={sim} /> : null)}
    </>
  );
}
