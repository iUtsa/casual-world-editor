/* Sweep-scanned objects: their real mesh (body frame, z up, centred like the box it replaces) and the texture
 * atlas made from the original camera pixels, loaded once per URL and shared by every view. */
import { useEffect, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

export interface Scanned { geometry: THREE.BufferGeometry; map: THREE.Texture | null }

const done = new Map<string, Scanned>();
const pending = new Map<string, Promise<Scanned>>();

export function scannedNow(url?: string | null): Scanned | null {
  return url ? done.get(url) ?? null : null;
}

export function loadScanned(url: string): Promise<Scanned> {
  const hit = done.get(url);
  if (hit) return Promise.resolve(hit);
  let p = pending.get(url);
  if (!p) {
    p = new GLTFLoader().loadAsync(url).then((g) => {
      let mesh: THREE.Mesh | null = null;
      g.scene.traverse((c) => { if (!mesh && (c as THREE.Mesh).isMesh) mesh = c as THREE.Mesh; });
      if (!mesh) throw new Error("scanned model has no mesh");
      const m = mesh as THREE.Mesh;
      m.updateWorldMatrix(true, false);
      const geometry = m.geometry.clone().applyMatrix4(m.matrixWorld);   // glTF is y-up: bake the node transform
      geometry.computeVertexNormals();
      const mat = m.material as THREE.MeshStandardMaterial;
      const map = mat?.map ?? null;
      if (map) { map.colorSpace = THREE.SRGBColorSpace; map.anisotropy = 4; }
      const s = { geometry, map };
      done.set(url, s);
      return s;
    });
    p.catch(() => pending.delete(url));
    pending.set(url, p);
  }
  return p;
}

/** Re-renders the caller when the object's scanned mesh has loaded. */
export function useScanned(o: { shape: string; mesh_url?: string | null }): Scanned | null {
  const url = o.shape === "mesh" ? o.mesh_url ?? null : null;
  const [s, setS] = useState<Scanned | null>(() => scannedNow(url));
  useEffect(() => {
    if (!url) { setS(null); return; }
    let live = true;
    loadScanned(url).then((x) => { if (live) setS(x); }).catch(() => undefined);
    return () => { live = false; };
  }, [url]);
  return s;
}
