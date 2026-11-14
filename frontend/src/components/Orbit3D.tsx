import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clock } from "../clock";
import { applyCaptureCamera, geometryFor, poseAt } from "../poses";
import { useScanned } from "../scanned";
import type { Scene, SceneObject, Sim } from "../types";

/** Inspectable 3D view of the explicit world model: primitives, table, the
 *  capture camera, and the measured support graph as arrows. */
function Controls() {
  const { camera, gl } = useThree();
  useEffect(() => {
    camera.up.set(0, 0, 1);
    const c = new OrbitControls(camera, gl.domElement);
    c.target.set(0, 0.05, 0.05);
    c.update();
    return () => c.dispose();
  }, [camera, gl]);
  return null;
}

function Content({ scene, sim, selected }: { scene: Scene; sim: Sim | null; selected: string | null }) {
  const refs = useRef<Record<string, THREE.Mesh | null>>({});
  const arrows = useRef<THREE.Group>(null);
  const idx = useMemo(() => Object.fromEntries((sim?.object_ids ?? []).map((id, i) => [id, i])), [sim]);
  const pos = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  const camHelper = useMemo(() => {
    const c = new THREE.PerspectiveCamera();
    applyCaptureCamera(c, scene.camera);
    c.far = 0.25;
    c.updateProjectionMatrix();
    return new THREE.CameraHelper(c);
  }, [scene.camera]);
  const edges = useMemo(() => {
    const out: { a: string; b: string }[] = [];
    for (const o of scene.objects) for (const s of o.supported_by) if (s !== "table" && s !== "floor") out.push({ a: o.id, b: s });
    return out;
  }, [scene]);

  useFrame(() => {
    for (const o of scene.objects) {
      const m = refs.current[o.id];
      if (!m) continue;
      if (sim && idx[o.id] !== undefined) {
        poseAt(sim, idx[o.id], clock.t, pos, quat);
        m.visible = !(sim.removed_at[o.id] !== undefined && clock.t >= sim.removed_at[o.id]);
      } else {
        pos.set(o.pos[0], o.pos[1], o.pos[2]);
        quat.setFromAxisAngle(new THREE.Vector3(0, 0, 1), o.yaw);
      }
      m.position.copy(pos);
      m.quaternion.copy(quat);
    }
    if (arrows.current) arrows.current.visible = !sim || clock.t === 0;
  });

  const t = scene.table;
  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[0.5, -0.6, 2]} intensity={1.6} />
      <mesh position={[(t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2, -0.01]}>
        <boxGeometry args={[t.x_max - t.x_min, t.y_max - t.y_min, 0.02]} />
        <meshStandardMaterial color="#a8a29e" />
      </mesh>
      <primitive object={camHelper} />
      {scene.objects.map((o) => (
        <OrbitObject key={o.id} o={o} selected={o.id === selected} meshRef={(m) => (refs.current[o.id] = m)} />
      ))}
      <group ref={arrows}>
        {edges.map(({ a, b }) => {
          const A = scene.objects.find((o) => o.id === a)!;
          const B = scene.objects.find((o) => o.id === b);
          if (!B) return null;
          const from = new THREE.Vector3(...A.pos);
          const to = new THREE.Vector3(B.pos[0], B.pos[1], B.pos[2] + B.size[2] / 2);
          const dir = to.clone().sub(from);
          const len = dir.length();
          return <primitive key={a + b} object={new THREE.ArrowHelper(dir.normalize(), from, len, 0xf97316, 0.02, 0.012)} />;
        })}
      </group>
    </>
  );
}

export default function Orbit3D(props: { scene: Scene; sim: Sim | null; selected: string | null }) {
  return (
    <Canvas camera={{ position: [0.45, -0.75, 0.55], fov: 45, near: 0.01, far: 20 }} style={{ background: "#0b1020" }}>
      <Controls />
      <Content {...props} />
    </Canvas>
  );
}


function OrbitObject({ o, selected, meshRef }: { o: SceneObject; selected: boolean; meshRef: (m: THREE.Mesh | null) => void }) {
  const scanned = useScanned(o);
  const geom = useMemo(() => geometryFor(o), [o, scanned]);
  return (
    <mesh ref={meshRef} geometry={geom}>
      <meshStandardMaterial color={scanned?.map ? "#ffffff" : new THREE.Color(...(o.mean_color as [number, number, number]))}
        map={scanned?.map ?? null} emissive={selected ? "#facc15" : "#000"} emissiveIntensity={0.35}
        transparent opacity={o.static ? 0.6 : 1} />
    </mesh>
  );
}
