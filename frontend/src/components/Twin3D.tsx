/* 🧊 3D twin of the mapped space: every real object and the visible table are painted with the camera's own pixels
 * (projective texturing from the capture pose; hidden faces borrow the mirrored front), so you can orbit around
 * your desk. Simulations play here too, frame-exact from the same MuJoCo run, with velocity arrows. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { clock } from "../clock";
import { applyCaptureCamera, geometryFor, observedMatrix, poseAt } from "../poses";
import { useScanned } from "../scanned";
import { frag, useTex, vert } from "./SimLayer";
import Spawned from "./Spawned";
import type { Asset, Scene, Sim } from "../types";

interface Props {
  scene: Scene; sim: Sim | null; frameUrl: string; idsUrl: string; assets: Record<string, Asset>;
  selected: string | null; onPick: (id: string | null) => void;
}

const tableVert = /* glsl */ `
  uniform mat4 uViewProj0;
  varying vec4 vClip0;
  void main() {
    vec4 w = modelMatrix * vec4(position, 1.0);
    vClip0 = uViewProj0 * w;
    gl_Position = projectionMatrix * viewMatrix * w;
  }`;
const tableFrag = /* glsl */ `
  uniform sampler2D uFrame;
  uniform sampler2D uIds;
  uniform vec3 uTable;
  varying vec4 vClip0;
  void main() {
    if (vClip0.w <= 0.0) discard;
    vec2 uv = vClip0.xy / vClip0.w * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) discard;     // only what the camera saw
    float id = floor(texture2D(uIds, uv).r * 255.0 + 0.5);
    vec3 c = id < 0.5 ? texture2D(uFrame, uv).rgb : uTable;                    // under an object: plain table colour
    float e = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    gl_FragColor = vec4(c, smoothstep(0.0, 0.05, e));
  }`;

/** Mean colour of the table pixels (id 0) in the scanned frame: fills the table under objects. */
function useTableColor(frameUrl: string, idsUrl: string) {
  const [c, setC] = useState(new THREE.Color(0.35, 0.3, 0.26));
  useEffect(() => {
    const a = new Image(), b = new Image();
    let n = 0;
    const done = () => {
      if (++n < 2) return;
      const w = 96, h = 54, cv = document.createElement("canvas");
      cv.width = w; cv.height = h;
      const x = cv.getContext("2d", { willReadFrequently: true })!;
      x.drawImage(a, 0, 0, w, h); const f = x.getImageData(0, 0, w, h).data;
      x.imageSmoothingEnabled = false; x.drawImage(b, 0, 0, w, h); const d = x.getImageData(0, 0, w, h).data;
      let r = 0, g = 0, bl = 0, k = 0;
      for (let i = Math.floor(h * 0.4) * w * 4; i < f.length; i += 4) if (d[i] === 0) { r += f[i]; g += f[i + 1]; bl += f[i + 2]; k++; }
      if (k) setC(new THREE.Color(r / k / 255, g / k / 255, bl / k / 255));
    };
    a.onload = done; b.onload = done; a.src = frameUrl; b.src = idsUrl;
  }, [frameUrl, idsUrl]);
  return c;
}

function Rig({ scene, target }: { scene: Scene; target: THREE.Vector3 }) {
  const { camera, gl } = useThree();
  const ctl = useRef<OrbitControls | null>(null);
  const intro = useRef({ t0: -1, from: new THREE.Vector3() });
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    applyCaptureCamera(cam, scene.camera);               // start exactly where your camera was
    cam.up.set(0, 0, 1);
    const c = new OrbitControls(cam, gl.domElement);
    c.target.copy(target); c.enableDamping = true; c.dampingFactor = 0.08; c.maxPolarAngle = Math.PI * 0.49;
    c.minDistance = 0.1; c.maxDistance = 8; c.update();
    ctl.current = c;
    intro.current = { t0: performance.now(), from: cam.position.clone() };
    return () => c.dispose();
  }, [camera, gl, scene.camera, target]);
  useFrame(() => {
    const c = ctl.current;
    if (!c) return;
    const it = intro.current;
    if (it.t0 > 0) {                                     // swing 35° around the desk so it reads as 3D at once
      const k = Math.min(1, (performance.now() - it.t0) / 1600), e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
      const v = it.from.clone().sub(c.target).applyAxisAngle(new THREE.Vector3(0, 0, 1), THREE.MathUtils.degToRad(35) * e);
      camera.position.copy(c.target).add(v);
      if (k >= 1) it.t0 = -1;
    }
    c.update();
  });
  return null;
}

function VelocityArrows({ sim }: { sim: Sim }) {
  const g = useRef<THREE.Group>(null);
  const arrows = useMemo(() => sim.object_ids.map(() => {
    const a = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.1, 0xfbbf24, 0.03, 0.02);
    a.visible = false; return a;
  }), [sim]);
  useEffect(() => { const grp = g.current; if (!grp) return; arrows.forEach((a) => grp.add(a)); return () => { arrows.forEach((a) => grp.remove(a)); }; }, [arrows]);
  useFrame(() => {
    const f = Math.min(sim.frames.length - 1, Math.max(0, Math.round(clock.t * sim.fps)));
    sim.object_ids.forEach((id, i) => {
      const v = sim.kinematics?.[id]?.v[f], a = arrows[i];
      const sp = v ? Math.hypot(v[0], v[1], v[2]) : 0;
      a.visible = !!v && sp > 0.06 && sim.removed_at[id] === undefined;
      if (!a.visible || !v) return;
      const p = sim.frames[f][i];
      a.position.set(p[0], p[1], p[2] + (sim.spawned?.[id] ? 0.03 : 0));
      a.setDirection(new THREE.Vector3(v[0], v[1], v[2]).normalize());
      a.setLength(Math.min(0.5, sp * 0.2), 0.03, 0.02);
    });
  });
  return <group ref={g} />;
}

function World({ scene, sim, frameUrl, idsUrl, assets, selected, onPick }: Props) {
  const frameTex = useTex(frameUrl, false);
  const idTex = useTex(idsUrl, true);
  const tableColor = useTableColor(frameUrl, idsUrl);
  const viewProj = useMemo(() => {
    const c = new THREE.PerspectiveCamera();
    applyCaptureCamera(c, scene.camera);
    return new THREE.Matrix4().multiplyMatrices(c.projectionMatrix, c.matrixWorldInverse);
  }, [scene.camera]);
  const light = useMemo(() => new THREE.Vector3(0.3, -0.4, 1).normalize(), []);
  const materials = useMemo(() => {
    const out: Record<string, THREE.ShaderMaterial> = {};
    if (!frameTex || !idTex) return out;
    for (const o of scene.objects) {
      const mir = new THREE.Vector3(scene.camera.t_wc[0] - o.pos[0], scene.camera.t_wc[1] - o.pos[1], 0);
      out[o.id] = new THREE.ShaderMaterial({
        vertexShader: vert, fragmentShader: frag,
        uniforms: {
          uFrame: { value: frameTex }, uIds: { value: idTex }, uIdx: { value: o.idx },
          uModel0: { value: observedMatrix(o) }, uViewProj0: { value: viewProj }, uViewProj0f: { value: viewProj },
          uC0: { value: new THREE.Vector3(o.pos[0], o.pos[1], o.pos[2]) },
          uMirN: { value: mir.lengthSq() > 0 ? mir.normalize() : new THREE.Vector3(0, -1, 0) },
          uColor: { value: new THREE.Color(o.mean_color[0], o.mean_color[1], o.mean_color[2]) },
          uCamPos: { value: new THREE.Vector3(...scene.camera.t_wc) }, uLight: { value: light }, uHighlight: { value: 0 },
        },
      });
    }
    return out;
  }, [scene, frameTex, idTex, viewProj, light]);
  const tableMat = useMemo(() => frameTex && idTex ? new THREE.ShaderMaterial({
    vertexShader: tableVert, fragmentShader: tableFrag, transparent: true, depthWrite: true,
    uniforms: { uFrame: { value: frameTex }, uIds: { value: idTex }, uViewProj0: { value: viewProj }, uTable: { value: tableColor } },
  }) : null, [frameTex, idTex, viewProj, tableColor]);

  const meshes = useRef<Record<string, THREE.Mesh | null>>({});
  const idx = useMemo(() => Object.fromEntries((sim?.object_ids ?? []).map((id, i) => [id, i])), [sim]);
  const pos = useMemo(() => new THREE.Vector3(), []), quat = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    for (const o of scene.objects) {
      const m = meshes.current[o.id];
      if (!m) continue;
      const i = idx[o.id];
      let vis = true;
      if (sim && i !== undefined) {
        poseAt(sim, i, clock.t, pos, quat);
        const rem = sim.removed_at[o.id];
        if (rem !== undefined && clock.t >= rem) vis = false;
      } else { pos.set(o.pos[0], o.pos[1], o.pos[2]); quat.setFromAxisAngle(new THREE.Vector3(0, 0, 1), o.yaw); }
      m.position.copy(pos); m.quaternion.copy(quat); m.visible = vis;
      const mat = materials[o.id];
      if (mat) mat.uniforms.uHighlight.value = o.id === selected ? 1 : 0;
    }
  });

  const T = scene.table;
  const target = useMemo(() => {
    const os = scene.objects;
    if (!os.length) return new THREE.Vector3((T.x_min + T.x_max) / 2, (T.y_min + T.y_max) / 2, 0);
    return new THREE.Vector3(os.reduce((s, o) => s + o.pos[0], 0) / os.length, os.reduce((s, o) => s + o.pos[1], 0) / os.length,
      Math.max(0.02, os.reduce((s, o) => s + o.pos[2], 0) / os.length));
  }, [scene, T]);
  const camHelper = useMemo(() => {
    const c = new THREE.PerspectiveCamera();
    applyCaptureCamera(c, scene.camera);
    c.far = Math.max(0.08, 0.12 * Math.hypot(...scene.camera.t_wc.map((v, k) => v - target.getComponent(k))));
    c.updateProjectionMatrix();
    const h = new THREE.CameraHelper(c);
    (h.material as THREE.LineBasicMaterial).color = new THREE.Color("#38bdf8");
    return h;
  }, [scene.camera, target]);
  const span = Math.max(T.x_max - T.x_min, T.y_max - T.y_min);
  const pick = (id: string) => (e: ThreeEvent<MouseEvent>) => { if (e.delta > 5) return; e.stopPropagation(); onPick(id === selected ? null : id); };

  return (
    <>
      <Rig scene={scene} target={target} />
      <hemisphereLight args={["#e0f2fe", "#1e293b", 0.9]} />
      <directionalLight position={[target.x + 0.5, target.y - 0.6, 2.2]} intensity={1.6} castShadow shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-span / 2} shadow-camera-right={span / 2} shadow-camera-top={span / 2} shadow-camera-bottom={-span / 2}
        shadow-camera-near={0.1} shadow-camera-far={6} shadow-bias={-0.0004} />
      <gridHelper args={[Math.max(2, span * 1.6), 40, "#1e3a5f", "#132238"]} rotation={[Math.PI / 2, 0, 0]} position={[target.x, target.y, -0.002]} />
      {tableMat && (
        <mesh position={[(T.x_min + T.x_max) / 2, (T.y_min + T.y_max) / 2, 0]} material={tableMat} renderOrder={-1}
          onClick={(e) => { if (e.delta <= 5) onPick(null); }}>
          <planeGeometry args={[T.x_max - T.x_min, T.y_max - T.y_min]} />
        </mesh>
      )}
      <mesh position={[(T.x_min + T.x_max) / 2, (T.y_min + T.y_max) / 2, 0.0006]} receiveShadow>
        <planeGeometry args={[T.x_max - T.x_min, T.y_max - T.y_min]} />
        <shadowMaterial opacity={0.35} transparent depthWrite={false} />
      </mesh>
      {scene.objects.map((o) => <ObjectMesh key={o.id} o={o} material={materials[o.id]} onClick={pick(o.id)}
        meshRef={(m) => (meshes.current[o.id] = m)} />)}
      <Spawned sim={sim} assets={assets} />
      {sim && <VelocityArrows sim={sim} />}
      <primitive object={camHelper} />
    </>
  );
}

function ObjectMesh({ o, material, meshRef, onClick }: { o: Scene["objects"][number]; material?: THREE.ShaderMaterial;
  meshRef: (m: THREE.Mesh | null) => void; onClick: (e: ThreeEvent<MouseEvent>) => void }) {
  const scanned = useScanned(o);
  const geom = useMemo(() => geometryFor(o), [o, scanned]);
  // a sweep-scanned object carries its own texture, correct from every side; others are painted from the camera
  const own = useMemo(() => scanned?.map ? new THREE.MeshStandardMaterial({ map: scanned.map, roughness: 0.75 }) : null, [scanned]);
  if (!material && !own) return null;
  return <mesh ref={meshRef} geometry={geom} material={own ?? material} castShadow receiveShadow onClick={onClick}
    onPointerOver={() => (document.body.style.cursor = "pointer")} onPointerOut={() => (document.body.style.cursor = "")} />;
}

export default function Twin3D(p: Props) {
  return (
    <div className="twin3d">
      <Canvas shadows="soft" gl={{ antialias: true }} onCreated={({ gl }) => { gl.toneMapping = THREE.NeutralToneMapping; }}
        onPointerMissed={() => p.onPick(null)}>
        <World {...p} />
      </Canvas>
      <div className="twin-hint">Drag to orbit · scroll to zoom · click an object · <span style={{ color: "#38bdf8" }}>blue frame</span> = where your camera was</div>
    </div>
  );
}
