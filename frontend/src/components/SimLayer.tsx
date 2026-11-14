import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { clock } from "../clock";
import { applyCaptureCamera, geometryFor, observedMatrix, poseAt } from "../poses";
import { useScanned } from "../scanned";
import type { Asset, Scene, SceneObject, Sim } from "../types";
import Spawned from "./Spawned";

/* Projective texturing: each moved object is painted with the real pixels it
 * had at capture time. A fragment takes the photo's colour only if, in the
 * capture pose, that surface faced the camera AND the id-map says the pixel
 * belonged to this object; hidden faces fall back to the object's mean colour
 * with simple shading. */
export const vert = /* glsl */ `
  uniform mat4 uModel0;
  uniform mat4 uViewProj0;
  varying vec4 vClip0;
  varying vec3 vN0;
  varying vec3 vP0;
  varying vec3 vNw;
  void main() {
    vec4 p0 = uModel0 * vec4(position, 1.0);
    vP0 = p0.xyz;
    vClip0 = uViewProj0 * p0;
    vN0 = normalize(mat3(uModel0) * normal);
    vNw = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
export const frag = /* glsl */ `
  uniform sampler2D uFrame;
  uniform sampler2D uIds;
  uniform float uIdx;
  uniform vec3 uColor;
  uniform vec3 uCamPos;
  uniform vec3 uLight;
  uniform float uHighlight;
  uniform mat4 uViewProj0f;
  uniform vec3 uC0;        // object centre at capture
  uniform vec3 uMirN;      // horizontal unit vector from the object toward the camera
  varying vec4 vClip0;
  varying vec3 vN0;
  varying vec3 vP0;
  varying vec3 vNw;
  // sample the photo at a capture-time surface point if the camera saw it there
  bool samplePhoto(vec3 p, vec3 n, out vec3 col) {
    vec4 c = uViewProj0f * vec4(p, 1.0);
    vec2 uv = c.xy / c.w * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) return false;
    if (dot(n, normalize(uCamPos - p)) <= 0.08) return false;
    float id = floor(texture2D(uIds, uv).r * 255.0 + 0.5);
    if (abs(id - uIdx) > 0.5) return false;
    col = texture2D(uFrame, uv).rgb;
    return true;
  }
  void main() {
    float shade = 0.62 + 0.38 * max(dot(normalize(vNw), uLight), 0.0);
    vec3 col = uColor * shade;
    vec3 photo;
    if (samplePhoto(vP0, vN0, photo)) {
      col = photo;
    } else {
      // hidden side: borrow the real pixels from the mirror point across the object's
      // centre plane (objects are roughly symmetric front/back), slightly darkened
      float k = dot(vP0 - uC0, uMirN);
      vec3 pm = vP0 - 2.0 * k * uMirN;
      vec3 nm = vN0 - 2.0 * dot(vN0, uMirN) * uMirN;
      if (samplePhoto(pm, nm, photo)) col = photo * (0.82 + 0.18 * shade);
    }
    col = mix(col, vec3(1.0, 0.85, 0.2), uHighlight * 0.35);
    gl_FragColor = vec4(col, 1.0);
  }`;

export function useTex(url: string | null, nearest: boolean) {
  return useMemo(() => {
    if (!url) return null;
    const t = new THREE.TextureLoader().load(url);
    t.colorSpace = THREE.NoColorSpace;
    t.generateMipmaps = !nearest;
    if (nearest) { t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter; }
    return t;
  }, [url, nearest]);
}

function CameraRig({ scene }: { scene: Scene }) {
  const { camera, size } = useThree();
  useEffect(() => {
    applyCaptureCamera(camera as THREE.PerspectiveCamera, scene.camera);
  }, [camera, scene.camera, size]);
  return null;
}

interface Props {
  scene: Scene; sim: Sim | null; frameUrl: string; idsUrl: string;
  selected: string | null; showFit: boolean; showGhosts: boolean; assets: Record<string, Asset>;
}

/** Image-based lighting from the captured frame: summoned objects reflect the
 *  real room, and a white surface lit by it reproduces the scene's own
 *  brightness, so exposure 1 matches the photo. Returns mean linear luminance. */
function FrameLighting({ frameUrl, onLum }: { frameUrl: string; onLum: (l: number) => void }) {
  const { gl, scene } = useThree();
  useEffect(() => {
    let disposed = false;
    const img = new Image();
    img.src = frameUrl;
    img.onload = () => {
      if (disposed) return;
      const c = document.createElement("canvas");
      c.width = 64; c.height = 36;
      const ctx = c.getContext("2d")!;
      ctx.drawImage(img, 0, 0, 64, 36);
      const d = ctx.getImageData(0, 0, 64, 36).data;
      let lum = 0;
      const lin = (v: number) => { v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
      for (let i = 0; i < d.length; i += 4) lum += 0.2126 * lin(d[i]) + 0.7152 * lin(d[i + 1]) + 0.0722 * lin(d[i + 2]);
      onLum(lum / (64 * 36));
      const tex = new THREE.Texture(img);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.needsUpdate = true;
      const pm = new THREE.PMREMGenerator(gl);
      const env = pm.fromEquirectangular(tex).texture;
      scene.environment = env;
      tex.dispose();
      pm.dispose();
    };
    return () => { disposed = true; };
  }, [frameUrl, gl, scene, onLum]);
  return null;
}

function World({ scene, sim, frameUrl, idsUrl, selected, showFit, showGhosts, assets }: Props) {
  const [lum, setLum] = useState(0.18);
  const frameTex = useTex(frameUrl, false);
  const idTex = useTex(idsUrl, true);
  const { camera } = useThree();
  const dynamic = useMemo(() => {
    if (!sim) return new Set<string>();
    return new Set([...sim.moved_ids, ...Object.keys(sim.removed_at)]);
  }, [sim]);
  const meshes = useRef<Record<string, THREE.Mesh | null>>({});
  const fits = useRef<Record<string, THREE.LineSegments | null>>({});
  const objIndex = useMemo(() => {
    const m: Record<string, number> = {};
    sim?.object_ids.forEach((id, i) => (m[id] = i));
    return m;
  }, [sim]);
  const light = useMemo(() => new THREE.Vector3(0.3, -0.4, 1).normalize(), []);

  const materials = useMemo(() => {
    const out: Record<string, THREE.ShaderMaterial> = {};
    if (!frameTex || !idTex) return out;
    const cam = camera as THREE.PerspectiveCamera;
    applyCaptureCamera(cam, scene.camera);
    const viewProj = new THREE.Matrix4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    for (const o of scene.objects) {
      out[o.id] = new THREE.ShaderMaterial({
        vertexShader: vert, fragmentShader: frag,
        uniforms: {
          uFrame: { value: frameTex }, uIds: { value: idTex }, uIdx: { value: o.idx },
          uModel0: { value: observedMatrix(o) }, uViewProj0: { value: viewProj }, uViewProj0f: { value: viewProj },
          uC0: { value: new THREE.Vector3(o.pos[0], o.pos[1], o.pos[2]) },
          uMirN: { value: (() => { const v = new THREE.Vector3(scene.camera.t_wc[0] - o.pos[0], scene.camera.t_wc[1] - o.pos[1], 0); return v.lengthSq() > 0 ? v.normalize() : new THREE.Vector3(0, -1, 0); })() },
          uColor: { value: new THREE.Color(o.mean_color[0], o.mean_color[1], o.mean_color[2]) },
          uCamPos: { value: new THREE.Vector3(...scene.camera.t_wc) }, uLight: { value: light },
          uHighlight: { value: 0 },
        },
      });
    }
    return out;
  }, [scene, frameTex, idTex, camera, light]);

  const pos = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);
  useFrame(() => {
    const t = clock.t;
    for (const o of scene.objects) {
      const mesh = meshes.current[o.id];
      const fit = fits.current[o.id];
      const i = objIndex[o.id];
      let visible = true;
      if (sim && i !== undefined) {
        poseAt(sim, i, t, pos, quat);
        const rem = sim.removed_at[o.id];
        if (rem !== undefined && t >= rem) visible = false;
      } else {
        pos.set(o.pos[0], o.pos[1], o.pos[2]);
        quat.setFromAxisAngle(new THREE.Vector3(0, 0, 1), o.yaw);
      }
      for (const obj of [mesh, fit]) {
        if (!obj) continue;
        obj.position.copy(pos);
        obj.quaternion.copy(quat);
      }
      if (mesh) mesh.visible = visible;
      if (fit) fit.visible = visible && showFit;
      const mat = materials[o.id];
      if (mat) mat.uniforms.uHighlight.value = o.id === selected ? 1 : 0;
    }
  });

  const t = scene.table;
  return (
    <>
      <CameraRig scene={scene} />
      <FrameLighting frameUrl={frameUrl} onLum={setLum} />
      <directionalLight position={[0.4, -0.3, 2.5]} intensity={Math.min(3, 1.6 * lum + 0.15)} castShadow
        shadow-mapSize={[2048, 2048]} shadow-camera-left={-1} shadow-camera-right={1}
        shadow-camera-top={1} shadow-camera-bottom={-1} shadow-camera-near={0.1} shadow-camera-far={6}
        shadow-bias={-0.0005} />
      {/* real table: invisible depth occluder + shadow catcher */}
      <mesh position={[(t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2, -0.022]} renderOrder={-2}>
        <boxGeometry args={[t.x_max - t.x_min, t.y_max - t.y_min, 0.04]} />
        <meshBasicMaterial colorWrite={false} />
      </mesh>
      <mesh position={[(t.x_min + t.x_max) / 2, (t.y_min + t.y_max) / 2, 0.0008]} receiveShadow renderOrder={-1}>
        <planeGeometry args={[t.x_max - t.x_min, t.y_max - t.y_min]} />
        <shadowMaterial opacity={0.32} transparent depthWrite={false} />
      </mesh>
      {scene.objects.map((o) => (
        <ObjectView key={o.id} o={o} dynamic={dynamic.has(o.id)} material={materials[o.id]}
          meshRef={(m) => (meshes.current[o.id] = m)} fitRef={(l) => (fits.current[o.id] = l)} />
      ))}
      <Spawned sim={sim} assets={assets} />
      {showGhosts && sim && Object.keys(sim.removed_at).map((id) => {
        const o = scene.objects.find((x) => x.id === id);
        return o ? <Ghost key={id} o={o} /> : null;
      })}
    </>
  );
}

function ObjectView({ o, dynamic, material, meshRef, fitRef }: {
  o: SceneObject; dynamic: boolean; material?: THREE.ShaderMaterial;
  meshRef: (m: THREE.Mesh | null) => void; fitRef: (l: THREE.LineSegments | null) => void;
}) {
  const scanned = useScanned(o);
  const geom = useMemo(() => geometryFor(o), [o, scanned]);
  const edges = useMemo(() => new THREE.EdgesGeometry(geom, 25), [geom]);
  // a moving sweep-scanned object shows its own texture: its back side is real, not a mirrored guess
  const own = useMemo(() => scanned?.map ? new THREE.MeshStandardMaterial({ map: scanned.map, roughness: 0.75 }) : null, [scanned]);
  return (
    <>
      {dynamic && (own || material) ? (
        <mesh ref={meshRef} geometry={geom} material={own ?? material} castShadow />
      ) : (
        // unmoved real object: invisible, but it occludes moving things behind it
        <mesh ref={meshRef} geometry={geom} renderOrder={-2}>
          <meshBasicMaterial colorWrite={false} />
        </mesh>
      )}
      <lineSegments ref={fitRef} geometry={edges}>
        <lineBasicMaterial color="#5eead4" transparent opacity={0.85} depthTest={false} />
      </lineSegments>
    </>
  );
}

function Ghost({ o }: { o: SceneObject }) {
  const scanned = useScanned(o);
  const geom = useMemo(() => new THREE.EdgesGeometry(geometryFor(o), 25), [o, scanned]);
  const m = useMemo(() => observedMatrix(o), [o]);
  return (
    <lineSegments geometry={geom} matrix={m} matrixAutoUpdate={false}
      onUpdate={(l) => l.computeLineDistances()}>
      <lineDashedMaterial color="#f87171" dashSize={0.01} gapSize={0.007} depthTest={false} />
    </lineSegments>
  );
}

export default function SimLayer(props: Props) {
  return (
    <Canvas className="gl-layer" shadows="soft"
      gl={{ alpha: true, antialias: true, preserveDrawingBuffer: true }}
      onCreated={({ gl }) => { gl.toneMapping = THREE.NeutralToneMapping; gl.toneMappingExposure = 1; }}
      camera={{ position: [0, -1, 1] }}
      style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      <World {...props} />
    </Canvas>
  );
}
