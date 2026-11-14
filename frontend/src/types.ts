export type Shape = "box" | "cylinder" | "mesh";

export interface Camera {
  width: number; height: number; fx: number; fy: number; cx: number; cy: number;
  hfov_deg: number; R_wc: number[][]; t_wc: number[];
}

export interface SceneObject {
  id: string; idx: number; label: string; material: string; shape: Shape;
  size: number[]; pos: number[]; yaw: number; profile?: number[][] | null;
  mesh_url?: string | null; hulls?: number[][][] | null; volume_m3?: number | null; mass: number; mass_source: string;
  friction: number; static: boolean; supported_by: string[]; supporting: string[];
  polygon: number[][]; bbox: number[]; area_px: number; mean_color: number[];
  score: number; label_source: string; notes: string[];
  stability: { moved_m: number; rotated_deg: number; stable: boolean };
}

export interface Scene {
  scene_id: string; source: string; camera: Camera;
  table: { x_min: number; x_max: number; y_min: number; y_max: number; height_above_floor: number; plane_rms_m: number };
  objects: SceneObject[]; timings_ms: Record<string, number>; warnings: string[]; labeler: string;
}

export interface Intervention { type: string; target: string | null; params: Record<string, unknown> }

export interface SimEvent {
  id: string; t: number; type: string; subject: string | null; other: string | null;
  [k: string]: unknown;
}

export interface CausalEdge { cause: string; effect: string; kind: string; t: number; evidence: string[] }

export interface Segment {
  index: number; intervention: Intervention; t_start: number; t_end: number;
  events: SimEvent[]; causal_edges: CausalEdge[];
  summary: Record<string, { label: string; displacement_m: number; drop_m: number; rotation_deg: number;
    max_speed_mps: number; toppled: boolean; fell_off_table: boolean; moved: boolean; cause: string | null }>;
  support_before: Record<string, { on: string; force_N: number; share: number }[]>;
  support_after: Record<string, { on: string; force_N: number; share: number }[]>;
  predicted_dependents: string[];
}

export interface Sim {
  fps: number; duration: number; object_ids: string[];
  frames: number[][][];            // [frame][object] = [x,y,z,qx,qy,qz,qw]
  removed_at: Record<string, number>; moved_ids: string[]; segments: Segment[];
  gravity: number; key: string; compute_ms: number;
  spawned: Record<string, { asset: string; label: string; t: number; kind: string;
    proxy?: { shape: string; dims: number[]; color: string; material: string; archetype?: string } }>;
  wheels: Record<string, number[][]>;     // per frame: [steer_fl, steer_fr, spin_fl, spin_fr, spin_rl, spin_rr]
  labels: Record<string, string>;
  kinematics: Record<string, { v: (number[] | null)[]; F_contact: (number | null)[]; F_objects: (number | null)[]; KE: (number | null)[]; PE: (number | null)[];
    c?: (number[] | null)[]; w?: (number[] | null)[] }>;
  masses: Record<string, number>;
  pair_forces?: Record<string, number[][]>;   // "a|b" -> [frame, mean |F|, peak |F| (N), impulse (N s)]
}

export interface Answer {
  answer: string; value: number | null; unit?: string; quantity?: string; subject?: string; subject_id?: string;
  method?: string; assumptions?: string[]; needs_experiment?: boolean;
  details?: Record<string, unknown>;
}

export interface Measure {
  t: number[]; speed: number[]; hspeed: number[]; acc: number[]; F: number[]; F_objects: number[]; KE: number[]; PE: number[];
  metrics: Record<string, number>; label: string; segment: { t_start: number; t_end: number };
}

export interface Asset {
  id: string; name: string; aliases: string[]; kind: "vehicle" | "rigid" | "procedural";
  url?: string; nodes?: string[] | null; gltf_to_body?: number[][]; dims: number[];
  mass: number; friction: number; bouncy: boolean; static: boolean;
  shape?: "box" | "sphere" | "wedge"; material?: string; thumb: string | null;
  license: string | null; source: string | null; authors: string[] | null;
  wheels?: { centres: number[][]; radius: number; width: number; order: string[];
    centres_gltf: number[][]; radius_gltf: number; width_gltf: number };
}

export interface Plan {
  planner: string; steps: string[]; notes: string[]; interventions: Intervention[]; request: string;
  generate: string | null; can_generate: boolean;
}

export interface ScenePayload {
  scene: Scene | null; interventions: Intervention[]; can_redo: boolean;
  depth_scale: number; has_clean_plate: boolean;
}

export interface Explanation {
  headline: string;
  causal_chain: { cause: string; effect: string; mechanism: string; principle: string; evidence: string[] }[];
  unaffected: string; graph_vs_physics: string; counterfactual: string;
  confidence: number; confidence_reason: string; assumptions: string[];
  source: string; compute_ms: number; segment: number;
}

export interface Health { models: string; device: string; claude: boolean; claude_model: string; fal: boolean; local_ai: string }

export type Tool = "select" | "remove" | "push" | "add" | "merge" | "place" | "driveto";

// ---------------------------------------------------------------- experiments
export interface ResultRow { label: string; value: number | null; unit: string; about: string; text: string; why?: string | null; picked?: boolean; quantity?: string }

export interface ExpObject {
  name: string; source: string; shape: string; dims: number[]; mass: number; material: string; color: string; note: string;
  sid: string | null; bounciness: number; self_propelled: boolean; habitat: string; typical_speed: number;
  asset_id: string | null; scene_id: string | null; fixed: boolean; support: boolean; archetype: string;
}

export interface ExpAction { do: string; a: string; b: string | null; speed: number | null; force: number | null; height: number | null;
  distance: number | null; angle_deg: number | null; rpm: number | null; stiffness: number | null; duration: number | null; from_ground?: boolean }

export interface ExperimentSpec {
  title: string; setting: string; objects: { name: string; what: string; count: number; size_m: number | null; mass_kg: number | null; bounciness: number | null }[];
  actions: ExpAction[]; measure: string[]; seconds: number | null; understood_by: string; notes: string[]; gravity: number | null;
}

export interface FxEvent { t: number; kind: "hit" | "ground" | "splash"; pos: number[]; force?: number; impulse?: number; speed?: number; a: string; b?: string; scale: number }

export interface ExpSim extends Sim {
  links: { id: string; a: string; b: string; kind: string; length: number; stiffness?: number }[];
  tensions: Record<string, number[][]>;
  env?: Record<string, number>;
}

export interface ExpRun {
  id: string; parent: string | null; changes: string[]; text: string; created: number; duration: number; compute_ms?: number;
  experiment: ExperimentSpec; setting: string; gravity: number; env: Record<string, number>;
  objects: ExpObject[]; notes: string[]; interventions: Intervention[]; sim: ExpSim;
  results: ResultRow[]; story: string[]; fx: FxEvent[]; catalog: ResultRow[];
  scene_table: { x_min: number; x_max: number; y_min: number; y_max: number; height_above_floor: number };
}

export interface RunSummary { id: string; text: string; setting: string; parent: string | null; changes: string[]; created: number; headline: ResultRow[] }

export interface ExpAnswer { answer: string; value?: number; unit?: string; text?: string; rows: ResultRow[]; why?: string[] }

export interface LiveStatus { version: number; state: string; last_ms: number | null; added: string[]; removed: string[]; error: string | null;
  frames_fused: number; enabled: boolean; rerun: string | null; buffered: number;
  phone: { connected: boolean; age_s: number | null; fps: number; meta: { steady?: boolean; motion?: number; change?: number } } }

export interface CircuitElement { id: string; kind: string; label: string; value: number; pos: { x: number; y: number; dir: string }; color: string;
  closed: boolean; blown: boolean; on: boolean; I: number[]; V: number[]; brightness: number; note?: string }
export interface CircuitRun { domain: "circuit"; id: string; parent: string | null; changes: string[]; text: string; source_text: string; created: number; compute_ms?: number;
  source: { label: string; V: number; r_int: number; pos: { x: number; y: number }; I: number[] };
  elements: CircuitElement[]; wires: { pts: number[][]; el: string; I: number[] }[]; board: { w: number; h: number };
  t: number[]; duration: number; dynamic: boolean; results: ResultRow[]; story: string[]; notes: string[] }
export interface SpaceBody { name: string; label?: string; kind?: string; mass: number; radius: number; color: string; rings?: number[] | null; temp?: number | null; parent?: string | null; pole?: number[] | null }
export interface SpaceBelt { name: string; color: string; a: number[]; e: number[]; i: number[]; node: number[]; argp: number[]; M0: number[]; mu: number; size: number; longitude_mode?: boolean }
export interface GravityRun { domain: "gravity"; id: string; parent: string | null; changes: string[]; text: string; source_text: string; created: number; compute_ms?: number;
  mode: string; bodies: SpaceBody[]; t: number[]; pos: number[][][];
  source?: string; date?: string | null; frame?: string | null; focus?: string[] | null; belts?: SpaceBelt[]; sun_index?: number | null;
  particles?: { n: number; frames: number; unit_m: number; data: string; colors: string; split: number };
  events: { t: number; kind: string; a: string; b: string; speed: number; energy: number; pos: number[]; small_radius: number; big_radius: number }[];
  alive: boolean[]; sim_seconds: number; duration: number; results: ResultRow[]; story: string[]; notes: string[];
  encounter?: string; view_extent_m?: number }
export type AnyRun = (ExpRun & { domain?: "mechanics" }) | CircuitRun | GravityRun;

export interface TransformerRun {
  text: string; tokens: string[]; n_layers: number; n_heads: number; temperature: number;
  attention: number[][][][];          // [layer][head][query][key]
  heads: { layer: number; head: number; prev: number; first: number; self: number; entropy: number; kind: string; off: boolean }[];
  lens: { layer: number; top: { token: string; p: number }[] }[];
  next: { token: string; p: number }[]; baseline_next: { token: string; p: number }[] | null;
  pca: number[][][]; norms: number[][]; attn_update: number[]; mlp_update: number[];
  ablate: number[][]; mask: number[]; params: string; model: string; story: string[]; compute_ms?: number;
}
export interface DiagramReading { kind: string; reading: string; lab?: string; labels?: string[]; error?: string; kind_raw?: string }
