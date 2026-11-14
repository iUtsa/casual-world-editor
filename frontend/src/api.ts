import type { AnyRun, DiagramReading, TransformerRun, Answer, Asset, ExpAnswer, ExperimentSpec, ExpRun, Explanation, Health, Intervention, LiveStatus, Measure, Plan, RunSummary, ScenePayload, Sim } from "./types";

async function j<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const b = await res.json(); if (b.detail) msg = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail); } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}
const post = (url: string, body?: unknown) =>
  fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: body === undefined ? undefined : JSON.stringify(body) });

export type SimPayload = ScenePayload & { sim: Sim | null };
export type ExpResponse =
  | { kind: "run"; run: AnyRun; runs: RunSummary[]; scene_payload?: ScenePayload }
  | ({ kind: "answer" } & ExpAnswer)
  | { kind: "chem"; text: string }
  | { kind: "ml"; text: string }
  | { kind: "unsupported"; text: string; what: string; message: string };

export interface SweepJob {
  id: string; state: "running" | "done" | "error"; stage: string; frac: number; message: string; log: string[];
  error?: string; scene_id?: string; keyframes?: number; pose_note?: string; objects?: number; meshes?: number;
  timings_s?: Record<string, number>;
}

export const api = {
  health: () => fetch("/api/health").then(j<Health>),
  synthetic: () => fetch("/api/synthetic").then(j<{ name: string; title: string; notes: string }[]>),
  async syntheticImage(name: string): Promise<{ blob: Blob; hfov: number }> {
    const r = await fetch(`/api/synthetic/${name}`);
    if (!r.ok) throw new Error("could not load synthetic scene");
    return { blob: await r.blob(), hfov: parseFloat(r.headers.get("X-HFOV") || "60") };
  },
  scan(image: Blob, hfov: number, source: string, useClaude: boolean, depth?: Blob | null) {
    const f = new FormData();
    if (depth) f.append("depth", depth, "depth.png");
    f.append("image", image, "frame.jpg");
    f.append("hfov", String(hfov));
    f.append("source", source);
    f.append("use_claude", String(useClaude));
    return fetch("/api/scan", { method: "POST", body: f }).then(j<ScenePayload>);
  },
  syntheticDepth: (name: string) => fetch(`/api/synthetic/${name}/depth.png`).then((r) => r.blob()),
  scanRemote(hfov: number, useClaude: boolean) {
    const f = new FormData();
    f.append("hfov", String(hfov)); f.append("use_claude", String(useClaude));
    return fetch("/api/scan_remote", { method: "POST", body: f }).then(j<ScenePayload>);
  },
  scene: () => fetch("/api/scene").then(j<ScenePayload>),
  sweep(video: Blob, name: string, hfov: number) {
    const f = new FormData();
    f.append("video", video, name);
    f.append("hfov", String(hfov));
    return fetch("/api/sweep", { method: "POST", body: f }).then(j<SweepJob>);
  },
  sweepStatus: (id: string) => fetch(`/api/sweep/${id}`).then(j<SweepJob>),
  click: (x: number, y: number) => post("/api/objects/click", { x, y }).then(j<ScenePayload & { id: string }>),
  merge: (ids: string[]) => post("/api/objects/merge", { ids }).then(j<ScenePayload>),
  discard: (id: string) => fetch(`/api/objects/${id}`, { method: "DELETE" }).then(j<ScenePayload>),
  patch: (id: string, body: Record<string, unknown>) =>
    fetch(`/api/objects/${id}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(j<ScenePayload>),
  calibrate: (object_id: string, dimension: string, value_m: number) => post("/api/calibrate", { object_id, dimension, value_m }).then(j<ScenePayload>),
  hfov: (hfov: number) => post("/api/hfov", { hfov }).then(j<ScenePayload>),
  cleanPlate(image: Blob) {
    const f = new FormData(); f.append("image", image, "plate.jpg");
    return fetch("/api/clean_plate", { method: "POST", body: f }).then(j<{ ok: boolean }>);
  },
  dropCleanPlate: () => fetch("/api/clean_plate", { method: "DELETE" }).then(j<{ ok: boolean }>),
  intervene: (type: string, target: string | null, params: Record<string, unknown> = {}) =>
    post("/api/interventions", { type, target, params }).then(j<SimPayload>),
  undo: () => post("/api/undo").then(j<SimPayload>),
  redo: () => post("/api/redo").then(j<SimPayload>),
  reset: () => post("/api/reset").then(j<SimPayload>),
  explain: (segment: number, use_claude: boolean) => post("/api/explain", { segment, use_claude }).then(j<Explanation>),
  assets: () => fetch("/api/assets").then(j<Asset[]>),
  plan: (text: string, selected: string | null, use_claude: boolean) => post("/api/plan", { text, selected, use_claude }).then(j<Plan>),
  liveCommand: (text: string, selected: string | null) => post("/api/live/command", { text, selected })
    .then(j<{ handled: false } | ({ handled: true; say: string; target: string; start: number[]; label: string } & SimPayload)>),
  applyPlan: (interventions: Intervention[], checkpoint = false) => post("/api/plan/apply", { interventions, checkpoint }).then(j<SimPayload>),
  spawnAt: (asset: string, pixel: number[] | null, checkpoint = false) => post("/api/spawn", { asset, pixel, checkpoint }).then(j<SimPayload>),
  generate: (name: string, size_m = 0.12) => post("/api/generate", { name, size_m }).then(j<{ job: string }>),
  generateStatus: (job: string) => fetch(`/api/generate/${job}`).then(j<{ state: string; asset?: string; error?: string; elapsed: number }>),
  ask: (question: string, selected: string | null) => post("/api/ask", { question, selected }).then(j<Answer>),
  measure: (oid: string) => fetch(`/api/measure/${oid}`).then(j<Measure>),
  experiment: (text: string, where: "auto" | "scene" | "sandbox", measures: string[] = [], follow_up = true) =>
    post("/api/experiment", { text, where, measures, follow_up }).then(j<ExpResponse>),
  rerun: (experiment: ExperimentSpec, parent: string | null, where: "auto" | "scene" | "sandbox", measures: string[] = []) =>
    post("/api/experiment/rerun", { experiment, parent, where, measures }).then(j<ExpResponse>),
  runs: () => fetch("/api/experiment/runs").then(j<{ runs: RunSummary[]; current: string | null }>),
  getRun: (id: string) => fetch(`/api/experiment/${id}`).then(j<{ run: AnyRun; scene_payload: ScenePayload | null }>),
  currentRun: () => fetch("/api/experiment/current").then(j<{ run: ExpRun | null; live: LiveStatus }>),
  askRun: (question: string, run_id: string | null) => post("/api/experiment/ask", { question, run_id }).then(j<ExpAnswer>),
  liveStatus: () => fetch("/api/live/status").then(j<LiveStatus>),
  liveRescan: (hfov: number, enabled?: boolean) => post("/api/live/rescan", { hfov, enabled }).then(j<LiveStatus>),
  transformer: (text: string, ablate: number[][] = [], mask: number[] = [], temperature = 1) =>
    post("/api/transformer", { text, ablate, mask, temperature }).then(j<TransformerRun>),
  diagram(image: Blob) {
    const f = new FormData(); f.append("image", image, "diagram.jpg");
    return fetch("/api/diagram", { method: "POST", body: f }).then(j<DiagramReading>);
  },
  plateUrl: (ids: string[], key: string) => `/api/plate.png?ids=${encodeURIComponent(ids.join(","))}&k=${key}`,
};
