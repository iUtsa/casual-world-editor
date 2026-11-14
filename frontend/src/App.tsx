import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type ExpResponse, type SimPayload, type SweepJob } from "./api";
import { clock } from "./clock";
import Stage, { type SourceView } from "./components/Stage";
import Source from "./components/Source";
import Inspector from "./components/Inspector";
import Explanation from "./components/Explanation";
import Orbit3D from "./components/Orbit3D";
import CommandBar from "./components/CommandBar";
import Library from "./components/Library";
import Measurements from "./components/Measurements";
import ChemView from "./components/ChemView";
import ConnectPhone from "./components/ConnectPhone";
import ExperimentView, { type ViewOptions } from "./components/ExperimentView";
import Player from "./components/Player";
import { DomainEdit, DomainResults, EditTab, ResultsTab, RunsTab } from "./components/LabPanel";
import CircuitView from "./components/CircuitView";
import SpaceView from "./components/SpaceView";
import LiveAR, { type ChatMsg } from "./components/LiveAR";
import type { HandAction } from "./components/HandLayer";
import TransformerLab from "./components/TransformerLab";
import type { AnyRun, Asset, DiagramReading, ExpAnswer, ExperimentSpec, ExpRun, Explanation as Ex, FxEvent, Health, LiveStatus, ResultRow, RunSummary, ScenePayload, Sim, Tool } from "./types";
import type { Pt } from "./geometry2d";

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: "select", label: "Select", hint: "click an object to inspect it" },
  { id: "remove", label: "Remove", hint: "click an object to delete it from the world" },
  { id: "push", label: "Push", hint: "drag on an object: direction and length set the push" },
  { id: "add", label: "Add / refine", hint: "click a missed object to segment it, or an existing one to refine its mask" },
  { id: "merge", label: "Merge", hint: "click pieces of one object, then press Merge" },
];
const DEFAULT_VIEW: ViewOptions = { slowmo: true, trails: true, arrows: true, labels: true, follow: false, forces: false };

function isMech(r: AnyRun): r is ExpRun & { domain?: "mechanics" } { return !r.domain || r.domain === "mechanics"; }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [payload, setPayload] = useState<ScenePayload | null>(null);
  const [sim, setSim] = useState<Sim | null>(null);
  const [ex, setEx] = useState<Ex | null>(null);
  const [exLoading, setExLoading] = useState(false);
  const [tool, setTool] = useState<Tool>("select");
  const [selected, setSelected] = useState<string | null>(null);
  const [mergeSet, setMergeSet] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"live" | "lab" | "ai" | "chem">("live");
  const [aiText, setAiText] = useState<string | null>(null);
  const [chat, setChat] = useState<ChatMsg[]>([]);
  const [tab, setTab] = useState<"results" | "edit" | "runs" | "scene">("results");
  const [sceneTab, setSceneTab] = useState<"objects" | "why" | "measure" | "add">("objects");
  const [showPhone, setShowPhone] = useState(false);
  const [showStart, setShowStart] = useState(false);
  const [hfov, setHfov] = useState(62);
  const [source, setSourceState] = useState<SourceView>({ kind: "none" });
  const [version, setVersion] = useState(0);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [placeAsset, setPlaceAsset] = useState<string | null>(null);
  const [measureFocus, setMeasureFocus] = useState<string | null>(null);
  // experiments
  const [run, setRun] = useState<AnyRun | null>(null);
  const [chemText, setChemText] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [answer, setAnswer] = useState<ExpAnswer | null>(null);
  const [where, setWhere] = useState<"auto" | "scene" | "sandbox">("auto");
  const [pinned, setPinned] = useState<string[]>([]);
  const [viewOpts, setViewOpts] = useState<ViewOptions>(DEFAULT_VIEW);
  const [camKey, setCamKey] = useState(0);
  const [show, setShow] = useState<"world" | "camera">("world");
  const [live, setLive] = useState<LiveStatus | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const liveVersion = useRef(0);
  const assetMap = useMemo(() => Object.fromEntries(assets.map((a) => [a.id, a])), [assets]);
  const preview = useRef<HTMLVideoElement | HTMLImageElement | null>(null);
  const stageVideo = useRef<HTMLVideoElement | null>(null);

  const scene = payload?.scene ?? null;
  const removed = useMemo(() => new Set(payload?.interventions.filter((i) => i.type === "remove").map((i) => i.target!) ?? []), [payload]);

  useEffect(() => {
    const poll = () => api.health().then(setHealth).catch(() => setHealth(null));
    poll();
    const iv = setInterval(poll, 5000);
    api.scene().then((p) => { if (p.scene) { setPayload(p); setVersion((v) => v + 1); setBootSceneId(p.scene.scene_id); } }).catch(() => {});
    api.assets().then(setAssets).catch(() => {});
    api.runs().then((r) => setRuns(r.runs)).catch(() => {});
    return () => clearInterval(iv);
  }, []);

  const [bootSceneId, setBootSceneId] = useState<string | null>(null);   // on the server before this page opened
  const [liveRun, setLiveRun] = useState(false);                          // the current run came from Live AR's chat (Studio's stay in Studio)
  const say = (m: string) => { setToast(m); setTimeout(() => setToast((t) => (t === m ? null : t)), 6000); };

  const run_ = useCallback(async <T,>(msg: string, fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(msg); setError(null);
    try { return await fn(); } catch (e) { setError((e as Error).message); return undefined; } finally { setBusy(null); }
  }, []);

  const fetchExplanation = useCallback(async () => {
    setExLoading(true);
    try { setEx(await api.explain(-1, !!health?.claude)); } catch (e) { setError((e as Error).message); } finally { setExLoading(false); }
  }, [health]);

  const applyPayload = useCallback((p: ScenePayload | SimPayload, simChanged: boolean) => {
    setPayload(p);
    setVersion((v) => v + 1);
    if (simChanged) {
      const s = (p as SimPayload).sim ?? null;
      setSim(s); setEx(null);
      if (s) { clock.rate = 1; clock.duration = s.duration; clock.play(s.segments[s.segments.length - 1].t_start); }
      else { clock.duration = 0; clock.set(0); clock.pause(); }
    }
  }, []);

  /** Show a finished experiment: in the real scene (AR) or in its own world. */
  const adoptRun = useCallback((r: AnyRun, scenePayload?: ScenePayload | null) => {
    setRun(r); setAnswer(null); setTab("results");
    if (r.domain === "circuit" || r.domain === "gravity") {
      setShow("world"); clock.rate = 1; clock.duration = r.duration; clock.play(0);
      return;
    }
    if (r.setting === "scene" && scenePayload) {
      setPayload(scenePayload); setVersion((v) => v + 1); setSim(r.sim); setEx(null); setShow("camera");
    } else setShow("world");
    clock.rate = 1;
    clock.duration = r.duration;
    const segs = r.sim.segments;
    clock.play(Math.max(0, segs[segs.length - 1].t_start - 0.25));
  }, []);

  const onResponse = useCallback((res: ExpResponse) => {
    if (res.kind === "answer") { setAnswer(res); return; }
    if (res.kind === "chem") { setChemText(res.text); setMode("chem"); return; }
    if (res.kind === "ml") { setAiText(mlSentence(res.text)); setMode("ai"); return; }
    if (res.kind === "unsupported") { setAnswer({ answer: res.message, rows: [] }); return; }
    setRuns(res.runs);
    adoptRun(res.run, res.scene_payload);
    if (res.run.changes.length) say(`Re-ran with: ${res.run.changes.join(" · ")}`);
  }, [adoptRun]);

  const submit = (text: string) => run_(`Simulating “${text}”…`, async () => { setLiveRun(false); onResponse(await api.experiment(text, where, pinned)); });

  /** Live AR chat: same engine; the reply is summarised as a chat message (headline numbers + why). */
  const chatSend = (text: string) => {
    const uid = Math.random().toString(36).slice(2);
    setChat((c) => [...c, { id: uid, role: "user", text }]);
    setBusy("Simulating…"); setError(null);
    const inScene = mode === "live" && !!scene;
    (async () => {
      // 1. a command on your mapped objects ("push the mug", "throw the box at the bottle"): acts on the world on screen
      if (inScene) {
        const c = await api.liveCommand(text, selected);
        if (c.handled) {
          if (run) setRun(null);
          setLiveRun(false);
          clock.set(clock.duration);
          applyPayload(c, true);
          const s = c.sim;
          setChat((m) => [...m, { id: uid + "r", role: "ai", text: s ? handReport({ ivs: [], note: c.say, target: c.target, speed: 0, start: c.start }, s, "✓") : c.say }]);
          return null;
        }
      }
      // 2. anything else is an experiment — in Live AR it happens IN your scene (toy-size if it's too big for the desk)
      return api.experiment(text, inScene ? "scene" : "auto", pinned);
    })().then((res) => {
      if (!res) return;
      if (res.kind === "run") setLiveRun(true);
      onResponse(res);
      const id = uid + "r";
      let msg: ChatMsg;
      if (res.kind === "answer") msg = { id, role: "ai", text: res.answer, tiles: res.rows.slice(0, 3), why: res.why?.filter((w) => w !== res.answer) };
      else if (res.kind === "chem") msg = { id, role: "ai", text: "That's chemistry — opening the reaction lab and simulating it with quantum chemistry…" };
      else if (res.kind === "ml") msg = { id, role: "ai", text: "That's about language models — opening the AI Lab, where a real transformer (GPT-2) runs on this Mac so you can watch it think." };
      else if (res.kind === "unsupported") msg = { id, role: "ai", text: res.message };
      else {
        const r = res.run;
        const where_ = isMech(r) ? (r.setting === "scene" ? "in your real scene" : "as a hologram on your table") : r.domain === "circuit" ? "as a circuit hologram" : "as a gravity hologram";
        const toy = isMech(r) ? (r.notes ?? []).filter((n) => /toy-size|Your desk has no|It starts from your table/.test(n)).join(" ") : "";
        msg = { id, role: "ai", text: `Here it is ${where_}. ${toy ? toy + " " : ""}${firstSentences(r.story[0] ?? "", 2)}`, tiles: keyResults(r.results.filter((x) => x.label !== "Agreement with theory")).slice(0, 3),
          results: r.results, why: r.story.slice(1), changes: r.changes, runId: r.id };
      }
      setChat((c) => [...c, msg]);
    }).catch((e) => setChat((c) => [...c, { id: uid + "e", role: "ai", text: `Couldn't run that: ${(e as Error).message}`, error: true }]))
      .finally(() => setBusy(null));
  };
  /** A diagram reading confirmed by the student goes to the right lab. */
  const runReading = (r: DiagramReading, text: string) => {
    if (r.lab === "transformer") { setAiText(mlSentence(text)); setMode("ai"); return; }
    if (r.lab === "chemistry") { setChemText(text); setMode("chem"); return; }
    if (mode === "live") chatSend(text); else submit(text);
  };
  const onReading = (r: DiagramReading) => setChat((c) => [...c, { id: Math.random().toString(36).slice(2), role: "ai",
    text: r.error ? `I couldn't read that photo: ${r.error}` : "", reading: r.error ? undefined : r, error: !!r.error }]);

  const liveScan = (blob: Blob | null) => run_("Mapping your space…", async () => {
    const p = blob ? await api.scan(blob, hfov, "camera", !!health?.claude) : await api.scanRemote(hfov, !!health?.claude);
    resetSimState(); setSelected(null);
    applyPayload(p, false);
    if (run && isMech(run) && run.setting === "scene") setRun(null);
    const n = p.scene?.objects.length ?? 0;
    setChat((c) => c.length && c[c.length - 1].text.startsWith("I can see") ? c : [...c, { id: Math.random().toString(36).slice(2), role: "ai",
      text: n ? `I can see ${n} real object${n === 1 ? "" : "s"}: ${p.scene!.objects.map((o) => o.label).join(", ")}. Try “drop a baseball on the ${p.scene!.objects[0].label}” — or anything else.`
        : "I've mapped the surface. Ask for any experiment and it will appear here." }]);
  });
  const sweepDone = async (job: SweepJob) => {
    const p = await api.scene();
    resetSimState(); setSelected(null);
    applyPayload(p, false);
    const n = p.scene?.objects.length ?? 0;
    setChat((c) => [...c, { id: Math.random().toString(36).slice(2), role: "ai",
      text: `3D scan done: ${job.meshes ?? 0} of ${n} object${n === 1 ? "" : "s"} are now real 3D models from ${job.keyframes} views ` +
        `(${p.scene!.objects.map((o) => o.label).join(", ")}). Open 🧊 3D to look around them — shapes, sizes and masses come from the scan.` }]);
  };
  const rerun = (spec: ExperimentSpec) => run && isMech(run) && run_("Re-simulating with your values…", async () => onResponse(await api.rerun(spec, run.id, where, pinned)));
  const openRun = (id: string) => run_("Loading…", async () => { const r = await api.getRun(id); adoptRun(r.run, r.scene_payload); });
  const getRun = useCallback(async (id: string) => (await api.getRun(id)).run, []);

  function setSource(s: SourceView, fov?: number) { setSourceState(s); if (fov) setHfov(fov); }

  async function capture(): Promise<Blob> {
    if (source.kind === "phone") return (await fetch("/api/remote/latest.jpg")).blob();
    const el = (source.kind !== "image" && stageVideo.current) || preview.current;
    if (!el) throw new Error("no source frame available");
    const w = el instanceof HTMLVideoElement ? el.videoWidth : el.naturalWidth;
    const h = el instanceof HTMLVideoElement ? el.videoHeight : el.naturalHeight;
    if (!w || !h) throw new Error("source not ready yet");
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    c.getContext("2d")!.drawImage(el, 0, 0, w, h);
    return new Promise((res, rej) => c.toBlob((b) => (b ? res(b) : rej(new Error("capture failed"))), "image/jpeg", 0.93));
  }
  const resetSimState = () => { setSim(null); setEx(null); clock.duration = 0; clock.pause(); clock.set(0); };

  const scan = () => run_("Building the 3D world from the camera…", async () => {
    const p = source.kind === "phone" ? await api.scanRemote(hfov, !!health?.claude)
      : await api.scan(await capture(), hfov, source.kind === "camera" ? "camera" : source.kind, !!health?.claude, source.kind === "image" ? source.depth : null);
    resetSimState(); setSelected(null); setMergeSet([]); setShowStart(false); setShow("camera"); setTab("scene");
    applyPayload(p, false);
  });
  const edit = (msg: string, fn: () => Promise<ScenePayload>) => run_(msg, async () => { const p = await fn(); resetSimState(); applyPayload(p, false); return p; });
  const intervene = (type: string, target: string | null, params: Record<string, unknown> = {}) =>
    run_("Simulating…", async () => { clock.set(clock.duration); applyPayload(await api.intervene(type, target, params), true); fetchExplanation(); });
  const onPick = (id: string | null) => {
    if (!id) { setSelected(null); return; }
    if (tool === "remove") { intervene("remove", id); return; }
    if (tool === "merge") { setMergeSet((m) => (m.includes(id) ? m.filter((x) => x !== id) : [...m, id])); return; }
    setSelected(id);
  };
  /** Live AR hands: the gesture became interventions; simulate them in the real scene and report what was measured. */
  const handAct = (a: HandAction) => run_("✋ Simulating your hand…", async () => {
    if (run && !(isMech(run) && run.setting === "scene")) setRun(null);
    clock.set(clock.duration);
    const p = await api.applyPlan(a.ivs, true);
    applyPayload(p, true);
    const s = p.sim;
    if (s) setChat((c) => [...c, { id: Math.random().toString(36).slice(2), role: "ai", text: handReport(a, s) }]);
  });
  /** Live AR: a real-size virtual baseball on a free spot of the desk, to grab and throw at the real things. */
  const addBall = () => run_("Adding a ball…", async () => {
    if (run && !(isMech(run) && run.setting === "scene")) setRun(null);
    clock.set(clock.duration);
    const p = await api.spawnAt("baseball_01", null, true);
    applyPayload(p, true);
    setChat((c) => [...c, { id: Math.random().toString(36).slice(2), role: "ai",
      text: "🎾 A baseball (145 g, 7.4 cm) is on your desk. Turn on ✋ Hands, pinch it, and flick it at one of your things — or drag it with the mouse, or say “throw the baseball at the …”." }]);
  });
  const onPush = (id: string, from: Pt, to: Pt) => intervene("push", id, { pixel_from: from, pixel_to: to });
  const onPlace = (pt: Pt) => {
    if (tool === "place" && placeAsset) run_("Placing…", async () => { clock.set(clock.duration); applyPayload(await api.spawnAt(placeAsset, pt), true); setTool("select"); setPlaceAsset(null); });
  };
  const onAdd = (pt: Pt) => edit("Segmenting…", async () => { const r = await api.click(pt[0], pt[1]); setSelected(r.id); return r; });

  // live sync: the phone keeps the 3D world current; changes re-run the active real-scene experiment
  useEffect(() => {
    const iv = setInterval(async () => {
      try {
        const st = await api.liveStatus();
        setLive(st);
        if (st.version !== liveVersion.current) {
          const first = liveVersion.current === 0;
          liveVersion.current = st.version;
          if (first && st.version > 0 && scene) return;
          const p = await api.scene();
          const sc = p.scene;
          const fresh = !!sc && sc.scene_id !== scene?.scene_id;       // not the one this page just mapped itself
          if (sc) { applyPayload(p, false); setShowStart(false); }
          if (sc && fresh && sc.source.startsWith("phone") && !st.rerun) setChat((c) => [...c, { id: Math.random().toString(36).slice(2), role: "ai",
            text: sc.objects.length ? `📱 Your phone mapped the space. I can see ${sc.objects.length} real object${sc.objects.length === 1 ? "" : "s"}: ${sc.objects.map((o) => o.label).join(", ")}. Try “drop a baseball on the ${sc.objects[0].label}” — or turn on ✋ Hands.`
              : "📱 Your phone mapped the surface. Ask for any experiment and it will appear here." }]);
          if (st.rerun) { const r = await api.getRun(st.rerun); adoptRun(r.run, r.scene_payload); }
          else if (!run || (isMech(run) && run.setting === "scene")) { setSim(null); setShow("camera"); }
          const parts = [st.added.length ? `+ ${st.added.join(", ")}` : "", st.removed.length ? `− ${st.removed.join(", ")}` : ""].filter(Boolean).join("  ");
          say(`Real scene updated${parts ? `: ${parts}` : ""} (${st.frames_fused} frame${st.frames_fused === 1 ? "" : "s"} fused, ${((st.last_ms ?? 0) / 1000).toFixed(1)} s)${st.rerun ? " — experiment re-run" : ""}`);
        }
      } catch { /* backend restarting */ }
    }, 1500);
    return () => clearInterval(iv);
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.code === "Space" && (sim || run)) { e.preventDefault(); clock.playing ? clock.pause() : clock.play(); }
      if ((e.key === "z" || e.key === "Z") && (e.metaKey || e.ctrlKey) && scene) {
        e.preventDefault();
        run_("Undoing…", async () => applyPayload(await (e.shiftKey ? api.redo() : api.undo()), true));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const frameKey = `${scene?.scene_id ?? "none"}-${version}`;
  const mech = run && isMech(run) ? run : null;
  const otherRun = run && !isMech(run) ? run : null;
  const worldView = (mech && mech.setting !== "scene" && (show === "world" || !scene)) || (otherRun && (show === "world" || !scene));
  const playDuration = otherRun ? otherRun.duration : worldView ? mech!.duration : sim?.duration ?? 0;
  const playStart = otherRun ? 0 : worldView ? Math.max(0, mech!.sim.segments[mech!.sim.segments.length - 1].t_start - 0.25) : sim ? sim.segments[sim.segments.length - 1].t_start : 0;
  const playFx: FxEvent[] = otherRun?.domain === "gravity"
    ? otherRun.events.map((e) => ({ t: (e.t / Math.max(1, otherRun.sim_seconds)) * otherRun.duration, kind: "hit", pos: e.pos, force: e.energy, a: e.a, scale: 1 }))
    : worldView && mech ? mech.fx : mech?.setting === "scene" ? mech.fx : [];
  const phoneOn = !!live?.phone.connected;

  const top = (
    <header>
      <h1 className="brand"><span className="logo">◎</span> Causal <span className="brand-sub">World Lab</span></h1>
      <div className="seg-toggle modes">
        <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>◉ Live AR</button>
        <button className={mode !== "live" ? "on" : ""} onClick={() => mode === "live" && setMode("lab")}>Studio</button>
      </div>
      {mode !== "live" && (
        <div className="seg-toggle studio-tabs" title="Studio: every lab in one place">
          <button className={mode === "lab" ? "on" : ""} onClick={() => setMode("lab")}>⚛ Physics</button>
          <button className={mode === "chem" ? "on" : ""} onClick={() => setMode("chem")}>🧪 Chemistry</button>
          <button className={mode === "ai" ? "on" : ""} onClick={() => setMode("ai")}>🧠 AI Lab</button>
        </div>
      )}
      <div className="grow" />
      {phoneOn && mode !== "live" && (
        <span className={`pill ${live!.state === "scanning" ? "warn" : "ok"}`} title="The phone is streaming; it rescans automatically when the scene changes and it's held steady">
          📱 {live!.state === "scanning" ? "updating 3D…" : live!.phone.meta?.steady ? "steady" : "moving"} · {live!.phone.fps} fps
          <label className="check" style={{ marginLeft: 6 }}><input type="checkbox" checked={live!.enabled}
            onChange={(e) => api.liveRescan(hfov, e.target.checked).then(setLive)} />live sync</label>
        </span>
      )}
      <span className={`dot ${health?.models === "ready" ? "ok" : ""}`} title={`vision models: ${health?.models ?? "offline"} (${health?.device ?? "?"})`} />
      <span className="small muted" title="free on-device language/vision model">AI {health?.claude ? "Claude" : health?.local_ai === "ready" ? "on this Mac" : health?.local_ai ?? "off"}</span>
      <button onClick={() => setShowPhone(true)}>📱 Connect phone</button>
    </header>
  );
  const phoneModal = showPhone && (
    <ConnectPhone onClose={() => setShowPhone(false)} onUse={() => { setSource({ kind: "phone" }, 66); setShowPhone(false); if (mode !== "live") setShowStart(true); }} />
  );

  if (mode === "live") {
    return (
      <div className="app2 live-mode">
        {top}
        <LiveAR scene={scene} payload={payload} sim={sim} run={liveRun ? run : null} assets={assetMap} source={source} setSource={(x, f) => setSource(x, f)} frameKey={frameKey} staleSceneId={bootSceneId} phoneOn={phoneOn} phoneFps={live?.phone.fps ?? 0}
          selected={selected} onSelect={setSelected}
          busy={busy} messages={chat} onSend={chatSend} onScan={liveScan} onSweepDone={sweepDone} hfov={hfov} onOpenStudio={() => setMode("lab")} onPhone={() => setShowPhone(true)}
          onReplay={() => { clock.rate = 1; clock.play(0); }} onReading={onReading} onRunReading={runReading} onHand={handAct} onBall={addBall} />
        {error && <div className="toast" onClick={() => setError(null)}>{error}</div>}
        {phoneModal}
      </div>
    );
  }

  if (mode === "ai") {
    return <div className="app2 ai-mode">{top}<div className="chem-body"><TransformerLab initial={aiText} /></div>{phoneModal}</div>;
  }

  if (mode === "chem") {
    return <div className="app2 chem-mode">{top}<div className="chem-body"><ChemView initial={chemText} /></div>{phoneModal}</div>;
  }

  const start = (!scene && !run) || showStart;
  return (
    <div className="app2">
      {top}
      <div className="body">
        <section className="center">
          <CommandBar busy={!!busy} hasRun={!!run} hasScene={!!scene} where={where} setWhere={setWhere} onSubmit={submit}
            answer={answer} onCloseAnswer={() => setAnswer(null)} onRunReading={runReading} onPin={(l) => { setPinned((p) => (p.includes(l) ? p : [...p, l])); setTab("results"); }} />
          {(scene || run) && !start && (
            <div className="toolbar">
              {scene && run && (otherRun || mech?.setting !== "scene") && (
                <div className="seg-toggle">
                  <button className={show === "world" ? "on" : ""} onClick={() => setShow("world")}>Experiment world</button>
                  <button className={show === "camera" ? "on" : ""} onClick={() => setShow("camera")}>My real scene</button>
                </div>
              )}
              {scene && !worldView && TOOLS.map((t) => (
                <button key={t.id} className={tool === t.id ? "on" : ""} onClick={() => { setTool(t.id); setMergeSet([]); }} title={t.hint}>{t.label}</button>
              ))}
              {tool === "merge" && !worldView && <button className="primary" disabled={mergeSet.length < 2} onClick={() => edit("Merging…", async () => { const r = await api.merge(mergeSet); setMergeSet([]); return r; })}>Merge {mergeSet.length}</button>}
              <div className="grow" />
              <button onClick={() => setShowStart(true)}>{scene ? "New scan" : "Use a camera"}</button>
            </div>
          )}
          <div className="stage-wrap">
            {!start && worldView && mech && <ExperimentView run={mech} assets={assetMap} opts={viewOpts} resetKey={camKey} />}
            {!start && worldView && otherRun?.domain === "circuit" && <CircuitView run={otherRun} resetKey={camKey} />}
            {!start && worldView && otherRun?.domain === "gravity" && <SpaceView run={otherRun} resetKey={camKey} />}
            {!start && !worldView && scene && (
              <div className="views composite">
                <Stage scene={scene} sim={sim} source={source} liveBackground={true} tool={tool} selected={selected}
                  mergeSet={mergeSet} showFit={false} showGhosts={true} showOutlines={!sim} frameKey={frameKey}
                  onPick={onPick} onPush={onPush} onAdd={onAdd} sourceVideoRef={(v) => (stageVideo.current = v)}
                  assets={assetMap} onPlace={onPlace} showVectors={viewOpts.arrows} />
              </div>
            )}
            {start && (
              <div className="start">
                <div className="hero-text">
                  <h2>Say any experiment. Watch it happen. See why.</h2>
                  <p>Type or speak anything — “two blue whales fight”, “drop a hammer and a feather on the Moon”, “a man pulls a truck” — and get a full 3D physics
                    simulation with real sizes and masses, every measurement you want, and the reasons behind the result. Point a camera (or your phone) at a table to run experiments in your real world.</p>
                </div>
                <Source source={source} setSource={setSource} hfov={hfov} setHfov={setHfov} previewRef={(v) => (preview.current = v)}
                  onScan={scan} busy={!!busy} scanned={!!scene} hasCleanPlate={!!payload?.has_clean_plate} onPhone={() => setShowPhone(true)}
                  onCleanPlate={() => run_("Saving clean plate…", async () => { await api.cleanPlate(await capture()); setPayload(await api.scene()); })}
                  onDropCleanPlate={() => run_("…", async () => { await api.dropCleanPlate(); setPayload(await api.scene()); })} />
                {(scene || run) && <button className="link" onClick={() => setShowStart(false)}>Back</button>}
              </div>
            )}
            {busy && <div className="busy-veil"><div className="spinner" /><span>{busy}</span></div>}
            {toast && <div className="toast" onClick={() => setToast(null)}>{toast}</div>}
          </div>
          {!start && (worldView || sim) && !(otherRun?.domain === "circuit" && !otherRun.dynamic) && (
            <Player duration={playDuration} fx={playFx} opts={worldView && mech ? viewOpts : null} setOpts={setViewOpts}
              onResetCamera={() => setCamKey((k) => k + 1)} start={playStart} />
          )}
          <div className="statusline small">
            {error && <span className="error">{error} <button className="link" onClick={() => setError(null)}>×</button></span>}
            {!error && !busy && scene && !worldView && <span className="muted">{TOOLS.find((t) => t.id === tool)?.hint ?? ""}</span>}
          </div>
        </section>

        {(run || scene) && (
          <aside className="drawer">
            <nav className="tabs">
              {run && <button className={tab === "results" ? "on" : ""} onClick={() => setTab("results")}>Results</button>}
              {run && <button className={tab === "edit" ? "on" : ""} onClick={() => setTab("edit")}>Change &amp; re-run</button>}
              {run && <button className={tab === "runs" ? "on" : ""} onClick={() => setTab("runs")}>Runs ({runs.length})</button>}
              {scene && <button className={tab === "scene" ? "on" : ""} onClick={() => setTab("scene")}>Real scene</button>}
            </nav>
            <div className="tab-body">
              {mech && tab === "results" && <ResultsTab run={mech} pinned={pinned} setPinned={setPinned} />}
              {mech && tab === "edit" && <EditTab run={mech} busy={!!busy} hasScene={!!scene} onRerun={rerun} />}
              {otherRun && tab === "results" && <DomainResults run={otherRun} pinned={pinned} setPinned={setPinned} />}
              {otherRun && tab === "edit" && <DomainEdit run={otherRun} busy={!!busy} onSay={submit} />}
              {run && tab === "runs" && <RunsTab runs={runs} current={run} onOpen={openRun} getRun={getRun} />}
              {scene && (tab === "scene" || !run) && (
                <>
                  <div className="seg-toggle small">
                    {([["objects", "Objects"], ["why", "Why"], ["measure", "Graphs"], ["add", "Add model"]] as const).map(([k, l]) => (
                      <button key={k} className={sceneTab === k ? "on" : ""} onClick={() => setSceneTab(k)}>{l}</button>
                    ))}
                  </div>
                  {sceneTab === "objects" && (
                    <>
                      <Inspector scene={scene} selected={selected} onSelect={setSelected} removed={removed} busy={!!busy}
                        depthScale={payload?.depth_scale ?? 1}
                        onPatch={(id, body) => edit("Rebuilding…", () => api.patch(id, body))}
                        onDiscard={(id) => edit("Rebuilding…", () => api.discard(id))}
                        onCalibrate={(id, dim, m) => edit("Rescaling world…", () => api.calibrate(id, dim, m))}
                        onIntervene={intervene} />
                      <section className="panel">
                        <h3>Undo &amp; gravity</h3>
                        <div className="row wrap">
                          <button disabled={!payload?.interventions.length || !!busy} onClick={() => run_("Undoing…", async () => applyPayload(await api.undo(), true))}>Undo</button>
                          <button disabled={!payload?.can_redo || !!busy} onClick={() => run_("Redoing…", async () => applyPayload(await api.redo(), true))}>Redo</button>
                          <button disabled={!payload?.interventions.length || !!busy} onClick={() => run_("Resetting…", async () => applyPayload(await api.reset(), true))}>Start over</button>
                        </div>
                        <div className="row wrap">
                          {[["Earth", 9.81], ["Mars", 3.71], ["Moon", 1.62], ["Jupiter", 24.8]].map(([n, g]) => (
                            <button key={n} disabled={!!busy} onClick={() => intervene("set_gravity", null, { g })}>{n}</button>
                          ))}
                        </div>
                      </section>
                      {scene.warnings.length > 0 && (
                        <details className="warnings"><summary>{scene.warnings.length} reconstruction note(s)</summary><ul>{scene.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul></details>
                      )}
                    </>
                  )}
                  {sceneTab === "why" && (
                    <Explanation ex={ex} loading={exLoading} sim={sim} claude={!!health?.claude} useClaude={!!health?.claude}
                      setUseClaude={() => fetchExplanation()} onHover={() => {}} />
                  )}
                  {sceneTab === "measure" && <Measurements sim={sim} scene={scene} focus={measureFocus} setFocus={setMeasureFocus} />}
                  {sceneTab === "add" && assets.length > 0 && (
                    <Library assets={assets} chosen={placeAsset} disabled={!!busy} onChoose={(id) => { setPlaceAsset(id); setTool(id ? "place" : "select"); }} />
                  )}
                  {sceneTab === "objects" && scene && <div className="orbit-mini"><Orbit3D scene={scene} sim={sim} selected={selected} /></div>}
                </>
              )}
            </div>
          </aside>
        )}
      </div>
      {phoneModal}
    </div>
  );
}


/** What the simulator measured after a hand action, in one or two sentences. */
function handReport(a: HandAction, s: Sim, icon = "✋") {
  const seg = s.segments[s.segments.length - 1];
  const f0 = Math.min(s.frames.length - 1, Math.max(0, Math.round(seg.t_start * s.fps)));
  const i = s.object_ids.indexOf(a.target);
  const out = [`${icon} ${a.note}.`];
  const k = s.kinematics[a.target];
  if (i >= 0 && k) {
    const p0 = a.start, p1 = s.frames[s.frames.length - 1][i];
    let top = 0;
    for (let f = f0; f < k.v.length; f++) { const v = k.v[f]; if (v) top = Math.max(top, Math.hypot(v[0], v[1], v[2])); }
    const d = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
    if (top > 0.02 || d > 0.005)
      out.push(`It reached ${top.toFixed(2)} m/s and ${p1[2] < -0.02 ? "fell off the edge" : `ended ${(d * 100).toFixed(1)} cm from where it started`}.`);
  }
  // impacts the hand caused: contacts between the moved object and something else (not the stack's static load)
  // impulse (momentum transferred over the whole contact) is the robust number; the peak depends on contact stiffness
  let peak = 0, who = "", J = 0;
  for (const [pair, rows] of Object.entries(s.pair_forces ?? {})) {
    const [x, y] = pair.split("|");
    if (x !== a.target && y !== a.target) continue;
    const rs = rows.filter((r) => r[0] >= f0);
    for (let i = 0; i < rs.length; i++) {
      if (rs[i][2] <= peak) continue;
      let lo = i, hi = i;                                     // the contact episode around this frame
      while (lo > 0 && rs[lo][0] - rs[lo - 1][0] <= 2) lo--;
      while (hi < rs.length - 1 && rs[hi + 1][0] - rs[hi][0] <= 2) hi++;
      peak = rs[i][2]; who = x === a.target ? y : x;
      // the impact's impulse, minus the resting load if it stays in contact afterwards (that's weight, not the hit)
      const ep = rs.slice(lo, Math.min(hi + 1, lo + 6));
      const rest = hi - lo >= 4 ? rs[Math.min(hi, lo + 8)][1] : 0;
      J = ep.reduce((t, r) => t + Math.max(0, r[3] - rest / s.fps), 0);
    }
  }
  if (peak > 0.05) out.push(`It hit the ${s.labels[who] ?? who}: impulse ${J.toFixed(2)} N·s, peak force ${peak.toFixed(0)} N.`);
  return out.join(" ");
}

function firstSentences(t: string, n: number) {
  const parts = t.split(/(?<=[.!?])\s+(?=[A-Z])/);       // sentence ends, not decimal points
  return parts.slice(0, n).join(" ").trim();
}
const KEY = [/acceleration down the ramp/i, /^Acceleration \(theory/i, /peak collision force/i, /speed just before impact/i, /impact speed/i, /top speed$/i, /period \(measured/i, /landed earlier/i, /tension \(average/i,
  /current from the battery/i, /brightness/i, /megatons/i, /jump height$/i, /terminal speed \(measured/i, /energy lost/i, /restitution/i, /moved$/i,
  /launch speed/i, /time to reach/i, /orbital speed/i, /distance covered/i];
function keyResults(rows: ResultRow[]) {
  const rank = (r: ResultRow) => { const i = KEY.findIndex((p) => p.test(r.label)); return i < 0 ? 99 : i; };
  return [...rows].map((r, i) => ({ r, i })).sort((a, b) => rank(a.r) - rank(b.r) || a.i - b.i).map((x) => x.r);
}

/** The sentence to feed the transformer: quoted text or text after a colon, else a classic example. */
function mlSentence(t: string) {
  const q = t.match(/["“']([^"”']{4,})["”']/) ?? t.match(/:\s*(.{4,})$/);
  return q ? q[1].trim() : "The Eiffel Tower is located in the city of";
}
