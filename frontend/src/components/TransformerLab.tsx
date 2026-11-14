/* AI Lab: watch a real transformer (GPT-2, running on this Mac) process your sentence.
 *
 * 3D: the 12 layers stacked bottom-to-top, one glowing node per token per layer (size = how big that
 * token's vector is). Arcs show real attention — which earlier words each word looks at — for the layer
 * the signal is passing through. At the top, the model's next-word guesses with their probabilities.
 * Panels: what's happening and why, next-word chart, the answer forming layer by layer (logit lens), a map
 * of all 144 heads (click to switch heads off), and experiments (hide a word, temperature) — every
 * experiment compared with the unmodified run. */
import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../api";
import type { TransformerRun } from "../types";

const PRESETS = ["The Eiffel Tower is located in the city of", "The capital of Japan is", "Once upon a time there was a little",
  "She opened the fridge and took out a cold", "1, 2, 3, 4, 5,"];
const DX = 1.15, DY = 1.0;

export default function TransformerLab({ initial }: { initial?: string | null }) {
  const [text, setText] = useState(initial || PRESETS[0]);
  const [run, setRun] = useState<TransformerRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [layer, setLayer] = useState(12);          // the layer the signal has reached (0 = embeddings)
  const [head, setHead] = useState<number | "all">("all");
  const [playing, setPlaying] = useState(false);
  const [tab, setTab] = useState<"why" | "next" | "lens" | "heads">("why");
  const [ablate, setAblate] = useState<number[][]>([]);
  const [mask, setMask] = useState<number[]>([]);
  const [temp, setTemp] = useState(1);
  const [sink, setSink] = useState(false);

  const go = async (t = text, a = ablate, m = mask, tp = temp, animate = true) => {
    setBusy(true); setErr(null);
    try {
      const r = await api.transformer(t, a, m, tp);
      setRun(r);
      if (animate) { setLayer(0); setPlaying(true); }
    } catch (e) { setErr((e as Error).message); } finally { setBusy(false); }
  };
  useEffect(() => { go(initial || PRESETS[0]); }, [initial]);   // eslint-disable-line react-hooks/exhaustive-deps
  // the signal rises one layer at a time
  useEffect(() => {
    if (!playing || !run) return;
    if (layer >= run.n_layers) { setPlaying(false); return; }
    const t = setTimeout(() => setLayer((l) => l + 1), 520);
    return () => clearTimeout(t);
  }, [playing, layer, run]);

  const toggleHead = (l: number, h: number) => {
    const has = ablate.some(([a, b]) => a === l && b === h);
    const nx = has ? ablate.filter(([a, b]) => !(a === l && b === h)) : [...ablate, [l, h]];
    setAblate(nx); go(text, nx, mask, temp, false);
  };
  const toggleMask = (i: number) => {
    if (!run || i >= run.tokens.length - 1) return;
    const nx = mask.includes(i) ? mask.filter((x) => x !== i) : [...mask, i];
    setMask(nx); go(text, ablate, nx, temp, false);
  };
  const reset = () => { setAblate([]); setMask([]); setTemp(1); go(text, [], [], 1, false); };

  return (
    <div className="ai-lab">
      <div className="lab-bar glass">
        <form className="cmd-row" onSubmit={(e) => { e.preventDefault(); setAblate([]); setMask([]); go(text, [], [], temp); }}>
          <input className="grow cmd-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a sentence — watch a real language model predict the next word" />
          <button className="primary narrow-btn" disabled={busy}>{busy ? "Thinking…" : "Run GPT-2"}</button>
        </form>
        <div className="row wrap small">
          {PRESETS.map((p) => <button key={p} className="chip" onClick={() => { setText(p); setAblate([]); setMask([]); go(p, [], [], temp); }}>{p}</button>)}
        </div>
      </div>
      {err && <div className="error small">{err}</div>}
      <div className="ai-body">
        <section className="ai-stage glass">
          {run && <Scene run={run} layer={layer} head={head} sink={sink} />}
          {run && <div className="xhud">
            <span className="pill dark">{run.model}</span>
            <span className="pill dark">{run.params} parameters · {run.n_layers} layers × {run.n_heads} heads</span>
            {run.compute_ms !== undefined && <span className="pill dark">{run.compute_ms} ms</span>}
          </div>}
          {busy && <div className="live-busy"><div className="spinner" /><span>Running the model…</span></div>}
          {run && (
            <div className="ai-controls">
              <button className="btn-glow small-btn" onClick={() => { setLayer(0); setPlaying(true); }}>▶ Watch it think</button>
              <label className="ai-slider">Layer <b>{layer === 0 ? "embeddings" : layer}</b>
                <input type="range" min={0} max={run.n_layers} value={layer} onChange={(e) => { setPlaying(false); setLayer(+e.target.value); }} /></label>
              <label className="ai-slider">Head
                <select value={String(head)} onChange={(e) => setHead(e.target.value === "all" ? "all" : +e.target.value)}>
                  <option value="all">all (average)</option>
                  {Array.from({ length: run.n_heads }, (_, i) => <option key={i} value={i}>head {i + 1}</option>)}
                </select></label>
              <label className="check" title="Many heads park attention on the first token when they have nothing to do; hiding those links shows the meaningful ones">
                <input type="checkbox" checked={sink} onChange={(e) => setSink(e.target.checked)} />show attention to the first token (the "sink")</label>
            </div>
          )}
        </section>
        <aside className="ai-side glass">
          <nav className="tabs">
            {([["why", "What's happening"], ["next", "Next word"], ["lens", "Layer by layer"], ["heads", "Heads & experiments"]] as const).map(([k, l]) =>
              <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{l}</button>)}
          </nav>
          <div className="tab-body">
            {run && tab === "why" && (
              <section className="panel why">
                {run.story.map((p, i) => <p key={i} className={i === 0 && run.baseline_next ? "exp-line" : ""}>{p}</p>)}
              </section>
            )}
            {run && tab === "next" && <NextWords run={run} />}
            {run && tab === "lens" && <Lens run={run} />}
            {run && tab === "heads" && (
              <>
                <section className="panel">
                  <h3>Hide a word from the others</h3>
                  <p className="muted small">Click a word: the other words can no longer attend to it. Watch what the model predicts without it.</p>
                  <div className="token-chips">
                    {run.tokens.map((t, i) => <button key={i} className={`tok ${mask.includes(i) ? "masked" : ""}`} disabled={busy || i === run.tokens.length - 1}
                      onClick={() => toggleMask(i)} title={i === run.tokens.length - 1 ? "the word being predicted from" : "hide / show"}>{t.replace(/ /g, "␣")}</button>)}
                  </div>
                </section>
                <section className="panel">
                  <h3>Temperature <span className="muted small">{temp.toFixed(2)}</span></h3>
                  <input type="range" min={0.1} max={3} step={0.05} value={temp} onChange={(e) => setTemp(+e.target.value)}
                    onPointerUp={() => go(text, ablate, mask, temp, false)} onKeyUp={() => go(text, ablate, mask, temp, false)} />
                  <p className="muted small">Low = confident and repetitive; high = surprising and random.</p>
                </section>
                <HeadMap run={run} ablate={ablate} onToggle={toggleHead} busy={busy} />
                {(ablate.length > 0 || mask.length > 0 || temp !== 1) && <button onClick={reset}>Undo all experiments</button>}
              </>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- 3D
function attnFor(run: TransformerRun, l: number, head: number | "all") {
  const A = run.attention[l];
  if (head !== "all") return A[head];
  const T = run.tokens.length;
  const out = Array.from({ length: T }, () => new Array(T).fill(0));
  for (const H of A) for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) out[i][j] += H[i][j] / A.length;
  return out;
}
const pos = (run: TransformerRun, i: number, l: number) => new THREE.Vector3((i - (run.tokens.length - 1) / 2) * DX, l * DY, 0);

function Scene({ run, layer, head, sink }: { run: TransformerRun; layer: number; head: number | "all"; sink: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  return (
    <>
      <Canvas dpr={[1, 2]} camera={{ position: [0, 6, 22], fov: 42 }} onCreated={({ gl, scene }) => { gl.toneMapping = THREE.ACESFilmicToneMapping; scene.background = null; }}
        gl={{ alpha: true, antialias: true }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[0, 14, 8]} intensity={60} color="#a5b4fc" />
        <pointLight position={[-8, 2, 6]} intensity={30} color="#38bdf8" />
        <Rig run={run} />
        <Nodes run={run} layer={layer} />
        <Arcs run={run} layer={layer} head={head} sink={sink} />
        <Slabs run={run} layer={layer} />
        <Labels run={run} host={host} layer={layer} />
      </Canvas>
      <div className="xlabels" ref={host} />
    </>
  );
}

function Rig({ run }: { run: TransformerRun }) {
  const { camera, gl } = useThree();
  const ctl = useRef<OrbitControls | null>(null);
  useEffect(() => {
    const w = run.tokens.length * DX;
    const d = Math.max(21, w * 1.35);
    camera.position.set(w * 0.12, run.n_layers * DY * 0.62, d);
    const oc = new OrbitControls(camera, gl.domElement);
    oc.target.set(0, run.n_layers * DY * 0.43, 0);          // leave room for the word labels above the control bar oc.enableDamping = true; oc.autoRotate = true; oc.autoRotateSpeed = 0.35;
    oc.update(); ctl.current = oc;
    return () => oc.dispose();
  }, [run.tokens.length, camera, gl, run.n_layers]);
  useFrame(() => ctl.current?.update());
  return null;
}

function Nodes({ run, layer }: { run: TransformerRun; layer: number }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const T = run.tokens.length, L = run.n_layers;
  const med = useMemo(() => run.norms.map((row) => { const s_ = [...row].sort((a, b) => a - b); return s_[Math.floor(s_.length / 2)] || 1; }), [run]);
  const col = useMemo(() => new THREE.Color(), []);
  useEffect(() => {
    const m = mesh.current; if (!m) return;
    const o = new THREE.Object3D();
    let k = 0;
    for (let l = 0; l <= L; l++) for (let i = 0; i < T; i++, k++) {
      const n = Math.min(2.2, run.norms[l][i] / med[l]);      // relative to the layer's typical token
      const r = 0.1 + 0.11 * n;
      o.position.copy(pos(run, i, l)); o.scale.setScalar(l <= layer ? r : r * 0.55); o.updateMatrix();
      m.setMatrixAt(k, o.matrix);
      const hue = 0.58 - 0.5 * (l / L);                   // cyan at the bottom -> magenta/amber at the top
      col.setHSL(hue, 0.95, l <= layer ? (run.mask.includes(i) ? 0.22 : 0.58) : 0.14);
      m.setColorAt(k, col);
    }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [run, layer, med, col, T, L]);
  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, (L + 1) * T]}>
      <sphereGeometry args={[1, 24, 16]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

/** Attention arcs into the current layer (bright) and the layers below (faint). */
function Arcs({ run, layer, head, sink }: { run: TransformerRun; layer: number; head: number | "all"; sink: boolean }) {
  const geo = useMemo(() => {
    const P: number[] = [], C: number[] = [];
    const T = run.tokens.length;
    for (let l = 1; l <= Math.min(layer, run.n_layers); l++) {
      const A = attnFor(run, l - 1, head);
      const focus = l === layer;
      for (let i = 0; i < T; i++) for (let j = 0; j <= i; j++) {
        const w = A[i][j];
        if (!sink && j === 0 && i > 0) continue;                 // attention parked on the first token ("sink") hides the interesting links
        if (w < (focus ? 0.04 : 0.15)) continue;
        const a = pos(run, j, l - 1), b = pos(run, i, l);
        const c = a.clone().lerp(b, 0.5).add(new THREE.Vector3(0, 0, 0.6 + 0.25 * (i - j)));
        const curve = new THREE.QuadraticBezierCurve3(a, c, b);
        const pts = curve.getPoints(16);
        const k = (focus ? 1 : 0.18) * Math.min(1, w * 1.6);
        const color = new THREE.Color().setHSL(0.55 - 0.45 * (l / run.n_layers), 0.9, 0.6).multiplyScalar(k);
        for (let s = 0; s < pts.length - 1; s++) {
          P.push(pts[s].x, pts[s].y, pts[s].z, pts[s + 1].x, pts[s + 1].y, pts[s + 1].z);
          C.push(color.r, color.g, color.b, color.r, color.g, color.b);
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(P, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(C, 3));
    return g;
  }, [run, layer, head, sink]);
  return <lineSegments geometry={geo}><lineBasicMaterial vertexColors transparent opacity={1} blending={THREE.AdditiveBlending} depthWrite={false} /></lineSegments>;
}

function Slabs({ run, layer }: { run: TransformerRun; layer: number }) {
  const w = run.tokens.length * DX + 0.8;
  return (
    <>
      {Array.from({ length: run.n_layers + 1 }, (_, l) => (
        <mesh key={l} position={[0, l * DY - 0.02, -0.4]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[w, 0.9]} />
          <meshBasicMaterial color={l === layer ? "#38bdf8" : "#6366f1"} transparent opacity={l === layer ? 0.16 : 0.035} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      ))}
    </>
  );
}

function Labels({ run, host, layer }: { run: TransformerRun; host: React.RefObject<HTMLDivElement>; layer: number }) {
  const { camera, size } = useThree();
  const els = useRef<{ toks: HTMLDivElement[]; layers: HTMLDivElement[]; preds: HTMLDivElement }>({ toks: [], layers: [], preds: document.createElement("div") });
  useEffect(() => {
    const h = host.current; if (!h) return;
    const toks = run.tokens.map((t, i) => { const d = document.createElement("div"); d.className = `tok-label ${run.mask.includes(i) ? "masked" : ""}`; d.textContent = t.replace(/ /g, "␣"); h.appendChild(d); return d; });
    const layers = Array.from({ length: run.n_layers + 1 }, (_, l) => { const d = document.createElement("div"); d.className = "layer-label"; d.textContent = l === 0 ? "embeddings" : `layer ${l}`; h.appendChild(d); return d; });
    const preds = document.createElement("div"); preds.className = "pred-cloud"; h.appendChild(preds);
    els.current = { toks, layers, preds };
    return () => { toks.forEach((d) => d.remove()); layers.forEach((d) => d.remove()); preds.remove(); };
  }, [run, host]);
  useEffect(() => {
    const p = els.current.preds;
    const top = run.lens[Math.min(layer, run.n_layers)].top;
    const list = layer >= run.n_layers ? run.next.slice(0, 5) : top;
    p.innerHTML = `<div class="pred-title">${layer >= run.n_layers ? "next word" : `best guess after ${layer === 0 ? "embeddings" : "layer " + layer}`}</div>` +
      list.map((x, k) => `<div class="pred ${k === 0 ? "first" : ""}"><b>${esc(x.token.trim() || "␣")}</b><span>${(x.p * 100).toFixed(1)}%</span><i style="width:${Math.max(3, x.p * 100 / Math.max(list[0].p, 1e-6) * 100)}%"></i></div>`).join("");
  }, [run, layer]);
  const v = useMemo(() => new THREE.Vector3(), []);
  useFrame(() => {
    const toScreen = (p: THREE.Vector3) => { v.copy(p).project(camera); return [((v.x + 1) / 2) * size.width, ((1 - v.y) / 2) * size.height]; };
    run.tokens.forEach((_, i) => { const d = els.current.toks[i]; if (!d) return; const [x, y] = toScreen(pos(run, i, 0).add(new THREE.Vector3(0, -0.55, 0))); d.style.transform = `translate(${x}px, ${y}px) translate(-50%, 0)`; });
    els.current.layers.forEach((d, l) => { const [x, y] = toScreen(pos(run, -0.9, l).add(new THREE.Vector3(-0.2, 0, 0))); d.style.transform = `translate(${x}px, ${y}px) translate(-100%, -50%)`; d.classList.toggle("on", l === layer); });
    const [x, y] = toScreen(pos(run, run.tokens.length - 1, Math.min(layer, run.n_layers)).add(new THREE.Vector3(0.9, 0.2, 0)));
    const card = els.current.preds.getBoundingClientRect();
    const cy = Math.min(Math.max(y, card.height / 2 + 52), size.height - card.height / 2 - 90);        // stay inside the stage, clear of the HUD
    const cx = Math.min(x, size.width - card.width - 16);
    els.current.preds.style.transform = `translate(${cx}px, ${cy}px) translate(0, -50%)`;
  });
  return null;
}
function esc(s: string) { return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!)); }

// ---------------------------------------------------------------- panels
function NextWords({ run }: { run: TransformerRun }) {
  const max = Math.max(...run.next.map((x) => x.p), 1e-6);
  const base = new Map((run.baseline_next ?? []).map((x) => [x.token, x.p]));
  return (
    <section className="panel">
      <h3>Next-word probabilities</h3>
      <p className="muted small">Out of all 50,257 tokens it knows — these are the 10 most likely.{run.baseline_next ? " The marker shows the value before your experiment." : ""}</p>
      <div className="bars">
        {run.next.map((x) => (
          <div key={x.token} className="bar-row" title={`${(x.p * 100).toFixed(2)}%${base.has(x.token) ? ` (was ${(base.get(x.token)! * 100).toFixed(2)}%)` : ""}`}>
            <span className="bar-label">{x.token.trim() || "␣"}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${(x.p / max) * 100}%` }} />
              {base.has(x.token) && <span className="bar-was" style={{ left: `${Math.min(100, (base.get(x.token)! / max) * 100)}%` }} />}</span>
            <span className="bar-val">{(x.p * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function Lens({ run }: { run: TransformerRun }) {
  const final = run.next[0].token;
  return (
    <section className="panel">
      <h3>The answer forming, layer by layer</h3>
      <p className="muted small">After each layer, we ask: if the model had to answer right now, what would it say? (the "logit lens")</p>
      <table className="mtable full lens">
        <tbody>
          {run.lens.map((r) => (
            <tr key={r.layer} className={r.top[0].token === final ? "on" : ""}>
              <td className="muted">{r.layer === 0 ? "embed" : `L${r.layer}`}</td>
              {r.top.map((x, k) => <td key={k}><b>{x.token.trim() || "␣"}</b> <span className="muted">{(x.p * 100).toFixed(1)}%</span></td>)}
            </tr>
          ))}
        </tbody>
      </table>
      <h4>How much each part changes the last word</h4>
      <div className="bars">
        {run.mlp_update.map((m, l) => {
          const a = run.attn_update[l], max = Math.max(...run.mlp_update, ...run.attn_update);
          return (
            <div key={l} className="bar-row duo" title={`attention ${a.toFixed(0)} · feed-forward ${m.toFixed(0)}`}>
              <span className="bar-label">L{l + 1}</span>
              <span className="bar-track"><span className="bar-fill" style={{ width: `${(a / max) * 100}%` }} /><span className="bar-fill b2" style={{ width: `${(m / max) * 100}%` }} /></span>
            </div>
          );
        })}
      </div>
      <div className="legend small"><span><i className="sw" /> attention</span><span><i className="sw b2" /> feed-forward (MLP)</span></div>
    </section>
  );
}

function HeadMap({ run, ablate, onToggle, busy }: { run: TransformerRun; ablate: number[][]; onToggle: (l: number, h: number) => void; busy: boolean }) {
  const [hover, setHover] = useState<TransformerRun["heads"][number] | null>(null);
  const kindColor: Record<string, string> = { "previous word": "#38bdf8", "first token (attention sink)": "#6366f1", itself: "#a78bfa",
    "spread out (averaging)": "#334155", "specific words": "#f59e0b" };
  return (
    <section className="panel">
      <h3>All 144 attention heads</h3>
      <p className="muted small">Colour = what each head measurably does on this sentence. Click a head to switch it off and see the effect.</p>
      <div className="headmap" onMouseLeave={() => setHover(null)}>
        {run.heads.map((h) => {
          const off = ablate.some(([a, b]) => a === h.layer && b === h.head);
          return <button key={`${h.layer}-${h.head}`} disabled={busy} className={`hcell ${off ? "off" : ""}`} style={{ background: kindColor[h.kind] }}
            onMouseEnter={() => setHover(h)} onClick={() => onToggle(h.layer, h.head)} aria-label={`layer ${h.layer + 1} head ${h.head + 1}`} />;
        })}
      </div>
      <div className="legend small">{Object.entries(kindColor).map(([k, c]) => <span key={k}><i className="sw" style={{ background: c }} />{k}</span>)}</div>
      <p className="small hover-line">{hover ? `Layer ${hover.layer + 1}, head ${hover.head + 1}: ${hover.kind} — ${(hover.prev * 100).toFixed(0)}% on previous word, ${(hover.first * 100).toFixed(0)}% on first token` : "Hover a head to see what it does."}</p>
    </section>
  );
}
