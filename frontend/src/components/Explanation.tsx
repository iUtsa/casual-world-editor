import { clock } from "../clock";
import type { Explanation as Ex, Sim } from "../types";

interface Props {
  ex: Ex | null; loading: boolean; sim: Sim | null; claude: boolean; useClaude: boolean;
  setUseClaude: (v: boolean) => void; onHover: (ids: string[]) => void;
}

export default function Explanation({ ex, loading, sim, claude, useClaude, setUseClaude, onHover }: Props) {
  const events = sim ? Object.fromEntries(sim.segments.flatMap((s) => s.events).map((e) => [e.id, e])) : {};
  return (
    <section className="panel explain">
      <h3>3 · Why it happened
        <label className="check small right" title={claude ? "" : "Set ANTHROPIC_API_KEY (or `ant auth login`) and restart the backend"}>
          <input type="checkbox" disabled={!claude} checked={useClaude && claude} onChange={(e) => setUseClaude(e.target.checked)} /> Claude
        </label>
      </h3>
      {!sim && <p className="muted small">Intervene on the scene to get a causal explanation grounded in the simulation's event log.</p>}
      {loading && <p className="muted">Reasoning over the event log…</p>}
      {ex && !loading && (
        <>
          <p className="headline">{ex.headline}</p>
          <ol className="chain">
            {ex.causal_chain.map((c, i) => (
              <li key={i} onMouseEnter={() => onHover(c.evidence)} onMouseLeave={() => onHover([])}>
                <div><b>{c.cause}</b> → <b>{c.effect}</b></div>
                <div>{c.mechanism}</div>
                <div className="muted small">{c.principle}</div>
                <div className="chips">
                  {c.evidence.map((id) => {
                    const e = events[id];
                    return (
                      <button key={id} className="chip" title={e ? `${e.type} ${e.subject ?? ""} ${e.other ?? ""}` : "unknown event"}
                        onClick={() => e && (clock.pause(), clock.set(Math.max(0, e.t - 0.05)))}>
                        {id}{e ? ` · ${e.type} @${e.t.toFixed(2)}s` : ""}
                      </button>
                    );
                  })}
                </div>
              </li>
            ))}
          </ol>
          <div className="kv">
            <span>unaffected</span><b>{ex.unaffected}</b>
            <span>graph vs physics</span><b>{ex.graph_vs_physics}</b>
            <span>try next</span><b>{ex.counterfactual}</b>
          </div>
          <div className="conf">
            <span>confidence</span>
            <div className="bar"><div style={{ width: `${ex.confidence * 100}%` }} /></div>
            <b>{Math.round(ex.confidence * 100)}%</b>
          </div>
          <p className="muted small">{ex.confidence_reason}</p>
          <details>
            <summary className="small">Assumptions ({ex.assumptions.length})</summary>
            <ul className="notes">{ex.assumptions.map((a, i) => <li key={i}>{a}</li>)}</ul>
          </details>
          <p className="muted small">source: {ex.source} · {ex.compute_ms} ms</p>
        </>
      )}
    </section>
  );
}
