import { useState } from "react";
import type { ResultRow } from "../types";

/** Pairs of (measured, theory) rows: "X (measured)" with "X (theory …)", or "Y" with "Y (theory …)". */
export function theoryPairs(rows: ResultRow[]) {
  const isTheory = (r: ResultRow) => /\(theory|theory:|\(kepler|agreement/i.test(r.label);
  const out: { what: string; measured: ResultRow; theory: ResultRow }[] = [];
  rows.forEach((t, i) => {
    if (!/\(theory|theory:|\(kepler/i.test(t.label) || /agreement/i.test(t.label)) return;
    const stem = t.label.replace(/\s*\((theory|kepler|…).*$/i, "").replace(/^…/, "").trim().toLowerCase();
    let m = stem ? rows.find((r) => !isTheory(r) && r.label.toLowerCase().replace(/\s*\(measured\)/, "").includes(stem)) : undefined;
    if (!m && !stem) m = [...rows.slice(0, i)].reverse().find((r) => !isTheory(r));      // "…(theory: …)" belongs to the row above
    if (m) out.push({ what: t.label, measured: m, theory: t });
  });
  return out;
}

/** "Impact speed (theory, Newton's 2nd law with air drag)" -> Impact speed / Newton's 2nd law with air drag */
function lawOf(p: { measured: ResultRow; theory: ResultRow }) {
  const m = p.theory.label.match(/^…?(.*?)\s*\((?:theory[,:]?\s*)?(.*)\)\s*$/i);
  const what = (m?.[1] || p.measured.label.replace(/\s*\(measured\)/i, "")).trim();
  const law = (m?.[2] || "").replace(/^theory[,:]?\s*/i, "").trim();
  return <><span className="theory-what">{what.charAt(0).toUpperCase() + what.slice(1)}</span>{law && <small className="theory-law">{law}</small>}</>;
}

export function agreementOf(rows: ResultRow[]): number | null {
  const a = rows.find((r) => r.label === "Agreement with theory");
  return a && a.value !== null ? a.value : null;
}

/** "✓ Matches theory 99.9 %" — the simulation checked against the physical law it should obey. */
export default function TheoryCheck({ rows, compact = false }: { rows: ResultRow[]; compact?: boolean }) {
  const [open, setOpen] = useState(false);
  const ag = agreementOf(rows);
  if (ag === null) return null;
  const pairs = theoryPairs(rows);
  const tone = ag >= 99 ? "ok" : ag >= 95 ? "warn" : "bad";
  return (
    <div className={`theory-check ${tone} ${compact ? "compact" : ""}`}>
      <button type="button" className="theory-badge" onClick={() => setOpen(!open)}
        title="The simulation's own measurements compared with the textbook law (solved exactly for the same set-up)">
        <span className="theory-icon">{ag >= 99 ? "✓" : "≈"}</span>
        <span>Matches theory <b>{ag >= 99.995 ? "100" : ag.toFixed(ag >= 99.9 ? 2 : 1)}%</b></span>
        {pairs.length > 0 && <span className="theory-more">{open ? "hide" : "see the check"}</span>}
      </button>
      {open && pairs.length > 0 && (
        <table className="theory-table">
          <thead><tr><th>Quantity</th><th>Simulation</th><th>Theory</th></tr></thead>
          <tbody>
            {pairs.map((p) => (
              <tr key={p.what}>
                <td title={p.theory.why ?? p.theory.label}>{lawOf(p)}</td>
                <td>{p.measured.text}</td>
                <td>{p.theory.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
