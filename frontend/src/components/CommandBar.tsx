import { useState } from "react";
import type { ExpAnswer } from "../types";
import VoiceButton from "./VoiceButton";
import DiagramButton from "./DiagramButton";
import { ReadingCard } from "./LiveAR";
import type { DiagramReading } from "../types";

const STARTERS = [
  "two baseballs clash into each other",
  "two blue whales fight underwater",
  "a dragon crashes into a castle",
  "drop a hammer and a feather on the Moon from 2 m",
  "a pendulum with a 2 m string pulled back 20 degrees",
  "a man pulls a truck with a rope",
  "kick a soccer ball at a chair really hard",
  "two cars crash head on at 50 km/h",
  "drop a rock and a beach ball in water",
  "a ball on a spring",
];
const FOLLOWUPS = ["what if it was twice as heavy?", "now try it on the Moon", "make it 3 times faster", "how hard did they hit?", "why?"];
const SCENE_STARTERS = ["drop a baseball on it", "roll a steel ball into the mug", "throw a tennis ball at the bottle"];

interface Props {
  busy: boolean; hasRun: boolean; hasScene: boolean;
  where: "auto" | "scene" | "sandbox"; setWhere: (w: "auto" | "scene" | "sandbox") => void;
  onSubmit: (text: string) => void; answer: ExpAnswer | null; onCloseAnswer: () => void; onPin: (label: string) => void;
  onRunReading: (r: DiagramReading, text: string) => void;
}

export default function CommandBar({ busy, hasRun, hasScene, where, setWhere, onSubmit, answer, onCloseAnswer, onPin, onRunReading }: Props) {
  const [text, setText] = useState("");
  const [reading, setReading] = useState<DiagramReading | null>(null);
  const submit = (t: string) => { if (!t.trim() || busy) return; setText(t); onSubmit(t.trim()); };
  const chips = hasRun ? FOLLOWUPS : hasScene ? [...SCENE_STARTERS, ...STARTERS.slice(0, 4)] : STARTERS;
  return (
    <div className="command">
      <form className="cmd-row" onSubmit={(e) => { e.preventDefault(); submit(text); }}>
        <input className="grow cmd-input" value={text} onChange={(e) => setText(e.target.value)} autoFocus
          placeholder={hasRun ? "Ask about it, say “what if…”, or describe a new experiment" : "Describe any experiment: “two elephants fight”, “drop an egg from 3 m”…"} />
        <VoiceButton disabled={busy} onText={(t) => submit(t)} />
        <DiagramButton onRead={setReading} label="📷 Diagram" />
        <button className="primary narrow-btn" disabled={busy || !text.trim()}>{busy ? "Simulating…" : "Run"}</button>
      </form>
      {reading && (
        <div className="plan answer">
          {reading.error ? <span className="error">{reading.error}</span> : <ReadingCard r={reading} onRun={(r, t) => { setReading(null); onRunReading(r, t); }} />}
          <button className="link" onClick={() => setReading(null)}>close</button>
        </div>
      )}
      <div className="row wrap small">
        {hasScene && (
          <div className="seg-toggle where" title="Where should the experiment happen?">
            {([["auto", "Auto"], ["scene", "In my real scene"], ["sandbox", "Imagined world"]] as const).map(([k, l]) => (
              <button key={k} type="button" className={where === k ? "on" : ""} onClick={() => setWhere(k)}>{l}</button>
            ))}
          </div>
        )}
        {chips.map((e) => <button key={e} type="button" className="chip" disabled={busy} onClick={() => submit(e)}>{e}</button>)}
      </div>
      {answer && (
        <div className="plan answer">
          <div className="row">
            {answer.text ? <span className="hero">{answer.text}</span> : answer.value !== undefined && answer.value !== null && <span className="hero">{fmt(answer.value)}<small> {answer.unit}</small></span>}
            <span className="grow">{answer.answer}</span>
            <button className="link" onClick={onCloseAnswer} aria-label="close">×</button>
          </div>
          {answer.why && answer.why.length > 0 && answer.why.join(" ") !== answer.answer && <p className="muted small">Why: {answer.why.join(" ")}</p>}
          {answer.rows.length > 0 && (
            <div className="row wrap small">{answer.rows.map((r) => <button key={r.label} className="chip" onClick={() => onPin(r.label)} title="pin to results">📌 {r.label}: {r.text}</button>)}</div>
          )}
        </div>
      )}
    </div>
  );
}

function fmt(v: number) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2); }
