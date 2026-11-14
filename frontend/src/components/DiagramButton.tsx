import { useRef, useState } from "react";
import { api } from "../api";
import type { DiagramReading } from "../types";

/** "Snap the board": read a photo of a diagram (from the live camera or an uploaded image) into an editable sentence. */
export default function DiagramButton({ onRead, grabFrame, label = "📷 Read a diagram", className = "" }: {
  onRead: (r: DiagramReading) => void; grabFrame?: () => Promise<Blob | null>; label?: string; className?: string;
}) {
  const file = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const read = async (b: Blob | null) => {
    setOpen(false);
    if (!b) return;
    setBusy(true);
    try { onRead(await api.diagram(b)); }
    catch (e) { onRead({ kind: "unknown", reading: "", error: (e as Error).message }); }
    finally { setBusy(false); }
  };
  return (
    <span className={`diagram-btn ${className}`}>
      <button type="button" disabled={busy} onClick={() => (grabFrame ? setOpen(!open) : file.current?.click())} title="Photograph a diagram (whiteboard, textbook, slide) and simulate it">
        {busy ? "Reading the diagram…" : label}
      </button>
      {open && (
        <span className="diagram-menu">
          <button type="button" onClick={async () => read(grabFrame ? await grabFrame() : null)}>Use what the camera sees now</button>
          <button type="button" onClick={() => { setOpen(false); file.current?.click(); }}>Upload a photo…</button>
        </span>
      )}
      <input ref={file} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) read(f); }} />
    </span>
  );
}
