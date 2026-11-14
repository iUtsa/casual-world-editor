import { useRef, useState } from "react";

/** Hold-to-talk (or click to start/stop). Captures 16 kHz mono PCM with Web Audio,
 *  sends a WAV to the local Whisper endpoint. No audio leaves the machine. */
export default function VoiceButton({ onText, disabled }: { onText: (t: string) => void; disabled: boolean }) {
  const [state, setState] = useState<"idle" | "rec" | "busy">("idle");
  const [err, setErr] = useState<string | null>(null);
  const rec = useRef<{ ctx: AudioContext; stream: MediaStream; chunks: Float32Array[]; node: ScriptProcessorNode } | null>(null);

  async function start() {
    setErr(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const ctx = new AudioContext({ sampleRate: 16000 });
      const src = ctx.createMediaStreamSource(stream);
      const node = ctx.createScriptProcessor(4096, 1, 1);
      const chunks: Float32Array[] = [];
      node.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
      src.connect(node);
      node.connect(ctx.destination);
      rec.current = { ctx, stream, chunks, node };
      setState("rec");
    } catch (e) {
      setErr(`Microphone unavailable: ${(e as Error).message}`);
    }
  }

  async function stop() {
    const r = rec.current;
    if (!r) return;
    rec.current = null;
    r.node.disconnect();
    r.stream.getTracks().forEach((t) => t.stop());
    const sr = r.ctx.sampleRate;
    await r.ctx.close();
    setState("busy");
    try {
      const n = r.chunks.reduce((a, c) => a + c.length, 0);
      const pcm = new Int16Array(n);
      let o = 0;
      for (const c of r.chunks) for (let i = 0; i < c.length; i++) pcm[o++] = Math.max(-1, Math.min(1, c[i])) * 32767;
      const f = new FormData();
      f.append("audio", wav(pcm, sr), "speech.wav");
      const res = await fetch("/api/transcribe", { method: "POST", body: f });
      const j = await res.json();
      if (!res.ok) throw new Error(j.detail || res.statusText);
      if (j.text) onText(j.text); else setErr("Didn't catch that — try again, a bit louder.");
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setState("idle");
    }
  }

  return (
    <>
      <button type="button" className={`mic ${state}`} disabled={disabled || state === "busy"}
        title="Speak the experiment (local Whisper; audio stays on this machine)"
        onClick={() => (state === "rec" ? stop() : start())}>
        {state === "rec" ? "■ Stop" : state === "busy" ? "…" : "🎙"}
      </button>
      {err && <span className="error small">{err}</span>}
    </>
  );
}

function wav(pcm: Int16Array, sr: number): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  new Int16Array(buf, 44).set(pcm);
  return new Blob([buf], { type: "audio/wav" });
}
