import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import type { SourceView } from "./Stage";

interface Props {
  source: SourceView;
  setSource: (s: SourceView, hfov?: number) => void;
  hfov: number; setHfov: (v: number) => void;
  previewRef: (v: HTMLVideoElement | HTMLImageElement | null) => void;
  onScan: () => void; onCleanPlate: () => void; hasCleanPlate: boolean; onDropCleanPlate: () => void;
  busy: boolean; scanned: boolean; onPhone: () => void;
}

export default function Source(p: Props) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [synth, setSynth] = useState<{ name: string; title: string; notes: string }[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => { api.synthetic().then(setSynth).catch(() => {}); }, []);

  const stopStream = () => { streamRef.current?.getTracks().forEach((t) => t.stop()); streamRef.current = null; };

  async function startCamera(id?: string) {
    setErr(null);
    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: id ? { exact: id } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      streamRef.current = stream;
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      setDevices(devs);
      setDeviceId(stream.getVideoTracks()[0]?.getSettings().deviceId ?? "");
      p.setSource({ kind: "camera", stream }, 62);
    } catch (e) {
      setErr(`Camera unavailable: ${(e as Error).message}`);
    }
  }

  function onFile(f: File) {
    stopStream();
    const url = URL.createObjectURL(f);
    if (f.type.startsWith("video/")) p.setSource({ kind: "video", videoUrl: url }, p.hfov);
    else p.setSource({ kind: "image", imageUrl: url }, p.hfov);
  }

  async function loadSynth(name: string) {
    stopStream();
    const { blob, hfov } = await api.syntheticImage(name);
    const depth = await api.syntheticDepth(name);
    p.setSource({ kind: "image", imageUrl: URL.createObjectURL(blob), depth, depthLabel: "renderer depth buffer (simulated RGB-D sensor)" }, hfov);
  }

  async function onDepthFile(f: File) {
    p.setSource({ ...p.source, depth: f, depthLabel: `sensor depth: ${f.name}` });
  }

  useEffect(() => () => stopStream(), []);

  return (
    <section className="panel">
      <h3>Choose a camera or picture</h3>
      <div className="row wrap">
        <button className={p.source.kind === "phone" ? "on" : ""} onClick={() => { stopStream(); p.onPhone(); }}>Phone (QR code)</button>
        <button className={p.source.kind === "camera" ? "on" : ""} onClick={() => startCamera(deviceId || undefined)}>This computer's camera</button>
        <label className="button">Image / video file
          <input type="file" accept="image/*,video/*" hidden onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
        </label>
      </div>
      {devices.length > 1 && p.source.kind === "camera" && (
        <select value={deviceId} onChange={(e) => startCamera(e.target.value)}>
          {devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId.slice(0, 8)}</option>)}
        </select>
      )}
      {synth.length > 0 && (
        <div className="row wrap small">
          <span className="muted">Demo scenes:</span>
          {synth.map((s) => <button key={s.name} className="link" title={s.notes} onClick={() => loadSynth(s.name)}>{s.title}</button>)}
        </div>
      )}
      {err && <div className="error">{err}</div>}
      {(p.source.kind === "image") && (
        <div className="row wrap small">
          {p.source.depth ? (
            <><span className="pill ok" title={p.source.depthLabel}>RGB-D</span>
              <span className="muted">{p.source.depthLabel}</span>
              <button className="link" onClick={() => p.setSource({ ...p.source, depth: null, depthLabel: undefined })}>use monocular instead</button></>
          ) : (
            <label className="link button" title="16-bit PNG, millimetres, aligned to the image">+ aligned depth PNG (RGB-D)
              <input type="file" accept="image/png" hidden onChange={(e) => e.target.files?.[0] && onDepthFile(e.target.files[0])} />
            </label>
          )}
        </div>
      )}
      <div className="preview">
        {p.source.kind === "camera" && (
          <video ref={(v) => { if (v && streamRef.current && v.srcObject !== streamRef.current) { v.srcObject = streamRef.current; v.play().catch(() => {}); } p.previewRef(v); }} muted playsInline autoPlay />
        )}
        {p.source.kind === "video" && <video ref={p.previewRef} src={p.source.videoUrl ?? undefined} controls muted playsInline />}
        {p.source.kind === "image" && <img ref={p.previewRef} src={p.source.imageUrl ?? undefined} alt="source" />}
        {p.source.kind === "phone" && <img src="/api/remote/stream" alt="phone camera" />}
      </div>
      <label className="field">Camera field of view (phones and laptops are ~60–70°)
        <span className="row"><input type="range" min={30} max={110} step={0.5} value={p.hfov} onChange={(e) => p.setHfov(+e.target.value)} />
          <b>{p.hfov.toFixed(1)}°</b></span>
      </label>
      <button className="primary" disabled={p.busy || p.source.kind === "none"} onClick={p.onScan}>
        {p.scanned ? "Rescan" : "Scan → 3D world"}
      </button>
      <div className="row wrap small">
        <button className="link" disabled={p.source.kind === "none"} onClick={p.onCleanPlate}
          title="Remove all objects, then capture: removed objects reveal the real table instead of inpainting">
          Capture empty-table plate</button>
        {p.hasCleanPlate && <><span className="pill ok">clean plate set</span><button className="link" onClick={p.onDropCleanPlate}>clear</button></>}
      </div>
    </section>
  );
}
