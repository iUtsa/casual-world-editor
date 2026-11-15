import { useEffect, useState } from "react";

interface Info { ok: boolean; public?: boolean; reason?: string; ip?: string; https_url?: string; http_url?: string; qr_svg?: string;
  status?: { connected: boolean; fps: number } }

/** QR code + live connection status for using a phone as the camera. */
export default function ConnectPhone({ onClose, onUse }: { onClose: () => void; onUse: () => void }) {
  const [info, setInfo] = useState<Info | null>(null);
  const [st, setSt] = useState<{ connected: boolean; fps: number } | null>(null);
  // opened through a public HTTPS address (the hosted demo): the phone goes there too, from any network
  const local = /^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(location.hostname);
  const q = !local && location.protocol === "https:" ? `?origin=${encodeURIComponent(location.origin)}` : "";
  useEffect(() => { fetch(`/api/connect${q}`).then((r) => r.json()).then(setInfo).catch(() => setInfo({ ok: false, reason: "backend unreachable" })); }, []);
  useEffect(() => {
    const iv = setInterval(() => fetch("/api/remote/status").then((r) => r.json()).then(setSt).catch(() => {}), 1000);
    return () => clearInterval(iv);
  }, []);
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="row"><h3 className="grow">Use your phone as the camera</h3><button className="link" onClick={onClose}>×</button></div>
        {!info ? <p className="muted">…</p> : !info.ok ? <p className="error">{info.reason}</p> : (
          <div className="phone-grid">
            <div className="qr" dangerouslySetInnerHTML={{ __html: info.qr_svg! }} />
            <div className="steps">
              {info.public ? (
                <ol>
                  <li>Scan the code with your phone camera (any network works).</li>
                  <li>Allow the camera. Tap <b>Live to Mac</b> to stream, or <b>Scan</b>.</li>
                </ol>
              ) : (
              <ol>
                <li>Phone on the same Wi-Fi as this Mac.</li>
                <li>Scan the code with the phone camera.</li>
                <li>The browser warns the connection isn't private (this Mac made its own certificate): tap <b>Show details → visit this website</b> / <b>Advanced → Proceed</b>.</li>
                <li>Allow the camera. Tap <b>Live to Mac</b> to stream, or <b>Scan</b>.</li>
              </ol>
              )}
              <p className="small mono">{info.https_url}</p>
              {!info.public && <p className="small muted">No camera permission over plain http? Open {info.http_url} and use “Photo”.</p>}
              <p className={`pill ${st?.connected ? "ok" : ""}`}>{st?.connected ? `phone streaming · ${st.fps} fps` : "waiting for the phone…"}</p>
              <button className="primary narrow-btn" disabled={!st?.connected} onClick={onUse}>Use the phone camera here</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
