/* causalworld.pages.dev: the landing page is served by Cloudflare (always online); everything else (the lab, the API,
   the live-hand WebSocket, the phone page, 3D assets) is passed through to the Mac that runs the app, over a
   Cloudflare tunnel. ORIGIN is that tunnel's address, written by deploy/go_public.sh at each start. */
import { ORIGIN } from "./_origin.js";

const OFFLINE = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lab offline · Causal World Editor</title><link rel="icon" href="/landing/favicon.svg" type="image/svg+xml">
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#f4f1e8;color:#1e2622;font:17px/1.55 system-ui,sans-serif;padding:16px;
background-image:radial-gradient(#d6cfbf 1px,transparent 1.2px);background-size:22px 22px}
.card{max-width:520px;background:#fffdf8;border:2px solid #1e2622;border-radius:14px;box-shadow:6px 6px 0 #d3ccc0;padding:28px}
h1{font-size:30px;line-height:1.1;margin:0 0 12px;letter-spacing:-.02em}p{margin:0 0 14px;color:#3a3731}
a{display:inline-block;margin-top:6px;background:#c8f135;color:#1e2622;font-weight:600;text-decoration:none;padding:12px 18px;border:2px solid #1e2622;border-radius:10px;box-shadow:4px 4px 0 #1e2622}</style>
</head><body><div class="card"><img src="/landing/brand/cwe-icon.svg" alt="" width="56" height="56" style="display:block;margin-bottom:18px"><h1>The lab is offline right now.</h1>
<p>Causal World Editor runs entirely on one laptop (free, on-device models), and that laptop isn't serving at the moment.</p>
<p>Try again in a little while, or run it on your own Mac from the quick start.</p>
<a href="/#start">Back to the home page</a></div></body></html>`;

function offline(request) {
  const html = (request.headers.get("Accept") || "").includes("text/html");
  return html
    ? new Response(OFFLINE, { status: 503, headers: { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" } })
    : new Response(JSON.stringify({ detail: "The lab is offline right now." }), { status: 503, headers: { "Content-Type": "application/json" } });
}

export async function onRequest({ request }) {
  if (!ORIGIN) return offline(request);
  const url = new URL(request.url);
  const target = new URL(url.pathname + url.search, ORIGIN);

  // live hand: the WebSocket is passed straight through
  if ((request.headers.get("Upgrade") || "").toLowerCase() === "websocket") {
    try { return await fetch(target.toString(), request); } catch { return offline(request); }
  }

  const hasBody = !["GET", "HEAD"].includes(request.method);
  let res;
  try {
    res = await fetch(target.toString(), {
      method: request.method, headers: request.headers, body: hasBody ? request.body : null, redirect: "manual",
    });
  } catch {
    return offline(request);
  }
  if ([502, 503, 504, 521, 522, 523, 530].includes(res.status) && !res.headers.get("content-type")?.includes("json")) return offline(request);

  // the app's own redirects (e.g. /lab -> /lab/) name the tunnel's host: send visitors to ours instead
  const loc = res.headers.get("Location");
  if (loc) {
    const l = new URL(loc, target);
    if (l.host === target.host) {
      l.protocol = url.protocol; l.host = url.host;
      const h = new Headers(res.headers);
      h.set("Location", l.toString());
      return new Response(res.body, { status: res.status, headers: h });
    }
  }
  return res;
}
