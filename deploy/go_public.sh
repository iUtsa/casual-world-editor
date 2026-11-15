#!/bin/bash
# Put Casual World Editor online at https://causalworld.pages.dev, served from this Mac.
#   - the landing page is hosted by Cloudflare Pages (stays up when the Mac is off)
#   - the lab, API, live hand and phone camera go through a Cloudflare quick tunnel to this Mac
# Quick tunnels can expire (after the Mac sleeps, Cloudflare answers "Tunnel not found"), so this script checks the
# tunnel every 30 s and, if it stops answering, starts a fresh one and redeploys the site to point at it.
# Usage: deploy/go_public.sh        (run ./restart.sh first; keep this terminal open, Ctrl-C to go offline)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"          # cloudflared lives here (official binary)
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PROJECT=causalworld
PAGES=$ROOT/deploy/pages
LOG=$ROOT/data/tunnel.log
mkdir -p "$ROOT/data"

for i in $(seq 1 600); do   # at login the app takes a minute or two to load its models
  curl -sf -o /dev/null http://127.0.0.1:8000/api/health && break
  [ "$i" -eq 1 ] && echo "waiting for the app on this Mac (./restart.sh, or the background service) ..."
  sleep 2
done
curl -sf -o /dev/null http://127.0.0.1:8000/api/health || { echo "The app isn't running: start it with ./restart.sh"; exit 1; }
command -v cloudflared >/dev/null || { echo "Install cloudflared first: brew install cloudflared"; exit 1; }

TUN=""; CAF=""
trap 'kill $TUN 2>/dev/null; kill $CAF 2>/dev/null || true' EXIT

start_tunnel() {   # a new tunnel to this Mac; its address changes every start
  [ -n "$TUN" ] && kill "$TUN" 2>/dev/null || true
  : > "$LOG"
  cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >"$LOG" 2>&1 &
  TUN=$!
  URL=""
  for _ in $(seq 1 60); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
  done
  [ -n "$URL" ] || { echo "The tunnel didn't start; see $LOG"; return 1; }
  echo "tunnel: $URL"
}

deploy_site() {    # landing page (static) + the pass-through function pointed at the current tunnel
  (cd "$ROOT/frontend" && npm run build >/dev/null)
  rm -rf "$PAGES/public"
  mkdir -p "$PAGES/public"
  cp "$ROOT/frontend/dist/index.html" "$PAGES/public/"
  cp -R "$ROOT/frontend/dist/landing" "$PAGES/public/landing"
  cat > "$PAGES/public/_routes.json" <<'EOF'
{ "version": 1, "include": ["/*"], "exclude": ["/", "/index.html", "/landing/*"] }
EOF
  echo "export const ORIGIN = \"$URL\";" > "$PAGES/functions/_origin.js"
  (cd "$PAGES" &&
    { npx --yes wrangler@3 pages project list 2>/dev/null | grep -q " $PROJECT " ||
      npx --yes wrangler@3 pages project create "$PROJECT" --production-branch main; } &&
    npx --yes wrangler@3 pages deploy public --project-name "$PROJECT" --branch main --commit-dirty=true)
}

start_tunnel
deploy_site

caffeinate -dims &   # stay awake while serving
CAF=$!
echo
echo "Live at https://$PROJECT.pages.dev  (lab: https://$PROJECT.pages.dev/lab/, phone: https://$PROJECT.pages.dev/phone)"
echo "Keep this terminal open. Ctrl-C takes the lab offline (the landing page stays up)."

# watchdog: three failed checks in a row (90 s) -> fresh tunnel + redeploy
fails=0
while true; do
  sleep 30
  if ! curl -sf -o /dev/null http://127.0.0.1:8000/api/health; then
    echo "$(date '+%H:%M:%S') the app on this Mac isn't answering (run ./restart.sh); the site shows 'lab offline' meanwhile"
    continue
  fi
  if curl -sf -o /dev/null --max-time 15 "$URL/api/health" && kill -0 "$TUN" 2>/dev/null; then
    fails=0; continue
  fi
  fails=$((fails + 1))
  [ "$fails" -lt 3 ] && continue
  echo "$(date '+%H:%M:%S') tunnel lost; starting a new one"
  if start_tunnel && deploy_site; then
    fails=0
    echo "$(date '+%H:%M:%S') back online at https://$PROJECT.pages.dev"
  fi
done
