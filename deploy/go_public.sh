#!/bin/bash
# Put Causal World Editor online at https://causalworld.pages.dev, served from this Mac.
#   - the landing page is hosted by Cloudflare Pages (stays up when the Mac is off)
#   - the lab, API, live hand and phone camera go through a Cloudflare quick tunnel to this Mac
# Usage: deploy/go_public.sh        (run ./restart.sh first; keep this terminal open, Ctrl-C to go offline)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"          # cloudflared lives here (official binary)
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PROJECT=causalworld
PAGES=$ROOT/deploy/pages
LOG=$ROOT/data/tunnel.log
mkdir -p "$ROOT/data"

curl -sf -o /dev/null http://127.0.0.1:8000/api/health || { echo "The app isn't running: start it with ./restart.sh"; exit 1; }
command -v cloudflared >/dev/null || { echo "Install cloudflared first: brew install cloudflared"; exit 1; }

# 1. the tunnel to this Mac (its address changes every start)
: > "$LOG"
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8000 >"$LOG" 2>&1 &
TUN=$!
trap 'kill $TUN 2>/dev/null; kill $CAF 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$LOG" | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done
[ -n "${URL:-}" ] || { echo "The tunnel didn't start; see $LOG"; exit 1; }
echo "tunnel: $URL"

# 2. the site: landing page (static) + the pass-through function pointed at this tunnel
(cd "$ROOT/frontend" && npm run build >/dev/null)
rm -rf "$PAGES/public"
mkdir -p "$PAGES/public"
cp "$ROOT/frontend/dist/index.html" "$PAGES/public/"
cp -R "$ROOT/frontend/dist/landing" "$PAGES/public/landing"
cat > "$PAGES/public/_routes.json" <<'EOF'
{ "version": 1, "include": ["/*"], "exclude": ["/", "/index.html", "/landing/*"] }
EOF
echo "export const ORIGIN = \"$URL\";" > "$PAGES/functions/_origin.js"
cd "$PAGES"
npx --yes wrangler@3 pages project list 2>/dev/null | grep -q " $PROJECT " ||
  npx --yes wrangler@3 pages project create "$PROJECT" --production-branch main
npx --yes wrangler@3 pages deploy public --project-name "$PROJECT" --branch main --commit-dirty=true

# 3. stay awake and keep serving
caffeinate -dims &
CAF=$!
echo
echo "Live at https://$PROJECT.pages.dev  (lab: https://$PROJECT.pages.dev/lab/, phone: https://$PROJECT.pages.dev/phone)"
echo "Keep this terminal open. Ctrl-C takes the lab offline (the landing page stays up)."
wait $TUN
