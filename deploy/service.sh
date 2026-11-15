#!/bin/bash
# Run Casual World Editor as two macOS background services, so it's always on while this Mac is on:
#   com.casualworld.app     the app (localhost:8000, phone camera on :8443)
#   com.casualworld.public  the public site: tunnel + watchdog that replaces an expired tunnel (deploy/go_public.sh)
# Both start at login and restart if they stop. No terminal needs to stay open.
# Usage: deploy/service.sh install | uninstall | status | restart
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
AGENTS="$HOME/Library/LaunchAgents"
UIDN=$(id -u)
PATHS="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

plist() {   # label, command, log
  cat > "$AGENTS/$1.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$ROOT/$2</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>$PATHS</string><key>HOME</key><string>$HOME</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$ROOT/data/$3</string>
  <key>StandardErrorPath</key><string>$ROOT/data/$3</string>
</dict></plist>
EOF
}

case "${1:-status}" in
  install)
    mkdir -p "$AGENTS" "$ROOT/data"
    # stop copies started by hand, so the services own the ports and the tunnel
    pkill -f "deploy/go_public.sh" || true; pkill -f "cloudflared tunnel --no-autoupdate" || true; pkill -f "caffeinate -dims" || true
    pkill -f "backend/serve.py" || true; pkill -f "backend/ai_worker.py" || true
    sleep 3
    plist com.casualworld.app run.sh server.log
    plist com.casualworld.public deploy/go_public.sh go_public.log
    for l in com.casualworld.app com.casualworld.public; do
      launchctl bootout "gui/$UIDN/$l" 2>/dev/null || true
      launchctl bootstrap "gui/$UIDN" "$AGENTS/$l.plist"
    done
    echo "Installed. The app and https://casual-world-editor.pages.dev now start at login and restart on their own."
    echo "Logs: data/server.log, data/go_public.log · Stop for good: deploy/service.sh uninstall"
    ;;
  uninstall)
    for l in com.casualworld.app com.casualworld.public; do
      launchctl bootout "gui/$UIDN/$l" 2>/dev/null || true
      rm -f "$AGENTS/$l.plist"
    done
    pkill -f "cloudflared tunnel --no-autoupdate" || true; pkill -f "caffeinate -dims" || true
    echo "Removed. Start by hand with ./restart.sh and deploy/go_public.sh."
    ;;
  restart)
    launchctl kickstart -k "gui/$UIDN/com.casualworld.app"
    echo "App restarting; the public site waits for it and stays pointed at it."
    ;;
  status)
    for l in com.casualworld.app com.casualworld.public; do
      if launchctl print "gui/$UIDN/$l" >/dev/null 2>&1; then
        echo "$l: $(launchctl print "gui/$UIDN/$l" | awk -F'= ' '/^\tstate/ {print $2; exit}')"
      else
        echo "$l: not installed"
      fi
    done
    printf "local app: "; curl -s -o /dev/null -w "%{http_code}\n" --max-time 5 http://127.0.0.1:8000/api/health || true
    printf "public lab: "; curl -s -o /dev/null -w "%{http_code}\n" --max-time 20 https://casual-world-editor.pages.dev/api/health || true
    ;;
  *) echo "usage: deploy/service.sh install | uninstall | status | restart"; exit 1 ;;
esac
