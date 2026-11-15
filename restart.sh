#!/bin/bash
# Stop any running instance, wait until the ports are really free, start, wait until healthy.
cd "$(dirname "$0")"
if launchctl print "gui/$(id -u)/com.causalworld.app" >/dev/null 2>&1; then   # running as a service (deploy/service.sh)
  pkill -f "backend/ai_worker.py"; launchctl kickstart -k "gui/$(id -u)/com.causalworld.app"
else
  pkill -f "backend/serve.py" ; pkill -f "uvicorn backend.app.main:app" ; pkill -f "backend/ai_worker.py"
  for i in $(seq 1 40); do
    [ $i -eq 16 ] && { pkill -9 -f "backend/serve.py"; pkill -9 -f "uvicorn backend.app.main:app"; pkill -9 -f "backend/ai_worker.py"; }
    if ! lsof -ti tcp:8000 -sTCP:LISTEN >/dev/null 2>&1 && ! lsof -ti tcp:8443 -sTCP:LISTEN >/dev/null 2>&1 && ! lsof -ti tcp:8011 -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    sleep 0.5
  done
  nohup ./run.sh > data/server.log 2>&1 &
fi
for i in $(seq 1 120); do
  s=$(curl -s -m 2 localhost:8000/api/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['models'], d.get('local_ai'))" 2>/dev/null)
  [ "$s" = "ready ready" ] && { echo "healthy: $s"; exit 0; }
  sleep 1
done
echo "not healthy: $s"; tail -20 data/server.log; exit 1
