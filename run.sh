#!/bin/bash
# http://localhost:8000 on this Mac, https://<LAN IP>:8443/phone for phones (same process, same session)
cd "$(dirname "$0")"
[ -d frontend/dist ] || (cd frontend && npm run build)
exec .venv/bin/python backend/serve.py
