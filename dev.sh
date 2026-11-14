#!/bin/bash
# Dev: backend with auto-reload on :8000, Vite UI with hot reload on :5173 (proxies /api).
cd "$(dirname "$0")"
.venv/bin/python -m uvicorn backend.app.main:app --port 8000 --reload --reload-dir backend &
BACK=$!
trap "kill $BACK" EXIT
cd frontend && npx vite
