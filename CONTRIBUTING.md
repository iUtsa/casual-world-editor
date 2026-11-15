# Contributing to Causal World Editor

Thanks for helping. A few things keep the project trustworthy:

1. **Measure, then claim.** Physics changes come with a test that checks the physics (a conservation law, a textbook
   formula, a replay matching its live run), not just that code runs. Numbers in the README come from a benchmark.
2. **Stay on-device.** Every feature must work offline, free, on an 8 GB Apple Silicon Mac. Cloud features are optional
   and off by default.
3. **Keep the camera honest.** Never let a model's guess silently shape physics; show what was measured and how sure
   we are.

## Setup

See [Quick start](README.md#-quick-start). Then:

```bash
./dev.sh                                         # backend :8000 + Vite UI :5173 with hot reload
.venv/bin/python -m pytest -q backend/tests      # all tests (~3 min)
cd frontend && npm run typecheck                 # TypeScript
.venv/bin/python backend/tests/e2e/ui_touch.py /tmp/e2e   # browser end-to-end (restart the server afterwards)
```

## Pull requests

- One focused change per PR, with a short description of what changed and how you verified it.
- Include before/after screenshots for UI changes (desktop and phone width).
- New design follows the paper-and-ink system in `frontend/src/theme.css` and `frontend/public/landing/`.
