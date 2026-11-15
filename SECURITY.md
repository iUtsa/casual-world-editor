# Security

Casual World Editor runs locally: camera frames, voice and diagrams stay on the machine, and the phone connects over
the local network on HTTPS. The optional public demo (`deploy/go_public.sh`) exposes the app through a Cloudflare
tunnel, so treat it as a demo, not a hardened service: run it only while you are presenting.

## Reporting a vulnerability

Please open a [private security advisory](https://github.com/iUtsa/causal-world-editor/security/advisories/new) on
GitHub instead of a public issue. Include steps to reproduce and the impact. We aim to reply within a week.
