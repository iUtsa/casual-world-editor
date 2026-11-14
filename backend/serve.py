"""Run the app on http://localhost:8000 (this Mac) and https://<LAN IP>:8443 (phones) from ONE
process, so both share the same session and the models are loaded once."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from backend.app import remote
from backend.app.main import app


async def main():
    key, crt = remote.ensure_cert()
    http = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=remote.HTTP_PORT, log_level="info",
                                         timeout_graceful_shutdown=3))
    https = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=remote.HTTPS_PORT, ssl_keyfile=str(key),
                                          ssl_certfile=str(crt), log_level="warning", lifespan="off",
                                          timeout_graceful_shutdown=3))
    await asyncio.gather(http.serve(), https.serve())


if __name__ == "__main__":
    asyncio.run(main())
