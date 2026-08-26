"""FastAPI web endpoints, served from Modal as a single ASGI app.

Phase 0 exposes only `/health`. `/submit`, `/status/{id}` and `/download/{id}`
arrive in Phase 3 once the pipeline exists.

Deploy: `modal deploy -m modal_app.api`
"""

import modal
from fastapi import FastAPI

from .app import APP_NAME, api_image, app

web = FastAPI(title="voice-convert API", version="0.1.0")


@web.get("/health")
async def health() -> dict:
    """Liveness probe. Deliberately touches no Volume and no GPU."""
    return {"status": "ok", "app": APP_NAME}


@app.function(image=api_image)
@modal.asgi_app()
def api():
    return web
