"""The deploy target: importing this registers every Modal object we ship.

A function is deployed only if the module defining it has been imported by the
time the App is deployed, so `modal deploy -m modal_app.api` would publish the
web endpoint and nothing else — the `storage.cleanup` cron and the
`VoiceConverter` class would silently not exist in the deployment.

Containers still import only the module their own function lives in, so this
does not drag the audio stack into the API image.

    modal deploy -m modal_app.deploy
"""

from . import (  # noqa: F401  (registration)
    api,
    conversion,
    pipeline,
    separation,
    storage,
    tts,
    watermark,
)
from .app import app

__all__ = ["app"]
