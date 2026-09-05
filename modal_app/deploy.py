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
    beatgen,
    conversion,
    pipeline,
    separation,
    storage,
    training,
    tts,
    watermark,
)
from .app import app

# The beat generator is registered only when the deployment asks for it.
#
# Not squeamishness — arithmetic about blast radius. `modal deploy` builds every
# registered image in one pass, so an image that fails to build does not fail
# its own function, it fails the deploy and takes every unrelated change in the
# same push with it. `beatgen_image` does not build today (see the comment on
# `BEATGEN_REQUIREMENTS`), and while it was imported here unconditionally three
# phases of working code sat undeployed behind it.
#
# So: an image nobody has watched build does not get to hold the deploy hostage.
# `beatgen` is imported above for `enabled()` alone — importing the module does
# not register anything, because the `@app.cls` that would is built inside
# `register()` rather than at module scope.
if beatgen.enabled():
    beatgen.register()

__all__ = ["app"]
