"""Where the browser learns whether this deployment can generate a beat.

The generator shipped once and stayed invisible. `/health` answered
`beat_generator: true`, the flag was on, the image had built — and the page
still showed only the file picker, because the browser's own cross-origin
`GET ${apiBase}/health` can fail on CORS or on a dropped connection and both
land in the same `catch` as an honest "no generator here". A false that means
"blocked" and a false that means "not deployed" are not the same fact, and the
UI had no way to tell them apart.

So the probe moved server side, to `/api/capabilities` — same origin for the
browser, no preflight, no browser cache. These tests hold that shape, and hold
it *apart* from `/api/config`: config is on the upload's critical path and a
slow probe there is a slow upload, which is why the deadline lives on the route
that nothing waits for.
"""

from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
CONFIG_ROUTE = WEB / "app" / "api" / "config" / "route.ts"
CAPABILITIES_ROUTE = WEB / "app" / "api" / "capabilities" / "route.ts"
API_CLIENT = WEB / "lib" / "api.ts"


def _code(source: str) -> str:
    """`source` without its comment lines, which are allowed to say "/health"."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("*", "//", "/*"))
    )


@pytest.fixture(scope="module")
def config() -> str:
    return CONFIG_ROUTE.read_text()


@pytest.fixture(scope="module")
def capabilities() -> str:
    return CAPABILITIES_ROUTE.read_text()


@pytest.fixture(scope="module")
def client() -> str:
    return API_CLIENT.read_text()


def test_the_server_is_what_probes_health(capabilities: str) -> None:
    assert "/health" in capabilities, "the server side is what asks the API what it can do"
    assert "beat_generator" in capabilities, "the API's field name"
    assert "beatGenerator" in capabilities, "the field the browser reads"


def test_the_probe_survives_a_cold_container(capabilities: str) -> None:
    """`api()` sets no `min_containers`, so the first ask of the day is slow.

    The browser-side probe had no deadline at all and was right not to. A
    deadline here is only safe because nothing waits on this route — so it gets
    a generous one, and a second attempt, since the first is what wakes the
    container up.
    """
    assert "HEALTH_TIMEOUT_MS = 12_000" in capabilities
    assert "HEALTH_ATTEMPTS = 2" in capabilities


def test_not_knowing_is_never_an_error(capabilities: str) -> None:
    """A failed probe is `beatGenerator: false`, never a failed request.

    The upload source works without the generator, so there is nothing the
    browser could do with a 5xx here except hide a control it was going to hide
    anyway.
    """
    assert "status: 5" not in capabilities
    assert "return false" in capabilities


def test_config_stays_off_the_probe(config: str) -> None:
    """`submit` waits on `/api/config` before a byte goes up.

    The probe lived here for one commit. A cold Modal container would have put
    its entire wake-up time between pressing the button and the upload starting.
    """
    assert "/health" not in _code(config)
    assert "AbortSignal" not in config
    assert "MODAL_API_URL is not configured" in config


def test_the_browser_does_not_ask_the_api_directly(client: str) -> None:
    assert "/health" not in _code(client), "the health probe belongs to the server now"
    assert 'fetch("/api/capabilities"' in client
