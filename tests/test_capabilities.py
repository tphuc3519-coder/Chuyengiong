"""Where the browser learns whether this deployment can generate a beat.

The generator shipped once and stayed invisible. `/health` answered
`beat_generator: true`, the flag was on, the image had built — and the page
still showed only the file picker, because the browser's own cross-origin
`GET ${apiBase}/health` can fail on CORS or on a dropped connection and both
land in the same `catch` as an honest "no generator here". A false that means
"blocked" and a false that means "not deployed" are not the same fact, and the
UI had no way to tell them apart.

So the probe moved server side, into `/api/config` — the request the page
already makes, same origin, no preflight. These tests hold that shape: the
config route does the asking, and the client does not go back to asking it
from the browser.
"""

from pathlib import Path

import pytest

WEB = Path(__file__).resolve().parent.parent / "web"
CONFIG_ROUTE = WEB / "app" / "api" / "config" / "route.ts"
API_CLIENT = WEB / "lib" / "api.ts"


@pytest.fixture(scope="module")
def route() -> str:
    return CONFIG_ROUTE.read_text()


@pytest.fixture(scope="module")
def client() -> str:
    return API_CLIENT.read_text()


def test_the_config_route_probes_health_itself(route: str) -> None:
    assert "/health" in route, "the server side is what asks the API what it can do"
    assert "beat_generator" in route, "the API's field name"
    assert "beatGenerator" in route, "the field the browser reads"


def test_the_probe_cannot_hold_the_form_open(route: str) -> None:
    """Nothing can be uploaded before this reply lands, so it needs a deadline.

    A Modal container that is scaling from zero answers in seconds; one that is
    wedged never answers at all, and without a timeout the page would wait for
    it before it could so much as show the upload button.
    """
    assert "AbortSignal.timeout" in route
    assert "HEALTH_TIMEOUT_MS" in route


def test_a_missing_api_url_is_still_the_only_way_config_fails(route: str) -> None:
    """A failed probe is `beatGenerator: false`, never a failed config.

    The upload source works without the generator. Turning an unreachable
    `/health` into a 503 here would take the whole form down over a feature the
    user may not even be using.
    """
    assert route.count("status: 503") == 1
    assert "MODAL_API_URL is not configured" in route


def _code(source: str) -> str:
    """`source` without its comment lines, which are allowed to say "/health"."""
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("*", "//", "/*"))
    )


def test_the_browser_does_not_ask_the_api_directly(client: str) -> None:
    assert "/health" not in _code(client), "the health probe belongs to the config route now"


def test_capabilities_comes_from_the_config_request(client: str) -> None:
    assert "export function capabilities()" in client
    assert "beatGenerator" in client
    # One request for both facts: a second round trip is a second thing that
    # can be blocked, cached, or answered stale.
    assert client.count('fetch("/api/config"') == 1
