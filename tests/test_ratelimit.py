"""The hourly submit cap.

Driven with a plain dict and an explicit clock, so nothing here needs Modal and
nothing depends on real time passing. What matters is that the window slides
rather than resetting on the hour, that a rejected request does not push the
next slot further away, and that the stored key cannot be read back as an
address.
"""

import pytest

from modal_app import ratelimit

# The tests below exercise the limiter as it behaves once a cap is configured;
# `JOBS_PER_HOUR` unset means no cap at all, which has its own tests at the end.
LIMIT = ratelimit.PLAN_MAX_JOBS

ADDRESS = "203.0.113.7"


@pytest.fixture
def store():
    return {}


def test_the_first_calls_are_allowed_and_count_down(store):
    key = ratelimit.client_key(ADDRESS)
    left = [ratelimit.check(key, store=store, now=1000.0, limit=LIMIT) for _ in range(LIMIT)]
    assert left == list(range(LIMIT - 1, -1, -1))


def test_one_over_the_cap_raises(store):
    key = ratelimit.client_key(ADDRESS)
    for _ in range(LIMIT):
        ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    with pytest.raises(ratelimit.RateLimited):
        ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)


def test_a_rejected_call_is_not_recorded(store):
    """Otherwise a client hammering the endpoint would never get back in: every
    rejection would push the oldest timestamp forward."""
    key = ratelimit.client_key(ADDRESS)
    for _ in range(LIMIT):
        ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    for _ in range(3):
        with pytest.raises(ratelimit.RateLimited):
            ratelimit.check(key, store=store, now=1500.0, limit=LIMIT)
    assert len(store[key]) == LIMIT


def test_the_window_slides_rather_than_resetting(store):
    key = ratelimit.client_key(ADDRESS)
    for _ in range(LIMIT):
        ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    # One second before the oldest expires, still full; one second after, free.
    assert (
        ratelimit.remaining(key, store=store, now=1000.0 + ratelimit.WINDOW_SEC - 1, limit=LIMIT)
        == 0
    )
    assert ratelimit.remaining(
        key, store=store, now=1000.0 + ratelimit.WINDOW_SEC + 1, limit=LIMIT
    ) == (LIMIT)


def test_retry_after_points_at_the_oldest_slot(store):
    key = ratelimit.client_key(ADDRESS)
    ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    for _ in range(LIMIT - 1):
        ratelimit.check(key, store=store, now=2000.0, limit=LIMIT)
    with pytest.raises(ratelimit.RateLimited) as raised:
        ratelimit.check(key, store=store, now=2000.0, limit=LIMIT)
    assert raised.value.retry_after == int(ratelimit.WINDOW_SEC - 1000.0)


def test_retry_after_is_zero_while_slots_are_free(store):
    key = ratelimit.client_key(ADDRESS)
    ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    assert ratelimit.retry_after(key, store=store, now=1000.0, limit=LIMIT) == 0


def test_retry_after_counts_from_the_oldest_request_not_the_newest(store):
    """A client one minute into its window waits 59, not the full hour."""
    key = ratelimit.client_key(ADDRESS)
    for _ in range(LIMIT):
        ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    wait = ratelimit.retry_after(key, store=store, now=1000.0 + 600, limit=LIMIT)
    assert wait == int(ratelimit.WINDOW_SEC) - 600


def test_remaining_records_nothing(store):
    key = ratelimit.client_key(ADDRESS)
    assert ratelimit.remaining(key, store=store, now=1000.0, limit=LIMIT) == LIMIT
    assert store == {}


def test_different_addresses_get_different_keys():
    assert ratelimit.client_key(ADDRESS) != ratelimit.client_key("198.51.100.9")
    assert ratelimit.client_key(ADDRESS) == ratelimit.client_key(f" {ADDRESS} ")


def test_the_address_is_not_stored(store):
    """Plan §8 item 5: the audit trail is job ids and timestamps, not who
    submitted them."""
    key = ratelimit.client_key(ADDRESS)
    ratelimit.check(key, store=store, now=1000.0, limit=LIMIT)
    assert ADDRESS not in repr(store)


def test_a_missing_address_falls_back_to_one_shared_bucket():
    assert ratelimit.client_key(None) == ratelimit.UNKNOWN_CLIENT
    assert ratelimit.client_key("") == ratelimit.UNKNOWN_CLIENT


def test_the_client_is_the_first_entry_in_x_forwarded_for():
    headers = {"x-forwarded-for": "203.0.113.7, 70.41.3.18, 150.172.238.178"}
    assert ratelimit.address_from_headers(headers) == ADDRESS


def test_the_fallback_is_used_when_the_header_is_absent_or_empty():
    assert ratelimit.address_from_headers({}, "10.0.0.1") == "10.0.0.1"
    assert ratelimit.address_from_headers({"x-forwarded-for": " "}, "10.0.0.1") == "10.0.0.1"
    assert ratelimit.address_from_headers({}, None) is None


def test_prune_drops_only_the_expired_windows(store):
    old, fresh = ratelimit.client_key("198.51.100.1"), ratelimit.client_key("198.51.100.2")
    ratelimit.check(old, store=store, now=1000.0, limit=LIMIT)
    ratelimit.check(fresh, store=store, now=1000.0 + ratelimit.WINDOW_SEC, limit=LIMIT)
    assert ratelimit.prune(store=store, now=1000.0 + ratelimit.WINDOW_SEC + 1) == 1
    assert list(store) == [fresh]


# --- no cap, which is what ships ------------------------------------------


def test_no_cap_is_what_an_unset_variable_means(monkeypatch):
    monkeypatch.delenv(ratelimit.ENV_LIMIT, raising=False)
    assert ratelimit.max_jobs() == ratelimit.NO_LIMIT


@pytest.mark.parametrize("value", ["", "  ", "0", "-3", "off", "five", "5.5"])
def test_only_a_positive_integer_turns_the_cap_on(value):
    """Failing open on a malformed value is deliberate: this is read at import
    time in every container, and a limiter that refuses to load takes the app
    down with it — which is a worse outcome than an uncapped one."""
    assert ratelimit.max_jobs(value) == ratelimit.NO_LIMIT


def test_a_number_turns_it_back_on(monkeypatch):
    monkeypatch.setenv(ratelimit.ENV_LIMIT, str(ratelimit.PLAN_MAX_JOBS))
    assert ratelimit.max_jobs() == ratelimit.PLAN_MAX_JOBS


def test_without_a_cap_nothing_is_counted_and_nothing_is_refused(store, monkeypatch):
    """The bug that took the cap off: every job spent a slot whatever became of
    it, so five *failures* locked the operator out of their own deployment."""
    monkeypatch.delenv(ratelimit.ENV_LIMIT, raising=False)
    key = ratelimit.client_key(ADDRESS)
    for _ in range(LIMIT * 4):
        assert ratelimit.check(key, store=store, now=1000.0) is None
    assert store == {}, "an uncapped limiter should not be growing a Dict nobody reads"
    assert ratelimit.remaining(key, store=store, now=1000.0) is None
    assert ratelimit.retry_after(key, store=store, now=1000.0) == 0


def test_the_browser_is_told_there_is_no_ceiling_rather_than_a_wrong_one(monkeypatch):
    """`remaining` reaches the page as `jobs_remaining`. null is what hides the
    "N lượt còn lại" line; a 0 would read as "you are out of attempts"."""
    monkeypatch.delenv(ratelimit.ENV_LIMIT, raising=False)
    assert ratelimit.remaining(ratelimit.client_key(ADDRESS), store={}) is None
