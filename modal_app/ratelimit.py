"""Per-client rate limit for `/submit`, stored in `modal.Dict`.

Plan §9 caps the MVP at 5 jobs per hour per IP, to bound GPU spend rather than
to fight abuse — a determined caller can change address, but an accidental loop
in a client cannot quietly run up a bill.

That cap is **off unless `JOBS_PER_HOUR` says otherwise** (see `max_jobs`). It
was charging a slot per job whatever became of it, so an operator debugging
their own deployment ran out of attempts on five *failures* and could not get
back in to fix anything. Off is the operator's call, and reversible: the
limiter below is unchanged and one variable puts it back.

It lives here rather than in the frontend's route handlers because the browser
uploads straight to Modal (a 3 minute mp3 is past Vercel's 4.5 MB body limit),
so `/submit` is the only place every request is guaranteed to pass through.

The address itself is never stored. Plan §8 item 5 wants an audit trail of job
ids and timestamps and explicitly not of user content; an IP is closer to
content than to a job id, so the key is a salted hash of it and the raw value
stays in the request.

Like `jobs`, every function takes an optional `store`, so the tests drive the
whole thing with a plain dict and no Modal credentials.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import MutableMapping
from typing import Any

from .app import rate_dict

# Plan §9 capped the MVP at 5 jobs/hour. That cap ships **off**: it was written
# to bound GPU spend, and what it actually did was charge an operator debugging
# their own deployment a slot for every job that crashed — five failures and the
# app locks you out of fixing it. Set `JOBS_PER_HOUR` to a positive integer to
# put it back; anything else, empty included, means no cap.
#
# Nothing else about the limiter changed, so turning it back on is one variable
# rather than a revert.
ENV_LIMIT = "JOBS_PER_HOUR"
NO_LIMIT = 0
# What plan §9 asks for, and what `JOBS_PER_HOUR=5` restores. Not a default —
# it is here so the tests and the docs have one number to point at.
PLAN_MAX_JOBS = 5
WINDOW_SEC = 3600.0


def max_jobs(value: str | None = None) -> int:
    """The cap in force, or `NO_LIMIT` when there is none.

    Only a positive integer turns the cap on. A malformed value means no cap
    rather than a guessed one: this is read at import time in every container,
    and a limiter that refuses to load takes the whole app down with it.
    """
    raw = (os.environ.get(ENV_LIMIT, "") if value is None else value).strip()
    if not raw:
        return NO_LIMIT
    try:
        return max(NO_LIMIT, int(raw))
    except ValueError:
        return NO_LIMIT


# Requests that reach Modal directly carry the client address in
# `x-forwarded-for`; the first entry is the client, the rest are proxies.
FORWARDED_HEADER = "x-forwarded-for"
UNKNOWN_CLIENT = "unknown"


class RateLimited(RuntimeError):
    """Raised by `check`. Carries the seconds until the next slot frees up."""

    def __init__(self, retry_after: int, limit: int) -> None:
        self.retry_after = max(1, retry_after)
        self.limit = limit
        super().__init__(
            f"rate limit reached: {limit} jobs per hour. "
            f"Try again in {self.retry_after // 60 + 1} minute(s)."
        )


def _store(store: MutableMapping[str, Any] | None) -> MutableMapping[str, Any]:
    return rate_dict if store is None else store


def _salt() -> str:
    """Hash salt, fixed by default so a key means the same in every container.

    The default is a constant rather than a secret, which is enough for what
    the key is for — grouping one client's requests without keeping the
    address. Set RATE_LIMIT_SALT on the Modal secret to make the hash
    unguessable as well, so nobody holding the Dict can test whether a
    particular address has submitted.
    """
    return os.environ.get("RATE_LIMIT_SALT", "voice-convert")


def client_key(address: str | None) -> str:
    """A stable, non-reversible key for one client address."""
    if not address:
        return UNKNOWN_CLIENT
    digest = hashlib.sha256(f"{_salt()}:{address.strip()}".encode()).hexdigest()
    return digest[:32]


def address_from_headers(headers: Any, fallback: str | None = None) -> str | None:
    """Pull the client address out of request headers, `None` if absent."""
    forwarded = headers.get(FORWARDED_HEADER) if headers is not None else None
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return fallback


def _recent(
    key: str,
    store: MutableMapping[str, Any],
    now: float,
    window_sec: float,
) -> list[float]:
    stamps = store.get(key) or []
    return [float(t) for t in stamps if now - float(t) < window_sec]


def check(
    key: str,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
    limit: int | None = None,
    window_sec: float = WINDOW_SEC,
) -> int | None:
    """Record one request against `key`. Returns how many remain in the window.

    Raises `RateLimited` and records nothing when the window is already full,
    so a rejected request does not push the next slot further away.

    `None` when no cap is configured, and then nothing is recorded either —
    there is no point growing a Dict nobody reads.
    """
    limit = max_jobs() if limit is None else limit
    if limit == NO_LIMIT:
        return None
    store = _store(store)
    stamp = time.time() if now is None else now
    stamps = _recent(key, store, stamp, window_sec)
    if len(stamps) >= limit:
        raise RateLimited(
            retry_after=retry_after(key, store, stamp, limit, window_sec), limit=limit
        )
    stamps.append(stamp)
    store[key] = stamps
    return limit - len(stamps)


def remaining(
    key: str,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
    limit: int | None = None,
    window_sec: float = WINDOW_SEC,
) -> int | None:
    """How many jobs `key` may still start. Read-only — nothing is recorded.

    `None` when no cap is configured. It reaches the browser as `null`, which is
    what makes the "N lượt còn lại" line disappear rather than quote a ceiling
    that is not there.
    """
    limit = max_jobs() if limit is None else limit
    if limit == NO_LIMIT:
        return None
    stamp = time.time() if now is None else now
    return max(0, limit - len(_recent(key, _store(store), stamp, window_sec)))


def retry_after(
    key: str,
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
    limit: int | None = None,
    window_sec: float = WINDOW_SEC,
) -> int:
    """Seconds until `key` has a slot again; 0 when it already has one.

    The wait is until the *oldest* recorded request leaves the window, which is
    usually far less than a full window — telling a client to come back in an
    hour when the next slot opens in five minutes is its own kind of wrong.

    With no cap configured there is never a wait, so this is always 0 and
    `_check_quota` waves every request through.
    """
    limit = max_jobs() if limit is None else limit
    if limit == NO_LIMIT:
        return 0
    stamp = time.time() if now is None else now
    stamps = _recent(key, _store(store), stamp, window_sec)
    if len(stamps) < limit:
        return 0
    return max(1, int(window_sec - (stamp - min(stamps)) + 0.5))


def prune(
    store: MutableMapping[str, Any] | None = None,
    now: float | None = None,
    window_sec: float = WINDOW_SEC,
) -> int:
    """Drop windows that have fully expired. Called by the cleanup cron."""
    store = _store(store)
    stamp = time.time() if now is None else now
    dead = [key for key in list(store.keys()) if not _recent(key, store, stamp, window_sec)]
    for key in dead:
        del store[key]
    return len(dead)
