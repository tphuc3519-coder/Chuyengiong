"""Audit trail: one structured line per job event (plan §8 item 5).

The plan asks for job ids and timestamps so a complaint about a voice can be
traced to a job, and asks explicitly for the audio itself never to be logged.
Both halves matter, so this module is the only place either happens and it is
built to make the second half hard to get wrong:

* **Fields are allowlisted.** `FIELDS` is the complete list of what may be
  written. A key that is not in it is dropped and its *name* is recorded, so a
  caller that tries to log something new sees it disappear instead of finding
  out later that a filename has been sitting in the logs for a month.
* **Only scalars survive.** `bytes` cannot be rendered at all, which is the
  audio rule enforced by construction rather than by everyone remembering it.
  Strings are stripped of newlines and capped, so nothing forges a second line.
* **`record` never raises.** Every call site is either on the request path or
  inside a pipeline's `except` block; an audit line failing to format must not
  be what takes a job down.

Output goes to stdout as ``[audit] {json}``, which is what Modal collects. It
is one line per event on purpose: `grep '\\[audit\\]' | cut -d' ' -f2- | jq` is
the whole tooling story at MVP, and nothing here needs a database.

What is deliberately *not* here: the client address (`ratelimit.client_key`
hashes it first, and only the hash is ever passed in), file names, error text
from a user's file, and any parameter that could carry text a user typed.
"""

from __future__ import annotations

import json
import re
import time

PREFIX = "[audit]"

# The events. A job produces `submit` then exactly one of `done`/`failed`, plus
# a `download` per fetch; `expire` is the cleanup cron and carries no job id.
SUBMIT = "submit"
DONE = "done"
FAILED = "failed"
DOWNLOAD = "download"
EXPIRE = "expire"
EVENTS = (SUBMIT, DONE, FAILED, DOWNLOAD, EXPIRE)

# Everything a line may contain, and nothing else.
#
#   client          salted hash of the address, from `ratelimit.client_key`
#   consent         what the consent gate was told (plan §8 item 1)
#   mode            "song" | "speech"
#   input_bytes     upload sizes — a size is not content
#   reference_bytes
#   output_bytes    what /download handed back
#   model           separation model id
#   steps           diffusion steps
#   shift           semitones applied, measured or given
#   cfg             how hard the sampler was pushed towards the reference
#   clarity         how much of the output filter chain ran
#   profile         *name* of the trained voice used, never the audio it learnt
#   language        which language was read, never the text that was read
#   emotion         which delivery it was read with — a setting, not content
#   beat_bytes      size of an uploaded replacement backing track
#   beat_source     where a replacement backing track came from, not what it said
#   watermark       whether the output carries an AudioSeal watermark
#   seconds         wall clock of the pipeline run
#   reason          exception *class* name on failure, never its message
#   jobs            cleanup: directories removed
#   records         cleanup: job records dropped
#   windows         cleanup: rate limit windows dropped
FIELDS = (
    "client",
    "consent",
    "mode",
    "input_bytes",
    "reference_bytes",
    "output_bytes",
    "model",
    "steps",
    "shift",
    "cfg",
    "clarity",
    "profile",
    "language",
    "emotion",
    "beat_bytes",
    "beat_source",
    "watermark",
    "seconds",
    "reason",
    "jobs",
    "records",
    "windows",
)

# Long enough for a model id or an exception class, short enough that nothing
# interesting fits.
MAX_TEXT = 64
# `storage.new_job_id` is `uuid4().hex`. Anything else reaching here came from a
# URL path rather than from us.
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
INVALID_JOB = "invalid"


class AuditError(ValueError):
    """Unknown event name. Raised by `line`, swallowed by `record`."""


def _text(value: str) -> str:
    """One line, bounded. Newlines would let a value forge its own record."""
    return " ".join(value.split())[:MAX_TEXT]


def _scalar(value: object) -> object | None:
    """The value as something JSON can hold, or None if it may not be logged.

    `bytes` falls through to None like every other non-scalar, which is the
    point: audio cannot be logged even by a caller who passes it deliberately.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        return _text(value)
    return None


def _job(job_id: str | None) -> str | None:
    if job_id is None:
        return None
    return job_id if _JOB_ID_RE.match(str(job_id)) else INVALID_JOB


def stamp(now: float | None = None) -> str:
    """UTC, to the second. Job records keep the epoch value; logs read better."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if now is None else now))


def event_line(event: str, job_id: str | None = None, now: float | None = None, **fields) -> str:
    """The line `record` would print. Pure, so the tests can assert on it."""
    if event not in EVENTS:
        raise AuditError(f"event must be one of {EVENTS}, got {event!r}")

    entry: dict[str, object] = {"ts": stamp(now), "event": event}
    if job_id is not None:
        entry["job"] = _job(job_id)

    dropped = []
    for key in FIELDS:
        if key not in fields:
            continue
        value = _scalar(fields[key])
        if value is None and fields[key] is not None:
            dropped.append(key)  # present but unloggable — a bytes field, say
            continue
        entry[key] = value
    # Names, never values: a key is written in our source, a value is not.
    dropped += [key for key in fields if key not in FIELDS]
    if dropped:
        entry["dropped"] = sorted(dropped)

    return f"{PREFIX} {json.dumps(entry, ensure_ascii=False, separators=(',', ':'))}"


def record(event: str, job_id: str | None = None, **fields) -> str:
    """Write one audit line. Returns it, and never raises."""
    try:
        text = event_line(event, job_id, **fields)
    except Exception as exc:  # a typo in an event name must not fail a job
        text = f'{PREFIX} {{"ts":"{stamp()}","event":"malformed","reason":"{type(exc).__name__}"}}'
    print(text, flush=True)
    return text
