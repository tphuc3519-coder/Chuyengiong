"""The audit trail, and above all what it refuses to write.

Plan §8 item 5 asks for two things at once — a trail of job ids and timestamps,
and no user content in the logs — so the tests that matter here are the
negative ones: audio handed to `record` on purpose does not come out the other
end, and neither does anything else the allowlist has not been told about.
"""

import json
import time

import pytest

from modal_app import audit

JOB = "0" * 32


def parsed(line: str) -> dict:
    assert line.startswith(audit.PREFIX + " ")
    return json.loads(line[len(audit.PREFIX) + 1 :])


def test_line_carries_job_id_and_timestamp():
    entry = parsed(audit.event_line(audit.SUBMIT, JOB, now=1_700_000_000.0, mode="song"))
    assert entry["event"] == audit.SUBMIT
    assert entry["job"] == JOB
    assert entry["mode"] == "song"
    # UTC, seconds precision, sortable as text.
    assert entry["ts"] == "2023-11-14T22:13:20Z"


def test_stamp_defaults_to_now():
    assert audit.stamp() == time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def test_audio_is_never_logged():
    """The rule the whole module exists for."""
    audio = b"RIFF" + b"\x00" * 4096
    entry = parsed(
        audit.event_line(audit.SUBMIT, JOB, input_bytes=audio, reference_bytes=len(audio))
    )
    assert "input_bytes" not in entry
    assert entry["reference_bytes"] == 4100  # a size is fine; the samples are not
    assert entry["dropped"] == ["input_bytes"]


def test_fields_outside_the_allowlist_are_dropped_by_name():
    entry = parsed(audit.event_line(audit.DOWNLOAD, JOB, filename="bí mật.mp3", output_bytes=12))
    assert "filename" not in entry
    assert json.dumps(entry, ensure_ascii=False).find("bí mật") == -1
    assert entry["dropped"] == ["filename"]  # the name is ours, the value is not


def test_text_is_capped_and_kept_on_one_line():
    entry = parsed(audit.event_line(audit.FAILED, JOB, reason="A\nB" + "x" * 200))
    assert entry["reason"] == "A B" + "x" * (audit.MAX_TEXT - 3)
    assert "\n" not in entry["reason"]


def test_none_is_a_value_not_a_drop():
    """`shift=None` means auto-detect never ran — that is worth recording."""
    entry = parsed(audit.event_line(audit.DONE, JOB, shift=None, seconds=1.23456))
    assert entry["shift"] is None
    assert entry["seconds"] == 1.235
    assert "dropped" not in entry


def test_job_id_from_a_url_path_cannot_forge_a_field():
    entry = parsed(audit.event_line(audit.DOWNLOAD, '../../etc", "consent": "true'))
    assert entry["job"] == audit.INVALID_JOB


def test_expire_has_no_job_id():
    entry = parsed(audit.event_line(audit.EXPIRE, jobs=3, records=4, windows=5))
    assert "job" not in entry
    assert (entry["jobs"], entry["records"], entry["windows"]) == (3, 4, 5)


def test_unknown_event_raises_but_record_does_not(capsys):
    with pytest.raises(audit.AuditError):
        audit.event_line("exploded", JOB)
    # `record` runs inside pipeline `except` blocks: it may never be the thing
    # that raises there.
    line = audit.record("exploded", JOB)
    assert parsed(line)["event"] == "malformed"
    assert capsys.readouterr().out.strip() == line


def test_record_prints_one_line(capsys):
    line = audit.record(audit.SUBMIT, JOB, mode="speech", consent=True)
    printed = capsys.readouterr().out
    assert printed == line + "\n"
    assert parsed(line)["consent"] is True
