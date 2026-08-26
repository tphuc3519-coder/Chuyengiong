"""Storage layer, driven against a tmp directory instead of the Volume."""

import os

import pytest

from modal_app import storage


def test_job_ids_are_hex_uuids():
    job_id = storage.new_job_id()
    assert storage.check_job_id(job_id) == job_id
    assert len(job_id) == 32
    assert storage.new_job_id() != job_id


@pytest.mark.parametrize(
    "bad",
    ["", "..", "../etc", "g" * 32, "0123456789abcdef", None, "/" + "0" * 31, "0" * 33],
)
def test_bad_job_ids_are_rejected(bad):
    """`job_id` arrives as a URL path segment; `..` must be impossible."""
    with pytest.raises(storage.StorageError):
        storage.check_job_id(bad)


@pytest.mark.parametrize("bad", ["", "../input.mp3", "input.exe", "in put.wav", "Input.wav"])
def test_bad_artifact_names_are_rejected(bad):
    with pytest.raises(storage.StorageError):
        storage.check_name(bad)


def test_layout_matches_the_plan(tmp_path):
    job_id = storage.new_job_id()
    path = storage.path_for(job_id, "output.mp3", root=tmp_path)
    assert path == tmp_path / job_id / "output.mp3"


def test_round_trip_is_byte_exact(tmp_path):
    job_id = storage.new_job_id()
    payload = os.urandom(50_000)
    storage.put(job_id, "input.mp3", payload, root=tmp_path)
    assert storage.get(job_id, "input.mp3", root=tmp_path) == payload
    assert storage.size(job_id, "input.mp3", root=tmp_path) == len(payload)
    assert storage.exists(job_id, "input.mp3", root=tmp_path)


def test_writes_leave_no_partial_file_behind(tmp_path):
    job_id = storage.new_job_id()
    storage.put(job_id, "vocal.wav", b"x", root=tmp_path)
    names = {p.name for p in (tmp_path / job_id).iterdir()}
    assert names == {"vocal.wav"}


def test_put_overwrites_in_place(tmp_path):
    job_id = storage.new_job_id()
    storage.put(job_id, "vocal.wav", b"first", root=tmp_path)
    storage.put(job_id, "vocal.wav", b"second", root=tmp_path)
    assert storage.get(job_id, "vocal.wav", root=tmp_path) == b"second"


def test_missing_file_raises(tmp_path):
    job_id = storage.new_job_id()
    with pytest.raises(storage.StorageError):
        storage.get(job_id, "output.mp3", root=tmp_path)
    assert not storage.exists(job_id, "output.mp3", root=tmp_path)


def test_delete_job(tmp_path):
    job_id = storage.new_job_id()
    storage.put(job_id, "input.mp3", b"data", root=tmp_path)
    assert storage.delete_job(job_id, root=tmp_path) is True
    assert not (tmp_path / job_id).exists()
    assert storage.delete_job(job_id, root=tmp_path) is False


def test_cleanup_removes_expired_and_keeps_fresh(tmp_path):
    fresh, stale = storage.new_job_id(), storage.new_job_id()
    storage.put(fresh, "input.mp3", b"new", root=tmp_path)
    storage.put(stale, "input.mp3", b"old", root=tmp_path)
    storage.backdate(stale, 7 * 3600, root=tmp_path)

    removed = storage.cleanup_expired(max_age_hours=6, root=tmp_path)

    assert removed == [stale]
    assert not (tmp_path / stale).exists()
    assert storage.get(fresh, "input.mp3", root=tmp_path) == b"new"


def test_cleanup_measures_age_from_the_last_write(tmp_path):
    """A job still being written must not be swept out from under itself."""
    job_id = storage.new_job_id()
    storage.put(job_id, "input.mp3", b"uploaded", root=tmp_path)
    storage.backdate(job_id, 7 * 3600, root=tmp_path)
    storage.put(job_id, "vocal.wav", b"just separated", root=tmp_path)

    assert storage.cleanup_expired(max_age_hours=6, root=tmp_path) == []
    assert storage.job_age_sec(job_id, root=tmp_path) < 60


def test_cleanup_ignores_anything_that_is_not_a_job(tmp_path):
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "some-other-dir").mkdir()
    os.utime(tmp_path / "some-other-dir", (0, 0))
    os.utime(tmp_path / "notes.txt", (0, 0))

    assert storage.cleanup_expired(max_age_hours=6, root=tmp_path) == []
    assert (tmp_path / "notes.txt").exists()
    assert (tmp_path / "some-other-dir").exists()


def test_cleanup_on_an_empty_root(tmp_path):
    assert storage.cleanup_expired(root=tmp_path) == []
    assert storage.cleanup_expired(root=tmp_path / "not-created-yet") == []
