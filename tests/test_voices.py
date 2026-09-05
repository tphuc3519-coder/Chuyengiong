"""Voice profile names and paths.

The name is the interesting half: it arrives from a form field and becomes a
directory, which is the shape of problem where "sanitise it" is the wrong
answer. A sanitiser turns `../../etc` into something that looks fine and points
elsewhere, and the caller never finds out the voice it asked for is not the
voice it got.
"""

import pytest

from modal_app import voices


def test_an_ordinary_name_is_accepted():
    assert voices.check_name("mai") == "mai"
    assert voices.check_name("Mai_2-b") == "Mai_2-b"


@pytest.mark.parametrize(
    "name",
    ["", "..", "../etc", "a/b", "voice name", "-leading", ".hidden", "x" * 49, "é"],
)
def test_anything_that_could_be_a_path_or_a_surprise_is_refused(name):
    with pytest.raises(voices.VoiceError):
        voices.check_name(name)


def test_a_non_string_is_refused_rather_than_coerced():
    with pytest.raises(voices.VoiceError):
        voices.check_name(None)


def test_clean_name_turns_an_unusable_name_into_no_profile():
    """`/submit` uses this: a job that names a voice that cannot exist should
    run zero-shot, which is what every job did before profiles existed."""
    assert voices.clean_name("../etc") == ""
    assert voices.clean_name(None) == ""
    assert voices.clean_name("mai") == "mai"


def test_the_layout_is_per_voice_and_per_mode(tmp_path):
    """`speech` and `singing` are different architectures at different sample
    rates. A profile trained for one does not load into the other — not worse,
    it will not load — so they cannot share a directory."""
    speech, _ = voices.files(str(tmp_path), "mai", "speech")
    singing, _ = voices.files(str(tmp_path), "mai", "singing")
    assert speech != singing


def _write_profile(root, name, mode):
    import os

    where = voices.directory(str(root), name, mode)
    os.makedirs(where, exist_ok=True)
    for filename in (voices.CHECKPOINT_NAME, voices.CONFIG_NAME):
        with open(os.path.join(where, filename), "wb") as out:
            out.write(b"x")


def test_a_profile_is_only_there_when_both_files_are(tmp_path):
    """A run that died before its final save leaves a directory and no
    checkpoint. The directory is the record, so a half-written one is simply
    not a profile and nothing has to be reconciled."""
    import os

    os.makedirs(voices.directory(str(tmp_path), "mai", "speech"))
    assert not voices.exists(str(tmp_path), "mai", "speech")
    _write_profile(tmp_path, "mai", "speech")
    assert voices.exists(str(tmp_path), "mai", "speech")


def test_no_voice_asked_for_is_not_an_error(tmp_path):
    assert voices.resolve(str(tmp_path), "", "speech") is None


def test_a_voice_that_was_asked_for_and_is_missing_fails_loudly(tmp_path):
    """The one case worth failing over: a job that asked to be converted in a
    trained voice and silently ran zero-shot instead produces a result that is
    plausible and wrong, and nobody would know to ask."""
    with pytest.raises(voices.VoiceError):
        voices.resolve(str(tmp_path), "mai", "speech")


def test_a_profile_trained_for_one_mode_is_not_offered_for_the_other(tmp_path):
    _write_profile(tmp_path, "mai", "speech")
    assert voices.resolve(str(tmp_path), "mai", "speech")
    with pytest.raises(voices.VoiceError):
        voices.resolve(str(tmp_path), "mai", "singing")


def test_listing_is_empty_on_a_deployment_that_never_trained_anything(tmp_path):
    assert voices.profiles(str(tmp_path)) == {}


def test_listing_reports_each_voice_with_the_modes_it_has(tmp_path):
    _write_profile(tmp_path, "mai", "speech")
    _write_profile(tmp_path, "mai", "singing")
    _write_profile(tmp_path, "nam", "speech")
    assert voices.profiles(str(tmp_path)) == {
        "mai": ["singing", "speech"],
        "nam": ["speech"],
    }
