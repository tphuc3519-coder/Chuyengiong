"""Text to speech: what happens to the text before a GPU ever sees it.

Synthesis itself needs the checkpoint and a container, so what is worth testing
here is everything around it — the validation the request is answered with, the
split that decides how many times the model is called, and the fact that the
text is treated as user data all the way through: on the Volume, under the same
TTL as the audio, and never in the audit log.
"""

import importlib

import pytest

from modal_app import jobs, pipeline, storage, tts

# --- language -------------------------------------------------------------


def test_vietnamese_is_the_default_and_is_offered():
    assert tts.DEFAULT_LANGUAGE == "vie"
    assert tts.DEFAULT_LANGUAGE in tts.LANGUAGES


def test_an_unknown_language_is_refused_rather_than_defaulted():
    """Reading Vietnamese with the English checkpoint produces confident
    nonsense, which is worse than an error."""
    with pytest.raises(tts.TtsError):
        tts.check_language("klingon")
    with pytest.raises(tts.TtsError):
        tts.check_language("")


def test_the_model_id_is_the_mms_checkpoint_for_the_language():
    assert tts.model_id("vie") == "facebook/mms-tts-vie"


# --- the text -------------------------------------------------------------


def test_empty_text_is_refused():
    for bad in ("", "   ", "\n\n"):
        with pytest.raises(tts.TtsError):
            tts.check_text(bad)


def test_text_past_the_limit_is_refused_with_the_number():
    with pytest.raises(tts.TtsError, match=str(tts.MAX_TEXT_CHARS)):
        tts.check_text("a" * (tts.MAX_TEXT_CHARS + 1))
    assert tts.check_text("a" * tts.MAX_TEXT_CHARS)


def test_text_with_no_letters_is_refused():
    """MMS tokenises characters and drops every one it has no token for, so a
    line of digits would synthesise to silence and arrive as a job that
    succeeded and produced nothing."""
    with pytest.raises(tts.TtsError, match="write them out"):
        tts.check_text("123 456 789!")


def test_check_text_returns_the_text_stripped():
    assert tts.check_text("  Xin chào.  ") == "Xin chào."


# --- splitting ------------------------------------------------------------


def test_split_is_by_sentence():
    assert tts.split_text("Một. Hai! Ba?") == ["Một.", "Hai!", "Ba?"]


def test_split_drops_blank_lines_and_keeps_the_order():
    assert tts.split_text("Một.\n\n\nHai.") == ["Một.", "Hai."]


def test_a_long_sentence_is_broken_at_a_clause():
    text = "một " * 15 + ", " + "hai " * 40
    segments = tts.split_text(text, max_chars=100)
    assert all(len(segment) <= 100 for segment in segments)
    # The comma stays with the clause it closes — the model reads it as a pause.
    assert segments[0].endswith(",")


def test_a_sentence_with_nowhere_to_break_is_still_split():
    """An unbroken run that long is not language, but it must not be handed to
    the model whole either."""
    segments = tts.split_text("x" * 500, max_chars=100)
    assert segments and all(len(segment) <= 100 for segment in segments)


def test_splitting_loses_no_words():
    text = "Xin chào các bạn. Hôm nay trời đẹp; mình đi chơi nhé!"
    assert " ".join(tts.split_text(text)).split() == text.split()


def test_every_segment_fits_the_default_budget():
    text = ("Câu này dài vừa phải, có dấu phẩy ở giữa và kết thúc bằng dấu chấm. ") * 20
    assert all(len(segment) <= tts.SEGMENT_MAX_CHARS for segment in tts.split_text(text))


# --- speaking rate --------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, tts.DEFAULT_SPEAKING_RATE),
        (0, tts.DEFAULT_SPEAKING_RATE),
        ("nonsense", tts.DEFAULT_SPEAKING_RATE),
        (9.0, tts.SPEAKING_RATE_MAX),
        (0.01, tts.SPEAKING_RATE_MIN),
        (1.2, 1.2),
    ],
)
def test_speaking_rate_is_clamped_not_rejected(given, expected):
    assert tts.clamp_speaking_rate(given) == pytest.approx(expected)


# --- the job it becomes ---------------------------------------------------


def test_tts_is_a_job_mode_that_converts_as_speech():
    assert "tts" in jobs.JOB_MODES
    assert jobs.CONVERSION_MODE["tts"] == "speech"


def test_a_tts_job_walks_queued_synthesizing_converting_done():
    store: dict = {}
    job_id = jobs.create("a" * 32, "tts", store=store)["id"]
    for status in (jobs.SYNTHESIZING, jobs.CONVERTING, jobs.DONE):
        assert jobs.update(job_id, status, store=store)["status"] == status
    assert jobs.find(job_id, store=store)["progress"] == 100


def test_clean_params_carries_the_language_and_the_rate():
    params = pipeline.clean_params("tts", {"language": "eng", "speaking_rate": 1.5})
    assert params["language"] == "eng"
    assert params["speaking_rate"] == 1.5
    # It converts as speech, so it gets speech's quality default and nothing
    # from the song branch.
    assert params["diffusion_steps"] == 25
    assert "separation_model" not in params
    assert "source_ext" not in params


def test_clean_params_defaults_to_vietnamese_and_normal_speed():
    params = pipeline.clean_params("tts")
    assert params["language"] == tts.DEFAULT_LANGUAGE
    assert params["speaking_rate"] == tts.DEFAULT_SPEAKING_RATE


def test_clean_params_refuses_a_language_we_do_not_speak():
    with pytest.raises(tts.TtsError):
        pipeline.clean_params("tts", {"language": "klingon"})


def test_auto_detect_reaches_the_tts_branch():
    """It converts as `speech`, and that is the mode `run_tts_pipeline` measures
    under — the synthetic voice has one register per language and the target is
    as likely to be an octave off it as not."""
    assert jobs.CONVERSION_MODE["tts"] in pipeline.AUTO_DETECT_MODES


def test_the_text_and_the_synthesised_voice_are_storable_artifacts():
    for name in (pipeline.TEXT, pipeline.SPOKEN):
        assert storage.check_name(name) == name


def test_the_text_lives_on_the_volume_under_the_same_ttl(tmp_path):
    """Not in the job record: the sweep that expires a job's audio after six
    hours has to take what the user wrote with it."""
    job_id = storage.new_job_id()
    storage.put(job_id, pipeline.TEXT, "Xin chào".encode(), root=tmp_path)
    storage.backdate(job_id, (storage.DEFAULT_MAX_AGE_HOURS + 1) * 3600, root=tmp_path)
    assert storage.cleanup_expired(root=tmp_path) == [job_id]
    assert not storage.exists(job_id, pipeline.TEXT, root=tmp_path)


def test_every_job_mode_has_a_pipeline():
    assert set(pipeline.PIPELINES) == set(jobs.JOB_MODES)


def test_the_synthesiser_container_stack_is_not_imported_at_module_scope():
    """`pipeline` imports this module on the small API image, which has neither
    transformers nor a model. Importing either at module scope would turn every
    cold start into an ImportError."""
    with open(importlib.import_module("modal_app.tts").__file__) as handle:
        top_level = [
            line
            for line in handle
            if line.startswith(("import ", "from "))
            and any(pkg in line for pkg in ("transformers", "torch", "numpy"))
        ]
    assert top_level == []
