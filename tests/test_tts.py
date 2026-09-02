"""Text to speech: what happens to the text before a GPU ever sees it.

Synthesis itself needs the checkpoint and a container, so what is worth testing
here is everything around it — the validation the request is answered with, the
split that decides how many times the model is called, and the fact that the
text is treated as user data all the way through: on the Volume, under the same
TTL as the audio, and never in the audit log.
"""

import importlib
import types
from unittest import mock

import pytest

from modal_app import jobs, pipeline, prosody, storage, tts

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


# --- Japanese -------------------------------------------------------------
#
# It is the one language that does not read through MMS, because MMS reads a
# non-Latin script by romanising it with `uroman` first and uroman renders
# Japanese kanji in Mandarin: 今日 ("kyou") comes out "jinri", 田中 ("tanaka")
# comes out "tianzhong". That is the text `facebook/mms-tts-jpn` was trained
# on, so no romanisation on our side rescues it. Kokoro's `misaki[ja]` front
# end is a dictionary-and-morphology G2P and reads both correctly.


def test_japanese_is_offered_and_does_not_read_through_mms():
    spec = tts.spec_for("jpn")
    assert spec.engine == tts.KOKORO
    assert spec.label == "日本語"


def test_asking_mms_for_japanese_is_an_error_not_a_silent_wrong_reading():
    with pytest.raises(tts.TtsError, match="kokoro"):
        tts.model_id("jpn")


def test_a_kokoro_language_carries_what_kokoro_needs():
    for code, spec in tts.LANGUAGES.items():
        if spec.engine != tts.KOKORO:
            continue
        assert spec.kokoro_code, code
        assert spec.voice, code


def test_japanese_splits_on_punctuation_that_carries_no_space():
    """Japanese writes 「です。」 and starts the next sentence immediately, so a
    splitter that needs whitespace after a full stop never fires."""
    assert tts.split_text("今日はいい天気ですね。私の名前は田中です。") == [
        "今日はいい天気ですね。",
        "私の名前は田中です。",
    ]


def test_japanese_breaks_a_long_sentence_at_the_ideographic_comma():
    text = "こんにちは、" + "あ" * 40
    segments = tts.split_text(text, max_chars=20)
    assert segments[0] == "こんにちは、"
    assert all(len(segment) <= 20 for segment in segments)


def test_japanese_gets_a_shorter_limit_because_a_character_says_more():
    """700 Japanese characters and 2000 Latin ones are the same two or three
    minutes of speech; the cap is on the recording, not on the typing."""
    assert tts.spec_for("jpn").max_chars < tts.spec_for("vie").max_chars
    assert tts.check_text("あ" * 700, "jpn")
    with pytest.raises(tts.TtsError):
        tts.check_text("あ" * 701, "jpn")
    # The same length is fine in a language that spends characters faster.
    assert tts.check_text("a" * 701, "vie")


def test_kana_and_kanji_count_as_words_to_read():
    """The "no letters" gate is about digits and symbols, and `str.isalpha` is
    true for both scripts — so Japanese passes it as written."""
    assert tts.check_text("こんにちは。", "jpn")
    assert tts.check_text("東京", "jpn")


def test_japanese_segments_stay_well_under_the_kokoro_truncation_point():
    assert tts.spec_for("jpn").segment_max_chars < tts.KOKORO_MAX_PHONEMES / 3


def test_transformers_is_pinned_for_both_engines():
    """`kokoro` asks for transformers unversioned, and the newest one requires
    torch >= 2.5 against `base_image`'s 2.4.0 — which it answers by disabling
    PyTorch and turning every model class into a stub that raises the first
    time something builds one. That is inside `@modal.enter()`, on the first
    Japanese request, long after a deploy has gone green."""
    assert tts.TRANSFORMERS_SPEC.startswith("transformers==")


# --- romaji ---------------------------------------------------------------
#
# Japanese typed without an IME is romaji, and Kokoro's front end hands Latin
# letters through untouched: `konnichiwa` reaches the model as eleven Latin
# characters. Romaji is phonetic, so spelling it back into kana costs nothing
# and is the difference between a reading and none.


def test_japanese_is_the_language_that_takes_romaji():
    assert tts.spec_for("jpn").romaji_input
    assert not tts.spec_for("vie").romaji_input


def test_romaji_becomes_kana():
    assert tts.to_kana("kyou wa ii tenki desu ne.") == "きょう わ いい てんき です ね."
    assert tts.to_kana("ohayou gozaimasu") == "おはよう ございます"


def test_a_doubled_n_is_the_syllabic_n_not_a_dropped_mora():
    """Wapuro romaji writes ん before a vowel as "nn"; jaconv reads only the
    apostrophe form, so "konnichiwa" came back こんいちわ — a different word."""
    assert tts.to_kana("konnichiwa") == "こんにちわ"
    assert tts.to_kana("onnanoko") == "おんなのこ"
    assert tts.to_kana("sennin") == "せんにん"
    # An n that already closes a syllable was never the broken case.
    assert tts.to_kana("shinbun") == "しんぶん"


def test_kana_and_kanji_pass_through_untouched():
    assert tts.to_kana("今日はいい天気ですね。") == "今日はいい天気ですね。"
    # A Latin word inside Japanese is read as Japanese, which is the right
    # answer for a name and the closest available one for anything else.
    assert tts.to_kana("私はTanakaです。") == "私はたなかです。"


def test_romaji_keeps_its_punctuation_so_the_split_still_works():
    """The splitter runs after the conversion, so a full stop that did not
    survive it would take the sentence boundary with it."""
    assert tts.split_text(tts.to_kana("konnichiwa. genki desu ka?")) == [
        "こんにちわ.",
        "げんき です か?",
    ]


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


# --- paragraphs -----------------------------------------------------------
#
# `split_text` throws away where the blank lines were, and two of the reading
# rules are about the start and the end of a paragraph — the pitch resets at one
# and the silence is three quarters of a second rather than four tenths. So the
# split keeps the structure and `split_text` is the flattened view of it.


def test_a_blank_line_starts_a_new_block():
    assert tts.split_blocks("Một. Hai.\n\nBa.") == [["Một.", "Hai."], ["Ba."]]


def test_a_single_line_break_is_a_sentence_boundary_not_a_paragraph():
    """Text pasted out of a document is full of wrapped lines. Only a blank
    line means the writer stopped."""
    assert tts.split_blocks("Một.\nHai.") == [["Một.", "Hai."]]


def test_split_text_is_split_blocks_flattened():
    text = "Một. Hai.\n\nBa.\n\n\nBốn."
    assert tts.split_text(text) == [
        segment for block in tts.split_blocks(text) for segment in block
    ]


def test_blocks_that_hold_nothing_are_dropped_rather_than_kept_empty():
    assert tts.split_blocks("\n\n   \n\nMột.\n\n\n") == [["Một."]]


# --- the reading ----------------------------------------------------------


def test_the_pauses_between_segments_are_the_plan_not_one_fixed_gap():
    """`_join` used to insert the same 0.25s everywhere. A comma, a full stop
    and a blank line are three different silences, and the file gets longer as
    they do."""
    import numpy as np

    from modal_app.audio_utils import decode_wav

    voice = [np.zeros(1000, dtype=np.float32) for _ in range(2)]
    voice[0][0] = 0.5
    short, _ = decode_wav(tts._join(voice, 16000, [0.1, 0.0]))
    long, _ = decode_wav(tts._join(voice, 16000, [0.6, 0.0]))
    assert len(long) - len(short) == pytest.approx(0.5 * 16000, abs=2)


def test_join_without_a_plan_still_reads_as_it_always_did():
    """Nothing else calls it that way, but the fallback is what keeps `_join`
    a wav writer rather than a second copy of the prosody rules."""
    import numpy as np

    from modal_app.audio_utils import decode_wav

    voice = [np.zeros(1000, dtype=np.float32) for _ in range(2)]
    voice[0][0] = 0.5
    plain, _ = decode_wav(tts._join(voice, 16000))
    planned, _ = decode_wav(tts._join(voice, 16000, [tts.SEGMENT_GAP_SEC, 0.0]))
    assert len(plain) == len(planned)


# --- Japanese pitch accent ------------------------------------------------
#
# 箸 and 橋 are both `hashi`; 雨 and 飴 are both `ame`. Where the pitch falls is
# what separates them, so getting it wrong is a different word rather than an
# accent. `kokoro.KPipeline` builds `misaki.ja.JAG2P()`, whose default is the
# first-generation front end — cutlet, which emits no accent marks at all.
# The second one does, and whether it can be used is a question about the
# checkpoint's vocabulary, so the code asks instead of assuming.


def _probe_accent(vocab, misaki=None):
    """`_use_accent_g2p` against a checkpoint whose vocab is `vocab`."""
    import sys
    from unittest import mock

    fake = types.SimpleNamespace(
        pipeline=types.SimpleNamespace(model=types.SimpleNamespace(vocab=vocab), g2p="cutlet")
    )
    # `None` in sys.modules makes `import misaki` raise ImportError, which is
    # what a box without misaki[ja] does and what CI is.
    modules = (
        {"misaki": misaki, "misaki.ja": getattr(misaki, "ja", None)} if misaki else {"misaki": None}
    )
    with mock.patch.dict(sys.modules, modules):
        used = tts.KokoroSynthesizer._use_accent_g2p(fake)
    return used, fake.pipeline.g2p


def test_the_accent_marks_are_the_ones_misaki_appends():
    """`_` low, `-` mid, `^` the fall — one character per phoneme, appended to
    the phoneme string as a parallel track."""
    assert set(tts.ACCENT_MARKS) == {"_", "-", "^"}


def test_a_checkpoint_with_no_id_for_the_marks_keeps_the_front_end_it_had():
    """`KModel.forward` maps phonemes through `vocab` and silently drops what
    it cannot find, so handing the pitch track to a checkpoint that was not
    trained on it deletes the marks and reads what is left — and `j`, the
    track's filler, is the IPA phoneme /j/. Quietly worse than not trying."""
    used, g2p = _probe_accent({"a": 1, "i": 2, "j": 3})
    assert used is False
    assert g2p == "cutlet"


def test_a_checkpoint_that_reads_the_marks_gets_the_accent_front_end():
    accent_g2p = object()
    misaki = types.ModuleType("misaki")
    misaki.ja = types.SimpleNamespace(JAG2P=lambda version: accent_g2p)
    vocab = {mark: i for i, mark in enumerate(tts.ACCENT_MARKS)}
    used, g2p = _probe_accent(vocab, misaki=misaki)
    assert used is True
    assert g2p is accent_g2p


def test_the_gate_never_stops_the_container_starting():
    """A reading with no accent marks is the reading this shipped with. A
    container that will not start is not."""
    # No vocab to ask, and a vocab that says yes but no misaki to import.
    assert _probe_accent(None) == (False, "cutlet")
    assert _probe_accent({mark: 1 for mark in tts.ACCENT_MARKS}) == (False, "cutlet")


def test_the_probe_sentence_is_ours_and_not_the_users():
    """It is read to the log on every cold start so the G2P chain is visible in
    a container's output. What the user wrote never goes there — the same rule
    the audit log has been under all along."""
    assert tts.ACCENT_PROBE and all(not ch.isascii() or ch == "。" for ch in tts.ACCENT_PROBE)


def test_the_natural_reading_is_what_a_request_that_says_nothing_gets():
    """Every existing caller — and every form that predates the field — sends
    no style at all, and has to keep getting the reading it was getting."""
    import inspect

    signature = inspect.signature(tts.synthesize)
    assert signature.parameters["emotion"].default == prosody.DEFAULT_EMOTION
    assert signature.parameters["expressiveness"].default == prosody.DEFAULT_EXPRESSIVENESS


def test_the_seam_hands_both_engines_the_same_style():
    """Which model reads is `tts`'s business and the style is `prosody`'s, so
    the argument list cannot differ between them — a paragraph read in Japanese
    gets the same pauses as one read in Vietnamese."""
    sent = {}

    class Fake:
        def __init__(self, language):
            sent["language"] = language

        @property
        def synthesize(self):
            return self

        def remote(self, **kwargs):
            sent.update(kwargs)
            return b"wav"

    for language, engine in (("vie", "Synthesizer"), ("jpn", "KokoroSynthesizer")):
        with mock.patch.object(tts, engine, Fake):
            tts.synthesize(language, "Xin chào.", 1.0, emotion="warm", expressiveness=0.5)
        assert sent["language"] == language
        assert (sent["emotion"], sent["expressiveness"]) == ("warm", 0.5)


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
    assert params["emotion"] == prosody.DEFAULT_EMOTION
    assert params["expressiveness"] == prosody.DEFAULT_EXPRESSIVENESS


def test_clean_params_carries_the_style_and_how_far_it_is_taken():
    params = pipeline.clean_params("tts", {"emotion": "warm", "expressiveness": 0.5})
    assert params["emotion"] == "warm"
    assert params["expressiveness"] == 0.5


def test_clean_params_falls_back_on_a_style_we_do_not_have():
    """A language we cannot read is refused because reading it anyway produces
    confident nonsense. A style we do not have costs nothing to ignore."""
    params = pipeline.clean_params("tts", {"emotion": "wistful", "expressiveness": 99})
    assert params["emotion"] == prosody.DEFAULT_EMOTION
    assert params["expressiveness"] == prosody.EXPRESSIVENESS_MAX


def test_the_style_is_not_a_setting_the_other_branches_carry():
    """There is nothing to read on a branch that starts from a recording."""
    for mode in ("song", "speech"):
        assert "emotion" not in pipeline.clean_params(mode)


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
