"""How the text is read: the plan, not the audio.

Everything in `prosody.plan` is pure Python against a list of sentences, which
is the whole reason it is a separate module — the rules that decide how a
paragraph is read can be checked here, exactly, without a checkpoint or a
container. The audio half (`shape`) needs librosa and only its level change is
exercised: what a phase vocoder does to a waveform is not something a unit test
can assert about, and the fallback for a box without librosa is.
"""

import numpy as np
import pytest

from modal_app import prosody
from modal_app.prosody import Beat

# --- the styles -----------------------------------------------------------


def test_natural_is_the_default_and_is_offered():
    assert prosody.DEFAULT_EMOTION in prosody.EMOTIONS
    assert prosody.EMOTIONS[prosody.DEFAULT_EMOTION].label


def test_the_natural_style_is_every_field_at_its_identity():
    """It has to be, or `expressiveness` has nothing to scale against and the
    default reading is somebody's idea of a mood rather than a plain one."""
    natural = prosody.EMOTIONS[prosody.DEFAULT_EMOTION]
    assert (natural.rate, natural.pitch_range, natural.pause) == (1.0, 1.0, 1.0)
    assert (natural.pitch, natural.gain_db) == (0.0, 0.0)
    assert (natural.variation, natural.duration_variation) == (1.0, 1.0)


def test_an_unknown_style_falls_back_rather_than_raising():
    """The opposite of what an unknown language does, and for the opposite
    reason: reading Vietnamese with the English checkpoint is confident
    nonsense, reading it with no style is the reading this app shipped with."""
    assert prosody.resolve_emotion("wistful") is prosody.EMOTIONS[prosody.DEFAULT_EMOTION]
    assert prosody.resolve_emotion(None) is prosody.EMOTIONS[prosody.DEFAULT_EMOTION]
    assert prosody.clean_emotion("wistful") == prosody.DEFAULT_EMOTION
    assert prosody.clean_emotion("sad") == "sad"


def test_every_style_carries_a_label_to_put_in_the_picker():
    assert all(style.label for style in prosody.EMOTIONS.values())


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, prosody.DEFAULT_EXPRESSIVENESS),
        ("nonsense", prosody.DEFAULT_EXPRESSIVENESS),
        (0, prosody.EXPRESSIVENESS_MIN),
        (-3, prosody.EXPRESSIVENESS_MIN),
        (9.0, prosody.EXPRESSIVENESS_MAX),
        (0.4, 0.4),
    ],
)
def test_expressiveness_is_clamped_not_rejected(given, expected):
    assert prosody.clamp_expressiveness(given) == pytest.approx(expected)


def test_the_styles_move_in_the_directions_the_recordings_do():
    """The acoustic-emotion literature is consistent about the direction of
    each of these even where it disagrees about the magnitude: high arousal is
    faster, higher and louder with a wider range and shorter gaps; sadness is
    all five the other way."""
    cheerful, sad = prosody.EMOTIONS["cheerful"], prosody.EMOTIONS["sad"]
    assert cheerful.rate > 1.0 > sad.rate
    assert cheerful.pitch > 0.0 > sad.pitch
    assert cheerful.gain_db > 0.0 > sad.gain_db
    assert cheerful.pitch_range > 1.0 > sad.pitch_range
    assert sad.pause > 1.0 > cheerful.pause


# --- what closes a sentence -----------------------------------------------


@pytest.mark.parametrize(
    ("segment", "kind"),
    [
        ("Xin chào.", prosody.SENTENCE),
        ("Thật à?", prosody.QUESTION),
        ("Tuyệt vời!", prosody.EXCLAMATION),
        ("Ừ thì…", prosody.TRAILING),
        ("Ừ thì...", prosody.TRAILING),
        ("Hôm nay trời đẹp,", prosody.CLAUSE),
        ("một câu bị cắt giữa chừng", prosody.RUN_ON),
        ("今日はいい天気ですね。", prosody.SENTENCE),
        ("元気ですか？", prosody.QUESTION),
    ],
)
def test_a_segment_is_classified_by_what_closes_it(segment, kind):
    assert prosody.classify(segment) == kind


def test_a_japanese_question_ending_in_ka_is_a_question_without_a_question_mark():
    """「そうですか。」 is a question and ends in a full stop. Only the final
    position is read, so the か that means "or" mid-sentence is untouched."""
    assert prosody.classify("そうですか。") == prosody.QUESTION
    assert prosody.classify("元気ですか") == prosody.QUESTION
    assert prosody.classify("コーヒーか紅茶をください。") == prosody.SENTENCE


def test_a_closing_quote_does_not_hide_the_punctuation_under_it():
    assert prosody.classify('"Thật à?"') == prosody.QUESTION
    assert prosody.classify("「本当ですか?」") == prosody.QUESTION


def test_the_pause_lengths_are_ordered_the_way_the_punctuation_is():
    """A comma is not a full stop and a full stop is not a blank line. Every
    one of these sat at a single 0.25s before there was a plan."""
    assert (
        prosody.PAUSE_SEC[prosody.RUN_ON]
        < prosody.PAUSE_SEC[prosody.CLAUSE]
        < prosody.PAUSE_SEC[prosody.SENTENCE]
        < prosody.PAUSE_SEC[prosody.TRAILING]
        < prosody.PARAGRAPH_PAUSE_SEC
    )


def test_the_pauses_stay_inside_the_published_bands():
    """120-300ms for a phrase break, 400-700ms for a paragraph or a dramatic
    one. The numbers are somebody else's measurements, not a guess."""
    assert 0.12 <= prosody.PAUSE_SEC[prosody.CLAUSE] <= 0.30
    assert 0.30 <= prosody.PAUSE_SEC[prosody.SENTENCE] <= 0.70
    assert 0.40 <= prosody.PARAGRAPH_PAUSE_SEC <= 0.80


# --- the plan -------------------------------------------------------------


def plan(blocks, **kwargs):
    return prosody.plan(blocks, **kwargs)


def test_one_beat_per_segment_in_the_order_they_were_written():
    beats = plan([["Một.", "Hai."], ["Ba."]])
    assert [beat.text for beat in beats] == ["Một.", "Hai.", "Ba."]


def test_nothing_follows_the_last_segment_so_nothing_pauses_after_it():
    beats = plan([["Một.", "Hai."]])
    assert beats[-1].pause_sec == 0.0
    assert beats[0].pause_sec > 0.0


def test_a_blank_line_is_a_longer_silence_than_a_full_stop():
    same = plan([["Một.", "Hai."]])[0].pause_sec
    across = plan([["Một."], ["Hai."]])[0].pause_sec
    assert across > same


def test_a_question_rises_and_a_statement_does_not():
    rises, flat = plan([["Thật à?", "Ừ."]])
    assert rises.rise > 0
    assert flat.rise == 0


def test_pitch_declines_across_a_paragraph_and_starts_again_at_the_next():
    first, _, last, opening = plan([["Một.", "Hai.", "Ba."], ["Bốn."]])
    assert first.pitch > last.pitch
    # The next paragraph opens back up rather than carrying on downward.
    assert opening.pitch > last.pitch


def test_the_declination_is_centred_so_it_does_not_move_the_measured_f0():
    """`pipeline._resolve_shift` measures the median F0 of the whole spoken wav
    against the reference. A paragraph that drifted downward overall would move
    that measurement, and the shift it produces is applied to everything."""
    beats = plan([["Một.", "Hai.", "Ba.", "Bốn.", "Năm."]])
    assert sum(beat.pitch for beat in beats) == pytest.approx(0.0, abs=0.4)


def test_the_last_sentence_of_a_paragraph_is_read_slightly_slower():
    beats = plan([["Một.", "Hai."]])
    assert beats[-1].rate < beats[0].rate


def test_an_exclamation_is_louder_and_an_ellipsis_is_quieter():
    loud, quiet = plan([["Tuyệt vời!", "Ừ thì…"]])
    assert loud.gain_db > 0 > quiet.gain_db
    assert quiet.rate < loud.rate


def test_the_speaking_rate_the_user_asked_for_is_what_the_style_multiplies():
    slow = plan([["Một."]], speaking_rate=0.8)[0].rate
    fast = plan([["Một."]], speaking_rate=1.4)[0].rate
    assert slow < fast
    assert slow == pytest.approx(0.8 * prosody.FINAL_LENGTHENING)


def test_a_sad_read_is_slower_with_longer_gaps_than_a_cheerful_one():
    sad = plan([["Một.", "Hai."]], emotion="sad")
    cheerful = plan([["Một.", "Hai."]], emotion="cheerful")
    assert sad[0].rate < cheerful[0].rate
    assert sad[0].pause_sec > cheerful[0].pause_sec
    assert sad[0].pitch < cheerful[0].pitch


def test_expressiveness_zero_is_the_flat_read_this_used_to_produce():
    beats = plan([["Thật à?", "Tuyệt vời!"], ["Ừ thì…"]], emotion="cheerful", expressiveness=0)
    assert all(beat.pitch == 0 and beat.gain_db == 0 and beat.rise == 0 for beat in beats)
    assert all(beat.rate == pytest.approx(1.0) for beat in beats)
    assert all(beat.variation == 1.0 for beat in beats)


def test_a_flat_read_still_pauses_where_the_punctuation_says_to():
    """The length of the silence after a comma is punctuation, not emotion.
    Turning the expression off is not a licence to stop reading the page."""
    beats = plan([["Một,", "Hai."], ["Ba."]], expressiveness=0)
    assert beats[0].pause_sec == pytest.approx(prosody.PAUSE_SEC[prosody.CLAUSE])
    assert beats[1].pause_sec == pytest.approx(prosody.PARAGRAPH_PAUSE_SEC)


def test_more_expressiveness_takes_every_deviation_further():
    half = plan([["Thật à?"]], emotion="cheerful", expressiveness=0.5)[0]
    whole = plan([["Thật à?"]], emotion="cheerful", expressiveness=1.0)[0]
    assert abs(half.pitch) < abs(whole.pitch)
    assert half.rise < whole.rise
    assert abs(half.gain_db) < abs(whole.gain_db)


def test_a_style_moves_the_noise_scales_as_multipliers_not_values():
    """The checkpoint's own defaults differ per language, so a style says "a
    bit more than whatever this one does" rather than naming a number."""
    natural = plan([["Một."]])[0]
    assert (natural.variation, natural.duration_variation) == (1.0, 1.0)
    assert plan([["Một."]], emotion="cheerful")[0].variation > 1.0
    assert plan([["Một."]], emotion="serious")[0].variation < 1.0


def test_an_empty_paragraph_contributes_nothing():
    assert plan([[], ["Một."], []]) == plan([["Một."]])


def test_a_one_sentence_paragraph_has_nowhere_to_decline_to():
    only = plan([["Một."]])[0]
    assert only.pitch == pytest.approx(-prosody.FINAL_FALL_ST)


# --- applying it ----------------------------------------------------------


def test_the_level_change_is_applied_to_the_audio():
    audio = np.full(1000, 0.5, dtype=np.float32)
    louder = prosody.shape(audio, 16000, Beat("x", prosody.SENTENCE, gain_db=6.0))
    assert float(np.abs(louder).max()) == pytest.approx(0.5 * 10 ** (6 / 20), rel=1e-3)


def test_a_beat_with_nothing_to_apply_leaves_the_audio_alone():
    audio = np.linspace(-0.4, 0.4, 500, dtype=np.float32)
    assert np.allclose(prosody.shape(audio, 16000, Beat("x", prosody.SENTENCE)), audio)


def test_shaping_never_clips_because_the_join_normalises_afterwards():
    """`tts._join` scales the finished wav to a fixed peak. Clipping here would
    throw away the headroom it is about to use."""
    audio = np.full(1000, 0.95, dtype=np.float32)
    assert (
        float(np.abs(prosody.shape(audio, 16000, Beat("x", prosody.SENTENCE, 1, 0, 6.0))).max())
        > 1.0
    )


def test_a_pitch_move_under_the_audible_threshold_is_not_worth_a_vocoder_pass():
    assert prosody.MIN_AUDIBLE_ST > 0
    audio = np.linspace(-0.4, 0.4, 500, dtype=np.float32)
    tiny = Beat("x", prosody.SENTENCE, pitch=prosody.MIN_AUDIBLE_ST / 2)
    assert np.allclose(prosody.shape(audio, 16000, tiny), audio)


def test_a_segment_too_short_to_have_a_tail_is_not_bent():
    """The rise is the last third of a second of a sentence. A segment barely
    longer than that would have all of itself transposed, which is not a final
    rise — it is a higher sentence."""
    audio = np.linspace(-0.4, 0.4, int(0.2 * 16000), dtype=np.float32)
    assert np.allclose(prosody.shape(audio, 16000, Beat("x", prosody.QUESTION, rise=2.0)), audio)


def test_the_pitch_moves_are_capped_for_a_tonal_language():
    """A whole sentence transposed is safe — the tone contours move with it.
    The final rise is a contour of its own laid over the last syllable, and a
    large one is the difference between asking a question and saying another
    word. Vietnamese is the default language of this app."""
    beats = plan(
        [["Thật à?", "Tuyệt vời!", "Ừ."]],
        emotion="cheerful",
        expressiveness=prosody.EXPRESSIVENESS_MAX,
    )
    assert all(abs(beat.pitch) <= prosody.MAX_SENTENCE_PITCH_ST for beat in beats)
    assert all(beat.rise <= prosody.MAX_RISE_ST for beat in beats)


@pytest.mark.parametrize(
    ("length", "window"),
    [(32000, 2048), (1400, 1024), (900, 512), (300, 256), (100, 256)],
)
def test_the_transposition_window_shrinks_to_fit_short_audio(length, window):
    """A glide step is ~90ms — 1400 samples at 16 kHz — and librosa's default
    2048 window is longer than that. A window longer than the signal is padding
    rather than analysis, and it warns on stderr once per step."""
    assert prosody._window(length) == window
