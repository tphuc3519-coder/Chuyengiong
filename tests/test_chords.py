"""Chord detection, on progressions whose answer is written in the notes.

Build a bar of C-E-G and it is a C major chord; there is no interpretation
involved and no model needed to check it. What the tests are really protecting
is the *restriction* — that every answer is diatonic to the detected key — since
that is what turns the failure mode from "a semitone clash under somebody's
singing" into "a reharmonisation".
"""

import numpy as np

from modal_app import analysis as an
from modal_app import chords as ch

SR = an.ANALYSIS_RATE


def note(midi: int, seconds: float) -> np.ndarray:
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(seconds * SR)) / SR
    partials = sum(0.6**k * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(4))
    return (0.25 * partials * np.exp(-t * 0.5)).astype(np.float32)


def beat_click(seconds: float) -> np.ndarray:
    """A pulse, so `analysis` has a tempo to hand the detector."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    length = int(0.02 * SR)
    envelope = np.exp(-np.arange(length) / (0.004 * SR))
    audio[:length] = (0.5 * envelope * np.sin(2 * np.pi * 1500 * np.arange(length) / SR)).astype(
        np.float32
    )
    return audio


def bar(midis: list[int], bpm: float = 120.0) -> np.ndarray:
    seconds = 4 * 60.0 / bpm
    voices = [note(midi, seconds) for midi in midis]
    length = min(len(voice) for voice in voices)
    chord = sum(voice[:length] for voice in voices)
    drums = np.zeros(length, dtype=np.float32)
    for index in range(4):
        at = int(index * 60.0 / bpm * SR)
        click = beat_click(0.1)
        drums[at : at + len(click)] += click[: max(0, length - at)]
    return (chord + drums).astype(np.float32)


def song(progression: list[list[int]], repeats: int = 4, bpm: float = 120.0) -> np.ndarray:
    return np.concatenate([np.concatenate([bar(c, bpm) for c in progression])] * repeats)


# C - Am - F - G, with a bass note under each.
I_VI_IV_V = [[48, 60, 64, 67], [45, 57, 60, 64], [41, 53, 57, 60], [43, 55, 59, 62]]
# Am - F - G - Am.
MINOR_LOOP = [[45, 57, 60, 64], [41, 53, 57, 60], [43, 55, 59, 62], [45, 57, 60, 64]]


# --- the vocabulary -------------------------------------------------------


def test_the_diatonic_triads_of_c_major_are_the_ones_everybody_knows():
    assert ch.diatonic(0, False) == (
        (0, False),  # C
        (2, True),  # Dm
        (4, True),  # Em
        (5, False),  # F
        (7, False),  # G
        (9, True),  # Am
        (11, True),  # B(dim, played as Bm)
    )


def test_every_key_has_seven_triads_and_they_are_all_in_the_key():
    for key in range(12):
        for minor in (False, True):
            triads = ch.diatonic(key, minor)
            assert len(triads) == 7
            assert len({root for root, _ in triads}) == 7


def test_a_chord_knows_which_notes_it_is():
    assert ch.Chord(0, False, 0.0, 1.0).semitones == (0, 4, 7)
    assert ch.Chord(9, True, 0.0, 1.0).semitones == (9, 0, 4)


# --- detection ------------------------------------------------------------


def test_a_major_progression_is_read_back_correctly():
    audio = song(I_VI_IV_V)
    chart = ch.detect(audio, an.analyse(audio))
    assert [chord.name for chord in chart.chords[:4]] == ["C", "Am", "F", "G"]


def test_a_minor_progression_is_read_back_correctly():
    audio = song(MINOR_LOOP)
    chart = ch.detect(audio, an.analyse(audio))
    assert [chord.name for chord in chart.chords[:3]] == ["Am", "F", "G"]


def test_every_chord_found_is_inside_the_detected_key():
    """The property the whole design rests on. Mistakes stay diatonic, so the
    worst case is a chord sharing two notes with the truth rather than one a
    semitone away."""
    audio = song(I_VI_IV_V)
    track = an.analyse(audio)
    chart = ch.detect(audio, track)
    allowed = set(ch.diatonic(track.key, track.minor))
    for chord in chart.chords:
        assert (chord.root, chord.minor) in allowed


def test_repeated_bars_are_merged_into_one_chord():
    """Two bars of Am is one Am, not four half-bars of it — otherwise the
    sketch retriggers the chord on every segment boundary."""
    audio = song([[45, 57, 60, 64], [45, 57, 60, 64], [41, 53, 57, 60], [41, 53, 57, 60]])
    chart = ch.detect(audio, an.analyse(audio))
    names = [chord.name for chord in chart.chords]
    assert names[:2] == ["Am", "F"]
    assert chart.chords[0].duration_sec > chart.bar_sec


def test_a_track_with_no_tempo_has_no_chart():
    """No bars means nothing to measure a chord over."""
    held = np.sin(2 * np.pi * 220 * np.arange(SR * 6) / SR).astype(np.float32)
    assert not ch.detect(held, an.analyse(held))


def test_noise_produces_no_chart_rather_than_a_wrong_one():
    """`MIN_CONFIDENCE`: an empty chart tells `sketch` to play drums, and drums
    under a voice cannot be in the wrong key."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 12) * 0.2).astype(np.float32)
    track = an.Track(
        bpm=120.0, beat_offset_sec=0.0, key=0, minor=False, key_margin=0.2, duration_sec=12.0
    )
    assert not ch.detect(noise, track)


# --- reading it back ------------------------------------------------------


def test_the_chart_loops_so_a_short_excerpt_can_fill_a_long_song():
    audio = song(I_VI_IV_V, repeats=1)
    chart = ch.detect(audio, an.analyse(audio))
    span = chart.chords[-1].start_sec + chart.chords[-1].duration_sec
    assert chart.at(0.1) is not None
    assert chart.at(span + 0.1).name == chart.at(0.1).name


def test_an_empty_chart_is_falsy_and_answers_nothing():
    empty = ch.Chart((), 0.0, 2.0)
    assert not empty
    assert empty.at(1.0) is None
    assert "no chart" in str(empty)


def test_the_chart_prints_as_something_a_log_line_can_use():
    audio = song(I_VI_IV_V)
    assert "C" in str(ch.detect(audio, an.analyse(audio)))
