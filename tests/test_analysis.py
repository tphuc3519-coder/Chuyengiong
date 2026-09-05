"""Tempo and key, measured against signals whose answer is arithmetic.

A click track at exactly 100 BPM is 100 BPM; a progression built only out of
the notes of C major is in C major. That is the whole test strategy here, and it
is available because both measurements are numpy and neither needs a model.

The tolerances are not decoration. 0.5% of tempo error is half a second of
drift over a three minute song, which is the difference between a beat sitting
under a voice and a beat sliding out from under it — so the tests assert
tenths of a percent, and they are the reason `_fit_grid` exists.
"""

import numpy as np
import pytest

from modal_app import analysis as an

SR = an.ANALYSIS_RATE


def clicks(bpm: float, seconds: float = 20.0, offset: float = 0.0) -> np.ndarray:
    """A metronome: short decaying blips at an exact tempo."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    length = int(0.02 * SR)
    envelope = np.exp(-np.arange(length) / (0.004 * SR))
    blip = (0.8 * envelope * np.sin(2 * np.pi * 1200 * np.arange(length) / SR)).astype(np.float32)
    time = offset
    while time < seconds:
        start = int(time * SR)
        audio[start : start + length] += blip[: max(0, len(audio) - start)]
        time += 60.0 / bpm
    return audio


def note(midi: int, seconds: float = 1.0) -> np.ndarray:
    """A plucked note with a few harmonics, so chroma has something real to read."""
    freq = 440.0 * 2 ** ((midi - 69) / 12)
    t = np.arange(int(seconds * SR)) / SR
    partials = sum(0.6**k * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(4))
    return (0.2 * partials * np.exp(-t * 0.8)).astype(np.float32)


def progression(chords: list[list[int]], repeats: int = 4) -> np.ndarray:
    """A chord loop, as the shortest thing that is unambiguously in a key."""
    bars = []
    for chord in chords:
        voices = [note(midi) for midi in chord]
        length = min(len(v) for v in voices)
        bars.append(sum(v[:length] for v in voices))
    return np.concatenate(bars * repeats)


# --- tempo ----------------------------------------------------------------


@pytest.mark.parametrize("bpm", [80, 100, 120, 128, 140, 175])
def test_the_tempo_is_found_to_within_a_tenth_of_a_percent(bpm):
    found, _ = an.tempo(clicks(bpm))
    assert found == pytest.approx(bpm, rel=0.001)


@pytest.mark.parametrize("offset", [0.0, 0.07, 0.19])
def test_the_first_beat_is_found_to_within_fifteen_milliseconds(offset):
    """Where the beat falls, not just how often. A bed aligned to the wrong
    part of the bar is not a backing track."""
    period = 60.0 / 120
    found, where = an.tempo(clicks(120, offset=offset))
    distance = min(abs(where - offset), period - abs(where - offset))
    assert distance < 0.015


def test_a_lag_supported_at_its_multiples_beats_one_that_is_not():
    """The error that ruined a real backing track, as a unit test.

    On a 154 BPM rock arrangement the autocorrelation peaked almost equally at
    the beat period and at one and a half times it — a backbeat puts strong
    onsets on both grids — and plain scoring picked the wrong one by 0.9%. A
    3:2 error is the one that cannot be lived with: two bars of a bed at 103
    span three bars of a song at 154, which is not drift, it is a different
    metre.

    Built here as a correlation rather than as audio, because what is being
    tested is the scoring rule: a true period has support at 1x, 2x, 3x and 4x
    of itself, while a lag half again as long shares only its even multiples.
    """
    period = 20
    correlation = np.zeros(400)
    for multiple in range(1, 20):
        correlation[period * multiple] = 0.9
    # …and a spurious peak at 1.5x, as strong as the real one.
    correlation[period * 3 // 2] = 0.9

    lags = np.arange(10, 60)
    scores = an._comb_score(correlation, lags)
    assert scores[lags == period][0] > scores[lags == period * 3 // 2][0]


def test_the_comb_does_not_simply_prefer_longer_lags():
    """The trap the first version fell into: lags are whole frames and periods
    are not, so scoring at exact integer multiples rewards long lags — their
    multiples land nearer the real peaks. It turned a 120 BPM click track into
    60. The windowed search is what fixes it, and this is the guard."""
    for bpm in (100, 120, 140):
        found, _ = an.tempo(clicks(bpm))
        assert found == pytest.approx(bpm, rel=0.01), f"{bpm} came back as {found}"


def test_a_track_with_no_pulse_says_so_rather_than_guessing():
    """Silence, and a held tone. Both are honest zeros — `beats.py` reads that
    as "do not stretch this" rather than stretching to a made-up number."""
    assert an.tempo(np.zeros(SR * 5, dtype=np.float32)) == (0.0, 0.0)
    # A held tone is the case that made `PULSE_FLOOR` necessary: nothing starts
    # in it, but the numerical dust in its flux autocorrelates as happily as
    # anything else and it used to come back at a confident 108 BPM.
    held = np.sin(2 * np.pi * 220 * np.arange(SR * 5) / SR).astype(np.float32)
    assert an.tempo(held)[0] == 0.0
    noise = (np.random.default_rng(3).standard_normal(SR * 8) * 0.1).astype(np.float32)
    assert an.tempo(noise)[0] == 0.0


def test_the_tempo_stays_inside_the_range_it_searches():
    """Everything musical is between 60 and 200, and a number outside it is a
    detection error rather than a very slow or very fast song."""
    for bpm in (70, 90, 110, 150, 190):
        found, _ = an.tempo(clicks(bpm))
        assert an.BPM_MIN <= found <= an.BPM_MAX


def test_the_grid_fit_is_what_buys_the_accuracy():
    """Without `_fit_grid` the period is one autocorrelation peak rounded to a
    23 ms frame, which is a few tenths of a percent out — inaudible in
    isolation and half a second of drift across a song. This is the test that
    fails if it is ever removed as an optimisation."""
    envelope = an.onset_envelope(clicks(128))
    frames_per_sec = SR / an.HOP
    exact = frames_per_sec * 60.0 / 128.0
    rough = exact * 1.004  # 0.4% out, which is where rounding to a frame leaves it
    period, _ = an._fit_grid(envelope, rough, 0.0)
    assert abs(period - exact) < abs(rough - exact)
    assert 60.0 * frames_per_sec / period == pytest.approx(128, rel=0.002)


def test_the_fit_can_only_refine_a_period_and_never_replace_it():
    """The fit locks onto whatever onsets sit near the grid it was handed, so a
    starting period that was badly wrong would come back as a confident wrong
    answer rather than as an error. The guard is that it may adjust by a
    quarter and no more — past that it is not refining, it is answering a
    different question."""
    envelope = an.onset_envelope(clicks(120))
    rough = SR / an.HOP * 60.0 / 120.0
    for wrong in (rough * 0.4, rough * 1.6, rough * 3.0):
        assert 0.75 * wrong <= an._fit_grid(envelope, wrong, 0.0)[0] <= 1.25 * wrong


# --- key ------------------------------------------------------------------


def test_a_major_progression_is_read_as_its_major_key():
    tonic, minor, margin = an.key(
        progression([[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]])
    )
    assert (an.NOTE_NAMES[tonic], minor) == ("C", False)
    assert margin > an.KEY_MIN_MARGIN


def test_a_minor_progression_is_read_as_its_minor_key():
    tonic, minor, _ = an.key(progression([[57, 60, 64], [62, 65, 69], [64, 68, 71], [57, 60, 64]]))
    assert (an.NOTE_NAMES[tonic], minor) == ("A", True)


def test_transposing_the_music_transposes_the_answer():
    """The strongest evidence that this reads harmony rather than a spectrum:
    the same progression a fifth up comes back a fifth up."""
    c_major = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]]
    g_major = [[midi + 7 for midi in chord] for chord in c_major]
    assert an.key(progression(g_major))[0] == (an.key(progression(c_major))[0] + 7) % 12


def test_noise_is_reported_as_having_no_key_worth_using():
    """`KEY_MIN_MARGIN` is the guard `beats.py` reads before transposing
    anything: a key and its relative share all seven notes, so a small margin
    is the normal way to be unsure rather than a rare one."""
    noise = (np.random.default_rng(0).standard_normal(SR * 5) * 0.1).astype(np.float32)
    assert an.key(noise)[2] < an.KEY_MIN_MARGIN


def test_silence_has_no_key_and_does_not_divide_by_zero():
    assert an.key(np.zeros(SR * 3, dtype=np.float32)) == (0, False, 0.0)
    assert not an.chroma(np.zeros(SR * 3, dtype=np.float32)).any()


def test_chroma_is_a_distribution():
    weights = an.chroma(progression([[60, 64, 67]]))
    assert len(weights) == 12
    assert weights.sum() == pytest.approx(1.0)


# --- the record -----------------------------------------------------------


def test_analyse_reports_both_measurements_and_the_length():
    track = an.analyse(clicks(120, seconds=12))
    assert track.bpm == pytest.approx(120, rel=0.001)
    assert track.duration_sec == pytest.approx(12, abs=0.1)
    assert track.key_name in {f"{name}{suffix}" for name in an.NOTE_NAMES for suffix in ("", "m")}


def test_a_track_prints_as_something_a_log_line_can_use():
    assert "BPM" in str(an.analyse(clicks(120, seconds=8)))
