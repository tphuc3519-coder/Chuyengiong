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


# --- the downbeat ---------------------------------------------------------


def kit(
    bpm: float,
    seconds: float = 24.0,
    offset: float = 0.0,
    kicks: tuple[int, ...] = (0, 2),
    snares: tuple[int, ...] = (1, 3),
    bass: bool = True,
) -> np.ndarray:
    """A drum kit playing 4/4, with the bar starting at `offset`.

    The point of building this rather than reusing `clicks` is that a click
    track has no bar in it: every beat is identical, so there is no phase to
    find and nothing to test. What makes a downbeat findable is that the
    instruments differ — a kick under 100 Hz on beat one, a band-limited snare
    two octaves above it on the backbeat — and that difference is exactly what
    `DOWNBEAT_HZ` is there to read.
    """
    rng = np.random.default_rng(0)
    audio = np.zeros(int(seconds * SR) + SR, dtype=np.float32)

    kick_t = np.arange(int(0.22 * SR)) / SR
    kick = (
        np.exp(-kick_t / 0.07)
        * np.sin(2 * np.pi * np.cumsum(110 * np.exp(-kick_t / 0.03) + 45) / SR)
    ).astype(np.float32)

    def band_noise(low: float, high: float, length: float, decay: float) -> np.ndarray:
        count = int(length * SR)
        spectrum = np.fft.rfft(rng.standard_normal(count))
        freqs = np.fft.rfftfreq(count, 1.0 / SR)
        spectrum[(freqs < low) | (freqs > high)] = 0
        shaped = np.fft.irfft(spectrum, count)
        peak = np.abs(shaped).max() or 1.0
        return (shaped / peak * np.exp(-np.arange(count) / (decay * SR))).astype(np.float32)

    snare = 0.6 * band_noise(180, 8000, 0.14, 0.03)
    hat = 0.15 * band_noise(4000, 10000, 0.05, 0.006)

    def add(sound: np.ndarray, at: float) -> None:
        start = int(at * SR)
        if 0 <= start < len(audio):
            audio[start : start + len(sound)] += sound[: len(audio) - start]

    period = 60.0 / bpm
    index, time = 0, offset
    while time < seconds - 0.3:
        if index % 4 in kicks:
            add(kick, time)
        if index % 4 in snares:
            add(snare, time)
        add(hat, time)
        add(hat, time + period / 2)
        if bass and index % 4 == 0:
            length = int(period * 2 * SR)
            envelope = np.exp(-np.arange(length) / (period * 1.2 * SR))
            add(
                (0.4 * envelope * np.sin(2 * np.pi * 55 * np.arange(length) / SR)).astype(
                    np.float32
                ),
                time,
            )
        time += period
        index += 1
    return audio[: int(seconds * SR)]


def bar_error(track, true_offset: float) -> float:
    """How far the found bar line is from the real one, the short way round."""
    bar = 4 * 60.0 / track.bpm
    found = track.downbeat_sec
    return min((found - true_offset) % bar, (true_offset - found) % bar)


@pytest.mark.parametrize("offset", [0.0, 0.13, 0.39, 0.75, 1.17, 1.82])
def test_the_bar_line_is_found_wherever_the_song_starts(offset):
    """The measurement this exists for. A bed aligned to the *beat* is in time
    and in the wrong place in the bar — its kick lands on the song's beat two
    and stays there — so the phase has to come out right for a song that does
    not begin at t=0, which is every song."""
    track = an.analyse(kit(120, offset=offset))
    assert track.has_downbeat
    assert bar_error(track, offset) < 0.1


def test_the_downbeat_is_one_of_the_beats_and_not_between_them():
    """It is a phase, not a search: whatever comes back has to sit on the grid
    `tempo()` already found."""
    track = an.analyse(kit(120, offset=0.39))
    period = 60.0 / track.bpm
    steps = (track.downbeat_sec - track.beat_offset_sec) / period
    assert steps == pytest.approx(round(steps), abs=0.01)


def test_a_kick_on_every_beat_with_no_harmony_is_declined_rather_than_guessed():
    """Four-on-the-floor gives every phase the same low end, and with nothing
    harmonic on top there is no second opinion to break the tie. Answering
    anyway would be a coin flip that then decides where a three minute bed
    sits."""
    track = an.analyse(kit(120, kicks=(0, 1, 2, 3), snares=(), bass=False))
    assert track.downbeat_margin < an.DOWNBEAT_MIN_MARGIN
    assert not track.has_downbeat


def test_nothing_in_the_low_end_and_no_chord_changes_means_no_bar_line():
    """Neither cue has anything to read, and saying so is the answer."""
    track = an.analyse(kit(120, kicks=(), snares=(1, 3), bass=False))
    assert not track.has_downbeat


def test_a_track_with_no_bar_line_falls_back_to_its_first_beat():
    """`bar_start_sec` is the single accessor every caller uses, so the
    fallback lives in one place and cannot be forgotten in another."""
    unsure = an.Track(
        bpm=120.0,
        beat_offset_sec=0.25,
        key=0,
        minor=False,
        key_margin=0.2,
        duration_sec=30.0,
        downbeat_sec=0.75,
        downbeat_margin=an.DOWNBEAT_MIN_MARGIN / 2,
    )
    assert unsure.bar_start_sec == 0.25
    sure = an.Track(**{**unsure.__dict__, "downbeat_margin": 0.4})
    assert sure.bar_start_sec == 0.75


def test_a_track_written_before_downbeats_existed_still_works():
    """The two fields default to "not measured", so every `Track(...)` in this
    codebase that predates them behaves exactly as it did."""
    old = an.Track(
        bpm=120.0, beat_offset_sec=0.4, key=0, minor=False, key_margin=0.2, duration_sec=10.0
    )
    assert not old.has_downbeat
    assert old.bar_start_sec == 0.4


def test_no_pulse_means_no_downbeat_to_look_for():
    silence = np.zeros(SR * 4, dtype=np.float32)
    assert an.downbeat(silence, 0.0, 0.0) == (0.0, 0.0)
    assert not an.analyse(silence).has_downbeat


def test_less_than_a_bar_of_audio_has_no_phase_to_find():
    assert an.downbeat(kit(120, seconds=1.2), 120.0, 0.0)[1] == 0.0


def test_the_low_band_envelope_is_the_only_thing_that_changed():
    """`max_hz` left out has to leave `tempo()` reading exactly the envelope it
    always did — the band limit is a new caller, not a new behaviour."""
    audio = kit(120, seconds=8)
    assert np.array_equal(an.onset_envelope(audio), an.onset_envelope(audio, SR, None))
    limited = an.onset_envelope(audio, SR, max_hz=an.DOWNBEAT_HZ)
    assert len(limited) == len(an.onset_envelope(audio))
    assert not np.array_equal(limited, an.onset_envelope(audio))


def chords_per_bar(
    bpm: float = 100.0,
    seconds: float = 24.0,
    offset: float = 0.0,
    kicks: tuple[int, ...] = (0, 1, 2, 3),
) -> np.ndarray:
    """An arrangement whose bar line is stated by the *harmony* and not by the
    drums: a chord change every bar over a kick on every beat.

    This is the case the low-band cue alone cannot answer and a listener finds
    trivially — four-on-the-floor is most of dance music, and the counting is
    done from the chords.
    """
    audio = np.zeros(int(seconds * SR) + SR, dtype=np.float32)
    progression = [[60, 64, 67], [57, 60, 64], [53, 57, 60], [55, 59, 62]]

    kick_t = np.arange(int(0.2 * SR)) / SR
    kick = (
        np.exp(-kick_t / 0.06)
        * np.sin(2 * np.pi * np.cumsum(110 * np.exp(-kick_t / 0.03) + 45) / SR)
    ).astype(np.float32)

    def add(sound: np.ndarray, at: float) -> None:
        start = int(at * SR)
        if 0 <= start < len(audio):
            audio[start : start + len(sound)] += sound[: len(audio) - start]

    period = 60.0 / bpm
    bar = period * 4
    index, time = 0, offset
    while time < seconds - 0.3:
        if index % 4 in kicks:
            add(kick, time)
        if index % 4 == 0:
            length = int(bar * 0.95 * SR)
            t = np.arange(length) / SR
            chord = sum(note(midi, bar * 0.95)[:length] for midi in progression[(index // 4) % 4])
            add((chord * np.minimum(t / 0.005, 1.0)).astype(np.float32), time)
        time += period
        index += 1
    return audio[: int(seconds * SR)]


@pytest.mark.parametrize("offset", [0.0, 0.23, 0.69, 1.15])
def test_the_bar_line_is_found_from_the_chords_when_the_drums_cannot_say(offset):
    """The whole reason there are two cues. A kick on all four beats says
    nothing about which one is beat one; a chord change once a bar says it
    plainly, and that is how a listener counts a dance track."""
    track = an.analyse(chords_per_bar(offset=offset))
    assert track.has_downbeat
    assert bar_error(track, offset) < 0.12


def test_the_harmonic_cue_alone_finds_what_the_low_band_alone_cannot():
    """Held apart rather than only tested through the combination, so a
    regression in either one is attributable."""
    audio = chords_per_bar(offset=0.23)
    bpm, beat = an.tempo(audio)
    _, start, period = an._grid(bpm, beat, SR / an.HOP)

    low = an._share(an._low_end_phases(audio, start, period, SR))
    harmonic = an._share(an._harmonic_phases(audio, start, period, SR))
    assert low is not None and harmonic is not None
    # A kick on every beat: the low end has no opinion worth acting on.
    assert float(low.max()) - float(np.sort(low)[-2]) < an.DOWNBEAT_MIN_MARGIN
    # The chords do.
    assert float(harmonic.max()) - float(np.sort(harmonic)[-2]) > an.DOWNBEAT_MIN_MARGIN


def test_the_two_cues_are_made_comparable_before_they_are_added():
    """One is onset energy and the other a distance between chroma vectors —
    unrelated units. Normalising each to sum to 1 is what makes the weighting
    mean "how strongly each cue prefers a phase" rather than "which cue got the
    bigger numbers"."""
    assert an._share(np.array([1.0, 2.0, 3.0, 4.0])).sum() == pytest.approx(1.0)
    assert an._share(np.array([100.0, 200.0, 300.0, 400.0])).sum() == pytest.approx(1.0)
    # Two vectors that prefer the same phase equally strongly must weigh the
    # same however loud the recording they came from was.
    assert an._share(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
        an._share(np.array([10.0, 20.0, 30.0, 40.0]))
    )
    assert an._share(np.zeros(4)) is None
    assert an._share(None) is None


def test_a_loud_chord_and_a_quiet_one_are_not_a_chord_change():
    """Chroma rows are normalised before differencing, so the harmonic cue
    measures *which notes* rather than how loud — otherwise it would be a
    second, worse copy of the low-band cue."""
    quiet = note(60, 1.0) + note(64, 1.0) + note(67, 1.0)
    loud = quiet * 4.0
    audio = np.concatenate([quiet, loud] * 8).astype(np.float32)
    scores = an._harmonic_phases(audio, 0.0, len(quiet) / an.HOP, SR)
    if scores is not None:
        share = an._share(scores)
        assert share is None or float(share.max()) - float(np.sort(share)[-2]) < 0.25


def test_too_little_audio_to_see_a_change_twice_says_nothing():
    """One chord change per bar means a single bar is a single sample of a
    single phase, which is not evidence."""
    assert an._harmonic_phases(chords_per_bar(seconds=2.0), 0.0, 21.5, SR) is None
