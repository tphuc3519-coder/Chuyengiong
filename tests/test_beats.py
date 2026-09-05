"""Fitting a beat under a voice: the plan, and then the audio.

Two halves, tested two ways. `plan_fit` is arithmetic on two `Track` records,
so every awkward case — a beat with no pulse, a song in no clear key, a minor
loop under a major song — is decided here rather than in a container. `fit`
runs ffmpeg for real, and the assertion that matters is measured rather than
described: put a 90 BPM beat under a 120 BPM song and the thing that comes back
has to be at 120, with its first beat where the song's first beat is.
"""

import shutil

import numpy as np
import pytest

from modal_app import analysis as an
from modal_app import beats
from modal_app.analysis import Track
from modal_app.audio_utils import decode_audio, encode_wav

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def track(bpm=120.0, offset=0.0, key=0, minor=False, margin=0.2, duration=30.0) -> Track:
    return Track(
        bpm=bpm,
        beat_offset_sec=offset,
        key=key,
        minor=minor,
        key_margin=margin,
        duration_sec=duration,
    )


def drums(bpm: float, seconds: float, offset: float = 0.0, tone: float = 110.0) -> np.ndarray:
    """A pulse with a pitch, so both measurements have something to read."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    length = int(0.05 * SR)
    envelope = np.exp(-np.arange(length) / (0.008 * SR))
    hit = (0.7 * envelope * np.sin(2 * np.pi * tone * np.arange(length) / SR)).astype(np.float32)
    time = offset
    while time < seconds:
        start = int(time * SR)
        audio[start : start + length] += hit[: max(0, len(audio) - start)]
        time += 60.0 / bpm
    return audio


# --- tempo folding --------------------------------------------------------


def test_a_beat_at_double_the_tempo_is_already_in_time():
    """The rule that stops the most destructive stretch this could do. A 140 BPM
    loop under a 70 BPM song is playing the same pulse twice as often, which is
    a thing music does on purpose — halving it would be ruining a beat to solve
    a problem it does not have."""
    assert beats.fold_tempo(0.5) == pytest.approx(1.0)
    assert beats.fold_tempo(2.0) == pytest.approx(1.0)
    assert beats.fold_tempo(4.0) == pytest.approx(1.0)


def test_folding_leaves_every_stretch_inside_what_still_sounds_like_music():
    """Whatever two tempos go in, the ratio that comes out is between 0.71 and
    1.41 — which is the range WSOLA survives."""
    for ratio in (0.3, 0.6, 0.9, 1.1, 1.7, 2.4, 3.3):
        folded = beats.fold_tempo(ratio)
        assert 1 / 2**0.5 <= folded <= 2**0.5


def test_nonsense_folds_to_no_change():
    assert beats.fold_tempo(0.0) == 1.0
    assert beats.fold_tempo(float("inf")) == 1.0


# --- transposition --------------------------------------------------------


def test_a_beat_is_moved_to_the_songs_key():
    shift, _ = beats.transpose_to(track(key=0), track(key=2))
    assert shift == 2


def test_the_shorter_way_round_is_taken():
    """Eleven semitones up is one semitone down, and one semitone down is what
    a listener would call it."""
    shift, _ = beats.transpose_to(track(key=0), track(key=11))
    assert shift == -1


def test_a_minor_beat_under_a_major_song_goes_to_the_relative_minor():
    """The rule that makes this musical rather than arithmetic. A minor loop
    under a C major song belongs at A minor, which shares all seven notes with
    it — moving it to C minor is a minor third out and sounds it."""
    shift, why = beats.transpose_to(track(key=0, minor=True), track(key=0, minor=False))
    assert (0 + shift) % 12 == 9  # A
    assert "relative" in why


def test_a_major_beat_under_a_minor_song_goes_to_the_relative_major():
    shift, why = beats.transpose_to(track(key=0, minor=False), track(key=9, minor=True))
    assert (0 + shift) % 12 == 0  # A minor's relative major is C
    assert "relative" in why


@pytest.mark.parametrize("margins", [(0.0, 0.2), (0.2, 0.0)])
def test_nothing_is_transposed_on_a_key_estimate_that_is_a_guess(margins):
    """A key and its relative share every note, so a small margin is the normal
    way for this to be unsure. Transposing on a coin flip moves a beat a minor
    third away from the song, which is worse than leaving it."""
    beat_margin, song_margin = margins
    shift, why = beats.transpose_to(
        track(key=0, margin=beat_margin), track(key=5, margin=song_margin)
    )
    assert shift == 0
    assert why


# --- the plan -------------------------------------------------------------


def test_a_beat_with_no_pulse_cannot_be_fitted_to_anything():
    with pytest.raises(beats.BeatError):
        beats.plan_fit(track(bpm=0.0), track())


def test_a_song_with_no_pulse_leaves_the_beat_at_its_own_tempo():
    """The other way round is recoverable: play the beat as it is rather than
    stretch it to a number that was never in the recording."""
    plan = beats.plan_fit(track(bpm=90), track(bpm=0.0))
    assert plan.tempo_ratio == 1.0
    assert plan.reasons


def test_the_loop_is_cut_to_whole_bars():
    """An arbitrary upload does not end where a bar does, and looping it puts a
    seam in the middle of a beat."""
    plan = beats.plan_fit(track(bpm=120, offset=0.3, duration=17.3), track(bpm=120))
    bar = 60.0 / 120 * beats.BEATS_PER_BAR
    assert plan.loop_start_sec == pytest.approx(0.3)
    assert plan.loop_length_sec % bar == pytest.approx(0.0, abs=1e-6)
    assert plan.loop_start_sec + plan.loop_length_sec <= 17.3


def test_a_beat_too_short_for_two_bars_is_looped_whole_rather_than_refused():
    plan = beats.plan_fit(track(bpm=120, duration=2.5), track(bpm=120))
    assert plan.loop_length_sec == pytest.approx(2.5)
    assert any("looped whole" in reason for reason in plan.reasons)


def test_the_plan_says_what_it_did_and_why():
    plan = beats.plan_fit(track(bpm=90, key=0, minor=True), track(bpm=120, key=0))
    assert "semitone" in str(plan)
    assert plan.reasons


def test_the_pitch_ratio_follows_the_semitones():
    assert beats.Fit(12, 1.0, 0, 4).pitch_ratio == pytest.approx(2.0)
    assert beats.Fit(0, 1.0, 0, 4).pitch_ratio == pytest.approx(1.0)


def test_the_tempo_filter_is_split_when_one_stage_cannot_reach():
    """ffmpeg's `atempo` takes 0.5 to 2.0 per instance. Octave folding keeps the
    ratio inside that, but dividing by the pitch shift can push it to the
    edge — so the guard is here rather than in a comment."""
    assert beats._atempo(1.2).count("atempo") == 1
    assert beats._atempo(3.0).count("atempo") == 2
    assert beats._atempo(0.2).count("atempo") == 2


# --- the audio ------------------------------------------------------------


@needs_ffmpeg
def test_a_fitted_beat_comes_out_at_the_songs_tempo():
    """The measurement the whole module exists for, taken end to end: a 90 BPM
    loop under a 120 BPM song has to come back at 120."""
    beat = encode_wav(drums(90, 16), SR)
    song = encode_wav(drums(120, 30, offset=0.25, tone=160.0), SR)
    bed, plan, source, target = beats.analyse_and_fit(beat, song)
    assert source.bpm == pytest.approx(90, rel=0.01)
    assert target.bpm == pytest.approx(120, rel=0.01)
    fitted, _ = an.tempo(decode_audio(bed, an.ANALYSIS_RATE), an.ANALYSIS_RATE)
    assert fitted == pytest.approx(120, rel=0.01)


@needs_ffmpeg
def test_a_fitted_beat_lands_on_the_songs_first_beat():
    beat = encode_wav(drums(90, 16), SR)
    song = encode_wav(drums(120, 30, offset=0.25, tone=160.0), SR)
    bed, _, _, target = beats.analyse_and_fit(beat, song)
    _, where = an.tempo(decode_audio(bed, an.ANALYSIS_RATE), an.ANALYSIS_RATE)
    period = 60.0 / target.bpm
    distance = min(
        abs(where - target.beat_offset_sec), period - abs(where - target.beat_offset_sec)
    )
    assert distance < 0.03


@needs_ffmpeg
def test_the_bed_is_as_long_as_the_song_however_short_the_loop_was():
    beat = encode_wav(drums(120, 8), SR)
    song = encode_wav(drums(120, 40, tone=160.0), SR)
    bed, _, _, _ = beats.analyse_and_fit(beat, song)
    assert len(decode_audio(bed, SR)) / SR == pytest.approx(40, abs=0.2)


@needs_ffmpeg
def test_an_explicit_length_overrides_the_songs_own():
    beat = encode_wav(drums(120, 8), SR)
    song = encode_wav(drums(120, 40, tone=160.0), SR)
    bed, _, _, _ = beats.analyse_and_fit(beat, song, duration_sec=12.0)
    assert len(decode_audio(bed, SR)) / SR == pytest.approx(12, abs=0.2)


@needs_ffmpeg
def test_the_bed_does_not_stop_dead():
    """A loop cut mid-decay and then trimmed ends on a click without the fade."""
    beat = encode_wav(drums(120, 8), SR)
    song = encode_wav(drums(120, 20, tone=160.0), SR)
    bed, _, _, _ = beats.analyse_and_fit(beat, song)
    audio = decode_audio(bed, SR)
    tail = audio[-int(0.05 * SR) :]
    assert float(np.abs(tail).max()) < 0.2 * float(np.abs(audio).max())


@needs_ffmpeg
def test_transposing_the_beat_moves_its_pitch():
    beat = encode_wav(drums(120, 8, tone=110.0), SR)
    plan = beats.Fit(semitones=7, tempo_ratio=1.0, loop_start_sec=0.0, loop_length_sec=8.0)
    moved = decode_audio(beats.stretch(beat, plan, SR), SR)

    def peak_hz(audio):
        spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
        return float(np.fft.rfftfreq(len(audio), 1 / SR)[int(spectrum.argmax())])

    assert peak_hz(moved) == pytest.approx(110 * 2 ** (7 / 12), rel=0.05)


@needs_ffmpeg
def test_an_empty_beat_is_a_beat_error_and_not_an_ffmpeg_one():
    with pytest.raises(beats.BeatError):
        beats.fit(b"", track(), track(), 10.0)


# --- tonal balance ---------------------------------------------------------


def low_share(audio: np.ndarray, cutoff: float = beats.LOW_HZ) -> float:
    """Share of the power sitting below `cutoff`. The number that went wrong."""
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    return float(spectrum[freqs < cutoff].sum() / spectrum.sum())


def bassy(seconds: float = 4.0) -> np.ndarray:
    """The shape the measurement found: one low note carrying almost everything.

    73 Hz because that is where the real job's loudest bin sat — D2, the root
    of the song it was made for.
    """
    time = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    audio = (
        np.sin(2 * np.pi * 73 * time)
        + 0.10 * np.sin(2 * np.pi * 440 * time)
        + 0.03 * rng.standard_normal(len(time))
    )
    return (audio / np.abs(audio).max() * 0.95).astype(np.float32)


@needs_ffmpeg
def test_a_bed_that_is_almost_all_bass_is_pulled_back_to_the_target():
    """The measured failure, as a test.

    A finished job came back with 71.9% of its power between 40 and 120 Hz
    where a human arrangement of the same song had 23.5%, and 5.3% in the
    2-6 kHz band where that reference had 16.0%. Mud, with the singer behind
    it.
    """
    before = bassy()
    assert low_share(before) > 0.9, "the fixture is not the failure being tested"

    fixed, note = beats.balance(encode_wav(before, SR), SR)
    after = decode_audio(fixed, SR)

    assert low_share(after) == pytest.approx(beats.LOW_SHARE_TARGET, abs=0.03)
    assert "->" in note and "shelf" in note


@needs_ffmpeg
def test_a_bed_that_is_already_balanced_keeps_its_shelf_flat():
    """No shelf where none is needed. This runs on every generated bed, and a
    filter that always fires is a filter that dulls the ones that were fine."""
    time = np.arange(4 * SR) / SR
    rng = np.random.default_rng(1)
    even = (0.3 * np.sin(2 * np.pi * 73 * time) + 0.4 * rng.standard_normal(len(time))).astype(
        np.float32
    )
    assert low_share(even) < beats.LOW_SHARE_TARGET

    fixed, note = beats.balance(encode_wav(even / np.abs(even).max() * 0.9, SR), SR)
    after = decode_audio(fixed, SR)
    # Untouched above the sub, so the share barely moves.
    assert low_share(after) == pytest.approx(low_share(even), abs=0.02)
    assert "+0.0 dB" in note


@needs_ffmpeg
def test_the_sub_goes_whatever_the_rest_of_the_spectrum_is_doing():
    """Rumble that survives into `mixing` is rumble `loudnorm` turns *up*:
    K-weighting barely hears it, so it costs headroom and buys nothing."""
    time = np.arange(4 * SR) / SR
    rumble = (0.6 * np.sin(2 * np.pi * 12 * time) + 0.4 * np.sin(2 * np.pi * 500 * time)).astype(
        np.float32
    )
    after = decode_audio(beats.balance(encode_wav(rumble, SR), SR)[0], SR)
    assert low_share(after, cutoff=20.0) < 0.01


@needs_ffmpeg
def test_the_level_is_set_by_rms_and_the_peak_is_only_a_ceiling():
    """The step that replaces peak normalisation, which is where this started.

    Two beds with the same RMS and very different peaks used to come out at
    very different loudnesses — the one with a single tall transient got turned
    down to fit it, and arrived under the vocal as a whisper.
    """
    time = np.arange(4 * SR) / SR
    rng = np.random.default_rng(2)
    even = 0.2 * rng.standard_normal(len(time))
    spiky = even.copy()
    spiky[SR] = 0.99  # one transient, nothing else different

    a = decode_audio(beats.balance(encode_wav(even.astype(np.float32), SR), SR)[0], SR)
    b = decode_audio(beats.balance(encode_wav(spiky.astype(np.float32), SR), SR)[0], SR)
    rms = lambda x: float(np.sqrt((x**2).mean()))  # noqa: E731
    assert rms(a) == pytest.approx(rms(b), rel=0.05)
    assert np.abs(a).max() <= beats.BALANCE_PEAK + 1e-3
    assert np.abs(b).max() <= beats.BALANCE_PEAK + 1e-3


@needs_ffmpeg
def test_silence_is_refused_rather_than_divided_by():
    with pytest.raises(beats.BeatError):
        beats.balance(encode_wav(np.zeros(SR, dtype=np.float32), SR), SR)
    with pytest.raises(beats.BeatError):
        beats.balance(b"")
