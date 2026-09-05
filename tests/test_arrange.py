"""The synthesised backing track: does it play the right thing, in time.

Nothing here can tell you whether it sounds good — that needs ears and is on the
verify list. What it can tell you is everything that would make it *wrong*: a
bed at the wrong tempo, chords that are not the chart's, drums that do not land
on the beat, a rhythm section that vanishes when the chart is empty, and a
render that takes longer than the pipeline has.
"""

import numpy as np
import pytest

from modal_app import analysis as an
from modal_app import arrange, chords
from modal_app.audio_utils import decode_wav_channels

SR = arrange.SAMPLE_RATE


def track(bpm=120.0, offset=0.0, key=0, minor=False, duration=30.0) -> an.Track:
    return an.Track(
        bpm=bpm,
        beat_offset_sec=offset,
        key=key,
        minor=minor,
        key_margin=0.2,
        duration_sec=duration,
    )


def chart(names: list[tuple[int, bool]], bar_sec: float = 2.0) -> chords.Chart:
    built = [
        chords.Chord(root=root, minor=minor, start_sec=index * bar_sec, duration_sec=bar_sec)
        for index, (root, minor) in enumerate(names)
    ]
    return chords.Chart(tuple(built), 0.2, bar_sec)


def band_energy(mono: np.ndarray, low: float, high: float) -> float:
    freqs = np.fft.rfftfreq(len(mono), 1.0 / SR)
    spectrum = np.abs(np.fft.rfft(mono.astype(np.float64)))
    inside = spectrum[(freqs >= low) & (freqs < high)]
    return float(np.sqrt((inside**2).mean())) if inside.size else 0.0


# --- shape and level ------------------------------------------------------


def test_the_bed_is_stereo_and_the_length_that_was_asked_for():
    """Stereo is not a luxury: `mixing.mix` negotiates one channel layout across
    its inputs and settles on the narrowest, so a mono bed would fold the whole
    mix to mono."""
    bed = arrange.render(chart([(0, False), (9, True)]), track(), duration_sec=8.0)
    assert bed.shape == (int(8.0 * SR), 2)


def test_the_bed_is_normalised_and_never_clips():
    bed = arrange.render(chart([(0, False)]), track(), duration_sec=6.0)
    assert float(np.abs(bed).max()) == pytest.approx(arrange.TARGET_PEAK, abs=1e-3)


def test_the_two_channels_differ_so_the_mix_has_a_width():
    bed = arrange.render(chart([(0, False), (5, False)]), track(), duration_sec=8.0)
    assert not np.allclose(bed[:, 0], bed[:, 1])


def test_rendering_is_fast_enough_for_a_whole_song():
    """The bed is built on the CPU container inside the mixing step, so a four
    minute render has to cost seconds and not minutes."""
    import time

    started = time.time()
    arrange.render(chart([(0, False), (9, True), (5, False), (7, False)]), track(), 60.0)
    assert time.time() - started < 20.0


# --- is it playing the chart ---------------------------------------------


def test_the_chord_that_is_playing_is_the_chord_in_the_chart():
    """A C major bar and an F# major bar have almost no notes in common, so the
    spectra have to differ where the roots are."""
    c_major = arrange.render(chart([(0, False)]), track(), 4.0, style="pop")
    f_sharp = arrange.render(chart([(6, False)]), track(), 4.0, style="pop")

    # The root of each, in the bass register the arranger voices it at.
    def root_hz(midi):
        return 440.0 * 2 ** ((midi - 69) / 12)

    c_bass = root_hz(arrange.BASS_MIDI)
    f_bass = root_hz(arrange.BASS_MIDI + 6)
    mono_c, mono_f = c_major.mean(axis=1), f_sharp.mean(axis=1)
    assert band_energy(mono_c, c_bass * 0.95, c_bass * 1.05) > band_energy(
        mono_f, c_bass * 0.95, c_bass * 1.05
    )
    assert band_energy(mono_f, f_bass * 0.95, f_bass * 1.05) > band_energy(
        mono_c, f_bass * 0.95, f_bass * 1.05
    )


def test_the_voicing_stays_in_one_octave_whatever_the_key():
    """Otherwise a chart in B is played an octave above the same chart in C,
    which is audible and has nothing to do with the music."""
    centres = []
    for root in range(12):
        notes = arrange.voicing(chords.Chord(root, False, 0.0, 1.0).semitones)
        centres.append(sum(notes) / len(notes))
    assert max(centres) - min(centres) < 12


def test_the_bed_comes_out_at_the_tempo_it_was_given():
    bed = arrange.render(chart([(0, False), (9, True)]), track(bpm=96.0), 24.0, style="pop")
    mono = bed.mean(axis=1)
    resampled = mono[:: int(SR / an.ANALYSIS_RATE)]
    found, _ = an.tempo(resampled.astype(np.float32), an.ANALYSIS_RATE)
    assert found == pytest.approx(96.0, rel=0.03)


def test_the_bed_starts_where_the_songs_first_beat_is():
    """A bed that starts at zero under a song whose first beat is half a second
    in is a bed half a second out."""
    late = arrange.render(chart([(0, False)]), track(offset=0.5), 6.0, style="pop")
    opening = np.abs(late[: int(0.4 * SR)]).max()
    assert opening < 0.05 * float(np.abs(late).max())


# --- the fallback ---------------------------------------------------------


def test_an_empty_chart_still_produces_a_rhythm_section():
    """`chords.detect` returns an empty chart when it was not confident, and
    drums under a voice cannot be in the wrong key while guessed chords can."""
    bed = arrange.render(chords.Chart((), 0.0, 2.0), track(), 8.0)
    assert float(np.abs(bed).max()) > 0.1


def test_the_rhythm_section_has_no_harmony_in_it():
    """It has to actually be drums — a bed that quietly kept playing a chord it
    was not confident about would defeat the fallback."""
    mono = arrange.render(chords.Chart((), 0.0, 2.0), track(), 8.0).mean(axis=1)
    with_chords = arrange.render(chart([(0, False)]), track(), 8.0).mean(axis=1)
    # The chord voicing lives around middle C; drums have very little there.
    assert band_energy(mono, 200, 500) < 0.5 * band_energy(with_chords, 200, 500)


# --- styles ---------------------------------------------------------------


def test_a_style_is_chosen_from_the_tempo_when_nobody_names_one():
    assert arrange.choose_style("auto", 70.0).label == "Ballad"
    assert arrange.choose_style("auto", 150.0).label == "Trap"
    assert arrange.choose_style("auto", 110.0).label == "Pop"


def test_the_tempo_ranges_are_disjoint_and_leave_no_tempo_unanswered():
    """`choose_style` walks the styles in dict order, so overlapping ranges would
    make the answer depend on where somebody inserted a style."""
    for bpm in range(40, 220, 2):
        matches = [
            style
            for style in arrange.STYLES.values()
            if style.tempo_range[0] <= bpm < style.tempo_range[1]
        ]
        assert len(matches) == 1, bpm


def test_naming_a_style_overrides_the_tempo():
    assert arrange.choose_style("lofi", 150.0).label == "Lo-fi"


def test_an_unknown_style_falls_back_rather_than_failing():
    """By the time this is read the GPU has already run. A mistyped style is
    not worth any of that."""
    assert arrange.choose_style("dubstep", 120.0) is not None


@pytest.mark.parametrize("style", sorted(arrange.STYLES))
def test_every_style_renders(style):
    bed = arrange.render(chart([(0, False), (9, True)]), track(), 6.0, style=style)
    assert float(np.abs(bed).max()) > 0.1


def test_every_style_puts_its_drums_on_sixteenths_of_a_bar():
    """The patterns are step numbers, and a step outside the bar is a hit that
    lands in the next one."""
    for style in arrange.STYLES.values():
        for pattern in (
            style.kick,
            style.snare,
            style.hat,
            style.hat_open,
            style.bass,
            style.chord,
        ):
            assert all(0 <= step < arrange.STEPS_PER_BAR for step in pattern)


# --- the wav --------------------------------------------------------------


def test_the_wav_round_trips_as_stereo_at_the_right_rate():
    data = arrange.render_wav(chart([(0, False)]), track(), 4.0)
    frames, rate = decode_wav_channels(data)
    assert rate == SR
    assert frames.shape[1] == 2


def test_a_song_with_no_tempo_cannot_be_arranged():
    with pytest.raises(ValueError):
        arrange.render(chart([(0, False)]), track(bpm=0.0), 8.0)


def test_no_duration_is_nothing_to_arrange():
    with pytest.raises(ValueError):
        arrange.render(chart([(0, False)]), track(), 0.0)
