"""The crude synth take that tells the model which chords to play.

Nobody hears this audio, so there is nothing here about whether it sounds
good — that was `arrange.py`'s bar and `arrange.py` lost. What is worth
holding is that the sketch is *harmonically legible and rhythmically square*,
because those are the two things `init_audio` carries into the output, and that
it stays out of the way on everything else.
"""

import numpy as np
import pytest

from modal_app import sketch
from modal_app.analysis import Track
from modal_app.chords import Chart, Chord

RATE = sketch.SAMPLE_RATE


def track(bpm: float = 120.0, offset: float = 0.37) -> Track:
    return Track(
        bpm=bpm, beat_offset_sec=offset, key=9, minor=True, key_margin=0.2, duration_sec=180.0
    )


def chart(*chords: tuple[int, bool]) -> Chart:
    return Chart(
        tuple(
            Chord(root=root, minor=minor, start_sec=index * 2.0, duration_sec=2.0)
            for index, (root, minor) in enumerate(chords)
        ),
        confidence=0.2,
        bar_sec=2.0,
    )


def spectrum(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = audio * np.hanning(len(audio))
    return np.fft.rfftfreq(len(audio), 1.0 / RATE), np.abs(np.fft.rfft(window))


def test_the_sketch_is_mono_at_the_models_own_rate():
    """`prepare_audio` resamples and re-channels whatever it is given, so any
    stereo image or rate conversion built here is work thrown away."""
    audio = sketch.render(chart((9, True)), track(), 4.0)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) == int(4.0 * RATE)


def test_it_leaves_headroom():
    """An init that clips gives the model distortion to imitate."""
    audio = sketch.render(chart((9, True), (5, False)), track(), 8.0)
    assert 0 < float(np.abs(audio).max()) <= sketch.TARGET_PEAK + 1e-6


def test_a_chord_recogniser_gets_the_chart_back_out_of_the_sketch():
    """The strongest thing worth asserting about this module, and the only one
    that matters: render a known chart, point `chords.detect` at the render,
    and get the same chords back.

    That is the whole contract with the model. `init_audio` carries harmony,
    and harmony the sketch failed to state clearly is harmony Stable Audio will
    replace with its own — at which point the detour through `chords.detect`
    bought nothing and the beat is a prompt-only beat wearing a costume.

    Measured through the same 22.05 kHz decode the real path uses, with the
    bass sounding underneath, because a triad that is only legible in isolation
    is not legible.
    """
    from modal_app.analysis import ANALYSIS_RATE
    from modal_app.audio_utils import decode_audio
    from modal_app.chords import detect

    written = chart((9, True), (5, False), (0, False), (7, False))
    audio = decode_audio(sketch.render_wav(written, track(), 16.0), ANALYSIS_RATE)
    # `render` starts at the top of the chart, so the reader must too.
    read = detect(audio, track(offset=0.0), ANALYSIS_RATE)

    assert read, f"nothing detected: {read}"
    played = [(chord.root, chord.minor) for chord in read.chords]
    assert played[: len(written.chords)] == [
        (chord.root, chord.minor) for chord in written.chords
    ], f"read {played} back from {[c.name for c in written.chords]}"


def test_a_chord_change_is_audible_as_one():
    """Two bars of different triads have to differ, or the chart was rendered
    but not played."""
    audio = sketch.render(chart((9, True), (5, False)), track(), 4.0)
    first = spectrum(audio[int(0.15 * RATE) : int(1.6 * RATE)])[1]
    second = spectrum(audio[int(2.15 * RATE) : int(3.6 * RATE)])[1]
    a, b = first / np.linalg.norm(first), second / np.linalg.norm(second)
    assert float(a @ b) < 0.9, "the two bars are the same chord"


def test_an_empty_chart_is_drums_rather_than_silence():
    """`chords.detect` returns one whenever it was not confident. Drums under a
    voice cannot be harmonically wrong; guessed chords very much can."""
    audio = sketch.render(Chart((), 0.0, 2.0), track(), 4.0)
    assert float(np.abs(audio).max()) > 0.1


def test_the_sketch_starts_at_the_top_rather_than_on_the_songs_first_beat():
    """`arrange.render` carried `beat_offset_sec` because its bed had to line up
    with a specific performance. This one is measured and fitted by `beats.fit`
    afterwards like any other beat, so an offset here would only be a fraction
    of a bar of silence for the model to imitate."""
    audio = sketch.render(chart((9, True)), track(), 2.0)
    lead_in = audio[: int(0.05 * RATE)]
    assert float(np.abs(lead_in).max()) > 0.05, "nothing on beat one"


def test_the_chords_are_not_buried_under_the_drums():
    """The balance is nearly the inverse of `arrange.py`'s, and on purpose: a
    finished bed puts the chords under the kick, an instruction does not."""
    assert sketch.CHORD_GAIN > sketch.KICK_GAIN
    assert sketch.BASS_GAIN > sketch.SNARE_GAIN


def test_the_tempo_picks_the_style_and_the_ranges_leave_no_gap():
    """Disjoint and covering, so the answer never depends on dict order."""
    edges = sorted(style.tempo_range for style in sketch.STYLES.values())
    assert edges[0][0] == 0.0
    for (_, high), (low, _) in zip(edges, edges[1:], strict=False):
        assert high == low
    assert edges[-1][1] >= 999.0
    assert sketch.choose_style(60).label == "Slow"
    assert sketch.choose_style(120).label == "Mid"
    assert sketch.choose_style(160).label == "Fast"


def test_a_track_with_no_tempo_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        sketch.render(chart((9, True)), track(bpm=0.0), 4.0)
    with pytest.raises(ValueError):
        sketch.render(chart((9, True)), track(), 0.0)


def test_render_wav_is_what_the_generator_takes():
    from modal_app.audio_utils import decode_audio

    data = sketch.render_wav(chart((9, True)), track(), 3.0)
    assert data[:4] == b"RIFF"
    decoded = decode_audio(data, RATE)
    assert abs(len(decoded) - 3 * RATE) < RATE * 0.05
