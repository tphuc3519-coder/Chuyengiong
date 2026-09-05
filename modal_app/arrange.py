"""Playing a chord chart back as a different record.

    Chart + Track + style ──► render() ──► bed.wav (44.1k stereo)

The third way to get a backing track, and the only one that is *the same song*.
`beats.py` fits a beat somebody else made; `beatgen.py` invents one that has
nothing to do with the original. This plays the original's own harmony, at the
original's own tempo, on instruments that were synthesised here — every sample
of it generated from arithmetic in this file and none of it copied from
anything.

**What that does and does not buy, stated plainly because the whole feature
turns on it.** It removes the *sound recording*: none of the original master
survives, so there is nothing for a fingerprint of the recording to match. It
does **not** remove the *composition* — the chords are still their chords and
the vocal on top is still singing their melody, which is what a cover is and
what publishers claim. The gain is real and it is narrower than it sounds:
covers are licensable, cheaply and often compulsorily, while masters are
licensed at somebody's discretion and usually not at all.

**And an honest ceiling.** What comes out is a well-programmed backing track,
not a production. Additive synthesis and a few noise bursts through a delay
network get you something clean, in tune and in time; they do not get you the
drum sound of a record somebody spent a week mixing. For hip-hop, lo-fi and
anything where the bed is meant to sit under a voice it is genuinely usable.
For a song whose *arrangement* was the point, it will sound like what it is.

Everything is additive or noise-shaped in the frequency domain — no filters
with feedback, no per-sample loops — so a four minute bed renders in a couple of
seconds of numpy on the CPU container, and every part of it is testable without
a GPU or a sample library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import Track
from .chords import Chart

SAMPLE_RATE = 44100
STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4

# Where the instruments sit, in MIDI note numbers. Bass an octave and a half
# below the chord voicing, which is where a bass sits.
BASS_MIDI = 36
CHORD_MIDI = 60
# The chord is voiced from its root upwards and then dropped so it lands around
# CHORD_MIDI wherever the root happens to be — a chart in B would otherwise
# creep an octave above one in C.
VOICING_CENTRE = 62

# Level per part, before the bus. Set by ear against the vocal that goes over
# it: the bed is a bed.
KICK_GAIN = 0.9
SNARE_GAIN = 0.5
HAT_GAIN = 0.16
BASS_GAIN = 0.55
CHORD_GAIN = 0.28
PAD_GAIN = 0.10

# The bed is normalised to this before it leaves; `loudnorm` sets the real level
# after the vocal is on top of it.
TARGET_PEAK = 0.85

# Reverb, as a handful of delay taps rather than a convolution. A 26k-sample
# impulse response convolved over a four minute track is a 16M-point FFT and a
# quarter of a gigabyte; four multiply-adds at fixed offsets is a rounding error
# and, on a bed that sits under a voice, sounds close enough to the same thing.
REVERB_TAPS_MS = (37.0, 61.0, 89.0, 127.0)
REVERB_GAINS = (0.24, 0.17, 0.12, 0.08)
# Slightly different offsets per channel is most of what makes it sound wide.
REVERB_SPREAD_MS = 7.0


@dataclass(frozen=True)
class Style:
    """One way of playing a chart: which steps of the bar each part lands on.

    Positions are sixteenth notes, 0 to 15. Writing patterns as step numbers
    rather than as note durations is what keeps this readable — a four-on-the
    floor kick is `(0, 4, 8, 12)` and a backbeat snare is `(4, 12)`, and those
    are the same two lines a drum machine has had since 1980.
    """

    label: str
    kick: tuple[int, ...]
    snare: tuple[int, ...]
    hat: tuple[int, ...] = ()
    hat_open: tuple[int, ...] = ()
    bass: tuple[int, ...] = (0,)
    chord: tuple[int, ...] = (0,)
    pad: bool = False
    # How far the off-beat sixteenths are pushed late, as a fraction of a step.
    # Straight is 0; a third of a step is a hard shuffle.
    swing: float = 0.0
    # Multiplies the chord and pad decay. A ballad holds; a trap stab does not.
    sustain: float = 1.0
    fills: bool = True
    # Which tempos `auto` hands to this style. The ranges are disjoint and
    # cover every tempo, so `choose_style` has exactly one answer and it does
    # not depend on what order the dict happens to be in — which is the sort of
    # thing that works until somebody adds a style in the middle.
    tempo_range: tuple[float, float] = field(default=(0.0, 999.0))


STYLES: dict[str, Style] = {
    "lofi": Style(
        "Lo-fi",
        kick=(0, 6, 10),
        snare=(4, 12),
        hat=(0, 2, 4, 6, 8, 10, 12, 14),
        bass=(0, 6, 10),
        chord=(0, 8),
        pad=True,
        swing=0.18,
        sustain=1.4,
        tempo_range=(76.0, 92.0),
    ),
    "boombap": Style(
        "Boom bap",
        kick=(0, 3, 8, 11),
        snare=(4, 12),
        hat=(0, 2, 4, 6, 8, 10, 12, 14),
        hat_open=(14,),
        bass=(0, 8),
        chord=(0, 4, 8, 12),
        swing=0.12,
        sustain=0.6,
        tempo_range=(92.0, 108.0),
    ),
    "pop": Style(
        "Pop",
        kick=(0, 8),
        snare=(4, 12),
        hat=(0, 2, 4, 6, 8, 10, 12, 14),
        bass=(0, 4, 8, 12),
        chord=(0, 4, 8, 12),
        pad=True,
        sustain=1.0,
        tempo_range=(108.0, 132.0),
    ),
    "trap": Style(
        "Trap",
        kick=(0, 7, 10),
        snare=(8,),
        hat=tuple(range(16)),
        hat_open=(6,),
        bass=(0, 7, 10),
        chord=(0,),
        sustain=0.5,
        tempo_range=(132.0, 999.0),
    ),
    "ballad": Style(
        "Ballad",
        kick=(0, 8),
        snare=(8,),
        hat=(0, 4, 8, 12),
        bass=(0, 8),
        chord=(0,),
        pad=True,
        sustain=2.2,
        fills=False,
        tempo_range=(0.0, 76.0),
    ),
}
DEFAULT_STYLE = "auto"


def choose_style(name: str, bpm: float) -> Style:
    """The named style, or the one whose tempo range the song is in.

    `auto` is the default because the tempo is already measured and it is a
    much better guess than any fixed choice: 72 BPM wants a ballad and 150 does
    not, and nobody should have to be told that twice.
    """
    if name in STYLES:
        return STYLES[name]
    for style in STYLES.values():
        low, high = style.tempo_range
        if low <= bpm < high:
            return style
    return STYLES["pop"]


# --- synthesis primitives -------------------------------------------------


def _envelope(length: int, attack_sec: float, decay_sec: float) -> np.ndarray:
    """A percussive envelope: short linear attack, exponential decay."""
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    time = np.arange(length) / SAMPLE_RATE
    decay = np.exp(-time / max(decay_sec, 1e-4))
    attack = np.clip(time / max(attack_sec, 1e-5), 0.0, 1.0)
    return (attack * decay).astype(np.float32)


def _tone(freq: float, length: int, harmonics: int, rolloff: float, phase: float = 0.0):
    """Additive synthesis: `harmonics` partials with `1/k**rolloff` amplitudes.

    Additive rather than a saw through a filter, and the reason is arithmetic
    rather than taste: a resonant filter is a per-sample feedback loop, which in
    numpy is a Python loop over a million samples. Choosing the partial
    amplitudes directly gives the same spectrum in one vectorised expression,
    and it cannot alias — nothing above Nyquist is ever generated to be folded
    back down.
    """
    if length <= 0 or freq <= 0:
        return np.zeros(max(0, length), dtype=np.float32)
    time = np.arange(length) / SAMPLE_RATE
    out = np.zeros(length, dtype=np.float64)
    for partial in range(1, harmonics + 1):
        if freq * partial >= SAMPLE_RATE / 2:
            break
        out += (partial**-rolloff) * np.sin(2 * np.pi * freq * partial * time + phase)
    return (out / max(1.0, out.max() or 1.0)).astype(np.float32)


def _shaped_noise(length: int, low: float, high: float, seed: int) -> np.ndarray:
    """White noise with everything outside `[low, high]` taken out.

    Done as one multiply in the frequency domain. A drum hit is a few thousand
    samples, so the round trip costs nothing and there is no filter to design.
    """
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    noise = np.random.default_rng(seed).standard_normal(length)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(length, 1.0 / SAMPLE_RATE)
    spectrum *= (freqs >= low) & (freqs <= high)
    shaped = np.fft.irfft(spectrum, n=length)
    peak = float(np.abs(shaped).max())
    return (shaped / peak).astype(np.float32) if peak > 0 else np.zeros(length, dtype=np.float32)


def kick(seed: int = 0) -> np.ndarray:
    """A sine whose pitch falls from 110 Hz to 45 Hz. That is the whole drum."""
    length = int(0.32 * SAMPLE_RATE)
    time = np.arange(length) / SAMPLE_RATE
    freq = 45.0 + 65.0 * np.exp(-time / 0.028)
    # Integrating the frequency rather than multiplying by it: a sweep written
    # as sin(2*pi*f(t)*t) is not the sweep it looks like, it bends twice as far.
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    body = np.sin(phase) * _envelope(length, 0.001, 0.075)
    click = _shaped_noise(int(0.006 * SAMPLE_RATE), 1000, 6000, seed) * 0.25
    body[: len(click)] += click
    return body.astype(np.float32)


def snare(seed: int = 1) -> np.ndarray:
    """Noise for the wires, a tone for the shell."""
    length = int(0.22 * SAMPLE_RATE)
    wires = _shaped_noise(length, 180, 8000, seed) * _envelope(length, 0.001, 0.055)
    shell = _tone(190.0, length, 3, 1.5) * _envelope(length, 0.001, 0.030)
    return (0.8 * wires + 0.5 * shell).astype(np.float32)


def hat(open_hat: bool = False, seed: int = 2) -> np.ndarray:
    length = int((0.26 if open_hat else 0.06) * SAMPLE_RATE)
    return (
        _shaped_noise(length, 6500, 16000, seed)
        * _envelope(length, 0.0005, 0.055 if open_hat else 0.012)
    ).astype(np.float32)


def _midi_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def bass_note(midi: int, length: int) -> np.ndarray:
    """Round and short. Eight partials rolling off steeply is a muted bass."""
    return (_tone(_midi_hz(midi), length, 8, 1.3) * _envelope(length, 0.004, 0.28)).astype(
        np.float32
    )


def voicing(chord_semitones: tuple[int, ...]) -> list[float]:
    """Chord tones as MIDI numbers, stacked around `VOICING_CENTRE`.

    Voiced upwards from the root and then transposed by whole octaves until the
    middle of it sits near the centre — otherwise a chart in B is played an
    octave above the same chart in C, which is audible and has nothing to do
    with the music.
    """
    notes = [CHORD_MIDI + step for step in chord_semitones]
    notes.append(notes[0] + 12)
    while sum(notes) / len(notes) > VOICING_CENTRE + 6:
        notes = [note - 12 for note in notes]
    while sum(notes) / len(notes) < VOICING_CENTRE - 6:
        notes = [note + 12 for note in notes]
    return [float(note) for note in notes]


def chord_stab(chord_semitones: tuple[int, ...], length: int, sustain: float) -> np.ndarray:
    """The chord, as two slightly detuned layers so it has a width to it."""
    out = np.zeros(length, dtype=np.float32)
    for note in voicing(chord_semitones):
        for detune, phase in ((0.997, 0.0), (1.003, 1.1)):
            out += _tone(_midi_hz(note) * detune, length, 6, 1.4, phase) * 0.5
    return (out * _envelope(length, 0.012, 0.36 * sustain)).astype(np.float32)


def pad_chord(chord_semitones: tuple[int, ...], length: int) -> np.ndarray:
    """The same notes held, quietly, with a slow attack. Glue, not a part."""
    out = np.zeros(length, dtype=np.float32)
    for note in voicing(chord_semitones):
        out += _tone(_midi_hz(note - 12), length, 10, 2.0)
    time = np.arange(length) / SAMPLE_RATE
    swell = np.clip(time / 0.25, 0.0, 1.0) * np.clip((length / SAMPLE_RATE - time) / 0.4, 0.0, 1.0)
    return (out * swell).astype(np.float32)


# --- putting it in the timeline -------------------------------------------


def _add(track: np.ndarray, sound: np.ndarray, at: int, gain: float) -> None:
    """Mix `sound` into `track` at a sample offset, clipped to the end."""
    if at >= len(track) or gain == 0:
        return
    end = min(len(track), at + len(sound))
    track[at:end] += sound[: end - at] * np.float32(gain)


def _reverb(mono: np.ndarray, offset_ms: float) -> np.ndarray:
    """Delay taps summed back in. Cheap, and enough space for a bed."""
    out = mono.copy()
    for delay_ms, gain in zip(REVERB_TAPS_MS, REVERB_GAINS, strict=True):
        delay = int((delay_ms + offset_ms) * SAMPLE_RATE / 1000)
        if delay <= 0 or delay >= len(mono):
            continue
        out[delay:] += mono[:-delay] * np.float32(gain)
    return out


def render(
    chart: Chart,
    track: Track,
    duration_sec: float,
    style: str = DEFAULT_STYLE,
    seed: int = 0,
) -> np.ndarray:
    """A new backing track for `track`, as `(frames, 2)` float32.

    Plays `chart` on loop for `duration_sec`, starting on the song's own first
    beat so the bed and the voice agree about where the bar is.

    An empty chart is not an error and not silence: it renders the rhythm
    section alone. `chords.detect` returns one whenever it was not confident,
    and drums under a voice cannot be harmonically wrong while guessed chords
    very much can.
    """
    if track.bpm <= 0:
        raise ValueError("cannot arrange without a tempo")
    if duration_sec <= 0:
        raise ValueError("nothing to arrange")

    plan = choose_style(style, track.bpm)
    beat = 60.0 / track.bpm
    step = beat / STEPS_PER_BEAT
    bar = beat * 4
    total = int(duration_sec * SAMPLE_RATE)

    drums = np.zeros(total, dtype=np.float32)
    lows = np.zeros(total, dtype=np.float32)
    mids = np.zeros(total, dtype=np.float32)

    kick_hit, snare_hit = kick(seed), snare(seed + 1)
    closed, opened = hat(False, seed + 2), hat(True, seed + 3)

    bars = int(np.ceil((duration_sec - track.beat_offset_sec) / bar)) + 1
    for index in range(max(0, bars)):
        bar_start = track.beat_offset_sec + index * bar
        if bar_start >= duration_sec:
            break
        # Every fourth bar ends with an extra pair of snares. One line, and it
        # is most of the difference between a loop and an arrangement.
        fill = plan.fills and index % 4 == 3

        for position in range(STEPS_PER_BAR):
            # Swing pushes the odd sixteenths late, which is the difference
            # between a drum machine and somebody playing one.
            late = plan.swing * step if position % 2 else 0.0
            at = int((bar_start + position * step + late) * SAMPLE_RATE)
            if position in plan.kick:
                _add(drums, kick_hit, at, KICK_GAIN)
            if position in plan.snare or (fill and position in (14,)):
                _add(drums, snare_hit, at, SNARE_GAIN)
            if position in plan.hat_open:
                _add(drums, opened, at, HAT_GAIN)
            elif position in plan.hat:
                _add(drums, closed, at, HAT_GAIN * (0.75 if position % 2 else 1.0))

        chord = chart.at(bar_start - track.beat_offset_sec) if chart else None
        if chord is None:
            continue
        semitones = chord.semitones
        for position in plan.bass:
            at = int((bar_start + position * step) * SAMPLE_RATE)
            _add(
                lows,
                bass_note(BASS_MIDI + semitones[0], int(0.5 * beat * SAMPLE_RATE)),
                at,
                BASS_GAIN,
            )
        for position in plan.chord:
            at = int((bar_start + position * step) * SAMPLE_RATE)
            length = int(min(bar, 2 * beat) * SAMPLE_RATE)
            _add(mids, chord_stab(semitones, length, plan.sustain), at, CHORD_GAIN)
        if plan.pad:
            _add(
                mids,
                pad_chord(semitones, int(bar * SAMPLE_RATE)),
                int(bar_start * SAMPLE_RATE),
                PAD_GAIN,
            )

    # Drums and bass down the middle, the harmony spread — which is where a
    # record puts them, and it leaves the centre clear for the voice.
    left = drums + lows + _reverb(mids, 0.0)
    right = drums + lows + _reverb(mids, REVERB_SPREAD_MS)
    stereo = np.stack([left, right], axis=1)
    peak = float(np.abs(stereo).max())
    if peak > 0:
        stereo = stereo * np.float32(TARGET_PEAK / peak)
    return stereo.astype(np.float32)


def render_wav(
    chart: Chart,
    track: Track,
    duration_sec: float,
    style: str = DEFAULT_STYLE,
    seed: int = 0,
) -> bytes:
    """`render`, as 16-bit stereo wav bytes ready for `mixing.mix`."""
    from .audio_utils import encode_wav_channels

    return encode_wav_channels(render(chart, track, duration_sec, style, seed), SAMPLE_RATE)
