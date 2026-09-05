"""A deliberately crude synth take of a song's own chords.

    Chart + Track ──► render() ──► sketch.wav ──► Stable Audio Open
                                                  (init_audio)
                                                        │
                                                   real instruments

**Nobody listens to what this module makes.** That is not modesty, it is the
design, and it is the reason this code exists after `arrange.py` — which is
where most of it comes from — was deleted for not being good enough.

`arrange.py` rendered the *finished* backing track: additive synthesis, a
reverb, five styles, and a mix. Held against a human arrangement of the same
song it put 55% of its energy below 120 Hz and had nothing above 4 kHz, and
that gap is not a tuning problem — it is the distance between generating
waveforms from arithmetic and a sampled instrument library. So it went.

What comes after it is diffusion, and diffusion changes what the synthesis is
*for*. `generate_diffusion_cond` takes an `init_audio`: start the sampler from
a noised version of this instead of from noise, and the output keeps the
harmony, the tempo and the shape of what it started from while the timbre is
re-imagined from scratch. So the sketch does not have to sound good. It has to
be **harmonically unambiguous and rhythmically square**, and then get out of
the way.

That is a much easier target than the one `arrange.py` missed, and it is why
the same oscillators are worth keeping.

**Three things are different from `arrange.py`, and each is because of the new
job:**

* *No reverb.* Tails are texture, and texture is exactly what the model is
  being asked to replace. A reverb here would be smeared into the output as
  something to keep rather than something to overwrite.
* *No pad, and the chords sit forward.* The one thing the sketch must
  communicate is which triad is sounding. `arrange.py` mixed the chords at 0.28
  under drums at 0.9 because that is where a bed puts them; here the balance is
  nearly the opposite.
* *Mono.* `prepare_audio` inside `stable_audio_tools` resamples and re-channels
  whatever it is given, so a stereo image built here is work thrown away.

**And the copyright argument, which is the whole point of the detour.** The
straightforward way to get the model to follow a song's harmony is to feed it
the song: separate the instrumental, pass *that* as `init_audio`. It works, and
what comes out is a derivative of the master recording — the exact thing this
branch exists to avoid, and the thing audio fingerprinting is built to find
through precisely this kind of transformation.

Going via a chart breaks that chain. `chords.detect` reads a sequence of
triads, which is the *composition* rather than the recording; `render` plays
them on oscillators that have never heard the song. What reaches the model is
audio this repository generated. The output is a cover — the songwriter's
right, which is licensable and in many places compulsory — and not a copy of
somebody's master.

numpy only. This runs on the CPU container beside `analysis` and `chords`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analysis import Track
from .chords import Chart

# The model's own rate, so nothing is resampled on the way in.
SAMPLE_RATE = 44100
STEPS_PER_BAR = 16
STEPS_PER_BEAT = 4
BEATS_PER_BAR = 4

# Where the parts sit. Bass an octave below middle C, chords around it.
BASS_MIDI = 36
CHORD_MIDI = 60
VOICING_CENTRE = 62

# The balance, and the one set of numbers that is genuinely different from
# `arrange.py`. There the chords sat under the drums because that is where a
# finished bed puts them; here the chords *are* the message and the drums are
# only there to say where the bar is. A model that cannot hear the triad will
# invent one.
KICK_GAIN = 0.42
SNARE_GAIN = 0.24
HAT_GAIN = 0.07
BASS_GAIN = 0.62
CHORD_GAIN = 0.85

# Left with headroom rather than pushed to the edge: this is going into an
# encoder, not into somebody's ears, and a sketch that clips gives the model
# distortion to imitate.
TARGET_PEAK = 0.70


@dataclass(frozen=True)
class Style:
    """One way of playing a chart: which sixteenths each part lands on.

    Trimmed from `arrange.py`'s version — no pad, no fills, no swing. All three
    were there to make a loop sound less like a machine, and this one is
    allowed to sound like a machine. Swing in particular is worth losing: it
    pushed the off-beats late, and late off-beats in an init are a groove the
    model has to be talked out of.
    """

    label: str
    kick: tuple[int, ...]
    snare: tuple[int, ...]
    hat: tuple[int, ...] = ()
    bass: tuple[int, ...] = (0,)
    chord: tuple[int, ...] = (0,)
    # Multiplies the chord decay. A slow song holds its chords; a fast one does
    # not, and a chord still ringing under the next one blurs the change.
    sustain: float = 1.0
    tempo_range: tuple[float, float] = field(default=(0.0, 999.0))


# Disjoint ranges covering every tempo, so `choose_style` has exactly one
# answer and it does not depend on dict order.
STYLES: dict[str, Style] = {
    "slow": Style(
        "Slow",
        kick=(0, 8),
        snare=(8,),
        hat=(0, 4, 8, 12),
        bass=(0, 8),
        chord=(0, 8),
        sustain=1.8,
        tempo_range=(0.0, 92.0),
    ),
    "mid": Style(
        "Mid",
        kick=(0, 8),
        snare=(4, 12),
        hat=(0, 2, 4, 6, 8, 10, 12, 14),
        bass=(0, 4, 8, 12),
        chord=(0, 8),
        sustain=1.0,
        tempo_range=(92.0, 132.0),
    ),
    "fast": Style(
        "Fast",
        kick=(0, 4, 8, 12),
        snare=(4, 12),
        hat=tuple(range(0, 16, 2)),
        bass=(0, 4, 8, 12),
        chord=(0, 4, 8, 12),
        sustain=0.6,
        tempo_range=(132.0, 999.0),
    ),
}


def choose_style(bpm: float) -> Style:
    """The style whose tempo range `bpm` falls in.

    No name to pass and no `auto`: the user's taste belongs in the text prompt,
    which is what the model actually listens to for character. All this has to
    decide is how busy the scaffolding is, and the tempo answers that.
    """
    for style in STYLES.values():
        low, high = style.tempo_range
        if low <= bpm < high:
            return style
    return STYLES["mid"]


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


def hat(seed: int = 2) -> np.ndarray:
    """Closed only. An open hat is a tail, and tails are the model's job."""
    length = int(0.06 * SAMPLE_RATE)
    return (_shaped_noise(length, 6500, 16000, seed) * _envelope(length, 0.0005, 0.012)).astype(
        np.float32
    )


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
    """The chord, plainly. One layer, no detune, no width.

    `arrange.py` stacked two detuned copies to give the chord a size to it.
    Detuning is a chorus, a chorus is a texture, and a texture in an init is
    something the model preserves. What is wanted here is the pitch content and
    nothing else attached to it.
    """
    out = np.zeros(length, dtype=np.float32)
    for note in voicing(chord_semitones):
        out += _tone(_midi_hz(note), length, 6, 1.4)
    peak = float(np.abs(out).max())
    if peak > 0:
        out /= peak
    return (out * _envelope(length, 0.010, 0.36 * sustain)).astype(np.float32)


def _add(track: np.ndarray, sound: np.ndarray, at: int, gain: float) -> None:
    """Mix `sound` into `track` at a sample offset, clipped to the end."""
    if at >= len(track) or gain == 0:
        return
    end = min(len(track), at + len(sound))
    track[at:end] += sound[: end - at] * np.float32(gain)


# --- the sketch -----------------------------------------------------------


def render(chart: Chart, track: Track, duration_sec: float, seed: int = 0) -> np.ndarray:
    """`duration_sec` of `chart`, played square, as mono float32.

    Starts at the top of the chart rather than at the song's own first beat,
    and that is the difference between this and `arrange.render`. That function
    made a bed to lie under a specific performance, so it had to agree with it
    about where the bar was. This one makes a loop for a model to re-render;
    what comes back is measured and fitted by `beats.fit` afterwards, exactly
    as an uploaded beat is. Carrying `beat_offset_sec` in here would only put a
    fraction of a bar of silence at the front for the model to imitate.

    An empty chart is not an error and not silence: it renders the rhythm
    section alone. `chords.detect` returns one whenever it was not confident,
    and a model given drums invents its own harmony — which is a worse beat
    than this branch is trying to make, but a better one than a bed confidently
    playing the wrong chords under the singer.
    """
    if track.bpm <= 0:
        raise ValueError("cannot sketch without a tempo")
    if duration_sec <= 0:
        raise ValueError("nothing to sketch")

    plan = choose_style(track.bpm)
    beat = 60.0 / track.bpm
    step = beat / STEPS_PER_BEAT
    bar = beat * BEATS_PER_BAR
    total = int(duration_sec * SAMPLE_RATE)
    out = np.zeros(total, dtype=np.float32)

    kick_hit, snare_hit, hat_hit = kick(seed), snare(seed + 1), hat(seed + 2)

    for index in range(int(np.ceil(duration_sec / bar))):
        bar_start = index * bar
        if bar_start >= duration_sec:
            break

        for position in range(STEPS_PER_BAR):
            at = int((bar_start + position * step) * SAMPLE_RATE)
            if position in plan.kick:
                _add(out, kick_hit, at, KICK_GAIN)
            if position in plan.snare:
                _add(out, snare_hit, at, SNARE_GAIN)
            if position in plan.hat:
                _add(out, hat_hit, at, HAT_GAIN * (0.75 if position % 2 else 1.0))

        chord = chart.at(bar_start) if chart else None
        if chord is None:
            continue
        semitones = chord.semitones
        for position in plan.bass:
            at = int((bar_start + position * step) * SAMPLE_RATE)
            _add(
                out,
                bass_note(BASS_MIDI + semitones[0], int(0.5 * beat * SAMPLE_RATE)),
                at,
                BASS_GAIN,
            )
        for position in plan.chord:
            at = int((bar_start + position * step) * SAMPLE_RATE)
            length = int(min(bar, 2 * beat) * SAMPLE_RATE)
            _add(out, chord_stab(semitones, length, plan.sustain), at, CHORD_GAIN)

    peak = float(np.abs(out).max())
    if peak > 0:
        out = out * np.float32(TARGET_PEAK / peak)
    return out.astype(np.float32)


def render_wav(chart: Chart, track: Track, duration_sec: float, seed: int = 0) -> bytes:
    """`render`, as 16-bit mono wav — what `BeatGenerator.generate` takes."""
    from .audio_utils import encode_wav

    return encode_wav(render(chart, track, duration_sec, seed), SAMPLE_RATE)
