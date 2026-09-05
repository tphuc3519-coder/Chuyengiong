"""What chords a track is playing, restricted to the key it is in.

    instrumental + Track ──► detect() ──► Chart([Chord, Chord, …])
                                                    │
                                          sketch.render() ──► init_audio

The measurement `sketch.py` needs and `analysis.py` does not provide: tempo
says *when* and key says *where*, and neither says what the harmony is doing
between one bar and the next.

**This module was deleted once and is back for a different job.** It fed
`arrange.py`, whose synthesised bed was the *finished* backing track — and that
bed lost to a human arrangement by a distance no tuning closes, so both went.
What is downstream of it now is `sketch.py`, which renders the same chart badly
on purpose and hands it to Stable Audio Open as `init_audio`. The bar moved:
the sketch is not what anybody hears, it is how the model is told which chords
to play. Ugly is fine there. Wrong is not, which is why the restriction below
still matters as much as it ever did.

**The one design decision that makes this usable is the restriction.** A general
chord recogniser picks from 24 triads (or 48, with sevenths) and gets a
respectable fraction of them right; the ones it gets wrong are wrong by
arbitrary amounts, and a chord a semitone away from the truth under somebody's
singing is a car alarm. Restricted to the seven triads that are diatonic to the
detected key, the recogniser is choosing between chords that all *belong*, so
its mistakes stay inside the key — a vi where the truth was a IV shares two of
three notes and passes as a reharmonisation rather than as an error.

That is the trade this module makes on purpose: fewer chords available, and no
catastrophic failure mode. Songs that leave their key (a borrowed IV minor, a
secondary dominant) get the nearest diatonic chord instead, which is what a
musician sight-reading a chart in that key would play too.

**Half a bar at a time**, and then runs of the same chord are merged. Pop
harmony moves on the bar or the half-bar far more often than on the beat, and
averaging chroma over two beats is a much steadier measurement than averaging
over one.

**Confidence is reported and acted on.** Below `MIN_CONFIDENCE` the chart is
empty rather than wrong, and `sketch.py` reads that as "play a rhythm section
and no chords" — a bed with drums and a root note cannot clash with a vocal,
while a bed playing the wrong chords very much can.

numpy only, like everything else that runs on the CPU container.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analysis import ANALYSIS_RATE, NOTE_NAMES, Track, chroma

# Scale degrees of the major and natural minor scales, and whether the triad
# built on each is minor. Diminished triads (vii° in major, ii° in minor) are
# in the list because leaving them out would silently reassign their bars to
# something a third away; they are rendered as their minor triad, which is what
# most pop arrangements play there anyway.
MAJOR_DEGREES = (0, 2, 4, 5, 7, 9, 11)
MAJOR_QUALITIES = (False, True, True, False, False, True, True)
MINOR_DEGREES = (0, 2, 3, 5, 7, 8, 10)
MINOR_QUALITIES = (True, True, False, True, True, False, False)

# Semitones above the root in a triad.
MAJOR_TRIAD = (0, 4, 7)
MINOR_TRIAD = (0, 3, 7)

# How much of a bar is measured at once. Two beats: pop harmony moves on the
# bar or the half-bar much more often than on the beat, and half a bar of
# chroma is a far steadier measurement than a quarter of one.
BEATS_PER_SEGMENT = 2
BEATS_PER_BAR = 4

# Below this margin between the best chord and the runner-up, averaged over the
# track, there is no chart worth playing. See the module docstring: an empty
# chart is a rhythm section, and a wrong chart is a clash.
MIN_CONFIDENCE = 0.02

# The root is what a listener hears the chord as, and it is usually the loudest
# thing in the bass. Weighting it above the third and fifth breaks ties between
# triads that share two notes — which every pair a third apart does, and those
# are exactly the confusions worth breaking.
ROOT_WEIGHT = 1.6
THIRD_WEIGHT = 1.0
FIFTH_WEIGHT = 0.8


@dataclass(frozen=True)
class Chord:
    """One chord and when it is played."""

    root: int
    minor: bool
    start_sec: float
    duration_sec: float

    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.root % 12]}{'m' if self.minor else ''}"

    @property
    def semitones(self) -> tuple[int, ...]:
        """The triad's pitch classes, root first."""
        intervals = MINOR_TRIAD if self.minor else MAJOR_TRIAD
        return tuple((self.root + step) % 12 for step in intervals)


@dataclass(frozen=True)
class Chart:
    """A track's harmony, or the honest absence of one.

    Empty means the detection was not confident enough to be worth playing —
    see `MIN_CONFIDENCE`. Callers read that as "no chords", not as "silence".
    """

    chords: tuple[Chord, ...]
    confidence: float
    bar_sec: float

    def __bool__(self) -> bool:
        return bool(self.chords)

    def at(self, seconds: float) -> Chord | None:
        """Which chord is sounding at `seconds`, looping if the track is longer.

        Looping is the point: a chart measured over a three minute song is used
        to render a bed for the same three minutes, but a chart measured over a
        thirty second excerpt should repeat rather than stop.
        """
        if not self.chords:
            return None
        span = self.chords[-1].start_sec + self.chords[-1].duration_sec
        if span <= 0:
            return None
        position = seconds % span
        for chord in self.chords:
            if chord.start_sec <= position < chord.start_sec + chord.duration_sec:
                return chord
        return self.chords[-1]

    def __str__(self) -> str:
        if not self.chords:
            return "no chart"
        names = " ".join(chord.name for chord in self.chords[:8])
        more = "…" if len(self.chords) > 8 else ""
        return f"{len(self.chords)} chords ({self.confidence:.3f}): {names}{more}"


def diatonic(key: int, minor: bool) -> tuple[tuple[int, bool], ...]:
    """The seven triads of `key`, as `(root, is_minor)`.

    This is the whole vocabulary the detector may choose from, and the reason
    its mistakes are survivable — every option here shares a scale with every
    other one.
    """
    degrees = MINOR_DEGREES if minor else MAJOR_DEGREES
    qualities = MINOR_QUALITIES if minor else MAJOR_QUALITIES
    return tuple(
        ((key + step) % 12, quality) for step, quality in zip(degrees, qualities, strict=True)
    )


def _templates(key: int, minor: bool) -> tuple[np.ndarray, tuple[tuple[int, bool], ...]]:
    """A 12-dimensional weighted template per candidate triad."""
    options = diatonic(key, minor)
    weights = np.zeros((len(options), 12))
    for index, (root, is_minor) in enumerate(options):
        intervals = MINOR_TRIAD if is_minor else MAJOR_TRIAD
        for step, weight in zip(intervals, (ROOT_WEIGHT, THIRD_WEIGHT, FIFTH_WEIGHT), strict=True):
            weights[index, (root + step) % 12] = weight
        weights[index] /= np.linalg.norm(weights[index])
    return weights, options


def _merge(chords: list[Chord]) -> tuple[Chord, ...]:
    """Runs of the same chord become one. Two bars of Am is one Am, not four."""
    merged: list[Chord] = []
    for chord in chords:
        last = merged[-1] if merged else None
        if last is not None and last.root == chord.root and last.minor == chord.minor:
            merged[-1] = Chord(
                root=last.root,
                minor=last.minor,
                start_sec=last.start_sec,
                duration_sec=last.duration_sec + chord.duration_sec,
            )
        else:
            merged.append(chord)
    return tuple(merged)


def detect(audio: np.ndarray, track: Track, sample_rate: int = ANALYSIS_RATE) -> Chart:
    """The chord chart of `audio`, given what `analysis` already measured.

    Needs a tempo: without one there are no bars to measure over, and a chord
    chart with no rhythm to it is not something `sketch` could play. A track
    with no pulse comes back as an empty chart, same as an unconfident one.
    """
    beat = 60.0 / track.bpm if track.bpm > 0 else 0.0
    segment = beat * BEATS_PER_SEGMENT
    if segment <= 0 or len(audio) < segment * sample_rate:
        return Chart((), 0.0, 0.0)

    weights, options = _templates(track.key, track.minor)
    start = max(0.0, track.beat_offset_sec)
    found: list[Chord] = []
    margins: list[float] = []

    while (start + segment) * sample_rate <= len(audio):
        window = audio[int(start * sample_rate) : int((start + segment) * sample_rate)]
        profile = chroma(window, sample_rate)
        if profile.any():
            scores = weights @ profile
            order = np.argsort(scores)[::-1]
            root, is_minor = options[int(order[0])]
            margins.append(float(scores[order[0]] - scores[order[1]]))
            found.append(Chord(root=root, minor=is_minor, start_sec=start, duration_sec=segment))
        start += segment

    if not found:
        return Chart((), 0.0, beat * BEATS_PER_BAR)

    confidence = float(np.mean(margins)) if margins else 0.0
    if confidence < MIN_CONFIDENCE:
        # Measured, and not good enough to play. `sketch` renders a rhythm
        # section instead, which cannot be harmonically wrong.
        return Chart((), confidence, beat * BEATS_PER_BAR)

    # Re-based to zero so `at()` loops cleanly: the first chord starts where the
    # first beat did, and the chart is a thing that repeats rather than a
    # timeline with a gap at the front.
    offset = found[0].start_sec
    rebased = [
        Chord(chord.root, chord.minor, chord.start_sec - offset, chord.duration_sec)
        for chord in found
    ]
    return Chart(_merge(rebased), confidence, beat * BEATS_PER_BAR)


def detect_bytes(data: bytes, track: Track) -> Chart:
    """`detect` for an encoded file, decoded through ffmpeg."""
    from .audio_utils import decode_audio

    return detect(decode_audio(data, ANALYSIS_RATE), track, ANALYSIS_RATE)
