"""Mixdown and final encode, done with ffmpeg.

Plain subprocess calls, no numpy: this is the one step where a filter graph
beats array arithmetic, and keeping it dependency-free means the separation
container (which has ffmpeg but not our audio stack) can reuse `sum_stems`.

Three rules, all from the plan:

* `amix=normalize=0` — the default normalize=1 divides every input by the
  number of inputs, which drops a two-input mix by 6 dB for no reason;
* `loudnorm=I=-14` last, so the mix lands on the streaming reference level
  instead of wherever the sum happened to end up;
* an `AI-generated` comment tag on every output we hand back (Phase 6 item 2,
  done here because this is the only place output bytes are created).

Both entry points also take a `clarity` amount, which is `enhance`'s chain
inserted **on the voice and nowhere else** — ahead of `amix` in `mix`, so a
song's instrumental reaches the mix exactly as the separator produced it. Zero
emits no filters at all, which is what keeps the old output reachable.

Because that chain is on one side only, its **latency** is on one side only
too, and a filter delay that lands on the voice alone is the singer arriving
late. `chain_latency` measures it and `mix` takes it back off — see those two
and `tests/test_alignment.py`, which holds the 25 ms this cost before anybody
went looking for it.

`mix` also takes an optional **bed profile** — a `styles.Mixdown` — and that is
where a beat stops being "a second piece of audio playing at the same time" and
starts being a backing track. Without one the graph is exactly what it always
was: two inputs, `amix`, `loudnorm`. With one, the bed is shelved, has a bell
cut where the consonants live, and is **ducked by the voice** through
`sidechaincompress`, which is the single filter that most separates a record
from a karaoke track. `mix_bed` builds that chain and `bed_graph` wires it, both
as strings, so the whole thing is testable without a GPU.

`mix` and `to_mp3` take an optional `watermark` callable, applied to the
normalised wav in between the mix and the encode (plan §8, "Cân nhắc thêm").
It is a callable rather than an import because the model behind it needs torch
and its own container: this module stays plain ffmpeg, which is what lets the
separation container reuse `sum_stems` and lets CI test the whole path with a
stand-in.
"""

from __future__ import annotations

import array
import io
import math
import subprocess
import tempfile
import wave
from collections.abc import Callable
from pathlib import Path

from . import enhance
from .styles import Mixdown

# -14 LUFS with 1 dB of headroom: the streaming reference, and the right target
# for something that will be listened to on a phone.
LOUDNORM = "loudnorm=I=-14:TP=-1.0"
# The mono vocal, copied to both channels at unity. See `mix`.
CENTRE = "pan=stereo|c0=c0|c1=c0"
OUTPUT_BITRATE = "192k"
# Beyond this the vocal is either buried or clipping the mix; the slider in the
# UI has no business going further.
MAX_VOCAL_GAIN_DB = 12.0
AI_COMMENT = "AI-generated voice conversion"

# --- the bed --------------------------------------------------------------

# The compressor the ducking is done with, and the two numbers that are not a
# style's business.
#
# **Ratio is fixed and the threshold is what a style moves.** Both change how
# far the bed comes down, so leaving both adjustable would be two controls for
# one result, and the pair that produces a given depth would not be unique.
# Ratio 4:1 is the ordinary answer for a bus compressor and it leaves the knee
# soft enough that the bed is heard going down rather than being switched off.
DUCK_RATIO = 4.0
# 10 ms is fast enough to be under the first syllable and slow enough that the
# very transient of a consonant survives it — the bed dips *behind* the word
# rather than being cut by it. 300 ms is about the gap between two sung phrases,
# which is what the bed should be back up for.
DUCK_ATTACK_MS = 10.0
DUCK_RELEASE_MS = 300.0
# Peak rather than RMS: the key is being used as a control signal, and what
# decides whether a word is happening is its peak, not the average of it and
# the silence around it.
DUCK_DETECTION = "peak"

# How the depth a style asks for becomes a threshold.
#
# For a key normalised to 0 dBFS peak, a compressor pulls a signal down by
# `overshoot × (1 - 1/ratio)`, and the overshoot is the whole distance from the
# threshold to 0. So the threshold that produces `duck_db` of ducking is
# `-duck_db / (1 - 1/ratio)`, which at 4:1 is `-duck_db / 0.75`.
#
# **Measured against that formula it comes back about 20% shallow**: asking for
# 6.0, 3.0, 1.5 and 9.0 dB on a steady key produced 4.9, 2.0, 1.0 and 7.9. The
# gap is the detector — it does not sit at the peak, it follows the signal — and
# it is not corrected here on purpose. A style's `duck_db` is a dial with a
# scale on it, not a contract; correcting the formula to hit the nominal number
# would be pretending this is calibrated when what it actually is, is
# monotonic, which is the property a dial needs.
DUCK_LAW = 1.0 - 1.0 / DUCK_RATIO

# How far the corrective gain may go when the bed is placed against the voice.
#
# The correction exists because the two sides arrive at unrelated levels: a
# generated bed comes out of `beats.balance` at a fixed RMS, an uploaded one is
# at whatever loudness somebody mastered it to, and a converted vocal is at
# whatever Seed-VC produced. Without measuring, a style's balance is a wish.
#
# 15 dB covers every real case — a quiet upload against a hot vocal is maybe
# 10 — and stops a nearly-silent bed from being amplified into its own noise
# floor, which is the failure mode of an unbounded correction.
MAX_BED_TRIM_DB = 15.0

# Everything below this in the bed is rumble, and rumble is expensive twice
# over: it eats headroom, and `loudnorm` at the end barely hears it (K-weighting
# rolls off hard down there) so it turns the *whole mix* down to make room for
# something nobody can hear.
#
# `beats.balance` already does this to a generated bed. Doing it here as well
# is not a duplicate: `balance` never runs on an uploaded beat, and an uploaded
# beat is exactly the file most likely to have a mastered-in sub nobody asked
# for. Idempotent on a bed that has already had it.
BED_HIGHPASS_HZ = 30.0

# Bell width for the vocal pocket, as **Q** — `equalizer` is given `t=q`, so
# this is a quality factor and not a number of octaves. 1.4 is a bandwidth of
# about one octave: narrow enough that the cut is a place rather than a tone
# control, wide enough that it does not ring.
POCKET_WIDTH = 1.4
# Where the bed's low shelf turns over. Below a male voice's fundamental, so
# what moves is the weight of the arrangement and not the part of it a listener
# hears as pitch.
SHELF_HZ = 110.0
SHELF_WIDTH = 0.7


class MixError(RuntimeError):
    """ffmpeg refused to produce output."""


def clamp_gain_db(gain_db: float) -> float:
    try:
        value = float(gain_db)
    except (TypeError, ValueError):
        return 0.0
    return max(-MAX_VOCAL_GAIN_DB, min(MAX_VOCAL_GAIN_DB, value))


def _sample_rate(wav: bytes) -> int:
    """The rate of a wav this app wrote itself.

    `wave` and not ffprobe because it is stdlib and this module stays free of
    everything else; safe because the only wav reaching here is one `encode_wav`
    produced, which is plain PCM by construction.
    """
    if not wav:
        raise MixError("the vocal is empty")
    try:
        with wave.open(io.BytesIO(wav), "rb") as src:
            return src.getframerate()
    except (wave.Error, EOFError) as exc:  # truncated headers raise EOFError
        raise MixError(f"cannot read the vocal's sample rate: {exc}") from exc


def _seconds(wav: bytes) -> float:
    """How long a wav this app wrote is. 0.0 when it cannot be read.

    For the log line in `mix` and nothing else, which is why it never raises:
    a mix must not fail over a number that was only going to be printed.
    """
    try:
        with wave.open(io.BytesIO(wav), "rb") as src:
            rate = src.getframerate()
            return src.getnframes() / float(rate) if rate else 0.0
    except (wave.Error, EOFError):
        return 0.0


def _channels(wav: bytes) -> int:
    """How many channels a wav this app produced has.

    Two, in the honest "I could not tell" case. Everything reaching `mix` as a
    bed is PCM — `beats.lay_under` and the separator both write one — but the
    consequence of guessing wrong matters, and the two guesses are not
    symmetric: calling a stereo file mono applies `CENTRE` and throws away its
    right channel, while calling a mono file stereo leaves it mono, which
    everything downstream already copes with.
    """
    try:
        with wave.open(io.BytesIO(wav), "rb") as src:
            return src.getnchannels()
    except (wave.Error, EOFError):
        return 2


def _stereo(wav: bytes) -> str:
    """The filter that makes `wav` two channels, or `""` if it already is.

    `pan` and never `aformat`, and the difference is measurable: an `aformat`
    upmix applies the -3.01 dB centre mix level, so a mono bed sent through it
    arrives 30% quiet and every level decision above it is off by that much.
    Measured on a 0.5 amplitude tone: `aformat` returns 0.354, `pan` returns
    0.500.
    """
    return "" if _channels(wav) >= 2 else CENTRE


def _peak_db(wav: bytes) -> float:
    """The loudest sample in `wav`, in dBFS. 0.0 when it cannot be measured.

    One `volumedetect` pass with no output file — it decodes and throws the
    audio away, which is a second of CPU on a three minute song and is the only
    way the ducking threshold means anything. `sidechaincompress` compares the
    key against an absolute level, so without normalising the key first the
    same style would duck a quiet vocal not at all and a loud one into the
    floor.

    0.0 on failure is the safe answer rather than the accurate one: it applies
    no correction, so a file this cannot read is ducked by whatever its own
    level happens to earn instead of by a number invented here.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.wav"
        path.write_bytes(wav)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-af",
                "volumedetect",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
        )
    for line in reversed(proc.stderr.decode("utf-8", "replace").splitlines()):
        if "max_volume:" in line:
            try:
                return float(line.split("max_volume:")[1].strip().split()[0])
            except (IndexError, ValueError):
                return 0.0
    return 0.0


def _loudness_db(wav: bytes) -> float | None:
    """Integrated loudness in LUFS, or `None` when it cannot be measured.

    `ebur128` rather than `volumedetect`, and the difference is the whole
    reason this is worth a second ffmpeg pass. A lead vocal is more silence
    than singing, and `mean_volume` averages the silence in: measured on a test
    vocal that is 60% gaps, `mean_volume` said -16.8 dB while the gated
    integrated loudness said -13.2 LUFS. Balancing a bed against the first
    number places it against how much the singer *rests*.

    The R128 gate is what fixes that — it discards everything more than 10 LU
    below the running average, so what comes back is the loudness of the parts
    that are actually playing. Which is the number a person means when they say
    one thing is louder than another.

    `None` for a file too short to gate (under ~0.4 s), for silence, and for
    anything ffmpeg refuses — and every caller treats it as "apply no
    correction" rather than guessing, because a balance computed from a
    measurement that did not happen is worse than the level the file came with.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "in.wav"
        path.write_bytes(wav)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(path),
                "-af",
                "ebur128=framelog=quiet",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
        )
    # The summary block prints "I:  -13.2 LUFS" a line or two after
    # "Integrated loudness:". Read the last one, because a graph with more than
    # one ebur128 in it would print more than one and the last is ours.
    lines = proc.stderr.decode("utf-8", "replace").splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if "Integrated loudness" not in lines[index]:
            continue
        for line in lines[index : index + 4]:
            stripped = line.strip()
            if stripped.startswith("I:"):
                try:
                    value = float(stripped.split()[1])
                except (IndexError, ValueError):
                    return None
                # Silence reads as -inf, or as a floor near -70 with nothing
                # above the absolute gate. Neither is a level to balance to.
                return value if math.isfinite(value) and value > -70.0 else None
        return None
    return None


# --- the vocal chain's own delay ------------------------------------------

# What one measured latency costs to find out, cached for the life of the
# container. Keyed by (chain, rate) because both change the answer: `afftdn`
# sizes its window from the sample rate, so the same chain came back 343
# samples late at 22.05 kHz and 1102 at 44.1 kHz.
_LATENCY: dict[tuple[str, int], int] = {}

# The probe. A burst of tone after a stretch of digital silence, so the onset
# in the output is unambiguous: nothing causal can start before it, and the
# distance between the two onsets is the delay.
_PROBE_SEC = 0.4
_PROBE_TONE_HZ = 1000.0
_PROBE_BURST_SEC = 0.05
# Where the burst starts inside the probe, and how loud a sample has to be to
# count as the onset — a fraction of that file's own peak, so a chain with
# gain in it is measured on its own terms.
_PROBE_START = 0.5
_PROBE_ONSET = 0.25


def _probe_wav(rate: int) -> tuple[bytes, int]:
    """`(wav bytes, onset sample)` for the latency probe at `rate`."""
    total = int(_PROBE_SEC * rate)
    onset = int(total * _PROBE_START)
    burst = int(_PROBE_BURST_SEC * rate)
    samples = array.array("h", bytes(2 * total))
    for i in range(burst):
        samples[onset + i] = int(24000 * math.sin(2 * math.pi * _PROBE_TONE_HZ * i / rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue(), onset


def _onset(wav: bytes) -> int:
    """The first sample in `wav` above `_PROBE_ONSET` of its own peak."""
    with wave.open(io.BytesIO(wav), "rb") as src:
        samples = array.array("h")
        samples.frombytes(src.readframes(src.getnframes()))
    peak = max((abs(v) for v in samples), default=0)
    if peak <= 0:
        return 0
    floor = peak * _PROBE_ONSET
    for i, value in enumerate(samples):
        if abs(value) >= floor:
            return i
    return 0


def chain_latency(chain: str, rate: int) -> int:
    """How many samples `chain` delays what goes through it. Measured, not assumed.

    **The vocal is the only side of a song mix that goes through a filter, so
    every sample of latency in it is the singer landing behind the band.**
    `afftdn` is where it comes from: an overlap-add FFT denoiser cannot answer
    for a sample until it has the rest of that sample's window, and ffmpeg does
    not compensate for it — measured with a click, the Phase 10 clarity chain
    put the voice **25 ms behind** the untouched instrumental at 44.1 kHz, and
    15.6 ms behind at 22.05 kHz. Nothing else in the chain contributes more
    than a millisecond; the biquads are 1.6 ms and the de-esser is none.

    Measured rather than tabulated because the number is not ours: it belongs
    to whichever ffmpeg the container was built with, and `afftdn` grew a
    `window_size` option between the build CI has and the next one. A table
    would be right until the image is rebuilt and then silently wrong, which is
    the worst of the two failure modes — a wrong compensation moves the voice
    the other way.

    One ffmpeg pass over 0.4 s of audio, cached per container. That is a few
    milliseconds of CPU against a job that has already spent minutes of GPU.
    0 for an empty chain, and 0 for anything this cannot measure: leaving the
    voice where it was is the behaviour this function replaced.
    """
    if not chain:
        return 0
    key = (chain, rate)
    if key in _LATENCY:
        return _LATENCY[key]
    probe, onset = _probe_wav(rate)
    try:
        filtered = _ffmpeg([probe], f"[0:a]{chain}anull[out]", ["-c:a", "pcm_s16le"], ".wav")
        latency = max(0, _onset(filtered) - onset)
    except (MixError, wave.Error, EOFError):
        latency = 0
    _LATENCY[key] = latency
    return latency


def advance(samples: int) -> str:
    """The filter that pulls a stream `samples` earlier, or `""` for none.

    `asetpts` is not optional next to `atrim`: `atrim` drops the samples but
    keeps the timestamps they had, so without it `amix` puts the voice back
    exactly where the trim just took it from.
    """
    if samples <= 0:
        return ""
    return f",atrim=start_sample={samples},asetpts=N/SR/TB"


def bed_trim_db(profile: Mixdown, vocal_wav: bytes, bed_wav: bytes, vocal_gain_db: float) -> float:
    """The gain that puts the bed where the style says, relative to the voice.

    This is the function that turns `bed_below_voice_db` from a wish into a
    setting. Both sides are measured, and the bed is moved so its loudness
    lands `bed_below_voice_db` under the voice's — so the same style produces
    the same balance whether the beat arrived from a GPU at a fixed RMS or as
    somebody's mastered upload eight dB louder.

    **The vocal gain slider is part of the target, not applied after it.** The
    voice the bed is being balanced against is the voice as the user asked for
    it, so turning the voice up moves the bed down with it and the slider does
    what a balance slider is expected to do. (The ducking is the one thing that
    slider deliberately does not touch — see `mix`.)

    Two honest limits, both stated rather than corrected for:

    * The bed is measured **before** its own pocket, shelf and ducking, so a
      style that ducks hard ends up a little quieter than the number says. The
      direction is right — heavier ducking should read as a bed further back —
      and correcting it would mean measuring a signal that does not exist until
      after the graph has run.
    * `bed_below_voice_db` itself is **reasoned, not measured**, exactly like
      the rest of the table. What this function buys is not a correct absolute
      balance, it is a *reproducible* one: whatever the right number turns out
      to be, it will mean the same thing on every job once somebody with ears
      has set it.

    0.0 whenever either side cannot be measured, which leaves the two files at
    the levels they came with — the behaviour this app had before.
    """
    voice = _loudness_db(vocal_wav)
    bed = _loudness_db(bed_wav)
    if voice is None or bed is None:
        return 0.0
    target = voice + clamp_gain_db(vocal_gain_db) - profile.clamped().bed_below_voice_db
    return max(-MAX_BED_TRIM_DB, min(MAX_BED_TRIM_DB, target - bed))


def mix_bed(profile: Mixdown, rate: int, stereo: str = "", trim_db: float = 0.0) -> str:
    """The bed's own filters, before the voice is anywhere near it.

    The shelf and the bell are omitted when their setting is zero — the same
    rule `enhance.chain` follows, and for the same reason: a style that asks
    for nothing should produce very nearly the graph this module had before
    styles existed, so "turn it off" stays reachable.

    `trim_db` is the measured placement from `bed_trim_db`, and it goes in
    ahead of the tone shaping so the shelf and the bell act on the bed at the
    level it will actually sit at rather than at the level it arrived with.

    `aresample` is not one of the three and is never omitted.
    `sidechaincompress` compares two streams sample for sample and refuses a
    pair that disagree about the rate, and the bed and the voice arrive from
    different places at different rates as a matter of course.
    """
    profile = profile.clamped()
    # Rumble first, before anything else looks at the bed: the shelf below is
    # fitted to what is left, and a sub nobody wants should not be part of what
    # it is fitted to.
    parts = [f"aresample={rate}", f"highpass=f={BED_HIGHPASS_HZ:.0f}"]
    if stereo:
        parts.insert(0, stereo)
    if trim_db:
        # The measured placement, applied before the tone shaping so the shelf
        # and the bell act on the bed at the level it will actually sit at.
        parts.append(f"volume={trim_db:.2f}dB")
    if profile.low_shelf_db:
        parts.append(f"bass=g={profile.low_shelf_db:.2f}:f={SHELF_HZ:.0f}:t=q:w={SHELF_WIDTH}")
    if profile.pocket_db:
        parts.append(
            f"equalizer=f={profile.pocket_hz:.0f}:t=q:w={POCKET_WIDTH}:g={profile.pocket_db:.2f}"
        )
    return ",".join(parts)


def ducks(profile: Mixdown | None) -> bool:
    """Whether this profile wants the voice to push the bed down at all.

    Asked in two places — here, before the sidechain is built, and in `mix`,
    before the key is split off — and it has to give the same answer in both.
    Hence a function rather than the comparison written twice.
    """
    return profile is not None and profile.clamped().duck_db > 0


def bed_graph(
    profile: Mixdown,
    rate: int,
    vocal_peak_db: float,
    stereo: str = "",
    trim_db: float = 0.0,
) -> str:
    """The whole second half of a ducked mix: `[1:a]` and `[key]` in, `[bed]` out.

    Expects a `[key]` label carrying the voice and leaves `[bed]` behind for
    `amix`. Written as its own function because it is a string, and a string is
    the only part of a mix CI can check without ears.

    Two details that are not decoration:

    * **The key is padded.** `sidechaincompress` is a framesync filter and
      framesync stops at the shorter input, so a vocal that ends before the
      song does would truncate the bed at that point — measured: an 8 second
      bed with a 4 second key came back 4 seconds long. `apad` makes the key
      endless, the bed becomes the shorter input, and the bed's own length is
      what survives. It also does the musically right thing at the end: the key
      goes silent rather than repeating its last frame, so the bed comes back
      up under the outro instead of staying ducked through it.
    * **The key is levelled, not the voice.** `volume` here moves the copy the
      compressor listens to and nothing that anybody hears, so the ducking
      depth a style asks for is the depth it gets whether the converted vocal
      came back hot or quiet. It also means the vocal gain slider changes the
      balance of the mix and not how hard the bed ducks, which are two
      different things a user should be able to set separately.
    """
    profile = profile.clamped()
    bed = mix_bed(profile, rate, stereo, trim_db)
    if not ducks(profile):
        # No ducking asked for, so no `[key]` is read. `mix` asks the same
        # question before it splits one off — an unconsumed label is not a
        # quiet no-op in ffmpeg, it is "Error : Invalid argument" and a failed
        # job at the mix step.
        return f"[1:a]{bed}[bed]"

    threshold = 10.0 ** (-(profile.duck_db / DUCK_LAW) / 20.0)
    return (
        f"[1:a]{bed}[bedraw];"
        f"[key]volume={-vocal_peak_db:.2f}dB,aresample={rate},apad[keypad];"
        f"[bedraw][keypad]sidechaincompress="
        f"threshold={threshold:.6f}:ratio={DUCK_RATIO}:"
        f"attack={DUCK_ATTACK_MS:.0f}:release={DUCK_RELEASE_MS:.0f}:"
        f"detection={DUCK_DETECTION}:makeup=1[bed]"
    )


def _ffmpeg(inputs: list[bytes], filter_complex: str, args: list[str], suffix: str) -> bytes:
    """Run one filter graph over N in-memory inputs and return the output file.

    Files rather than pipes on both ends: the mp3 muxer wants to seek back and
    write its header, and a wav input needs a size in its header before ffmpeg
    will treat it as anything but a stream.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        for i, data in enumerate(inputs):
            if not data:
                raise MixError(f"input {i} is empty")
            path = tmpdir / f"in{i}.wav"
            path.write_bytes(data)
            cmd += ["-i", str(path)]
        out = tmpdir / f"out{suffix}"
        cmd += ["-filter_complex", filter_complex, "-map", "[out]", *args, str(out)]

        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not out.is_file():
            detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
            raise MixError(f"ffmpeg failed: {detail[-1] if detail else 'no output'}")
        return out.read_bytes()


def _mp3_args() -> list[str]:
    return [
        "-c:a",
        "libmp3lame",
        "-b:a",
        OUTPUT_BITRATE,
        "-metadata",
        f"comment={AI_COMMENT}",
    ]


def _apply(mixed_wav: bytes, watermark: Callable[[bytes], bytes] | None) -> bytes:
    """Watermark if asked, then encode. The one place that order is decided.

    After `loudnorm`, never before: loudnorm applies a gain and a limiter, and
    a watermark added ahead of it would be rescaled by both. Before the encode,
    so what ships is what was marked.

    The encode is a separate ffmpeg pass even when nothing is watermarked. It
    could be folded back into the mix graph in that case, but then turning
    `WATERMARK` off would exercise a different code path than the one we test —
    and the cost of the extra pass is one 16-bit round trip of a file that came
    from 16-bit stems.
    """
    if watermark is None:
        return encode_mp3(mixed_wav)
    marked = watermark(mixed_wav)
    if not marked:
        raise MixError("watermarking returned no audio")
    return encode_mp3(marked)


def encode_mp3(audio_wav: bytes) -> bytes:
    """Tag and encode, no filtering. `anull` because `_ffmpeg` maps `[out]`."""
    return _ffmpeg([audio_wav], "[0:a]anull[out]", _mp3_args(), ".mp3")


def mix(
    vocal_wav: bytes,
    instrumental_wav: bytes,
    vocal_gain_db: float = 0.0,
    watermark: Callable[[bytes], bytes] | None = None,
    clarity: float | None = enhance.DEFAULT_CLARITY,
    bed: Mixdown | None = None,
) -> bytes:
    """Converted vocal over the original instrumental -> mp3 bytes.

    The two inputs are rarely the same length — conversion resamples and joins
    chunks — so `amix` runs to the longer of the two rather than truncating the
    song at whichever stem ends first.

    The `pan` is not cosmetic. `amix` negotiates one channel layout across its
    inputs and settles on the narrowest, so a mono vocal — and Seed-VC only
    ever returns mono — silently folded the instrumental's stereo image down
    with it. Copying the vocal to both channels first leaves the backing track
    as wide as it arrived. `pan` rather than an `aformat` upmix because
    ffmpeg's mono-to-stereo conversion applies the -3 dB centre mix level,
    which would quietly move the vocal back down in the mix.

    `clarity` runs first of all, on the vocal alone. Before the gain rather
    than after it, because the de-esser and the denoiser both have thresholds
    and a gain applied ahead of them would move what they act on.

    **`bed` is what makes a replacement backing track sound like one.** With a
    profile the bed is measured against the voice and *placed* rather than
    merely gained — see `bed_trim_db` — then shelved, pocketed and ducked. When
    it is None this function is exactly what it was before styles existed
    — same graph, same two filters, same output. That is deliberate and is
    worth keeping true: `song` mode mixes a vocal back over the instrumental it
    was separated from, which was balanced by whoever made the record, and a
    duck applied to it would be this app disagreeing with them. A *replacement*
    bed has no such claim on the mix: nothing about it was chosen with this
    singer in mind, which is why the beat branches pass a profile and this one
    does not.
    """
    gain = clamp_gain_db(vocal_gain_db)
    # `loudnorm` runs at 192 kHz internally and hands that rate on, so the wav
    # coming out is four times the size it needs to be and — because ffmpeg
    # describes a rate that high with an explicit channel layout — carries a
    # WAVE_FORMAT_EXTENSIBLE header. The watermark step then could not read its
    # own input ("unknown format: 65534") and the job died one stage from done.
    # Putting the rate back settles both: `amix` had already agreed on the
    # vocal's, which is what this restores.
    rate = _sample_rate(vocal_wav)
    # Measured before the graph is built, because it is an argument to the
    # graph: two `ebur128` passes that decode and throw the audio away, which
    # is a couple of seconds of CPU against a job that has already spent
    # minutes of GPU.
    trim = 0.0 if bed is None else bed_trim_db(bed, vocal_wav, instrumental_wav, gain)
    clarity_chain = enhance.chain(clarity, ",")
    # …and the same for the delay that chain adds. The instrumental goes into
    # `amix` unfiltered, so an uncompensated one is the singer arriving late on
    # every job — see `chain_latency`.
    late = chain_latency(clarity_chain, rate)
    voice = f"{clarity_chain}volume={gain:.2f}dB,{CENTRE}{advance(late)}"
    # The two durations a listener hears as "the voice does not sit on the
    # beat", printed side by side so a real job answers the question instead of
    # a pair of ears having to.
    print(
        f"[mix] voice {_seconds(vocal_wav):.2f}s over bed {_seconds(instrumental_wav):.2f}s "
        f"at {rate} Hz, {enhance.describe(clarity)}, voice pulled {late / rate * 1000:.1f} ms "
        "earlier to undo its own filter delay"
    )
    if bed is None:
        graph = (
            f"[0:a]{voice}[v];"
            f"[v][1:a]amix=inputs=2:normalize=0:duration=longest[m];"
            f"[m]{LOUDNORM},aresample={rate}[out]"
        )
    else:
        # The key is split off **before** the voice chain, at the input, which
        # is the one place its level is known: `_peak_db` measured that file
        # and nothing else. Taken after the gain instead, the vocal gain slider
        # would silently double as a ducking control — turn the voice up 6 dB
        # and the bed ducks 6 dB harder, which is not what a level slider
        # means and is not something a user could ever infer.
        split = "[0:a]asplit=2[raw][key];" if ducks(bed) else "[0:a]anull[raw];"
        graph = (
            f"{split}"
            f"[raw]{voice}[v];"
            f"{bed_graph(bed, rate, _peak_db(vocal_wav), _stereo(instrumental_wav), trim)};"
            f"[v][bed]amix=inputs=2:normalize=0:duration=longest[m];"
            f"[m]{LOUDNORM},aresample={rate}[out]"
        )
    mixed = _ffmpeg([vocal_wav, instrumental_wav], graph, ["-c:a", "pcm_s16le"], ".wav")
    return _apply(mixed, watermark)


def to_mp3(
    audio_wav: bytes,
    gain_db: float = 0.0,
    watermark: Callable[[bytes], bytes] | None = None,
    clarity: float | None = enhance.DEFAULT_CLARITY,
) -> bytes:
    """The output step for every branch with nothing to mix into.

    `speech`, `tts` and `vocal` all end here: no instrumental, same level, same
    tagging, and the same clarity chain the song path applies to its vocal.
    """
    gain = clamp_gain_db(gain_db)
    graph = f"[0:a]{enhance.chain(clarity, ',')}volume={gain:.2f}dB,{LOUDNORM}[out]"
    levelled = _ffmpeg([audio_wav], graph, ["-c:a", "pcm_s16le"], ".wav")
    return _apply(levelled, watermark)


def sum_stems(stems: list[bytes]) -> bytes:
    """Add N stems back into one wav, for models that split further than 2 ways.

    HTDemucs returns drums/bass/other where we want a single instrumental.
    `normalize=0` again: these stems sum back to the original mix by
    construction, and scaling them down would make the backing track quiet.
    """
    if not stems:
        raise MixError("nothing to sum")
    if len(stems) == 1:
        return stems[0]
    labels = "".join(f"[{i}:a]" for i in range(len(stems)))
    graph = f"{labels}amix=inputs={len(stems)}:normalize=0:duration=longest[out]"
    return _ffmpeg(stems, graph, ["-c:a", "pcm_s16le"], ".wav")
