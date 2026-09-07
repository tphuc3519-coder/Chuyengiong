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

import io
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


def mix_bed(profile: Mixdown, rate: int, stereo: str = "") -> str:
    """The bed's own filters, before the voice is anywhere near it.

    Three, and every one of them is omitted when its setting is zero — the same
    rule `enhance.chain` follows, and for the same reason: a style that asks
    for nothing should produce the graph this module had before styles existed,
    so "turn it off" is always reachable and always exactly the old behaviour.

    `aresample` is not one of the three and is never omitted.
    `sidechaincompress` compares two streams sample for sample and refuses a
    pair that disagree about the rate, and the bed and the voice arrive from
    different places at different rates as a matter of course.
    """
    profile = profile.clamped()
    parts = [f"aresample={rate}"]
    if stereo:
        parts.insert(0, stereo)
    if profile.low_shelf_db:
        parts.append(f"bass=g={profile.low_shelf_db:.2f}:f={SHELF_HZ:.0f}:t=q:w={SHELF_WIDTH}")
    if profile.pocket_db:
        parts.append(
            f"equalizer=f={profile.pocket_hz:.0f}:t=q:w={POCKET_WIDTH}:g={profile.pocket_db:.2f}"
        )
    if profile.bed_gain_db:
        parts.append(f"volume={profile.bed_gain_db:.2f}dB")
    return ",".join(parts)


def ducks(profile: Mixdown | None) -> bool:
    """Whether this profile wants the voice to push the bed down at all.

    Asked in two places — here, before the sidechain is built, and in `mix`,
    before the key is split off — and it has to give the same answer in both.
    Hence a function rather than the comparison written twice.
    """
    return profile is not None and profile.clamped().duck_db > 0


def bed_graph(profile: Mixdown, rate: int, vocal_peak_db: float, stereo: str = "") -> str:
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
    bed = mix_bed(profile, rate, stereo)
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

    **`bed` is what makes a replacement backing track sound like one**, and
    when it is None this function is exactly what it was before styles existed
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
    voice = f"{enhance.chain(clarity, ',')}volume={gain:.2f}dB,{CENTRE}"
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
            f"{bed_graph(bed, rate, _peak_db(vocal_wav), _stereo(instrumental_wav))};"
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
