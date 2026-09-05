"""The clarity chain: what happens to the converted voice before it is encoded.

    converted.wav ──► enhance.chain() ──► mix / loudnorm ──► output.mp3

One ffmpeg filter graph, applied to the **voice only** — in `mixing.mix` it
sits on the vocal input, ahead of `amix`, so a song's backing track is never
touched by it. Everything here is a fixed, mild correction of something Seed-VC
reliably leaves behind, scaled by a single number the user controls.

Why any of it, when the model is supposed to produce a voice: a diffusion
decoder and a neural vocoder are very good at the middle of the spectrum and
much less careful at both ends of it. What comes back has

* **rumble and DC drift under 70 Hz** — nothing a voice produced, but it eats
  headroom, and `loudnorm` at the end of the pipeline turns the whole file down
  to make room for it;
* **a broadband haze** at maybe -50 dBFS: the vocoder's own noise floor, which
  is not the same thing as the source's and does not go away by cleaning the
  input. It is what makes a converted voice sound like it is behind something;
* **a build-up around 250–350 Hz**, the region every "muddy" complaint is
  about, and a corresponding dip in the 3–4 kHz band that carries consonants —
  which together are most of what "unclear" means when somebody says a voice is
  not `trong`;
* **sibilance the vocoder has exaggerated**, because /s/ and /ʃ/ are noise and
  the model reconstructs noise by generating it.

None of these is subtle enough to need a model to fix and none is severe enough
to need more than a couple of dB. What matters much more than the numbers is
that **the whole chain scales to nothing.** At `clarity=0` this function
returns no filters at all, so the output is bit-for-bit what the pipeline
produced before this module existed, and any complaint about it has a
one-slider answer.

Plain ffmpeg, like `mixing`: no numpy, no torch, and the filters are all in the
build every ffmpeg since 4.3 ships. It is a string builder, which is why CI can
test the whole of it for real.
"""

from __future__ import annotations

CLARITY_MIN = 0.0
CLARITY_MAX = 1.0
# Half of the available correction. The chain is deliberately built so that its
# full strength is more than most material wants — a slider whose top end is
# the right answer has nowhere to go when it is not.
DEFAULT_CLARITY = 0.5

# Everything below is the amount applied at `clarity = 1`, scaled linearly.

# Broadband noise reduction, in dB. Gentle on purpose: `afftdn` above ~8 dB
# starts to audibly pump on breaths and reverb tails, and what is being removed
# here is a vocoder floor, not a bad recording.
DENOISE_DB = 7.0
# Track the floor as it moves rather than measuring it once. A converted vocal
# is not stationary — the model's noise follows the level of what it is
# generating — so a fixed estimate is wrong for most of the file.
DENOISE_TRACK = True
# Where the voice stops and the desk starts.
HIGHPASS_HZ = 70.0
# The mud cut and the presence lift, as (frequency, Q, dB).
MUD = (300.0, 1.1, -2.5)
PRESENCE = (3400.0, 1.2, 2.5)
# Air, as a high shelf. Smaller than the presence lift: this is the band a
# 22.05 kHz checkpoint has the least of to give, and asking a shelf for what
# was never generated only raises the noise that is there instead.
AIR_HZ = 8000.0
AIR_DB = 1.5
# De-esser intensity, 0 to 1 in ffmpeg's own units. Applied *after* the
# presence lift, which is what made the sibilance worth attending to.
DEESS = 0.35


class EnhanceError(ValueError):
    """A clarity value that is not a number at all."""


def clamp_clarity(value: float | None) -> float:
    """How much of the chain to apply. Out of range is clamped, never refused.

    Same rule as every other slider in this app: a value that arrives wrong is
    a client bug, and failing a job that has already paid for a GPU over one is
    not a trade worth making.
    """
    try:
        amount = DEFAULT_CLARITY if value is None else float(value)
    except (TypeError, ValueError):
        return DEFAULT_CLARITY
    return max(CLARITY_MIN, min(CLARITY_MAX, amount))


def _eq(band: tuple[float, float, float], amount: float) -> str:
    frequency, q, gain = band
    return f"equalizer=f={frequency:.0f}:t=q:w={q:.2f}:g={gain * amount:.2f}"


def filters(clarity: float | None = DEFAULT_CLARITY) -> list[str]:
    """The chain as a list of ffmpeg filters, in the order they must run.

    Empty at zero, which is the whole contract: no filters means `chain` emits
    nothing and the caller's graph is exactly what it was.

    The order is not arrangeable. Denoising first, because every filter after
    it has a gain and would otherwise amplify the floor it is about to remove.
    The high pass next, so the equalisers are not spending their range on
    frequencies that are on their way out. Then the two peaking filters. The
    de-esser last of all, because the presence lift is what it is correcting
    for — de-essing before the lift measures sibilance that has not happened
    yet.
    """
    amount = clamp_clarity(clarity)
    if amount <= 0:
        return []
    # Float samples throughout: the peaking filters have gain, and a graph that
    # negotiated 16-bit in the middle would clip on the loudest syllable before
    # `loudnorm` ever got the chance to set the level.
    chain = ["aformat=sample_fmts=fltp"]
    chain.append(f"afftdn=nr={DENOISE_DB * amount:.2f}:nf=-45{':tn=1' if DENOISE_TRACK else ''}")
    chain.append(f"highpass=f={HIGHPASS_HZ:.0f}:poles=2")
    chain.append(_eq(MUD, amount))
    chain.append(_eq(PRESENCE, amount))
    chain.append(f"treble=g={AIR_DB * amount:.2f}:f={AIR_HZ:.0f}")
    chain.append(f"deesser=i={DEESS * amount:.2f}:m=0.5:f=0.5")
    return chain


def chain(clarity: float | None = DEFAULT_CLARITY, suffix: str = "") -> str:
    """The filters as one comma-separated ffmpeg fragment, or `""`.

    `suffix` is appended when there is anything to append to, so a caller can
    write `f"[0:a]{enhance.chain(c, ',')}volume=…"` and get a valid graph both
    ways round without testing for the empty case itself. That is the entire
    reason it exists: a filter graph with a stray comma is an ffmpeg error at
    run time, in a container, on somebody's job.
    """
    parts = filters(clarity)
    return ",".join(parts) + suffix if parts else ""


def describe(clarity: float | None = DEFAULT_CLARITY) -> str:
    """One line for the container log: what was applied, at what strength."""
    amount = clamp_clarity(clarity)
    if amount <= 0:
        return "clarity off"
    return (
        f"clarity {amount:.2f}: denoise {DENOISE_DB * amount:.1f}dB, "
        f"mud {MUD[2] * amount:+.1f}dB, presence {PRESENCE[2] * amount:+.1f}dB, "
        f"air {AIR_DB * amount:+.1f}dB, de-ess {DEESS * amount:.2f}"
    )
