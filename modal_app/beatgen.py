"""Making a beat that did not exist before.

    "rock, guitar méo, trống thật" ──► BeatGenerator ──► beat.wav
                                             ▲                │
                       sketch.render() ──────┘                │
                        (vòng hợp âm bài)      beats.fit ──────┘

The other half of `beats.py`: that module can put *a* beat under a voice, and
this one is where a beat comes from when nobody has one to upload.

**Why this exists at all is a copyright question, so it is worth being exact.**
Editing an existing beat — retuning it, restretching it, filtering it — makes a
derivative work, which is still the original owner's; and the fingerprinting
systems that look for it are built to survive precisely those edits, so it does
not even work on its own terms. Music that was never copied from anything has
neither problem.

**The model here is ACE-Step, and it is the second one this module has used.**
The first was Stable Audio Open, and swapping it out was not a matter of taste
— it was two structural limits that no parameter reached:

  * *A 47 second window.* The bed was generated once at 30 seconds and then
    **looped** for the length of the song. A 90 second song was the same 30
    seconds three times: no intro, no verse that differs from a chorus, no
    lift. Held against a human arrangement of the same song, that is not a
    tuning gap, it is a missing dimension. ACE-Step generates up to four
    minutes in one pass, so the bed is the whole track and `beats.lay_under`
    has nothing left to repeat.
  * *No structure control.* ACE-Step is a song model: it writes an
    arrangement that goes somewhere, and takes `[verse]`/`[chorus]` markers
    when asked.

**And the licence got simpler on the way.** Stable Audio Open is gated on
Hugging Face under Stability's community terms, which meant an account, an
accepted licence and an `HF_TOKEN` on the deployment before the feature worked
at all. ACE-Step is **Apache 2.0, code and weights**, ungated. Nothing to
accept and no token to set — `HF_TOKEN` is still forwarded when present because
an authenticated download is rate-limited less kindly, but it is no longer a
requirement and `load` no longer refuses without one.

MusicGen remains the obvious candidate that cannot be used: its weights are
CC-BY-NC, so making music you are allowed to release with a model you are not
allowed to release from defeats the purpose in a way that is easy to miss.

**One limit no amount of prompting fixes, and the way around it.** A loop
generated from a sentence can be in the right key and at the right tempo; it
cannot know the *chord progression* of the song it is going under. Where a
vocal moves through changes, a bed sitting on its own harmony clashes at the
bars where they diverge.

`audio2audio` is the way around it, and it is the reason `generate` takes an
init. Given reference audio, the model follows that recording's harmony and bar
lines while writing its own instruments over them — `ref_audio_strength` says
how closely, and **higher means closer to the reference**, which is the
opposite direction from the `init_noise_level` this module used to pass. What
gets passed in decides what this branch *is*:

  * `sketch.render` — this repository's own oscillators playing the chart that
    `chords.detect` read off the song. The model never hears the original, so
    the output is a cover of the composition, not a derivative of the master.
    This is the path the feature is built around.
  * the separated instrumental itself — musically the closest match, and a
    derivative work of the recording, which is the thing the paragraph above
    exists to avoid. Offered, off by default, and labelled as what it is.

**Accuracy of the prompt does not matter much**, which is the nice consequence
of having built `beats.py` first. Ask for 90 BPM and get 94; the beat is
measured and fitted afterwards, so the prompt only has to get the *character*
right and the arithmetic is somebody else's job.

Smoke test (needs Modal credentials and a GPU; no token, no licence to accept):

    modal run -m modal_app.beatgen --prompt "boom bap hip hop beat, 90 BPM"
"""

# NB: no `from __future__ import annotations` — modal.parameter() reads the raw
# class annotation and cannot resolve a stringified one.
import modal

from .app import MODEL_DIR, app, config_secret, model_vol

# Apache 2.0, and **not gated** — no licence to accept, no token to hold. The
# pipeline fetches this into `CHECKPOINT_DIR` on the model Volume the first
# time a container runs, and every container after that finds it there.
MODEL_REPO = "ACE-Step/ACE-Step-v1-3.5B"
CHECKPOINT_DIR = f"{MODEL_DIR}/acestep"

# The reference implementation, pinned to a commit like seed-vc's is: there are
# no releases on this repository, so a tag is not available to pin to and the
# default branch is not a version.
ACESTEP_REPO = "https://github.com/ace-step/ACE-Step.git"
ACESTEP_COMMIT = "1bee4c9f5b43e30995f8d4d33b3919197ce1bd68"

# **This list once had `einops==0.8.0` in it and that one line failed every
# deploy for three phases**, back when this module ran Stable Audio Open:
#
#     ERROR: Cannot install einops==0.8.0 and stable-audio-tools==0.0.16
#     The conflict is caused by:
#         stable-audio-tools 0.0.16 depends on einops==0.7.0
#
# `modal deploy` builds every registered image in one pass, so that took the
# whole deployment down with it rather than just this feature — which is why
# `enabled()` exists below and why it stays. The lesson survives the model
# swap: this set was resolved with `pip install --dry-run` before it was
# written down (116 packages, exit 0), and it should be again before it moves.
#
# What is pinned here and why:
#
#  * The three torch wheels are pinned together because they have to agree.
#    Left to itself the resolver takes `torchvision` latest, which requires
#    `torch==2.14.0`, and the pin below would lose. ACE-Step asks for all three
#    without versions.
#  * transformers and diffusers are **not** pinned here: ACE-Step pins
#    transformers itself (4.50.0) and floors diffusers, and the package wins.
BEATGEN_REQUIREMENTS = [
    f"ace-step @ git+{ACESTEP_REPO}@{ACESTEP_COMMIT}",
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "torchvision==0.19.0",
]

BEAT_GENERATOR_ENV = "BEAT_GENERATOR"


def enabled() -> bool:
    """Whether this deployment ships the beat generator at all.

    Read at import time by `deploy.py`, which is what keeps the image out of a
    deploy that has not asked for it. Read again by `api.submit`, so a request
    for a generated beat is refused with a sentence rather than accepted into a
    pipeline that has no container to run it.
    """
    import os

    return os.environ.get(BEAT_GENERATOR_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# How long a bed can be. Four minutes is the model's own ceiling and it is the
# number that changed this feature: the previous model generated 47 seconds at
# most, so a three minute song got one loop repeated four times. `pipeline`
# asks for the song's own length, so nothing is repeated and nothing is cut.
MAX_SECONDS = 240.0
DEFAULT_SECONDS = 60.0
MIN_SECONDS = 8.0

# Sampler settings, at the library's own defaults. 60 steps is where ACE-Step's
# examples sit; `guidance_scale` 15 likewise.
DEFAULT_STEPS = 60
MIN_STEPS = 20
MAX_STEPS = 200
GUIDANCE_SCALE = 15.0
SCHEDULER = "euler"

# How much of `ref_audio_input` survives, and **higher means closer to it** —
# the opposite direction from the noise level this module passed to Stable
# Audio Open, which is exactly the kind of detail that silently inverts a
# feature when a model is swapped.
#
# The two init sources want opposite ends of the range:
#
#  * A **sketch** is four oscillators and a drum machine. Almost none of it
#    should survive — only the harmony, the tempo and where the bar is.
#  * The **original instrumental** is already a real arrangement, and
#    re-writing it that hard throws away the thing it was passed in for.
#
# Neither number is measured. They cannot be, from here: judging them needs a
# GPU and a pair of ears, and they are parameters for exactly that reason.
INIT_STRENGTH_MIN = 0.05
INIT_STRENGTH_MAX = 0.95
SKETCH_STRENGTH = 0.35
ORIGINAL_STRENGTH = 0.65

# A prompt is a description of music, not an essay.
MAX_PROMPT_CHARS = 300
# Appended to whatever the user wrote, because the thing being made is a bed
# for somebody else's voice: a generated vocal underneath a converted one is
# the single worst failure this branch has available to it.
PROMPT_SUFFIX = "instrumental, no vocals"
# ACE-Step's own way of being told there is nothing to sing. Belt and braces
# with `PROMPT_SUFFIX`: the tag is the documented switch, the words in the
# prompt are what the style encoder reads.
INSTRUMENTAL_LYRICS = "[inst]"


def beatgen_image():
    """The image, built only when something asks for it.

    A function and not a module-level object, so that a deployment with the
    generator switched off never even constructs it. Modal builds the images
    its registered objects reference and an orphan one should be ignored — but
    "should be ignored" is a claim about somebody else's internals, and the
    thing that went wrong here was a deploy failing on an image nobody wanted.
    Not creating it is a guarantee; not attaching it is an expectation.

    **Built from `debian_slim` rather than from `base_image`, and that is not
    tidiness.** `base_image` pins `numpy<2` for the librosa/soundfile stack the
    conversion containers run. ACE-Step pulls `spacy`, `spacy` pulls `thinc`,
    and `thinc` requires `numpy>=2.0.0` — so inheriting that pin makes the
    image unresolvable, full stop:

        ERROR: Cannot install ace-step, ace-step==0.2.0 and numpy<2 because
        these package versions have conflicting dependencies.

    Nothing this container runs needs the numpy 1 stack. `audio_utils` is the
    only module of ours it imports, and what it uses there — `wave`, ffmpeg and
    plain arrays — is numpy 2 clean.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ffmpeg", "git", "libsndfile1")
        .pip_install(*BEATGEN_REQUIREMENTS)
        .env({"HF_HOME": MODEL_DIR})
        # `base_image` bakes the source in for the same reason: Modal forbids
        # further build steps after a mounted `add_local_*`.
        .add_local_python_source("modal_app", copy=True)
    )


class BeatGenError(RuntimeError):
    """No prompt, or the model would not load."""


def clean_prompt(prompt: str) -> str:
    """The description, trimmed, with the instrumental instruction appended.

    Refused when empty, because an empty prompt to this model does not produce
    nothing — it produces something arbitrary, which is worse: a job that
    succeeded and returned music nobody asked for.
    """
    text = (prompt or "").strip()
    if not text:
        raise BeatGenError("describe the beat you want, e.g. 'boom bap, 90 BPM, piano'")
    text = text[:MAX_PROMPT_CHARS].strip()
    return f"{text}, {PROMPT_SUFFIX}"


def describe(track) -> str:
    """A prompt written from what `analysis` already measured.

    The derive path has a chart, a tempo and a key before it has a word from
    the user, so an empty description is not a reason to refuse — it is a
    reason to write the obvious one. Names the tempo and the key and stops
    there: what genre a song is is not something `analysis` measures, and a
    guess at it would be a guess printed in a box the user then has to correct.

    `beats.fit` re-measures and re-fits whatever comes back, so the numbers
    here are a nudge rather than a contract.
    """
    return f"{track.bpm:.0f} BPM, key of {track.key_name}, drums and bass"


def clamp_init_strength(strength, fallback: float = SKETCH_STRENGTH) -> float:
    """How much of the reference audio survives, kept inside a usable range.

    Neither end is allowed. At 0 the reference is ignored and the whole point
    of the derive path — following the song's own harmony — is gone; at 1 the
    model returns what it was given, which on the `original` init means handing
    back the master recording as the new beat.
    """
    try:
        value = fallback if strength is None else float(strength)
    except (TypeError, ValueError):
        return fallback
    return max(INIT_STRENGTH_MIN, min(INIT_STRENGTH_MAX, value))


def clamp_seconds(seconds) -> float:
    try:
        value = DEFAULT_SECONDS if seconds is None else float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_SECONDS
    return max(MIN_SECONDS, min(MAX_SECONDS, value))


def clamp_steps(steps) -> int:
    try:
        value = DEFAULT_STEPS if not steps else int(steps)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, value))


class BeatGenerator:
    """ACE-Step, loaded once per container.

    Not decorated at module scope, and that is the whole point of `register()`:
    `@app.cls(image=...)` attaches the image to the App as soon as the module is
    imported, and `modal deploy` then builds it whether or not this deployment
    wants the feature. A deployment that has not switched the generator on
    should not be building — or failing on — an image it will never run.
    """

    @modal.enter()
    def load(self) -> None:
        import os
        import time

        import torch
        from acestep.pipeline_ace_step import ACEStepPipeline

        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=CHECKPOINT_DIR,
            dtype="bfloat16" if self.device == "cuda" else "float32",
            torch_compile=False,
        )

        # **`__init__` does not fetch the weights.** `__call__` does, lazily:
        #
        #     if not self.loaded:
        #         logger.warning("Checkpoint not loaded, loading checkpoint...")
        #         self.load_checkpoint(self.checkpoint_dir)
        #
        # Left alone that is two bugs at once. The ~7 GB `snapshot_download`
        # would happen inside somebody's job rather than in container startup,
        # and — worse — the `model_vol.commit()` below would run *before*
        # anything had been written, so the Volume would never keep a copy and
        # **every cold container would download it again**. Calling it here
        # puts the fetch where a cold start belongs and the commit after the
        # thing it is committing.
        started = time.time()
        self.pipeline.load_checkpoint(CHECKPOINT_DIR)
        model_vol.commit()
        print(
            f"[BeatGenerator] {MODEL_REPO} ready in {time.time() - started:.1f}s "
            f"device={self.device} dir={CHECKPOINT_DIR}"
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        seconds: float = DEFAULT_SECONDS,
        steps: int = DEFAULT_STEPS,
        seed: int = -1,
        init_wav: bytes = None,
        init_strength: float = None,
    ) -> bytes:
        """One instrumental, as 16-bit PCM wav.

        Mono on the way out, because that is what everything downstream of it
        takes: `analysis` measures mono, `beats` fits mono, and `mixing` puts
        the vocal in the centre of a stereo pair anyway. A stereo bed is worth
        having and is a change to the mix stage, not to this one.

        `seconds` is the song's own length rather than a loop length, which is
        the point of this model: what comes back is one continuous
        arrangement, so `beats.lay_under` has nothing to repeat and there is no
        seam to hear every thirty seconds.

        `seed` of -1 is a fresh beat every time, which is the right default for
        something you are auditioning; a fixed seed is how you get the same one
        back after changing nothing else.

        **`init_wav` is what makes this follow a song rather than a sentence.**
        Given one, the model writes over that recording's harmony and bar lines
        instead of inventing its own. `init_strength` says how closely, and
        **higher is closer** — `pipeline` picks the default from which source
        the audio came from.
        """
        import tempfile
        import time
        from pathlib import Path

        from .audio_utils import decode_audio, encode_wav

        text = clean_prompt(prompt)
        length = clamp_seconds(seconds)
        total_steps = clamp_steps(steps)
        strength = clamp_init_strength(init_strength)

        started = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = None
            if init_wav:
                ref_path = str(Path(tmp) / "ref.wav")
                Path(ref_path).write_bytes(init_wav)

            # A **directory**, and the file name comes back rather than being
            # guessed. `save_path` is overloaded — a directory gets a
            # timestamped name built inside it, anything else is treated as the
            # file itself and has the format appended — and the call already
            # answers the question:
            #
            #     return output_paths + [input_params_json]
            #
            # Reading `output_paths[0]` is the only version of this that cannot
            # go looking for a file the library decided to call something else.
            result = self.pipeline(
                save_path=tmp,
                format="wav",
                audio_duration=length,
                prompt=text,
                # The documented switch for "nobody sings on this". The words
                # in `PROMPT_SUFFIX` say the same thing to the style encoder.
                lyrics=INSTRUMENTAL_LYRICS,
                infer_step=total_steps,
                guidance_scale=GUIDANCE_SCALE,
                scheduler_type=SCHEDULER,
                manual_seeds=[int(seed)] if seed is not None and seed >= 0 else None,
                audio2audio_enable=bool(init_wav),
                ref_audio_input=ref_path,
                ref_audio_strength=strength,
            )
            written = next(
                (
                    Path(item)
                    for item in (result or [])
                    if isinstance(item, str) and Path(item).is_file()
                ),
                None,
            )
            if written is None or not written.stat().st_size:
                raise BeatGenError("the generator produced no audio")
            audio = decode_audio(written.read_bytes(), 44100)

        # Level is `beats.balance`'s job and it runs after `stretch` has
        # transposed this; all that happens here is a guard against a file that
        # would clip on the way into it.
        import numpy as np

        peak = float(np.abs(audio).max())
        if peak > 0.95:
            audio = audio / peak * 0.95

        source = f"init {strength:.2f}" if init_wav else "from scratch"
        print(
            f"[BeatGenerator] {length:.0f}s in {time.time() - started:.1f}s "
            f"({total_steps} steps, {source}): {text!r}"
        )
        return encode_wav(np.asarray(audio, dtype=np.float32), 44100)


# The decorated class, once something has asked for it. `None` means this
# deployment does not ship the generator — and `pipeline` never reaches for it,
# because `api.submit` refused the request long before.
_REGISTERED = None


def register():
    """Attach `BeatGenerator` to the App and return it. `deploy` calls it.

    Idempotent, because `deploy` is imported by the tests as well as by the
    deploy itself, and decorating the same class twice is not something Modal
    enjoys. The class keeps its own name, so what lands in the deployment is
    `BeatGenerator` exactly as it would have been with a decorator on it.
    """
    global _REGISTERED
    if _REGISTERED is None:
        _REGISTERED = app.cls(
            image=beatgen_image(),
            gpu="A10G",
            volumes={MODEL_DIR: model_vol},
            secrets=[config_secret()],
            scaledown_window=300,
            # Four minutes of audio at 60 steps is seconds of compute on an
            # A10G; the ceiling here is the first cold container fetching ~7 GB
            # of weights onto the Volume.
            timeout=1800,
            max_containers=2,
        )(BeatGenerator)
    return _REGISTERED


def generator():
    """The `BeatGenerator` that has `.remote` on it. **Not the class above.**

    This function exists because of one crash, and it is worth spelling out
    because the shape that caused it looks completely fine:

        AttributeError: 'function' object has no attribute 'remote'

    `app.cls(...)` does not decorate `BeatGenerator` in place — it *returns a
    new object* and leaves the class alone. So the module-level name stays a
    plain Python class, `BeatGenerator().generate` is an ordinary bound method,
    and `.remote` on it is an attribute that was never there. Every call site
    that reached for the imported name got that, and the first person to run a
    generated beat got it three minutes into a job.

    Preferring `_REGISTERED` is not enough on its own either. `register()` is
    called by `deploy.py`, in the deploy process; `run_beat_pipeline` runs in a
    container that carries no `BEAT_GENERATOR` in its environment and has no
    reason to have imported `deploy` at all, so `_REGISTERED` there is `None`.
    That is why the fallback is a lookup by name against the deployed App
    rather than another attempt to register locally: the class is already
    deployed, and this is how a container refers to one it did not define.

    Fails loudly if there is nothing deployed under that name, which is the
    honest error — `api.submit` refuses these jobs when `enabled()` is false,
    so getting here at all means the deployment claimed to have one.
    """
    if _REGISTERED is not None:
        return _REGISTERED
    import modal as _modal

    from .app import APP_NAME

    return _modal.Cls.from_name(APP_NAME, BeatGenerator.__name__)


@app.local_entrypoint()
def make_beat(
    prompt: str,
    output: str = "beat.wav",
    seconds: float = DEFAULT_SECONDS,
    steps: int = DEFAULT_STEPS,
    seed: int = -1,
) -> None:
    """Standalone smoke test: a description in, one instrumental out.

        modal run -m modal_app.beatgen --prompt "boom bap hip hop beat, 90 BPM"
        modal run -m modal_app.beatgen --prompt "lo-fi house, 124 BPM" --seed 7

    Drives the *deployed* class, through `generator()` — so it needs a deploy
    with `BEAT_GENERATOR` on, and it is a smoke test of what users actually hit
    rather than of a copy built for the occasion.

    What to listen for is not whether it is at the BPM you asked for — it will
    not be, and `beats.py` fixes that — but whether it is the *kind* of music
    you asked for, whether anybody is singing on it, and whether it goes
    anywhere over its length rather than repeating.
    """
    from pathlib import Path

    Path(output).write_bytes(
        generator()().generate.remote(prompt=prompt, seconds=seconds, steps=steps, seed=seed)
    )
    print(f"wrote {output}")
