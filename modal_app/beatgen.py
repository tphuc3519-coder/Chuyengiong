"""Making a beat that did not exist before.

    "boom bap, 90 BPM, dusty piano loop" ──► BeatGenerator ──► beat.wav
                                                                  │
                                          beats.fit(bài) ─────────┘

The other half of `beats.py`: that module can put *a* beat under a voice, and
this one is where a beat comes from when nobody has one to upload.

**Why this exists at all is a copyright question, so it is worth being exact.**
Editing an existing beat — retuning it, restretching it, filtering it — makes a
derivative work, which is still the original owner's; and the fingerprinting
systems that look for it are built to survive precisely those edits, so it does
not even work on its own terms. Music that was never copied from anything has
neither problem. That is the whole argument for generating, and it is why the
model choice below is not a matter of taste.

**The model choice is part of the argument.** MusicGen is the obvious candidate
and its weights are CC-BY-NC: using it to make music you can release is not
allowed by the licence of the thing making it, which defeats the purpose in a
way that is easy to miss. Stable Audio Open is licensed for commercial use
under Stability's community terms, and — the part that matters here — it was
trained on Freesound and the Free Music Archive, which is to say on audio that
was licensed for it. The provenance of the model is part of the provenance of
what it makes.

**And one limit that no amount of prompting fixes.** A generated loop can be in
the right key and at the right tempo; it cannot know the *chord progression* of
the song it is going under. Where a vocal moves through changes, a loop that
sits on one harmony will clash at the bars where they diverge. That is why this
works on rap, hip-hop and most electronic music — where the bed is a loop and
the vocal is rhythmic — and gets progressively worse the more melodic the
vocal is. It is a property of putting a loop under a melody, not a bug to be
tuned out.

**Accuracy of the prompt does not matter much**, which is the nice consequence
of having built `beats.py` first. Ask for 90 BPM and get 94; the beat is
measured and fitted afterwards, so the prompt only has to get the *character*
right and the arithmetic is somebody else's job.

Smoke test (needs Modal credentials, a GPU, and `HF_TOKEN`):

    modal run -m modal_app.beatgen --prompt "boom bap hip hop beat, 90 BPM"
"""

# NB: no `from __future__ import annotations` — modal.parameter() reads the raw
# class annotation and cannot resolve a stringified one.
import modal

from .app import MODEL_DIR, app, base_image, config_secret, model_vol

# Gated on Hugging Face: somebody has to accept Stability's terms with the
# account whose token this runs under. That is a deliberate part of the licence
# and not an obstacle to route around — see `load` for what happens without it.
MODEL_REPO = "stabilityai/stable-audio-open-1.0"

# `stable-audio-tools` is the reference implementation and MIT licensed; it is
# the weights that carry Stability's own terms. Pinned because its inference
# entry point has moved between releases, exactly like seed-vc's.
#
# **This list once had `einops==0.8.0` in it and that one line failed every
# deploy for three phases.** `stable-audio-tools` depends on `einops==0.7.0`
# exactly, so pip had nothing to resolve:
#
#     ERROR: Cannot install einops==0.8.0 and stable-audio-tools==0.0.16
#     The conflict is caused by:
#         stable-audio-tools 0.0.16 depends on einops==0.7.0
#
# `modal deploy` builds every registered image in one pass, so that took the
# whole deployment down with it rather than just this feature — which is why
# `enabled()` exists below and why it stays.
#
# What is pinned here and why:
#
#  * torch and torchaudio are repeated at `base_image`'s versions so the
#    resolver cannot pull a different build in behind them. Left alone it takes
#    torchaudio 2.11 against torch 2.4, which installs and does not run.
#  * transformers is held at 4.46.3 for the reason `tts.py` documents at
#    length: 5.x wants torch >= 2.5 and, finding 2.4, prints one line to the
#    build log and carries on with every model class replaced by a stub. Stable
#    Audio Open conditions on T5, so that stub is the whole feature.
#  * einops is *not* pinned, because the package pins it and the package wins.
BEATGEN_REQUIREMENTS = [
    "stable-audio-tools==0.0.16",
    "torch==2.4.0",
    "torchaudio==2.4.0",
    "transformers==4.46.3",
]

# Installed *after* the list above, in its own layer, and that is not tidiness.
#
# `descript-audiotools` — which `stable-audio-tools` pulls in — caps protobuf
# below 3.20 for a tensorboard logger nothing here touches. Modal's own agent
# runs inside this image and reads an attribute protobuf grew in 3.20, so under
# that cap every container dies before a line of this module runs. Inside the
# same `pip_install` the resolver would have to satisfy the cap; in a later
# layer it lands on top of it instead. pip prints a conflict warning, which is
# the intended outcome — `conversion.py` does exactly this for seed-vc and
# carries the longer version of the story.
PROTOBUF_SPEC = "protobuf>=3.20,<7"

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


# What the model can produce in one pass. Stable Audio Open generates a fixed
# window — about 47 seconds at 44.1 kHz — and asking for less does not make it
# faster, it makes it pad. So the whole window is always generated and the
# usable part is trimmed out of it.
MAX_SECONDS = 47.0
DEFAULT_SECONDS = 30.0
MIN_SECONDS = 8.0

# Sampler settings. 100 steps is where Stability's own example sits and where
# the quality stops moving; `cfg_scale` 7 is the same.
DEFAULT_STEPS = 100
MIN_STEPS = 20
MAX_STEPS = 250
CFG_SCALE = 7.0
SIGMA_MIN = 0.3
SIGMA_MAX = 500.0
SAMPLER = "dpmpp-3m-sde"

# A prompt is a description of music, not an essay.
MAX_PROMPT_CHARS = 300
# Appended to whatever the user wrote, because the thing being made is a bed
# for somebody else's voice: a generated vocal underneath a converted one is
# the single worst failure this branch has available to it.
PROMPT_SUFFIX = "instrumental, no vocals, looping"


def beatgen_image():
    """The image, built only when something asks for it.

    A function and not a module-level object, so that a deployment with the
    generator switched off never even constructs it. Modal builds the images
    its registered objects reference and an orphan one should be ignored — but
    "should be ignored" is a claim about somebody else's internals, and the
    thing that went wrong here was a deploy failing on an image nobody wanted.
    Not creating it is a guarantee; not attaching it is an expectation.
    """
    return (
        base_image.pip_install(*BEATGEN_REQUIREMENTS)
        .pip_install(PROTOBUF_SPEC)
        .env({"HF_HOME": MODEL_DIR})
    )


class BeatGenError(RuntimeError):
    """No prompt, or no access to the weights."""


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


def clamp_seconds(seconds: float | None) -> float:
    try:
        value = DEFAULT_SECONDS if seconds is None else float(seconds)
    except (TypeError, ValueError):
        return DEFAULT_SECONDS
    return max(MIN_SECONDS, min(MAX_SECONDS, value))


def clamp_steps(steps: int | None) -> int:
    try:
        value = DEFAULT_STEPS if not steps else int(steps)
    except (TypeError, ValueError):
        return DEFAULT_STEPS
    return max(MIN_STEPS, min(MAX_STEPS, value))


class BeatGenerator:
    """Stable Audio Open, loaded once per container.

    Not decorated at module scope, and that is the whole point of `register()`:
    `@app.cls(image=...)` attaches the image to the App as soon as the module is
    imported, and `modal deploy` then builds it whether or not this deployment
    wants the feature. A deployment that has not switched the generator on
    should not be building — or failing on — an image it will never run.
    """

    @modal.enter()
    def load(self) -> None:
        import os

        import torch
        from stable_audio_tools import get_pretrained_model

        os.makedirs(MODEL_DIR, exist_ok=True)
        # Gated weights: `huggingface_hub` reads HF_TOKEN from the environment,
        # which `config_secret` forwards from the deploy. Said plainly here
        # because the failure without it is a 401 from inside a library, on a
        # GPU, minutes into somebody's job.
        if not os.environ.get("HF_TOKEN"):
            raise BeatGenError(
                f"{MODEL_REPO} is gated: accept its licence on Hugging Face and set "
                "HF_TOKEN on the deployment before using generated beats"
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.config = get_pretrained_model(MODEL_REPO)
        self.model = self.model.to(self.device)
        self.sample_rate = int(self.config["sample_rate"])
        self.sample_size = int(self.config["sample_size"])
        model_vol.commit()
        print(
            f"[BeatGenerator] {MODEL_REPO} sr={self.sample_rate} "
            f"window={self.sample_size / self.sample_rate:.1f}s device={self.device}"
        )

    @modal.method()
    def generate(
        self,
        prompt: str,
        seconds: float = DEFAULT_SECONDS,
        steps: int = DEFAULT_STEPS,
        seed: int = -1,
    ) -> bytes:
        """One instrumental, as 16-bit PCM wav at the model's own rate.

        Mono on the way out, because that is what everything downstream of it
        takes: `analysis` measures mono, `beats` fits mono, and `mixing` puts
        the vocal in the centre of a stereo pair anyway. A stereo bed is worth
        having and is a change to the mix stage, not to this one.

        `seed` of -1 is a fresh beat every time, which is the right default for
        something you are auditioning; a fixed seed is how you get the same one
        back after changing nothing else.
        """
        import time

        import numpy as np
        import torch
        from stable_audio_tools.inference.generation import generate_diffusion_cond

        from .audio_utils import encode_wav

        text = clean_prompt(prompt)
        length = clamp_seconds(seconds)
        total_steps = clamp_steps(steps)
        if seed is not None and seed >= 0:
            torch.manual_seed(int(seed))

        started = time.time()
        # `seconds_total` tells the model how long the thing it is describing
        # is, which shapes what it writes — a 30 second loop and a 47 second
        # one are different pieces of music, not the same one cut short.
        conditioning = [{"prompt": text, "seconds_start": 0, "seconds_total": length}]
        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model,
                steps=total_steps,
                cfg_scale=CFG_SCALE,
                conditioning=conditioning,
                sample_size=self.sample_size,
                sigma_min=SIGMA_MIN,
                sigma_max=SIGMA_MAX,
                sampler_type=SAMPLER,
                device=self.device,
            )

        # (batch, channels, samples) -> mono float32, and only the seconds that
        # were asked for: the window is fixed and the tail of it is padding.
        audio = output.to(torch.float32).cpu().numpy()
        audio = audio.reshape(-1, audio.shape[-1]).mean(axis=0)
        audio = audio[: int(length * self.sample_rate)]
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak * 0.95

        print(
            f"[BeatGenerator] {length:.0f}s in {time.time() - started:.1f}s "
            f"({total_steps} steps): {text!r}"
        )
        return encode_wav(np.asarray(audio, dtype=np.float32), self.sample_rate)


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
            timeout=900,
            max_containers=2,
        )(BeatGenerator)
    return _REGISTERED


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

    What to listen for is not whether it is at the BPM you asked for — it will
    not be, and `beats.py` fixes that — but whether it is the *kind* of music
    you asked for, and whether anybody is singing on it.
    """
    from pathlib import Path

    Path(output).write_bytes(
        BeatGenerator().generate.remote(prompt=prompt, seconds=seconds, steps=steps, seed=seed)
    )
    print(f"wrote {output}")
