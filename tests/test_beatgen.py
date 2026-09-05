"""The parts of beat generation that do not need a GPU.

Which is the prompt and the clamps. Everything else is Stable Audio Open behind
a gated download, so what is worth covering here is the handful of decisions
made before the weights are touched — and one of them is load-bearing: an empty
prompt to this model does not produce nothing, it produces something arbitrary,
which reaches the user as a job that succeeded and handed back music nobody
asked for.
"""

import pytest

from modal_app import beatgen


def test_a_prompt_is_kept_and_the_instrumental_instruction_appended():
    """The appended part is not decoration: a generated vocal underneath a
    converted one is the worst thing this branch can produce."""
    prompt = beatgen.clean_prompt("boom bap, 90 BPM, dusty piano")
    assert prompt.startswith("boom bap, 90 BPM, dusty piano")
    assert "no vocals" in prompt


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_empty_prompt_is_refused_rather_than_sent(empty):
    with pytest.raises(beatgen.BeatGenError):
        beatgen.clean_prompt(empty)


def test_an_over_long_prompt_is_cut_rather_than_refused():
    prompt = beatgen.clean_prompt("x" * 5000)
    assert len(prompt) <= beatgen.MAX_PROMPT_CHARS + len(beatgen.PROMPT_SUFFIX) + 2


def test_the_length_is_clamped_to_what_the_model_can_make():
    assert beatgen.clamp_seconds(None) == beatgen.DEFAULT_SECONDS
    assert beatgen.clamp_seconds(0.5) == beatgen.MIN_SECONDS
    assert beatgen.clamp_seconds(600) == beatgen.MAX_SECONDS
    assert beatgen.clamp_seconds("long") == beatgen.DEFAULT_SECONDS


def test_the_window_the_model_generates_is_not_exceeded():
    """Stable Audio Open produces a fixed window and asking for more than it
    returns padding, not music."""
    assert beatgen.MAX_SECONDS <= 47.0


def test_steps_are_clamped_to_a_useful_range():
    assert beatgen.clamp_steps(None) == beatgen.DEFAULT_STEPS
    assert beatgen.clamp_steps(0) == beatgen.DEFAULT_STEPS
    assert beatgen.clamp_steps(1) == beatgen.MIN_STEPS
    assert beatgen.clamp_steps(10_000) == beatgen.MAX_STEPS


def test_nothing_in_the_requirements_fights_the_package_itself():
    """The one line that failed every deploy for three phases:

        ERROR: Cannot install einops==0.8.0 and stable-audio-tools==0.0.16
        The conflict is caused by:
            stable-audio-tools 0.0.16 depends on einops==0.7.0

    `einops` was pinned here to a version the package forbids. The package pins
    its own dependencies; this list may pin the things `base_image` already
    holds, and nothing else."""
    allowed = {"stable-audio-tools", "torch", "torchaudio", "transformers"}
    for requirement in beatgen.BEATGEN_REQUIREMENTS:
        name = requirement.split("==")[0].split(">")[0].split("<")[0].strip()
        assert name in allowed, f"{name} is not ours to pin"


def test_protobuf_is_forced_after_the_package_and_not_beside_it():
    """`descript-audiotools` caps protobuf below 3.20 for a logger nothing here
    touches, and Modal's own agent needs 3.20. In the same `pip_install` the
    resolver has to satisfy the cap; in a later layer it lands on top of it.
    `conversion.py` carries the long version of this story."""
    assert beatgen.PROTOBUF_SPEC not in beatgen.BEATGEN_REQUIREMENTS
    assert beatgen.PROTOBUF_SPEC.startswith("protobuf>=3.20")


def test_the_prompt_limit_matches_the_one_the_pipeline_enforces():
    """`pipeline` runs on the API image and validates the prompt without the
    generator's constants in front of it, so the number exists twice. This is
    the test that keeps the copies equal."""
    from modal_app import pipeline

    assert pipeline.BEAT_PROMPT_CHARS == beatgen.MAX_PROMPT_CHARS


def test_the_class_in_this_module_has_no_remote_on_it():
    """The failure this is here about, in the shape the user saw it:

        AttributeError: 'function' object has no attribute 'remote'

    `BeatGenerator` is undecorated at module scope on purpose (see `register`),
    so `@modal.method()` on it stays an ordinary function until something wraps
    the class. A caller that imports the class and calls `.generate.remote()`
    gets that `AttributeError` — not at import, not in a test, but on a GPU-less
    container minutes into somebody's job, where `pipeline` turns it into the
    sentence explaining why their song failed."""
    with pytest.raises(AttributeError):
        beatgen.BeatGenerator().generate.remote  # noqa: B018


def test_the_pipeline_asks_the_deployment_for_the_generator():
    """So the only handle anything outside the deploy uses is this one, which
    is a real Modal class and does have `.remote` under it."""
    import modal

    assert isinstance(beatgen.deployed(), modal.Cls)
