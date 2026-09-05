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
    """`pipeline` runs on the API image and cannot import this module, so the
    number exists twice. This is the test that keeps the copies equal."""
    from modal_app import pipeline

    assert pipeline.BEAT_PROMPT_CHARS == beatgen.MAX_PROMPT_CHARS


# --- init_audio: what turns a described beat into a derived one -------------


def test_the_noise_level_is_clamped_away_from_zero():
    """Zero is not the bottom of the range, and that is not an off-by-one.

    `init_noise_level` becomes the sampler's `sigma_max`, so at zero there is
    no noise to remove and the sampler returns its input unchanged — which on
    the `original` init means handing back the master recording as the new
    beat."""
    assert beatgen.clamp_noise_level(0) == beatgen.INIT_NOISE_MIN
    assert beatgen.clamp_noise_level(-5) == beatgen.INIT_NOISE_MIN
    assert beatgen.clamp_noise_level(1e9) == beatgen.INIT_NOISE_MAX
    assert beatgen.clamp_noise_level("loud") == beatgen.SKETCH_NOISE_LEVEL
    assert beatgen.clamp_noise_level(None) == beatgen.SKETCH_NOISE_LEVEL
    assert beatgen.clamp_noise_level(None, beatgen.ORIGINAL_NOISE_LEVEL) == (
        beatgen.ORIGINAL_NOISE_LEVEL
    )
    assert beatgen.clamp_noise_level(40) == 40.0


def test_the_two_init_sources_sit_at_opposite_ends_of_the_range():
    """A sketch is four oscillators: almost none of it should survive, only the
    harmony and where the bar is. The original instrumental is already a real
    arrangement, and re-rendering it that hard throws away the thing it was
    passed in for."""
    assert beatgen.SKETCH_NOISE_LEVEL > beatgen.ORIGINAL_NOISE_LEVEL
    for level in (beatgen.SKETCH_NOISE_LEVEL, beatgen.ORIGINAL_NOISE_LEVEL):
        assert beatgen.INIT_NOISE_MIN <= level <= beatgen.INIT_NOISE_MAX


def test_a_prompt_can_be_written_from_a_measurement_alone():
    """`derive` has a tempo and a key before it has a word from the user, so an
    empty description is a reason to write the obvious prompt rather than to
    refuse the job."""
    from modal_app.analysis import Track

    track = Track(
        bpm=153.6, beat_offset_sec=0.0, key=9, minor=True, key_margin=0.1, duration_sec=180.0
    )
    written = beatgen.describe(track)
    assert "154 BPM" in written
    assert "Am" in written
    # And it has to survive the same cleaning every other prompt goes through.
    assert beatgen.clean_prompt(written).endswith(beatgen.PROMPT_SUFFIX)


def test_generate_accepts_an_init_without_requiring_one():
    """The signature is the contract `pipeline` relies on: `upload` and
    `generate` pass nothing, `derive` passes two more arguments."""
    import inspect

    # `@modal.method()` wraps the function; the signature lives on the original.
    signature = inspect.signature(beatgen.BeatGenerator.generate._get_raw_f())
    assert signature.parameters["init_wav"].default is None
    assert signature.parameters["init_noise_level"].default is None
