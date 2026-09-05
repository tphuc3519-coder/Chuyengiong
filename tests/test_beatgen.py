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


def test_the_prompt_limit_matches_the_one_the_pipeline_enforces():
    """`pipeline` runs on the API image and cannot import this module, so the
    number exists twice. This is the test that keeps the copies equal."""
    from modal_app import pipeline

    assert pipeline.BEAT_PROMPT_CHARS == beatgen.MAX_PROMPT_CHARS
