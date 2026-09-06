"""The parts of beat generation that do not need a GPU.

Which is the prompt and the clamps. Everything else is ACE-Step behind a 7 GB
download, so what is worth covering here is the handful of decisions made
before the weights are touched — and one of them is load-bearing: an empty
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


def test_the_bed_can_be_as_long_as_a_song():
    """The number the model swap was *for*.

    Stable Audio Open produced a fixed 47 second window, so a three minute song
    got one 30 second loop repeated four times — no intro, no chorus that
    lifts, and a seam every thirty seconds. Anything at or below the old
    ceiling here means the swap bought nothing.
    """
    assert beatgen.MAX_SECONDS >= 240.0
    assert beatgen.clamp_seconds(200.0) == 200.0


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

    `einops` was pinned there to a version the package forbade. The package
    pins its own dependencies; this list may pin the wheels that have to agree
    with each other, and nothing else. The lesson survived the model swap even
    though the package did not."""
    allowed = {"ace-step", "torch", "torchaudio", "torchvision"}
    for requirement in beatgen.BEATGEN_REQUIREMENTS:
        name = requirement.split("@")[0].split("==")[0].split(">")[0].split("<")[0].strip()
        assert name in allowed, f"{name} is not ours to pin"


def test_the_three_torch_wheels_are_pinned_together():
    """Left alone the resolver takes `torchvision` latest, which requires
    `torch==2.14.0` — and then the torch pin beside it loses. ACE-Step asks for
    all three without versions, so the agreement has to be made here."""
    pins = {
        r.split("==")[0]: r.split("==")[1]
        for r in beatgen.BEATGEN_REQUIREMENTS
        if "==" in r and "@" not in r
    }
    assert pins == {"torch": "2.4.0", "torchaudio": "2.4.0", "torchvision": "0.19.0"}


def test_the_model_is_pinned_to_a_commit_rather_than_a_branch():
    """No releases on that repository, so a tag is not available to pin to and
    the default branch is not a version."""
    spec = next(r for r in beatgen.BEATGEN_REQUIREMENTS if r.startswith("ace-step"))
    assert "@" in spec.split("git+")[1]
    commit = spec.rsplit("@", 1)[1]
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)


def test_the_image_does_not_inherit_the_numpy_pin_that_makes_it_unresolvable():
    """`base_image` pins `numpy<2` for the librosa stack the conversion
    containers run. ACE-Step pulls `spacy`, `spacy` pulls `thinc`, and `thinc`
    requires `numpy>=2.0.0`:

        ERROR: Cannot install ace-step, ace-step==0.2.0 and numpy<2 because
        these package versions have conflicting dependencies.

    So this image has its own root. Read as source because the failure is a
    build failure — there is nothing to assert against at import time."""
    import inspect

    # The body, past the docstring and without its comments — both name
    # `base_image` to explain why it is not the root.
    body = "\n".join(
        line
        for line in inspect.getsource(beatgen.beatgen_image).split('"""')[2].splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "base_image" not in body
    assert "debian_slim" in body
    assert "add_local_python_source" in body


def test_the_prompt_limit_matches_the_one_the_pipeline_enforces():
    """`pipeline` runs on the API image and cannot import this module, so the
    number exists twice. This is the test that keeps the copies equal."""
    from modal_app import pipeline

    assert pipeline.BEAT_PROMPT_CHARS == beatgen.MAX_PROMPT_CHARS


# --- init_audio: what turns a described beat into a derived one -------------


def test_the_init_strength_is_clamped_away_from_both_ends():
    """Neither end is allowed, and the reasons are different.

    `ref_audio_strength` runs the other way from the `init_noise_level` this
    module used to pass — **higher is closer to the reference** — which is
    exactly the kind of detail that silently inverts a feature when a model is
    swapped. At 0 the reference is ignored and the derive path stops deriving;
    at 1 the model hands back what it was given, which on the `original` init
    means returning the master recording as the new beat."""
    assert beatgen.clamp_init_strength(0) == beatgen.INIT_STRENGTH_MIN
    assert beatgen.clamp_init_strength(-5) == beatgen.INIT_STRENGTH_MIN
    assert beatgen.clamp_init_strength(1.0) == beatgen.INIT_STRENGTH_MAX
    assert beatgen.clamp_init_strength(9) == beatgen.INIT_STRENGTH_MAX
    assert beatgen.clamp_init_strength("close") == beatgen.SKETCH_STRENGTH
    assert beatgen.clamp_init_strength(None) == beatgen.SKETCH_STRENGTH
    assert beatgen.clamp_init_strength(None, beatgen.ORIGINAL_STRENGTH) == (
        beatgen.ORIGINAL_STRENGTH
    )
    assert beatgen.clamp_init_strength(0.4) == 0.4


def test_the_original_init_is_followed_more_closely_than_the_sketch():
    """A sketch is four oscillators: almost none of it should survive, only the
    harmony and where the bar is. The original instrumental is already a real
    arrangement, and re-writing it that hard throws away the thing it was
    passed in for.

    Higher is closer here, so this reads the opposite way round from the old
    noise levels — and that inversion is the assertion."""
    assert beatgen.ORIGINAL_STRENGTH > beatgen.SKETCH_STRENGTH
    for level in (beatgen.SKETCH_STRENGTH, beatgen.ORIGINAL_STRENGTH):
        assert beatgen.INIT_STRENGTH_MIN <= level <= beatgen.INIT_STRENGTH_MAX


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
    assert signature.parameters["init_strength"].default is None


# --- the two things the library does not do where it looks like it does ----


def test_the_weights_are_fetched_in_startup_and_committed_after():
    """`ACEStepPipeline.__init__` does **not** download the checkpoints.

    `__call__` does, lazily:

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            self.load_checkpoint(self.checkpoint_dir)

    Left alone that is two bugs at once. The ~7 GB `snapshot_download` happens
    inside somebody's job instead of in container startup, and `model_vol
    .commit()` runs *before* anything has been written — so the Volume never
    keeps a copy and every cold container downloads it again.

    Read as source order because there is no library here to run: the assertion
    is that the fetch is called, and that the commit comes after it.
    """
    import inspect

    # `@modal.enter()` wraps it, same as `@modal.method()` does `generate`.
    body = inspect.getsource(beatgen.BeatGenerator.load._get_raw_f())
    assert "load_checkpoint" in body, "the fetch is left to the first job"
    assert body.index("load_checkpoint(") < body.index("model_vol.commit()"), (
        "the Volume is committed before the weights land in it"
    )


def test_the_output_file_is_the_one_the_call_named():
    """`save_path` is overloaded and guessing at it is how a job dies at the end.

    A directory gets a timestamped name built inside it; anything else is
    treated as the file itself with the format appended. The call already
    answers the question — `return output_paths + [input_params_json]` — so the
    path comes from the result rather than from a filename we invented.
    """
    import inspect

    body = inspect.getsource(beatgen.BeatGenerator.generate._get_raw_f())
    assert "result = self.pipeline(" in body
    assert "beat.wav" not in body, "an output filename is being guessed at"
    assert "save_path=tmp" in body, "save_path must be the directory, not a file"
