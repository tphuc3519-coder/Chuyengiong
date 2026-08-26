"""Wiring checks for the Seed-VC container definition.

The conversion itself needs a GPU, so what is testable here is that the module
imports without the audio stack and that the choices the plan calls risky —
the pinned commit, the checkpoint per mode, the defaults — are the ones we
actually ship.
"""

import re

from modal_app import audio_utils as au
from modal_app import conversion


def test_seed_vc_is_pinned_to_a_commit():
    assert re.fullmatch(r"[0-9a-f]{40}", conversion.SEED_VC_COMMIT)


def test_both_modes_have_a_checkpoint_choice():
    assert set(conversion.F0_CONDITION) == set(au.MODES)
    # Singing needs F0 conditioning to keep the melody; speech does not.
    assert conversion.F0_CONDITION["singing"] is True
    assert conversion.F0_CONDITION["speech"] is False


def test_requirements_drop_the_nightly_torch_index():
    joined = " ".join(conversion.SEED_VC_REQUIREMENTS)
    assert "--index-url" not in joined
    assert "nightly" not in joined
    assert "torch==2.4.0" in conversion.SEED_VC_REQUIREMENTS


def test_requirements_drop_gui_and_eval_only_packages():
    names = {req.split("==")[0].split(">=")[0] for req in conversion.SEED_VC_REQUIREMENTS}
    assert names.isdisjoint(
        {"gradio", "FreeSimpleGUI", "sounddevice", "jiwer", "modelscope", "funasr"}
    )
    # ...but keep what the DiT / vocoder / content encoder actually import.
    assert {"munch", "einops", "descript-audio-codec", "transformers"} <= names


def test_defaults_match_the_plan_table():
    assert au.DEFAULT_DIFFUSION_STEPS == {"speech": 25, "singing": 50}
    assert au.MAX_SEMITONE_SHIFT == {"speech": 8, "singing": 12}


def test_reference_cap_leaves_room_for_source_audio():
    """Source and reference share one 30s window — a 30s reference leaves none."""
    assert au.REFERENCE_MAX_SEC <= conversion.CONTEXT_WINDOW_SEC - 10
    assert au.REFERENCE_MIN_SEC < au.REFERENCE_MAX_SEC
