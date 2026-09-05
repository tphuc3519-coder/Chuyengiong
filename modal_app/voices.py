"""Where a trained voice lives, and what counts as one.

    /models/voices/<name>/<mode>/ft_model.pth        the fine-tuned weights
                                 config.yml          the architecture they are for

Path arithmetic and a name rule, and deliberately nothing else: `training.py`
writes this layout on a GPU and `conversion.py` reads it on another one, and
neither should have to agree with the other by remembering a convention. It
imports nothing — not modal, not torch, not numpy — so the API container can
answer "which voices exist" without any of that, and CI can test the rules.

A profile is per *mode*, not per voice. `speech` and `singing` are different
checkpoints with different sample rates and different architectures (one is F0
conditioned, one is not), so a voice trained for one is not a thing that can be
loaded into the other — it is not merely worse, it will not load. Two
directories keeps that a fact about the filesystem rather than a rule somebody
has to know.
"""

from __future__ import annotations

import os
import re

# Under `MODEL_DIR`, so a trained voice survives a container restart the same
# way downloaded weights do.
VOICES_SUBDIR = "voices"
# What `train.py` calls its final save, and the config it copies beside it.
CHECKPOINT_NAME = "ft_model.pth"
CONFIG_NAME = "config.yml"

# A profile name is a directory name and reaches this module from a form field,
# so it is checked rather than trusted: letters, digits, dash, underscore, and
# nothing that could be a path.
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,47}$")


class VoiceError(ValueError):
    """A voice name that is not usable as one, or a profile that is not there."""


def check_name(name: str) -> str:
    """The name, or a refusal. Empty is not a name — callers mean `None`.

    Refused rather than sanitised. A sanitiser turns `../../etc` into something
    that looks fine and points somewhere else, and the caller never learns that
    the voice it asked for is not the voice it got.
    """
    if not isinstance(name, str) or not NAME_PATTERN.match(name):
        raise VoiceError(
            f"voice name must be 1-48 characters of letters, digits, - or _, got {name!r}"
        )
    return name


def clean_name(name: str | None) -> str:
    """`check_name`, but an unusable name is simply "no voice profile".

    `/submit` uses this: a job that asks for a profile that cannot exist should
    run zero-shot, which is what every job did before profiles existed, rather
    than fail. `resolve` is where a *missing* profile is reported, and it says
    which one.
    """
    try:
        return check_name(name or "")
    except VoiceError:
        return ""


def root(model_dir: str) -> str:
    return os.path.join(model_dir, VOICES_SUBDIR)


def directory(model_dir: str, name: str, mode: str) -> str:
    """Where the profile for `name` in `mode` is, whether or not it exists."""
    return os.path.join(root(model_dir), check_name(name), mode)


def files(model_dir: str, name: str, mode: str) -> tuple[str, str]:
    """`(checkpoint, config)` paths for a profile. No existence check."""
    where = directory(model_dir, name, mode)
    return os.path.join(where, CHECKPOINT_NAME), os.path.join(where, CONFIG_NAME)


def exists(model_dir: str, name: str, mode: str) -> bool:
    checkpoint, config = files(model_dir, name, mode)
    return os.path.isfile(checkpoint) and os.path.isfile(config)


def resolve(model_dir: str, name: str, mode: str) -> tuple[str, str] | None:
    """The profile's files if it is complete, `None` if the name is empty.

    Raises for a name that was given and is not there, which is the one case
    worth failing over: a job that asked to be read in a trained voice and
    silently ran zero-shot instead produces a result that is *plausible* and
    wrong, and nobody would know to ask.
    """
    if not name:
        return None
    if not exists(model_dir, name, mode):
        raise VoiceError(f"no trained voice {name!r} for {mode}; train it first")
    return files(model_dir, name, mode)


def profiles(model_dir: str) -> dict[str, list[str]]:
    """Every trained voice on the volume, as `{name: [mode, …]}`.

    Reads the filesystem rather than a manifest on purpose: the directory is
    the record, so a half-written profile — a training run that died before its
    final save — is simply not listed, and nothing has to be reconciled.
    """
    found: dict[str, list[str]] = {}
    base = root(model_dir)
    if not os.path.isdir(base):
        return found
    for name in sorted(os.listdir(base)):
        try:
            check_name(name)
        except VoiceError:
            continue
        modes = sorted(
            mode for mode in os.listdir(os.path.join(base, name)) if exists(model_dir, name, mode)
        )
        if modes:
            found[name] = modes
    return found
