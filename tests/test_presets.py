"""The preset voice manifest, checked against plan §8 item 4.

Presets are the one place this app would ship a voice rather than receive one,
which makes them the one place a provenance rule can be enforced by CI instead
of by remembering. So: every clip listed has to say where it came from, and no
audio may sit in the directory without being listed.

The manifest ships empty, and an empty manifest passes — the UI hides the
preset row entirely, which is a supported state rather than a broken one.
"""

import json
from pathlib import Path

import pytest

PRESET_DIR = Path(__file__).resolve().parent.parent / "web" / "public" / "presets"
MANIFEST = PRESET_DIR / "index.json"
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac"}
# `loadPresets` in web/lib/presets.ts drops any entry missing one of these.
REQUIRED = ("id", "name", "file", "license")


@pytest.fixture(scope="module")
def presets() -> list[dict]:
    entries = json.loads(MANIFEST.read_text())
    assert isinstance(entries, list), "the manifest is a JSON array"
    return entries


def test_every_preset_says_where_the_voice_came_from(presets):
    """Plan §8 item 4. A clip nobody can point to the source of does not ship,
    and no celebrity voice ships at all — `license` is where that is declared."""
    for entry in presets:
        for key in REQUIRED:
            value = entry.get(key)
            assert isinstance(value, str) and value.strip(), f"{entry.get('id')}: {key} is required"


def test_preset_ids_are_unique(presets):
    ids = [entry["id"] for entry in presets]
    assert len(ids) == len(set(ids))


def test_every_listed_file_is_actually_there(presets):
    for entry in presets:
        assert (PRESET_DIR / entry["file"]).is_file(), f"missing audio: {entry['file']}"
        # The path is joined into a URL by the client; keep it a bare name.
        assert "/" not in entry["file"] and ".." not in entry["file"]


def test_no_unlisted_audio_sits_in_the_directory(presets):
    """The reverse rule, and the one that catches a clip dropped in by hand:
    audio the manifest does not mention is audio nobody has vouched for."""
    listed = {entry["file"] for entry in presets}
    stray = sorted(
        path.name
        for path in PRESET_DIR.iterdir()
        if path.suffix.lower() in AUDIO_SUFFIXES and path.name not in listed
    )
    assert stray == [], f"unlisted audio in web/public/presets: {stray}"
