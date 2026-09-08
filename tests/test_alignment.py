"""Giọng phải nằm đúng chỗ nó vốn nằm — trên trục thời gian, không chỉ trên trục cao độ.

Bài học: hai đường đi của một bản mix `song` không đối xứng. Nhạc nền đi thẳng
từ máy tách stem vào `amix`; giọng thì đi qua Seed-VC rồi qua cả dây lọc
`enhance`. Mỗi mili-giây lệch trong nhánh giọng là một mili-giây ca sĩ hát sau
ban nhạc, và không có gì trong `mix` báo lại chuyện đó.

Hai chỗ đã đo được là lệch, và cả hai đều **không phải lỗi của model**:

* `afftdn` là bộ khử ồn FFT chồng-cộng, nên nó chỉ trả lời được cho một mẫu sau
  khi có đủ cả cửa sổ chứa mẫu đó. ffmpeg không bù lại. Đo bằng một tiếng tách:
  giọng ra **chậm 25 ms** so với nhạc nền ở 44.1 kHz (15.6 ms ở 22.05 kHz).
* Seed-VC làm việc theo khung mel, `floor(len / hop)` khung, nên mỗi khúc trả
  về **hụt tới một hop** (511 mẫu, 11.6 ms ở 44.1 kHz). `crossfade_concat` nối
  theo một độ chồng cố định, nên chỗ hụt đó không tự mất đi — nó kéo mọi thứ
  phía sau lên sớm dần, cộng dồn qua từng mối nối.

Cả hai đều nhỏ đến mức nghe một lần không chắc, và cộng lại thì đủ để một người
nói "giọng không khớp beat". Nên chúng được giữ ở đây bằng phép đo chứ không
bằng tai.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from modal_app import mixing, styles
from modal_app.audio_utils import (
    CHUNK_OVERLAP_SEC,
    crossfade_concat,
    decode_wav,
    encode_wav,
    fit_length,
    split_at_silence,
)

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")

# Where each side's marker sits in the probe song, and how far apart that puts
# them. The mix may move the pair — `loudnorm` and the mp3 encoder both do —
# but nothing is allowed to move one without the other.
VOCAL_MARK_SEC = 1.0
BED_MARK_SEC = 2.5
PROBE_SEC = 4.0

# The whole point is sub-frame accuracy, so the tolerance is one mp3 granule
# (1152 samples, 26 ms) rather than "close enough". Anything looser would pass
# the 25 ms bug this file exists for.
TOLERANCE_SEC = 0.010


def marker(at_sec: float, seconds: float = PROBE_SEC, hz: float = 1000.0) -> np.ndarray:
    """Digital silence with one short tone burst in it, starting at `at_sec`."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    burst = int(0.02 * SR)
    t = np.arange(burst, dtype=np.float32) / SR
    audio[int(at_sec * SR) : int(at_sec * SR) + burst] = 0.6 * np.sin(2 * np.pi * hz * t)
    return audio


def onsets(mp3: bytes) -> list[float]:
    """The start of every burst in a finished mp3, in seconds."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "in.mp3").write_bytes(mp3)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path / "in.mp3"),
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-c:a",
                "pcm_s16le",
                str(path / "out.wav"),
            ],
            check=True,
        )
        audio, _ = decode_wav((path / "out.wav").read_bytes())
    loud = np.flatnonzero(np.abs(audio) >= 0.25 * np.abs(audio).max())
    # A burst is one run of loud samples; anything more than 100 ms after the
    # last one is the next burst.
    starts = [loud[0]]
    starts += [b for a, b in zip(loud[:-1], loud[1:], strict=True) if b - a > 0.1 * SR]
    return [start / SR for start in starts]


@needs_ffmpeg
@pytest.mark.parametrize("clarity", [0.0, 0.5, 1.0])
def test_clarity_never_moves_the_voice_off_the_beat(clarity):
    """The gap between the two markers survives the clarity chain, at every strength.

    Before the latency of `afftdn` was compensated this failed at 0.5 and 1.0
    by 25 ms and passed at 0.0, which is exactly the shape of the bug: turning
    the slider up moved the singer back.
    """
    mixed = mixing.mix(
        encode_wav(marker(VOCAL_MARK_SEC), SR),
        encode_wav(marker(BED_MARK_SEC), SR),
        clarity=clarity,
    )
    found = onsets(mixed)
    assert len(found) == 2, f"expected two markers, found {found}"
    gap = found[1] - found[0]
    assert abs(gap - (BED_MARK_SEC - VOCAL_MARK_SEC)) < TOLERANCE_SEC


@needs_ffmpeg
def test_a_bed_profile_does_not_move_the_voice_either():
    """The ducked graph splits the voice off before the chain; the trim is after it."""
    mixed = mixing.mix(
        encode_wav(marker(VOCAL_MARK_SEC), SR),
        encode_wav(marker(BED_MARK_SEC), SR),
        clarity=0.5,
        bed=styles.mix_for("ballad"),
    )
    found = onsets(mixed)
    assert len(found) == 2, f"expected two markers, found {found}"
    assert abs((found[1] - found[0]) - (BED_MARK_SEC - VOCAL_MARK_SEC)) < TOLERANCE_SEC


@needs_ffmpeg
def test_an_empty_chain_is_not_probed_and_is_not_late():
    assert mixing.chain_latency("", SR) == 0
    assert mixing.advance(0) == ""
    assert mixing.advance(-5) == ""


@needs_ffmpeg
def test_the_clarity_chain_is_measured_the_same_way_twice():
    """The measurement is cached, and a cache that returns something else is worse
    than no cache. Also pins the sign: a chain can only ever be *late*."""
    chain = mixing.enhance.chain(mixing.enhance.DEFAULT_CLARITY, ",")
    first = mixing.chain_latency(chain, SR)
    assert first >= 0
    assert mixing.chain_latency(chain, SR) == first


def test_advance_resets_timestamps_as_well_as_trimming():
    """`atrim` alone drops samples and keeps their timestamps, so `amix` puts the
    voice straight back where the trim just took it from."""
    filters = mixing.advance(1102)
    assert "atrim=start_sample=1102" in filters
    assert "asetpts" in filters


# --- the other half: chunks that come back short --------------------------

# Seed-VC's mel hop at 44.1 kHz. `to_mel` runs `center=False` over audio
# pre-padded by `(n_fft - hop) / 2` each side, which is exactly
# `floor(len / hop)` frames, and the vocoder gives back `hop` samples per
# frame — so a chunk comes back with its last partial frame missing.
MEL_HOP = 512


def as_the_model_returns_it(chunk: np.ndarray) -> np.ndarray:
    """A converted chunk's length, without a GPU: whole mel frames only."""
    return chunk[: len(chunk) // MEL_HOP * MEL_HOP]


def a_song_worth_chunking() -> np.ndarray:
    """Three minutes with quiet patches, so `split_at_silence` really splits it."""
    rng = np.random.default_rng(7)
    audio = rng.standard_normal(int(180 * SR)).astype(np.float32) * 0.2
    # A gap every 25 s for the splitter to find, offset so the cut points are
    # not multiples of the hop.
    for start in range(int(12.5 * SR), len(audio), int(25 * SR)):
        audio[start : start + int(0.6 * SR)] = 0.0
    return audio


def test_unpadded_chunks_pull_the_whole_vocal_earlier():
    """The bug, kept as a measurement: without the padding the join drifts.

    Not a regression test for code that exists — it is the arithmetic that
    makes `fit_length` necessary, asserted so nobody removes the padding on the
    grounds that eleven milliseconds cannot matter.
    """
    source = a_song_worth_chunking()
    chunks = split_at_silence(source, SR)
    assert len(chunks) > 1

    naive = crossfade_concat([as_the_model_returns_it(c) for c in chunks], SR, CHUNK_OVERLAP_SEC)
    lost = len(source) - len(naive)
    # One hop per chunk at worst, and in practice most of them.
    assert 0 < lost <= len(chunks) * MEL_HOP


def test_padding_each_chunk_back_puts_the_vocal_exactly_where_it_was():
    source = a_song_worth_chunking()
    chunks = split_at_silence(source, SR)

    joined = crossfade_concat(
        [fit_length(as_the_model_returns_it(c), len(c)) for c in chunks], SR, CHUNK_OVERLAP_SEC
    )
    assert len(joined) == len(source)


def test_fit_length_pads_with_silence_and_trims_without_asking():
    audio = np.ones(10, dtype=np.float32)
    assert len(fit_length(audio, 4)) == 4
    padded = fit_length(audio, 14)
    assert len(padded) == 14
    assert np.all(padded[10:] == 0.0)
    assert fit_length(audio, 10) is not None and len(fit_length(audio, 10)) == 10
