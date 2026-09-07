"""The ffmpeg filter graphs, run for real.

These need ffmpeg — CI installs it, so the mix is exercised rather than
described. What is worth asserting is not "ffmpeg ran": it is the three
decisions that make a mix sound wrong when they are missed. A mix that is 6 dB
quieter than its inputs means `normalize=0` was dropped; a mix as long as the
shorter input means `duration=longest` was; a file with no comment tag means an
output left our hands without saying it was AI-generated.
"""

import shutil
import subprocess

import numpy as np
import pytest

from modal_app import mixing, styles
from modal_app.audio_utils import (
    decode_audio,
    decode_wav_channels,
    encode_wav,
    encode_wav_channels,
    to_pcm_wav,
)

SR = 44100

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(seconds: float, freq: float = 220.0, amplitude: float = 0.2) -> bytes:
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    return encode_wav(amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32), SR)


def duration_of(data: bytes) -> float:
    return len(decode_audio(data, SR)) / SR


def stereo_tone(seconds: float, left: float = 220.0, right: float = 330.0) -> bytes:
    """A wav with a different tone per channel, so a downmix is visible."""
    t = np.arange(int(seconds * SR), dtype=np.float32) / SR
    frames = np.stack(
        [0.2 * np.sin(2 * np.pi * left * t), 0.2 * np.sin(2 * np.pi * right * t)], axis=1
    )
    return encode_wav_channels(frames.astype(np.float32), SR)


def channels_of(data: bytes) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=nw=1:nk=1",
            "-",
        ],
        input=data,
        capture_output=True,
        check=True,
    )
    return int(out.stdout.decode().strip())


def bin_at(audio: np.ndarray, freq: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    return float(spectrum[np.argmin(abs(freqs - freq))])


def comment_of(data: bytes, tmp_path) -> str:
    path = tmp_path / "probe.mp3"
    path.write_bytes(data)
    out = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-show_entries",
            "format_tags=comment",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode().strip()


# --- gain clamping (no ffmpeg needed) -------------------------------------


def test_gain_is_clamped_to_the_usable_range():
    assert mixing.clamp_gain_db(40) == mixing.MAX_VOCAL_GAIN_DB
    assert mixing.clamp_gain_db(-40) == -mixing.MAX_VOCAL_GAIN_DB
    assert mixing.clamp_gain_db(3.5) == 3.5


def test_a_missing_or_unparseable_gain_is_no_gain():
    assert mixing.clamp_gain_db(None) == 0.0
    assert mixing.clamp_gain_db("loud") == 0.0


# --- the real thing --------------------------------------------------------


@needs_ffmpeg
def test_mix_runs_to_the_longer_input(tmp_path):
    """A converted vocal is never exactly as long as the instrumental it came
    from; truncating to the shorter one would cut the end off the song."""
    out = mixing.mix(tone(1.0), tone(2.0, freq=440.0))
    assert duration_of(out) == pytest.approx(2.0, abs=0.15)


@needs_ffmpeg
def test_mix_does_not_halve_the_level():
    """`amix` defaults to normalize=1, which divides by the input count."""
    quiet = decode_audio(mixing.mix(tone(2.0), tone(2.0, freq=440.0)), SR)
    assert float(np.abs(quiet).max()) > 0.3


@needs_ffmpeg
def test_vocal_gain_changes_the_balance():
    loud = decode_audio(mixing.mix(tone(2.0), tone(2.0, 440.0), vocal_gain_db=6), SR)
    soft = decode_audio(mixing.mix(tone(2.0), tone(2.0, 440.0), vocal_gain_db=-6), SR)

    # loudnorm pulls both to the same level, so compare the vocal's share of it:
    # the 220 Hz bin against the 440 Hz one.
    def share(audio):
        spectrum = np.abs(np.fft.rfft(audio[:SR]))
        freqs = np.fft.rfftfreq(SR, 1 / SR)
        return spectrum[np.argmin(abs(freqs - 220))] / spectrum[np.argmin(abs(freqs - 440))]

    assert share(loud) > share(soft)


@needs_ffmpeg
def test_every_output_is_tagged_ai_generated(tmp_path):
    assert comment_of(mixing.mix(tone(1.0), tone(1.0)), tmp_path) == mixing.AI_COMMENT
    assert comment_of(mixing.to_mp3(tone(1.0)), tmp_path) == mixing.AI_COMMENT


@needs_ffmpeg
def test_speech_output_is_an_mp3_of_the_same_length():
    assert duration_of(mixing.to_mp3(tone(1.5))) == pytest.approx(1.5, abs=0.15)


@needs_ffmpeg
def test_summing_four_stems_gives_one_instrumental():
    summed = mixing.sum_stems([tone(1.0, 220.0), tone(1.0, 440.0), tone(1.0, 660.0)])
    assert duration_of(summed) == pytest.approx(1.0, abs=0.1)


def test_summing_one_stem_is_a_no_op():
    """Cheap path for the 2-stem models, and it must not need ffmpeg."""
    data = tone(0.1)
    assert mixing.sum_stems([data]) is data


def test_empty_input_is_rejected_before_ffmpeg_sees_it():
    with pytest.raises(mixing.MixError):
        mixing.mix(b"", tone(0.1))
    with pytest.raises(mixing.MixError):
        mixing.sum_stems([])


# --- the watermark hook ----------------------------------------------------
#
# The model needs torch and its own container, so what is testable here is the
# contract `mixing` offers it: what it is handed, when it is called, and what
# happens to what it gives back.


@needs_ffmpeg
def test_the_watermark_is_handed_readable_audio_not_an_mp3():
    """It runs on the mixed wav, after loudnorm and before the encode. Handing
    it the encoded mp3 would mean marking a file that is already final."""
    seen = []

    def watermark(data: bytes) -> bytes:
        seen.append(decode_audio(data, SR))
        return data

    mixing.mix(tone(1.0), tone(1.0, 440.0), watermark=watermark)
    assert len(seen) == 1
    assert len(seen[0]) == pytest.approx(SR, abs=SR * 0.15)


@needs_ffmpeg
def test_what_the_watermark_returns_is_what_gets_encoded():
    """The whole point: the shipped mp3 is the marked audio, not the input."""
    marked = tone(1.0, freq=880.0)
    out = decode_audio(mixing.mix(tone(1.0), tone(1.0, 440.0), watermark=lambda _: marked), SR)

    spectrum = np.abs(np.fft.rfft(out[:SR]))
    freqs = np.fft.rfftfreq(SR, 1 / SR)
    peak = freqs[np.argmax(spectrum)]
    assert peak == pytest.approx(880.0, abs=10.0)


@needs_ffmpeg
def test_a_watermarked_output_is_still_tagged_ai_generated(tmp_path):
    out = mixing.mix(tone(1.0), tone(1.0), watermark=lambda data: data)
    assert comment_of(out, tmp_path) == mixing.AI_COMMENT


@needs_ffmpeg
def test_the_speech_branch_gets_the_same_hook():
    seen = []
    mixing.to_mp3(tone(1.0), watermark=lambda data: seen.append(data) or data)
    assert len(seen) == 1


@needs_ffmpeg
def test_a_watermark_that_produces_nothing_fails_the_job():
    """Rather than quietly shipping the unmarked mix: an output that claims to
    be watermarked and is not is worse than no output."""
    with pytest.raises(mixing.MixError):
        mixing.mix(tone(1.0), tone(1.0), watermark=lambda _: b"")


@needs_ffmpeg
def test_a_stereo_mix_reaches_the_watermark_as_stereo():
    """The watermark step reads the mix and writes it back. If it were handed a
    downmix, the song would come back mono."""
    seen = []
    mixing.mix(tone(1.0), stereo_tone(1.0), watermark=lambda data: seen.append(data) or data)
    assert channels_of(seen[0]) == 2


# --- channel layout --------------------------------------------------------


@needs_ffmpeg
def test_a_mono_vocal_does_not_drag_the_backing_track_down_to_mono():
    """`amix` settles on the narrowest layout across its inputs, and Seed-VC
    only ever returns mono — so without the `pan`, every song shipped with the
    instrumental's stereo image folded away."""
    out = mixing.mix(tone(1.0), stereo_tone(1.0, left=220.0, right=880.0))
    assert channels_of(out) == 2
    # The side that was only in the right channel is still there.
    left, right = _split(out)
    assert bin_at(right, 880.0) > 10 * bin_at(left, 880.0)


@needs_ffmpeg
def test_the_vocal_lands_in_both_channels_at_the_same_level():
    """`pan` rather than an `aformat` upmix: ffmpeg's mono-to-stereo conversion
    applies the -3 dB centre mix level, which would move the vocal down in the
    mix as a side effect of fixing the layout."""
    path_out = mixing.mix(tone(1.0, freq=220.0), stereo_tone(1.0, left=440.0, right=880.0))
    left, right = _split(path_out)
    assert bin_at(left, 220.0) == pytest.approx(bin_at(right, 220.0), rel=0.05)
    # ...and each side still carries only its own half of the backing track.
    assert bin_at(left, 440.0) > 10 * bin_at(left, 880.0)
    assert bin_at(right, 880.0) > 10 * bin_at(right, 440.0)


def _split(mp3: bytes) -> tuple[np.ndarray, np.ndarray]:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(SR),
            "pipe:1",
        ],
        input=mp3,
        capture_output=True,
        check=True,
    ).stdout
    frames = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    return frames[:, 0].copy(), frames[:, 1].copy()


# --- what the watermark step is handed ------------------------------------


def fmt_tag_and_rate(wav: bytes) -> tuple[int, int]:
    """`(wFormatTag, sample_rate)` straight out of the header."""
    at = wav.find(b"fmt ") + 8
    return (
        int.from_bytes(wav[at : at + 2], "little"),
        int.from_bytes(wav[at + 4 : at + 8], "little"),
    )


@needs_ffmpeg
def test_the_mix_handed_to_the_watermark_is_plain_pcm_at_the_vocals_rate():
    """The bug this exists for: `loudnorm` runs at 192 kHz and passes that rate
    on, which makes ffmpeg write a WAVE_FORMAT_EXTENSIBLE header. `wave` refuses
    that outright, so the watermark step could not read its own input and every
    song job died one stage from done with "unknown format: 65534"."""
    seen: dict = {}

    def watermark(wav: bytes) -> bytes:
        seen["tag"], seen["rate"] = fmt_tag_and_rate(wav)
        frames, rate = decode_wav_channels(wav)
        seen["channels"] = frames.shape[1]
        return encode_wav_channels(frames, rate)

    mixing.mix(tone(1.0), stereo_tone(1.0), 0.0, watermark=watermark)
    assert seen["tag"] == 1, "the watermark was handed something `wave` cannot read"
    assert seen["rate"] == SR, "loudnorm's 192 kHz was passed on instead of the mix rate"
    assert seen["channels"] == 2, "the stereo mix reached the watermark folded to mono"


@needs_ffmpeg
def test_the_mix_is_not_four_times_the_size_it_needs_to_be():
    """The same 192 kHz, from the other side: it quadrupled every intermediate
    wav and every sample the watermark model had to walk."""
    captured: list[bytes] = []
    mixing.mix(tone(1.0), stereo_tone(1.0), 0.0, watermark=lambda wav: captured.append(wav) or wav)
    # 1s of 16-bit stereo at SR, plus a header.
    assert len(captured[0]) < SR * 2 * 2 * 1.1


# --- the bed ---------------------------------------------------------------


def bursts(seconds: float, freq: float = 900.0, on: float = 1.0) -> bytes:
    """A voice that starts and stops, so ducking has gaps to be measured in."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    window = np.hanning(int(on * SR)).astype(np.float32)
    tone = np.sin(2 * np.pi * freq * np.arange(int(on * SR)) / SR).astype(np.float32)
    for start in range(0, int(seconds), 2):
        audio[start * SR : start * SR + len(window)] = 0.6 * window * tone
    return encode_wav(audio, SR)


def band_power_db(data: bytes, start: float, end: float, low: float, high: float) -> float:
    """How much energy sits in one band over one slice, in dB."""
    audio = decode_audio(data, SR)[int(start * SR) : int(end * SR)]
    power = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1.0 / SR)
    return 10 * np.log10(max(float(power[(freqs > low) & (freqs < high)].sum()), 1e-12))


@needs_ffmpeg
def test_a_mix_with_no_bed_profile_is_the_graph_this_module_always_had():
    """The property that keeps the old output reachable: `song` mode mixes a
    vocal back over the instrumental it was separated from, and nothing in a
    style has any business touching that balance."""
    voice, bed = bursts(6), tone(6, freq=110.0)
    plain = mixing.mix(voice, bed)
    steady = band_power_db(plain, 0.35, 0.75, 90, 130)
    gap = band_power_db(plain, 1.35, 1.75, 90, 130)
    assert steady == pytest.approx(gap, abs=0.2)


@needs_ffmpeg
@pytest.mark.parametrize("style_id", ["ballad", "auto", "trap"])
def test_the_bed_gets_out_of_the_way_while_the_voice_is_singing(style_id):
    """The filter that most separates a record from a karaoke track. Measured
    in the bed's own band so the voice on top of it is not what is being
    read."""
    voice, bed = bursts(6), tone(6, freq=110.0)
    mixed = mixing.mix(voice, bed, bed=styles.mix_for(style_id))
    ducked = band_power_db(mixed, 0.35, 0.75, 90, 130)
    open_ = band_power_db(mixed, 1.35, 1.75, 90, 130)
    assert open_ - ducked > 0.5


@needs_ffmpeg
def test_asking_for_more_ducking_gets_more_ducking():
    """`duck_db` is a dial with a scale on it rather than a calibrated number —
    what has to hold is that it is monotonic."""
    voice, bed = bursts(6), tone(6, freq=110.0)

    def depth(duck_db: float) -> float:
        profile = styles.Mixdown(duck_db=duck_db, pocket_db=0.0)
        mixed = mixing.mix(voice, bed, bed=profile)
        return band_power_db(mixed, 1.35, 1.75, 90, 130) - band_power_db(mixed, 0.35, 0.75, 90, 130)

    assert depth(0.0) == pytest.approx(0.0, abs=0.2)
    assert depth(6.0) > depth(3.0) > depth(1.5) > 0.3


@needs_ffmpeg
def test_the_vocal_gain_slider_is_a_level_and_not_a_ducking_control():
    """The key is split off at the input, before the gain, so turning the voice
    up changes the balance and not how hard the bed ducks. Taken after the gain
    the two would be the same knob, which is not something a user could infer."""
    voice, bed = bursts(6), tone(6, freq=110.0)
    profile = styles.mix_for("lofi")

    def depth(gain_db: float) -> float:
        mixed = mixing.mix(voice, bed, vocal_gain_db=gain_db, bed=profile)
        return band_power_db(mixed, 1.35, 1.75, 90, 130) - band_power_db(mixed, 0.35, 0.75, 90, 130)

    assert depth(6.0) == pytest.approx(depth(0.0), abs=0.6)


@needs_ffmpeg
def test_a_quiet_voice_ducks_the_bed_as_far_as_a_loud_one_does():
    """`sidechaincompress` compares the key against an absolute level, so
    without normalising it first the same style would duck a quiet vocal not at
    all and a loud one into the floor."""
    bed = tone(6, freq=110.0)
    loud = bursts(6)
    quiet = encode_wav((decode_audio(loud, SR) * 0.1).astype(np.float32), SR)
    profile = styles.mix_for("lofi")

    def depth(voice: bytes) -> float:
        mixed = mixing.mix(voice, bed, bed=profile)
        return band_power_db(mixed, 1.35, 1.75, 90, 130) - band_power_db(mixed, 0.35, 0.75, 90, 130)

    assert depth(quiet) == pytest.approx(depth(loud), abs=0.6)


@needs_ffmpeg
def test_the_bed_survives_a_voice_that_stops_before_the_song_does():
    """`sidechaincompress` is a framesync filter and framesync stops at the
    shorter input — so without `apad` on the key, a vocal that ends early cuts
    the backing track off at that point. Measured: an 8 second bed with a 4
    second key came back 4 seconds long."""
    short_voice = bursts(3)
    long_bed = tone(8, freq=110.0)
    mixed = mixing.mix(short_voice, long_bed, bed=styles.mix_for("trap"))
    assert duration_of(mixed) == pytest.approx(8.0, abs=0.2)
    # And the bed is open again under the outro rather than stuck ducked.
    assert band_power_db(mixed, 5.0, 6.0, 90, 130) > -60


@needs_ffmpeg
def test_the_pocket_takes_the_bed_out_of_the_band_the_voice_lives_in():
    """The static half of making room, next to the dynamic half. It carves the
    *place* the voice occupies rather than the moments it occupies it."""
    voice = bursts(6)
    time = np.arange(int(6 * SR), dtype=np.float32) / SR
    wide = (0.2 * np.sin(2 * np.pi * 2400 * time) + 0.2 * np.sin(2 * np.pi * 200 * time)).astype(
        np.float32
    )
    bed = encode_wav(wide, SR)

    flat = mixing.mix(voice, bed, bed=styles.Mixdown(duck_db=0.0, pocket_db=0.0))
    carved = mixing.mix(
        voice, bed, bed=styles.Mixdown(duck_db=0.0, pocket_db=-5.0, pocket_hz=2400.0)
    )

    # The bell against the band three and a half octaves below it, as a ratio —
    # `loudnorm` sits at the end of both mixes and hands back whatever make-up
    # gain the file needs, so an absolute level here would be measuring that
    # rather than the filter.
    def tilt(mixed: bytes) -> float:
        return band_power_db(mixed, 1.2, 1.8, 2300, 2500) - band_power_db(mixed, 1.2, 1.8, 180, 220)

    assert tilt(flat) - tilt(carved) > 3.0


@needs_ffmpeg
def test_a_mono_bed_is_widened_at_unity_and_not_three_decibels_down():
    """An `aformat` upmix applies the -3.01 dB centre mix level. Measured on a
    0.5 amplitude tone: `aformat` returns 0.354, `pan` returns 0.500."""
    mono = tone(2, freq=110.0, amplitude=0.5)
    assert mixing._channels(mono) == 1
    assert mixing._stereo(mono) == mixing.CENTRE
    assert mixing._stereo(stereo_tone(2)) == ""


def test_the_bed_chain_omits_every_filter_it_was_not_asked_for():
    """The same rule `enhance.chain` follows: a style that asks for nothing has
    to produce the graph this module had before styles existed."""
    nothing = mixing.mix_bed(styles.Mixdown(duck_db=0, pocket_db=0, low_shelf_db=0), 44100)
    assert nothing == "aresample=44100,highpass=f=30"
    everything = mixing.mix_bed(
        styles.Mixdown(duck_db=3, pocket_db=-3, low_shelf_db=2), 44100, trim_db=1.5
    )
    for expected in ("bass=", "equalizer=", "volume="):
        assert expected in everything


def test_the_measured_placement_goes_in_before_the_tone_shaping():
    """The shelf and the bell are fitted to the bed at the level it will sit
    at, not at the level it happened to arrive with."""
    chain = mixing.mix_bed(styles.Mixdown(low_shelf_db=2, pocket_db=-3), 44100, trim_db=-4.0)
    assert chain.index("volume=") < chain.index("bass=") < chain.index("equalizer=")


def test_every_bed_loses_its_rumble_whatever_it_came_from():
    """`beats.balance` does this to a generated bed and never runs on an
    uploaded one — and an uploaded beat is exactly the file most likely to
    carry a mastered-in sub that `loudnorm` then turns the whole mix down for."""
    assert f"highpass=f={mixing.BED_HIGHPASS_HZ:.0f}" in mixing.mix_bed(styles.NEUTRAL, 44100)


def test_no_ducking_asked_for_builds_no_sidechain_at_all():
    """A null compressor is still a compressor: it would resample, buffer and
    round-trip the bed for nothing."""
    graph = mixing.bed_graph(styles.Mixdown(duck_db=0.0), 44100, -3.0)
    assert "sidechaincompress" not in graph
    assert graph.endswith("[bed]")


def test_a_deeper_duck_is_a_lower_threshold():
    """The formula in `DUCK_LAW`, checked as a direction rather than a value."""

    def threshold(duck_db: float) -> float:
        graph = mixing.bed_graph(styles.Mixdown(duck_db=duck_db), 44100, 0.0)
        return float(graph.split("threshold=")[1].split(":")[0])

    assert threshold(9.0) < threshold(6.0) < threshold(3.0) < threshold(1.0) <= 1.0


def test_the_key_is_padded_so_framesync_cannot_truncate_the_bed():
    assert "apad" in mixing.bed_graph(styles.Mixdown(duck_db=3.0), 44100, -3.0)


@needs_ffmpeg
def test_a_peak_that_cannot_be_read_applies_no_correction_rather_than_a_wrong_one():
    assert mixing._peak_db(b"not audio") == 0.0
    assert mixing._peak_db(tone(1, amplitude=0.5)) == pytest.approx(-6.0, abs=0.3)


# --- placing the bed against the voice ------------------------------------


@needs_ffmpeg
def test_loudness_gates_the_silence_a_singer_leaves():
    """The reason this is `ebur128` and not `volumedetect`. A lead vocal is more
    silence than singing, and an ungated average measures how much the singer
    *rests* — which is not what anybody means by how loud they are."""
    gappy = decode_audio(bursts(20, on=1.5), SR)
    steady = np.sin(2 * np.pi * 260 * np.arange(len(gappy)) / SR).astype(np.float32)
    steady *= float(np.abs(gappy).max())

    gated = mixing._loudness_db(encode_wav(gappy, SR))
    solid = mixing._loudness_db(encode_wav(steady, SR))
    assert gated is not None and solid is not None
    # Two signals at the same peak. Ungated, the gappy one would read many dB
    # quieter; gated, it is within a few of the steady one.
    assert abs(gated - solid) < 6.0


@needs_ffmpeg
def test_nothing_measurable_is_reported_as_nothing_rather_than_guessed():
    assert mixing._loudness_db(encode_wav(np.zeros(2 * SR, dtype=np.float32), SR)) is None
    assert mixing._loudness_db(b"not audio") is None


@needs_ffmpeg
def test_the_same_style_places_the_bed_the_same_however_loud_it_arrived():
    """The whole point of measuring. A generated bed leaves `beats.balance` at a
    fixed RMS and an uploaded one is at whatever somebody mastered it to — so
    without this a style's balance is a wish rather than a setting."""
    voice = bursts(20, on=1.5)
    profile = styles.mix_for("auto")
    landed = []
    for amplitude in (0.6, 0.2, 0.9):
        bed = tone(20, freq=110.0, amplitude=amplitude)
        trim = mixing.bed_trim_db(profile, voice, bed, 0.0)
        landed.append(mixing._loudness_db(bed) + trim)
    assert max(landed) - min(landed) < 0.5

    # And it landed where the style asked, not merely consistently.
    voice_lufs = mixing._loudness_db(voice)
    assert landed[0] == pytest.approx(voice_lufs - profile.bed_below_voice_db, abs=0.5)


@needs_ffmpeg
def test_a_style_that_wants_the_bed_in_front_gets_it_in_front():
    """Negative means the bed is the point and the voice rides on it, which is
    the honest description of trap and of club music."""
    voice, bed = bursts(20, on=1.5), tone(20, freq=110.0)
    forward = mixing.bed_trim_db(styles.mix_for("trap"), voice, bed, 0.0)
    behind = mixing.bed_trim_db(styles.mix_for("ballad"), voice, bed, 0.0)
    assert forward > behind
    assert forward - behind == pytest.approx(
        styles.find("ballad").mix.bed_below_voice_db - styles.find("trap").mix.bed_below_voice_db,
        abs=0.01,
    )


@needs_ffmpeg
def test_the_vocal_gain_slider_moves_the_balance_with_it():
    """Turning the voice up should move the bed down by the same amount — that
    is what a balance slider means. It is also the one thing that slider must
    *not* change: the ducking depth stays where the style put it."""
    voice, bed = bursts(20, on=1.5), tone(20, freq=110.0)
    profile = styles.mix_for("auto")
    quiet = mixing.bed_trim_db(profile, voice, bed, -6.0)
    loud = mixing.bed_trim_db(profile, voice, bed, 6.0)
    assert loud - quiet == pytest.approx(12.0, abs=0.01)


@needs_ffmpeg
def test_a_bed_that_cannot_be_measured_is_left_at_the_level_it_came_with():
    """No correction rather than a guess: the behaviour this app had before
    anything was measured."""
    voice = bursts(6)
    silent = encode_wav(np.zeros(2 * SR, dtype=np.float32), SR)
    assert mixing.bed_trim_db(styles.NEUTRAL, voice, silent, 0.0) == 0.0
    assert mixing.bed_trim_db(styles.NEUTRAL, silent, tone(6), 0.0) == 0.0


@needs_ffmpeg
def test_the_correction_is_bounded_so_a_near_silent_bed_is_not_amplified():
    voice = bursts(20, on=1.5)
    whisper = tone(20, freq=110.0, amplitude=0.002)
    assert mixing.bed_trim_db(styles.NEUTRAL, voice, whisper, 0.0) == mixing.MAX_BED_TRIM_DB


@needs_ffmpeg
def test_a_quiet_beat_and_a_loud_one_end_up_in_the_same_mix():
    """End to end, through the whole graph: the measurement has to survive the
    shelf, the pocket, the duck and `loudnorm`."""
    voice = bursts(20, on=1.5)
    profile = styles.mix_for("lofi")
    levels = [
        band_power_db(
            mixing.mix(voice, tone(20, freq=110.0, amplitude=a), bed=profile), 2.2, 2.8, 90, 130
        )
        for a in (0.6, 0.15)
    ]
    assert abs(levels[0] - levels[1]) < 1.5


# --- a stereo bed ---------------------------------------------------------


@needs_ffmpeg
def test_a_stereo_bed_arrives_in_the_mix_still_stereo():
    """The vocal is mono and centred, so a stereo bed is the difference between
    a voice in front of an arrangement and a voice on top of a thing in exactly
    the same place in the image."""
    voice = bursts(4)
    mixed = mixing.mix(voice, stereo_tone(4, left=110.0, right=170.0), bed=styles.mix_for("auto"))
    audio, _ = decode_wav_channels(to_pcm_wav(mixed, SR))
    assert audio.shape[1] == 2
    assert not np.allclose(audio[:, 0], audio[:, 1])
