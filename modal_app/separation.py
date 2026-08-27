"""Vocal/instrumental separation, ported from `chamaya00/tachnhac`.

The model choice, the checkpoint file names, the CUDA base image and the output
name normalisation all come across unchanged from that app's `modal_app.py` —
it is tuned and it works, and this module is not the place to relitigate it.
What changed is I/O only, as the plan requires: the old app read and wrote the
job directory on its own Volume and served stems to a browser, this one takes
bytes and gives back bytes so `pipeline.py` owns storage.

Two carried-over details that look incidental and are not:

* the image is `nvidia/cuda:...-cudnn-...`, not `debian_slim`. `onnxruntime-gpu`
  needs `libcudnn.so.9`; without it separation silently falls back to CPU, where
  BS-Roformer is tens of times slower;
* `clang` and `build-essential` are installed because demucs pulls in `diffq`,
  which has no wheel and must compile against Modal's clang-built Python.

Unlike the old app this writes WAV, not MP3: these stems feed Seed-VC and the
final mix, so there is no reason to put a lossy generation in the middle.

Smoke test (needs Modal credentials and a GPU):

    modal run -m modal_app.separation::separate_files --source song.mp3
"""

# No `from __future__ import annotations`: modal.parameter() reads the raw
# class annotation and cannot resolve a stringified one.
import modal

from .app import MODEL_DIR, app, model_vol

# Ported verbatim from the old app. `file` is what audio-separator resolves
# against its model list, so these strings are not ours to prettify.
SEPARATION_MODELS = {
    # 2 stem, best vocal quality available — the default, per the plan's risk
    # table ("dùng BS-Roformer thay HTDemucs" when the vocal has artifacts).
    "roformer": {
        "file": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "stems": ["Vocals", "Instrumental"],
        "label": "BS-Roformer (2 stem)",
    },
    # 4 stem, roughly 4x faster; drums/bass/other are summed back into one
    # instrumental below.
    "htdemucs": {
        "file": "htdemucs.yaml",
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "label": "HTDemucs (4 stem, fast)",
    },
}
DEFAULT_SEPARATION_MODEL = "roformer"

VOCAL_STEM = "vocals"
INSTRUMENTAL_STEM = "instrumental"

# audio-separator picks its decoder from the file extension, so the temp file
# it reads has to keep the one the upload arrived with.
ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".aac", ".wma"}
DEFAULT_EXT = ".mp3"

# Named constants rather than literals inside the builder: these two strings
# are the whole reason separation runs on a GPU at all, and a test asserts on
# them (a Modal Image does not expose its own build steps).
CUDA_IMAGE_TAG = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04"
# >=0.28 is the release that added `custom_output_names`. The `[gpu]` extra is
# what pulls onnxruntime-gpu instead of the CPU build.
AUDIO_SEPARATOR_SPEC = "audio-separator[gpu]>=0.28,<1.0"

separation_image = (
    modal.Image.from_registry(CUDA_IMAGE_TAG, add_python="3.11")
    .apt_install("ffmpeg", "git", "clang", "build-essential")
    .pip_install(AUDIO_SEPARATOR_SPEC)
    .env({"AUDIO_SEPARATOR_MODEL_DIR": MODEL_DIR})
    .add_local_python_source("modal_app")
)


class SeparationError(RuntimeError):
    """The model ran but produced nothing we can use."""


def check_model(name: str) -> str:
    if name not in SEPARATION_MODELS:
        raise SeparationError(
            f"separation model must be one of {tuple(SEPARATION_MODELS)}, got {name!r}"
        )
    return name


def safe_ext(name: str | None) -> str:
    """The extension we recognise in `name`, else `.mp3`.

    Takes either a file name (`song.m4a`) or a bare extension (`.m4a`): the API
    extracts it from the upload and the pipeline re-checks the stored value, and
    both go through here, so it has to be idempotent.
    """
    import os

    text = (name or "").strip().lower()
    ext = os.path.splitext(text)[1] or text
    return ext if ext in ALLOWED_EXTS else DEFAULT_EXT


def point_output_at(separator, out_dir: str) -> None:
    """Send the next `separate()` call's stems to `out_dir`.

    Both the wrapper and the architecture instance, not just the wrapper.
    `load_model` copies the directory it was constructed with into the
    per-architecture separator's config, and *that* copy is the one the stem
    write joins against — so moving the wrapper alone leaves the stems in
    MODEL_DIR, the caller collects an empty directory, and the job fails as
    "produced no vocal stem" with the model working perfectly. audio-separator
    does the same two-step wherever it redirects its own output.
    """
    separator.output_dir = out_dir
    instance = getattr(separator, "model_instance", None)
    if instance is not None:
        instance.output_dir = out_dir


def _collect_stems(out_dir, stems: list[str]) -> dict[str, str]:
    """Rename whatever the model wrote to `<stem>.<ext>`. Ported as-is.

    `custom_output_names` does not always take: depending on the architecture
    the file can come out as `input_(Vocals)_model_bs_roformer_....wav`. The
    caller needs predictable names, so this matches on the stem name appearing
    anywhere in the file name and falls back to positional assignment.
    """
    import os
    import re
    import shutil

    found: dict[str, str] = {}
    leftovers: list[str] = []

    for name in sorted(os.listdir(out_dir)):
        if not name.lower().endswith((".wav", ".mp3", ".flac")):
            continue
        base, ext = os.path.splitext(name)
        low = base.lower()

        match = None
        for stem in stems:
            key = stem.lower()
            if low == key or re.search(rf"[(\[_\-]{key}[)\]_\-]?", low):
                match = key
                break

        if match is None or match in found:
            leftovers.append(name)
            continue

        target = f"{match}{ext.lower()}"
        if name != target:
            shutil.move(os.path.join(out_dir, name), os.path.join(out_dir, target))
        found[match] = target

    # Nothing matched by name: serve them in declaration order rather than
    # failing the whole job over a file naming scheme.
    for name in leftovers:
        remaining = [s.lower() for s in stems if s.lower() not in found]
        if not remaining:
            break
        ext = os.path.splitext(name)[1].lower()
        target = f"{remaining[0]}{ext}"
        shutil.move(os.path.join(out_dir, name), os.path.join(out_dir, target))
        found[remaining[0]] = target

    return {s.lower(): found[s.lower()] for s in stems if s.lower() in found}


@app.cls(
    image=separation_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,
    timeout=1800,
    max_containers=4,
)
class Separator:
    model: str = modal.parameter(default=DEFAULT_SEPARATION_MODEL)

    @modal.enter()
    def load(self) -> None:
        """Load the checkpoint once per container; weights cache in the Volume."""
        import os

        import torch
        from audio_separator.separator import Separator as AudioSeparator

        check_model(self.model)
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Log the device we really got. On CPU this model is slow enough to hit
        # the pipeline timeout, so a silent fallback must not stay silent.
        self.device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        cfg = SEPARATION_MODELS[self.model]
        self._separator = AudioSeparator(
            model_file_dir=MODEL_DIR,
            output_dir=MODEL_DIR,  # replaced per call; audio-separator needs one now
            output_format="WAV",
        )
        self._separator.load_model(model_filename=cfg["file"])
        model_vol.commit()
        print(f"[Separator] model={self.model} device={self.device}")

    @modal.method()
    def separate(self, audio: bytes, ext: str = DEFAULT_EXT) -> dict:
        """Split `audio` into `{"vocals": wav bytes, "instrumental": wav bytes}`.

        Bytes in and out rather than Volume paths: it keeps this callable
        standalone, and Modal moves anything over 2 MiB through blob storage
        anyway, so the argument size is not the thing to optimise here.
        """
        import tempfile
        import time
        from pathlib import Path

        from .mixing import sum_stems

        if not audio:
            raise SeparationError("no input audio")

        cfg = SEPARATION_MODELS[check_model(self.model)]
        started = time.time()

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            src = tmpdir / f"input{safe_ext(ext)}"
            src.write_bytes(audio)
            out_dir = tmpdir / "stems"
            out_dir.mkdir()

            # audio-separator takes its output directory from the instance,
            # not from separate(), so point it at this call's temp dir.
            point_output_at(self._separator, str(out_dir))
            names = {}
            for stem in cfg["stems"]:
                names[stem] = stem.lower()
                names[stem.lower()] = stem.lower()
            try:
                self._separator.separate(str(src), custom_output_names=names)
            except TypeError:
                # audio-separator older than 0.28: no custom_output_names, but
                # _collect_stems normalises the names it does produce.
                self._separator.separate(str(src))

            produced = _collect_stems(out_dir, cfg["stems"])
            if VOCAL_STEM not in produced:
                raise SeparationError(
                    f"model produced no vocal stem (got {sorted(produced) or 'nothing'})"
                )
            data = {name: (out_dir / file).read_bytes() for name, file in produced.items()}

        vocals = data.pop(VOCAL_STEM)
        rest = [data[name] for name in sorted(data)]
        if not rest:
            raise SeparationError("model produced a vocal stem and nothing else")
        instrumental = data.get(INSTRUMENTAL_STEM) or sum_stems(rest)

        print(
            f"[Separator] {len(audio) / 1e6:.1f} MB -> vocals + instrumental "
            f"in {time.time() - started:.1f}s on {self.device}"
        )
        return {VOCAL_STEM: vocals, INSTRUMENTAL_STEM: instrumental}


@app.local_entrypoint()
def separate_files(source: str, model: str = DEFAULT_SEPARATION_MODEL, out: str = ".") -> None:
    """Standalone smoke test: one local audio file in, two wavs out."""
    from pathlib import Path

    src = Path(source)
    stems = Separator(model=model).separate.remote(src.read_bytes(), src.suffix)
    for name, data in stems.items():
        path = Path(out) / f"{name}.wav"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data) / 1e6:.1f} MB)")
