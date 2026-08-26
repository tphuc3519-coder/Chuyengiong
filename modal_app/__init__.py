"""Modal backend for the voice conversion app.

`app.py` holds the App/Volume/Dict declarations and the shared images, `api.py`
the web endpoints, `pipeline.py` the orchestration that chains one job together,
and `separation.py` / `conversion.py` / `mixing.py` the three steps it chains:
stems on a GPU, Seed-VC on a GPU, ffmpeg on CPU. `audio_utils.py` holds the
plain-numpy audio helpers, `storage.py` and `jobs.py` the state underneath.

Nothing here imports its submodules: the containers each import only what they
run, and the API image has neither torch nor audio-separator in it.
"""
