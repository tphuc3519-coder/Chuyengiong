"""
RVC mode for chuyengiong.

Ba endpoint:
  POST /add-model   -> tải model .zip từ URL (voice-models.com) vào Volume
  GET  /models      -> liệt kê model đã có
  POST /convert     -> đổi giọng cho một file vocal

Deploy:  modal deploy modal_rvc.py

App này đứng riêng khỏi `modal_app/`: nó không dùng chung Volume, job state hay
image nào với pipeline Seed-VC, và deploy bằng workflow riêng. Xem
docs/rvc-mode.md.
"""

import base64
import io
import pathlib
import shutil
import zipfile

import modal

app = modal.App("chuyengiong-rvc")

# Volume giữ các model .pth/.index — không mất khi container tắt
models_vol = modal.Volume.from_name("rvc-models", create_if_missing=True)
MODELS_DIR = "/models"


def _download_base_models():
    """Kéo hubert_base.pt + rmvpe.pt vào image lúc build.
    Thiếu hai file này RVC không chạy được."""
    import rvc_python
    from rvc_python.infer import RVCInference

    # "cpu" đúng chính tả, không phải "cpu:0": Config bật/tắt fp16 bằng đúng
    # phép so sánh `device != "cpu"`, nên "cpu:0" sẽ chạy half precision trên CPU.
    RVCInference(device="cpu")  # constructor tự tải base models

    # download_rvc_models() nuốt lỗi HTTP: non-200 thì nó chỉ in một dòng rồi
    # đi tiếp, nên build vẫn xanh với image thiếu weights và chết ở lần convert
    # đầu tiên. Kiểm lại ở đây để hỏng thì hỏng ngay lúc build.
    base = pathlib.Path(rvc_python.__file__).parent / "base_model"
    for filename in ("hubert_base.pt", "rmvpe.pt"):
        f = base / filename
        if not f.exists() or f.stat().st_size < 1_000_000:
            raise RuntimeError(f"Tải hụt base model {filename} — xem log ở trên")


image = (
    modal.Image.debian_slim(python_version="3.10")
    # build-essential vì fairseq 0.12.2 (rvc-python ghim cứng) chỉ có wheel
    # cho cp36/37/38 — trên 3.10 nó phải compile ba extension C++ từ source,
    # mà debian_slim không có g++.
    .apt_install("ffmpeg", "git", "build-essential")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # rvc-python ghim omegaconf==2.0.6, và metadata của bản đó khai
    # "PyYAML (>=5.1.*)" — specifier không hợp chuẩn. pip >= 24.1 bỏ qua hẳn
    # mọi distribution như vậy, nên báo "No matching distribution found for
    # omegaconf==2.0.6" chứ không phải lỗi resolve. Hạ pip là cách duy nhất
    # còn lại: phiên bản omegaconf là pin cứng của rvc-python, sửa không được.
    .run_commands("python -m pip install 'pip<24.1'")
    .pip_install(
        "rvc-python==0.1.5",
        "pydub==0.25.1",
        "soundfile==0.12.1",
        "numpy<2",
        "requests==2.32.3",
        "fastapi[standard]",
    )
    .run_function(_download_base_models)
)


# ----------------------------------------------------------------------
# 1. Thêm model từ URL
# ----------------------------------------------------------------------


@app.function(
    image=image,
    volumes={MODELS_DIR: models_vol},
    timeout=600,
)
@modal.fastapi_endpoint(method="POST")
def add_model(payload: dict):
    """
    Body: {"url": "https://...", "name": "son-tung"}

    Tải zip, giải nén, chỉ giữ .pth và .index.
    Đặt tên bằng chữ thường không dấu, không khoảng trắng.
    """
    import requests

    url = payload["url"]
    name = payload["name"].strip().lower().replace(" ", "-")

    dest = pathlib.Path(MODELS_DIR) / name
    if dest.exists():
        return {"ok": False, "error": f"Model '{name}' đã tồn tại"}
    dest.mkdir(parents=True)

    r = requests.get(url, stream=True, timeout=300, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    raw = io.BytesIO(r.content)

    pth = index = None
    try:
        with zipfile.ZipFile(raw) as z:
            for member in z.namelist():
                low = member.lower()
                if low.endswith(".pth") and pth is None:
                    pth = dest / f"{name}.pth"
                    pth.write_bytes(z.read(member))
                elif low.endswith(".index") and index is None:
                    index = dest / f"{name}.index"
                    index.write_bytes(z.read(member))
    except zipfile.BadZipFile:
        # Có link trả thẳng file .pth chứ không phải zip
        if url.lower().endswith(".pth"):
            pth = dest / f"{name}.pth"
            pth.write_bytes(r.content)

    if pth is None:
        shutil.rmtree(dest)
        return {"ok": False, "error": "Không tìm thấy file .pth trong link này"}

    models_vol.commit()
    return {
        "ok": True,
        "name": name,
        "has_index": index is not None,
        "size_mb": round(pth.stat().st_size / 1e6, 1),
    }


# ----------------------------------------------------------------------
# 2. Liệt kê model
# ----------------------------------------------------------------------


@app.function(image=image, volumes={MODELS_DIR: models_vol})
@modal.fastapi_endpoint(method="GET")
def list_models():
    models_vol.reload()
    root = pathlib.Path(MODELS_DIR)
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir():
            continue
        pth = next(d.glob("*.pth"), None)
        if pth:
            out.append(
                {
                    "name": d.name,
                    "has_index": next(d.glob("*.index"), None) is not None,
                    "size_mb": round(pth.stat().st_size / 1e6, 1),
                }
            )
    return {"models": out}


# ----------------------------------------------------------------------
# 3. Đổi giọng
# ----------------------------------------------------------------------


@app.function(
    image=image,
    gpu="A10G",
    volumes={MODELS_DIR: models_vol},
    timeout=1800,
    scaledown_window=300,  # giữ container ấm 5 phút, đỡ chờ load model
)
@modal.fastapi_endpoint(method="POST")
def convert(payload: dict):
    """
    Body:
      {
        "audio_b64": "...",       # vocal ĐÃ tách stem, wav hoặc mp3
        "model": "son-tung",
        "pitch": 0,               # nửa cung. Nam -> nữ: +12, nữ -> nam: -12
        "index_rate": 0.6,
        "protect": 0.33,
        "f0_method": "rmvpe"
      }
    Trả về: {"ok": true, "audio_b64": "..."}
    """
    from pydub import AudioSegment, silence
    from rvc_python.infer import RVCInference

    models_vol.reload()

    name = payload["model"]
    model_dir = pathlib.Path(MODELS_DIR) / name
    pth = next(model_dir.glob("*.pth"), None)
    if pth is None:
        return {"ok": False, "error": f"Không có model '{name}'"}
    index = next(model_dir.glob("*.index"), None)

    work = pathlib.Path("/tmp/work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    src = work / "in.wav"
    src.write_bytes(base64.b64decode(payload["audio_b64"]))

    # Chuẩn hoá về mono 44.1k trước khi đưa vào RVC
    audio = AudioSegment.from_file(src).set_channels(1).set_frame_rate(44100)

    rvc = RVCInference(device="cuda:0")
    rvc.load_model(str(pth), index_path=str(index) if index else "")
    # Tên tham số là f0up_key/f0method, KHÔNG phải f0_up_key/f0_method:
    # set_params() lọc theo whitelist rồi chỉ print warning cho tên lạ, nên
    # gõ sai là bị bỏ qua im lặng — pitch luôn 0 và f0 method luôn là
    # "harvest" mặc định, đúng cái làm giọng ra nghe như robot.
    rvc.set_params(
        f0up_key=int(payload.get("pitch", 0)),
        f0method=payload.get("f0_method", "rmvpe"),
        index_rate=float(payload.get("index_rate", 0.6)),
        protect=float(payload.get("protect", 0.33)),
        filter_radius=3,
        rms_mix_rate=0.25,
        resample_sr=0,
    )

    # Bài dài -> cắt theo khoảng lặng rồi ghép lại.
    # Cắt theo độ dài cố định sẽ để lại vết nối nghe rõ.
    CHUNK_LIMIT_MS = 90_000

    if len(audio) <= CHUNK_LIMIT_MS:
        segments = [audio]
    else:
        pieces = silence.split_on_silence(
            audio,
            min_silence_len=400,
            silence_thresh=audio.dBFS - 16,
            keep_silence=200,
        )
        if not pieces:
            pieces = [audio[i : i + CHUNK_LIMIT_MS] for i in range(0, len(audio), CHUNK_LIMIT_MS)]
        # Gộp các mảnh nhỏ lại cho đến gần giới hạn
        segments, buf = [], pieces[0]
        for p in pieces[1:]:
            if len(buf) + len(p) < CHUNK_LIMIT_MS:
                buf += p
            else:
                segments.append(buf)
                buf = p
        segments.append(buf)

    outputs = []
    for i, seg in enumerate(segments):
        seg_in = work / f"seg_{i}.wav"
        seg_out = work / f"seg_{i}_out.wav"
        seg.export(seg_in, format="wav")
        rvc.infer_file(input_path=str(seg_in), output_path=str(seg_out))
        outputs.append(AudioSegment.from_wav(seg_out))

    # Ghép, crossfade 30ms để không nghe thấy chỗ nối
    merged = outputs[0]
    for seg in outputs[1:]:
        merged = merged.append(seg, crossfade=min(30, len(seg) - 1))

    final = work / "out.wav"
    merged.export(final, format="wav")

    return {
        "ok": True,
        "chunks": len(segments),
        "audio_b64": base64.b64encode(final.read_bytes()).decode(),
    }
