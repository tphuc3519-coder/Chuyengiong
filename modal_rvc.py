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

import io
import json
import pathlib
import re
import shutil
import unicodedata
import zipfile
from typing import Annotated

import modal
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

app = modal.App("chuyengiong-rvc")

# Trình duyệt POST thẳng vào /convert (xem docstring của nó), nên response phải
# nói rõ là đọc được từ origin khác. Mở cho mọi origin: ba endpoint này vốn
# không có xác thực, nên siết CORS chẳng bảo vệ được gì mà chỉ làm preview
# deploy của Vercel hỏng — CORS chặn trình duyệt, không chặn curl.
_CORS = {"Access-Control-Allow-Origin": "*"}


def _json_error(message: str, status: int) -> JSONResponse:
    """Lỗi cũng phải kèm CORS, không thì trình duyệt chỉ thấy 'failed to fetch'
    và người dùng mất luôn câu giải thích."""
    return JSONResponse({"ok": False, "error": message}, status_code=status, headers=_CORS)


# Volume giữ các model .pth/.index — không mất khi container tắt
models_vol = modal.Volume.from_name("rvc-models", create_if_missing=True)
MODELS_DIR = "/models"

# Tên model trở thành tên thư mục trên Volume, và ba endpoint đều mở không xác
# thực — nên tên phải là *không thể* chứa dấu phân cách đường dẫn, không phải
# *thường là không*. Bắt đầu bằng chữ/số nên ".." cũng không lọt.
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")


def _safe_name(raw: str) -> str | None:
    """Chuẩn hoá tên model, trả về None nếu không còn gì dùng được.

    Bỏ dấu thay vì từ chối: người dùng gõ "Sơn Tùng MTP" là chuyện bình thường,
    và trả về lỗi cho một cái tên hợp lý là cách chắc chắn làm họ bỏ cuộc.
    Hàm này phải khớp với `normalise()` bên web/app/rvc/page.tsx — trang đó
    hiện trước tên sẽ lưu, lệch nhau là hứa một đằng làm một nẻo.
    """
    lowered = raw.strip().lower().replace("đ", "d")
    bare = "".join(c for c in unicodedata.normalize("NFD", lowered) if not unicodedata.combining(c))
    name = re.sub(r"[^a-z0-9._-]+", "-", bare)
    name = re.sub(r"-{2,}", "-", name).strip("-._")
    return name if _SAFE_NAME.fullmatch(name) else None


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


def _checkpoint_info(pth: pathlib.Path) -> dict:
    """Đọc ba thứ trong header checkpoint mà mọi thứ sau đó phụ thuộc vào.

    `version` là quan trọng nhất, và nó là chỗ hỏng im lặng nhất trong cả
    rvc-python: `get_vc()` lấy version từ *tham số truyền vào* chứ không đọc từ
    checkpoint (nhánh dọn dẹp ngay phía trên nó thì lại đọc — hai nhánh không
    khớp nhau), mặc định là "v2". Rồi ngay dưới là
    `load_state_dict(..., strict=False)`.

    Nên nạp một model v1 mà không nói rõ v1 thì: kiến trúc dựng sai chiều
    (256 vs 768), weight không khớp bị **bỏ qua không báo gì**, hubert lấy nhầm
    layer (9 vs 12) và bỏ qua `final_proj`. Không có lỗi nào được ném ra, chỉ
    có âm thanh ra nghe như robot — ở mọi pitch, nên chỉnh tham số không bao
    giờ cứu được.
    """
    import torch

    # weights_only=False vì checkpoint RVC là dict có metadata, không chỉ tensor
    # — cũng đúng cách rvc-python tự load nó ngay sau đây.
    cpt = torch.load(pth, map_location="cpu", weights_only=False)
    return {
        "version": cpt.get("version", "v1"),
        "f0": int(cpt.get("f0", 1)),
        "sr": int(cpt["config"][-1]),
    }


def _info_for(model_dir: pathlib.Path, pth: pathlib.Path) -> dict:
    """`_checkpoint_info` có nhớ: đọc một checkpoint 55MB mỗi lần liệt kê là phí."""
    cache = model_dir / "info.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    info = _checkpoint_info(pth)
    cache.write_text(json.dumps(info))
    models_vol.commit()
    return info


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
    name = _safe_name(payload["name"])
    if name is None:
        return {"ok": False, "error": "Tên chỉ được dùng a-z, 0-9, dấu chấm, gạch ngang, gạch dưới"}

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

    result = {
        "ok": True,
        "name": name,
        "has_index": index is not None,
        "size_mb": round(pth.stat().st_size / 1e6, 1),
    }
    try:
        result.update(_info_for(dest, pth))
    except Exception:
        # File tải về không phải checkpoint RVC hợp lệ. Vẫn giữ lại — convert sẽ
        # nói rõ hơn — nhưng đừng để việc thêm giọng chết ở đây.
        pass
    return result


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
            entry = {
                "name": d.name,
                "has_index": next(d.glob("*.index"), None) is not None,
                "size_mb": round(pth.stat().st_size / 1e6, 1),
            }
            # Đọc được thì kèm theo, hỏng thì thôi: liệt kê giọng không được
            # chết chỉ vì một checkpoint lạ.
            try:
                entry.update(_info_for(d, pth))
            except Exception:
                pass
            out.append(entry)
    return {"models": out}


# Bài dài phải cắt nhỏ mới đưa qua GPU được. 90 giây là mức đã chạy ổn.
CHUNK_LIMIT_MS = 90_000


def _chunk_bounds(
    total_ms: int, quiet: list[tuple[int, int]], limit_ms: int = CHUNK_LIMIT_MS
) -> list[tuple[int, int]]:
    """Chia [0, total_ms) thành các đoạn LIỀN NHAU, cắt giữa khoảng lặng.

    Điểm sống còn là *liền nhau*: các đoạn phủ kín từ đầu đến cuối, không chồng
    nhau và không chừa khoảng nào, nên ghép lại đúng bằng độ dài ban đầu.

    Bản trước dùng `split_on_silence`, mà đó là hàm **vứt bỏ** khoảng lặng —
    `keep_silence=200` chỉ chừa 200ms mỗi đầu, nên mọi quãng nghỉ dài trong bài
    bị nuốt. Một bài 1:31 ra 1:09: lời vẫn đủ, nhưng vocal không còn khớp với
    nhạc nền được nữa — mà ghép lại với nhạc nền chính là mục đích.

    Nên ở đây khoảng lặng chỉ dùng để *chọn chỗ cắt*, không phải để bỏ đi.
    Không có khoảng lặng nào trong tầm thì cắt cứng đúng giới hạn: thà một vết
    nối khẽ còn hơn tràn bộ nhớ GPU.
    """
    if total_ms <= limit_ms:
        return [(0, total_ms)]

    mids = sorted((start + end) // 2 for start, end in quiet)

    bounds = [0]
    while total_ms - bounds[-1] > limit_ms:
        start = bounds[-1]
        # Khoảng lặng xa nhất mà vẫn trong giới hạn: cắt ít lần nhất có thể.
        reachable = [m for m in mids if start < m <= start + limit_ms]
        bounds.append(reachable[-1] if reachable else start + limit_ms)
    bounds.append(total_ms)

    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _speak(text: str, language: str) -> bytes:
    """Đọc văn bản bằng đúng engine mà app chính đã deploy sẵn.

    Gọi chéo sang app `voice-convert` chứ không dựng lại engine trong image
    này. Image TTS bên đó có một loạt thứ phải đúng mới chạy — `unidic` đè
    `unidic-lite` làm chết container trước khi đọc được chữ nào, `open_jtalk`
    compile từ sdist, `transformers` phải ghim — và `prosody.py` là cả một kế
    hoạch đọc (ngắt theo dấu câu, hạ dần cao độ, câu hỏi lên giọng) mà chép
    lại là chép sai.

    Ghép lỏng, cố ý: chỉ tra tên lúc chạy, không import. `voice-convert` chưa
    deploy thì hỏng ở đây với một câu đọc được, chứ không làm chết build của
    app này.
    """
    # Quy tắc chọn engine là của modal_app/tts.py (`spec_for`): chỉ tiếng Nhật
    # đọc qua Kokoro, còn lại qua MMS. Nhân đúng một dòng ở đây thay vì kéo cả
    # modal_app vào image này — nếu bên đó đổi quy tắc thì đây phải đổi theo.
    cls_name = "KokoroSynthesizer" if language == "jpn" else "Synthesizer"
    synthesizer = modal.Cls.from_name("voice-convert", cls_name)
    return synthesizer(language=language).synthesize.remote(text=text)


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
def convert(
    model: Annotated[str, Form()],
    audio: Annotated[UploadFile | None, File()] = None,
    text: Annotated[str, Form()] = "",
    language: Annotated[str, Form()] = "vie",
    pitch: Annotated[int, Form()] = 0,
    index_rate: Annotated[float, Form()] = 0.6,
    protect: Annotated[float, Form()] = 0.33,
    f0_method: Annotated[str, Form()] = "rmvpe",
):
    """Đổi giọng cho một file vocal ĐÃ tách stem.

    Nhận multipart và trả thẳng bytes wav, không phải JSON+base64, vì hai lý do
    ăn nhau:

    1. Trình duyệt gọi thẳng vào đây chứ không qua route của Vercel — body của
       serverless function bị chặn ở 4.5MB *cả hai chiều*, mà wav trả về của
       một bài 4 phút đã hơn 40MB. Proxy không tải nổi.
    2. multipart/form-data là content-type nằm trong danh sách an toàn của CORS,
       nên POST kiểu này không sinh preflight. Chỉ cần đúng một header
       Access-Control-Allow-Origin ở response là trình duyệt đọc được.

    base64 cũng biến mất luôn, đỡ 33% dung lượng ở cả hai chiều.
    """
    from pydub import AudioSegment, silence
    from rvc_python.infer import RVCInference

    models_vol.reload()

    name = _safe_name(model)
    if name is None:
        return _json_error("Tên model không hợp lệ", 400)
    model_dir = pathlib.Path(MODELS_DIR) / name
    pth = next(model_dir.glob("*.pth"), None)
    if pth is None:
        return _json_error(f"Không có model '{name}'", 404)
    index = next(model_dir.glob("*.index"), None)

    work = pathlib.Path("/tmp/work")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # Nguồn giọng: một file đã có, hoặc một đoạn văn bản được đọc ra ngay tại
    # đây. Cùng một endpoint vì phần sau giống hệt nhau — và vì thêm endpoint
    # thứ tư nghĩa là thêm một URL nữa phải cấu hình trên Vercel.
    if text.strip():
        try:
            source = _speak(text.strip(), language)
        except Exception as err:
            return _json_error(f"Không đọc được văn bản: {err}", 502)
    elif audio is not None:
        source = audio.file.read()
    else:
        return _json_error("Cần một file giọng hoặc một đoạn văn bản", 400)

    src = work / "in"
    src.write_bytes(source)

    # Chuẩn hoá về mono 44.1k trước khi đưa vào RVC
    try:
        segment = AudioSegment.from_file(src).set_channels(1).set_frame_rate(44100)
    except Exception:
        return _json_error("Không đọc được file audio này", 400)

    info = _info_for(model_dir, pth)

    rvc = RVCInference(device="cuda:0")
    # version PHẢI lấy từ checkpoint — xem _checkpoint_info(). Bỏ tham số này
    # là nạp mọi model dưới dạng v2, và model v1 sẽ ra tiếng robot ở mọi pitch.
    rvc.load_model(str(pth), version=info["version"], index_path=str(index) if index else "")
    # Tên tham số là f0up_key/f0method, KHÔNG phải f0_up_key/f0_method:
    # set_params() lọc theo whitelist rồi chỉ print warning cho tên lạ, nên
    # gõ sai là bị bỏ qua im lặng — pitch luôn 0 và f0 method luôn là
    # "harvest" mặc định, đúng cái làm giọng ra nghe như robot.
    rvc.set_params(
        f0up_key=int(pitch),
        f0method=f0_method,
        index_rate=float(index_rate),
        protect=float(protect),
        filter_radius=3,
        rms_mix_rate=0.25,
        resample_sr=0,
    )

    # detect_silence CHỈ để chọn chỗ cắt. Khoảng lặng nằm nguyên trong đoạn nó
    # thuộc về — xem _chunk_bounds() về chuyện này từng sai thế nào.
    quiet: list[tuple[int, int]] = []
    if len(segment) > CHUNK_LIMIT_MS:
        quiet = silence.detect_silence(
            segment, min_silence_len=300, silence_thresh=segment.dBFS - 16
        )
    bounds = _chunk_bounds(len(segment), quiet)

    outputs = []
    for i, (start, end) in enumerate(bounds):
        seg_in = work / f"seg_{i}.wav"
        seg_out = work / f"seg_{i}_out.wav"
        segment[start:end].export(seg_in, format="wav")
        rvc.infer_file(input_path=str(seg_in), output_path=str(seg_out))
        outputs.append(AudioSegment.from_wav(seg_out))

    # Nối thẳng, KHÔNG crossfade: crossfade ăn mất 30ms ở mỗi mối, tức là lại
    # rút ngắn bài thêm lần nữa. Chỗ cắt vốn nằm trong khoảng lặng nên không có
    # gì để hoà vào nhau; 3ms fade hai bên đủ chặn tiếng "tách" ở lần cắt cứng,
    # và fade thì không đổi độ dài.
    merged = outputs[0]
    for seg in outputs[1:]:
        merged = merged.fade_out(3) + seg.fade_in(3)

    final = work / "out.wav"
    merged.export(final, format="wav")

    return Response(
        content=final.read_bytes(),
        media_type="audio/wav",
        headers={
            **_CORS,
            # Trình duyệt chỉ đọc được header nào được liệt kê ở đây; thiếu nó
            # thì fetch() thấy response 200 mà không thấy một header nào.
            "Access-Control-Expose-Headers": (
                "X-Rvc-Chunks, X-Rvc-Version, X-Rvc-F0, X-Rvc-In-Ms, X-Rvc-Out-Ms"
            ),
            "X-Rvc-Chunks": str(len(bounds)),
            # Hai số này phải gần bằng nhau. Lệch nhiều nghĩa là khoảng lặng
            # lại bị nuốt, và vocal sẽ không ghép được với nhạc nền nữa.
            "X-Rvc-In-Ms": str(len(segment)),
            "X-Rvc-Out-Ms": str(len(merged)),
            "X-Rvc-Version": info["version"],
            "X-Rvc-F0": str(info["f0"]),
        },
    )
