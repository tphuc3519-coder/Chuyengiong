# Chuyengiong

Voice conversion web app: đưa vào một bài hát (hoặc đoạn thoại) + một giọng mẫu,
nhận về bản đã đổi sang giọng đó, nhạc nền giữ nguyên.

Kế hoạch chi tiết theo từng phase: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Scaffold + deploy pipeline | 🟡 code xong, chờ verify deploy thật |
| 1 | Seed-VC + chunking | 🟡 code xong, chờ chạy thật trên GPU |
| 2 | Storage + job state | ⬜ |
| 3 | Nối pipeline hoàn chỉnh | ⬜ |
| 4 | Frontend | ⬜ |
| 5 | Pitch auto-detect | ⬜ |
| 6 | Consent gate | ⬜ |

## Cấu trúc

```
modal_app/
├── app.py          # Modal App, Volumes, Dict, images — nguồn duy nhất cho phần stateful
├── api.py          # FastAPI ASGI app; Phase 0 chỉ có /health
├── audio_utils.py  # chunk theo silence, crossfade, validate — numpy thuần, test được
└── conversion.py   # Seed-VC trên GPU: VoiceConverter (@app.cls)
tests/          # chạy bằng pytest, không cần Modal credentials và không cần GPU
.github/workflows/
├── ci.yml            # ruff + pytest trên mọi push/PR
└── deploy-modal.yml  # modal deploy khi main đổi
```

Container dependencies khai báo trong `modal_app/app.py` (Modal tự build image).
`requirements.txt` chỉ dành cho tooling local và CI.

## Chạy local

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ruff check . && ruff format --check .
pytest -q
```

## Deploy

CI/CD lo phần này: push lên `main` chạm `modal_app/**` → workflow `Deploy Modal`
chạy `modal deploy -m modal_app.api`.

Cần hai secret trong repo settings (Settings → Secrets and variables → Actions):

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Lấy bằng `modal token new` rồi đọc `~/.modal.toml`. Deploy tay:

```bash
modal deploy -m modal_app.api
```

Sau khi deploy, endpoint health:

```bash
curl https://<workspace>--voice-convert-api.modal.run/health
# {"status":"ok","app":"voice-convert"}
```

> URL khác một chút so với plan (`...-api-health.modal.run`): endpoint được phục vụ
> dưới dạng một ASGI app duy nhất tên `api`, nên `/submit`, `/status`, `/download`
> ở Phase 3 dùng chung một URL gốc thay vì mỗi cái một domain.

## Phase 1 — Seed-VC

`modal_app/conversion.py` port từ [`Plachtaa/seed-vc`](https://github.com/Plachtaa/seed-vc),
**pin ở commit `51383ef`** (không dùng `main`). Phần load model dùng lại nguyên
`inference.load_models` của repo gốc; phần chuyển đổi từng chunk là port của thân
`inference.main`, tách ra để reference chỉ encode một lần.

Chạy thử standalone (cần Modal credentials + GPU):

```bash
modal run -m modal_app.conversion --source song.mp3 --reference voice.wav \
  --mode singing --semitone-shift 0
```

### Checkpoint theo mode

| Mode | Checkpoint | SR | Content encoder |
|---|---|---|---|
| `singing` | seed-uvit-whisper-base (F0 conditioned) | 44100 | Whisper-small |
| `speech` | seed-uvit-whisper-small-wavenet | 22050 | Whisper-small |

Cả hai đều dùng Whisper-small nên phủ ~99 ngôn ngữ — **không** đổi sang checkpoint
`xlsr tiny` chỉ vì nó nhanh hơn.

### Ba chỗ khác plan, cố ý

1. **`mode` là class parameter, không phải tham số của `convert()`.** Hai mode dùng
   checkpoint, sample rate và vocoder khác nhau; `VoiceConverter(mode="singing")` để
   `@modal.enter()` load đúng một bộ model và giữ container ấm riêng cho từng mode.
   Phase 3 gọi `VoiceConverter(mode="singing").convert.remote(...)`.
2. **Reference cắt ở 20s, không phải 30s.** Seed-VC nhét cả source lẫn reference vào
   *một* context window 30s (`max_context_window` trong `inference.py`), nên reference
   30s sẽ không còn chỗ cho source. 20s để lại 10s source mỗi forward pass.
3. **Không cài nguyên `requirements.txt` của seed-vc.** Bỏ 3 dòng torch nightly
   (`--index-url`), bỏ gradio/FreeSimpleGUI/sounddevice (GUI) và
   jiwer/modelscope/funasr/resemblyzer (eval + training). Danh sách còn lại giữ
   nguyên pin của upstream — xem `SEED_VC_REQUIREMENTS`.

### Còn phải verify bằng tai và bằng GPU

Test trong `tests/` chỉ phủ phần logic thuần (chunk, crossfade, validate). Các mục
acceptance còn lại của Phase 1 trong plan cần file audio thật:

- [ ] file 8 phút chạy xong không OOM
- [ ] điểm nối chunk không click/pop, không hụt âm lượng
- [ ] tone nhất quán từ đầu đến cuối bài
- [ ] 4 ngôn ngữ khác họ chữ viết + cross-lingual (giọng Việt · bài Anh)
- [ ] lần gọi thứ hai nhanh hơn rõ rệt (container ấm), weights không tải lại
