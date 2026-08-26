# Chuyengiong

Voice conversion web app: đưa vào một bài hát (hoặc đoạn thoại) + một giọng mẫu,
nhận về bản đã đổi sang giọng đó, nhạc nền giữ nguyên.

Kế hoạch chi tiết theo từng phase: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Scaffold + deploy pipeline | 🟡 code xong, chờ verify deploy thật |
| 1 | Seed-VC + chunking | 🟡 code xong, chờ chạy thật trên GPU |
| 2 | Storage + job state | 🟡 code xong, chờ chạy `modal run -m modal_app.verify` |
| 3 | Nối pipeline hoàn chỉnh | 🟡 code xong, chờ chạy end-to-end thật |
| 4 | Frontend | 🟡 code xong, chờ test trên iOS Safari thật |
| 5 | Pitch auto-detect | ⬜ |
| 6 | Consent gate | ⬜ |

## Cấu trúc

```
modal_app/
├── app.py          # Modal App, Volumes, Dict, images — nguồn duy nhất cho phần stateful
├── api.py          # FastAPI ASGI app: /health, /submit, /status, /download
├── audio_utils.py  # chunk theo silence, crossfade, validate — numpy thuần, test được
├── separation.py   # tách stem trên GPU: Separator (@app.cls) — port từ tachnhac
├── conversion.py   # Seed-VC trên GPU: VoiceConverter (@app.cls)
├── mixing.py       # ffmpeg: mix vocal + nhạc nền, encode mp3
├── pipeline.py     # orchestration: spawn + nối các bước, cập nhật job state
├── storage.py      # file trên Volume + cron dọn rác
├── jobs.py         # state machine của job, lưu trong modal.Dict
├── ratelimit.py    # 5 job/giờ mỗi client, khoá là hash của địa chỉ
├── deploy.py       # target deploy duy nhất — import mọi module để đăng ký
└── verify.py       # acceptance Phase 2 chạy trên hạ tầng thật (không cần GPU)
web/            # Next.js 15 trên Vercel — xem web/README.md
tests/          # chạy bằng pytest, không cần Modal credentials và không cần GPU
.github/workflows/
├── ci.yml            # python: ruff + pytest · web: eslint + prettier + tsc + build
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

Frontend chạy riêng, xem [`web/README.md`](web/README.md):

```bash
cd web && npm install && npm run dev
```

## Deploy

CI/CD lo phần này: push lên `main` chạm `modal_app/**` → workflow `Deploy Modal`
chạy `modal deploy -m modal_app.deploy`.

> Deploy qua `modal_app.deploy`, **không** phải `modal_app.api`. Modal chỉ publish
> những function mà module định nghĩa chúng đã được import; `api.py` không import
> `conversion.py` hay `storage.py`, nên deploy thẳng vào `api` sẽ lặng lẽ thiếu
> cron cleanup và cả class GPU. `deploy.py` import hết một lượt. Container thì vẫn
> chỉ import module chứa function của nó, nên `api_image` không phải kéo theo
> torch/numpy.

Cần hai secret trong repo settings (Settings → Secrets and variables → Actions):

- `MODAL_TOKEN_ID`
- `MODAL_TOKEN_SECRET`

Lấy bằng `modal token new` rồi đọc `~/.modal.toml`. Deploy tay:

```bash
modal deploy -m modal_app.deploy
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

Ngoài ra `huggingface.co` bị chặn ở môi trường viết code, nên tên file checkpoint
singing chưa được xác nhận là resolve được — nó là thứ mà chính `inference.py` của
upstream request ở commit đã pin, không phải thứ ta tự đặt.

## Phase 2 — Storage & job state

`storage.py` giữ file người dùng trên Volume `vc-data`, layout đúng như plan:
`/data/{job_id}/{input,reference,vocal,instrumental,converted,output}.{wav,mp3}`.
`jobs.py` giữ trạng thái job trong `modal.Dict` `vc-jobs`:

```
queued → separating → converting → mixing → done
                                         ↘ failed
```

Chuyển trạng thái được phép **nhảy tới** nhưng không được lùi — nhánh `speech`
không có separation và mixing nên chạy `queued → converting → done` qua đúng máy
trạng thái đó. `progress` không bao giờ giảm.

Bốn chỗ đáng chú ý:

1. **`job_id` và tên file được validate, không phải nội suy.** `job_id` đi thẳng
   từ path segment của URL (`/status/{id}`, `/download/{id}`) xuống filesystem, nên
   `..` phải là *không thể*, không phải *khó xảy ra*. Job id là `uuid4().hex` — 32
   ký tự hex, không có gì trong đó có thể là dấu phân cách đường dẫn.
2. **Ghi file là atomic** (`.part` rồi rename). Container khác không bao giờ đọc
   phải một `output.mp3` mới ghi được một nửa.
3. **Tuổi job tính từ lần ghi cuối, không phải lúc tạo.** Pipeline đang ghi stem thì
   tự giữ thư mục của nó sống, nên cron không thể xoá mất file mà job sắp đọc.
4. **`cleanup_expired()` trả về list id, không phải `int` như chữ ký trong plan.**
   Cron cần chính các id đó để xoá record tương ứng trong Dict; `len()` cho ra con
   số plan yêu cầu. Cron cũng xoá cả record mồ côi (job fail trước khi kịp ghi file
   nào) theo `created_at`.

`mode` ở tầng job là `song`/`speech`, khác `mode` của `VoiceConverter` là
`singing`/`speech` — `jobs.CONVERSION_MODE` map giữa hai cái, Phase 3 dùng nó chứ
đừng truyền thẳng `"song"` xuống GPU.

### Acceptance Phase 2

Unit test phủ logic; hai thứ chỉ hạ tầng thật mới trả lời được — Volume ghi ở
container này đọc được ở container kia, và `modal.Dict` có thực sự cư xử như
mapping mà `jobs.py` giả định (`in`, `.get`, `.keys()`, `del`) — nằm trong:

```bash
modal run -m modal_app.verify
```

Chạy trên CPU, tốn vài giây GPU-free. In ra từng check kèm `ok`/`FAIL`:

- `round_trip_is_byte_exact` — ghi rồi `data_vol.reload()` rồi đọc lại, đúng bytes
- `expired_job_removed` / `fresh_job_kept` — cron xoá đúng job quá hạn, giữ job mới
- `record_is_queued` / `progress_advances_in_order` / `done_is_terminal`
- `expiry_finds_the_record` / `record_is_removable`

## Phase 3 — Pipeline hoàn chỉnh

```
song:    input ──► separate ──► vocal ──► convert ──► mix ──► output.mp3
                        └────► instrumental ───────────┘
speech:  input ─────────────────────────► convert ──► encode ──► output.mp3
```

`pipeline.py` chạy trên image CPU nhỏ, tự nó không xử lý audio: separation và
conversion nằm ở container GPU riêng, gọi bằng `.remote()` nên hàm điều phối
không giữ GPU trong lúc chờ. `/submit` dùng `.spawn()` và trả về ngay.

### Separation — port từ [`chamaya00/tachnhac`](https://github.com/chamaya00/tachnhac)

Model, tên checkpoint, base image CUDA và đoạn chuẩn hoá tên file đầu ra bê
nguyên từ `modal_app.py` của app đó. Chỉ phần I/O đổi: app cũ đọc/ghi thẳng
Volume của nó và phục vụ stem cho trình duyệt, ở đây nhận bytes trả bytes để
`pipeline.py` giữ quyền quản lý storage.

| Model | Stem | Ghi chú |
|---|---|---|
| `roformer` (mặc định) | Vocals + Instrumental | BS-Roformer, chất lượng vocal tốt nhất |
| `htdemucs` | Vocals + Drums + Bass + Other | nhanh hơn ~4x; 3 stem còn lại cộng lại thành instrumental |

Hai chi tiết mang theo từ app cũ trông như phụ nhưng không phải:

- image là `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`, **không** phải
  `debian_slim` — `onnxruntime-gpu` cần `libcudnn.so.9`, thiếu là nó lặng lẽ tụt
  về CPU và BS-Roformer chậm hàng chục lần;
- `clang` + `build-essential` vì demucs kéo theo `diffq`, gói này không có wheel.

Khác app cũ một chỗ: stem ghi ra **WAV chứ không phải MP3**. Stem ở đây là đầu
vào của Seed-VC và của bản mix cuối, không có lý do nhét một thế hệ lossy vào
giữa.

```bash
modal run -m modal_app.separation::separate_files --source song.mp3
```

### Mixing

`mixing.py` gọi ffmpeg, không dùng thư viện audio Python — đúng như plan. Ba thứ
quyết định bản mix nghe đúng hay sai:

- `amix=normalize=0` — mặc định `normalize=1` chia biên độ cho số input, mix hai
  đường bị tụt 6 dB không vì lý do gì;
- `duration=longest` — vocal sau chuyển đổi không bao giờ dài đúng bằng
  instrumental, cắt theo đường ngắn hơn là cụt đuôi bài;
- `loudnorm=I=-14:TP=-1.0` ở cuối, chuẩn streaming.

### API

```
POST /submit          multipart: input, reference, mode, consent, params  →  {job_id}
GET  /status/{id}                                    →  {status, progress, error, ...}
GET  /download/{id}                                  →  audio/mpeg
```

```bash
BASE=https://<workspace>--voice-convert-api.modal.run

curl -sS -X POST "$BASE/submit" \
  -F "input=@song.mp3" -F "reference=@voice.wav" \
  -F "mode=song" -F "consent=true" -F "semitone_shift=0"
# {"job_id":"...","status":"queued","mode":"song"}

curl -sS "$BASE/status/$JOB_ID"
curl -sS -o output.mp3 "$BASE/download/$JOB_ID"
```

`/download` trả `409` khi job chưa xong (kèm status hiện tại) hoặc đã fail (kèm
message), `410` khi record còn nhưng file đã bị cron dọn — không bao giờ trả một
file rỗng.

### Sáu chỗ khác plan, cố ý

1. **Hai hàm pipeline, không phải một.** `run_song_pipeline` và
   `run_speech_pipeline`. Nhánh `speech` chạy `queued → converting → done` đúng
   như docstring của `jobs.py`: không separation, không mix, bước encode mp3 nằm
   trong `converting` vì nó chỉ tốn một giây.
2. **`Separator` là `@app.cls`, không phải `@app.function`.** Cùng lý do với
   `VoiceConverter`: load checkpoint một lần trong `@modal.enter()` rồi giữ
   container ấm 5 phút.
3. **`api_image` giờ có ffmpeg và numpy.** Nó chạy cả web endpoint, cron cleanup
   lẫn pipeline. Vẫn không có torch — phần nặng ở container khác.
4. **Consent gate làm luôn ở đây** (Phase 6 mục 1). Checkbox mà chỉ frontend tự
   kiểm thì không phải cổng chặn: `/submit` từ chối khi thiếu `consent=true`.
   Kèm luôn mục 2 — mọi file output đóng dấu `-metadata comment=AI-generated…`,
   vì `mixing.py` là chỗ duy nhất sinh ra bytes output.
5. **CORS mở sẵn.** Bài 3 phút là 4–7 MB, vượt giới hạn body 4.5 MB của Vercel,
   nên client phải POST thẳng lên Modal (plan §6). Giới hạn origin là chuyện cấu
   hình lúc deploy, chưa hardcode domain nào.
6. **`semitone_shift` lấy từ request.** Phase 5 mới tự tính từ vocal stem;
   `pipeline.py` đã để sẵn đúng một chỗ nhận giá trị đó và truyền chung cho cả
   bài — không bao giờ tính lại theo từng chunk.

### Còn phải verify bằng hạ tầng thật

`tests/` phủ phần chạy được không cần GPU: clamp tham số, đặt tên stem, filter
graph ffmpeg (chạy ffmpeg thật trong CI), và toàn bộ validate/status code của
API qua `TestClient`. Các mục acceptance còn lại của Phase 3 cần deploy thật:

- [ ] end-to-end: bài 3 phút + giọng mẫu 15s → ra file cover hoàn chỉnh
- [ ] nhạc nền không méo, vocal không chìm cũng không chói
- [ ] job fail giữa chừng → status `failed` có message, không treo vô hạn
- [ ] `/submit` trả về trong < 2 giây với file 7 MB
- [x] rate limit theo IP (plan §9) — làm ở Phase 4, xem `modal_app/ratelimit.py`

Validate độ dài file (reference 5–30s, nguồn ≤ 15 phút) nằm ở container GPU chứ
không ở `/submit`: chỉ chỗ đó mới decode được audio. Nghĩa là file sai độ dài
báo lỗi qua `status.error` sau vài giây, không phải ngay lúc submit.

---

## Phase 4 — Frontend

Next.js 15 App Router trong [`web/`](web/), deploy trên Vercel với Root
Directory là `web`. Một trang duy nhất, đi đúng thứ tự plan §6:

```
kiểu nội dung → file nguồn → giọng mẫu → tinh chỉnh → đồng thuận
      → convert → tiến trình → nghe thử → tải về
```

Không routing, không state global: mọi bước hiện cùng lúc để trên màn hình điện
thoại không có bước nào nấp sau nút "tiếp theo", và lúc chạy thì thẻ tiến trình
thay chỗ form chứ không chuyển trang.

### Đường đi của request — chỗ khác plan rõ nhất

```
trình duyệt ──POST /submit────────────────► Modal      (4–60 MB)
            ──GET  /api/status/{id}──► Vercel ──► Modal (vài trăm byte)
            ──GET  /download/{id}─────────────► Modal
            ──GET  /api/config───────► Vercel
```

Plan §6 nói hai điều cùng lúc: URL Modal phải nằm trong env var chứ không
hardcode vào client bundle, **và** file lớn phải bỏ qua Vercel mà POST thẳng lên
Modal. Muốn POST thẳng thì trình duyệt buộc phải biết URL đó. Lối ra là phục vụ
URL **lúc chạy** qua `/api/config` thay vì `NEXT_PUBLIC_` (vốn bị nhúng vào
bundle lúc build) — vẫn là env var, và tiện thêm: một build dùng được cho mọi
môi trường.

### Bốn chỗ đáng chú ý

1. **Upload dùng `XMLHttpRequest`, không phải `fetch`.** Lý do duy nhất: `fetch`
   tới giờ vẫn không báo được tiến trình upload, mà file 7 MB trên mạng di động
   không có phản hồi thì trông như treo.
2. **Thanh tiến trình có hai pha.** Upload là pha riêng vì nó phụ thuộc đường
   truyền của người dùng; thanh chỉ bắt đầu khi server đã nhận file thì đứng im
   cả phút (acceptance: không đứng im quá 20 giây).
3. **Poll fail ≠ job fail.** Mạng rớt giữa chừng thì đếm và bỏ qua tới 5 lần
   liên tiếp — job đang chạy trên GPU 3 phút không đáng chết vì một request
   hỏng. Nhịp poll: 2s, backoff sau 60s, chặn trên 10s, bỏ cuộc ở 15 phút, mọi
   thứ treo trên một `AbortController` mà lúc unmount sẽ cắt.
4. **Ghi âm viết theo iOS Safari.** `audio/mp4` chứ không phải webm, không
   truyền `timeslice` (iOS chỉ trả data lúc `stop()`), độ dài đo bằng đồng hồ
   (Safari trả `duration = Infinity` cho blob của MediaRecorder), `AudioContext`
   `resume()` sau cử chỉ người dùng. Không có `MediaRecorder` thì tab tự ẩn.

### Rate limit — plan §9, nợ từ Phase 3

`modal_app/ratelimit.py`: 5 job/giờ cho mỗi client, cửa sổ trượt. Nằm ở backend
chứ không ở route handler của Vercel, vì upload đi thẳng lên Modal nên frontend
không nằm trên đường đó.

Hai chi tiết cố ý:

- **Không lưu địa chỉ IP.** Khoá là `sha256(salt:address)` cắt 32 ký tự. Plan §8
  mục 5 muốn audit trail là job id + timestamp và rõ ràng *không* phải nội dung
  người dùng; IP gần nội dung hơn là gần job id.
- **Request bị từ chối không tốn lượt.** `/submit` đọc quota trước (read-only),
  chỉ trừ lượt ngay trước khi spawn. Ngược lại thì gõ sai một tham số cũng mất
  một lượt, và client hammer endpoint sẽ tự đẩy lượt kế tiếp ra xa mãi.

`ALLOWED_ORIGINS` (phẩy ngăn cách) trên Modal secret khoá CORS lại khi đã có
domain Vercel; chưa đặt thì vẫn mở `*`.

### Còn phải verify bằng thiết bị thật

CI chạy eslint + prettier + tsc + `next build`. Các mục acceptance của Phase 4
cần máy thật và một deployment thật:

- [ ] chạy trọn vẹn trên Safari iOS
- [ ] ghi âm hoạt động trên iOS Safari
- [ ] file 7 MB upload thành công
- [ ] progress bar không đứng im quá 20 giây
- [ ] chưa có preset giọng nào (`web/public/presets/index.json` rỗng) — cần 4–6
      clip tự thu hoặc có licence rõ ràng, plan §8 mục 4
