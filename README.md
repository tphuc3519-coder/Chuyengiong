# Chuyengiong

Voice conversion web app: đưa vào một bài hát (hoặc một bản hát đã tách sẵn,
hoặc đoạn thoại, hoặc một đoạn văn bản) + một giọng mẫu, nhận về bản đã đổi sang
giọng đó — nhạc nền giữ nguyên, hoặc thay hẳn bằng một beat khác.

Kế hoạch chi tiết theo từng phase: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Scaffold + deploy pipeline | 🟡 code xong, chờ verify deploy thật |
| 1 | Seed-VC + chunking | 🟡 code xong, chờ chạy thật trên GPU |
| 2 | Storage + job state | 🟡 code xong, chờ chạy `modal run -m modal_app.verify` |
| 3 | Nối pipeline hoàn chỉnh | 🟡 code xong, chờ chạy end-to-end thật |
| 4 | Frontend | 🟡 code xong, chờ test trên iOS Safari thật |
| 5 | Pitch auto-detect | 🟡 code xong, acceptance §7 pass bằng tone tổng hợp |
| 6 | Consent gate & an toàn | 🟡 code xong, chờ verify watermark trên hạ tầng thật |
| 7 | Audio watermark (§8 "cân nhắc thêm") | 🟡 code xong, chờ chạy checkpoint thật |
| 8 | Text to speech theo giọng mẫu (có tiếng Nhật) | 🟡 code xong, chờ nghe thật trên container |
| 9 | Đọc có ngữ điệu — ngắt nghỉ, lên xuống, cảm xúc | 🟡 code xong, chờ nghe thật trên container |
| 10 | Giọng trong hơn, bớt "AI" — làm sạch giọng mẫu, hậu kỳ, train giọng riêng, mode không tách nhạc | 🟡 code xong, chờ nghe thật trên GPU |
| 11 | Đổi beat — đo BPM/tông, khớp beat có sẵn hoặc sinh beat mới | 🟡 code xong, chờ nghe thật trên GPU |
| 12 | Phối lại bài gốc — đọc vòng hợp âm rồi dựng lại bằng tiếng tự tổng hợp | 🔴 **đã gỡ** — không đạt chất lượng, xem 13.2 |
| 13 | Mode "Đổi beat" đứng riêng — giữ nguyên giọng gốc, không cần giọng mẫu | 🟡 code xong, chờ chạy thật |
| 13.1 | Sửa deploy đỏ: image của beatgen chặn cả ba phase | 🟢 deploy xanh lại, sinh beat tắt sau cờ |
| 13.2 | Nghe thật lần đầu: gỡ "Phối lại", sửa tempo sai 3:2 | 🟢 đo trên nhạc thật |
| 13.3 | Sửa đúng dòng làm chết deploy, bật lại "máy làm beat" | 🟡 pip resolve xanh, chờ build thật |

## Cấu trúc

```
modal_app/
├── app.py          # Modal App, Volumes, Dict, images — nguồn duy nhất cho phần stateful
├── api.py          # FastAPI ASGI app: /health, /submit, /status, /download
├── audio_utils.py  # chunk theo silence, crossfade, validate — numpy thuần, test được
├── separation.py   # tách stem trên GPU: Separator (@app.cls) — port từ tachnhac
├── conversion.py   # Seed-VC trên GPU: VoiceConverter (@app.cls)
├── tts.py          # văn bản → wav trên CPU: Synthesizer (MMS) + KokoroSynthesizer (tiếng Nhật)
├── prosody.py      # đọc thế nào: ngắt nghỉ theo dấu câu, ngữ điệu, cảm xúc — Python thuần
├── reference.py    # làm sạch giọng mẫu trước khi nó thành timbre — numpy thuần, test được
├── enhance.py      # chuỗi lọc "độ trong" cho giọng đã convert — ffmpeg thuần
├── analysis.py     # đo BPM, vị trí phách và tông của một bản nhạc — numpy thuần
├── beats.py        # cắt/dịch/kéo/lặp một beat cho khớp bài — ffmpeg thuần
├── beatgen.py      # sinh beat mới từ mô tả (Stable Audio Open) trên GPU
├── training.py     # fine-tune Seed-VC cho một giọng riêng trên GPU (công cụ vận hành)
├── voices.py       # giọng đã train nằm ở đâu, tên nào hợp lệ — không import gì cả
├── mixing.py       # ffmpeg: mix vocal + nhạc nền, encode mp3
├── pipeline.py     # orchestration: spawn + nối các bước, cập nhật job state
├── storage.py      # file trên Volume + cron dọn rác
├── jobs.py         # state machine của job, lưu trong modal.Dict
├── pitch.py        # YIN + gợi ý dịch cao độ — port từ thanh-pitch
├── audit.py        # nhật ký job: mã job + thời điểm, không bao giờ có audio
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
`/data/{job_id}/{input,reference,vocal,instrumental,spoken,converted,output}.{wav,mp3}`,
cộng thêm `input.txt` cho nhánh `tts`.
`jobs.py` giữ trạng thái job trong `modal.Dict` `vc-jobs`:

```
queued → separating   ⎫
         synthesizing ⎬→ converting → mixing → done
                      ⎭                     ↘ failed
```

Chuyển trạng thái được phép **nhảy tới** nhưng không được lùi — nhánh `speech`
không có separation và mixing nên chạy `queued → converting → done` qua đúng máy
trạng thái đó, còn `tts` chạy `queued → synthesizing → converting → done`.
`separating` và `synthesizing` là bước chuẩn bị của hai nhánh khác nhau, không
job nào đi qua cả hai. `progress` không bao giờ giảm.

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
tts:     input.txt ──► synthesize ──────► convert ──► encode ──► output.mp3
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
POST /submit    multipart: input | text, reference, mode, consent, params  →  {job_id}
GET  /status/{id}                                    →  {status, progress, error, ...}
GET  /download/{id}                                  →  audio/mpeg
```

```bash
BASE=https://<workspace>--voice-convert-api.modal.run

curl -sS -X POST "$BASE/submit" \
  -F "input=@song.mp3" -F "reference=@voice.wav" \
  -F "mode=song" -F "consent=true" -F "semitone_shift=0"
# {"job_id":"...","status":"queued","mode":"song"}

# Mode `tts` gửi chữ thay cho file — `input` khi đó không cần có (Phase 8):
curl -sS -X POST "$BASE/submit" \
  -F "text=Xin chào, đây là giọng của tôi." -F "reference=@voice.wav" \
  -F "mode=tts" -F "language=vie" -F "consent=true"

curl -sS "$BASE/status/$JOB_ID"
curl -sS -o output.mp3 "$BASE/download/$JOB_ID"
```

`/download` trả `409` khi job chưa xong (kèm status hiện tại) hoặc đã fail (kèm
message), `410` khi record còn nhưng file đã bị cron dọn — không bao giờ trả một
file rỗng.

### Sáu chỗ khác plan, cố ý

1. **Hai hàm pipeline, không phải một** (Phase 8 thêm cái thứ ba,
   `run_tts_pipeline`). `run_song_pipeline` và `run_speech_pipeline`. Nhánh
   `speech` chạy `queued → converting → done` đúng như docstring của `jobs.py`:
   không separation, không mix, bước encode mp3 nằm trong `converting` vì nó chỉ
   tốn một giây.
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

### Cấu hình lúc deploy

Hai biến, cả hai đều tuỳ chọn, đi từ máy chạy `modal deploy` vào container qua
`app.config_secret()`:

| Biến | Đặt ở đâu | Không đặt thì |
|---|---|---|
| `ALLOWED_ORIGINS` | GitHub **variable** | CORS mở `*` |
| `RATE_LIMIT_SALT` | GitHub **secret** | salt là hằng số cố định |

`Secret.from_dict` chứ không phải `Secret.from_name`: secret có tên phải tồn
tại trước khi tra được, nên một bản clone mới deploy lần đầu sẽ fail vì thứ vốn
hoàn toàn tuỳ chọn. Rỗng nghĩa là "chưa cấu hình", và cả hai chỗ đọc đều hiểu
như vậy.

`ALLOWED_ORIGINS` (phẩy ngăn cách) khoá CORS lại khi đã có domain Vercel.
`RATE_LIMIT_SALT` làm hash của rate limit không đoán được — không có nó thì key
vẫn không đảo ngược được, nhưng ai cầm được Dict vẫn thử được xem một địa chỉ
cụ thể đã submit hay chưa.

### Còn phải verify bằng thiết bị thật

CI chạy eslint + prettier + tsc + `next build`. Các mục acceptance của Phase 4
cần máy thật và một deployment thật:

- [ ] chạy trọn vẹn trên Safari iOS
- [ ] ghi âm hoạt động trên iOS Safari
- [ ] file 7 MB upload thành công
- [ ] progress bar không đứng im quá 20 giây
- [ ] chưa có preset giọng nào (`web/public/presets/index.json` rỗng) — cần 4–6
      clip tự thu hoặc có licence rõ ràng, plan §8 mục 4

---

## Phase 5 — Pitch auto-detect

### Port từ [`chamaya00/thanh-pitch`](https://github.com/chamaya00/thanh-pitch)

`modal_app/pitch.py` là hàm `detectPitch` trong `index.html` của repo đó, giữ
nguyên thuật toán và **nguyên mọi hằng số đã tune**: ngưỡng tuyệt đối 0.13, gate
RMS 0.006, gate clarity 0.72, trần fallback 0.55, quy tắc "cực tiểu cục bộ đầu
tiên dưới ngưỡng", nội suy parabol, và hai dải `RANGES.speak` 60–500 Hz /
`RANGES.sing` 60–1200 Hz. Không có con số nào ở đây là đoán lại từ đầu.

Hai chỗ đổi, đều vì chạy trên cả file thay vì từng frame analyser:

1. **Chạy theo lô.** Bản trình duyệt duyệt vòng lặp thẳng trên 4096 mẫu mỗi
   animation frame. Bài 15 phút là ~18000 frame, nên hàm sai khác được tính qua
   FFT theo block 2000 frame — cùng công thức (`d(τ) = Σx² + Σx²ₜ − 2Σxxₜ`),
   chỉ là phần tương quan giao cho FFT.
2. **Bỏ bộ lọc trung vị 3 frame.** Nó tồn tại để kim trên màn hình không nhảy
   quãng tám. Lấy trung vị trên *mọi* frame hữu thanh của cả file — thứ mà gợi ý
   dịch cao độ vốn cần — làm việc đó tốt hơn.

Tốc độ: bài 8,5 phút mất ~4s trên CPU container. Không cần GPU, không cần
librosa (`api_image` vẫn chỉ có ffmpeg + numpy, `/status` vẫn khởi động trong
vài giây).

### `None` không phải `0`

Đây là chỗ cả tính năng dựa vào. `/submit` **bỏ trống** `semitone_shift` nghĩa
là "tự đo"; gửi `0` nghĩa là "giữ nguyên cao độ". `clean_params` giữ `None`
nguyên vẹn thay vì `or 0`, và `pipeline._resolve_shift` là chỗ duy nhất biến nó
thành số.

Đo **sau separation, trước khi chunk**, trên vocal stem chứ không phải bản mix —
đo bản mix là đo cả nhạc nền lẫn người hát. Nhánh `speech` không có separation
nên đo thẳng file nguồn. Một giá trị cho cả bài, không bao giờ tính lại theo
từng chunk: đó là bug mà plan §10 gọi là phổ biến nhất của loại app này.

### Ba chỗ khác plan, cố ý

1. **Slider không mở sẵn bằng giá trị gợi ý — vì chưa có giá trị nào.** Plan §7
   nói hiển thị gợi ý làm mặc định trên slider, nhưng cũng nói gợi ý được tính
   trong `pipeline.py` sau separation. Hai điều đó không xảy ra cùng lúc: lúc
   người dùng nhìn slider thì chưa tách nhạc. Nên "Tự động" là một *chế độ* bật
   sẵn, và giá trị đo được hiện ra khi job chạy tới đó; tắt "Tự động" thì slider
   mở đúng ở con số lần trước đã áp dụng.
2. **`jobs.public()` lộ đúng một tham số.** `semitone_shift` là thứ duy nhất
   trong `params` đi ra ngoài, vì với auto-detect thì client không chọn nó —
   `/status` là đường duy nhất để thấy đã áp dụng bao nhiêu.
3. **Hai bên đo bằng cùng một dải F0.** Dải chỉ giới hạn khoảng tìm chu kỳ chứ
   không lọc kết quả, nên tiếng cao hơn trần sẽ bị gập xuống quãng tám dưới
   (700 Hz đọc bằng dải `speech` ra 350). Đo hai bên bằng hai dải khác nhau sẽ
   lệch một cách khó thấy.

### Acceptance §7

Cả hai mục pass bằng test tổng hợp (`tests/test_pitch.py`) — tone có tần số biết
trước là một trong số ít thứ trong pipeline audio có đáp án đúng kiểm được:

- [x] gợi ý cho cặp nam→nữ ra khoảng +10 đến +14 (130 Hz → 245 Hz cho +11)
- [x] bài có intro nhạc dài không làm lệch kết quả — 45s im lặng + 20s hát ở
      220 Hz vẫn ra 220,0 Hz
- [ ] còn phải nghe bằng tai trên giọng thật: tone tổng hợp không có vibrato,
      không có luyến, và không có tiếng nhạc rò sang vocal stem

---

## Phase 6 — Consent gate & an toàn

Plan §8 nói thẳng: rủi ro lớn nhất của app loại này là pháp lý chứ không phải
kỹ thuật. Năm mục bắt buộc, và chỗ mỗi mục thật sự được thi hành:

| Mục plan §8 | Ở đâu | Ghi chú |
|---|---|---|
| 1 · checkbox không tick sẵn | `web/app/components/ConsentGate.tsx` + `api.submit` | Làm từ Phase 3 — cổng thật nằm ở `/submit`, không ở trình duyệt |
| 2 · metadata `AI-generated` | `modal_app/mixing.py` | Làm từ Phase 3 — `mixing` là nơi duy nhất sinh bytes output |
| 3 · terms of service | `web/app/terms/page.tsx` | Link từ ô đồng thuận và footer |
| 4 · không giọng người nổi tiếng | `web/public/presets/` + `tests/test_presets.py` | Luật thành test, không phải lời hứa |
| 5 · audit trail | `modal_app/audit.py` | Mã job + thời điểm; audio không bao giờ vào log |

### Audit trail — `audit.py`

Một dòng JSON cho mỗi sự kiện, ra stdout (Modal gom log), tiền tố `[audit]`:

```
[audit] {"ts":"2026-08-26T09:14:02Z","event":"submit","job":"a1b2…","mode":"song",
         "client":"9f3c…","consent":true,"input_bytes":5242880,"reference_bytes":180224}
[audit] {"ts":"2026-08-26T09:15:37Z","event":"done","job":"a1b2…","mode":"song",
         "seconds":94.3,"shift":11,"steps":50,"model":"htdemucs","reason":null}
```

Năm sự kiện: `submit` (`api.submit`, sau khi job đã start), `done`/`failed`
(`pipeline._finished`), `download` (`api.download`, chỉ khi thật sự trả file), và
`expire` (cron `storage.cleanup`, không có mã job — nó nói cả lô đã bị xoá).
Tra một khiếu nại: `grep '[audit]' | cut -d' ' -f2- | jq 'select(.job=="…")'`.

Nửa sau của yêu cầu — *không* log nội dung — được ép bằng cấu trúc chứ không
bằng trí nhớ:

1. **Allowlist tên trường.** `audit.FIELDS` là danh sách đầy đủ những gì được
   ghi. Trường lạ bị bỏ và **tên** của nó (tên do code đặt, không phải dữ liệu
   người dùng) hiện trong `dropped` — người viết thấy ngay, thay vì để tên file
   nằm trong log ba tháng rồi mới phát hiện.
2. **Chỉ scalar sống sót.** `bytes` không render được ở đâu cả, nên đưa cả file
   wav vào `record()` cũng không ra được audio. Chuỗi bị gộp xuống một dòng và
   cắt ở 64 ký tự — không có giá trị nào tự bịa ra được một dòng log thứ hai.
3. **`record()` không bao giờ raise.** Mọi chỗ gọi nó đều nằm trên đường request
   hoặc trong `except` của pipeline; một dòng log hỏng định dạng mà làm chết job
   thì tệ hơn hẳn việc mất dòng log đó.

Ba thứ cố ý không có: địa chỉ IP (chỉ có hash từ `ratelimit.client_key`), tên
file người dùng gửi lên, và **message** của exception — `failed` ghi tên *class*
thôi, vì message có thể trích nguyên câu ffmpeg nói về file của người dùng.
`/status` vẫn mang message đầy đủ tới đúng một người có quyền đọc nó.

### Presets — luật §8 mục 4 viết thành test

`tests/test_presets.py` chạy trong CI Python và chặn hai chiều: mỗi entry trong
`web/public/presets/index.json` phải có `license` nói rõ nguồn, và **không được
có file audio nào nằm trong thư mục mà manifest không nhắc tới**. Chiều thứ hai
mới là chiều bắt được lỗi thật — một clip thả tay vào thư mục rồi quên khai.

Manifest ship rỗng vẫn pass: không có giọng nào thì UI ẩn hẳn hàng "Giọng có
sẵn". Đó là trạng thái hợp lệ, tốt hơn là ship một giọng không ai chỉ được nguồn.

### Terms — `web/app/terms/page.tsx`

Ngắn, và chỉ viết những gì code thật sự làm: cam kết ở ô đồng thuận, cấm mạo
danh, metadata AI trên file tải về, TTL 6 giờ, nhật ký chỉ có mã job, không có
giọng người nổi tiếng, 5 lượt/giờ, nguồn ≤ 15 phút, giọng mẫu 5–30 giây. Một
điều khoản không tương ứng với hành vi nào của hệ thống thì chỉ là trang trí.

Địa chỉ khiếu nại đọc từ `CONTACT_EMAIL` lúc chạy (`lib/server.ts`), để trống thì
trang nói thẳng là chưa công bố — không in link chết. Vì thế `/terms` là route
động chứ không prerender.

### Còn lại trước khi mở public

- [x] audio watermark — làm ở Phase 7 bên dưới, `modal_app/watermark.py`
- [ ] điền `CONTACT_EMAIL` trên Vercel và rà lại ngày "cập nhật" của trang terms
- [ ] nếu thêm preset: 4–6 clip tự thu hoặc có licence rõ ràng
      (`web/public/presets/README.md`)

---

## Phase 7 — Audio watermark

Plan §8 xếp mục này vào "cân nhắc thêm": không bắt buộc ở MVP, nhưng **nên thêm
trước khi mở public**. Đây là mục đó. [AudioSeal](https://github.com/facebookresearch/audioseal)
của Meta (MIT), checkpoint `audioseal_wm_16bits`, nhúng một message 16 bit không
nghe thấy được và sống sót qua encode mp3.

Nó bù đúng chỗ hai tính năng an toàn kia hụt:

| | Chứng minh được gì | Hụt ở đâu |
|---|---|---|
| metadata `AI-generated` | file này do AI sinh | mất sạch sau lần re-encode đầu tiên |
| audit trail | *ta* đã chạy job nào, lúc nào | không nói gì về một file người khác cầm tới |
| **watermark** | **file này ra từ deployment này** | cần model để đọc |

Ba cái cộng lại mới trả lời được câu hỏi thật sự của một khiếu nại: "file này có
phải các anh làm không". Tra bằng:

```bash
modal run -m modal_app.watermark --path suspect.mp3
```

```json
{"watermarked": true, "ours": true, "probability": 0.9861, "matching_bits": 16,
 "message": 51113, "expected_message": 51113, "seconds": 184.2}
```

### Model chỉ chạy 16kHz — và không tự resample nữa

Đây là rủi ro kiểu "risk #1" của plan §10, lần này ở AudioSeal thay vì Seed-VC.
Bản 0.2 **bỏ** resample nội bộ; đọc thẳng trong `models.py` của repo:

> Starting from AudioSeal 0.2+, audio is not resampled internally to 16kHz or
> some predefined sample rates. The user is responsible for providing the
> correct sample rate to the model.

Đưa thẳng bài 44.1kHz vào sẽ không lỗi — nó chỉ tạo ra watermark sai và bản dò
không tìm lại được. Nên `watermark.py` làm đúng việc bản <0.2 từng làm, nhưng
làm ở chỗ nhìn thấy được:

```
mix 44.1kHz stereo ──► view 16kHz mono ──► AudioSeal ──► wm 16kHz mono
        │                                                     │
        │                                              resample lên 44.1k
        ▼                                                     ▼
   cộng vào cả hai kênh ◄─────────────────────────────────────┘
```

Nhạc **không** bị downsample. Nó chỉ được cộng thêm một tín hiệu giới hạn dưới
8kHz. Bản dò cũng downmix về 16kHz mono nên tìm lại đúng cái đã cộng vào.

`resample()` đi qua ffmpeg (wav round trip) chứ không nội suy tuyến tính, và
**chuẩn hoá tín hiệu lên full scale trước khi round trip rồi trả lại thang sau**
— watermark nằm khoảng −45 dBFS, round trip 16 bit ở đúng thang của nó sẽ ném nó
xuống sát sàn lượng tử. Phép nhân/chia này tuyến tính tuyệt đối nên không mất gì.

### Ba chỗ khác plan, cố ý

1. **Chạy trong container CPU riêng, không phải trong `mixing.py`.** Plan nói
   "chỗ nhét là `mixing.py`" và về *thứ tự* thì đúng thế — nhưng AudioSeal cần
   torch, còn `mixing.py` cố tình không có dependency nào ngoài ffmpeg (đó là lý
   do container separation dùng lại được `sum_stems`). Nên `mix()`/`to_mp3()`
   nhận một **callable** `watermark`, gọi sau `loudnorm` và trước encode mp3.
   Model nằm sau callable đó, ở `Watermarker` — CPU 4 core, không GPU.

   Đổi lại: CI test được toàn bộ đường đi bằng một callable giả, không cần torch.

2. **Sau `loudnorm`, không phải trước.** `loudnorm` có gain và limiter; watermark
   cộng vào trước đó sẽ bị cả hai chỉnh lại thang. Vì thế encode mp3 tách thành
   một pass ffmpeg riêng — kể cả khi không watermark, để tắt `WATERMARK` không
   đẩy job sang một đường code khác với đường được test.

3. **Chunk 30s, ghép bằng fade tuyến tính.** Cùng lý do Phase 1 phải chunk: một
   forward pass cho bài 8 phút ngốn hàng GB trong stack SEANet. Khác Phase 1 ở
   chỗ fade: `audio_utils` dùng equal-power vì hai mép mối nối là cùng đoạn audio
   qua diffusion hai lần nên lệch pha; ở đây hai mép là watermark của **cùng**
   đoạn audio nên tương quan — equal-power sẽ *nhô* lên ở mối nối, tuyến tính
   mới giữ phẳng. Có test cho đúng tính chất đó.

### Message là id của deployment, không phải mã job

16 bit không chứa nổi một job id 128 bit. `WATERMARK_MESSAGE` chỉ tách "ra từ
deployment này" khỏi "ra từ chỗ khác" và khỏi dương tính giả — audit trail mới là
thứ map file về job. Giá trị chọn có bit trộn (không phải `0x0000`/`0xFFFF`), vì
file không watermark hay decode ra đúng hai mẫu đó.

### Bật/tắt, và tại sao hỏng là hỏng to

`WATERMARK` (env, `app.CONFIG_KEYS`) — **mặc định bật**, đặt `0`/`false`/`no`/`off`
để tắt. Không có trạng thái thứ ba: watermark bật mà lỗi thì **job fail**. Ship
một file *không* watermark trong khi log ghi là có thì tệ hơn hẳn không ship gì.

Cờ được chốt **lúc tạo job** (`clean_params`) chứ không lúc mix, nên `params` và
dòng audit `done` nói đúng cái đã làm với file đó, không phải cái config đang nói
một tuần sau. Nó không phải tuỳ chọn của client — client gửi lên cũng bị bỏ qua.

Watermark bài dài mất khoảng một phút CPU nằm gọn trong bước `mixing`, nên
`pipeline._watermark` đẩy progress lên 85 khi đi qua — plan §6 đòi thanh tiến
trình không đứng im quá 20 giây, và đây là chỗ duy nhất ranh giới của bước không
tự cho ra một mốc.

### Sửa kèm: mix đang bị bóp về mono

Test stereo của watermark lôi ra một lỗi có sẵn **từ Phase 3**: `amix` thương
lượng một channel layout chung cho mọi input và chọn cái *hẹp nhất*. Vocal từ
Seed-VC luôn mono (`encode_wav` ép thế), nên mọi bài hát ship ra đều bị fold
instrumental stereo xuống mono — mất hẳn stereo image của nhạc nền.

Sửa bằng `pan=stereo|c0=c0|c1=c0` trên vocal trước khi `amix`. Dùng `pan` chứ
không phải upmix `aformat` vì ffmpeg khi mono→stereo áp mức centre mix −3 dB
(đo được: hệ số 0.707), tức là sẽ lặng lẽ dìm vocal xuống trong lúc sửa layout.

Lưu ý khi nghe lại: cân bằng vocal/nhạc **đổi** so với bản cũ. Trước đây
instrumental bị downmix về mono (−3 dB công suất) rồi mới cộng vocal; giờ nó giữ
nguyên mức. Đây là cân bằng đúng, và slider `vocal_gain_db` vẫn ở đó.

### Còn phải verify bằng hạ tầng thật

CI chạy toàn bộ phần quanh model — cờ bật/tắt, message, cửa sổ, mối nối, resample,
layout kênh, và hợp đồng giữa `mixing` và callable — bằng ffmpeg thật và một
callable giả. Bản thân model thì chưa: `huggingface.co` không tới được từ máy
build, nên checkpoint chưa từng được nạp. API đã đọc từ source của
`audioseal` 0.2.0 (`models.py`, `loader.py`, model card) chứ không đoán theo README.

- [ ] `modal run -m modal_app.watermark --path <file đã convert>` → `ours: true`
- [ ] nghe kỹ: watermark phải **không nghe thấy** trên bài thật, nhất là đoạn
      nhạc dạo và đoạn im lặng
- [ ] file convert xong, tải về, re-encode lại một lần nữa → vẫn `watermarked: true`
- [ ] file **không** phải của ta → `ours: false` (kiểm tra dương tính giả)
- [ ] đo thời gian thật của bước này trên bài 3 phút và 8 phút
- [ ] checkpoint chỉ tải một lần: container thứ hai không tải lại (Volume chạy đúng)

## Phase 8 — Text to speech theo giọng mẫu

Mode thứ ba, `tts`: thay vì file nguồn thì gõ chữ, và nhận về bản đọc bằng
giọng mẫu.

```
tts:  input.txt ──► Synthesizer ──► spoken.wav ──► VoiceConverter ──► output.mp3
                    (MMS · 8 ngôn ngữ chữ Latin)
                    KokoroSynthesizer
                    (tiếng Nhật)
```

Điểm chính của thiết kế: **không có model cloning thứ hai**. Bước đọc chữ chỉ
tạo ra một bản thu bằng giọng tổng hợp sẵn có; cái làm nó thành giọng của người
dùng vẫn là Seed-VC ở nhánh `speech`, y nguyên code đang chạy. Nhờ vậy pitch
auto-detect, chuẩn hoá độ ồn, watermark, consent gate, TTL — tất cả là code cũ
chứ không phải bản sao thứ hai của chúng.

`tts.synthesize(language, ...)` là chỗ duy nhất chọn engine; `pipeline.py` chỉ
biết ngôn ngữ, không biết model nào đọc.

### Vì sao MMS-TTS chứ không phải một TTS zero-shot

`facebook/mms-tts-<iso639-3>` là checkpoint VITS chạy thẳng bằng `transformers`:
~145 MB, không cần vocoder rời, không cần phonemizer, và có tiếng Việt tử tế.
Một TTS zero-shot (XTTS, F5) sẽ làm lại đúng việc mà bước sau đã làm — clone
timbre — bằng một bộ weight nữa, một GPU nữa, và khả năng phủ ngôn ngữ kém hơn
Whisper-small mà Seed-VC đang dùng.

`Synthesizer` chạy **CPU**, không GPU: VITS đọc một câu nhanh hơn thời gian
thực trên vài core, nên GPU sẽ dành phần lớn thời gian để cold start. GPU trong
pipeline này thuộc về bước chuyển giọng.

### Ngôn ngữ: một cái gate chứ không phải một danh sách

MMS phủ ~1100 ngôn ngữ nhưng thứ ngoài chữ Latin phải romanise bằng `uroman`
trước, và checkpoint nhận chữ chưa romanise thì trả về **im lặng chứ không phải
lỗi**. Nên nhánh MMS của `LANGUAGES` chỉ liệt kê những thứ đọc được nguyên văn
(vie, eng, ind, fra, spa, deu, por, ita), backend từ chối ngôn ngữ không có
trong bảng, và `Synthesizer.load` còn kiểm tra `tokenizer.is_uroman` một lần
nữa. Thêm ngôn ngữ = thêm vào bảng rồi chạy smoke test, không phải hy vọng.

Ngôn ngữ sai cũng bị từ chối chứ không rơi về mặc định: đọc tiếng Việt bằng
checkpoint tiếng Anh cho ra một bản thu trôi chảy của thứ vô nghĩa, tệ hơn lỗi.

### Tiếng Nhật: engine riêng, và đây là số đo chứ không phải phỏng đoán

`facebook/mms-tts-jpn` là ngõ cụt, kiểm chứng được bằng chính `uroman` — thứ
MMS dùng cả lúc train lẫn lúc chạy:

```
今日はいい天気ですね。  ->  jinrihaiitianqidesune.     (đúng: kyou wa ii tenki desu ne)
私の名前は田中です。    ->  sinomingqianhatianzhongdesu. (đúng: watashi no namae wa tanaka desu)
```

Kanji ra **âm Hán ngữ**: 今日 thành "jinri", 天気 thành "tianqi", 田中 thành
"tianzhong"; truyền `lcode='jpn'` không đổi gì. Kana thì ổn, nhưng văn xuôi
tiếng Nhật khoảng một nửa là kanji. Đó chính là văn bản mà `mms-tts-jpn` học từ
đó — nên đưa romaji đúng vào cũng lệch phân phối, mà đưa output của uroman vào
thì lệch ngôn ngữ. Không có đường nào ra tiếng Nhật thật.

Nên tiếng Nhật đọc bằng **Kokoro** (`hexgrad/Kokoro-82M`, Apache-2.0, 82M
tham số, chạy CPU), với front end `misaki[ja]` — G2P có từ điển và phân tích
hình thái thật:

```
今日はいい天気ですね。      ->  kʲoː βa iː teŋkʲi desɨ ne.
私の名前は田中です。        ->  βatakɯɕi no namae βa tanaka desɨ.
よろしくお願いします。      ->  joɾoɕikɯ oneɡai ɕi masɨ.
```

Một engine thì gọn hơn. Hai engine là thứ ngôn ngữ này cần.

**Bẫy `unidic`.** `misaki[ja]` khai báo phụ thuộc `unidic` — package này *không*
chứa từ điển, nó chỉ là bộ tải về, mà `fugashi` lại ưu tiên nó hơn `unidic-lite`
khi có cả hai. Container chết ngay trong `Tagger()`:

```
param.cpp(69) [ifs] no such file or directory: .../unidic/dicdir/mecabrc
```

Nên image cài `unidic-lite` rồi `pip uninstall -y unidic`. Cách khác là chạy
`python -m unidic download` lúc build, tốn ~700 MB image cho những cách đọc mà
`unidic-lite` đã có sẵn.

**Bẫy `transformers` không pin.** `kokoro` khai báo `transformers` không kèm
version, mà `base_image` pin torch 2.4.0 và không có transformers — nên
resolver lấy bản mới nhất (5.x), thứ đòi torch ≥ 2.5. Nó **không** báo lỗi lúc
build, chỉ in một dòng vào log:

```
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.4.0
```

rồi chạy tiếp không có torch, và mọi class model của nó thành stub ném
`requires the PyTorch library` ở lần đầu có thứ gì dựng nó lên — tức là trong
`@modal.enter()`, ở request tiếng Nhật đầu tiên, rất lâu sau khi deploy đã
xanh. `TRANSFORMERS_SPEC` pin cả hai image vào 4.46.3, đúng bản `conversion.py`
đang dùng.

**Giới hạn ký tự theo ngôn ngữ.** Ký tự không phải đơn vị của lời nói: 2000 ký
tự tiếng Việt và 700 ký tự tiếng Nhật là cùng khoảng 2–3 phút audio. Giới hạn
đặt lên bản ghi chứ không đặt lên bàn phím, nên nó nằm trong `Language` cùng với
`segment_max_chars` (200 vs 80 — đo trên G2P thật thì 80 ký tự tiếng Nhật ra
~200 phoneme, còn Kokoro cắt cụt ở 510).

**Romaji cũng đọc được.** Không phải máy nào cũng có IME, nên gõ `konnichiwa`
phải ra được こんにちわ. Front end của Kokoro cho chữ Latin đi thẳng qua không
đổi, nên `tts.to_kana` chuyển romaji sang kana *trước* khi cắt câu (kana mới là
đơn vị của `segment_max_chars`). Romaji là chữ ghi âm nên đây là đổi chính tả
chứ không phải dịch — `kyou wa ii tenki desu ne.` ra đúng bộ phoneme
`kʲoː βa iː teŋkʲi desɨ ne.` mà `今日はいい天気ですね。` cho.

Một cái bẫy nhỏ trong đó: romaji kiểu wapuro viết ん trước nguyên âm là `nn`
hoặc `n'`, mà `jaconv` chỉ đọc dạng có dấu nháy — `konnichiwa` ra こん**い**ちわ,
thiếu một mora và thành từ khác. `_ROMAJI_DOUBLE_N` viết lại `nn` thành `n'n`
trước khi chuyển.

**Cắt câu.** Tiếng Nhật viết 「です。」rồi vào câu sau, không có dấu cách nào cả —
regex cũ đòi khoảng trắng sau dấu chấm nên nguyên đoạn văn sẽ thành một "câu"
700 ký tự. `_SENTENCE_END` có thêm nhánh khớp rỗng sau `。！？`, và `_CLAUSE_BREAKS`
có thêm `、`.

### Trọng âm tiếng Nhật — trần của cách làm hiện tại

箸 và 橋 đều là `hashi`. 雨 và 飴 đều là `ame`. Cái phân biệt chúng là chỗ cao độ
rơi xuống, nên **đọc sai trọng âm không phải là giọng lạ, mà là ra từ khác**.

`kokoro.KPipeline` dựng `misaki.ja.JAG2P()` — và mặc định của nó là
`version='cutlet'`, tức front end **đời một**:

| | chuỗi | trọng âm |
|---|---|---|
| đời 1 (`cutlet`) | cutlet → fugashi → mecab → unidic-lite | **không có** |
| đời 2 (`pyopenjtalk`) | `pyopenjtalk.run_frontend` → nhân trọng âm theo từ điển Open JTalk | có, dạng dải `_` thấp / `-` giữa / `^` chỗ rơi, một ký tự mỗi phoneme |

Đời hai rõ ràng là cái nên dùng. Nhưng **dùng được hay không là câu hỏi về
checkpoint chứ không phải về thư viện**: `KModel.forward` map phoneme qua
`self.vocab` và **bỏ im lặng** mọi thứ không có id. Đưa dải trọng âm cho một
checkpoint không được huấn luyện với nó thì không lỗi — nó lặng lẽ xoá các dấu
đi và đọc phần còn lại, mà `j` (ký tự độn của dải) lại đúng là phoneme IPA /j/.
Đọc còn tệ hơn là không thử.

Nên `KokoroSynthesizer.load` **hỏi thay vì đoán**, đúng kiểu cổng `is_uroman`
của MMS: nếu `model.vocab` có id cho `_ - ^` thì đổi sang front end đời hai;
không có thì giữ nguyên và **nói ra trong log**. Kèm một câu thăm dò cố định
(`今日はいい天気ですね。`, không phải chữ của người dùng) in ra chuỗi phoneme.

**Câu trả lời đã tra được, không cần đoán.** `config.json` của Kokoro-82M có 114
ký hiệu (đọc từ bản ONNX port trên PyPI, `kokoro-onnx`, vì `huggingface.co`
không tới được từ máy viết code):

```
;:,.!?—…"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊabcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤəɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴøɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ
```

`_`, `-`, `^`, `↑` — **không có cái nào**. `j` thì có (id 52). Nghĩa là đưa dải
trọng âm đời hai cho v1.0 sẽ xoá sạch các dấu và đọc thừa một tràng /j/. Cổng ở
trên sẽ báo `accent=False`, và nó đúng.

### Đã đổi sang Open JTalk, rồi đổi lại về Kokoro

Đoạn này giữ nguyên cả hai nửa, vì nửa sau chỉ có nghĩa khi đọc cùng nửa trước.

**Lý lẽ lúc đổi sang Open JTalk.** `hexgrad/Kokoro-82M` không thể được cho biết
cao độ rơi ở đâu — đó là trần của nó, không sửa được trong engine. Còn
`pyopenjtalk.run_frontend` đọc nhân trọng âm ra từ **từ điển** của Open JTalk,
và backend HTS dựng ra âm thanh từ đó. Ba lý do nó hợp ở đây hơn là nghe qua
tưởng:

- **Đã có sẵn trong image.** `misaki[ja]==0.9.4` khai `Requires-Dist:
  pyopenjtalk`, nên nó vốn đã được build vào container từ đầu. Giọng
  (`mei_normal.htsvoice`) và từ điển đều nằm trong package, không tải gì.
- **`speed` và `half_tone` là tham số tổng hợp.** Cao độ theo câu đặt được ngay
  lúc dựng tiếng, tính bằng nửa cung: không resample, không kéo formant theo,
  không phải trả lại độ dài. Đây là engine duy nhất nhận `Beat.rate` thay vì
  `Beat.synth_rate`, và `prosody.shape` chỉ còn phải chỉnh mức.
- **Tưởng là đúng tiêu chí repo.** Giọng tổng hợp là *người đóng thế*, Seed-VC
  mới là thứ làm nó thành giọng người dùng, nên checkpoint chọn vì đúng chứ
  không vì đẹp — và cái dở nhất của HTS, tiếng rè của nguồn kích thích, lại
  đúng là phần Seed-VC vứt đi.

**Lý do đổi lại.** Gạch đầu dòng thứ ba sai, và nó là gạch đầu dòng mang cả
quyết định. "Người đóng thế" đúng với *timbre* và chỉ với timbre: Seed-VC thay
người nói, nó không thay cách nói. Nhịp, cách nhả chữ, độ cứng của từng âm tiết
— tất cả đi thẳng từ `spoken.wav` ra bản cuối. Giọng HTS vào máy đã nghe như
máy thì ra khỏi máy vẫn nghe như máy, chỉ là một cái máy khác. Nghe thử là biết,
và đó là thứ duy nhất có thể phân xử được chuyện này — mục "Seed-VC convert từ
giọng HTS máy móc có ra hồn không" trong danh sách verify ở dưới chính là rủi ro
đó, và câu trả lời là không.

Nên tiếng Nhật đọc lại bằng `KokoroSynthesizer`, và **cái mất được ghi ra chứ
không giấu đi**: 箸 với 橋 vẫn là chỗ Kokoro đoán. Đổi lại là một bản đọc người
ta chịu nghe, mà một bản đọc không ai muốn nghe thì không phải là một bản đọc
đúng hơn.

`OpenJTalkSynthesizer` ở nguyên đó, làm bản đối chiếu — cách nghe xem cái mặc
định đang đoán gì:

```
modal run -m modal_app.tts --language jpn --output kokoro.wav --text "箸と橋、雨と飴。"
modal run -m modal_app.tts --language jpn --output openjtalk.wav --engine openjtalk --text "箸と橋、雨と飴。"
```

`--engine` chỉ có ở smoke test, không request nào đặt được: nó là một lựa chọn
kỹ thuật cần tai chứ không phải một nút cho người dùng. Và nhớ rằng cả hai file
đó mới là **đầu vào** của conversion — cái quyết định là bản nào ra khỏi Seed-VC
mà nghe được.

**`--voice` cũng vậy.** Kokoro-82M v1.0 có năm giọng tiếng Nhật
(`KOKORO_VOICES`): `jf_alpha` (mặc định), `jf_gongitsune`, `jf_nezumi`,
`jf_tebukuro`, `jm_kumo`. Nó không phải nút cho người dùng vì Seed-VC thay timbre
ngay sau đó — năm giọng sẽ ra cùng một giọng. Nhưng *chọn* cái mặc định lại là
quyết định về cách đọc chứ không phải về chất giọng — cách ngắt cụm, tốc độ,
nhả chữ rõ hay lướt đều sống sót qua conversion — nên nó phải thử được bằng một
flag thay vì một lần deploy:

```
modal run -m modal_app.tts --language jpn --voice jf_gongitsune --text "今日はいい天気ですね。"
```

Tên giọng lạ thì rơi về mặc định chứ không làm hỏng job (`kokoro_voice`): lúc
đọc tới đó thì container đã lên, text đã nằm trên Volume và GPU đã đặt chỗ rồi.

### Số và ký hiệu không được đọc

Tokenizer của MMS làm việc trên ký tự, theo một bảng từ vựng chỉ có chữ cái và
dấu câu. Chữ số không nằm trong đó và bị **bỏ đi không kêu một tiếng** — "25
tuổi" đọc thành "tuổi". Hai chỗ xử lý:

- `check_text` từ chối đoạn không có lấy một chữ cái, để một dòng toàn số không
  trở thành job chạy thành công và cho ra file im lặng;
- form nhập nói thẳng ra điều đó, ngay cạnh ô gõ chữ.

Cố ý **không** tự đọc số hộ: "25" đúng ra là "hai mươi lăm", không phải "hai
năm", và ngữ pháp số theo từng ngôn ngữ là một việc lớn hơn nó trông nhiều.

### Ba chỗ đáng chú ý

1. **Cắt câu trước khi đọc.** VITS dự đoán độ dài cho từng token và sai số cộng
   dồn, nên một đoạn dài đưa vào nguyên khối sẽ trôi nhịp dần. `split_text` cắt
   ở ranh giới câu, rồi cắt tiếp ở mệnh đề nếu vẫn quá 200 ký tự, và nối lại
   với 0.25s im lặng — đúng chỗ dấu chấm vốn đã là chỗ nghỉ.
2. **Văn bản nằm trên Volume (`input.txt`), không nằm trong job record.** Cùng
   một lý do với audio: cron dọn rác 6 giờ xoá cả thư mục job, nên cái người
   dùng viết ra biến mất cùng lúc với file họ tải lên. Audit log ghi độ dài và
   ngôn ngữ, không bao giờ ghi nội dung.
3. **Pitch vẫn auto-detect.** Giọng tổng hợp của MMS chỉ có một quãng giọng cho
   mỗi ngôn ngữ, và giọng mẫu có thể cách nó cả quãng tám — nên `tts` đi qua
   đúng `_resolve_shift` mà nhánh `speech` dùng, đo trên `spoken.wav`.

### Còn phải verify bằng hạ tầng thật

Test trong CI phủ phần thuần logic — validate, cắt câu, clamp tốc độ, tham số
job, và việc `/submit` nhận text thay cho file. Ngoài ra, những thứ dưới đây đã
được **chạy thật** lúc viết code (PyPI tới được, chỉ HF là không): `uroman` trên
tiếng Nhật, `misaki[ja]` G2P trên chính các segment mà `split_text` sinh ra,
lỗi `unidic` và cách sửa, và việc resolve được cả `kokoro` lẫn `torch==2.4.0` +
`transformers==4.46.3` + `numpy<2` trong cùng một môi trường.

Còn checkpoint thì chưa từng được nạp ở đây (`huggingface.co` không tới được từ
máy viết code):

- [ ] `modal run -m modal_app.tts --text "Xin chào, đây là một câu thử."` → nghe được
- [ ] tiếng Việt có dấu đọc đúng thanh điệu, không nuốt dấu
- [ ] `--language jpn --text "今日はいい天気ですね。"` → ra tiếng Nhật thật, kanji
      đọc đúng (đây là mục quan trọng nhất của cả phase này)
- [ ] `--language jpn --text "kyou wa ii tenki desu ne."` → nghe giống hệt câu trên
- [ ] `hexgrad/Kokoro-82M` tải được cả `kokoro-v1_0.pth` lẫn `voices/jf_alpha.pt`
- [ ] chạy end-to-end với giọng mẫu thật: bản ra nghe giống giọng mẫu chứ không
      phải giọng tổng hợp pha
- [ ] đoạn dài kịch giới hạn (2000 ký tự Latin · 700 ký tự Nhật): mối nối giữa
      các câu không cụt, nhịp không trôi
- [ ] tốc độ đọc 0.5× và 2× vẫn hiểu được sau khi qua Seed-VC
- [ ] container thứ hai không tải lại weights (HF_HOME trên Volume chạy đúng)
- [ ] đo thời gian thật của bước synthesize trên CPU cho cả hai engine, để biết
      `cpu=4` là đủ hay thừa


## Phase 9 — Đọc có ngữ điệu

Phase 8 đọc được chữ. Nó đọc **đều**: câu nào cũng một tốc độ, một độ cao, một
độ to, và giữa hai câu bất kỳ — dấu phẩy, dấu chấm, hay hết cả một đoạn — đúng
0.25 giây im lặng như nhau. Nghe ra ngay: đó là máy đọc cho xong chứ không phải
người đọc.

Phase này thêm `modal_app/prosody.py`, và **không** thêm model nào.

```
text ──► split_blocks ──► plan() ──► [Beat, Beat, …] ──► engine + shape()
```

`tts.py` giữ phần "ai đọc", `prosody.py` giữ phần "đọc thế nào".

### Vì sao không phải một model có cảm xúc

Cả MMS-TTS lẫn Kokoro đều nói bằng đúng một giọng cố định, không có đầu vào
emotion nào cả. Muốn có emotion thật theo kiểu conditioning thì phải thay engine
lần thứ ba — thêm weight, thêm cold start, và mất luôn phần phủ ngôn ngữ mà MMS
đang cho. Trong khi phần lớn cái tai nghe ra là "đọc có hồn" lại là những thứ
viết ra được thành luật:

- **Ngắt nghỉ dài ngắn theo dấu câu.** Hướng dẫn phổ biến của mấy hệ TTS thương
  mại: 120–300 ms cho một chỗ ngắt trong câu, 400–700 ms cho hết đoạn hoặc chỗ
  ngắt có chủ ý. Nên dấu phẩy (0.20s), dấu chấm (0.40s), dấu ba chấm (0.60s) và
  dòng trống (0.75s) là bốn khoảng lặng khác nhau, thay cho một con số duy nhất.
  Chỗ bị cắt giữa câu vì quá dài thì ngắn nhất (0.12s) — chỗ đó vốn không có
  dấu nghỉ nào.
- **Cao độ trôi xuống dần trong một đoạn rồi bắt lại ở đoạn sau** (declination).
  Đây đúng là thứ khiến một dãy câu tổng hợp rời nhau nghe như đọc danh sách.
  Nó được đặt **cân đối quanh 0** — câu đầu cao hơn nửa quãng, câu cuối thấp hơn
  nửa quãng — để F0 trung vị của cả file không đổi, vì `pipeline._resolve_shift`
  đo đúng con số đó để quyết định dịch giọng.
- **Câu hỏi đọc cao hơn, câu kể xuống giọng.** Câu kể xuống là phần đuôi của
  declination. Câu hỏi thì *cả câu* cao hơn chứ không phải vuốt lên ở cuối —
  vuốt cuối câu mới đúng là cách người ta hỏi, nhưng vẽ một đường cong bên
  trong một câu thì chỉ có phase vocoder làm được, và đó đúng là thứ đã làm
  hỏng tiếng Nhật (xem mục dưới). F0 trung bình của câu hỏi vốn cũng cao hơn,
  nên đây vẫn là một dấu hiệu thật, chỉ là nhẹ hơn. Tiếng Nhật hỏi bằng か cuối
  câu và thường không có dấu hỏi (「そうですか。」), nên chỗ đó cũng được đọc là
  câu hỏi — nhưng **chỉ ở chỗ câu thật sự kết thúc**: か là âm tiết bình thường
  trong từ bình thường (しずか, なんとか, たしか), mà tiếng Nhật không có dấu
  cách nên `_wrap` cắt câu dài ở một ký tự bất kỳ, và một mẩu cắt ra kết thúc
  bằng か thì rất dễ gặp.
- **Cảm xúc là năm thứ đi cùng nhau**: tốc độ, cao độ trung bình, biên độ lên
  xuống, độ to, và độ dài khoảng lặng. Các nghiên cứu acoustic về giọng cảm xúc
  thống nhất với nhau về *hướng* của cả năm — vui/giận thì F0 cao hơn, biên độ
  rộng hơn, nói nhanh hơn và to hơn; buồn thì ngược lại cả năm và nghỉ nhiều
  hơn. Nên một "cảm xúc" ở đây là năm con số theo đúng các hướng đó, biên độ cố
  ý nhỏ hơn số đo trên giọng diễn: đây là giọng đọc, không phải diễn.

### Đưa vào bằng cửa nào

| Chỗ | Cái có sẵn |
|---|---|
| MMS (VITS) | `speaking_rate`, `noise_scale`, `noise_scale_duration` |
| Kokoro | `speed` |
| Open JTalk (chỉ dùng để A/B) | `speed`, `half_tone` — nửa cung đặt ngay lúc tổng hợp, nên nó là engine duy nhất nhận `Beat.rate` và `prosody.shape(engine_pitch=True)` |
| Sau khi tổng hợp | gain, và dịch cao độ bằng **resample** |

`noise_scale` là mức cho phép prior dao động — tức cao độ và năng lượng của bản
đọc — còn `noise_scale_duration` là cùng thứ đó cho bộ dự đoán độ dài, tức nhịp.
Style chỉnh chúng bằng **hệ số nhân** trên giá trị của chính checkpoint chứ
không ghi đè bằng một hằng số: mặc định khác nhau theo từng checkpoint.

Ba thuộc tính đó được đọc lúc forward, nên đặt lại cho **từng câu** không tốn gì
thêm — một đoạn văn vẫn đúng bằng số lần gọi model như trước.

Lưu ý cái rate nào đi vào engine: `Beat.synth_rate`, **không phải** `Beat.rate`.
Dịch cao độ ở đây làm bằng resample nên nó đổi luôn độ dài; engine được yêu cầu
đọc chậm đi đúng bằng phần mà resample sẽ tăng lên, và câu ra đúng nhịp cần
nghe. Đưa nhầm `rate` vào thì mọi câu có dịch cao độ đều sai độ dài.

### Vì sao không dùng phase vocoder — bài học phải trả bằng tiếng Nhật

Bản đầu của module này dịch cao độ bằng `librosa.effects.pitch_shift`, tức
time-stretch bằng phase vocoder rồi resample lại cho đúng độ dài. **Nó đọc sai
tiếng Nhật.**

Phase vocoder phân tích theo cửa sổ — mặc định của librosa là 2048 mẫu, tức
**85 ms** ở 24 kHz của Kokoro — và bôi nhoè mọi thứ ngắn hơn một cửa sổ. Tiếng
Nhật đặt nghĩa đúng vào cỡ đó: っ là một quãng lặng, ー là nguyên âm kéo dài, ん
là một mora riêng, và きて với きって khác nhau ở độ dài một khoảng lặng. Đo
được:

| | quãng lặng 80 ms |
|---|---|
| gốc | 80.0 ms |
| qua phase vocoder (0.6 nửa cung) | **36.9 ms** |
| qua resample (0.6 nửa cung) | 77.3 ms (đúng bằng 80 ÷ 2^(0.6/12)) |

Và nó chạy trên **mọi câu**, để đổi cao độ chưa tới một nửa cung — thứ mà không
có thì cũng chẳng ai nhận ra. Câu hỏi còn tệ hơn: bốn lượt vocoder chồng lên
0.35 giây cuối, đúng chỗ trọng âm cuối câu nằm.

Resample không có cửa sổ và không phân tích gì cả. Nó là đổi tốc độ băng: mọi
tỉ lệ trong tín hiệu giữ nguyên chính xác, nên đường thanh điệu tiếng Việt,
đường trọng âm cao độ tiếng Nhật và một phụ âm đôi đều sống sót. Đổi lại nó
đổi độ dài (đã trả bằng `synth_rate`) và dịch formant theo cao độ — lý do
`MAX_SENTENCE_PITCH_ST` để nhỏ, và cũng là lý do nó ít quan trọng ở đây: Seed-VC
thay timbre ngay bước sau.

Cái mất: không vẽ được đường cong cao độ *bên trong* một câu nữa, nên vuốt cuối
câu hỏi thành cả câu đọc cao hơn. Đó là đánh đổi có ý thức — dấu hiệu yếu hơn,
nhưng không phá chữ.

### Hai giới hạn cố ý

**Chỉ ở mức câu.** Nhấn một từ có nghĩa là cắt câu ra làm đôi và tổng hợp riêng
từng nửa, mất coarticulation ở chỗ nối — đọc dở đi chứ không hay lên.

**Mọi độ lệch đều nhân với một con số duy nhất.** `expressiveness` (0 → 1.5) là
khoảng cách giữa bản đọc phẳng và style đã chọn. 0 không phải "không có style",
nó chính là bản đọc mà Phase 8 cho ra. Chỉ *độ lệch* mới co lại: khoảng lặng sau
dấu phẩy là dấu câu chứ không phải tâm trạng, nên nó sống sót ở mức 0.

### Test

`tests/test_prosody.py` phủ toàn bộ phần ra quyết định — nó là Python thuần trên
một danh sách câu, chạy được trong CI không cần checkpoint, không cần container.
Phần DSP có test thật nhưng cần librosa (nằm trong `base_image`, không nằm
trong `requirements.txt`) nên CI skip: cao độ dịch đúng số nửa cung đã yêu cầu,
độ dài đổi đúng bằng phần `synth_rate` trả lại, và quãng lặng 80 ms vẫn là
quãng lặng — kèm luôn phép đo cho thấy phase vocoder không giữ được nó, để
không ai đưa nó trở lại.

### Còn phải verify bằng tai

- [ ] `modal run -m modal_app.tts --emotion cheerful --text "Chào cậu! Cậu khoẻ
      không? Lâu rồi không gặp…"` → ba câu, ba khoảng lặng khác nhau, đúng một
      câu lên giọng ở cuối
- [ ] cùng đoạn văn qua cả năm style: nghe ra khác nhau, và không style nào
      nghe như đang diễn
- [ ] `--expressiveness 0` → đúng bằng bản đọc Phase 8 (trừ ngắt nghỉ theo dấu câu)
- [x] **so `openjtalk.wav` với `kokoro.wav` trên 箸/橋, 雨/飴** — trước và sau
      Seed-VC. Đây là mục quyết định giọng tiếng Nhật đi đường nào
- [x] Seed-VC convert từ giọng HTS máy móc có ra hồn không — rủi ro duy nhất
      chưa đo được của việc đổi engine. **Không.** Nên mặc định quay về Kokoro;
      xem "Đã đổi sang Open JTalk, rồi đổi lại về Kokoro"
- [ ] năm giọng Nhật của Kokoro (`--voice`) — cái nào ngắt cụm và nhả chữ rõ
      nhất *sau* Seed-VC, vì đó mới là phần giọng nào cũng giữ lại
- [ ] `pyopenjtalk==0.4.1` build được trong image (PyPI chỉ có sdist, không có
      wheel) — bước `pyopenjtalk.g2p` lúc build sẽ làm deploy đỏ nếu không
- [ ] **tiếng Nhật đọc đúng chữ** — đây là mục quan trọng nhất của phase này,
      vì bản đầu đã sai đúng chỗ đó: っ và ー còn nguyên độ dài, きって không
      thành きて
- [ ] tiếng Việt: dịch cao độ theo câu không làm méo thanh điệu (đây là lý do
      các con số đều dưới 1.5 nửa cung)
- [ ] `そうですか。` được đọc như câu hỏi, còn một mẩu cắt ra kết thúc bằng か
      (「きょうはとてもしずか」) thì không
- [ ] độ dài câu vẫn đúng sau khi resample — nếu nghe nhanh/chậm bất thường ở
      câu có dịch cao độ thì `synth_rate` bị đưa nhầm chiều
- [ ] khoảng lặng sau `。` (0.40s) cộng với khoảng lặng Kokoro tự sinh ra từ dấu
      câu có bị thành lê thê không — chưa kiểm chứng được ở đây vì `misaki[ja]`
      không cài được trên máy viết code
- [ ] sau Seed-VC, ngữ điệu còn giữ được — nếu conversion nuốt mất phần lên
      xuống thì phải đo lại xem nên đẩy biên độ lên bao nhiêu
- [ ] đo thêm thời gian CPU: mỗi câu thêm một lần pitch_shift, câu hỏi thêm bốn

---

## Phase 10 — Giọng trong hơn và bớt "AI"

Yêu cầu vào phase này là một câu: *"làm sao để giọng trong hơn và nghe không bị
AI hoá"*. Nó không phải một bug để sửa mà là bốn chỗ khác nhau cùng góp phần, và
phase này đụng vào cả bốn — cộng thêm một mode mới đi kèm cùng lý do.

Nguyên tắc xuyên suốt: **mọi thứ thêm vào đều kéo về 0 được.** `Độ trong` = 0
nghĩa là không có một filter nào chạy, `Bám giọng mẫu` giữ nguyên mặc định
upstream, `Độ biểu cảm` = 0 vẫn đọc phẳng đúng như trước. Bản audio mà app này
trả về trước Phase 10 vẫn lấy lại được bằng cách kéo slider, chứ không phải bằng
cách revert code.

### 1. Giọng mẫu bẩn thì giọng ra bẩn — `reference.py`

Đây là chỗ đóng góp lớn nhất và cũng là chỗ trước giờ không ai đụng tới.

Seed-VC **không sao chép một giọng, nó sao chép một mẫu ghi âm**. Tiếng ồn nền,
tiếng quạt laptop, tiếng ù 50 Hz từ củ sạc, tiếng vọng của một phòng nhỏ — với
model đó không phải là "nền", đó là một phần của timbre nó được yêu cầu tái tạo.
Nên nó dính vào từng âm tiết của kết quả, và kết quả nghe *đã qua xử lý*: rè,
lạo xạo, nhân tạo. Chính xác là cái người ta gọi là "nghe bị AI".

Trước khi mẫu được encode, nó đi qua ba phép sửa, đúng thứ tự đó:

| Bước | Làm gì | Vì sao |
|---|---|---|
| Cắt ù | tắt hẳn dưới 40 Hz, mở dần tới 75 Hz | dưới 75 Hz không có gì của giọng người, mà có rất nhiều tiếng gõ bàn, bước chân, ù điện |
| Trừ nhiễu | ước lượng sàn nhiễu từng dải tần bằng phân vị 15 theo thời gian, trừ đi với hệ số 1.8 và **sàn -14 dB** | tiếng nói thì to và ngắt quãng, tiếng phòng thì nhỏ và liên tục — phân vị rơi đúng vào tiếng phòng. Sàn -14 dB là thứ phân biệt "giảm ồn" với cái tiếng ọc ạch dưới nước mà ai cũng nhận ra: gate một dải về 0 rồi mở lại ở frame sau chính là cách tạo ra nó |
| Đặt mức | chuẩn hoá RMS của *phần có tiếng* về ~-24 dBFS, không cho vượt đỉnh | không phải để to hơn — `loudnorm` cuối pipeline lo việc đó — mà vì mẫu và nguồn tới model ở mức của hai cái micro khác nhau, và mẫu nhỏ tiếng là mẫu model nhìn thấy ít hơn |

Toàn bộ bằng numpy, một lượt STFT làm cả ba việc, không scipy không librosa —
nên `tests/test_reference.py` chạy trong CI trên máy trần và đo bằng số thật:
ù giảm còn dưới 5%, nhiễu băng rộng giảm hơn 40%, dải giọng giữ trên 70%.

### 2. Chọn nhầm đoạn thì làm sạch cũng vô ích — `speech_flags`

`usable_reference_window` xưa nay chấm điểm mỗi cửa sổ 20 giây bằng **mức to**.
Có đúng một trường hợp nó sai và trường hợp đó không hiếm chút nào: bản ghi mở
đầu bằng một tiếng động lớn — kéo ghế, gõ cửa, preamp vặn to trước khi có ai nói
— thì đoạn to nhất được chấm là đoạn giọng đẹp nhất, và model nhận đúng 20 giây
tiếng động đó.

Không có ngưỡng âm lượng nào phân biệt được hai thứ, nên phải hỏi thêm một câu
khác: **có tuần hoàn không**. Một frame giọng hữu thanh cắt trục 0 đúng hai lần
mỗi chu kỳ — 0.006 số mẫu ở 140 Hz và 44.1 kHz — còn nhiễu, tiếng quạt và méo
xén cắt cỡ một nửa số mẫu. Không có mức khuếch đại nào làm nhiễu trở nên tuần
hoàn.

Kèm một chi tiết nhỏ mà thiếu nó thì hỏng cả: **sàn âm lượng đo theo đỉnh của
các frame có giọng, không theo đỉnh của cả file.** Nếu đo theo cả file thì một
tiếng sập cửa làm mọi *từ* nói nhỏ sau đó rơi xuống dưới sàn, và cửa sổ được
chọn lại chính là cái cửa. Và là phân vị 95 chứ không phải max, vì một đoạn
nhiễu dài thì kiểu gì cũng có đúng một frame lọt qua bài kiểm tra tuần hoàn.

### 3. Một cửa sổ là một mẫu của người, không phải là người — style trung bình

Speaker embedding lấy từ một cửa sổ mang theo cả cái riêng của *cửa sổ đó* lẫn
cái riêng của người. Trung bình vài cửa sổ thì phần thứ nhất triệt tiêu, phần
thứ hai còn lại — đó chính xác là cách speaker verification vẫn enrol.

Nên `reference.prepare` trả về `(cửa sổ chính, các cửa sổ phụ)`: mel, content và
độ dài vẫn lấy từ một chỗ duy nhất (trung bình hai câu khác nhau thì vô nghĩa),
chỉ riêng embedding CAM++ được lấy trung bình. Các cửa sổ phụ không chồng lên
nhau, không chồng lên cửa sổ chính, và một cửa sổ dưới 50% là giọng thì không
được nhận — trộn tiếng phòng vào embedding là đẩy nó ra xa người nói.

Bản ghi ngắn (dưới 40 giây) không có cửa sổ phụ nào và không có gì thay đổi, tức
là phần lớn các lần chạy. Cái này ăn tiền đúng ở trường hợp `prepare_reference`
vẫn mời gọi từ đầu: *"đưa cả phút cũng được"*.

### 4. Vocoder để lại dấu vết — `enhance.py`

Decoder diffusion + neural vocoder rất giỏi ở giữa dải tần và cẩu thả ở hai đầu.
Cái quay về đều đặn có: ù và trôi DC dưới 70 Hz, một lớp mờ băng rộng cỡ -50
dBFS (sàn nhiễu của chính vocoder — làm sạch đầu vào không xử lý được nó), một
cục dồn quanh 250–350 Hz, và tiếng gió /s/ bị thổi phồng vì nhiễu thì model tái
tạo bằng cách sinh ra nhiễu.

Một chuỗi ffmpeg, chỉ áp lên **giọng** — trong mode `song` nó nằm trước `amix`
nên nhạc nền không bị đụng tới:

```
aformat=fltp → afftdn (bám sàn nhiễu) → highpass 70 Hz
             → -2.5 dB @ 300 Hz → +2.5 dB @ 3.4 kHz → +1.5 dB shelf 8 kHz → de-esser
```

Thứ tự không đổi chỗ được: khử nhiễu trước mọi thứ có gain (nếu không thì boost
lên chính cái sàn sắp bị khử), và de-esser sau cùng vì nó sửa hậu quả của cú
nâng 3.4 kHz — de-ess trước cú nâng là đo tiếng gió chưa xảy ra.

Con số ở trên là mức tại `Độ trong` = 100%; slider nhân tuyến tính, mặc định
50%. Tại 0% hàm trả về **rỗng**, không phải "nhẹ" — `enhance.chain(0)` là chuỗi
rỗng và graph của caller y hệt lúc chưa có module này.

### 5. Không câu nào đọc giống câu nào — `prosody._jitter`

Cái đuôi cuối cùng của "nghe như máy" nằm ở chỗ mọi luật trong `prosody.py` đều
được áp *chính xác*: hai câu cùng loại, cùng vị trí trong đoạn thì ra cùng tốc
độ, cùng cao độ, cùng khoảng lặng, tới từng mẫu. Không ai đọc như thế. Biến
thiên tốc độ giữa các câu liền nhau của cùng một người là vài phần trăm và cao
độ trung bình lệch nhau cỡ nửa cung — đó không phải nhiễu, đó là khác biệt giữa
một người và một máy đếm nhịp.

`_jitter` trả lại một lượng nhỏ: ±3% tốc độ, ±0.25 nửa cung, ±15% độ dài khoảng
lặng. Bằng **hash của chính câu đó** chứ không phải random — cùng một văn bản
phải cho ra cùng một bản thu, nếu không thì không so được với lần trước và một
lời phàn nàn về một câu cụ thể không tái hiện được để mà sửa. Và nó nhân với
`expressiveness` như mọi độ lệch khác, nên ở mức 0 bản đọc vẫn phẳng đúng bằng
0.

### 6. `Bám giọng mẫu` — cái nút xưa nay bị đóng cứng

`inference_cfg_rate` là classifier-free guidance: cân giữa dự đoán *có nhìn thấy*
giọng mẫu và dự đoán không nhìn thấy. Trước đây nó là `0.7` hardcode trong chữ
ký hàm và không đường nào chạm tới.

Kéo lên: giống mẫu hơn, và cũng lộ chất máy hơn — artefact của một model
diffusion là artefact của conditioning, ép conditioning mạnh lên là ép chúng
mạnh lên theo. Kéo xuống: còn lại nhiều nét của người trên bản gốc. 0.7 vẫn là
mặc định vì nó là giữa khoảng dùng được, và đây là slider nằm trong `Tinh chỉnh`
chứ không phải ngoài trang chính: hai đầu đều là setting thật và đều tệ hơn ở
giữa với phần lớn material.

### 7. Train giọng riêng — `training.py`, `voices.py`

Zero-shot là nhìn một người trong 20 giây rồi làm người đó trong 4 phút. Đó là
một việc đáng nể và nó **có trần**, và cái trần đúng là thứ người ta mô tả khi
nói kết quả *gần giống* nhưng vẫn nghe như máy đang mặc một giọng: 20 giây không
chứa hết quãng giọng của người đó, không chứa cái hơi ở đáy quãng, không chứa
phụ âm của họ khi nói nhanh — nên model điền nốt bằng trung bình của tất cả
những người nó từng học, và *trung bình của tất cả mọi người* chính là âm thanh
của một giọng AI.

Fine-tune thay chỗ đoán đó bằng dữ liệu. `train.py` của upstream được dùng
nguyên vẹn như một thư viện, và đáng nói rõ nó làm gì vì nó không giống nghĩa
thông thường của "train một model giọng":

* nó fine-tune **DiT** (decoder diffusion) và length regulator, hết. Content
  encoder (Whisper), speaker encoder (CAM++) và vocoder giữ nguyên;
* nó **không cần transcript, không cần nhãn**, chỉ cần audio của một người. Tín
  hiệu học là self-supervised: mỗi clip được đẩy qua tone converter của
  OpenVoice sang một timbre ngẫu nhiên khác, rồi model phải dựng lại bản gốc từ
  content của bản đã đổi cộng speaker embedding của bản gốc. Nên nó học cách đưa
  *bất cứ thứ gì* về giọng này;
* vài phút audio và vài trăm step là khoảng làm việc. Đây là adaptation chứ
  không phải train từ đầu — một giờ audio không hơn năm phút mười lần, và 5000
  step không hơn 500 step năm lần. Overfit một giọng ở đây dễ hơn underfit nhiều.

Profile lưu theo **từng mode**: `speech` và `singing` là hai kiến trúc khác nhau
ở hai sample rate khác nhau, một profile train cho cái này không phải là "kém
hơn" khi nạp vào cái kia — nó không nạp được. Hai thư mục biến điều đó thành một
sự thật của filesystem thay vì một luật ai đó phải nhớ.

**Đây là công cụ vận hành, không phải endpoint.** Không có gì trong `api.py` khởi
động một lần train: nó là mười phút GPU, nó cần audio mà chưa ai đặt rate limit
lên, và câu hỏi đồng thuận cho "giữ giọng người này trên server vĩnh viễn" là
một câu hỏi khác với câu mà form submit đang hỏi. Người trả tiền GPU chạy nó từ
shell:

```bash
modal run -m modal_app.training --voice mai --audio ./mai.wav
modal run -m modal_app.training --voice mai --audio ./clips --mode singing

# nghe thử, có/không profile, cùng source cùng reference:
modal run -m modal_app.conversion --source a.wav --reference b.wav --mode speech
modal run -m modal_app.conversion --source a.wav --reference b.wav --mode speech --voice mai
```

Job hỏi bằng tên: `voice_profile=mai` trên `/submit`, và `GET /voices` liệt kê
những gì deployment này đang có. Tên không dùng được thì bị bỏ qua (chạy
zero-shot như mọi job trước khi có tính năng này); tên **dùng được mà không có
profile** thì job fail trong container GPU — vì một job xin giọng đã train mà âm
thầm chạy zero-shot sẽ cho ra kết quả *nghe có lý* và sai, và không ai biết mà
hỏi.

Audio dùng để train bị xoá ngay sau khi lưu checkpoint, cùng chỗ với thư mục
run: một profile giọng không phải là lý do để giữ tiếng của ai đó trên server vô
thời hạn — phần còn lại của app hết hạn audio của user sau 6 giờ.

### 8. Mode `vocal` — chuyển giọng, không tách nhạc nền

Đúng là `song` bỏ đi bước tách. Nghe như một tối ưu tốc độ, nhưng lý do chính
nằm ở chất lượng: **separation không miễn phí theo cả hai chiều.** Nó tốn một
lượt GPU, và thứ nó giao cho converter là một stem mang theo artefact mà mọi
model tách nguồn đều để lại — transient bị nhoè, bóng mờ của bản phối còn sót
lại — rồi những thứ đó được convert *cùng với* giọng. Một phần không nhỏ của
việc "bài hát convert xong nghe như đã qua máy" nằm ở đấy.

File đã là giọng không nên trả cả hai cái giá đó. Nên `vocal` đi thẳng vào
conversion với checkpoint `singing` (có điều kiện F0, 44.1 kHz) và trả về đúng
cái đó: không stem, không mix, không dịch cao độ tự động — bản hát có key của
nó.

Đưa cả một bản mix đầy đủ vào đây thì được phép và không phải mục đích của nó:
nhạc nền sẽ đi qua Seed-VC cùng với giọng. Đó là thứ đáng nghe thử một lần chứ
không phải một cách convert bài hát.

### Ba chỗ đáng chú ý

**`enhance.chain(clarity, ",")` trả về chuỗi rỗng khi clarity = 0, kể cả dấu
phẩy.** Nghe vụn vặt, nhưng một filter graph thừa dấu phẩy là lỗi ffmpeg lúc
chạy, trong container, trên job của người đang đợi — không phải lúc build.

**Hai slider mới luôn được gửi, không bao giờ bỏ trống.** Với `semitone_shift`,
"vắng mặt" mang nghĩa *tự đo* và khác hẳn 0. Với hai cái này thì 0 là setting
thật (không guidance, không hậu kỳ) nên form luôn có câu trả lời, và backend
phân biệt `None` với `0` chứ không dùng `or`.

**`speech` và `vocal` chạy chung một hàm.** Chúng chỉ khác nhau đúng ở checkpoint
nào convert (`jobs.CONVERSION_MODE`), nên `_convert_uploaded` viết một lần. Bản
copy-paste của nó sẽ là bản mà một trong hai nhánh lặng lẽ ngừng được đóng dấu
watermark.

### Test

`tests/test_reference.py`, `tests/test_enhance.py`, `tests/test_voices.py`,
`tests/test_training.py` — cộng thêm phần mới trong `test_audio_utils`,
`test_prosody`, `test_pipeline`, `test_api`, `test_deploy`. Tất cả chạy trong CI
không cần GPU: phần DSP đo trên tín hiệu có đáp án tính được bằng số học, phần
ffmpeg chạy ffmpeg thật, phần training test đúng cái nửa quyết định GPU được xem
gì (độ dài clip mà `ft_dataset` lặng lẽ bỏ qua nếu sai, khoảng lặng, và việc từ
chối tiêu GPU cho một bản ghi quá ngắn để học được ai).

### Còn phải verify bằng tai và bằng GPU

- [ ] **A/B giọng mẫu bẩn**: cùng một câu, mẫu thu bằng điện thoại trong phòng
      có tiếng quạt, chạy trước và sau `reference.clean`. Đây là mục quan trọng
      nhất của phase — nếu nó không nghe khác thì cả phần 1 không đáng giá
- [ ] mẫu dài hơn 40 giây: embedding trung bình nhiều cửa sổ có ổn định hơn
      thật không, hay chỉ là một phép trung bình không ai nghe ra
- [ ] `Độ trong` ở 0 / 50 / 100% trên cùng một bản: 50% có phải chỗ đúng để đặt
      mặc định không, và 100% đã quá tay chưa
- [ ] `afftdn` trên giọng **hát** — nó bám sàn nhiễu, và đuôi reverb của một bản
      hát có thể bị nó coi là nhiễu
- [ ] `Bám giọng mẫu` ở 0.4 / 0.7 / 1.0: mô tả trong slider hint ("giống hơn
      nhưng lộ chất máy hơn") là suy luận từ cách CFG hoạt động, chưa phải là
      một phép nghe
- [ ] train một giọng thật: 500 step trên A10G có phải ~10 phút không, và
      checkpoint ra có nạp lại được bằng `--voice` không. **Toàn bộ nhánh
      training chưa từng chạy** — `Trainer` được gọi đúng chữ ký của upstream ở
      commit đã pin, nhưng chữ ký đúng không phải là đã chạy
- [ ] profile `singing`: preset config ghi `pretrained_model` là bản `ft_ema`
      còn inference dùng `ft_ema_v2`. Ở đây truyền thẳng checkpoint của
      inference vào để hai bên xuất phát từ cùng một chỗ — nếu kiến trúc lệch
      thì `load_checkpoint` sẽ báo, và fallback là bỏ `PRETRAINED` đi
- [ ] một giọng train xong có thật sự hơn zero-shot không, và hơn bao nhiêu —
      đây là câu hỏi quyết định tính năng này có đáng giữ hay không
- [ ] mode `vocal` trên một bản a cappella: so với `song` trên bản mix đầy đủ
      của cùng bài, phần giọng có sạch hơn thấy rõ không (giả thuyết của cả
      mode nằm ở đây)
- [ ] jitter ±3%/±0.25 nửa cung: có nghe ra "người hơn" hay chỉ là không nghe
      thấy gì — nếu là vế sau thì tăng dần chứ đừng bỏ

---

## Phase 11 — Đổi beat

Câu hỏi mở ra phase này là *"có cách nào đổi beat để tránh bản quyền không,
kiểu làm ra một beat mới luôn"*. Câu trả lời có hai nửa và nửa đầu quan trọng
hơn nửa sau.

### Sửa beat cũ không thoát được, và đây là lý do

Đổi tone, đổi tempo, EQ hay thêm hiệu ứng lên một bản beat có sẵn tạo ra **tác
phẩm phái sinh** — vẫn thuộc quyền của chủ sở hữu gốc. Về kỹ thuật cũng không
thoát: các hệ nhận dạng dấu vân tay được thiết kế để sống sót qua đúng những
phép biến đổi đó.

Và một chuyện hay bị bỏ sót: **một bài hát có ít nhất hai quyền tách rời** —
bản ghi (master) và tác phẩm (giai điệu + lời). Tách nhạc nền rồi thay beat chỉ
đụng tới cái thứ nhất. Nếu phần hát vẫn đi đúng giai điệu và lời bài gốc thì
phần đó vẫn nguyên vẹn là của người ta, dù beat có mới 100%; app này đổi
*timbre* người hát chứ không đổi giai điệu.

Nên mode `beat` sạch khi phần vocal là rap hoặc lời của chính người dùng, hoặc
khi họ đã có license cho bản cover. Nó **không** là cách hợp thức hoá việc ghép
giọng lên đúng giai điệu bài gốc, và README nói thẳng ra vì UI không phải chỗ
để giải thích luật.

### Kiến trúc: đo trước, rồi mới làm

Chỗ khó không phải sinh ra beat mà là **khớp nó với phần hát đã có**. Nên phase
này chia ba module theo đúng ba việc khác nhau, và hai trong ba không cần GPU:

```
bài ──► tách ──► vocal ─────────────► convert ──────────────┐
             └─► instrumental ──► analysis.py (BPM, tông)   │
                                          │                  ├──► mix ──► mp3
beat tải lên ────────────────► beats.py ──┘                  │
mô tả ──► beatgen.py ────────► beats.py ─────────────────────┘
```

Bản instrumental gốc được tách ra, **đo**, rồi vứt đi. Nó là mốc thời gian và
mốc tông, không phải một phần của output. Đo trên instrumental chứ không đo
trên vocal là cố ý: giọng hát tách riêng gần như không có nhịp để bắt — trống
mới là thứ giữ nhịp — và đoán tông từ một bè giai điệu là đoán mò.

### `analysis.py` — BPM và tông, numpy thuần

**Tempo.** Spectral flux → autocorrelation → nội suy parabol → **khớp lại lưới
phách bằng bình phương tối thiểu**. Bước cuối là bước đáng nói: autocorrelation
cho chu kỳ sai vài phần mười phần trăm, nghe thì không ra gì và trên bài 3 phút
là **nửa giây trôi** giữa beat và giọng — mà trôi là kiểu sai duy nhất người
nghe không bỏ qua được. Đi dọc lưới, tìm onset mạnh nhất gần mỗi phách dự đoán,
rồi fit một đường thẳng qua chúng: hệ số góc là chu kỳ, tung độ gốc là pha, cả
hai giờ được ước lượng từ cả trăm phách thay vì từ một đỉnh tương quan.

Đo trong CI trên click track: **sai số BPM < 0.02%, sai số vị trí phách < 11ms.**

Hai cái bẫy đã cắn và đã viết lại thành hằng số có tên:

* **`ONSET_LEAD_FRAMES`** — spectral flux báo onset *sớm* hơn tiếng thật, và
  lượng sớm đó suy ra được chứ không phải fudge: entry `i` so frame `i+1` với
  frame `i`, nên phần mẫu chỉ có ở một bên là `[i*HOP + FRAME, (i+1)*HOP +
  FRAME)`. Không trừ đi thì mọi phách lệch sớm 70ms.
* **`PULSE_FLOOR`** — không có gate thì `tempo()` **luôn** trả lời. Một hợp âm
  organ ngân dài ra 108 BPM: flux của nó là bụi số học, và autocorrelation của
  bụi thì cũng tuần hoàn như mọi thứ khác. Envelope được chia cho mức phổ trung
  bình nên ngưỡng là một tỷ lệ, không phải một mức: click track ~6000, hợp âm
  ~3000, tiếng ngân 3, nhiễu trắng 21. Ngưỡng 50 nằm giữa một khoảng trống rộng
  hai bậc — và đặt thấp có chủ ý, vì bỏ sót một nhịp chỉ làm beat mất khớp
  tempo, còn bịa ra một nhịp làm beat mất tempo.

**Tông.** Chroma 12 bậc → tương quan với 24 profile Krumhansl-Schmuckler. Trả
về kèm **biên độ thắng** (`key_margin`), vì điểm yếu đã biết của phương pháp
này đúng là chỗ quan trọng nhất: một tông và tông tương đối của nó dùng chung
cả bảy nốt, nên trên bài đi qua cả hai thì hai đáp án tương quan gần bằng nhau.
Margin nhỏ nghĩa là **đừng dịch tông dựa trên cái này**.

### `beats.py` — bốn quyết định, ffmpeg thuần

* **Cắt loop tròn ô nhịp.** File tải lên không kết thúc đúng chỗ hết ô nhịp, nên
  lặp thẳng là đặt một mối nối vào giữa một phách. Cắt từ phách đầu của nó tới
  ô nhịp trọn vẹn cuối cùng tốn một giây audio và mua được điểm lặp nằm đúng chỗ
  tai người chờ nó.
* **Dịch tông theo tông tương đối, không theo chủ âm.** Loop thứ đặt dưới bài
  trưởng không muốn về chủ âm trưởng đó — nó muốn về tông thứ tương đối, vốn
  dùng chung cả bảy nốt. Dịch về chủ âm là lệch một quãng ba thứ và nghe đúng
  như thế.
* **Khớp tempo theo quãng tám gần nhất.** Beat 140 BPM dưới bài 70 BPM là *đã*
  đúng nhịp; bắt nó chia đôi là phá beat để sửa một lỗi nó không có. Gập tỷ lệ
  theo bội 2 cho tới khi gần 1 nhất giữ mọi phép kéo trong khoảng 0.71x–1.41x,
  là khoảng WSOLA còn nghe ra nhạc.
* **Tách cao độ khỏi tempo.** `asetrate` là đổi tốc độ băng — nó nhân cả hai. Nên
  `atempo` sau đó phải trả lại đúng phần cao độ đã lấy: `atempo = tỷ_lệ / cao_độ`.
  Sai dấu chỗ này thì đúng tông sai tempo, hoặc ngược lại.

Đo end-to-end trong CI: beat 90 BPM đặt dưới bài 120 BPM ra đúng 120 BPM, phách
đầu lệch dưới 30ms.

### `beatgen.py` — và vì sao chọn model là một phần của câu trả lời

MusicGen là ứng viên hiển nhiên và weights của nó là **CC-BY-NC**: dùng nó để
làm nhạc phát hành được là điều chính license của thứ đang làm ra nhạc không
cho phép — tự mâu thuẫn đúng ở chỗ dễ bỏ qua nhất. Stable Audio Open được cấp
phép cho dùng thương mại theo community license của Stability, và — phần quan
trọng ở đây — được train trên Freesound và Free Music Archive, tức là trên audio
đã được cấp phép cho việc đó. Nguồn gốc của model là một phần nguồn gốc của thứ
nó làm ra.

Weights **gated** trên Hugging Face: phải có tài khoản đã chấp nhận điều khoản
và một `HF_TOKEN` đặt lên deployment. Đó là một phần của license chứ không phải
chướng ngại để đi vòng, và `load()` báo thẳng khi thiếu thay vì để 401 nổ ra từ
trong thư viện, trên GPU, giữa job của ai đó.

Prompt **không cần chính xác**, đó là hệ quả dễ chịu của việc viết `beats.py`
trước: xin 90 BPM ra 94 cũng không sao, beat sinh xong được đo lại rồi khớp,
nên mô tả chỉ cần đúng *chất* nhạc. `PROMPT_SUFFIX` gắn thêm "instrumental, no
vocals" vào mọi prompt — một giọng hát do model sinh ra nằm dưới một giọng đã
convert là thứ tệ nhất nhánh này có thể tạo ra.

### Giới hạn không prompt nào sửa được

Một loop sinh ra có thể đúng tông và đúng tempo; nó **không thể biết vòng hợp
âm** của bài nó sắp nằm dưới. Chỗ nào giọng hát đi qua các hợp âm khác nhau, một
loop ngồi trên một hợp âm sẽ chỏi ở đúng những ô nhịp đó.

Nên nó chạy tốt với rap, hip-hop và phần lớn nhạc điện tử — nơi phần nền là một
loop và phần giọng thiên về nhịp — và tệ dần khi giọng càng đi giai điệu. Đó là
tính chất của việc đặt một loop dưới một giai điệu, không phải một tham số chờ
được chỉnh. UI nói đúng câu đó ngay dưới ô mô tả.

### Ba chỗ đáng chú ý

**Đúng một nguồn beat, và chỉ `api.py` biết được điều đó.** `clean_params` thấy
prompt nhưng không bao giờ thấy upload. Có cả hai thì bị từ chối chứ không giải
quyết bằng một luật ưu tiên — không có thứ tự nào giữa "file tôi vừa tải lên" và
"beat tôi vừa mô tả" mà người dùng đoán được.

**Trạng thái `generating` là trạng thái thật.** Chỉ nhánh `beat` vào nó, và chỉ
khi beat được mô tả chứ không phải tải lên — beat tải lên không cần GPU nào và
đi thẳng sang `converting`.

**`beat` không tự dò cao độ giọng.** Bed đang được kéo về tông của người hát,
nên dịch luôn người hát là khớp hai thứ vào nhau cùng lúc.

### Test

`tests/test_analysis.py`, `tests/test_beats.py`, `tests/test_beatgen.py`, cộng
phần thêm trong `test_pipeline`, `test_api`, `test_jobs`, `test_deploy`. Tất cả
chạy trong CI không cần GPU, và cái đáng nói là **chúng đo chứ không mô tả**:
click track đúng 100 BPM thì phải ra 100 BPM, vòng hợp âm chỉ dùng nốt của Đô
trưởng thì phải ra Đô trưởng, và beat 90 BPM sau khi khớp phải đo lại được đúng
120 BPM. Dung sai không phải trang trí — 0.5% sai số tempo là nửa giây trôi trên
một bài 3 phút, nên test đòi phần mười phần trăm, và đó là lý do `_fit_grid` tồn
tại.

### Còn phải verify bằng tai và bằng GPU

- [ ] **`analysis.py` trên nhạc thật.** Toàn bộ số đo ở trên là trên tín hiệu
      tổng hợp. Click track không có swing, không có ghost note, không có đoạn
      chuyển — ba thứ làm beat tracking sai trên nhạc thật
- [ ] tông của một bài pop thật: `key_margin` có đủ lớn để dịch tông không, hay
      thực tế nó luôn dưới ngưỡng và beat không bao giờ được dịch
- [ ] `PULSE_FLOOR = 50` trên một bản ballad phối thưa — đây là chỗ dễ báo
      "không có nhịp" nhất, và hậu quả là beat giữ nguyên tempo của nó
- [ ] beat 90 BPM dưới bài 128 BPM: tỷ lệ 1.42 là gần mép của khoảng gập, nghe
      còn ra nhạc không
- [ ] **`beatgen.py` chưa từng chạy một lần nào** — weights gated, cần `HF_TOKEN`
      và một GPU. `stable-audio-tools==0.0.16` được gọi theo đúng ví dụ của
      Stability, nhưng gọi đúng không phải là đã chạy
- [ ] beat sinh ra có thật sự không có giọng hát trong đó không —
      `PROMPT_SUFFIX` là một lời nhắc cho model, không phải một ràng buộc
- [ ] điểm lặp: cắt tròn ô nhịp có đủ để mối nối không nghe thấy không, hay còn
      cần crossfade ở chỗ lặp
- [ ] rap/hip-hop so với ballad — giả thuyết "loop hợp với nhạc nhịp, chỏi với
      nhạc giai điệu" cần được nghe chứ không phải được lập luận

---

## Phase 12 — Phối lại bài gốc

Phase 11 cho hai cách thay nhạc nền và cả hai đều **không phải cùng một bài**:
beat tải lên là nhạc của người khác, beat sinh ra là nhạc chưa từng có. Yêu cầu
còn thiếu là cái ở giữa — *"đổi từ beat gốc sang beat khác tone y chang nhưng
phối nhạc khác"*: giữ nguyên tông và vòng hợp âm của bài, thay toàn bộ tiếng.

### Nó gỡ được gì, và không gỡ được gì

Nói trước vì cả tính năng xoay quanh điểm này, và vì UI không phải chỗ giải
thích luật.

Dựng lại bản phối gỡ được quyền **bản ghi**: không còn một mẫu nào của master
gốc trong output, nên fingerprint của *bản thu* không còn gì để khớp.

Nó **không** gỡ được quyền **tác phẩm**. Hợp âm vẫn là hợp âm của họ và giọng
bên trên vẫn hát đúng giai điệu của họ — đó chính là định nghĩa của một bản
cover, và là thứ publisher claim. Cái được thật sự hẹp hơn chữ "tránh bản
quyền" nhiều: **cover thì xin license được, rẻ, và ở nhiều nơi là license bắt
buộc** (bên giữ quyền không được từ chối); còn license bản ghi thì tuỳ hứng
người giữ và thường là không.

Đó là một bước tiến thật. Nó không phải là miễn phí.

### `chords.py` — đọc vòng hợp âm, và một quyết định làm nó dùng được

Chroma từng nửa ô nhịp → so với template các hợp âm ba nốt → gộp các đoạn giống
nhau. Phần đáng nói không phải thuật toán mà là **giới hạn danh sách**.

Một bộ nhận diện hợp âm tổng quát chọn trong 24 hợp âm và đúng một tỷ lệ khá;
những cái nó sai thì sai **tuỳ tiện**, và một hợp âm lệch nửa cung nằm dưới
giọng hát là một cái còi báo động. Giới hạn trong **bảy hợp âm diatonic của tông
đã đo**, bộ nhận diện đang chọn giữa những hợp âm đều *thuộc về* bài — nên cái
sai của nó nằm trong tông: đoán vi trong khi thật ra là IV thì hai trong ba nốt
vẫn đúng, nghe như một cách phối lại chứ không như một lỗi.

Đó là đánh đổi có chủ ý: ít lựa chọn hơn, đổi lấy việc không có kiểu sai thảm
hoạ nào. Bài đi ra ngoài tông (một IV thứ mượn, một dominant phụ) sẽ nhận hợp âm
diatonic gần nhất — đúng bằng thứ một nhạc công đọc chart trong tông đó sẽ chơi.

**Không chắc thì trả về chart rỗng.** Dưới `MIN_CONFIDENCE`, `detect` trả về
rỗng và `arrange` đọc đó là "chỉ chơi trống và bass". Trống nằm dưới giọng thì
không thể sai tông; hợp âm đoán bừa thì rất có thể.

Đo trong CI: `C Am F G` ra `C Am F G`, `Am F G Am` ra `Am F G Am`, và mọi hợp âm
tìm được đều nằm trong tông đã đo.

### `arrange.py` — chơi lại chart đó bằng tiếng tự tổng hợp

Trống, bass, hợp âm, pad — mọi mẫu đều sinh từ số học trong file này, không một
sample nào copy từ đâu.

Toàn bộ là **cộng hợp (additive) hoặc tạo hình nhiễu trong miền tần số**, không
có bộ lọc hồi tiếp nào, và lý do là số học chứ không phải khẩu vị: một filter
cộng hưởng là vòng lặp hồi tiếp từng mẫu, mà trong numpy nghĩa là một vòng lặp
Python chạy qua một triệu mẫu. Chọn thẳng biên độ từng hoạ âm cho ra đúng phổ đó
trong một biểu thức vector hoá — và không bao giờ aliasing, vì không có gì trên
Nyquist được sinh ra để mà gập xuống.

Vài chỗ đáng ghi lại:

* **Kick là một sin có cao độ rơi từ 110 Hz xuống 45 Hz.** Viết sweep thành
  `sin(2πf(t)·t)` là sai — nó bẻ gấp đôi quãng cần bẻ. Phải tích phân tần số rồi
  mới lấy sin.
* **Voicing được kéo về một quãng tám cố định.** Không thì chart ở B được chơi
  cao hơn chart ở C đúng một quãng tám, nghe rõ và chẳng liên quan gì tới âm
  nhạc.
* **Reverb là vài tap trễ chứ không phải convolution.** Một IR 26k mẫu chập với
  bài 4 phút là FFT 16 triệu điểm và một phần tư gigabyte; bốn phép nhân-cộng ở
  các độ trễ cố định là số lẻ, và trên một bản nền nằm dưới giọng thì nghe gần
  như nhau.
* **Swing đẩy các nốt móc kép lẻ trễ lại**, và mỗi ô nhịp thứ tư có thêm một
  cặp snare. Hai dòng code, và là gần hết phần khác nhau giữa một cái loop và
  một bản phối.
* **Năm kiểu phối, khoảng tempo rời nhau.** `auto` chọn theo tốc độ đã đo — 72
  BPM muốn ballad, 150 thì không. Khoảng rời nhau và phủ kín để `choose_style`
  có đúng một câu trả lời, không phụ thuộc vào việc ai đó chèn style vào giữa
  dict.

### Trần chất lượng, nói thẳng

Thứ ra lò là một **bản phối lập trình sạch sẽ**, không phải một bản production.
Tổng hợp cộng hợp và vài cú nhiễu qua mạng trễ cho ra thứ sạch, đúng tông, đúng
nhịp; nó không cho ra tiếng trống của một bản thu ai đó ngồi mix một tuần.

Với hip-hop, lo-fi và mọi bài mà phần nền chỉ để đỡ giọng thì dùng được thật.
Với bài mà *bản phối* mới là cái hay, nó sẽ nghe đúng như bản chất của nó.

### Ba nguồn beat, và tại sao phải gọi tên

`beat_source` là một lựa chọn có tên chứ không suy ra từ trường nào được điền:
`remake` **không gửi cả file lẫn mô tả**, nên không có gì để suy ra. Mỗi nguồn có
điều kiện riêng và `api.py` kiểm — vì chỉ tầng đó nhìn thấy file upload:

| Nguồn | Cần | Từ chối |
|---|---|---|
| `upload` | file beat | thiếu file |
| `generate` | mô tả | thiếu mô tả, hoặc gửi kèm cả file |
| `remake` | không gì cả | gửi kèm file (nếu bỏ qua im lặng thì người dùng sẽ ngồi hỏi sao không nghe thấy beat mình vừa tải lên) |

### Sửa kèm: allowlist của audit đã bỏ sót từ Phase 8

`audit.FIELDS` là danh sách trắng, và `language`, `emotion` chưa bao giờ nằm
trong đó — nên từ Phase 8 tới giờ mọi dòng audit đều ghi tên chúng vào `dropped`
thay vì ghi giá trị. Nó suy biến an toàn đúng như thiết kế, nên không ai thấy;
phát hiện ra khi các field mới của Phase 11 và 12 rơi vào đúng cái bẫy đó. Đã
thêm `cfg`, `clarity`, `profile`, `language`, `emotion`, `beat_bytes`,
`beat_source` — toàn là *cài đặt*, không có gì là nội dung của người dùng.

### Test

`tests/test_chords.py`, `tests/test_arrange.py`, cộng phần thêm trong
`test_pipeline` và `test_api`. Không cần GPU, và vẫn đo chứ không mô tả: dựng
một ô nhịp C-E-G thì phải đọc ra C, mọi hợp âm tìm được phải nằm trong tông đã
đo, bản phối dựng ra phải đo lại được đúng tempo đã cho, và chart rỗng phải cho
ra một bộ trống thật chứ không phải im lặng.

### Còn phải verify bằng tai

- [ ] **`chords.py` trên nhạc thật.** Vòng hợp âm tổng hợp không có đảo âm,
      không có nốt ngoài hợp âm, không có phần đệm — ba thứ làm chord detection
      sai trên bản thu thật. Đây là mục quan trọng nhất
- [ ] tỷ lệ bài thật rơi xuống dưới `MIN_CONFIDENCE` — nếu đa số thì tính năng
      thực tế là "trống và bass", và phải nói lại điều đó trong UI
- [ ] **bản phối nghe có dùng được không.** Đây là câu hỏi quyết định phase này
      đáng giữ hay không, và không có cách nào trả lời bằng test
- [ ] năm kiểu phối: khoảng tempo có chia đúng chỗ không, hay `auto` chọn sai
      trên bài thật
- [ ] mức giữa các bè (`KICK_GAIN` … `PAD_GAIN`) — đặt bằng suy luận, chưa bằng
      tai, và bản nền phải nằm *dưới* giọng chứ không tranh chỗ
- [ ] chỗ nối khi chart lặp lại: hợp âm cuối về hợp âm đầu có nghe thành một
      vòng không, hay nghe như bị cắt
- [ ] thời gian dựng trên bài 4-5 phút thật — test đo 60 giây mất dưới 20 giây,
      nhưng nó chạy trong bước `mixing` trên container CPU cùng lúc với những
      việc khác
- [ ] swing 0.18 của lo-fi có quá tay không

---

## Phase 13 — "Đổi beat" tách khỏi "đổi giọng"

Phase 11 và 12 để việc thay nhạc nền nằm trong mode `beat`, mà `beat` là mode
**đổi giọng có thay nền**. Hệ quả: muốn đổi mỗi cái beat thì vẫn phải nộp giọng
mẫu, vẫn phải tick một câu cam kết về quyền sử dụng giọng của người khác, và vẫn
phải trả một lượt GPU chuyển giọng — cho một việc không đụng gì tới giọng.

Nên `rebeat` là một mode riêng: **tách nền, giữ nguyên người hát, thay nhạc**.

### Mode duy nhất không convert gì cả

```
rebeat:  input ──► separate ──► vocal ─────────────────────► mix ──► output.mp3
                        └────► instrumental ──► (đo BPM/key) ↑ beat mới
```

`queued → separating → [generating] → mixing → done`, và không có `converting`
trong đó vì không có gì để convert. Rẻ hơn hẳn một chặng GPU, và ngắn hơn đúng
bằng mọi bước có thể hỏng trong chặng đó.

Ba nguồn beat của Phase 11–12 dùng lại y nguyên (tải lên / tự sinh / phối lại),
vì `_beat_bed` và `_generate_beat` được tách ra dùng chung cho cả hai nhánh.
Bản copy-paste của chúng sẽ là bản mà một trong hai nhánh lặng lẽ ngừng được
đóng dấu watermark.

### `CONVERSION_MODE` không có `rebeat`, và đó là chủ ý

Chỗ dễ làm sai nhất của phase này là nhét một entry giả vào bảng ánh xạ cho đủ
bộ. `rebeat` **không có** conversion mode vì nó không convert, và một
placeholder ở đó là một lời nói dối mà `clean_params` sẽ đọc như một câu trả
lời thật.

Nên bảng chỉ chứa các mode thật sự convert, `CONVERTING_MODES` là danh sách
đó, và `clean_params` chỉ thêm `semitone_shift`, `diffusion_steps`, `cfg_rate`,
`voice_profile` cho những mode nằm trong nó. Job `rebeat` không mang những số
đó — vì ghi vào thì `/status` sẽ báo cáo những con số không ai đọc.

### Giọng mẫu: không cần, và bị từ chối nếu gửi

`reference` thành optional trong chữ ký và bắt buộc theo từng mode ở thân hàm,
nên thiếu nó là **400 nói rõ mode nào cần** thay vì 422 về một form field.

Gửi giọng mẫu cho `rebeat` thì bị từ chối chứ không bị bỏ qua. Bỏ qua im lặng
chính là cách một người ngồi hết cả lượt chạy để tự hỏi sao giọng không đổi.

### Câu cam kết đổi theo mode

Câu đồng thuận cũ là *"tôi có quyền sử dụng giọng nói trong file tham chiếu"*.
Với `rebeat` thì không có file tham chiếu nào — bắt ai đó xác nhận quyền với
một file họ chưa từng được hỏi là một cái checkbox không ai tick thật lòng
được, và một cái gate không ai đọc thì không phải gate.

Nên `rebeat` hỏi câu đúng với việc nó làm: *"tôi có quyền sử dụng bản ghi này"*.
Cổng vẫn nằm ở backend và vẫn từ chối job không kèm cam kết — đổi là đổi câu
chữ, không phải đổi luật.

### UI

Sáu mode giờ chia thành hai cặp rõ ràng:

| Mode | Giọng | Nhạc nền |
|---|---|---|
| Bài hát | đổi sang giọng mẫu | giữ nguyên |
| **Đổi beat** | **giữ nguyên** | **thay** |
| Đổi beat + giọng | đổi sang giọng mẫu | thay |
| Giọng hát | đổi sang giọng mẫu | không có |
| Giọng nói | đổi sang giọng mẫu | không có |
| Văn bản | đọc bằng giọng mẫu | không có |

Bước "Giọng mẫu" bị **ẩn** ở `rebeat` chứ không phải disable: backend từ chối
reference gửi tới mode đó, nên một ô trống ở đây là lời mời tới một cái 400. Và
trong "Tinh chỉnh", dịch cao độ / chất lượng / bám giọng mẫu đều biến mất — còn
lại đúng hai thứ áp dụng cho mọi bản mix: độ trong và âm lượng giọng.

### Test

Thêm trong `test_api`, `test_pipeline`, `test_jobs`, `test_deploy`: `rebeat`
chạy được không cần reference, từ chối reference gửi kèm, không mang một tham số
chuyển giọng nào, vẫn cần cam kết, vẫn tách nền, vẫn watermark — và
`CONVERSION_MODE` là tập con thật sự của `JOB_MODES` với đúng một phần tử thiếu.

### Còn phải verify

- [ ] chạy thật một job `rebeat`: thời gian có ngắn hơn `beat` đúng bằng chặng
      GPU đã bỏ không
- [ ] giọng gốc đi thẳng từ separator vào mix — chuỗi `enhance` giờ chạy trên
      output của separator chứ không phải của Seed-VC, và hai thứ đó có
      artefact khác nhau. Có thể mặc định `Độ trong` cho mode này phải khác
- [ ] `vocal_gain_db` mặc định 0: giọng gốc đã được mix sẵn trong bài, nên tỷ lệ
      giọng/nền có thể lệch so với khi giọng là bản convert

---

## 13.1 — Một image hỏng chặn ba phase

Phase 11, 12 và 13 đều đã merge và **không có phase nào lên production**. API
thật vẫn trả lời bằng code của Phase 10, nên một job `rebeat` gửi lên nhận
`422`: phiên bản đó chưa biết mode ấy, và ở đó `reference` còn là field bắt
buộc nên FastAPI từ chối ngay khi validate, trước cả khi vào thân hàm.

Nguyên nhân là một dòng trong log deploy:

```
failed to run builder command "python -m pip install einops==0.8.0
  'protobuf>=3.20,<7' stable-audio-tools==0.0.16": container exit status: 1
Image build for im-O19YwE2Ch5GisKJiWZjztC failed.
```

`stable-audio-tools` là một package **training**: nó kéo theo
pytorch-lightning, wandb, gradio, encodec, laion-clap, k-diffusion và hơn chục
thứ nữa, vài trong số đó tự pin torch — trên một base image đã có torch 2.4.0
và numpy<2. Gỡ được chỗ đó là một việc thật, và nó cần output pip thật
(`modal image logs <id>`) chứ không phải một phỏng đoán.

### Bài học là về bán kính ảnh hưởng, không phải về pip

`modal deploy` build **mọi image đã đăng ký trong một lượt**. Nên một image
hỏng không làm hỏng function của chính nó — nó làm hỏng cả lần deploy, và kéo
theo mọi thay đổi không liên quan trong cùng push. Ba phase code chạy được nằm
im sau một tính năng chưa ai từng nhìn thấy build xong.

Sửa bằng cấu trúc chứ không bằng cách vá pin:

* `beatgen.enabled()` đọc `BEAT_GENERATOR`, mặc định **tắt** — ngược với
  `watermark.enabled()`, và ngược có lý do: watermark bật trừ khi tắt đi, còn
  cái này tắt trừ khi có người bật, vì chưa ai xem image của nó build xong.
* `BeatGenerator` không còn `@app.cls` ở module scope. Nó được gắn vào App bên
  trong `register()`, và `deploy.py` chỉ gọi `register()` khi cờ bật.
* `beatgen_image` cũng thành một hàm. Modal chỉ build image mà object đã đăng
  ký tham chiếu tới, nên một image mồ côi *lẽ ra* bị bỏ qua — nhưng "lẽ ra bị
  bỏ qua" là một khẳng định về nội tại của người khác, mà thứ vừa hỏng chính là
  một lần deploy chết vì image không ai cần. Không tạo ra nó là một bảo đảm;
  không gắn nó vào là một kỳ vọng.
* `api.submit` từ chối `beat_source=generate` bằng 400 có câu chữ, thay vì để
  job chạy ba phút rồi chết vì không có container nào.
* UI **ẩn** nút "Tự sinh beat" khi cờ tắt — không phải disable, vì backend từ
  chối thẳng nguồn đó và một nút xám là quảng cáo cho thứ không có.

Hai test giữ cho nó không tái diễn: `BeatGenerator` **không** được đăng ký
trong một deploy chưa bật cờ, và nó **có** được đăng ký khi bật.

### Bật lại khi nào

Khi ai đó ngồi gỡ `BEATGEN_REQUIREMENTS` và **nhìn thấy image build xong**.
Lúc đó: `BEAT_GENERATOR=1` và `HF_TOKEN` trên deployment, và
`BEAT_GENERATOR_ENABLED = true` trong `web/lib/params.ts`.

Hai nguồn beat còn lại — tải lên và phối lại — không cần model nào, không cần
image nào, và chạy được ngay.

---

## 13.2 — Lần đầu nghe trên nhạc thật

Một bản Blue Bird chạy qua mode đổi beat, đặt cạnh bản phối rock của Ryu No
Kage. Nhận xét của người dùng: *"beat hoàn toàn lạc quẻ"*, và *"giọng AI người
ta trong veo, còn bản mình thì…"*. Đo hai file cạnh nhau thì thấy cả hai đều
đúng, và số đo chỉ thẳng vào chỗ hỏng:

| | Ryu No Kage | Bản của app |
|---|---|---|
| Rolloff 99% | 9 324 Hz | **4 027 Hz** |
| Năng lượng < 120 Hz | 23.5 % | **55.5 %** |
| 500 Hz – 2 kHz | 37.7 % | 19.6 % |
| 2 – 6 kHz (độ rõ chữ) | 16.0 % | **5.2 %** |
| Đỉnh | 0.763 | **1.280** (+2.1 dBFS) |
| Tempo | ~154 BPM | **76 BPM** |

### Gỡ "Phối lại bài gốc"

Phase 12 dựng lại bản phối bằng tổng hợp cộng hợp — cộng các hoạ âm hình sin.
Thứ nó bị đem ra so là một bản phối rock có guitar điện, trống thật, bass thật.
Khoảng cách đó không phải khoảng cách tinh chỉnh: nó là khoảng cách giữa *sinh
sóng bằng số học* và *một thư viện sample thu từ nhạc cụ thật*, và không có hằng
số nào sửa được.

README của Phase 12 đã viết đúng cái trần đó ("một bản phối lập trình sạch sẽ,
không phải một bản production"). Nhưng viết trần vào README không cứu được việc
nó được đưa ra như một lựa chọn ngang hàng trong UI, và người dùng có lý khi
mong đợi hơn thế. Nên nó bị **gỡ**, không phải đánh bóng: `arrange.py`,
`chords.py` và test của chúng đi hẳn.

Còn lại hai nguồn beat, và `upload` mới là nguồn chạm được tới cái bar đó — vì ở
đó bản phối là do người làm ra. App chỉ làm phần nó làm được: đo, khớp tông và
nhịp, ghép.

### Sửa lỗi tempo 3:2 — phần phải sửa dù có gỡ hay không

`analysis.py` là nền của **cả** đường tải beat lên, nên lỗi của nó không đi cùng
`arrange.py`.

Trên bản rock 154 BPM, autocorrelation đỉnh gần bằng nhau ở chu kỳ phách và ở
**một rưỡi** lần chu kỳ đó — backbeat đặt onset mạnh lên cả hai lưới. Chấm điểm
cũ chọn 103.4 BPM (score 0.4817) thay vì 152.0 (0.4774): **sai vì 0.9%.**

Và sai 3:2 là kiểu sai không sống chung được. Hai ô nhịp của bed ở 103 trải dài
đúng ba ô nhịp của bài ở 154 — không phải trôi dần, mà là khác nhịp hẳn. Sai
quãng tám thì lành hơn nhiều: bed ở nửa tốc độ rơi đúng vào phách chẵn, đó là
half-time, và `beats.fold_tempo` hấp thụ nó sẵn rồi.

Cách sửa là chấm điểm mỗi ứng viên **kèm các bội của nó**. Chu kỳ đúng có điểm
tựa ở 1x, 2x, 3x, 4x; một lag dài gấp rưỡi chỉ chia được các bội chẵn nên các
bội lẻ sụp. Trên đúng bài đó: 152 được 0.506/0.485/0.378/0.504, còn 103 được
0.493/0.378/0.252/0.505.

Nó **không** đụng tới nhập nhằng quãng tám, và như thế là đúng: mọi bội của 2P
cũng là bội của P, nên không phép chấm điểm nào phân biệt được, và không nên giả
vờ là được.

**Một cái bẫy trong chính cách sửa đó**, đã cắn một lần và giờ có test riêng:
lag là số nguyên khung còn chu kỳ thì không. Chu kỳ thật 21.53 khung được tìm ở
lag 22, mà bội hai của lag 22 là 44 trong khi đỉnh thật ở 43. Ứng viên dài gấp
đôi (lag 43) thì các bội 86, 129, 172 lại khớp đẹp — nên chấm điểm ở đúng bội
nguyên **thưởng cho lag dài một cách có hệ thống**, và bản đầu tiên biến click
track 120 BPM thành 60. Phải tìm mỗi bội trong một cửa sổ rộng `ceil(k/2)`, đúng
bằng quãng đường mà nửa khung làm tròn đi được tới bội thứ k.

Kết quả trên chính bản Ryu No Kage: **153.0 BPM** (trước là 102.7), và 8/11 đoạn
20 giây đọc đúng thay vì lật liên tục.

### Còn nợ

- [ ] 55% năng lượng dưới 120 Hz và clipping +2.1 dBFS là số đo của bản **có
      bed tổng hợp**. Phải đo lại một bản đường `upload` để biết `mixing` và
      `enhance` có phần lỗi trong đó không, hay tất cả là do bed
- [ ] margin của phép đo tông trên nhạc thật là 0.003–0.09, ngưỡng đang là 0.04.
      Guard hiện chạy đúng (không dịch tông khi không chắc) nhưng nó đúng vì
      may, không phải vì tin cậy
- [ ] 3/11 đoạn vẫn đọc 103 BPM. Cả bài thì đúng, nhưng nếu sau này cần đo theo
      đoạn thì chưa đủ

---

## 13.3 — Một dòng, ba phase

Sau khi gỡ "Phối lại", app không còn cách nào **tự** làm ra beat: nguồn duy nhất
còn lại là người dùng tải beat lên. Phản hồi đúng và thẳng: *"sao bảo tôi đưa
beat khác vào, bạn phải phối chứ"*.

Thứ chặn đường đó là image của `beatgen` — và nguyên nhân, khi cuối cùng chạy
được trình giải phụ thuộc, là một dòng:

```
ERROR: Cannot install einops==0.8.0 and stable-audio-tools==0.0.16
The conflict is caused by:
    stable-audio-tools 0.0.16 depends on einops==0.7.0
```

`einops` bị ghim ở 0.8.0 trong cùng lệnh cài, trong khi package ghim chính nó ở
0.7.0. Không có gì để giải. Và vì `modal deploy` build mọi image trong một lượt,
một dòng đó kéo theo cả Phase 11, 12 và 13.

### Luật rút ra, viết thành test

**Danh sách này chỉ được ghim những thứ `base_image` đã giữ.** Package tự ghim
phụ thuộc của nó, và package thắng. `test_beatgen.py` giờ từ chối bất kỳ tên nào
ngoài `stable-audio-tools`, `torch`, `torchaudio`, `transformers`.

Ba cái còn lại có lý do riêng, đã đo bằng `pip install --dry-run`:

* **torch/torchaudio** lặp lại ở đúng version của `base_image`. Để yên thì trình
  giải lấy torchaudio 2.11 đặt cạnh torch 2.4 — cài được và không chạy được.
* **transformers 4.46.3** vì lý do `tts.py` đã viết dài: 5.x đòi torch ≥ 2.5, gặp
  2.4 thì in **một dòng** vào log build rồi đi tiếp với mọi model class thành
  stub. Stable Audio Open điều kiện hoá bằng T5, nên cái stub đó là cả tính năng.
* **protobuf ở layer riêng**, sau. `descript-audiotools` chặn protobuf dưới 3.20
  cho một logger không ai dùng, còn agent của Modal cần ≥ 3.20. Cùng một
  `pip_install` thì trình giải phải thoả cái chặn; ở layer sau thì nó đè lên.
  `conversion.py` đã làm đúng thế cho seed-vc và mang bản dài của câu chuyện.

### Cờ tính năng: hỏi backend, đừng ghim vào bundle

`BEAT_GENERATOR_ENABLED` từng là hằng số trong `web/lib/params.ts` — hai cờ, ở
hai nơi, phải lật cùng lúc. Một UI mời một nguồn mà API từ chối còn tệ hơn việc
một trong hai cờ sai.

Giờ `/health` trả về `beat_generator`, và **`/api/capabilities` hỏi hộ trình
duyệt** một lần mỗi lần tải trang.

Bản đầu tiên để chính trình duyệt gọi `GET ${apiBase}/health`, và nó im lặng
đúng một vòng: cờ đã bật, image đã build, `/health` trả `true` — mà trang vẫn
chỉ hiện ô kéo-thả file. Một request cross-origin có hai kiểu chết, CORS và
mạng, mà cả hai rơi vào cùng cái `catch` với "bản này không có generator". Ba
sự thật khác nhau, một chữ `false`, không cách nào phân biệt từ trong UI.

Nên phép thử chuyển về phía server: trình duyệt gọi cùng origin, không
preflight, không cache chen vào, còn phía server gọi Modal thẳng.

**Và nó là route riêng, không nhét vào `/api/config`.** Đây là chỗ suýt sai lần
thứ hai. `submit` chờ `/api/config` trước khi đẩy byte đầu tiên lên, nên mọi thứ
chậm thêm vào đó là thời gian chết giữa lúc bấm nút và lúc upload chạy. Mà
`api()` không đặt `min_containers`: container nguội thì `/health` trả lời sau
hàng chục giây. Ghép hai thứ vào một route thì hoặc deadline ngắn (giết đúng
lần mở trang đầu tiên trong ngày) hoặc deadline dài (giết upload). Tách ra thì
không ai phải chọn: `/api/capabilities` không có gì chờ nó ngoài cái nút nguồn
beat, nên nó được 12 giây và hai lần thử — lần đầu chính là lần đánh thức
container.

Probe hỏng thì `beatGenerator: false`, và route này không có mã lỗi nào cả:
nguồn "tải beat lên" vẫn chạy được mà không cần generator, nên chẳng có gì để
trình duyệt làm với một cái 5xx ngoài việc ẩn đúng cái nút nó vốn sẽ ẩn.

Bản triển khai cũ quá đến mức không trả lời được thì coi như không có generator
— cùng một kết luận, bằng đường khác.

Và nguồn beat mặc định đổi thành **"Máy làm beat"**, vì đó là điều mode này hứa.
`effectiveBeatSource()` kẹp nó về `upload` ở nơi không có generator — tính ra chứ
không ghi ngược vào state, để không có khoảnh khắc nào form tin một đằng và
request gửi một nẻo.

### Cái này vẫn không phải "phối lại bài của bạn"

Nói rõ vì đây là chỗ dễ hiểu nhầm nhất, và UI cũng nói đúng câu này: Stable Audio
Open **sáng tác theo mô tả**, nó không đọc vòng hợp âm bài gốc. Nó cho ra nhạc
thật — nhạc cụ thật, bản phối thật, khác hẳn máy đánh trống của Phase 12 — rồi
`beats.py` kéo về đúng tốc độ và tông của bài. Nhưng nó không đi theo các đoạn
chuyển hợp âm.

Hợp với rap, hip-hop, nhạc điện tử. Bài mà giọng đi giai điệu nhiều thì vẫn chỏi
ở những ô nhịp đổi hợp âm, và không prompt nào sửa được điều đó.

*(Phase 14 sửa đúng câu này. Đoạn trên giữ nguyên vì nó vẫn đúng với nguồn
`generate`.)*

### Bật "máy làm beat" trên deployment

Hai biến, đặt trong **Settings → Secrets and variables → Actions** của repo:

| Tên | Đặt ở tab | Giá trị |
|---|---|---|
| `BEAT_GENERATOR` | **Variables** | `1` |
| `HF_TOKEN` | **Secrets** | token đọc của Hugging Face |

Token lấy thế này, và **thứ tự quan trọng** — chấp nhận điều khoản trước, tạo
token sau:

1. đăng nhập huggingface.co, mở `stabilityai/stable-audio-open-1.0`, bấm nút
   chấp nhận điều khoản trên trang model (weights là gated, không có bước này
   thì token hợp lệ vẫn nhận 401);
2. Settings → Access Tokens → New token, quyền **read** là đủ;
3. dán vào `HF_TOKEN` ở tab Secrets.

Rồi chạy lại workflow **Deploy Modal** (Actions → Deploy Modal → Run workflow),
hoặc push bất cứ gì vào `main`.

Nếu deploy đỏ: xoá biến `BEAT_GENERATOR` đi là mọi thứ còn lại trở về xanh ngay.
Đó là lý do cái cờ tồn tại.

### Còn phải verify

- [ ] **Image có build được thật không.** `pip install --dry-run` xanh với đúng
      bộ pin này, và đó mới là *giải được phụ thuộc* — chưa phải là *build xong*
      trên Modal. Cờ `BEAT_GENERATOR` vẫn mặc định tắt đúng vì lý do đó: bật lên,
      deploy, nếu đỏ thì tắt lại và không kéo theo gì khác
- [ ] `HF_TOKEN` phải có, weights của Stable Audio Open là gated
- [ ] beat sinh ra có thật sự không có giọng hát trong đó không
- [ ] nghe thử: nó có hơn hẳn máy đánh trống Phase 12 không, và hơn tới đâu

---

## Phase 14 — Phối lại bài, không phải sáng tác cạnh bài

Phase 13.3 làm cho "máy làm beat" hiện ra được. Rồi câu hỏi tiếp theo, đúng và
đau:

> *"tôi bảo là dựa trên beat gốc thì bạn tạo ra beat mới khác khác nhưng vẫn hợp
> tone, mà cái này là như nào"*

Đúng. Nút đó chưa bao giờ làm việc ấy.

```
Trước:  bạn gõ mô tả ──► model sáng tác từ số 0 ──┐
        bài gốc ──► đo BPM + tông ───────────────┴──► beats.fit kéo cho khớp
                    (model chưa từng nghe bài)
```

Nó **hợp tông** thật, nhưng chỉ vì `beats.fit` ép nó khớp *sau khi* sinh xong.
Model không biết bài đang chạy hợp âm gì, nên ở mỗi ô nhịp đổi hợp âm nó chỏi —
đúng cái giới hạn mà mục "Cái này vẫn không phải phối lại bài của bạn" ở trên đã
ghi ra và coi là không sửa được.

Sửa được. Chỉ là chỗ sửa không nằm ở prompt.

### `init_audio`: cho sampler xuất phát từ nhạc chứ không từ nhiễu

`generate_diffusion_cond` trong `stable-audio-tools` nhận `init_audio` và
`init_noise_level`. Cho vào một đoạn audio, sampler bắt đầu từ bản đã nhiễu hoá
của đoạn đó thay vì từ nhiễu thuần: đầu ra **giữ hoà thanh, giữ tốc độ, giữ vạch
ô nhịp** của đầu vào, còn nhạc cụ thì được vẽ lại từ đầu.

Đọc trong code của thư viện, `init_noise_level` chính là `sigma_max` cho lần
chạy đó:

```python
if init_audio is not None:
    sampler_kwargs["sigma_max"] = init_noise_level
```

Nên nó là **một cái núm duy nhất quyết định giữ lại bao nhiêu**. Sinh thường
dùng 500 (không giữ gì); UI variation của chính thư viện mặc định 0.1 (giữ gần
hết).

Câu hỏi thật sự không phải "dùng `init_audio` hay không". Là **đưa cái gì vào**.

### Hai đường, và chúng khác nhau ở luật chứ không ở nhạc

Cách hiển nhiên: tách nhạc nền bài gốc ra rồi đưa thẳng vào. Nó chạy, và cái ra
là **tác phẩm phái sinh của chính bản ghi** — đúng thứ mà cả nhánh "đổi beat"
sinh ra để tránh, và đúng kiểu biến đổi mà hệ thống nhận diện audio được xây để
xuyên qua. Nhạc hay hơn, bản quyền tệ hơn.

Đường vòng phá được chuỗi đó:

```
nhạc nền ──► chords.detect ──► Chart (vòng hợp âm = phần *sáng tác*)
                                  │
                          sketch.render ──► synth thô, xấu, đúng hợp âm
                                  │
                       init_audio ──► Stable Audio Open ──► nhạc cụ thật
                                  │
                             beats.fit ──► khớp bài
```

`chords.detect` đọc ra một **chuỗi hợp âm**, tức là phần sáng tác chứ không phải
bản ghi. `sketch.render` đánh chuỗi đó bằng oscillator chưa bao giờ nghe bài.
Thứ chạm tới model là audio do repo này sinh ra. Đầu ra vì thế là một bản
**cover** — quyền tác giả, thứ xin phép được và ở nhiều nơi là cưỡng chế cấp
phép — chứ không phải bản sao master của ai.

Đường thẳng vẫn có, sau một ô tick, mặc định tắt, và UI gọi đúng tên nó.

### `sketch.py` — và vì sao `arrange.py` bị xoá lại quay về được

Phase 13.2 xoá `arrange.py` với kết luận thẳng: additive synthesis ra máy đánh
trống, đem so với bản phối rock người làm thì thua bằng một khoảng cách không
tune được.

Kết luận đó vẫn đúng. Nhưng nó là kết luận về **synth làm sản phẩm cuối**.

Sau diffusion thì synth không còn là sản phẩm. Nó là *câu lệnh*. Không ai nghe
`sketch.py`. Nó chỉ phải **nói rõ hợp âm và vuông vắn nhịp**, rồi biến đi. Đó là
một cái bar hoàn toàn khác, và là bar mà bốn oscillator đạt được.

Ba chỗ `sketch.py` khác `arrange.py`, mỗi chỗ vì công việc đã đổi:

* **Không reverb.** Đuôi vang là texture, mà texture chính là thứ model được
  giao vẽ lại. Reverb ở đây sẽ bị bôi vào đầu ra như một thứ cần giữ.
* **Không pad, và hợp âm đẩy lên trước.** `arrange.py` để chord ở 0.28 dưới kick
  0.9 vì bed thật để vậy. Ở đây gần như ngược lại — `CHORD_GAIN` 0.85 trên
  `KICK_GAIN` 0.42. Model không nghe rõ triad thì nó tự bịa một cái.
* **Mono, không swing, không fill.** `prepare_audio` tự resample và đổi kênh nên
  stereo là công toi; còn swing với fill là để loop bớt giống máy, mà cái này
  được phép giống máy.

**Test mạnh nhất của module này là một vòng khứ hồi.** Dựng chart Am–F–C–G, cho
`chords.detect` đọc lại chính bản render (qua đúng đường decode 22.05 kHz thật,
có bass kêu bên dưới), và đòi lấy lại đúng bốn hợp âm đó. Nếu bộ dò hợp âm không
đọc nổi sketch thì model cũng không, và cả đường vòng qua `chords.detect` không
mua được gì.

```
tests/test_sketch.py::test_a_chord_recogniser_gets_the_chart_back_out_of_the_sketch
```

### Hai giá trị nhiễu, và chúng chưa được đo

`SKETCH_NOISE_LEVEL = 65`, `ORIGINAL_NOISE_LEVEL = 28`.

Sketch là bốn oscillator: gần như không có gì trong đó nên sống sót, chỉ hoà
thanh, tốc độ và vị trí vạch nhịp — nên nó ở gần đầu tự do. Nhạc nền gốc đã là
bản phối thật; vẽ lại mạnh tay thế thì vứt mất chính thứ nó được đưa vào để giữ
— nên nó thấp hơn nhiều.

**Không con số nào trong hai con số này được đo.** Từ đây không đo được: muốn
biết thì cần một GPU và một đôi tai. Chúng là tham số, và ghi ra đây là để lần
sau có người sửa thì biết mình đang sửa cái gì.

Sàn kẹp là `0.1` chứ không phải `0` — không phải lỗi lệch một. `sigma_max` bằng
0 thì sampler không có nhiễu nào để gỡ và trả lại nguyên đầu vào, mà trên nhánh
`original` thì "trả lại nguyên đầu vào" nghĩa là đưa lại chính bản master làm
beat mới.

### Kẹp về phía an toàn, không phải về phía mặc định

`beat_init` không hợp lệ thì rơi về `sketch`, không phải về `original`:

```python
params["beat_init"] = init if init in BEAT_INITS else DEFAULT_BEAT_INIT
```

Gõ sai, client cũ, form viết tay — mọi đường đều dẫn về nhánh không sao chép gì.
Đây là chỗ duy nhất trong repo mà một giá trị bị kẹp theo hướng *luật* chứ không
theo hướng *chạy được*, nên nó có test riêng.

### Nguồn beat: từ hai thành ba

| | Model nghe gì | Đi theo hợp âm bài | Sản phẩm là |
|---|---|---|---|
| `upload` | — | — | file của bạn, đã khớp bài |
| `generate` | không gì | không | nhạc mới, không liên quan bài |
| `derive` + `sketch` | synth của app | **có** | cover phần sáng tác |
| `derive` + `original` | nhạc nền gốc | có | phái sinh của bản ghi |

`derive` là mặc định, vì đó là điều mode này hứa. `generate` vẫn còn cho ai muốn
một beat không dính gì tới bài. `upload` vẫn là nguồn duy nhất chạm được bar
"người phối", vì ở đó có người phối thật.

### Prompt: ô trống là một câu trả lời

Trên `generate`, mô tả là toàn bộ đầu vào — trống là 400.

Trên `derive` thì không. Ở đó đã có bài, và `beatgen.describe()` viết prompt từ
đúng cái đã đo:

```
"154 BPM, key of Am, drums and bass"
```

Từ chối một job vì ô trống, trong khi phép đo đã nằm sẵn trong tay, là từ chối
dùng thứ mình vừa đo. Ô mô tả ở nhánh này chỉ còn quyết định **nhạc cụ và chất
nhạc** — hợp âm với tốc độ đã lấy từ bài rồi, và UI nói đúng câu đó.

### Test

`tests/test_sketch.py` (10) mới, `tests/test_chords.py` (12) khôi phục, cộng
thêm ở `test_beatgen.py`, `test_api.py`, `test_pipeline.py`. Những cái đáng giữ:

* bộ dò hợp âm đọc lại được chart ra khỏi sketch — hợp đồng thật với model
* `beat_init` không hợp lệ kẹp về `sketch`, kiểm cả `"ORIGINAL_"` và `"  "`
* `clamp_noise_level(0)` là `INIT_NOISE_MIN` chứ không phải 0
* `derive` không cần mô tả, `generate` thì cần
* `derive` cùng lúc gửi file beat là 400, như mọi cặp nguồn khác
* các dải tempo của `STYLES` rời nhau và phủ kín, nên câu trả lời không phụ
  thuộc thứ tự dict
* sketch để lại headroom — init mà clip thì model có méo tiếng để bắt chước

**605 passed, 3 skipped.**

### Còn phải verify bằng tai và bằng GPU

- [ ] `init_noise_level` 65 cho sketch: nghe ra vẫn còn đúng vòng hợp âm không,
      hay model đã đi mất
- [ ] `init_noise_level` 28 cho `original`: có thật sự khác bản gốc đủ nhiều
      không, hay chỉ là bản gốc bị lọc
- [ ] `chords.detect` trên nhạc thật — độ tin cậy đo được ở Phase 12 là
      0.003–0.09 với ngưỡng 0.04, tức là ngưỡng đó đang đúng *vì may*
- [ ] bài đổi hợp âm nhiều: `derive` có thật sự hết chỏi so với `generate` không
- [ ] cửa sổ 47 giây của model so với bài 3 phút — `beats.fit` lặp nó, và lặp
      một vòng 8 ô nhịp lên cả bài là điều chưa nghe thử bao giờ

---

## 14.1 — `AttributeError: 'function' object has no attribute 'remote'`

Lần đầu có người bấm chạy một beat sinh ra, và nó chết. Ba phút vào một job đã
trả tiền GPU.

```
AttributeError: 'function' object has no attribute 'remote'
```

Chỗ gây lỗi trông y hệt bốn dòng bên cạnh nó:

```python
stems = Separator(model=...).separate.remote(...)  # chạy
converted = VoiceConverter(mode=...).convert.remote(...)  # chạy
beat = BeatGenerator().generate.remote(...)  # nổ
```

### `app.cls()` không decorate tại chỗ

`@app.cls(...)` **trả về một object mới**, nó không sửa class được truyền vào.
Với ba class kia thì không thấy được vì chúng viết dạng decorator ở module
scope — tên trong module *chính là* object Modal trả về:

```python
Separator        modal.cls.Cls
VoiceConverter   modal.cls.Cls
Watermarker      modal.cls.Cls
```

`BeatGenerator` thì không. Phase 13.1 để nó **không decorate** ở module scope,
có chủ ý: đó là thứ khiến một deploy không bật generator sẽ không bao giờ build
image của nó — bài học của "một image hỏng chặn ba phase". `register()` gói nó
lại và cất vào `_REGISTERED`, còn cái tên `BeatGenerator` trong module vẫn là
một class Python trần. `BeatGenerator().generate` là bound method thường, và
`.remote` trên đó là thuộc tính chưa từng tồn tại.

Không có gì ở chỗ gọi nói ra điều đó. Bản chạy được và bản hỏng khác nhau đúng
một dòng `import`, và cả hai đều compile.

### Vì sao không phải là "cứ dùng `_REGISTERED`"

Vì trong container chạy pipeline thì `_REGISTERED` là `None`.

`register()` được gọi bởi `deploy.py`, trong tiến trình deploy.
`run_beat_pipeline` chạy ở container khác, và container đó:

```python
@app.function(image=api_image, volumes={DATA_DIR: data_vol}, timeout=PIPELINE_TIMEOUT, retries=0)
```

— **không có `secrets=`**, nên `BEAT_GENERATOR` không nằm trong env của nó, và
nó cũng không có lý do gì phải import `deploy`. Mọi cách sửa dựa vào việc
`register()` đã chạy trong container đó đều là đoán.

Nên `beatgen.generator()` dùng `_REGISTERED` khi có, còn không thì **tra cứu
theo tên trên App đã deploy**:

```python
modal.Cls.from_name(APP_NAME, BeatGenerator.__name__)
```

Class đã deploy sẵn rồi. Đây là cách một container trỏ tới class nó không tự
định nghĩa, và nó đúng bất kể container ấy import module nào.

### Test bắt được đúng cái crash đó

Cái đáng giữ là test hành vi, không phải test chính tả: `_generate_beat` chạy
thật với một `generator()` giả. Bỏ bản sửa ra thì nó dựng lại đúng câu lỗi trên
màn hình người dùng.

```
tests/test_pipeline.py::test_the_generator_is_reached_through_the_lookup_not_the_imported_class
```

Thêm một lưới chặn cho *loại* lỗi này chứ không riêng lần này: mọi entry point
GPU mà pipeline với tay tới đều phải là object do Modal tạo ra. Nếu sau này có
ai thêm một class nữa rồi quên decorate, `test_deploy.py` đỏ trước khi người
dùng gặp.

**610 passed, 3 skipped.**

### Kèm: `ruff` được ghim cứng

CI của chính commit này đỏ, và không phải vì code:

```
--> README.md:2291:52
1 file would be reformatted, 56 files already formatted
```

`requirements.txt` để `ruff>=0.6`. Runner giải ra 0.16.6, mà 0.16 bắt đầu
format cả Python nằm trong fence của Markdown — nên `ruff format --check` đỏ
trên một cái README vừa viết, theo một luật chưa tồn tại lúc viết nó, trong khi
đúng câu lệnh đó xanh ở máy local đang chạy 0.15.

Một luật lint mới xuất hiện là một phát hiện đáng có. Một formatter tự đổi cách
format là một diff không ai yêu cầu, rơi vào đúng PR nào đang mở tuần đó. Nên
`ruff==0.16.6`, ghim cứng, khác với mọi dòng còn lại trong file — và có test giữ
điều đó. Nâng thì nâng có chủ ý: sửa dòng ghim, chạy `ruff format .`, và bản
reformat nằm trong một commit nói rõ nó là bản reformat.

**611 passed, 3 skipped.**
