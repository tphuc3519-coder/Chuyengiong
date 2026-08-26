# Voice Conversion Web App — Implementation Plan

> Tài liệu này viết để feed vào Claude Code. Mỗi Phase là một đơn vị công việc độc lập,
> có acceptance criteria rõ ràng. **Không nhảy sang Phase sau khi Phase trước chưa pass.**

---

## 0. Tổng quan

**Sản phẩm:** Web app nhận một bài hát (hoặc đoạn thoại) + một giọng mẫu, trả về bản
đã đổi sang giọng mẫu đó, nhạc nền giữ nguyên.

**Nguyên tắc kiến trúc:**
- Modal làm *toàn bộ* phần nặng: storage, GPU, orchestration. Vercel chỉ là UI mỏng.
- Không thêm S3/R2 ở MVP. File tạm nằm trong Modal Volume, key là UUID, có TTL cleanup.
- Async job từ đầu. Không có endpoint nào chạy đồng bộ quá 30s.
- Zero-shot trước (Seed-VC). Training-based (RVC) là Phase sau, **không** nằm trong MVP.

**Pipeline:**

```
input.mp3 ──► [separation]  ──► vocal.wav ──► [Seed-VC] ──► converted.wav
                    │                                              │
                    └────────► instrumental.wav ───────────────────┤
                                                                   ▼
                                                          [ffmpeg mix] ──► output.mp3
              reference_voice.wav ──────────────────────────────┘
```

Nhánh `speech` bỏ qua separation và mix — chạy thẳng Seed-VC.

---

## 1. Stack & repo layout

**Stack:** Modal (Python 3.11) · Seed-VC · ffmpeg · Next.js 15 App Router · Vercel · GitHub Actions

```
voice-convert/
├── modal_app/
│   ├── __init__.py
│   ├── app.py              # Modal App, images, volumes, secrets
│   ├── storage.py          # đọc/ghi/cleanup Volume
│   ├── jobs.py             # job state machine (modal.Dict)
│   ├── separation.py       # PORT TỪ REPO CŨ — không viết lại
│   ├── conversion.py       # Seed-VC class
│   ├── mixing.py           # ffmpeg
│   ├── pipeline.py         # orchestration: spawn + chain
│   └── api.py              # FastAPI web endpoints
├── web/                    # Next.js
│   ├── app/
│   │   ├── page.tsx
│   │   ├── api/
│   │   │   ├── submit/route.ts
│   │   │   ├── status/[jobId]/route.ts
│   │   │   └── download/[jobId]/route.ts
│   │   └── components/
│   └── lib/modal.ts
├── .github/workflows/
│   ├── deploy-modal.yml
│   └── ci.yml
└── requirements.txt
```

**Lưu ý:** `separation.py` phải là **port trực tiếp** từ app tách stem đã có
(BS-Roformer/HTDemucs trên Modal). Trước khi viết code Phase 3, hãy đọc repo cũ và
copy nguyên logic sang. Đừng implement lại.

---

## 2. Phase 0 — Scaffold & xác minh môi trường

### Việc cần làm
1. Tạo repo, dựng cây thư mục trên.
2. `modal_app/app.py`: khai báo App, Volume, Secret.
3. Tạo `.github/workflows/deploy-modal.yml` chạy `modal deploy modal_app/api.py`.
4. Secrets GitHub: `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
5. Deploy một hello-world web endpoint, gọi thử bằng curl.

### Code sườn

```python
# modal_app/app.py
import modal

app = modal.App("voice-convert")

# Volume cho model weights (persist, tải 1 lần duy nhất)
model_vol = modal.Volume.from_name("vc-models", create_if_missing=True)
# Volume cho file người dùng (ephemeral, có cleanup)
data_vol  = modal.Volume.from_name("vc-data", create_if_missing=True)
# Job state
job_dict  = modal.Dict.from_name("vc-jobs", create_if_missing=True)

MODEL_DIR = "/models"
DATA_DIR  = "/data"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.4.0", "torchaudio==2.4.0",
        "huggingface_hub", "soundfile", "librosa",
        "numpy<2", "scipy", "fastapi[standard]",
    )
)
```

### ⚠️ Bắt buộc trước khi code Phase 1
Seed-VC thay đổi API giữa các version. **Fetch và đọc `README.md` + `requirements.txt`
của `github.com/Plachtaa/seed-vc` trước khi viết `conversion.py`.** Pin về một commit
SHA cụ thể trong Dockerfile — đừng clone `main`. Các tên arg trong plan này
(`--diffusion-steps`, `--f0-condition`, `--semi-tone-shift`, `--auto-f0-adjust`) là
tham chiếu, phải verify lại với repo thực tế.

### Acceptance
- [ ] `curl https://<user>--voice-convert-api-health.modal.run` trả `{"status":"ok"}`
- [ ] Push lên `main` → GitHub Actions deploy thành công, không cần chạm terminal

---

## 3. Phase 1 — Seed-VC inference function

Đây là phần lõi. Làm chạy được **standalone** trước khi nối vào bất cứ thứ gì.

### Thiết kế

Dùng `@app.cls()` chứ không phải `@app.function()`, để load model một lần trong
`@modal.enter()` và tái sử dụng qua nhiều request. Đây là khác biệt lớn nhất về
chi phí và latency.

```python
# modal_app/conversion.py
import modal
from .app import app, model_vol, MODEL_DIR, base_image

vc_image = (
    base_image
    .run_commands(
        "git clone https://github.com/Plachtaa/seed-vc /opt/seed-vc",
        "cd /opt/seed-vc && git checkout <PIN_COMMIT_SHA>",
        "cd /opt/seed-vc && pip install -r requirements.txt",
    )
    .env({"HF_HOME": MODEL_DIR, "PYTHONPATH": "/opt/seed-vc"})
)

@app.cls(
    image=vc_image,
    gpu="A10G",
    volumes={MODEL_DIR: model_vol},
    scaledown_window=300,    # giữ container ấm 5 phút
    timeout=900,
)
class VoiceConverter:

    @modal.enter()
    def load(self):
        """Chạy 1 lần khi container khởi động. Weights cache trong Volume."""
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load checkpoint singing-capable (có F0 conditioning).
        # Verify tên model + hàm load với README repo.
        self.model = ...  # seed-vc loader
        model_vol.commit()


    @modal.method()
    def convert(
        self,
        source_wav: bytes,
        reference_wav: bytes,
        semitone_shift: int = 0,   # TÍNH SẴN Ở NGOÀI, không auto-detect trong đây
        diffusion_steps: int = 30,
        mode: str = "speech",      # "speech" | "singing"
    ) -> bytes:
        """
        Orchestrator. Cắt chunk → convert từng chunk → crossfade ghép.
        Đây là entry point duy nhất. KHÔNG expose _convert_chunk ra ngoài.
        """
        chunks = split_at_silence(source_wav, target_sec=30, max_sec=40)
        ref_embed = self._encode_reference(reference_wav)   # encode 1 lần, dùng lại
        out = []
        for c in chunks:
            out.append(self._convert_chunk(
                c, ref_embed, semitone_shift, diffusion_steps, mode
            ))
        return crossfade_concat(out, overlap_ms=200)

    def _convert_chunk(self, chunk, ref_embed, shift, steps, mode) -> bytes:
        """Convert đúng một chunk. Không gọi trực tiếp từ ngoài."""
        ...
```

### Tham số mặc định theo mode

| Mode | diffusion_steps | f0_condition | Ghi chú |
|---|---|---|---|
| `speech` | 25 | False | Nhanh, ~5s cho 30s audio |
| `singing` | 50 | True | Cần F0 để giữ giai điệu. Chậm hơn ~2x |

Cho phép user chỉnh `diffusion_steps` trong khoảng 10–100. Trên 100 gần như không
cải thiện thêm mà tốn GPU tuyến tính.

### 🔴 Chunking — thiết kế bắt buộc, không phải tối ưu hoá sau

**Đưa nguyên file dài vào một lần sẽ OOM trên A10G.** Phải chunk ngay từ đầu. Viết
`convert()` không chunk rồi sửa sau nghĩa là đập đi làm lại — vì `semitone_shift` và
reference embedding phải được nâng lên tầng orchestrator, không nằm trong hàm convert.

**Ba quy tắc:**

**1. Cắt tại silence, không cắt theo thời gian cố định.**
Cắt giữa một nốt đang ngân sẽ nghe rõ chỗ nối, dù có crossfade.

```python
def split_at_silence(wav: bytes, target_sec=30, max_sec=40, min_sec=10) -> list:
    """
    Tìm điểm cắt trong cửa sổ [target_sec-8, target_sec+8].
    Chọn frame có RMS thấp nhất trong cửa sổ đó.
    Nếu không có điểm nào dưới ngưỡng silence → cắt cứng tại max_sec.
    Chunk cuối ngắn hơn min_sec thì gộp vào chunk trước.
    """
```

Với vocal stem sau separation, silence detection dễ hơn nhiều so với bài full mix —
đây là lý do nữa để chunk *sau* separation chứ không phải trước.

**2. Crossfade 200ms khi ghép.** Equal-power (`sqrt`) fade, không phải linear —
linear fade làm âm lượng bị hõm ở giữa điểm nối.

**3. `semitone_shift` tính MỘT LẦN cho cả bài, áp dụng chung cho mọi chunk.**
Nếu để mỗi chunk tự auto-detect F0, tone sẽ nhảy giữa các đoạn. Lỗi này nghe rất rõ
và là bug phổ biến nhất của loại app này. Vì vậy `convert()` **nhận** `semitone_shift`
như tham số bắt buộc, không tự tính bên trong.

Tương tự, reference embedding encode một lần rồi tái sử dụng — vừa nhanh hơn vừa đảm
bảo timbre nhất quán xuyên suốt.

### 🌏 Yêu cầu đa ngôn ngữ — ảnh hưởng tới việc chọn checkpoint

**Ràng buộc sản phẩm: hát được mọi ngôn ngữ.** May mắn là VC vốn language-agnostic —
nội dung và cách phát âm đến từ source, model chỉ transfer timbre. Giọng mẫu tiếng
Việt + bài tiếng Nhật vẫn ra tiếng Nhật, chỉ đổi chất giọng.

Nhưng **content encoder có bias theo ngôn ngữ training**. Seed-VC có nhiều checkpoint
với encoder khác nhau — phải chọn đúng:

| Encoder | Độ phủ ngôn ngữ | Dùng khi |
|---|---|---|
| Whisper-based | ~99 ngôn ngữ | ✅ **Chọn cái này.** Phủ rộng nhất |
| XLSR-based (tiny) | Hẹp hơn | Chỉ khi cần real-time — không phải case của ta |

**Không chọn checkpoint tiny vì nó nhanh.** Ta chạy offline, độ phủ ngôn ngữ quan
trọng hơn latency. Verify tên checkpoint chính xác với README repo.

**Ba điều đã tự đúng, không cần code thêm:**
- Chunking theo silence — không phụ thuộc ngôn ngữ
- YIN pitch detection — làm việc trên F0, không phải phoneme
- Tiếng có thanh điệu (Việt, Thái, Quan thoại): khi hát, giai điệu đã lấn át thanh
  điệu. `semitone_shift` dịch đều toàn bộ F0 nên **contour thanh điệu được giữ nguyên**
  — không làm sai nghĩa

**Chỗ cần cẩn thận:** mode `speech` với tiếng có thanh điệu. Shift lớn (> ±8 semitone)
có thể đẩy F0 ra ngoài vùng tự nhiên và nghe méo thanh. Giới hạn slider ở ±8 cho
`speech`, giữ ±12 cho `singing`.

### Ràng buộc input — validate ở đây, không phải ở frontend
- **Reference: 5–30 giây.** Đây là giọng mẫu, không phải file nguồn. Dưới 5s chất
  lượng tệ, trên 30s không tốt hơn — dài hơn thì tự cắt lấy 30s đầu.
- **File nguồn: không giới hạn theo phút, giới hạn theo tổng thời lượng.** Reject
  trên 15 phút (lý do là chi phí và timeout, không phải giới hạn kỹ thuật). Bài 3–8
  phút là case chính, phải chạy trơn.
- Resample mọi input về sample rate mà model yêu cầu trước khi đưa vào.
- Convert về mono nếu model không nhận stereo.

### Acceptance
- [ ] `modal run modal_app/conversion.py` với 2 file wav local → ra output nghe được
- [ ] Lần gọi thứ hai (container còn ấm) nhanh hơn lần đầu rõ rệt
- [ ] Weights không tải lại sau khi restart container (Volume hoạt động)
- [ ] Mode `singing` giữ đúng giai điệu, không bị lệch tone
- [ ] **File 8 phút chạy xong không OOM** — test bằng file dài thật, đừng chỉ test 30s
- [ ] **Nghe kỹ các điểm nối chunk: không có click, pop, hay hụt âm lượng**
- [ ] **Tone nhất quán từ đầu đến cuối bài** — không nhảy giọng giữa các đoạn
- [ ] Đoạn nhạc dạo không lời không sinh ra tiếng lạ
- [ ] **Test tối thiểu 4 ngôn ngữ khác họ chữ viết**: Việt, Anh, Nhật/Hàn, và một
      tiếng ít tài nguyên hơn (Thái, Indo). Nghe rõ chữ ở cả bốn.
- [ ] **Cross-lingual: giọng mẫu tiếng Việt + bài tiếng Anh** → phát âm tiếng Anh
      giữ nguyên từ bản gốc, chỉ đổi chất giọng
- [ ] Bài tiếng Việt mode `singing` không bị sai thanh điệu sau khi shift

---

## 4. Phase 2 — Storage & job state

### Storage

```python
# modal_app/storage.py
def put(job_id: str, name: str, data: bytes) -> str
def get(job_id: str, name: str) -> bytes
def cleanup_expired(max_age_hours: int = 6) -> int
```

Layout trên Volume: `/data/{job_id}/{input,reference,vocal,instrumental,converted,output}.{wav,mp3}`

Thêm một cron xoá job cũ:

```python
@app.function(schedule=modal.Period(hours=6), volumes={DATA_DIR: data_vol})
def cleanup():
    from .storage import cleanup_expired
    n = cleanup_expired(max_age_hours=6)
    data_vol.commit()
    print(f"removed {n} jobs")
```

### Job state machine

Lưu trong `modal.Dict`. Các trạng thái:

```
queued → separating → converting → mixing → done
                                          ↘ failed
```

Record:
```python
{
  "id": str, "status": str, "progress": int,   # 0-100
  "mode": "song" | "speech",
  "created_at": float, "error": str | None,
  "params": {...},
}
```

Ước lượng `progress` theo bước (separating=30, converting=75, mixing=90, done=100) —
đừng cố tính chính xác, chỉ cần thanh tiến trình không đứng im.

### Acceptance
- [ ] Ghi rồi đọc lại được file qua Volume, đúng bytes
- [ ] Cron cleanup chạy, xoá đúng job quá hạn, không xoá job mới
- [ ] Job record cập nhật đúng thứ tự trạng thái

---

## 5. Phase 3 — Nối pipeline hoàn chỉnh

### Port separation
Đọc repo tách stem hiện có. Copy sang `separation.py` **giữ nguyên** model choice
(BS-Roformer cho chất lượng, HTDemucs cho tốc độ) và các tham số đã tune. Chỉ sửa
phần I/O để đọc/ghi qua `storage.py`.

### Orchestration

```python
# modal_app/pipeline.py
@app.function(volumes={DATA_DIR: data_vol}, timeout=1800)
def run_song_pipeline(job_id: str, params: dict):
    from .storage import get, put
    from .jobs import update
    try:
        update(job_id, "separating", 5)
        stems = separate.remote(get(job_id, "input.mp3"))
        put(job_id, "vocal.wav", stems["vocals"])
        put(job_id, "instrumental.wav", stems["instrumental"])

        update(job_id, "converting", 30)
        converted = VoiceConverter().convert.remote(
            source_wav=stems["vocals"],
            reference_wav=get(job_id, "reference.wav"),
            semitone_shift=params["semitone_shift"],
            diffusion_steps=params["diffusion_steps"],
            mode="singing",
        )
        put(job_id, "converted.wav", converted)

        update(job_id, "mixing", 75)
        out = mix(converted, stems["instrumental"], vocal_gain_db=params.get("vocal_gain", 0))
        put(job_id, "output.mp3", out)

        update(job_id, "done", 100)
    except Exception as e:
        update(job_id, "failed", error=str(e))
        raise
```

Endpoint submit dùng `.spawn()` để trả ngay:

```python
run_song_pipeline.spawn(job_id, params)
```

### Mixing

```python
# modal_app/mixing.py — ffmpeg, không dùng thư viện Python audio cho bước này
# amix với normalize=0 để không bị giảm âm lượng, rồi loudnorm ở cuối
```

Lệnh tham khảo:
```
ffmpeg -i converted.wav -i instrumental.wav \
  -filter_complex "[0:a]volume={gain}dB[v];[v][1:a]amix=inputs=2:normalize=0[m];[m]loudnorm=I=-14:TP=-1.0[out]" \
  -map "[out]" -c:a libmp3lame -b:a 192k output.mp3
```

`-14 LUFS` là chuẩn streaming, hợp lý cho output nghe trên điện thoại.

### API endpoints

```
POST /submit          multipart: input, reference, mode, params  →  {job_id}
GET  /status/{id}                                                →  {status, progress, error}
GET  /download/{id}                                              →  audio/mpeg stream
```

### Acceptance
- [ ] End-to-end: submit bài 3 phút + giọng mẫu 15s → ra file cover hoàn chỉnh
- [ ] Nhạc nền không bị méo, vocal không bị chìm hoặc chói
- [ ] Job fail giữa chừng → status `failed` có message, không treo vô hạn
- [ ] Submit trả về trong < 2 giây

---

## 6. Phase 4 — Frontend

### Flow màn hình

```
[1] Chọn mode: Bài hát / Giọng nói
[2] Upload nguồn (drag-drop + file picker)
[3] Giọng mẫu: upload HOẶC ghi âm trực tiếp (MediaRecorder) HOẶC chọn preset
[4] Tinh chỉnh (collapsed mặc định): pitch shift, chất lượng
[5] Checkbox đồng thuận  ← bắt buộc, không được bỏ qua
[6] Convert → progress → preview player → download
```

### Chi tiết quan trọng

**Ghi âm trong browser:** dùng `MediaRecorder`. Hiển thị countdown 15s và waveform
đơn giản. Người dùng ngại upload file — cho ghi âm trực tiếp làm tăng conversion rate
mạnh hơn bất kỳ tính năng nào khác ở giai đoạn này.

**Preset giọng:** chuẩn bị sẵn 4–6 giọng (tự thu hoặc dataset có license rõ ràng) để
user thử ngay mà không cần upload gì. Đây là cách duy nhất để có first-run experience
dưới 30 giây.

**Polling:** interval 2s, exponential backoff sau 60s, timeout ở 15 phút. Dùng
`AbortController` khi unmount.

**Vercel proxy:** route handler ở `/api/*` forward sang Modal. Modal URL để trong env
var, **không** hardcode vào client bundle. File lớn → dùng streaming, đừng buffer
toàn bộ vào memory của serverless function (giới hạn body Vercel là 4.5MB — với file
lớn hơn phải upload thẳng lên Modal từ client bằng presigned pattern hoặc cho Modal
nhận trực tiếp qua CORS).

> Quyết định sớm: nếu file người dùng thường > 4.5MB (bài hát 3 phút MP3 ≈ 4–7MB),
> **bỏ qua Vercel cho đường upload**. Cho client POST thẳng lên Modal endpoint, bật
> CORS cho domain Vercel. Vercel chỉ phục vụ UI và status polling.

### Acceptance
- [ ] Chạy trọn vẹn trên Safari iOS (đây là thiết bị test chính)
- [ ] Ghi âm hoạt động trên iOS Safari — kiểm tra kỹ, MediaRecorder trên iOS có quirk
- [ ] File 7MB upload thành công
- [ ] Progress bar không đứng im quá 20 giây

---

## 7. Phase 5 — Pitch handling

Đây là thứ quyết định output nghe hay hay dở, quan trọng hơn cả model.

**Chạy ở đâu:** trên **toàn bộ vocal stem, TRƯỚC khi chunk**, trong `pipeline.py`
ngay sau bước separation. Kết quả truyền xuống `convert()` như một giá trị duy nhất
cho cả bài. Xem lại quy tắc 3 ở Phase 1 — đây là chỗ giá trị đó được sinh ra.

### Auto-detect gợi ý pitch shift

Tái sử dụng YIN từ app pitch detection đã có:

```python
def suggest_semitone_shift(source_wav: bytes, reference_wav: bytes) -> int:
    """Tính median F0 của cả hai (chỉ trên voiced frames), trả về số semitone."""
    f0_src = median_f0_voiced(source_wav)   # YIN
    f0_ref = median_f0_voiced(reference_wav)
    if not f0_src or not f0_ref:
        return 0
    shift = round(12 * log2(f0_ref / f0_src))
    return max(-12, min(12, shift))
```

**Chỉ tính trên voiced frames** — lấy median cả đoạn có silence sẽ ra sai hoàn toàn.

Hiển thị giá trị này làm mặc định trên slider, cho user override. Nam→nữ thường
+12, nữ→nam thường −12, nhưng auto-detect chính xác hơn nhiều so với đoán theo giới tính.

### Acceptance
- [ ] Gợi ý cho cặp nam→nữ ra khoảng +10 đến +14
- [ ] Bài có intro nhạc dài không làm lệch kết quả (silence bị loại đúng)

---

## 8. Phase 6 — Consent gate & an toàn

**Đây không phải phần optional.** Rủi ro lớn nhất của app loại này là pháp lý, không
phải kỹ thuật.

### Bắt buộc
1. Checkbox không tick sẵn: *"Tôi xác nhận có quyền sử dụng giọng nói trong file tham
   chiếu, hoặc đó là giọng của chính tôi."* Không cho submit nếu chưa tick.
2. Ghi metadata `AI-generated` vào file output (ffmpeg `-metadata comment=`).
3. Terms of service ngắn, nêu rõ cấm dùng giọng người khác khi không được phép.
4. Không cung cấp preset là giọng người nổi tiếng. Không xây thư viện giọng do người
   dùng chia sẻ công khai ở MVP — đó là nơi vấn đề bùng phát.
5. Log job ID + timestamp (không log nội dung audio) để có audit trail nếu bị khiếu nại.

### Cân nhắc thêm
Audio watermark (AudioSeal của Meta, MIT license) — không bắt buộc ở MVP nhưng nên
thêm trước khi mở public.

---

## 9. Chi phí

Ước tính cho bài 3 phút trên A10G (~$1.10/giờ):

| Bước | GPU time | Chi phí |
|---|---|---|
| Separation | ~30s | $0.009 |
| Seed-VC (singing, 50 steps) | ~60s | $0.018 |
| Mixing (CPU) | ~5s | ~$0 |
| **Tổng** | **~1.5 phút** | **~$0.03/bài** |

Cold start thêm ~30–60s lần đầu. Với `scaledown_window=300`, các request trong 5 phút
kế tiếp không chịu chi phí này.

**Kiểm soát chi phí:**
- Rate limit theo IP: 5 job/giờ ở MVP
- Giới hạn độ dài input 10 phút
- `max_containers` trên VoiceConverter để tránh burst tốn tiền ngoài dự kiến

---

## 10. Rủi ro & phương án

| Rủi ro | Dấu hiệu | Xử lý |
|---|---|---|
| Seed-VC API khác plan này | Import error, sai tên arg | Đọc README repo trước khi code Phase 1. Đây là rủi ro số 1. |
| Output singing bị lệch tone | Nghe sai giai điệu | Bật `f0_condition`, tăng diffusion_steps lên 50–80 |
| Vocal sau separation có artifact | Nghe rè, có tiếng nhạc lẫn | Dùng BS-Roformer thay HTDemucs, chấp nhận chậm hơn |
| Cold start làm user bỏ đi | Job đầu tiên chờ > 90s | Tăng `scaledown_window`, hoặc `min_containers=1` giờ cao điểm |
| Vercel body limit 4.5MB | Upload fail file lớn | Bypass Vercel cho upload (mục Phase 4) |
| OOM với file dài | CUDA OOM ở bài > 2 phút | Chunk ngay từ Phase 1. Giảm `target_sec` xuống 20 nếu vẫn OOM |
| Nghe rõ điểm nối chunk | Click/pop định kỳ mỗi ~30s | Cắt tại silence thay vì thời gian cố định; equal-power crossfade |
| Tone nhảy giữa các đoạn | Giọng đổi cao độ giữa bài | `semitone_shift` global, không per-chunk. Bug phổ biến nhất. |
| Reference quá ngắn/nhiễu | Giọng ra không giống | Validate ở backend, báo lỗi cụ thể chứ đừng convert rồi ra kết quả tệ |

---

## 11. Không làm ở MVP

Ghi rõ để tránh scope creep:

- ❌ RVC training (upload 30 phút audio → train model riêng) — Phase sau
- ❌ Real-time conversion
- ❌ Thư viện giọng công khai / user-shared voices
- ❌ Tài khoản, thanh toán
- ❌ TTS, chế nhạc — sản phẩm riêng, không nhét chung
- ❌ Batch processing

---

## 12. Thứ tự thực hiện

```
Phase 0  scaffold + deploy pipeline      →  nửa ngày
Phase 1  Seed-VC + chunking standalone   →  2–3 ngày   ← rủi ro tập trung ở đây
Phase 2  storage + job state             →  nửa ngày
Phase 3  nối pipeline hoàn chỉnh         →  1 ngày
Phase 4  frontend                        →  2 ngày
Phase 5  pitch auto-detect               →  nửa ngày
Phase 6  consent gate                    →  nửa ngày
```

Nếu Phase 1 quá 3 ngày, dừng lại và cân nhắc fallback: dùng RVC v2 với vài model
pretrained có sẵn thay vì zero-shot. Chất lượng vẫn tốt, chỉ mất tính năng "upload
giọng bất kỳ".
