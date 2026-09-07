# Chỉnh hệ thống đổi beat bằng tai

Trạng thái sống của việc đang dở. `README.md` (Phase 16, 16.1, 16.2) kể chuyện
đã làm gì và tại sao; **file này chỉ nói còn phải làm gì tiếp**.

Cập nhật lần cuối: sau khi job sinh beat đầu tiên chạy được trên GPU thật.

---

## 1. Trạng thái, và một cái bẫy

Nhánh đang làm: `claude/beat-switching-system-6lgymx`.

| commit | nội dung | đã ở `main`? | đã deploy? |
|---|---|---|---|
| `4bb5390` | khớp ô nhịp, né theo giọng, 12 kiểu nhạc | ✅ | ✅ |
| `963e7e0` | picker kiểu nhạc trên UI | ✅ | ✅ |
| `1e7642e` | bed stereo, cân bằng bằng phép đo, luật vạch nhịp thứ hai | ✅ | ✅ |
| `db5dcfb` | **ghim `diffusers==0.36.0`** | ❌ | ✅ (deploy từ nhánh) |
| `c8d753b` | tick các mục đã verify trong README | ❌ | — |
| `2660f36` | `ORIGINAL_STRENGTH` 0.65 → 0.45, sửa nhãn UI | ❌ | ❌ |
| `568d8e9` | slider "Bám bài bao nhiêu" (`beat_follow`) | ❌ | ❌ |

### ⚠️ Bẫy: `main` chưa có bản vá diffusers

`main` đang ở trạng thái mà container `BeatGenerator` **không khởi động được**
— nó chết trong `@modal.enter()` với `ValueError: infer_schema(func)` (xem
README 16.2). Job chạy được là vì deploy đó xuất phát từ **nhánh**, không phải
từ `main`.

`.github/workflows/deploy-modal.yml` tự chạy khi push vào `main` có đụng
`modal_app/**`. Nên **bất cứ push nào vào `main` bây giờ cũng deploy lại đúng
bản hỏng đó.**

**Việc đầu tiên của session sau: merge nhánh này vào `main`.** Không phải để
"xong việc" — để `main` thôi mang một bản deploy chết.

---

## 2. Vòng lặp duy nhất còn lại

Mọi thứ đo được bằng máy đã đo xong (mục 4). Cái còn lại chỉ nghe mới biết, và
vòng lặp là:

```
sửa hằng số  ──►  merge/deploy  ──►  chạy 1 job  ──►  nghe  ──►  người dùng tả lại
      ▲                                                                  │
      └──────────────────────────────────────────────────────────────────┘
```

Mỗi vòng tốn của người dùng một lượt deploy và một lượt nghe. **Vì thế: nếu một
hằng số cần dò bằng tai, hãy biến nó thành một cái núm trên UI thay vì đoán hộ
một con số.** `beat_follow` đã đi đường đó rồi; những cái còn lại trong mục 3
có thể cần đi theo.

### Chạy một job thử

App → **Đổi beat** (hoặc **Đổi beat + giọng**) → **Phối lại bài này** → chọn một
kiểu nhạc → chạy.

Xem log ở Modal, function `BeatGenerator`. Hai dòng đáng đọc:

```
[derive] <job>: <bpm/tông/vạch nhịp> / <vòng hợp âm dò được>
[BeatGenerator] 230s in 18.0s (60 steps, 2ch, init 0.65): '<prompt>'
```

- `init 0.65` → nhánh `original` (ô "nghe thẳng nhạc nền gốc" đang bật)
- `init 0.35` → nhánh `sketch` (mặc định)
- `2ch` → bed ra stereo
- dòng `[derive]` → **vòng hợp âm app dò được**. Nếu nó sai thì không con số nào
  cứu được: máy đang được bảo chơi sai bài.

---

## 3. Đang mở: beat nghe chưa hay

Hai báo cáo bằng tai, trên bài thật, và chúng hỏng theo hai kiểu **ngược nhau**:

| nhánh | `init_strength` | người dùng nói |
|---|---|---|
| `original` (máy nghe nhạc nền gốc) | 0.65 | *"nghe k khác bản gốc mấy"* |
| `sketch` (máy nghe hợp âm app tự đánh) | 0.35 | *"nghe siêu kém"* |

**Hai con số này không kẹp được một giá trị ở giữa** — chúng là hai *nguồn* khác
nhau, không phải hai điểm trên một trục. Đừng nội suy.

Đã làm vì hai báo cáo này: hạ `ORIGINAL_STRENGTH` 0.65 → 0.45
(`modal_app/beatgen.py`), và đưa `beat_follow` ra thành slider. **Cả hai chưa
deploy.**

### Câu cần hỏi tiếp

Nhánh `sketch` "kém" theo kiểu nào? Hai triệu chứng, hai hướng sửa ngược nhau:

| nghe thấy | nghĩa là | kéo slider |
|---|---|---|
| beat đi lạc, chỏi hợp âm, không ra bài của mình | strength **quá thấp**, model bỏ qua sketch | **lên** 0.5–0.6 |
| nghe như MIDI rẻ tiền, như nhạc chuông | strength **quá cao**, model chép cả chất tổng hợp thô | **xuống** 0.15–0.2 |

Nếu là vế thứ hai thì vấn đề nằm ở **`sketch.py` quá thô**, không nằm ở con số —
và đó là một việc sửa khác hẳn (xem README Phase 13.2, chỗ `arrange.py` bị gỡ vì
đúng lý do này).

### Đường chưa thử

**"Beat mới từ mô tả"** (`beat_source="generate"`). Không đưa gì cho model bám,
nên ACE-Step tự do hoàn toàn — nó vốn giỏi nhất ở đây, và beat ra vẫn được
`beats.fit` kéo về đúng tông và tốc độ của bài. Cái mất là nó không biết vòng
hợp âm.

Nếu đường này nghe hay hơn hẳn hai đường kia, kết luận là **sketch mới là khâu
yếu**, không phải mấy con số.

---

## 3b. Lấy lại kết quả khi mất trang

Job chạy ở Modal, không chạy trong trình duyệt — **reload trang không giết
job**, và file nằm trên Volume 6 tiếng (`storage.DEFAULT_MAX_AGE_HOURS`).

Trang giờ tự nhớ job id vào `localStorage` và tự nối lại khi mở lại. Nếu vì lý
do gì đó không nối được (trình duyệt chặn site data, đổi máy, quá 6 tiếng), lấy
tay:

1. Job id in trong log Modal — tìm dòng `[beat] <id>:` hoặc `[derive] <id>:`
2. Địa chỉ API: mở `https://<app>/api/config`, nó trả `{"apiBase": "..."}`
3. Tải: mở `<apiBase>/download/<job_id>`

`/status/<job_id>` cũng nhận cùng id nếu muốn xem job xong chưa.

---

## 4. Đã đo xong — đừng đo lại

| câu hỏi | trả lời | bằng chứng |
|---|---|---|
| ACE-Step có trả stereo không | **có** | log `2ch` |
| bed có dài bằng cả bài không | **có**, 230s một lần sinh | log `230s in 18.0s` |
| prompt của style tới nơi nguyên vẹn không | **có** | log in đúng `STYLES` + 2 suffix |
| image có build và khởi động thật không | **có** | job `Succeeded`, 18.16s |
| `diffusers` bản nào chạy với torch 2.4.0 | **≤ 0.36.0** | bisect, README 16.2 |
| trọng số hai luật vạch nhịp | **0.4** | 180 phép đo, README 16.1 |
| cắt băng cho luật vạch nhịp | **120 Hz** | 4 ngưỡng đo margin, README 16 |
| độ sâu né theo giọng có đơn điệu không | **có**, ~80% danh nghĩa | README 16 |
| cân bằng có lặp lại được không | **có**, lệch < 0.5 dB | README 16.1 |

---

## 5. Còn mở, chưa ai nghe

Xếp theo mức đáng nghi:

1. **12 hồ sơ phối trong `modal_app/styles.py`** — chiều thì chắc (rock né sâu
   hơn bolero), **độ lớn thì chưa đo**. Cả 12 nằm gọn trong một bảng.
   - `bed_below_voice_db` — bed đứng sau giọng bao nhiêu dB
   - `duck_db` — né bao sâu khi có người hát
   - `pocket_db` / `pocket_hz` — khoét ở dải nào, sâu bao nhiêu
   - `low_shelf_db` — giữ lại bao nhiêu tiếng trầm
2. **Vạch nhịp trên nhạc thật.** Mới chỉ gặp năm kiểu phối tổng hợp. Nghe xem
   kick của beat có rơi đúng nhịp bài không.
3. **Hốc giọng 2.2–2.8 kHz** — đúng chỗ cho tiếng Việt chưa, hay còn thấp.
4. **Bed vào từ giây 0** — nghe tự nhiên hay cụt đầu ô nhịp.
5. **Cắt 30 Hz trên beat upload** — có ai thấy mất lực không.
6. **`SKETCH_STRENGTH` / `ORIGINAL_STRENGTH`** — xem mục 3.

---

## 6. Bom hẹn giờ đã biết

`BEATGEN_REQUIREMENTS` giờ ghim hết, nhưng `requirements.txt` của ACE-Step vẫn
**floor** mấy dòng này, tức là mỗi lần build lại có thể lấy bản mới:

`gradio` · `peft` · `numba` · `tensorboard` · `tensorboardX` · `click` · `cutlet`

Chưa cái nào hỏng. **Chưa ghim vì chưa có bằng chứng** — ghim đoán trước là một
cách khác để hỏng. Nhưng nếu deploy đỏ lần nữa với lỗi lúc import, hãy nhìn
danh sách này trước.

Và bài học của 16.2, viết lại cho gọn: **resolve được không có nghĩa là import
được.** `pip install --dry-run` xanh vẫn có thể chết ở `@modal.enter()`. Muốn
chắc thì cài thật rồi chạy đúng dòng import trong `load()`.
