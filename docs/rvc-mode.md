# RVC mode

Mode đổi giọng bằng model RVC đã train sẵn (tải từ voice-models.com), chạy song
song với pipeline Seed-VC chứ không thay thế nó. Plan gốc (`implementation-plan.md`
§0) để RVC ra ngoài MVP; đây là phần đó, làm thành một Modal app đứng riêng.

Làm hoàn toàn từ điện thoại. Không cần terminal.

## Đặt file ở đâu

```
modal_rvc.py                           <- gốc repo
web/app/api/rvc/[action]/route.ts      <- tạo thư mục lồng nhau trong GitHub web editor
.github/workflows/deploy-rvc.yml
```

> Khác plan một chỗ: route nằm dưới `web/`, không phải `app/` ở gốc repo. Next.js
> của repo này ở trong `web/` (đó là root directory của project trên Vercel), nên
> `app/api/...` ở gốc repo sẽ không được build và cũng không thành route nào cả.

Mẹo tạo thư mục lồng nhau trên GitHub: bấm **Add file → Create new file**, rồi gõ
cả đường dẫn `web/app/api/rvc/[action]/route.ts` vào ô tên file. GitHub tự tạo thư mục.

## Bước 1 — Deploy Modal

Không có terminal thì dùng GitHub Actions: `.github/workflows/deploy-rvc.yml`.
Chạy bằng tab Actions → Deploy RVC → Run workflow, hoặc để nó tự chạy khi `main`
chạm `modal_rvc.py`.

Workflow này tách hẳn khỏi `Deploy Modal`: hai app khác nhau, hai image khác nhau,
một cái hỏng không kéo cái kia xuống. Dùng chung `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET` đã có sẵn, không cần thêm secret nào.

Lần đầu mất khoảng 10-15 phút vì phải build image có torch + CUDA. Các lần sau
nhanh hơn nhiều. Log cuối sẽ in ra 3 URL. Chép lại.

## Bước 2 — Biến môi trường trên Vercel

Settings → Environment Variables, thêm:

```
MODAL_RVC_CONVERT_URL
MODAL_RVC_LIST_URL
MODAL_RVC_ADD_URL
```

Ba URL riêng chứ không phải một base như `MODAL_API_URL`, vì `modal_rvc.py`
publish ba `fastapi_endpoint` độc lập — mỗi cái một domain. Redeploy Vercel sau
khi thêm. Thiếu biến nào thì đúng action đó trả 500 kèm tên biến còn thiếu, phần
còn lại của app vẫn chạy.

## Bước 3 — Nạp model đầu tiên

Vào voice-models.com, tìm giọng, copy link download. Rồi từ trình duyệt điện
thoại mở app của bạn và gọi:

```js
await fetch("/api/rvc/add-model", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ url: "<link vừa copy>", name: "ten-giong" }),
});
```

Hoặc gắn tạm hai ô input + một nút trong UI để khỏi phải mở console.

Kiểm tra: `GET /api/rvc/models`.

## Bước 4 — Nối vào pipeline hiện có

Luồng đầy đủ:

```
file nhạc
  -> BS-Roformer (đã có)  -> vocal.wav + instrumental.wav
  -> POST /api/rvc/convert với vocal.wav
  -> trộn output với instrumental.wav
```

Bước trộn cuối làm bằng ffmpeg.wasm ngay trên trình duyệt, hoặc thêm một Modal
function nhỏ nữa. Đừng đưa instrumental qua RVC — chỉ vocal thôi.

Bước này **chưa làm**. `modal_app/separation.py` và `modal_app/mixing.py` đã có sẵn
hai đầu, nhưng chúng chạy trong job state machine của `modal_app/jobs.py`, còn
`/convert` của RVC là một lệnh gọi đồng bộ không có job id. Nối vào là việc riêng.

## Tham số cần chỉnh

| Tham số | Ý nghĩa | Gợi ý |
|---|---|---|
| `pitch` | dịch cao độ, đơn vị nửa cung | nam→nữ `+12`, nữ→nam `-12`, cùng giới `0` |
| `index_rate` | bám đặc trưng giọng gốc của model | `0.5-0.7`. Cao quá sẽ có tiếng rè |
| `protect` | giữ phụ âm, tránh méo | `0.33`. Nếu lời bị nhoè thì tăng lên `0.5` |
| `f0_method` | thuật toán dò cao độ | `rmvpe` cho hát. `harvest` chậm hơn nhưng đôi khi mượt hơn ở giọng trầm |

`pitch` là thứ ảnh hưởng nhiều nhất. Sai quãng thì giọng ra nghe như robot,
chỉnh đúng thì tự nhiên ngay.

`modal_app/pitch.py` đã đo được cao độ giọng nguồn — có thể dùng nó để gợi ý
`pitch` thay vì bắt người dùng đoán.

## Chỗ dễ hỏng

- **Model không có file `.index`** — vẫn chạy được nhưng giọng kém giống hơn.
  Nhiều model trên voice-models.com thiếu file này.
- **Link download không phải zip** — một số link trỏ tới trang trung gian
  (Mega, Pixeldrain) chứ không phải file trực tiếp. Phải mở link, bấm tải,
  rồi lấy link thật.
- **Lần gọi đầu chậm** — cold start Modal cộng thời gian nạp model. Đã đặt
  `scaledown_window=300` nên trong 5 phút sau đó sẽ nhanh.
- **Tên tham số của `set_params()`** — là `f0up_key` và `f0method`, không có
  gạch dưới ở giữa. Hàm này lọc theo whitelist rồi chỉ `print` cảnh báo cho tên
  lạ, nên gõ thành `f0_up_key`/`f0_method` là bị bỏ qua im lặng: pitch luôn 0,
  f0 method luôn là `harvest`. Giọng ra nghe như robot mà log thì sạch.
- **Bài dài** — code đã tự cắt theo khoảng lặng khi vượt 90 giây. Nếu vocal
  gần như không có khoảng lặng, nó rơi về cắt theo độ dài cố định và chỗ nối
  có thể nghe thấy nhẹ.
- **Nội dung đi qua base64 trong body JSON** — cả chiều lên lẫn chiều về. Bài dài
  thì body phình to; `maxDuration = 300` của route chỉ lo thời gian, không lo
  kích thước. Nếu đụng giới hạn body của Vercel thì phải đổi sang upload thẳng
  lên Modal như `modal_app/storage.py` đang làm.
- **Chưa có consent gate và watermark** — hai thứ này nằm ở `modal_app/`
  (Phase 6, Phase 7) và app RVC không đi qua chúng. Trước khi mở cho người khác
  dùng thì phải tính.

## Vì sao image phải hạ pip xuống dưới 24.1

Lần deploy đầu đỏ ở đây, và lỗi trông như link hỏng chứ không phải như thật:

```
ERROR: No matching distribution found for omegaconf==2.0.6
```

omegaconf 2.0.6 có trên PyPI. Vấn đề là metadata của nó khai
`PyYAML (>=5.1.*)` — `>=` đi với `.*` không hợp chuẩn. pip 24.1 bắt đầu **bỏ qua
hẳn** distribution có metadata như vậy thay vì chỉ cảnh báo, nên với pip của
image (25.1.1) thì omegaconf 2.0.6 coi như không tồn tại. Đây là pin cứng của
`rvc-python`, không đổi phiên bản được, nên hạ pip là đường còn lại.

Chỗ này dễ tự lừa mình: chạy thử ở máy có pip 24.0 thì resolve xanh, vì 24.0
đứng ngay dưới mốc đổi hành vi. Phải đúng pip ≥ 24.1 mới thấy.

Kèm theo đó, `rvc-python` ghim `fairseq==0.12.2`, mà fairseq chỉ có wheel cho
cp36/37/38. Trên Python 3.10 nó compile ba extension C++ từ source, nên image
cần `build-essential` — `debian_slim` không có `g++`.

Đã kiểm trên Python 3.10 thật: tái hiện đúng lỗi với pip 25.1.1, pip 24.0 thì
resolve sạch cả cây (`numpy 1.23.5`, `omegaconf 2.0.6`, `hydra-core 1.0.7`,
`faiss-cpu 1.7.3`, `fairseq 0.12.2`), và fairseq build ra đủ ba file `.so`.

## Chi phí

A10G trên Modal khoảng 1,10 USD/giờ. Một bài 4 phút xử lý mất chừng 40-60
giây, tính ra khoảng 0,02 USD mỗi bài. Đặt spending limit trong dashboard
Modal trước khi mở cho người khác dùng.

## Còn phải verify thật

- [ ] workflow `Deploy RVC` chạy xanh và in ra 3 URL
- [ ] `add-model` nuốt được một link voice-models.com thật (cả trường hợp có
      `.index` lẫn không)
- [ ] `/convert` chạy hết một bài 4 phút, nghe thử chỗ nối chunk
- [ ] đo lại thời gian và chi phí thực tế so với ước lượng ở trên
