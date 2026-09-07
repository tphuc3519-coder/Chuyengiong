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

Mở **`/rvc`** trên app. Dán link tải, đặt tên, bấm **Thêm giọng**. Danh sách
giọng đã có nằm ngay dưới.

Link phải trỏ tới đúng file (`.zip` hoặc `.pth`), không phải trang tải. Nếu
voice-models.com đưa sang Mega hay Pixeldrain: bấm tải cho nó chạy, rồi vào
`chrome://downloads`, chuột phải mục vừa tải → Copy link address.

Tên gõ có dấu cũng được — cả trang lẫn server đều bỏ dấu như nhau
("Sơn Tùng MTP" → `son-tung-mtp`), và trang hiện trước tên sẽ lưu. Hai bản
chuẩn hoá đó phải khớp nhau: `normalise()` trong `web/app/rvc/page.tsx` và
`_safe_name()` trong `modal_rvc.py`, có test ở `tests/test_rvc_names.py`.

Muốn gọi thẳng API thì vẫn được: `POST /api/rvc/add-model` với
`{"url": "...", "name": "..."}`, và `GET /api/rvc/models` để xem danh sách.

## Đổi giọng thử — cũng ở `/rvc`

Cùng trang, phần dưới: chọn file **giọng đã tách sẵn**, chọn model, kéo pitch,
bấm **Đổi giọng**. Nghe ngay trên trang, tải về được.

Đây là đường đi tắt để nghe thử, chưa phải bước 4. Nó nhận vocal đã tách và trả
lại vocal đã đổi giọng — không tự tách nhạc nền, không tự ghép lại.

### Vì sao `/convert` không đi qua route của Vercel

Hai endpoint kia (`models`, `add-model`) đi qua `app/api/rvc/[action]/route.ts`
vì chúng chỉ trao đổi vài trăm byte JSON. `convert` thì không: body của
serverless function trên Vercel bị chặn ở **4.5MB cả hai chiều**, mà một vocal
4 phút đã vượt, còn wav trả về thì hơn 40MB. Proxy không tải nổi audio theo
chiều nào.

Nên trình duyệt POST thẳng lên Modal, đúng cách `web/lib/api.ts` đã làm với
pipeline chính (`/api/config` đưa `apiBase` cho trình duyệt, upload đi thẳng).
Địa chỉ lấy ở `GET /api/rvc/config`, không nhúng vào bundle.

Gửi bằng `multipart/form-data` chứ không phải JSON+base64, có lý do: multipart
nằm trong danh sách content-type an toàn của CORS nên POST kiểu này **không
sinh preflight** — chỉ cần đúng một header `Access-Control-Allow-Origin` ở
response là xong, không phải dựng ASGI app có CORS middleware. base64 cũng biến
mất, đỡ 33% dung lượng cả hai chiều.

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
- **`add-model` có thể timeout ở route** — gói Hobby của Vercel chặn hàm ở 60
  giây, mà tải một model 100MB từ Mega cộng cold start có thể lâu hơn. Modal
  vẫn tải xong và ghi vào Volume; bấm **Đọc lại** sau một phút là thấy. Đừng
  thêm lại, sẽ báo trùng tên.
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

## Nghe như robot ở mọi pitch — gần như chắc là `version`

Triệu chứng: vocal sạch, kéo pitch từ đầu đến cuối dải, lần nào cũng ra tiếng
máy móc, và không có một dòng lỗi nào.

Nguyên nhân nằm trong `rvc_python/modules/vc/modules.py`. `get_vc()` lấy
`version` từ **tham số truyền vào** chứ không đọc từ checkpoint — trong khi
nhánh dọn dẹp ngay phía trên nó lại đọc từ `cpt.get("version", "v1")`. Hai
nhánh không khớp nhau. Mặc định của `load_model()` là `"v2"`.

Ngay dưới đó là `load_state_dict(..., strict=False)`.

Nên nạp một model v1 mà không nói rõ v1 thì ba thứ cùng sai một lúc:

| Chỗ | v1 | Bị ép thành v2 |
|---|---|---|
| Kiến trúc | `SynthesizerTrnMs256NSFsid` | `...Ms768...` — weight không khớp bị **bỏ qua im lặng** |
| Layer hubert | 9 | 12 |
| `final_proj` | có | không |

Không có exception nào được ném ra. Chỉ có âm thanh sai, ở mọi pitch — nên
chỉnh tham số không bao giờ cứu được, và đó chính là chỗ dễ mất hàng giờ.

`_checkpoint_info()` đọc `version` thẳng từ checkpoint rồi truyền vào
`load_model()`. Nó cũng đọc `f0`: model có `f0 = 0` là loại train không kèm
pitch, kéo thanh cao độ sẽ không có tác dụng gì — trang `/rvc` nói thẳng điều
đó thay vì để người dùng ngồi đoán.

## Ba endpoint đang mở

`fastapi_endpoint` không tự chặn ai. Route proxy chỉ giấu URL khỏi trình duyệt
chứ không bảo vệ chúng — ai có URL là POST thẳng vào `convert` được, và mỗi lần
gọi là A10G quay. Đặt spending limit trong dashboard Modal trước khi đưa link
app cho người khác.

Tên model thì đã chặn: `_safe_name()` không cho nó chứa dấu phân cách đường dẫn
nữa. Nhưng đó là vá một lỗ, không phải là xác thực.

## Chi phí

A10G trên Modal khoảng 1,10 USD/giờ. Một bài 4 phút xử lý mất chừng 40-60
giây, tính ra khoảng 0,02 USD mỗi bài. Đặt spending limit trong dashboard
Modal trước khi mở cho người khác dùng.

## Còn phải verify thật

- [x] workflow `Deploy RVC` chạy xanh và in ra 3 URL — run #2, build 194 giây,
      base models tải đủ
- [ ] `add-model` nuốt được một link voice-models.com thật (cả trường hợp có
      `.index` lẫn không)
- [ ] `/convert` chạy hết một bài 4 phút, nghe thử chỗ nối chunk
- [ ] nghe thử pitch: cùng file, `0` và `+12` phải ra khác nhau rõ rệt (đây là
      chỗ lỗi `f0up_key` từng ẩn)
- [ ] đo lại thời gian và chi phí thực tế so với ước lượng ở trên
