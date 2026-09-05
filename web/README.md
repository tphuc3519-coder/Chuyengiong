# web/ — giao diện

Next.js 15 (App Router) + React 19, không dùng framework CSS. Deploy trên Vercel
với **Root Directory = `web`**.

```bash
cd web
npm install
cp .env.example .env.local     # điền URL Modal đã deploy
npm run dev                     # http://localhost:3000
```

Kiểm tra như CI chạy:

```bash
npm run lint && npm run format:check && npm run typecheck && npm run build
```

## Biến môi trường

| Tên             | Bắt buộc | Ý nghĩa                                     |
| --------------- | -------- | ------------------------------------------- |
| `MODAL_API_URL` | ✅       | Base URL của Modal API, không có `/` ở cuối |
| `CONTACT_EMAIL` | —        | Địa chỉ nhận khiếu nại, hiện trên `/terms`  |

Không biến nào có tiền tố `NEXT_PUBLIC_`: xem mục "Đường đi của request" bên
dưới. `CONTACT_EMAIL` để trống thì trang điều khoản nói thẳng là chưa công bố
kênh liên hệ, chứ không in ra một link chết.

## Đường đi của request

```
trình duyệt ──POST /submit────────────────► Modal      (upload, 4–60 MB)
            ──GET  /api/status/{id}──► Vercel ──► Modal (poll, vài trăm byte)
            ──GET  /download/{id}─────────────► Modal   (mp3 kết quả)
            ──GET  /api/config───────► Vercel          (URL của Modal)
```

Upload và download đi thẳng lên Modal vì body limit của Vercel là 4.5 MB còn bài
3 phút là 4–7 MB (plan §6). Status poll đi qua Vercel: payload nhỏ, và giữ vòng
lặp poll cùng origin với trang.

Nghĩa là trình duyệt phải biết URL của Modal. Plan yêu cầu URL nằm trong env var
chứ không hardcode vào client bundle — nên nó được phục vụ **lúc chạy** qua
`/api/config` (`lib/server.ts` đọc `process.env`), không phải `NEXT_PUBLIC_` vốn
bị nhúng thẳng vào bundle lúc build. Hệ quả tiện: cùng một build chạy được cho cả
preview lẫn production.

## Hai chế độ beat

`rebeat` ("Đổi beat") giữ nguyên giọng gốc và chỉ thay nhạc nền; `beat` ("Đổi
beat + giọng") làm cả hai. Chúng dùng chung đúng một component nguồn beat, và
khác nhau ở ba chỗ trong UI:

- bước "Giọng mẫu" bị **ẩn** ở `rebeat` — không phải disable, vì backend từ
  chối reference gửi tới mode đó và một ô trống ở đây là lời mời tới một cái 400;
- trong "Tinh chỉnh", dịch cao độ / chất lượng / bám giọng mẫu biến mất, còn lại
  độ trong và âm lượng giọng;
- câu cam kết đổi sang _"tôi có quyền sử dụng bản ghi này"_, vì không có file
  tham chiếu nào để cam kết về. Bắt ai đó xác nhận quyền với một file họ chưa
  từng được hỏi là một checkbox không ai tick thật lòng được.

`convertsVoice(mode)` trong `lib/params.ts` là chỗ duy nhất quyết định cả ba.

## Chế độ "Đổi beat"

`app/components/BeatSource.tsx`. Mode `beat` thêm đúng một bước vào form (2b) và
giữ nguyên mọi bước khác: nguồn beat là radio group hai lựa chọn — tải lên hoặc
tự sinh. Nơi deployment không bật phần sinh beat thì chỉ còn một nguồn, và
nhóm radio biến mất hẳn thay vì hiển thị một lựa chọn duy nhất.

Từng có lựa chọn thứ ba, "phối lại bài này". Nó bị gỡ: đem so với một bản phối
rock do người làm, bản nó dựng ra có 55% năng lượng dưới 120 Hz và không có gì
trên 4 kHz — khoảng cách đó là giữa tổng hợp sóng bằng số học và một thư viện
sample thu thật, không phải chuyện tinh chỉnh.

"Tải beat lên" là mặc định vì nó luôn chạy được: không cần GPU, không cần weights
gated, và license của thứ ra lò là license người dùng đã chọn khi lấy file.

Dòng chữ dưới mỗi lựa chọn không phải disclaimer — mỗi dòng là thứ dễ làm đúng
người chọn nó thất vọng nhất, và nói trước một lần rẻ hơn để mỗi người tự phát
hiện một lần.

Với **tự sinh**: beat sinh ra không biết vòng hợp âm của bài, nên nó hợp với rap
và nhạc điện tử và chỏi dần khi giọng đi giai điệu nhiều.

Với **phối lại**: nó gỡ được quyền _bản ghi_ chứ không gỡ được quyền _tác phẩm_ —
hợp âm vẫn của bài gốc, giọng vẫn hát giai điệu đó, nên kết quả là một bản cover.
Đây là chỗ duy nhất trong app nói về bản quyền bằng chữ, và nó nói vì người dùng
chọn cái nút này chính là để giải quyết chuyện đó.

## Chế độ "Giọng hát"

Mode `vocal` là `song` bỏ bước tách nhạc nền, và khác biệt duy nhất ở frontend
là nhãn của bước 2 (`SOURCE_LABEL` trong `app/page.tsx`) — cùng một ô chọn file,
cùng giọng mẫu, cùng slider. Nó tồn tại vì tách nguồn tốn cả GPU lẫn chất lượng:
stem đi ra mang theo transient bị nhoè và bóng mờ của bản phối, rồi những thứ đó
được convert cùng với giọng. File đã là giọng thì không nên trả cái giá đó.

Nó ăn theo `song` ở chỗ giữ nguyên tone (`keepsKey` trong `Advanced.tsx`) và ở
số bước diffusion mặc định — cả hai đều convert bằng checkpoint `singing`.

## Chế độ "Văn bản"

`app/components/TextInput.tsx`. Mode `tts` gõ chữ thay cho chọn file —
bước 2 của form đổi, phần còn lại giữ nguyên, vì backend đọc chữ ra rồi mới đưa
qua đúng bước chuyển giọng của mode `speech`.

Chọn ngôn ngữ nằm ngay cạnh ô gõ chứ không nằm trong "Tinh chỉnh": checkpoint
đọc tiếng Việt không đọc được tiếng Đức, nên chọn sai không phải là chỉnh cho
hay hơn mà là ra một bản đọc trôi chảy của thứ vô nghĩa. `LANGUAGES` trong
`lib/params.ts` là bản sao của `modal_app/tts.py` — thêm ngôn ngữ thì sửa cả hai,
và backend là bên từ chối.

`maxChars` khác nhau theo ngôn ngữ vì ký tự không phải đơn vị của lời nói: 2000
ký tự tiếng Việt và 700 ký tự tiếng Nhật là cùng khoảng 2–3 phút audio. Đổi
ngôn ngữ có thể làm đoạn đang gõ vượt giới hạn mới — `maxLength` không lấy chữ
về được, nên bộ đếm chuyển sang "thừa N ký tự" và nút chuyển giọng khoá lại cho
tới khi cắt bớt.

Ô gõ nói thẳng rằng số và ký hiệu sẽ bị bỏ qua khi đọc. Đó không phải lời khuyên
phong cách: model tokenise theo ký tự trên một bảng từ vựng không có chữ số, nên
"25 tuổi" đọc thành "tuổi".

## Ghi âm trong trình duyệt

`app/components/Recorder.tsx`. Thiết bị test chính là iOS Safari, và nó quyết
định gần hết thiết kế của component: container `audio/mp4` (không phải webm),
`ondataavailable` chỉ đáng tin lúc `stop()` nên không truyền `timeslice`, độ dài
đo bằng đồng hồ chứ không đọc `duration` của blob (Safari trả `Infinity`), và
`AudioContext` phải `resume()` sau cử chỉ người dùng thì thanh mức mới nhảy.
Trình duyệt không có `MediaRecorder` thì tab "Ghi âm" tự ẩn.

## Giọng có sẵn

`public/presets/index.json` ship rỗng — không có giọng nào thì hàng "Giọng có
sẵn" tự ẩn. Cách thêm và hai ràng buộc bắt buộc: `public/presets/README.md`.

## Điều khoản sử dụng

`app/terms/page.tsx` — plan §8 mục 3. Link từ hai chỗ: ô đồng thuận (mở tab
mới, để form không mất file đang chọn) và footer trang chủ. Render lúc chạy chứ
không prerender, để đổi `CONTACT_EMAIL` không phải build lại.

Trang này chỉ viết ra những gì code thật sự làm: cổng đồng thuận ở `/submit`,
TTL 6 giờ, metadata `AI-generated` trên mọi file, và nhật
ký chỉ có mã job — không có nội dung audio, tên file hay địa chỉ IP. Đổi một
trong các con số đó thì sửa cả hai nơi.
