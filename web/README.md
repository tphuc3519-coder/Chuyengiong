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

Chỉ một biến, và nó **không** có tiền tố `NEXT_PUBLIC_`: xem mục "Đường đi của
request" bên dưới.

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
