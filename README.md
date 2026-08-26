# Chuyengiong

Voice conversion web app: đưa vào một bài hát (hoặc đoạn thoại) + một giọng mẫu,
nhận về bản đã đổi sang giọng đó, nhạc nền giữ nguyên.

Kế hoạch chi tiết theo từng phase: [`docs/implementation-plan.md`](docs/implementation-plan.md).

## Trạng thái

| Phase | Nội dung | Trạng thái |
|---|---|---|
| 0 | Scaffold + deploy pipeline | 🟡 code xong, chờ verify deploy thật |
| 1 | Seed-VC + chunking | ⬜ chưa bắt đầu |
| 2 | Storage + job state | ⬜ |
| 3 | Nối pipeline hoàn chỉnh | ⬜ |
| 4 | Frontend | ⬜ |
| 5 | Pitch auto-detect | ⬜ |
| 6 | Consent gate | ⬜ |

## Cấu trúc

```
modal_app/
├── app.py      # Modal App, Volumes, Dict, images — nguồn duy nhất cho phần stateful
└── api.py      # FastAPI ASGI app; Phase 0 chỉ có /health
tests/          # chạy bằng pytest, không cần Modal credentials
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

## Trước khi code Phase 1

Seed-VC đổi API giữa các version. Đọc `README.md` + `requirements.txt` của
[`Plachtaa/seed-vc`](https://github.com/Plachtaa/seed-vc) và pin về một commit SHA
cụ thể trước khi viết `conversion.py` — xem mục cảnh báo trong plan.
