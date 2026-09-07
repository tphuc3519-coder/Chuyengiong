# Chuyengiong — ghi chú cho session sau

App đổi giọng / đổi beat: FastAPI + GPU trên Modal, Next.js trên Vercel.
`README.md` là nhật ký kỹ thuật theo từng Phase — đọc phase liên quan trước khi
sửa thứ nó nói tới, vì phần lớn con số trong repo này có lý do viết sẵn ở đó.

## Việc đang dở

**`docs/beat-tuning.md`** là trạng thái sống của việc đang làm: chỉnh hệ thống
đổi beat bằng tai. Mở nó trước khi đụng vào `beats.py`, `styles.py`,
`beatgen.py` hay `mixing.py` — nó ghi cái gì đã đo xong (đừng đo lại), cái gì
còn mở, và con số nào đang chờ một lượt nghe.

## Chạy kiểm tra

```bash
# Python — CI chạy đúng ba dòng này
ruff check . && ruff format --check . && pytest -q
# ffmpeg là bắt buộc; thiếu nó thì ~40 test tự skip chứ không đỏ
apt-get install -y ffmpeg

# Web
cd web && npm ci && npm run lint && npm run format:check && npm run typecheck && npm run build
```

Không cần credentials Modal và không cần GPU để chạy test.

## Cách viết ở repo này

Đây không phải sở thích, nó là thứ giữ cho repo đọc được sau sáu tháng:

- **Docstring và comment giải thích *tại sao*, không phải *cái gì*.** Một hằng
  số không có lý do bên cạnh là một hằng số sẽ bị ai đó sửa bừa. Nếu con số đến
  từ một phép đo, chép cả phép đo vào.
- **Nói thật cái gì chưa đo được.** Repo này có nhiều hằng số "suy luận, chưa
  đo" và chúng được ghi rõ là thế. Đừng viết như thể đã đo.
- **Bài học biến thành test.** Mỗi lần deploy hỏng ở đây đều để lại một test giữ
  đúng cái luật rút ra được. Xem `tests/test_beatgen.py` cho ví dụ.
- Docstring tiếng Anh, chữ trên UI và README tiếng Việt.
- README được nối thêm một mục mỗi phase, không sửa lại lịch sử cũ.

## Trước khi tự tay chỉnh một hằng số nghe được bằng tai

Kiểm tra xem nó đã là một cái núm trên UI chưa. Vài thứ tưởng là hằng số đã
thành setting rồi (`beat_follow`), vì đoán một con số cho người khác tốn của họ
một lượt deploy cộng một lượt nghe.
