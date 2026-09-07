"""Tên model thành tên thư mục trên Volume, và ba endpoint RVC đều mở không
xác thực — nên phần chuẩn hoá tên là chỗ đáng test nhất trong modal_rvc.py.

Bản sao của cùng thuật toán nằm ở web/app/rvc/page.tsx (`normalise`) để trang
hiện trước tên sẽ lưu. Sửa một bên thì sửa cả hai.
"""

from modal_rvc import _safe_name


def test_bo_dau_tieng_viet_thay_vi_tu_choi():
    # Người dùng gõ tên có dấu là chuyện bình thường; từ chối là cách chắc chắn
    # làm họ bỏ cuộc giữa chừng.
    assert _safe_name("Sơn Tùng MTP") == "son-tung-mtp"
    assert _safe_name("Mỹ Tâm") == "my-tam"


def test_chu_d_gach_ngang():
    # "đ" không tách ra dấu tổ hợp như các nguyên âm, NFD để nguyên nó.
    assert _safe_name("Đông Nhi") == "dong-nhi"


def test_khoang_trang_va_chu_hoa():
    assert _safe_name("  Son Tung  ") == "son-tung"
    assert _safe_name("SON TUNG") == "son-tung"


def test_khong_the_thoat_ra_khoi_thu_muc_models():
    # Đây là lý do hàm này tồn tại: name đi thẳng vào pathlib.Path(MODELS_DIR) / name.
    assert _safe_name("../../etc/passwd") == "etc-passwd"
    assert _safe_name("..") is None
    assert _safe_name("/") is None
    # ".." còn sót lại ở giữa là vô hại: kết quả vẫn là *một* đoạn tên, không
    # còn dấu phân cách nào để leo lên thư mục cha.
    assert _safe_name("a/../../b") == "a-..-..-b"

    # Tính chất thật sự cần giữ, viết thẳng ra để lần sửa sau không làm hỏng:
    # kết quả không bao giờ chứa dấu phân cách, và không bao giờ là "." hay "..".
    for name in ("../../etc/passwd", "a/../../b", "..", "/", "....//....", "\\", "a\x00b"):
        out = _safe_name(name)
        assert out is None or ("/" not in out and "\\" not in out and out not in {".", ".."})


def test_tra_ve_none_khi_khong_con_gi_dung_duoc():
    assert _safe_name("") is None
    assert _safe_name("   ") is None
    assert _safe_name("!!!") is None
    assert _safe_name("...") is None


def test_khong_de_lai_gach_thua():
    assert _safe_name("a   b") == "a-b"
    assert _safe_name("-son-tung-") == "son-tung"
    # Gạch dưới là ký tự hợp lệ cho tên thư mục, không đụng vào.
    assert _safe_name("son___tung") == "son___tung"


def test_giu_nguyen_ten_da_dung_chuan():
    assert _safe_name("son-tung") == "son-tung"
    assert _safe_name("model_v2.1") == "model_v2.1"
