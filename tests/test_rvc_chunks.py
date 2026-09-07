"""Cắt khúc bài dài phải giữ nguyên tổng độ dài.

Bản đầu dùng `pydub.silence.split_on_silence`, mà đó là hàm *vứt bỏ* khoảng
lặng. Một bài 1:31 ra 1:09 — lời vẫn đủ nên nghe qua tưởng đúng, nhưng vocal
không còn khớp với nhạc nền, mà ghép lại với nhạc nền chính là mục đích.

Nên tính chất cần giữ không phải "chỗ nối nghe êm" mà là "không mất một
mili-giây nào". Test viết quanh đúng điều đó.
"""

import pytest

from modal_rvc import _chunk_bounds

LIMIT = 90_000


def _covers(bounds, total):
    """Các đoạn liền nhau, phủ kín [0, total), không chồng, không hở."""
    assert bounds[0][0] == 0
    assert bounds[-1][1] == total
    for (_, end), (nxt, _) in zip(bounds[:-1], bounds[1:], strict=True):
        assert end == nxt
    assert sum(end - start for start, end in bounds) == total


def test_ngan_hon_gioi_han_thi_khong_cat():
    assert _chunk_bounds(60_000, [], LIMIT) == [(0, 60_000)]


def test_dung_bang_gioi_han_thi_khong_cat():
    assert _chunk_bounds(LIMIT, [], LIMIT) == [(0, LIMIT)]


def test_bai_1_31_giu_nguyen_do_dai():
    # Đúng ca đã hỏng: 1:31 = 91 giây, vượt mốc 90 giây nên rơi vào nhánh cắt.
    total = 91_000
    quiet = [(20_000, 24_000), (45_000, 52_000), (70_000, 73_000)]
    bounds = _chunk_bounds(total, quiet, LIMIT)
    _covers(bounds, total)
    # 91 giây chứ không phải 69 giây, dù có 14 giây khoảng lặng trong đó.
    assert sum(end - start for start, end in bounds) == 91_000


def test_cat_giua_khoang_lang():
    bounds = _chunk_bounds(120_000, [(80_000, 90_000)], LIMIT)
    _covers(bounds, 120_000)
    assert bounds[0][1] == 85_000  # điểm giữa của (80_000, 90_000)


def test_chon_khoang_lang_xa_nhat_con_trong_tam_de_cat_it_lan_nhat():
    quiet = [(10_000, 11_000), (50_000, 51_000), (85_000, 86_000)]
    bounds = _chunk_bounds(150_000, quiet, LIMIT)
    _covers(bounds, 150_000)
    assert bounds[0][1] == 85_500


def test_khong_co_khoang_lang_thi_cat_cung_nhung_van_du_do_dai():
    total = 200_000
    bounds = _chunk_bounds(total, [], LIMIT)
    _covers(bounds, total)
    assert all(end - start <= LIMIT for start, end in bounds)


def test_khoang_lang_ngoai_tam_khong_keo_doan_dai_qua_gioi_han():
    # Khoảng lặng duy nhất nằm quá xa: phải cắt cứng chứ không được với tới nó.
    bounds = _chunk_bounds(200_000, [(150_000, 160_000)], LIMIT)
    _covers(bounds, 200_000)
    assert all(end - start <= LIMIT for start, end in bounds)


@pytest.mark.parametrize("total", [90_001, 91_000, 180_000, 240_000, 500_000])
def test_moi_do_dai_deu_phu_kin(total):
    quiet = [(i, i + 800) for i in range(30_000, total, 37_000)]
    _covers(_chunk_bounds(total, quiet, LIMIT), total)
