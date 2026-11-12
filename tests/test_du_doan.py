"""Đề xuất đo phải bỏ mức đã đo, và dự đoán lỗi không được tự sửa."""
import numpy as np
import pytest

from ml.du_doan import de_xuat_muc_do, du_doan_luoi
from ml.giao_dien import khung_du_doan


class MoHinhSdBietTruoc:
    ten = "gpr"
    ho_tro_khoang = True

    def __init__(self, khoi_luong):
        self.khoi_luong = khoi_luong
        self.dong_da_du_doan = None

    def predict(self, X):
        dong = X.J.to_numpy() * self.khoi_luong * 1000
        self.dong_da_du_doan = dong
        # Mức đã đo 200 mA có sd lớn nhất, nhưng tuyệt đối không được đề xuất lại.
        sd = np.where(np.isclose(dong, 200), 10.0,
                      np.where(np.isclose(dong, 70), 0.9, 0.1))
        return khung_du_doan(X, np.ones(len(X)), np.ones(len(X)), sd, sd)


def test_de_xuat_bo_muc_da_do_va_chon_argmax_sd(bang):
    g = bang.loc[bang.ratio == 0.1].sort_values("J")
    khoi_luong = 0.010 / float(g.J.iloc[0])
    mo_hinh = MoHinhSdBietTruoc(khoi_luong)
    kq = de_xuat_muc_do(mo_hinh, bang, 0.1, np.arange(10, 201, 10))
    da_do = g.J.to_numpy() * khoi_luong * 1000
    assert kq["dong_mA"] == 70
    assert kq["logCsp_sd"] == pytest.approx(0.9)
    assert kq["J"] == pytest.approx(0.070 / khoi_luong)
    assert not np.isclose(kq["dong_mA"], da_do).any()
    assert not np.isclose(mo_hinh.dong_da_du_doan[:, None], da_do[None, :]).any()
    assert len(mo_hinh.dong_da_du_doan) == 14


def test_de_xuat_khong_con_ung_vien(bang):
    g = bang.loc[bang.ratio == 0.1].sort_values("J")
    mo_hinh = MoHinhSdBietTruoc(0.010 / float(g.J.iloc[0]))
    kq = de_xuat_muc_do(mo_hinh, bang, 0.1, [10, 20, 50, 80, 100, 200])
    assert "dong_mA" not in kq
    assert "Không còn" in kq["ly_do"]
    assert mo_hinh.dong_da_du_doan is None


class MoHinhGiaTriLoi:
    ten = "ridge"
    ho_tro_khoang = False

    def predict(self, X):
        return khung_du_doan(X, [0, -2, np.nan, np.inf, 5, 6], [1, 1, 1, 1, 0, 1])


def test_danh_dau_loi_khong_sua_gia_tri():
    with np.errstate(divide="ignore", invalid="ignore"):
        kq = du_doan_luoi(MoHinhGiaTriLoi(), 0.1, [10, 20, 30, 40, 50, 60])
    assert kq.khong_hop_le.tolist() == [True, True, True, True, True, False]
    np.testing.assert_equal(kq.Csp.to_numpy(), [0, -2, np.nan, np.inf, 5, 6])
    assert kq.Vbar.iloc[4] == 0


def test_loi_ti_le_hien_so_thuong():
    with pytest.raises(ValueError, match="chọn một trong") as exc:
        du_doan_luoi(MoHinhGiaTriLoi(), 0.123, [10])
    assert "np.float" not in str(exc.value)
    assert "0.025, 0.05, 0.075, 0.1" in str(exc.value)


def test_tb_theo_muc_bao_thieu_muc_cho_j_moi():
    mo_hinh = MoHinhGiaTriLoi()
    mo_hinh.ten = "tb_theo_muc"
    with pytest.raises(ValueError, match="cần cột muc"):
        du_doan_luoi(mo_hinh, 0.1, [11])


def test_gpr_c_canh_bao_muc_chua_do(bang):
    g = bang.loc[bang.ratio == 0.1].sort_values("J")
    mo_hinh = MoHinhSdBietTruoc(0.010 / float(g.J.iloc[0]))
    mo_hinh.thong_tin_khop_ = {"cach": "C"}
    with pytest.warns(UserWarning, match="GPR.*cách C"):
        du_doan_luoi(mo_hinh, 0.1, [11])
