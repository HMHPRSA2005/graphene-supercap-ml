"""So sánh cùng J: miền đo, nguồn bất định, thứ hạng và xác suất thắng."""
import json

import numpy as np
import pandas as pd
import pytest

from ml import vat_ly
from ml.danh_gia import khoang_95
from ml.giao_dien import khung_du_doan
from ml.so_sanh import (KHONG_DAI, NGUON_SD, NGUON_SEED, _gom_seed,
                        du_doan_luoi_chung, luoi_J, tom_tat, xac_suat_tot_nhat,
                        xep_hang)


class MoHinhBietTruoc:
    ten = "gia"
    ho_tro_khoang = False

    def predict(self, X):
        csp = X.ratio.map({0.05: 20.0, 0.1: 40.0, 0.5: 10.0}).to_numpy()
        vbar = X.ratio.map({0.05: 1.2, 0.1: 0.8, 0.5: 1.0}).to_numpy()
        return khung_du_doan(X, csp, vbar)


@pytest.fixture
def bang_gia():
    return pd.DataFrame({"ratio": [0.05, 0.05, 0.1, 0.1, 0.5, 0.5],
                         "J": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0]})


def test_luoi_chung_trong_mien_cua_moi_dien_cuc(bang, bang_gia):
    for that in (bang, bang_gia):
        j = luoi_J(that, "chung", 25)
        bien = that.groupby("ratio").J.agg(["min", "max"])
        assert len(j) == 25
        assert j[0] == bien["min"].max()
        assert j[-1] == bien["max"].min()
        np.testing.assert_allclose(np.diff(np.log(j)), np.diff(np.log(j))[0])
        for g in (that.loc[that.ratio == r] for r in that.ratio.unique()):
            assert np.all((j >= g.J.min()) & (j <= g.J.max()))
    kq = du_doan_luoi_chung(MoHinhBietTruoc(), bang_gia, luoi_J(bang_gia))
    assert not kq.ngoai_mien_do.any()


def test_toan_bo_co_ngoai_mien_dung(bang_gia):
    j = luoi_J(bang_gia, "toan_bo", 17)
    assert j[0] == 1 and j[-1] == 30
    kq = du_doan_luoi_chung(MoHinhBietTruoc(), bang_gia, j)
    for r, g in kq.groupby("ratio"):
        that = bang_gia.loc[bang_gia.ratio == r]
        dung = (g.J < that.J.min()) | (g.J > that.J.max())
        np.testing.assert_array_equal(g.ngoai_mien_do, dung)
    assert kq.ngoai_mien_do.any()
    # Điểm biên chính xác không phải ngoại suy.
    assert not kq.loc[(kq.ratio == 0.05) & (kq.J == 1), "ngoai_mien_do"].item()
    assert not kq.loc[(kq.ratio == 0.5) & (kq.J == 30), "ngoai_mien_do"].item()


def test_xac_suat_tong_mot_tat_dinh_va_vuot_troi():
    mu = np.array([[1.0, 1.1, 1.2], [1.0, 5.0, 0.5], [1.5, 1.6, 1.4]])
    sd = np.full_like(mu, 0.08)
    p = xac_suat_tot_nhat(mu, sd, 2000, seed=17)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_array_equal(p, xac_suat_tot_nhat(mu, sd, 2000, seed=17))
    assert not np.array_equal(p, xac_suat_tot_nhat(mu, sd, 2000, seed=18))
    assert p[1, 1] > 0.999
    # Không thiên vị điện cực đầu khi hòa hoàn toàn.
    np.testing.assert_allclose(xac_suat_tot_nhat([1, 1], [0, 0], 20), [0.5, 0.5])
    assert np.isnan(xac_suat_tot_nhat([1, np.nan], [0.1, 0.1], 20)).all()


def test_xep_hang_csp_pd_biet_truoc(bang_gia):
    kq = du_doan_luoi_chung(MoHinhBietTruoc(), bang_gia, [3.0, 5.0, 10.0])
    hang = xep_hang(kq)
    for _, g in hang.groupby("J"):
        g = g.set_index("ratio")
        assert g.hang_Csp.to_dict() == {0.05: 2, 0.1: 1, 0.5: 3}
        assert g.hang_Pd.to_dict() == {0.05: 1, 0.1: 3, 0.5: 2}
        assert g.xac_suat_tot_nhat_Csp.isna().all()
    np.testing.assert_allclose(kq.Ed_hat, vat_ly.ed_tu_csp(kq.Csp_hat))
    np.testing.assert_allclose(kq.Pd_hat, vat_ly.pd_tu_vbar(kq.J, kq.Vbar_hat))


def test_sd_mo_hinh_dung_nguon_va_bien_vat_ly(bang_gia):
    class MoHinhSd(MoHinhBietTruoc):
        ho_tro_khoang = True

        def predict(self, X):
            y = super().predict(X)
            y["logCsp_sd"] = 0.1
            y["logVbar_sd"] = 0.03
            return y

    kq = du_doan_luoi_chung(MoHinhSd(), bang_gia, [3.0, 5.0])
    assert (kq.nguon_bat_dinh == NGUON_SD).all()
    for cot, sd in (("Csp", 0.1), ("Vbar", 0.03)):
        lo, hi = khoang_95(np.log(kq[cot + "_hat"]), sd)
        np.testing.assert_allclose(kq[cot + "_L95"], lo)
        np.testing.assert_allclose(kq[cot + "_U95"], hi)
    for bien in ("L95", "U95"):
        np.testing.assert_allclose(kq["Ed_" + bien], vat_ly.ed_tu_csp(kq["Csp_" + bien]))
        np.testing.assert_allclose(kq["Pd_" + bien], vat_ly.pd_tu_vbar(kq.J, kq["Vbar_" + bien]))
    hang = xep_hang(kq, so_mau=500)
    np.testing.assert_allclose(hang.groupby("J").xac_suat_tot_nhat_Csp.sum(), 1)
    assert hang.nguon_xac_suat.str.startswith("Monte Carlo").all()


def test_phan_tan_seed_khong_gia_lam_sd(bang_gia):
    cac_bang = []
    for seed, he_so in enumerate((0.8, 1, 1.3)):
        b = du_doan_luoi_chung(MoHinhBietTruoc(), bang_gia, [3.0, 5.0])
        b["Csp_hat"] *= he_so
        b["Vbar_hat"] *= he_so
        b["seed"], b["mo_hinh"] = seed, "gia"
        cac_bang.append(b)
    kq = _gom_seed(cac_bang, "C")
    assert (kq.nguon_bat_dinh == NGUON_SEED).all()
    assert kq.logCsp_sd.isna().all()
    lo, hi = np.quantile(np.stack([b.Csp_hat for b in cac_bang]), [0.025, 0.975], axis=0)
    np.testing.assert_allclose(kq.Csp_L95, lo)
    np.testing.assert_allclose(kq.Csp_U95, hi)
    hang = xep_hang(kq, du_doan_seeds=pd.concat(cac_bang))
    assert (hang.loc[hang.ratio == 0.1, "xac_suat_tot_nhat_Csp"] == 1).all()
    np.testing.assert_allclose(hang.groupby("J").xac_suat_tot_nhat_Csp.sum(), 1)
    assert hang.nguon_xac_suat.str.startswith("Tần suất").all()
    khong_dai = _gom_seed(cac_bang, "A")
    assert (khong_dai.nguon_bat_dinh == KHONG_DAI).all()
    assert "Csp_L95" not in khong_dai


def test_tom_tat_dong_thuan_va_doan_J(bang_gia):
    b = du_doan_luoi_chung(MoHinhBietTruoc(), bang_gia, [3.0, 5.0, 10.0])
    b["mo_hinh"] = "mot"
    c = b.copy()
    c["mo_hinh"] = "hai"
    c.loc[(c.J == 10) & (c.ratio == 0.5), "Csp_hat"] = 100
    tom = tom_tat(xep_hang(pd.concat([b, c]))).set_index("mo_hinh")
    assert tom.loc["mot", "ti_le_luoi_01_tot_nhat"] == 1
    assert tom.loc["hai", "ti_le_luoi_01_tot_nhat"] == pytest.approx(2 / 3)
    np.testing.assert_allclose(tom.ti_le_dong_thuan_toan_bo, 2 / 3)
    np.testing.assert_allclose(tom.dong_thuan_da_so_trung_binh, 5 / 6)
    doan = json.loads(tom.loc["hai", "doan_J_tot_nhat_Csp"])
    assert len(doan) == 2
    assert doan[0] == {"J_tu": 3, "J_den": 5, "ratio_tot_nhat": [0.1]}
    assert doan[1]["ratio_tot_nhat"] == [0.5]


def test_giu_loi_khong_nang_hang_du_doan_sai(bang_gia):
    class MoHinhLoi(MoHinhBietTruoc):
        def predict(self, X):
            y = super().predict(X)
            y.loc[X.ratio == 0.1, "Csp_hat"] = -2
            return y

    kq = du_doan_luoi_chung(MoHinhLoi(), bang_gia, [3.0])
    assert kq.loc[kq.ratio == 0.1, "khong_hop_le"].item()
    assert kq.loc[kq.ratio == 0.1, "Csp_hat"].item() == -2
    hang = xep_hang(kq)
    assert hang.loc[hang.ratio == 0.1, "hang_Csp"].isna().all()
