"""Bảng thân chương pivot đúng cấu hình; không đổi bảng đầy đủ."""
import numpy as np
import pandas as pd

from ml.ve_hinh_ch5 import (
    DauRa, THU_TU_GON, _bang_53_gon, bang_chinh, do_phu, kb4,
)


def test_luu_bang_mac_dinh_giu_dinh_dang_cu(tmp_path):
    out = DauRa(tmp_path)
    bang = pd.DataFrame({"mo_hinh": ["chon_dang_ham"], "MAPE_Csp": [np.nan]})
    out.luu_bang(bang, "ngan")
    assert (out.bang / "ngan.tex").read_text(encoding="utf-8") == (
        "% Cần \\usepackage{booktabs,longtable}\n% \n"
        "\\begin{longtable}{p{0.450\\linewidth}p{0.450\\linewidth}}\n"
        "\\toprule\nmo\\_hinh & MAPE\\_Csp \\\\ \\midrule \\endhead\n"
        "chon\\_dang\\_ham & — \\\\\n\\bottomrule\n\\end{longtable}\n"
    )
    out.luu_bang(pd.DataFrame([{f"cot_{i}": i for i in range(7)}]), "rong")
    assert "Bản ghi 1:" in (out.bang / "rong.tex").read_text(encoding="utf-8")


def _diem(kb, mo_hinh, cach="C", n_ao=100, sigma_lap=0.03):
    for seed in (0, 1):
        for ratio in (0.025, 0.1, 1.75):
            for muc in (2, 3):
                csp = 100.0 + muc
                yield dict(kb=kb, mo_hinh=mo_hinh, cach=cach, n_ao=n_ao,
                           sigma_lap=sigma_lap, seed=seed, ratio=ratio, muc=muc,
                           Csp=csp, Csp_hat=csp * (1.02 if kb == "KB1" else 1.04),
                           Pd=20.0, Pd_hat=21.0)


def _tex_gon(out, ten):
    tex = (out.bang / f"{ten}.tex").read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex
    assert "longtable" not in tex
    assert r"\centering" in tex and r"\small" in tex
    assert tex.index(r"\end{tabular}") < tex.index(r"\footnotesize")
    assert "chon_dang_ham" not in tex
    assert "n_ao" not in tex and "sigma_lap" not in tex
    return tex


def test_bang_52_pivot_tam_mo_hinh_khong_co_kb2_bo(tmp_path):
    hang = [r for kb in ("KB1", "KB2", "KB2_BO_0.075")
            for mh in (*THU_TU_GON[:8], "gng_r") for r in _diem(kb, mh)]
    out = DauRa(tmp_path)
    bang_chinh(pd.DataFrame(hang), out)
    day_du = pd.read_csv(out.bang / "bang_5_2.csv")
    assert len(day_du) == 24
    gon = pd.read_csv(out.bang / "bang_5_2_gon.csv")
    assert gon.shape == (8, 6)
    assert gon.mo_hinh.tolist() == ["Hàm mũ", "Chọn dạng hàm", "Ridge", "SVR", "RF", "GB", "MLP", "GPR"]
    assert gon.KB1_Csp.eq("2,00").all()
    assert gon.KB2_Csp.eq("4,00").all()
    tex = _tex_gon(out, "bang_5_2_gon")
    assert "KB2_BO" not in tex and r"$\pm$" not in tex.split(r"\end{tabular}")[0]
    assert tex.count(r"p{\dimexpr") == 6


def test_bang_52_khong_doi_cau_hinh_chuan_thanh_gan_nhat(tmp_path):
    out = DauRa(tmp_path)
    bang_chinh(pd.DataFrame(_diem("KB1", "gpr", n_ao=250, sigma_lap=0.07)), out)
    assert (out.bang / "bang_5_2.csv").exists()
    assert not (out.bang / "bang_5_2_gon.tex").exists()


def test_bang_53_pivot_chi_trung_binh_va_moc_tu_so_lieu(tmp_path):
    hang = []
    for kb, moc in (("KB1", 9.328227), ("KB2", 112.537223)):
        for mh in ("gpr", "mlp", "svr"):
            for i, cach in enumerate("ABCD"):
                hang.append(dict(kb=kb, mo_hinh=mh, cach=cach, n_ao=100,
                                 sigma_lap=0.03, MAPE_Csp_tb=10+i, MAPE_Csp_sd=1))
            hang.append(dict(kb=kb, mo_hinh=mh, cach="C", n_ao=250,
                             sigma_lap=0.03, MAPE_Csp_tb=999, MAPE_Csp_sd=1))
        hang.append(dict(kb=kb, mo_hinh="chon_dang_ham", cach="moc_dang_ham", n_ao=0,
                         sigma_lap=0, MAPE_Csp_tb=moc, MAPE_Csp_sd=0))
    out = DauRa(tmp_path)
    _bang_53_gon(pd.DataFrame(hang), out)
    gon = pd.read_csv(out.bang / "bang_5_3_gon.csv")
    assert gon.shape == (6, 6)
    assert gon.mo_hinh.tolist() == ["GPR", "MLP", "SVR"] * 2
    np.testing.assert_allclose(gon[list("ABCD")], np.tile([10, 11, 12, 13], (6, 1)))
    tex = _tex_gon(out, "bang_5_3_gon")
    assert "9,33" in tex and "112,54" in tex
    assert r"$\pm$" not in tex and "999" not in tex


def test_bang_56_pivot_bien_trong_va_doi_chung(tmp_path):
    hang = [r for mh in THU_TU_GON for r in _diem("KB4", mh)]
    hang.extend(_diem("KB4", "gpr", cach="A"))
    out = DauRa(tmp_path)
    kb4(pd.DataFrame(hang), out)
    gon = pd.read_csv(out.bang / "bang_5_6_gon.csv")
    assert gon.shape == (10, 5)
    assert gon.mo_hinh.iloc[-2:].tolist() == ["TB hình học", "TB theo mức"]
    assert len(pd.read_csv(out.bang / "bang_5_6.csv")) == 22
    tex = _tex_gon(out, "bang_5_6_gon")
    assert "0,025 và 1,75" in tex


def test_do_phu_chi_sinh_bang_day_du_giu_cac_cau_hinh(tmp_path):
    hang = [dict(kb="KB3", mo_hinh=mh, cach="C", n_ao=100, sigma_lap=0.03,
                 chinh_sach=cs, k=k, seed=seed, ratio=0.1, Csp=100.0,
                 Csp_L95=90.0, Csp_U95=110.0)
            for mh in ("gpr", "mlp") for cs in (None, "trai_deu", "thap_nhat")
            for k in (1, 2) for seed in (0, 1)]
    out = DauRa(tmp_path)
    do_phu(pd.DataFrame(hang), out)
    assert len(pd.read_csv(out.bang / "do_phu.csv")) == 12
    assert not (out.bang / "do_phu_gon.tex").exists()


def test_tabular_dau_phay_ti_le_va_gia_tri_thieu(tmp_path):
    out = DauRa(tmp_path)
    bang = pd.DataFrame({"ratio": [0.025], "Ed": [np.nan], "MAPE": [1.2345]})
    out.luu_tabular(bang, "gon", "Ghi chú.", {"ratio": "Tỉ lệ", "Ed": "Năng lượng", "MAPE": "Sai số"})
    tex = _tex_gon(out, "gon")
    assert r"0,025 & — & 1,23 \\" in tex
    pd.testing.assert_frame_equal(pd.read_csv(out.bang / "gon.csv"), bang)
