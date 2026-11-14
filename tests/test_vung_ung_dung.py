"""Biên Δt, miền đo, khoảng rời nhau và lựa chọn điểm ứng dụng."""
import numpy as np
import pandas as pd
import pytest

from ml import vung_ung_dung as vud
from ml.giao_dien import khung_du_doan


@pytest.fixture
def luoi():
    # Các mốc đúng 10/100 s, thêm điểm nằm ngay hai phía của mốc.
    dt = np.array([9.0, 10.0, 11.0, 99.0, 100.0, 101.0])
    ed = np.array([1.0, 3.0, 2.0, 4.0, 5.0, 6.0])
    return pd.DataFrame({"ratio": 0.1, "J": np.arange(1.0, 7.0),
                         "dong_mA": np.arange(1.0, 7.0)*10,
                         "Csp": ed/0.6125, "Vbar": 1.0,
                         "Ed": ed, "Pd": 3600*ed/dt, "dt": dt,
                         "khong_hop_le": False})


def test_truy_van_giu_dung_bien_dong(luoi):
    doan = vud.cac_khoang(luoi, dt_min=10, dt_max=100)
    assert len(doan) == 1
    assert doan[0].dt.tolist() == [10, 11, 99, 100]
    assert vud.cac_khoang(luoi, dt_min=10, dt_max=10)[0].dt.tolist() == [10]
    assert vud.cac_khoang(luoi, dt_max=10)[0].dt.tolist() == [9, 10]
    assert vud.cac_khoang(luoi, dt_min=100)[0].dt.tolist() == [100, 101]


def test_phan_vung_khong_lap_moc(luoi):
    bang = vud.bang_vung({0.1: luoi})
    assert bang.J.tolist() == ["1–1", "2–4", "5–6"]
    assert bang.vung.tolist() == list(vud.TEN_VUNG)
    assert len(bang.columns) == 6
    assert bang.Pd.iloc[0] == "0,4–0,4"
    tuy_chon = vud.bang_vung({0.1: luoi}, (11, 101))
    assert tuy_chon.J.tolist() == ["1–2", "3–5", "6–6"]


class MoHinhDuong:
    ten = "ridge"

    def predict(self, X):
        return khung_du_doan(X, np.full(len(X), 100.0), np.full(len(X), 0.8))


def test_luoi_va_truy_van_khong_vuot_mien_do(bang):
    for ratio, that in bang.groupby("ratio"):
        du_doan = vud.du_doan_mien_do(MoHinhDuong(), bang, ratio)
        assert len(du_doan) >= 200
        assert du_doan.J.min() == that.J.min()
        assert du_doan.J.max() == that.J.max()
        np.testing.assert_allclose(np.diff(np.log(du_doan.J)), np.diff(np.log(du_doan.J))[0])
        np.testing.assert_allclose(du_doan.dong_mA.iloc[[0, -1]], [10, 200])
        for doan in vud.cac_khoang(du_doan, dt_max=100):
            assert doan.J.between(that.J.min(), that.J.max()).all()
            assert (doan.dt <= 100).all()
        kq = vud.bang_truy_van({ratio: du_doan}, pd_min=1)
        assert that.J.min() <= kq["J* (A/g)"].iloc[0] <= that.J.max()


def test_rong_khi_khong_thoa(luoi):
    assert vud.cac_khoang(luoi, dt_max=1) == []
    assert vud.cac_khoang(luoi, pd_min=1e20) == []
    assert vud.cac_khoang(luoi, ed_min=1e20) == []
    kq = vud.bang_truy_van({0.1: luoi}, dt_min=1000)
    assert kq["J (A/g)"].iloc[0] == "không có trong miền đo"
    assert kq["J* (A/g)"].iloc[0] == "—"
    assert vud.bang_vung({0.1: luoi.iloc[:0]}).empty


def test_khong_noi_qua_loi_hoac_diem_khong_thoa(luoi):
    luoi.loc[1, "khong_hop_le"] = True
    luoi.loc[3, "Ed"] = np.inf
    luoi.loc[4, "Pd"] = 0
    doan = vud.cac_khoang(luoi, dt_max=100)
    assert [g.J.tolist() for g in doan] == [[1], [3]]
    luoi.loc[1, "khong_hop_le"] = False
    luoi.loc[1, "Ed"] = -1
    assert [g.J.tolist() for g in vud.cac_khoang(luoi)] == [[1], [3], [6]]


def test_cung_pd_noi_suy_log_log_khong_ngoai_suy(luoi):
    luoi["Pd"] = np.array([2500, 5000, 10000, 20000, 40000, 80000], dtype=float)
    luoi["Ed"] = np.sqrt(luoi.Pd)
    gon = vud.bang_cung_pd({0.1: luoi, 0.5: luoi.iloc[1:-1]}, (2.5, 5, 7, 80, 100))
    np.testing.assert_allclose(gon.iloc[0, 1:5].to_numpy(float),
                               np.sqrt([2500, 5000, 7000, 80000]))
    assert np.isnan(gon.Ed_100).all()
    assert np.isnan(gon.loc[1, "Ed_2.5"]) and np.isnan(gon.loc[1, "Ed_80"])
    luoi.loc[2, "khong_hop_le"] = True
    gon = vud.bang_cung_pd({0.1: luoi}, (7,))
    assert np.isnan(gon.Ed_7.iloc[0])


def test_chon_ed_lon_nhat_va_pd_khi_loc_ed(luoi):
    kq = vud.bang_truy_van({0.1: luoi}, dt_max=100)
    assert kq["Ed* (Wh/kg)"].iloc[0] == 5
    assert kq["J* (A/g)"].iloc[0] == 5
    kq = vud.bang_truy_van({0.1: luoi}, ed_min=2)
    assert kq["Pd* (W/kg)"].iloc[0] == luoi.loc[1, "Pd"]
    assert kq["J* (A/g)"].iloc[0] == 2
    kq = vud.bang_truy_van({0.1: luoi}, pd_min=500)
    assert kq["J* (A/g)"].iloc[0] == 2


def test_cli_tao_du_hinh_bang_va_loc_ti_le(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vud, "tai_best", lambda p: (
        MoHinhDuong(), {"muc_tieu": "chung", "cau_hinh": {"mo_hinh": "ridge", "cach": "A"}}))
    argv = ["--best", "best.joblib", "--thu-muc", str(tmp_path), "--ti-le", "0.1"]
    vud.main(argv)
    bang = pd.read_csv(tmp_path / "bang_ch5/vung_ung_dung.csv")
    assert bang.ratio.unique().tolist() == [0.1]
    assert bang.columns.tolist() == vud.COT_BANG
    tex = (tmp_path / "bang_ch5/vung_ung_dung.tex").read_text(encoding="utf-8")
    assert "khớp lại trên 48 điểm để sử dụng, chỉ trong miền đo" in tex
    assert tex.count(r"\dimexpr") == 6
    assert r"\begin{tabular}" in tex and "longtable" not in tex
    assert "kW/kg" in tex
    assert tex.index(r"\end{tabular}") < tex.index(r"\footnotesize")
    assert "dấu ;" not in tex
    cung_pd = pd.read_csv(tmp_path / "bang_ch5/ragone_cung_pd.csv")
    assert cung_pd.ratio.tolist() == [0.1]
    assert list(cung_pd) == ["ratio", "Ed_5", "Ed_10", "Ed_20", "Ed_40", "Ed_80"]
    tex_pd = (tmp_path / "bang_ch5/ragone_cung_pd.tex").read_text(encoding="utf-8")
    assert "log–log" in tex_pd and r"\begin{tabular}" in tex_pd
    for duoi in ("pdf", "png"):
        assert (tmp_path / f"hinh_ch5/ragone_vung_ung_dung.{duoi}").stat().st_size > 0
    assert vud.GHI_CHU in capsys.readouterr().out
    vud.main([*argv, "--dt-min", "1000000"])
    assert "không có trong miền đo" in capsys.readouterr().out


def test_hinh_nhan_thap_phan_va_hai_vi_du(tmp_path, bang, monkeypatch):
    out = vud.OutCh5(tmp_path)
    luoi = vud.du_doan_mien_do(MoHinhDuong(), bang, 0.1)
    # Bảo đảm có hai vùng để kiểm tra nhãn độc lập với mô hình giả lập.
    luoi["dt"] = np.geomspace(1, 1000, len(luoi))
    luoi["Ed"] = luoi.Pd * luoi.dt / 3600
    hinh = []
    monkeypatch.setattr(out, "luu_hinh", lambda fig, *args: hinh.append(fig))
    vud.ve_ragone(out, {0.1: luoi}, bang, (10, 100))
    ax = hinh[0].axes[0]
    assert "r = 0,1" in ax.get_legend_handles_labels()[1]
    nhan = [t.get_text() for t in ax.texts]
    assert "phanh tái sinh (r = 0,1)" in nhan
    assert "dự phòng (r = 0,1)" in nhan
    for dt in (1, 10, 100):
        assert f"Δt = {dt} s" in nhan
    vud.plt.close(hinh[0])


@pytest.mark.parametrize("them", [
    ["--dt-min", "100", "--dt-max", "10"],
    ["--dt-moc", "100", "10"],
    ["--dt-max", "nan"],
    ["--pd-min", "0"],
    ["--dt-max", "10", "--ed-min", "5"],
    ["--pd-min", "100", "--ed-min", "5"],
    ["--ti-le", "0.123"],
])
def test_cli_bao_dieu_kien_khong_hop_le(tmp_path, them):
    with pytest.raises(SystemExit) as exc:
        vud.main(["--best", "khong_can_nap.joblib", "--thu-muc", str(tmp_path), *them])
    assert exc.value.code == 2
