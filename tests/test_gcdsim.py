"""Kiểm tra số lần đo ảo mặc định, công thức lab và khả năng tái lập."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from gcdsim import gcdsim as gs
from openpyxl import load_workbook


@pytest.fixture(scope="module")
def mo_phong():
    goc = Path(__file__).resolve().parents[1]
    bang = gs.read_lab_excel(str(goc / "data" / "so_lieu_GCD.xlsx"))
    fits = gs.attach_measured(gs.fit_electrodes(bang), bang)
    return bang, fits


@pytest.fixture(scope="module")
def bang_ao_mac_dinh(mo_phong):
    bang, fits = mo_phong
    return gs.generate_table(fits, gs.SimConfig(), bang)


def test_mac_dinh_du_10000_lan_moi_dien_cuc(bang_ao_mac_dinh):
    ao = bang_ao_mac_dinh
    assert len(ao) == 480000
    nhom = ao.groupby(["ratio", "run_id"])
    assert nhom.ngroups == 8 * 10000
    assert nhom.size().eq(6).all()
    assert nhom.J_nominal.nunique().eq(6).all()
    assert ao.groupby("ratio").run_id.nunique().eq(10000).all()


def test_nang_luong_dung_cong_thuc_lab(bang_ao_mac_dinh):
    np.testing.assert_allclose(
        bang_ao_mac_dinh.Ed, bang_ao_mac_dinh.Csp * 2.1 ** 2 / 7.2,
        rtol=1e-14, atol=0,
    )


def test_cung_seed_tai_lap_toan_bo_bang(mo_phong):
    bang, fits = mo_phong
    cfg = gs.SimConfig(n_runs=5, seed=42)
    lan_1 = gs.generate_table(fits, cfg, bang)
    lan_2 = gs.generate_table(fits, cfg, bang)
    pd.testing.assert_frame_equal(lan_1, lan_2, check_exact=True)
    khac_seed = gs.generate_table(fits, gs.SimConfig(n_runs=5, seed=43), bang)
    assert not np.array_equal(lan_1.Csp, khac_seed.Csp)


def test_xuat_excel_giu_du_lieu_va_cac_sheet_kiem_chung(tmp_path, monkeypatch):
    goc = Path(__file__).resolve().parents[1]
    dau_ra = tmp_path / "du_lieu_ao.xlsx"
    monkeypatch.setattr(sys, "argv", [
        "gcdsim", str(goc / "data" / "so_lieu_GCD.xlsx"),
        "--out", str(dau_ra), "--n", "3", "--seed", "42",
    ])
    gs.main()
    wb = load_workbook(dau_ra, read_only=True, data_only=True)
    try:
        assert wb.sheetnames == [
            "du_lieu_ao", "du_lieu_goc", "quy_luat_khop", "cau_hinh",
            "kiem_chung_tung_diem", "kiem_chung_tong_hop",
        ]
        ao = pd.DataFrame(wb["du_lieu_ao"].values)
        ao.columns = ao.iloc[0]
        ao = ao.iloc[1:]
        assert len(ao) == 48 * 3
        np.testing.assert_allclose(
            ao.Ed.to_numpy(float), ao.Csp.to_numpy(float) * 2.1 ** 2 / 7.2,
        )
        cau_hinh = list(wb["cau_hinh"].values)
        assert dict(zip(cau_hinh[0], cau_hinh[1]))["n_runs"] == 3
        assert len(list(wb["kiem_chung_tung_diem"].values)) == 49
        tong_hop = dict(list(wb["kiem_chung_tong_hop"].values)[1:])
        assert tong_hop["so_diem_that"] == 48
        assert tong_hop["Ed_bang_0_6125_Csp"] is True
    finally:
        wb.close()


@pytest.mark.parametrize("duoi", [".xlsx", ".csv"])
def test_loi_thay_the_giu_nguyen_tep_cu_va_don_tep_tam(tmp_path, monkeypatch, duoi):
    goc = Path(__file__).resolve().parents[1]
    dau_ra = tmp_path / f"du_lieu_ao{duoi}"
    dau_ra.write_bytes(b"san pham cu")
    monkeypatch.setattr(sys, "argv", [
        "gcdsim", str(goc / "data" / "so_lieu_GCD.xlsx"),
        "--out", str(dau_ra), "--n", "2",
    ])

    def loi_thay_the(tep_tam, tep_dich):
        assert Path(tep_tam) == tmp_path / f"du_lieu_ao.tmp{duoi}"
        assert Path(tep_tam).stat().st_size > 0
        assert Path(tep_dich) == dau_ra
        raise OSError("Không thay được tệp đích")

    monkeypatch.setattr(gs.os, "replace", loi_thay_the)
    with pytest.raises(OSError, match="Không thay được tệp đích"):
        gs.main()
    assert dau_ra.read_bytes() == b"san pham cu"
    assert not (tmp_path / f"du_lieu_ao.tmp{duoi}").exists()
