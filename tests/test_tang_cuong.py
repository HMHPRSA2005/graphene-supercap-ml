"""Theo dấu J danh định, không nhầm nhiễu cân với mức đo bị giấu."""
import numpy as np
import pandas as pd
import pytest

from ml.du_lieu import chia_kb1, chia_kb2, chia_kb3, chia_kb4, chia_trong
from ml.tang_cuong import CauHinhTangCuong, tao_tap_hoc


@pytest.mark.parametrize("cach", ["C", "D"])
@pytest.mark.parametrize("truong_hop", ["kb1", "kb2", "kb3_1", "kb3_2", "kb4", "trong_kb2"])
def test_chi_sinh_tu_phan_hoc(bang, cach, truong_hop):
    if truong_hop == "kb1":
        lan = chia_kb1(bang)[0]
    elif truong_hop == "kb2":
        lan = chia_kb2(bang)[0]
    elif truong_hop.startswith("kb3"):
        lan = chia_kb3(bang, k_list=(int(truong_hop[-1]),), chinh_sach=("thap_nhat",))[0]
    elif truong_hop == "kb4":
        lan = chia_kb4(bang)[0]
    else:
        lan = chia_trong(bang, chia_kb2(bang)[0])[0]
    that = bang.loc[lan.hoc]
    kt = bang.loc[lan.kiem_tra]
    tap = tao_tap_hoc(that, CauHinhTangCuong(cach=cach, n_ao=2), seed=17)
    goc = tap.loc[tap.nguon == "that"]
    ao = tap.loc[tap.nguon == "ao"]
    assert len(goc) == len(that) and len(ao) > 0
    pd.testing.assert_frame_equal(goc[["ratio", "J", "Csp", "Vbar"]].reset_index(drop=True),
                                  that[["ratio", "J", "Csp", "Vbar"]].reset_index(drop=True))
    assert set(tap.ratio) == set(that.ratio)
    assert tap.diem_goc.notna().all()
    assert set(goc.diem_goc) == set(that.diem_id)
    assert set(tap.diem_goc).isdisjoint(kt.diem_id)
    # D được nội suy từ đường khớp; không được tái sử dụng điểm kiểm tra.
    # Với các phép chia trên, lưới D không trùng mức bị giấu.
    for diem in kt.itertuples():
        trung = np.isclose(tap.ratio, diem.ratio, rtol=0, atol=1e-10)
        trung &= np.isclose(tap.J_nominal, diem.J, rtol=1e-10, atol=1e-10)
        assert not trung.any()
    if cach == "C":
        assert set(ao.diem_goc).issubset(that.diem_id)
        assert len(ao) == 2 * len(that)
    else:
        for ratio, nhom in ao.groupby("ratio"):
            j_hoc = that.loc[that.ratio == ratio, "J"]
            assert nhom.J_nominal.min() >= j_hoc.min() - 1e-9
            assert nhom.J_nominal.max() <= j_hoc.max() + 1e-9
            if j_hoc.nunique() == 1:
                assert nhom.J_nominal.nunique() == 1
        if truong_hop in ("kb2", "trong_kb2"):
            for ratio in kt.ratio.unique():
                assert ao.loc[ao.ratio == ratio, "J_nominal"].max() < kt.loc[kt.ratio == ratio, "J"].min()
