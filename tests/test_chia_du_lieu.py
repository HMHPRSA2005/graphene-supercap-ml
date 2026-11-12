"""Kịch bản chia số đo và các fold chọn tham số phải độc lập kiểm tra ngoài."""
import numpy as np
import pytest

from ml.du_lieu import (chia_kb1, chia_kb2, chia_kb3, chia_kb4, chia_kb5,
                        chia_trong, muc_giu_kb3)


def _kiem_tra_phan_hoach(hoc, kiem_tra, tat_ca):
    a, b = set(hoc), set(kiem_tra)
    assert len(a) == len(hoc) and len(b) == len(kiem_tra)
    assert a and b and a.isdisjoint(b)
    assert a | b == set(tat_ca)


def _fold_mong_doi(bang, lan):
    hoc = bang.loc[lan.hoc]
    if lan.kb in ("KB1", "KB3"):
        lo = hoc.groupby("ratio")["muc"].transform("min")
        hi = hoc.groupby("ratio")["muc"].transform("max")
        nhom = [hoc.index[(hoc.muc == m) & (lo < m) & (m < hi)] for m in range(2, 6)]
    else:
        ratios = sorted(hoc.ratio.unique())
        if lan.kb == "KB2":
            ratios = [r for r in ratios if r != lan.thong_tin["ratio_dich"]]
        nhom = []
        for i in range(4):
            mask = hoc.ratio.isin(ratios[i::4])
            if lan.kb == "KB2":
                mask &= hoc.muc == 6
            nhom.append(hoc.index[mask])
    return [set(n) for n in nhom if len(n)]


@pytest.mark.parametrize("kb", ["KB1", "KB2", "KB2_bo", "KB3", "KB4"])
def test_chia_ngoai_va_chia_trong(bang, kb):
    if kb == "KB1":
        cac_lan, so_lan = chia_kb1(bang), 32
    elif kb == "KB2":
        cac_lan, so_lan = chia_kb2(bang), 8
    elif kb == "KB2_bo":
        cac_lan, so_lan = chia_kb2(bang, bo_ti_le=(0.075,)), 7
    elif kb == "KB3":
        cac_lan, so_lan = chia_kb3(bang), 80
    else:
        cac_lan, so_lan = chia_kb4(bang), 8
    assert len(cac_lan) == so_lan
    assert len({lan.ma for lan in cac_lan}) == so_lan
    tat_ca = bang.index if kb != "KB2_bo" else bang.index[bang.ratio != 0.075]
    for lan in cac_lan:
        _kiem_tra_phan_hoach(lan.hoc, lan.kiem_tra, tat_ca)
        hoc, kt = bang.loc[lan.hoc], bang.loc[lan.kiem_tra]
        ratio = lan.thong_tin["ratio_dich"]
        assert kt.ratio.eq(ratio).all()
        if kb == "KB1":
            assert len(hoc) == 47 and len(kt) == 1
            assert 2 <= kt.muc.iloc[0] <= 5
        elif kb.startswith("KB2"):
            assert len(hoc) == len(tat_ca) - 1 and len(kt) == 1
            assert kt.muc.eq(6).all()
            if kb == "KB2_bo":
                assert not hoc.ratio.eq(0.075).any()
        elif kb == "KB3":
            k = lan.thong_tin["k"]
            giu = muc_giu_kb3(k, lan.thong_tin["chinh_sach"])
            assert len(hoc) == 42 + k and len(kt) == 6 - k
            assert sorted(hoc.loc[hoc.ratio == ratio, "muc"]) == giu
            assert set(kt.muc) == set(range(1, 7)) - set(giu)
        else:
            assert len(hoc) == 42 and len(kt) == 6
            assert not hoc.ratio.eq(ratio).any()
            assert lan.thong_tin["kieu_ma_hoa"] == "ti_le"
            assert lan.thong_tin["bien"] == (ratio in (bang.ratio.min(), bang.ratio.max()))
        folds = chia_trong(bang, lan)
        mong_doi = _fold_mong_doi(bang, lan)
        assert len(folds) == len(mong_doi)
        for fold, kt_mong_doi in zip(folds, mong_doi):
            _kiem_tra_phan_hoach(fold.hoc, fold.kiem_tra, lan.hoc)
            assert set(fold.kiem_tra) == kt_mong_doi
            assert set(fold.hoc).isdisjoint(lan.kiem_tra)
            assert set(fold.kiem_tra).isdisjoint(lan.kiem_tra)
            assert fold.thong_tin["kieu_ma_hoa"] == lan.thong_tin["kieu_ma_hoa"]


def test_muc_trai_deu():
    for k, mong_doi in enumerate(([1], [1, 6], [1, 4, 6], [1, 3, 4, 6], [1, 2, 4, 5, 6]), 1):
        assert muc_giu_kb3(k, "trai_deu") == mong_doi
        assert muc_giu_kb3(k, "thap_nhat") == list(range(1, k + 1))


def test_kb5_kiem_tra_co_dinh(bang):
    cac_dot, kiem_tra = chia_kb5(bang)
    assert len(cac_dot) == 4 and len(kiem_tra) == 16
    assert set(bang.loc[kiem_tra, "muc"]) == {3, 5}
    for hoc, so_diem, ratios in zip(cac_dot, (12, 8, 12, 32),
                                   ({0.025, 0.05, 0.075}, {0.1, 0.25}, {0.5, 1.0, 1.75}, set(bang.ratio))):
        assert len(hoc) == so_diem and len(set(hoc)) == so_diem
        assert set(hoc).isdisjoint(kiem_tra)
        assert set(bang.loc[hoc, "muc"]) == {1, 2, 4, 6}
        assert set(bang.loc[hoc, "ratio"]) == ratios
    assert all(set(cac_dot[i]).isdisjoint(cac_dot[j]) for i in range(3) for j in range(i))
    assert set(np.concatenate(cac_dot[:3])) == set(cac_dot[3])
    assert set(cac_dot[3]) | set(kiem_tra) == set(bang.index)
