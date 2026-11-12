"""Chọn best chỉ từ cấu hình đủ lần chia; không xếp hạng kết quả dở dang."""
import json

import pandas as pd

from ml import du_lieu
from ml.luu_best import bang_diem, chon_best, khop_tu_cau_hinh


def _hang(bang, kb, cac_lan, mo_hinh, sai_so, tham_so='{"alpha": 0.01, "degree": 1}'):
    hang = []
    for f in cac_lan:
        for _, r in bang.loc[f.kiem_tra].iterrows():
            hang.append(dict(kb=kb, ma_lan_chia=f.ma, mo_hinh=mo_hinh, cach="A", n_ao=0, sigma_lap=0.03,
                             kieu_ma_hoa="onehot", cau_hinh_chon="{}", seed=0, Csp=r.Csp,
                             Csp_hat=r.Csp * (1 + sai_so), Pd=r.Pd, Pd_hat=r.Pd * (1 + sai_so),
                             khong_hop_le=False, tham_so=tham_so))
    return hang


def test_chon_best_bo_cau_hinh_thieu_lan_chia(bang):
    kb1, kb2 = du_lieu.chia_kb1(bang), du_lieu.chia_kb2(bang)
    kq = pd.DataFrame(
        _hang(bang, "KB1", kb1, "ridge", 0.10) + _hang(bang, "KB2", kb2, "ridge", 0.20)
        # mlp tốt hơn nhưng mới chạy một lần chia KB1: không được chọn.
        + _hang(bang, "KB1", kb1[:1], "mlp", 0.01) + _hang(bang, "KB2", kb2, "mlp", 0.01)
        # Đồ thị chỉ thuộc phụ lục: dù có điểm thấp và đủ fold cũng không lọt best chính.
        + _hang(bang, "KB1", kb1, "pgcn_r", 0.001) + _hang(bang, "KB2", kb2, "pgcn_r", 0.001))
    diem = bang_diem(kq, bang)
    tot, rong = chon_best(diem, "chung")
    assert tot["mo_hinh"] == "ridge"
    assert len(rong) == 1
    assert abs(tot["chi_tiet_kb"]["KB1"]["MAPE_Csp"] - 10) < 1e-9
    tot2, _ = chon_best(diem, "KB2")
    assert tot2["mo_hinh"] == "mlp"
    assert chon_best(diem, "co_khoang")[0] is None


def test_khop_tu_cau_hinh_du_doan_duoc(bang):
    mh = khop_tu_cau_hinh(dict(mo_hinh="ridge", cach="A", n_ao=0, sigma_lap=0.03, kieu_ma_hoa="onehot",
                               cau_hinh_chon="{}", tham_so=json.dumps({"alpha": 0.01, "degree": 1})))
    dd = mh.predict(bang[["ratio", "J"]])
    assert (dd.Csp_hat > 0).all() and len(dd) == len(bang)
