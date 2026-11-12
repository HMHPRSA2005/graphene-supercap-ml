"""Mô hình nền chỉ khớp số đo thật; dự phòng ít điểm không nhìn tập kiểm tra.

Nếu không có điện cực khác đủ dữ liệu, số mũ/độ dốc chung mặc định bằng 0
(hàm hằng). Thuộc tính ``du_phong`` trên ElectrodeFit ghi lý do dự phòng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from gcdsim import gcdsim as gs
from .giao_dien import MoHinhCoSo, khung_du_doan
from .vat_ly import vbar_tu_pd

_SO_THAM_SO = {"exp": 3, "power": 2, "dunn": 2}


def _vbar(bang):
    return (bang["Vbar"].to_numpy(float) if "Vbar" in bang
            else vbar_tu_pd(bang["J"], bang["Pd"]))


def _hop_le(y):
    return np.all(np.isfinite(y) & (np.asarray(y) > 0))


def _he_so_chung(bang, ratio):
    so_mu, do_doc = [], []
    for r, g in bang.groupby("ratio", sort=True):
        if r == ratio:
            continue
        j = g["J"].to_numpy(float)
        if len(g) >= 3:
            n = gs.fit_power(j, g["Csp"].to_numpy(float))[1]
            if np.isfinite(n):
                so_mu.append(n)
        if len(g) >= 2 and np.ptp(j) > 0:
            d = np.polyfit(j, _vbar(g), 1)[0]
            if np.isfinite(d):
                do_doc.append(d)
    return (float(np.median(so_mu)) if so_mu else 0.0,
            float(np.median(do_doc)) if do_doc else 0.0)


def khop_dien_cuc(bang_dien_cuc, dang="auto", bang_that_hoc=None) -> gs.ElectrodeFit:
    """Khớp một điện cực; bảng tham chiếu chỉ gồm số đo thật phần học.

    LOOCV chỉ xét dạng còn đủ số tham số khi bỏ một điểm. Dạng được yêu cầu
    không đủ điểm cũng dùng dự phòng lũy thừa chung, không khớp thiếu hạng.
    """
    if dang != "auto" and dang not in gs.MODELS:
        raise ValueError(f"Dạng hàm không hợp lệ: {dang}")
    g = bang_dien_cuc.sort_values("J")
    if g.empty or g["ratio"].nunique() != 1:
        raise ValueError("Cần số đo của đúng một điện cực")
    ratio = float(g["ratio"].iloc[0])
    j, y = g["J"].to_numpy(float), g["Csp"].to_numpy(float)
    v = _vbar(g)
    if not (_hop_le(j) and _hop_le(y) and _hop_le(v)):
        raise ValueError("Số đo J, Csp, Vbar phải hữu hạn và dương")
    tham_chieu = g if bang_that_hoc is None else bang_that_hoc
    n_chung, doc_chung = _he_so_chung(tham_chieu, ratio)
    scores, cac_tham_so = {}, {}
    for name in (gs.MODELS if dang == "auto" else (dang,)):
        if len(g) - 1 < _SO_THAM_SO[name]:
            continue
        _, fit, pred = gs.MODELS[name]
        try:
            errs = []
            for k in range(len(g)):
                mask = np.arange(len(g)) != k
                p = fit(j[mask], y[mask])
                yh = pred(p, j[k:k + 1])
                if not (np.all(np.isfinite(p)) and _hop_le(yh)):
                    raise ValueError("Khớp LOOCV không hợp lệ")
                errs.append(abs(yh[0] - y[k]) / y[k])
            p = fit(j, y)
            if not (np.all(np.isfinite(p)) and _hop_le(pred(p, np.geomspace(j.min(), j.max(), 32)))):
                raise ValueError("Đường khớp không hợp lệ")
            scores[name] = float(np.mean(errs) * 100)
            cac_tham_so[name] = tuple(map(float, p))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, OverflowError):
            scores[name] = float("inf")
    hop_le = [name for name in cac_tham_so if np.isfinite(scores[name])]
    du_phong = []
    if hop_le:
        name = min(hop_le, key=scores.get)
        params = cac_tham_so[name]
    else:
        name = "power"
        a = float(np.exp(np.mean(np.log(y) + n_chung * np.log(j))))
        params = (a, n_chung)
        du_phong.append("Csp: luy_thua_so_mu_chung")
    if len(g) >= 2 and np.ptp(j) > 0:
        v_params = tuple(map(float, np.polyfit(j, v, 1)[::-1]))
    else:
        v_params = (float(np.mean(v - doc_chung * j)), doc_chung)
        du_phong.append("Vbar: do_doc_chung")
    yh = gs.MODELS[name][2](params, j)
    resid = np.log(y / yh)
    sigma = float(np.sqrt(np.sum(resid ** 2) / max(len(g) - len(params), 1)))
    f = gs.ElectrodeFit(ratio, name, params, sigma, scores, v_params,
                        float(j.min()), float(j.max()))
    f.du_phong = tuple(du_phong)
    gs.attach_measured({ratio: f}, g)
    return f


def khop_tat_ca(bang_that_hoc, dang="auto") -> dict[float, gs.ElectrodeFit]:
    """Khớp từng điện cực, chia sẻ số mũ/độ dốc chỉ từ phần học hiện hành."""
    return {float(r): khop_dien_cuc(g, dang, bang_that_hoc)
            for r, g in bang_that_hoc.groupby("ratio", sort=True)}


class MoHinhDangHam(MoHinhCoSo):
    def __init__(self, dang="exp"):
        if dang not in ("auto", *gs.MODELS):
            raise ValueError(f"Dạng hàm không hợp lệ: {dang}")
        self.dang = dang
        self.ten = "chon_dang_ham" if dang == "auto" else ("ham_mu" if dang == "exp" else dang)

    def fit(self, bang_hoc, bang_that_hoc):
        self.khop_ = khop_tat_ca(bang_that_hoc, self.dang)
        self.so_hang_hoc_ = len(bang_that_hoc)
        return self

    def predict(self, X):
        csp, vb = np.full(len(X), np.nan), np.full(len(X), np.nan)
        ratios, j = X["ratio"].to_numpy(float), X["J"].to_numpy(float)
        for r, f in self.khop_.items():
            mask = ratios == r
            csp[mask] = f.csp(j[mask])
            # Không dùng f.vbar ở đây: GCD-Sim chặn miền điện áp khi sinh mẫu,
            # còn đánh giá phải giữ nguyên cả dự đoán tuyến tính không hợp lệ.
            vb[mask] = f.v_params[0] + f.v_params[1] * j[mask]
        return khung_du_doan(X, csp, vb)


class MoHinhTrungBinhHinhHoc(MoHinhCoSo):
    ten = "tb_hinh_hoc"

    def fit(self, bang_hoc, bang_that_hoc):
        self.trung_binh_ = np.exp(np.mean(np.log(np.column_stack(
            [bang_that_hoc["Csp"].to_numpy(float), _vbar(bang_that_hoc)])), axis=0))
        self.so_hang_hoc_ = len(bang_that_hoc)
        return self

    def predict(self, X):
        return khung_du_doan(X, np.full(len(X), self.trung_binh_[0]),
                             np.full(len(X), self.trung_binh_[1]))


class MoHinhTrungBinhTheoMuc(MoHinhCoSo):
    ten = "tb_theo_muc"

    def fit(self, bang_hoc, bang_that_hoc):
        bang = pd.DataFrame({"muc": bang_that_hoc["muc"].to_numpy(),
                             "Csp": np.log(bang_that_hoc["Csp"].to_numpy(float)),
                             "Vbar": np.log(_vbar(bang_that_hoc))})
        self.theo_muc_ = np.exp(bang.groupby("muc", sort=True)[["Csp", "Vbar"]].mean())
        self.so_hang_hoc_ = len(bang_that_hoc)
        return self

    def predict(self, X):
        tb = self.theo_muc_.reindex(X["muc"].to_numpy())
        return khung_du_doan(X, tb["Csp"].to_numpy(), tb["Vbar"].to_numpy())
