"""Thước đo Chương 5; dự đoán không hữu hạn/không dương được đếm, không sửa.

Các thước đo điểm nhận ``tra_so_khong_hop_le=True`` để trả (giá trị, số dự
đoán lỗi). Chúng chỉ tính trên cặp số đo/dự đoán hữu hạn, dương. Tập rỗng
và R² khi số đo không có phương sai trả NaN, không giả thành kết quả tốt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def khong_hop_le(yhat):
    a = np.asarray(yhat, float)
    return ~np.isfinite(a) | (a <= 0)


def sai_so_tuong_doi(y, yhat):
    y, yh = np.broadcast_arrays(np.asarray(y, float), np.asarray(yhat, float))
    hop_le = ~khong_hop_le(y) & ~khong_hop_le(yh)
    kq = np.full(y.shape, np.nan)
    np.divide(np.abs(yh - y) * 100, y, out=kq, where=hop_le)
    return kq


def _cap(y, yhat):
    y, yh = np.broadcast_arrays(np.asarray(y, float), np.asarray(yhat, float))
    loi = khong_hop_le(yh)
    giu = ~loi & ~khong_hop_le(y)
    return y[giu], yh[giu], int(loi.sum())


def _tra(gia_tri, n, tra_so):
    gia_tri = float(gia_tri)
    return (gia_tri, n) if tra_so else gia_tri


def mape(y, yhat, tra_so_khong_hop_le=False):
    y, yh, n = _cap(y, yhat)
    return _tra(np.mean(np.abs(yh - y) / y) * 100 if y.size else np.nan,
                n, tra_so_khong_hop_le)


def mae(y, yhat, tra_so_khong_hop_le=False):
    y, yh, n = _cap(y, yhat)
    return _tra(np.mean(np.abs(yh - y)) if y.size else np.nan, n, tra_so_khong_hop_le)


def rmse(y, yhat, tra_so_khong_hop_le=False):
    y, yh, n = _cap(y, yhat)
    return _tra(np.sqrt(np.mean((yh - y) ** 2)) if y.size else np.nan, n, tra_so_khong_hop_le)


def rmse_log(y, yhat, tra_so_khong_hop_le=False):
    y, yh, n = _cap(y, yhat)
    return _tra(np.sqrt(np.mean((np.log(yh) - np.log(y)) ** 2)) if y.size else np.nan,
                n, tra_so_khong_hop_le)


def r2(y, yhat, tra_so_khong_hop_le=False):
    y, yh, n = _cap(y, yhat)
    sst = float(np.sum((y - np.mean(y)) ** 2)) if y.size >= 2 else 0.0
    return _tra(1 - np.sum((yh - y) ** 2) / sst if sst > 0 else np.nan,
                n, tra_so_khong_hop_le)


def khoang_95(mu_log, sd_log):
    mu, sd = np.broadcast_arrays(np.asarray(mu_log, float), np.asarray(sd_log, float))
    dung = np.isfinite(mu) & np.isfinite(sd) & (sd >= 0)
    l = np.full(mu.shape, np.nan)
    u = np.full(mu.shape, np.nan)
    l[dung] = np.exp(mu[dung] - 1.96 * sd[dung])
    u[dung] = np.exp(mu[dung] + 1.96 * sd[dung])
    return l, u


def _khoang(y, L, U):
    y, l, u = np.broadcast_arrays(np.asarray(y, float), np.asarray(L, float), np.asarray(U, float))
    giu = ~khong_hop_le(y) & ~khong_hop_le(l) & ~khong_hop_le(u) & (l <= u)
    return y[giu], l[giu], u[giu]


def picp(y, L, U):
    """Tỉ lệ phủ trên mọi điểm; khoảng/số đo không hợp lệ tính là không phủ."""
    y, l, u = np.broadcast_arrays(np.asarray(y, float), np.asarray(L, float), np.asarray(U, float))
    hop_le = ~khong_hop_le(y) & ~khong_hop_le(l) & ~khong_hop_le(u) & (l <= u)
    return float(np.mean(hop_le & (l <= y) & (y <= u))) if y.size else float("nan")


def w_rel(y, L, U):
    """Độ rộng tương đối trung bình (U−L)/y, không nhân 100."""
    y, l, u = _khoang(y, L, U)
    return float(np.mean((u - l) / y)) if y.size else float("nan")


def tong_hop_theo_dien_cuc(df, cot, nhom="ratio"):
    """Trọng số bằng nhau giữa điện cực; một điện cực toàn NaN trả NaN.

    NaN lẻ được bỏ khi lấy trung bình trong điện cực. Cần đọc kết quả kèm
    số dự đoán không hợp lệ để tránh đánh giá quá lạc quan.
    """
    theo_dien_cuc = df.groupby(nhom, sort=True)[cot].mean()
    return float(theo_dien_cuc.mean(skipna=False))


def e_bar_f3(E: dict[tuple[int, int], float]):
    return (float(np.mean([E[3, i] for i in (1, 2, 3)])),
            float(np.mean([E[3, i] - E[i, i] for i in (1, 2)])))


def wilcoxon_holm(cap: list[tuple[str, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """Wilcoxon hai phía trên cặp hữu hạn, chênh lệch = mảng thứ nhất − thứ hai.

    Cặp hoàn toàn bằng nhau có p=1. Hiệu chỉnh Holm theo toàn bộ so sánh có
    p hữu hạn; so sánh không có cặp dữ liệu trả NaN và n=0.
    """
    rows = []
    for ten, a, b in cap:
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        giu = np.isfinite(a) & np.isfinite(b)
        d = (a - b)[giu].ravel()
        p = (1.0 if np.all(d == 0) else float(wilcoxon(d, alternative="two-sided").pvalue)) if d.size else np.nan
        rows.append(dict(so_sanh=ten, trung_vi_chenh=float(np.median(d)) if d.size else np.nan,
                         p=p, p_holm=np.nan, n=int(d.size)))
    bang = pd.DataFrame(rows, columns=["so_sanh", "trung_vi_chenh", "p", "p_holm", "n"])
    thu_tu = bang.index[bang["p"].notna()].to_numpy()
    thu_tu = thu_tu[np.argsort(bang.loc[thu_tu, "p"].to_numpy(), kind="stable")]
    if len(thu_tu):
        p = bang.loc[thu_tu, "p"].to_numpy() * np.arange(len(thu_tu), 0, -1)
        bang.loc[thu_tu, "p_holm"] = np.minimum(1.0, np.maximum.accumulate(p))
    return bang
