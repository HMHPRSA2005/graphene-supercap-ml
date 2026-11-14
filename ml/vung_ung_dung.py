"""Ragone và chọn dòng theo vùng ứng dụng, chỉ trong miền J đã đo."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .du_doan import GHI_CHU, _dien_cuc, du_doan_luoi
from .du_lieu import doc_so_lieu
from .luu_best import tai_best
from .ve_hinh_ch5 import DauRa as OutCh5, TEN_GON

SO_DIEM = 2001
TEN_VUNG = ("công suất", "trung gian", "năng lượng")
COT_BANG = ["ratio", "vung", "J", "dong_mA", "Ed", "Pd"]
MOC_PD_KW = (5, 10, 20, 40, 80)
GIOI_HAN = ("Giá trị vật liệu điện cực trong hệ ba điện cực, chuẩn hóa theo khối lượng "
            "hoạt chất; không phải thiết bị đóng gói, chỉ so sánh và định hướng.")


def du_doan_mien_do(mo_hinh, bang_that, ti_le, so_diem=SO_DIEM):
    """Lưới log-đều kín [Jmin, Jmax]; suy m (g) từ mức đo đầu 10 mA."""
    if so_diem < 200:
        raise ValueError("Lưới phải có ít nhất 200 điểm.")
    g = _dien_cuc(bang_that, ti_le)
    j = np.geomspace(float(g.J.min()), float(g.J.max()), so_diem)
    bang = du_doan_luoi(mo_hinh, float(g.ratio.iloc[0]), j)
    khoi_luong_g = 0.010 / float(g.J.iloc[0])
    bang["dong_mA"] = bang.J * khoi_luong_g * 1000
    cot = ["J", "Csp", "Vbar", "Ed", "Pd", "dt", "dong_mA"]
    gia_tri = bang[cot].to_numpy(float)
    bang["khong_hop_le"] |= ~np.all(np.isfinite(gia_tri) & (gia_tri > 0), axis=1)
    return bang


def cac_khoang(bang, *, dt_min=None, dt_max=None, pd_min=None, ed_min=None,
               dt_max_mo=False):
    """Các đoạn lưới liên tiếp thỏa điều kiện; không nối qua điểm lỗi/không thỏa.

    Truy vấn dùng biên đóng. dt_max_mo dành cho phân vùng nửa kín, để
    Δt đúng mốc chỉ thuộc vùng bên trên. Đầu mút chỉ xấp xỉ theo lưới.
    """
    cot = ["J", "dong_mA", "Csp", "Vbar", "Ed", "Pd", "dt"]
    gia_tri = bang[cot].to_numpy(float)
    giu = np.all(np.isfinite(gia_tri) & (gia_tri > 0), axis=1)
    giu &= ~bang.khong_hop_le.to_numpy(bool)
    for ten, nguong in (("dt", dt_min), ("Pd", pd_min), ("Ed", ed_min)):
        if nguong is not None:
            giu &= bang[ten].to_numpy() >= nguong
    if dt_max is not None:
        dt = bang.dt.to_numpy()
        giu &= dt < dt_max if dt_max_mo else dt <= dt_max
    vi_tri = np.flatnonzero(giu)
    if not len(vi_tri):
        return []
    return [bang.iloc[doan] for doan in np.split(vi_tri, np.flatnonzero(np.diff(vi_tri) != 1) + 1)]


def _so(v):
    """Ba chữ số có nghĩa, không dùng dạng mũ, để khoảng vừa cột bảng."""
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.3g}"


def _khoang(doan, cot, *, trinh_bay=False):
    def so(v):
        if trinh_bay:
            return (f"{v / 1000:.1f}" if cot == "Pd" else _so(v)).replace(".", ",")
        return _so(v)
    return "; ".join(f"{so(g[cot].min())}–{so(g[cot].max())}" for g in doan)


def bang_vung(cac_bang, dt_moc=(10.0, 100.0)):
    """Một hàng/(tỉ lệ, vùng); dấu ; ngăn các khoảng rời nhau, theo thứ tự J."""
    hang = []
    for ratio, bang in cac_bang.items():
        for ten, lo, hi in zip(TEN_VUNG, (None, *dt_moc), (*dt_moc, None)):
            doan = cac_khoang(bang, dt_min=lo, dt_max=hi, dt_max_mo=True)
            if doan:
                hang.append({"ratio": ratio, "vung": ten,
                             **{c: _khoang(doan, c, trinh_bay=True) for c in COT_BANG[2:]}})
    return pd.DataFrame(hang, columns=COT_BANG)


def bang_truy_van(cac_bang, **dieu_kien):
    """Khoảng J/I thỏa truy vấn và cực đại trên toàn bộ điểm lưới hợp lệ."""
    hang = []
    muc_tieu = "Pd" if dieu_kien.get("ed_min") is not None else "Ed"
    for ratio, bang in cac_bang.items():
        doan = cac_khoang(bang, **dieu_kien)
        dong = {"ratio": ratio, "J (A/g)": "không có trong miền đo", "I (mA)": "—",
                **{c: "—" for c in ("J* (A/g)", "I* (mA)", "Ed* (Wh/kg)", "Pd* (W/kg)", "dt* (s)")}}
        if doan:
            hop = pd.concat(doan, ignore_index=True)
            diem = hop.loc[hop[muc_tieu].idxmax()]
            dong.update({"J (A/g)": _khoang(doan, "J"), "I (mA)": _khoang(doan, "dong_mA"),
                         "J* (A/g)": diem.J, "I* (mA)": diem.dong_mA,
                         "Ed* (Wh/kg)": diem.Ed, "Pd* (W/kg)": diem.Pd, "dt* (s)": diem["dt"]})
        hang.append(dong)
    return pd.DataFrame(hang)


def bang_cung_pd(cac_bang, moc_pd_kw=MOC_PD_KW):
    """Nội suy tuyến tính log Ed theo log Pd, không vượt đoạn lưới hợp lệ."""
    hang = []
    for ratio, bang in cac_bang.items():
        r = {"ratio": ratio, **{f"Ed_{p:g}": np.nan for p in moc_pd_kw}}
        for doan in cac_khoang(bang):
            d = doan.sort_values("Pd").drop_duplicates("Pd")
            p, e = d.Pd.to_numpy(float), d.Ed.to_numpy(float)
            for kw in moc_pd_kw:
                muc = kw * 1000
                if p[0] <= muc <= p[-1]:
                    r[f"Ed_{kw:g}"] = float(np.exp(np.interp(np.log(muc), np.log(p), np.log(e))))
        hang.append(r)
    return pd.DataFrame(hang, columns=["ratio", *(f"Ed_{p:g}" for p in moc_pd_kw)])


def ve_ragone(out, cac_bang, bang_that, dt_moc):
    fig, ax = plt.subplots(figsize=(12, 8))
    mau = plt.get_cmap("tab10")
    for i, (ratio, bang) in enumerate(cac_bang.items()):
        that = _dien_cuc(bang_that, ratio)
        hop_le = ~bang.khong_hop_le
        # NaN ngắt đường tại điểm lỗi, không nối tắt qua khoảng không hợp lệ.
        ax.plot(bang.Pd.where(hop_le), bang.Ed.where(hop_le), color=mau(i),
                label=f"r = {ratio:g}".replace(".", ","), linewidth=1.8)
        giu = np.isfinite(that.Pd) & np.isfinite(that.Ed) & (that.Pd > 0) & (that.Ed > 0)
        ax.scatter(that.loc[giu, "Pd"], that.loc[giu, "Ed"], color=mau(i), s=22,
                   edgecolors="white", linewidths=0.4, zorder=4)
        if np.isclose(ratio, 0.1):
            for dieu_kien, nhan, vi_tri in (
                    ({"dt_max": 10, "dt_max_mo": True}, "phanh tái sinh", (-90, -38)),
                    ({"dt_min": 100}, "dự phòng", (30, -35))):
                doan = cac_khoang(bang, **dieu_kien)
                for g in doan:
                    ax.plot(g.Pd, g.Ed, color=mau(i), linewidth=4.5, zorder=5)
                if doan:
                    g = max(doan, key=len)
                    diem = g.iloc[len(g) // 2]
                    ax.annotate(nhan + " (r = 0,1)", (diem.Pd, diem.Ed), xytext=vi_tri,
                                textcoords="offset points", fontsize=12, fontweight="bold",
                                arrowprops={"arrowstyle": "->", "color": mau(i)},
                                bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                                zorder=6)
    ax.set(xscale="log", yscale="log", xlabel=r"$P_d$ (W/kg)", ylabel=r"$E_d$ (Wh/kg)",
           title="Ragone theo vùng ứng dụng — điểm đo và mô hình trong miền đo")
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    p = np.geomspace(xmin, xmax, 400)
    a, b = dt_moc
    for lo, hi, color in ((np.full_like(p, ymin), p*a/3600, "#f4d6b0"),
                          (p*a/3600, p*b/3600, "#dce5ef"),
                          (p*b/3600, np.full_like(p, ymax), "#d8ead8")):
        lo, hi = np.maximum(lo, ymin), np.minimum(hi, ymax)
        ax.fill_between(p, lo, hi, where=hi >= lo, color=color, alpha=0.55, zorder=0)
    for dt in sorted({1.0, 10.0, 100.0, *dt_moc}):
        ax.plot(p, p*dt/3600, "--", color="0.45", linewidth=0.8, zorder=1)
        x = np.clip(np.sqrt(ymin*ymax)*3600/dt, xmin*1.2, xmax/1.8)
        y = x*dt/3600
        if ymin < y < ymax:
            ax.annotate(f"Δt = {dt:g} s", (x, y), xytext=(3, 4),
                        textcoords="offset points", fontsize=12, color="0.2",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    for i, (ten, dieu_kien, color) in enumerate(zip(
            TEN_VUNG, (f"Δt < {a:g} s", f"{a:g} ≤ Δt < {b:g} s", f"Δt ≥ {b:g} s"),
            ("#f4d6b0", "#dce5ef", "#d8ead8"))):
        ax.text(0.02, 0.97-i*0.06, f"vùng {ten}: {dieu_kien}", transform=ax.transAxes,
                va="top", fontsize=12, bbox={"facecolor": color, "edgecolor": "none", "alpha": 0.85})
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(which="both", alpha=0.18)
    ax.tick_params(labelsize=12)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(14)
    ax.legend(title="Điểm: đo; nét: mô hình", fontsize=12, title_fontsize=12,
              loc="lower left", ncol=4)
    out.luu_hinh(fig, "ragone_vung_ung_dung",
                 "Ngưỡng vùng là quy ước minh họa, không phải chuẩn.\n"
                 "Hệ ba điện cực / khối lượng hoạt chất, không phải thiết bị đóng gói; " + GHI_CHU + ".")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best", type=Path, required=True, help="Chỉ nạp best joblib/JSON tin cậy; kiểm phiên bản qua tai_best.")
    parser.add_argument("--thu-muc", type=Path, required=True)
    parser.add_argument("--dt-moc", type=float, nargs=2, default=(10.0, 100.0), metavar=("THAP", "CAO"))
    parser.add_argument("--dt-min", type=float)
    parser.add_argument("--dt-max", type=float)
    parser.add_argument("--pd-min", type=float)
    parser.add_argument("--ed-min", type=float)
    parser.add_argument("--ti-le", type=float, nargs="+")
    args = parser.parse_args(argv)
    dieu_kien = {c: getattr(args, c) for c in ("dt_min", "dt_max", "pd_min", "ed_min")}
    try:
        a, b = args.dt_moc
        if not (np.isfinite(a) and np.isfinite(b) and 0 < a < b):
            raise ValueError("--dt-moc cần hai mốc hữu hạn, 0 < THAP < CAO.")
        for ten, v in dieu_kien.items():
            if v is not None and (not np.isfinite(v) or v <= 0):
                raise ValueError(f"--{ten.replace('_', '-')} phải hữu hạn và dương.")
        if args.dt_min is not None and args.dt_max is not None and args.dt_min > args.dt_max:
            raise ValueError("--dt-min không được lớn hơn --dt-max.")
        so_loai = sum((args.dt_min is not None or args.dt_max is not None,
                       args.pd_min is not None, args.ed_min is not None))
        if so_loai > 1:
            raise ValueError("Chọn khoảng Δt hoặc --pd-min hoặc --ed-min, không kết hợp các loại.")
        that = doc_so_lieu()
        ratios = sorted(that.ratio.unique()) if args.ti_le is None else sorted({
            float(_dien_cuc(that, r).ratio.iloc[0]) for r in args.ti_le})
        mo_hinh, thong_tin = tai_best(args.best)
        print(f"Mô hình best ({thong_tin['muc_tieu']}): {thong_tin['cau_hinh']}")
        print(GHI_CHU)
        print(GIOI_HAN)
        print(f"Lưới {SO_DIEM} điểm J log-đều/điện cực trong [Jmin, Jmax] đã đo; không ngoại suy. "
              "Khoảng/cực đại xấp xỉ theo lưới; dấu ; tách các khoảng rời nhau.")
        cac_bang = {r: du_doan_mien_do(mo_hinh, that, r) for r in ratios}
        for r, bang in cac_bang.items():
            n_loi = int(bang.khong_hop_le.sum())
            if n_loi:
                print(f"Cảnh báo tỉ lệ {r:g}: bỏ {n_loi}/{len(bang)} dự đoán không hữu hạn/≤0 hoặc bị đánh dấu lỗi.")
        if so_loai:
            muc_tieu = "Pd" if args.ed_min is not None else "Ed"
            print("Điều kiện: " + ", ".join(f"{k}={v:g}" for k, v in dieu_kien.items() if v is not None))
            print(f"Dấu *: điểm có {muc_tieu} lớn nhất trong các điểm lưới thỏa điều kiện của điện cực.")
            print(bang_truy_van(cac_bang, **dieu_kien).to_string(index=False, float_format=lambda v: f"{v:.6g}"))
        else:
            out = OutCh5(args.thu_muc)
            bang = bang_vung(cac_bang, args.dt_moc)
            ch = thong_tin["cau_hinh"]
            ten = TEN_GON.get(ch.get("mo_hinh", mo_hinh.ten), mo_hinh.ten)
            ghi_chu = (f"{ten}, cách {ch.get('cach')} ×{float(ch.get('n_ao', 0)):g}, "
                       f"σ = {100 * float(ch.get('sigma_lap', 0)):g}%; khớp lại trên {len(that)} điểm để sử dụng, chỉ trong miền đo. "
                       f"Ngưỡng {a:g}/{b:g} s chỉ minh họa; vật liệu trong hệ ba điện cực, không phải thiết bị đóng gói.")
            if bang[COT_BANG[2:]].apply(lambda c: c.str.contains(";", regex=False)).any().any():
                ghi_chu += " Dấu ; tách các khoảng rời."
            out.luu_tabular(bang, "vung_ung_dung", ghi_chu,
                           ten_cot={"ratio": "Tỉ lệ", "vung": "Vùng", "J": "J (A/g)",
                                    "dong_mA": "I (mA)", "Ed": r"$E_d$ (Wh/kg)", "Pd": r"$P_d$ (kW/kg)"},
                           rong_cot=[0.08, 0.15, 0.19, 0.19, 0.20, 0.19])
            cung_pd = bang_cung_pd(cac_bang)
            out.luu_tabular(cung_pd, "ragone_cung_pd",
                           "Ed (Wh/kg) tại cùng Pd (kW/kg), nội suy log–log trên đường mô hình "
                           "trong miền đo từng điện cực; — là ngoài miền đo. " + ghi_chu,
                           ten_cot={"ratio": "Tỉ lệ", **{f"Ed_{p}": f"{p} kW/kg" for p in MOC_PD_KW}})
            print("Ed (Wh/kg) tại cùng Pd = 5, 10, 20, 40, 80 kW/kg (—: ngoài miền đo):")
            print(cung_pd.to_string(index=False, na_rep="—", float_format=lambda v: f"{v:.12g}"))
            ve_ragone(out, cac_bang, that, args.dt_moc)
            print("Đơn vị khoảng: J A/g; I mA; Ed Wh/kg; Pd kW/kg.")
            print(bang.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
            for p in out.tep:
                print(f"Đã ghi: {p}")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
