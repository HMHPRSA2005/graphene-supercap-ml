"""Hình KB5 cho báo cáo: MAPE Csp (trục log, trung vị qua seed) và số nút đồ thị.

Đọc bảng theo bước do ``ml.ve_hinh_ch5`` sinh từ ket_qua_kb5.csv, không tính lại sai số.
Cách chạy (từ Code/):
    python -X utf8 -m ml.ve_hinh_kb5_bao_cao --thu-muc ket_qua/bo_sung --out ../Report/Report/ket_qua_ch5/hinh_kb5.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

TEN = {"mlp_hoc_lai": "MLP học lại", "mlp_tinh_chinh": "MLP tinh chỉnh",
       "gng_r": "GNG-R", "pgcn_r": "PGCN-R"}
MAU = {"mlp_hoc_lai": "tab:orange", "mlp_tinh_chinh": "tab:green",
       "gng_r": "tab:blue", "pgcn_r": "tab:red"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thu-muc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    b = pd.read_csv(a.thu_muc / "bang_ch5" / "phu_luc_kb5_theo_buoc.csv")
    b["cuoi"] = b.cuoi_moi_truong.astype(str).str.lower().isin(["true", "1"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    for pp in TEN:
        g = b.loc[b.phuong_phap == pp].groupby("buoc").MAPE_Csp
        q = g.quantile([0.25, 0.5, 0.75]).unstack()
        ax1.plot(q.index, q[0.5], color=MAU[pp], label=TEN[pp], lw=1.6)
        ax1.fill_between(q.index, q[0.25], q[0.75], color=MAU[pp], alpha=0.15)
    # Vạch chuyển đợt: trung vị vị trí cuối đợt 1-3 của MLP học lại (cùng dữ liệu cho mọi phương pháp).
    moc = b.loc[b.cuoi & (b.moi_truong < 4) & (b.phuong_phap == "mlp_hoc_lai")]
    for mt, d in moc.groupby("moi_truong"):
        x = d.buoc.median()
        ax1.axvline(x, color="gray", ls=":", lw=1)
        ax1.text(x, 0.03, f" đợt {mt + 1}", transform=ax1.get_xaxis_transform(),
                 ha="left", fontsize=8, color="gray")
    ax1.set_yscale("log")
    ax1.set(xlabel="Số mẫu đã học cộng dồn", ylabel="MAPE $C_{\\mathrm{sp}}$ (%), điện cực đã thấy")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_title("(a) Sai số điện dung (trung vị, dải 25–75%)", fontsize=10)

    for pp in ("gng_r", "pgcn_r"):
        g = b.loc[b.phuong_phap == pp].groupby("buoc").so_nut.median()
        ax2.plot(g.index, g.values, color=MAU[pp], label=TEN[pp], lw=1.6)
    ax2.set(xlabel="Số mẫu đã học cộng dồn", ylabel="Số nút")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    ax2.set_title("(b) Số nút của đồ thị (trung vị)", fontsize=10)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".png"), dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
