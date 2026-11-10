"""Vẽ các hình kiểm chứng dữ liệu ảo cho báo cáo (Chương 3).
Chạy:  python ve_hinh.py data/so_lieu_GCD.xlsx out/hinh
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gcdsim as gs

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})
REAL, SYN = "#a0782b", "#1f8a5b"


def main(inp, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = gs.read_lab_excel(inp)
    fits = gs.attach_measured(gs.fit_electrodes(df), df)
    cfg = gs.SimConfig()
    syn = gs.generate_table(fits, cfg, df)
    ratios = sorted(df.ratio.unique())

    # Hình 3.x: Csp theo J – số liệu gốc và đám mây dữ liệu ảo, 8 điện cực
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=False)
    for ax, r in zip(axes.ravel(), ratios):
        s = syn[syn.ratio == r]
        ax.scatter(s.J, s.Csp, s=2, alpha=0.08, color=SYN, rasterized=True)
        g = df[df.ratio == r]
        ax.plot(g.J, g.Csp, "o-", color=REAL, ms=5, lw=1.5)
        ax.set_title(f"Tỉ lệ {r:g}")
        ax.set_xlabel("J (A/g)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Csp (F/g)")
    fig.legend(handles=[plt.Line2D([], [], marker="o", color=REAL, label="Số liệu gốc (6 điểm)"),
                        plt.Line2D([], [], marker="o", ls="", color=SYN, alpha=.5,
                                   label="Dữ liệu ảo (1000 lần đo ảo)")],
               loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(f"{outdir}/hinh_csp_theo_J.png")
    plt.close(fig)

    # Ragone gốc vs ảo
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(ratios):
        s = syn[syn.ratio == r].sample(600, random_state=0)
        ax.scatter(s.Pd, s.Ed, s=3, alpha=0.15, color=cmap(i), rasterized=True)
        g = df[df.ratio == r]
        ax.plot(g.Pd, g.Ed, "o-", color=cmap(i), ms=4, lw=1.2, label=f"{r:g}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Mật độ công suất Pd (W/kg)"); ax.set_ylabel("Mật độ năng lượng Ed (Wh/kg)")
    ax.set_title("Đường Ragone: đường nối = số liệu gốc, chấm mờ = dữ liệu ảo")
    ax.legend(title="Tỉ lệ", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(f"{outdir}/hinh_ragone_goc_va_ao.png"); plt.close(fig)

    # Phân bố tại một điểm đo
    r0 = 0.1
    jj = sorted(df[df.ratio == r0].J)[0]
    s = syn[(syn.ratio == r0) & np.isclose(syn.J_nominal, jj)]
    real = df[(df.ratio == r0) & np.isclose(df.J, jj)].Csp.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(s.Csp, bins=40, color=SYN, alpha=.7)
    ax.axvline(real, color=REAL, lw=2.5, label=f"Số gốc = {real:.1f} F/g")
    lo, hi = np.percentile(s.Csp, [5, 95])
    ax.axvspan(lo, hi, color=SYN, alpha=.12, label=f"Dải 5–95%: {lo:.0f}–{hi:.0f}")
    ax.set_xlabel("Csp (F/g)"); ax.set_ylabel("Số mẫu ảo")
    ax.set_title(f"1000 giá trị ảo tại tỉ lệ {r0:g}, J = {jj:.2f} A/g")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/hinh_phan_bo_mot_diem.png"); plt.close(fig)

    # Đường cong GCD ảo
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(7, 4))
    g = syn[(syn.ratio == r0)]
    for k, jn in enumerate(sorted(g.J_nominal.unique())[:4]):
        rows = g[np.isclose(g.J_nominal, jn)].head(5)
        for _, row in rows.iterrows():
            t, v, tc = gs.gcd_curve(row.Csp, row.J, row.Vbar, rng=rng)
            ax.plot(t, v, color=plt.get_cmap("viridis")(k / 4), lw=0.8, alpha=0.6)
        ax.plot([], [], color=plt.get_cmap("viridis")(k / 4), label=f"J = {jn:.1f} A/g")
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("Điện thế (V)")
    ax.set_title(f"Đường sạc–xả ảo của điện cực tỉ lệ {r0:g} (5 lần đo ảo mỗi mức dòng)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{outdir}/hinh_duong_gcd_ao.png"); plt.close(fig)

    point, summ = gs.validate(df, syn, fits, cfg)
    pd.Series(summ).to_csv(f"{outdir}/kiem_chung_tong_hop.csv", header=["gia_tri"])
    print(summ)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "out/hinh")
