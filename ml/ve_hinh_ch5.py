"""Sinh lại bảng/hình Chương 5 từ kết quả điểm kiểm tra.

Chạy: python -X utf8 -m ml.ve_hinh_ch5 --thu-muc ket_qua/thu
Thêm --ung-dung để khớp lại 48 số đo và tạo minh họa 5.9–5.10 (không
phải kết quả kiểm tra). MAPE/RMSE-log tính từng điện cực, trung bình điện
cực trong mỗi seed, sau đó báo trung bình ± SD mẫu qua seed (một seed:
SD chưa xác định). CSV giữ cả số liệu số và ô trình bày; .tex dùng booktabs
và longtable (bảng rộng chuyển thành chỉ tiêu/giá trị để vừa trang A4),
không cần Jinja2. Dữ liệu thiếu được thông báo, không thay bằng số giả.
Ứng dụng mặc định cách A; đổi bằng --cach-ung-dung A|B|C|D.
KB5 và mô hình đồ thị được xuất riêng với tiền tố phu_luc_ (tham khảo);
nhóm chính khảo sát dữ liệu ảo chỉ gồm MLP, SVR, GPR.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
import numpy as np
import pandas as pd

from . import danh_gia as dg
from .ma_hoa import DANH_MUC_TI_LE

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.unicode_minus": False, "savefig.dpi": 200})
TEN = {"ham_mu": "Hàm mũ", "chon_dang_ham": "Chọn dạng hàm", "ridge": "Ridge",
       "svr": "SVR", "rf": "Random Forest", "gb": "Gradient Boosting",
       "mlp": "MLP", "gpr": "GPR", "gng_r": "GNG-R", "pgcn_r": "PGCN-R",
       "tb_hinh_hoc": "TB hình học", "tb_theo_muc": "TB theo mức",
       "mlp_hoc_lai": "MLP học lại", "mlp_tinh_chinh": "MLP tinh chỉnh"}
CAU_HINH = ["kb", "mo_hinh", "cach", "n_ao", "sigma_lap"]
NHOM_GOC = ((0.025, 0.05, 0.075), (0.1, 0.25), (0.5, 1.0, 1.75))
MO_HINH_DU_LIEU_AO = ("mlp", "svr", "gpr")
GHI_CHU_KB5 = "Phụ lục: học liên tục (tham khảo)"
GIOI_HAN_WILCOXON = ("Chỉ 8 điện cực: công suất kiểm định thấp; p hai phía nhỏ nhất "
                     "0,0078125 khi 8 hiệu khác 0 đồng dấu, trước hiệu chỉnh Holm.")


def thong_bao(ten, ly_do):
    print(f"Bỏ qua {ten}: {ly_do}.")


def du_cot(bang, cot, ten):
    thieu = sorted(set(cot) - set(bang.columns))
    if thieu:
        thong_bao(ten, "thiếu cột " + ", ".join(thieu))
        return False
    if bang.empty:
        thong_bao(ten, "không có nhóm dữ liệu phù hợp")
        return False
    return True


def _tex(s):
    thay = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "±": r"$\pm$"}
    return "".join(thay.get(c, c) for c in str(s))


class DauRa:
    def __init__(self, thu_muc):
        self.bang = Path(thu_muc) / "bang_ch5"
        self.hinh = Path(thu_muc) / "hinh_ch5"
        self.tep = []

    def luu_bang(self, bang, ten, ghi_chu="", cot_tex=None):
        if bang.empty:
            thong_bao(ten, "bảng rỗng")
            return
        self.bang.mkdir(parents=True, exist_ok=True)
        csv = self.bang / f"{ten}.csv"
        bang.to_csv(csv, index=False, encoding="utf-8-sig")
        tex = csv.with_suffix(".tex")
        # CSV giữ mọi cột; TeX bỏ cột số lặp lại ô trình bày và xoay bảng rộng.
        cot = [c for c in bang if not (c.endswith(("_tb", "_sd")) and c[:-3] in bang)]
        if cot_tex is not None:
            cot = [c for c in cot_tex if c in bang]
        dong = ["% Cần \\usepackage{booktabs,longtable}", "% " + ghi_chu]
        if ghi_chu:
            dong.append(r"\par\smallskip\noindent " + _tex(ghi_chu) + r"\par")
        if len(cot) > 6:
            dong.extend([r"\begin{longtable}{p{0.43\linewidth}p{0.50\linewidth}}",
                         r"\toprule Chỉ tiêu & Giá trị \\ \midrule \endhead"])
            dinh_danh = [c for c in ("kb", "mo_hinh", "phuong_phap", "thu_tu", "cach",
                                    "n_ao", "sigma_lap", "seed", "ratio", "k", "chinh_sach") if c in cot]
            for i, (_, hang) in enumerate(bang.iterrows(), 1):
                nhan = f"Bản ghi {i}: " + "; ".join(f"{c}={hang[c]}" for c in dinh_danh)
                dong.append(r"\multicolumn{2}{p{0.95\linewidth}}{\textbf{" + _tex(nhan) + r"}} \\")
                for c in cot:
                    if c in dinh_danh:
                        continue
                    v = hang[c]
                    gia_tri = "—" if pd.isna(v) else f"{v:.6g}" if isinstance(v, float) else str(v)
                    dong.append(_tex(c).replace(r"\_", r"\_\allowbreak ") + " & " + _tex(gia_tri) + r" \\")
                dong.append(r"\midrule")
            dong.extend([r"\bottomrule", r"\end{longtable}", ""])
        else:
            rong = 0.90 / len(cot)
            dong.extend([r"\begin{longtable}{" + (f"p{{{rong:.3f}\\linewidth}}" * len(cot)) + "}",
                         r"\toprule", " & ".join(_tex(c) for c in cot) + r" \\ \midrule \endhead"])
            for hang in bang[cot].itertuples(index=False, name=None):
                dong.append(" & ".join(_tex("—" if pd.isna(v) else f"{v:.6g}" if isinstance(v, float) else v)
                                       for v in hang) + r" \\")
            dong.extend([r"\bottomrule", r"\end{longtable}", ""])
        tex.write_text("\n".join(dong), encoding="utf-8")
        self.tep.extend([csv, tex])

    def luu_hinh(self, fig, ten, ghi_chu=""):
        self.hinh.mkdir(parents=True, exist_ok=True)
        if ghi_chu:
            fig.text(0.5, 0.008, ghi_chu, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035 if ghi_chu else 0, 1, 1))
        for duoi in ("png", "pdf"):
            p = self.hinh / f"{ten}.{duoi}"
            fig.savefig(p, dpi=200)
            self.tep.append(p)
        plt.close(fig)


def doc_ket_qua(p):
    if not p.exists():
        thong_bao(p.name, "chưa có tệp")
        return pd.DataFrame()
    try:
        bang = pd.read_csv(p)
    except pd.errors.EmptyDataError:
        thong_bao(p.name, "tệp rỗng")
        return pd.DataFrame()
    for c in ("cach", "kb"):
        if c in bang:
            bang[c] = bang[c].astype(str).str.upper()
    return bang


def chon(bang, **dieu_kien):
    for c, v in dieu_kien.items():
        if c not in bang:
            return bang.iloc[:0]
        if isinstance(v, (tuple, list)):
            bang = bang.loc[bang[c].isin(v)]
        elif isinstance(v, float):
            bang = bang.loc[np.isclose(bang[c], v)]
        else:
            bang = bang.loc[bang[c] == v]
    return bang


def theo_dien_cuc(bang, nhom, bo_sung=False):
    """Một hàng/điện cực/seed; số lỗi không bị mất khi lấy trung bình."""
    hang = []
    for khoa, g in bang.groupby(nhom + ["seed", "ratio"], dropna=False, sort=True):
        r = dict(zip(nhom + ["seed", "ratio"], khoa))
        for c in ("Csp", "Pd"):
            if {c, c + "_hat"}.issubset(g.columns):
                r["MAPE_" + c] = dg.mape(g[c], g[c + "_hat"])
                r["loi_" + c] = int(dg.khong_hop_le(g[c + "_hat"]).sum())
        if {"Csp", "Csp_hat"}.issubset(g.columns):
            r["RMSE_log_Csp"] = dg.rmse_log(g.Csp, g.Csp_hat)
            if bo_sung:
                r["MAE_Csp"] = dg.mae(g.Csp, g.Csp_hat)
                r["RMSE_Csp"] = dg.rmse(g.Csp, g.Csp_hat)
                r["R2_Csp"] = dg.r2(g.Csp, g.Csp_hat)
        loi = np.zeros(len(g), dtype=bool)
        for c in ("Csp_hat", "Pd_hat", "Vbar_hat"):
            if c in g:
                loi |= dg.khong_hop_le(g[c])
        if "khong_hop_le" in g:
            loi |= g.khong_hop_le.astype(str).str.lower().isin(["true", "1"]).to_numpy()
        r["so_khong_hop_le"], r["so_diem"] = int(loi.sum()), len(g)
        hang.append(r)
    return pd.DataFrame(hang)


def _o(tb, sd):
    return "—" if not np.isfinite(tb) else f"{tb:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{tb:.3f} ± chưa xác định"


def tong_hop(bang_dc, nhom):
    cot = [c for c in ("MAPE_Csp", "MAPE_Pd", "RMSE_log_Csp", "MAE_Csp", "RMSE_Csp", "R2_Csp") if c in bang_dc]
    if bang_dc.empty:
        return pd.DataFrame()
    seed = bang_dc.groupby(nhom + ["seed"], dropna=False)[cot].mean().reset_index()
    dem = bang_dc.groupby(nhom + ["seed"], dropna=False)[["so_khong_hop_le", "so_diem"]].sum().reset_index()
    seed = seed.merge(dem, on=nhom + ["seed"])
    hang = []
    for khoa, g in seed.groupby(nhom, dropna=False):
        if len(nhom) == 1:
            khoa = (khoa,) if not isinstance(khoa, tuple) else khoa
        r = dict(zip(nhom, khoa))
        r["so_seed"] = len(g)
        for c in cot:
            tb, sd = g[c].mean(), g[c].std(ddof=1)
            r[c], r[c + "_tb"], r[c + "_sd"] = _o(tb, sd), tb, sd
        r["so_khong_hop_le"] = int(g.so_khong_hop_le.sum())
        r["tong_du_doan"] = int(g.so_diem.sum())
        hang.append(r)
    return pd.DataFrame(hang)

def _ghep_thoi_gian(ket_qua, bang, nhom):
    """Mỗi lần học chỉ đếm một lần, không nhân với số điểm kiểm tra."""
    cot = [c for c in ("t_sinh_s", "t_hoc_s", "t_du_doan_s") if c in bang]
    if not cot or "ma_lan_chia" not in bang:
        thong_bao("Thời gian Bảng 5.2", "thiếu thời gian hoặc ma_lan_chia")
        return ket_qua
    for c in set(("t_sinh_s", "t_hoc_s", "t_du_doan_s")) - set(cot):
        thong_bao("Thời gian Bảng 5.2 " + c, "thiếu cột")
    lan = bang.drop_duplicates(nhom + ["seed", "ma_lan_chia"])
    seed = lan.groupby(nhom + ["seed"], dropna=False)[cot].mean().reset_index()
    hang = []
    for khoa, g in seed.groupby(nhom, dropna=False):
        r = dict(zip(nhom, khoa))
        for c in cot:
            tb, sd = g[c].mean(), g[c].std(ddof=1)
            r[c], r[c + "_tb"], r[c + "_sd"] = _o(tb, sd), tb, sd
        hang.append(r)
    return ket_qua.merge(pd.DataFrame(hang), on=nhom, how="left")

def _chon_c_gan_nhat(bang, nhom):
    """Chọn C gần (100, 0.03) theo khoảng cách log n và sigma tương đối."""
    g = chon(bang, cach="C")
    if not {"n_ao", "sigma_lap", *nhom}.issubset(g.columns) or g.empty:
        return g.iloc[:0]
    cac_nhom = []
    for _, d in g.groupby(nhom, dropna=False):
        muc = d[["n_ao", "sigma_lap"]].dropna().drop_duplicates()
        muc = muc.loc[(muc.n_ao > 0) & (muc.sigma_lap >= 0)].copy()
        if muc.empty:
            continue
        muc["khoang_cach"] = np.abs(np.log(muc.n_ao / 100)) + np.abs(muc.sigma_lap - 0.03) / 0.03
        n, sigma = muc.sort_values(["khoang_cach", "n_ao", "sigma_lap"]).iloc[0][["n_ao", "sigma_lap"]]
        cac_nhom.append(chon(d, n_ao=float(n), sigma_lap=float(sigma)))
    return pd.concat(cac_nhom, ignore_index=True) if cac_nhom else g.iloc[:0]


def _nhan_cau_hinh(g):
    return "; ".join(f"C ×{r.n_ao:g}, σ={100*r.sigma_lap:g}%"
                     for r in g[["n_ao", "sigma_lap"]].drop_duplicates().itertuples(index=False))


def _nhan_seed(g, ve_sd=True):
    if "so_seed" in g:
        dem = g.so_seed
    else:
        nhom = [c for c in ("mo_hinh", "phuong_phap") if c in g]
        dem = g.groupby(nhom).seed.nunique() if nhom else pd.Series([g.seed.nunique()])
    lo, hi = int(dem.min()), int(dem.max())
    nhan = f"n={lo} seed" if lo == hi else f"n={lo}–{hi} seed/cấu hình"
    if not ve_sd:
        return nhan
    return nhan + ("; một seed: SD chưa xác định, không vẽ thanh/dải SD" if lo == 1 else "; thanh/dải: ± SD qua seed")


def _ve_sai_so(ax, x, d, ten):
    line, = ax.plot(x, d.MAPE_Csp_tb, marker="o", label=ten)
    hop_le = d.MAPE_Csp_sd.notna() & (d.so_seed >= 2)
    if hop_le.any():
        ax.errorbar(np.asarray(x)[hop_le], d.loc[hop_le, "MAPE_Csp_tb"],
                    yerr=d.loc[hop_le, "MAPE_Csp_sd"], fmt="none",
                    color=line.get_color(), capsize=3)


def bang_chinh(bang, out):
    g = _chon_c_gan_nhat(chon(bang, kb=["KB1", "KB2", "KB2_BO_0.075"]), ["kb", "mo_hinh"])
    if not du_cot(g, CAU_HINH + ["seed", "ratio", "Csp", "Csp_hat", "Pd", "Pd_hat"], "Bảng 5.2"):
        return
    nhom = CAU_HINH
    dc = theo_dien_cuc(g, nhom, bo_sung=True)
    ket_qua = _ghep_thoi_gian(tong_hop(dc, nhom), g, nhom)
    bo_0075 = dc.loc[(dc.kb == "KB1") & ~np.isclose(dc.ratio, 0.075)]
    if not bo_0075.empty:
        nhay = tong_hop(bo_0075, nhom)
        cot_nhay = [c + duoi for c in ("MAPE_Csp", "MAPE_Pd") for duoi in ("", "_tb", "_sd")]
        doi_ten = {c + duoi: c + "_bo_0075" + duoi
                   for c in ("MAPE_Csp", "MAPE_Pd") for duoi in ("", "_tb", "_sd")}
        ket_qua = ket_qua.merge(nhay[nhom + cot_nhay].rename(columns=doi_ten), on=nhom, how="left")
    else:
        for c in ("MAPE_Csp_bo_0075", "MAPE_Pd_bo_0075"):
            ket_qua[c] = "—"
            ket_qua[c + "_tb"], ket_qua[c + "_sd"] = np.nan, np.nan
    la_do_thi = ket_qua.mo_hinh.isin(["gng_r", "pgcn_r"])
    ket_qua["ten_bang"] = [f"Bảng 5.2 — C ×{r.n_ao:g}, σ={100*r.sigma_lap:g}%"
                          for r in ket_qua.itertuples(index=False)]
    ghi_chu = _nhan_cau_hinh(g) + "; " + _nhan_seed(g, ve_sd=False)
    ghi_chu += ("; sai số trung bình điện cực rồi seed; số lỗi cộng qua seed. "
                "MAE/RMSE Csp: F/g; R² không xác định nếu ít hơn 2 điểm hợp lệ hoặc số đo không biến thiên. "
                "Thời gian (s): trung bình mỗi lần học trong seed rồi trung bình ± SD qua seed, không đếm lặp điểm kiểm tra. "
                "KB1 bỏ 0,075: chỉ loại điện cực này khi tổng hợp, không khớp lại; KB2 là thí nghiệm riêng.")
    cot_chinh_tex = nhom + ["so_seed", "MAPE_Csp", "MAPE_Pd", "RMSE_log_Csp", "so_khong_hop_le",
                            "t_sinh_s", "t_hoc_s", "t_du_doan_s"]
    if (~la_do_thi).any():
        out.luu_bang(ket_qua.loc[~la_do_thi], "bang_5_2", ghi_chu, cot_tex=cot_chinh_tex)
    if la_do_thi.any():
        phu_luc = ket_qua.loc[la_do_thi].copy()
        phu_luc["ten_bang"] = "Phụ lục: mô hình đồ thị (tham khảo)"
        out.luu_bang(phu_luc, "phu_luc_bang_do_thi", "Phụ lục: mô hình đồ thị (tham khảo). " + ghi_chu)
    phu_luc_dc = _ghep_thoi_gian(tong_hop(dc, nhom + ["ratio"]), g, nhom + ["ratio"])
    out.luu_bang(phu_luc_dc, "phu_luc_sai_so_dien_cuc", ghi_chu)
    kb1 = g.loc[(g.kb == "KB1") & ~g.mo_hinh.isin(["gng_r", "pgcn_r"])]
    if du_cot(kb1, ["muc"], "Bảng KB1 theo mức dòng"):
        kb1 = kb1.loc[kb1.muc.isin([2, 3, 4, 5])]
        if not kb1.empty:
            theo_muc = tong_hop(theo_dien_cuc(kb1, nhom + ["muc"]), nhom + ["muc"])
            out.luu_bang(theo_muc, "bang_kb1_theo_muc",
                         _nhan_cau_hinh(kb1) + "; mức 2–5; MAPE trung bình điện cực rồi seed; cùng cấu hình Bảng 5.2.")


def hinh_du_doan(bang, out):
    g = _chon_c_gan_nhat(chon(bang, kb="KB1"), ["mo_hinh"])
    if not du_cot(g, ["mo_hinh", "ratio", "seed", "J", "Csp", "Csp_hat"], "Hình 5.2–5.3"):
        return
    dc = theo_dien_cuc(g, ["mo_hinh"])
    tot = tong_hop(dc, ["mo_hinh"]).sort_values("MAPE_Csp_tb").dropna(subset=["MAPE_Csp_tb"]).mo_hinh.head(3).tolist()
    if not tot:
        thong_bao("Hình 5.2–5.3", "không có dự đoán hợp lệ để xếp hạng")
        return
    g = g.loc[g.mo_hinh.isin(tot)].copy()
    so_bo = int(dg.khong_hop_le(g.Csp_hat).sum())
    ghi_chu = _nhan_cau_hinh(g) + "; " + _nhan_seed(g, ve_sd=False)
    ghi_chu += f". Bỏ {so_bo}/{len(g)} dự đoán Csp không hợp lệ (gộp seed)."
    g.loc[dg.khong_hop_le(g.Csp_hat), "Csp_hat"] = np.nan
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for ax, ratio in zip(axs.flat, DANH_MUC_TI_LE):
        r = g.loc[np.isclose(g.ratio, ratio)]
        if r.empty:
            ax.set(title=f"Tỉ lệ {ratio:g}", xticks=[], yticks=[])
            ax.text(0.5, 0.5, "Chưa có điểm kiểm tra", ha="center", va="center", transform=ax.transAxes)
            continue
        that = r.groupby("J").Csp.first().sort_index()
        ax.plot(that.index, that, "ko-", label="Số đo kiểm tra")
        for ten in tot:
            dd = r.loc[r.mo_hinh == ten].groupby("J").Csp_hat.mean().sort_index()
            ax.plot(dd.index, dd, "o--", label=TEN.get(ten, ten))
        ax.set(title=f"Tỉ lệ {ratio:g}", xlabel="J (A/g)", ylabel="Csp (F/g)")
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=7)
    out.luu_hinh(fig, "hinh_5_2", ghi_chu + "\nDự đoán trung bình qua seed; chỉ vẽ điểm kiểm tra.")
    fig, ax = plt.subplots(figsize=(7, 6))
    gia_tri = []
    for ten in tot:
        d = g.loc[g.mo_hinh == ten].groupby(["ratio", "J"]).agg(Csp=("Csp", "first"), Csp_hat=("Csp_hat", "mean")).dropna()
        d = d.loc[(d.Csp > 0) & (d.Csp_hat > 0)]
        ax.scatter(d.Csp, d.Csp_hat, label=TEN.get(ten, ten), alpha=0.7)
        gia_tri.extend(d[["Csp", "Csp_hat"]].to_numpy().ravel())
    if gia_tri:
        lo, hi = min(gia_tri) * 0.85, max(gia_tri) * 1.15
        ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
        ax.set(xlim=(lo, hi), ylim=(lo, hi))
    ax.set(xscale="log", yscale="log", xlabel="Csp thật (F/g)", ylabel="Csp dự đoán (F/g)", title="KB1 — dự đoán và số đo")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    out.luu_hinh(fig, "hinh_5_3", ghi_chu + "\nTrung bình dự đoán qua seed; điểm không còn dự đoán hợp lệ không được vẽ.")


def anh_huong_ao(bang, out):
    can = CAU_HINH + ["ratio", "seed", "Csp", "Csp_hat"]
    if not du_cot(bang, can, "Ảnh hưởng dữ liệu ảo"):
        return
    g = chon(bang, kb=["KB1", "KB2"], sigma_lap=0.03, mo_hinh=MO_HINH_DU_LIEU_AO)
    cac_nhom = []
    for (kb, mh), d in g.groupby(["kb", "mo_hinh"]):
        cac_muc = d.loc[d.cach.isin(["B", "C", "D"]) & (d.n_ao != 5), "n_ao"].dropna()
        if cac_muc.empty:
            cac_muc = d.n_ao.dropna()
        if cac_muc.empty:
            continue
        n = 100 if (cac_muc == 100).any() else cac_muc.min()
        if n != 100:
            print(f"Bảng 5.3 {kb}/{mh}: dùng n_ao={n:g}; ghi rõ cấu hình trong bảng.")
        a = d.loc[d.cach == "A"]
        na = n if (a.n_ao == n).any() else (a.n_ao.min() if not a.empty else n)
        cac_nhom.append(d.loc[((d.cach == "A") & (d.n_ao == na)) |
                              (d.cach.isin(["B", "C", "D"]) & (d.n_ao == n))])
    chinh = pd.concat(cac_nhom, ignore_index=True) if cac_nhom else g.iloc[:0]
    # A không sinh mẫu; chỉ giữ một bản của mỗi điểm kiểm tra.
    if not chinh.empty:
        a = chinh.loc[chinh.cach == "A"].sort_values("n_ao", ascending=False)
        khoa = [c for c in ["kb", "mo_hinh", "seed", "ratio", "muc", "J", "ma_lan_chia"] if c in a]
        a = a.drop_duplicates(khoa)
        chinh = pd.concat([a, chinh.loc[chinh.cach != "A"]], ignore_index=True)
        dc = theo_dien_cuc(chinh, CAU_HINH)
        bang_53 = tong_hop(dc, CAU_HINH)
        bang_53["vai_tro"] = "Mô hình học máy"
        moc = chon(bang, kb=["KB1", "KB2"], mo_hinh="chon_dang_ham").copy()
        if not moc.empty:
            # Mô hình nền chỉ học số đo thật: dùng một bản/điểm/seed, ưu tiên A.
            moc["uu_tien"] = moc.cach.map({"A": 0, "C": 1, "B": 2, "D": 3}).fillna(4)
            khoa_moc = [c for c in ("kb", "seed", "ma_lan_chia", "ratio", "muc", "J") if c in moc]
            moc = moc.sort_values(["uu_tien", "n_ao", "sigma_lap"]).drop_duplicates(khoa_moc)
            moc["cach"], moc["n_ao"], moc["sigma_lap"] = "moc_dang_ham", 0, 0.0
            bang_moc = tong_hop(theo_dien_cuc(moc, CAU_HINH), CAU_HINH)
            bang_moc["vai_tro"] = "Đường khớp tạo dữ liệu cách D (mốc, chỉ học số đo thật)"
            bang_53 = pd.concat([bang_53, bang_moc], ignore_index=True)
        else:
            thong_bao("Mốc đường khớp trong Bảng 5.3", "thiếu chon_dang_ham")
        out.luu_bang(bang_53, "bang_5_3",
                     "MAPE theo điện cực rồi seed; A chỉ dùng số đo thật. Mốc chon_dang_ham là đường khớp sinh dữ liệu D, "
                     "không phụ thuộc tăng cường: một bản/điểm/seed, ưu tiên kết quả A; n_ao=0, sigma=0 nghĩa là không áp dụng.")
        cac_kiem_dinh = []
        for (kb, ten), d in dc.groupby(["kb", "mo_hinh"]):
            # Họ Holm riêng cho (KB, mô hình), không gộp các mô hình độc lập.
            cap, thong_tin = [], []
            p = d.groupby(["ratio", "cach"]).MAPE_Csp.mean().unstack("cach")
            for x, y in (("C", "A"), ("C", "B"), ("D", "A")):
                if x not in p or y not in p:
                    thong_bao(f"Wilcoxon {kb}/{ten}/{x}−{y}", "thiếu cách huấn luyện")
                    continue
                doi = p[[x, y]].dropna()
                cap.append((f"{kb}/{ten}/{x}−{y}", doi[x].to_numpy(), doi[y].to_numpy()))
                thong_tin.append(int((doi[x] < doi[y]).sum()))
            if cap:
                kq = dg.wilcoxon_holm(cap)
                kq["ho_holm"] = f"{kb}/{ten}"
                kq["so_doi_chieu_co_du_lieu"] = len(cap)
                kq["so_dien_cuc_cai_thien"] = thong_tin
                kq["co_y_nghia_005"] = kq.p_holm < 0.05
                kq["ghi_chu_cong_suat"] = GIOI_HAN_WILCOXON
                cac_kiem_dinh.append(kq)
        if cac_kiem_dinh:
            out.luu_bang(pd.concat(cac_kiem_dinh, ignore_index=True), "bang_5_3_wilcoxon_holm",
                         "Holm riêng theo (KB, mô hình), các đối chiếu định trước C−A, C−B, D−A. " + GIOI_HAN_WILCOXON)
    else:
        thong_bao("Bảng 5.3", "thiếu cấu hình A/B/C/D")
    b5 = chon(g, cach="B", n_ao=5)
    if not b5.empty:
        out.luu_bang(tong_hop(theo_dien_cuc(b5, CAU_HINH), CAU_HINH), "bang_5_3_tham_chieu_B5")
    so_mau = g.loc[(g.cach == "C") | (g.cach == "A")].copy()
    so_mau.loc[so_mau.cach == "A", "n_ao"] = 0
    khoa = [c for c in ["kb", "mo_hinh", "cach", "n_ao", "seed", "ratio", "muc", "J", "ma_lan_chia"] if c in so_mau]
    so_mau = so_mau.drop_duplicates(khoa)
    _duong_sai_so(so_mau, "n_ao", "Số mẫu ảo mỗi điểm thật (×)", "hinh_5_4", out,
                  "A = ×0; C ở các mức còn lại; σ = 3%.")
    sigma = chon(bang, kb=["KB1", "KB2"], cach="C", n_ao=100, mo_hinh=MO_HINH_DU_LIEU_AO)
    _duong_sai_so(sigma, "sigma_lap", "σ_lặp (%)", "hinh_5_5", out, "C ×100.")


def _duong_sai_so(g, truc, nhan, ten, out, ghi_chu):
    if g.empty:
        thong_bao(ten, "thiếu nhóm/cấu hình")
        return
    nhom = ["kb", "mo_hinh", truc]
    t = tong_hop(theo_dien_cuc(g, nhom), nhom)
    ghi_chu += " " + _nhan_seed(t)
    out.luu_bang(t, ten + "_so_lieu", ghi_chu)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), squeeze=False)
    for ax, kb in zip(axs.flat, ("KB1", "KB2")):
        for mh, d in t.loc[t.kb == kb].groupby("mo_hinh"):
            d = d.sort_values(truc)
            x = d[truc].to_numpy(float) * (100 if truc == "sigma_lap" else 1)
            _ve_sai_so(ax, x, d, TEN.get(mh, mh))
        ax.set(title=kb, xlabel=nhan, ylabel="MAPE Csp (%)")
        if truc == "n_ao":
            ax.set_xscale("symlog", linthresh=1)
            ticks = sorted(g.n_ao.unique())
            ax.set_xticks(ticks, [f"×{int(x)}" for x in ticks])
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend(fontsize=8)
    out.luu_hinh(fig, ten, ghi_chu)


def _cau_hinh_ve(g, ten):
    """Chọn một cấu hình sẵn có, không trộn các mức thí nghiệm."""
    g = chon(g, cach="C", sigma_lap=0.03)
    if g.empty or "n_ao" not in g:
        return g
    n = 100 if (g.n_ao == 100).any() else sorted(g.n_ao.dropna().unique())[0]
    if n != 100:
        print(f"{ten}: dùng cấu hình có sẵn C ×{n:g}, σ=3% (không phải cấu hình chính ×100).")
    return chon(g, n_ao=n)


def kb3(bang, out):
    g = _cau_hinh_ve(chon(bang, kb="KB3"), "KB3")
    if not du_cot(g, CAU_HINH + ["k", "chinh_sach", "ratio", "seed", "Csp", "Csp_hat"], "KB3"):
        return
    nhom = CAU_HINH + ["chinh_sach", "k"]
    dc = theo_dien_cuc(g, nhom)
    t = tong_hop(dc, nhom)
    out.luu_bang(t, "hinh_5_6_so_lieu", "Chỉ chấm các mức dòng bị giấu; trung bình điện cực trước seed.")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for ax, cs, nhan in zip(axs.flat, ("thap_nhat", "trai_deu"), ("Giữ k mức thấp nhất", "Giữ k mức trải đều")):
        for mh, d in t.loc[t.chinh_sach == cs].groupby("mo_hinh"):
            d = d.sort_values("k")
            _ve_sai_so(ax, d.k, d, TEN.get(mh, mh))
        ax.axhline(10, color="black", linestyle="--", label="Ngưỡng 10%")
        ax.set(title=nhan, xlabel="Số mức dòng đã đo k", ylabel="MAPE Csp phần bị giấu (%)", xlim=(1, 5), xticks=range(1, 6))
        ax.secondary_xaxis("top", functions=(lambda k: (6-k)/6, lambda q: 6-6*q)).set_xlabel("q(k) = (6 − k)/6: phần phép đo tiết kiệm")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    out.luu_hinh(fig, "hinh_5_6", f"C ×{g.n_ao.iloc[0]:g}, σ=3%; ngưỡng áp dụng cho MAPE trung bình.\n" + _nhan_seed(t))
    hang = []
    for (cs, mh), d in t.groupby(["chinh_sach", "mo_hinh"]):
        dat = d.loc[d.MAPE_Csp_tb < 10].sort_values("k")
        r = dict(chinh_sach=cs, mo_hinh=mh, n_ao=g.n_ao.iloc[0], k_nho_nhat=np.nan, q=np.nan,
                 MAPE_tb=np.nan, MAPE_dien_cuc_lon_nhat=np.nan, sai_so_diem_lon_nhat=np.nan)
        if not dat.empty:
            k = int(dat.k.iloc[0])
            e = dc.loc[(dc.chinh_sach == cs) & (dc.mo_hinh == mh) & (dc.k == k)]
            p = g.loc[(g.chinh_sach == cs) & (g.mo_hinh == mh) & (g.k == k)]
            r.update(k_nho_nhat=k, q=(6-k)/6, MAPE_tb=dat.MAPE_Csp_tb.iloc[0],
                     MAPE_dien_cuc_lon_nhat=e.groupby("ratio").MAPE_Csp.mean().max(),
                     sai_so_diem_lon_nhat=pd.Series(dg.sai_so_tuong_doi(p.Csp, p.Csp_hat)).max())
        hang.append(r)
    out.luu_bang(pd.DataFrame(hang), "kb3_k_dat_nguong", "k nhỏ nhất quan sát được với MAPE trung bình <10%; trống = chưa đạt, không ngoại suy k thiếu.")


def kb4(bang, out):
    g = chon(bang, kb="KB4").copy()
    if not du_cot(g, CAU_HINH + ["seed", "ratio", "Csp", "Csp_hat", "Pd", "Pd_hat"], "Bảng 5.6"):
        return
    g["vung"] = np.where(np.isclose(g.ratio, DANH_MUC_TI_LE[0]) | np.isclose(g.ratio, DANH_MUC_TI_LE[-1]), "biên", "trong")
    nhom = CAU_HINH + ["vung"]
    out.luu_bang(tong_hop(theo_dien_cuc(g, nhom), nhom), "bang_5_6", "Giấu trọn điện cực; không gộp cấu hình; biên = tỉ lệ 0.025 và 1.75.")


def do_phu(bang, out):
    cot = CAU_HINH + ["seed", "ratio", "Csp", "Csp_L95", "Csp_U95"]
    if not du_cot(bang, cot, "Bảng độ phủ"):
        return
    nhom = CAU_HINH + [c for c in ("k", "chinh_sach") if c in bang]
    hang = []
    for khoa, g in bang.groupby(nhom + ["seed"], dropna=False):
        y, l, u = (g[c].to_numpy(float) for c in ("Csp", "Csp_L95", "Csp_U95"))
        hop_le = ~dg.khong_hop_le(y) & ~dg.khong_hop_le(l) & ~dg.khong_hop_le(u) & (l <= u)
        if not hop_le.any():
            continue
        r = dict(zip(nhom + ["seed"], khoa))
        d = g.loc[hop_le, ["ratio"]].copy()
        d["phu"] = ((l[hop_le] <= y[hop_le]) & (y[hop_le] <= u[hop_le])).astype(float)
        d["rong"] = (u[hop_le] - l[hop_le]) / y[hop_le]
        r.update(PICP=dg.tong_hop_theo_dien_cuc(d, "phu"), W_rel=dg.tong_hop_theo_dien_cuc(d, "rong"),
                 so_duoc_phu=int(d.phu.sum()), tong_khoang_hop_le=int(hop_le.sum()), tong_diem=len(g),
                 khoang_thieu_hoac_loi=int((~hop_le).sum()),
                 PICP_gop_diem=dg.picp(y, l, u), W_rel_gop_diem=dg.w_rel(y, l, u))
        hang.append(r)
    if not hang:
        thong_bao("Bảng độ phủ", "không có khoảng 95% hợp lệ")
        return
    chi_tiet = pd.DataFrame(hang)
    out.luu_bang(chi_tiet, "do_phu_tung_seed", "PICP/W_rel theo điện cực; số phủ/tổng là số điểm kiểm tra, mỗi seed riêng.")
    hang = []
    for khoa, g in chi_tiet.groupby(nhom, dropna=False):
        r = dict(zip(nhom, khoa))
        for c in ("PICP", "W_rel"):
            r[c] = _o(g[c].mean(), g[c].std(ddof=1))
            r[c + "_tb"], r[c + "_sd"] = g[c].mean(), g[c].std(ddof=1)
        for c in ("so_duoc_phu", "tong_khoang_hop_le", "tong_diem", "khoang_thieu_hoac_loi"):
            r[c] = int(g[c].sum())
        r["so_seed"] = len(g)
        hang.append(r)
    out.luu_bang(pd.DataFrame(hang), "do_phu", "PICP/W_rel: điện cực trước rồi seed; số phủ/tổng cộng qua seed, không phải điểm vật liệu độc lập.")


def _thu_tu(s):
    so = tuple(int(x) for x in re.findall(r"[1-3]", str(s)))
    if len(so) == 3 and set(so) == {1, 2, 3}:
        return so
    if str(s).lower() in ("mac_dinh", "chuan", "default"):
        return (1, 2, 3)
    return None


def kb5(bang, out):
    cot = ["phuong_phap", "seed", "thu_tu", "moi_truong", "buoc", "cuoi_moi_truong", "ratio", "Csp", "Csp_hat"]
    if not du_cot(bang, cot, "KB5"):
        return
    bang = bang.copy()
    bang["cuoi_moi_truong"] = bang.cuoi_moi_truong.astype(str).str.lower().isin(["true", "1"])
    thong_so = [c for c in ("so_nut", "so_canh", "t_hoc_s", "t_du_doan_s") if c in bang]
    for c in set(("so_nut", "so_canh", "t_hoc_s", "t_du_doan_s")) - set(thong_so):
        thong_bao("KB5 " + c, "thiếu cột")
    # MLP không có đồ thị: số 0 do bộ chạy ghi là không áp dụng, không phải mạng rỗng.
    for c in ("so_nut", "so_canh"):
        if c in bang:
            bang.loc[~bang.phuong_phap.isin(["gng_r", "pgcn_r"]), c] = np.nan
    bang["loi"] = dg.sai_so_tuong_doi(bang.Csp, bang.Csp_hat)
    bang["loi_du_doan"] = dg.khong_hop_le(bang.Csp_hat).astype(int)
    for c in ("Vbar_hat", "Pd_hat"):
        if c in bang:
            bang["loi_du_doan"] |= dg.khong_hop_le(bang[c]).astype(int)
    b54 = []
    dong, moc = [], []
    for tt, all_g in bang.groupby("thu_tu"):
        thu_tu = _thu_tu(tt)
        if thu_tu is None:
            thong_bao(f"KB5 thứ tự {tt}", "không xác định được ba nhóm điện cực từ thu_tu")
            continue
        nhom = [NHOM_GOC[i-1] for i in thu_tu]
        for i in range(1, 5):
            b54.append(dict(thu_tu=tt, moi_truong=i, ti_le="; ".join(f"{r:g}" for r in (nhom[i-1] if i < 4 else DANH_MUC_TI_LE))))
        for (pp, seed, mt, buoc), g in all_g.groupby(["phuong_phap", "seed", "moi_truong", "buoc"]):
            da_thay = [r for n in nhom[:min(int(mt), 3)] for r in n]
            giu = g.loc[g.ratio.isin(da_thay)]
            r = dict(thu_tu=tt, phuong_phap=pp, seed=seed, moi_truong=mt, buoc=buoc,
                     MAPE_Csp=dg.tong_hop_theo_dien_cuc(giu, "loi"),
                     so_khong_hop_le=int(giu.loi_du_doan.sum()), so_diem=len(giu),
                     cuoi_moi_truong=bool(g.cuoi_moi_truong.any()))
            for c in thong_so:
                r[c] = g[c].iloc[0]
            dong.append(r)
        cuoi = all_g.loc[all_g.cuoi_moi_truong]
        for (pp, seed), g in cuoi.groupby(["phuong_phap", "seed"]):
            e, tai_nguyen = {}, {}
            for mt, d in g.groupby("moi_truong"):
                d = d.loc[d.buoc == d.buoc.max()]
                da_thay = [r for n in nhom[:min(int(mt), 3)] for r in n]
                cham = d.loc[d.ratio.isin(da_thay)]
                tai_nguyen[f"so_khong_hop_le_t{int(mt)}"] = int(cham.loi_du_doan.sum())
                tai_nguyen[f"so_diem_t{int(mt)}"] = len(cham)
                for i in range(1, min(int(mt), 3)+1):
                    e[int(mt), i] = dg.tong_hop_theo_dien_cuc(d.loc[d.ratio.isin(nhom[i-1])], "loi")
                for c in thong_so:
                    tai_nguyen[f"{c}_t{int(mt)}"] = d[c].iloc[0]
            r = dict(thu_tu=tt, phuong_phap=pp, seed=seed, **tai_nguyen)
            for t, i in ((1, 1), (2, 2), (3, 1), (3, 2), (3, 3)):
                r[f"E_{t}_{i}"] = e.get((t, i), np.nan)
            if all(k in e for k in ((3, 1), (3, 2), (3, 3), (1, 1), (2, 2))):
                r["E3_bar"], r["F3"] = dg.e_bar_f3(e)
            for i in (1, 2, 3):
                r[f"E_4_{i}_tru_E_3_{i}"] = e.get((4, i), np.nan) - e.get((3, i), np.nan)
            for c in ("so_nut", "so_canh"):
                if f"{c}_t3" in tai_nguyen and f"{c}_t4" in tai_nguyen:
                    r[c + "_tang_dot4"] = tai_nguyen[f"{c}_t4"] - tai_nguyen[f"{c}_t3"]
            moc.append(r)
    if not dong:
        return
    out.luu_bang(pd.DataFrame(b54).drop_duplicates().assign(ghi_chu=GHI_CHU_KB5),
                 "phu_luc_bang_kb5_moi_truong", GHI_CHU_KB5 + ". Nhóm mới theo thứ tự thực nghiệm; đợt 4 là toàn bộ điện cực quay lại.")
    lich = pd.DataFrame(dong)
    out.luu_bang(lich.assign(ghi_chu=GHI_CHU_KB5), "phu_luc_kb5_theo_buoc",
                 GHI_CHU_KB5 + ". Chỉ chấm điện cực đã xuất hiện; lỗi dự đoán được đếm riêng.")
    if moc:
        chi_tiet = pd.DataFrame(moc)
        out.luu_bang(chi_tiet.assign(ghi_chu=GHI_CHU_KB5), "phu_luc_bang_kb5_tung_seed", GHI_CHU_KB5)
        rows = []
        for (tt, pp), g in chi_tiet.groupby(["thu_tu", "phuong_phap"]):
            r = dict(thu_tu=tt, phuong_phap=pp, so_seed=len(g))
            for c in chi_tiet.columns.difference(["thu_tu", "phuong_phap", "seed"]):
                r[c], r[c + "_tb"], r[c + "_sd"] = _o(g[c].mean(), g[c].std(ddof=1)), g[c].mean(), g[c].std(ddof=1)
            rows.append(r)
        out.luu_bang(pd.DataFrame(rows).assign(ghi_chu=GHI_CHU_KB5), "phu_luc_bang_kb5",
                     GHI_CHU_KB5 + ". MAPE E_t,i trên nhóm i sau đợt t; E3_bar/F3 chỉ dùng 3 đợt đầu; thời gian học cộng dồn. Số không hợp lệ: mỗi mốc cuối đợt, chỉ điện cực đã xuất hiện; trung bình ± SD qua seed.")
    else:
        thong_bao("Bảng phụ lục KB5", "chưa có mốc cuối môi trường")
    for tt, g in lich.groupby("thu_tu"):
        duoi = "" if lich.thu_tu.nunique() == 1 else "_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(tt))
        fig, ax = plt.subplots(figsize=(9, 5))
        _ve_lich(ax, g, "MAPE_Csp")
        _vach_dot(ax, g)
        ax.set(xlabel="Số mẫu đã học cộng dồn", ylabel="MAPE Csp (%)",
               title=f"{GHI_CHU_KB5}\nĐiện cực đã xuất hiện — thứ tự {tt}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        out.luu_hinh(fig, "phu_luc_hinh_kb5_sai_so" + duoi,
                     GHI_CHU_KB5 + ". Trung bình điện cực rồi seed; không phải khoảng dự đoán.\n" + _nhan_seed(g))
        if not {"so_nut", "so_canh"}.issubset(g.columns):
            thong_bao("Hình phụ lục KB5 nút/cạnh", "thiếu số nút/cạnh")
            continue
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, c, nhan in zip(axs, ("so_nut", "so_canh"), ("Số nút", "Số cạnh")):
            _ve_lich(ax, g, c)
            _vach_dot(ax, g)
            ax.set(xlabel="Số mẫu đã học cộng dồn", ylabel=nhan,
                   title=f"{GHI_CHU_KB5}\n{nhan} — thứ tự {tt}")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=8)
        out.luu_hinh(fig, "phu_luc_hinh_kb5_nut_canh" + duoi,
                     GHI_CHU_KB5 + ". Chỉ mô hình đồ thị; " + _nhan_seed(g))


def _ve_lich(ax, g, cot):
    for pp, d in g.groupby("phuong_phap"):
        d = d.groupby("buoc")[cot].agg(["mean", "std"]).dropna(subset=["mean"]).sort_index()
        if d.empty:
            continue
        x, y, sd = d.index.to_numpy(float), d["mean"].to_numpy(), d["std"].to_numpy()
        line, = ax.plot(x, y, marker=".", label=TEN.get(pp, pp))
        if np.isfinite(sd).any():
            ax.fill_between(x, y-sd, y+sd, alpha=0.13, color=line.get_color())


def _vach_dot(ax, g):
    # Phương pháp có số bước khác nhau: vạch riêng, không lấy trung bình vị trí chuyển đợt.
    cuoi = g.loc[g.cuoi_moi_truong & (g.moi_truong < 4)]
    for (pp, mt), d in cuoi.groupby(["phuong_phap", "moi_truong"]):
        for b in sorted(d.buoc.unique()):
            ax.axvline(b, color="gray", linestyle=":", alpha=0.35)
    if not cuoi.empty:
        ax.plot([], [], color="gray", linestyle=":", label="Cuối đợt (từng phương pháp/seed)")


def ung_dung(out, ten="gpr", n_ao=100, cach="A"):
    from . import du_doan as dd
    from .du_lieu import doc_so_lieu
    that = doc_so_lieu()
    print(dd.GHI_CHU)
    mh = dd.khop_toan_bo(ten, cach, n_ao, 0)
    ghi_chu = f"{dd.GHI_CHU}; mô hình {ten}, cách {cach}" + (f" ×{n_ao}" if cach != "A" else " (chỉ số đo thật)")
    ratios = [DANH_MUC_TI_LE[i] for i in (0, 3, 7)]
    fig9, axs9 = plt.subplots(1, 3, figsize=(15, 4.8))
    fig10, axs10 = plt.subplots(1, 3, figsize=(15, 4.8))
    bang, de_xuat = [], []
    for ratio, ax9, ax10 in zip(ratios, axs9, axs10):
        g = that.loc[np.isclose(that.ratio, ratio)]
        m = 0.010 / g.J.min()
        j = np.geomspace(g.J.min(), max(g.J.max(), 0.200/m), 120)
        p = dd.du_doan_luoi(mh, ratio, j)
        p["ghi_chu"] = ghi_chu
        bang.append(p)
        ax9.plot(p.Pd, p.Ed, label="Dự đoán")
        if {"Ed_L95", "Ed_U95"}.issubset(p.columns):
            ax9.fill_between(p.Pd, p.Ed_L95, p.Ed_U95, alpha=0.2, label="Biên Ed 95%")
        if {"Pd_L95", "Pd_U95"}.issubset(p.columns):
            ax9.fill_betweenx(p.Ed, p.Pd_L95, p.Pd_U95, alpha=0.12, label="Biên Pd 95%")
        ax9.scatter(g.Pd, g.Ed, marker="x", color="black", label="Số đo thật")
        ax9.set(xscale="log", yscale="log", xlabel="Pd (W/kg)", ylabel="Ed (Wh/kg)", title=f"Tỉ lệ {ratio:g}")
        ax9.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=6))
        ax9.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax9.xaxis.set_minor_formatter(NullFormatter())
        ax10.plot(p.J, p.Csp, label="Dự đoán")
        if {"Csp_L95", "Csp_U95"}.issubset(p.columns):
            ax10.fill_between(p.J, p.Csp_L95, p.Csp_U95, alpha=0.2, label="Khoảng dự đoán 95%")
        ax10.scatter(g.J, g.Csp, marker="x", color="black", label="Đã đo")
        dx = dd.de_xuat_muc_do(mh, that, ratio, np.arange(10, 201, 10))
        de_xuat.append(dict(ratio=ratio, **dx, ghi_chu=ghi_chu))
        if "J" in dx:
            ax10.axvline(dx["J"], color="red", linestyle="--", label=f"Đề xuất {dx['dong_mA']:g} mA")
        else:
            print(f"Tỉ lệ {ratio:g}: {dx['ly_do']}")
        ax10.set(xlabel="J (A/g)", ylabel="Csp (F/g)", title=f"Tỉ lệ {ratio:g}")
        for ax in (ax9, ax10):
            ax.grid(which="both", alpha=0.2)
            ax.legend(fontsize=7)
    out.luu_bang(pd.concat(bang, ignore_index=True), "ung_dung_du_doan", ghi_chu)
    out.luu_bang(pd.DataFrame(de_xuat), "ung_dung_de_xuat", ghi_chu)
    out.luu_hinh(fig9, "hinh_5_9", ghi_chu + "\nBiên Ed/Pd riêng, không phải khoảng đồng thời.")
    out.luu_hinh(fig10, "hinh_5_10", ghi_chu + "\nĐề xuất: σ_logCsp lớn nhất trong các mức chưa đo.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thu-muc", type=Path, required=True)
    parser.add_argument("--ung-dung", action="store_true", help="Khớp lại toàn bộ 48 điểm để tạo Hình 5.9–5.10")
    parser.add_argument("--mo-hinh-ung-dung", default="gpr")
    parser.add_argument("--cach-ung-dung", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--n-ao-ung-dung", type=int, default=100)
    args = parser.parse_args(argv)
    out = DauRa(args.thu_muc)
    b = doc_ket_qua(args.thu_muc / "ket_qua_kb1_kb4.csv")
    if not b.empty:
        if {"kb", "ma_lan_chia"}.issubset(b.columns):
            bo = (b.kb == "KB2") & b.ma_lan_chia.astype(str).str.endswith("_bo_0.075")
            b.loc[bo, "kb"] = "KB2_BO_0.075"
        bang_chinh(b, out)
        hinh_du_doan(b, out)
        anh_huong_ao(b, out)
        kb3(b, out)
        kb4(b, out)
        do_phu(b, out)
    kb5(doc_ket_qua(args.thu_muc / "ket_qua_kb5.csv"), out)
    if args.ung_dung:
        ung_dung(out, args.mo_hinh_ung_dung, args.n_ao_ung_dung, args.cach_ung_dung)
    print(f"Đã ghi {len(out.tep)} tệp:")
    for p in out.tep:
        print(f"  {p}")
    return out.tep


if __name__ == "__main__":
    main()
