"""So sánh điện cực tại cùng J; khớp lại để sử dụng, không đánh giá tổng quát hóa.

Chạy: python -X utf8 -m ml.so_sanh --thu-muc ket_qua/thu
Lưới ``chung`` là giao các khoảng đo; ``toan_bo`` là hợp và có cờ ngoại suy.
Khoảng từ sd log là biên dự đoán 95%; phân vị qua seed chỉ là phân tán giữa
các lần khớp, không phải khoảng dự đoán. Với mô hình có sd, dùng seed đầu
(tức một phân bố dự đoán, không trộn sd mô hình với phân tán giữa seed).
Xác suất Monte Carlo giả định các điện cực độc lập; không khẳng định nhân quả
hoặc tìm tỉ lệ tối ưu ngoài 8 điện cực đã đo. Hạng bằng nhau được giữ nguyên;
xác suất thắng khi hòa được chia đều, không thiên vị thứ tự điện cực.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from . import vat_ly
from .danh_gia import khong_hop_le, khoang_95
from .du_doan import (TEN_DO_THI, TEN_NEN, _canh_bao_noi_suy, khop_toan_bo)
from .du_lieu import doc_so_lieu
from .mo_hinh_ml import TEN_MO_HINH

GHI_CHU = "Khớp lại để sử dụng; độ tin cậy dự đoán xem kết quả KB1/KB2."
NGUON_SD = "Khoảng dự đoán biên 95% từ sd mô hình"
NGUON_SEED = "Phân tán qua seed, không phải khoảng dự đoán"
KHONG_DAI = "Không có dải bất định"


def luoi_J(bang, mien="chung", so_diem=60):
    """Trả lưới log đều trong giao/hợp các khoảng J đo của từng điện cực."""
    if mien not in ("chung", "toan_bo"):
        raise ValueError("Miền phải là chung hoặc toan_bo.")
    if so_diem < 2:
        raise ValueError("Lưới cần ít nhất 2 điểm J.")
    if bang.empty or khong_hop_le(bang.J).any():
        raise ValueError("Bảng đo cần các J hữu hạn, dương.")
    bien = bang.groupby("ratio").J.agg(["min", "max"])
    lo = bien["min"].max() if mien == "chung" else bien["min"].min()
    hi = bien["max"].min() if mien == "chung" else bien["max"].max()
    if lo > hi:
        raise ValueError("Các điện cực không có khoảng J chung.")
    return np.unique(np.geomspace(lo, hi, so_diem))


def du_doan_luoi_chung(mo_hinh, bang, J):
    """Dự đoán mọi điện cực trên cùng J, suy Ed/Pd và giữ nguyên giá trị lỗi."""
    j = np.asarray(J, float).reshape(-1)
    if not len(j) or khong_hop_le(j).any():
        raise ValueError("J phải hữu hạn, dương và không rỗng.")
    cac_bang = []
    co_muc_moi = False
    for ratio, g in bang.groupby("ratio", sort=True):
        x = pd.DataFrame({"ratio": ratio, "J": j})
        trung = np.isclose(j[:, None], g.J.to_numpy()[None, :], rtol=1e-8, atol=1e-10)
        co_muc_moi |= not trung.any(axis=1).all()
        if mo_hinh.ten == "tb_theo_muc":
            if not trung.any(axis=1).all():
                raise ValueError("tb_theo_muc không dự đoán được tại J chưa đo.")
            x["muc"] = g.muc.to_numpy()[trung.argmax(axis=1)]
        cac_bang.append(x)
    _canh_bao_noi_suy(mo_hinh, co_muc_moi)
    x = pd.concat(cac_bang, ignore_index=True)
    y = mo_hinh.predict(x)
    kq = x[["ratio", "J"]].copy()
    for cot, gia_tri in vat_ly.suy_ra(x.J, y.Csp_hat, y.Vbar_hat).items():
        if cot != "dt":
            kq[cot + "_hat"] = gia_tri
    kq["khong_hop_le"] = khong_hop_le(y.Csp_hat) | khong_hop_le(y.Vbar_hat)
    bien = bang.groupby("ratio").J.agg(["min", "max"])
    kq["ngoai_mien_do"] = ((x.J < x.ratio.map(bien["min"])) |
                             (x.J > x.ratio.map(bien["max"])))
    co_sd = False
    for cot, sd_cot in (("Csp", "logCsp_sd"), ("Vbar", "logVbar_sd")):
        sd = y[sd_cot].to_numpy(float)
        kq[sd_cot] = sd
        if np.any(np.isfinite(sd) & (sd >= 0)):
            co_sd = True
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = khoang_95(np.log(kq[cot + "_hat"]), sd)
            kq[cot + "_L95"], kq[cot + "_U95"] = lo, hi
    _suy_bien(kq)
    kq["nguon_bat_dinh"] = NGUON_SD if co_sd else KHONG_DAI
    return kq


def _suy_bien(bang):
    """Biên Ed theo Csp, biên Pd theo V̄; không tạo khoảng đồng thời."""
    for bien in ("L95", "U95"):
        if "Csp_" + bien in bang:
            bang["Ed_" + bien] = vat_ly.ed_tu_csp(bang["Csp_" + bien])
        if "Vbar_" + bien in bang:
            bang["Pd_" + bien] = vat_ly.pd_tu_vbar(bang.J, bang["Vbar_" + bien])


def xac_suat_tot_nhat(mu_log, sd_log, so_mau=4000, seed=0):
    """Xác suất thắng; trục cuối là điện cực, các trục trước là điểm J.

    So sánh mẫu trên thang log tương đương lấy mẫu log-normal rồi so sánh,
    tránh tràn exp. Một J thiếu mu/sd của bất kỳ điện cực nào trả toàn NaN.
    """
    mu, sd = np.broadcast_arrays(np.asarray(mu_log, float), np.asarray(sd_log, float))
    if mu.ndim == 0 or not mu.shape[-1] or so_mau < 1:
        raise ValueError("Cần trục điện cực không rỗng và số mẫu MC dương.")
    kq = np.full(mu.shape, np.nan)
    rng = np.random.default_rng(seed)
    for m, s, p in zip(mu.reshape(-1, mu.shape[-1]),
                       sd.reshape(-1, sd.shape[-1]), kq.reshape(-1, mu.shape[-1])):
        if not (np.isfinite(m).all() and np.isfinite(s).all() and (s >= 0).all()):
            continue
        mau = rng.normal(m, s, size=(so_mau, len(m)))
        thang = mau == mau.max(axis=1, keepdims=True)
        p[:] = (thang / thang.sum(axis=1, keepdims=True)).mean(axis=0)
    return kq


def _tan_suat_seed(bang_seed, j, ratios):
    cac_xac_suat = []
    for _, g in bang_seed.loc[bang_seed.J == j].groupby("seed", sort=True):
        csp = g.set_index("ratio").Csp_hat.reindex(ratios).to_numpy(float)
        if khong_hop_le(csp).any():
            return np.full(len(ratios), np.nan)
        thang = csp == csp.max()
        cac_xac_suat.append(thang / thang.sum())
    return np.mean(cac_xac_suat, axis=0) if cac_xac_suat else np.full(len(ratios), np.nan)


def xep_hang(du_doan, so_mau=4000, seed=0, du_doan_seeds=None):
    """Hạng giảm dần theo Csp/Pd; Ed cùng hạng Csp theo công thức vật lý.

    Bảng nhiều mô hình được nhóm riêng. Có sd thì dùng MC, không có sd thì
    dùng tần suất qua ít nhất hai seed; nếu không, xác suất để trống.
    """
    bang = du_doan.copy()
    if "mo_hinh" not in bang:
        bang["mo_hinh"] = "mo_hinh"
    cac_bang = []
    for i, ((ten, j), g) in enumerate(bang.groupby(["mo_hinh", "J"], sort=True)):
        g = g.sort_values("ratio").copy()
        for cot in ("Csp", "Pd"):
            gia_tri = g[cot + "_hat"].mask(khong_hop_le(g[cot + "_hat"]))
            g["hang_" + cot] = gia_tri.rank(method="min", ascending=False)
        p = np.full(len(g), np.nan)
        nguon = "Không có xác suất; chỉ thứ hạng"
        if "logCsp_sd" in g and np.isfinite(g.logCsp_sd).any():
            with np.errstate(divide="ignore", invalid="ignore"):
                p = xac_suat_tot_nhat(np.log(g.Csp_hat), g.logCsp_sd, so_mau, seed + i)
            nguon = "Monte Carlo log-normal; các điện cực độc lập"
        elif du_doan_seeds is not None:
            b = du_doan_seeds.loc[du_doan_seeds.mo_hinh == ten]
            if b.seed.nunique() > 1:
                p = _tan_suat_seed(b, j, g.ratio)
                nguon = "Tần suất tốt nhất qua seed, không phải xác suất dự đoán"
        g["xac_suat_tot_nhat_Csp"] = p
        g["nguon_xac_suat"] = nguon
        cot = ["mo_hinh", "J", "ratio", "hang_Csp", "hang_Pd",
               "xac_suat_tot_nhat_Csp", "nguon_xac_suat"]
        cac_bang.append(g[cot])
    return pd.concat(cac_bang, ignore_index=True)


def _gom_seed(cac_bang, cach):
    """Trung vị qua seed; chỉ lấy phân vị khi không có sd và cách khác A."""
    kq = cac_bang[0].copy()
    if len(cac_bang) == 1:
        return kq
    for cot in ("Csp", "Vbar"):
        gia_tri = np.stack([b[cot + "_hat"].to_numpy() for b in cac_bang])
        # Không bỏ seed lỗi để tạo một dự đoán tổng hợp có vẻ hợp lệ.
        gia_tri = np.where(khong_hop_le(gia_tri), np.nan, gia_tri)
        kq[cot + "_hat"] = np.median(gia_tri, axis=0)
        if cach != "A":
            kq[cot + "_L95"], kq[cot + "_U95"] = np.quantile(gia_tri, [0.025, 0.975], axis=0)
    kq["Ed_hat"] = vat_ly.ed_tu_csp(kq.Csp_hat)
    kq["Pd_hat"] = vat_ly.pd_tu_vbar(kq.J, kq.Vbar_hat)
    kq["khong_hop_le"] = np.logical_or.reduce([b.khong_hop_le.to_numpy() for b in cac_bang])
    if cach != "A":
        _suy_bien(kq)
        kq["nguon_bat_dinh"] = NGUON_SEED
    return kq


def tom_tat(xep):
    """Mỗi mô hình một hàng; đoạn gồm các nút lưới liên tiếp cùng tập thắng.

    Đồng thuận toàn bộ = tỉ lệ J có cùng tập điện cực hạng 1 ở mọi mô hình.
    Đồng thuận đa số = trung bình tỉ lệ mô hình chọn tập thắng phổ biến nhất.
    Điện cực hòa hạng 1 đều được tính là tốt nhất; không tự phá hòa.
    """
    tot = {(ten, j): tuple(g.loc[g.hang_Csp == 1, "ratio"])
           for (ten, j), g in xep.groupby(["mo_hinh", "J"], sort=True)}
    ten_mh = sorted(xep.mo_hinh.unique())
    js = sorted(xep.J.unique())
    dong_thuan, da_so = [], []
    for j in js:
        tap = [tot[(ten, j)] for ten in ten_mh]
        hop_le = all(tap)
        dong_thuan.append(hop_le and len(set(tap)) == 1)
        da_so.append(max(tap.count(t) for t in tap) / len(tap) if hop_le else 0.0)
    hang = []
    for ten, g in xep.groupby("mo_hinh", sort=True):
        doan = []
        for j in js:
            thang = list(tot[(ten, j)])
            if doan and doan[-1]["ratio_tot_nhat"] == thang:
                doan[-1]["J_den"] = float(j)
            else:
                doan.append(dict(J_tu=float(j), J_den=float(j), ratio_tot_nhat=thang))
        p = g.groupby("ratio").xac_suat_tot_nhat_Csp.mean()
        hang.append(dict(
            mo_hinh=ten, doan_J_tot_nhat_Csp=json.dumps(doan, ensure_ascii=False),
            ti_le_luoi_01_tot_nhat=np.mean([0.1 in tot[(ten, j)] for j in js]),
            xac_suat_trung_binh_01=p.get(0.1, np.nan),
            xac_suat_trung_binh=json.dumps({f"{r:g}": None if pd.isna(v) else float(v)
                                           for r, v in p.items()}, ensure_ascii=False),
            ti_le_dong_thuan_toan_bo=float(np.mean(dong_thuan)),
            dong_thuan_da_so_trung_binh=float(np.mean(da_so)),
            so_mo_hinh=len(ten_mh), ghi_chu=GHI_CHU))
    return pd.DataFrame(hang)


def ve_hinh(du_doan, xep, bang_that, thu_muc, mien, cach):
    """Ba hình chung, mỗi mô hình một ô; biên Ragone là biên riêng Ed/Pd."""
    ten_mh = list(du_doan.mo_hinh.unique())
    ratios = sorted(bang_that.ratio.unique())
    mau = dict(zip(ratios, plt.get_cmap("tab10").colors))
    chu_giai = [Line2D([0], [0], color=mau[r], label=f"Tỉ lệ {r:g}") for r in ratios]
    mien_chu = "J chung (nội suy)" if mien == "chung" else "J toàn bộ (có ngoại suy)"
    cau_hinh = f"Cách {cach} · Miền {mien_chu}"
    loai_hinh = (("csp_theo_J_luoi_chung", "Điện dung tại cùng mật độ dòng"),
                 ("ragone_lien_tuc", "Ragone liên tục tại cùng mật độ dòng"),
                 ("xac_suat_tot_nhat", "Xác suất điện cực tốt nhất theo Csp"))
    for ten_hinh, tieu_de in loai_hinh:
        fig, axes = plt.subplots(1, len(ten_mh), figsize=(6 * len(ten_mh), 6.1), squeeze=False)
        for ax, ten in zip(axes.flat, ten_mh):
            b = du_doan.loc[du_doan.mo_hinh == ten]
            nguon = b.nguon_bat_dinh.iloc[0]
            for r in ratios:
                g = b.loc[b.ratio == r].sort_values("J")
                that = bang_that.loc[bang_that.ratio == r]
                if ten_hinh == "xac_suat_tot_nhat":
                    g = xep.loc[(xep.mo_hinh == ten) & (xep.ratio == r)].sort_values("J")
                    ax.plot(g.J, g.xac_suat_tot_nhat_Csp, color=mau[r])
                    continue
                g = g.loc[~g.khong_hop_le]
                if ten_hinh == "csp_theo_J_luoi_chung":
                    ax.plot(g.J, g.Csp_hat, color=mau[r])
                    if "Csp_L95" in g:
                        ax.fill_between(g.J.to_numpy(), g.Csp_L95.to_numpy(), g.Csp_U95.to_numpy(),
                                        color=mau[r], alpha=0.12)
                    ax.scatter(that.J, that.Csp, color=mau[r], marker="x", s=22)
                else:
                    ax.plot(g.Pd_hat, g.Ed_hat, color=mau[r])
                    if "Ed_L95" in g:
                        ax.fill_between(g.Pd_hat.to_numpy(), g.Ed_L95.to_numpy(), g.Ed_U95.to_numpy(),
                                        color=mau[r], alpha=0.12)
                    if "Pd_L95" in g:
                        ax.fill_betweenx(g.Ed_hat.to_numpy(), g.Pd_L95.to_numpy(), g.Pd_U95.to_numpy(),
                                         color=mau[r], alpha=0.07)
                    ax.scatter(that.Pd, that.Ed, color=mau[r], marker="x", s=22)
            if ten_hinh == "xac_suat_tot_nhat":
                prob = xep.loc[xep.mo_hinh == ten]
                co_xs = prob.xac_suat_tot_nhat_Csp.notna().any()
                if not co_xs:
                    ax.text(0.5, 0.5, "Không có xác suất\nChỉ có thứ hạng (xep_hang.csv)",
                            transform=ax.transAxes, ha="center", va="center", fontsize=10)
                nguon_xs = prob.nguon_xac_suat.iloc[0]
                phu_de = ("MC log-normal, điện cực độc lập" if nguon_xs.startswith("Monte") else
                          "Tần suất qua seed, không phải xác suất dự đoán" if co_xs else "Không có sd / nhiều seed")
                ax.set(xlabel="Mật độ dòng J (A/g)", ylabel="Xác suất / tần suất tốt nhất",
                       xscale="log", xlim=(b.J.min(), b.J.max()), ylim=(-0.02, 1.02))
                ax.set_title(f"{ten}\n{phu_de}", fontsize=10)
            else:
                ax.set_title(f"{ten}\n{nguon}", fontsize=10)
                if ten_hinh == "csp_theo_J_luoi_chung":
                    ax.set(xlabel="Mật độ dòng J (A/g)", ylabel="Điện dung riêng Csp (F/g)", xscale="log")
                else:
                    ax.set(xlabel="Mật độ công suất Pd (W/kg)", ylabel="Mật độ năng lượng Ed (Wh/kg)",
                           xscale="log", yscale="log")
            ax.grid(True, which="both", alpha=0.2)
        fig.suptitle(f"{tieu_de}\n{cau_hinh}", fontsize=13)
        fig.legend(handles=chu_giai, loc="lower center", bbox_to_anchor=(0.5, 0.082), ncol=4, fontsize=9)
        ghi_chu = "×: số đo thật. Dải: biên riêng, không phải khoảng đồng thời Ed–Pd."
        if ten_hinh == "xac_suat_tot_nhat":
            ghi_chu = "MC: bất định mô hình; tần suất seed: độ ổn định giữa các lần khớp."
        fig.text(0.5, 0.048, ghi_chu, ha="center", fontsize=9)
        fig.text(0.5, 0.016, GHI_CHU, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.19, 1, 0.91))
        for duoi in ("png", "pdf"):
            fig.savefig(Path(thu_muc) / f"{ten_hinh}.{duoi}", dpi=200)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thu-muc", type=Path, default=Path("ket_qua/thu"))
    parser.add_argument("--mo-hinh", nargs="+", choices=tuple(t for t in (*TEN_MO_HINH, *TEN_NEN, *TEN_DO_THI)
                                                          if t != "tb_theo_muc"),
                        default=["gpr", "chon_dang_ham", "mlp"])
    parser.add_argument("--cach", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--n-ao", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--so-diem-J", type=int, default=60)
    parser.add_argument("--mien", choices=("chung", "toan_bo"), default="chung")
    parser.add_argument("--so-mau-mc", type=int, default=4000)
    parser.add_argument("--tham-so", help="JSON tham số dùng chung cho các mô hình được chọn")
    args = parser.parse_args(argv)
    if args.n_ao < 0 or args.so_mau_mc < 1:
        parser.error("--n-ao phải không âm, --so-mau-mc phải dương.")
    try:
        tham_so = None if args.tham_so is None else json.loads(args.tham_so)
        if tham_so is not None and not isinstance(tham_so, dict):
            raise ValueError("--tham-so phải là đối tượng JSON.")
        that = doc_so_lieu()
        j_chung = luoi_J(that, "chung", args.so_diem_J)
        j = luoi_J(that, args.mien, args.so_diem_J)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(GHI_CHU)
    print(f"Khoảng J chung: [{j_chung.min():.6g}, {j_chung.max():.6g}] A/g.")
    print(f"Miền {args.mien}: [{j.min():.6g}, {j.max():.6g}] A/g; {len(j)} điểm.")
    du_doan, tat_ca_seed = [], []
    seeds = list(dict.fromkeys(args.seeds))
    for ten in dict.fromkeys(args.mo_hinh):
        cac_bang = []
        for seed in seeds:
            mo_hinh = khop_toan_bo(ten, args.cach, args.n_ao, seed, tham_so)
            b = du_doan_luoi_chung(mo_hinh, that, j)
            b["mo_hinh"], b["cach"], b["seed"] = ten, args.cach, seed
            cac_bang.append(b)
            # Không trộn phân tán seed với sd; A + mô hình tất định không có dải.
            if mo_hinh.ho_tro_khoang or (args.cach == "A" and ten in (*TEN_NEN, "ridge", "svr")):
                break
        g = _gom_seed(cac_bang, args.cach)
        g = g.drop(columns="seed")
        g["seeds_su_dung"] = " ".join(str(int(b.seed.iloc[0])) for b in cac_bang)
        g["ghi_chu"] = GHI_CHU
        du_doan.append(g)
        tat_ca_seed.extend(cac_bang)
        print(f"{ten}: {g.nguon_bat_dinh.iloc[0]}; seed {g.seeds_su_dung.iloc[0]}.")
    du_doan = pd.concat(du_doan, ignore_index=True)
    hang = xep_hang(du_doan, args.so_mau_mc, seed=0,
                    du_doan_seeds=pd.concat(tat_ca_seed, ignore_index=True))
    tom = tom_tat(hang)
    out = args.thu_muc / "so_sanh"
    out.mkdir(parents=True, exist_ok=True)
    for ten, bang in (("du_doan_luoi", du_doan), ("xep_hang", hang), ("tom_tat", tom)):
        bang.to_csv(out / f"{ten}.csv", index=False, encoding="utf-8-sig")
    ve_hinh(du_doan, hang, that, out, args.mien, args.cach)
    for dong in tom.itertuples():
        print(f"{dong.mo_hinh}: điện cực tốt nhất theo Csp tại các nút lưới:")
        for doan in json.loads(dong.doan_J_tot_nhat_Csp):
            ratios = ", ".join(f"{r:g}" for r in doan["ratio_tot_nhat"]) or "không có dự đoán hợp lệ"
            print(f"  J ∈ [{doan['J_tu']:.6g}, {doan['J_den']:.6g}] A/g: {ratios}")
        print(f"  Tỉ lệ lưới mà 0,1 tốt nhất: {dong.ti_le_luoi_01_tot_nhat:.1%}; "
              f"xác suất/tần suất trung bình của 0,1: {dong.xac_suat_trung_binh_01:.4g}.")
    print(f"Đồng thuận toàn bộ: {tom.ti_le_dong_thuan_toan_bo.iloc[0]:.1%}; "
          f"đồng thuận đa số trung bình: {tom.dong_thuan_da_so_trung_binh.iloc[0]:.1%} "
          f"({len(tom)} mô hình).")
    print(f"Ngoài miền đo: {int(du_doan.ngoai_mien_do.sum())}; "
          f"không hợp lệ: {int(du_doan.khong_hop_le.sum())}. Đã ghi: {out}")


if __name__ == "__main__":
    main()
