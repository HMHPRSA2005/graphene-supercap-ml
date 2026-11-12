"""Khớp toàn bộ để sử dụng (mục 4.8), không dùng làm kết quả kiểm tra."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import vat_ly
from .danh_gia import khoang_95, khong_hop_le
from .du_lieu import doc_so_lieu
from .mo_hinh_ml import LUOI, TEN_MO_HINH, TOI_DA_HANG, tao_mo_hinh
from .mo_hinh_nen import (MoHinhDangHam, MoHinhTrungBinhHinhHoc,
                          MoHinhTrungBinhTheoMuc)
from .tang_cuong import CauHinhTangCuong, lay_mau_con, tao_tap_hoc

GHI_CHU = "khớp lại để sử dụng, không phải kết quả kiểm tra"
TEN_NEN = ("ham_mu", "chon_dang_ham", "tb_hinh_hoc", "tb_theo_muc")
TEN_DO_THI = ("gng_r", "pgcn_r")


def khop_toan_bo(ten, cach, n_ao, seed, tham_so=None):
    """Trả mô hình đã fit; mã hóa/chuẩn hóa được thực hiện bên trong mô hình."""
    bang = doc_so_lieu()
    if ten in TEN_MO_HINH:
        p = dict(LUOI[ten][0]) if tham_so is None else tham_so
        mo_hinh = tao_mo_hinh(ten, p, seed=seed)
    elif ten in TEN_DO_THI:
        from .gng_r import LUOI_DO_THI, MoHinhDoThi
        p = dict(LUOI_DO_THI[ten][0]) if tham_so is None else tham_so
        mo_hinh = MoHinhDoThi(ten, p, seed=seed)
    elif ten in TEN_NEN:
        if tham_so:
            raise ValueError("Mô hình nền không nhận --tham-so.")
        p = {}
        if ten in ("ham_mu", "chon_dang_ham"):
            mo_hinh = MoHinhDangHam("exp" if ten == "ham_mu" else "auto")
        elif ten == "tb_hinh_hoc":
            mo_hinh = MoHinhTrungBinhHinhHoc()
        else:
            mo_hinh = MoHinhTrungBinhTheoMuc()
    else:
        raise ValueError(f"Mô hình không được hỗ trợ: {ten}")
    bang_hoc = tao_tap_hoc(bang, CauHinhTangCuong(cach=cach, n_ao=n_ao), seed)
    if ten in TOI_DA_HANG:
        bang_hoc = lay_mau_con(bang_hoc, TOI_DA_HANG[ten], seed)
    mo_hinh.fit(bang_hoc, bang)
    mo_hinh.thong_tin_khop_ = dict(ghi_chu=GHI_CHU, mo_hinh=ten, cach=cach,
                                  n_ao=n_ao, seed=seed, tham_so=p,
                                  so_diem_that=len(bang), so_hang_sinh=len(bang_hoc))
    return mo_hinh


def _dien_cuc(bang, ti_le):
    g = bang.loc[np.isclose(bang["ratio"], ti_le, rtol=0, atol=1e-9)].sort_values("J")
    if g.empty:
        danh_muc = ", ".join(f"{float(r):g}" for r in sorted(bang.ratio.unique()))
        raise ValueError(f"Tỉ lệ {ti_le:g} chưa có số đo; chọn một trong: {danh_muc}.")
    return g


def _dong_duong(values):
    a = np.asarray(values, dtype=float).reshape(-1)
    if not len(a) or not np.all(np.isfinite(a) & (a > 0)):
        raise ValueError("Dòng điện/mật độ dòng phải hữu hạn, dương và không rỗng.")
    return a


def _canh_bao_noi_suy(mo_hinh, co_muc_moi):
    thong_tin = getattr(mo_hinh, "thong_tin_khop_", {})
    if mo_hinh.ten == "gpr" and thong_tin.get("cach") == "C" and co_muc_moi:
        warnings.warn(
            "Cảnh báo: GPR + cách C chỉ sinh mẫu quanh các mức dòng đã đo; "
            "độ dài tương quan theo J có thể co nhỏ, gây dao động mạnh giữa các mức. "
            "Dự đoán và đề xuất ở mức chưa đo có thể thiếu tin cậy. "
            "Mặc định dùng A; chọn cách tăng cường theo kết quả Bảng 5.3.",
            UserWarning)


def du_doan_luoi(mo_hinh, ti_le, J):
    """Bảng thang gốc và khoảng biên 95% nếu có sd; không bịa bất định Δt."""
    j = _dong_duong(J)
    g = _dien_cuc(doc_so_lieu(), ti_le)
    x = pd.DataFrame({"ratio": float(g.ratio.iloc[0]), "J": j})
    # Trung bình theo mức chỉ có nghĩa ở đúng mức dòng đã đo.
    khop = np.isclose(j[:, None], g.J.to_numpy()[None, :], rtol=1e-8, atol=1e-10)
    x["muc"] = np.where(khop.any(axis=1), g.muc.to_numpy()[khop.argmax(axis=1)], np.nan)
    if mo_hinh.ten == "tb_theo_muc" and x.muc.isna().any():
        raise ValueError("tb_theo_muc cần cột muc xác định từ đúng mức dòng đã đo; "
                         "không gán được muc cho J mới. Hãy chọn --J hoặc --dong-mA trùng mức đã đo.")
    _canh_bao_noi_suy(mo_hinh, not khop.any(axis=1).all())
    y = mo_hinh.predict(x)
    bang = x[["ratio", "J"]].copy()
    for cot, gia_tri in vat_ly.suy_ra(j, y.Csp_hat, y.Vbar_hat).items():
        bang[cot] = gia_tri
    bang["khong_hop_le"] = khong_hop_le(y.Csp_hat) | khong_hop_le(y.Vbar_hat)
    for cot, sd_cot in (("Csp", "logCsp_sd"), ("Vbar", "logVbar_sd")):
        sd = y[sd_cot].to_numpy(float)
        if np.any(np.isfinite(sd) & (sd >= 0)):
            bang[sd_cot] = sd
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = khoang_95(np.log(bang[cot].to_numpy()), sd)
            bang[f"{cot}_L95"], bang[f"{cot}_U95"] = lo, hi
    for bien in ("L95", "U95"):
        if "Csp_" + bien in bang:
            bang["Ed_" + bien] = vat_ly.ed_tu_csp(bang["Csp_" + bien])
        if "Vbar_" + bien in bang:
            bang["Pd_" + bien] = vat_ly.pd_tu_vbar(j, bang["Vbar_" + bien])
    bang.attrs["ghi_chu"] = GHI_CHU
    return bang


def de_xuat_muc_do(mo_hinh, bang, ti_le, dong_mA):
    """Chọn argmax σ_logCsp trong các dòng ứng viên chưa đo; không thay bằng heuristic."""
    g = _dien_cuc(bang, ti_le)
    # 10 mA = 0,010 A; khối lượng suy ra tính bằng g.
    khoi_luong = 0.010 / float(g.J.iloc[0])
    dong = np.unique(_dong_duong(dong_mA))
    da_do = g.J.to_numpy(float) * khoi_luong * 1000
    dong = dong[~np.isclose(dong[:, None], da_do[None, :], rtol=1e-7, atol=1e-7).any(axis=1)]
    if not mo_hinh.ho_tro_khoang:
        return {"ly_do": "Cần mô hình có độ bất định (ví dụ gpr) để đề xuất theo σ_logCsp; mô hình này không có sd."}
    if not len(dong):
        return {"ly_do": "Không còn mức dòng ứng viên chưa đo."}
    _canh_bao_noi_suy(mo_hinh, True)
    j = dong / (1000 * khoi_luong)
    x = pd.DataFrame({"ratio": float(g.ratio.iloc[0]), "J": j})
    sd = mo_hinh.predict(x)["logCsp_sd"].to_numpy(float)
    hop_le = np.isfinite(sd) & (sd >= 0)
    if not hop_le.any():
        return {"ly_do": "Mô hình không trả sd logCsp hợp lệ cho các mức ứng viên."}
    chon = int(np.argmax(np.where(hop_le, sd, -np.inf)))
    return dict(dong_mA=float(dong[chon]), J=float(j[chon]), logCsp_sd=float(sd[chon]),
                khoi_luong_g=khoi_luong, ly_do="Mức chưa đo có σ_logCsp lớn nhất.")


def ve_ragone(bang_du_doan, bang_that, ti_le, out):
    """Vẽ biên Ed/Pd riêng, không xem hình chữ nhật là khoảng đồng thời 95%."""
    g = _dien_cuc(bang_that, ti_le)
    b = bang_du_doan.sort_values("Pd")
    fig, ax = plt.subplots(figsize=(7, 5))
    hop_le = np.isfinite(b.Ed) & np.isfinite(b.Pd) & (b.Ed > 0) & (b.Pd > 0)
    b = b.loc[hop_le]
    ax.plot(b.Pd, b.Ed, label="Dự đoán (khớp toàn bộ)")
    if {"Ed_L95", "Ed_U95"}.issubset(b.columns):
        ax.fill_between(b.Pd.to_numpy(), b.Ed_L95.to_numpy(), b.Ed_U95.to_numpy(),
                        alpha=0.22, label="Biên Ed 95% tại Pd trung vị")
    if {"Pd_L95", "Pd_U95"}.issubset(b.columns):
        ax.fill_betweenx(b.Ed.to_numpy(), b.Pd_L95.to_numpy(), b.Pd_U95.to_numpy(),
                         alpha=0.12, label="Biên Pd 95% tại Ed trung vị")
    ax.scatter(g.Pd, g.Ed, marker="x", color="black", label="Số đo thật")
    ax.set(xscale="log", yscale="log", xlabel="Pd (W/kg)", ylabel="Ed (Wh/kg)",
           title=f"Ragone — tỉ lệ {ti_le:g}")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)
    fig.text(0.5, 0.01, GHI_CHU, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    duong_dan = Path(out)
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(duong_dan, dpi=160)
    plt.close(fig)
    return duong_dan


def main(argv=None, *, mo_hinh=None, tt_best=None):
    """CLI hoặc xuất dự đoán bằng mô hình đã nạp để dùng lại qua nhiều điện cực."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ti-le", type=float, required=True)
    dong = parser.add_mutually_exclusive_group()
    dong.add_argument("--J", type=float, nargs="+")
    dong.add_argument("--dong-mA", type=float, nargs="+")
    parser.add_argument("--mo-hinh", choices=(*TEN_NEN, *TEN_MO_HINH, *TEN_DO_THI), default="gpr")
    parser.add_argument("--cach", choices=("A", "B", "C", "D"), default="A",
                        help="Mặc định A (chỉ số đo thật); chọn cách theo kết quả Bảng 5.3.")
    parser.add_argument("--n-ao", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tham-so", help="Đối tượng JSON chứa tham số mô hình")
    parser.add_argument("--best", type=Path,
                        help="Nạp mô hình đã lưu bởi ml.luu_best (best_*.joblib); "
                             "bỏ qua --mo-hinh/--cach/--n-ao/--seed/--tham-so")
    parser.add_argument("--out", type=Path, default=Path("ket_qua/du_doan/du_doan.xlsx"))
    args = parser.parse_args(argv)
    if args.out.suffix.lower() not in (".xlsx", ".csv"):
        parser.error("--out phải có đuôi .xlsx hoặc .csv")
    if args.n_ao < 0:
        parser.error("--n-ao phải không âm")
    try:
        p = None if args.tham_so is None else json.loads(args.tham_so)
        if p is not None and not isinstance(p, dict):
            raise ValueError("--tham-so phải là đối tượng JSON")
        that = doc_so_lieu()
        g = _dien_cuc(that, args.ti_le)
        khoi_luong = 0.010 / float(g.J.iloc[0])
        if args.best is not None and mo_hinh is None:
            from .luu_best import tai_best
            mo_hinh, tt_best = tai_best(args.best)
        if tt_best is not None:
            print(f"Mô hình best ({tt_best['muc_tieu']}): {tt_best['cau_hinh']}")
        ten_mo_hinh = mo_hinh.ten if mo_hinh is not None else args.mo_hinh
        if args.J is not None:
            j = _dong_duong(args.J)
        elif args.dong_mA is not None:
            j = _dong_duong(args.dong_mA) / (1000 * khoi_luong)
        elif ten_mo_hinh == "tb_theo_muc":
            j = g.J.to_numpy(float)
        else:
            j = np.geomspace(g.J.min(), g.J.max(), 100)
        if mo_hinh is None:
            mo_hinh = khop_toan_bo(args.mo_hinh, args.cach, args.n_ao, args.seed, p)
        bang = du_doan_luoi(mo_hinh, args.ti_le, j)
        de_xuat = de_xuat_muc_do(mo_hinh, that, args.ti_le, np.arange(10, 201, 10))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    bang["dong_mA"] = bang.J * khoi_luong * 1000
    bang["ghi_chu"] = GHI_CHU
    print(GHI_CHU)
    so_loi = int(bang.khong_hop_le.sum())
    if so_loi:
        print(f"Cảnh báo: {so_loi}/{len(bang)} dòng dự đoán có Csp/Vbar không hữu hạn hoặc ≤ 0; "
              "đã đánh dấu khong_hop_le, giữ nguyên giá trị.")
    print("Đơn vị: J A/g; Csp F/g; Vbar V; Ed Wh/kg; Pd W/kg; dt s. Khoảng 95% là biên từng đại lượng, không suy khoảng Δt.")
    print(bang.drop(columns="ghi_chu").to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print("Đề xuất mức đo:", de_xuat["ly_do"])
    if "dong_mA" in de_xuat:
        print(f"  I = {de_xuat['dong_mA']:g} mA; J = {de_xuat['J']:.6g} A/g; σ_logCsp = {de_xuat['logCsp_sd']:.6g}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix.lower() == ".xlsx":
        thong_tin = {**mo_hinh.thong_tin_khop_, "ti_le": args.ti_le,
                     "khoi_luong_g": khoi_luong,
                     "don_vi": "J A/g; Csp F/g; Vbar V; Ed Wh/kg; Pd W/kg; dt s",
                     "khoang_95": "Biên riêng Csp, Vbar, Ed, Pd; không phải khoảng đồng thời, không suy khoảng dt."}
        with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
            bang.to_excel(writer, sheet_name="du_doan", index=False)
            pd.DataFrame([{"muc": k, "gia_tri": json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else v}
                          for k, v in thong_tin.items()]).to_excel(writer, sheet_name="thong_tin", index=False)
            pd.DataFrame([de_xuat]).to_excel(writer, sheet_name="de_xuat", index=False)
    else:
        bang.to_csv(args.out, index=False, encoding="utf-8-sig")
    hinh = ve_ragone(bang, that, args.ti_le, args.out.with_suffix(".png"))
    print(f"Đã ghi: {args.out}\nHình: {hinh}")


if __name__ == "__main__":
    main()
