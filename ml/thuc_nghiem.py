"""Chọn tham số lồng nhau và chạy từng đơn vị; không dùng số đo kiểm tra để học."""
from __future__ import annotations

from dataclasses import replace
import json
from time import perf_counter

import numpy as np

from . import danh_gia, du_lieu, mo_hinh_ml, mo_hinh_nen, tang_cuong, vat_ly
from .gng_r import LUOI_DO_THI, MoHinhDoThi
from .cau_hinh import ma_cau_hinh_chon

COT_KET_QUA = "nhom kb ma_lan_chia mo_hinh cach n_ao sigma_lap seed kieu_ma_hoa k chinh_sach ratio muc J Csp Csp_hat Vbar Vbar_hat Ed Ed_hat Pd Pd_hat logCsp_sd logVbar_sd Csp_L95 Csp_U95 khong_hop_le tham_so cau_hinh_chon so_hang_hoc t_sinh_s t_hoc_s t_du_doan_s".split()
MO_HINH_NEN = ("ham_mu", "chon_dang_ham", "tb_hinh_hoc", "tb_theo_muc")
TEN_MO_HINH = (*MO_HINH_NEN, *mo_hinh_ml.TEN_MO_HINH, *LUOI_DO_THI)


def tao_mo_hinh_theo_ten(ten, tham_so=None, kieu_ma_hoa="onehot", lam=1.0, seed=0):
    """Điều phối tới một giao diện fit/predict thống nhất."""
    if ten in ("ham_mu", "chon_dang_ham"):
        return mo_hinh_nen.MoHinhDangHam("exp" if ten == "ham_mu" else "auto")
    if ten == "tb_hinh_hoc":
        return mo_hinh_nen.MoHinhTrungBinhHinhHoc()
    if ten == "tb_theo_muc":
        return mo_hinh_nen.MoHinhTrungBinhTheoMuc()
    if ten in mo_hinh_ml.TEN_MO_HINH:
        return mo_hinh_ml.tao_mo_hinh(ten, tham_so, kieu_ma_hoa=kieu_ma_hoa, lam=lam, seed=seed)
    if ten in LUOI_DO_THI:
        if kieu_ma_hoa != "onehot":
            raise ValueError("Mô hình đồ thị hiện chỉ hỗ trợ onehot; không áp dụng KB4")
        return MoHinhDoThi(ten, tham_so, lam=lam, seed=seed)
    raise ValueError(f"Tên mô hình không hợp lệ: {ten}")


def luoi_cua(ten, rut_gon=False):
    """Trả bản sao lưới để bên gọi không thay đổi cấu hình chung."""
    if ten in MO_HINH_NEN:
        luoi = [{}]
    elif ten in mo_hinh_ml.LUOI:
        luoi = mo_hinh_ml.LUOI[ten]
    elif ten in LUOI_DO_THI:
        luoi = LUOI_DO_THI[ten]
    else:
        raise ValueError(f"Không có lưới cho {ten}")
    return [dict(p) for p in (luoi[:2] if rut_gon else luoi)]


def toi_da_hang(ten):
    return mo_hinh_ml.TOI_DA_HANG.get(ten)


def _tap_hoc(that, cau_hinh, ten, seed, gioi_han_hang=None):
    hoc = tang_cuong.tao_tap_hoc(that, cau_hinh, seed)
    gioi_han = toi_da_hang(ten)
    if gioi_han_hang is not None:
        gioi_han = min(gioi_han, gioi_han_hang) if gioi_han else gioi_han_hang
    return tang_cuong.lay_mau_con(hoc, gioi_han, seed) if gioi_han else hoc


def chon_tham_so(bang, lan_chia, ten, cau_hinh_tc, n_ao_chon=10, rut_gon=False, gioi_han_hang=None, ngan_sach_chon=None):
    """Sinh lại dữ liệu chỉ từ phần học trong; chọn bằng RMSE-log hai đầu ra."""
    luoi = luoi_cua(ten, rut_gon)
    if len(luoi) == 1:
        return luoi[0], float("nan")
    folds = du_lieu.chia_trong(bang, lan_chia)
    if not folds:
        raise ValueError(f"Không có fold nội bộ cho {lan_chia.ma}")
    cfg = replace(cau_hinh_tc, n_ao=min(cau_hinh_tc.n_ao, n_ao_chon))
    # Các tổ hợp dùng cùng tập sinh seed=0 trong mỗi fold, không khớp lại bộ sinh
    # theo tổ hợp và tuyệt đối không sinh từ phần kiểm định trong.
    du_lieu_fold = [(f, _tap_hoc(bang.loc[f.hoc], cfg, ten, 0, gioi_han_hang)) for f in folds]
    diem = []
    for p in luoi:
        loi = []
        for f, hoc in du_lieu_fold:
            p_trong = {**p, **(ngan_sach_chon or {})}
            mh = tao_mo_hinh_theo_ten(ten, p_trong, f.thong_tin.get("kieu_ma_hoa", "onehot"), 1.0, 0)
            mh.fit(hoc, bang.loc[f.hoc])
            that = bang.loc[f.kiem_tra]
            dd = mh.predict(that)
            if danh_gia.khong_hop_le(dd[["Csp_hat", "Vbar_hat"]]).any():
                loi.append(float("inf"))
            else:
                loi.append(np.mean([danh_gia.rmse_log(that[c], dd[c + "_hat"]) for c in ("Csp", "Vbar")]))
        diem.append(float(np.mean(loi)))
    i = int(np.argmin(diem))
    return luoi[i], diem[i]


def chay_don_vi(bang, lan_chia, ten, cau_hinh_tc, seeds, n_ao_chon=10, rut_gon=False, nhom="chinh", gioi_han_hang=None, ngan_sach_chon=None):
    """Một lần chia/cấu hình, chọn tham số một lần rồi chạy mọi seed được giao."""
    if (lan_chia.kb == "KB4" and ten in ("ham_mu", "chon_dang_ham")) or (lan_chia.kb != "KB4" and ten.startswith("tb_")):
        return []
    p, _ = chon_tham_so(bang, lan_chia, ten, cau_hinh_tc, n_ao_chon, rut_gon, gioi_han_hang, ngan_sach_chon)
    that_hoc, that_kt = bang.loc[lan_chia.hoc], bang.loc[lan_chia.kiem_tra]
    kieu = lan_chia.thong_tin.get("kieu_ma_hoa", "onehot")
    ket_qua = []
    ma_chon = ma_cau_hinh_chon(rut_gon, n_ao_chon, gioi_han_hang, ngan_sach_chon)
    for seed in seeds:
        t = perf_counter()
        hoc = _tap_hoc(that_hoc, cau_hinh_tc, ten, seed, gioi_han_hang)
        t_sinh = perf_counter() - t
        mh = tao_mo_hinh_theo_ten(ten, p, kieu, 1.0, seed)
        t = perf_counter()
        mh.fit(hoc, that_hoc)
        t_hoc = perf_counter() - t
        t = perf_counter()
        dd = mh.predict(that_kt)
        t_dd = perf_counter() - t
        ed = vat_ly.ed_tu_csp(dd.Csp_hat)
        pd = vat_ly.pd_tu_vbar(that_kt.J, dd.Vbar_hat)
        with np.errstate(divide="ignore", invalid="ignore"):
            l, u = danh_gia.khoang_95(np.log(dd.Csp_hat), dd.logCsp_sd)
        loi = danh_gia.khong_hop_le(dd[["Csp_hat", "Vbar_hat"]]).any(axis=1)
        for i, (_, r) in enumerate(that_kt.iterrows()):
            ket_qua.append(dict(nhom=nhom, kb=lan_chia.kb, ma_lan_chia=lan_chia.ma,
                mo_hinh=ten, cach=cau_hinh_tc.cach, n_ao=cau_hinh_tc.n_ao,
                sigma_lap=cau_hinh_tc.sigma_lap, seed=int(seed), kieu_ma_hoa=kieu,
                k=lan_chia.thong_tin.get("k"), chinh_sach=lan_chia.thong_tin.get("chinh_sach"),
                ratio=r.ratio, muc=int(r.muc), J=r.J, Csp=r.Csp, Csp_hat=dd.Csp_hat.iloc[i],
                Vbar=r.Vbar, Vbar_hat=dd.Vbar_hat.iloc[i], Ed=r.Ed, Ed_hat=ed[i],
                Pd=r.Pd, Pd_hat=pd[i], logCsp_sd=dd.logCsp_sd.iloc[i], logVbar_sd=dd.logVbar_sd.iloc[i],
                Csp_L95=l[i], Csp_U95=u[i], khong_hop_le=bool(loi[i]),
                tham_so=json.dumps(p, ensure_ascii=False, sort_keys=True),
                cau_hinh_chon=ma_chon,
                so_hang_hoc=int(getattr(mh, "so_hang_hoc_", len(hoc))),
                t_sinh_s=t_sinh, t_hoc_s=t_hoc, t_du_doan_s=t_dd))
    return ket_qua
