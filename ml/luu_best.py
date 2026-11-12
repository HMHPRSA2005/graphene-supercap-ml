"""Chọn cấu hình tốt nhất từ kết quả KB1/KB2 rồi lưu mô hình ``best_*.joblib``.

Quy trình (chạy được sau MỖI phiên Colab trong Giai đoạn 1; file best tự cập nhật):
  1. Đọc ``<thu_muc>/ket_qua_kb1_kb4.csv`` (sai số trên điểm thật bị giấu).
  2. Gom theo cấu hình (mô hình, cách, n_ao, sigma_lap, cau_hinh_chon); chỉ xét
     cấu hình đã chạy ĐỦ mọi lần chia của KB đó (tránh xếp hạng trên kết quả dở dang).
     Bỏ đối chứng ``_bo_0.075``, mô hình đồ thị chỉ thuộc phụ lục và dự đoán không hợp lệ.
  3. Điểm = trung bình (MAPE_Csp + MAPE_Pd)/2, MAPE tính theo seed rồi lấy trung bình seed.
  4. Tham số = tổ hợp được vòng chọn lồng nhau chọn NHIỀU NHẤT qua các lần chia.
  5. Khớp lại cấu hình đó trên toàn bộ 48 số đo (khớp để sử dụng, không phải kết quả
     kiểm tra) và lưu ``best/best_<muc_tieu>.joblib`` + ``.json`` mô tả.

Mục tiêu: KB1 (nội suy), KB2 (ngoại suy), chung (trung bình KB1, KB2),
co_khoang (tốt nhất trong mô hình có độ bất định, dùng cho dải 95% và đề xuất đo).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from . import du_lieu
from .tang_cuong import CauHinhTangCuong

GHI_CHU = "khớp lại toàn bộ 48 số đo để sử dụng; độ tin cậy xem MAPE kiểm tra trong .json"
KHOA = ["mo_hinh", "cach", "n_ao", "sigma_lap", "kieu_ma_hoa", "cau_hinh_chon"]
MUC_TIEU = {"KB1": ("KB1",), "KB2": ("KB2",), "chung": ("KB1", "KB2"), "co_khoang": ("KB1", "KB2")}
CO_KHOANG = ("gpr",)
THU_VIEN = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")


def _phien_ban():
    return {"python": platform.python_version(),
            **{t: importlib.metadata.version(t) for t in THU_VIEN}}


def _ma_day_du(bang):
    return {"KB1": {f.ma for f in du_lieu.chia_kb1(bang)},
            "KB2": {f.ma for f in du_lieu.chia_kb2(bang)}}


def bang_diem(kq: pd.DataFrame, bang_that: pd.DataFrame, cho_phep_thieu=False) -> pd.DataFrame:
    """Một hàng/(cấu hình, KB); loại gng_r/pgcn_r vì chỉ thuộc phụ lục."""
    kq = kq.loc[kq.kb.isin(["KB1", "KB2"])
                & ~kq.ma_lan_chia.str.contains("_bo_", regex=False)
                & ~kq.mo_hinh.isin(["gng_r", "pgcn_r"])].copy()
    if kq.empty:
        return pd.DataFrame()
    kq["ape_Csp"] = (kq.Csp_hat - kq.Csp).abs() / kq.Csp * 100
    kq["ape_Pd"] = (kq.Pd_hat - kq.Pd).abs() / kq.Pd * 100
    day_du = _ma_day_du(bang_that)
    hang = []
    for (*khoa, kb), g in kq.groupby([*KHOA, "kb"], dropna=False):
        ma_co = set(g.ma_lan_chia)
        du = day_du[kb] <= ma_co
        if not du and not cho_phep_thieu:
            continue
        loi = bool(g.khong_hop_le.astype(str).str.lower().eq("true").any()
                   or not np.isfinite(g[["ape_Csp", "ape_Pd"]].to_numpy()).all())
        theo_seed = g.groupby("seed")[["ape_Csp", "ape_Pd"]].mean()
        # Tham số do vòng chọn lồng nhau quyết định ở từng lần chia; lấy tổ hợp phổ biến nhất.
        dem = Counter(g.drop_duplicates("ma_lan_chia").tham_so)
        tham_so, so_lan = dem.most_common(1)[0]
        hang.append(dict(zip(KHOA, khoa), kb=kb, MAPE_Csp=theo_seed.ape_Csp.mean(),
                         MAPE_Csp_sd=theo_seed.ape_Csp.std(ddof=0), MAPE_Pd=theo_seed.ape_Pd.mean(),
                         MAPE_Pd_sd=theo_seed.ape_Pd.std(ddof=0), so_seed=len(theo_seed),
                         so_lan_chia=len(ma_co), du_lan_chia=du, khong_hop_le=loi,
                         tham_so=tham_so, ti_le_chon_tham_so=so_lan / sum(dem.values())))
    d = pd.DataFrame(hang)
    if not d.empty:
        d["diem"] = (d.MAPE_Csp + d.MAPE_Pd) / 2
    return d


def chon_best(diem: pd.DataFrame, muc_tieu: str):
    """Trả (hàng tốt nhất, bảng xếp hạng) cho một mục tiêu; None nếu chưa có ứng viên."""
    if diem.empty:
        return None, pd.DataFrame()
    cac_kb = MUC_TIEU[muc_tieu]
    d = diem.loc[diem.kb.isin(cac_kb) & ~diem.khong_hop_le]
    if muc_tieu == "co_khoang":
        d = d.loc[d.mo_hinh.isin(CO_KHOANG)]
    if d.empty:
        return None, pd.DataFrame()
    rong = d.pivot_table(index=KHOA, columns="kb", values=["diem", "MAPE_Csp", "MAPE_Pd"], aggfunc="first")
    rong.columns = [f"{a}_{b}" for a, b in rong.columns]
    rong = rong.dropna(subset=[f"diem_{kb}" for kb in cac_kb]).reset_index()
    if rong.empty:
        return None, rong
    rong["diem"] = rong[[f"diem_{kb}" for kb in cac_kb]].mean(axis=1)
    rong = rong.sort_values("diem", kind="stable").reset_index(drop=True)
    tot = rong.iloc[0].to_dict()
    # Tham số: lấy từ KB đầu tiên của mục tiêu (KB1 cho chung) vì tham số chọn theo từng KB.
    cung = np.logical_and.reduce([d[k] == tot[k] for k in KHOA])
    g = d.loc[cung].set_index("kb")
    tot["tham_so"] = g.loc[cac_kb[0], "tham_so"]
    tot["ti_le_chon_tham_so"] = float(g.loc[cac_kb[0], "ti_le_chon_tham_so"])
    tot["so_seed"] = int(g.so_seed.min())
    tot["chi_tiet_kb"] = {kb: {c: float(g.loc[kb, c]) for c in ("MAPE_Csp", "MAPE_Csp_sd", "MAPE_Pd", "MAPE_Pd_sd", "diem")}
                          for kb in cac_kb}
    return tot, rong


def khop_tu_cau_hinh(cau_hinh: dict, seed: int = 0):
    """Khớp lại trên toàn bộ số đo thật đúng như một đơn vị thực nghiệm (cùng giới hạn hàng)."""
    from .thuc_nghiem import _tap_hoc, tao_mo_hinh_theo_ten
    bang = du_lieu.doc_so_lieu()
    ten = cau_hinh["mo_hinh"]
    cfg = CauHinhTangCuong(cach=cau_hinh["cach"], n_ao=int(cau_hinh["n_ao"]),
                           sigma_lap=float(cau_hinh["sigma_lap"]))
    chon = json.loads(cau_hinh.get("cau_hinh_chon") or "{}")
    p = json.loads(cau_hinh.get("tham_so") or "{}")
    hoc = _tap_hoc(bang, cfg, ten, seed, chon.get("gioi_han_hang"))
    mo_hinh = tao_mo_hinh_theo_ten(ten, p, cau_hinh.get("kieu_ma_hoa", "onehot"), 1.0, seed)
    mo_hinh.fit(hoc, bang)
    mo_hinh.thong_tin_khop_ = dict(ghi_chu=GHI_CHU, mo_hinh=ten, cach=cfg.cach, n_ao=cfg.n_ao,
                                  sigma_lap=cfg.sigma_lap, seed=seed, tham_so=p,
                                  so_diem_that=len(bang), so_hang_sinh=len(hoc))
    return mo_hinh


def tai_best(duong_dan):
    """So phiên bản JSON trước khi nạp; lệch NumPy/sklearn thì khớp lại một lần."""
    duong_dan = Path(duong_dan)
    tep_json = duong_dan.with_suffix(".json")
    thong_tin = json.loads(tep_json.read_text(encoding="utf-8")) if tep_json.exists() else None
    if thong_tin is not None:
        pb = thong_tin.get("phien_ban", {})
        lech = [f"{ten}: đã lưu={pb.get(ten)}, local={importlib.metadata.version(ten)}"
                for ten in ("numpy", "scikit-learn")
                if pb.get(ten) != importlib.metadata.version(ten)]
        if lech:
            print(f"Không nạp pickle {duong_dan.name} vì lệch phiên bản ({'; '.join(lech)}); "
                  f"khớp lại một lần từ {tep_json.name} để sử dụng, không phải kết quả kiểm tra.")
            return khop_tu_cau_hinh(thong_tin["cau_hinh"], thong_tin["seed_khop"]), thong_tin
    try:
        goi = joblib.load(duong_dan)
        return goi["mo_hinh"], goi["thong_tin"]
    except Exception as exc:
        if thong_tin is None:
            raise
        print(f"Không nạp được {duong_dan.name} ({type(exc).__name__}); khớp lại một lần từ {tep_json.name}.")
        return khop_tu_cau_hinh(thong_tin["cau_hinh"], thong_tin["seed_khop"]), thong_tin


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thu-muc", default="ket_qua", help="Thư mục chứa ket_qua_kb1_kb4.csv")
    parser.add_argument("--muc-tieu", nargs="+", choices=tuple(MUC_TIEU), default=list(MUC_TIEU))
    parser.add_argument("--seed", type=int, default=0, help="Seed khi khớp lại toàn bộ")
    parser.add_argument("--cho-phep-thieu", action="store_true",
                        help="Cho xếp hạng cả cấu hình chưa đủ lần chia (chỉ để thử với nhóm thu)")
    args = parser.parse_args(argv)
    thu_muc = Path(args.thu_muc)
    nguon = thu_muc / "ket_qua_kb1_kb4.csv"
    if not nguon.exists():
        parser.error(f"Chưa có {nguon}; chạy ml.chay_kb1_kb4 trước.")
    kq = pd.read_csv(nguon)
    bang = du_lieu.doc_so_lieu()
    diem = bang_diem(kq, bang, args.cho_phep_thieu)
    ra = thu_muc / "best"
    ra.mkdir(parents=True, exist_ok=True)
    if diem.empty:
        print("Chưa có cấu hình KB1/KB2 nào chạy đủ lần chia. Chạy xong nhóm `chinh` rồi thử lại "
              "(hoặc thêm --cho-phep-thieu khi chỉ thử với nhóm thu).")
        return
    diem.sort_values(["kb", "diem"]).to_csv(ra / "diem_cau_hinh.csv", index=False, encoding="utf-8-sig")
    for muc_tieu in args.muc_tieu:
        tot, rong = chon_best(diem, muc_tieu)
        if tot is None:
            print(f"[{muc_tieu}] chưa có ứng viên hợp lệ, bỏ qua.")
            continue
        rong.to_csv(ra / f"xep_hang_{muc_tieu}.csv", index=False, encoding="utf-8-sig")
        cau_hinh = {k: (v.item() if hasattr(v, "item") else v) for k, v in tot.items()
                    if k in (*KHOA, "tham_so")}
        mo_hinh = khop_tu_cau_hinh(cau_hinh, args.seed)
        thong_tin = dict(muc_tieu=muc_tieu, kb_danh_gia=list(MUC_TIEU[muc_tieu]), ghi_chu=GHI_CHU,
                         cau_hinh=cau_hinh, seed_khop=args.seed, diem=float(tot["diem"]),
                         chi_tiet_kb=tot["chi_tiet_kb"], so_seed=tot["so_seed"],
                         ti_le_chon_tham_so=tot["ti_le_chon_tham_so"],
                         so_ung_vien=len(rong), so_hang_hoc=int(mo_hinh.thong_tin_khop_["so_hang_sinh"]),
                         ho_tro_khoang=bool(mo_hinh.ho_tro_khoang), phien_ban=_phien_ban(),
                         thoi_diem=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         nguon=str(nguon))
        joblib.dump({"mo_hinh": mo_hinh, "thong_tin": thong_tin}, ra / f"best_{muc_tieu}.joblib", compress=3)
        (ra / f"best_{muc_tieu}.json").write_text(json.dumps(thong_tin, ensure_ascii=False, indent=2), encoding="utf-8")
        ct = " | ".join(f"{kb}: MAPE Csp {v['MAPE_Csp']:.2f}%, Pd {v['MAPE_Pd']:.2f}%"
                        for kb, v in thong_tin["chi_tiet_kb"].items())
        print(f"[{muc_tieu}] {cau_hinh['mo_hinh']} cách {cau_hinh['cach']} n_ao={cau_hinh['n_ao']} "
              f"σ={cau_hinh['sigma_lap']:g} tham số {cau_hinh['tham_so']} → {ct} "
              f"({len(rong)} ứng viên) → {ra / f'best_{muc_tieu}.joblib'}")
    print(GHI_CHU)


if __name__ == "__main__":
    main()
