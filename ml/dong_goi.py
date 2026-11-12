"""Kiểm kê Giai đoạn 1 và đóng gói kết quả, không coi kết quả thiếu là đầy đủ."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import zipfile
import zlib

from .cau_hinh import khoa_don_vi, liet_ke_don_vi
from .chay_kb1_kb4 import diem_kiem_tra, doc_tien_do, ghi_json_nguyen_tu
from .du_lieu import doc_so_lieu

NHOM_CHINH = ("chinh", "du_lieu_ao", "so_mau", "sigma", "kb3", "kb4")
THU_VIEN = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "openpyxl")


def phien_ban():
    kq = {"python": platform.python_version()}
    for ten in THU_VIEN:
        try:
            kq[ten] = importlib.metadata.version(ten)
        except importlib.metadata.PackageNotFoundError:
            kq[ten] = None
    return kq


def sha256(tep):
    bam = hashlib.sha256()
    with Path(tep).open("rb") as f:
        for khoi in iter(lambda: f.read(1024 * 1024), b""):
            bam.update(khoi)
    return bam.hexdigest()


def xlsx_nguyen_ven(tep):
    """Chỉ chấp nhận XLSX là ZIP đọc được và không lỗi CRC."""
    try:
        if not zipfile.is_zipfile(tep):
            return False
        with zipfile.ZipFile(tep) as z:
            return z.testzip() is None
    except (OSError, zipfile.BadZipFile, RuntimeError, zlib.error):
        return False


def tien_do(thu_muc):
    bang = doc_so_lieu()
    _, da_co = doc_tien_do(Path(thu_muc) / "ket_qua_kb1_kb4.csv")
    ket_qua = {}
    for nhom in (*NHOM_CHINH, "thu"):
        viec = liet_ke_don_vi(bang, nhom)
        diem = diem_kiem_tra(bang, viec)
        du = sum(all(da_co.get((*khoa_don_vi(d), s)) == diem[khoa_don_vi(d)]
                     for s in d["seeds"]) for d in viec)
        ket_qua[nhom] = dict(da_du=du, can_co=len(viec))
    return ket_qua


def in_tien_do(tien_do_nhom, thieu=()):
    print("Đơn vị đủ toàn bộ seed / đơn vị cần có (nhóm có đơn vị trùng nhau):")
    for nhom, g in tien_do_nhom.items():
        print(f"  {nhom}: {g['da_du']}/{g['can_co']}")
    if thieu:
        print("CẢNH BÁO: sản phẩm chưa đầy đủ:")
        for muc in thieu:
            print("  - " + muc)
    else:
        print("Đã có đầy đủ các mục được kiểm kê.")


def muc_thieu(thu_muc, tien_do_nhom):
    r = Path(thu_muc)
    thieu = []
    for nhom in NHOM_CHINH:
        g = tien_do_nhom[nhom]
        if g["da_du"] != g["can_co"]:
            thieu.append(f"{nhom} chưa đủ; chạy python -X utf8 -m ml.chay_kb1_kb4 --nhom {nhom} --thu-muc \"{r}\" --n-jobs -1")
    yeu_cau = [(f"best/best_{m}.{d}", "ml.luu_best")
               for m in ("KB1", "KB2", "chung", "co_khoang") for d in ("joblib", "json")]
    yeu_cau += [("ket_qua_kb1_kb4.csv", "ml.chay_kb1_kb4 --nhom chinh"),
                ("moi_truong.json", "ml.chay_kb1_kb4 --nhom chinh"),
                ("bang_ch5/*.tex", "ml.ve_hinh_ch5 --ung-dung"),
                ("bang_ch5/*.csv", "ml.ve_hinh_ch5 --ung-dung"),
                ("hinh_ch5/*.pdf", "ml.ve_hinh_ch5 --ung-dung"),
                ("hinh_ch5/*.png", "ml.ve_hinh_ch5 --ung-dung"),
                ("so_sanh/*.pdf", "ml.so_sanh"),
                ("so_sanh/*.csv", "ml.so_sanh")]
    for mau, cli in yeu_cau:
        if not any(p.is_file() and p.stat().st_size for p in r.glob(mau)):
            thieu.append(f"Thiếu {mau}; chạy python -X utf8 -m {cli} --thu-muc \"{r}\"")
    gcd = [("gcdsim/du_lieu_ao_480000.xlsx", f'gcdsim.gcdsim data/so_lieu_GCD.xlsx --out "{r}/gcdsim/du_lieu_ao_480000.xlsx" --seed 42'),
           ("gcdsim/hinh/*", f'gcdsim.ve_hinh data/so_lieu_GCD.xlsx "{r}/gcdsim/hinh"'),
           ("gcdsim/do_nhay.csv", f'gcdsim.do_nhay data/so_lieu_GCD.xlsx "{r}/gcdsim/do_nhay.csv"')]
    for mau, cli in gcd:
        if not any(p.is_file() and p.stat().st_size for p in r.glob(mau)):
            thieu.append(f"Thiếu {mau}; chạy python -X utf8 -m {cli}")
    xlsx = r / "gcdsim/du_lieu_ao_480000.xlsx"
    if xlsx.exists() and not xlsx_nguyen_ven(xlsx):
        thieu.append("XLSX GCD-Sim hỏng/ghi dở: gcdsim/du_lieu_ao_480000.xlsx; chạy lại ô «GCD-Sim — N=10000 mỗi điện cực».")
    csv = r / "ket_qua_kb1_kb4.csv"
    if csv.is_file():
        cac_o = {"best": "Lưu best sau phiên / best cuối",
                 "bang_ch5": "Cuối Phiên 3 — so sánh và sinh tư liệu Chương 5",
                 "hinh_ch5": "Cuối Phiên 3 — so sánh và sinh tư liệu Chương 5",
                 "so_sanh": "Cuối Phiên 3 — so sánh và sinh tư liệu Chương 5"}
        for thu, o in cac_o.items():
            cu = [p.relative_to(r).as_posix() for p in sorted((r / thu).glob("*"))
                  if p.is_file() and p.stat().st_mtime < csv.stat().st_mtime]
            if cu:
                thieu.append(f"Sản phẩm cũ hơn ket_qua_kb1_kb4.csv: {', '.join(cu)}; chạy lại ô «{o}» rồi đóng gói lại.")
    return thieu


def dong_goi(thu_muc, out, cho_phep_thieu=False):
    r, out = Path(thu_muc).resolve(), Path(out).resolve()
    if not r.is_dir():
        raise ValueError(f"Chưa có thư mục kết quả {r}.")
    td = tien_do(r)
    thieu = muc_thieu(r, td)
    in_tien_do(td, thieu)
    if thieu and not cho_phep_thieu:
        raise ValueError("Từ chối đóng gói thiếu. Chạy các lệnh trên; chỉ dùng --cho-phep-thieu khi chạy thử local.")
    tep = []
    for p in sorted(r.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Không đóng gói liên kết tượng trưng: {p}")
        if p.is_file() and p != out and p != r / "manifest.json":
            tep.append(p)
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    manifest = dict(phien_ban_manifest=1, thoi_diem=datetime.now(timezone.utc).isoformat(),
                    git_commit=git, phien_ban=phien_ban(), tien_do=td, thieu=thieu,
                    tep=[dict(ten=p.relative_to(r).as_posix(), kich_thuoc=p.stat().st_size,
                              sha256=sha256(p)) for p in tep])
    ghi_json_nguyen_tu(r / "manifest.json", manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in tep:
            z.write(p, p.relative_to(r).as_posix())
        z.write(r / "manifest.json", "manifest.json")
    print(f"Đã ghi {out} ({out.stat().st_size:,} byte).")
    print('Tiếp theo trên máy local: python -X utf8 -m ml.giai_doan_2 tat_ca "' + str(out) + '" --thu-muc ket_qua --bao-cao ../Report/Report')
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--thu-muc", default="ket_qua")
    p.add_argument("--out", default="san_pham_giai_doan_1.zip")
    p.add_argument("--chi-tien-do", action="store_true")
    p.add_argument("--cho-phep-thieu", action="store_true")
    a = p.parse_args(argv)
    try:
        if a.chi_tien_do:
            td = tien_do(a.thu_muc)
            in_tien_do(td, muc_thieu(a.thu_muc, td))
        else:
            dong_goi(a.thu_muc, a.out, a.cho_phep_thieu)
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
