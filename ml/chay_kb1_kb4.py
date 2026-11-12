"""CLI song song, ghi kết quả ở tiến trình chính và tiếp tục sau gián đoạn.

CSV có nhật ký giao dịch cạnh tệp: lần khởi động sau hoàn tất đúng giao dịch
đang dở, kể cả khi mất kết nối giữa lúc ghi. Không chạy hai CLI cùng thư mục.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter


def ghi_json_nguyen_tu(duong_dan, noi_dung):
    """Thay JSON nguyên tử cùng thư mục, đồng bộ trước khi đổi tên."""
    dich = Path(duong_dan)
    tam = dich.with_suffix(dich.suffix + ".tmp")
    with tam.open("w", encoding="utf-8") as tep:
        json.dump(noi_dung, tep, ensure_ascii=False, indent=2)
        tep.flush()
        os.fsync(tep.fileno())
    os.replace(tam, dich)


def ghi_moi_truong(duong_dan_thu_muc, cli=None, argv=None, **tuy_chon):
    """Ghi nối lịch sử môi trường cho từng lần gọi cả hai CLI."""
    phien_ban = {}
    for ten in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib", "openpyxl"):
        try:
            phien_ban[ten] = importlib.metadata.version(ten)
        except importlib.metadata.PackageNotFoundError:
            phien_ban[ten] = None
    doi_so = list(sys.argv[1:] if argv is None else argv)
    noi_dung = dict(python=platform.python_version(), thu_vien=phien_ban,
                   he_dieu_hanh=platform.platform(), cpu=platform.processor(),
                   so_nhan=os.cpu_count(), thoi_diem=datetime.now(timezone.utc).isoformat(),
                   cli=cli, argv=doi_so,
                   lenh=[sys.executable, "-X", "utf8", "-m", cli or __name__, *doi_so],
                   tuy_chon=tuy_chon)
    dich = Path(duong_dan_thu_muc) / "moi_truong.json"
    lich_su = json.loads(dich.read_text(encoding="utf-8")) if dich.exists() else []
    if isinstance(lich_su, dict):
        lich_su = [lich_su]
    if not isinstance(lich_su, list):
        raise ValueError(f"{dich}: lịch sử môi trường phải là danh sách JSON")
    ghi_json_nguyen_tu(dich, [*lich_su, noi_dung])


def phuc_hoi_csv(duong_dan):
    """Phát lại giao dịch đã ghi nhật ký nhưng chưa xác nhận xong."""
    dich = Path(duong_dan)
    nhat_ky = dich.with_suffix(".pending.json")
    if nhat_ky.exists():
        gd = json.loads(nhat_ky.read_text(encoding="utf-8"))
        with dich.open("r+b" if dich.exists() else "w+b") as tep:
            tep.truncate(gd["vi_tri"])
            tep.seek(gd["vi_tri"])
            tep.write(gd["noi_dung"].encode("utf-8"))
            tep.flush()
            os.fsync(tep.fileno())
        nhat_ky.unlink()


def ghi_noi_tiep(duong_dan, hang, cot):
    """Một giao dịch nối tiếp bền vững; chỉ tiến trình chính gọi hàm này."""
    if not hang:
        return
    dich = Path(duong_dan)
    phuc_hoi_csv(dich)
    vi_tri = dich.stat().st_size if dich.exists() else 0
    chuoi = io.StringIO(newline="")
    bo_ghi = csv.DictWriter(chuoi, fieldnames=cot, extrasaction="raise")
    if not vi_tri:
        bo_ghi.writeheader()
    bo_ghi.writerows(hang)
    noi_dung = chuoi.getvalue()
    nhat_ky = dich.with_suffix(".pending.json")
    tam = nhat_ky.with_suffix(".tmp")
    with tam.open("w", encoding="utf-8") as tep:
        json.dump(dict(vi_tri=vi_tri, noi_dung=noi_dung), tep, ensure_ascii=False)
        tep.flush()
        os.fsync(tep.fileno())
    os.replace(tam, nhat_ky)
    with dich.open("ab") as tep:
        tep.write(noi_dung.encode("utf-8"))
        tep.flush()
        os.fsync(tep.fileno())
    nhat_ky.unlink()


def kiem_tra_header(duong_dan, cot):
    """Từ chối CSV khác schema trước khi đọc hoặc nối thêm hàng."""
    dich = Path(duong_dan)
    if dich.exists() and dich.stat().st_size:
        with dich.open(encoding="utf-8-sig", newline="") as tep:
            header = next(csv.reader(tep), [])
        if header != list(cot):
            raise ValueError(f"Header CSV {dich} không khớp schema hiện tại. "
                             "Hãy dùng thư mục kết quả khác hoặc chuyển đổi dữ liệu có kiểm soát; "
                             f"cần {list(cot)}, nhận {header}")


def kiem_tra_ghi(thu_muc):
    """Thử ghi, đổi tên và phục hồi giao dịch thật ngay trên ổ đích."""
    try:
        Path(thu_muc).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="_kiem_tra_ghi_", dir=thu_muc) as tam:
            dich = Path(tam) / "kiem_tra.csv"
            ghi_noi_tiep(dich, [dict(id=1, ten="điện cực")], ["id", "ten"])
            vi_tri = dich.stat().st_size
            noi_dung = "2,tiếp tục\r\n"
            ghi_json_nguyen_tu(dich.with_suffix(".pending.json"),
                              dict(vi_tri=vi_tri, noi_dung=noi_dung))
            with dich.open("ab") as tep:
                tep.write(noi_dung.encode("utf-8")[:5])
                tep.flush()
                os.fsync(tep.fileno())
            phuc_hoi_csv(dich)
            phuc_hoi_csv(dich)
            ghi_noi_tiep(dich, [dict(id=3, ten="hoàn tất")], ["id", "ten"])
            kiem_tra_header(dich, ["id", "ten"])
            with dich.open(encoding="utf-8", newline="") as tep:
                hang = list(csv.DictReader(tep))
            if hang != [dict(id="1", ten="điện cực"), dict(id="2", ten="tiếp tục"),
                        dict(id="3", ten="hoàn tất")]:
                raise ValueError("Nội dung phục hồi khác dữ liệu đã ghi")
            thay = Path(tam) / "doi_ten.csv"
            du_lieu = dich.read_bytes()
            thay.write_text("bản cũ cần được thay thế", encoding="utf-8")
            os.replace(dich, thay)
            if not thay.exists() or dich.exists() or thay.read_bytes() != du_lieu:
                raise ValueError("os.replace không thay thế tệp như yêu cầu")
        print(f"OK: ghi nối tiếp, phục hồi CSV UTF-8 và os.replace tại {thu_muc}", flush=True)
        return True
    except Exception as exc:
        print(f"Lỗi kiểm tra ghi tại {thu_muc}: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        return False


def _ghi_lai_csv(duong_dan, bang):
    """Loại đơn vị hỏng bằng thay tệp nguyên tử, không ghi đè CSV tại chỗ."""
    dich = Path(duong_dan)
    tam = dich.with_suffix(".rewrite.tmp")
    with tam.open("w", encoding="utf-8", newline="") as tep:
        bang.to_csv(tep, index=False)
        tep.flush()
        os.fsync(tep.fileno())
    os.replace(tam, dich)


def _chay(bang, don_vi):
    from .thuc_nghiem import chay_don_vi
    t = perf_counter()
    return don_vi, chay_don_vi(bang, **don_vi), perf_counter() - t


COT_KHOA = ["kb", "ma_lan_chia", "mo_hinh", "cach", "n_ao", "sigma_lap",
            "kieu_ma_hoa", "cau_hinh_chon", "seed"]


def doc_tien_do(duong_dan):
    """Đọc cùng khóa/điểm kiểm tra cho resume và đóng gói sản phẩm."""
    import pandas as pd
    from .thuc_nghiem import COT_KET_QUA
    dich = Path(duong_dan)
    phuc_hoi_csv(dich)
    kiem_tra_header(dich, COT_KET_QUA)
    cu = pd.read_csv(dich) if dich.exists() and dich.stat().st_size else pd.DataFrame()
    da_co = {}
    if not cu.empty:
        for khoa, nhom in cu.groupby(COT_KHOA, dropna=False):
            da_co[khoa] = (set(zip(nhom.ratio.round(9), nhom.muc.astype(int))), len(nhom))
    return cu, da_co


def diem_kiem_tra(bang, cac_viec):
    from .cau_hinh import khoa_don_vi
    ket_qua = {}
    for d in cac_viec:
        f = d["lan_chia"]
        diem = set(zip(bang.loc[f.kiem_tra, "ratio"].round(9), bang.loc[f.kiem_tra, "muc"].astype(int)))
        ket_qua[khoa_don_vi(d)] = (diem, len(diem))
    return ket_qua


def main(argv=None):
    # cau_hinh nhập bộ sinh numpy/scipy; giới hạn luồng trước khi nhập.
    for ten in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[ten] = "1"
    from .cau_hinh import TEN_NHOM, khoa_don_vi, liet_ke_don_vi
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhom", action="append", choices=TEN_NHOM)
    parser.add_argument("--thu-muc", default="ket_qua")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--mo-hinh", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--toi-da-don-vi", type=int)
    parser.add_argument("--bo-don-vi-do-dang", action="store_true",
                        help="Loại toàn bộ hàng của đơn vị đang chọn bị thiếu/trùng điểm rồi chạy lại")
    parser.add_argument("--kiem-tra-ghi", action="store_true",
                        help="Tự thử ghi/phục hồi/đổi tên tại thư mục đích rồi thoát, không học mô hình")
    args = parser.parse_args(argv)
    if args.kiem_tra_ghi:
        if not kiem_tra_ghi(args.thu_muc):
            raise SystemExit(1)
        return
    if not args.nhom:
        parser.error("cần --nhom khi không dùng --kiem-tra-ghi")
    if args.n_jobs == 0:
        parser.error("--n-jobs không được bằng 0")
    if args.toi_da_don_vi is not None and args.toi_da_don_vi < 1:
        parser.error("--toi-da-don-vi phải dương")
    from joblib import Parallel, delayed, parallel_config
    from .du_lieu import doc_so_lieu
    from .thuc_nghiem import COT_KET_QUA, TEN_MO_HINH
    if args.mo_hinh and (sai := sorted(set(args.mo_hinh) - set(TEN_MO_HINH))):
        parser.error(f"--mo-hinh không hợp lệ: {', '.join(sai)}; chọn trong {TEN_MO_HINH}")

    thu_muc = Path(args.thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    ghi_moi_truong(thu_muc, cli="ml.chay_kb1_kb4", argv=argv, **vars(args))
    dich = thu_muc / "ket_qua_kb1_kb4.csv"
    try:
        cu, da_co = doc_tien_do(dich)
    except ValueError as exc:
        parser.error(str(exc))
    bang = doc_so_lieu()
    cac_viec = liet_ke_don_vi(bang, args.nhom, args.mo_hinh, args.seeds)
    cot_khoa = COT_KHOA
    diem_don_vi = diem_kiem_tra(bang, cac_viec)
    do_dang = {khoa[:-1] for khoa, diem in da_co.items()
               if khoa[:-1] in diem_don_vi and diem != diem_don_vi[khoa[:-1]]}
    if do_dang:
        if not args.bo_don_vi_do_dang:
            parser.error(f"Có {len(do_dang)} đơn vị CSV không trọn vẹn hoặc trùng điểm, không có nhật ký. "
                         "Dùng --bo-don-vi-do-dang để loại các đơn vị đó rồi chạy lại.")
        giu = [k not in do_dang for k in cu[cot_khoa[:-1]].itertuples(index=False, name=None)]
        _ghi_lai_csv(dich, cu.loc[giu])
        da_co = {k: v for k, v in da_co.items() if k[:-1] not in do_dang}
        print(f"Đã loại {len(do_dang)} đơn vị dở dang để chạy lại toàn bộ seed được yêu cầu", flush=True)
    con_lai = []
    for d in cac_viec:
        khoa = khoa_don_vi(d)
        thieu = [s for s in d["seeds"] if da_co.get((*khoa, s)) != diem_don_vi[khoa]]
        if thieu:
            con_lai.append({**d, "seeds": thieu})
    so_da_du = len(cac_viec) - len(con_lai)
    if args.toi_da_don_vi is not None:
        con_lai = con_lai[:args.toi_da_don_vi]
    print(f"Đã đủ: {so_da_du}/{len(cac_viec)}; cần chạy: {len(con_lai)} đơn vị", flush=True)
    if not con_lai:
        return
    bat_dau = perf_counter()
    # generator_unordered giải phóng kết quả từng đơn vị, không đợi cả lô.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        try:
            bo_chay = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered", batch_size=1)
        except ValueError:
            # joblib 1.3 có generator có thứ tự, chưa có generator_unordered.
            bo_chay = Parallel(n_jobs=args.n_jobs, return_as="generator", batch_size=1)
        ket_qua = bo_chay(delayed(_chay)(bang, d) for d in con_lai)
        for i, (d, hang, giay) in enumerate(ket_qua, 1):
            ghi_noi_tiep(dich, hang, COT_KET_QUA)
            thoi_gian = dict(nhom=d["nhom"], kb=d["lan_chia"].kb, ma_lan_chia=d["lan_chia"].ma,
                             mo_hinh=d["ten"], cach=d["cau_hinh_tc"].cach,
                             n_ao=d["cau_hinh_tc"].n_ao, so_seed=len(d["seeds"]),
                             rut_gon=d["rut_gon"], t_don_vi_s=giay)
            ghi_noi_tiep(thu_muc / "thoi_gian_don_vi.csv", [thoi_gian], list(thoi_gian))
            da_qua = perf_counter() - bat_dau
            eta = da_qua / i * (len(con_lai) - i)
            print(f"[{i}/{len(con_lai)}] {d['lan_chia'].kb} {d['ten']} {d['cau_hinh_tc'].cach}: {giay:.1f}s; đã chạy {da_qua:.1f}s; ETA {eta / 60:.1f} phút", flush=True)
    print(f"Hoàn tất {len(con_lai)} đơn vị trong {perf_counter() - bat_dau:.1f}s: {dich}", flush=True)


if __name__ == "__main__":
    main()
