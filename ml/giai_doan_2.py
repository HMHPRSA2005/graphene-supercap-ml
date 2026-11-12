"""Nhận sản phẩm Colab, chạy ứng dụng terminal và chép tư liệu Chương 5."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

from .dong_goi import in_tien_do, muc_thieu, phien_ban, sha256, tien_do

DAU_BAO_CAO = "Tư liệu Chương 5 nhận từ sản phẩm Giai đoạn 1."


def _ten_an_toan(ten):
    p = PurePosixPath(ten)
    if (not ten or "\\" in ten or ":" in ten or p.is_absolute()
            or any(c in ("", ".", "..") for c in ten.rstrip("/").split("/"))):
        raise ValueError(f"Đường dẫn ZIP không an toàn: {ten!r}")
    return p


def nhan(tep_zip, thu_muc, ghi_de=False, *, da_nap=None):
    """Kiểm ZIP trước khi ghi; mặc định từ chối mọi tệp đích đã tồn tại."""
    dich = Path(thu_muc)
    with zipfile.ZipFile(tep_zip) as z, tempfile.TemporaryDirectory(prefix="nhan_colab_") as t:
        tam = Path(t)
        ten_da_co = set()
        for muc in z.infolist():
            p = _ten_an_toan(muc.filename)
            khoa = p.as_posix().casefold()
            if khoa in ten_da_co or stat.S_ISLNK(muc.external_attr >> 16):
                raise ValueError(f"ZIP có tên trùng hoặc liên kết tượng trưng: {muc.filename}")
            ten_da_co.add(khoa)
            z.extract(muc, tam)
            if not muc.is_dir():
                moc = datetime(*muc.date_time).timestamp()
                os.utime(tam / p, (moc, moc))
        if not (tam / "manifest.json").is_file():
            raise ValueError("ZIP thiếu manifest.json ở gốc; tạo lại bằng ml.dong_goi.")
        m = json.loads((tam / "manifest.json").read_text(encoding="utf-8"))
        if m.get("phien_ban_manifest") != 1:
            raise ValueError("Phiên bản manifest không được hỗ trợ.")
        da_ke = set()
        for muc in m["tep"]:
            ten = _ten_an_toan(muc["ten"]).as_posix()
            if ten == "manifest.json" or ten.casefold() in da_ke:
                raise ValueError(f"Tên manifest trùng/không hợp lệ: {ten}")
            da_ke.add(ten.casefold())
            p = tam / ten
            if (not p.is_file() or p.stat().st_size != muc["kich_thuoc"]
                    or sha256(p) != muc["sha256"]):
                raise ValueError(f"Sai SHA256 hoặc kích thước/thiếu tệp: {ten}; không nhận sản phẩm.")
        thuc = {p.relative_to(tam).as_posix().casefold() for p in tam.rglob("*")
                if p.is_file() and p != tam / "manifest.json"}
        if thuc != da_ke:
            raise ValueError("ZIP chứa tệp không được kiểm kê trong manifest.")
        cac_tep = [p for p in tam.rglob("*") if p.is_file()]
        xung_dot = []
        for p in cac_tep:
            out = dich / p.relative_to(tam)
            if not out.resolve().is_relative_to(dich.resolve()):
                raise ValueError(f"Đường dẫn đích vượt thư mục nhận: {out}")
            if out.exists():
                xung_dot.append(out)
        if xung_dot and not ghi_de:
            raise ValueError("Từ chối ghi đè các tệp đã có:\n  " + "\n  ".join(map(str, xung_dot))
                             + "\nDùng thư mục mới/rỗng hoặc sao lưu rồi thêm --ghi-de.")
        if xung_dot:
            print("Đã cho phép --ghi-de; các tệp sẽ thay thế:")
            for out in xung_dot:
                print(f"  {out}")
        td = tien_do(tam)
        thieu = [s.replace(str(tam), str(dich)) for s in muc_thieu(tam, td)]
        if td != m["tien_do"] or thieu != m["thieu"]:
            print("Kiểm kê lại từ cây đã giải nén khác manifest; giá trị ghi trong manifest:")
            in_tien_do(m["tien_do"], m["thieu"])
        print("Tiến độ/mục thiếu tính lại từ cây đã giải nén:")
        in_tien_do(td, thieu)
        dich.mkdir(parents=True, exist_ok=True)
        for p in cac_tep:
            out = dich / p.relative_to(tam)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
    hien_tai = phien_ban()
    for ten, pb in m["phien_ban"].items():
        if hien_tai.get(ten) != pb:
            print(f"Cảnh báo phiên bản {ten}: Colab={pb}, local={hien_tai.get(ten)}; best có thể phải khớp lại từ JSON.")
    from .luu_best import tai_best
    print("Chỉ nhận ZIP do chính bạn tạo/tin cậy: joblib có thể thực thi mã khi nạp; SHA256 không xác thực tác giả.")
    for p in sorted((dich / "best").glob("best_*.joblib")):
        goi = tai_best(p)
        if da_nap is not None:
            da_nap[p.resolve()] = goi
        print(f"Nạp được: {p}")
    print(f"Đã nhận và kiểm SHA256: {dich}. Tiếp theo: python -X utf8 -m ml.giai_doan_2 ung_dung --thu-muc \"{dich}\"")
    return m


def ung_dung(thu_muc, *, da_nap=None):
    from .du_doan import main as du_doan
    from .du_lieu import doc_so_lieu
    from .luu_best import tai_best
    r = Path(thu_muc)
    bests = [r / "best" / f"best_{m}.joblib" for m in ("chung", "co_khoang")]
    for p in bests:
        if not p.is_file() or not p.with_suffix(".json").is_file():
            raise ValueError(f"Thiếu {p} hoặc JSON; trên Colab chạy ml.luu_best rồi đóng gói lại.")
    ratios = sorted(doc_so_lieu().ratio.unique())
    for p in bests:
        goi = None if da_nap is None else da_nap.get(p.resolve())
        mo_hinh, tt_best = tai_best(p) if goi is None else goi
        for ratio in ratios:
            out = r / "ung_dung" / p.stem / f"ti_le_{ratio:g}.xlsx"
            du_doan(["--ti-le", str(ratio), "--best", str(p), "--out", str(out)],
                    mo_hinh=mo_hinh, tt_best=tt_best)
    print(f"Ứng dụng: {r / 'ung_dung'} (16 XLSX + 16 PNG; gồm đề xuất đo khi có sd).")
    print(f'Tiếp theo: python -X utf8 -m ml.giai_doan_2 bao_cao --thu-muc "{r}" --bao-cao ../Report/Report')


def bao_cao(thu_muc, thu_muc_bao_cao):
    r = Path(thu_muc)
    dich = Path(thu_muc_bao_cao) / "ket_qua_ch5"
    nguon = [(p, p.name) for p in sorted((r / "bang_ch5").glob("*.tex"))]
    hinh = {p.stem: p for p in (r / "hinh_ch5").glob("*.png")}
    hinh.update({p.stem: p for p in (r / "hinh_ch5").glob("*.pdf")})
    nguon += [(hinh[k], hinh[k].name) for k in sorted(hinh)]
    nguon += [(p, "so_sanh_" + p.name) for p in sorted((r / "so_sanh").glob("*.pdf"))]
    if not nguon:
        raise ValueError("Không có bảng/hình để chép; chạy ml.so_sanh và ml.ve_hinh_ch5 --ung-dung trên Colab rồi đóng gói lại.")
    if dich.exists():
        marker = dich / "README.txt"
        if (dich.is_symlink() or not marker.is_file()
                or not marker.read_text(encoding="utf-8").startswith(DAU_BAO_CAO + "\n")):
            raise ValueError(f"Từ chối thay thư mục {dich}: không có README.txt đánh dấu do công cụ tạo. Sao lưu hoặc chọn thư mục báo cáo khác.")
        print(f"Thay sạch tư liệu Chương 5 đã sinh: {dich}")
        shutil.rmtree(dich)
    dich.mkdir(parents=True, exist_ok=True)
    dong = [DAU_BAO_CAO,
            "Chỉ chép tệp; chưa sửa nội dung luận văn, chưa biên dịch LaTeX.",
            "Hình ưu tiên PDF; chỉ dùng PNG nếu không có PDF cùng tên.",
            "Khớp lại để sử dụng, không phải kết quả kiểm tra (các hình ứng dụng/so sánh).", ""]
    for p, ten in nguon:
        shutil.copy2(p, dich / ten)
        dong.append(f"{ten} <- {p.resolve()}")
        print(f"Chép: {p} -> {dich / ten}")
    (dich / "README.txt").write_text("\n".join(dong) + "\n", encoding="utf-8")
    print(f"Tiếp theo: đọc {dich / 'README.txt'}, chèn bảng bằng \\input{{ket_qua_ch5/<tên>.tex}} và hình bằng \\includegraphics{{ket_qua_ch5/<tên>.pdf}}; đối chiếu dữ liệu thiếu trong manifest trước khi viết kết luận, rồi biên dịch báo cáo.")
    return [ten for _, ten in nguon]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="lenh", required=True)
    for ten in ("nhan", "ung_dung", "bao_cao", "tat_ca"):
        c = sub.add_parser(ten)
        if ten in ("nhan", "tat_ca"):
            c.add_argument("zip", help="ZIP sản phẩm do chính bạn tạo/tin cậy")
            c.add_argument("--ghi-de", action="store_true",
                           help="Cho thay các tệp đích đã có; liệt kê tệp thay thế. Mặc định từ chối: nên nhận vào thư mục mới/rỗng hoặc sao lưu trước.")
        c.add_argument("--thu-muc", default="ket_qua")
        if ten in ("bao_cao", "tat_ca"):
            c.add_argument("--bao-cao", required=True)
    a = p.parse_args(argv)
    da_nap = {}
    try:
        if a.lenh in ("nhan", "tat_ca"):
            nhan(a.zip, a.thu_muc, a.ghi_de, da_nap=da_nap)
        if a.lenh in ("ung_dung", "tat_ca"):
            ung_dung(a.thu_muc, da_nap=da_nap)
        if a.lenh in ("bao_cao", "tat_ca"):
            bao_cao(a.thu_muc, a.bao_cao)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
