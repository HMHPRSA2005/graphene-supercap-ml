"""Các rủi ro thật của trao đổi sản phẩm Colab/local."""
import hashlib
import json
import zipfile

import pytest

from ml.dong_goi import dong_goi
from ml.giai_doan_2 import nhan


def test_so_mau_co_day_du_bon_muc():
    from ml.cau_hinh import NHOM
    assert all(v["n_ao"] == [10, 100, 1000, 10000] for v in NHOM["so_mau"])



def test_san_pham_thieu_bi_tu_choi(tmp_path):
    r = tmp_path / "ket_qua"
    r.mkdir()
    out = tmp_path / "san_pham.zip"
    with pytest.raises(ValueError, match="Từ chối đóng gói thiếu"):
        dong_goi(r, out)
    assert not out.exists()
    m = dong_goi(r, out, cho_phep_thieu=True)
    assert m["thieu"]
    assert m["tien_do"]["so_mau"]["da_du"] == 0
    assert out.is_file()


def test_phat_hien_sha256_sai_truoc_khi_ghi_dich(tmp_path):
    out = tmp_path / "sai_sha.zip"
    manifest = dict(phien_ban_manifest=1, tep=[dict(ten="ket_qua.csv", kich_thuoc=3,
                    sha256=hashlib.sha256(b"abc").hexdigest())])
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("manifest.json", json.dumps(manifest))
        z.writestr("ket_qua.csv", "xyz")
    dich = tmp_path / "nhan"
    with pytest.raises(ValueError, match="SHA256"):
        nhan(out, dich)
    assert not dich.exists()


@pytest.mark.parametrize("ten", ["../thoat.txt", "a/../../thoat.txt", "/thoat.txt",
                                     "C:/thoat.txt", "a\\..\\thoat.txt"])
def test_zip_tu_choi_vuot_thu_muc(tmp_path, ten):
    out = tmp_path / "khong_an_toan.zip"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr(ten, "không được ghi")
    with pytest.raises(ValueError, match="không an toàn"):
        nhan(out, tmp_path / "nhan")
    assert not (tmp_path / "nhan").exists()
    assert not (tmp_path / "thoat.txt").exists()


def test_nhan_tu_choi_ghi_de_tru_khi_co_co(tmp_path, capsys):
    nguon = tmp_path / "nguon"
    nguon.mkdir()
    (nguon / "ghi_chu.txt").write_text("bản mới", encoding="utf-8")
    tep_zip = tmp_path / "san_pham.zip"
    dong_goi(nguon, tep_zip, cho_phep_thieu=True)
    dich = tmp_path / "nhan"
    dich.mkdir()
    tep_cu = dich / "ghi_chu.txt"
    tep_cu.write_text("bản cần giữ", encoding="utf-8")
    with pytest.raises(ValueError, match="Từ chối ghi đè") as loi:
        nhan(tep_zip, dich)
    assert "ghi_chu.txt" in str(loi.value)
    assert tep_cu.read_text(encoding="utf-8") == "bản cần giữ"
    assert not (dich / "manifest.json").exists()
    nhan(tep_zip, dich, ghi_de=True)
    assert tep_cu.read_text(encoding="utf-8") == "bản mới"
    assert "các tệp sẽ thay thế" in capsys.readouterr().out


def test_dong_goi_bao_san_pham_cu_hon_csv(tmp_path):
    import os
    from ml.dong_goi import muc_thieu, NHOM_CHINH
    nguon = tmp_path / "ket_qua_kb1_kb4.csv"
    nguon.write_text("dữ liệu mới", encoding="utf-8")
    bang = tmp_path / "bang_ch5" / "bang_5_2.tex"
    bang.parent.mkdir()
    bang.write_text("bảng đã cũ", encoding="utf-8")
    os.utime(bang, (100, 100))
    os.utime(nguon, (200, 200))
    td = {g: dict(da_du=1, can_co=1) for g in NHOM_CHINH}
    thieu = muc_thieu(tmp_path, td)
    cu = [s for s in thieu if "cũ hơn ket_qua_kb1_kb4.csv" in s]
    assert len(cu) == 1
    assert "bang_ch5/bang_5_2.tex" in cu[0]
    assert "Cuối Phiên 3" in cu[0]


@pytest.mark.parametrize("loai", ["khong_phai_zip", "crc_sai"])
def test_dong_goi_bao_xlsx_hong(tmp_path, loai):
    from ml.dong_goi import muc_thieu, NHOM_CHINH, xlsx_nguyen_ven
    tep = tmp_path / "gcdsim" / "du_lieu_ao_480000.xlsx"
    tep.parent.mkdir()
    if loai == "khong_phai_zip":
        tep.write_bytes(b"ghi do")
    else:
        with zipfile.ZipFile(tep, "w", compression=zipfile.ZIP_STORED) as z:
            z.writestr("xl/workbook.xml", b"NOI_DUNG_CAN_KIEM_CRC")
        tep.write_bytes(tep.read_bytes().replace(b"NOI_DUNG_CAN_KIEM_CRC", b"NOI_DUNG_DA_SAI___CRC"))
    assert not xlsx_nguyen_ven(tep)
    td = {g: dict(da_du=1, can_co=1) for g in NHOM_CHINH}
    assert any("XLSX GCD-Sim hỏng" in s for s in muc_thieu(tmp_path, td))
