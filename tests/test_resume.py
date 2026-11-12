"""Chạy hai lần đúng CLI công khai: lần hai không chạy lại đơn vị đã đủ seed, không nhân đôi hàng."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


GOC = Path(__file__).resolve().parents[1]


def test_chay_lai_khong_nhan_doi(tmp_path):
    # Chạy trọn nhóm nhỏ (mô hình nền, 1 seed) để lần hai không còn đơn vị nào cần chạy.
    lenh = [sys.executable, "-X", "utf8", "-m", "ml.chay_kb1_kb4",
            "--nhom", "thu", "--thu-muc", str(tmp_path), "--n-jobs", "1",
            "--mo-hinh", "ham_mu", "--seeds", "0"]
    for lan in range(2):
        p = subprocess.run(lenh, cwd=GOC, capture_output=True,
                           text=True, encoding="utf-8", timeout=120)
        assert p.returncode == 0, p.stdout + p.stderr
        bang = pd.read_csv(tmp_path / "ket_qua_kb1_kb4.csv")
        assert len(bang) > 0
        if lan == 0:
            lan_dau = bang.copy()
        else:
            assert "cần chạy: 0 đơn vị" in p.stdout, p.stdout
            pd.testing.assert_frame_equal(bang, lan_dau)


@pytest.mark.parametrize("so_byte_da_ghi", [0, 5, None])
@pytest.mark.parametrize("co_tep_cu", [False, True])
def test_phat_lai_nhat_ky_khong_nhan_doi(tmp_path, monkeypatch, so_byte_da_ghi, co_tep_cu):
    from ml.chay_kb1_kb4 import ghi_noi_tiep, phuc_hoi_csv

    dich = tmp_path / "ket_qua.csv"
    nhat_ky = dich.with_suffix(".pending.json")
    cot = ["ma", "ghi_chu"]
    hang_cu = [{"ma": 1, "ghi_chu": "đã ghi"}] if co_tep_cu else []
    ghi_noi_tiep(dich, hang_cu, cot)
    vi_tri = dich.stat().st_size if dich.exists() else 0
    unlink_goc = Path.unlink

    def ngat_truoc_xac_nhan(tep, *args, **kwargs):
        if tep == nhat_ky:
            raise OSError("Giả lập mất kết nối sau ghi, trước xác nhận")
        return unlink_goc(tep, *args, **kwargs)

    hang_moi = [{"ma": 2, "ghi_chu": "dữ liệu mới"}]
    with monkeypatch.context() as m:
        m.setattr(Path, "unlink", ngat_truoc_xac_nhan)
        with pytest.raises(OSError, match="mất kết nối"):
            ghi_noi_tiep(dich, hang_moi, cot)
    assert nhat_ky.exists()
    gd = json.loads(nhat_ky.read_text(encoding="utf-8"))
    assert gd["vi_tri"] == vi_tri
    if so_byte_da_ghi is not None:
        # Mô phỏng ngắt trước ghi hoặc giữa giao dịch; None = ghi đủ nhưng chưa xác nhận.
        with dich.open("r+b") as tep:
            tep.truncate(vi_tri + so_byte_da_ghi)
    phuc_hoi_csv(dich)
    noi_dung_lan_dau = dich.read_bytes()
    phuc_hoi_csv(dich)
    assert dich.read_bytes() == noi_dung_lan_dau
    assert not nhat_ky.exists()
    mong_doi = pd.DataFrame(hang_cu + hang_moi)
    pd.testing.assert_frame_equal(pd.read_csv(dich), mong_doi)
    # Giao dịch kế tiếp phải nối đúng sau phần vừa phục hồi.
    hang_tiep = [{"ma": 3, "ghi_chu": "giao dịch kế"}]
    ghi_noi_tiep(dich, hang_tiep, cot)
    pd.testing.assert_frame_equal(pd.read_csv(dich), pd.DataFrame(hang_cu + hang_moi + hang_tiep))
