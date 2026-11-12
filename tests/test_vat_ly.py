"""Kiểm tra cả 48 số đo, không dùng dữ liệu sinh thay số đo gốc."""
import numpy as np

from ml.vat_ly import dt_xa, ed_tu_csp, pd_tu_vbar, suy_ra, vbar_tu_pd


def test_tai_tao_ed_pd_48_diem(bang):
    assert len(bang) == 48
    vbar = vbar_tu_pd(bang.J, bang.Pd)
    np.testing.assert_allclose(ed_tu_csp(bang.Csp), bang.Ed, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pd_tu_vbar(bang.J, vbar), bang.Pd, rtol=1e-12, atol=1e-12)
    ket_qua = suy_ra(bang.J, bang.Csp, vbar)
    np.testing.assert_allclose(ket_qua["Ed"], bang.Ed, rtol=1e-12)
    np.testing.assert_allclose(ket_qua["Pd"], bang.Pd, rtol=1e-12)
    np.testing.assert_allclose(ket_qua["dt"], dt_xa(bang.Ed, bang.Pd), rtol=1e-12)
