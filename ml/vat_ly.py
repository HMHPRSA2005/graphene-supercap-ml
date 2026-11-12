"""Công thức của phòng thí nghiệm: suy Ed, Pd, Δt từ Csp và V̄ (mục 4.2).

Mô hình chỉ học 2 đại lượng Csp và V̄; Ed, Pd, Δt luôn tính bằng các công thức dưới đây
nên kết quả dự đoán tự nhất quán về vật lý.
"""
from __future__ import annotations

import numpy as np

DV = 2.1  # cửa sổ điện áp (V), giống gcdsim.DV_DEFAULT


def ed_tu_csp(csp, dv: float = DV):
    """Ed (Wh/kg) = Csp·ΔV²/7,2  (= 0,6125·Csp khi ΔV = 2,1 V)."""
    return np.asarray(csp, float) * dv ** 2 / 7.2


def pd_tu_vbar(J, vbar):
    """Pd (W/kg) = 1000·J·V̄."""
    return 1000.0 * np.asarray(J, float) * np.asarray(vbar, float)


def vbar_tu_pd(J, pd_):
    """V̄ (V) = Pd / (1000·J)."""
    return np.asarray(pd_, float) / (1000.0 * np.asarray(J, float))


def dt_xa(ed, pd_):
    """Δt (s) = 3600·Ed / Pd."""
    return 3600.0 * np.asarray(ed, float) / np.asarray(pd_, float)


def suy_ra(J, csp, vbar, dv: float = DV) -> dict:
    """Trả về dict Csp, Vbar, Ed, Pd, dt từ (J, Csp, V̄)."""
    ed = ed_tu_csp(csp, dv)
    p = pd_tu_vbar(J, vbar)
    return {"Csp": np.asarray(csp, float), "Vbar": np.asarray(vbar, float),
            "Ed": ed, "Pd": p, "dt": dt_xa(ed, p)}
