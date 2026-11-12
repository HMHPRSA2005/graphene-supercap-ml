"""Giao diện chung cho mọi mô hình (mô hình nền, scikit-learn, GNG-R, PGCN-R).

Hợp đồng:
  fit(bang_hoc, bang_that_hoc) -> self
      bang_hoc      : dữ liệu được phép học (điểm thật phần học + mẫu ảo nếu có);
                      cột tối thiểu ratio, J, Csp, Vbar, nguon ('that'|'ao'), diem_goc.
      bang_that_hoc : chỉ các SỐ ĐO THẬT của phần học (cột ratio, J, Csp, Vbar, muc ...).
                      Dùng để học thang chuẩn hóa và cho mô hình nền khớp dạng hàm.
  predict(X) -> DataFrame cùng index với X (X có cột ratio, J), các cột:
      Csp_hat, Vbar_hat            (> 0, thang gốc; NaN/≤0 = dự đoán không hợp lệ, KHÔNG tự sửa)
      logCsp_sd, logVbar_sd        (độ lệch chuẩn dự đoán trên thang log; NaN nếu mô hình không có)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COT_DU_DOAN = ["Csp_hat", "Vbar_hat", "logCsp_sd", "logVbar_sd"]


class MoHinhCoSo:
    ten: str = "co_so"
    ho_tro_khoang: bool = False  # True nếu predict trả logCsp_sd/logVbar_sd hữu hạn

    def fit(self, bang_hoc: pd.DataFrame, bang_that_hoc: pd.DataFrame) -> "MoHinhCoSo":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


def khung_du_doan(X: pd.DataFrame, csp, vbar, sd_csp=None, sd_vbar=None) -> pd.DataFrame:
    """Đóng gói dự đoán theo đúng hợp đồng cột."""
    n = len(X)
    nan = np.full(n, np.nan)
    return pd.DataFrame({
        "Csp_hat": np.asarray(csp, float),
        "Vbar_hat": np.asarray(vbar, float),
        "logCsp_sd": nan if sd_csp is None else np.asarray(sd_csp, float),
        "logVbar_sd": nan if sd_vbar is None else np.asarray(sd_vbar, float),
    }, index=X.index)
