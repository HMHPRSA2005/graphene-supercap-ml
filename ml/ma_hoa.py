"""Biểu diễn đầu vào và đầu ra (mục 4.3, Bảng 4.1).

Đầu vào:
  - kieu="onehot": [λ·one-hot(điện cực) (8 cột), J~]  — KB1, KB2, KB3, KB5
  - kieu="ti_le" : [r~, J~] với r~ = log r chuẩn hóa      — KB4 (điện cực mới)
  J~ = (log J − log Jmin) / (log Jmax − log Jmin); Jmin, Jmax lấy từ SỐ ĐO THẬT của phần học.
  Không chặn J~ vào [0, 1] khi ngoại suy.
Đầu ra học: y = [log Csp, log V̄]; giải mã bằng exp.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Danh mục 8 điện cực cố định trước thí nghiệm (chỉ là định danh, không chứa số đo).
DANH_MUC_TI_LE: tuple[float, ...] = (0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 1.75)


class BoMaHoa:
    def __init__(self, kieu: str = "onehot", lam: float = 1.0,
                 danh_muc: tuple[float, ...] = DANH_MUC_TI_LE):
        if kieu not in ("onehot", "ti_le"):
            raise ValueError(f"kieu phải là 'onehot' hoặc 'ti_le', nhận {kieu!r}")
        self.kieu = kieu
        self.lam = float(lam)
        self.danh_muc = tuple(float(r) for r in danh_muc)
        self._vi_tri = {round(r, 9): i for i, r in enumerate(self.danh_muc)}

    @staticmethod
    def _thang(v: np.ndarray) -> tuple[float, float]:
        lo, hi = float(np.min(v)), float(np.max(v))
        return lo, (hi - lo) if hi > lo else 1.0

    def fit(self, bang_that: pd.DataFrame) -> "BoMaHoa":
        """Học thang chuẩn hóa từ số đo thật của phần học (cột ratio, J)."""
        self.logJ0_, self.logJw_ = self._thang(np.log(bang_that["J"].to_numpy(float)))
        if self.kieu == "ti_le":
            self.logr0_, self.logrw_ = self._thang(np.log(bang_that["ratio"].to_numpy(float)))
        return self

    def j_chuan_hoa(self, J) -> np.ndarray:
        return (np.log(np.asarray(J, float)) - self.logJ0_) / self.logJw_

    def transform(self, bang: pd.DataFrame) -> np.ndarray:
        jt = self.j_chuan_hoa(bang["J"].to_numpy(float))[:, None]
        r = bang["ratio"].to_numpy(float)
        if self.kieu == "ti_le":
            rt = ((np.log(r) - self.logr0_) / self.logrw_)[:, None]
            return np.hstack([rt, jt])
        idx = np.empty(len(r), int)
        for k, v in enumerate(r):
            key = round(float(v), 9)
            if key not in self._vi_tri:
                raise ValueError(f"Tỉ lệ {v} không thuộc danh mục điện cực {self.danh_muc}")
            idx[k] = self._vi_tri[key]
        oh = np.zeros((len(r), len(self.danh_muc)))
        oh[np.arange(len(r)), idx] = self.lam
        return np.hstack([oh, jt])

    def fit_transform(self, bang_that: pd.DataFrame, bang: pd.DataFrame | None = None) -> np.ndarray:
        self.fit(bang_that)
        return self.transform(bang_that if bang is None else bang)

    @property
    def so_chieu(self) -> int:
        return 2 if self.kieu == "ti_le" else len(self.danh_muc) + 1


def ma_hoa_y(bang: pd.DataFrame) -> np.ndarray:
    """(n, 2): [log Csp, log V̄]."""
    return np.column_stack([np.log(bang["Csp"].to_numpy(float)),
                            np.log(bang["Vbar"].to_numpy(float))])


def giai_ma_y(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trả (Csp, V̄) trên thang gốc. exp(μ) là trung vị log-normal (mục 4.2)."""
    Y = np.asarray(Y, float)
    return np.exp(Y[:, 0]), np.exp(Y[:, 1])
