"""Các mô hình đối chứng scikit-learn (mục 4.4, lưới Bảng 4.2).

Tham số dùng tên của scikit-learn; riêng Ridge có thêm ``degree`` và MLP
có ``so_buoc`` (mặc định 6000, không thuộc LUOI). MLP dùng batch_size =
min(200, n), so_lo = ceil(n / batch_size), max_iter = max(1,
ceil(so_buoc / so_lo)): ngân sách bước Adam chỉ làm tròn lên một vòng học,
thay vì giữ số vòng cố định khi tăng số mẫu ảo. GPR tối ưu một lần
(không khởi động lại), RF chạy một luồng để bộ chạy có thể song song ở
mức lần chia. Mô hình không tự lấy mẫu con: bộ chạy áp dụng TOI_DA_HANG
trước fit và phải giữ đủ điểm thật.
"""
from __future__ import annotations

from itertools import product
from math import ceil
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from .giao_dien import MoHinhCoSo, khung_du_doan
from .ma_hoa import BoMaHoa, giai_ma_y, ma_hoa_y


TEN_MO_HINH = ("ridge", "svr", "rf", "gb", "mlp", "gpr")
TOI_DA_HANG = {"gpr": 600, "svr": 2000}
LUOI: dict[str, list[dict]] = {
    "ridge": [dict(degree=bac, alpha=alpha)
              for bac, alpha in product((1, 2, 3), (1e-4, 1e-3, 1e-2, 1e-1, 1.0))],
    "svr": [dict(C=c, gamma=gamma, epsilon=epsilon)
            for c, gamma, epsilon in product((1, 10, 100), (0.1, 1, 10), (0.01, 0.05))],
    "rf": [dict(n_estimators=n, max_depth=sau, min_samples_leaf=la)
           for n, sau, la in product((200, 500), (3, 5, 8, None), (1, 3))],
    "gb": [dict(n_estimators=n, learning_rate=lr, max_depth=sau)
           for n, lr, sau in product((100, 300), (0.03, 0.1), (2, 3, 4))],
    "mlp": [dict(hidden_layer_sizes=lop, alpha=alpha, learning_rate_init=lr)
            for lop, alpha, lr in product(((16,), (32,), (32, 16), (64, 32)),
                                          (1e-4, 1e-3, 1e-2), (1e-3, 1e-2))],
    "gpr": [{}],
}


class MoHinhSklearn(MoHinhCoSo):
    """Học hai đầu ra log, chuẩn hóa y chỉ từ bảng học của lần fit.

    X chỉ được chuẩn hóa thêm trong pipeline đa thức Ridge. GPR trả độ
    lệch chuẩn của quan sát mới (WhiteKernel nằm trong phương sai dự đoán),
    đổi về đơn vị log bằng độ lệch chuẩn y, không cộng nhiễu lần thứ hai.
    ``hoc_tiep`` giữ nguyên bộ mã hóa, thang y và trạng thái Adam ban đầu.
    """

    def __init__(self, ten: str, tham_so: dict | None = None,
                 kieu_ma_hoa: str = "onehot", lam: float = 1.0, seed: int = 0):
        if ten not in TEN_MO_HINH:
            raise ValueError(f"Mô hình {ten!r} không hợp lệ; chọn trong {TEN_MO_HINH}")
        self.ten = ten
        self.tham_so = {**LUOI[ten][0], **(tham_so or {})}
        self.kieu_ma_hoa = kieu_ma_hoa
        self.lam = lam
        self.seed = seed
        self.ho_tro_khoang = ten == "gpr"

    def _tao_bo_hoi_quy(self, so_hang: int):
        """Tạo mới bộ học để gọi fit lần nữa không giữ trạng thái cũ."""
        p = dict(self.tham_so)
        if self.ten == "ridge":
            bac = p.pop("degree")
            p["random_state"] = self.seed
            return make_pipeline(PolynomialFeatures(degree=bac, include_bias=False),
                                 StandardScaler(), Ridge(**p))
        if self.ten == "svr":
            return MultiOutputRegressor(SVR(**p), n_jobs=1)
        if self.ten == "rf":
            p.setdefault("n_jobs", 1)
            p["random_state"] = self.seed
            return RandomForestRegressor(**p)
        if self.ten == "gb":
            p["random_state"] = self.seed
            return MultiOutputRegressor(GradientBoostingRegressor(**p), n_jobs=1)
        if self.ten == "mlp":
            so_buoc = p.pop("so_buoc", 6000)
            if not isinstance(so_buoc, (int, np.integer)) or so_buoc < 1:
                raise ValueError("so_buoc phải là số nguyên dương")
            co_lo = min(200, so_hang)
            so_lo = ceil(so_hang / co_lo)
            so_vong = max(1, ceil(so_buoc / so_lo))
            # Không chia ngẫu nhiên các hàng ảo và không dừng sớm theo loss.
            p.update(batch_size=co_lo, max_iter=so_vong, solver="adam",
                     early_stopping=False, tol=0.0,
                     n_iter_no_change=so_vong + 1, random_state=self.seed)
            return MLPRegressor(**p)
        p.setdefault("kernel", ConstantKernel(1.0, (1e-3, 1e3))
                     * RBF(np.ones(self.bo_ma_hoa_.so_chieu), (1e-2, 1e3))
                     + WhiteKernel(0.01, (1e-6, 1.0)))
        p.setdefault("n_restarts_optimizer", 0)
        if not 0 <= p["n_restarts_optimizer"] <= 3:
            raise ValueError("GPR chỉ hỗ trợ n_restarts_optimizer từ 0 đến 3")
        # y đã được chuẩn hóa bên ngoài; không chuẩn hóa lại trong GPR.
        p.update(normalize_y=False, random_state=self.seed)
        return [GaussianProcessRegressor(**p) for _ in range(2)]

    def fit(self, bang_hoc: pd.DataFrame,
            bang_that_hoc: pd.DataFrame) -> "MoHinhSklearn":
        self.bo_ma_hoa_ = BoMaHoa(self.kieu_ma_hoa, self.lam).fit(bang_that_hoc)
        x = self.bo_ma_hoa_.transform(bang_hoc)
        y = ma_hoa_y(bang_hoc)
        self.y_trung_binh_ = y.mean(axis=0)
        self.y_do_lech_ = y.std(axis=0)
        self.y_do_lech_[self.y_do_lech_ == 0] = 1.0
        y = (y - self.y_trung_binh_) / self.y_do_lech_
        self.mo_hinh_ = self._tao_bo_hoi_quy(len(bang_hoc))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.ten == "gpr":
                for cot, mo_hinh in enumerate(self.mo_hinh_):
                    mo_hinh.fit(x, y[:, cot])
            else:
                self.mo_hinh_.fit(x, y)
        self.so_hang_hoc_ = len(bang_hoc)
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "so_hang_hoc_"):
            raise NotFittedError("Cần gọi fit trước predict")
        x = self.bo_ma_hoa_.transform(X)
        if self.ten == "gpr":
            du_doan = [mo_hinh.predict(x, return_std=True) for mo_hinh in self.mo_hinh_]
            y = np.column_stack([cap[0] for cap in du_doan])
            sd_log = np.column_stack([cap[1] for cap in du_doan]) * self.y_do_lech_
        else:
            y = self.mo_hinh_.predict(x)
            sd_log = None
        csp, vbar = giai_ma_y(y * self.y_do_lech_ + self.y_trung_binh_)
        if sd_log is None:
            return khung_du_doan(X, csp, vbar)
        return khung_du_doan(X, csp, vbar, sd_log[:, 0], sd_log[:, 1])

    def hoc_tiep(self, bang_hoc_moi: pd.DataFrame, so_vong: int = 1) -> float:
        """MLP học tiếp đủ so_vong; trả MSE chuẩn hóa trên bảng mới."""
        if self.ten != "mlp":
            raise ValueError(f"hoc_tiep chỉ hỗ trợ 'mlp', không hỗ trợ {self.ten!r}")
        if not hasattr(self, "so_hang_hoc_"):
            raise NotFittedError("Cần gọi fit trước hoc_tiep")
        if not isinstance(so_vong, (int, np.integer)) or so_vong < 1:
            raise ValueError("so_vong phải là số nguyên dương")
        x = self.bo_ma_hoa_.transform(bang_hoc_moi)
        y = (ma_hoa_y(bang_hoc_moi) - self.y_trung_binh_) / self.y_do_lech_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(so_vong):
                self.mo_hinh_.partial_fit(x, y)
        return float(np.mean((self.mo_hinh_.predict(x) - y) ** 2))


def tao_mo_hinh(ten: str, tham_so: dict | None = None, **kw) -> MoHinhSklearn:
    """Tạo mô hình theo tên thống nhất của bộ chạy."""
    return MoHinhSklearn(ten, tham_so, **kw)
