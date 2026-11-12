"""GNG-R: đồ thị Fritzke với lớp ra hồi quy chuẩn hóa (4.7)–(4.8).

Nút mới nằm giữa nút lỗi lớn nhất và láng giềng lỗi lớn nhất; tâm và trọng số
ra đều được nội suy. Lỗi tăng trưởng GNG là khoảng cách đầu vào bình phương,
không phải lỗi hồi quy. Độ rộng Gaussian bằng trung bình độ dài cạnh hiện tại,
có sàn ``sigma_min``. Cạnh không tồn tại có tuổi -1; mảng cấp phát theo ngân sách
nút, các nút đang dùng nằm trong một tiền tố liên tục.

Dùng ``MoHinhDoThi('gng_r'|'pgcn_r', tham_so=..., seed=...)`` cho bảng dữ liệu.
Các tham số lõi dùng tên không dấu; ``lam_chen`` là chu kỳ chèn, KHÔNG phải
``lam`` của bộ mã hóa one-hot. Không có khoảng dự đoán xác suất.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .giao_dien import MoHinhCoSo, khung_du_doan
from .ma_hoa import BoMaHoa, giai_ma_y, ma_hoa_y


class GNGR:
    """Lõi numpy; ``fit(X, y)``, ``hoc_tiep(X, y, so_vong=1)``, ``predict(X)``.

    X hữu hạn (n, d), y hữu hạn (n, 2); lần fit đầu cần ít nhất hai mẫu.
    MSE là trung bình trên cả hai đầu ra, được đo lại cuối mỗi vòng trên đúng
    tập đang học. Dừng sớm không giữ lại dữ liệu hay xem tập kiểm tra.
    """

    def __init__(self, eta_b=0.05, eta_n=0.005, eta_o=0.2,
                 lam_chen=100, tuoi_canh_max=100, so_nut_max=300,
                 alpha=0.5, he_so_giam_loi=0.995, sigma_min=0.025,
                 so_vong=4, nguong_mse=0.002, seed=0):
        for ten, gia_tri in (("eta_b", eta_b), ("eta_n", eta_n), ("eta_o", eta_o),
                             ("alpha", alpha), ("he_so_giam_loi", he_so_giam_loi)):
            if not np.isfinite(gia_tri) or not 0 < gia_tri <= 1:
                raise ValueError(f"{ten} phải nằm trong (0, 1]")
        for ten, gia_tri, can in (("lam_chen", lam_chen, 1),
                                  ("tuoi_canh_max", tuoi_canh_max, 1),
                                  ("so_nut_max", so_nut_max, 2), ("so_vong", so_vong, 1)):
            if not isinstance(gia_tri, (int, np.integer)) or gia_tri < can:
                raise ValueError(f"{ten} phải là số nguyên >= {can}")
        if not np.isfinite(sigma_min) or sigma_min <= 0:
            raise ValueError("sigma_min phải dương và hữu hạn")
        if not np.isfinite(nguong_mse) or nguong_mse < 0:
            raise ValueError("nguong_mse phải không âm và hữu hạn")
        self.eta_b, self.eta_n, self.eta_o = eta_b, eta_n, eta_o
        self.lam_chen, self.tuoi_canh_max = lam_chen, tuoi_canh_max
        self.so_nut_max, self.alpha = so_nut_max, alpha
        self.he_so_giam_loi, self.sigma_min = he_so_giam_loi, sigma_min
        self.so_vong, self.nguong_mse, self.seed = so_vong, nguong_mse, seed

    @staticmethod
    def _kiem_tra(X, y=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X phải là ma trận hữu hạn, có ít nhất một cột")
        if y is None:
            return X
        y = np.asarray(y, dtype=float)
        if not len(X) or y.shape != (len(X), 2) or not np.isfinite(y).all():
            raise ValueError("y phải hữu hạn, dạng (n, 2), n > 0")
        return X, y

    def _khoi_tao(self, X, y):
        m, d = self.so_nut_max, X.shape[1]
        self.c = np.zeros((m, d))
        self.w = np.zeros((m, 2))
        self.loi = np.zeros(m)
        self.luc_sinh = np.zeros(m, dtype=np.int64)
        self.tuoi_canh = np.full((m, m), -1, dtype=np.int64)
        self._do_dai = np.zeros((m, m))
        self._tong_dai = np.zeros(m)
        self._bac = np.zeros(m, dtype=np.int64)
        self._hieu = np.empty((m, d))
        self._kc = np.empty(m)
        self._a = np.empty(m)
        self._sigma = np.empty(m)
        self._delta_w = np.empty((m, 2))
        self._toc_do = np.ones(m)
        self._o = np.empty(2)
        self._du = np.empty(2)
        self.so_nut, self.so_buoc = 2, 0
        self.c[:2], self.w[:2] = X[:2], y[:2]
        self._noi(0, 1)
        self._bi_keo = np.empty(m, dtype=np.intp)
        self._hieu_canh = np.empty((m, m, d))
        self._kc_canh = np.empty((m, m))
        self._rng = np.random.default_rng(self.seed)

    @property
    def so_canh(self):
        return int(self._bac[:self.so_nut].sum() // 2) if hasattr(self, "_bac") else 0

    def _noi(self, i, j):
        if self.tuoi_canh[i, j] < 0:
            dai = float(np.linalg.norm(self.c[i] - self.c[j]))
            self._do_dai[i, j] = self._do_dai[j, i] = dai
            self._tong_dai[i] += dai
            self._tong_dai[j] += dai
            self._bac[i] += 1
            self._bac[j] += 1
        self.tuoi_canh[i, j] = self.tuoi_canh[j, i] = 0

    def _cat(self, i, j):
        dai = self._do_dai[i, j]
        self._tong_dai[i] -= dai
        self._tong_dai[j] -= dai
        self._bac[i] -= 1
        self._bac[j] -= 1
        self._do_dai[i, j] = self._do_dai[j, i] = 0.0
        self.tuoi_canh[i, j] = self.tuoi_canh[j, i] = -1

    def _keo(self, s, ke, x):
        """Kéo cùng lúc và cập nhật độ dài cạnh chỉ ở các hàng bị tác động."""
        n, k = self.so_nut, len(ke) + 1
        self.c[s] += self.eta_b * self._toc_do[s] * (x - self.c[s])
        self.c[ke] += self.eta_n * self._toc_do[ke, None] * (x - self.c[ke])
        self._bi_keo[0], self._bi_keo[1:k] = s, ke
        idx = self._bi_keo[:k]
        hieu, dai = self._hieu_canh[:k, :n], self._kc_canh[:k, :n]
        np.subtract(self.c[idx, None, :], self.c[None, :n, :], out=hieu)
        np.einsum("ijk,ijk->ij", hieu, hieu, out=dai)
        np.sqrt(dai, out=dai)
        dai *= self.tuoi_canh[idx, :n] >= 0
        self._do_dai[idx, :n] = dai
        self._do_dai[:n, idx] = dai.T
        np.sum(self._do_dai[:n, :n], axis=1, out=self._tong_dai[:n])

    def _kich_hoat(self, x):
        n = self.so_nut
        np.subtract(self.c[:n], x, out=self._hieu[:n])
        np.einsum("ij,ij->i", self._hieu[:n], self._hieu[:n], out=self._kc[:n])
        np.divide(self._tong_dai[:n], np.maximum(self._bac[:n], 1), out=self._sigma[:n])
        np.maximum(self._sigma[:n], self.sigma_min, out=self._sigma[:n])
        np.square(self._sigma[:n], out=self._sigma[:n])
        np.divide(self._kc[:n], self._sigma[:n], out=self._a[:n])
        self._a[:n] *= -0.5
        np.exp(self._a[:n], out=self._a[:n])
        tong = self._a[:n].sum()
        if tong < np.finfo(float).tiny:
            self._a[:n] = 0.0
            self._a[int(np.argmin(self._kc[:n]))] = 1.0
        else:
            self._a[:n] /= tong
        np.dot(self._a[:n], self.w[:n], out=self._o)

    def _toc_do_nut(self):
        self._toc_do[:self.so_nut] = 1.0

    def _chon_chen(self, thang, mse):
        if self.so_buoc % self.lam_chen == 0:
            return int(np.argmax(self.loi[:self.so_nut]))
        return -1

    def _nut_moi(self, r, q, f):
        """Điểm mở rộng cho thống kê ngắn/dài hạn của PGCN-R."""

    def _chen(self, q):
        ke = np.flatnonzero(self.tuoi_canh[q, :self.so_nut] >= 0)
        if not len(ke) or self.so_nut >= self.so_nut_max:
            return
        f = int(ke[np.argmax(self.loi[ke])])
        r = self.so_nut
        self.c[r] = (self.c[q] + self.c[f]) * 0.5
        self.w[r] = (self.w[q] + self.w[f]) * 0.5
        self.luc_sinh[r] = self.so_buoc
        self.loi[q] *= self.alpha
        self.loi[f] *= self.alpha
        self.loi[r] = self.loi[q]
        self.so_nut += 1
        self._cat(q, f)
        self._noi(q, r)
        self._noi(f, r)
        self._nut_moi(r, q, f)

    def _xoa_nut(self, i):
        """Xóa nút cô lập bằng cách chuyển nút cuối vào chỗ trống."""
        cuoi = self.so_nut - 1
        if i != cuoi:
            self.c[i], self.w[i] = self.c[cuoi], self.w[cuoi]
            self.loi[i], self.luc_sinh[i] = self.loi[cuoi], self.luc_sinh[cuoi]
            self._tong_dai[i], self._bac[i] = self._tong_dai[cuoi], self._bac[cuoi]
            self.tuoi_canh[i, :] = self.tuoi_canh[cuoi, :]
            self.tuoi_canh[:, i] = self.tuoi_canh[:, cuoi]
            self._do_dai[i, :] = self._do_dai[cuoi, :]
            self._do_dai[:, i] = self._do_dai[:, cuoi]
            self.tuoi_canh[i, i] = -1
            self._do_dai[i, i] = 0.0
        self.tuoi_canh[cuoi, :] = self.tuoi_canh[:, cuoi] = -1
        self._do_dai[cuoi, :] = self._do_dai[:, cuoi] = 0.0
        self._tong_dai[cuoi], self._bac[cuoi] = 0.0, 0
        self.so_nut -= 1

    def _buoc(self, x, y):
        self._kich_hoat(x)
        n = self.so_nut
        s = int(np.argmin(self._kc[:n]))
        khoang_cach = self._kc[s]
        self._kc[s] = np.inf
        t = int(np.argmin(self._kc[:n]))
        np.subtract(y, self._o, out=self._du)
        mse = float(np.dot(self._du, self._du) * 0.5)
        self.so_buoc += 1
        self._toc_do_nut()
        np.multiply(self._a[:n, None], self._du, out=self._delta_w[:n])
        self._delta_w[:n] *= self.eta_o * self._toc_do[:n, None]
        self.w[:n] += self._delta_w[:n]
        self.loi[s] += khoang_cach
        ke = np.flatnonzero(self.tuoi_canh[s, :n] >= 0)
        self.tuoi_canh[s, ke] += 1
        self.tuoi_canh[ke, s] = self.tuoi_canh[s, ke]
        self._noi(s, t)
        ke = np.flatnonzero(self.tuoi_canh[s, :n] >= 0)
        self._keo(s, ke, x)
        q = self._chon_chen(s, mse)
        if q >= 0:
            self._chen(q)
        het_han = ke[self.tuoi_canh[s, ke] > self.tuoi_canh_max]
        for i in het_han:
            self._cat(s, i)
        if len(het_han):
            for i in np.flatnonzero(self._bac[:self.so_nut] == 0)[::-1]:
                self._xoa_nut(int(i))
        self.loi[:self.so_nut] *= self.he_so_giam_loi

    def fit(self, X, y):
        X, y = self._kiem_tra(X, y)
        if len(X) < 2:
            raise ValueError("Khởi tạo đồ thị cần ít nhất hai mẫu")
        self._khoi_tao(X, y)
        self.hoc_tiep(X, y, self.so_vong)
        return self

    def hoc_tiep(self, X, y, so_vong=1):
        if not hasattr(self, "c"):
            raise RuntimeError("Cần fit trước hoc_tiep")
        X, y = self._kiem_tra(X, y)
        if X.shape[1] != self.c.shape[1]:
            raise ValueError("Số chiều X khác lần fit đầu")
        if not isinstance(so_vong, (int, np.integer)) or so_vong < 1:
            raise ValueError("so_vong phải là số nguyên dương")
        self.lich_su_mse_ = []
        for _ in range(so_vong):
            for k in self._rng.permutation(len(X)):
                self._buoc(X[k], y[k])
            self.mse_ = float(np.mean((self.predict(X) - y) ** 2))
            self.lich_su_mse_.append(self.mse_)
            if self.mse_ < self.nguong_mse:
                break
        return self.mse_

    def predict(self, X):
        if not hasattr(self, "c"):
            raise RuntimeError("Cần fit trước predict")
        X = self._kiem_tra(X)
        if X.shape[1] != self.c.shape[1]:
            raise ValueError("Số chiều X khác lần fit đầu")
        n = self.so_nut
        sigma = np.maximum(self._tong_dai[:n] / np.maximum(self._bac[:n], 1), self.sigma_min)
        du_doan = np.empty((len(X), 2))
        for dau in range(0, len(X), 1024):
            x = X[dau:dau + 1024]
            kc = np.sum((x[:, None, :] - self.c[None, :n, :]) ** 2, axis=2)
            a = np.exp(-0.5 * kc / sigma ** 2)
            tong = a.sum(axis=1)
            rong = tong < np.finfo(float).tiny
            a[rong] = 0.0
            a[np.flatnonzero(rong), np.argmin(kc[rong], axis=1)] = 1.0
            tong[rong] = 1.0
            du_doan[dau:dau + len(x)] = (a / tong[:, None]) @ self.w[:n]
        return du_doan


LUOI_DO_THI = {
    "gng_r": [dict(eta_b=b, lam_chen=l, tuoi_canh_max=t)
              for b in (0.03, 0.08) for l in (50, 150) for t in (50, 150)],
    "pgcn_r": [dict(eta_b=b, t_dai=d, tuoi_nut_moi=a, tuoi_canh_max=t)
               for b in (0.03, 0.08) for d in (200, 800)
               for a, t in ((30, 50), (50, 100), (100, 150))],
}


class MoHinhDoThi(MoHinhCoSo):
    """Bọc GNG-R/PGCN-R; cố định mã hóa và thang y sau fit đầu.

    ``tham_so`` chuyển tới GNGR/PGCNR, trừ ``so_mau_hoc`` (mặc định 20000):
    fit đặt số vòng = max(1, ceil(so_mau_hoc / n_hang)). Nhờ vậy ngân sách
    trình bày mẫu không tăng tuyến tính theo số bản nhiễu. Luôn học trọn vòng,
    nên có thể vượt ngân sách dưới n_hang mẫu; khi n_hang > ngân sách vẫn có
    một vòng. Nếu cung cấp ``so_vong`` trong tham_so thì đó là cận trên bổ sung;
    nếu không, mặc định 4 vòng của lõi được thay bằng ngân sách trên. Dừng sớm
    theo MSE vẫn có hiệu lực. ``seed`` do lớp bọc quản lý.
    fit khởi tạo lại; hoc_tiep giữ nghĩa so_vong, không áp dụng so_mau_hoc,
    chỉ xem bảng mới và không phát lại dữ liệu cũ.
    """

    def __init__(self, ten, tham_so=None, lam=1.0, seed=0):
        if ten not in ("gng_r", "pgcn_r"):
            raise ValueError("ten phải là 'gng_r' hoặc 'pgcn_r'")
        self.ten, self.tham_so = ten, dict(tham_so or {})
        self.lam, self.seed = lam, seed

    def fit(self, bang_hoc: pd.DataFrame, bang_that_hoc: pd.DataFrame):
        tham_so = dict(self.tham_so)
        so_mau_hoc = tham_so.pop("so_mau_hoc", 20000)
        if not isinstance(so_mau_hoc, (int, np.integer)) or so_mau_hoc < 1:
            raise ValueError("so_mau_hoc phải là số nguyên dương")
        if not len(bang_hoc):
            raise ValueError("bang_hoc phải có ít nhất hai mẫu")
        so_vong = max(1, (so_mau_hoc + len(bang_hoc) - 1) // len(bang_hoc))
        if "so_vong" in tham_so:
            gioi_han = tham_so["so_vong"]
            if not isinstance(gioi_han, (int, np.integer)) or gioi_han < 1:
                raise ValueError("so_vong phải là số nguyên dương")
            so_vong = min(so_vong, gioi_han)
        tham_so["so_vong"] = so_vong
        self.bo_ma_hoa_ = BoMaHoa(kieu="onehot", lam=self.lam).fit(bang_that_hoc)
        X, y = self.bo_ma_hoa_.transform(bang_hoc), ma_hoa_y(bang_hoc)
        self.y_tb_ = y.mean(axis=0)
        self.y_sd_ = y.std(axis=0)
        self.y_sd_[self.y_sd_ == 0] = 1.0
        if self.ten == "pgcn_r":
            from .pgcn_r import PGCNR
            lop = PGCNR
        else:
            lop = GNGR
        self.loi_ = lop(**tham_so, seed=self.seed)
        self.loi_.fit(X, (y - self.y_tb_) / self.y_sd_)
        self.so_hang_hoc_ = len(bang_hoc)
        return self

    def predict(self, X: pd.DataFrame):
        if not hasattr(self, "loi_"):
            raise RuntimeError("Cần fit trước predict")
        y = self.loi_.predict(self.bo_ma_hoa_.transform(X)) * self.y_sd_ + self.y_tb_
        csp, vbar = giai_ma_y(y)
        return khung_du_doan(X, csp, vbar)

    def hoc_tiep(self, bang_hoc_moi: pd.DataFrame, so_vong=1):
        if not hasattr(self, "loi_"):
            raise RuntimeError("Cần fit trước hoc_tiep")
        X = self.bo_ma_hoa_.transform(bang_hoc_moi)
        y = (ma_hoa_y(bang_hoc_moi) - self.y_tb_) / self.y_sd_
        mse = self.loi_.hoc_tiep(X, y, so_vong)
        self.so_hang_hoc_ += len(bang_hoc_moi)
        return mse

    @property
    def so_nut(self):
        return self.loi_.so_nut if hasattr(self, "loi_") else 0

    @property
    def so_canh(self):
        return self.loi_.so_canh if hasattr(self, "loi_") else 0

    @property
    def so_buoc(self):
        return self.loi_.so_buoc if hasattr(self, "loi_") else 0
