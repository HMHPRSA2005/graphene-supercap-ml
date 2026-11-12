"""PGCN-R tự cài theo GNG/LLCS, KHÔNG phải mã PGCN gốc của Tân.

Mã gốc PGCN của Tân chưa có trong kho. Vì vậy bản này cần được đối chiếu với
khóa luận Tân trước khi diễn giải là tái lập thuật toán gốc.

Kế thừa: đồ thị GNG, hai nút thắng, cạnh có tuổi, xóa cạnh già/nút cô lập,
chèn tại trung điểm với láng giềng lỗi lớn, và cơ chế kiểu LLCS gồm lỗi ngắn/
dài hạn, điều kiện không cải thiện, tuổi nút tối thiểu, ổn định hóa nút cũ.
Phần đổi: đầu ra hai chiều hồi quy chuẩn hóa log(Csp), log(Vbar), Gaussian
chuẩn hóa và quy tắc delta (4.7)–(4.8), thay cho đầu ra phân loại. GNGR cung
cấp lớp hồi quy chung cho đối chứng; PGCNR thêm giảm tốc độ học theo từng nút
cho cả tâm và trọng số đầu ra nhằm hạn chế quên.

Bảng tên tham số CẦN ĐỐI CHIẾU với khóa luận Tân (giá trị mặc định tự chọn):

=================== =================== ====================================
Ký hiệu             Tên Python          Mặc định / ý nghĩa
=================== =================== ====================================
η_b                 eta_b               0.05, tốc độ kéo nút thắng
η_n                 eta_n               0.005, tốc độ kéo láng giềng
η_o                 eta_o               0.2, tốc độ cập nhật lớp ra hồi quy
T_S                 t_ngan              20, cửa sổ EMA ngắn (số lần thắng)
T_L                 t_dai               200, cửa sổ EMA dài (số lần thắng)
θ_chèn              nguong_chen         0.02, MSE dài hạn chuẩn hóa mỗi nút
δ                   delta               0.05, dung sai không cải thiện
Mnew_node_age       tuoi_nut_moi        50, số lần nút thắng kể từ lúc tạo
T_d                 tuoi_canh_max       100, tuổi cạnh theo lần nút đầu thắng
Số nút tối đa       so_nut_max          300, còn chỗ tăng trưởng sau đợt đầu
Ngưỡng MSE          nguong_mse          0.002, MSE tập học cuối mỗi vòng
=================== =================== ====================================

Quyết định tự cài: EMA có hệ số 1/T và khởi tạo bằng lỗi đầu tiên; lỗi là MSE
trên hai đầu ra đã chuẩn hóa, trước cập nhật. Chỉ nút thắng có tuổi đủ lớn,
eL > θ_chèn và eS >= eL*(1-δ) được kích hoạt chèn (không dùng chu kỳ GNG).
Nút mới nhận EMA trung bình hai cha nhân alpha; EMA của hai cha cũng nhân
alpha như lỗi tích lũy GNG. Tuổi dùng cho chèn/ổn định là số lần chính nút đó
thắng, không phải số mẫu toàn cục; không đặt lại bộ đếm cha khi chèn. Chỉ nút
đã có lỗi và đang học tốt (eS <= eL) giảm hệ số học theo
max(he_so_hoc_min, 1/(1+so_lan_thang/T_L)); mặc định he_so_hoc_min=0.2.
Nút chưa ổn định dùng hệ số 1. Không bình phương, không đóng băng nút già;
việc học ở vùng khác không làm già nút chưa tham gia. Hệ số có sàn áp dụng
cho cả tâm và lớp ra. Không cam kết chống quên hoàn toàn. Dừng theo MSE tập học,
không phải tập kiểm tra, không phát lại dữ liệu cũ. ``lam_chen`` của lõi không điều
khiển chèn PGCN-R. ``so_vong``, ``sigma_min``, ``alpha`` và
``he_so_giam_loi`` giữ nghĩa trong GNGR.

MoHinhDoThi và LUOI_DO_THI được xuất từ cả gng_r và pgcn_r; chỉ có một lớp
bọc dùng chung. Ví dụ: MoHinhDoThi('pgcn_r', {'t_ngan': 20, 't_dai': 200}).
Lớp bọc nhận ``tham_so['so_mau_hoc']`` (mặc định 20000), tính số vòng fit
bằng max(1, ceil(so_mau_hoc/n_hang)); ``tham_so['so_vong']`` nếu được cung cấp
là cận trên bổ sung. Luôn học nguyên vòng, nên ngân sách có thể vượt dưới
n_hang mẫu. hoc_tiep vẫn nhận số vòng trực tiếp, không áp dụng ngân sách fit.
"""
from __future__ import annotations

import numpy as np

from .gng_r import GNGR, LUOI_DO_THI, MoHinhDoThi

__all__ = ["PGCNR", "MoHinhDoThi", "LUOI_DO_THI"]


class PGCNR(GNGR):
    """Mạng tăng trưởng có giám sát lỗi và giảm tốc độ học theo tuổi nút."""

    def __init__(self, t_ngan=20, t_dai=200, nguong_chen=0.02,
                 delta=0.05, tuoi_nut_moi=50, he_so_hoc_min=0.2, **kw):
        if not (np.isfinite(t_ngan) and np.isfinite(t_dai) and 1 <= t_ngan < t_dai):
            raise ValueError("Cần 1 <= t_ngan < t_dai, đều hữu hạn")
        if not np.isfinite(nguong_chen) or nguong_chen < 0:
            raise ValueError("nguong_chen phải không âm, hữu hạn")
        if not np.isfinite(delta) or not 0 <= delta < 1:
            raise ValueError("delta phải nằm trong [0, 1)")
        if not isinstance(tuoi_nut_moi, (int, np.integer)) or tuoi_nut_moi < 1:
            raise ValueError("tuoi_nut_moi phải là số nguyên dương")
        if not np.isfinite(he_so_hoc_min) or not 0 < he_so_hoc_min <= 1:
            raise ValueError("he_so_hoc_min phải nằm trong (0, 1]")
        super().__init__(**kw)
        self.t_ngan, self.t_dai = t_ngan, t_dai
        self.nguong_chen, self.delta = nguong_chen, delta
        self.tuoi_nut_moi = tuoi_nut_moi
        self.he_so_hoc_min = he_so_hoc_min

    def _khoi_tao(self, X, y):
        super()._khoi_tao(X, y)
        self.e_ngan = np.zeros(self.so_nut_max)
        self.e_dai = np.zeros(self.so_nut_max)
        self._da_thang = np.zeros(self.so_nut_max, dtype=bool)
        self.so_lan_thang = np.zeros(self.so_nut_max, dtype=np.int64)

    def _toc_do_nut(self):
        n = self.so_nut
        np.divide(self.so_lan_thang[:n], self.t_dai, out=self._toc_do[:n])
        self._toc_do[:n] += 1.0
        np.reciprocal(self._toc_do[:n], out=self._toc_do[:n])
        np.maximum(self._toc_do[:n], self.he_so_hoc_min, out=self._toc_do[:n])
        on_dinh = self._da_thang[:n] & (self.e_ngan[:n] <= self.e_dai[:n])
        self._toc_do[:n][~on_dinh] = 1.0

    def _chon_chen(self, thang, mse):
        self.so_lan_thang[thang] += 1
        if not self._da_thang[thang]:
            self.e_ngan[thang] = self.e_dai[thang] = mse
            self._da_thang[thang] = True
        else:
            self.e_ngan[thang] += (mse - self.e_ngan[thang]) / self.t_ngan
            self.e_dai[thang] += (mse - self.e_dai[thang]) / self.t_dai
        if (self.so_lan_thang[thang] >= self.tuoi_nut_moi
                and self.e_dai[thang] > self.nguong_chen
                and self.e_ngan[thang] >= self.e_dai[thang] * (1 - self.delta)):
            return thang
        return -1

    def _nut_moi(self, r, q, f):
        self.e_ngan[q] *= self.alpha
        self.e_ngan[f] *= self.alpha
        self.e_dai[q] *= self.alpha
        self.e_dai[f] *= self.alpha
        self.e_ngan[r] = (self.e_ngan[q] + self.e_ngan[f]) * 0.5
        self.e_dai[r] = (self.e_dai[q] + self.e_dai[f]) * 0.5
        self._da_thang[r] = True
        self.so_lan_thang[r] = 0

    def _xoa_nut(self, i):
        cuoi = self.so_nut - 1
        self.e_ngan[i], self.e_dai[i] = self.e_ngan[cuoi], self.e_dai[cuoi]
        self._da_thang[i] = self._da_thang[cuoi]
        self.so_lan_thang[i] = self.so_lan_thang[cuoi]
        self.so_lan_thang[cuoi] = 0
        self.e_ngan[cuoi] = self.e_dai[cuoi] = 0.0
        self._da_thang[cuoi] = False
        super()._xoa_nut(i)
