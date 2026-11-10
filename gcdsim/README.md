# GCD-Sim: công cụ sinh dữ liệu giả lập sạc–xả (GCD)

Sinh dữ liệu ảo từ số liệu thực nghiệm của điện cực siêu tụ graphene/hạt nano (lab NTM.Hồng & LD.Phúc).
Mặc định: 48 điểm gốc × 1000 lần đo ảo = **48.000 mẫu**. Mỗi mẫu ảo mô phỏng độ bất định quanh số gốc theo các mức sai số giả định.

Giao diện sử dụng chính là terminal. Không cần trình duyệt hoặc máy chủ web.

## Cài đặt
```
python -m pip install -r requirements.txt
```
Cần Python 3.10 trở lên và NumPy 2.0 trở lên. Không cần thư viện học máy hay mạng nơ-ron.

## Chạy
```
# Sinh 48.000 mẫu ảo, xuất Excel nhiều sheet (dữ liệu ảo, dữ liệu gốc, cấu hình, kiểm chứng)
python gcdsim.py data/so_lieu_GCD.xlsx --out du_lieu_ao.xlsx

# Xuất CSV, đổi số lần đo ảo và mức sai số
python gcdsim.py data/so_lieu_GCD.xlsx --out du_lieu_ao.csv --n 1000 --repeat-err 0.03 --mass-err 0.02 --meas-err 0.01

# Chế độ theo quy luật khớp (sinh ở các mức dòng chưa đo, 12 mức trong khoảng đã đo)
python gcdsim.py data/so_lieu_GCD.xlsx --mode fit --j-mode range --out du_lieu_ao_noi_suy.csv

# Vẽ hình kiểm chứng cho báo cáo
python ve_hinh.py data/so_lieu_GCD.xlsx out/hinh
```
Xem toàn bộ tùy chọn: `python gcdsim.py --help`. Trên Windows, có thể dùng `python -X utf8` để in tiếng Việt ổn định.

Các tệp `web/` và ảnh chụp giao diện cũ được giữ lại làm bản lưu, không thuộc quy trình chạy hiện tại.

## Thuật toán (chế độ mặc định "neo theo số gốc")
Với mỗi điện cực và mỗi lần đo ảo r = 1..N:
1. Sai số cân khối lượng δ ~ N(0, σ_m), **dùng chung cho cả 6 mức dòng** (mỗi điện cực chỉ cân một lần). Hệ số g = 1/(1+δ).
2. Với từng mức dòng J_k của số gốc:
   - Csp* = Csp_gốc · exp(N(0, σ_lặp)) · (1 + N(0, σ_tb)) · g
   - J* = J_k · g
   - V̄* = V̄_gốc · exp(N(0, σ_lặp)) · (1 + N(0, σ_tb)),  với V̄_gốc = Pd/(1000·J)
3. Tính các đại lượng còn lại bằng đúng công thức của lab:
   Ed = Csp·ΔV²/7,2 ;  Pd = 1000·J·V̄ ;  Δt = Ed·3600/Pd.

Mặc định: σ_lặp = 3 %, σ_m = 2 %, σ_tb = 1 %, ΔV = 2,1 V. **Các mức này là giả định, cần cô Hồng xác nhận** (lý tưởng nhất là đo lặp 3 lần một điện cực để ước lượng thật).

Chế độ `fit`: thay số gốc bằng giá trị trên đường khớp Csp(J) (chọn giữa hàm mũ, lũy thừa, Dunn theo sai số kiểm tra chéo LOOCV) để sinh ở các mức dòng chưa đo.

Đường cong V(t): nhánh xả V(t) = Vs·(1 − t/Δt)^p, với p chọn để điện thế trung bình bằng V̄, nên diện tích dưới đường cong khớp đúng Csp. Hình dạng này là giả định; sẽ thay bằng đường cong thật khi có file thô từ máy đo.

## Các cột của file dữ liệu ảo
| Cột | Ý nghĩa |
|---|---|
| sample_id | số thứ tự mẫu ảo |
| ratio | tỉ lệ hạt nano/graphene của điện cực |
| run_id | số thứ tự lần đo ảo (0..N−1) |
| J_nominal | mật độ dòng danh định (đúng mức đo gốc) |
| J | mật độ dòng sau khi tính sai số cân (A/g) |
| Csp, Ed, Pd | điện dung riêng (F/g), mật độ năng lượng (Wh/kg), mật độ công suất (W/kg) |
| Vbar | điện thế xả trung bình (V) |
| dt_discharge | thời gian xả (s) |
| mass_error | sai số cân khối lượng của lần đo ảo |
| extrapolated | True nếu mức dòng nằm ngoài khoảng đã đo (chỉ ở chế độ fit) |

## Kiểm chứng (số liệu cô Hồng, seed = 42)
- 48 điểm gốc → 48.000 mẫu ảo; độ phân tán quanh mỗi điểm gốc ≈ 3,7 %
- Trung vị dữ liệu ảo lệch số gốc: 0,08 % (Csp)
- 96,2 % lần đo ảo giữ đúng xu hướng Csp theo J như số gốc
- 100 % lần đo ảo vẫn cho tỉ lệ 0,1 là điện cực tốt nhất
- Tính lại Csp từ đường cong ảo bằng công thức lab: sai số lớn nhất 0,09 %
- Ed = 0,6125 × Csp đúng ở mọi mẫu (như số gốc)
