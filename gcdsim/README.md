# GCD-Sim: sinh dữ liệu giả lập sạc–xả (GCD)

GCD-Sim mô phỏng độ bất định quanh số liệu của điện cực siêu tụ graphene/hạt nano (lab NTM.Hồng & LD.Phúc), không tạo thêm bằng chứng thực nghiệm. Mặc định: **8 điện cực × 6 điểm × 10000 lần đo ảo = 480.000 mẫu**. Một lần đo ảo của một điện cực gồm cả 6 mức dòng, dùng chung sai số cân khối lượng.

Chỉ có giao diện dòng lệnh Python; không cần trình duyệt hoặc máy chủ web.

## Cài đặt

Dùng Python 3.10+ (máy local của dự án dùng Python 3.12), NumPy 2.0+ để có `np.trapezoid`. Chạy mọi lệnh từ **thư mục `Code/`**, nơi chứa `data/`, `gcdsim/`, `ml/`:

```bash
python -X utf8 -m pip install -r requirements.txt
```

## Ba lệnh tái lập chuẩn

```bash
# 480.000 mẫu, seed 42; bỏ --n nghĩa là 10000 lần đo ảo/điện cực
python -X utf8 -m gcdsim.gcdsim data/so_lieu_GCD.xlsx --out ket_qua/gcdsim/du_lieu_ao_480000.xlsx --seed 42

# Bốn hình kiểm chứng Chương 3 và bảng kiểm chứng tổng hợp
python -X utf8 -m gcdsim.ve_hinh data/so_lieu_GCD.xlsx ket_qua/gcdsim/hinh

# Độ nhạy: 15 cấu hình, mỗi cấu hình N=10000, seed 42
python -X utf8 -m gcdsim.do_nhay data/so_lieu_GCD.xlsx ket_qua/gcdsim/do_nhay.csv
```

Excel có các sheet `du_lieu_ao`, `du_lieu_goc`, `quy_luat_khop`, `cau_hinh`, `kiem_chung_tung_diem`, `kiem_chung_tong_hop`. Chương trình in JSON kiểm chứng và số mẫu đã ghi. Excel được ghi tuần tự để không giữ toàn bộ hơn 6 triệu ô trong bộ nhớ; kiểm tra đường cong chỉ lấy 200 mẫu, không sinh 480.000 đường cong.

Xuất Excel/CSV vào tệp tạm cùng thư mục (ví dụ `du_lieu_ao_480000.tmp.xlsx`), chỉ thay tệp đích bằng `os.replace` khi ghi xong. Nếu có lỗi, chương trình dọn tệp tạm và giữ nguyên sản phẩm cũ thay vì để lại tệp đích ghi dở.

**Vị trí dữ liệu:** tệp Excel seed 42 hiện có kích thước **53.520.383 byte (53,52 MB; 51,04 MiB)**, vượt ngưỡng 50 MB nên chỉ nằm ở `ket_qua/gcdsim/du_lieu_ao_480000.xlsx`, không có bản sao trong `data/` và không đưa vào Git. Dùng lệnh đầu để sinh lại, hoặc lấy `gcdsim/du_lieu_ao_480000.xlsx` trong sản phẩm **Giai đoạn 1** (`san_pham_giai_doan_1.zip`) nếu Colab đã chạy ba lệnh chuẩn trước khi đóng gói.

Trong **Giai đoạn 1**, chạy ba lệnh này trên Google Colab và thay `ket_qua` bằng thư mục kết quả `THU_MUC` trên Drive để đóng gói cùng sản phẩm huấn luyện. **Giai đoạn 2** là nhận sản phẩm Colab trên máy local để sử dụng ứng dụng terminal và hoàn thiện báo cáo; không cần sinh lại nếu sản phẩm đã chứa đủ tệp. Xem quy trình tổng thể và `ml.giai_doan_2` trong `../README.md`.

### Tùy chọn khác

```bash
# CSV dạng dài làm đầu vào, đổi mức nhiễu
python -X utf8 -m gcdsim.gcdsim data/so_lieu_goc_dang_dai.csv --out ket_qua/gcdsim/ao_rep5.csv --repeat-err 0.05 --mass-err 0.02 --meas-err 0.01

# Chế độ khớp: sinh ở 12 mức dòng trong khoảng đã đo
python -X utf8 -m gcdsim.gcdsim data/so_lieu_goc_dang_dai.csv --mode fit --j-mode range --out ket_qua/gcdsim/ao_noi_suy.csv

python -X utf8 -m gcdsim.gcdsim --help
```

`--n` là số lần đo ảo **mỗi điện cực**, không phải tổng số hàng. Các mức nhiễu nhập theo tỉ lệ (`0.03` = 3 %). CSV không chứa các sheet kiểm chứng như Excel.

## Thuật toán

Chế độ mặc định `anchor` neo đúng vào số đo gốc. Với mỗi điện cực và mỗi lần đo ảo r = 1..N:

1. Lấy sai số cân khối lượng δ ~ N(0, σ_m), **dùng chung cho cả 6 mức dòng**. Hệ số g = 1/(1+δ).
2. Ở từng mức dòng J_k:
   - Csp* = Csp_gốc · exp(N(0, σ_lặp)) · (1 + N(0, σ_tb)) · g.
   - J* = J_k · g.
   - V̄* = V̄_gốc · exp(N(0, σ_lặp)) · (1 + N(0, σ_tb)), với V̄_gốc = Pd/(1000·J). Chặn V̄* trong [0,02; 0,98·ΔV].
3. Tính đúng công thức lab: Ed = Csp·ΔV²/7,2; Pd = 1000·J·V̄; Δt = Ed·3600/Pd.

Mặc định σ_lặp = 3 %, σ_m = 2 %, σ_tb = 1 %, ΔV = 2,1 V. **Các mức nhiễu là giả định cần phòng thí nghiệm xác nhận**, lý tưởng bằng phép đo lặp. Tăng N chỉ làm ổn định mô phỏng, không tăng số điểm thực nghiệm độc lập.

Chế độ `fit` dùng đường khớp Csp(J) thay điểm neo: chọn hàm mũ, lũy thừa hoặc Dunn có MAPE LOOCV nhỏ nhất; khớp tuyến tính V̄(J). Dùng để nội suy hoặc khảo sát ngoại suy; cần kiểm tra đường khớp trước khi sử dụng, không tự loại điểm khác xu hướng khi chưa được lab xác nhận. `--repeat-from-data` dùng độ lệch sau khớp làm nhiễu lặp thay mức cố định.

Đường cong xả giả định V(t) = Vs·(1 − t/Δt)^p, chọn p để điện thế trung bình bằng V̄. Tính lại Csp bằng tích phân công thức lab. Đây là **giả định hình dạng**, không phải đường cong GCD đo thật; nhiễu điện thế và tích phân số làm sai số kiểm chứng khác 0.

## Các cột dữ liệu ảo

| Cột | Ý nghĩa |
|---|---|
| `sample_id` | Số thứ tự mẫu ảo trong bảng |
| `ratio` | Tỉ lệ hạt nano/graphene của điện cực |
| `run_id` | Lần đo ảo (0..N−1), xác định cùng `ratio` |
| `J_nominal` | Mật độ dòng danh định (A/g); ở chế độ anchor là mức đo gốc |
| `J` | Mật độ dòng sau sai số cân (A/g) |
| `Csp`, `Ed`, `Pd` | Điện dung riêng (F/g), mật độ năng lượng (Wh/kg), mật độ công suất (W/kg) |
| `Vbar` | Điện thế xả trung bình (V) |
| `dt_discharge` | Thời gian xả (s) |
| `mass_error` | Sai số cân tương đối, chung cho cả lần đo ảo |
| `extrapolated` | True nếu mức dòng vượt khoảng đã đo (chỉ chế độ fit) |
| `source` | `synthetic`: dữ liệu giả lập |

## Kiểm chứng: N=10000, seed 42

Các giá trị sau được đo từ lần sinh chuẩn nêu trên, không phải kết quả dự đoán học máy. Các chỉ số kiểm chứng lấy từ sheet **`kiem_chung_tong_hop`** của lần chạy này; riêng độ phân tán lấy ở dòng mặc định (σ_lặp = 3 %, σ_m = 2 %) của `do_nhay.csv` vì sheet tổng hợp không chứa chỉ số đó.

| Chỉ số | Kết quả mới |
|---|---|
| Số điểm thật → số mẫu ảo | 48 → 480.000 |
| Độ phân tán Csp quanh điểm neo | 3,7465 % (xấp xỉ 3,75 %) |
| MAPE của trung vị Csp ảo so với số thật | 0,03 % |
| Điểm thật trong dải 5–95 % của Csp / Pd ảo | 100,0 % / 100,0 % (48/48 cho mỗi đại lượng) |
| Lần đo ảo giữ đúng mọi bước chuyển xu hướng Csp theo J | 95,8 % (trung bình trên 8 điện cực) |
| Giữ điện cực 0,1 tốt nhất | 100,0 % |
| Sai số tính lại Csp từ đường cong, lớn nhất / trung bình | 0,085 % / 0,013 % (200 mẫu) |
| Ed = 0,6125·Csp | Đúng ở mọi mẫu |

Độ phân tán = trung bình trên 48 điểm của `100 × std(Csp_ảo, ddof=1) / Csp_gốc`, nhóm theo `(ratio, J_nominal)`. MAPE trung vị là **trung bình sai số tuyệt đối tương đối của 48 trung vị**, không phải trung vị của các sai số. Xu hướng xét đồng thời cả 5 bước chuyển của mỗi lần đo ảo; riêng điện cực 1,75 giữ xu hướng 67,76 %. Thứ hạng so ở **mức dòng đầu của từng điện cực**, không phải cùng J.

Trong 15 cấu hình độ nhạy (σ_lặp = 1–5 %, σ_m = 2–4,5 %, σ_tb = 1 %), độ phân tán từ 2,4547 % đến 6,8449 %; giữ xu hướng từ 98,2 % xuống 92,9 %; điện cực 0,1 vẫn đứng đầu 100,0 %. Đây là độ nhạy đối với giả định nhiễu, không chứng minh các mức nhiễu đó đúng với thiết bị lab.

Thời gian đo: **~59 giây local cho cả lệnh sinh Excel (sau khi ghi tuần tự)**, cụ thể 59,26 giây; cả lệnh tạo hình và kiểm chứng 6,21 giây; cả lệnh độ nhạy 14,89 giây. Thời gian phụ thuộc máy, không phải cam kết hiệu năng.

## Lưu ý cho học máy

**Không chia ngẫu nhiên 480.000 mẫu ảo thành tập học và tập kiểm tra.** Phải chia số đo thật trước, rồi chỉ khớp hàm, chuẩn hóa và sinh mẫu từ phần huấn luyện; gói `ml/` thực hiện tại từng lần chia. Bảng Chương 3 neo trên cả 48 điểm không được dùng làm tập học cho phép kiểm tra trên chính 48 điểm ấy.

`sample_id`, `run_id`, `J_nominal`, `source` là dấu vết nguồn sinh, không phải đầu vào vật lý của mô hình. `ratio` là thông tin thành phần điện cực và được gói ML dùng làm đặc trưng; việc chia nhóm phải ngăn rò rỉ giữa điểm thật và các bản ảo của nó. Xem hợp đồng đặc trưng của `ml/` trước khi sử dụng.
