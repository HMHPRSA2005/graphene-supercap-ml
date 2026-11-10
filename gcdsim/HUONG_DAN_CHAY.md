# Chạy GCD-Sim bằng terminal

GCD-Sim sử dụng giao diện dòng lệnh Python. Thư mục triển khai chính là `Code/gcdsim`; không cần chạy frontend hoặc máy chủ web.

## Cài đặt

Cần Python 3.10 trở lên. Mở PowerShell tại thư mục dự án:

```powershell
cd D:\Project\ScientificProject\Code\gcdsim
python -m pip install -r requirements.txt
```

NumPy phải từ 2.0 vì chương trình dùng `np.trapezoid`. `openpyxl` dùng để đọc và ghi Excel; nếu chưa cài gói này, có thể chạy với CSV. `matplotlib` dùng cho bước vẽ hình.

## Sinh dữ liệu

Sinh 48.000 mẫu từ 48 điểm gốc bằng cấu hình mặc định:

```powershell
python -X utf8 gcdsim.py data/so_lieu_goc_dang_dai.csv --out out/du_lieu_ao.csv --n 1000 --seed 42
```

Đọc và xuất Excel nhiều sheet:

```powershell
python -X utf8 gcdsim.py data/so_lieu_GCD.xlsx --out out/du_lieu_ao_48000.xlsx --n 1000 --seed 42
```

Terminal hiển thị bảng khớp quy luật, các chỉ số kiểm chứng và đường dẫn tệp đã ghi. Tệp Excel có các sheet dữ liệu ảo, dữ liệu gốc, quy luật khớp, cấu hình và kiểm chứng. CSV chỉ chứa bảng dữ liệu ảo; có thể lưu thông tin terminal để đối chiếu:

```powershell
python -X utf8 gcdsim.py data/so_lieu_goc_dang_dai.csv --out out/du_lieu_ao.csv | Tee-Object -FilePath out/ket_qua_terminal.txt
```

Với seed 42 và cấu hình mặc định, lần chạy kiểm chứng cho sai lệch trung vị điện dung 0,08%, giữ chiều biến đổi 96,2% và sai số tích phân lớn nhất khoảng 0,089%. Chỉ số thứ hạng so sánh mức dòng đầu của từng điện cực, không phải cùng mật độ dòng.

## Đổi cấu hình và xem trợ giúp

```powershell
python -X utf8 gcdsim.py --help
python -X utf8 gcdsim.py data/so_lieu_goc_dang_dai.csv --out out/ao_rep5.csv --repeat-err 0.05 --mass-err 0.02 --meas-err 0.01
python -X utf8 gcdsim.py data/so_lieu_goc_dang_dai.csv --mode fit --j-mode range --out out/ao_noi_suy.csv
```

Các tham số sai số nhập theo tỉ lệ: `0.03` nghĩa là 3%. Chế độ `anchor` sinh quanh điểm gốc; `fit` sinh theo đường khớp và cần kiểm tra độ phù hợp trước khi dùng. Các điểm khác xu hướng không nên tự loại khi chưa được phòng thí nghiệm xác nhận.

## Vẽ hình cho báo cáo

```powershell
python -X utf8 ve_hinh.py data/so_lieu_GCD.xlsx out/hinh
```

Lệnh ghi các hình PNG vào `out/hinh`, không cần mở giao diện đồ họa.

## Lưu ý khi huấn luyện học máy

Tách tập số đo thật để kiểm tra trước khi sinh dữ liệu. Chỉ khớp hàm, chuẩn hóa và sinh mẫu trên phần huấn luyện. Không chia ngẫu nhiên 48.000 mẫu ảo vào cả tập học và tập kiểm tra. Giữ các cột `ratio`, `J_nominal`, `run_id` để truy vết nhóm, và không dùng mã định danh làm đặc trưng dự đoán.

Mức sai số 3% / 2% / 1% vẫn là giả định. Dữ liệu ảo không thay thế việc đo các điện cực mới. Các tệp web và ảnh giao diện cũ chỉ được giữ làm bản lưu.
