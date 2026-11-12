# Dữ liệu

Số liệu đo sạc–xả dòng không đổi (GCD) của 8 điện cực siêu tụ graphene/hạt nano, do nhóm NTM.Hồng & LD.Phúc cung cấp. Mỗi điện cực đo ở 6 mức dòng tuyệt đối 10, 20, 50, 80, 100, 200 mA; mật độ dòng J khác nhau giữa các điện cực vì khối lượng điện cực khác nhau.

| Tệp | Nội dung |
|---|---|
| `so_lieu_GCD.xlsx` | Bảng gốc của phòng thí nghiệm: 3 khối Điện dung riêng, Mật độ năng lượng, Mật độ công suất; mỗi điện cực 2 cột J (A/g) và giá trị |
| `so_lieu_goc_dang_dai.csv` | Cùng 48 điểm ở dạng dài, là đầu vào chuẩn của gói `ml/` |
| `../ket_qua/gcdsim/du_lieu_ao_480000.xlsx` | Dữ liệu ảo Chương 3: 48 điểm × 10000 lần đo ảo, seed 42; Excel nhiều sheet gồm dữ liệu, cấu hình và kiểm chứng. Tệp 53.520.383 byte (53,52 MB), vượt 50 MB nên không lưu trong `data/` hoặc Git |

## Nhận hoặc sinh lại dữ liệu ảo

Từ thư mục `Code/`, chạy:

```bash
python -X utf8 -m gcdsim.gcdsim data/so_lieu_GCD.xlsx --out ket_qua/gcdsim/du_lieu_ao_480000.xlsx --seed 42
```

Không cần truyền `--n`: mặc định là 10000 lần đo ảo mỗi điện cực, tổng cộng **480.000 hàng**. Tệp sinh sẵn cũ đã được bỏ; không có bản sao Excel ảo trong `data/`.

Trong **Giai đoạn 1**, Colab có thể sinh tệp này vào thư mục kết quả Drive rồi đóng gói trong `san_pham_giai_doan_1.zip`. **Giai đoạn 2** nhận sản phẩm đó trên máy local; nếu đã có `gcdsim/du_lieu_ao_480000.xlsx` thì không cần sinh lại. Xem `../gcdsim/README.md` để vẽ hình, tái lập độ nhạy và hiểu các giả định; xem `../README.md` cho quy trình hai giai đoạn.

Bảng ảo dùng toàn bộ 48 điểm để minh họa Chương 3, **không phải tập kiểm tra ML độc lập**. Khi đánh giá ML phải chia số thật trước rồi chỉ sinh ảo từ phần học.

## Cột của `so_lieu_goc_dang_dai.csv`
| Cột | Đơn vị | Ý nghĩa |
|---|---|---|
| ratio | – | tỉ lệ hạt nano/graphene của điện cực (0,025; 0,05; 0,075; 0,1; 0,25; 0,5; 1; 1,75) |
| J | A/g | mật độ dòng |
| Csp | F/g | điện dung riêng |
| Ed | Wh/kg | mật độ năng lượng, Ed = Csp·ΔV²/7,2 với ΔV = 2,1 V |
| Pd | W/kg | mật độ công suất, Pd = 1000·J·V̄ (V̄: điện thế xả trung bình) |

Lưu ý: điểm 0,075 ở mức dòng cao nhất có Csp = 2,06 F/g (rất nhỏ so với các mức khác); điện cực 0,1 ở mức 2 và 1,75 ở mức 2 có V̄ thấp bất thường. Các điểm này chưa được phòng thí nghiệm xác nhận là lỗi nên vẫn giữ nguyên.
