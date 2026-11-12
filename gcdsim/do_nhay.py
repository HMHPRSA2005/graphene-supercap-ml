"""Tái lập độ nhạy GCD-Sim (Chương 3), không thay đổi bộ sinh.

Chạy từ gốc kho mã::

    python -X utf8 -m gcdsim.do_nhay data/so_lieu_GCD.xlsx ket_qua/gcdsim/do_nhay.csv

Độ phân tán Csp (%) = trung bình trên các điểm gốc của
100 × std(Csp ảo, ddof=1) / Csp gốc, nhóm theo (ratio, J_nominal).
Đây là độ lệch chuẩn tương đối quanh điểm neo, không phải MAPE hay độ rộng
khoảng phân vị. MAPE trung vị = trung bình sai số tuyệt đối tương đối của
trung vị ảo tại từng điểm so với số gốc, theo hàm validate hiện có.
Xu hướng yêu cầu cả năm bước chuyển giữ nguyên chiều; tỉ lệ là trung bình
trên tám điện cực. Thứ hạng chỉ xét mức dòng đầu của từng điện cực, không
phải so sánh cùng mật độ dòng J. Các mức nhiễu vẫn là giả định cần đo lặp.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import gcdsim as gs

SIGMA_LAP = (0.01, 0.02, 0.03, 0.04, 0.05)
SIGMA_M = (0.02, 0.03, 0.045)


def do_phan_tan_csp(bang_that, bang_ao):
    """Trung bình độ lệch chuẩn mẫu quanh mỗi điểm neo, chia Csp gốc (%)."""
    nhom = (bang_ao.groupby(["ratio", "J_nominal"]).Csp.std(ddof=1)
            .rename("Csp_sd").reset_index())
    nhom = nhom.merge(bang_that[["ratio", "J", "Csp"]],
                      left_on=["ratio", "J_nominal"], right_on=["ratio", "J"],
                      how="left", validate="one_to_one")
    return float((100 * nhom.Csp_sd / nhom.Csp).mean())


def do_nhay(bang_that):
    """Sinh lại mỗi cấu hình với cùng seed 42, N=10000, nhiễu thiết bị 1%."""
    fits = gs.attach_measured(gs.fit_electrodes(bang_that), bang_that)
    cac_dong = []
    for sigma_lap in SIGMA_LAP:
        for sigma_m in SIGMA_M:
            cfg = gs.SimConfig(n_runs=10000, mode="anchor", repeat_err=sigma_lap,
                               mass_err=sigma_m, meas_err=0.01, seed=42)
            bang_ao = gs.generate_table(fits, cfg, bang_that)
            _, tong_hop = gs.validate(bang_that, bang_ao, fits, cfg)
            # validate xác định điện cực tốt nhất từ số thật, không gán nhãn 0,1
            # cho một điện cực khác nếu người dùng truyền bộ số liệu khác.
            if tong_hop["dien_cuc_tot_nhat_so_lieu_goc"] != 0.1:
                raise ValueError("Số liệu đầu vào phải có điện cực 0,1 tốt nhất ở mức dòng đầu.")
            cac_dong.append(dict(
                sigma_lap_pct=100 * sigma_lap,
                sigma_m_pct=100 * sigma_m,
                sigma_tb_pct=100 * cfg.meas_err,
                n_runs=cfg.n_runs, seed=cfg.seed, mode=cfg.mode,
                so_mau_ao=len(bang_ao),
                do_phan_tan_Csp_pct=do_phan_tan_csp(bang_that, bang_ao),
                giu_xu_huong_pct=tong_hop["lan_do_ao_giu_dung_xu_huong_Csp_theo_J_%"],
                giu_01_tot_nhat_pct=tong_hop["lan_do_ao_giu_dien_cuc_tot_nhat_%"],
                MAPE_trung_vi_Csp_pct=tong_hop["MAPE_trung_vi_ao_vs_that_Csp_%"]))
    return pd.DataFrame(cac_dong)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dau_vao", type=Path, help="Excel của lab hoặc CSV dạng dài")
    parser.add_argument("dau_ra", type=Path, help="Bảng độ nhạy CSV")
    args = parser.parse_args(argv)
    bang_that = (gs.read_lab_excel(str(args.dau_vao))
                 if args.dau_vao.suffix.lower() in (".xlsx", ".xls")
                 else gs.read_long_csv(str(args.dau_vao)))
    ket_qua = do_nhay(bang_that)
    args.dau_ra.parent.mkdir(parents=True, exist_ok=True)
    ket_qua.to_csv(args.dau_ra, index=False, encoding="utf-8-sig")
    print("Độ phân tán = trung bình 100 × std(Csp ảo, ddof=1) / Csp gốc tại mỗi điểm neo.")
    print("Giữ xu hướng: mọi bước chuyển; giữ 0,1 tốt nhất: mức dòng đầu mỗi điện cực, không cùng J.")
    print(ket_qua.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    mac_dinh = ket_qua.loc[(ket_qua.sigma_lap_pct == 3) & (ket_qua.sigma_m_pct == 2)].iloc[0]
    print(f"Mặc định (3%, 2%): độ phân tán {mac_dinh.do_phan_tan_Csp_pct:.4f}% "
          f"→ {mac_dinh.do_phan_tan_Csp_pct:.1f}%.")
    for lap, mass in ((1, 2), (5, 4.5)):
        dong = ket_qua.loc[(ket_qua.sigma_lap_pct == lap) & (ket_qua.sigma_m_pct == mass)].iloc[0]
        print(f"Góc ({lap:g}%, {mass:g}%): phân tán {dong.do_phan_tan_Csp_pct:.4f}%; "
              f"giữ xu hướng {dong.giu_xu_huong_pct:.1f}%; "
              f"giữ 0,1 tốt nhất {dong.giu_01_tot_nhat_pct:.1f}%.")
    print(f"Đã ghi {len(ket_qua)} cấu hình: {args.dau_ra}")


if __name__ == "__main__":
    main()
