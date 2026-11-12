"""Preset Chương 5; nhóm thử giữ mọi kịch bản nhưng lấy lần chia đại diện.

KB2 bỏ điện cực có hậu tố riêng trong mã lần chia. Cách A dùng n_ao=0 vì
không sinh mẫu; nhờ đó cấu hình thật trùng giữa các nhóm chỉ tính một lần.
Để chạy Colab 2 nhân: nhóm đầy đủ chọn tham số với n_ao_chon=3, chỉ
lấy mẫu con GPR (300 hàng) và SVR (theo TOI_DA_HANG). Các mô hình khác
giữ đủ dữ liệu, dùng ngân sách bước tối ưu của chính mô hình; không cắt
lưới, seed hay lần chia. so_hang_hoc ghi số hàng thực dùng.
Vòng nội bộ dùng cùng ngân sách cho mọi tổ hợp: MLP 2000 bước Adam,
đồ thị 4000 mẫu trình bày; học ngoài giữ mặc định 6000 bước / 20000 mẫu.
Đồ thị chỉ giữ ở phu_luc_do_thi, tham khảo khóa luận Tân; không thuộc
các nhóm chính hoặc nhóm thu, không chạy mặc định.
"""
from __future__ import annotations

from dataclasses import replace
from itertools import product
import json

from . import du_lieu
from .tang_cuong import CauHinhTangCuong

TEN_NHOM = ("thu", "chinh", "du_lieu_ao", "so_mau", "sigma", "kb3", "kb4", "phu_luc_do_thi")
CAC_SEED = list(range(10))
CHINH = ["ham_mu", "chon_dang_ham", "ridge", "svr", "rf", "gb", "mlp", "gpr"]
AO = ["mlp", "svr", "gpr"]
NGAN_SACH_CHON = {
    "mlp": {"so_buoc": 2000},
    "gng_r": {"so_mau_hoc": 4000},
    "pgcn_r": {"so_mau_hoc": 4000},
}


def _viec(kb, mo_hinh, cach=("C",), n_ao=(100,), sigma_lap=(0.03,), seeds=None, **tuy_chon):
    return dict(kb=kb, mo_hinh=list(mo_hinh), cach=list(cach), n_ao=list(n_ao),
                sigma_lap=list(sigma_lap), seeds=list(CAC_SEED if seeds is None else seeds), tuy_chon=tuy_chon)


NHOM = {
    "thu": [
        _viec(kb, ["ham_mu", "chon_dang_ham", "ridge", "mlp", "gpr"],
              ("A", "C"), (10,), seeds=[0, 1], dai_dien=True,
              **({"k_list": (2, 4)} if kb == "KB3" else {})) for kb in ("KB1", "KB2", "KB3")
    ] + [_viec("KB4", ["tb_hinh_hoc", "ridge", "mlp", "gpr"], ("A", "C"), (10,), seeds=[0, 1], dai_dien=True)],
    "chinh": [_viec(kb, CHINH, **tc) for kb, tc in (("KB1", {}), ("KB2", {}), ("KB2", {"bo_ti_le": (0.075,)}))],
    "du_lieu_ao": [_viec(kb, AO, ("A", "B", "C", "D")) for kb in ("KB1", "KB2")]
        + [_viec(kb, AO, ("B",), (5,)) for kb in ("KB1", "KB2")],
    "so_mau": [_viec(kb, AO, n_ao=(10, 100, 1000, 10000)) for kb in ("KB1", "KB2")],
    "sigma": [_viec(kb, AO, sigma_lap=(0.01, 0.03, 0.05, 0.10)) for kb in ("KB1", "KB2")],
    "kb3": [_viec("KB3", ["chon_dang_ham", "ridge", "svr", "mlp", "gpr"])],
    "kb4": [_viec("KB4", ["tb_hinh_hoc", "tb_theo_muc", "ridge", "svr", "rf", "gb", "mlp", "gpr"], ("A", "C"))],
    "phu_luc_do_thi": [_viec(kb, ["gng_r", "pgcn_r"]) for kb in ("KB1", "KB2")],
}

def ma_cau_hinh_chon(rut_gon=False, n_ao_chon=10, gioi_han_hang=None, ngan_sach_chon=None):
    """Mã JSON tất định; ngân sách khác là đơn vị khác khi tiếp tục CSV."""
    return json.dumps(dict(rut_gon=bool(rut_gon), n_ao_chon=n_ao_chon,
                           gioi_han_hang=gioi_han_hang, ngan_sach_chon=ngan_sach_chon),
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))



def khoa_don_vi(don_vi):
    """Khóa không chứa nhóm/seed; seed là số lần lặp của cùng đơn vị."""
    f, c = don_vi["lan_chia"], don_vi["cau_hinh_tc"]
    return (f.kb, f.ma, don_vi["ten"], c.cach, c.n_ao, c.sigma_lap,
            f.thong_tin.get("kieu_ma_hoa", "onehot"),
            ma_cau_hinh_chon(**{k: don_vi[k] for k in
                ("rut_gon", "n_ao_chon", "gioi_han_hang", "ngan_sach_chon") if k in don_vi}))


def liet_ke_don_vi(bang, nhom=("chinh",), mo_hinh=None, seeds=None):
    """Liệt kê theo thứ tự nhóm và gộp seed của các đơn vị trùng khóa."""
    if isinstance(nhom, str):
        nhom = [nhom]
    da_co = {}
    for ten_nhom in nhom:
        for viec in NHOM[ten_nhom]:
            tc = dict(viec["tuy_chon"])
            dai_dien = tc.pop("dai_dien", False)
            cac_lan = getattr(du_lieu, "chia_" + viec["kb"].lower())(bang, **tc)
            if tc.get("bo_ti_le"):
                hau_to = "_bo_" + "_".join(f"{r:g}" for r in tc["bo_ti_le"])
                cac_lan = [replace(f, ma=f.ma + hau_to) for f in cac_lan]
            if dai_dien:
                # Mỗi (k, chính sách) của KB3 đều có mặt, thay vì lấy lát đầu.
                giu = {}
                for f in cac_lan:
                    giu.setdefault((f.thong_tin.get("k"), f.thong_tin.get("chinh_sach")), f)
                cac_lan = list(giu.values())
            for f, ten, cach, n, sigma in product(cac_lan, viec["mo_hinh"], viec["cach"], viec["n_ao"], viec["sigma_lap"]):
                if mo_hinh is not None and ten not in mo_hinh:
                    continue
                c = CauHinhTangCuong(cach=cach, n_ao=0 if cach == "A" else n, sigma_lap=sigma)
                d = dict(lan_chia=f, ten=ten, cau_hinh_tc=c,
                         seeds=list(viec["seeds"] if seeds is None else seeds),
                         n_ao_chon=10 if ten_nhom == "thu" else 3,
                         gioi_han_hang=300 if ten_nhom != "thu" and ten == "gpr" else None,
                         ngan_sach_chon=None if ten_nhom == "thu" else NGAN_SACH_CHON.get(ten),
                         rut_gon=ten_nhom == "thu", nhom=ten_nhom)
                khoa = khoa_don_vi(d)
                if khoa in da_co:
                    da_co[khoa]["seeds"] = sorted(set(da_co[khoa]["seeds"]) | set(d["seeds"]))
                else:
                    da_co[khoa] = d
    return list(da_co.values())
