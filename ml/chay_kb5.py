"""Học liên tục KB5, cố định mã hóa/thang y từ đợt đầu, lưu CSV có resume.

Mốc buoc đếm mẫu mới được đưa vào (không nhân số epoch). Mỗi khối kết thúc
ở bội moi_buoc hoặc cuối đợt: học tiếp chỉ khối mới; đối chứng học lại khớp
MLP mới trên toàn bộ mẫu đã thấy tại mốc đó, bao gồm cuối mỗi đợt. Thang
chuẩn hóa của mọi phương pháp luôn thuộc toàn bộ dữ liệu đợt đầu.
MLP mới có ngân sách 6000 bước Adam (số epoch phụ thuộc số batch), thay
vì tăng chi phí theo số hàng tích lũy. Tinh chỉnh giữ 10 vòng trên khối mới.
"""
from __future__ import annotations

import argparse
import inspect
from itertools import permutations
import json
import os
from pathlib import Path
from time import perf_counter
import warnings

PHUONG_PHAP = ("mlp_hoc_lai", "mlp_tinh_chinh", "gng_r", "pgcn_r")
COT_KB5 = "phuong_phap seed thu_tu moi_truong buoc cuoi_moi_truong ratio muc Csp Csp_hat Vbar Vbar_hat Pd Pd_hat so_nut so_canh t_hoc_s t_du_doan_s".split()
THAM_SO_MLP = dict(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
                   solver="adam", batch_size="auto", early_stopping=False, tol=0.0,
                   so_buoc=6000, so_vong_tinh_chinh=10)


def cau_hinh_chay(n_ao, moi_buoc, tham_so_pgcn, nhanh):
    """Chụp tham số thực dùng; bộ lọc seed/phương pháp/thứ tự được phép mở rộng."""
    from dataclasses import asdict
    from .tang_cuong import CauHinhTangCuong
    from .gng_r import GNGR
    from .pgcn_r import PGCNR
    def mac_dinh(lop):
        return {ten: p.default for ten, p in inspect.signature(lop.__init__).parameters.items()
                if ten != "seed" and p.default is not inspect.Parameter.empty}
    gng = mac_dinh(GNGR)
    pgcn = {**gng, **mac_dinh(PGCNR), **tham_so_pgcn}
    cau_hinh = dict(n_ao=n_ao, moi_buoc=moi_buoc, tham_so_pgcn=tham_so_pgcn,
                    tang_cuong=asdict(CauHinhTangCuong(cach="C", n_ao=n_ao)),
                    tham_so_mlp=THAM_SO_MLP,
                    tham_so_do_thi=dict(gng_r={**gng, "so_mau_hoc": 20000},
                        pgcn_r={"so_mau_hoc": 20000, **pgcn}),
                    quy_tac_so_vong_fit="ceil(so_mau_hoc/n_khoi_dau), chan boi so_vong neu chi dinh",
                    so_vong_hoc_tiep=dict(gng_r=4, pgcn_r=tham_so_pgcn.get("so_vong", 4)),
                    nhanh=bool(nhanh))
    return json.loads(json.dumps(cau_hinh, ensure_ascii=False, sort_keys=True))


def kiem_tra_cau_hinh(thu_muc, cau_hinh):
    """Không gán nhãn cấu hình mới cho CSV cũ không có nguồn gốc tham số."""
    from .chay_kb1_kb4 import ghi_json_nguyen_tu
    dich = Path(thu_muc) / "cau_hinh_kb5.json"
    csv_path = Path(thu_muc) / "ket_qua_kb5.csv"
    if dich.exists():
        cu = json.loads(dich.read_text(encoding="utf-8"))
        if cu != cau_hinh:
            khac = sorted(k for k in set(cu) | set(cau_hinh) if cu.get(k) != cau_hinh.get(k))
            raise ValueError(f"Cấu hình KB5 khác lần chạy trước ({', '.join(khac)}). "
                             f"Không trộn --nhanh/đầy đủ hoặc tham số khác trong {csv_path}; "
                             "hãy dùng --thu-muc khác.")
    elif csv_path.exists() and csv_path.stat().st_size:
        raise ValueError(f"Không thể resume {csv_path}: thiếu {dich.name}; "
                         "không xác định được tham số đã dùng. Hãy dùng --thu-muc khác.")
    else:
        ghi_json_nguyen_tu(dich, cau_hinh)


class _HocCoDinh:
    """Dùng lõi thật với một thang chung; không fit lại thang ở mỗi mốc."""

    def __init__(self, phuong_phap, that_dau, hoc_dau, seed, tham_so_pgcn):
        import numpy as np
        from .ma_hoa import BoMaHoa, ma_hoa_y
        self.phuong_phap, self.seed = phuong_phap, seed
        self.bo_ma_hoa = BoMaHoa("onehot").fit(that_dau)
        y = ma_hoa_y(hoc_dau)
        self.y_tb, self.y_sd = y.mean(axis=0), y.std(axis=0)
        self.y_sd[self.y_sd == 0] = 1.0
        self.tham_so_pgcn = tham_so_pgcn
        self.so_vong_tiep = tham_so_pgcn.get("so_vong", 4) if phuong_phap == "pgcn_r" else 4
        self.loi = None

    def hoc(self, bang):
        """MLP mới: 6000 bước; đồ thị fit 20000 mẫu, học tiếp tối đa 4 vòng."""
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor
        from .ma_hoa import ma_hoa_y
        from .gng_r import GNGR
        from .pgcn_r import PGCNR
        x = self.bo_ma_hoa.transform(bang)
        y = (ma_hoa_y(bang) - self.y_tb) / self.y_sd
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            if self.phuong_phap == "mlp_hoc_lai" or self.loi is None:
                if self.phuong_phap.startswith("mlp_"):
                    p = dict(THAM_SO_MLP)
                    so_buoc = p.pop("so_buoc")
                    p.pop("so_vong_tinh_chinh")
                    so_batch = (len(bang) + 199) // 200
                    so_vong = (so_buoc + so_batch - 1) // so_batch
                    self.loi = MLPRegressor(**p, max_iter=so_vong,
                                           n_iter_no_change=so_vong + 1, random_state=self.seed)
                else:
                    p = dict(self.tham_so_pgcn) if self.phuong_phap == "pgcn_r" else {}
                    so_mau_hoc = p.pop("so_mau_hoc", 20000)
                    if not isinstance(so_mau_hoc, int) or so_mau_hoc < 1:
                        raise ValueError("so_mau_hoc phải là số nguyên dương")
                    so_vong = (so_mau_hoc + len(bang) - 1) // len(bang)
                    p["so_vong"] = min(so_vong, p.get("so_vong", so_vong))
                    self.loi = (PGCNR(**p, seed=self.seed) if self.phuong_phap == "pgcn_r"
                                else GNGR(**p, seed=self.seed))
                self.loi.fit(x, y)
            elif self.phuong_phap == "mlp_tinh_chinh":
                for _ in range(THAM_SO_MLP["so_vong_tinh_chinh"]):
                    self.loi.partial_fit(x, y)
            else:
                self.loi.hoc_tiep(x, y, so_vong=self.so_vong_tiep)

    def predict(self, bang):
        from .giao_dien import khung_du_doan
        from .ma_hoa import giai_ma_y
        y = self.loi.predict(self.bo_ma_hoa.transform(bang)) * self.y_sd + self.y_tb
        return khung_du_doan(bang, *giai_ma_y(y))

    @property
    def so_nut(self):
        return getattr(self.loi, "so_nut", 0)

    @property
    def so_canh(self):
        return getattr(self.loi, "so_canh", 0)


def chay_chuoi(bang, phuong_phap, seed, thu_tu=(1, 2, 3, 4), n_ao=1000,
               moi_buoc=2000, tham_so_pgcn=None):
    """Sinh từng đợt chỉ từ số đo của đợt; trả các mốc để ghi ngay."""
    import numpy as np
    import pandas as pd
    from .du_lieu import chia_kb5
    from .tang_cuong import CauHinhTangCuong, tao_tap_hoc
    from .vat_ly import pd_tu_vbar
    chi_so, kt = chia_kb5(bang)
    kiem_tra = bang.loc[kt]
    mo_hinh = None
    da_thay = []
    buoc, t_hoc = 0, 0.0
    for t, i in enumerate(thu_tu, 1):
        that = bang.loc[chi_so[i - 1]]
        hoc = tao_tap_hoc(that, CauHinhTangCuong(cach="C", n_ao=n_ao), seed * 10 + i)
        if mo_hinh is None:
            mo_hinh = _HocCoDinh(phuong_phap, that, hoc, seed, tham_so_pgcn or {})
        # Trộn chỉ phần học; không thay đổi tập kiểm tra hoặc chia ngẫu nhiên
        # mẫu ảo để chọn tham số. Mốc đếm số mẫu đã tiếp nhận.
        hoc = hoc.iloc[np.random.default_rng(seed * 10 + i).permutation(len(hoc))]
        dau = 0
        while dau < len(hoc):
            so = min(moi_buoc - buoc % moi_buoc, len(hoc) - dau)
            khoi = hoc.iloc[dau:dau + so]
            if phuong_phap == "mlp_hoc_lai":
                da_thay.append(khoi)
                tap_hoc = pd.concat(da_thay, ignore_index=True)
            else:
                tap_hoc = khoi
            bat_dau = perf_counter()
            mo_hinh.hoc(tap_hoc)
            t_hoc += perf_counter() - bat_dau
            buoc += so
            dau += so
            bat_dau = perf_counter()
            dd = mo_hinh.predict(kiem_tra)
            t_dd = perf_counter() - bat_dau
            pd_hat = pd_tu_vbar(kiem_tra.J, dd.Vbar_hat)
            hang = []
            for j, (_, r) in enumerate(kiem_tra.iterrows()):
                hang.append(dict(phuong_phap=phuong_phap, seed=seed,
                    thu_tu="-".join(map(str, thu_tu)), moi_truong=t, buoc=buoc,
                    cuoi_moi_truong=dau == len(hoc), ratio=r.ratio, muc=int(r.muc),
                    Csp=r.Csp, Csp_hat=dd.Csp_hat.iloc[j], Vbar=r.Vbar,
                    Vbar_hat=dd.Vbar_hat.iloc[j], Pd=r.Pd, Pd_hat=pd_hat[j],
                    so_nut=mo_hinh.so_nut, so_canh=mo_hinh.so_canh,
                    t_hoc_s=t_hoc, t_du_doan_s=t_dd))
            yield hang


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thu-muc", default="ket_qua")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--phuong-phap", nargs="+", choices=PHUONG_PHAP, default=list(PHUONG_PHAP))
    parser.add_argument("--doi-thu-tu", action="store_true")
    parser.add_argument("--n-ao", type=int)
    parser.add_argument("--moi-buoc", type=int, default=2000)
    parser.add_argument("--tham-so-pgcn", default="{}")
    parser.add_argument("--nhanh", action="store_true")
    args = parser.parse_args(argv)
    n_ao = args.n_ao if args.n_ao is not None else (50 if args.nhanh else 1000)
    seeds = args.seeds if args.seeds is not None else ([0] if args.nhanh else list(range(10)))
    if n_ao < 0 or args.moi_buoc < 2:
        parser.error("n_ao không âm và moi_buoc >= 2")
    try:
        tham_so = json.loads(args.tham_so_pgcn)
        if not isinstance(tham_so, dict):
            raise ValueError("cần đối tượng JSON")
    except (ValueError, json.JSONDecodeError) as exc:
        parser.error(f"--tham-so-pgcn: {exc}")
    for ten in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[ten] = "1"
    import pandas as pd
    from .du_lieu import doc_so_lieu
    from .chay_kb1_kb4 import ghi_moi_truong, ghi_noi_tiep, phuc_hoi_csv, kiem_tra_header
    thu_muc = Path(args.thu_muc)
    thu_muc.mkdir(parents=True, exist_ok=True)
    ghi_moi_truong(thu_muc, cli="ml.chay_kb5", argv=argv, **vars(args))
    dich = thu_muc / "ket_qua_kb5.csv"
    phuc_hoi_csv(dich)
    try:
        kiem_tra_header(dich, COT_KB5)
        kiem_tra_cau_hinh(thu_muc, cau_hinh_chay(n_ao, args.moi_buoc, tham_so, args.nhanh))
    except ValueError as exc:
        parser.error(str(exc))
    da_xong = set()
    da_ghi = set()
    bang = doc_so_lieu()
    if dich.exists() and dich.stat().st_size:
        cu = pd.read_csv(dich)
        for khoa, nhom in cu.groupby(["phuong_phap", "seed", "thu_tu"], sort=False):
            for moc, g in nhom.groupby(["moi_truong", "buoc"]):
                if len(g) != 16 or g[["ratio", "muc"]].drop_duplicates().shape[0] != 16:
                    raise ValueError("Mốc KB5 không đủ 16 điểm; cần phục hồi bản sao")
                da_ghi.add((*khoa, *moc))
            cuoi = nhom.loc[(nhom.moi_truong == 4) & nhom.cuoi_moi_truong.astype(str).str.lower().eq("true")]
            if len(cuoi) == 16:
                da_xong.add(khoa)
    cac_thu_tu = [(*p, 4) for p in permutations((1, 2, 3))] if args.doi_thu_tu else [(1, 2, 3, 4)]
    cac_viec = [(pp, s, tt) for tt in cac_thu_tu for s in seeds for pp in args.phuong_phap
                if (pp, s, "-".join(map(str, tt))) not in da_xong]
    t0 = perf_counter()
    print(f"KB5 cần chạy {len(cac_viec)} chuỗi; n_ao={n_ao}", flush=True)
    for j, (pp, s, tt) in enumerate(cac_viec, 1):
        for hang in chay_chuoi(bang, pp, s, tt, n_ao, args.moi_buoc, tham_so):
            r = hang[0]
            khoa = (pp, s, r["thu_tu"], r["moi_truong"], r["buoc"])
            # Phát lại tất định trạng thái từ đầu nếu mất kết nối giữa chuỗi,
            # nhưng không ghi lại các mốc đã có trong CSV.
            if khoa not in da_ghi:
                ghi_noi_tiep(dich, hang, COT_KB5)
                da_ghi.add(khoa)
            print(f"{pp} seed={s} thứ tự={r['thu_tu']} đợt={r['moi_truong']} bước={r['buoc']} nút/cạnh={r['so_nut']}/{r['so_canh']} học={r['t_hoc_s']:.1f}s", flush=True)
        da_qua = perf_counter() - t0
        print(f"[{j}/{len(cac_viec)}] đã chạy {da_qua:.1f}s; ETA {da_qua / j * (len(cac_viec) - j) / 60:.1f} phút", flush=True)
    print(f"KB5 hoàn tất trong {perf_counter() - t0:.1f}s: {dich}", flush=True)


if __name__ == "__main__":
    main()
