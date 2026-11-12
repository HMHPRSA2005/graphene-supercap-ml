"""Tăng cường chỉ từ phần học thật; mọi nhóm lấy mẫu đều có nguồn rõ ràng.

C/D giữ nhiễu cân của GCD-Sim: J là dòng hiệu dụng, J_nominal là dòng lưới
(trong miền học ở D). Không chặn J hiệu dụng vì sẽ phá quan hệ vật lý.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from gcdsim import gcdsim as gs
from .mo_hinh_nen import khop_tat_ca
from .vat_ly import ed_tu_csp, pd_tu_vbar


@dataclass
class CauHinhTangCuong:
    cach: str = "C"
    n_ao: int = 100
    sigma_lap: float = 0.03
    mass_err: float = 0.02
    meas_err: float = 0.01
    nhieu_B: float = 0.02
    n_j_D: int = 12


def _bang_that(bang):
    that = bang.copy()
    that["nguon"] = "that"
    if "diem_id" in that:
        that["diem_goc"] = that["diem_id"].astype(str)
    else:
        that["diem_goc"] = [f"{r:g}_J{j:.17g}" for r, j in zip(that.ratio, that.J)]
    that["J_nominal"] = that["J"]
    if "Ed" not in that:
        that["Ed"] = ed_tu_csp(that["Csp"])
    if "Pd" not in that:
        that["Pd"] = pd_tu_vbar(that["J"], that["Vbar"])
    return that


def tao_tap_hoc(bang_that_hoc, cau_hinh, seed) -> pd.DataFrame:
    """A: thật; B: nhiễu tương đối; C: neo số đo; D: lưới đường khớp.

    D có một mức J khi điện cực chỉ có một mức thật. Dùng J_nominal để
    xác định miền sinh/điểm gốc, không dùng J đã chịu sai số cân khối lượng.
    """
    cfg = cau_hinh
    if cfg.cach not in ("A", "B", "C", "D"):
        raise ValueError(f"Cách tăng cường không hợp lệ: {cfg.cach}")
    that = _bang_that(bang_that_hoc)
    if cfg.cach == "A" or cfg.n_ao == 0 or that.empty:
        return that.reset_index(drop=True)
    if cfg.n_ao < 0 or (cfg.cach == "D" and cfg.n_j_D < 1):
        raise ValueError("n_ao không âm, n_j_D dương")
    if cfg.cach == "B":
        ao = that.loc[that.index.repeat(cfg.n_ao)].reset_index(drop=True)
        rng = np.random.default_rng(seed)
        ao["Csp"] = ao["Csp"].to_numpy(float) * (1 + rng.normal(0, cfg.nhieu_B, len(ao)))
        ao["Vbar"] = ao["Vbar"].to_numpy(float) * (1 + rng.normal(0, cfg.nhieu_B, len(ao)))
        ao["Ed"] = ed_tu_csp(ao["Csp"])
        ao["Pd"] = pd_tu_vbar(ao["J"], ao["Vbar"])
    else:
        sim = gs.SimConfig(n_runs=cfg.n_ao, mode="anchor" if cfg.cach == "C" else "fit",
                           j_mode="measured" if cfg.cach == "C" else "range",
                           n_j_range=cfg.n_j_D, repeat_err=cfg.sigma_lap,
                           mass_err=cfg.mass_err, meas_err=cfg.meas_err, seed=seed)
        if cfg.cach == "C":
            # Mode anchor không dùng đường Csp/Vbar; không khớp hàm thừa.
            fits = {float(r): gs.ElectrodeFit(float(r), "power", (1.0, 0.0), 0.0, {},
                                             (1.0, 0.0), float(g.J.min()), float(g.J.max()))
                    for r, g in that.groupby("ratio", sort=True)}
            gs.attach_measured(fits, that)
            ao = gs.generate_table(fits, sim, df_real=that)
            nguon = that.set_index(["ratio", "J"])["diem_goc"]
            keys = pd.MultiIndex.from_arrays([ao["ratio"], ao["J_nominal"]])
            ao["diem_goc"] = nguon.reindex(keys).to_numpy()
        else:
            fits = khop_tat_ca(that)
            # Tách theo số mức để không tạo n_j_D bản trùng ở điện cực một điểm.
            thuong = {r: f for r, f in fits.items() if f.J_min < f.J_max}
            mot_muc = {r: f for r, f in fits.items() if f.J_min == f.J_max}
            nhom = [(thuong, cfg.n_j_D), (mot_muc, 1)]
            seeds = np.random.SeedSequence(seed).spawn(2)
            cac_bang = [gs.generate_table(fs, replace(sim, n_j_range=n_j,
                           seed=int(s.generate_state(1)[0])))
                        for (fs, n_j), s in zip(nhom, seeds) if fs]
            ao = pd.concat(cac_bang, ignore_index=True)
            ao["diem_goc"] = [f"{r:g}_luoi_J{j:.17g}"
                              for r, j in zip(ao["ratio"], ao["J_nominal"])]
    ao["nguon"] = "ao"
    return pd.concat([that, ao], ignore_index=True, sort=False)


def lay_mau_con(bang_hoc, toi_da, seed) -> pd.DataFrame:
    """Giữ toàn bộ điểm thật, chia đều phần còn lại theo nhóm điểm gốc.

    Nhóm nhỏ được giữ hết rồi phân phối lại hạn mức. Nếu riêng số điểm thật
    đã vượt toi_da, trả toàn bộ điểm thật (ưu tiên bất biến giữ số đo).
    """
    if len(bang_hoc) <= toi_da:
        return bang_hoc
    that = bang_hoc["nguon"].eq("that").to_numpy()
    vi_tri_that = np.flatnonzero(that)
    han_muc = max(0, int(toi_da) - len(vi_tri_that))
    ao = bang_hoc.iloc[np.flatnonzero(~that)]
    nhom = list(ao.groupby("diem_goc", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    nhom = [nhom[i] for i in rng.permutation(len(nhom))]
    so_luong = np.array([len(v) for v in nhom], dtype=int)
    lay = np.zeros(len(nhom), dtype=int)
    con = min(han_muc, len(ao))
    while con:
        chua_day = np.flatnonzero(lay < so_luong)
        deu, du = divmod(con, len(chua_day))
        them = np.minimum(so_luong[chua_day] - lay[chua_day],
                          deu + (np.arange(len(chua_day)) < du))
        lay[chua_day] += them
        con -= int(them.sum())
    vi_tri_ao = np.flatnonzero(~that)
    chon = [vi_tri_that]
    for vi_tri, n in zip(nhom, lay):
        if n:
            chon.append(vi_tri_ao[rng.choice(vi_tri, size=n, replace=False)])
    return bang_hoc.iloc[np.sort(np.concatenate(chon))].copy()
