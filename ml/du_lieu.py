"""Đọc số đo và chia kịch bản Chương 4–5, giữ nguyên nhãn điểm thật."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .ma_hoa import DANH_MUC_TI_LE
from .vat_ly import vbar_tu_pd

DUONG_DAN_GOC = Path(__file__).resolve().parents[1] / "data" / "so_lieu_goc_dang_dai.csv"
MOI_TRUONG_KB5 = ((0.025, 0.05, 0.075), (0.1, 0.25), (0.5, 1.0, 1.75), DANH_MUC_TI_LE)
MUC_KIEM_TRA_KB5 = (3, 5)


@dataclass(frozen=True, eq=False)
class LanChia:
    kb: str
    ma: str
    hoc: np.ndarray
    kiem_tra: np.ndarray
    thong_tin: dict


def doc_so_lieu(duong_dan=None) -> pd.DataFrame:
    """Đọc CSV dài, xếp tỉ lệ/dòng tăng và gắn định danh không phụ thuộc phép chia."""
    bang = pd.read_csv(DUONG_DAN_GOC if duong_dan is None else duong_dan)
    bang = bang.sort_values(["ratio", "J"]).reset_index(drop=True)
    bang["Vbar"] = vbar_tu_pd(bang["J"], bang["Pd"])
    bang["muc"] = bang.groupby("ratio", sort=True).cumcount() + 1
    bang["dien_cuc"] = pd.factorize(bang["ratio"], sort=True)[0]
    bang["diem_id"] = [f"{r:g}_m{m}" for r, m in zip(bang["ratio"], bang["muc"])]
    return bang


def _lan(bang, kb, ma, kiem_tra, **thong_tin):
    kt = bang.index[kiem_tra].to_numpy()
    hoc = bang.index[~kiem_tra].to_numpy()
    tt = dict(ratio_dich=None, muc_giau=[], k=None, chinh_sach=None,
              kieu_ma_hoa="onehot", bien=False)
    tt.update(thong_tin)
    return LanChia(kb, ma, hoc, kt, tt)


def chia_kb1(bang) -> list[LanChia]:
    return [_lan(bang, "KB1", f"r{r:g}_m{m}", (bang.ratio == r) & (bang.muc == m),
                 ratio_dich=float(r), muc_giau=[m])
            for r in sorted(bang.ratio.unique()) for m in range(2, 6)]


def chia_kb2(bang, bo_ti_le=()) -> list[LanChia]:
    bang = bang.loc[~bang.ratio.isin(bo_ti_le)]
    return [_lan(bang, "KB2", f"r{r:g}_m6", (bang.ratio == r) & (bang.muc == 6),
                 ratio_dich=float(r), muc_giau=[6], bo_ti_le=tuple(bo_ti_le))
            for r in sorted(bang.ratio.unique())]


def muc_giu_kb3(k, chinh_sach):
    if k not in range(1, 6):
        raise ValueError("k phải nằm trong 1..5")
    if chinh_sach == "thap_nhat":
        return list(range(1, k + 1))
    if chinh_sach == "trai_deu":
        return sorted(set(np.rint(np.linspace(1, 6, k)).astype(int).tolist()))
    raise ValueError("chinh_sach phải là 'thap_nhat' hoặc 'trai_deu'")


def chia_kb3(bang, k_list=(1, 2, 3, 4, 5), chinh_sach=("thap_nhat", "trai_deu")):
    cac_lan = []
    for k in k_list:
        for cs in chinh_sach:
            giu = muc_giu_kb3(k, cs)
            giau = [m for m in range(1, 7) if m not in giu]
            for r in sorted(bang.ratio.unique()):
                cac_lan.append(_lan(bang, "KB3", f"r{r:g}_k{k}_{cs}",
                                    (bang.ratio == r) & ~bang.muc.isin(giu),
                                    ratio_dich=float(r), muc_giau=giau, k=k, chinh_sach=cs))
    return cac_lan


def chia_kb4(bang):
    ratios = sorted(bang.ratio.unique())
    return [_lan(bang, "KB4", f"r{r:g}", bang.ratio == r, ratio_dich=float(r),
                 muc_giau=list(range(1, 7)), kieu_ma_hoa="ti_le", bien=r in (ratios[0], ratios[-1]))
            for r in ratios]


def chia_kb5(bang):
    """Đợt 4 tái hiện toàn bộ điện cực; kiểm tra cố định mức 3 và 5."""
    hoc = [bang.index[bang.ratio.isin(mt) & ~bang.muc.isin(MUC_KIEM_TRA_KB5)].to_numpy()
           for mt in MOI_TRUONG_KB5]
    return hoc, bang.index[bang.muc.isin(MUC_KIEM_TRA_KB5)].to_numpy()


def chia_trong(bang, lan_chia, so_fold=4):
    """Chia nội bộ trên phần học ngoài; không đọc số đo kiểm tra ngoài."""
    if so_fold < 1:
        raise ValueError("so_fold phải dương")
    hoc = bang.loc[lan_chia.hoc]
    masks = []
    if lan_chia.kb in ("KB1", "KB3"):
        lo = hoc.groupby("ratio")["muc"].transform("min")
        hi = hoc.groupby("ratio")["muc"].transform("max")
        masks = [(hoc.muc == m) & (lo < m) & (hi > m) for m in range(2, 6)]
    elif lan_chia.kb in ("KB2", "KB4"):
        ratios = sorted(hoc.ratio.unique())
        if lan_chia.kb == "KB2":
            ratios = [r for r in ratios if r != lan_chia.thong_tin["ratio_dich"]
                      and ((hoc.ratio == r) & (hoc.muc == 6)).any()]
        for f in range(so_fold):
            mask = hoc.ratio.isin(ratios[f::so_fold])
            if lan_chia.kb == "KB2":
                mask &= hoc.muc == 6
            masks.append(mask)
    else:
        raise ValueError(f"Không có chia trong cho {lan_chia.kb}")
    ket_qua = []
    for f, mask in enumerate(masks, 1):
        if not mask.any():
            continue
        tt = dict(lan_chia.thong_tin)
        tt["fold"] = f
        ket_qua.append(LanChia(lan_chia.kb, f"{lan_chia.ma}_trong{f}",
                              hoc.index[~mask].to_numpy(), hoc.index[mask].to_numpy(), tt))
    return ket_qua
