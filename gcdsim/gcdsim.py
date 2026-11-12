"""
GCD-Sim: sinh dữ liệu giả lập (dữ liệu ảo) cho phép đo sạc–xả dòng không đổi (GCD)
từ bảng số liệu thực nghiệm của điện cực siêu tụ graphene/hạt nano.

Ý tưởng (giải thích được bằng vật lý, không dùng mô hình sinh "hộp đen"):
  1. Với mỗi điện cực, khớp quy luật Csp giảm theo mật độ dòng J bằng 3 dạng hàm
     quen thuộc trong điện hóa, chọn dạng có sai số kiểm tra chéo (LOOCV) nhỏ nhất.
  2. Khớp điện thế xả trung bình V̄(J) = Pd / (1000·J) bằng đường thẳng.
  3. Sinh mẫu ảo = giá trị trên đường khớp × các nguồn sai số thực tế của phép đo:
       - sai số mô hình (độ lệch còn lại sau khi khớp),
       - sai số cân khối lượng điện cực (làm J, Csp, Ed, Pd cùng lệch một tỉ lệ),
       - nhiễu thiết bị đo.
  4. Tính Ed, Pd, Δt bằng đúng công thức lab dùng → dữ liệu ảo luôn tự nhất quán.
  5. (Tùy chọn) Sinh đường cong GCD V(t) có diện tích tích phân khớp với Csp.
  6. Kiểm chứng: tính lại Csp từ đường cong; tỉ lệ điểm thật nằm trong dải ảo.

Chỉ dùng numpy, pandas, openpyxl, matplotlib.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

DV_DEFAULT = 2.1  # cửa sổ điện áp (V), theo tài liệu của lab


# ----------------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------------
def read_lab_excel(path: str) -> pd.DataFrame:
    """Đọc file Excel dạng của lab (các khối 'Điện dung riêng', 'Mật độ năng lượng',
    'Mật độ công suất'; mỗi điện cực 2 cột J | giá trị). Trả về bảng dạng dài:
    ratio, J, Csp, Ed, Pd."""
    raw = pd.read_excel(path, header=None)
    blocks = {}
    for r in range(raw.shape[0]):
        label = str(raw.iat[r, 0]).strip().lower() if pd.notna(raw.iat[r, 0]) else ""
        key = None
        if label.startswith("điện dung"):
            key = "Csp"
        elif label.startswith("mật độ năng"):
            key = "Ed"
        elif label.startswith("mật độ công"):
            key = "Pd"
        if key is None:
            continue
        rows = []
        for c in range(1, raw.shape[1]):
            ratio = raw.iat[r, c]
            if not isinstance(ratio, (int, float)) or pd.isna(ratio):
                continue
            # cột J nằm ở c, cột giá trị ở c+1 (hàng tiêu đề 'J (A/g)' ở r+1)
            jcol, vcol = c, c + 1
            if not str(raw.iat[r + 1, jcol]).startswith("J"):
                jcol, vcol = c + 1, c + 2
            rr = r + 2
            while rr < raw.shape[0] and pd.notna(raw.iat[rr, jcol]):
                rows.append((float(ratio), float(raw.iat[rr, jcol]), float(raw.iat[rr, vcol])))
                rr += 1
        blocks[key] = pd.DataFrame(rows, columns=["ratio", "J", key])
    df = blocks["Csp"]
    for k in ("Ed", "Pd"):
        if k in blocks:
            df = df.merge(blocks[k], on=["ratio", "J"], how="left")
    return df.sort_values(["ratio", "J"]).reset_index(drop=True)


def read_long_csv(path: str) -> pd.DataFrame:
    """CSV dạng dài với các cột ratio,J,Csp,Pd (Ed tùy chọn)."""
    df = pd.read_csv(path)
    return df.sort_values(["ratio", "J"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Khớp quy luật Csp(J) — bình phương tối thiểu có trọng số 1/y² (sai số tương đối)
# ----------------------------------------------------------------------------
def _wls(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w = 1.0 / np.maximum(np.abs(y), 1e-9)
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta


def fit_exp(J, y):
    """Csp = a·exp(−b·J) + c ; dò b trên lưới, a, c giải tuyến tính."""
    best = None
    jmax = J.max()
    for b in np.geomspace(0.01 / jmax, 20 / jmax, 400):
        X = np.column_stack([np.exp(-b * J), np.ones_like(J)])
        a, c = _wls(X, y)
        if a <= 0 or c < 0:
            continue
        pred = a * np.exp(-b * J) + c
        err = np.mean(((pred - y) / y) ** 2)
        if best is None or err < best[0]:
            best = (err, (a, b, c))
    if best is None:  # không tìm được nghiệm hợp lệ -> fallback c = 0
        b = 1 / jmax
        a = _wls(np.exp(-b * J)[:, None], y)[0]
        return (a, b, 0.0)
    return best[1]


def pred_exp(p, J):
    a, b, c = p
    return a * np.exp(-b * J) + c


def fit_power(J, y):
    """Csp = a·J^(−n) ; tuyến tính trong log."""
    X = np.column_stack([np.ones_like(J), np.log(J)])
    beta, *_ = np.linalg.lstsq(X, np.log(y), rcond=None)
    return (np.exp(beta[0]), -beta[1])


def pred_power(p, J):
    a, n = p
    return a * J ** (-n)


def fit_dunn(J, y):
    """Csp = a + b/√J (điện dung bề mặt + phần khuếch tán, kiểu Trasatti/Dunn)."""
    X = np.column_stack([np.ones_like(J), 1 / np.sqrt(J)])
    return tuple(_wls(X, y))


def pred_dunn(p, J):
    a, b = p
    return np.maximum(a + b / np.sqrt(J), 1e-6)


MODELS = {
    "exp": ("Hàm mũ  Csp = a·e^(−bJ) + c", fit_exp, pred_exp),
    "power": ("Lũy thừa  Csp = a·J^(−n)", fit_power, pred_power),
    "dunn": ("Dunn  Csp = a + b/√J", fit_dunn, pred_dunn),
}


def loocv_mape(name, J, y):
    _, fit, pred = MODELS[name]
    errs = []
    for k in range(len(J)):
        m = np.ones(len(J), bool)
        m[k] = False
        p = fit(J[m], y[m])
        errs.append(abs(pred(p, J[k:k + 1])[0] - y[k]) / y[k])
    return float(np.mean(errs) * 100), np.array(errs) * 100


@dataclass
class ElectrodeFit:
    ratio: float
    model: str
    params: tuple
    resid_sigma: float          # độ lệch chuẩn sai số tương đối (log) sau khớp
    loocv: dict                 # MAPE LOOCV của từng dạng hàm (%)
    v_params: tuple             # V̄ = v0 + v1·J
    J_min: float
    J_max: float
    excluded: list = field(default_factory=list)

    def csp(self, J):
        return MODELS[self.model][2](self.params, np.asarray(J, float))

    def vbar(self, J, dv=DV_DEFAULT):
        v0, v1 = self.v_params
        return np.clip(v0 + v1 * np.asarray(J, float), 0.05, 0.95 * dv)


def fit_electrodes(df: pd.DataFrame, model="auto", exclude=None,
                   outlier_threshold=None) -> dict[float, ElectrodeFit]:
    """exclude: list các cặp (ratio, J) bỏ khỏi bước khớp (điểm nghi lỗi).
    outlier_threshold: nếu đặt (vd 50), tự bỏ điểm có sai số LOOCV > ngưỡng %."""
    exclude = {(round(r, 6), round(j, 6)) for r, j in (exclude or [])}
    fits = {}
    for ratio, g in df.groupby("ratio"):
        g = g.sort_values("J")
        keep = np.array([(round(ratio, 6), round(j, 6)) not in exclude for j in g.J])
        J, y = g.J.values[keep], g.Csp.values[keep]
        excluded = list(g.J.values[~keep])
        if outlier_threshold is not None and len(J) > 4:
            _, errs = loocv_mape("exp", J, y)
            bad = errs > outlier_threshold
            # chỉ bỏ tối đa 1 điểm tệ nhất
            if bad.any():
                k = int(np.argmax(errs))
                excluded.append(float(J[k]))
                J, y = np.delete(J, k), np.delete(y, k)
        scores = {m: loocv_mape(m, J, y)[0] for m in MODELS}
        name = min(scores, key=scores.get) if model == "auto" else model
        params = MODELS[name][1](J, y)
        pred = MODELS[name][2](params, J)
        resid = np.log(y / pred)
        dof = max(len(J) - len(params), 1)
        sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
        # V̄(J)
        if "Pd" in g and g.Pd.notna().all():
            gk = g[keep] if outlier_threshold is None else g[g.J.isin(J)]
            vb = gk.Pd.values / (1000 * gk.J.values)
            v_params = tuple(np.polyfit(gk.J.values, vb, 1)[::-1])
        else:
            v_params = (0.6, 0.0)
        fits[float(ratio)] = ElectrodeFit(float(ratio), name, tuple(map(float, params)),
                                          sigma, scores, tuple(map(float, v_params)),
                                          float(g.J.min()), float(g.J.max()), excluded)
    return fits


# ----------------------------------------------------------------------------
# 3. Sinh dữ liệu ảo dạng bảng
# ----------------------------------------------------------------------------
@dataclass
class SimConfig:
    n_runs: int = 10000            # lần đo ảo/điện cực (48 điểm × 10000 = 480.000 mẫu)
    mode: str = "anchor"           # "anchor": neo theo số đo gốc | "fit": theo quy luật khớp Csp(J)
    j_mode: str = "measured"       # (chỉ cho mode=fit) "measured" | "range" | "extrapolate"
    n_j_range: int = 12
    extrap_factor: float = 1.5
    repeat_err: float = 0.03       # sai số lặp lại giữa các lần đo (tương đối)
    mass_err: float = 0.02         # sai số cân khối lượng điện cực (dùng chung cho cả lần đo)
    meas_err: float = 0.01         # nhiễu thiết bị trên từng điểm
    repeat_from_data: bool = False # True: lấy sai số lặp lại = độ lệch sau khớp của từng điện cực
    dv: float = DV_DEFAULT
    seed: int = 42


def _j_grid(f, cfg):
    if cfg.mode == "anchor" or cfg.j_mode == "measured":
        return f._measured
    if cfg.j_mode == "range":
        return np.geomspace(f.J_min, f.J_max, cfg.n_j_range)
    return np.geomspace(f.J_min, f.J_max * cfg.extrap_factor, cfg.n_j_range)


def generate_table(fits: dict[float, ElectrodeFit], cfg: SimConfig, df_real=None) -> pd.DataFrame:
    """Sinh n_runs lần đo ảo cho mỗi điện cực. Trong một lần đo ảo:
       - khối lượng điện cực bị cân lệch một lượng δ (chung cho mọi mức dòng),
       - mỗi mức dòng có thêm sai số lặp lại và nhiễu thiết bị riêng.
    mode='anchor': tâm là đúng số đo gốc (Csp, V̄ = Pd/1000J).
    mode='fit'   : tâm là giá trị trên đường khớp (dùng để sinh ở J chưa đo)."""
    rng = np.random.default_rng(cfg.seed)
    out = []
    for ratio, f in fits.items():
        js = np.asarray(_j_grid(f, cfg), float)
        if cfg.mode == "anchor":
            g = df_real[df_real.ratio == ratio].set_index("J").loc[js]
            c_center = g.Csp.values
            v_center = (g.Pd / (1000 * g.index.values)).values
        else:
            c_center = f.csp(js)
            v_center = f.vbar(js, cfg.dv)
        R, K = cfg.n_runs, len(js)
        rep = f.resid_sigma if cfg.repeat_from_data else cfg.repeat_err
        delta_m = rng.normal(0, cfg.mass_err, (R, 1))          # mỗi lần đo cân một lần
        gm = 1.0 / (1.0 + delta_m)
        csp = c_center * np.exp(rng.normal(0, rep, (R, K))) * (1 + rng.normal(0, cfg.meas_err, (R, K))) * gm
        J = js * gm
        vbar = np.clip(v_center * np.exp(rng.normal(0, rep, (R, K))) * (1 + rng.normal(0, cfg.meas_err, (R, K))),
                       0.02, 0.98 * cfg.dv)
        ed = csp * cfg.dv ** 2 / 7.2                            # công thức của lab
        pd_ = 1000 * J * vbar
        dt = ed * 3600 / pd_
        out.append(pd.DataFrame({
            "ratio": ratio,
            "run_id": np.repeat(np.arange(R), K),
            "J_nominal": np.tile(js, R), "J": J.ravel(), "Csp": csp.ravel(), "Ed": ed.ravel(),
            "Pd": pd_.ravel(), "Vbar": vbar.ravel(), "dt_discharge": dt.ravel(),
            "mass_error": np.repeat(delta_m.ravel(), K),
            "extrapolated": np.tile(js > f.J_max * 1.0001, R),
            "source": "synthetic",
        }))
    df = pd.concat(out, ignore_index=True)
    df.insert(0, "sample_id", np.arange(len(df)))
    return df


def attach_measured(fits, df_real):
    for ratio, f in fits.items():
        f._measured = np.sort(df_real.loc[df_real.ratio == ratio, "J"].values)
    return fits


# ----------------------------------------------------------------------------
# 4. Sinh đường cong GCD
# ----------------------------------------------------------------------------
def gcd_curve(csp, J, vbar, dv=DV_DEFAULT, ir_frac=0.03, j_ref=None, eta=0.95,
              noise_v=0.002, n_pts=400, rng=None):
    """Một chu kỳ sạc–xả. Nhánh xả: V(t) = Vs·(1 − t/Δt)^p, với p chọn để
    trung bình V bằng V̄ (Vs/(p+1) = V̄). Δt suy từ Csp:  Csp = 2·J·∫V dt / ΔV².
    Nhánh sạc: tăng gần tuyến tính lên ΔV trong Δt/η (η: hiệu suất coulomb).
    Đây là GIẢ ĐỊNH hình dạng; thay bằng đường cong thật khi có file thô."""
    rng = rng or np.random.default_rng()
    j_ref = j_ref or J
    ir = min(ir_frac * dv * J / j_ref, 0.5 * (dv - vbar))
    vs = dv - ir
    vbar = min(vbar, 0.95 * vs)
    p = vs / vbar - 1
    dt = csp * dv ** 2 / (2 * J * vbar)
    tc = dt / eta
    t_ch = np.linspace(0, tc, n_pts)
    v_ch = dv * (t_ch / tc) ** 0.85
    t_dis = np.linspace(0, dt, n_pts)
    v_dis = vs * (1 - t_dis / dt) ** p
    t = np.concatenate([t_ch, tc + t_dis])
    v = np.concatenate([v_ch, v_dis]) + rng.normal(0, noise_v, 2 * n_pts)
    return t, v, tc  # tc = thời điểm bắt đầu xả


def csp_from_curve(t, v, t_start_dis, J, dv=DV_DEFAULT):
    """Công thức của lab: Csp = 2·I·∫V dt / (m·ΔV²) = 2·J·∫V dt / ΔV²."""
    m = t >= t_start_dis
    integral = np.trapezoid(np.clip(v[m], 0, None), t[m])
    return 2 * J * integral / dv ** 2


# ----------------------------------------------------------------------------
# 5. Kiểm chứng
# ----------------------------------------------------------------------------
def validate(df_real, df_syn, fits, cfg: SimConfig, n_curve_check=200):
    rng = np.random.default_rng(cfg.seed + 1)
    rows = []
    for _, r in df_real.iterrows():
        s = df_syn[(df_syn.ratio == r.ratio) & np.isclose(df_syn.J_nominal, r.J)]
        if s.empty:
            continue
        lo, hi = np.percentile(s.Csp, [5, 95])
        plo, phi = np.percentile(s.Pd, [5, 95])
        rows.append({
            "ratio": r.ratio, "J": r.J, "Csp_real": r.Csp,
            "Csp_syn_median": float(np.median(s.Csp)), "Csp_p5": lo, "Csp_p95": hi,
            "in_band_Csp": bool(lo <= r.Csp <= hi),
            "Pd_real": r.Pd, "Pd_syn_median": float(np.median(s.Pd)),
            "in_band_Pd": bool(plo <= r.Pd <= phi),
            "excluded_from_fit": r.J in fits[r.ratio].excluded,
        })
    point = pd.DataFrame(rows)
    # kiểm tra công thức trên đường cong
    sub = df_syn.sample(min(n_curve_check, len(df_syn)), random_state=cfg.seed)
    errs = []
    for _, s in sub.iterrows():
        t, v, tc = gcd_curve(s.Csp, s.J, s.Vbar, cfg.dv, rng=rng)
        errs.append(abs(csp_from_curve(t, v, tc, s.J, cfg.dv) - s.Csp) / s.Csp * 100)
    used = point[~point.excluded_from_fit] if cfg.mode == "fit" else point
    # Xu hướng: trong mỗi lần đo ảo, Csp có giảm khi J tăng giống số liệu gốc không?
    trend_same, rank_same = [], []
    for ratio, g in df_syn[df_syn.ratio.isin(df_real.ratio.unique())].groupby("ratio"):
        real = df_real[df_real.ratio == ratio].sort_values("J").Csp.values
        real_sign = np.sign(np.diff(real))
        M = g.sort_values(["run_id", "J_nominal"]).Csp.values.reshape(-1, len(real)) \
            if len(g) % len(real) == 0 else None
        if M is not None:
            trend_same.append(np.mean(np.all(np.sign(np.diff(M, axis=1)) == real_sign, axis=1)))
    # Thứ hạng điện cực tại J nhỏ nhất: điện cực tốt nhất ở số liệu gốc có giữ ngôi đầu?
    first = df_real.sort_values("J").groupby("ratio").head(1).set_index("ratio").Csp
    best_real = first.idxmax()
    sJ = df_syn.sort_values("J_nominal").groupby(["ratio", "run_id"]).head(1)
    piv = sJ.pivot(index="run_id", columns="ratio", values="Csp")
    rank_same = float((piv.idxmax(axis=1) == best_real).mean() * 100)
    summary = {
        "so_diem_that": int(len(point)),
        "ti_le_diem_that_trong_dai_5_95_Csp_%": round(100 * used.in_band_Csp.mean(), 1),
        "ti_le_diem_that_trong_dai_5_95_Pd_%": round(100 * used.in_band_Pd.mean(), 1),
        "MAPE_trung_vi_ao_vs_that_Csp_%": round(float(np.mean(
            np.abs(used.Csp_syn_median - used.Csp_real) / used.Csp_real) * 100), 2),
        "sai_so_tinh_lai_Csp_tu_duong_cong_max_%": round(float(np.max(errs)), 3),
        "sai_so_tinh_lai_Csp_tu_duong_cong_tb_%": round(float(np.mean(errs)), 3),
        "lan_do_ao_giu_dung_xu_huong_Csp_theo_J_%": round(float(np.mean(trend_same) * 100), 1) if trend_same else None,
        "lan_do_ao_giu_dien_cuc_tot_nhat_%": round(rank_same, 1),
        "dien_cuc_tot_nhat_so_lieu_goc": float(best_real),
        "Ed_bang_0_6125_Csp": bool(np.allclose(df_syn.Ed, df_syn.Csp * cfg.dv ** 2 / 7.2)),
    }
    return point, summary


def fits_table(fits):
    rows = []
    for r, f in fits.items():
        rows.append({"ratio": r, "dang_ham": f.model, "tham_so": [round(x, 4) for x in f.params],
                     "sai_so_mo_hinh_sigma": round(f.resid_sigma, 4),
                     **{f"LOOCV_{k}_%": round(v, 1) for k, v in f.loocv.items()},
                     "V̄=v0+v1·J": [round(x, 4) for x in f.v_params],
                     "diem_bo_qua_J": [round(x, 2) for x in f.excluded]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Sinh dữ liệu ảo GCD từ số liệu thực nghiệm")
    ap.add_argument("input", help="file Excel của lab hoặc CSV dạng dài (ratio,J,Csp,Pd)")
    ap.add_argument("--out", default="du_lieu_ao.csv")
    ap.add_argument("--n", type=int, default=10000, help="số lần đo ảo cho mỗi điện cực (mặc định 10000)")
    ap.add_argument("--mode", default="anchor", choices=["anchor", "fit"])
    ap.add_argument("--repeat-err", type=float, default=0.03)
    ap.add_argument("--repeat-from-data", action="store_true")
    ap.add_argument("--j-mode", default="measured", choices=["measured", "range", "extrapolate"])
    ap.add_argument("--model", default="auto", choices=["auto", *MODELS])
    ap.add_argument("--mass-err", type=float, default=0.02)
    ap.add_argument("--meas-err", type=float, default=0.01)
    ap.add_argument("--auto-outlier", type=float, default=None,
                    help="tự bỏ 1 điểm/điện cực nếu sai số LOOCV vượt ngưỡng %% (vd 60)")
    ap.add_argument("--exclude", default="", help="bỏ điểm, dạng 'ratio:J;ratio:J'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--export-long", default=None, help="xuất dữ liệu gốc dạng dài ra CSV")
    a = ap.parse_args()

    df = read_lab_excel(a.input) if a.input.lower().endswith((".xlsx", ".xls")) else read_long_csv(a.input)
    if a.export_long:
        df.to_csv(a.export_long, index=False)
    excl = [tuple(map(float, x.split(":"))) for x in a.exclude.split(";") if x]
    fits = attach_measured(fit_electrodes(df, a.model, excl, a.auto_outlier), df)
    cfg = SimConfig(n_runs=a.n, mode=a.mode, j_mode=a.j_mode, repeat_err=a.repeat_err,
                    mass_err=a.mass_err, meas_err=a.meas_err,
                    repeat_from_data=a.repeat_from_data, seed=a.seed)
    syn = generate_table(fits, cfg, df)
    print(fits_table(fits).to_string(index=False))
    point = summ = None
    if a.mode == "anchor" or a.j_mode == "measured":
        point, summ = validate(df, syn, fits, cfg)
        print(json.dumps(summ, ensure_ascii=False, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stem, ext = os.path.splitext(a.out)
    tmp_out = f"{stem}.tmp{ext}"
    try:
        if a.out.lower().endswith(".xlsx"):
            from openpyxl import Workbook

            # Ghi tuần tự để không giữ hơn 6 triệu ô Excel trong bộ nhớ ở N=10000.
            wb = Workbook(write_only=True)
            sheets = [
                ("du_lieu_ao", syn),
                ("du_lieu_goc", df),
                ("quy_luat_khop", fits_table(fits).astype(str)),
                ("cau_hinh", pd.DataFrame([asdict(cfg)])),
            ]
            if point is not None:
                sheets.append(("kiem_chung_tung_diem", point))
            for name, table in sheets:
                ws = wb.create_sheet(name)
                ws.append(list(table.columns))
                for row in table.itertuples(index=False, name=None):
                    ws.append(row)
            if summ is not None:
                ws = wb.create_sheet("kiem_chung_tong_hop")
                ws.append([None, "gia_tri"])
                for key, value in summ.items():
                    ws.append([key, value])
            wb.save(tmp_out)
        else:
            syn.to_csv(tmp_out, index=False)
        os.replace(tmp_out, a.out)
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    print(f"Đã ghi {len(syn)} mẫu ảo vào {a.out}")


if __name__ == "__main__":
    main()
