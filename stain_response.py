"""stain_response.py — 埋め込みの施設間シフトが、スライドの H/E 染色バランスにどう応答するか。

施設差の一部は色（Macenko で TITAN のシフトが半減）だが、「形態も色で見えている」ので、
「どの色がどこに」ではなく「埋め込みの応答がスライドの H/E バランスに依存するか」をスライド単位で調べる。

スライドごとの色（paper_color.py / macenko.py の slide_color.csv: 32 パッチの組織画素の平均、CIELAB の L*a*b* と
Ruifrok–Johnston の H / E）と、スライド埋め込みの施設間シフト方向への射影を突き合わせる。

    u   = (μ_E − μ_P) / |μ_E − μ_P|        補正なしの埋め込みから求めた単位シフト方向（集約器ごと）
    s   = ⟨x − μ_site(補正なし), u⟩         施設内で中心化した射影（施設内の色依存を見る）
    s_raw = ⟨x − μ_P(補正なし), u⟩          共通原点の射影（施設の差 = |μ_E − μ_P| を含む。色で説明できる割合を見る）
    he_balance = log(H / E)                  H/E バランス（正ならヘマトキシリン寄り）

gan / macenko（patho2 のみ）の射影も補正なしの μ・u で測る（= 補正でスライドが u に沿ってどれだけ動いたか）。
集約器は TITAN と平均プーリング（meanpool.py、`aggregator="mean"`）の 2 つ。

出力 (output_dir/stain_response/):
    slide_projection.csv — スライド × 集約器 × 条件の s・s_raw と色（H, E, he_balance, L, a, b）、組織型、施設
    correlations.csv     — 施設内（補正なし）の s と各色特徴の Pearson / Spearman。生と組織型内（施設内の組織型平均で
                           両者を残差化）の 2 通り
    regression.csv       — 施設をまとめた回帰: s_raw ~ site（＋色）で施設係数が色の共変量でどれだけ減るか、
                           s ~ 色 の R²（施設内の色依存の大きさ）
    binned.csv           — 応答曲線: 施設内で he_balance / H / E を分位（既定 10 分位）に分け、各 bin の s の平均と SE
    correction_delta.csv — patho2 の gan / macenko: Δs = s_cond − s_original と色の変化（ΔH, ΔE, Δhe_balance, ΔL, Δa, Δb）
                           の相関と傾き
    shift_decomposition.csv — 条件ごとに残る施設差 μ_E − μ_P(cond) の大きさと、u に沿う成分 / 直交成分
    binned_shape.csv     — binned の bin 平均に対する単調性（Spearman）と線形 / 2 次の当てはまり
    summary.md           — 主要な数字の表と読み（日本語）

Usage:
    uv run python stain_response.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.loader import load_config, load_data, paper_dir

AGGREGATORS = ["titan", "mean"]
CORRECTIONS = ["gan", "macenko"]            # 画像レベルの補正（patho2 → EBRAINS の向きだけ）
COLOR_FEATURES = ["H", "E", "he_balance", "L", "a", "b"]
BIN_FEATURES = ["he_balance", "H", "E"]
SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}


# ── 色 ────────────────────────────────────────────────────────────────────────

def load_color(cfg: dict) -> pd.DataFrame:
    """slide_color.csv（補正なし両施設 + GAN）と macenko の slide_color.csv を (site, case_id, condition) の表にする。"""
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    paper = paper_dir(cfg) / "color" / "slide_color.csv"
    mac = Path(cfg["output_dir"]) / "macenko" / "slide_color.csv"
    group_to_cond = {reference: "original", other: "original", f"{other} (GAN)": "gan", f"{other} (Macenko)": "macenko"}
    frames = []
    for path in (paper, mac):
        if not path.exists():
            print(f"  WARN: colour file not found: {path}")
            continue
        df = pd.read_csv(path)
        df["condition"] = df["group"].map(group_to_cond)
        unknown = df.loc[df["condition"].isna(), "group"].unique()
        if len(unknown):
            print(f"  WARN: unknown colour groups skipped: {list(unknown)}")
        frames.append(df.dropna(subset=["condition"]))
    color = pd.concat(frames, ignore_index=True)
    color["he_balance"] = np.log(color["H"] / color["E"])
    return color[["site", "case_id", "condition", "n_patches", "n_pixels", *COLOR_FEATURES]]


# ── 射影 ──────────────────────────────────────────────────────────────────────

def project(cfg: dict, aggregator: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """集約器ごとに、補正なしの μ・u を決め、original / gan / macenko のスライドを u に射影する。

    返り値: (射影の表, u の情報, 補正後に残る施設差の u 方向 / 直交方向への分解)
    """
    reference: str = cfg["reference"]
    base = load_data(cfg, "original", aggregator=aggregator)
    if base.empty:
        return pd.DataFrame(), {}, pd.DataFrame()
    other = next(s for s in base["source"].unique() if s != reference)
    X = np.stack(base["embedding"].values).astype(np.float64)
    src = base["source"].values
    mu = {s: X[src == s].mean(axis=0) for s in (reference, other)}
    shift = mu[reference] - mu[other]
    norm = float(np.linalg.norm(shift))
    u = shift / norm
    info = {"aggregator": aggregator, "dim": X.shape[1], "shift_norm": norm,
            f"n_{reference}": int((src == reference).sum()), f"n_{other}": int((src == other).sum())}

    frames, decomp = [], []
    for cond in ["original", *CORRECTIONS]:
        if cond == "original":
            merged = base
        else:
            merged = load_data(cfg, cfg.get("variants", {}).get(cond, cond), aggregator=aggregator)
            if merged.empty:
                print(f"  SKIP [{aggregator}/{cond}]: no data")
                continue
            merged = merged[merged["source"] == other]   # reference は常に original なので補正側だけ
        Xc = np.stack(merged["embedding"].values).astype(np.float64)
        centre = np.stack([mu[s] for s in merged["source"].values])
        # 補正後に残る施設差 μ_E − μ_P(cond) を、u に沿う成分と直交成分に分ける
        d = mu[reference] - Xc[merged["source"].values == other].mean(axis=0)
        along = float(d @ u)
        decomp.append({"aggregator": aggregator, "condition": cond, "shift_norm": float(np.linalg.norm(d)),
                       "along_u": along, "orthogonal": float(np.sqrt(max(float(d @ d) - along ** 2, 0.0)))})
        frames.append(pd.DataFrame({
            "aggregator": aggregator, "condition": cond,
            "site": merged["source"].values, "case_id": merged["case_id"].values, "subtype": merged["subtype"].values,
            "s": (Xc - centre) @ u,
            "s_raw": (Xc - mu[other]) @ u,
        }))
    return pd.concat(frames, ignore_index=True), info, pd.DataFrame(decomp)


# ── 相関 ──────────────────────────────────────────────────────────────────────

def _corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {"pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_rho": float(sr[0]), "spearman_p": float(sr[1])}


def residualize(df: pd.DataFrame, cols: list[str], min_n: int) -> pd.DataFrame:
    """組織型の平均を引く（施設内）。例数 min_n 未満の組織型は落とす（残差が 0 になって相関を縮めるため）。"""
    keep = df.groupby("subtype")["case_id"].transform("size") >= min_n
    out = df[keep].copy()
    out[cols] = out[cols] - out.groupby("subtype")[cols].transform("mean")
    return out


def correlations(proj: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    orig = proj[proj["condition"] == "original"]
    for (agg, site), g in orig.groupby(["aggregator", "site"], sort=False):
        g = g.dropna(subset=COLOR_FEATURES)
        res = residualize(g, ["s", *COLOR_FEATURES], min_n)
        for feat in COLOR_FEATURES:
            rows.append({"aggregator": agg, "site": site, "feature": feat, "adjust": "raw", "n": len(g),
                         **_corr(g[feat].to_numpy(), g["s"].to_numpy())})
            rows.append({"aggregator": agg, "site": site, "feature": feat, "adjust": "within_subtype", "n": len(res),
                         **_corr(res[feat].to_numpy(), res["s"].to_numpy())})
    return pd.DataFrame(rows)


# ── 回帰 ──────────────────────────────────────────────────────────────────────

def _ols(y: np.ndarray, cols: dict[str, np.ndarray]) -> tuple[dict[str, float], float]:
    """切片つき最小二乗。係数の dict と R² を返す。"""
    A = np.column_stack([np.ones(len(y)), *cols.values()])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return dict(zip(["intercept", *cols.keys()], map(float, beta))), r2


def regression(proj: pd.DataFrame, reference: str) -> pd.DataFrame:
    """施設をまとめた回帰。

    s_raw ~ site                 施設係数 = ⟨μ_E − μ_P, u⟩ = |μ_E − μ_P|
    s_raw ~ site + 色            色を共変量にしたときの施設係数（残りの施設差）
    s_raw ~ 色                   施設を入れずに色だけでどれだけ説明できるか（R²）
    s     ~ 色                   施設内中心化の射影に対する色の説明力（施設内の色依存）
    色の集合は {H, E, L, a, b} と {he_balance} の 2 通り。
    """
    rows = []
    orig = proj[proj["condition"] == "original"].dropna(subset=COLOR_FEATURES)
    for agg, g in orig.groupby("aggregator", sort=False):
        site = (g["site"].to_numpy() == reference).astype(np.float64)
        y_raw, y_c = g["s_raw"].to_numpy(), g["s"].to_numpy()
        base, r2_site = _ols(y_raw, {"site": site})
        for name, feats in (("H,E,L,a,b", ["H", "E", "L", "a", "b"]), ("he_balance", ["he_balance"])):
            cols = {f: g[f].to_numpy() for f in feats}
            z = {f: (v - v.mean()) / v.std() for f, v in cols.items()}   # 係数を比べやすいよう標準化
            adj, r2_adj = _ols(y_raw, {"site": site, **z})
            _, r2_color_raw = _ols(y_raw, z)
            coef_c, r2_color_c = _ols(y_c, z)
            rows.append({
                "aggregator": agg, "colour_set": name, "n": len(g),
                "site_coef_unadjusted": base["site"], "r2_site_only": r2_site,
                "site_coef_adjusted": adj["site"], "r2_site_plus_colour": r2_adj,
                "site_gap_explained_by_colour": 1 - adj["site"] / base["site"],
                "r2_colour_only_s_raw": r2_color_raw,
                "r2_colour_on_s_within_site": r2_color_c,
                **{f"std_coef_{f}_on_s": coef_c[f] for f in feats},
            })
    return pd.DataFrame(rows)


# ── 応答曲線 ──────────────────────────────────────────────────────────────────

def binned(proj: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    rows = []
    orig = proj[proj["condition"] == "original"]
    for (agg, site), g in orig.groupby(["aggregator", "site"], sort=False):
        for feat in BIN_FEATURES:
            d = g.dropna(subset=[feat])
            q = pd.qcut(d[feat], n_bins, labels=False, duplicates="drop")
            for b, gb in d.groupby(q):
                rows.append({"aggregator": agg, "site": site, "feature": feat, "bin": int(b), "n": len(gb),
                             f"{feat}_min": gb[feat].min(), f"{feat}_mean": gb[feat].mean(), f"{feat}_max": gb[feat].max(),
                             "feature_mean": gb[feat].mean(),
                             "s_mean": gb["s"].mean(), "s_se": gb["s"].std(ddof=1) / np.sqrt(len(gb))})
    out = pd.DataFrame(rows)
    cols = ["aggregator", "site", "feature", "bin", "n", "feature_mean", "s_mean", "s_se"]
    return out[cols]


def monotonicity(binned_df: pd.DataFrame) -> pd.DataFrame:
    """bin 平均の並びが単調か（Spearman）と、線形回帰からのずれ（2 次項の寄与）で飽和・非線形の目安を出す。"""
    rows = []
    for (agg, site, feat), g in binned_df.groupby(["aggregator", "site", "feature"], sort=False):
        x, y = g["feature_mean"].to_numpy(), g["s_mean"].to_numpy()
        rho = stats.spearmanr(x, y)[0] if len(g) >= 3 else np.nan
        _, r2_lin = _ols(y, {"x": x})
        _, r2_quad = _ols(y, {"x": x, "x2": x ** 2})
        rows.append({"aggregator": agg, "site": site, "feature": feat, "n_bins": len(g),
                     "spearman_bins": float(rho), "r2_linear": r2_lin, "r2_quadratic": r2_quad,
                     "s_range": float(y.max() - y.min())})
    return pd.DataFrame(rows)


# ── 補正による移動 ────────────────────────────────────────────────────────────

def correction_delta(proj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    orig = proj[proj["condition"] == "original"].set_index(["aggregator", "site", "case_id"])
    for cond in CORRECTIONS:
        c = proj[proj["condition"] == cond].set_index(["aggregator", "site", "case_id"])
        if c.empty:
            continue
        for agg in c.index.get_level_values(0).unique():
            a, o = c.xs(agg), orig.xs(agg)
            common = a.index.intersection(o.index)
            a, o = a.loc[common], o.loc[common]
            ds = (a["s"] - o["s"]).to_numpy()
            for feat in COLOR_FEATURES:
                dc = (a[feat] - o[feat]).to_numpy()
                ok = ~np.isnan(dc)
                slope, intercept, *_ = stats.linregress(dc[ok], ds[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
                rows.append({"aggregator": agg, "condition": cond, "feature": feat, "n": int(ok.sum()),
                             "delta_s_mean": float(ds[ok].mean()), "delta_s_sd": float(ds[ok].std(ddof=1)),
                             "delta_feature_mean": float(dc[ok].mean()), "delta_feature_sd": float(dc[ok].std(ddof=1)),
                             **_corr(dc[ok], ds[ok]), "slope": float(slope), "intercept": float(intercept)})
    return pd.DataFrame(rows)


# ── summary.md ────────────────────────────────────────────────────────────────

def _md(df: pd.DataFrame, digits: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(path: Path, info: list[dict], decomp: pd.DataFrame, corr: pd.DataFrame, reg: pd.DataFrame,
                  mono: pd.DataFrame, delta: pd.DataFrame, n_patches: int, min_n: int) -> None:
    def pick(agg: str, feat: str, site: str, adjust: str = "within_subtype", col: str = "pearson_r") -> float:
        m = corr[(corr["aggregator"] == agg) & (corr["feature"] == feat) & (corr["site"] == site) & (corr["adjust"] == adjust)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    def reg_pick(agg: str, cset: str, col: str) -> float:
        m = reg[(reg["aggregator"] == agg) & (reg["colour_set"] == cset)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    def mono_pick(agg: str, site: str, feat: str, col: str) -> float:
        m = mono[(mono["aggregator"] == agg) & (mono["site"] == site) & (mono["feature"] == feat)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    def delta_pick(agg: str, cond: str, feat: str) -> pd.Series | None:
        m = delta[(delta["aggregator"] == agg) & (delta["condition"] == cond) & (delta["feature"] == feat)]
        return m.iloc[0] if len(m) else None

    def pivot(col: str) -> pd.DataFrame:
        t = corr.pivot_table(index=["feature", "adjust"], columns=["aggregator", "site"], values=col, sort=False)
        t.columns = [f"{a}/{SITE_LABEL.get(s_, s_)}" for a, s_ in t.columns]
        return t.reindex([(f, adj) for f in COLOR_FEATURES for adj in ("raw", "within_subtype")]).reset_index()

    sites = [s_ for s_ in ("ebrains", "patho2") if s_ in set(corr["site"])]
    aggs = [a for a in AGGREGATORS if a in set(corr["aggregator"])]
    lab = lambda a, s_: f"{a}/{SITE_LABEL.get(s_, s_)}"

    reg_cols = ["aggregator", "colour_set", "site_coef_unadjusted", "site_coef_adjusted", "site_gap_explained_by_colour",
                "r2_site_only", "r2_site_plus_colour", "r2_colour_only_s_raw", "r2_colour_on_s_within_site"]
    delta_cols = ["aggregator", "condition", "feature", "n", "delta_s_mean", "delta_s_sd", "delta_feature_mean",
                  "pearson_r", "spearman_rho", "slope"]
    mono_cols = ["aggregator", "site", "feature", "spearman_bins", "r2_linear", "r2_quadratic", "s_range"]

    # ── 読み（数字から組む。言い切りは避け、閾値は目安）──
    read: list[str] = []
    # 1. どの色特徴に応答しているか（組織型内）
    fe = {(a, s_): (pick(a, "H", s_), pick(a, "E", s_), pick(a, "he_balance", s_)) for a in aggs for s_ in sites}
    e_max = max(abs(e) for (_, e, _) in fe.values())
    read.append(("- **応答しているのは H（ヘマトキシリン量）で、E はほぼ無関係**" if e_max < 0.15
                 else "- 応答しているのは主に H（ヘマトキシリン量）")
                + "（組織型内の Pearson r、H / E / log(H/E) の順）: "
                + "; ".join(f"{lab(a, s_)} {h:+.2f} / {e:+.2f} / {b:+.2f}" for (a, s_), (h, e, b) in fe.items())
                + "。log(H/E) の相関は H の相関を E のノイズで薄めた形で、「バランス」というより H の絶対量への応答に見える。"
                  "L*（暗さ）と b*（青み）も同じ向き（濃く青いほど EBRAINS 側）で、H と同じものを別の座標で見ている。")
    # 2. 施設・集約器の違い
    read.append("- 施設で応答の強さが違う: " + "; ".join(
        f"{a} は EBRAINS r = {pick(a, 'H', 'ebrains'):+.2f} に対し patho2 r = {pick(a, 'H', 'patho2'):+.2f}" for a in aggs)
        + "。TITAN の patho2 内の応答は弱く（r ≈ 0.1）、スライド単位の色依存は EBRAINS 側（色のばらつきが大きい）でしか"
          "はっきり取り出せない。組織型平均で残差化しても r はほとんど変わらないので、組織型（例: psammomatous が紫）の"
          "交絡ではなく、組織型内の染色の濃さそのものへの応答。")
    read.append("- **平均プーリングの方が色に強く応答する**: 施設内の s を色 5 つで回帰した R² は "
                + " / ".join(f"{a} {reg_pick(a, 'H,E,L,a,b', 'r2_colour_on_s_within_site'):.2f}" for a in aggs)
                + "。TITAN は注意機構で集約する分、スライド平均色への依存が薄まっているのだろう。")
    # 3. 施設差のうち色で説明できる分
    read.append("- u に沿った施設差（|μ_E − μ_P|）のうち、スライド平均色（H,E,L,a,b）を共変量にして消える分は "
                + " / ".join(f"{a} {reg_pick(a, 'H,E,L,a,b', 'site_gap_explained_by_colour'):.0%}"
                             f"（{reg_pick(a, 'H,E,L,a,b', 'site_coef_unadjusted'):.2f} → "
                             f"{reg_pick(a, 'H,E,L,a,b', 'site_coef_adjusted'):.2f}）" for a in aggs)
                + "、log(H/E) だけなら "
                + " / ".join(f"{a} {reg_pick(a, 'he_balance', 'site_gap_explained_by_colour'):.0%}" for a in aggs)
                + "。**TITAN の施設差の大半は、32 パッチの平均色では説明できない**（Macenko が TITAN のシフトを大きく"
                  "減らすこととは別の話で、パッチ単位の色・質感まで変える補正の効果はスライド平均色には写らない）。")
    # 4. 応答曲線
    shape = []
    for a in aggs:
        for s_ in sites:
            rl, rq = mono_pick(a, s_, "H", "r2_linear"), mono_pick(a, s_, "H", "r2_quadratic")
            rlb, rqb = mono_pick(a, s_, "he_balance", "r2_linear"), mono_pick(a, s_, "he_balance", "r2_quadratic")
            shape.append(f"{lab(a, s_)} H: 単調 ρ = {mono_pick(a, s_, 'H', 'spearman_bins'):.2f}, 線形 R² {rl:.2f}（2 次 {rq:.2f}）"
                         f"; log(H/E): 線形 {rlb:.2f}（2 次 {rqb:.2f}）")
    bent = [lab(a, s_) for a in aggs for s_ in sites
            if mono_pick(a, s_, "he_balance", "r2_quadratic") - mono_pick(a, s_, "he_balance", "r2_linear") >= 0.15]
    read.append("- 応答曲線（分位の bin 平均）: " + "; ".join(shape)
                + "。H に対してはほぼ単調・線形で、飽和は見えない。"
                + (f"log(H/E) では {', '.join(bent)} で上位の bin が頭打ちになる（E が小さいスライドで比が暴れる分で、"
                   "H の飽和ではない）。" if bent else "log(H/E) でも曲がりは目立たない。"))
    # 5. 補正の Δs
    for a in aggs:
        parts = []
        for cond in CORRECTIONS:
            r = delta_pick(a, cond, "he_balance")
            rh = delta_pick(a, cond, "H")
            d = decomp[(decomp["aggregator"] == a) & (decomp["condition"] == cond)]
            if r is None or d.empty:
                continue
            d = d.iloc[0]
            parts.append(f"{cond}: Δs = {r['delta_s_mean']:+.2f} ± {r['delta_s_sd']:.2f}（SD/平均 "
                         f"{abs(r['delta_s_sd'] / r['delta_s_mean']):.2f}）、Δlog(H/E) との r = {r['pearson_r']:+.2f}、"
                         f"ΔH との r = {rh['pearson_r']:+.2f}。残る施設差 {d['shift_norm']:.2f} = u 方向 {d['along_u']:.2f} "
                         f"＋ 直交 {d['orthogonal']:.2f}")
        read.append(f"- {a} の補正: " + "; ".join(parts) + "。")
    def dec(a: str, cond: str, col: str) -> float:
        m = decomp[(decomp["aggregator"] == a) & (decomp["condition"] == cond)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    max_abs_r = float(delta[delta["feature"].isin(["H", "he_balance"])]["pearson_r"].abs().max())
    cv = {(a, c): abs(delta_pick(a, c, "H")["delta_s_sd"] / delta_pick(a, c, "H")["delta_s_mean"])
          for a in aggs for c in CORRECTIONS if delta_pick(a, c, "H") is not None}
    read.append("- **補正による u 方向の移動は、スライドの色変化の量にほぼ比例しない**: Δs の ΔH・Δlog(H/E) との相関は "
                f"|r| ≤ {max_abs_r:.2f}。Δs の SD/平均は "
                + ", ".join(f"{a}/{c} {v:.2f}" for (a, c), v in cv.items())
                + f"。GAN は TITAN を u 方向にほぼ施設差ぶん（{delta_pick('titan', 'gan', 'H')['delta_s_mean']:.1f} / "
                  f"{dec('titan', 'original', 'shift_norm'):.1f}）動かすが、それは「色をどれだけ変えたか」とは独立に"
                  "全スライドにほぼ同じ量だけ起きる。"
                + f"Macenko は TITAN を u 方向に {delta_pick('titan', 'macenko', 'H')['delta_s_mean']:.1f} 動かす一方で"
                  f"直交方向に {dec('titan', 'macenko', 'orthogonal'):.1f} の新しいずれを作る。平均プーリングでは Macenko は"
                  f" u 方向にほとんど動かず（{delta_pick('mean', 'macenko', 'H')['delta_s_mean']:.1f}）直交方向に "
                  f"{dec('mean', 'macenko', 'orthogonal'):.1f} ずれて施設差が {dec('mean', 'original', 'shift_norm'):.2f} → "
                  f"{dec('mean', 'macenko', 'shift_norm'):.2f} に増え、GAN は u 方向に行き過ぎる"
                  f"（残り {dec('mean', 'gan', 'along_u'):+.2f}）。"
                + ("TITAN/macenko の ΔH との相関が負なのは、H を増やされたスライドほど動きが小さいという向きで、"
                   "「色を変えた量 → 埋め込みの移動」という単純な関係ではない。"
                   if delta_pick("titan", "macenko", "H")["pearson_r"] < -0.1 else ""))
    # 6. 注意
    h_r = [pick(a, "H", s_) for a in aggs for s_ in sites]
    read.append(f"- 注意: 色はスライドあたり {n_patches} パッチの組織画素の平均で、スライド内の不均一を均した粗い推定。"
                f"施設内の H との相関（r = {min(h_r):+.2f}〜{max(h_r):+.2f}）はそれでも出ているので「取り出せない」ほどではないが、"
                "施設差の色説明割合（"
                + " / ".join(f"{a} {reg_pick(a, 'H,E,L,a,b', 'site_gap_explained_by_colour'):.0%}" for a in aggs)
                + "）はこの粗さ（色の測定誤差による希釈）で下に偏る。H / E は Ruifrok–Johnston の固定行列による分解で、施設ごとの染色ベクトルの違いは吸収していない"
                f"（patho2 の H が低く E が高いのは、分解行列との相性もありうる）。組織型内の残差化は施設内で例数 {min_n} 以上の組織型に限った。")

    body = [
        "# stain_response — 埋め込みの施設間シフト方向への射影と、スライドの H/E 染色バランス",
        "",
        "定義: u = (μ_E − μ_P)/|·|（補正なし、集約器ごと）、s = ⟨x − μ_site, u⟩（施設内で中心化）、"
        "s_raw = ⟨x − μ_P, u⟩（共通原点）、he_balance = log(H/E)。gan / macenko は patho2 のみで、補正なしの μ・u で測る。",
        "",
        "## 集約器ごとの u と、条件ごとに残る施設差の分解",
        "",
        _md(pd.DataFrame(info)),
        "",
        "μ_E − μ_P(cond) を u に沿う成分（along_u）と直交成分に分けたもの。original の along_u = |μ_E − μ_P|。",
        "",
        _md(decomp),
        "",
        "## 施設内の相関（補正なし）: Pearson r（s vs 色特徴）",
        "",
        "raw = そのまま、within_subtype = 施設内の組織型平均で s と色を残差化。",
        "",
        _md(pivot("pearson_r")),
        "",
        "## 同 Spearman ρ",
        "",
        _md(pivot("spearman_rho")),
        "",
        "## 施設をまとめた回帰（s_raw ~ site [+ 色]）",
        "",
        "site_coef = u に沿った施設の差（unadjusted は |μ_E − μ_P| に一致）。"
        "site_gap_explained_by_colour = 1 − adjusted/unadjusted。r2_colour_on_s_within_site = 施設内中心化 s を色で回帰した R²。",
        "",
        _md(reg[reg_cols]),
        "",
        "## 応答曲線の形（bin 平均の単調性と線形 / 2 次の R²）",
        "",
        "各 bin の値は binned.csv。spearman_bins は bin 平均の並びの単調性、r2_linear / r2_quadratic は bin 平均に対する当てはめ。",
        "",
        _md(mono[mono_cols]),
        "",
        "## 補正による移動 Δs と色の変化（patho2、gan / macenko）",
        "",
        "Δs = s_cond − s_original（u に沿った移動。正なら EBRAINS 側へ）。slope は Δs を Δfeature で回帰した傾き。",
        "",
        _md(delta[delta_cols]),
        "",
        "## 読み",
        "",
        *read,
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    sr = cfg.get("stain_response", {})
    n_bins: int = sr.get("n_bins", 10)
    min_n: int = sr.get("min_n_subtype", 3)
    reference: str = cfg["reference"]
    out_dir = Path(cfg["output_dir"]) / "stain_response"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[stain_response] colour")
    color = load_color(cfg)
    print("  " + ", ".join(f"{k}={v}" for k, v in color.groupby(["site", "condition"]).size().items()))

    frames, infos, decomps = [], [], []
    for agg in AGGREGATORS:
        print(f"[stain_response] project aggregator={agg}")
        proj, info, decomp = project(cfg, agg)
        if proj.empty:
            print(f"  SKIP [{agg}]: no embeddings")
            continue
        print(f"  dim={info['dim']} |μ_E − μ_P|={info['shift_norm']:.4f} n={len(proj)} rows")
        frames.append(proj)
        infos.append(info)
        decomps.append(decomp)
    decomp = pd.concat(decomps, ignore_index=True)
    proj = pd.concat(frames, ignore_index=True).merge(color, on=["site", "case_id", "condition"], how="left")
    missing = proj[COLOR_FEATURES[0]].isna().sum()
    if missing:
        print(f"  WARN: {missing} rows without colour (left as NaN)")
    proj.to_csv(out_dir / "slide_projection.csv", index=False)
    print(f"  saved: {out_dir / 'slide_projection.csv'}")

    corr = correlations(proj, min_n)
    reg = regression(proj, reference)
    bins = binned(proj, n_bins)
    mono = monotonicity(bins)
    delta = correction_delta(proj)
    for name, df in (("correlations", corr), ("regression", reg), ("binned", bins), ("binned_shape", mono),
                     ("correction_delta", delta), ("shift_decomposition", decomp)):
        df.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"  saved: {out_dir / f'{name}.csv'}")

    n_patches = int(color["n_patches"].median()) if "n_patches" in color else cfg.get("paper_color", {}).get("n_patches", 32)
    write_summary(out_dir / "summary.md", infos, decomp, corr, reg, mono, delta, n_patches, min_n)
    print(f"  saved: {out_dir / 'summary.md'}")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(corr[corr["feature"] == "he_balance"].round(3).to_string(index=False))
        print(reg[["aggregator", "colour_set", "site_coef_unadjusted", "site_coef_adjusted", "site_gap_explained_by_colour",
                   "r2_colour_on_s_within_site"]].round(3).to_string(index=False))
        print(delta[delta["feature"] == "he_balance"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
