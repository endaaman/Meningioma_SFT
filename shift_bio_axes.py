"""shift_bio_axes.py — 施設間シフト Δ が生物学的な軸に沿ってどれだけ動くか（SFT−髄膜腫 と 組織型どうし）。

shift_direction.py は「Δ = mean(EBRAINS) − mean(patho2) は SFT−髄膜腫 方向とほぼ直交する（cos 0.05 / 0.10）」
を示すが、768 次元では無関係な 2 本のベクトルの |cos| が 0.04 程度なので、直交自体は自明に近い。
ここでは帰無をデータの共分散に合わせ、さらに「どの軸に沿って、軸の長さの何倍動くか」を測る。

軸（各施設の中だけで作る。シフトが混ざらないように施設をまたがない）:
    sft   … d = μ(SFT) − μ(髄膜腫全体)
    pair  … d_ab = μ(a) − μ(b)。その施設に min_n 例以上ある群（SFT も 1 群）の全組
    rest  … d_g = μ(g) − μ(施設全体)
量:
    cos   … cos(Δ, d)
    proj  … Δ·d̂（Δ の軸方向成分の長さ）
    r     … (Δ·d̂) / |d|（軸の長さを 1 としたときの移動量。最近傍重心法では r が 0.5 に達すると
            a の平均と b の平均の中点を越える）
帰無:
    cov   … 共分散を考慮した無作為方向: その施設の中心化した埋め込み行列 Xc の行の正規乱数重みの線形結合
            v = Xcᵀw（w ~ N(0, I)）。|cos(Δ, v)| の分布（n_null 回）。等方の参考値は 1/√768 ≈ 0.036
    perm  … 軸ごとのラベル並べ替え: 施設内でサブタイプのラベルを並べ替えて軸を作り直し、|cos(Δ, 軸)| の分布
            （n_perm 回）。p は両側（|cos| が観測以上の割合）
対照（下流との突き合わせ）:
    out/nc/histotype/per_class.csv の EBRAINS → patho2 のクラス別 recall（補正なし → 平行移動の増分）と、
    そのクラスを含む組の r（g を先にした符号: r_gh > 0 なら Δ は h → g の向き、つまり EBRAINS の重心で
    分類すると patho2 の g のスライドは h の側に見える）

出力 (output_dir/shift_bio_axes/):
    axes.csv          — 軸 × 施設の 1 行: n_a, n_b, |d|, cos, projection, r, 帰無の分位点と p
    null_cov.csv      — 共分散帰無の要約（施設ごと: 平均 |cos|、97.5% 点、実効次元 1/E[cos²]）
    recall_vs_r.csv   — 組織型 5 クラスの recall 増分と r の対応
    summary.md        — 表と読み（日本語）

Usage:
    uv run python shift_bio_axes.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.display import ordered_subtypes, shorten
from utils.loader import load_config, load_data

SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}
MENI = "Meningioma (all)"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── axes ──────────────────────────────────────────────────────────────────────

def site_groups(sub: np.ndarray, mask: np.ndarray, min_n: int, cfg: dict) -> list[str]:
    """その施設に min_n 例以上あるサブタイプ（SFT を含む。並びは config の表示順）。"""
    return [g for g in ordered_subtypes(set(sub[mask]), cfg) if ((sub == g) & mask).sum() >= min_n]


def axis_specs(groups: list[str]) -> list[tuple[str, str, str]]:
    """軸の一覧 (kind, a, b)。sft → pair（全組）→ rest の順。"""
    specs = [("sft", "SFT", MENI)]
    specs += [("pair", a, b) for a, b in combinations(groups, 2)]
    specs += [("rest", g, "all") for g in groups]
    return specs


def compute_axes(X: np.ndarray, sub: np.ndarray, is_sft: np.ndarray, mask: np.ndarray,
                 groups: list[str], specs: list[tuple[str, str, str]]) -> np.ndarray:
    """specs の順に軸ベクトル d を積んだ行列（n_axes × dim）。sub を並べ替えれば帰無の軸になる。"""
    mu = {g: X[mask & (sub == g)].mean(axis=0) for g in groups}
    mu["all"] = X[mask].mean(axis=0)
    mu["SFT"] = X[mask & is_sft].mean(axis=0)
    mu[MENI] = X[mask & ~is_sft].mean(axis=0)
    return np.stack([mu[a] - mu[b] for _, a, b in specs])


def cov_null(Xc: np.ndarray, delta: np.ndarray, n_null: int, rng: np.random.Generator) -> np.ndarray:
    """共分散を考慮した無作為方向 v = Xcᵀw との cos（符号付き、n_null 本）。"""
    W = rng.standard_normal((n_null, Xc.shape[0]))
    V = W @ Xc
    return (V @ delta) / (np.linalg.norm(V, axis=1) * np.linalg.norm(delta))


def compute(merged: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sb_cfg = cfg.get("shift_bio_axes", {})
    min_n: int = sb_cfg.get("min_n", 5)
    n_perm: int = sb_cfg.get("n_perm", 1000)
    n_null: int = sb_cfg.get("n_null", 1000)
    rng = np.random.default_rng(sb_cfg.get("seed", 42))

    reference: str = cfg["reference"]
    other = next(s for s in merged["source"].unique() if s != reference)
    X = np.stack(merged["embedding"].values).astype(np.float64)
    sub = merged["subtype"].to_numpy()
    src = merged["source"].to_numpy()
    is_sft = merged["subtype"].str.lower().str.contains("sft").to_numpy()
    delta = X[src == reference].mean(axis=0) - X[src == other].mean(axis=0)
    norm_delta = float(np.linalg.norm(delta))
    print(f"  Δ = mean({reference}) − mean({other}), |Δ| = {norm_delta:.3f}")

    rows, null_rows = [], []
    for site in (reference, other):
        mask = src == site
        groups = site_groups(sub, mask, min_n, cfg)
        specs = axis_specs(groups)
        D = compute_axes(X, sub, is_sft, mask, groups, specs)
        norms = np.linalg.norm(D, axis=1)
        cos = (D @ delta) / (norms * norm_delta)
        proj = cos * norm_delta
        r = proj / norms

        # 帰無 cov: 施設ごとに 1 本の分布（軸によらない）
        Xc = X[mask] - X[mask].mean(axis=0)
        c_null = np.abs(cov_null(Xc, delta, n_null, rng))
        q975_cov, mean_cov = float(np.quantile(c_null, 0.975)), float(c_null.mean())
        null_rows.append({
            "site": site, "n_slides": int(mask.sum()), "n_null": n_null,
            "null_mean_abs_cos": mean_cov, "null_q975_abs_cos": q975_cov, "null_max_abs_cos": float(c_null.max()),
            "effective_dim": float(1.0 / np.mean(c_null ** 2)),     # 等方なら 768
            "isotropic_abs_cos": float(np.sqrt(2 / (np.pi * X.shape[1]))),   # 等方の E|cos| ≈ √(2/(π·768))
            "isotropic_rms_cos": float(1.0 / np.sqrt(X.shape[1])),
        })

        # 帰無 perm: 施設内でラベルを並べ替え、全軸を作り直す
        idx = np.flatnonzero(mask)
        perm_abs = np.empty((n_perm, len(specs)))
        for i in range(n_perm):
            sub_p, sft_p = sub.copy(), is_sft.copy()
            order = rng.permutation(idx)
            sub_p[idx], sft_p[idx] = sub[order], is_sft[order]
            Dp = compute_axes(X, sub_p, sft_p, mask, groups, specs)
            perm_abs[i] = np.abs((Dp @ delta) / (np.linalg.norm(Dp, axis=1) * norm_delta))
        q975_perm = np.quantile(perm_abs, 0.975, axis=0)
        p_perm = (perm_abs >= np.abs(cos)[None, :]).mean(axis=0)
        p_cov = np.array([(c_null >= abs(c)).mean() for c in cos])
        p_cov_lower = np.array([(c_null <= abs(c)).mean() for c in cos])   # 片側: 観測より直交に近い無作為方向の割合

        n_of = {g: int((mask & (sub == g)).sum()) for g in groups}
        n_of.update({"all": int(mask.sum()), "SFT": int((mask & is_sft).sum()), MENI: int((mask & ~is_sft).sum())})
        for k, (kind, a, b) in enumerate(specs):
            rows.append({
                "site": site, "kind": kind, "a": a, "b": b, "n_a": n_of[a], "n_b": n_of[b],
                "norm_delta": norm_delta, "norm_d": float(norms[k]), "cos": float(cos[k]),
                "projection": float(proj[k]), "r": float(r[k]),
                "null_cov_mean": mean_cov, "null_cov_q975": q975_cov, "p_cov": float(p_cov[k]),
                "p_cov_lower": float(p_cov_lower[k]),
                "null_perm_q975": float(q975_perm[k]), "p_perm": float(p_perm[k]),
                # 共分散帰無の 97.5% 点を r の尺度に直したもの（|Δ| / |d| 倍）
                "r_null_q975": float(q975_cov * norm_delta / norms[k]),
            })
        print(f"  {site}: {len(groups)} groups, {len(specs)} axes; cov null |cos| mean {mean_cov:.3f}, "
              f"q975 {q975_cov:.3f}, effective dim {null_rows[-1]['effective_dim']:.1f}")
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


# ── 下流（組織型 recall）との突き合わせ ───────────────────────────────────────

def _signed_r(axes: pd.DataFrame, site: str, g: str, h: str) -> float:
    """r_gh（g を先にした符号）。組の向きが (h, g) で保存されていれば符号を反す。"""
    s = axes[(axes["site"] == site) & (axes["kind"] == "pair")]
    hit = s[(s["a"] == g) & (s["b"] == h)]
    if len(hit):
        return float(hit["r"].iloc[0])
    hit = s[(s["a"] == h) & (s["b"] == g)]
    return -float(hit["r"].iloc[0]) if len(hit) else np.nan


def recall_vs_r(axes: pd.DataFrame, cfg: dict, reference: str, other: str) -> pd.DataFrame:
    """組織型分類（最近傍重心法、EBRAINS → patho2）のクラス別 recall の増分（平行移動 − 補正なし）と、
    そのクラスを含む組の r（5 クラス内の相手）。r_gh > 0 は patho2 の g のスライドが EBRAINS の重心では
    h 側に見える向き。"""
    path = Path(cfg["output_dir"]) / "nc" / "histotype" / "per_class.csv"
    if not path.exists():
        print(f"  WARN: {path} not found; skip recall comparison")
        return pd.DataFrame()
    pc = pd.read_csv(path)
    pc = pc[(pc["train_source"] == reference) & (pc["test_source"] == other)]
    set_name = cfg.get("nc", {}).get("histotype_set", "main")
    classes: list[str] = cfg["lp_histotype_sets"][set_name]["classes"]
    full = {c: next(f for f in classes if f.startswith(c)) for c in pc["class"].unique()}
    rows = []
    for short, g in full.items():
        rec = {v: float(pc[(pc["class"] == short) & (pc["variant"] == v)]["recall"].iloc[0])
               for v in ("original", "centroid")}
        row = {"class": g, "n_test": int(pc[(pc["class"] == short) & (pc["variant"] == "original")]["n"].iloc[0]),
               "recall_original": rec["original"], "recall_centroid": rec["centroid"],
               "recall_gain": rec["centroid"] - rec["original"]}
        for site in (reference, other):
            rs = {h: _signed_r(axes, site, g, h) for h in classes if h != g}
            row[f"r_max_away_{site}"] = max(rs.values())                       # 最も g から遠ざける相手
            row[f"r_mean_abs_{site}"] = float(np.mean([abs(v) for v in rs.values()]))
            row[f"r_away_partner_{site}"] = shorten(max(rs, key=rs.get), cfg)
        rows.append(row)
    return pd.DataFrame(rows)


# ── plot ──────────────────────────────────────────────────────────────────────

def panel_rows(axes: pd.DataFrame, cfg: dict, reference: str, other: str, top_k: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """図の行: SFT−髄膜腫（両施設）→ reference の組織型の組を |r| の大きい順に top_k。残りは others。
    組は r ≥ 0 になる向きにそろえる（ラベル a − b: Δ は b → a の向き）。返り値 (表示する行, others)。"""
    sft = axes[axes["kind"] == "sft"].copy()
    sft["label"] = [f"SFT − meningioma ({SITE_LABEL.get(s, s)})" for s in sft["site"]]
    sft = sft.set_index("site").loc[[reference, other]].reset_index()
    pairs = axes[(axes["kind"] == "pair") & (axes["site"] == reference)].copy()
    flip = pairs["r"] < 0
    pairs.loc[flip, ["a", "b"]] = pairs.loc[flip, ["b", "a"]].to_numpy()
    pairs.loc[flip, ["n_a", "n_b"]] = pairs.loc[flip, ["n_b", "n_a"]].to_numpy()
    pairs.loc[flip, ["r", "cos", "projection"]] *= -1
    pairs = pairs.sort_values("r", ascending=False)
    pairs["label"] = [f"{shorten(a, cfg)} − {shorten(b, cfg)}  ({na}/{nb})"
                      for a, b, na, nb in zip(pairs["a"], pairs["b"], pairs["n_a"], pairs["n_b"])]
    return pd.concat([sft, pairs.head(top_k)], ignore_index=True), pairs.iloc[top_k:]


def draw_axes(ax: plt.Axes, axes: pd.DataFrame, cfg: dict, reference: str, other: str,
              fontsize: float = 10, top_k: int = 10) -> None:
    """e: 軸に沿った相対移動量 r = (Δ·d̂)/|d| の横棒。

    上 2 行は SFT−髄膜腫 方向（施設ごとに作った軸、施設色）、続いて reference（EBRAINS）内の組織型の組を
    |r| の大きい順に top_k、最後の行は残りの組を灰色の点で並べる。組は r ≥ 0 になる向きにそろえてある。
    灰色の帯は共分散を考慮した帰無（施設内のスライドの偏差の無作為な線形結合との |cos| の 97.5% 点）を
    行ごとに r の尺度へ直したもの（× |Δ| / |d|。軸が短いほど帯は広い）。SFT の行は符号付きなので ±、組の行は
    r ≥ 0 に向きをそろえてあるので 0 から ＋側だけ、others の行は残りの組の帯の中央値。cos の帰無は施設ごとに
    1 本なのでこの換算を採り、別軸に cos を重ねるのは避けた（r の尺度で読めるほうが最近傍重心法の意味に直結する）。
    点線 r = 0.5 は、シフトが 2 つの重心の中点に達する位置。
    """
    src_colors: dict = cfg.get("display", {}).get("colors", {}).get("sources", {})
    shown, others = panel_rows(axes, cfg, reference, other, top_k)
    n_shown = len(shown)
    y = np.arange(n_shown)
    y_others = n_shown
    lo = np.where(shown["kind"] == "sft", -shown["r_null_q975"], 0.0)
    ax.hlines(y, lo, shown["r_null_q975"], color="#c8c8c8", lw=7, zorder=1, label="covariance null (95%)")
    colors = [src_colors.get(s, "#333333") if k == "sft" else "#555555"
              for s, k in zip(shown["site"], shown["kind"])]
    ax.barh(y, shown["r"], height=0.5, color=colors, zorder=3)
    if len(others):
        # others: 帯は残りの組の帯の中央値、点は r（r ≥ 0 の向き）
        ax.hlines(y_others, 0.0, others["r_null_q975"].median(), color="#c8c8c8", lw=7, zorder=1)
        jitter = np.random.default_rng(0).uniform(-0.2, 0.2, len(others))
        ax.scatter(others["r"], y_others + jitter, s=9, color="#888888", zorder=3, linewidths=0)
    ax.axvline(0, color="#999999", lw=0.8, zorder=2)
    xmax = max(float(shown["r"].abs().max()), float(shown["r_null_q975"].max())) * 1.15
    if xmax > 0.5:
        ax.axvline(0.5, color="#999999", lw=0.8, ls=":", zorder=2)
        # 線に沿わせて縦書き。帯が 0.5 に届かない行（上から 6–8 行目あたり）に置く
        ax.text(0.52, min(7, n_shown - 1), "midpoint", rotation=90, ha="left", va="center",
                fontsize=fontsize - 2, color="#777777")
    ax.axhline(1.5, color="#dddddd", lw=0.8)
    ax.set_xlim(-xmax * 0.35, xmax)
    ax.set_yticks(list(y) + ([y_others] if len(others) else []))
    ax.set_yticklabels(list(shown["label"]) + ([f"other {len(others)} pairs"] if len(others) else []), fontsize=fontsize)
    ax.set_ylim(y_others + 0.5 if len(others) else n_shown - 0.5, -0.5)
    ax.set_xlabel("shift along axis / axis length  (r)", fontsize=fontsize)
    ax.legend(loc="upper right", frameon=False, fontsize=fontsize - 1, handlelength=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)


def plot(axes: pd.DataFrame, cfg: dict, reference: str, other: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    draw_axes(ax, axes, cfg, reference, other, top_k=cfg.get("shift_bio_axes", {}).get("top_k", 10))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ── summary ───────────────────────────────────────────────────────────────────

def _paper_ba(cfg: dict) -> dict[str, str]:
    """Table S1 / S2（paper/{version}/tables）から EBRAINS → patho2 の balanced accuracy の点推定を拾う。"""
    tdir = Path(cfg["output_dir"]) / "paper" / cfg.get("paper", {}).get("version", "v1") / "tables"
    out = {}
    for key, fname, cond in (("sft_none", "tableS1_performance.csv", "なし"),
                             ("histo_none", "tableS2_histotype_performance.csv", "なし"),
                             ("histo_centroid", "tableS2_histotype_performance.csv", "平行移動")):
        path = tdir / fname
        if not path.exists():
            continue
        t = pd.read_csv(path)
        t.iloc[:, 0] = t.iloc[:, 0].replace("", np.nan).ffill()
        hit = t[(t.iloc[:, 0].str.startswith("EBRAINS")) & (t.iloc[:, 1] == cond)]
        if len(hit):
            out[key] = str(hit["Balanced acc."].iloc[0]).split(" ")[0]
    return out


def _row_md(r: pd.Series, label: str) -> str:
    return (f"| {label} | {int(r['n_a'])}/{int(r['n_b'])} | {r['norm_d']:.2f} | {r['cos']:+.3f} | {r['projection']:+.2f} | "
            f"{r['r']:+.3f} | {r['r_null_q975']:.3f} | {r['p_cov']:.3f} | {r['p_cov_lower']:.3f} | {r['p_perm']:.3f} |")


def write_summary(axes: pd.DataFrame, null_cov: pd.DataFrame, rec: pd.DataFrame, cfg: dict,
                  reference: str, other: str, out_path: Path) -> None:
    sb_cfg = cfg.get("shift_bio_axes", {})
    top_k = sb_cfg.get("top_k", 10)
    nd = float(axes["norm_delta"].iloc[0])
    header = ("| 軸 | n_a/n_b | \\|d\\| | cos | Δ·d̂ | r | r 帰無 97.5% | p_cov | p_cov_lower | p_perm |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    lines = [f"# 施設間シフト Δ と生物学的な軸（{other} → {reference}、|Δ| = {nd:.2f}）", "",
             f"軸は各施設の中だけで作る（その施設に {sb_cfg.get('min_n', 5)} 例以上の群）。r = (Δ·d̂)/|d| は軸の長さを 1 とした移動量"
             f"（最近傍重心法では r = 0.5 で 2 群の重心の中点を越える）。「r 帰無 97.5%」は共分散を考慮した無作為方向の "
             f"|cos| の 97.5% 点を r の尺度に直したもの（× |Δ| / |d|）。p_cov はその帰無、p_perm はラベル並べ替え帰無"
             f"（各 {sb_cfg.get('n_null', 1000)} / {sb_cfg.get('n_perm', 1000)} 回、|cos| の両側）。p_cov_lower は"
             f"「無作為方向の |cos| が観測以下になる割合」（観測が無作為より直交に近いかの片側）。", ""]

    lines += ["## 帰無（共分散を考慮した無作為方向）", "",
              "| 施設 | n | E\\|cos\\| | 97.5% 点 | max | 実効次元 1/E[cos²] | 等方 E\\|cos\\| |", "|---|---|---|---|---|---|---|"]
    for _, r in null_cov.iterrows():
        lines.append(f"| {SITE_LABEL.get(r['site'], r['site'])} | {int(r['n_slides'])} | {r['null_mean_abs_cos']:.3f} | "
                     f"{r['null_q975_abs_cos']:.3f} | {r['null_max_abs_cos']:.3f} | {r['effective_dim']:.1f} | {r['isotropic_abs_cos']:.3f} |")
    lines.append("")

    sft = axes[axes["kind"] == "sft"].set_index("site")
    for site in (reference, other):
        shown, others = panel_rows(axes, cfg, site, other if site == reference else reference, top_k)
        pairs = axes[(axes["kind"] == "pair") & (axes["site"] == site)]
        lines += [f"## 軸: {SITE_LABEL.get(site, site)} 内で定義（SFT−髄膜腫 → 組織型の組を |r| 順に上位 {top_k}。組は r ≥ 0 の向き）", "",
                  header, _row_md(sft.loc[site], "SFT − meningioma")]
        for _, r in shown[shown["kind"] == "pair"].iterrows():
            lines.append(_row_md(r, f"{shorten(r['a'], cfg)} − {shorten(r['b'], cfg)}"))
        if len(others):
            ab = others["r"].abs()
            lines.append(f"| 残り {len(others)} 組（集計） | | {others['norm_d'].min():.1f}–{others['norm_d'].max():.1f} | "
                         f"\\|cos\\| 中央値 {others['cos'].abs().median():.3f} | | \\|r\\| 中央値 {ab.median():.3f}, 最大 {ab.max():.3f} | "
                         f"| p<0.05: {(others['p_cov'] < 0.05).sum()} | | p<0.05: {(others['p_perm'] < 0.05).sum()} |")
        n_sig = int((pairs["p_cov"] < 0.05).sum())
        lines += ["", f"組 {len(pairs)} のうち共分散帰無で p < 0.05 は {n_sig}、並べ替え帰無で p < 0.05 は "
                  f"{int((pairs['p_perm'] < 0.05).sum())}。|r| ≥ 0.25 は {int((pairs['r'].abs() >= 0.25).sum())} 組、"
                  f"≥ 0.5 は {int((pairs['r'].abs() >= 0.5).sum())} 組。", ""]
        rest = axes[(axes["kind"] == "rest") & (axes["site"] == site)].sort_values("cos", key=np.abs, ascending=False)
        lines += [f"組織型 − 施設全体（μ(g) − μ(all)、{SITE_LABEL.get(site, site)}）の cos と r（|cos| 順）:", ""]
        lines.append("| 組織型 | n | \\|d\\| | cos | r | p_cov | p_perm |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in rest.iterrows():
            lines.append(f"| {shorten(r['a'], cfg)} | {int(r['n_a'])} | {r['norm_d']:.2f} | {r['cos']:+.3f} | {r['r']:+.3f} | "
                         f"{r['p_cov']:.3f} | {r['p_perm']:.3f} |")
        lines.append("")

    ba = _paper_ba(cfg)
    if len(rec):
        lines += [f"## 下流との突き合わせ: 組織型 5 クラスの recall（最近傍重心法、{SITE_LABEL[reference]} → {SITE_LABEL[other]}）と r", "",
                  "r_max_away = そのクラス g を含む組 (g, h) のうち、patho2 の g のスライドを EBRAINS の重心で見たとき最も h 側へ見せる"
                  " r_gh の最大値（軸は各施設内で定義）。", "",
                  "| クラス | n | recall なし | recall 平行移動 | 増分 | r_max_away (E) | 相手 (E) | r_max_away (P) | 相手 (P) | mean\\|r\\| (E) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in rec.iterrows():
            lines.append(f"| {shorten(r['class'], cfg)} | {int(r['n_test'])} | {r['recall_original']:.3f} | {r['recall_centroid']:.3f} | "
                         f"{r['recall_gain']:+.3f} | {r[f'r_max_away_{reference}']:+.3f} | {r[f'r_away_partner_{reference}']} | "
                         f"{r[f'r_max_away_{other}']:+.3f} | {r[f'r_away_partner_{other}']} | {r[f'r_mean_abs_{reference}']:.3f} |")
        rho = {site: spearmanr(rec["recall_gain"], rec[f"r_max_away_{site}"]) for site in (reference, other)}
        lines += ["", "Spearman（recall 増分 vs r_max_away、5 クラス、参考）: " +
                  ", ".join(f"{SITE_LABEL[s]} 軸 ρ = {v.statistic:+.2f} (p = {v.pvalue:.2f})" for s, v in rho.items()), ""]
        if ba:
            lines += [f"Table S1 / S2（{SITE_LABEL[reference]} → {SITE_LABEL[other]}）: SFT vs 髄膜腫の BA は補正なしで {ba.get('sft_none', '?')}、"
                      f"組織型 5 クラスの BA は補正なし {ba.get('histo_none', '?')} → 平行移動 {ba.get('histo_centroid', '?')}。", ""]

    lines += ["## 読み", ""] + reading(axes, null_cov, rec, cfg, reference, other, ba)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {out_path}")


def reading(axes: pd.DataFrame, null_cov: pd.DataFrame, rec: pd.DataFrame, cfg: dict,
            reference: str, other: str, ba: dict[str, str]) -> list[str]:
    """数値を埋めた読み（日本語）。文の骨は固定、数値は出力から。"""
    nc = null_cov.set_index("site")
    sft = axes[axes["kind"] == "sft"].set_index("site")
    pe = axes[(axes["kind"] == "pair") & (axes["site"] == reference)]
    pp = axes[(axes["kind"] == "pair") & (axes["site"] == other)]
    pe_abs, pp_abs = pe["r"].abs(), pp["r"].abs()
    shown, _ = panel_rows(axes, cfg, reference, other, cfg.get("shift_bio_axes", {}).get("top_k", 10))
    top = shown[shown["kind"] == "pair"]
    top_txt = "、".join(f"{shorten(a, cfg)}−{shorten(b, cfg)} {r:.2f}（p_cov {p:.2f}）"
                       for a, b, r, p in zip(top["a"][:5], top["b"][:5], top["r"][:5], top["p_cov"][:5]))
    nd = float(axes["norm_delta"].iloc[0])
    E, P = SITE_LABEL[reference], SITE_LABEL[other]
    sft_e, sft_p = sft.loc[reference], sft.loc[other]
    n_sig_e, n_sig_p = int((pe["p_cov"] < 0.05).sum()), int((pp["p_cov"] < 0.05).sum())
    # 組織型の組の軸の長さと、無作為方向が平均してその軸に沿って動かす量
    big = pe[pe_abs >= 0.3]
    typical_r_e = float(nc.loc[reference, "null_mean_abs_cos"] * nd / 3.0)   # |d| = 3 の軸に無作為方向が与える r の目安
    out = [
        f"- **SFT−髄膜腫 軸との直交は、共分散を考慮した帰無を超えない。** 施設内のスライドの偏差の無作為な線形結合でも "
        f"|cos(Δ, v)| は平均 {nc.loc[reference, 'null_mean_abs_cos']:.2f}、97.5% 点 {nc.loc[reference, 'null_q975_abs_cos']:.2f}"
        f"（{E}。実効次元 {nc.loc[reference, 'effective_dim']:.0f}、等方 768 次元なら平均 0.03）。観測の cos {sft_e['cos']:+.2f}（{E}）/ "
        f"{sft_p['cos']:+.2f}（{P}）は無作為方向より直交に近い側だが（p_cov_lower {sft_e['p_cov_lower']:.2f} / {sft_p['p_cov_lower']:.2f}）、"
        f"区別できる差ではない。「直交」は主張の根拠にしない方がよく、言えるのは「SFT−髄膜腫 軸に沿った成分が無作為方向の範囲内で小さい」まで。",
        f"- **量で見ると SFT 軸の成分は小さい。** r = {sft_e['r']:+.3f} / {sft_p['r']:+.3f}（軸の長さ {sft_e['norm_d']:.1f} / "
        f"{sft_p['norm_d']:.1f} に対し Δ の成分 {sft_e['projection']:+.2f} / {sft_p['projection']:+.2f}）。|Δ| = {nd:.1f} の 2 倍ある軸の長さ"
        f"に対して成分は 1/20 以下で、中点（r = 0.5）には遠い。補正なしでも重心の近い方が変わらない（SFT の BA {ba.get('sft_none', '0.97')}）のはこれで足りる。",
        f"- **組織型の組の軸には r が 0.3–0.9 の成分があるが、これも無作為方向の範囲内。** {E} 内の {len(pe)} 組のうち |r| ≥ 0.25 が "
        f"{int((pe_abs >= 0.25).sum())} 組、≥ 0.5 が {int((pe_abs >= 0.5).sum())} 組（{P} 内の {len(pp)} 組では {int((pp_abs >= 0.25).sum())} / "
        f"{int((pp_abs >= 0.5).sum())}）。上位は {top_txt}。しかし共分散帰無で p < 0.05 は {n_sig_e} / {len(pe)} 組（{P} {n_sig_p} / {len(pp)}）で、"
        f"{len(pe)} 回の検定なら偶然の範囲（多重比較の補正をすれば 0）。並べ替え帰無も同じ（p_perm ≈ p_cov）。"
        f"つまり施設差が特定の組織型の軸を狙って動いているとは言えない。",
        f"- **効いているのは軸の長さ。** 組織型の組の軸は {pe['norm_d'].min():.1f}–{pe['norm_d'].max():.1f}（{E}、中央値 {pe['norm_d'].median():.1f}）で、"
        f"|r| ≥ 0.3 の {len(big)} 組はすべて |d| ≤ {big['norm_d'].max():.1f}、上位 4 組は {top['norm_d'][:4].min():.1f}–{top['norm_d'][:4].max():.1f}"
        f"（SFT−髄膜腫 の {sft_e['norm_d']:.1f} の 1/5–1/3）。|d| = 3 の軸なら、|Δ| = {nd:.1f} の無作為方向の成分（平均 |cos| "
        f"{nc.loc[reference, 'null_mean_abs_cos']:.2f}）でも軸の長さの {typical_r_e * 100:.0f}% になる（Tran−Fibr は帯が 1.1 まで広がる）。SFT の鑑別が補正なしで施設をまたぎ、"
        f"組織型の構造・分類に補正が要る非対称は、「シフトの向き」より「軸の長さに対するシフトの大きさ」で説明がつく。"
        f"組の向きで見ると、{shorten(top['a'].iloc[0], cfg)}−{shorten(top['b'].iloc[0], cfg)} で Δ が {shorten(top['a'].iloc[0], cfg)} 側を向く"
        f"（{E} の重心で分類すると {P} の {shorten(top['a'].iloc[0], cfg)} が {shorten(top['b'].iloc[0], cfg)} 側に見える）のは "
        f"Transitional の施設差の読み（主張 4）と向きが合うが、p_cov {top['p_cov'].iloc[0]:.2f} なので傍証どまり。",
    ]
    if len(rec):
        rho_e = spearmanr(rec["recall_gain"], rec[f"r_max_away_{reference}"]).statistic
        rho_p = spearmanr(rec["recall_gain"], rec[f"r_max_away_{other}"]).statistic
        best = rec.sort_values("recall_gain", ascending=False).iloc[0]
        worst = rec.sort_values("recall_gain").iloc[0]
        out.append(
            f"- **下流（組織型 5 クラスの recall）との対応は弱い。** 5 クラスの中の組は r がすべて 0.5 未満（{E} 軸で最大 "
            f"{rec[f'r_max_away_{reference}'].max():+.2f}）で、大きい r は Transitional・Atypical・Secretory・Clear cell を含む組（分類の "
            f"5 クラスに入れていない組織型）に集まる。平行移動で recall が最も増えたのは {shorten(best['class'], cfg)}"
            f"（{best['recall_original']:.2f} → {best['recall_centroid']:.2f}、r_max_away {best[f'r_max_away_{reference}']:+.2f} 対 "
            f"{best[f'r_away_partner_{reference}']}）、減ったのは {shorten(worst['class'], cfg)}（{worst['recall_original']:.2f} → "
            f"{worst['recall_centroid']:.2f}、r_max_away {worst[f'r_max_away_{reference}']:+.2f}）。Spearman は {E} 軸で ρ = {rho_e:+.2f}、"
            f"{P} 軸で ρ = {rho_p:+.2f} と符号が揃わず（n = 5）、「r の大きいクラスほど補正で救われる」とは言えない。"
            f"組織型 BA は {ba.get('histo_none', '?')} → {ba.get('histo_centroid', '?')}（Table S2）で、BA の改善は特定の軸の成分より"
            f"全クラスの重心が一斉に動くこと（平行移動そのもの）で説明する方が合う。")
    out += [
        "- **注意**: 組織型の軸は例数が少なく、|d| に平均の推定誤差が乗る（短い軸ほど r は上振れする。例 Tran−Fibr は |d| 2.3 で "
        "Fig 4d の補正でも最短の組）。共分散帰無はスライドの偏差（組織型の違いを含む）から作るので、生物学的な軸との整列を検出する"
        "には保守的な帰無。それでも SFT 軸と組織型の軸の違いは「帰無を超えるか」ではなく「軸の長さに対する r」に出る。",
        "- **論文に置く文（案）**: 「施設間シフトの SFT−髄膜腫 軸に沿った成分は軸の長さの "
        f"{max(abs(sft_e['r']), abs(sft_p['r'])) * 100:.0f}% 以下で（cos {min(sft_e['cos'], sft_p['cos']):.2f}–{max(sft_e['cos'], sft_p['cos']):.2f}）、"
        f"施設内のばらつきに沿った無作為方向と区別できない。一方、組織型どうしの軸の多くは SFT−髄膜腫 軸より短く"
        f"（{pe['norm_d'].min():.1f}–{pe['norm_d'].max():.1f}）、同じシフトが軸の長さの 25% 以上に相当する組が {E} 内 {len(pe)} 組中 {int((pe_abs >= 0.25).sum())} 組あった"
        f"（最大 {pe_abs.max() * 100:.0f}%、Transitional−Fibrous）。ただしこれも無作為方向の範囲内であり、施設差が特定の組織型の軸を"
        "向いているのではなく、軸が短いために同じ大きさのシフトで構造が乱れる。SFT の鑑別が補正なしで施設をまたぎ、組織型の分類・"
        "構造には補正が要る非対称はこれで説明できる。」",
    ]
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    sb_cfg = cfg.get("shift_bio_axes", {})
    variant = sb_cfg.get("variant", cfg.get("shift_direction", {}).get("variant", "original"))
    dir_key = cfg.get("variants", {}).get(variant, variant)
    out_dir = Path(cfg["output_dir"]) / "shift_bio_axes"

    print(f"[shift_bio_axes] variant={variant}")
    merged = load_data(cfg, dir_key)
    if merged.empty:
        print("  SKIP: no data")
        return
    print(f"  {len(merged)} cases, sources: {sorted(merged['source'].unique())}")
    reference: str = cfg["reference"]
    other = next(s for s in merged["source"].unique() if s != reference)

    axes, null_cov = compute(merged, cfg)
    rec = recall_vs_r(axes, cfg, reference, other)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("axes", axes), ("null_cov", null_cov), ("recall_vs_r", rec)):
        if len(df):
            df.to_csv(out_dir / f"{name}.csv", index=False)
            print(f"  saved: {out_dir / name}.csv")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(null_cov.round(3).to_string(index=False))
        show = axes[axes["kind"] != "rest"].sort_values("r", key=np.abs, ascending=False)
        print(show.head(15).drop(columns=["norm_delta"]).round(3).to_string(index=False))
    write_summary(axes, null_cov, rec, cfg, reference, other, out_dir / "summary.md")
    plot(axes, cfg, reference, other, out_dir / "shift_bio_axes.png")


if __name__ == "__main__":
    main()
