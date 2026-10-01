"""harmonization.py — 施設差補正の効き具合の指標（補正前 / centroid / GAN）。

各状態のスライド埋め込み（reference 施設は常に original、他施設だけが補正 variant）について:
    ASW-batch  … 施設ラベルのシルエット係数（0 に近いほど施設が混ざる）
    ASW-bio    … サブタイプラベルのシルエット係数（大きいほどサブタイプが分かれる）
    ASW-class  … SFT / 髄膜腫 の 2 値ラベルのシルエット係数
    iLISI      … 近傍の施設ラベルの逆シンプソン指数の平均（1〜施設数。大きいほど混ざる）
    cLISI      … 近傍のサブタイプラベルの逆シンプソン指数の平均（1 に近いほどサブタイプが純粋）

LISI は k 近傍の一様重みなので値が離散的になり、中央値は同値に潰れやすい。図には平均を使い、
CSV には中央値（*_median）も残す。

ASW・LISI はともに centroid.py と同じく PCA（centroid.pca_n_components 次元）に落としてから計算する。
LISI は Korsunsky et al. (Harmony) の perplexity 重み付きではなく、k 近傍の一様重みによる簡易版。

施設差の帰無分布（施設ラベルの並べ替え検定）:
    各状態で施設ラベルを n_perm 回並べ替え、ASW-batch / iLISI の帰無分布を作る。
      free           … 全体でシャッフル
      within_subtype … サブタイプ内でシャッフル（施設ごとのサブタイプ構成差は保つ。本文の基準）
    p は「帰無が観測以上に施設が分かれている（ASW は ≥、iLISI は ≤）」割合（+1 補正）。
    残った施設差の割合 = (観測 − 帰無平均) / (補正なしの観測 − 補正なしの帰無平均)。
    補正なしを 100%、偶然のレベルを 0% とする。

出力 (output_dir/harmonization/):
    metrics.csv
    metrics.png
    site_null.csv          … state, null, metric, obs, null_mean, null_q025, null_q975, p
    residual_fraction.csv  … state, null, metric, residual（0〜1）

Usage:
    uv run python harmonization.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from utils.display import CONDITION_COLORS, CONDITION_ORDER, order_conditions
from utils.loader import load_config, load_data

STATE_LABELS = {"original": "None", "gan": "GAN", "centroid": "Centroid", "combat": "ComBat"}  # 補正の種類
STATE_COLORS = CONDITION_COLORS
METRIC_INFO = {
    # key: (表示名, 良い向きの注記)
    "asw_batch": ("ASW (site)", "→ 0: sites mixed"),
    "ilisi":     ("iLISI (site)", "↑: sites mixed"),
    "asw_bio":   ("ASW (subtype)", "↑: subtypes separated"),
    "clisi":     ("cLISI (subtype)", "↓ 1: subtypes separated"),
    "asw_class": ("ASW (SFT vs meningioma)", "↑: classes separated"),
}


def _reduce(X: np.ndarray, n: int | None) -> np.ndarray:
    if n and X.shape[1] > n:
        return PCA(n_components=n, random_state=42).fit_transform(X)
    return X


def _lisi(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """各点の k 近傍（自分を除く）におけるラベルの逆シンプソン指数。"""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]
    codes, inv = np.unique(labels, return_inverse=True)
    neigh = inv[idx]
    p = np.stack([(neigh == c).mean(axis=1) for c in range(len(codes))], axis=1)
    return 1.0 / (p ** 2).sum(axis=1)


def compute_metrics(merged: pd.DataFrame, cfg: dict) -> dict[str, float]:
    h_cfg = cfg.get("harmonization", {})
    pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
    k = h_cfg.get("lisi_k", 30)
    X = _reduce(np.stack(merged["embedding"].values).astype(np.float64), pre_n)
    src = merged["source"].values
    sub = merged["subtype"].values
    cls = np.where(merged["subtype"].str.lower().str.contains("sft"), "SFT", "Meningioma")
    il, cl = _lisi(X, src, k), _lisi(X, sub, k)
    return {
        "n": len(merged),
        "asw_batch": float(silhouette_score(X, src)),
        "asw_bio":   float(silhouette_score(X, sub)),
        "asw_class": float(silhouette_score(X, cls)),
        "ilisi": float(il.mean()),
        "clisi": float(cl.mean()),
        "ilisi_median": float(np.median(il)),
        "clisi_median": float(np.median(cl)),
    }


def _site_values(X: np.ndarray, src: np.ndarray, k: int) -> tuple[float, float]:
    return float(silhouette_score(X, src)), float(_lisi(X, src, k).mean())


def site_null(merged: pd.DataFrame, cfg: dict, rng: np.random.Generator) -> list[dict]:
    """1 状態について施設ラベルを並べ替えた ASW-batch / iLISI の帰無分布を作り、観測と比べる。"""
    h_cfg = cfg.get("harmonization", {})
    n_cfg = h_cfg.get("site_null", {})
    pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
    k = h_cfg.get("lisi_k", 30)
    n_perm = n_cfg.get("n_perm", 1000)
    X = _reduce(np.stack(merged["embedding"].values).astype(np.float64), pre_n)
    src = merged["source"].values
    sub = merged["subtype"].values
    groups = [np.where(sub == s)[0] for s in np.unique(sub)]
    obs = dict(zip(("asw_batch", "ilisi"), _site_values(X, src, k)))
    rows = []
    for null in n_cfg.get("nulls", ["within_subtype", "free"]):
        draws = []
        for _ in range(n_perm):
            perm = src.copy()
            if null == "free":
                rng.shuffle(perm)
            else:
                for g in groups:
                    perm[g] = rng.permutation(perm[g])
            draws.append(_site_values(X, perm, k))
        draws = np.array(draws)
        for j, metric in enumerate(("asw_batch", "ilisi")):
            d = draws[:, j]
            # 施設が分かれている向き: ASW は大きいほど、iLISI は小さいほど
            extreme = (d >= obs[metric]) if metric == "asw_batch" else (d <= obs[metric])
            rows.append({"null": null, "metric": metric, "obs": obs[metric], "null_mean": float(d.mean()),
                         "null_q025": float(np.quantile(d, 0.025)), "null_q975": float(np.quantile(d, 0.975)),
                         "p": float((1 + extreme.sum()) / (n_perm + 1))})
    return rows


def residual_fraction(null_table: pd.DataFrame) -> pd.DataFrame:
    """補正なしを 1、帰無平均を 0 とした「残った施設差の割合」。"""
    t = null_table.assign(excess=null_table["obs"] - null_table["null_mean"])
    base = t[t["state"] == "original"].set_index(["null", "metric"])["excess"]
    t["residual"] = [r.excess / base[(r.null, r.metric)] for r in t.itertuples()]
    return t[["state", "null", "metric", "residual"]]


def compute_null_all(cfg: dict) -> pd.DataFrame:
    n_cfg = cfg.get("harmonization", {}).get("site_null", {})
    rng = np.random.default_rng(n_cfg.get("seed", 42))
    variants: list[str] = order_conditions(cfg.get("harmonization", {}).get("variants", ["original", "gan", "centroid"]))
    rows = []
    for variant in variants:
        merged = load_data(cfg, cfg.get("variants", {}).get(variant, variant))
        if merged.empty:
            continue
        for r in site_null(merged, cfg, rng):
            rows.append({"state": variant, **r})
            print(f"  [{variant}/{r['null']}] {r['metric']}: obs={r['obs']:.4f} "
                  f"null={r['null_mean']:.4f} [{r['null_q025']:.4f}, {r['null_q975']:.4f}] p={r['p']:.4f}")
    return pd.DataFrame(rows)


def compute_all(cfg: dict) -> pd.DataFrame:
    variants: list[str] = order_conditions(cfg.get("harmonization", {}).get("variants", ["original", "gan", "centroid"]))
    rows = []
    for variant in variants:
        dir_key = cfg.get("variants", {}).get(variant, variant)
        merged = load_data(cfg, dir_key)
        if merged.empty:
            print(f"  SKIP [{variant}]: no data")
            continue
        counts = merged["source"].value_counts().to_dict()
        m = compute_metrics(merged, cfg)
        print(f"  [{variant}] {counts}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k != "n"))
        rows.append({"state": variant, **{f"n_{s}": c for s, c in counts.items()}, **m})
    return pd.DataFrame(rows)


def draw_metric(ax: plt.Axes, table: pd.DataFrame, key: str, show_note: bool = True,
                null: dict | None = None, null_style: str = "band", states: list[str] | None = None) -> None:
    """1 指標の棒グラフ（状態ごと）を ax に描く。

    states: 描く状態（既定は本番の CONDITION_ORDER にあるものだけ。combat 等の検討用の状態は明示したときだけ描く）。

    null: {"mean", "q025", "q975"}（施設ラベルの並べ替えによる偶然のレベル）。null_style は
    "band"（95% 範囲の灰色の横帯）/ "line"（平均の破線）。
    """
    if states is None:
        states = [s for s in CONDITION_ORDER if s in set(table["state"])]
    table = table.set_index("state").loc[states].reset_index()
    vals = table[key].values
    x = np.arange(len(states))
    ax.bar(x, vals, color=[STATE_COLORS.get(s, "#888888") for s in states], width=0.62)
    span = max(abs(vals).max(), 1e-9)
    for xi, v in zip(x, vals):
        ax.text(xi, v + span * 0.03 * (1 if v >= 0 else -1), f"{v:.3f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([STATE_LABELS.get(s, s) for s in states], fontsize=8)
    name, note = METRIC_INFO[key]
    ax.set_title(f"{name}\n{note}" if show_note else name, fontsize=9)
    if key.startswith("asw"):
        ax.axhline(0, color="#555555", lw=0.6)
    lo = min(0.0, vals.min() * 1.25)
    hi = vals.max() * 1.22 if vals.max() > 0 else 0.01
    if key in ("ilisi", "clisi"):
        lo = 1.0  # LISI の下限は 1
        hi = max(hi, 1.0 + (vals.max() - 1.0) * 1.25)
    if null is not None:
        if null_style == "band":
            ax.axhspan(null["q025"], null["q975"], color="#BBBBBB", alpha=0.45, lw=0, zorder=0)
            ax.text(1.01, null["mean"], "chance\n(95%)", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color="#666666", ha="left", va="center", clip_on=False)
        else:
            ax.axhline(null["mean"], color="#555555", lw=0.9, ls="--", zorder=0)
            ax.text(1.01, null["mean"], "chance", transform=ax.get_yaxis_transform(),
                    fontsize=6.5, color="#555555", ha="left", va="center", clip_on=False)
        if key in ("ilisi", "clisi"):
            hi = max(hi, null["q975"] + (null["q975"] - 1.0) * 0.08)
    ax.set_ylim(lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def plot(table: pd.DataFrame, out_path: Path) -> None:
    keys = list(METRIC_INFO)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        draw_metric(ax, table, key, states=order_conditions(table["state"].tolist()))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "harmonization"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[harmonization]")
    table = compute_all(cfg)
    path = out_dir / "metrics.csv"
    table.to_csv(path, index=False)
    print(f"  saved: {path}")
    print(table.round(4).to_string(index=False))
    plot(table, out_dir / "metrics.png")

    print("[harmonization] site-label permutation null")
    nt = compute_null_all(cfg)
    nt.to_csv(out_dir / "site_null.csv", index=False)
    rf = residual_fraction(nt)
    rf.to_csv(out_dir / "residual_fraction.csv", index=False)
    print(f"  saved: {out_dir / 'site_null.csv'} / residual_fraction.csv")
    print(rf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
