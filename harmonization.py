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

出力 (output_dir/harmonization/):
    metrics.csv
    metrics.png

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

from utils.loader import load_config, load_data

STATE_LABELS = {"original": "None", "centroid": "Centroid", "gan": "GAN"}  # 補正の種類
STATE_COLORS = {"original": "#9a9a9a", "centroid": "#4477AA", "gan": "#CCBB44"}
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


def compute_all(cfg: dict) -> pd.DataFrame:
    variants: list[str] = cfg.get("harmonization", {}).get("variants", ["original", "centroid", "gan"])
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


def draw_metric(ax: plt.Axes, table: pd.DataFrame, key: str, show_note: bool = True) -> None:
    """1 指標の棒グラフ（状態ごと）を ax に描く。"""
    states = table["state"].tolist()
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
    ax.set_ylim(lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def plot(table: pd.DataFrame, out_path: Path) -> None:
    keys = list(METRIC_INFO)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 3.2))
    for ax, key in zip(axes, keys):
        draw_metric(ax, table, key)
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


if __name__ == "__main__":
    main()
