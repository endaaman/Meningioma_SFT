"""施設間の分類性能: 混同行列（a）と balanced accuracy・SFT 感度の点推定＋95% CI（b）。

    a: 混同行列（2 方向 × 補正なし / GAN / centroid）。セルは件数、色は行（真のクラス）ごとの割合。
    b: balanced accuracy と SFT 感度の点推定 + 95% CI（患者単位 bootstrap）。

入力: output_dir/lp/trained_by_*/{variant}/predictions.csv（lp.py）, output_dir/lp/bootstrap.csv（lp_bootstrap.py）
出力: fig/{fig n}_classification.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.classification
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from figs import label
from figs.common import SITE_DISPLAY, VARIANT_COLOR, VARIANT_DISPLAY, config, order_conditions, out_root, save

NAME = "classification"
CLASS_NAMES = ["Meningioma", "SFT"]


def _confusion(df: pd.DataFrame) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(df["true"], df["pred"]):
        cm[t, p] += 1
    return cm


def _draw_cm(ax: plt.Axes, cm: np.ndarray, title: str, show_ylabel: bool, show_xlabel: bool) -> None:
    """show_xlabel は軸タイトル "Predicted" のみ（目盛りラベルは常に出す）。"""
    frac = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({frac[i, j]:.0%})", ha="center", va="center", fontsize=9,
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES, fontsize=8)
    ax.set_yticks([0, 1], CLASS_NAMES if show_ylabel else ["", ""], fontsize=8)
    if show_xlabel:
        ax.set_xlabel("Predicted", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    lp_root = out_root(cfg) / "lp"
    variants = order_conditions(cfg["lp"].get("variants", ["original"]))
    sources = list(cfg["embedding"].keys())
    directions = [(s, next(t for t in sources if t != s)) for s in sources]
    bs = pd.read_csv(lp_root / "bootstrap.csv")

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1.15], hspace=0.6)

    # ── a: 混同行列 ──────────────────────────────────────────────────────────
    for r, (train_src, test_src) in enumerate(directions):
        sub = gs[r].subgridspec(1, len(variants), wspace=0.15)
        for c, variant in enumerate(variants):
            ax = fig.add_subplot(sub[c])
            df = pd.read_csv(lp_root / f"trained_by_{train_src}" / variant / "predictions.csv")
            _draw_cm(ax, _confusion(df), VARIANT_DISPLAY.get(variant, variant),
                     show_ylabel=(c == 0), show_xlabel=(r == len(directions) - 1))
            if c == 0:
                ax.annotate(f"{SITE_DISPLAY[train_src]} → {SITE_DISPLAY[test_src]}  (n={len(df)})",
                            xy=(0, 1.32), xycoords="axes fraction", fontsize=9, fontweight="bold")
                if r == 0:
                    ax.annotate("a", xy=(-0.55, 1.32), xycoords="axes fraction", fontsize=12, fontweight="bold")

    # ── b: BA と感度の点推定 + 95% CI ─────────────────────────────────────────
    sub = gs[2].subgridspec(1, 2, wspace=0.08)
    metrics = [("balanced_accuracy", "Balanced accuracy"), ("sensitivity", "Sensitivity (SFT)")]
    ylabels, ypos = [], []
    for k, (metric, xlabel) in enumerate(metrics):
        ax = fig.add_subplot(sub[k])
        y = 0
        for train_src, test_src in directions:
            for variant in variants:
                row = bs[(bs.train_source == train_src) & (bs.variant == variant) & (bs.metric == metric)]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.errorbar(row.value, -y, xerr=[[row.value - row.ci_low], [row.ci_high - row.value]],
                            fmt="o", color=VARIANT_COLOR[variant], ecolor=VARIANT_COLOR[variant],
                            markersize=5, capsize=2, lw=1.4, markeredgecolor="black", markeredgewidth=0.4)
                if k == 0:
                    ylabels.append(f"{VARIANT_DISPLAY[variant]}")
                    ypos.append(-y)
                y += 1
            y += 0.6  # 方向の間の余白
        ax.set_xlim(0.45, 1.02)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(ypos, ylabels if k == 0 else [""] * len(ypos))
        if k == 0:
            n_v = len(variants)
            for d, (train_src, test_src) in enumerate(directions):
                mid = -(d * (n_v + 0.6) + (n_v - 1) / 2)
                ax.annotate(f"{SITE_DISPLAY[train_src]} →\n{SITE_DISPLAY[test_src]}", xy=(-0.62, mid),
                            xycoords=("axes fraction", "data"), fontsize=8, fontweight="bold", va="center")
            ax.annotate("b", xy=(-0.62, 1.05), xycoords="axes fraction", fontsize=12, fontweight="bold")

    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
