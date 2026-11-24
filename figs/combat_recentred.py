"""v4 Fig S（Fig 5 の対照）借りた ComBat の中心を当て先で取り直す（TITAN）。観察として示す（良し悪しの語は入れない）。

借りた ComBat（推定元の腫瘍で推定した位置と倍率をそのまま当てる）と、同じ倍率のまま中心だけを当て先の腫瘍の施設平均で
取り直したもの（transfer_correction.py の combat_recentred）と、当て先の腫瘍自身で推定した ComBat を並べる。
当て先 ← 推定元 = グリオーマ ← 髄膜腫、髄膜腫 ← グリオーマ、リンパ腫 ← グリオーマ、転移 ← グリオーマ。

a 2 施設の中心の距離（群の中央値、施設内の広がり = 1）
b 近傍の混ざり（両施設 5 枚以上の群の中央値、1 = 施設を見分けられない）

入力: output_dir/transfer/correction_titan.csv・correction_titan_groups.csv
出力: fig/{fig Sn}_combat_recentred.{png,pdf}、同 _values.csv

Usage:
    uv run python -m figs.combat_recentred
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figs import label
from figs.borrowed_scaling import MIN_N, NAME_T, load
from figs.common import CONDITION_COLORS, FONT, config, fig_path, panel, save

NAME = "combat_recentred"
PAIRS = [("glioma", "meningioma"), ("meningioma", "glioma"), ("lymphoma", "glioma"), ("metastasis", "glioma")]
BARS = [("combat", None, "ComBat, borrowed", CONDITION_COLORS["combat"]),
        ("combat_recentred", None, "ComBat, borrowed, centre re-estimated on the target", "#9e9ac8"),
        ("combat", "own", "ComBat, estimated on the target tumour", "#1baf7a")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    val, knn5 = load(cfg)
    gap = val.xs("group_gap_sd", level="metric")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.3})
    rows = []
    w = 0.26
    for ax, (metric, src_tab, ylab, letter) in zip(axes, (
            ("group_gap_sd", gap, "Site-centre distance\n(group median; within-site spread = 1)", "a"),
            ("knn_mix", knn5, f"Neighbour mixing (groups with ≥ {MIN_N}\nslides per site; 1 = sites indistinguishable)", "b"))):
        for j, (method, fixed_src, lab, col) in enumerate(BARS):
            ys = [float(src_tab.get((t, fixed_src or s, method), np.nan)) for t, s in PAIRS]
            xs = np.arange(len(PAIRS)) + (j - 1) * w
            ax.bar(xs, ys, width=w * 0.92, color=col, label=lab)
            for xv, yv in zip(xs, ys):   # 群が 1 つの当て先は中心を取り直すと距離が 0 になり棒が見えないので、値を書く
                if yv < 0.03:
                    ax.text(xv, yv + 0.01, f"{yv:.2f}", ha="center", va="bottom", fontsize=FONT - 3.5, color=col)
            rows += [{"metric": metric, "target": t, "source": s, "bar": lab, "value": y} for (t, s), y in zip(PAIRS, ys)]
        if metric == "knn_mix":
            ax.axhline(1, color="0.5", lw=0.7, ls=":")
        ax.set_xticks(range(len(PAIRS)))
        ax.set_xticklabels([f"{NAME_T[t]}\n← {NAME_T[s].lower()}" for t, s in PAIRS], fontsize=FONT - 2)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(ylab, fontsize=FONT - 1)
        ax.tick_params(axis="y", labelsize=FONT - 2)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, letter, x=-0.2, y=1.02)
    axes[0].legend(fontsize=FONT - 2.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3)
    save(fig, cfg, NAME)
    v = pd.DataFrame(rows)
    v.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    print(v.pivot_table(index=["metric", "target", "source"], columns="bar", values="value", sort=False).round(3).to_string())


if __name__ == "__main__":
    main()
