"""SFT からの距離（Supplementary）: 施設内の SFT と各髄膜腫サブタイプの平均ベクトル間 euc_mean を、
Fig 6 の centroid 樹形図と同じ最大値で割った値（施設ごとの点 ＋ 施設平均の棒）。施設内の距離なので補正の前後で不変。

入力:
    output_dir/sft_distance/sft_distance_raw.csv                               （sft_distance_bar.py）
    output_dir/dendrogram/centroid/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py、正規化の分母）
出力: fig/{fig Sn}_sft_distance.{png,pdf}, 同 .csv（番号は figs/__init__.py）

Usage:
    uv run python -m figs.sft_distance
"""
from __future__ import annotations

import matplotlib.pyplot as plt

import sft_distance_bar
from figs import label
from figs.common import FONT, config, fig_path, save
from figs.subtype_structure import sft_table
from utils.display import shorten

NAME = "sft_distance"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    tab = sft_table(cfg, "centroid")
    fig, ax = plt.subplots(figsize=(3.2, 3.6))
    sft_distance_bar.draw_raw(ax, tab, cfg, fontsize=FONT)
    ax.set_xlabel("Normalized distance to SFT\n(euc_mean, within site)", fontsize=FONT)
    ax.set_xlim(0, None)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    names = {"ebrains": "EBRAINS", "patho2": "patho2"}
    ax.legend(handles, [names.get(t, t) for t in labels], loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2, frameon=False, fontsize=FONT)
    tab.to_csv(fig_path(cfg, NAME, ".csv"), index=False)
    save(fig, cfg, NAME)
    order = tab.groupby("subtype")["distance"].mean().sort_values()
    print("  SFT に近い順: " + ", ".join(f"{shorten(k, cfg)} {v:.3f}" for k, v in order.items()))


if __name__ == "__main__":
    main()
