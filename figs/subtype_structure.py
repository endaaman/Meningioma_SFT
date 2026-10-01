"""サブタイプ構造: 施設 × サブタイプ平均の生のユークリッド距離の clustered heatmap（a, Ward, centroid 後）と
SFT からの距離（b, 施設内の生値）。

入力:
    output_dir/dendrogram/{variant}/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py）
    output_dir/dendrogram/pairs.csv                                            （dendrogram.py）
    output_dir/sft_distance/sft_distance_raw.csv                               （sft_distance_bar.py）
出力: fig/{fig n}_subtype_structure.{png,pdf}, 同 _pairs.csv, 同 _sft_distance.csv（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure
"""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import dendrogram
import sft_distance_bar
from figs import label
from figs.common import FONT, config, fig_path, out_root, panel_fig, save, src_markers
from utils.display import shorten

NAME = "subtype_structure"


def draw(cfg: dict, variant: str, with_b: bool) -> plt.Figure:
    """a: clustered heatmap（variant の平均ベクトル）、with_b なら b: SFT からの距離も描く。"""
    groups, D, Z = dendrogram.load_raw(out_root(cfg) / "dendrogram" / cfg["variants"].get(variant, variant))
    if with_b:
        fig = plt.figure(figsize=(7.2, 5.4))
        gs = GridSpec(1, 2, figure=fig, width_ratios=[2.3, 1], wspace=0.95)
    else:
        fig = plt.figure(figsize=(5.2, 5.4))
        gs = GridSpec(1, 1, figure=fig)
    ax_a = dendrogram.draw_clustered(fig, gs[0, 0], groups, D, Z, cfg, fontsize=FONT)
    src_mkr = src_markers(cfg, sorted(groups["source"].unique()))
    handles = [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=4, label=s)
               for s, m in src_mkr.items()]
    ax_a.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.2), ncol=2,
                frameon=False, fontsize=FONT - 1, handletextpad=0.2, columnspacing=0.8,
                title="(n) = slides", title_fontsize=FONT - 1, alignment="left")
    if with_b:
        fig.canvas.draw()
        top = ax_a.get_position().y1
        ax_b = fig.add_subplot(gs[0, 1])
        raw = pd.read_csv(out_root(cfg) / "sft_distance" / "sft_distance_raw.csv")
        sft_distance_bar.draw_raw(ax_b, raw, cfg, fontsize=FONT)
        ax_b.get_legend().remove()  # 施設マーカーは a の凡例と共通
        # b の縦範囲をヒートマップに揃える
        pb, pa = ax_b.get_position(), ax_a.get_position()
        ax_b.set_position([pb.x0, pa.y0, pb.width, pa.height])
        panel_fig(fig, pa.x0 - 0.13, fig.axes[0].get_position().y1 + 0.005, "a")
        panel_fig(fig, pb.x0 - 0.12, top + 0.005, "b")
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(draw(cfg, "centroid", with_b=True), cfg, NAME)
    root = out_root(cfg)
    shutil.copy2(root / "dendrogram" / "pairs.csv", fig_path(cfg, NAME, "_pairs.csv"))
    raw = pd.read_csv(root / "sft_distance" / "sft_distance_raw.csv")
    raw.to_csv(fig_path(cfg, NAME, "_sft_distance.csv"), index=False)
    order = raw.groupby("subtype")["distance"].mean().sort_values()
    print("  SFT に近い順: " + ", ".join(f"{shorten(k, cfg)} {v:.2f}" for k, v in order.items()))
    print(pd.read_csv(root / "dendrogram" / "pairs.csv").to_string(index=False))


if __name__ == "__main__":
    main()
