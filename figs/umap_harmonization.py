"""施設差と補正: UMAP（補正なし / centroid / GAN）と施設差・クラス分離の指標。

入力:
    output_dir/umap/{original,centroid,gan}/coords.csv  （umap_plot.py）
    output_dir/harmonization/metrics.csv               （harmonization.py）
出力: fig/{fig n}_umap_harmonization.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.umap_harmonization
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import harmonization
from figs import label
from figs.common import FONT, config, out_root, panel, save, src_markers
from utils.display import ordered_subtypes, shorten, subtype_color_map

NAME = "umap_harmonization"
VARIANTS = ["original", "centroid", "gan"]
TITLES = {"original": "Uncorrected", "centroid": "Centroid-corrected", "gan": "GAN-corrected"}
METRIC_KEYS = ["asw_batch", "ilisi", "asw_class"]


def _draw_umap(ax: plt.Axes, coords: pd.DataFrame, sub_map: dict, src_mkr: dict) -> None:
    for src, mkr in src_mkr.items():
        for sub, color in sub_map.items():
            m = (coords["source"] == src) & (coords["subtype"] == sub)
            if m.any():
                ax.scatter(coords.loc[m, "umap1"], coords.loc[m, "umap2"], c=[color], marker=mkr,
                           s=6, alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    root = out_root(cfg)
    coords = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
              for v in VARIANTS}
    metrics = pd.read_csv(root / "harmonization" / "metrics.csv")
    subs = ordered_subtypes(set(coords["original"]["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))

    fig = plt.figure(figsize=(7.2, 5.9))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.15, 0.2, 1], hspace=0.3)
    top = gs[0].subgridspec(1, len(VARIANTS), wspace=0.12)
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(top[0, i])
        _draw_umap(ax, coords[v], sub_map, src_mkr)
        ax.set_title(TITLES[v])
        if i:
            ax.set_ylabel("")
        panel(ax, "abc"[i], x=-0.04)

    # 凡例は UMAP の下に横並び（a–c 共通）
    ax_leg = fig.add_subplot(gs[1]); ax_leg.axis("off")
    handles = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in subs]
    handles += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=s)
                for s, m in src_mkr.items()]
    ax_leg.legend(handles=handles, loc="center", ncol=10, frameon=False, handlelength=1.0,
                  handleheight=0.9, columnspacing=0.9, handletextpad=0.4, borderaxespad=0,
                  fontsize=FONT - 1)

    bottom = gs[2].subgridspec(1, len(METRIC_KEYS), wspace=0.45)
    for i, key in enumerate(METRIC_KEYS):
        ax = fig.add_subplot(bottom[0, i])
        harmonization.draw_metric(ax, metrics, key)
        ax.tick_params(axis="x", labelsize=FONT - 1)
        if i == 0:
            panel(ax, "d", x=-0.22, y=1.22)
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
