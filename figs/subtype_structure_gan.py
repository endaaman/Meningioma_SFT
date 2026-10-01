"""サブタイプ構造（GAN 補正後）: subtype_structure の樹形図と同じ形式（euc_mean・元の図と同じ正規化、Ward）（Supplementary）。

入力: output_dir/dendrogram/gan/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py）
出力: fig/{fig Sn}_subtype_structure_gan.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure_gan
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from figs import label
from figs.common import FONT, config, save
from figs.subtype_structure import draw_trees, legend_handles, load

NAME = "subtype_structure_gan"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    fig = plt.figure(figsize=(5.6, 2.5))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.28], hspace=0.35)
    draw_trees(fig, [gs[0, 0]], ["gan"], cfg, "a")
    ax_l = fig.add_subplot(gs[1, 0]); ax_l.axis("off")
    groups, _ = load(cfg, "gan")
    ax_l.legend(handles=legend_handles(groups, cfg), loc="center", ncol=8, frameon=False,
                fontsize=FONT - 1, handlelength=1.2, handletextpad=0.4, columnspacing=0.8)
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
