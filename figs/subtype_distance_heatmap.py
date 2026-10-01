"""サブタイプ間距離のヒートマップ（補正前 / centroid 補正後、Supplementary）。

**本文 Fig（subtype_structure）とは距離の示し方が違う**: こちらは平均ベクトル間の **生の** ユークリッド距離
（正規化しない）を、Ward 法の葉順に並べた clustered heatmap。樹形図の形（併合の順）は本文と同じ
（正規化は最大値で割るだけなので Ward の木は変わらない）が、色の値は生の距離。

入力: output_dir/dendrogram/{original,centroid}/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py）
出力: fig/{fig Sn}_subtype_distance_heatmap.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_distance_heatmap
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import dendrogram
from figs import label
from figs.common import FONT, VARIANT_DISPLAY, config, out_root, panel_fig, save

NAME = "subtype_distance_heatmap"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    fig = plt.figure(figsize=(10.4, 5.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.55)
    for i, (v, letter) in enumerate((("original", "a"), ("centroid", "b"))):
        groups, D, Z = dendrogram.load_raw(out_root(cfg) / "dendrogram" / cfg["variants"].get(v, v))
        before = set(fig.axes)
        ax = dendrogram.draw_clustered(fig, gs[0, i], groups, D, Z, cfg, fontsize=FONT)
        fig.canvas.draw()
        pa = ax.get_position()
        top = max(a.get_position().y1 for a in fig.axes if a not in before)
        panel_fig(fig, pa.x0 - 0.09, top + 0.01, letter)
        fig.text(pa.x0 + pa.width / 2, top + 0.02,
                 f"{VARIANT_DISPLAY.get(v, v)}  (raw Euclidean distance)", ha="center", fontsize=FONT + 1)
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
