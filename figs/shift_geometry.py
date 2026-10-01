"""施設差の幾何: 群ごとのシフトの向き（a）・大きさ（b）、シフトと SFT − 髄膜腫 方向（c）、
組織型の平均どうしの距離の施設間比較（d: 形を保った一様な縮み）。

入力: output_dir/shift_direction/{shift_direction,bio_direction,subtype_distance_pairs,subtype_distance_scaling}_{variant}.csv
      （shift_direction.py）
出力: fig/{fig n}_shift_geometry.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.shift_geometry
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

import shift_direction
from figs import label
from figs.common import FONT, config, out_root, panel, panel_fig, save


NAME = "shift_geometry"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    sd_dir = out_root(cfg) / "shift_direction"
    variant = cfg.get("shift_direction", {}).get("variant", "original")
    table = pd.read_csv(sd_dir / f"shift_direction_{variant}.csv")
    bio = pd.read_csv(sd_dir / f"bio_direction_{variant}.csv")
    pairs = pd.read_csv(sd_dir / f"subtype_distance_pairs_{variant}.csv")
    scaling = pd.read_csv(sd_dir / f"subtype_distance_scaling_{variant}.csv")

    fig = plt.figure(figsize=(7.2, max(0.24 * len(table) + 1.3, 5.6)))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.3, 1.9], wspace=0.18)
    left = outer[0, 0].subgridspec(1, 2, width_ratios=[2.2, 1.2], wspace=0.08)
    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left[0, 1], sharey=ax_a)
    shift_direction.draw_cos(ax_a, table, cfg, fontsize=FONT)
    shift_direction.draw_norm(ax_b, table, cfg, fontsize=FONT)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.35)
    ax_c = fig.add_subplot(right[0, 0])
    shift_direction.draw_bio(ax_c, bio, cfg, fontsize=FONT)
    ax_d = fig.add_subplot(right[1, 0])
    shift_direction.draw_scaling(ax_d, pairs, scaling, fontsize=FONT)
    panel(ax_a, "a", x=-0.02, y=1.01); panel(ax_b, "b", x=0.02, y=1.01)
    # c・d は等倍軸で位置がずれるので、軸の実位置から図座標で置く
    fig.canvas.draw()
    panel_fig(fig, ax_c.get_position().x0, ax_a.get_position().y1 + 0.01, "c")
    pos_d = ax_d.get_position()
    panel_fig(fig, pos_d.x0 - 0.06, pos_d.y1 + 0.02, "d")
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
