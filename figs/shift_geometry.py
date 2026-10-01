"""施設差の幾何: 群ごとのシフトの向き（a）・大きさ（b）と、シフトと SFT − 髄膜腫 方向（c）。

入力: output_dir/shift_direction/{shift_direction,bio_direction}_{variant}.csv  （shift_direction.py）
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

    fig = plt.figure(figsize=(7.2, 0.24 * len(table) + 1.3))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.3, 1.9], wspace=0.12)
    left = outer[0, 0].subgridspec(1, 2, width_ratios=[2.2, 1.2], wspace=0.08)
    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left[0, 1], sharey=ax_a)
    shift_direction.draw_cos(ax_a, table, cfg, fontsize=FONT)
    shift_direction.draw_norm(ax_b, table, cfg, fontsize=FONT)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_c = fig.add_subplot(outer[0, 1])
    shift_direction.draw_bio(ax_c, bio, cfg, fontsize=FONT)
    panel(ax_a, "a", x=-0.02, y=1.01); panel(ax_b, "b", x=0.02, y=1.01)
    # c は等倍軸で縦位置がずれるので、a/b の上端に揃えて図座標で置く
    fig.canvas.draw()
    panel_fig(fig, ax_c.get_position().x0, ax_a.get_position().y1 + 0.01, "c")
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
