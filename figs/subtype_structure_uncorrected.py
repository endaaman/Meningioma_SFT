"""サブタイプ構造（補正前）: subtype_structure の a と同形式の clustered heatmap（Supplementary）。

入力: output_dir/dendrogram/original/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py）
出力: fig/{fig Sn}_subtype_structure_uncorrected.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure_uncorrected
"""
from __future__ import annotations

from figs import label
from figs.common import config, save
from figs.subtype_structure import draw

NAME = "subtype_structure_uncorrected"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(draw(cfg, "original", with_b=False), cfg, NAME)


if __name__ == "__main__":
    main()
