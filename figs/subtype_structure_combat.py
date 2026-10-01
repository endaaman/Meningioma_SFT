"""ComBat 補正後のサブタイプ構造（Supplementary）: Fig 6 と同じ樹形図（euc_mean、Ward、点線の四角 = 施設間ペア）。

ComBat は Fig 3 / Fig 5 / Table 2 で評価し、組織型のまとまりを崩すことを示したうえで Fig 6（構造の解析）からは外す。
隠していないことを示すため、樹形図はここに置く（ken 決定 2026-10-02）。

入力: output_dir/dendrogram/combat/euc_mean_raw_groups.csv / euc_mean_raw.npz（dendrogram.py）
出力: fig/{fig Sn}_subtype_structure_combat.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure_combat
"""
from __future__ import annotations

from figs import label, stem
from figs.common import config, fig_dir
from figs.subtype_structure import save_tree_figure, tree_figure

NAME = "subtype_structure_combat"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    fig_dir(cfg).mkdir(parents=True, exist_ok=True)
    save_tree_figure(tree_figure(["combat"], cfg), fig_dir(cfg) / stem(NAME))


if __name__ == "__main__":
    main()
