"""組織型（主要 5 クラス）の施設間分類の混同行列（Supplementary）。

対象は config の lp_histotype_sets.main（Meningothelial, Fibrous, Psammomatous, Angiomatous, Microcystic）。
Transitional（meningothelial と fibrous の混合型で、施設間の診断基準の不一致を距離解析で確認）と
Secretory（patho2 が 6 例で、patho2 で学習する方向が成り立たない）は除外。組織型の例数は Table 1。

評価は訓練不要の最近傍重心法（学習施設の組織型平均のうち最も近いものに割り当てる）。
条件は本文の 4 条件に相似変換（一様なスケール＋平行移動、倍率はラベル不要）を足した 5 条件。

入力: output_dir/nc/histotype/trained_by_*/*/predictions.csv（nc.py）
出力: fig/{fig S n}_histotype.png / .pdf（番号は figs/__init__.py）

Usage:
    uv run python -m figs.histotype
"""
from __future__ import annotations

from figs import label
from figs.common import CONDITIONS_SUPP, config, save
from figs.histotype_preview import cm_figure
from histotype_labels import apply_set

NAME = "histotype"
CLASS_SET = "main"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = apply_set(config(), CLASS_SET)
    cfg["lp_histotype"] = {**cfg["lp_histotype"], "output_subdir": "nc/histotype", "variants": CONDITIONS_SUPP}
    save(cm_figure(cfg), cfg, NAME)


if __name__ == "__main__":
    main()
