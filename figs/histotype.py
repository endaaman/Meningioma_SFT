"""組織型（主要 5 クラス）の施設間分類の混同行列（Supplementary）。

対象は config の lp_histotype_sets.main（Meningothelial, Fibrous, Psammomatous, Angiomatous, Microcystic）。
Transitional（meningothelial と fibrous の混合型で、施設間の診断基準の不一致を距離解析で確認）と
Secretory（patho2 が 6 例で、patho2 で学習する方向が成り立たない）は除外。組織型の例数は Table 1。

入力: output_dir/lp_histotype_main/trained_by_*/*/predictions.csv（lp.py --task histotype --set main）
出力: fig/{fig S n}_histotype.png / .pdf（番号は figs/__init__.py）

Usage:
    uv run python -m figs.histotype
"""
from __future__ import annotations

from figs import label
from figs.common import config, save
from figs.histotype_preview import cm_figure
from histotype_labels import apply_set

NAME = "histotype"
CLASS_SET = "main"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = apply_set(config(), CLASS_SET)
    save(cm_figure(cfg), cfg, NAME)


if __name__ == "__main__":
    main()
