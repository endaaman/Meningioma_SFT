"""figs — 論文の図表を out/ の解析結果から組み立てる（重い計算はしない）。

スクリプト名は内容名（番号を持たない）。図表番号は下の MAIN / TABLES / SUPP の順序だけで決まり、
並べ替えはこの定義を変えるだけで済む。出力ファイル名は原稿で参照しやすいよう番号付き
（例 fig/fig3_umap_harmonization.png）。

    uv run python -m figs.umap_harmonization
    uv run python -m figs.all        # fig/・tables/ の自分の出力を掃除してから全部作る

入力は解析スクリプトが out/ に書いた CSV / npz、出力は output_dir/paper/{paper.version}/{fig,tables}/。
"""

# 本文の図（順 = Fig 番号）。design は手描き、tissue は未作成（スクリプトが無いものは all で飛ばす）
MAIN = [
    "design",               # Fig 1 研究デザイン（手描き）
    "tissue",               # Fig 2 組織画像・GAN 変換・色
    "umap_harmonization",   # Fig 3 UMAP（補正なし / centroid / GAN）と施設差指標
    "shift_geometry",       # Fig 4 施設差の幾何
    "classification",       # Fig 5 施設間の分類性能
    "subtype_structure",    # Fig 6 サブタイプ構造（clustered heatmap）と SFT からの距離
]
# 表（順 = Table 番号）
TABLES = [
    "cohort",               # Table 1 データセット
    "performance",          # Table 2 分類性能
]
# Supplementary（順 = Fig S 番号）
SUPP = [
    "subtype_structure_uncorrected",  # 補正前の clustered heatmap
]


def stem(name: str) -> str:
    """出力ファイル名の幹（例: fig3_umap_harmonization / table1_cohort / figS1_subtype_structure_uncorrected）。"""
    for prefix, names in (("fig", MAIN), ("table", TABLES), ("figS", SUPP)):
        if name in names:
            return f"{prefix}{names.index(name) + 1}_{name}"
    raise KeyError(f"figs: '{name}' is not registered in MAIN / TABLES / SUPP")


def label(name: str) -> str:
    """原稿での呼び名（例: Fig 3 / Table 1 / Fig S1）。"""
    for prefix, names in (("Fig ", MAIN), ("Table ", TABLES), ("Fig S", SUPP)):
        if name in names:
            return f"{prefix}{names.index(name) + 1}"
    raise KeyError(name)
