"""figs — 論文の図表を out/ の解析結果から組み立てる（重い計算はしない）。

スクリプト名は内容名（番号を持たない）。図表番号は下の MAIN / TABLES / SUPP / SUPP_TABLES の順序だけで決まり、
並べ替えはこの定義を変えるだけで済む。出力ファイル名は原稿で参照しやすいよう番号付き
（例 fig/fig3_umap_harmonization.png）。

    uv run python -m figs.umap_harmonization
    uv run python -m figs.all        # fig/・tables/ の自分の出力を掃除してから全部作る

入力は解析スクリプトが out/ に書いた CSV / npz、出力は output_dir/paper/{paper.version}/{fig,tables}/。
"""

# 本文の図（順 = Fig 番号）。design は手描き（スクリプトが無いものは all で飛ばす）
MAIN = [
    "design",               # Fig 1 研究デザイン（手描き）
    "tissue",               # Fig 2 組織パッチ（patho2 / patho2→GAN / EBRAINS × サブタイプ）と色の定量
    "umap_harmonization",   # Fig 3 UMAP と指標（補正なし / GAN / ComBat / centroid の 4 条件）
    "shift_geometry",       # Fig 4 施設差の幾何
    "subtype_structure",    # Fig 5 サブタイプ構造（樹形図 補正なし / GAN / ComBat / centroid の 4 段、euc_mean）
    "classification",       # Fig 6 施設間の分類性能（4 条件）
    "pathology_checks",     # Fig 7 病理医の判断との対応: held-out 症例（旧 Fig S5、umap_held_out.py の描画）＋ STAT6 を染めた patho2 症例の UMAP 上の位置と距離差（stat6_case.py）
]
# 表（順 = Table 番号）
TABLES = [
    "cohort",               # Table 1 データセット
]
# Supplementary（順 = Fig S 番号）
SUPP = [
    "subtype_distance_heatmap",       # サブタイプ間距離のヒートマップ 4 枚（補正なし / GAN / ComBat / centroid、euc_mean、固定順）
    "sft_distance",                   # SFT からの距離（euc_mean、施設内。旧 Fig 6c）
    "site_axis_stain",                # 施設軸と染色の関係: 施設シフト方向への射影 s vs H / E（十分位の bin 平均）、画像レベル補正後に残る施設差の u 方向 / 直交分解、平均色で説明される割合（stain_response.py）
    "histotype",                      # 組織型（主要 5 クラス）の施設間分類の混同行列（2 方向 × 5 条件、Macenko を含む）
    "shift_sample_size",              # 平行移動のシフト推定に要る例数（無作為 / 単一組織型 × n。shift_estimation.py）
]
# Supplementary の表（順 = Table S 番号）
SUPP_TABLES = [
    "meanpool_comparison",            # 補正法の比較（平均プーリング、5 条件。meanpool.py）
    "knn_performance",                # 施設間の分類性能（スライド単位の k 近傍法、k = 1 / 5 / 15、5 条件。Table S1 の平均に依らない対照。knn.py）
    "method_comparison",              # 補正法の比較（全指標、Macenko・相似変換を含む 6 条件。相似変換はここだけ）
    "performance",                    # 施設間の分類性能（SFT vs 髄膜腫、5 条件、旧 Table 2。主な数値は Results 本文と Fig 5b）
    "histotype_performance",          # 組織型分類の balanced accuracy・macro F1（95% CI、5 条件）
    "stat6_groups",                   # STAT6 の状態の群 × 条件: SFT 割り当て n/N と距離差の中央値 [IQR]（EBRAINS 重心。stat6_case.py）
]
# 登録を外したもの（スクリプトは残す、figs.all では作らないが、番号付きの古い出力は all の掃除で消す）
UNREGISTERED = [
    "meanpool_geometry",              # 施設差の幾何: 平均プーリング vs TITAN。数値は Results 本文に書く（ken 決定 2026-10-20）
    "umap_held_out",                  # 旧 Fig S5。Fig 7a–b として pathology_checks に取り込んだ（ken 決定 2026-10-24）。描画コードはそのまま使う
]


def stem(name: str) -> str:
    """出力ファイル名の幹（例: fig3_umap_harmonization / table1_cohort / figS1_subtype_distance_heatmap / tableS1_performance）。"""
    for prefix, names in (("fig", MAIN), ("table", TABLES), ("figS", SUPP), ("tableS", SUPP_TABLES)):
        if name in names:
            return f"{prefix}{names.index(name) + 1}_{name}"
    raise KeyError(f"figs: '{name}' is not registered in MAIN / TABLES / SUPP / SUPP_TABLES")


def label(name: str) -> str:
    """原稿での呼び名（例: Fig 3 / Table 1 / Fig S1）。"""
    for prefix, names in (("Fig ", MAIN), ("Table ", TABLES), ("Fig S", SUPP), ("Table S", SUPP_TABLES)):
        if name in names:
            return f"{prefix}{names.index(name) + 1}"
    raise KeyError(name)
