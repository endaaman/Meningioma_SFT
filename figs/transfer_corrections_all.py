"""v4 Supplementary Data（CSV 1 枚、表にしない）: 借りた補正の全数値（TITAN）。

1 行 = 当て先 × 推定元 × 補正。当て先 6 つ（髄膜腫、グリオーマ、リンパ腫、転移、髄膜腫＋SFT、5 つの腫瘍全体）、推定元 = 補正なし・
同じ腫瘍（参考）と、別の腫瘍（単独の腫瘍、複数の腫瘍をまとめたもの、腫瘍ごとの平均）、補正 = 平行移動 / 相似変換 / ComBat /
中心を当て先で取り直した ComBat。列名は英語（下の COLUMNS）。当て先に無い指標は空欄（組の数と距離の保存は群が 2 つ以上の当て先だけ、
分類は当て先ごとの課題だけ）。

近傍の混ざりは両施設 5 枚以上の群の中央値（本文の約束）に揃え、全部の群の中央値も別の列に残す。

入力: output_dir/transfer/correction_titan.csv・correction_titan_groups.csv（transfer_correction.py）
出力: tables/{data Sn}_transfer_corrections_all.csv

Usage:
    uv run python -m figs.transfer_corrections_all
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS, config, out_root, table_path
from utils.display import FAMILY_LABELS

NAME = "transfer_corrections_all"
MIN_N = 5
TARGETS = ["meningioma", "glioma", "lymphoma", "metastasis", "paper", "all"]
SOURCES = ["none", "own", "meningioma", "paper", "glioma", "SFT", "lymphoma", "metastasis", "others", "others_mean"]
METHODS = ["centroid", "affine_free", "combat", "combat_recentred"]
GROUP_LABEL = {**FAMILY_LABELS, "paper": "Meningioma + SFT", "all": "All 5 tumours", "none": "(no correction)", "own": "(same tumour)",
               "others": "Other tumours, pooled", "others_mean": "Other tumours, mean of per-tumour shifts"}
METHOD_LABEL = {**CONDITION_LABELS, "combat_recentred": "ComBat, centre re-estimated on the target"}
# (correction_titan.csv の metric, 出力の列名)
COLUMNS = [("pairs_adjacent", "cross_site_pairs"), ("pairs_n", "n_pairs"),
           ("group_gap_sd", "site_centre_distance_median_all_groups"),
           ("knn_mix", "neighbour_mixing_median_all_groups"),
           ("asw_site", "site_silhouette"), ("ilisi_site", "ilisi_site"),
           ("residual_norm", "site_mean_distance"), ("distortion", "distance_preservation_ratio"),
           ("nc_histotype_ba_E2P", "ba_meningioma_5subtypes_ebrains_to_inhouse"), ("nc_histotype_ba_P2E", "ba_meningioma_5subtypes_inhouse_to_ebrains"),
           ("nc_glioma3_ba_E2P", "ba_glioma_3groups_ebrains_to_inhouse"), ("nc_glioma3_ba_P2E", "ba_glioma_3groups_inhouse_to_ebrains"),
           ("nc_sft_ba_E2P", "ba_sft_vs_meningioma_ebrains_to_inhouse"), ("nc_sft_ba_P2E", "ba_sft_vs_meningioma_inhouse_to_ebrains"),
           ("nc_family5_ba_E2P", "ba_5tumours_ebrains_to_inhouse"), ("nc_family5_ba_P2E", "ba_5tumours_inhouse_to_ebrains"),
           ("scale", "fitted_scale"), ("shift_norm", "fitted_shift_norm")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    t = out_root(cfg) / "transfer"
    c = pd.read_csv(t / "correction_titan.csv")
    g = pd.read_csv(t / "correction_titan_groups.csv")
    wide = c.pivot_table(index=["target", "source", "method"], columns="metric", values="value", aggfunc="first")
    n_slides = c.groupby(["target", "source", "method"])[["n_ebrains", "n_patho2", "n_src_ebrains", "n_src_patho2"]].first()
    k = g[(g["metric"] == "knn_mix") & (g["n_ebrains"] >= MIN_N) & (g["n_patho2"] >= MIN_N)]
    knn5 = k.groupby(["target", "source", "method"])["value"].agg(["median", "size"])
    rows = []
    for tg in TARGETS:
        for src in SOURCES:
            for m in METHODS:
                key = (tg, src, m)
                if key not in wide.index or (src == "none" and m != "centroid"):   # 補正なしは 1 行だけ（法によらず同じ値）
                    continue
                w, n = wide.loc[key], n_slides.loc[key]
                rows.append({"target": tg, "target_label": GROUP_LABEL[tg], "source": src, "source_label": GROUP_LABEL[src],
                             "method": "none" if src == "none" else m, "method_label": "No correction" if src == "none" else METHOD_LABEL[m],
                             "n_ebrains_target": int(n["n_ebrains"]), "n_inhouse_target": int(n["n_patho2"]),
                             "n_ebrains_source": n["n_src_ebrains"], "n_inhouse_source": n["n_src_patho2"],
                             "neighbour_mixing_median_groups_min5": knn5.loc[key, "median"] if key in knn5.index else float("nan"),
                             "n_groups_min5": int(knn5.loc[key, "size"]) if key in knn5.index else 0,
                             **{col: w.get(metric, float("nan")) for metric, col in COLUMNS}})
    d = pd.DataFrame(rows)
    order = ["target", "target_label", "source", "source_label", "method", "method_label", "n_ebrains_target", "n_inhouse_target",
             "n_ebrains_source", "n_inhouse_source", "cross_site_pairs", "n_pairs", "site_centre_distance_median_all_groups",
             "neighbour_mixing_median_groups_min5", "n_groups_min5", "neighbour_mixing_median_all_groups"]
    d = d[order + [col for _, col in COLUMNS if col not in order]]
    path = table_path(cfg, NAME, ".csv")
    d.to_csv(path, index=False, float_format="%.4f")
    borrowed = d[~d["source"].isin(["none", "own"])]
    print(f"  saved: {path}（{len(d)} 行、うち借りる {len(borrowed)} 行 = 当て先 × 推定元 "
          f"{borrowed[['target', 'source']].drop_duplicates().shape[0]} 通り × 法）")


if __name__ == "__main__":
    main()
