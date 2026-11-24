"""v4 Supplementary の表: 髄膜腫の補正法の比較（全指標、TITAN、髄膜腫だけ = EBRAINS 519 / 院内 555 枚、両施設にあるサブタイプ 12 組）。

行 = 補正なし / Macenko / GAN / ComBat / 平行移動 / 相似変換（CONDITIONS_SUPP の並び）。どの補正も髄膜腫自身で推定。
列 = 2 施設の平均どうしの距離、施設のシルエット係数、iLISI、隣り合う組（12 組中）、2 施設の中心の距離（12 群全部の中央値 / 両施設
5 枚以上の 9 群の中央値）、近傍の混ざり（同）、主要 5 サブタイプの最近傍重心法の balanced accuracy（E→P / P→E）。

入力: output_dir/method_compare_meningioma/summary.csv（method_compare_meningioma.py、definition = meningioma の行）
出力: tables/{table Sn}_meningioma_methods.{csv,md}

Usage:
    uv run python -m figs.meningioma_methods
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS, CONDITIONS_SUPP, config, out_root, table_path

NAME = "meningioma_methods"
COLS = [("site_shift_norm", "Site-mean distance", 2), ("asw_site", "Site silhouette", 3), ("ilisi_site", "iLISI (site)", 3),
        ("pairs_adjacent", "Cross-site pairs (/12)", 0),
        ("group_gap_sd", "Site-centre distance, 12 groups", 3), ("group_gap_sd_min5", "Site-centre distance, 9 groups ≥ 5", 3),
        ("knn_mix", "Neighbour mixing, 12 groups", 3), ("knn_mix_min5", "Neighbour mixing, 9 groups ≥ 5", 3),
        ("nc_histotype_ba_E2P", "Subtype BA, E→P", 3), ("nc_histotype_ba_P2E", "Subtype BA, P→E", 3)]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    s = pd.read_csv(out_root(cfg) / "method_compare_meningioma" / "summary.csv")
    s = s[s["definition"] == "meningioma"].set_index("condition")
    assert (s["n_groups_min5"] == 9).all() and (s["pairs_n"] == 12).all()
    rows = [{"Correction": CONDITION_LABELS[c], **{lab: round(float(s.loc[c, k]), d) if d else int(s.loc[c, k]) for k, lab, d in COLS}}
            for c in CONDITIONS_SUPP]
    table = pd.DataFrame(rows)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.{d}f}" if isinstance(r[c], float) else str(r[c])
                                       for c, d in zip(cols, [0] + [d for _, _, d in COLS])) + " |")
    n_e, n_p = int(s["n_ebrains"].iloc[0]), int(s["n_patho2"].iloc[0])
    notes = [f"TITAN、髄膜腫だけ（EBRAINS {n_e} 枚 / 院内 {n_p} 枚）。どの補正も髄膜腫自身で推定。GAN と Macenko は院内の画像を EBRAINS 側へ"
             "寄せてから埋め込みを作り直したもの、ComBat・平行移動・相似変換は埋め込みでの補正。",
             "Site-mean distance = 2 施設の平均ベクトルどうしの距離（埋め込みの長さの単位）。Site silhouette = 施設を群としたシルエット"
             "係数（0 に近いほど施設で分かれない）。iLISI = 近傍の施設の多様度（最大 2）。",
             "Cross-site pairs = 施設 × サブタイプの平均の Ward 樹形図で、同じサブタイプの 2 施設が隣り合う組の数（両施設にある 12 組中）。",
             "Site-centre distance = サブタイプごとの 2 施設の中心の距離を施設内の広がりで割った値（1 = 施設内の広がりと同じ）の中央値。"
             "Neighbour mixing = 院内のスライドの近傍に EBRAINS のスライドが入る割合を、施設を無作為に入れ替えたときの期待値で割った値"
             "（1 = 施設を見分けられない）の中央値。12 groups = 両施設にある 12 サブタイプ全部、9 groups ≥ 5 = 両施設に 5 枚以上ある 9 サブタイプ"
             "（本文の値）。",
             "Subtype BA = 主要 5 サブタイプ（Meningothelial、Fibrous、Psammomatous、Angiomatous、Microcystic）の最近傍重心法の"
             "balanced accuracy。E→P は EBRAINS の重心で院内のスライドを、P→E はその逆。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
