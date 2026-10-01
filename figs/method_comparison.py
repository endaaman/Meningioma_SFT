"""補正法の比較（全指標）の表（Supplementary）。

行: 補正なし / GAN / ComBat / 平行移動 / 相似変換 / 相似変換（正しい倍率・参考）
列: 施設の混ざり方（ASW・iLISI）、診断の分離（ASW SFT vs 髄膜腫）、組織型のまとまり（ASW・cLISI）、
    樹形図の隣接ペア数、対象施設内の SFT − 髄膜腫 距離、組織型の平均どうしの距離の施設間の相対ずれ

入力: output_dir/method_compare/summary.csv（method_compare.py）、
      output_dir/harmonization/site_null.csv（施設指標の chance level、harmonization.py）
出力: tables/{table S n}_method_comparison.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.method_comparison
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, SITE_DISPLAY, config, out_root, table_path

NAME = "method_comparison"
ROWS = ["original", "gan", "combat", "centroid", "affine_free", "affine_oracle"]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    root = out_root(cfg)
    s = pd.read_csv(root / "method_compare" / "summary.csv").set_index("variant")
    null = pd.read_csv(root / "harmonization" / "site_null.csv")
    null = null[null["null"] == "within_subtype"]
    chance = null.groupby("metric")["null_mean"].mean()
    ref = cfg["reference"]
    other = next(k for k in cfg["embedding"] if k != ref)
    ref_d, oth_d = SITE_DISPLAY.get(ref, ref), SITE_DISPLAY.get(other, other)
    n_sub = int(s["n_subtypes"].iloc[0])

    cols = {
        "asw_batch": "ASW 施設 ↓", "ilisi": "iLISI 施設 ↑", "asw_class": "ASW SFT vs 髄膜腫 ↑",
        "asw_bio": "ASW 組織型 ↑", "clisi": "cLISI 組織型 ↓", "adjacent_pairs": f"隣接ペア（/{n_sub}）",
        f"sft_men_dist_{other}": f"{oth_d} 内 SFT−髄膜腫 距離（{ref_d} {s[f'sft_men_dist_{ref}'].iloc[0]:.2f}）",
        "subtype_dist_mismatch": "組織型間距離の施設差",
    }
    fmt = {"asw_batch": "{:.4f}", "ilisi": "{:.3f}", "asw_class": "{:.3f}", "asw_bio": "{:+.4f}", "clisi": "{:.3f}",
           "adjacent_pairs": "{:d}", f"sft_men_dist_{other}": "{:.2f}", "subtype_dist_mismatch": "{:.1%}"}
    records = []
    for v in [r for r in ROWS if r in s.index]:
        rec = {"補正": CONDITION_LABELS_JA[v] + ("†" if v == "affine_oracle" else "")}
        for k, c in cols.items():
            val = s.loc[v, k]
            rec[c] = fmt[k].format(int(val) if k == "adjacent_pairs" else float(val))
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)

    head = list(table.columns)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in head) + " |" for _, r in table.iterrows()]
    lines += ["", ": 補正法の比較（全指標）。平行移動 = 各施設の平均を揃える位置のみの補正、相似変換 = 一様なスケール＋平行移動"
              "（倍率は施設平均まわりの広がりの比で、ラベル不要）、ComBat = 次元ごとの位置・尺度の補正。"
              "ASW・iLISI・cLISI は PCA 50 次元・k = 30。施設の指標の chance level（施設ラベルを組織型内で入れ替えた 1000 回の平均、"
              f"条件平均）は ASW {chance.get('asw_batch', float('nan')):.4f}、iLISI {chance.get('ilisi', float('nan')):.3f}。"
              "隣接ペアは施設 × 組織型の平均ベクトルの Ward 樹形図で同じ組織型の 2 施設が隣り合う数（Fig 6 と同じ）。"
              f"組織型間距離の施設差は、両施設で 5 例以上の組織型の平均どうしの距離の |{oth_d} − {ref_d}| / {ref_d} の中央値。"
              "†倍率を組織型ラベル（評価施設を含む）から推定した参考値（循環のため上限の参考）。 {#tbl:method_comparison}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
