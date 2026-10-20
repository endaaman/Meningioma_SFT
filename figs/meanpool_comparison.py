"""補正法の比較（平均プーリング）の表（Supplementary）。

Table S3（TITAN）と同じ指標を、学習なしの平均プーリングで作ったスライド埋め込み（meanpool.py）で並べる。
補正法どうしの関係（画像レベルの補正は施設差を部分的にしか消さない、平行移動で施設差の大半が消える等）が
集約器によらないかを見る。

行: 補正なし / 色正規化（Macenko）/ GAN / ComBat / 平行移動（CONDITIONS_FIG3 の順）
列: 施設平均どうしの距離、施設の混ざり方（ASW・iLISI）、樹形図の隣接ペア数、対象施設内の SFT − 髄膜腫 距離、
    組織型のまとまり（ASW・cLISI）、組織型の平均どうしの距離の施設間の相対ずれ

入力: output_dir/meanpool/summary.csv、output_dir/meanpool/site_null.csv（meanpool.py）、
      output_dir/stain_response/shift_decomposition.csv（stain_response.py。Macenko の行の説明に使う、aggregator=mean）
出力: tables/{table S n}_meanpool_comparison.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.meanpool_comparison
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, SITE_DISPLAY, config, out_root, table_path
from utils.display import CONDITIONS_FIG3

NAME = "meanpool_comparison"
ROWS = CONDITIONS_FIG3


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    root = out_root(cfg)
    s = pd.read_csv(root / "meanpool" / "summary.csv").set_index("variant")
    null = pd.read_csv(root / "meanpool" / "site_null.csv")
    null = null[null["null"] == "within_subtype"]
    chance = null.groupby("metric")["null_mean"].mean()
    ref = cfg["reference"]
    other = next(k for k in cfg["embedding"] if k != ref)
    ref_d, oth_d = SITE_DISPLAY.get(ref, ref), SITE_DISPLAY.get(other, other)
    n_sub = int(s["n_subtypes"].iloc[0])
    # Macenko で施設平均どうしの距離がむしろ増える理由: 補正によるシフトの変化を、補正なしの施設シフト方向 u に沿う成分と
    # それに直交する成分に分けた値（stain_response.py）
    dec = pd.read_csv(root / "stain_response" / "shift_decomposition.csv")
    dec = dec[dec["aggregator"] == "mean"].set_index("condition")
    mac_along, mac_orth = float(dec.loc["macenko", "along_u"]), float(dec.loc["macenko", "orthogonal"])
    raw_norm = float(dec.loc["original", "shift_norm"])

    cols = {
        "site_shift_norm": "施設平均どうしの距離 ↓", "asw_batch": "ASW 施設 ↓", "ilisi": "iLISI 施設 ↑",
        "adjacent_pairs": f"隣接ペア（/{n_sub}）",
        f"sft_men_dist_{other}": f"{oth_d} 内 SFT−髄膜腫 距離（{ref_d} {s[f'sft_men_dist_{ref}'].iloc[0]:.2f}）",
        "asw_bio": "ASW 組織型 ↑", "clisi": "cLISI 組織型 ↓", "subtype_dist_mismatch": "組織型間距離の施設差",
    }
    fmt = {"site_shift_norm": "{:.2f}", "asw_batch": "{:.4f}", "ilisi": "{:.3f}", "adjacent_pairs": "{:d}",
           f"sft_men_dist_{other}": "{:.2f}", "asw_bio": "{:+.4f}", "clisi": "{:.3f}", "subtype_dist_mismatch": "{:.1%}"}
    records = []
    for v in [r for r in ROWS if r in s.index]:
        rec = {"補正": CONDITION_LABELS_JA[v]}
        for k, c in cols.items():
            val = s.loc[v, k]
            rec[c] = fmt[k].format(int(val) if k == "adjacent_pairs" else float(val))
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)

    head = list(table.columns)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in head) + " |" for _, r in table.iterrows()]
    lines += ["", ": 補正法の比較（平均プーリング）。スライド埋め込みを TITAN の集約器ではなく、同じパッチ埋め込み（CONCH v1.5）の"
              "学習なしの単純平均（768 次元）にして、Table S3 と同じ指標を求めた。画像レベルの補正（色正規化・GAN）は"
              f"正規化・変換後のパッチの埋め込みを平均し直したもの、ComBat・平行移動は平均プーリングの埋め込みに同じ手順"
              "（ComBat = 次元ごとの位置・尺度、平行移動 = 各施設の平均を揃える）をかけたもの。"
              f"施設平均どうしの距離 = mean({oth_d}) − mean({ref_d}) のノルム（平行移動の残り）。"
              "ASW・iLISI・cLISI は PCA 50 次元・k = 30。施設の指標の chance level（施設ラベルを組織型内で入れ替えた 1000 回の平均、"
              f"補正なし・平行移動の平均）は ASW {chance.get('asw_batch', float('nan')):.4f}、iLISI {chance.get('ilisi', float('nan')):.3f}。"
              "隣接ペアは施設 × 組織型の平均ベクトルの Ward 樹形図で同じ組織型の 2 施設が隣り合う数。"
              f"組織型間距離の施設差は、両施設で 5 例以上の組織型の平均どうしの距離の |{oth_d} − {ref_d}| / {ref_d} の中央値。"
              f"色正規化で施設平均どうしの距離が増えるのは、補正後の施設シフトを補正なしのシフト方向に沿う成分と直交する成分に分けると"
              f"、沿う成分は {raw_norm:.1f} → {mac_along:.1f} と {raw_norm - mac_along:.1f} しか減らない一方で、直交する成分が {mac_orth:.1f} と新たに生じるため"
              "（色正規化は平均プーリングの埋め込みを施設差の方向にはあまり動かさず、別の向きのずれを作る）。"
              " {#tbl:meanpool_comparison}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
