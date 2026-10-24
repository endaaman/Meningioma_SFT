"""STAT6 の状態の群 × 補正条件の表（Supplementary）: SFT 割り当て n/N と、距離差の中央値 [IQR]。

patho2 の各スライドを EBRAINS の SFT / 髄膜腫の重心（最近傍重心法、nc.py と同じ）に当てたときの
距離差 d(髄膜腫重心) − d(SFT 重心)（> 0 で SFT 側）を、STAT6 免疫染色の状態の群ごとにまとめる。
群はレポート本文の読み（stat6_case.py の group_text = grouping "text"）: SFT STAT6 核陽性 / SFT 未施行 /
髄膜腫 STAT6 記載あり / 髄膜腫 記載なし。条件は CONDITIONS_FIG3（補正なし / Macenko / GAN / ComBat / 平行移動）。

入力: output_dir/stat6/summary.csv（stat6_case.py）
出力: tables/{table S n}_stat6_groups.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.stat6_groups
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, SITE_DISPLAY, config, out_root, table_path
from utils.display import CONDITIONS_FIG3

NAME = "stat6_groups"
GROUPING = "text"   # レポート本文の読み（CSV 列の値 "archive" は否定の読み落としがあるので使わない。decisions.md 2026-10-22）
GROUPS = [("SFT_STAT6pos", "SFT（STAT6 核陽性）"), ("SFT_STAT6neg", "SFT（STAT6 核陰性）"),
          ("SFT_untested", "SFT（STAT6 未施行）"), ("Men_STAT6tested", "髄膜腫（STAT6 記載あり）"),
          ("Men_untested", "髄膜腫（STAT6 記載なし）")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    ref, site = cfg["reference"], cfg["stat6"]["site"]
    s = pd.read_csv(out_root(cfg) / "stat6" / "summary.csv")
    s = s[(s["grouping"] == GROUPING) & (s["centroid_site"] == ref) & s["condition"].isin(CONDITIONS_FIG3)]
    s = s.set_index(["group", "condition"])
    conds = [c for c in CONDITIONS_FIG3 if c in s.index.get_level_values("condition")]
    groups = [(g, ja) for g, ja in GROUPS if g in s.index.get_level_values("group")]
    records = []
    for g, ja in groups:
        n = int(s.loc[(g, conds[0]), "n"])
        rec = {"群": ja, "n": n}
        for c in conds:
            r = s.loc[(g, c)]
            rec[CONDITION_LABELS_JA[c]] = f"{int(r['n_sft_assigned'])}/{n}; {r['median']:.2f} [{r['q25']:.2f}, {r['q75']:.2f}]"
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)

    head = list(table.columns)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in head) + " |" for _, r in table.iterrows()]
    ref_d, site_d = SITE_DISPLAY.get(ref, ref), SITE_DISPLAY.get(site, site)
    lines += ["", f": STAT6 免疫染色の状態別にみた {site_d} スライドの最近傍重心法の結果。"
              f"各セルは SFT に割り当てられた数 / 群の例数; 距離差 d(髄膜腫重心) − d(SFT 重心) の中央値 [IQR]"
              f"（重心は {ref_d} の SFT・髄膜腫の平均ベクトル。> 0 で SFT 側）。"
              "群は病理レポート本文の STAT6 の記載から読んだもの（SFT の STAT6 核陽性 / 未施行、髄膜腫の STAT6 の記載あり / なし。"
              "記載ありの髄膜腫 = 病理医が SFT を鑑別に挙げた髄膜腫）。"
              "条件は補正なし、色正規化（Macenko）、GAN、ComBat、平行移動（いずれも patho2 → EBRAINS の向き）。 {#tbl:stat6_groups}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
