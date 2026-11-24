"""v4 Supplementary の表: SFT 対 髄膜腫の施設間分類（最近傍重心法、訓練なし）。v3 の表（figs/performance.py、v3 Table S4）を
v4 用に組み直したもの: 条件を 補正なし / GAN / ComBat / 平行移動（CONDITIONS_FULL）に絞り、施設名を EBRAINS / In-house、列名を英語に。
セルの作り方（比率の指標は 該当数/例数 [Wilson 95% CI]、ほかは点推定 [患者単位ブートストラップ 95% CI]）は figs/performance.py の関数を
そのまま使う（v3 の表は変えない）。

入力: output_dir/nc/binary/{bootstrap,confusion}.csv（nc.py。髄膜腫＋SFT、EBRAINS 551 / 院内 577 枚）
出力: tables/{table Sn}_sft_cross_site.{csv,md}

Usage:
    uv run python -m figs.sft_cross_site
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS, CONDITIONS_FULL, config, out_root, table_path
from figs.performance import format_cell, proportion_counts

NAME = "sft_cross_site"
SITE = {"ebrains": "EBRAINS", "patho2": "In-house"}
METRICS = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("sensitivity", "Sensitivity (SFT)"),
           ("specificity", "Specificity"), ("f1_macro", "Macro F1"), ("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    nc = out_root(cfg) / "nc" / "binary"
    bs = pd.read_csv(nc / "bootstrap.csv")
    counts = proportion_counts(pd.read_csv(nc / "confusion.csv"), ["train_source", "test_source", "variant"])
    n_boot = cfg.get("lp_bootstrap", {}).get("n_boot", 2000)
    rows, ns = [], {}
    for tr, te in (("ebrains", "patho2"), ("patho2", "ebrains")):
        for v in CONDITIONS_FULL:
            g = bs[(bs["train_source"] == tr) & (bs["test_source"] == te) & (bs["variant"] == v)]
            rec = {"Train → test": f"{SITE[tr]} → {SITE[te]}", "Correction": CONDITION_LABELS[v]}
            for m, lab in METRICS:
                rec[lab] = format_cell(g[g["metric"] == m].iloc[0], counts[(tr, te, v)])
            rows.append(rec)
            ns[(tr, te)] = (int(g["n"].iloc[0]), int(g["n_sft"].iloc[0]))
    table = pd.DataFrame(rows)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    prev = None
    for r in rows:
        cells = [str(r[c]) for c in cols]
        if cells[0] == prev:
            cells[0] = ""
        else:
            prev = cells[0]
        lines.append("| " + " | ".join(cells) + " |")
    (n1, s1), (n2, s2) = ns[("ebrains", "patho2")], ns[("patho2", "ebrains")]
    notes = [f"TITAN、髄膜腫＋SFT（評価側: 院内 {n1} 枚うち SFT {s1}、EBRAINS {n2} 枚うち SFT {s2}）。最近傍重心法 = 学習側の施設の SFT と"
             "髄膜腫の平均ベクトルのうち近い方に割り当てる（訓練・閾値なし）。",
             "Accuracy・Sensitivity・Specificity は 該当数/例数 [Wilson score 95% 信頼区間]（スライドを独立とした二項比率）、ほかは点推定 "
             f"[95% 信頼区間]（評価側の施設の患者単位ブートストラップ {n_boot} 回）。Sensitivity は SFT を陽性とした値。AUC のスコアは "
             "d(髄膜腫の平均) − d(SFT の平均)。",
             "補正はどれも髄膜腫＋SFT の全スライドで推定（GAN は院内の画像を EBRAINS 側へ寄せてから埋め込みを作り直したもの）。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")
    print(table[["Train → test", "Correction", "Balanced accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
