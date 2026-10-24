"""施設間の分類性能の表（スライド単位の k 近傍法、点推定 [95% CI]）。Table S1（最近傍重心法）の対照。
セルの書式は Table S1 と同じ（比率の指標は n/N [Wilson 95% CI]、他は点推定 [患者単位 bootstrap 95% CI]）。

入力: output_dir/knn/binary/{bootstrap,confusion}.csv  （knn.py）
出力: tables/{table n}_knn_performance.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.knn_performance
"""
from __future__ import annotations

import csv

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, CONDITIONS_FIG3, SITE_DISPLAY, config, order_conditions, out_root, table_path
from figs.performance import METRIC_JA, format_cell, proportion_counts

NAME = "knn_performance"
K_MAIN = 5


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    bs = pd.read_csv(out_root(cfg) / "knn" / "binary" / "bootstrap.csv")
    bs = bs[bs["variant"].isin(CONDITIONS_FIG3)]  # Table S1 と同じ 5 条件
    counts = proportion_counts(pd.read_csv(out_root(cfg) / "knn" / "binary" / "confusion.csv"),
                               ["train_source", "test_source", "variant", "k"])
    src_rank = {s: i for i, s in enumerate(dict.fromkeys(bs["train_source"]))}
    var_rank = {v: i for i, v in enumerate(order_conditions(bs["variant"].unique()))}
    ks = sorted(bs["k"].unique())
    bs = bs.assign(_s=bs["train_source"].map(src_rank), _v=bs["variant"].map(var_rank)).sort_values(
        ["_s", "k", "_v"], kind="stable").drop(columns=["_s", "_v"])
    n_boot = cfg.get("lp_bootstrap", {}).get("n_boot", 2000)
    records = []
    for (train_src, test_src, k, variant), g in bs.groupby(["train_source", "test_source", "k", "variant"], sort=False):
        rec = {"学習 → 評価": f"{SITE_DISPLAY.get(train_src, train_src)} → {SITE_DISPLAY.get(test_src, test_src)}",
               "k": int(k), "補正": CONDITION_LABELS_JA.get(variant, variant)}
        for _, r in g.iterrows():
            rec[METRIC_JA[r["metric"]]] = format_cell(r, counts[(train_src, test_src, variant, k)])
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False, quoting=csv.QUOTE_MINIMAL)

    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    prev = (None, None)
    for _, r in table.iterrows():
        cells = [str(r[c]) for c in cols]
        key = (cells[0], cells[1])
        if key == prev:
            cells[0], cells[1] = "", ""
        elif cells[0] == prev[0]:
            cells[0] = ""
            prev = key
        else:
            prev = key
        lines.append("| " + " | ".join(cells) + " |")
    k_others = " / ".join(str(k) for k in ks if k != K_MAIN)
    lines += ["", f": 施設間の分類性能（スライド単位の k 近傍法: 評価施設の各スライドを、学習施設のスライドのうちユークリッド距離で近い "
              f"k 枚の多数決で SFT / 髄膜腫に割り当てる。訓練・閾値なし）。最近傍重心法（Table S1）が平均ベクトルに依存するのに対し、"
              f"平均を使わない対照。主な k は {K_MAIN}、k = {k_others} は感度分析。"
              "accuracy・感度・特異度は 該当数/例数 [Wilson score 95% CI]、"
              f"その他は点推定 [95% CI]（テスト側施設の患者単位 bootstrap、{n_boot} 回）。"
              "感度は SFT を陽性とした値。AUC のスコアは k 近傍のうち SFT の割合（k = 1 では 0/1 なので AUC は balanced accuracy に一致）。"
              "条件は Table S1 と同じ 5 つ。 {#tbl:knn_performance}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
