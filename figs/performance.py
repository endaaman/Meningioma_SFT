"""施設間の分類性能の表（訓練不要の最近傍重心法、点推定 [95% CI]、患者単位 bootstrap）。

入力: output_dir/nc/binary/bootstrap.csv  （nc.py）
出力: tables/{table n}_performance.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.performance
"""
from __future__ import annotations

import csv

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, CONDITIONS_SUPP_REF, SITE_DISPLAY, config, order_conditions, out_root, table_path

NAME = "performance"
METRIC_JA = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
    "sensitivity": "感度（SFT）", "specificity": "特異度", "f1_macro": "F1 macro",
    "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
}
VARIANT_JA = CONDITION_LABELS_JA


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    bs = pd.read_csv(out_root(cfg) / "nc" / "binary" / "bootstrap.csv")
    bs = bs[bs["variant"].isin(CONDITIONS_SUPP_REF)]  # 本文の 4 条件 ＋ 相似変換・相似変換（参考）
    src_rank = {s: i for i, s in enumerate(dict.fromkeys(bs["train_source"]))}
    var_rank = {v: i for i, v in enumerate(order_conditions(bs["variant"].unique()))}
    bs = bs.assign(_s=bs["train_source"].map(src_rank), _v=bs["variant"].map(var_rank)).sort_values(
        ["_s", "_v"], kind="stable").drop(columns=["_s", "_v"])
    n_boot = cfg.get("lp_bootstrap", {}).get("n_boot", 2000)
    records = []
    for (train_src, test_src, variant), g in bs.groupby(["train_source", "test_source", "variant"], sort=False):
        rec = {"学習 → 評価": f"{SITE_DISPLAY.get(train_src, train_src)} → {SITE_DISPLAY.get(test_src, test_src)}",
               "補正": VARIANT_JA.get(variant, variant) + ("†" if variant == "affine_oracle" else "")}
        for _, r in g.iterrows():
            rec[METRIC_JA[r["metric"]]] = f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False, quoting=csv.QUOTE_MINIMAL)

    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    prev = None
    for _, r in table.iterrows():
        cells = [str(r[c]) for c in cols]
        if cells[0] == prev:
            cells[0] = ""
        else:
            prev = cells[0]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", f": 施設間の分類性能（最近傍重心法: 学習施設の SFT・髄膜腫の平均ベクトルのうち近い方に割り当てる。訓練・閾値なし）。"
              f"各値は点推定 [95% CI]（テスト側施設の患者単位 bootstrap、{n_boot} 回）。"
              "感度は SFT を陽性とした値。AUC のスコアは d(髄膜腫平均) − d(SFT 平均)。"
              "相似変換 = 一様なスケール＋平行移動（倍率はラベル不要の推定）。"
              "†倍率を組織型ラベル（評価施設を含む）から推定した参考値（循環のため上限の参考）。 {#tbl:performance}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
