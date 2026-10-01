"""施設間の分類性能の表（点推定 [95% CI]、患者単位 bootstrap）。

入力: output_dir/lp/bootstrap.csv  （lp_bootstrap.py）
出力: tables/{table n}_performance.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.performance
"""
from __future__ import annotations

import csv

import pandas as pd

from figs import label
from figs.common import SITE_DISPLAY, config, order_conditions, out_root, table_path

NAME = "performance"
METRIC_JA = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
    "sensitivity": "感度（SFT）", "specificity": "特異度", "f1_macro": "F1 macro",
}
VARIANT_JA = {"original": "なし", "gan": "GAN", "centroid": "centroid"}


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    bs = pd.read_csv(out_root(cfg) / "lp" / "bootstrap.csv")
    src_rank = {s: i for i, s in enumerate(dict.fromkeys(bs["train_source"]))}
    var_rank = {v: i for i, v in enumerate(order_conditions(bs["variant"].unique()))}
    bs = bs.assign(_s=bs["train_source"].map(src_rank), _v=bs["variant"].map(var_rank)).sort_values(
        ["_s", "_v"], kind="stable").drop(columns=["_s", "_v"])
    n_boot = cfg.get("lp_bootstrap", {}).get("n_boot", 2000)
    records = []
    for (train_src, test_src, variant), g in bs.groupby(["train_source", "test_source", "variant"], sort=False):
        rec = {"学習 → 評価": f"{SITE_DISPLAY.get(train_src, train_src)} → {SITE_DISPLAY.get(test_src, test_src)}",
               "補正": VARIANT_JA.get(variant, variant)}
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
    lines += ["", f": 施設間の分類性能。各値は点推定 [95% CI]（テスト側施設の患者単位 bootstrap、{n_boot} 回）。"
              "感度は SFT を陽性とした値。 {#tbl:performance}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
