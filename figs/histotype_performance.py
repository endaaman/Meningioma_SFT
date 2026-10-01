"""組織型（主要 5 クラス）の施設間分類の指標表（Supplementary）。

訓練不要の最近傍重心法。balanced accuracy・macro F1・AUC（OvR、クラスの確率 = softmax(−距離)）の点推定 [95% CI]
（テスト側施設の患者単位 bootstrap）。2 方向 × 4 条件。

入力: output_dir/nc/histotype/bootstrap.csv（nc.py）
出力: tables/{table S n}_histotype_performance.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.histotype_performance
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, CONDITIONS_FULL, SITE_DISPLAY, config, order_conditions, out_root, table_path
from histotype_labels import apply_set

NAME = "histotype_performance"
CLASS_SET = "main"
METRICS = [("balanced_accuracy", "Balanced acc."), ("f1_macro", "Macro F1"), ("auc_ovr_macro", "AUC (OvR)")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = apply_set(config(), CLASS_SET)
    root = out_root(cfg) / "nc" / "histotype"
    bs = pd.read_csv(root / "bootstrap.csv")
    classes = [c.replace(" meningioma", "") for c in cfg["lp_histotype"]["classes"]]
    n_boot = cfg.get("lp_bootstrap", {}).get("n_boot", 2000)
    sources = list(cfg["embedding"].keys())
    records = []
    for tr in sources:
        te = next(s for s in sources if s != tr)
        for v in order_conditions([v for v in CONDITIONS_FULL if v in set(bs["variant"])]):
            sub = bs[(bs.train_source == tr) & (bs.variant == v)].set_index("metric")
            if sub.empty:
                continue
            rec = {"学習 → 評価": f"{SITE_DISPLAY[tr]} → {SITE_DISPLAY[te]}", "補正": CONDITION_LABELS_JA[v]}
            for m, col in METRICS:
                r = sub.loc[m]
                rec[col] = f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            records.append(rec)
    table = pd.DataFrame(records)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)

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
    lines += ["", f": 組織型（{len(classes)} クラス: {', '.join(classes)}）の施設間分類（最近傍重心法: 学習施設の組織型平均のうち最も近いものに割り当てる。訓練なし）。"
              f"AUC は one-vs-rest のクラス平均で、各クラスの確率を softmax(−距離) とした。各値は点推定 [95% CI]"
              f"（テスト側施設の患者単位 bootstrap、{n_boot} 回）。偶然の balanced accuracy は 1/{len(classes)} = "
              f"{1 / len(classes):.2f}。 {{#tbl:histotype_performance}}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
