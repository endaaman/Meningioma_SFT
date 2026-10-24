"""施設間の分類性能の表（訓練不要の最近傍重心法、点推定 [95% CI]）。

比率の指標（accuracy・感度・特異度）は n/N [Wilson score 95% CI]（confusion.csv の件数から計算。
患者単位 bootstrap の percentile 区間は 32/32 のような標本で [1.000, 1.000] に退化するため。ken 決定 2026-10-24）。
その他の指標（balanced accuracy・F1・AUC）は点推定 [患者単位 bootstrap 95% CI] のまま。

入力: output_dir/nc/binary/{bootstrap,confusion}.csv  （nc.py）
出力: tables/{table n}_performance.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.performance
"""
from __future__ import annotations

import csv
import math

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS_JA, CONDITIONS_FIG3, SITE_DISPLAY, config, order_conditions, out_root, table_path

NAME = "performance"
METRIC_JA = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
    "sensitivity": "感度（SFT）", "specificity": "特異度", "f1_macro": "F1 macro",
    "roc_auc": "ROC AUC", "pr_auc": "PR AUC",
}
VARIANT_JA = CONDITION_LABELS_JA
# 比率として n/N [Wilson 95% CI] で出す指標（残りは点推定 [bootstrap 95% CI]）
PROPORTION_METRICS = ("accuracy", "sensitivity", "specificity")
Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """二項比率 k/n の Wilson score 区間（連続性補正なし）。k = n でも上限は 1 のまま下限だけ縮む。"""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion_counts(confusion: pd.DataFrame, keys: list[str]) -> dict[tuple, dict[str, tuple[int, int]]]:
    """confusion.csv（true, pred, n の長い形式）から、keys ごとの {metric: (k, N)} を作る。感度は SFT（true = 1）が陽性。"""
    out: dict[tuple, dict[str, tuple[int, int]]] = {}
    for key, g in confusion.groupby(keys, sort=False):
        cm = {(int(t), int(p)): int(n) for t, p, n in zip(g["true"], g["pred"], g["n"])}
        tp, fn = cm.get((1, 1), 0), cm.get((1, 0), 0)
        tn, fp = cm.get((0, 0), 0), cm.get((0, 1), 0)
        out[tuple(key) if isinstance(key, tuple) else (key,)] = {
            "accuracy": (tp + tn, tp + tn + fp + fn), "sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp)}
    return out


def format_cell(r: pd.Series, counts: dict[str, tuple[int, int]] | None) -> str:
    """bootstrap.csv の 1 行を表のセルにする。比率の指標は n/N [Wilson]、それ以外は点推定 [bootstrap CI]。"""
    if counts is not None and r["metric"] in PROPORTION_METRICS:
        k, n = counts[r["metric"]]
        assert abs(k / n - r["value"]) < 1e-9, f"{r['metric']}: {k}/{n} != {r['value']}"
        lo, hi = wilson(k, n)
        return f"{k}/{n} [{lo:.3f}, {hi:.3f}]"
    return f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    bs = pd.read_csv(out_root(cfg) / "nc" / "binary" / "bootstrap.csv")
    counts = proportion_counts(pd.read_csv(out_root(cfg) / "nc" / "binary" / "confusion.csv"),
                               ["train_source", "test_source", "variant"])
    bs = bs[bs["variant"].isin(CONDITIONS_FIG3)]  # 本文の 4 条件 ＋ 色正規化（Macenko）。相似変換は Table S3 だけ
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
            rec[METRIC_JA[r["metric"]]] = format_cell(r, counts[(train_src, test_src, variant)])
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
              "accuracy・感度・特異度は 該当数/例数 [Wilson score 95% CI]（スライドを独立とした二項比率の区間）、"
              f"その他は点推定 [95% CI]（テスト側施設の患者単位 bootstrap、{n_boot} 回）。"
              "感度は SFT を陽性とした値。AUC のスコアは d(髄膜腫平均) − d(SFT 平均)。"
              "色正規化（Macenko）= patho2 のパッチを Macenko 法で EBRAINS の染色に合わせてから埋め込みを作り直したもの"
              "（GAN と同じ画像レベルの補正）。 {#tbl:performance}"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
