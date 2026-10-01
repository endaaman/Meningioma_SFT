"""lp_auc.py — SFT の確率から閾値に依存しない指標（ROC AUC・PR AUC）を出し、重みなし / 重み付きを比べる。

補正が「分離そのもの（順位）」を変えているのか、「閾値の位置」を直しているだけなのかを見るため。
入力: output_dir/{lp,lp_weighted}/trained_by_<src>/<variant>/predictions.csv（case_id, true, pred, prob）
      output_dir/{lp,lp_weighted}/bootstrap.csv（lp_bootstrap.py の出力）
出力: output_dir/lp_weighted/auc.csv、output_dir/lp_weighted/summary.md

Usage:
    uv run python lp_auc.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from lp_bootstrap import _load_patients
from utils.display import order_conditions
from utils.loader import load_config

RUNS = {"unweighted": "lp", "weighted": "lp_weighted"}


def _auc_ci(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, tuple[float, float, float]]:
    t, s = df["true"].to_numpy(), df["prob"].to_numpy()
    point = {"roc_auc": roc_auc_score(t, s), "pr_auc": average_precision_score(t, s)}
    groups = [g.index.to_numpy() for _, g in df.groupby("patient")]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {"roc_auc": [], "pr_auc": []}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(t[idx])) < 2:
            continue
        samples["roc_auc"].append(roc_auc_score(t[idx], s[idx]))
        samples["pr_auc"].append(average_precision_score(t[idx], s[idx]))
    return {m: (point[m], float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in samples.items()}


def main() -> None:
    cfg = load_config()
    out = Path(cfg["output_dir"])
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot, seed = bs_cfg.get("n_boot", 2000), bs_cfg.get("seed", 42)
    variants = order_conditions(cfg["lp"].get("variants", ["original"]))
    sources = list(cfg["embedding"].keys())

    rows, preds = [], {}
    for run, sub in RUNS.items():
        for train_src in sources:
            test_src = next(s for s in sources if s != train_src)
            for v in variants:
                path = out / sub / f"trained_by_{train_src}" / v / "predictions.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path, dtype={"case_id": str})
                df["patient"] = _load_patients(cfg, test_src, df["case_id"]).to_numpy()
                preds[(run, train_src, v)] = df
                for m, (pt, lo, hi) in _auc_ci(df, n_boot, seed).items():
                    rows.append({"run": run, "train_source": train_src, "test_source": test_src,
                                 "variant": v, "metric": m, "value": pt, "ci_low": lo, "ci_high": hi})
    auc = pd.DataFrame(rows)
    (out / "lp_weighted").mkdir(parents=True, exist_ok=True)
    auc.to_csv(out / "lp_weighted" / "auc.csv", index=False)

    # ── summary.md ──
    bs = pd.concat([pd.read_csv(out / sub / "bootstrap.csv").assign(run=run) for run, sub in RUNS.items()])
    allm = pd.concat([bs[["run", "train_source", "test_source", "variant", "metric", "value", "ci_low", "ci_high"]],
                      auc])
    fmt = lambda r: f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"  # noqa: E731
    lines = ["# SFT 分類: クラス重みなし vs 重み付き（balanced）", "",
             "値は点推定 [95% CI]（評価施設の患者単位 bootstrap、2000 回）。AUC は SFT の確率から計算（閾値に依存しない）。", ""]
    for train_src in sources:
        test_src = next(s for s in sources if s != train_src)
        lines += [f"## {train_src} → {test_src}", "",
                  "| 条件 | 重み | BA | SFT 感度 | 特異度 | ROC AUC | PR AUC |", "|---|---|---|---|---|---|---|"]
        for v in variants:
            for run in RUNS:
                sel = allm[(allm.run == run) & (allm.train_source == train_src) & (allm.variant == v)].set_index("metric")
                if sel.empty:
                    continue
                cell = {m: fmt(sel.loc[m]) for m in ["balanced_accuracy", "sensitivity", "specificity", "roc_auc", "pr_auc"]}
                lines.append(f"| {v} | {run} | {cell['balanced_accuracy']} | {cell['sensitivity']} | "
                             f"{cell['specificity']} | {cell['roc_auc']} | {cell['pr_auc']} |")
        lines.append("")
        # McNemar（重み付き）
        mc = pd.read_csv(out / "lp_weighted" / "mcnemar.csv")
        mc = mc[mc.train_source == train_src]
        lines += ["McNemar（重み付き）: " + "、".join(
            f"{r.variant_a} vs {r.variant_b} {r.a_correct_only}/{r.b_correct_only} p={r.p_exact:.3f}" for r in mc.itertuples()), ""]
        # 誤分類の比較: 重みなしで取りこぼした SFT が重み付きで拾えたか
        lines += ["誤分類の比較（SFT の取りこぼし = 真 SFT を髄膜腫と判定、偽陽性 = 真髄膜腫を SFT と判定）:", ""]
        for v in variants:
            k_u, k_w = ("unweighted", train_src, v), ("weighted", train_src, v)
            if k_u not in preds or k_w not in preds:
                continue
            u, w = preds[k_u].set_index("case_id"), preds[k_w].set_index("case_id")
            fn_u = set(u.index[(u.true == 1) & (u.pred == 0)]); fn_w = set(w.index[(w.true == 1) & (w.pred == 0)])
            fp_u = set(u.index[(u.true == 0) & (u.pred == 1)]); fp_w = set(w.index[(w.true == 0) & (w.pred == 1)])
            lines.append(f"- {v}: 取りこぼし 重みなし {len(fn_u)} → 重み付き {len(fn_w)}"
                         f"（拾えた {sorted(fn_u - fn_w)}、残った {sorted(fn_u & fn_w)}、新たに落とした {sorted(fn_w - fn_u)}）／"
                         f"偽陽性 {len(fp_u)} → {len(fp_w)}（新規 {sorted(fp_w - fp_u)}）")
        lines.append("")
    (out / "lp_weighted" / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
