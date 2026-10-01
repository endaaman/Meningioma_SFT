"""lp_bootstrap.py — Linear Probe のテスト予測に対する患者単位 bootstrap と McNemar 検定。

lp.py が書き出す trained_by_{src}/{variant}/predictions.csv を読み、
テスト側施設の患者単位で復元抽出して各指標の 95% CI を求める。
患者 ID は config の label.<site>.patient（無い施設は 1 症例 = 1 患者）。

出力 (output_dir/lp/):
    bootstrap.csv  — 条件 × 指標の点推定と 95% CI
    mcnemar.csv    — 同じテスト集合での variant 間の McNemar exact 検定
論文 Table 2 は figs/table2.py が bootstrap.csv から組む。

Usage:
    uv run python lp_bootstrap.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

from utils.loader import load_config

METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro"]


def _metrics(t: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pos, neg = t == 1, t == 0
    return {
        "accuracy":          float((t == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "sensitivity":       float((p[pos] == 1).mean()) if pos.any() else np.nan,
        "specificity":       float((p[neg] == 0).mean()) if neg.any() else np.nan,
        "f1_macro":          float(f1_score(t, p, average="macro", zero_division=0)),
    }


def _load_patients(cfg: dict, site: str, case_ids: pd.Series) -> pd.Series:
    path = cfg["label"][site].get("patient")
    if path is None:
        return case_ids.astype(str)
    mapping = pd.read_csv(path, dtype=str).set_index("case_id")["patient_id"]
    patients = case_ids.map(mapping)
    if patients.isna().any():
        raise ValueError(f"[{site}] patient_id missing for {int(patients.isna().sum())} cases in {path}")
    return patients


def _bootstrap(df: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    groups = [g.index.to_numpy() for _, g in df.groupby("patient")]
    t_all, p_all = df["true"].to_numpy(), df["pred"].to_numpy()
    samples: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        t, p = t_all[idx], p_all[idx]
        if len(np.unique(t)) < 2:  # 片方のクラスが抜けた標本は指標が定義できないので捨てる
            continue
        for m, v in _metrics(t, p).items():
            samples[m].append(v)
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in samples.items()}


def main() -> None:
    cfg = load_config()
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot = bs_cfg.get("n_boot", 2000)
    seed = bs_cfg.get("seed", 42)
    variants = cfg["lp"].get("variants", ["original"])
    sources = list(cfg["embedding"].keys())
    lp_root = Path(cfg["output_dir"]) / "lp"

    rows, mc_rows = [], []
    for train_src in sources:
        test_src = next(s for s in sources if s != train_src)
        preds: dict[str, pd.DataFrame] = {}
        for variant in variants:
            path = lp_root / f"trained_by_{train_src}" / variant / "predictions.csv"
            if not path.exists():
                print(f"  SKIP: {path} not found")
                continue
            df = pd.read_csv(path, dtype={"case_id": str})
            df["patient"] = _load_patients(cfg, test_src, df["case_id"]).to_numpy()
            preds[variant] = df
            point = _metrics(df["true"].to_numpy(), df["pred"].to_numpy())
            ci = _bootstrap(df, n_boot, np.random.default_rng(seed))
            n_sft = int((df["true"] == 1).sum())
            print(f"[{train_src} -> {test_src} / {variant}] n={len(df)} patients={df['patient'].nunique()} SFT={n_sft}")
            for m in METRICS:
                rows.append({"train_source": train_src, "test_source": test_src, "variant": variant,
                             "metric": m, "value": point[m], "ci_low": ci[m][0], "ci_high": ci[m][1],
                             "n": len(df), "n_patients": df["patient"].nunique(), "n_sft": n_sft})
                print(f"  {m:18s} {point[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]")

        for a, b in combinations([v for v in variants if v in preds], 2):
            m = preds[a][["case_id", "true", "pred"]].merge(
                preds[b][["case_id", "pred"]], on="case_id", suffixes=("_a", "_b"))
            ok_a, ok_b = m["pred_a"] == m["true"], m["pred_b"] == m["true"]
            only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
            p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            mc_rows.append({"train_source": train_src, "test_source": test_src, "variant_a": a, "variant_b": b,
                            "a_correct_only": only_a, "b_correct_only": only_b, "p_exact": p})
            print(f"  McNemar {a} vs {b}: {a} only={only_a}, {b} only={only_b}, p={p:.4f}")

    bs = pd.DataFrame(rows)
    bs.to_csv(lp_root / "bootstrap.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(lp_root / "mcnemar.csv", index=False)
    print(f"  saved: {lp_root / 'bootstrap.csv'}\n  saved: {lp_root / 'mcnemar.csv'}")


if __name__ == "__main__":
    main()
