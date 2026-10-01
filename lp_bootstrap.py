"""lp_bootstrap.py — Linear Probe のテスト予測に対する患者単位 bootstrap と McNemar 検定。

lp.py が書き出す trained_by_{src}/{variant}/predictions.csv を読み、
テスト側施設の患者単位で復元抽出して各指標の 95% CI を求める。
患者 ID は config の label.<site>.patient（無い施設は 1 症例 = 1 患者）。

出力 (output_dir/lp/):
    bootstrap.csv  — 条件 × 指標の点推定と 95% CI
    mcnemar.csv    — 同じテスト集合での variant 間の McNemar exact 検定
論文 Table 2 は figs/table2.py が bootstrap.csv から組む。

--task histotype（組織型の多クラス分類、lp.py --task histotype の出力 output_dir/lp_histotype/ を読む）:
    指標は accuracy / balanced_accuracy / f1_macro / auc_ovr_macro（one-vs-rest AUC のクラス平均、
    sklearn roc_auc_score(multi_class="ovr", average="macro")。bootstrap 標本にテスト側で欠けるクラスがあると
    AUC は定義できないのでその標本は AUC だけ欠測扱い）。加えて per_class.csv（クラスごとの recall の点推定と CI）。

Usage:
    uv run python lp_bootstrap.py
    uv run python lp_bootstrap.py --task histotype
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from utils.display import order_conditions
from histotype_labels import add_set_arg, apply_set
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


METRICS_MULTI = ["accuracy", "balanced_accuracy", "f1_macro", "auc_ovr_macro"]


def _metrics_multi(t: np.ndarray, p: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    k = prob.shape[1]
    auc = np.nan
    if len(np.unique(t)) == k:
        auc = float(roc_auc_score(t, prob, multi_class="ovr", average="macro", labels=list(range(k))))
    return {
        "accuracy":          float((t == p).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "f1_macro":          float(f1_score(t, p, average="macro", labels=list(range(k)), zero_division=0)),
        "auc_ovr_macro":     auc,
    }


def _recall_per_class(t: np.ndarray, p: np.ndarray, k: int) -> dict[str, float]:
    return {f"recall_{c}": (float((p[t == c] == c).mean()) if (t == c).any() else np.nan) for c in range(k)}


def _load_patients(cfg: dict, site: str, case_ids: pd.Series) -> pd.Series:
    path = cfg["label"][site].get("patient")
    if path is None:
        return case_ids.astype(str)
    mapping = pd.read_csv(path, dtype=str).set_index("case_id")["patient_id"]
    patients = case_ids.map(mapping)
    if patients.isna().any():
        raise ValueError(f"[{site}] patient_id missing for {int(patients.isna().sum())} cases in {path}")
    return patients


def _bootstrap(df: pd.DataFrame, n_boot: int, rng: np.random.Generator,
               metric_fn=None) -> dict[str, tuple[float, float]]:
    """metric_fn(idx) -> dict。省略時は 2 値の _metrics（従来どおり）。"""
    groups = [g.index.to_numpy() for _, g in df.groupby("patient")]
    t_all, p_all = df["true"].to_numpy(), df["pred"].to_numpy()
    if metric_fn is None:
        metric_fn = lambda idx: _metrics(t_all[idx], p_all[idx])  # noqa: E731
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(t_all[idx])) < 2:  # 片方のクラスしかない標本は指標が定義できないので捨てる
            continue
        for m, v in metric_fn(idx).items():
            samples.setdefault(m, []).append(v)
    return {m: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for m, v in samples.items()}


def main_histotype(cfg: dict) -> None:
    """組織型の多クラス分類（lp.py --task histotype の出力）の bootstrap と McNemar。"""
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot, seed = bs_cfg.get("n_boot", 2000), bs_cfg.get("seed", 42)
    h_cfg = cfg["lp_histotype"]
    classes = [c.replace(" meningioma", "") for c in h_cfg["classes"]]
    k = len(classes)
    variants = order_conditions(h_cfg.get("variants", ["original"]))
    sources = list(cfg["embedding"].keys())
    lp_root = Path(cfg["output_dir"]) / h_cfg.get("output_subdir", "lp_histotype")

    rows, pc_rows, mc_rows = [], [], []
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
            t, p = df["true"].to_numpy(), df["pred"].to_numpy()
            prob = df[[f"prob_{i}" for i in range(k)]].to_numpy()
            point = _metrics_multi(t, p, prob)
            ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                            metric_fn=lambda idx: _metrics_multi(t[idx], p[idx], prob[idx]))
            print(f"[{train_src} -> {test_src} / {variant}] n={len(df)} patients={df['patient'].nunique()}")
            for m in METRICS_MULTI:
                rows.append({"train_source": train_src, "test_source": test_src, "variant": variant,
                             "metric": m, "value": point[m], "ci_low": ci[m][0], "ci_high": ci[m][1],
                             "n": len(df), "n_patients": df["patient"].nunique()})
                print(f"  {m:18s} {point[m]:.3f} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]")
            rec = _recall_per_class(t, p, k)
            rec_ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                                metric_fn=lambda idx: _recall_per_class(t[idx], p[idx], k))
            for c in range(k):
                key = f"recall_{c}"
                pc_rows.append({"train_source": train_src, "test_source": test_src, "variant": variant,
                                "class": classes[c], "n": int((t == c).sum()), "recall": rec[key],
                                "ci_low": rec_ci[key][0], "ci_high": rec_ci[key][1]})

        for a, b in combinations([v for v in variants if v in preds], 2):
            m = preds[a][["case_id", "true", "pred"]].merge(
                preds[b][["case_id", "pred"]], on="case_id", suffixes=("_a", "_b"))
            ok_a, ok_b = m["pred_a"] == m["true"], m["pred_b"] == m["true"]
            only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
            pv = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
            mc_rows.append({"train_source": train_src, "test_source": test_src, "variant_a": a, "variant_b": b,
                            "a_correct_only": only_a, "b_correct_only": only_b, "p_exact": pv})

    pd.DataFrame(rows).to_csv(lp_root / "bootstrap.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(lp_root / "per_class.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(lp_root / "mcnemar.csv", index=False)
    print(f"  saved: {lp_root}/{{bootstrap,per_class,mcnemar}}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sft", "histotype"], default="sft")
    add_set_arg(parser)
    args = parser.parse_args()
    cfg = apply_set(load_config(), args.set)
    if args.task == "histotype":
        main_histotype(cfg)
        return
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot = bs_cfg.get("n_boot", 2000)
    seed = bs_cfg.get("seed", 42)
    variants = order_conditions(cfg["lp"].get("variants", ["original"]))
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
