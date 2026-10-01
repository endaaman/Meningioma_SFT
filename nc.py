"""nc.py — 訓練不要の施設間分類（最近傍重心法）。

学習施設で各クラスの平均ベクトル（スライド埋め込みのユークリッド平均）を求め、評価施設の各スライドを
最も近いクラス平均に割り当てる。学習するパラメータも閾値もない（「学習」はクラス平均を取るだけ）。

タスク:
    binary    — SFT vs 髄膜腫（全スライド）。スコア = d(髄膜腫平均) − d(SFT 平均)、> 0 で SFT。
                指標: accuracy / balanced_accuracy / sensitivity（SFT）/ specificity / f1_macro / roc_auc / pr_auc
    histotype — 組織型（config の lp_histotype_sets.<set> のクラス、既定 main = 5 クラス）。最も近い組織型平均。
                各クラスの確率は softmax(−d)（d = クラス平均までのユークリッド距離）と定義し、AUC（one-vs-rest の
                クラス平均）はこの確率から計算する。指標: accuracy / balanced_accuracy / f1_macro / auc_ovr_macro、
                クラスごとの recall
条件: config の nc.variants（既定は original / gan / combat / centroid / affine_free / affine_oracle）。
    reference 施設は補正 variant でも常に original（load_data と同じ規則）。
CI: 評価施設の患者単位 bootstrap（lp_bootstrap.py と同じ実装・同じ回数と seed）。条件間は McNemar exact 検定。

出力 (output_dir/nc/{binary,histotype}/):
    trained_by_{src}/{variant}/predictions.csv — case_id, true, pred, prob（binary: SFT スコア / histotype: 予測クラスの確率）
                                                  histotype は prob_0.. も
    bootstrap.csv, mcnemar.csv, confusion.csv（histotype は per_class.csv も）

Usage:
    uv run python nc.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score

from histotype_labels import apply_set
from lp_bootstrap import _bootstrap, _load_patients, _metrics, _metrics_multi, _recall_per_class
from utils.display import order_conditions
from utils.loader import load_config, load_data

DEFAULT_VARIANTS = ["original", "gan", "combat", "centroid", "affine_free", "affine_oracle"]


def _load(cfg: dict, variant: str) -> pd.DataFrame:
    df = load_data(cfg, cfg.get("variants", {}).get(variant, variant))
    return df.sort_values(["source", "case_id"]).reset_index(drop=True)


def _directions(cfg: dict) -> list[tuple[str, str]]:
    sources = list(cfg["embedding"].keys())
    return [(s, next(t for t in sources if t != s)) for s in sources]


def _dist(Z: np.ndarray, C: np.ndarray) -> np.ndarray:
    """各行（スライド）から各クラス平均へのユークリッド距離（n × k）。"""
    return np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1))


def _binary_metrics(t: np.ndarray, p: np.ndarray, s: np.ndarray) -> dict[str, float]:
    out = _metrics(t, p)
    two = len(np.unique(t)) == 2
    out["roc_auc"] = float(roc_auc_score(t, s)) if two else np.nan
    out["pr_auc"] = float(average_precision_score(t, s)) if two else np.nan
    return out


def _mcnemar(preds: dict[str, pd.DataFrame], tr: str, te: str) -> list[dict]:
    rows = []
    for a, b in combinations(list(preds), 2):
        m = preds[a][["case_id", "true", "pred"]].merge(preds[b][["case_id", "pred"]], on="case_id",
                                                         suffixes=("_a", "_b"))
        ok_a, ok_b = m["pred_a"] == m["true"], m["pred_b"] == m["true"]
        only_a, only_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        pv = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        rows.append({"train_source": tr, "test_source": te, "variant_a": a, "variant_b": b,
                     "a_correct_only": only_a, "b_correct_only": only_b, "p_exact": pv})
    return rows


def run_binary(cfg: dict, variants: list[str], out_dir: Path, n_boot: int, seed: int) -> None:
    rows, mc_rows, cm_rows = [], [], []
    for tr, te in _directions(cfg):
        preds: dict[str, pd.DataFrame] = {}
        for v in variants:
            d = _load(cfg, v)
            Z = np.stack(d["embedding"].values).astype(np.float64)
            y = d["subtype"].str.contains("sft", case=False).to_numpy().astype(int)
            src = d["source"].to_numpy()
            trm, tem = src == tr, src == te
            C = np.stack([Z[trm & (y == 0)].mean(0), Z[trm & (y == 1)].mean(0)])  # 0 = 髄膜腫, 1 = SFT
            D = _dist(Z[tem], C)
            score = D[:, 0] - D[:, 1]
            t, p = y[tem], (score > 0).astype(int)
            df = pd.DataFrame({"case_id": d.loc[tem, "case_id"].to_numpy(), "true": t, "pred": p, "prob": score})
            sub = out_dir / f"trained_by_{tr}" / v
            sub.mkdir(parents=True, exist_ok=True)
            df.to_csv(sub / "predictions.csv", index=False)
            df["patient"] = _load_patients(cfg, te, df["case_id"]).to_numpy()
            preds[v] = df
            point = _binary_metrics(t, p, score)
            ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                            metric_fn=lambda idx: _binary_metrics(t[idx], p[idx], score[idx]))
            n_sft = int(t.sum())
            for m, val in point.items():
                rows.append({"train_source": tr, "test_source": te, "variant": v, "metric": m, "value": val,
                             "ci_low": ci[m][0], "ci_high": ci[m][1], "n": len(df),
                             "n_patients": df["patient"].nunique(), "n_sft": n_sft})
            for i in range(2):
                for j in range(2):
                    cm_rows.append({"train_source": tr, "test_source": te, "variant": v, "true": i, "pred": j,
                                    "n": int(((t == i) & (p == j)).sum())})
            print(f"  [binary {tr}→{te} / {v}] BA {point['balanced_accuracy']:.3f} "
                  f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]  "
                  f"sens {point['sensitivity']:.3f}  AUC {point['roc_auc']:.3f} "
                  f"[{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}]")
        mc_rows += _mcnemar(preds, tr, te)
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "mcnemar.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion.csv", index=False)
    print(f"  saved: {out_dir}/{{bootstrap,mcnemar,confusion}}.csv")


def run_histotype(cfg: dict, variants: list[str], out_dir: Path, n_boot: int, seed: int) -> None:
    classes: list[str] = cfg["lp_histotype"]["classes"]
    k = len(classes)
    short = [c.replace(" meningioma", "") for c in classes]
    rows, pc_rows, mc_rows, cm_rows = [], [], [], []
    for tr, te in _directions(cfg):
        preds: dict[str, pd.DataFrame] = {}
        for v in variants:
            d = _load(cfg, v)
            d = d[d["subtype"].isin(classes)].reset_index(drop=True)
            Z = np.stack(d["embedding"].values).astype(np.float64)
            y = d["subtype"].map({c: i for i, c in enumerate(classes)}).to_numpy()
            src = d["source"].to_numpy()
            trm, tem = src == tr, src == te
            C = np.stack([Z[trm & (y == c)].mean(0) for c in range(k)])
            D = _dist(Z[tem], C)
            logit = -D - (-D).max(1, keepdims=True)
            prob = np.exp(logit) / np.exp(logit).sum(1, keepdims=True)
            t, p = y[tem], D.argmin(1)
            df = pd.DataFrame({"case_id": d.loc[tem, "case_id"].to_numpy(), "true": t, "pred": p,
                               "prob": prob[np.arange(len(p)), p]})
            for c in range(k):
                df[f"prob_{c}"] = prob[:, c]
            sub = out_dir / f"trained_by_{tr}" / v
            sub.mkdir(parents=True, exist_ok=True)
            df.to_csv(sub / "predictions.csv", index=False)
            df["patient"] = _load_patients(cfg, te, df["case_id"]).to_numpy()
            preds[v] = df
            point = _metrics_multi(t, p, prob)
            ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                            metric_fn=lambda idx: _metrics_multi(t[idx], p[idx], prob[idx]))
            for m, val in point.items():
                rows.append({"train_source": tr, "test_source": te, "variant": v, "metric": m, "value": val,
                             "ci_low": ci[m][0], "ci_high": ci[m][1], "n": len(df),
                             "n_patients": df["patient"].nunique()})
            rec = _recall_per_class(t, p, k)
            rec_ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                                metric_fn=lambda idx: _recall_per_class(t[idx], p[idx], k))
            for c in range(k):
                key = f"recall_{c}"
                pc_rows.append({"train_source": tr, "test_source": te, "variant": v, "class": short[c],
                                "n": int((t == c).sum()), "recall": rec[key],
                                "ci_low": rec_ci[key][0], "ci_high": rec_ci[key][1]})
                for j in range(k):
                    cm_rows.append({"train_source": tr, "test_source": te, "variant": v, "true": short[c],
                                    "pred": short[j], "n": int(((t == c) & (p == j)).sum())})
            print(f"  [histotype {tr}→{te} / {v}] BA {point['balanced_accuracy']:.3f} "
                  f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]  "
                  f"F1 {point['f1_macro']:.3f}  AUC {point['auc_ovr_macro']:.3f}")
        mc_rows += _mcnemar(preds, tr, te)
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap.csv", index=False)
    pd.DataFrame(pc_rows).to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "mcnemar.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion.csv", index=False)
    print(f"  saved: {out_dir}/{{bootstrap,per_class,mcnemar,confusion}}.csv")


def main() -> None:
    cfg = load_config()
    nc_cfg = cfg.get("nc", {})
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot, seed = bs_cfg.get("n_boot", 2000), bs_cfg.get("seed", 42)
    variants = order_conditions(nc_cfg.get("variants", DEFAULT_VARIANTS))
    root = Path(cfg["output_dir"]) / "nc"
    for task in ("binary", "histotype"):
        out_dir = root / task
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[nc / {task}]")
        if task == "binary":
            run_binary(cfg, variants, out_dir, n_boot, seed)
        else:
            run_histotype(apply_set(cfg, nc_cfg.get("histotype_set", "main")), variants, out_dir, n_boot, seed)


if __name__ == "__main__":
    main()
