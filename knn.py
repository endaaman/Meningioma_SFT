"""knn.py — 施設間分類のスライド単位 k 近傍法（最近傍重心法 nc.py の対照、査読対応）。

nc.py の評価（最近傍重心法・平均ベクトルの樹形図・施設平均間の距離）はどれも 1 次モーメント（平均）の指標で、
平均を揃える平行移動に構造的に有利。この対照は平均を使わない: 評価施設の各スライドを、学習施設のスライドのうち
ユークリッド距離（768 次元の TITAN 空間、読み込みは nc.py と同じ）で近い k 枚の多数決で SFT / 髄膜腫に割り当てる。
学習するパラメータも閾値もない（k は奇数で同票なし）。

    スコア = k 近傍のうち SFT の割合（ROC / PR AUC はこれから。k = 1 では 0/1 なので AUC = balanced accuracy）
    予測   = 割合 > 0.5 で SFT
    指標   = nc.py の binary と同じ（accuracy / balanced_accuracy / sensitivity（SFT）/ specificity / f1_macro / roc_auc / pr_auc）
    CI     = 評価施設の患者単位 bootstrap（lp_bootstrap._bootstrap、nc.py と同じ回数と seed）。条件間は McNemar exact 検定（k ごと）

条件: 既定は utils/display.py の CONDITIONS_FIG3（original / macenko / gan / combat / centroid。Table S1 と同じ 5 条件）。
reference 施設は補正 variant でも常に original（load_data と同じ規則）。両方向（EBRAINS → patho2 / patho2 → EBRAINS）。
k: 既定 1 / 5 / 15（主は 5、1 と 15 は感度分析）。

出力 (output_dir/knn/binary/。nc.py の out/nc/binary と同じ配置に k の列を足したもの):
    trained_by_{src}/{variant}/predictions.csv — case_id, true, pred_k{k}, prob_k{k}（prob = SFT の割合）
    bootstrap.csv（k 列あり）, mcnemar.csv（k 列あり、同じ k の条件間）, confusion.csv（k 列あり）
    summary_table.md — k ごとの要約表（読みと原稿の文案は手書きの summary.md）

Usage:
    uv run python knn.py
    uv run python knn.py --k 1 5 15 --variants original centroid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lp_bootstrap import _bootstrap, _load_patients
from nc import _binary_metrics, _directions, _load, _mcnemar
from utils.display import CONDITION_LABELS_JA, CONDITIONS_FIG3, order_conditions
from utils.loader import load_config

DEFAULT_KS = [1, 5, 15]
SITE_DISPLAY = {"ebrains": "EBRAINS", "patho2": "patho2"}


def knn_vote(Z_train: np.ndarray, y_train: np.ndarray, Z_test: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """評価スライドごとに、学習スライドのうち近い k 枚に占める SFT（y = 1）の割合を返す（k → n_test）。"""
    # (n_test × n_train) のユークリッド距離。768 次元 × 600 枚程度なので素直に計算する
    D = np.sqrt(np.maximum(((Z_test ** 2).sum(1)[:, None] + (Z_train ** 2).sum(1)[None, :]
                            - 2.0 * Z_test @ Z_train.T), 0.0))
    order = np.argsort(D, axis=1, kind="stable")
    return {k: y_train[order[:, :k]].mean(1) for k in ks}


def run_binary(cfg: dict, variants: list[str], ks: list[int], out_dir: Path, n_boot: int, seed: int) -> None:
    rows, mc_rows, cm_rows = [], [], []
    for tr, te in _directions(cfg):
        preds_by_k: dict[int, dict[str, pd.DataFrame]] = {k: {} for k in ks}
        for v in variants:
            d = _load(cfg, v)
            Z = np.stack(d["embedding"].values).astype(np.float64)
            y = d["subtype"].str.contains("sft", case=False).to_numpy().astype(int)
            src = d["source"].to_numpy()
            trm, tem = src == tr, src == te
            frac = knn_vote(Z[trm], y[trm], Z[tem], ks)
            t = y[tem]
            case_ids = d.loc[tem, "case_id"].to_numpy()
            out = pd.DataFrame({"case_id": case_ids, "true": t})
            patients = _load_patients(cfg, te, out["case_id"]).to_numpy()
            n_sft = int(t.sum())
            for k in ks:
                score = frac[k]
                p = (score > 0.5).astype(int)
                out[f"pred_k{k}"] = p
                out[f"prob_k{k}"] = score
                df = pd.DataFrame({"case_id": case_ids, "true": t, "pred": p, "prob": score, "patient": patients})
                preds_by_k[k][v] = df
                point = _binary_metrics(t, p, score)
                ci = _bootstrap(df, n_boot, np.random.default_rng(seed),
                                metric_fn=lambda idx, t=t, p=p, score=score: _binary_metrics(t[idx], p[idx], score[idx]))
                for m, val in point.items():
                    rows.append({"train_source": tr, "test_source": te, "variant": v, "k": k, "metric": m,
                                 "value": val, "ci_low": ci[m][0], "ci_high": ci[m][1], "n": len(df),
                                 "n_patients": df["patient"].nunique(), "n_sft": n_sft})
                for i in range(2):
                    for j in range(2):
                        cm_rows.append({"train_source": tr, "test_source": te, "variant": v, "k": k, "true": i,
                                        "pred": j, "n": int(((t == i) & (p == j)).sum())})
                print(f"  [knn {tr}→{te} / {v} / k={k}] BA {point['balanced_accuracy']:.3f} "
                      f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]  "
                      f"sens {point['sensitivity']:.3f} ({int((p[t == 1] == 1).sum())}/{n_sft})  "
                      f"spec {point['specificity']:.3f}  AUC {point['roc_auc']:.3f}")
            sub = out_dir / f"trained_by_{tr}" / v
            sub.mkdir(parents=True, exist_ok=True)
            out.to_csv(sub / "predictions.csv", index=False)
        for k in ks:
            mc_rows += [{**r, "k": k} for r in _mcnemar(preds_by_k[k], tr, te)]
    pd.DataFrame(rows).to_csv(out_dir / "bootstrap.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "mcnemar.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion.csv", index=False)
    print(f"  saved: {out_dir}/{{bootstrap,mcnemar,confusion}}.csv")


def write_summary(out_dir: Path, ks: list[int], n_boot: int) -> None:
    """k ごとの要約表（条件 × 方向 → BA [CI]、感度 n/N、特異度、AUC）を summary.md に書く。読みは手で足す。"""
    bs = pd.read_csv(out_dir / "bootstrap.csv")
    cm = pd.read_csv(out_dir / "confusion.csv")
    lines = ["# k 近傍法（スライド単位）の施設間分類 — 要約", "",
             f"`knn.py` の出力。各値は点推定 [95% CI]（評価施設の患者単位 bootstrap、{n_boot} 回）。"
             "感度は SFT を陽性とした n/N、AUC のスコアは k 近傍のうち SFT の割合。", ""]
    for k in ks:
        lines += [f"## k = {k}", "", "| 学習 → 評価 | 補正 | Balanced acc. | 感度（SFT） n/N | 特異度 | ROC AUC |",
                  "|---|---|---|---|---|---|"]
        for tr, te in [(a, b) for a, b in dict.fromkeys(zip(bs["train_source"], bs["test_source"]))]:
            for v in order_conditions(bs["variant"].unique()):
                g = bs[(bs["train_source"] == tr) & (bs["variant"] == v) & (bs["k"] == k)].set_index("metric")
                c = cm[(cm["train_source"] == tr) & (cm["variant"] == v) & (cm["k"] == k)]
                tp = int(c[(c["true"] == 1) & (c["pred"] == 1)]["n"].iloc[0])
                n_sft = int(c[c["true"] == 1]["n"].sum())
                tn = int(c[(c["true"] == 0) & (c["pred"] == 0)]["n"].iloc[0])
                n_men = int(c[c["true"] == 0]["n"].sum())

                def fmt(m: str) -> str:
                    r = g.loc[m]
                    return f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"

                lines.append(f"| {SITE_DISPLAY.get(tr, tr)} → {SITE_DISPLAY.get(te, te)} | {CONDITION_LABELS_JA.get(v, v)} | "
                             f"{fmt('balanced_accuracy')} | {fmt('sensitivity')} ({tp}/{n_sft}) | "
                             f"{fmt('specificity')} ({tn}/{n_men}) | {fmt('roc_auc')} |")
        lines.append("")
    path = out_dir / "summary_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_KS, help="近傍数（奇数。既定 1 5 15、主は 5）")
    ap.add_argument("--variants", nargs="+", default=None, help="条件（既定は CONDITIONS_FIG3）")
    args = ap.parse_args()
    if any(k % 2 == 0 for k in args.k):
        ap.error("--k は奇数（多数決の同票を避ける）")
    cfg = load_config()
    bs_cfg = cfg.get("lp_bootstrap", {})
    n_boot, seed = bs_cfg.get("n_boot", 2000), bs_cfg.get("seed", 42)
    variants = order_conditions(args.variants or CONDITIONS_FIG3)
    out_dir = Path(cfg["output_dir"]) / "knn" / "binary"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[knn / binary] k = {args.k}  variants = {variants}")
    run_binary(cfg, variants, args.k, out_dir, n_boot, seed)
    write_summary(out_dir, args.k, n_boot)


if __name__ == "__main__":
    main()
