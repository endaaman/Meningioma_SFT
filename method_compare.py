"""method_compare.py — 補正法の比較（全指標、Supplementary の表の元）。

条件: 補正なし / GAN / ComBat / 平行移動（centroid）/ 相似変換（affine_free）/ 相似変換（正しい倍率・参考、affine_oracle）
指標:
    施設の混ざり方   ASW(site)・iLISI(site)（harmonization.py と同じ定義: PCA 50 次元、k = 30）
    診断の分離       ASW(SFT vs 髄膜腫)
    組織型のまとまり ASW(組織型)・cLISI(組織型)
    サブタイプ構造   樹形図で同じ組織型の 2 施設が隣接する数（dendrogram.py と同じ: 平均ベクトル、Ward）
    形の一致         対象施設内の SFT − 髄膜腫 の平均間距離（reference 施設の値と比べる）、
                     組織型の平均どうしの距離の施設間の相対ずれ（|d_対象 − d_reference| / d_reference の中央値、
                     両施設で min_n 例以上の組織型）

出力: output_dir/method_compare/summary.csv

Usage:
    uv run python method_compare.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from dendrogram import group_means, pair_stats, raw_distance, ward_linkage
from harmonization import compute_metrics
from utils.loader import load_config, load_data

VARIANTS = ["original", "gan", "combat", "centroid", "affine_free", "affine_oracle"]


def shape_metrics(merged: pd.DataFrame, cfg: dict) -> dict[str, float]:
    reference: str = cfg["reference"]
    other = next(s for s in merged["source"].unique() if s != reference)
    min_n: int = cfg.get("shift_direction", {}).get("min_n", 5)
    X = np.stack(merged["embedding"].values).astype(np.float64)
    src, sub = merged["source"].values, merged["subtype"].values
    is_sft = merged["subtype"].str.contains("sft", case=False).values
    d_sft = {s: float(np.linalg.norm(X[(src == s) & is_sft].mean(axis=0) - X[(src == s) & ~is_sft].mean(axis=0)))
             for s in (reference, other)}
    subs = [t for t in np.unique(sub)
            if ((sub == t) & (src == reference)).sum() >= min_n and ((sub == t) & (src == other)).sum() >= min_n]
    mu = {(t, s): X[(sub == t) & (src == s)].mean(axis=0) for t in subs for s in (reference, other)}
    rel = [abs(np.linalg.norm(mu[(a, other)] - mu[(b, other)]) - np.linalg.norm(mu[(a, reference)] - mu[(b, reference)]))
           / np.linalg.norm(mu[(a, reference)] - mu[(b, reference)]) for a, b in combinations(subs, 2)]
    return {f"sft_men_dist_{other}": d_sft[other], f"sft_men_dist_{reference}": d_sft[reference],
            "subtype_dist_mismatch": float(np.median(rel))}


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "method_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in VARIANTS:
        merged = load_data(cfg, cfg.get("variants", {}).get(v, v))
        if merged.empty:
            print(f"  SKIP [{v}]: no data")
            continue
        m = compute_metrics(merged, cfg)
        g = group_means(merged, cfg)
        D = raw_distance(g)
        ps = pair_stats(D, ward_linkage(D), g)
        rows.append({"variant": v, **m, "adjacent_pairs": ps["adjacent_pairs"], "n_subtypes": ps["n_subtypes"],
                     "nn_any_same_other": ps["nn_any_same_other"], "n_groups": ps["n_groups"],
                     **shape_metrics(merged, cfg)})
        print(f"  [{v}] " + "  ".join(f"{k}={val:.4f}" if isinstance(val, float) else f"{k}={val}"
                                      for k, val in rows[-1].items() if k != "variant"))
    path = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved: {path}")


if __name__ == "__main__":
    main()
