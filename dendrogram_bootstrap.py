"""dendrogram_bootstrap.py — 樹形図の「隣接ペア」数の患者単位 bootstrap（Fig 6 / Table S3 の不確かさ）。

dendrogram.py の pairs.csv（施設 × 組織型の平均ベクトルを Ward 法で束ねたとき、同じ組織型の 2 施設が葉の並びで
隣り合う数 adjacent_pairs、最近傍が「同じ組織型・別施設」である群の数 nn_any_same_other）は点推定だけなので、
施設ごとに患者を復元抽出して群平均 → Ward 連結 → pair_stats をやり直し、95% 区間（2.5 / 97.5 パーセンタイル）を出す。

- 患者: EBRAINS は label.*.patient（1 患者に複数スライド）、patho2 は 1 症例 = 1 患者（lp_bootstrap._load_patients と同じ）。
- 群の作り方は dendrogram.group_means と同じ（両施設にある組織型、例数の下限なし）。抽出で片方の施設から消えた
  組織型は、その回の群から外れる（n_subtypes が減る）。ペア数は残った群で数え、n_subtypes の分布も併せて出す。
- 条件: utils.display.CONDITIONS_FIG3（補正なし / Macenko / GAN / ComBat / 平行移動）。回数・seed は config の
  dendrogram_bootstrap（既定 1000 回、seed 42）。

出力 (output_dir/dendrogram/):
    pairs_bootstrap.csv        — 条件ごとの点推定（全データ）と bootstrap の中央値・q025・q975（adjacent_pairs /
                                 nn_any_same_other / n_subtypes）、n_boot
    pairs_bootstrap_draws.csv  — 回ごとの値（condition, draw, n_subtypes, n_groups, adjacent_pairs, sister_pairs,
                                 nn_other_site_same, nn_any_same_other）

Usage:
    uv run python dendrogram_bootstrap.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dendrogram import group_means, pair_stats, raw_distance, ward_linkage
from lp_bootstrap import _load_patients
from utils.display import CONDITIONS_FIG3
from utils.loader import load_config, load_data

STATS = ["adjacent_pairs", "nn_any_same_other", "n_subtypes"]


def patient_groups(merged: pd.DataFrame, cfg: dict) -> dict[str, list[np.ndarray]]:
    """施設ごとに、患者 → その患者のスライドの行番号（merged の位置）のリスト。"""
    out: dict[str, list[np.ndarray]] = {}
    for src, g in merged.groupby("source"):
        patients = _load_patients(cfg, src, g["case_id"]).to_numpy()
        pos = pd.Series(np.arange(len(merged))[g.index.to_numpy()], index=patients)
        out[src] = [v.to_numpy() for _, v in pos.groupby(level=0)]
    return out


def resample(groups: dict[str, list[np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    """施設ごとに患者を同数だけ復元抽出して、行番号を連結する。"""
    return np.concatenate([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])
                           for g in groups.values()])


def stats_of(merged: pd.DataFrame, cfg: dict) -> dict:
    g = group_means(merged, cfg)
    D = raw_distance(g)
    return pair_stats(D, ward_linkage(D), g)


def main() -> None:
    cfg = load_config()
    b_cfg = cfg.get("dendrogram_bootstrap", {})
    n_boot, seed = int(b_cfg.get("n_boot", 1000)), int(b_cfg.get("seed", 42))
    out_dir = Path(cfg["output_dir"]) / "dendrogram"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, draws = [], []
    for v in CONDITIONS_FIG3:
        merged = load_data(cfg, cfg.get("variants", {}).get(v, v)).reset_index(drop=True)
        if merged.empty:
            print(f"  SKIP [{v}]: no data")
            continue
        point = stats_of(merged, cfg)
        groups = patient_groups(merged, cfg)
        rng = np.random.default_rng(seed)
        rows = []
        for b in range(n_boot):
            idx = resample(groups, rng)
            rows.append({"condition": v, "draw": b, **stats_of(merged.iloc[idx].reset_index(drop=True), cfg)})
        df = pd.DataFrame(rows)
        draws.append(df)
        rec = {"condition": v, "n_boot": n_boot,
               "n_patients": {s: len(g) for s, g in groups.items()}.__repr__()}
        for k in STATS:
            rec[f"{k}_point"] = point[k]
            rec[f"{k}_median"] = float(df[k].median())
            rec[f"{k}_q025"] = float(np.percentile(df[k], 2.5))
            rec[f"{k}_q975"] = float(np.percentile(df[k], 97.5))
        rec["n_subtypes_min"], rec["n_subtypes_max"] = int(df["n_subtypes"].min()), int(df["n_subtypes"].max())
        summary.append(rec)
        print(f"  [{v}] " + "  ".join(f"{k}={point[k]} (median {rec[f'{k}_median']:.0f}, "
                                      f"95% {rec[f'{k}_q025']:.0f}–{rec[f'{k}_q975']:.0f})" for k in STATS))
    pd.DataFrame(summary).to_csv(out_dir / "pairs_bootstrap.csv", index=False)
    pd.concat(draws, ignore_index=True).to_csv(out_dir / "pairs_bootstrap_draws.csv", index=False)
    print(f"  saved: {out_dir / 'pairs_bootstrap.csv'} / pairs_bootstrap_draws.csv")


if __name__ == "__main__":
    main()
