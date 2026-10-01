"""combat_decompose.py — ComBat の補正を「平行移動の成分」と「拡大縮小の成分」に分け、centroid と比べる。

ComBat（combat.py、共変量なし・reference 施設は不変）は、次元ごとに施設の足し算（平行移動）と掛け算（拡大縮小）を
除く。平均の移動が centroid と同じなら、ComBat は「centroid ＋ 次元ごとの拡大縮小」とみなせる（アブレーション）。

出力: output_dir/combat/decomposition.csv（metric, value の縦長）
    shift_norm_centroid / shift_norm_combat / shift_cos  … 補正される施設の平均の移動（大きさと向きの一致）
    scale_ratio_{median,q05,q95} / scale_gt1_frac         … 次元ごとの SD 比（ComBat 後 / 原）
    move_median / scale_part_median / scale_part_ratio     … スライドごとの移動量と、そのうち centroid との差（拡大縮小の分）
    sft_men_dist_{site}_{original,centroid,combat}        … 施設内の SFT − 髄膜腫 平均ベクトル間距離
    ref_max_abs_change                                     … reference 施設の値の変化（0 のはず）

Usage:
    uv run python combat_decompose.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.loader import load_config, load_data


def _load(cfg: dict, variant: str) -> pd.DataFrame:
    df = load_data(cfg, cfg["variants"].get(variant, variant))
    return df.sort_values(["source", "case_id"]).reset_index(drop=True)


def decompose(cfg: dict) -> dict[str, float]:
    ref = cfg["reference"]
    o, c = _load(cfg, "original"), _load(cfg, "combat")
    assert (o["case_id"].values == c["case_id"].values).all(), "original と combat の症例が一致しない"
    X = np.stack(o["embedding"].values).astype(np.float64)
    Y = np.stack(c["embedding"].values).astype(np.float64)
    e = (o["source"] == ref).values
    p = ~e
    sft = o["subtype"].str.contains("SFT", case=False).values

    d_cen = X[e].mean(0) - X[p].mean(0)          # centroid の平行移動
    d_cb = Y[p].mean(0) - X[p].mean(0)           # ComBat による平均の移動
    ratio = Y[p].std(0, ddof=1) / X[p].std(0, ddof=1)
    Zc = X[p] + d_cen                            # centroid 補正後
    move = np.linalg.norm(Y[p] - X[p], axis=1)
    scale_part = np.linalg.norm(Y[p] - Zc, axis=1)

    out = {
        "ref_max_abs_change": float(np.abs(Y[e] - X[e]).max()),
        "shift_norm_centroid": float(np.linalg.norm(d_cen)),
        "shift_norm_combat": float(np.linalg.norm(d_cb)),
        "shift_cos": float(d_cen @ d_cb / np.linalg.norm(d_cen) / np.linalg.norm(d_cb)),
        "scale_ratio_median": float(np.median(ratio)),
        "scale_ratio_q05": float(np.quantile(ratio, 0.05)),
        "scale_ratio_q95": float(np.quantile(ratio, 0.95)),
        "scale_gt1_frac": float((ratio > 1).mean()),
        "move_median": float(np.median(move)),
        "scale_part_median": float(np.median(scale_part)),
        "scale_part_ratio": float(np.median(scale_part / move)),
    }
    target = next(s for s in o["source"].unique() if s != ref)
    for site, mask in ((ref, e), (target, p)):
        variants = {"original": X[mask], "centroid": X[mask] + (d_cen if site == target else 0), "combat": Y[mask]}
        s = sft[mask]
        for name, Z in variants.items():
            out[f"sft_men_dist_{site}_{name}"] = float(np.linalg.norm(Z[s].mean(0) - Z[~s].mean(0)))
    return out


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "combat"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = decompose(cfg)
    df = pd.DataFrame({"metric": list(res), "value": list(res.values())})
    df.to_csv(out_dir / "decomposition.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  saved: {out_dir / 'decomposition.csv'}")


if __name__ == "__main__":
    main()
