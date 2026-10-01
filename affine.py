"""affine.py — 係数 1 つのアフィン補正（平行移動 ＋ 全体の一様な拡大縮小）をスライド埋め込みにかけ、HDF5 に書き出す。

reference 以外の施設の埋め込みを x' = (x − μ_site)·s + μ_ref に移す（reference は不変）。
centroid は s = 1 の場合に当たる。s の決め方で 2 つの variant を作る:

- affine_free  : s = reference と対象施設の、施設平均まわりの RMS 距離の比（ラベル不要）。
- affine_oracle: s = 組織型内の RMS 距離の比（両施設で min_n 例以上の組織型について、各組織型の平均まわりの
                 二乗距離を施設ごとに平均し、その比の平方根）。組織型ラベル（評価施設のものを含む）を使って推定するので、
                 組織型分類の評価に使うと循環になる。「ラベルを使った場合の上限の参考」としてのみ使う。

出力:
    embedding.{src}.affine_free/ , embedding.{src}.affine_oracle/ の HDF5（キー {tile_model}/{keys.slide_feature}）
    {output_dir}/affine/scale.csv — variant ごとの s と推定方法

Usage:
    uv run python affine.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from utils.loader import load_config, load_data


def scale_free(X: np.ndarray, ref: np.ndarray, oth: np.ndarray) -> float:
    rms = lambda Z: np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())  # noqa: E731
    return float(rms(X[ref]) / rms(X[oth]))


def scale_oracle(X: np.ndarray, src: np.ndarray, sub: np.ndarray, ref_name: str, oth_name: str, min_n: int) -> tuple[float, list[str]]:
    subs = [s for s in np.unique(sub)
            if ((sub == s) & (src == ref_name)).sum() >= min_n and ((sub == s) & (src == oth_name)).sum() >= min_n]

    def within_ms(site: str) -> float:
        return float(np.mean([((X[(sub == s) & (src == site)] - X[(sub == s) & (src == site)].mean(0)) ** 2).sum(1).mean()
                              for s in subs]))

    return float(np.sqrt(within_ms(ref_name) / within_ms(oth_name))), subs


def write_variant(merged: pd.DataFrame, Y: np.ndarray, variant: str, cfg: dict) -> None:
    overwrite = cfg.get("affine", {}).get("overwrite", True)
    for src in merged["source"].unique():
        h5_key = f"{cfg['embedding'][src]['tile_model']}/{cfg['keys']['slide_feature']}"
        out_dir = Path(cfg["embedding"][src][variant])
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i in np.where(merged["source"].values == src)[0]:
            path = out_dir / f"{merged['case_id'].iloc[i]}.h5"
            if path.exists() and not overwrite:
                continue
            with h5py.File(path, "w") as f:
                f.create_dataset(h5_key, data=Y[i].astype(np.float32))
            n += 1
        print(f"  [{variant}/{src}] written={n} -> {out_dir}")


def main() -> None:
    cfg = load_config()
    a_cfg = cfg.get("affine", {})
    min_n = a_cfg.get("min_n", 5)
    merged = load_data(cfg, "original").reset_index(drop=True)
    X = np.stack(merged["embedding"].values).astype(np.float64)
    src = merged["source"].values
    sub = merged["subtype"].values
    ref_name = cfg["reference"]
    others = [s for s in pd.unique(src) if s != ref_name]
    assert len(others) == 1, "施設は 2 つを前提とする"
    oth_name = others[0]
    ref, oth = src == ref_name, src == oth_name
    mu_ref, mu_oth = X[ref].mean(0), X[oth].mean(0)

    s_free = scale_free(X, ref, oth)
    s_oracle, subs = scale_oracle(X, src, sub, ref_name, oth_name, min_n)
    scales = {"affine_free": (s_free, "施設平均まわりの RMS 距離の比（reference / 対象）、ラベル不要"),
              "affine_oracle": (s_oracle, f"組織型内 RMS 距離の比（reference / 対象）、両施設で {min_n} 例以上の {len(subs)} 組織型。"
                                          "組織型ラベルを使う（組織型分類の評価では循環、上限の参考）")}
    rows = []
    for variant, (s, how) in scales.items():
        Y = X.copy()
        Y[oth] = (X[oth] - mu_oth) * s + mu_ref
        print(f"  {variant}: s = {s:.4f}  ({how})")
        write_variant(merged, Y, variant, cfg)
        rows.append({"variant": variant, "scale": s, "method": how, "reference": ref_name, "target": oth_name})
    out_dir = Path(cfg["output_dir"]) / "affine"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "scale.csv", index=False)
    print(f"  saved: {out_dir / 'scale.csv'}")


if __name__ == "__main__":
    main()
