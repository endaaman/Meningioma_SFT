"""combat.py — ComBat（neuroCombat）による施設補正をスライド埋め込みにかけ、HDF5 に書き出す。

centroid.py と同じく、補正済みのスライド埋め込みを embedding.{src}.combat/ の HDF5
（キー {tile_model}/{keys.slide_feature}）に書く。lp.py・umap_plot.py 等は variant "combat" として読む。

- 入力: 各施設の original のスライド埋め込み（ラベルのある症例のみ）
- 補正: 次元ごとに施設の平行移動（足し算）と拡大縮小（掛け算）を見積もり、経験ベイズで安定化して除く。
  ref_batch = config の reference（その施設は不変）。共変量は既定でなし（評価施設のラベルを使わないため）。
  config の combat.covariates に subtype を入れると組織型を共変量にする（ラベルを使う参考用）。
- reference 施設の HDF5 も書く（値は original と一致するはず。一致を確認して表示する）。

Usage:
    uv run python combat.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

np.int = int  # neuroCombat が numpy の旧い別名（np.int）を使うため、import より前に定義する

from neuroCombat import neuroCombat  # noqa: E402

from utils.loader import load_config, load_data  # noqa: E402


def run_combat(merged: pd.DataFrame, cfg: dict) -> np.ndarray:
    c_cfg = cfg.get("combat", {})
    covariates: list[str] = c_cfg.get("covariates", []) or []
    X = np.stack(merged["embedding"].values).astype(np.float64)
    covars = pd.DataFrame({"batch": merged["source"].values})
    for col in covariates:
        covars[col] = merged[col].values
    out = neuroCombat(dat=X.T, covars=covars, batch_col="batch",
                      categorical_cols=covariates or None, ref_batch=cfg.get("reference"))
    return out["data"].T


def write_embeddings(merged: pd.DataFrame, Y: np.ndarray, cfg: dict) -> None:
    overwrite = cfg.get("combat", {}).get("overwrite", False)
    for src in merged["source"].unique():
        h5_key = f"{cfg['embedding'][src]['tile_model']}/{cfg['keys']['slide_feature']}"
        out_dir = Path(cfg["embedding"][src]["combat"])
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = np.where(merged["source"].values == src)[0]
        n_written = n_skipped = 0
        for i in idx:
            out_path = out_dir / f"{merged['case_id'].iloc[i]}.h5"
            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue
            with h5py.File(out_path, "w") as f:
                f.create_dataset(h5_key, data=Y[i].astype(np.float32))
            n_written += 1
        print(f"  [{src}] written={n_written}  skipped={n_skipped}  -> {out_dir}")


def main() -> None:
    cfg = load_config()
    merged = load_data(cfg, "original").reset_index(drop=True)
    print(f"  slides: {merged['source'].value_counts().to_dict()}  covariates: {cfg.get('combat', {}).get('covariates', [])}")
    Y = run_combat(merged, cfg)

    X = np.stack(merged["embedding"].values).astype(np.float64)
    ref = merged["source"].values == cfg.get("reference")
    print(f"  reference ({cfg.get('reference')}) max |combat - original| = {np.abs(Y[ref] - X[ref]).max():.2e}")
    other = ~ref
    print(f"  other: mean |shift| = {np.linalg.norm(Y[other].mean(0) - X[other].mean(0)):.4f}  "
          f"SD ratio (combat/original, median over dims) = "
          f"{np.median(Y[other].std(0, ddof=1) / X[other].std(0, ddof=1)):.3f}")
    write_embeddings(merged, Y, cfg)


if __name__ == "__main__":
    main()
