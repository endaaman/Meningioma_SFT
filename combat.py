"""combat.py — ComBat（neuroCombat）による施設補正をスライド埋め込みにかけ、HDF5 に書き出す。

centroid.py と同じく、補正済みのスライド埋め込みを embedding.{src}.{variant}/ の HDF5
（キー {tile_model}/{keys.slide_feature}）に書く。lp.py・umap_plot.py 等は variant "combat" として読む。

- 入力: 各施設の original のスライド埋め込み（ラベルのある症例のみ）
- 補正: 次元ごとに施設の平行移動（足し算）と拡大縮小（掛け算）を見積もり、経験ベイズで安定化して除く。
  ref_batch = config の reference（その施設は不変）。共変量は既定でなし（評価施設のラベルを使わないため）。
- 共変量あり（--covariates subtype、または config の combat.covariates）: 組織型を共変量にして施設効果を見積もる
  （評価施設のラベルを使う参考用。「共変量を渡さなかったから過補正した」への反論）。出力先は
  embedding.{src}.{combat.covariate_variant}（既定 combat_cov）で、共変量なしの combat/ は触らない。
  組織型はそのまま 17 水準で渡す（希少型があっても neuroCombat は落ちない。2026-10-21 確認）。落ちる場合の
  逃げ道として combat.covariate_min_n > 0 にすると、片方の施設でその例数に満たない組織型を共変量としてだけ
  "other" にまとめる（埋め込みもラベル CSV も変えない。既定 0 = まとめない）。
- reference 施設の HDF5 も書く（値は original と一致するはず。一致を確認して表示する）。

Usage:
    uv run python combat.py                        # 共変量なし → combat/
    uv run python combat.py --covariates subtype   # 組織型を共変量 → combat_cov/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

np.int = int  # neuroCombat が numpy の旧い別名（np.int）を使うため、import より前に定義する

from neuroCombat import neuroCombat  # noqa: E402

from utils.loader import load_config, load_data  # noqa: E402


def covariate_frame(merged: pd.DataFrame, covariates: list[str], cfg: dict) -> pd.DataFrame:
    """neuroCombat に渡す共変量表（batch ＋ 共変量）。covariate_min_n > 0 なら希少な水準を "other" にまとめる。"""
    min_n: int = cfg.get("combat", {}).get("covariate_min_n", 0) or 0
    covars = pd.DataFrame({"batch": merged["source"].values})
    for col in covariates:
        values = merged[col].astype(str)
        counts = pd.crosstab(values, merged["source"])
        rare = sorted(counts.index[(counts < min_n).any(axis=1)]) if min_n > 0 else []
        print(f"  covariate {col}: {values.nunique() - len(rare) + bool(rare)} levels")
        if rare:
            print(f"  covariate {col}: {len(rare)} classes with < {min_n} slides in either site -> 'other': {rare}")
        covars[col] = values.where(~values.isin(rare), "other").values
    return covars


def run_combat(merged: pd.DataFrame, cfg: dict, covariates: list[str]) -> np.ndarray:
    X = np.stack(merged["embedding"].values).astype(np.float64)
    covars = covariate_frame(merged, covariates, cfg)
    out = neuroCombat(dat=X.T, covars=covars, batch_col="batch",
                      categorical_cols=covariates or None, ref_batch=cfg.get("reference"))
    return out["data"].T


def write_embeddings(merged: pd.DataFrame, Y: np.ndarray, cfg: dict, variant: str) -> None:
    overwrite = cfg.get("combat", {}).get("overwrite", False)
    for src in merged["source"].unique():
        h5_key = f"{cfg['embedding'][src]['tile_model']}/{cfg['keys']['slide_feature']}"
        out_dir = Path(cfg["embedding"][src][variant])
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--covariates", nargs="*", default=None,
                    help="共変量の列（例 subtype）。省略時は config の combat.covariates（既定なし）")
    ap.add_argument("--variant", default=None,
                    help="書き出す variant（embedding.*.<variant>）。既定: 共変量なし combat / あり combat.covariate_variant")
    args = ap.parse_args()
    cfg = load_config()
    c_cfg = cfg.get("combat", {})
    covariates: list[str] = args.covariates if args.covariates is not None else (c_cfg.get("covariates", []) or [])
    variant: str = args.variant or (c_cfg.get("covariate_variant", "combat_cov") if covariates else "combat")

    merged = load_data(cfg, "original").reset_index(drop=True)
    print(f"  slides: {merged['source'].value_counts().to_dict()}  covariates: {covariates}  variant: {variant}")
    Y = run_combat(merged, cfg, covariates)

    X = np.stack(merged["embedding"].values).astype(np.float64)
    ref = merged["source"].values == cfg.get("reference")
    print(f"  reference ({cfg.get('reference')}) max |combat - original| = {np.abs(Y[ref] - X[ref]).max():.2e}")
    other = ~ref
    print(f"  other: mean |shift| = {np.linalg.norm(Y[other].mean(0) - X[other].mean(0)):.4f}  "
          f"SD ratio (combat/original, median over dims) = "
          f"{np.median(Y[other].std(0, ddof=1) / X[other].std(0, ddof=1)):.3f}")
    write_embeddings(merged, Y, cfg, variant)


if __name__ == "__main__":
    main()
