"""umap_plot.py — TITAN スライド埋め込みの UMAP 可視化。

original / centroid / gan ディレクトリからそれぞれ読み込み、
サブフォルダ別に出力する。UMAP 座標は {variant}/coords.csv に保存し、論文図（figs/）が読む。

config に held_out があれば、fit 後に held-out スライド（コホート外。EBRAINS の original、読み方は held_out_case.py と同じ）を
out-of-sample で transform し、{variant}/heldout_coords.csv（case_id, diagnosis, umap1, umap2）に書く（Fig S5）。
fit はコホートだけで行うので coords.csv は held_out の有無によらず同じ。

Usage:
    uv run python umap_plot.py
    uv run python umap_plot.py --variants original centroid   # variant を絞る
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap as umap_lib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.display import ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data


def run_umap(merged: pd.DataFrame, out_dir: Path, cfg: dict, held: pd.DataFrame | None = None) -> None:
    umap_cfg = cfg.get("umap", {})
    src_colors: dict = cfg.get("display", {}).get("colors", {}).get("sources", {})

    print("  fitting UMAP...")
    X = np.stack(merged["embedding"].values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        reducer = umap_lib.UMAP(
            n_components=2,
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            min_dist=umap_cfg.get("min_dist", 0.01),
            random_state=umap_cfg.get("random_state", 42),
            n_jobs=1,
        )
        coords = reducer.fit_transform(X)

    coords_path = out_dir / "coords.csv"
    pd.DataFrame({"case_id": merged["case_id"].values, "source": merged["source"].values,
                  "subtype": merged["subtype"].values,
                  "umap1": coords[:, 0], "umap2": coords[:, 1]}).to_csv(coords_path, index=False)
    print(f"  saved: {coords_path}")

    # held-out スライド（コホート外）を同じ UMAP に out-of-sample で写像する（fit には使わない）
    if held is not None and not held.empty:
        leak = set(merged["case_id"]) & set(held["case_id"])
        if leak:
            raise RuntimeError(f"held-out slides are in the cohort (label CSV not excluded?): {sorted(leak)}")
        H = np.stack(held["embedding"].values).astype(X.dtype)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            hc = reducer.transform(H)
        held_path = out_dir / "heldout_coords.csv"
        pd.DataFrame({"case_id": held["case_id"].values, "diagnosis": held["diagnosis"].values,
                      "umap1": hc[:, 0], "umap2": hc[:, 1]}).to_csv(held_path, index=False)
        print(f"  saved: {held_path}")

    subtypes = merged["subtype"].values
    sources  = merged["source"].values
    sub_list = ordered_subtypes(set(subtypes), cfg)
    src_list = sorted(set(sources))
    sub_map  = subtype_color_map(sub_list, cfg)
    markers = cfg.get("umap", {}).get("markers", ["o", "s", "^", "D", "v", "P", "X", "*"])
    src_marker = {s: markers[i % len(markers)] for i, s in enumerate(src_list)}

    # plot 1: color=subtype, shape=source
    fig, ax = plt.subplots(figsize=(14, 11))
    for src in src_list:
        for sub in sub_list:
            mask = (subtypes == sub) & (sources == src)
            if not np.any(mask):
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       facecolors=sub_map[sub], edgecolors="none",
                       alpha=0.7, s=80, marker=src_marker[src], zorder=2)

            # # temporary annotation
            # case_ids = merged["case_id"].values
            # for i in np.where(np.array(["sft" in s.lower() for s in subtypes]))[0]:
            #     ax.annotate(case_ids[i], (coords[i, 0], coords[i, 1]),
            #     fontsize=5, alpha=0.8, va="bottom", ha="center")

    color_handles = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in sub_list]
    shape_handles = [Line2D([0], [0], marker=src_marker[s], color="gray",
                            linestyle="None", markersize=22, label=s) for s in src_list]
    ax.legend(handles=color_handles + shape_handles, loc="best", fontsize=22, ncol=2)
    ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "umap_subtype_source.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")

    # plot 2: color=source
    fig, ax = plt.subplots(figsize=(12, 10))
    for src in src_list:
        mask = sources == src
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   facecolors=src_colors.get(src, "gray"), edgecolors="none",
                   alpha=0.6, s=80, marker="o", label=src, zorder=2)
    ax.legend(loc="best", fontsize=22)
    ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "umap_source.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


def _load_held_out(cfg: dict) -> pd.DataFrame | None:
    """config の held_out（コホート外スライド）の original 埋め込み。無ければ None。読み方は held_out_case.py を再利用。"""
    if not cfg.get("held_out"):
        return None
    from held_out_case import held_out_cases, load_held_out
    cases = held_out_cases(cfg)
    if cases.empty:
        return None
    held = load_held_out(cfg, cases, "titan")
    print(f"[held_out] {len(held)} slides: " + ", ".join(f"{r.case_id} ({r.diagnosis})" for r in held.itertuples()))
    return held


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=None, help="処理する variant（既定: config の variants 全部）")
    args = ap.parse_args()
    cfg = load_config()
    out_root = Path(cfg["output_dir"]) / "umap"
    variants: dict[str, str] = cfg.get("variants", {"original": "original"})
    if args.variants:
        unknown = [v for v in args.variants if v not in variants]
        if unknown:
            raise SystemExit(f"unknown variants (not in config.variants): {unknown}")
        variants = {v: variants[v] for v in args.variants}
    held = _load_held_out(cfg)

    for variant, dir_key in variants.items():
        print(f"\n[{variant}]")
        merged = load_data(cfg, dir_key)
        if merged.empty:
            print("  SKIP: no data")
            continue
        print(f"  {len(merged)} cases")
        out_dir = out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        run_umap(merged, out_dir, cfg, held)


if __name__ == "__main__":
    main()
