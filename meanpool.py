"""meanpool.py — 平均プーリングのスライド埋め込み（TITAN の対照）を作り、同じ解析にかける。

「スライド埋め込みの施設差は平行移動」という主張が TITAN の集約器（学習済み attention）に固有のものでは
ないことを示すための対照。同じパッチ埋め込み（CONCH v1.5、{tile_model}/features）を学習なしの単純平均で
集約したスライド埋め込み（768 次元）について、Fig 4（施設差の幾何）と Fig 3f/g・Table S3（補正法の比較）と
同じ指標を出す。Table S4・S5 の元。

手順:
    build    各施設・各 variant（original / gan / macenko のうち config にあるもの）の全 h5 から
             {tile_model}/features を読み、パッチ方向の平均（float64 で計算、float32 で保存）を
             {meanpool.root}/{variant}/{case}.h5 のキー {tile_model}/{meanpool.slide_feature} に書く
             （attrs: n_patches, source_variant）。既存ファイルは meanpool.overwrite が true のときだけ作り直す。
             続けて平均プーリングの original から centroid（centroid.py と同じ: 各施設の平均を揃える平行移動、
             両方向）と combat（combat.py と同じ: neuroCombat、ref_batch = reference、共変量なし）を派生させて
             {meanpool.root}/{centroid,combat}/ に書く（派生は軽いので常に作り直す）。affine_* は作らない。
    analyze  meanpool.variants の各条件について method_compare.py と同じ指標 → output_dir/meanpool/summary.csv。
             original については shift_direction.py と同じ計算 → shift_direction_original.csv ほか、
             original / centroid について harmonization.py の施設ラベル並べ替え検定 → site_null.csv、
             TITAN の全体シフトとの cos（同じ 768 次元だが別の空間なので弱い比較）→ shift_vs_titan.csv。
    all      build → analyze

出力 (output_dir/meanpool/):
    summary.csv                           … method_compare/summary.csv と同じ列
    shift_direction_original.csv / bio_direction_original.csv / null_shuffle_original.csv /
    subtype_distance_pairs_original.csv / subtype_distance_scaling_original.csv
                                          … shift_direction/ と同じ形式
    site_null.csv                         … harmonization/site_null.csv と同じ形式（state = original / centroid）
    shift_vs_titan.csv                    … 全体シフト（mean(reference) − mean(other)）の TITAN との cos とノルム

Usage:
    uv run python -u meanpool.py build
    uv run python -u meanpool.py analyze
    uv run python -u meanpool.py all
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from centroid import compute_all_shifts
from combat import run_combat
from dendrogram import group_means, pair_stats, raw_distance, ward_linkage
from harmonization import compute_metrics, site_null
from method_compare import shape_metrics, site_metrics
from shift_direction import compute as compute_shift_direction, compute_scaling
from utils.loader import load_config, load_data

BUILD_VARIANTS = ["original", "gan", "macenko"]   # パッチ埋め込みから平均を取る variant（config にあるものだけ）
NULL_STATES = ["original", "centroid"]            # site_null を回す条件
ROWS_PER_READ = 8192                               # features を一度に読む行数（NFS 越しの読みを大きな塊にする）


def _root(cfg: dict, site: str) -> Path:
    return Path(cfg["meanpool"]["root"].format(site=site))


def _slide_key(cfg: dict, site: str) -> str:
    return f"{cfg['embedding'][site]['tile_model']}/{cfg['meanpool']['slide_feature']}"


def _write_slide(out_path: Path, key: str, vec: np.ndarray, attrs: dict) -> None:
    """1 スライドの埋め込みを .tmp に書いてから rename する（途中で落ちても壊れたファイルを残さない）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        d = f.create_dataset(key, data=vec.astype(np.float32))
        for k, v in attrs.items():
            d.attrs[k] = v
    os.replace(tmp, out_path)


def _mean_features(f: h5py.File, key: str) -> tuple[np.ndarray, int]:
    """パッチ埋め込み (n, d) の平均を float64 で返す（大きな h5 でも全体をメモリに載せない）。"""
    ds = f[key]
    n = ds.shape[0]
    acc = np.zeros(ds.shape[1], dtype=np.float64)
    for s in range(0, n, ROWS_PER_READ):
        acc += ds[s:s + ROWS_PER_READ].astype(np.float64).sum(axis=0)
    return acc / n, n


# ── build ───────────────────────────────────────────────────────────────────────

def build_means(cfg: dict) -> None:
    overwrite: bool = cfg["meanpool"].get("overwrite", False)
    for site, ds_cfg in cfg["embedding"].items():
        feat_key = f"{ds_cfg['tile_model']}/{cfg['keys']['tile_features']}"
        slide_key = _slide_key(cfg, site)
        for variant in BUILD_VARIANTS:
            if variant not in ds_cfg:
                continue
            src_dir, dst_dir = Path(ds_cfg[variant]), _root(cfg, site) / variant
            files = sorted(src_dir.glob("*.h5"))
            todo = [p for p in files if overwrite or not (dst_dir / p.name).exists()]
            print(f"\n[build] {site}/{variant}: {len(files)} files, todo {len(todo)}  {src_dir} -> {dst_dir}", flush=True)
            n_done, n_missing, t0 = 0, 0, time.time()
            for p in todo:
                with h5py.File(p, "r") as f:
                    if feat_key not in f:
                        n_missing += 1
                        print(f"  WARN {p.name}: no '{feat_key}' — skip", flush=True)
                        continue
                    vec, n = _mean_features(f, feat_key)
                _write_slide(dst_dir / p.name, slide_key, vec, {"n_patches": n, "source_variant": variant})
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo) - n_missing:
                    el = time.time() - t0
                    eta = el / n_done * (len(todo) - n_missing - n_done)
                    print(f"  [{n_done}/{len(todo)}] {p.name}: n_patches={n} |mean|={np.linalg.norm(vec):.3f}"
                          f"  ({el / 60:.1f} min, eta {eta / 60:.0f} min)", flush=True)
            n_out = len(list(dst_dir.glob("*.h5"))) if dst_dir.exists() else 0
            print(f"  written={n_done} skipped={len(files) - len(todo)} missing={n_missing}  total in dir={n_out}", flush=True)


def build_centroid(cfg: dict) -> None:
    """centroid.py と同じ: shifts[src] = mean(other) − mean(src)（ラベルのある症例の平均）、original の全 h5 に足す。"""
    merged = load_data(cfg, "original", aggregator="mean")
    print(f"\n[centroid] {merged['source'].value_counts().to_dict()}")
    shifts = compute_all_shifts(merged)
    for src, shift in shifts.items():
        key = _slide_key(cfg, src)
        raw_dir, out_dir = _root(cfg, src) / "original", _root(cfg, src) / "centroid"
        n = 0
        for p in sorted(raw_dir.glob("*.h5")):
            with h5py.File(p, "r") as f:
                emb, n_patches = f[key][:], int(f[key].attrs.get("n_patches", 0))
            _write_slide(out_dir / p.name, key, emb + shift, {"n_patches": n_patches, "source_variant": "original"})
            n += 1
        print(f"  [{src}] written={n} -> {out_dir}")


def build_combat(cfg: dict) -> None:
    """combat.py と同じ: neuroCombat（ref_batch = reference、combat.covariates）。ラベルのある症例だけ書く。"""
    merged = load_data(cfg, "original", aggregator="mean").reset_index(drop=True)
    print(f"\n[combat] {merged['source'].value_counts().to_dict()}  covariates: {cfg.get('combat', {}).get('covariates', [])}")
    Y = run_combat(merged, cfg, [])   # 共変量なし（combat.py の既定と同じ）
    X = np.stack(merged["embedding"].values).astype(np.float64)
    ref = merged["source"].values == cfg.get("reference")
    print(f"  reference ({cfg.get('reference')}) max |combat - original| = {np.abs(Y[ref] - X[ref]).max():.2e}")
    other = ~ref
    print(f"  other: mean |shift| = {np.linalg.norm(Y[other].mean(0) - X[other].mean(0)):.4f}  "
          f"SD ratio (combat/original, median over dims) = "
          f"{np.median(Y[other].std(0, ddof=1) / X[other].std(0, ddof=1)):.3f}")
    for src in merged["source"].unique():
        key = _slide_key(cfg, src)
        out_dir = _root(cfg, src) / "combat"
        idx = np.where(merged["source"].values == src)[0]
        for i in idx:
            _write_slide(out_dir / f"{merged['case_id'].iloc[i]}.h5", key, Y[i], {"source_variant": "original"})
        print(f"  [{src}] written={len(idx)} -> {out_dir}")


def build(cfg: dict) -> None:
    build_means(cfg)
    build_centroid(cfg)
    build_combat(cfg)


# ── analyze ─────────────────────────────────────────────────────────────────────

def _load(cfg: dict, variant: str) -> pd.DataFrame:
    return load_data(cfg, cfg.get("variants", {}).get(variant, variant), aggregator="mean")


def analyze_summary(cfg: dict, out_dir: Path) -> None:
    rows = []
    for v in cfg["meanpool"].get("variants", ["original", "centroid"]):
        merged = _load(cfg, v)
        if merged.empty:
            print(f"  SKIP [{v}]: no data")
            continue
        m = compute_metrics(merged, cfg)
        g = group_means(merged, cfg)
        D = raw_distance(g)
        ps = pair_stats(D, ward_linkage(D), g)
        rows.append({"variant": v, **m, "adjacent_pairs": ps["adjacent_pairs"], "n_subtypes": ps["n_subtypes"],
                     "nn_any_same_other": ps["nn_any_same_other"], "n_groups": ps["n_groups"],
                     **shape_metrics(merged, cfg), **site_metrics(merged, cfg)})
        print(f"  [{v}] " + "  ".join(f"{k}={val:.4f}" if isinstance(val, float) else f"{k}={val}"
                                      for k, val in rows[-1].items() if k != "variant"))
    path = out_dir / "summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved: {path}")


def analyze_shift_direction(cfg: dict, out_dir: Path) -> None:
    merged = _load(cfg, "original")
    table, bio_table, null_table = compute_shift_direction(merged, cfg)
    pairs, scaling = compute_scaling(merged, cfg)
    for name, df in (("shift_direction", table), ("bio_direction", bio_table), ("null_shuffle", null_table),
                     ("subtype_distance_pairs", pairs), ("subtype_distance_scaling", scaling)):
        path = out_dir / f"{name}_original.csv"
        df.to_csv(path, index=False)
        print(f"  saved: {path}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
        print(bio_table.round(3).to_string(index=False))
        print(scaling.round(3).to_string(index=False))


def analyze_site_null(cfg: dict, out_dir: Path) -> None:
    n_cfg = cfg.get("harmonization", {}).get("site_null", {})
    rng = np.random.default_rng(n_cfg.get("seed", 42))
    rows = []
    for state in NULL_STATES:
        merged = _load(cfg, state)
        if merged.empty:
            continue
        t0 = time.time()
        for r in site_null(merged, cfg, rng):
            rows.append({"state": state, **r})
            print(f"  [{state}/{r['null']}] {r['metric']}: obs={r['obs']:.4f} "
                  f"null={r['null_mean']:.4f} [{r['null_q025']:.4f}, {r['null_q975']:.4f}] p={r['p']:.4f}")
        print(f"  [{state}] {(time.time() - t0) / 60:.1f} min", flush=True)
    path = out_dir / "site_null.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  saved: {path}")


def analyze_vs_titan(cfg: dict, out_dir: Path) -> None:
    """全体シフト mean(reference) − mean(other) の、平均プーリングと TITAN の cos。

    2 つの埋め込みは別の空間（TITAN は学習済み集約器の出力）なので、次元が同じでも向きの比較は弱い参考値。
    次元が違えば cos は NaN にする。ノルムの尺度も併記する。
    """
    reference: str = cfg["reference"]
    rows = []
    vec = {}
    for agg in ("mean", "titan"):
        merged = load_data(cfg, "original", aggregator=agg)
        X = np.stack(merged["embedding"].values).astype(np.float64)
        is_ref = (merged["source"] == reference).values
        other = next(s for s in merged["source"].unique() if s != reference)
        vec[agg] = X[is_ref].mean(axis=0) - X[~is_ref].mean(axis=0)
        rows.append({"aggregator": agg, "dim": X.shape[1], "n": len(merged),
                     "shift_norm": float(np.linalg.norm(vec[agg])),
                     "emb_norm_mean": float(np.linalg.norm(X, axis=1).mean()),
                     "emb_norm_sd": float(np.linalg.norm(X, axis=1).std(ddof=1)),
                     f"within_site_sd_{reference}": float(np.sqrt(((X[is_ref] - X[is_ref].mean(0)) ** 2).sum(1).mean())),
                     f"within_site_sd_{other}": float(np.sqrt(((X[~is_ref] - X[~is_ref].mean(0)) ** 2).sum(1).mean()))})
    same_dim = vec["mean"].shape == vec["titan"].shape
    cos = float(vec["mean"] @ vec["titan"] / (np.linalg.norm(vec["mean"]) * np.linalg.norm(vec["titan"]))) if same_dim else np.nan
    table = pd.DataFrame(rows)
    table["cos_shift_mean_vs_titan"] = cos
    table["note"] = ("weak comparison: different embedding spaces, same dimension" if same_dim
                     else "dimension differs: cos not defined")
    path = out_dir / "shift_vs_titan.csv"
    table.to_csv(path, index=False)
    print(table.round(4).to_string(index=False))
    print(f"  saved: {path}")


def analyze(cfg: dict, out_dir: Path) -> None:
    print("\n[analyze] summary (method_compare と同じ指標)")
    analyze_summary(cfg, out_dir)
    print("\n[analyze] shift_direction (original)")
    analyze_shift_direction(cfg, out_dir)
    print("\n[analyze] shift vs TITAN (original)")
    analyze_vs_titan(cfg, out_dir)
    print("\n[analyze] site-label permutation null (original / centroid)")
    analyze_site_null(cfg, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "analyze", "all"])
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "meanpool"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.cmd in ("build", "all"):
        build(cfg)
    if args.cmd in ("analyze", "all"):
        analyze(cfg, out_dir)
    print(f"\ndone ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
