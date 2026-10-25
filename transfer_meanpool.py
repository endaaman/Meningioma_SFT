"""transfer_meanpool.py — 転移テストの全スライド（out/transfer/manifest.csv、2,301 枚）の平均プーリング埋め込み（学習なしの集約）。

目的: 腫瘍の科をまたぐ解析（shift_transfer.py の共通の平行移動、transfer_correction.py の「平行移動は移り倍率は移らない」、
dispersion_ratio.py の「ばらつきの比は腫瘍に固有」）が、TITAN の学習済み集約器に固有のものでないかを見る。論文コホートの対照
（meanpool.py、Table S4・S5）と同じ定義を、グリオーマ側のスライドにも広げる。

定義（meanpool.py と同じ）: CONCH v1.5 のパッチ埋め込み {tile_model}/features（768 次元）のパッチ方向の平均
    （float64 で計算、float32 で保存。計算は meanpool._mean_features をそのまま使う）。
    論文コホート（科 meningioma / SFT）: meanpool.py が書いた既存の {meanpool.root}/original/<slide_id>.h5 の
        {tile_model}/{meanpool.slide_feature} を読む（計算し直さない。n_patches は attrs から）。
        定義が合っているかは --check 枚（既定 5、両施設から）について h5_titan（論文の original h5）の features から
        平均を取り直し、cos・max |差|・パッチ数の一致を表示する。
    グリオーマ側（科 glioma / lymphoma / metastasis）: h5_titan（EBRAINS_embeddings / Patho2Gliomas_embeddings の h5）の
        {tile_model}/features から平均を取る。tile_model は ebrains_embeddings.tile_model / patho2_gliomas.tile_model。
        NFS の読みは 1 枚ずつ（並列にしない）。

出力 (output_dir/transfer/):
    embeddings_meanpool.npy       (2301, 768) float32、manifest の並び
    embeddings_meanpool_ids.csv   slide_id, site, n_patches, source（existing_h5 | computed）、manifest の並び
    既にあれば作り直さない（--reload で作り直す）。

読む側（transfer_correction.py / shift_transfer.py の --encoder meanpool）は load_cache() を使う。

Usage:
    uv run python -u transfer_meanpool.py build [--reload] [--check 5]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from meanpool import _mean_features
from utils.loader import load_config

PAPER_FAMILIES = ("meningioma", "SFT")   # 既存の平均プーリング h5（meanpool.py）がある科
SITES = ("ebrains", "patho2")
SEED = 42


def _paths(cfg: dict) -> tuple[Path, Path]:
    out = Path(cfg["output_dir"]) / "transfer"
    return out / "embeddings_meanpool.npy", out / "embeddings_meanpool_ids.csv"


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"].get("manifest", "transfer/manifest.csv"),
                       dtype={"slide_id": str, "patient": str})


def load_cache(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(manifest, 埋め込み (n, 768) float64、ids)。キャッシュが無いか manifest の並びと合わなければ止まる。"""
    npy, ids_csv = _paths(cfg)
    man = load_manifest(cfg)
    if not (npy.exists() and ids_csv.exists()):
        raise SystemExit(f"[meanpool] キャッシュが無い: {npy}。先に `uv run python -u transfer_meanpool.py build` を回すこと")
    ids = pd.read_csv(ids_csv, dtype={"slide_id": str})
    if not ids[["site", "slide_id"]].equals(man[["site", "slide_id"]].reset_index(drop=True)):
        raise SystemExit(f"[meanpool] {ids_csv} の並びが manifest と合わない。`transfer_meanpool.py build --reload` で作り直すこと")
    X = np.load(npy)
    if len(X) != len(man):
        raise SystemExit(f"[meanpool] {npy} の行数 {len(X)} が manifest の {len(man)} と合わない")
    return man, X.astype(np.float64), ids


def _paper_h5(cfg: dict, site: str, slide_id: str) -> tuple[Path, str]:
    """論文コホートの既存の平均プーリング h5 とキー（meanpool.py の _root / _slide_key と同じ）。"""
    root = Path(cfg["meanpool"]["root"].format(site=site)) / "original"
    return root / f"{slide_id}.h5", f"{cfg['embedding'][site]['tile_model']}/{cfg['meanpool']['slide_feature']}"


def _feature_key(cfg: dict, family: str, site: str) -> str:
    """h5_titan の中のパッチ埋め込みのキー。"""
    if family in PAPER_FAMILIES:
        tm = cfg["embedding"][site]["tile_model"]
    else:
        tm = cfg["ebrains_embeddings"]["tile_model"] if site == "ebrains" else cfg["patho2_gliomas"]["tile_model"]
    return f"{tm}/{cfg['keys']['tile_features']}"


def _compute(path: str, key: str) -> tuple[np.ndarray, int]:
    with h5py.File(path, "r") as f:
        if key not in f:
            raise SystemExit(f"[meanpool] '{key}' が無い: {path}")
        return _mean_features(f, key)


def check_definition(cfg: dict, man: pd.DataFrame, X: np.ndarray, n_patches: np.ndarray, n_check: int) -> pd.DataFrame:
    """論文コホートの数枚で、既存の平均（meanpool.py）と h5_titan の features から取り直した平均を比べる。"""
    rng = np.random.default_rng(SEED)
    picks = []
    for k, site in enumerate(SITES):   # 両施設から（ebrains を先に、端数は ebrains に）
        idx = np.where(man["family"].isin(PAPER_FAMILIES).to_numpy() & (man["site"] == site).to_numpy())[0]
        n = (n_check + 1) // 2 if k == 0 else n_check // 2
        picks += sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    rows = []
    for i in picks:
        r = man.iloc[i]
        v, n = _compute(r["h5_titan"], _feature_key(cfg, r["family"], r["site"]))
        s = X[i].astype(np.float64)
        rows.append(dict(slide_id=r["slide_id"], site=r["site"], family=r["family"], n_patches_stored=int(n_patches[i]),
                         n_patches_recomputed=int(n), cos=float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s))),
                         max_abs_diff=float(np.abs(v.astype(np.float32) - X[i]).max())))
    return pd.DataFrame(rows)


def build(cfg: dict, reload: bool, n_check: int) -> None:
    npy, ids_csv = _paths(cfg)
    man = load_manifest(cfg)
    if not reload and npy.exists() and ids_csv.exists():
        _, X, ids = load_cache(cfg)
        print(f"  cache: {npy} {X.shape}（--reload で作り直す）")
        X = X.astype(np.float32)
    else:
        X = np.zeros((len(man), 768), dtype=np.float32)
        n_patches = np.zeros(len(man), dtype=np.int64)
        source = np.empty(len(man), dtype=object)
        todo = [i for i, f in enumerate(man["family"]) if f not in PAPER_FAMILIES]
        print(f"  {len(man)} slides: existing_h5 {len(man) - len(todo)}, computed {len(todo)}")
        for i, r in enumerate(man.itertuples(index=False)):   # 論文コホート: 既存の h5（Sync 側、軽い）
            if r.family not in PAPER_FAMILIES:
                continue
            p, key = _paper_h5(cfg, r.site, r.slide_id)
            if not p.exists():
                raise SystemExit(f"[meanpool] 既存の平均プーリング h5 が無い: {p}（meanpool.py build）")
            with h5py.File(p, "r") as f:
                X[i] = f[key][:]
                n_patches[i] = int(f[key].attrs.get("n_patches", 0))
            source[i] = "existing_h5"
        t0 = time.time()
        for k, i in enumerate(todo, 1):   # グリオーマ側: h5_titan の features の平均（NFS、1 枚ずつ）
            r = man.iloc[i]
            v, n = _compute(r["h5_titan"], _feature_key(cfg, r["family"], r["site"]))
            X[i], n_patches[i], source[i] = v.astype(np.float32), n, "computed"
            if k % 50 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  [{k}/{len(todo)}] {r['site']}/{r['family']}/{r['slide_id']}: n_patches={n} |mean|={np.linalg.norm(v):.3f}"
                      f"  ({el / 60:.1f} min, eta {el / k * (len(todo) - k) / 60:.1f} min)", flush=True)
        ids = pd.DataFrame({"slide_id": man["slide_id"], "site": man["site"], "n_patches": n_patches, "source": source})
        npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy, X)
        ids.to_csv(ids_csv, index=False)
        print(f"  saved: {npy} {X.shape}\n  saved: {ids_csv}")

    d = man[["family", "site"]].assign(n_patches=ids["n_patches"].to_numpy(), source=ids["source"].to_numpy(),
                                       norm=np.linalg.norm(X.astype(np.float64), axis=1))
    print("\n[counts] family × site（枚数・パッチ数の中央値・|mean| の平均・出どころ）")
    tab = d.groupby(["family", "site"]).agg(slides=("n_patches", "size"), n_patches_median=("n_patches", "median"),
                                             norm_mean=("norm", "mean"), source=("source", lambda s: "/".join(sorted(set(s)))))
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    if (ids["n_patches"] <= 0).any():
        print(f"  WARN n_patches <= 0: {int((ids['n_patches'] <= 0).sum())} 枚")

    print(f"\n[check] 論文コホートの {n_check} 枚: 既存の平均（meanpool.py）vs h5_titan の features から取り直した平均")
    chk = check_definition(cfg, man, X, ids["n_patches"].to_numpy(), n_check)
    print(chk.to_string(index=False, float_format=lambda v: f"{v:.8f}" if abs(v) < 1e-3 or v > 0.99 else f"{v:.4f}"))
    if (chk["cos"] < 0.999999).any() or (chk["n_patches_stored"] != chk["n_patches_recomputed"]).any():
        print("  WARN 定義が合わないスライドがある")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--reload", action="store_true", help="キャッシュを捨てて作り直す")
    ap.add_argument("--check", type=int, default=5, help="定義の確認に使う論文コホートの枚数（両施設から）")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    print("[transfer_meanpool / build]")
    build(cfg, args.reload, args.check)
    print(f"\ndone ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
