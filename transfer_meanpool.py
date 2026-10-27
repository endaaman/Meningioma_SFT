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

--source virchow（エンコーダ名 meanpool_virchow。PRISM の集約器の対照: PRISM の入力と同じパッチ特徴の単純平均）:
    定義: manifest の h5_prism の virchow_2560/features（224 px / 0.5 mpp のタイルごとの 2560 次元 = Virchow v1 の
        CLS 1280 ＋ パッチトークン平均 1280 の concat、float32。キーの接頭辞は transfer.prism_key の先頭 = PRISM が読んだもの）
        のタイル方向の平均。計算は CONCH と同じ meanpool._mean_features（ROWS_PER_READ 行ずつ読んで float64 で足し、
        float32 で保存）。論文コホートにも既存の平均は無いので、全 2,301 枚を h5 から計算する。
    NFS の読みはプロセス --workers 並列（既定 4、NFS を他と分け合うので 4 まで）。済んだスライドは途中経過
        （embeddings_meanpool_virchow.partial.npy ＋ _partial.csv、1 枚ごとに書く）に残し、止めても続きから再開する
        （--reload で途中経過も捨てる）。全部済んだら下の 2 ファイルに書き、途中経過を消す。
    定義の確認: --check 枚（既定 5、両施設・全科から）について h5 の features を一度に読んで float64 で平均し直し
        （読み方が別の経路）、cos・max |差|・パッチ数（features の行数 = coordinates の行数）の一致を表示する。
    出力 (output_dir/transfer/):
        embeddings_meanpool_virchow.npy       (2301, 2560) float32、manifest の並び
        embeddings_meanpool_virchow_ids.csv   slide_id, site, n_patches, source（computed）、manifest の並び
    読む側（transfer_correction.py / shift_transfer.py の --encoder meanpool_virchow）は load_cache(cfg, "virchow")。

Usage:
    uv run python -u transfer_meanpool.py build [--reload] [--check 5]
    uv run python -u transfer_meanpool.py build --source virchow [--workers 4] [--reload] [--check 5]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from meanpool import _mean_features
from utils.loader import load_config

PAPER_FAMILIES = ("meningioma", "SFT")   # 既存の平均プーリング h5（meanpool.py）がある科
SITES = ("ebrains", "patho2")
SEED = 42
ENCODER_NAME = {"conch": "meanpool", "virchow": "meanpool_virchow"}   # --source → エンコーダ名（キャッシュのファイル名）
MAX_WORKERS = 4                          # NFS を他のジョブと分け合うので 4 まで


def _paths(cfg: dict, source: str = "conch") -> tuple[Path, Path]:
    out = Path(cfg["output_dir"]) / "transfer"
    name = ENCODER_NAME[source]
    return out / f"embeddings_{name}.npy", out / f"embeddings_{name}_ids.csv"


def load_manifest(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"].get("manifest", "transfer/manifest.csv"),
                       dtype={"slide_id": str, "patient": str})


def load_cache(cfg: dict, source: str = "conch") -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(manifest, 埋め込み (n, 768) float64（virchow は (n, 2560)）、ids)。キャッシュが無いか manifest の並びと合わなければ止まる。"""
    npy, ids_csv = _paths(cfg, source)
    tag, opt = ENCODER_NAME[source], "" if source == "conch" else f" --source {source}"
    man = load_manifest(cfg)
    if not (npy.exists() and ids_csv.exists()):
        raise SystemExit(f"[{tag}] キャッシュが無い: {npy}。先に `uv run python -u transfer_meanpool.py build{opt}` を回すこと")
    ids = pd.read_csv(ids_csv, dtype={"slide_id": str})
    if not ids[["site", "slide_id"]].equals(man[["site", "slide_id"]].reset_index(drop=True)):
        raise SystemExit(f"[{tag}] {ids_csv} の並びが manifest と合わない。`transfer_meanpool.py build{opt} --reload` で作り直すこと")
    X = np.load(npy)
    if len(X) != len(man):
        raise SystemExit(f"[{tag}] {npy} の行数 {len(X)} が manifest の {len(man)} と合わない")
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


# ── --source virchow（meanpool_virchow）─────────────────────────────────────────

def _virchow_key(cfg: dict) -> str:
    """h5_prism の中の Virchow のタイル特徴のキー（transfer.prism_key の先頭 = virchow_2560 の下の keys.tile_features）。"""
    return f"{cfg['transfer']['prism_key'].split('/')[0]}/{cfg['keys']['tile_features']}"


def _partial_paths(cfg: dict) -> tuple[Path, Path]:
    npy, _ = _paths(cfg, "virchow")
    return npy.with_name(npy.stem + ".partial.npy"), npy.with_name(npy.stem + "_partial.csv")


def _virchow_one(i: int, path: str, key: str) -> tuple[int, np.ndarray, int, int]:
    """1 枚: (manifest の行番号, 平均 float64, パッチ数, coordinates の行数)。ワーカーのプロセスで動く。"""
    with h5py.File(path, "r") as f:
        if key not in f:
            raise RuntimeError(f"'{key}' が無い: {path}")
        v, n = _mean_features(f, key)
        coords = f"{key.rsplit('/', 1)[0]}/coordinates"
        n_coords = int(f[coords].shape[0]) if coords in f else -1
    return i, v, n, n_coords


def check_virchow(cfg: dict, man: pd.DataFrame, X: np.ndarray, n_patches: np.ndarray, n_check: int) -> pd.DataFrame:
    """数枚（両施設・全科から）で、キャッシュの平均と h5 の features を一度に読んで float64 で平均し直した値を比べる。"""
    key = _virchow_key(cfg)
    rng = np.random.default_rng(SEED)
    picks = []
    for k, site in enumerate(SITES):   # ebrains を先に、端数は ebrains に
        idx = np.where((man["site"] == site).to_numpy())[0]
        n = (n_check + 1) // 2 if k == 0 else n_check // 2
        picks += sorted(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
    rows = []
    for i in picks:
        r = man.iloc[i]
        with h5py.File(r["h5_prism"], "r") as f:
            F = f[key][:]
            n_coords = int(f[f"{key.rsplit('/', 1)[0]}/coordinates"].shape[0])
        v = F.astype(np.float64).mean(axis=0)
        s = X[i].astype(np.float64)
        rows.append(dict(slide_id=r["slide_id"], site=r["site"], family=r["family"], dtype=str(F.dtype), dim=F.shape[1],
                         n_patches_stored=int(n_patches[i]), n_patches_recomputed=int(F.shape[0]), n_coordinates=n_coords,
                         cos=float(v @ s / (np.linalg.norm(v) * np.linalg.norm(s))),
                         max_abs_diff=float(np.abs(v.astype(np.float32) - X[i]).max())))
    return pd.DataFrame(rows)


def build_virchow(cfg: dict, reload: bool, n_check: int, workers: int) -> None:
    npy, ids_csv = _paths(cfg, "virchow")
    part_npy, part_csv = _partial_paths(cfg)
    man = load_manifest(cfg)
    key = _virchow_key(cfg)
    if not reload and npy.exists() and ids_csv.exists():
        _, X, ids = load_cache(cfg, "virchow")
        print(f"  cache: {npy} {X.shape}（--reload で作り直す）")
        X = X.astype(np.float32)
    else:
        if reload:
            for p in (part_npy, part_csv):
                p.unlink(missing_ok=True)
        with h5py.File(man["h5_prism"].iloc[0], "r") as f:   # 次元はファイルから
            dim = int(f[key].shape[1])
        npy.parent.mkdir(parents=True, exist_ok=True)
        if part_npy.exists() and part_csv.exists():
            P = np.lib.format.open_memmap(part_npy, mode="r+")
            if P.shape != (len(man), dim):
                raise SystemExit(f"[meanpool_virchow] 途中経過 {part_npy} の形 {P.shape} が ({len(man)}, {dim}) と違う（--reload で捨てる）")
            done = pd.read_csv(part_csv, dtype={"slide_id": str})
            ident = man[["site", "slide_id"]].reset_index(drop=True)
            bad = [int(r.idx) for r in done.itertuples(index=False)
                   if not (0 <= r.idx < len(man) and (ident.loc[r.idx, "site"], ident.loc[r.idx, "slide_id"]) == (r.site, r.slide_id))]
            if bad:
                raise SystemExit(f"[meanpool_virchow] 途中経過 {part_csv} の {len(bad)} 行が manifest と合わない（--reload で捨てる）")
            done = done.drop_duplicates("idx", keep="last")
        else:
            part_npy.unlink(missing_ok=True)
            part_csv.unlink(missing_ok=True)
            P = np.lib.format.open_memmap(part_npy, mode="w+", dtype=np.float32, shape=(len(man), dim))
            P.flush()
            done = pd.DataFrame(columns=["idx", "slide_id", "site", "n_patches", "n_coordinates"])
            done.to_csv(part_csv, index=False)
        finished = set(done["idx"].astype(int))
        todo = [i for i in range(len(man)) if i not in finished]
        print(f"  {len(man)} slides × {dim} dims ({key}): done {len(finished)}, todo {len(todo)}  workers={workers}", flush=True)
        t0, n_err = time.time(), 0
        if todo:
            with open(part_csv, "a", newline="") as fo, ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_virchow_one, i, man["h5_prism"].iloc[i], key): i for i in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    i = futs[fut]
                    r = man.iloc[i]
                    try:
                        _, v, n, n_coords = fut.result()
                    except Exception as e:  # noqa: BLE001  1 枚の失敗で止めない（次に回したときにやり直す）
                        n_err += 1
                        print(f"  ERROR {r['site']}/{r['slide_id']}: {e!r}", flush=True)
                        continue
                    P[i] = v.astype(np.float32)
                    P.flush()                # 行を書いてから済みの印を付ける（落ちても印の無い行はやり直すだけ）
                    fo.write(f"{i},{r['slide_id']},{r['site']},{n},{n_coords}\n")
                    fo.flush()
                    os.fsync(fo.fileno())
                    if k % 25 == 0 or k == len(todo):
                        el = time.time() - t0
                        print(f"  [{k}/{len(todo)}] {r['site']}/{r['family']}/{r['slide_id']}: n_patches={n} |mean|={np.linalg.norm(v):.3f}"
                              f"  ({el / 60:.1f} min, eta {el / k * (len(todo) - k) / 60:.1f} min)", flush=True)
        done = pd.read_csv(part_csv, dtype={"slide_id": str}).drop_duplicates("idx", keep="last").set_index("idx")
        if n_err or len(done) < len(man):
            raise SystemExit(f"[meanpool_virchow] 失敗 {n_err} 枚、済み {len(done)} / {len(man)}。もう一度回すと残りだけやる")
        mism = done[(done["n_coordinates"] >= 0) & (done["n_patches"] != done["n_coordinates"])]
        if len(mism):
            print(f"  WARN features と coordinates の行数が違う: {len(mism)} 枚")
        X = np.array(P, dtype=np.float32)
        del P
        ids = pd.DataFrame({"slide_id": man["slide_id"], "site": man["site"],
                            "n_patches": done.loc[range(len(man)), "n_patches"].to_numpy(dtype=np.int64), "source": "computed"})
        np.save(npy, X)
        ids.to_csv(ids_csv, index=False)
        part_npy.unlink()
        part_csv.unlink()
        print(f"  saved: {npy} {X.shape}\n  saved: {ids_csv}（途中経過は消した）")

    d = man[["family", "site"]].assign(n_patches=ids["n_patches"].to_numpy(), source=ids["source"].to_numpy(),
                                       norm=np.linalg.norm(X.astype(np.float64), axis=1))
    print("\n[counts] family × site（枚数・パッチ数の中央値・|mean| の平均・出どころ）")
    tab = d.groupby(["family", "site"]).agg(slides=("n_patches", "size"), n_patches_median=("n_patches", "median"),
                                             norm_mean=("norm", "mean"), source=("source", lambda s: "/".join(sorted(set(s)))))
    print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
    if (ids["n_patches"] <= 0).any():
        print(f"  WARN n_patches <= 0: {int((ids['n_patches'] <= 0).sum())} 枚")
    if not np.isfinite(X).all():
        print(f"  WARN 有限でない値: {int((~np.isfinite(X)).any(axis=1).sum())} 枚")

    print(f"\n[check] {n_check} 枚: キャッシュの平均 vs h5 の {key} を一度に読んで float64 で平均し直した値")
    chk = check_virchow(cfg, man, X, ids["n_patches"].to_numpy(), n_check)
    print(chk.to_string(index=False, float_format=lambda v: f"{v:.8f}" if abs(v) < 1e-3 or v > 0.99 else f"{v:.4f}"))
    if ((chk["cos"] < 0.999999).any() or (chk["n_patches_stored"] != chk["n_patches_recomputed"]).any()
            or (chk["n_patches_recomputed"] != chk["n_coordinates"]).any()):
        print("  WARN 定義が合わないスライドがある")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--source", choices=list(ENCODER_NAME), default="conch",
                    help="パッチ特徴: conch = CONCH v1.5（meanpool、既定）、virchow = Virchow v1 の concat 2560 次元（meanpool_virchow）")
    ap.add_argument("--reload", action="store_true", help="キャッシュを捨てて作り直す（virchow は途中経過も捨てる）")
    ap.add_argument("--check", type=int, default=5, help="定義の確認に使う枚数（両施設から。conch は論文コホート）")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS, help=f"virchow の NFS の読みの並列数（{MAX_WORKERS} まで）")
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    if args.source == "conch":
        print("[transfer_meanpool / build]")
        build(cfg, args.reload, args.check)
    else:
        if not 1 <= args.workers <= MAX_WORKERS:
            raise SystemExit(f"--workers は 1〜{MAX_WORKERS}（NFS を他と分け合う）")
        print("[transfer_meanpool / build / virchow]")
        build_virchow(cfg, args.reload, args.check, args.workers)
    print(f"\ndone ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
