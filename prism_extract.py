"""prism_extract.py — 転移テストの全スライド（out/transfer/manifest.csv）に Virchow → PRISM の埋め込みを足す。

第 2 のスライドエンコーダ（TITAN と系統の違う PRISM）で論文の幾何と転移テストを再現するためのデータ作り。
    タイル: Virchow v1（paige-ai/Virchow、ViT-H/14、224 px / 0.5 mpp）、出力 concat = CLS ＋ パッチ平均（2560 次元）
            → h5 の virchow_2560/{features, coordinates}（attrs preset='virchow', output='concat', patch_size=224）
    スライド: PRISM（paige-ai/Prism の Perceiver、座標は使わない）→ virchow_2560/aggregates/prism/feature（1280 次元）
計算は wsi-toolbox の `wt extract --preset virchow --outputs concat -S 224 --mpp 0.5` と `wt aggregate --slide-preset prism` と同じ
（FeatureExtractionCommand と presets.slide.aggregate_slide を直接呼ぶ。PRISM のモデルは 1 回だけ読む）。

書き先は manifest の h5_prism = **既存の h5**（ken 決定 2026-10-26）: EBRAINS → EBRAINS_embeddings の h5、院内グリオーマ →
Patho2Gliomas_embeddings の h5、院内の論文コホート → 論文の original h5。既存のグループ（conch15 / conch15_768 / TITAN）は触らず
virchow_2560 を足すだけ（h5 のルート attrs（cols / rows / patch_count / mpp）は最初の抽出のまま。各グループの attrs に 224 px の格子が付く）。
PRISM が既にあるスライドはスキップ、virchow_2560/features だけあるスライドは集約だけ。途中で止めても再実行で続きから進む。

順番: EBRAINS（論文コホート → グリオーマ → DLBCL → 転移）→ 院内グリオーマ・DLBCL・転移 → 院内の論文コホート。
      --sites / --families で絞れる（院内の論文コホートは ken の確認後に回す）。

ログ: output_dir/transfer/prism_log.csv（time, status, seconds, n_tiles, site, family, slide_id, h5）。

Usage:
    uv run python -u prism_extract.py --dry-run                 # 対象と状態の集計だけ
    uv run python -u prism_extract.py --sites ebrains --limit 2  # 試し
    uv run python -u prism_extract.py --sites ebrains            # EBRAINS 全部（約 1,357 枚、1〜3 分/枚）
    uv run python -u prism_extract.py                            # 全部
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from utils.loader import load_config

TILE_PRESET = "virchow"
TILE_OUTPUT = "concat"
TILE_GROUP = "virchow_2560"            # = {model}{key_suffix(concat)}
SLIDE_PRESET = "prism"
SLIDE_KEY = f"aggregates/{SLIDE_PRESET}/feature"   # config の keys.slide_feature は TITAN 用（aggregates/titan/feature）なので使わない
PATCH_SIZE = 224
TARGET_MPP = 0.5
FAMILY_ORDER = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
LOG_FIELDS = ["time", "status", "seconds", "n_tiles", "site", "family", "slide_id", "h5"]


def resolve_device(spec: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else spec


def state_of(h5_path: str, keys: dict) -> str:
    """'done'（PRISM あり）/ 'aggregate'（Virchow の特徴だけ）/ 'extract'（何も無い）/ 'no_h5' / 'error'。"""
    if not h5_path or not Path(h5_path).exists():
        return "no_h5"
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(TILE_GROUP)
            if grp is None or keys["tile_features"] not in grp or keys["tile_coords"] not in grp:
                return "extract"
            return "done" if SLIDE_KEY in grp else "aggregate"
    except OSError:
        return "error"


def order_targets(df: pd.DataFrame) -> pd.DataFrame:
    """EBRAINS → 院内の順、科は論文コホート → グリオーマ → DLBCL → 転移。院内の論文コホート（original h5 に書く）は最後。"""
    site_rank = {"ebrains": 0, "patho2": 1}
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    key = [(site_rank[s], (fam_rank[f] + 10) if (s == "patho2" and f in ("meningioma", "SFT")) else fam_rank[f], i)
           for s, f, i in zip(df["site"], df["family"], df["slide_id"])]
    return df.assign(_k=key).sort_values("_k").drop(columns="_k")


def aggregate_one(model, h5_path: str, keys: dict, device: str) -> int:
    """virchow_2560/features → virchow_2560/aggregates/prism/feature。wt aggregate と同じ計算。戻り値はタイル数。"""
    from wsi_toolbox.presets.slide import aggregate_slide

    with h5py.File(h5_path, "r") as f:
        grp = f[TILE_GROUP]
        feats = grp[keys["tile_features"]][:]
        coords = grp[keys["tile_coords"]][:]
        patch_size = int(grp.attrs["patch_size"])
    feat_t = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    coord_t = torch.from_numpy(coords).long().unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = aggregate_slide(SLIDE_PRESET, model, feat_t, coord_t, patch_size)
    emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
    target = f"{TILE_GROUP}/{SLIDE_KEY}"
    with h5py.File(h5_path, "a") as f:
        if target in f:
            del f[target]
        f.create_dataset(target, data=emb_np)
    return int(feats.shape[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="対象と状態の集計だけ（GPU を使わない）")
    ap.add_argument("--sites", nargs="*", default=None, help="ebrains / patho2 で絞る")
    ap.add_argument("--families", nargs="*", default=None, help=f"科で絞る: {FAMILY_ORDER}")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 枚だけ")
    ap.add_argument("--batch-size", type=int, default=None, help="Virchow のバッチ（既定: config transfer.prism_batch_size か 256）")
    args = ap.parse_args()

    cfg = load_config()
    tcfg = cfg["transfer"]
    keys = cfg["keys"]
    out_dir = Path(cfg["output_dir"]) / "transfer"
    manifest = pd.read_csv(Path(cfg["output_dir"]) / tcfg.get("manifest", "transfer/manifest.csv"), dtype=str).fillna("")

    df = order_targets(manifest)
    if args.sites:
        df = df[df["site"].isin(args.sites)]
    if args.families:
        df = df[df["family"].isin(args.families)]
    df = df.assign(state=[state_of(p, keys) for p in df["h5_prism"]])
    print(df.groupby(["site", "family", "state"], observed=True).size().unstack(fill_value=0).to_string())
    todo = df[df["state"].isin(("extract", "aggregate"))]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"\nto do: {len(todo)} ({(todo['state'] == 'extract').sum()} extract + aggregate, {(todo['state'] == 'aggregate').sum()} aggregate only); "
          f"done {(df['state'] == 'done').sum()}, no_h5 {(df['state'] == 'no_h5').sum()}, error {(df['state'] == 'error').sum()}")
    if args.dry_run or todo.empty:
        return

    import wsi_toolbox as wt
    from wsi_toolbox.presets.slide import create_slide_preset_model

    wt.set_default_progress("none")
    device = resolve_device(tcfg.get("device", cfg.get("titan", {}).get("device", "auto")))
    batch = args.batch_size or int(tcfg.get("prism_batch_size", 256))
    print(f"device: {device}  batch: {batch}  tile: {TILE_PRESET}/{TILE_OUTPUT} {PATCH_SIZE} px @ {TARGET_MPP} mpp -> {TILE_GROUP}  slide: {SLIDE_PRESET}")
    extract = wt.FeatureExtractionCommand(model=TILE_PRESET, preset=TILE_PRESET, outputs=(TILE_OUTPUT,), device=device,
                                          batch_size=batch, overwrite=False, patch_size=PATCH_SIZE, target_mpp=TARGET_MPP)
    model = create_slide_preset_model(SLIDE_PRESET).to(device).eval()

    log_path = out_dir / "prism_log.csv"
    new_log = not log_path.exists()
    counts: dict[str, int] = {}
    t_start = time.time()
    with open(log_path, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        for i, r in enumerate(todo.itertuples(index=False), 1):
            t0 = time.time()
            try:
                if r.state == "extract":
                    if not r.wsi or not Path(r.wsi).exists():
                        raise FileNotFoundError(f"WSI not found: {r.wsi}")
                    extract(r.h5_prism, wsi_path=r.wsi)
                n = aggregate_one(model, r.h5_prism, keys, device)
                status = "ok" if r.state == "extract" else "ok_agg"
            except Exception as e:  # 1 本の失敗で止めない
                status, n = f"error: {type(e).__name__}: {str(e)[:80]}", 0
            dt = time.time() - t0
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            print(f"  [{i}/{len(todo)}] {status:6s} {dt:6.1f}s N={n:6d}  {r.site}/{r.family}/{r.slide_id}")
            w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "status": status, "seconds": f"{dt:.1f}", "n_tiles": n,
                        "site": r.site, "family": r.family, "slide_id": r.slide_id, "h5": r.h5_prism})
            lf.flush()
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min: {counts}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
