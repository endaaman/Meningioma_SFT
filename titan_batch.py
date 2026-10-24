"""titan_batch.py — 外部の h5 の木に TITAN スライド埋め込みを書き込む（EBRAINS 全体 = EBRAINS_embeddings 用）。

titan.py は config の embedding.*（論文のコホート）だけを対象にする。こちらは任意のディレクトリ木を再帰的に走査し、
{tile_model}/features と coordinates があって {tile_model}/aggregates/titan/feature の無い h5 に TITAN を書く。
出力キーと計算は titan.py / `wt aggregate` と同じ（EBRAINS_embeddings の a194f779 で論文の埋め込みと cos 1.000000、2026-10-25）。

対象（config の ebrains_embeddings）:
    root:     /mnt/server42/AI/Datasets/EBRAINS_embeddings/<診断名>/<uuid>_hdf.h5
              学生が `wt extract --preset conch15` で作った CONCH v1.5 の特徴（conch15 = cls 1024 次元、conch15_768 = TITAN 入力の 768 次元、
              512 px / 0.5 mpp、パッチ画像なし）。3,114 uuid（EBRAINS 3,115 枚のうち壊れている a1982bd3 を除く全部）。
              「1p/19q」の診断名はディレクトリが入れ子になっていて、7 uuid は親と子の両方にある（両方に書く。中身は同じ特徴）。
    priority: 先に回す診断名ディレクトリ（グリオーマ）。残りはその後に名前順。

ログ: output_dir/ebrains_embeddings/titan_log.csv に 1 行ずつ追記（time, status, seconds, n_patches, dir, file）。
既存の TITAN はスキップする（--overwrite で作り直し）ので、途中で止めても再実行すれば続きから進む。

Usage:
    uv run python -u titan_batch.py --dry-run          # 対象と順番だけ表示（GPU を使わない）
    uv run python -u titan_batch.py --limit 3          # 先頭 3 本だけ
    uv run python -u titan_batch.py --priority-only    # 優先ディレクトリ（グリオーマ）だけ
    uv run python -u titan_batch.py                    # 全部
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.loader import load_config

LOG_FIELDS = ["time", "status", "seconds", "n_patches", "dir", "file"]


def resolve_device(spec: str) -> str:
    return ("cuda" if torch.cuda.is_available() else "cpu") if spec == "auto" else spec


def load_titan(device: str):
    from wsi_toolbox.presets.slide.titan import create_titan_model
    return create_titan_model().to(device).eval()


def titan_status(h5_path: Path, tile_model: str, keys: dict) -> str:
    """'done'（TITAN あり）/ 'todo'（特徴はあるが TITAN 無し）/ 'no_features' / 'error'。"""
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(tile_model)
            if grp is None or keys["tile_features"] not in grp or keys["tile_coords"] not in grp:
                return "no_features"
            return "done" if keys["slide_feature"] in grp else "todo"
    except OSError:
        return "error"


def encode_file(model, h5_path: Path, tile_model: str, keys: dict, device: str, overwrite: bool = False) -> tuple[str, int]:
    """1 本の h5 に TITAN を書く。titan.py と同じ計算・同じキー。戻り値は (status, n_patches)。"""
    target_key = f"{tile_model}/{keys['slide_feature']}"
    with h5py.File(h5_path, "r") as f:
        grp = f.get(tile_model)
        if grp is None or keys["tile_features"] not in grp:
            return "no_features", 0
        if target_key in f and not overwrite:
            return "skip", int(grp[keys["tile_features"]].shape[0])
        features = grp[keys["tile_features"]][:]
        coords = grp[keys["tile_coords"]][:]
        patch_size_lv0 = int(grp.attrs["patch_size"])   # coordinates は抽出レベルの画素座標なので、そのままの patch_size を渡す（titan.py / wt aggregate と同じ）
    if features.shape[0] == 0:
        return "empty", 0

    feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
    coord_t = torch.from_numpy(coords).long().unsqueeze(0).to(device)
    with torch.inference_mode():
        emb = model.encode_slide_from_patch_features(feat_t, coord_t, patch_size_lv0)
    emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)

    with h5py.File(h5_path, "a") as f:
        if target_key in f:
            del f[target_key]
        f.create_dataset(target_key, data=emb_np)
    return "ok", int(features.shape[0])


def list_targets(root: Path, priority: list[str]) -> list[tuple[int, Path]]:
    """root 以下の全 h5 を (優先順位, パス) で返す。優先順位は最上位ディレクトリ名が priority の何番目か（無ければ末尾）。"""
    rank = {name: i for i, name in enumerate(priority)}
    items = []
    for p in root.rglob("*.h5"):
        if p.name.startswith(".") or p.suffix != ".h5":
            continue
        top = p.relative_to(root).parts[0] if len(p.relative_to(root).parts) > 1 else ""
        items.append((rank.get(top, len(priority)), p))
    items.sort(key=lambda t: (t[0], str(t[1].relative_to(root))))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="config の ebrains_embeddings.root を上書き")
    ap.add_argument("--dry-run", action="store_true", help="対象と順番だけ表示")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 本だけ処理（dry-run では表示だけ）")
    ap.add_argument("--priority-only", action="store_true", help="priority のディレクトリだけ")
    ap.add_argument("--overwrite", action="store_true", help="既存の TITAN を作り直す")
    args = ap.parse_args()

    cfg = load_config()
    ecfg = cfg["ebrains_embeddings"]
    root = Path(args.root or ecfg["root"])
    tile_model: str = ecfg.get("tile_model", "conch15_768")
    priority: list[str] = ecfg.get("priority", [])
    keys = cfg["keys"]
    log_path = Path(cfg["output_dir"]) / ecfg.get("log", "ebrains_embeddings/titan_log.csv")

    items = list_targets(root, priority)
    if args.priority_only:
        items = [t for t in items if t[0] < len(priority)]
    if args.limit:
        items = items[: args.limit]

    n_prio = sum(1 for r, _ in items if r < len(priority))
    print(f"root: {root}\nfiles: {len(items)} (priority {n_prio}, rest {len(items) - n_prio})  tile_model: {tile_model}")
    if args.dry_run:
        by_dir: dict[str, int] = {}
        for _, p in items:
            d = str(p.relative_to(root).parent)
            by_dir[d] = by_dir.get(d, 0) + 1
        for i, (d, n) in enumerate(by_dir.items()):
            mark = "*" if d.split("/")[0] in priority else " "
            print(f"  {mark} {n:5d}  {d}")
        print("(* = priority)")
        print("status of first 5:", [(p.name[:8], titan_status(p, tile_model, keys)) for _, p in items[:5]])
        return

    device = resolve_device(ecfg.get("device", cfg.get("titan", {}).get("device", "auto")))
    print(f"device: {device}\nlog: {log_path}")
    model = load_titan(device)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    counts: dict[str, int] = {}
    t_start = time.time()
    with open(log_path, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        for i, (_, p) in enumerate(items, 1):
            t0 = time.time()
            try:
                status, n = encode_file(model, p, tile_model, keys, device, args.overwrite)
            except Exception as e:  # 1 本の失敗で止めない
                status, n = f"error: {type(e).__name__}: {str(e)[:80]}", 0
            dt = time.time() - t0
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            if status != "skip" or i % 200 == 0:
                print(f"  [{i}/{len(items)}] {status:6s} {dt:5.1f}s N={n:5d}  {p.relative_to(root)}")
            if status != "skip":
                w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "status": status, "seconds": f"{dt:.2f}",
                            "n_patches": n, "dir": str(p.relative_to(root).parent), "file": p.name})
                lf.flush()
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min: {counts}")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
