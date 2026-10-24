"""patho2_gliomas_extract.py — 院内グリオーマ（Patho2Gliomas、旧 BrainTumor）の HE WSI から CONCH v1.5 パッチ特徴と TITAN を作る。

入力（config の patho2_gliomas）:
    wsi_root/<群>/<症例番号>/*.ndpi
      群: A = IDH 変異型星細胞腫（ast）、G = IDH 野生型（glm）、O = 乏突起膠腫（odg）、L = リンパ腫（lym）、M = 転移（met）。
      略号は山田さんの WSI_BT（folder_labels.txt）と同じ。B（剖検様番号）と unused / todo / その他 / small / other は対象外。
      症例番号のディレクトリは `17-0395` / `N22-263` の形（空のディレクトリもある）。
    HE の判定はファイル名で行う: 日付と症例番号・ブロック番号を除いた残り（染色名）が HE を含む（"#1_HE", "RECUT HE", "-2HE"）か、
      染色名が無い（"16-1831.ndpi", "n16-177-1.ndpi", "18-3780_1.ndpi"）ものを HE とみなす。それ以外（MIB-1, GFAP, OLIG2, IDH-1, ...）は
      免疫染色として除外。判定は全ファイル分を index CSV に残す（監査用）。
出力:
    out_root/<群>/<症例番号>/<元ファイル名>.h5
      論文と同じキー: conch15_768/{features,coordinates}（512 px / 0.5 mpp。wsi-toolbox の FeatureExtractionCommand =
      `wt extract --preset conch15_768 -S 512 --mpp 0.5`。パッチ画像のキャッシュは持たない）と conch15_768/aggregates/titan/feature
      （titan_batch.encode_file = titan.py / wt aggregate と同じ計算）。
      1 症例に HE が複数あるときは全部作る。どのスライドを症例の代表にするかは解析側で選ぶ（index CSV の n_patches・recut を材料にする）。
    output_dir/patho2_gliomas/slides.csv   … 全 ndpi の一覧（group, label, case, file, stain, is_he, recut, h5, n_patches, status）
    output_dir/patho2_gliomas/extract_log.csv … 処理ログ（1 本ずつ追記）
既存の h5 は、特徴と TITAN が揃っていればスキップ、TITAN だけ無ければ TITAN だけ書く、特徴が無ければ作り直す。

Usage:
    uv run python -u patho2_gliomas_extract.py --dry-run           # HE 判定と件数だけ（index CSV は書く。GPU を使わない）
    uv run python -u patho2_gliomas_extract.py --groups A G O      # グリオーマ 3 群だけ
    uv run python -u patho2_gliomas_extract.py --limit 1
    uv run python -u patho2_gliomas_extract.py                      # config の priority 順に全群
"""
from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from titan_batch import encode_file, load_titan, resolve_device, titan_status
from utils.loader import load_config

SKIP_DIRS = {"unused", "todo", "その他", "small", "other"}
CASE_RE = re.compile(r"^[nN]?\d{2}-\d{3,4}$")
DATE_RE = re.compile(r"\s*-?\s*20\d{2}-\d{2}-\d{2}[ _]\d{2}\.\d{2}\.\d{2}$")   # " - 2019-03-18 16.21.03" / " 2019-10-30 12.28.33"
ID_RE = re.compile(r"^[nN]?\d{2}-\d{3,4}")
BLOCK_RE = re.compile(r"^[\s#_\-]*[\d,]*[\s#_\-]*")                           # "#1_", "-2", "_1", "#1,2_", " "
HE_RE = re.compile(r"(?<![A-Z])HE(?![A-Z])")                                   # HE をトークンとして含む（HER2 は含まない）
LOG_FIELDS = ["time", "status", "seconds", "n_patches", "group", "case", "file"]


def parse_stain(stem: str) -> str:
    """ファイル名（拡張子なし）から染色名を取り出す。無ければ ''（= 染色名の無い HE）。"""
    s = DATE_RE.sub("", stem).strip()
    m = ID_RE.match(s)
    rest = s[m.end():] if m else s
    rest = BLOCK_RE.sub("", rest)
    return rest.strip().upper()


def classify(stem: str) -> tuple[str, bool, bool]:
    stain = parse_stain(stem)
    compact = stain.replace(" ", "").replace("_", "")
    is_he = stain == "" or bool(HE_RE.search(stain)) or compact in ("HERECUT", "RECUTHE")   # "HERECUT" は区切りなしの再切片 HE
    return stain, is_he, "RECUT" in stain


def build_index(wsi_root: Path, out_root: Path, groups: dict[str, str], order: list[str]) -> pd.DataFrame:
    rows = []
    for g in order:
        gdir = wsi_root / g
        if not gdir.is_dir():
            print(f"WARN: group dir not found: {gdir}")
            continue
        for cdir in sorted(gdir.iterdir()):
            if not cdir.is_dir() or cdir.name in SKIP_DIRS or not CASE_RE.match(cdir.name):
                continue
            for ndpi in sorted(cdir.glob("*.ndpi")):
                stain, is_he, recut = classify(ndpi.stem)
                rows.append({
                    "group": g, "label": groups[g], "case": cdir.name, "file": ndpi.name,
                    "stain": stain, "is_he": is_he, "recut": recut,
                    "wsi": str(ndpi), "h5": str(out_root / g / cdir.name / f"{ndpi.stem}.h5"),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--groups", nargs="+", default=None, help="処理する群（既定: config の priority 順に全部）")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 本だけ")
    ap.add_argument("--overwrite", action="store_true", help="既存の h5 を作り直す")
    args = ap.parse_args()

    cfg = load_config()
    pcfg = cfg["patho2_gliomas"]
    wsi_root, out_root = Path(pcfg["wsi_root"]), Path(pcfg["out_root"])
    groups: dict[str, str] = pcfg["groups"]
    order: list[str] = args.groups or pcfg.get("priority", list(groups))
    tile_model: str = pcfg.get("tile_model", "conch15_768")
    keys = cfg["keys"]
    out_dir = Path(cfg["output_dir"]) / "patho2_gliomas"
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(wsi_root, out_root, groups, order)
    he = index[index["is_he"]]
    print(f"wsi_root: {wsi_root}\nout_root: {out_root}")
    summary = index.groupby("group").agg(cases=("case", "nunique"), files=("file", "size"), he=("is_he", "sum")).reindex(order)
    summary["he_cases"] = he.groupby("group")["case"].nunique().reindex(order)
    summary["recut"] = he.groupby("group")["recut"].sum().reindex(order)
    print(summary.fillna(0).astype(int).to_string())
    if args.dry_run:
        non_he = index.loc[~index["is_he"], "stain"].value_counts()
        print(f"\nnon-HE stain names ({len(non_he)}): {', '.join(f'{k}×{v}' for k, v in non_he.head(40).items())}")
        print(f"HE stain names: {he['stain'].value_counts().to_dict()}")
        index.to_csv(out_dir / "slides.csv", index=False)
        print(f"\nindex -> {out_dir / 'slides.csv'} ({len(index)} rows, HE {len(he)})")
        return

    targets = he.copy()
    if args.limit:
        targets = targets.head(args.limit)

    import wsi_toolbox as wt
    wt.set_default_progress("none")
    device = resolve_device(pcfg.get("device", "auto"))
    print(f"\ndevice: {device}  targets: {len(targets)} HE slides")
    extract = wt.FeatureExtractionCommand(
        model=tile_model, preset=tile_model, device=device,
        batch_size=int(pcfg.get("batch_size", 256)), overwrite=True,
        patch_size=int(pcfg.get("patch_size", 512)), target_mpp=float(pcfg.get("target_mpp", 0.5)),
    )
    titan = load_titan(device)

    log_path = out_dir / "extract_log.csv"
    new_log = not log_path.exists()
    counts: dict[str, int] = {}
    t_start = time.time()
    with open(log_path, "a", newline="") as lf:
        w = csv.DictWriter(lf, fieldnames=LOG_FIELDS)
        if new_log:
            w.writeheader()
        for i, r in enumerate(targets.itertuples(index=False), 1):
            h5_path, wsi_path = Path(r.h5), Path(r.wsi)
            t0 = time.time()
            try:
                state = titan_status(h5_path, tile_model, keys) if h5_path.exists() else "missing"
                if state == "done" and not args.overwrite:
                    status, n = "skip", 0
                else:
                    if state in ("missing", "no_features", "error") or args.overwrite:
                        h5_path.parent.mkdir(parents=True, exist_ok=True)
                        if state == "error":
                            h5_path.unlink()
                        extract(str(h5_path), wsi_path=str(wsi_path))
                    status, n = encode_file(titan, h5_path, tile_model, keys, device, overwrite=True)
                    status = "ok" if status == "ok" else f"titan_{status}"
            except Exception as e:  # 1 本の失敗で止めない
                status, n = f"error: {type(e).__name__}: {str(e)[:80]}", 0
            dt = time.time() - t0
            counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
            print(f"  [{i}/{len(targets)}] {status:6s} {dt:6.1f}s N={n:5d}  {r.group}/{r.case}/{r.file}")
            if status != "skip":
                w.writerow({"time": datetime.now().isoformat(timespec="seconds"), "status": status, "seconds": f"{dt:.1f}",
                            "n_patches": n, "group": r.group, "case": r.case, "file": r.file})
                lf.flush()

    # index を更新（h5 の有無・パッチ数・状態）
    import h5py
    n_patches, states = [], []
    for r in index.itertuples(index=False):
        p = Path(r.h5)
        st = titan_status(p, tile_model, keys) if (r.is_he and p.exists()) else ("" if not r.is_he else "missing")
        states.append(st)
        n = ""
        if st in ("done", "todo"):
            with h5py.File(p, "r") as f:
                n = int(f[tile_model][keys["tile_features"]].shape[0])
        n_patches.append(n)
    index["status"], index["n_patches"] = states, n_patches
    index.to_csv(out_dir / "slides.csv", index=False)
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min: {counts}\nindex -> {out_dir / 'slides.csv'}")

    del titan
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
