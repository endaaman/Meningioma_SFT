"""patho2_gliomas_extract.py — 院内グリオーマ（Patho2Gliomas、旧 BrainTumor）の HE WSI から CONCH v1.5 パッチ特徴と TITAN を作る。

入力（config の patho2_gliomas）:
    wsi_root/<群>/<症例番号>/*.ndpi
      コピーが途中で切れているファイルがある（N20-097#1_HE: 295 MB、正本は 4,952 MB。openslide も tifffile も開けない）。review_root
      （server41/脳腫瘍レビュー、院内 WSI の正本）に同名でより大きいファイルがあればそちらを読む（index の wsi_source = review）。
      群: A = IDH 変異型星細胞腫（ast）、G = IDH 野生型（glm）、O = 乏突起膠腫（odg）、L = リンパ腫（lym）、M = 転移（met）。
      略号は山田さんの WSI_BT（folder_labels.txt）と同じ。B（剖検様番号）と unused / todo / その他 / small / other は対象外。
      症例番号のディレクトリは `17-0395` / `N22-263` の形（空のディレクトリもある）。
    HE の判定はファイル名で行う: 日付と症例番号・ブロック番号を除いた残り（染色名）が HE を含む（"#1_HE", "RECUT HE", "-2HE"）か、
      染色名が無い（"16-1831.ndpi", "n16-177-1.ndpi", "18-3780_1.ndpi"）ものを HE とみなす。それ以外（MIB-1, GFAP, OLIG2, IDH-1, ...）は
      免疫染色として除外。判定は全ファイル分を index CSV に残す（監査用）。
出力:
    out_root/<群>/<症例番号>/<元ファイル名>.h5
      EBRAINS_embeddings（学生が作った EBRAINS 全体の特徴）と同じ構成: wsi-toolbox 0.7.0 の `wt extract --preset conch15 --outputs cls,proj
      -S 512 --mpp 0.5` 相当（FeatureExtractionCommand(outputs=("cls","proj"))）で、1 回の forward から
      conch15/{features(1024 = cls),coordinates} と conch15_768/{features(768 = attn-pool 射影、TITAN 入力),coordinates} の両方を書く
      （パッチ画像のキャッシュは持たない）。続けて conch15_768/aggregates/titan/feature（titan_batch.encode_file = titan.py / wt aggregate と同じ計算）。
      1 症例に HE が複数あるときは全部作る。どのスライドを症例の代表にするかは解析側で選ぶ（index CSV の n_patches・recut を材料にする）。
    output_dir/patho2_gliomas/slides.csv   … 全 ndpi の一覧（group, label, case, file, stain, is_he, recut, h5, n_patches, status）
    output_dir/patho2_gliomas/extract_log.csv … 処理ログ（1 本ずつ追記）
既存の h5 は、両方の特徴と TITAN が揃っていればスキップ、TITAN だけ無ければ TITAN だけ書く、特徴がどちらか欠けていれば作り直す
（0.6.2 の conch15_768 だけの h5 も作り直しの対象）。

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

import h5py
import pandas as pd
import torch

from titan_batch import encode_file, load_titan, resolve_device
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


def h5_state(h5_path: Path, feature_groups: list[str], tile_model: str, keys: dict) -> str:
    """'done'（全グループの特徴 + TITAN）/ 'todo'（特徴は揃っているが TITAN 無し）/ 'no_features'（どれか欠ける）/ 'error' / 'missing'。"""
    if not h5_path.exists():
        return "missing"
    try:
        with h5py.File(h5_path, "r") as f:
            for g in feature_groups:
                grp = f.get(g)
                if grp is None or keys["tile_features"] not in grp or keys["tile_coords"] not in grp:
                    return "no_features"
            return "done" if keys["slide_feature"] in f[tile_model] else "todo"
    except OSError:
        return "error"


def review_twins(review_root: Path | None) -> dict[str, list[tuple[int, Path]]]:
    """院内 WSI の正本（server41/脳腫瘍レビュー）の ndpi をファイル名 → [(size, path)] で引けるようにする（Patho2Gliomas の途中で切れたコピーの代替用）。"""
    twins: dict[str, list[tuple[int, Path]]] = {}
    if review_root is None or not review_root.is_dir():
        return twins
    for p in review_root.rglob("*.ndpi"):
        twins.setdefault(p.name, []).append((p.stat().st_size, p))
    return twins


def build_index(wsi_root: Path, out_root: Path, groups: dict[str, str], order: list[str],
                review_root: Path | None = None) -> pd.DataFrame:
    twins = review_twins(review_root)
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
                # Patho2Gliomas のコピーが正本（同名ファイル）より小さければ途中で切れたコピーなので、正本の方を読む
                src, source = ndpi, "patho2_gliomas"
                bigger = [(s, p) for s, p in twins.get(ndpi.name, []) if s > ndpi.stat().st_size]
                if bigger:
                    src, source = max(bigger)[1], "review"
                rows.append({
                    "group": g, "label": groups[g], "case": cdir.name, "file": ndpi.name,
                    "stain": stain, "is_he": is_he, "recut": recut,
                    "wsi": str(src), "wsi_source": source, "h5": str(out_root / g / cdir.name / f"{ndpi.stem}.h5"),
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
    tile_model: str = pcfg.get("tile_model", "conch15_768")           # TITAN の入力グループ
    preset: str = pcfg.get("preset", "conch15")
    outputs: tuple[str, ...] = tuple(pcfg.get("outputs", ["cls", "proj"]))
    feature_groups = [preset if o == "cls" else f"{preset}_768" for o in outputs if o in ("cls", "proj")]   # 0.7.0 の書き先: cls → {preset}/、proj → {preset}_768/
    keys = cfg["keys"]
    out_dir = Path(cfg["output_dir"]) / "patho2_gliomas"
    out_dir.mkdir(parents=True, exist_ok=True)

    review_root = Path(pcfg["review_root"]) if pcfg.get("review_root") else None
    index = build_index(wsi_root, out_root, groups, order, review_root)
    he = index[index["is_he"]]
    print(f"wsi_root: {wsi_root}\nout_root: {out_root}")
    n_rev = int((index["wsi_source"] == "review").sum())
    if n_rev:
        print(f"途中で切れたコピーの代わりに正本（{review_root}）を読むファイル: {n_rev}")
        print(index.loc[index["wsi_source"] == "review", ["group", "case", "file"]].to_string(index=False))
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
    print(f"\ndevice: {device}  targets: {len(targets)} HE slides  preset: {preset} outputs: {outputs} -> {feature_groups}")
    extract = wt.FeatureExtractionCommand(
        model=preset, preset=preset, outputs=outputs, device=device,
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
                state = h5_state(h5_path, feature_groups, tile_model, keys)
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
    n_patches, states = [], []
    for r in index.itertuples(index=False):
        p = Path(r.h5)
        st = h5_state(p, feature_groups, tile_model, keys) if r.is_he else ""
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
