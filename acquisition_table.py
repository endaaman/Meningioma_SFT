"""acquisition_table.py — 撮像と標本の条件の表（事実の整理。解析ではない。v4 の転移テストの 2,290 枚、2026-11-25）。

施設 × 腫瘍ごとに、記録から機械的に取れるものだけを並べる。取れない項目は「記録なし」と書き、推測で埋めない。

読むもの（どれも読むだけ。NFS には書かない）:
    out/transfer/manifest.csv           施設・腫瘍・患者・WSI のパス・h5 のパス
    WSI（NDPI）の先頭の IFD の TIFF タグ（tifffile。画像は読まない。openslide は NFS 越しに画像まで読みに行って遅いので使わない）:
        Make / Model（スキャナの製造元と型番。型番から製品名への読み替えはしない）、Software、DateTime（撮像の日時）、
        XResolution（撮像の µm/px = 1e4 / 画素数毎 cm）、NDPI の Magnification タグ（対物の倍率）
    h5（manifest の h5_titan）の conch15_768 の属性: mpp（パッチを切った段の µm/px）、patch_count、patch_size（512）
        組織面積 = patch_count × (patch_size × mpp)²（組織と判定されたパッチの面積の和、mm²。パッチ抽出の組織判定に依る）
    院内の症例番号（slide_id）の先頭の年 = shift_time.py と同じ読み方（`[Nn]?YY-数字` の YY → 20YY、受付の年）
スライドの選び方は manifest の患者あたりの枚数から機械的に書く（全員 1 枚なら「1 患者 1 枚」）。標本を作った施設・染色の手順は
記録が無いので「記録なし」。

出力: output_dir/acquisition/slide_metadata.csv（スライドごと、キャッシュ。--reload で読み直す）、acquisition_table.{csv,md}

Usage:
    uv run python acquisition_table.py [--reload] [--workers 8]
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

from utils.loader import load_config

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
SITES = [("ebrains", "EBRAINS"), ("patho2", "In-house")]
ID_RE = re.compile(r"^[Nn]?(\d{2})-\d+")
TAGS = {"make": "Make", "model": "Model", "software": "Software", "scan_datetime": "DateTime"}
NONE = "記録なし"


def one(row: pd.Series) -> dict:
    out = {"site": row["site"], "slide_id": row["slide_id"]}
    try:
        with tifffile.TiffFile(row["wsi"]) as tf:
            pg = tf.pages[0]
            tags = {t.name: t.value for t in pg.tags.values()}
            out.update({k: tags.get(v) for k, v in TAGS.items()})
            xr, unit = tags.get("XResolution"), tags.get("ResolutionUnit")
            if xr and int(unit) == 3:   # 画素数毎 cm
                out["scan_mpp"] = 1e4 / (xr[0] / xr[1])
            nd = pg.ndpi_tags or {}
            out["objective"] = nd.get("Magnification")
    except Exception as e:   # noqa: BLE001 読めなければ空欄（記録なし）にして理由を残す
        out["wsi_error"] = repr(e)[:200]
    try:
        with h5py.File(row["h5_titan"], "r") as f:
            a = f["conch15_768"].attrs
            out.update({"patch_mpp": float(a["mpp"]), "patch_count": int(a["patch_count"]),
                        "patch_size": int(a.get("patch_size", 512))})
    except Exception as e:   # noqa: BLE001
        out["h5_error"] = repr(e)[:200]
    return out


def collect(man: pd.DataFrame, path: Path, reload: bool, workers: int) -> pd.DataFrame:
    if path.exists() and not reload:
        meta = pd.read_csv(path, dtype={"slide_id": str})
        if set(zip(meta["site"], meta["slide_id"])) == set(zip(man["site"], man["slide_id"])):
            print(f"  cache: {path}")
            return meta
    print(f"  reading {len(man)} WSI headers and h5 attributes ({workers} threads) ...")
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, [r for _, r in man.iterrows()]))
    meta = pd.DataFrame(rows)
    meta.to_csv(path, index=False)
    print(f"  saved: {path}")
    return meta


def _counts(v: pd.Series) -> str:
    v = v.dropna().astype(str)
    if not len(v):
        return NONE
    c = v.value_counts()
    return "; ".join(f"{k} ({n})" for k, n in c.items())


def _med(v: pd.Series, d: int = 0, iqr: bool = True) -> str:
    v = pd.to_numeric(v, errors="coerce").dropna()
    if not len(v):
        return NONE
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return f"{q2:,.{d}f}" + (f" [{q1:,.{d}f}–{q3:,.{d}f}]" if iqr else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["output_dir"]) / "acquisition"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"].get("manifest", "transfer/manifest.csv"),
                      dtype={"slide_id": str, "patient": str})
    meta = collect(man, out / "slide_metadata.csv", args.reload, args.workers)
    d = man.merge(meta, on=["site", "slide_id"], how="left", validate="one_to_one")
    d["area_mm2"] = d["patch_count"] * (d["patch_size"] * d["patch_mpp"] / 1000.0) ** 2
    d["scan_year"] = pd.to_datetime(d["scan_datetime"], format="%Y:%m:%d %H:%M:%S", errors="coerce").dt.year
    d["id_year"] = d["slide_id"].map(lambda s: 2000 + int(m.group(1)) if (m := ID_RE.match(s)) else np.nan)
    d.loc[d["site"] == "ebrains", "id_year"] = np.nan   # EBRAINS の ID は uuid（年を含まない）

    rows = []
    for fam in FAMILIES:
        for site, sname in SITES:
            g = d[(d["family"] == fam) & (d["site"] == site)]
            spp = g.groupby("patient").size()
            sel = ("1 patient, 1 slide (all patients)" if (spp == 1).all()
                   else f"all slides of each patient: 1 slide {int((spp == 1).sum())} / ≥ 2 slides {int((spp >= 2).sum())} patients (max {int(spp.max())})")
            sy = g["scan_year"].dropna()
            iy = g["id_year"].dropna()
            rows.append({
                "Tumour": fam, "Site": sname, "Slides": len(g), "Patients": g["patient"].nunique(),
                "Slides per patient": sel,
                "Scanner (tiff.Make tiff.Model)": _counts(g["make"].fillna("") + " " + g["model"].fillna("")) if g["model"].notna().any() else NONE,
                "Scanner software": _counts(g["software"]),
                "Objective (×)": _counts(g["objective"]),
                "Scan µm/px": _counts(pd.to_numeric(g["scan_mpp"], errors="coerce").round(4)),
                "Patch µm/px": _counts(g["patch_mpp"].round(4)),
                "Scan year (tiff.DateTime)": f"{int(sy.min())}–{int(sy.max())}" if len(sy) else NONE,
                "Case-number year (in-house ID)": f"{int(iy.min())}–{int(iy.max())}" if len(iy) else NONE,
                "Diagnosis year": NONE if site == "ebrains" else "same as case-number year (accession); no separate record",
                "Patches (512 px), median [IQR]": _med(g["patch_count"]),
                "Tissue area mm², median [IQR]": _med(g["area_mm2"], 1),
                "Specimen preparation site / staining protocol": NONE,
                "WSI header unreadable": int(g.get("wsi_error", pd.Series(dtype=str)).notna().sum()) if "wsi_error" in g else 0,
                "h5 unreadable": int(g.get("h5_error", pd.Series(dtype=str)).notna().sum()) if "h5_error" in g else 0,
            })
    t = pd.DataFrame(rows)
    t.to_csv(out / "acquisition_table.csv", index=False)
    lines = []
    for _, r in t.iterrows():
        lines.append(f"### {r['Tumour']} / {r['Site']}")
        lines += [f"- {k}: {v}" for k, v in r.items() if k not in ("Tumour", "Site")]
        lines.append("")
    notes = ["型番（tiff.Model）は WSI のヘッダにある文字列のまま（製品名への読み替えはしない）。",
             "Patch µm/px = h5 のパッチを切った段の解像度（目標 0.5 µm/px に最も近い段）。組織面積は組織と判定されたパッチの面積の和。",
             "Case-number year = 院内の症例番号の先頭 2 桁（shift_time.py と同じ読み方）。EBRAINS の ID は uuid で年を含まず、診断の年の記録は無い。",
             "標本を作った施設、染色の手順、固定の条件は、手元のどの表にも記録が無い（記録なし）。"]
    (out / "acquisition_table.md").write_text("# 撮像と標本の条件（v4、転移テストの全スライド）\n\n" + "\n".join(lines) + "\n"
                                              + "\n".join(f"- {n}" for n in notes) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(t.drop(columns=["Slides per patient"]).to_string(index=False))
    print(f"saved: {out}/acquisition_table.{{csv,md}}")


if __name__ == "__main__":
    main()
