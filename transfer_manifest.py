"""transfer_manifest.py — 転移テスト（3 法 × 推定元 × 当て先）で使う全スライドの一覧を作り、埋め込みの揃い具合を確認する。

対象（2 施設 × 5 科）:
    meningioma / SFT : 論文のコホート（config の label.*.subtype と embedding.*.original。EBRAINS n551、patho2 n577）
    glioma (ast / glm / odg) / lymphoma (DLBCL) / metastasis:
        patho2  = Patho2Gliomas の代表 HE 1 症例 1 枚（patho2_gliomas.representatives → out/patho2_gliomas/slides.csv）
        ebrains = annotation.csv の diagnosis を shift_transfer.ebrains_groups で科に対応づけ（control は除く）、EBRAINS_embeddings の h5

列: slide_id, site, family, group（科の中の群: 髄膜腫サブタイプ / ast / glm / odg / lym / met）, diagnosis（元のラベル）,
    patient, wsi（元 WSI。Virchow 抽出に使う）, h5_titan（TITAN が入っている h5）, titan（あるか）, h5_prism, prism（あるか）

出力: output_dir/transfer/manifest.csv と、科 × 施設ごとの枚数・患者数・埋め込みの有無の要約（標準出力）。

Usage:
    uv run python transfer_manifest.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd

from utils.loader import load_config

SITES = ("ebrains", "patho2")
GLIOMA_GROUPS = ("ast", "glm", "odg")
FAMILY_OF_GROUP = {"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"}


def _has_key(h5_path: str | None, key: str) -> bool:
    if not h5_path or not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return key in f
    except OSError:
        return False


def paper_rows(cfg: dict) -> list[dict]:
    """論文コホート: ラベル CSV の症例 × embedding.*.original の h5。WSI は paper_patches.wsi_index から引く。"""
    rows: list[dict] = []
    wsi_index = cfg.get("paper_patches", {}).get("wsi_index", {})
    for site in SITES:
        sub = pd.read_csv(cfg["label"][site]["subtype"], dtype=str)
        emb_dir = Path(cfg["embedding"][site]["original"])
        pat_csv = cfg["label"][site].get("patient")
        patients = (pd.read_csv(pat_csv, dtype=str).set_index("case_id")["patient_id"].to_dict()
                    if pat_csv and Path(pat_csv).exists() else {})
        wi = wsi_index.get(site, {})
        if "path_col" in wi:   # patho2: annotation.csv の case_id → wsi_path
            ann = pd.read_csv(wi["csv"], dtype=str)
            wsi_of = ann.set_index(wi["id_col"])[wi["path_col"]].to_dict()
        else:                  # ebrains: annotation.csv の uuid（先頭 8 文字が case_id）→ root/<diagnosis>/<uuid>.ndpi
            ann = pd.read_csv(wi["csv"], dtype=str).dropna(subset=["uuid", "diagnosis"])
            root = Path(wi["root"])
            wsi_of = {u[:8]: str(root / d / f"{u}.ndpi") for u, d in zip(ann["uuid"], ann["diagnosis"])}
        for r in sub.itertuples(index=False):
            is_sft = "sft" in str(r.subtype).lower()
            rows.append({
                "slide_id": r.case_id, "site": site, "family": "SFT" if is_sft else "meningioma",
                "group": "SFT" if is_sft else r.subtype, "diagnosis": r.subtype,
                "patient": patients.get(r.case_id, r.case_id), "wsi": wsi_of.get(r.case_id, ""),
                "h5_titan": str(emb_dir / f"{r.case_id}.h5"),
            })
    return rows


def glioma_rows(cfg: dict) -> list[dict]:
    """グリオーマ・DLBCL・転移: patho2 は代表 HE（representatives.csv × slides.csv）、ebrains は annotation.csv × EBRAINS_embeddings。"""
    rows: list[dict] = []
    pg, st = cfg["patho2_gliomas"], cfg["shift_transfer"]
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv", dtype=str)
    reps = pd.read_csv(pg["representatives"], dtype=str)
    reps = reps[reps["note"] != "exclude"]
    merged = reps.merge(slides[["label", "case", "file", "wsi", "h5"]], on=["case", "file"], how="left")
    for r in merged.itertuples(index=False):
        rows.append({
            "slide_id": r.case, "site": "patho2", "family": FAMILY_OF_GROUP[r.label], "group": r.label, "diagnosis": r.label,
            "patient": r.case, "wsi": r.wsi if isinstance(r.wsi, str) else "", "h5_titan": r.h5 if isinstance(r.h5, str) else "",
        })

    ann = pd.read_csv(st["ebrains_annotation"], dtype=str).dropna(subset=["uuid", "diagnosis"])
    ann = ann[ann["control"] != "1"]
    diag2group = {d: g for g, ds in st["ebrains_groups"].items() for d in ds}
    emb_root = Path(cfg["ebrains_embeddings"]["root"])
    wsi_root = Path(cfg["paper_patches"]["wsi_index"]["ebrains"]["root"])
    index: dict[str, Path] = {}
    for p in sorted(emb_root.rglob("*_hdf.h5"), key=lambda q: len(q.parts)):
        index.setdefault(p.name[: -len("_hdf.h5")], p)
    for r in ann[ann["diagnosis"].isin(diag2group)].itertuples(index=False):
        g = diag2group[r.diagnosis]
        rows.append({
            "slide_id": r.uuid[:8], "site": "ebrains", "family": FAMILY_OF_GROUP[g], "group": g, "diagnosis": r.diagnosis,
            "patient": str(r.pat_id), "wsi": str(wsi_root / r.diagnosis / f"{r.uuid}.ndpi"),
            "h5_titan": str(index[r.uuid]) if r.uuid in index else "",
        })
    return rows


def main() -> None:
    cfg = load_config()
    tcfg = cfg["transfer"]
    out_path = Path(cfg["output_dir"]) / tcfg.get("manifest", "transfer/manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(paper_rows(cfg) + glioma_rows(cfg))
    # PRISM（Virchow → PRISM）は既存の h5 に足す（ken 2026-10-26）: EBRAINS は EBRAINS_embeddings の h5（論文コホートの EBRAINS 分も
    # 山田さんの original ではなくこちら）、院内は TITAN と同じ h5（グリオーマ = Patho2Gliomas_embeddings、論文コホート = original）
    emb_root = Path(cfg["ebrains_embeddings"]["root"])
    by_prefix: dict[str, str] = {}
    for p in sorted(emb_root.rglob("*_hdf.h5"), key=lambda q: len(q.parts)):
        by_prefix.setdefault(p.name[:8], str(p))
    df["h5_prism"] = [by_prefix.get(i, "") if s == "ebrains" else h for s, i, h in zip(df["site"], df["slide_id"], df["h5_titan"])]
    print(f"checking {len(df)} h5 files ...")
    df["titan"] = [_has_key(p, tcfg["titan_key"]) for p in df["h5_titan"]]
    df["prism"] = [_has_key(p, tcfg["prism_key"]) for p in df["h5_prism"]]
    df["wsi_exists"] = [bool(p) and Path(p).exists() for p in df["wsi"]]
    df = df[["slide_id", "site", "family", "group", "diagnosis", "patient", "wsi", "wsi_exists", "h5_titan", "titan", "h5_prism", "prism"]]
    df.to_csv(out_path, index=False)

    fam_order = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
    summary = (df.assign(family=pd.Categorical(df["family"], fam_order))
                 .groupby(["family", "site"], observed=True)
                 .agg(slides=("slide_id", "size"), patients=("patient", "nunique"), wsi=("wsi_exists", "sum"),
                      titan=("titan", "sum"), prism=("prism", "sum")))
    print(summary.to_string())
    dup = df[df.duplicated(["site", "slide_id"], keep=False)]
    if len(dup):
        print(f"\nWARN duplicated slide_id within a site: {sorted(set(dup['slide_id']))[:10]} ...")
    print(f"\ntotal {len(df)} slides; TITAN missing {int((~df['titan']).sum())}, WSI missing {int((~df['wsi_exists']).sum())}, PRISM done {int(df['prism'].sum())}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
