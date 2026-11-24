"""transfer_manifest.py — 転移テスト（3 法 × 推定元 × 当て先）で使う全スライドの一覧を作り、埋め込みの揃い具合を確認する。

対象（2 施設 × 5 科）:
    meningioma / SFT : 髄膜腫+SFT コホート（config の label.*.subtype と embedding.*.original。EBRAINS n551、patho2 n577）
    glioma (ast / glm / odg) / lymphoma (DLBCL) / metastasis:
        patho2  = Patho2Gliomas の代表 HE 1 症例 1 枚（patho2_gliomas.representatives → out/patho2_gliomas/slides.csv）。
                  症例の表（patho2_gliomas.cohort、cohort_enabled が true のとき。utils/cohort.py）の include に絞り、群を表の group にする
                  （無効・表が無いときは代表 HE の全 260 例をフォルダの群で）
        ebrains = annotation.csv の diagnosis を shift_transfer.ebrains_groups で科に対応づけ（control は除く）、EBRAINS_embeddings の h5

列: slide_id, site, family, group（科の中の群: 髄膜腫サブタイプ / ast / glm / odg / lym / met）, diagnosis（元のラベル）,
    patient, wsi（元 WSI。Virchow 抽出に使う）, h5_titan（TITAN が入っている h5）, titan（あるか）, h5_prism, prism（あるか）

出力: output_dir/transfer/manifest.csv と、科 × 施設ごとの枚数・患者数・埋め込みの有無の要約（標準出力）。

症例の表を作る（--build-cohort、1 回だけ。既にあれば --force が無いと止まる。ken 決定 2026-11-24）: 代表 HE の 260 例について
    1. 過去の論文（Enda K, et al. Pathol Int 2026;76(2):e70098）の症例に限る（status = include、群はフォルダの群のまま）。症例の
       一覧は patho2_gliomas.past_paper_tiles（{A,G,O,L,M}/<症例名>/ のタイル置き場）から作るときに 1 回だけ読む。無い症例は exclude
       過去の論文にあって代表 HE がまだ無い症例も include で載せる（代表が選ばれれば自動で解析に入る）。表に無い症例（過去の論文に
       無い新しい症例）は解析で外れる（utils.cohort.apply_cohort）
    2. 個別の除外・割り付け変更はしない。再確認で疑問が付いた 7 例（FLAGGED）は flag に理由を書く（include のまま。
       patho2_gliomas.exclude_flagged: true のときだけ外す = 感度分析）
    → patho2_gliomas.cohort（列は utils/cohort.COHORT_COLS）。

Usage:
    uv run python transfer_manifest.py [--build-cohort [--force]]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import pandas as pd

from utils.cohort import COHORT_COLS, apply_cohort, cohort_table
from utils.loader import load_config

SITES = ("ebrains", "patho2")
GLIOMA_GROUPS = ("ast", "glm", "odg")
FAMILY_OF_GROUP = {"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"}
# 症例の表の規則（ken 決定 2026-11-24「これだけ除く意義は薄い。元の論文からコホート操作すると切りがない」）:
# 過去の論文の症例を群もそのまま使う。個別の除外・割り付け変更はしない。再確認で疑問が付いた症例は flag に理由を書くだけ
# （patho2_gliomas.exclude_flagged: true で外した一覧になる = 感度分析用）
PAST_PAPER = "過去の論文（Enda K, et al. Pathol Int 2026;76(2):e70098）"
FLAGGED = {
    "N21-134": "IDH 野生型の星細胞腫。膠芽腫の所見（壊死・微小血管増生）が乏しく、TERT 未検査",
    "N21-163": "IDH 野生型の星細胞腫。膠芽腫の所見（壊死・微小血管増生）が乏しく、TERT 未検査（小脳）",
    "N21-170": "型の違う病変が 2 つ（乏突起膠腫と星細胞腫。症例表に「2つは別病変」）",
    "N17-075": "群 A だが報告書は 1p/19q 共欠失の乏突起膠腫",
    "18-3306": "乏突起膠腫の疑い（報告は Oligodendroglioma, suspected）。1p/19q 未確認",
    "N21-202": "小児例（IDH・H3 野生型、類上皮型）。CNS5 の膠芽腫 IDH 野生型は成人型",
    "20-3590": "症例表で壊死・微小血管増生なし、表の診断は未確定（最終報告は膠芽腫）",
}


def _has_key(h5_path: str | None, key: str) -> bool:
    if not h5_path or not Path(h5_path).exists():
        return False
    try:
        with h5py.File(h5_path, "r") as f:
            return key in f
    except OSError:
        return False


def paper_rows(cfg: dict) -> list[dict]:
    """髄膜腫+SFT コホート: ラベル CSV の症例 × embedding.*.original の h5。WSI は paper_patches.wsi_index から引く。"""
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
    merged = apply_cohort(merged, cfg, case_col="case", group_cols=["label"])   # 症例の表（無効なら 260 例のまま）
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


def build_cohort(cfg: dict, force: bool) -> None:
    """症例の表を作る（docstring の規則）。過去の論文の症例一覧（past_paper_tiles）はここで 1 回だけ読む。"""
    pg = cfg["patho2_gliomas"]
    out = Path(pg["cohort"])
    if out.exists() and not force:
        raise SystemExit(f"{out} は既にある（作り直すなら --force）")
    root = Path(os.path.expanduser(pg["past_paper_tiles"]))
    past = {(g, c) for g in pg["groups"] if (root / g).is_dir() for c in os.listdir(root / g) if (root / g / c).is_dir()}
    print(f"past paper ({root}): " + ", ".join(f"{g} {sum(1 for x, _ in past if x == g)}" for g in pg["groups"]) + f"  total {len(past)}")
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv", dtype=str)
    reps = pd.read_csv(pg["representatives"], dtype=str)
    reps = reps[reps["note"] != "exclude"].merge(slides[["group", "label", "case", "file"]], on=["case", "file"], how="left")
    rows = []
    for r in reps.itertuples(index=False):
        status, reason = "include", f"{PAST_PAPER}の症例"
        if (r.group, r.case) not in past:
            status, reason = "exclude", f"{PAST_PAPER}の症例に無い" + (
                f"（{PAST_PAPER}では群 {next(g for g, c in past if c == r.case)}）" if any(c == r.case for _, c in past) else "")
        rows.append({"case": r.case, "folder_group": r.group, "label": r.label, "group": r.label, "status": status, "reason": reason,
                     "flag": FLAGGED.get(r.case, "")})
    # 過去の論文にあって代表 HE がまだ無い症例も include で載せておく（representatives.csv に入れば自動で解析に入る）
    have = {r["case"] for r in rows}
    for g, c in sorted(past, key=lambda x: (list(pg["groups"]).index(x[0]), x[1])):
        if c not in have:
            rows.append({"case": c, "folder_group": g, "label": pg["groups"][g], "group": pg["groups"][g], "status": "include",
                         "reason": f"{PAST_PAPER}の症例（代表 HE は未選択）", "flag": FLAGGED.get(c, "")})
    tbl = pd.DataFrame(rows, columns=COHORT_COLS)
    for c in FLAGGED:
        if c not in set(tbl["case"]):
            print(f"  WARN flag の {c} は代表 HE の症例に無い")
        elif tbl.loc[tbl["case"] == c, "status"].iloc[0] != "include":
            print(f"  WARN flag の {c} は{PAST_PAPER}の症例に無い（exclude）")
    out.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out, index=False)
    inc = tbl[tbl["status"] == "include"]
    print(f"cases {len(tbl)} (representatives {len(have)}): include {len(inc)} / exclude {int((tbl['status'] == 'exclude').sum())}")
    print("  folder × status:\n" + tbl.groupby(["label", "status"]).size().unstack(fill_value=0).to_string())
    print("  included by group: " + inc["group"].value_counts().reindex(list(FAMILY_OF_GROUP)).to_string().replace("\n", ", "))
    fl = inc[inc["flag"] != ""]
    print(f"  flagged (include; excluded only with exclude_flagged): {len(fl)} " + ", ".join(f"{c}（{g}）" for c, g in zip(fl["case"], fl["group"])))
    past_only = sorted(f"{g}/{c}" for g, c in past if c not in have)
    print(f"  past-paper cases not among the representatives ({len(past_only)}, listed as include): {past_only}")
    print(f"saved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-cohort", action="store_true", help="症例の表（patho2_gliomas.cohort）を作って終わる")
    ap.add_argument("--force", action="store_true", help="--build-cohort で既存の表を上書きする")
    args = ap.parse_args()
    cfg = load_config()
    if args.build_cohort:
        build_cohort(cfg, args.force)
        return
    tbl = cohort_table(cfg)
    print("cohort: " + ("disabled（代表 HE の全症例をフォルダの群で）" if tbl is None else
                        f"{cfg['patho2_gliomas']['cohort']}（include {int((tbl['status'] == 'include').sum())} / {len(tbl)} 例）"))
    tcfg = cfg["transfer"]
    out_path = Path(cfg["output_dir"]) / tcfg.get("manifest", "transfer/manifest.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(paper_rows(cfg) + glioma_rows(cfg))
    # PRISM（Virchow → PRISM）は既存の h5 に足す（ken 2026-10-26）: EBRAINS は EBRAINS_embeddings の h5（髄膜腫+SFT コホートの EBRAINS 分も
    # 山田さんの original ではなくこちら）、院内は TITAN と同じ h5（グリオーマ = Patho2Gliomas_embeddings、髄膜腫+SFT コホート = original）
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
