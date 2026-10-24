"""patho2_gliomas_select.py — 院内グリオーマ（Patho2Gliomas）で 1 症例 1 枚の代表 HE を選ぶ。

1 症例に HE が複数あるとき全部使うと腫瘍の写っていない切片（辺縁など）を含めてしまうので、1 症例 1 枚にする（ken 2026-10-25）。
規則（patho2_gliomas_extract.py が書く out/patho2_gliomas/slides.csv の HE だけが対象）:
    1. 再切片 HE（RECUT）があればそれ。再切片は免疫染色のために選んだ腫瘍の確実なブロックから切るので、腫瘍が確実に載っていて切片もきれい。
       再切片が複数なら組織パッチ数が最大のもの。
    2. 無ければ組織パッチ数（n_patches、512 px / 0.5 mpp の組織パッチの数）が最大の HE。同数ならファイル名順で先頭（ブロック番号の小さい方）。
    3. 手動の上書き: config の patho2_gliomas.override（CSV: case, file）があればそれを優先する（腫瘍が写っていない等を見て差し替える用）。
症例の HE に抽出の済んでいないもの（status が done 以外）があれば、その症例は pending として選ばない（全部揃ってから比べる）。

出力: output_dir/patho2_gliomas/representatives.csv
    group, label, case, file, h5, n_patches, n_he, n_recut, reason（recut / max_patches / override）, status（selected / pending）

Usage:
    uv run python patho2_gliomas_select.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.loader import load_config


def select_case(df: pd.DataFrame, override_file: str | None) -> tuple[pd.Series | None, str]:
    """1 症例分の HE（slides.csv の行）から代表 1 行と理由を返す。揃っていなければ (None, 'pending')。"""
    if (df["status"] != "done").any() or df["n_patches"].isna().any():
        return None, "pending"
    if override_file is not None:
        hit = df[df["file"] == override_file]
        if len(hit) == 1:
            return hit.iloc[0], "override"
    pool = df[df["recut"]] if df["recut"].any() else df
    reason = "recut" if df["recut"].any() else "max_patches"
    pool = pool.sort_values(["n_patches", "file"], ascending=[False, True], kind="stable")
    return pool.iloc[0], reason


def main() -> None:
    cfg = load_config()
    pcfg = cfg["patho2_gliomas"]
    out_dir = Path(cfg["output_dir"]) / "patho2_gliomas"
    slides = pd.read_csv(out_dir / "slides.csv")
    he = slides[slides["is_he"]].copy()
    he["status"] = he["status"].fillna("")

    overrides: dict[str, str] = {}
    ov_path = pcfg.get("override")
    if ov_path and Path(ov_path).exists():
        ov = pd.read_csv(ov_path)
        overrides = dict(zip(ov["case"].astype(str), ov["file"].astype(str)))
        print(f"override: {len(overrides)} cases from {ov_path}")

    rows = []
    for (group, case), df in he.groupby(["group", "case"], sort=False):
        rep, reason = select_case(df, overrides.get(str(case)))
        base = {"group": group, "label": df["label"].iloc[0], "case": case, "n_he": len(df), "n_recut": int(df["recut"].sum())}
        if rep is None:
            rows.append({**base, "file": "", "h5": "", "n_patches": "", "reason": "", "status": "pending"})
        else:
            rows.append({**base, "file": rep["file"], "h5": rep["h5"], "n_patches": int(rep["n_patches"]), "reason": reason, "status": "selected"})
    reps = pd.DataFrame(rows)
    reps.to_csv(out_dir / "representatives.csv", index=False)

    summary = reps.groupby("group").agg(cases=("case", "size"), selected=("status", lambda s: int((s == "selected").sum())),
                                        recut=("reason", lambda s: int((s == "recut").sum())),
                                        max_patches=("reason", lambda s: int((s == "max_patches").sum())),
                                        override=("reason", lambda s: int((s == "override").sum())))
    print(summary.to_string())
    print(f"\n-> {out_dir / 'representatives.csv'} ({(reps['status'] == 'selected').sum()} selected, {(reps['status'] == 'pending').sum()} pending)")


if __name__ == "__main__":
    main()
