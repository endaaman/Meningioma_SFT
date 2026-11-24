"""cohort_table.py — Table 1 用: 施設 × 群の枚数と患者数（転移テストの manifest、症例の表を当てた後）。

群: 髄膜腫のサブタイプ（両施設にあるもの、片方の施設にしか無いもの）、SFT、グリオーマ 3 群（ast / glm / odg）、リンパ腫、転移。
科ごとの小計と全体の行も付ける。患者数は manifest の patient 列（EBRAINS は pat_id、院内の髄膜腫+SFT コホートは label の患者表、
院内のグリオーマ・リンパ腫・転移は症例番号）の重複を除いた数。

出力: output_dir/cohort_table/table1_counts.csv
    family, group, display（表示名）, both_sites（両施設にあるか）, ebrains_slides, ebrains_patients, patho2_slides, patho2_patients

Usage:
    uv run python cohort_table.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.display import ordered_subtypes
from utils.loader import load_config

FAMILY_ORDER = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
GLIOMA_ORDER = ["ast", "glm", "odg"]


def main() -> None:
    cfg = load_config()
    ref = cfg["reference"]
    out = Path(cfg["output_dir"])
    man = pd.read_csv(out / cfg["transfer"].get("manifest", "transfer/manifest.csv"), dtype={"slide_id": str, "patient": str})
    oth = next(s for s in man["site"].unique() if s != ref)
    names = cfg.get("shift_transfer", {}).get("group_names", {})
    rows = []

    def row(family: str, group: str, display: str, d: pd.DataFrame) -> dict:
        e, p = d[d["site"] == ref], d[d["site"] == oth]
        return {"family": family, "group": group, "display": display, "both_sites": bool(len(e) and len(p)),
                f"{ref}_slides": len(e), f"{ref}_patients": e["patient"].nunique(), f"{oth}_slides": len(p), f"{oth}_patients": p["patient"].nunique()}

    for f in FAMILY_ORDER:
        d = man[man["family"] == f]
        groups = ordered_subtypes(set(d["group"]), cfg) if f == "meningioma" else (GLIOMA_ORDER if f == "glioma" else sorted(set(d["group"])))
        if f == "meningioma":   # 両施設にあるサブタイプを先に、片方だけのものを後に
            both = [g for g in groups if d[(d["group"] == g)]["site"].nunique() == 2]
            groups = both + [g for g in groups if g not in both]
        if len(groups) > 1:
            rows += [row(f, g, names.get(g, g), d[d["group"] == g]) for g in groups]
        rows.append(row(f, "(all)", {"meningioma": "Meningioma (all)", "SFT": "SFT", "glioma": "Glioma (all)", "lymphoma": names.get("lym", "lym"),
                                    "metastasis": names.get("met", "met")}[f], d))
    rows.append(row("(all)", "(all)", "Total", man))
    tab = pd.DataFrame(rows)
    path = out / "cohort_table" / "table1_counts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(path, index=False)
    print(tab.to_string(index=False))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
