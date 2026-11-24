"""v4 Table 1: コホート（腫瘍 × 施設の枚数と患者数）。髄膜腫のサブタイプ別は Supplementary の表（figs.meningioma_subtype_counts）。

入力: output_dir/cohort_table/table1_counts.csv（cohort_table.py。転移テストの manifest = 249 例版、2,290 枚）
出力: tables/{table n}_cohort_tumours.{csv,md}（番号は figs/__init__.py）

表示名は utils/display.py（FAMILY_LABELS・GLIOMA_GROUP_LABELS）。値は「枚数（患者数）」。

Usage:
    uv run python -m figs.cohort_tumours
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import config, out_root, table_path
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_LABELS

NAME = "cohort_tumours"
SITES = [("ebrains", "EBRAINS"), ("patho2", "In-house")]


def counts(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(out_root(cfg) / "cohort_table" / "table1_counts.csv")


def cell(r: pd.Series, site: str) -> str:
    return f"{int(r[f'{site}_slides']):,} ({int(r[f'{site}_patients']):,})"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    t = counts(cfg)
    rows = []
    for fam in ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]:
        r = t[(t["family"] == fam) & (t["group"] == "(all)")].iloc[0]
        item = FAMILY_LABELS[fam]
        if fam == "meningioma":
            n_both = int(t[(t["family"] == fam) & (t["group"] != "(all)") & t["both_sites"]].shape[0])
            item += f" ({n_both} subtypes at both sites)"
        rows.append({"tumour": item, **{name: cell(r, s) for s, name in SITES}})
        if fam == "glioma":
            for g in ("ast", "glm", "odg"):
                rg = t[(t["family"] == fam) & (t["group"] == g)].iloc[0]
                rows.append({"tumour": f"  {GLIOMA_GROUP_LABELS[g]}", **{name: cell(rg, s) for s, name in SITES}})
    tot = t[t["family"] == "(all)"].iloc[0]
    rows.append({"tumour": "Total", **{name: cell(tot, s) for s, name in SITES}})
    table = pd.DataFrame(rows)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)

    fam_sum = int(t[(t["group"] == "(all)") & (t["family"] != "(all)")]["ebrains_patients"].sum())
    lines = ["| Tumour | EBRAINS, slides (patients) | In-house, slides (patients) |", "|---|---|---|"]
    for r in table.itertuples():
        item = f"&emsp;{r.tumour.strip()}" if r.tumour.startswith("  ") else r.tumour
        lines.append(f"| {item} | {r.EBRAINS} | {getattr(r, '_3')} |")
    notes = [f"EBRAINS の患者の合計 {int(tot['ebrains_patients']):,} は腫瘍ごとの和 {fam_sum:,} より 1 人少ない"
             "（1 人が髄膜腫と乏突起膠腫の両方のスライドを持つ）。",
             "院内は 1 症例 1 枚（グリオーマ・リンパ腫・転移は過去の論文の 249 例の代表 HE）。髄膜腫のサブタイプ別は Supplementary の表。",
             "脚注に入れるもの（原稿側）: 診断基準（院内 CNS5、EBRAINS WHO 2016 を CNS5 の群に割り付け）、診断の期間、スキャナ。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
