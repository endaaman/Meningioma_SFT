"""v4 Supplementary の表: 髄膜腫のサブタイプ別の枚数と患者数（Table 1 から分けた）。

入力: output_dir/cohort_table/table1_counts.csv（cohort_table.py）
出力: tables/{table Sn}_meningioma_subtype_counts.{csv,md}

両施設にあるサブタイプ（樹形図の 12 組）を先に、片方の施設にしか無いものを後に。値は「枚数（患者数）」、無い施設は 0。

Usage:
    uv run python -m figs.meningioma_subtype_counts
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import config, table_path
from figs.cohort_tumours import SITES, cell, counts

NAME = "meningioma_subtype_counts"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    t = counts(cfg)
    m = t[t["family"] == "meningioma"]
    rows = [{"subtype": r.display if r.group != "(all)" else "Meningioma (all)", "both_sites": bool(r.both_sites),
             **{name: cell(pd.Series(r._asdict()), s) for s, name in SITES}} for r in m.itertuples()]
    table = pd.DataFrame(rows)
    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    lines = ["| Subtype | EBRAINS, slides (patients) | In-house, slides (patients) |", "|---|---|---|"]
    for r in rows:
        name = r["subtype"] if r["both_sites"] or r["subtype"] == "Meningioma (all)" else f"{r['subtype']} †"
        lines.append(f"| {name} | {r['EBRAINS']} | {r['In-house']} |")
    n_both = int(m[(m["group"] != "(all)") & m["both_sites"]].shape[0])
    notes = [f"両施設にあるサブタイプ {n_both} 組が樹形図の組（Fig 2・4）。† = 片方の施設にしか無い（樹形図と組の数には入らない）。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
