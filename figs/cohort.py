"""データセットの表（施設ごとのスライド数・患者数・SFT/髄膜腫・年齢性別・サブタイプ内訳）。

入力: output_dir/dataset/table1.csv / table1_notes.json  （dataset.py）
出力: tables/{table n}_cohort.{csv,md}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.cohort
"""
from __future__ import annotations

import json

import pandas as pd

from figs import label
from figs.common import config, out_root, table_path

NAME = "cohort"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    src = out_root(cfg) / "dataset"
    table = pd.read_csv(src / "table1.csv", dtype=str, keep_default_na=False)
    notes_in = json.loads((src / "table1_notes.json").read_text())

    table.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    sources = [c for c in table.columns if c != "item"]
    lines = ["| | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for _, r in table.iterrows():
        item = r["item"]
        item = f"&emsp;{item.strip()}" if item.startswith("  ") else item
        lines.append(f"| {item} | " + " | ".join(str(r[s]) for s in sources) + " |")
    notes = []
    for ds, pts in notes_in.get("both_class_patients", {}).items():
        if pts:
            notes.append(f"{ds}: {len(pts)} 名（patient_id {', '.join(map(str, pts))}）は SFT と髄膜腫の両方の"
                         "スライドを持ち、両方の行に数えている。")
    notes.append("サブタイプ内訳はスライド数。年齢・性別は患者単位（複数スライドの患者は最も若い年齢）。")
    md = "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    path = table_path(cfg, NAME, ".md")
    path.write_text(md)
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
