"""v4 Supplementary の表: 撮像と標本の条件（事実の整理）。acquisition_table.py の出力を表にするだけ。

行 = 腫瘍 × 施設。列 = 枚数（患者数）、1 患者あたりの枚数、スキャナ（WSI のヘッダの型番のまま。製品名に読み替えない）、対物の倍率、
撮像の µm/px、パッチを切った µm/px、撮像の年、院内の症例番号の年、パッチ数と組織面積の中央値 [四分位]。記録が無いものは「記録なし」。

入力: output_dir/acquisition/acquisition_table.csv（acquisition_table.py）
出力: tables/{table Sn}_acquisition_conditions.{csv,md}

Usage:
    uv run python -m figs.acquisition_conditions
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import config, out_root, table_path
from utils.display import FAMILY_LABELS

NAME = "acquisition_conditions"
COLS = [("Slides", "Slides"), ("Patients", "Patients"), ("Slides per patient", "Slides per patient"),
        ("Scanner (tiff.Make tiff.Model)", "Scanner model (as recorded)"), ("Scanner software", "Scanner software"),
        ("Objective (×)", "Objective"), ("Scan µm/px", "Scan µm/px"), ("Patch µm/px", "Patch µm/px"),
        ("Scan year (tiff.DateTime)", "Scan year"), ("Case-number year (in-house ID)", "Case-number year"),
        ("Patches (512 px), median [IQR]", "Patches, median [IQR]"), ("Tissue area mm², median [IQR]", "Tissue area mm², median [IQR]")]
TUMOUR = {"meningioma": "Meningioma", "SFT": "SFT", "glioma": "Glioma", "lymphoma": FAMILY_LABELS["lymphoma"], "metastasis": "Metastasis"}


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    a = pd.read_csv(out_root(cfg) / "acquisition" / "acquisition_table.csv")
    t = pd.DataFrame({"Tumour": a["Tumour"].map(TUMOUR), "Site": a["Site"], **{new: a[old] for old, new in COLS}})
    t["Objective"] = t["Objective"].str.replace(".0 (", "× (", regex=False)
    t.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    prev = None
    for _, r in t.iterrows():
        cells = [str(r[c]) for c in cols]
        if cells[0] == prev:
            cells[0] = ""
        else:
            prev = cells[0]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ["スキャナの型番・ソフト・対物の倍率・撮像の µm/px・撮像の日時は、各 WSI（NDPI）の TIFF タグ（Make / Model / Software / "
             "Magnification / XResolution / DateTime）をそのまま数えた（括弧は枚数）。型番から製品名への読み替えはしていない。",
             "Patch µm/px = パッチ（512 px）を切った段の解像度（目標 0.5 µm/px に最も近い段）。組織面積 = 組織と判定されたパッチの面積の和。",
             "Case-number year = 院内の症例番号の先頭 2 桁（受付の年）。EBRAINS の ID は年を含まず、診断の年の記録は手元に無い（記録なし）。",
             "標本を作った施設、固定・染色の手順は、手元の記録に無い（記録なし）。院内は 1 患者 1 枚、EBRAINS は配布されたその患者の全スライド。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")


if __name__ == "__main__":
    main()
