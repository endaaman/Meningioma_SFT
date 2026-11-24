"""v4 Supplementary の表: EBRAINS 側の構成を変えた感度（TITAN）。院内の群は固定し、対応させる EBRAINS の診断の集め方を変えたときの、
その群の施設のずれ（長さと、髄膜腫だけのずれとの cos）。

行 = 院内の群（星細胞腫 IDH 変異型、膠芽腫 IDH 野生型、乏突起膠腫、グリオーマ全体）× EBRAINS 側の変種（baseline = 本文の割り付け）。
列 = EBRAINS の枚数（患者数）、院内の枚数、ずれの長さ ‖Δ‖（生 / 推定の雑音を差し引いた値）、髄膜腫だけのずれとの cos（生 / 雑音を差し
引いた値）、EBRAINS の診断名の内訳。
基準は髄膜腫だけのずれ（composition.csv の *_vs_meni の列。reference_meningioma/ には composition の出力が無いので既定の出力の列を使う）。
cos の 95% 区間は composition.csv には髄膜腫＋SFT の基準の分しか無いので載せない（新しい解析はしない）。

入力: output_dir/shift_transfer/composition.csv（shift_transfer.py）
出力: tables/{table Sn}_ebrains_composition.{csv,md}

Usage:
    uv run python -m figs.ebrains_composition
"""
from __future__ import annotations

import re

import pandas as pd

from figs import label
from figs.common import config, out_root, table_path
from utils.display import GLIOMA_GROUP_LABELS

NAME = "ebrains_composition"
GROUP_LABEL = {"glioma (all)": "Glioma (all)"}


def _variant(v: str) -> str:
    v = re.sub(r"\s*\(config\)", "", v).strip().replace("ast + glm + odg", "the three glioma groups")
    return "Baseline" + v[len("baseline"):] if v.startswith("baseline") else v


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    c = pd.read_csv(out_root(cfg) / "shift_transfer" / "composition.csv")
    rows = []
    for r in c.itertuples():
        rows.append({"In-house group": GROUP_LABEL.get(r.patho2_group, GLIOMA_GROUP_LABELS.get(r.patho2_group, r.patho2_group)),
                     "EBRAINS composition": _variant(r.variant),
                     "EBRAINS slides (patients)": f"{r.n_ebrains} ({r.n_pat_ebrains})", "In-house slides": r.n_patho2,
                     "Shift length, raw": round(r.shift_norm, 2), "Shift length, noise-corrected": round(r.shift_norm_corrected, 2),
                     "cos with meningioma shift, raw": round(r.cos_vs_meni, 3),
                     "cos with meningioma shift, noise-corrected": round(r.cos_corrected_vs_meni, 3),
                     "EBRAINS diagnoses (slides)": r.ebrains_diagnoses, "WHO 2016 grades (slides)": r.ebrains_grades})
    t = pd.DataFrame(rows)
    t.to_csv(table_path(cfg, NAME, ".csv"), index=False)
    show = [c for c in t.columns if c != "EBRAINS diagnoses (slides)"]
    lines = ["| " + " | ".join(show) + " |", "|" + "|".join(["---"] * len(show)) + "|"]
    prev = None
    for r in rows:
        cells = [(f"{r[k]:.3f}" if k.startswith("cos") else f"{r[k]:.2f}") if isinstance(r[k], float) else str(r[k]) for k in show]
        if cells[0] == prev:
            cells[0] = ""
        else:
            prev = cells[0]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ["TITAN。院内の群は固定し、EBRAINS 側で対応させる診断名の集め方だけを変えた。Baseline は本文の割り付け"
             "（config の shift_transfer.ebrains_groups）。(c) の IDH 野生型の（退形成性）星細胞腫は CNS5 では分子所見で膠芽腫に当たりうる群。",
             "Shift length = 2 施設の平均の差 ‖Δ‖（埋め込みの長さの単位）。noise-corrected = 平均の推定の雑音（各施設の分散 / 枚数の和）を差し引いた値。",
             "cos with meningioma shift = その群のずれと、髄膜腫だけ（SFT を除く）のずれのコサイン類似度。noise-corrected は長さを雑音で補正した値。",
             "EBRAINS の診断名ごとの枚数は CSV の列 EBRAINS diagnoses (slides) にある。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}")
    print(t[show[:2] + show[4:8]].to_string(index=False))


if __name__ == "__main__":
    main()
