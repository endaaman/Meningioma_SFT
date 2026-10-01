"""dataset.py — データセット構成（症例数円グラフ と Table 1）。

出力 (output_dir/dataset/):
    pie_{source}.png / legend.png — サブタイプ構成の円グラフ
    table1.csv / table1.md        — 施設ごとのスライド数・患者数・SFT/髄膜腫・年齢性別・サブタイプ内訳

患者数は label.<source>.patient（case_id,patient_id）があればそれで数え、無ければ 1 症例 = 1 患者。
年齢・性別は dataset.demographics.<source> の CSV（EBRAINS annotation.csv 形式: uuid, pat_id, age, sex）
から患者単位で集計する（同一患者に複数スライドがあれば最も若い年齢＝初回標本を採る）。無い施設は「—」。

Usage:
    uv run python dataset.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_counts(cfg: dict) -> dict[str, pd.Series]:
    col_candidates = cfg.get("subtype_cols", ["subtype"])
    result = {}
    for ds_name, ds_cfg in cfg["label"].items():
        df = pd.read_csv(ds_cfg["subtype"])
        col = next((c for c in col_candidates if c in df.columns), None)
        if col is None:
            raise ValueError(f"[{ds_name}] subtype column not found in {ds_cfg['subtype']}")
        result[ds_name] = df[col].value_counts()
    return result


def build_piecharts(cfg: dict) -> None:
    shortened: dict[str, str] = cfg["display"]["shortened"]["subtypes"]
    colors_map: dict[str, str] = cfg["display"]["colors"]["subtypes"]
    sources: list[str] = list(cfg["label"].keys())

    counts = load_counts(cfg)
    out_dir = Path(cfg["output_dir"]) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in sources:
        series = counts[ds]
        st_list = [st for st in shortened if series.get(st, 0) > 0]
        values  = [int(series[st]) for st in st_list]
        colors  = [colors_map.get(st, "#aaaaaa") for st in st_list]
        abbrs   = [shortened[st] for st in st_list]
        total   = sum(values)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(values, colors=colors, startangle=90)
        ax.set_aspect("equal")
        fig.tight_layout(pad=0)

        out = out_dir / f"pie_{ds}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"保存先: {out}  (n={total})")

    # 凡例: 全 source に登場するサブタイプをまとめて別ファイル出力
    all_st = [st for st in shortened if any(counts[ds].get(st, 0) > 0 for ds in sources)]
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=colors_map.get(st, "#aaaaaa"))
        for st in all_st
    ]
    legend_texts = [f"{shortened[st]}  {st}" for st in all_st]

    fig, ax = plt.subplots(figsize=(4, len(all_st) * 0.28 + 0.3))
    ax.set_axis_off()
    ax.legend(patches, legend_texts, loc="center", fontsize=8, frameon=False,
              handlelength=1.2, handleheight=1.0, borderpad=0)
    fig.tight_layout(pad=0.2)

    out = out_dir / "legend.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"保存先: {out}")


# ── Table 1 ───────────────────────────────────────────────────────────────────

def _load_cases(cfg: dict, ds: str) -> pd.DataFrame:
    """case_id / subtype / is_sft / patient_id の表。"""
    ds_cfg = cfg["label"][ds]
    col_candidates = cfg.get("subtype_cols", ["subtype"])
    df = pd.read_csv(ds_cfg["subtype"], dtype={"case_id": str})
    col = next(c for c in col_candidates if c in df.columns)
    df = df.rename(columns={col: "subtype"})[["case_id", "subtype"]]
    df["is_sft"] = df["subtype"].str.lower().str.contains("sft")
    if ds_cfg.get("patient"):
        pts = pd.read_csv(ds_cfg["patient"], dtype=str)
        df = df.merge(pts, on="case_id", how="left")
        if df["patient_id"].isna().any():
            raise ValueError(f"[{ds}] patient_id missing for {df['patient_id'].isna().sum()} cases")
    else:
        df["patient_id"] = df["case_id"]
    return df


def _demographics(cfg: dict, ds: str, cases: pd.DataFrame) -> dict[str, str]:
    path = cfg.get("dataset", {}).get("demographics", {}).get(ds)
    if not path:
        return {"age": "—", "sex": "—"}
    ann = pd.read_csv(path)
    ann["case_id"] = ann["uuid"].str[:8]
    d = cases[["case_id"]].merge(ann[["case_id", "pat_id", "age", "sex"]], on="case_id", how="left")
    per_pt = d.sort_values("age").groupby("pat_id").agg(age=("age", "first"), sex=("sex", "first"))
    q1, med, q3 = np.nanquantile(per_pt["age"], [0.25, 0.5, 0.75])
    n_m = int((per_pt["sex"] == "male").sum())
    n_f = int((per_pt["sex"] == "female").sum())
    return {"age": f"{med:.0f} [{q1:.0f}–{q3:.0f}]", "sex": f"{n_m} / {n_f}"}


def build_table1(cfg: dict) -> pd.DataFrame:
    shortened: dict[str, str] = cfg["display"]["shortened"]["subtypes"]
    sources: list[str] = list(cfg["label"].keys())
    cases = {ds: _load_cases(cfg, ds) for ds in sources}

    rows: list[tuple[str, dict[str, str]]] = []
    rows.append(("Slides", {ds: f"{len(c)}" for ds, c in cases.items()}))
    rows.append(("Patients", {ds: f"{c['patient_id'].nunique()}" for ds, c in cases.items()}))
    for name, m in (("SFT", True), ("Meningioma", False)):
        rows.append((f"{name}, slides / patients", {
            ds: f"{int((c['is_sft'] == m).sum())} / {c.loc[c['is_sft'] == m, 'patient_id'].nunique()}"
            for ds, c in cases.items()}))
    demo = {ds: _demographics(cfg, ds, c) for ds, c in cases.items()}
    rows.append(("Age, median [IQR] (patients)", {ds: demo[ds]["age"] for ds in sources}))
    rows.append(("Sex, male / female (patients)", {ds: demo[ds]["sex"] for ds in sources}))
    # サブタイプ内訳（スライド数）。表示順は config の順
    all_st = [st for st in shortened if any((c["subtype"] == st).any() for c in cases.values())]
    others = sorted(set().union(*[set(c["subtype"]) for c in cases.values()]) - set(all_st))
    rows.append(("Diagnosis / subtype (slides)", {ds: "" for ds in sources}))
    for st in all_st + others:
        rows.append((f"  {st}", {ds: f"{int((c['subtype'] == st).sum())}" for ds, c in cases.items()}))

    table = pd.DataFrame([{"item": item, **vals} for item, vals in rows])
    # 両方のクラスに属する患者（同一患者に SFT と髄膜腫の診断）を脚注用に数える
    table.attrs["both_class_patients"] = {
        ds: sorted(set(c.loc[c["is_sft"], "patient_id"]) & set(c.loc[~c["is_sft"], "patient_id"]))
        for ds, c in cases.items()}
    return table


def save_table1(table: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table1.csv", index=False)
    sources = [c for c in table.columns if c != "item"]
    lines = ["| | " + " | ".join(sources) + " |", "|---|" + "---|" * len(sources)]
    for _, r in table.iterrows():
        item = r["item"]
        item = f"&emsp;{item.strip()}" if item.startswith("  ") else item
        lines.append(f"| {item} | " + " | ".join(str(r[s]) for s in sources) + " |")
    notes = []
    for ds, pts in table.attrs.get("both_class_patients", {}).items():
        if pts:
            notes.append(f"{ds}: {len(pts)} 名（patient_id {', '.join(pts)}）は SFT と髄膜腫の両方の"
                         "スライドを持ち、両方の行に数えている。")
    notes.append("サブタイプ内訳はスライド数。年齢・性別は患者単位（複数スライドの患者は最も若い年齢）。")
    md = "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    (out_dir / "table1.md").write_text(md)
    print(f"保存先: {out_dir / 'table1.csv'} / table1.md")


if __name__ == "__main__":
    cfg = load_config()
    build_piecharts(cfg)
    t1 = build_table1(cfg)
    print(t1.to_string(index=False))
    save_table1(t1, Path(cfg["output_dir"]) / "dataset")
