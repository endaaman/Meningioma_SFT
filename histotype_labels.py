"""histotype_labels.py — 組織型（サブタイプ）分類用のラベル CSV を作る。

対象は config の lp_histotype.classes（両施設に 5 例以上ある主要な組織型）。
SFT、grade を規定する診断名（Atypical / Anaplastic）、希少な組織型・NOS はラベルを付けない（学習・評価から除外）。

入力: label.<site>.subtype（case_id, ..., subtype）
出力: lp_histotype.label_dir/<site>/case_histotype_n<件数>.csv（case_id, label）
      lp_histotype.label_dir/<site>/counts.csv（全 subtype の例数と採否）
クラス番号は classes の並び順（0 始まり）。lp.py --task histotype が読む。

クラス集合の変種（例: Transitional を除く no_tran）は config の lp_histotype_sets.<name> に置き、
--set <name> で lp_histotype の値を上書きする（classes / label_dir / output_subdir / preview_prefix）。

Usage:
    uv run python histotype_labels.py [--set no_tran]
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

from utils.loader import load_config


def apply_set(cfg: dict, name: str | None) -> dict:
    """lp_histotype を lp_histotype_sets.<name> で上書きした cfg を返す（name が None なら元のまま）。"""
    if not name:
        return cfg
    new = copy.deepcopy(cfg)
    new["lp_histotype"] = {**cfg["lp_histotype"], **cfg["lp_histotype_sets"][name]}
    return new


def add_set_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--set", default=None, help="lp_histotype_sets のクラス集合の変種名（例 no_tran）")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_set_arg(parser)
    args = parser.parse_args()
    cfg = apply_set(load_config(), args.set)
    h_cfg = cfg["lp_histotype"]
    classes: list[str] = h_cfg["classes"]
    index = {c: i for i, c in enumerate(classes)}
    root = Path(h_cfg["label_dir"])
    col_candidates = cfg.get("subtype_cols", ["subtype"])

    for site, l_cfg in cfg["label"].items():
        df = pd.read_csv(l_cfg["subtype"], dtype={"case_id": str})
        col = next(c for c in col_candidates if c in df.columns)
        out_dir = root / site
        out_dir.mkdir(parents=True, exist_ok=True)

        counts = df[col].value_counts().rename_axis("subtype").reset_index(name="n")
        counts["used"] = counts["subtype"].isin(classes)
        counts.to_csv(out_dir / "counts.csv", index=False)

        kept = df[df[col].isin(classes)]
        labels = pd.DataFrame({"case_id": kept["case_id"], "label": kept[col].map(index)})
        for old in out_dir.glob("case_histotype_n*.csv"):
            old.unlink()
        path = out_dir / f"case_histotype_n{len(labels)}.csv"
        labels.to_csv(path, index=False)
        print(f"[{site}] {len(labels)}/{len(df)} cases -> {path}")
        for c in classes:
            print(f"  {c:28s} {int((kept[col] == c).sum())}")


if __name__ == "__main__":
    main()
