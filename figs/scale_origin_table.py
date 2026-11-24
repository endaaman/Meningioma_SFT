"""v4 Supplementary の表: 倍率を掛ける中心の全数値（TITAN、Fig 5e の全部）。

1 行 = 当て先 × 推定元（当て先 4 つ: グリオーマ、髄膜腫、リンパ腫、転移。推定元は同じ腫瘍以外）。列 = 借りた倍率 s と当て先自身の s、
5 つの変種ごとの 2 施設の中心の距離 / 近傍の混ざり / 隣り合う組:
    平行移動を借りる / 相似変換を借りる（推定元の院内平均を中心に広げる）/ ComBat を借りる /
    借りた倍率で、広げる中心だけ当て先の院内スライド全体の平均にしたもの / 倍率も当て先自身の値にしたもの（中心は当て先の院内平均）
中心の距離は群の中央値、近傍の混ざりは両施設 5 枚以上の群の中央値（本文の約束。out/_diagnostics/scale_origin/groups.csv から集計）、
組の数はグリオーマ（3 組）と髄膜腫（12 組）だけ。
md には別表で、借りた相似変換で中心の差の二乗が増えた分のうち −(s−1)(μ_P,g − μ_P,src) の項の二乗で説明される割合
（推定元 × 群の和。transfer_scale_origin.py の summary.md の「分解」と同じ集計）。

入力: output_dir/_diagnostics/scale_origin/{summary,groups,decomposition}.csv（transfer_scale_origin.py）
出力: tables/{table Sn}_scale_origin_table.{csv,md}、同 _decomposition.csv

Usage:
    uv run python -m figs.scale_origin_table
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from figs import label
from figs.common import config, out_root, table_path
from figs.transfer_corrections_all import GROUP_LABEL

NAME = "scale_origin_table"
MIN_N = 5
TARGETS = ["glioma", "meningioma", "lymphoma", "metastasis"]
VARIANTS = [("translation", "Translation, borrowed"), ("affine_borrowed", "Similarity, borrowed"), ("combat_borrowed", "ComBat, borrowed"),
            ("affine_tgt_center", "Borrowed s, centred at target in-house mean"), ("ownscale_tgt_center", "Target's own s, centred at target in-house mean")]
METRICS = [("group_gap_sd", "centre_distance"), ("knn5", "neighbour_mixing"), ("pairs_adjacent", "cross_site_pairs")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    d = out_root(cfg) / "_diagnostics" / "scale_origin"
    s = pd.read_csv(d / "summary.csv")
    g = pd.read_csv(d / "groups.csv")
    k = g[(g["n_ebrains"] >= MIN_N) & (g["n_patho2"] >= MIN_N)].groupby(["target", "source", "variant"])["knn_mix"].median().rename("knn5")
    s = s.join(k, on=["target", "source", "variant"]).set_index(["target", "source", "variant"])
    rows = []
    for tg in TARGETS:
        srcs = [x for x in pd.unique(s.loc[tg].index.get_level_values("source")) if x != "own"]
        for src in srcs:
            r0 = s.loc[(tg, src, "translation")]
            row = {"target": tg, "target_label": GROUP_LABEL[tg], "source": src, "source_label": GROUP_LABEL[src],
                   "s_borrowed": r0["s_src"], "s_target": r0["s_tgt"], "n_pairs": r0["pairs_n"]}
            for v, _ in VARIANTS:
                r = s.loc[(tg, src, v)]
                for m, col in METRICS:
                    row[f"{v}__{col}"] = r[m]
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(table_path(cfg, NAME, ".csv"), index=False, float_format="%.4f")

    head = ["Target ← source", "s borrowed / target"] + [lab for _, lab in VARIANTS]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in t.itertuples():
        cells = [f"{r.target_label} ← {r.source_label}", f"{r.s_borrowed:.3f} / {r.s_target:.3f}"]
        for v, _ in VARIANTS:
            gd, kn, pr = getattr(r, f"{v}__centre_distance"), getattr(r, f"{v}__neighbour_mixing"), getattr(r, f"{v}__cross_site_pairs")
            cells.append(f"{gd:.3f} / {kn:.2f}" + ("" if pd.isna(pr) else f" / {int(pr)}/{int(r.n_pairs)}"))
        lines.append("| " + " | ".join(cells) + " |")

    dec = pd.read_csv(d / "decomposition.csv")
    dd = dec[dec["source"] != "own"]
    agg = dd.groupby("target").agg(pairs_x_groups=("group", "size"), increase_sq=("diff_sq_aff", "sum"),
                                   term_sq=("e_aff", lambda v: float((v ** 2).sum())), cross=("cross_aff", "sum"))
    agg["share_explained_by_term"] = agg["term_sq"] / agg["increase_sq"]
    agg = agg.reindex(TARGETS)
    assert np.allclose(agg["term_sq"] + agg["cross"], agg["increase_sq"], atol=1e-6)
    agg.reset_index().to_csv(table_path(cfg, NAME, "_decomposition.csv"), index=False, float_format="%.4f")
    lines += ["", "| Target | Source × group terms | Increase in squared centre gap | Squared −(s−1)(μ_P,g − μ_P,src) | Cross term | Share |",
              "|---|---|---|---|---|---|"]
    for tg, r in agg.iterrows():
        lines.append(f"| {GROUP_LABEL[tg]} | {int(r.pairs_x_groups)} | {r.increase_sq:.2f} | {r.term_sq:.2f} | {r.cross:.2f} | "
                     f"{r.share_explained_by_term:.0%} |")
    notes = ["TITAN。各セルは 2 施設の中心の距離（群の中央値、施設内の広がり = 1）/ 近傍の混ざり（両施設 5 枚以上の群の中央値、1 = 施設を"
             "見分けられない）/ 隣り合う組（グリオーマ 3 組・髄膜腫 12 組だけ）。",
             "s = 倍率（EBRAINS / 院内の施設内の広がりの比）。借りた s は推定元で、当て先の s は当て先自身で推定したもの。",
             "Similarity, borrowed は推定元の院内平均を中心に s 倍に広げてから平均を揃える。Borrowed s, centred at target in-house mean は"
             "同じ s を当て先の院内スライド全体の平均を中心に掛けたもの。Target's own s は倍率も当て先自身の値にしたもの。"
             "リンパ腫・転移は群が 1 つなので、後ろ 2 つの中心の位置は平行移動と同じで、違いは施設内の広がり（分母）による。",
             "下の表: 借りた相似変換で中心の差の二乗が平行移動より増えた分（推定元 × 群の和）= −(s−1)(μ_P,g − μ_P,src) の二乗 ＋ 交差項"
             "（恒等式）。Share = 前者の割合。"]
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}（{len(t)} 行）")
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
