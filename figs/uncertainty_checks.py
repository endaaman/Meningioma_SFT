"""v4 Supplementary の表: 主要な指標の不確かさと、査読で問われやすい 2 つの確認（TITAN）。数字は transfer_uncertainty.py の出力を並べるだけ。

A 患者単位のブートストラップ（施設 × 群の中で患者を復元抽出、1,000 回、補正は毎回当てはめ直す）。当て先グリオーマ（← 髄膜腫）と
  髄膜腫（← グリオーマ）の 補正なし / その腫瘍自身 / 借りる × 平行移動・相似変換・ComBat、鑑別の腫瘍（SFT ← 髄膜腫、リンパ腫・
  転移 ← グリオーマ）の 補正なし / 借りる。隣り合う組・中心の距離・近傍の混ざり（両施設 5 枚以上の群）。点推定（本文の値）と、
  再抽出の中央値 [2.5%, 97.5%]。借りた平行移動 − 補正なし の対応のある差
B 参照コホートへの診断名の検索の 1 位一致（4 つの腫瘍）。補正なし / 髄膜腫から借りる / その腫瘍自身 / EBRAINS の中（患者単位で除く）、
  借りる − 補正なしの差、良くなった・悪くなった枚数と McNemar の正確検定
C 「同じ腫瘍で推定」の当てはめと評価の分離（患者単位の半分、200 回、評価はすべて同じ半分）
D SFT と同じ枚数（EBRAINS 32 / 院内 22）で髄膜腫・グリオーマのずれを推定した場合（500 回）と、SFT から借りた実測値

入力: output_dir/transfer_uncertainty/{bootstrap_summary, bootstrap_pairs_freq, bootstrap_contrasts, retrieval_tumours_summary,
      retrieval_tumours_paired, split_summary, subsample_summary}.csv、点推定は output_dir/transfer/correction_titan.csv・
      correction_titan_groups.csv・rare_target_titan.csv
出力: tables/{table Sn}_uncertainty_checks.{csv,md}

Usage:
    uv run python -m figs.uncertainty_checks
"""
from __future__ import annotations

import pandas as pd

from figs import label
from figs.common import CONDITION_LABELS, config, out_root, table_path
from utils.display import FAMILY_LABELS

NAME = "uncertainty_checks"
MIN_N = 5
T_LABEL = {"glioma": "Glioma ← meningioma", "meningioma": "Meningioma ← glioma", "SFT": "SFT ← meningioma",
           "lymphoma": f"{FAMILY_LABELS['lymphoma']} ← glioma", "metastasis": "Metastasis ← glioma"}
C_LABEL = {"none": "No correction", "own": "Same tumour", "borrowed": "Borrowed", "own_split": "Same tumour, fitted on the other half",
           "own_insample_half": "Same tumour, fitted on the evaluated half"}
M_LABEL = {"none": "", **CONDITION_LABELS}
METRIC_LABEL = {"pairs_adjacent": "Cross-site pairs", "group_gap_sd": "Site-centre distance", "knn_mix": "Neighbour mixing (≥ 5 per site)",
                "pairs_n": "Pairs available"}


def ci(r, d: int = 3) -> str:
    return f"{r['median']:.{d}f} [{r['q025']:.{d}f}, {r['q975']:.{d}f}]"


def points(cfg: dict) -> dict:
    t = out_root(cfg) / "transfer"
    c = pd.read_csv(t / "correction_titan.csv").set_index(["target", "source", "method", "metric"])["value"]
    g = pd.read_csv(t / "correction_titan_groups.csv")
    g = g[(g["metric"] == "knn_mix") & (g["n_ebrains"] >= MIN_N) & (g["n_patho2"] >= MIN_N)]
    k5 = g.groupby(["target", "source", "method"])["value"].median()
    rt = pd.read_csv(t / "rare_target_titan.csv")
    rt = rt[(rt["target"] == "paper") & (rt["group"] == "SFT")].set_index(["source", "method", "metric"])["value"]
    src = {"glioma": "meningioma", "meningioma": "glioma", "SFT": "meningioma", "lymphoma": "glioma", "metastasis": "glioma"}
    out = {}
    for tg, s in src.items():
        for cond in ("none", "own", "borrowed"):
            so = {"none": "none", "own": "own", "borrowed": s}[cond]
            for m in ("centroid", "affine_free", "combat"):
                for metric in ("pairs_adjacent", "group_gap_sd", "knn_mix"):
                    try:
                        if tg == "SFT":
                            v = rt[(so, m, metric)] if metric != "pairs_adjacent" else None
                        elif metric == "knn_mix":
                            v = k5[(tg, so, m)]
                        else:
                            v = c[(tg, so, m, metric)]
                    except KeyError:
                        v = None
                    out[(tg, cond, "none" if cond == "none" else m, metric)] = v
    return out


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    u = out_root(cfg) / "transfer_uncertainty"
    P = points(cfg)
    rows, md = [], ["## A. Patient-level bootstrap (1,000 draws, corrections refitted each draw): point estimate; median [2.5%, 97.5%] of the draws", "",
                    "| Target ← source | Condition | Method | Cross-site pairs | Site-centre distance | Neighbour mixing (≥ 5 per site) |",
                    "|---|---|---|---|---|---|"]
    bs = pd.read_csv(u / "bootstrap_summary.csv")
    for (t, cnd, m), g in bs.groupby(["target", "condition", "method"], sort=False):
        gg = g.set_index("metric")
        cells = []
        for metric in ("pairs_adjacent", "group_gap_sd", "knn_mix"):
            if metric not in gg.index:
                cells.append("—")
                continue
            r = gg.loc[metric]
            pt = P.get((t, cnd, m, metric))
            d = 0 if metric == "pairs_adjacent" else 3
            cells.append((f"{pt:.{d}f}; " if pt is not None else "") + ci(r, d))
            rows.append({"section": "A bootstrap", "target": T_LABEL[t], "condition": C_LABEL[cnd], "method": M_LABEL[m],
                         "metric": METRIC_LABEL[metric], "point": pt, "median": r["median"], "q025": r["q025"], "q975": r["q975"]})
        md.append(f"| {T_LABEL[t]} | {C_LABEL[cnd]} | {M_LABEL[m]} | {' | '.join(cells)} |")
    con = pd.read_csv(u / "bootstrap_contrasts.csv")
    con = con[con["contrast"] == "borrowed centroid − none"]
    md += ["", "Paired difference, borrowed translation − no correction (same draw): median [2.5%, 97.5%]", "",
           "| Target ← source | Site-centre distance | Neighbour mixing |", "|---|---|---|"]
    for t in T_LABEL:
        cc = con[con["target"] == t].set_index("metric")
        md.append(f"| {T_LABEL[t]} | {ci(cc.loc['group_gap_sd'])} | {ci(cc.loc['knn_mix'])} |")
        for metric in ("group_gap_sd", "knn_mix"):
            r = cc.loc[metric]
            rows.append({"section": "A paired difference", "target": T_LABEL[t], "condition": "Borrowed − no correction", "method": "Translation",
                         "metric": METRIC_LABEL[metric], "median": r["median"], "q025": r["q025"], "q975": r["q975"]})

    rs = pd.read_csv(u / "retrieval_tumours_summary.csv")
    rs = rs[(rs["group"] == "(all)") & (rs["metric"] == "top1")]
    rp = pd.read_csv(u / "retrieval_tumours_paired.csv")
    rp = rp[rp["metric"] == "top1"].set_index("tumour")
    rc = {"none": "No correction", "meningioma": "Borrowed from meningioma", "family": "Same tumour", "ebrains_lpo": "Within EBRAINS (patient left out)"}
    md += ["", "## B. Diagnosis retrieval, correct diagnosis ranked first: point estimate; median [2.5%, 97.5%]", "",
           "| Tumour | No correction | Borrowed from meningioma | Same tumour | Within EBRAINS | Borrowed − none | Improved / worsened | McNemar exact p |",
           "|---|---|---|---|---|---|---|---|"]
    for t, tl in (("glioma", "Glioma"), ("lymphoma", FAMILY_LABELS["lymphoma"]), ("metastasis", "Metastasis"), ("SFT", "SFT")):
        cells = []
        for cnd in ("none", "meningioma", "family", "ebrains_lpo"):
            r = rs[(rs["tumour"] == t) & (rs["condition"] == cnd)].iloc[0]
            cells.append(f"{r['point']:.3f} [{r['q025']:.3f}, {r['q975']:.3f}]")
            rows.append({"section": "B retrieval top-1", "target": tl, "condition": rc[cnd], "metric": "Top-1", "point": r["point"],
                         "median": r["median"], "q025": r["q025"], "q975": r["q975"], "n": int(r["n"])})
        p = rp.loc[t]
        cells += [f"{p['diff']:+.3f} [{p['diff_q025']:+.3f}, {p['diff_q975']:+.3f}]", f"{int(p['improved'])} / {int(p['worsened'])}",
                  f"{p['mcnemar_exact_p']:.2g}"]
        rows.append({"section": "B retrieval paired", "target": tl, "condition": "Borrowed − none", "metric": "Top-1", "point": p["diff"],
                     "q025": p["diff_q025"], "q975": p["diff_q975"], "improved": int(p["improved"]), "worsened": int(p["worsened"]),
                     "mcnemar_exact_p": p["mcnemar_exact_p"]})
        md.append(f"| {tl} (in-house {int(p['n'])}) | " + " | ".join(cells) + " |")

    sp = pd.read_csv(u / "split_summary.csv")
    md += ["", "## C. Fitting and evaluation separated (patient-level halves, 200 splits; every condition evaluated on the same half)", "",
           "| Target ← source | Condition | Method | Cross-site pairs | Site-centre distance | Neighbour mixing |", "|---|---|---|---|---|---|"]
    for (t, cnd, m), g in sp.groupby(["target", "condition", "method"], sort=False):
        gg = g.set_index("metric")
        md.append(f"| {T_LABEL[t]} | {C_LABEL[cnd]} | {M_LABEL[m]} | {ci(gg.loc['pairs_adjacent'], 0)} of {ci(gg.loc['pairs_n'], 0)} | "
                  f"{ci(gg.loc['group_gap_sd'])} | {ci(gg.loc['knn_mix'])} |")
        for metric in ("pairs_adjacent", "group_gap_sd", "knn_mix"):
            r = gg.loc[metric]
            rows.append({"section": "C split", "target": T_LABEL[t], "condition": C_LABEL[cnd], "method": M_LABEL[m], "metric": METRIC_LABEL[metric],
                         "median": r["median"], "q025": r["q025"], "q975": r["q975"]})

    ss = pd.read_csv(u / "subsample_summary.csv")
    ml = {"pairs_adjacent": "Cross-site pairs", "group_gap_sd": "Site-centre distance", "knn_mix": "Neighbour mixing (≥ 5 per site)",
          "shift_norm": "Shift length", "cos_vs_rest_source": "cos with the rest of the source tumour's shift", "cos_vs_sft": "cos with the SFT shift"}
    md += ["", "## D. Shift estimated from as many slides as SFT has (EBRAINS 32 / in-house 22), 500 draws", "",
           "| Source → target | Metric | Subsampled source: median [2.5%, 97.5%] | Borrowed from SFT (observed) | Source, all slides | Draws at least as extreme as SFT |",
           "|---|---|---|---|---|---|"]
    for r in ss.itertuples():
        if r.metric not in ml:
            continue
        worse = {"pairs_adjacent": r.frac_le_sft, "group_gap_sd": r.frac_ge_sft, "knn_mix": r.frac_le_sft, "shift_norm": r.frac_ge_sft,
                 "cos_vs_rest_source": r.frac_le_sft, "cos_vs_sft": float("nan")}[r.metric]
        d = 0 if r.metric == "pairs_adjacent" else 3
        md.append(f"| {r.source_family.capitalize()} → {r.target} | {ml[r.metric]} | {r.median:.{d}f} [{r.q025:.{d}f}, {r.q975:.{d}f}] | "
                  f"{r.sft_observed:.{d}f} | {r.full_source:.{d}f} | {'' if pd.isna(worse) else f'{worse:.3f}'} |")
        rows.append({"section": "D SFT-sized subsample", "target": f"{r.source_family} → {r.target}", "metric": ml[r.metric], "median": r.median,
                     "q025": r.q025, "q975": r.q975, "sft_observed": r.sft_observed, "full_source": r.full_source, "frac_as_extreme": worse})
    notes = ["TITAN。Site-centre distance = サブタイプ（群）ごとの 2 施設の中心の距離を施設内の広がりで割った値の中央値。Neighbour mixing = "
             "院内のスライドの近傍 10 枚に EBRAINS が入る割合 ÷ 期待値（1 = 施設を見分けられない）の、両施設 5 枚以上の群の中央値"
             "（再抽出では、同じスライドの複製は近傍から除く）。",
             "再抽出ではサブタイプの平均に再抽出の雑音が加わるので、中心の距離は点推定より大きめ、隣り合う組は少なめに出る"
             "（例: 髄膜腫 ← グリオーマの 7 組は再抽出の中央値 5、7 組以上は 13%）。区間は再抽出の分布として読む。",
             "B の点推定と、良くなった・悪くなった枚数、McNemar の正確検定は元のスライド（再抽出なし）。参照（EBRAINS の 78 診断の重心）は固定。",
             "C の Pairs は評価した半分にある組の数のうちの隣り合う組（半分に分けると、例数の少ないサブタイプが片方の施設に無いことがある）。",
             "D の Draws at least as extreme = SFT から借りた実測値と同じかそれより悪い（組・混ざり・cos は以下、距離・長さは以上）抜き取りの割合。"]
    t = pd.DataFrame(rows)
    t.to_csv(table_path(cfg, NAME, ".csv"), index=False, float_format="%.4f")
    path = table_path(cfg, NAME, ".md")
    path.write_text("\n".join(md) + "\n\n" + "\n".join(f"- {n}" for n in notes) + "\n")
    print(f"  saved: {table_path(cfg, NAME, '.csv')} / {path.name}（{len(t)} 行）")


if __name__ == "__main__":
    main()
