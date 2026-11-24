"""review_tables.py — 査読で問われやすい 2 点を、既存の出力から表にする（新しい計算はしない、TITAN、2026-11-25）。

lengths   群ごとの施設のずれの長さ ‖Δ‖ の、推定の雑音の補正（cos と同じ考え方: 期待される雑音の二乗 tr V = Σ_d var_E,d / n_E
          + var_P,d / n_P を ‖Δ̂‖² から引いて平方根）。生の値と並べ、髄膜腫（SFT を除く全体）に対する比を両方で出す。
          入力: output_dir/shift_transfer/groups.csv（shift_norm / noise_trace / shift_norm_corrected。shift_transfer.shift_stats）
retrieval Fig 7 の「EBRAINS の中での値」の除き方の確認。atlas_retrieval.py の ebrains_loo は問い合わせるスライド自身だけを
          その診断名の重心から除く（スライド単位）。同じ患者の他のスライドも除いた版（ebrains_lpo、患者単位）は既に出力されているので、
          両方を並べる（1 位・5 位以内の一致、3 群の balanced accuracy）。
          入力: output_dir/atlas/retrieval_titan.csv・glioma_confusion_titan.csv

出力: output_dir/review_checks/{shift_length_noise.csv, retrieval_ebrains_patient_out.csv, summary.md}

Usage:
    uv run python review_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.loader import load_config

MENI = "Meningioma (all)"
ROWS = ["Meningioma (all)", "SFT", "Astrocytoma, IDH-mutant", "Glioblastoma, IDH-wildtype", "Oligodendroglioma", "Glioma (all)",
        "DLBCL", "Metastasis", "Meningothelial meningioma", "Fibrous meningioma", "Transitional meningioma", "Angiomatous meningioma",
        "Microcystic meningioma", "Atypical meningioma"]


def balanced_acc(c: pd.DataFrame) -> float:
    t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=["ast", "glm", "odg"]).fillna(0)
    return float(np.mean([t.loc[g, g] / t.loc[g].sum() if g in t.columns and t.loc[g].sum() else 0.0 for g in t.index]))


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["output_dir"])
    out = out_root / "review_checks"
    out.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(out_root / "shift_transfer" / "groups.csv").set_index("group")
    m_raw, m_cor = g.loc[MENI, "shift_norm"], g.loc[MENI, "shift_norm_corrected"]
    L = pd.DataFrame([{"group": r, "n_ebrains": int(g.loc[r, "n_ebrains"]), "n_inhouse": int(g.loc[r, "n_patho2"]),
                       "shift_norm_raw": g.loc[r, "shift_norm"], "noise_sq_expected": g.loc[r, "noise_trace"],
                       "shift_norm_corrected": g.loc[r, "shift_norm_corrected"],
                       "ratio_to_meningioma_raw": g.loc[r, "shift_norm"] / m_raw,
                       "ratio_to_meningioma_corrected": g.loc[r, "shift_norm_corrected"] / m_cor} for r in ROWS])
    L.to_csv(out / "shift_length_noise.csv", index=False)

    r = pd.read_csv(out_root / "atlas" / "retrieval_titan.csv")
    r = r[(r["retrieval"] == "nc_euclid") & (r["query_label"] == "(all)") & r["condition"].isin(["ebrains_loo", "ebrains_lpo"])
          & r["query_family"].isin(["glioma", "ast", "glm", "odg"]) & r["metric"].isin(["top1", "top5"])]
    R = r.pivot_table(index=["metric", "query_family"], columns="condition", values="value").reset_index()
    n = r[r["metric"] == "top1"].pivot_table(index="query_family", columns="condition", values="n")
    c = pd.read_csv(out_root / "atlas" / "glioma_confusion_titan.csv")
    c = c[(c["retrieval"] == "nc_euclid") & (c["scope"] == "full")]
    ba = {cond: balanced_acc(c[c["condition"] == cond]) for cond in ("ebrains_loo", "ebrains_lpo")}
    R = pd.concat([R, pd.DataFrame([{"metric": "balanced_accuracy_3groups", "query_family": "glioma", **ba}])], ignore_index=True)
    R.to_csv(out / "retrieval_ebrains_patient_out.csv", index=False)

    md = ["# 査読の確認（既存の出力から）", "", "## 施設のずれの長さと推定の雑音（shift_transfer/groups.csv）", "",
          "| Group | EBRAINS | In-house | ‖Δ‖ raw | tr V | ‖Δ‖ noise-corrected | ratio to meningioma, raw | ratio, corrected |",
          "|---|---|---|---|---|---|---|---|"]
    for _, x in L.iterrows():
        md.append(f"| {x.group} | {x.n_ebrains} | {x.n_inhouse} | {x.shift_norm_raw:.2f} | {x.noise_sq_expected:.2f} | "
                  f"{x.shift_norm_corrected:.2f} | {x.ratio_to_meningioma_raw:.2f} | {x.ratio_to_meningioma_corrected:.2f} |")
    other = L[L["group"].isin(["SFT", "Astrocytoma, IDH-mutant", "Glioblastoma, IDH-wildtype", "Oligodendroglioma", "DLBCL", "Metastasis"])]
    md += ["", f"- 髄膜腫以外の 6 群の比: 生 {other.ratio_to_meningioma_raw.min():.2f}–{other.ratio_to_meningioma_raw.max():.2f}、"
               f"補正後 {other.ratio_to_meningioma_corrected.min():.2f}–{other.ratio_to_meningioma_corrected.max():.2f}",
           "", "## Fig 7 の EBRAINS の中での値: スライド単位（ebrains_loo）と患者単位（ebrains_lpo）", "",
           "| metric | group | slide-out (Fig 7) | patient-out | n |", "|---|---|---|---|---|"]
    for _, x in R.iterrows():
        nn = int(n.loc[x.query_family, "ebrains_loo"]) if x.metric != "balanced_accuracy_3groups" else ""
        md.append(f"| {x.metric} | {x.query_family} | {x.ebrains_loo:.3f} | {x.ebrains_lpo:.3f} | {nn} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    pd.set_option("display.width", 200)
    print(L.round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
