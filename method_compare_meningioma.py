"""method_compare_meningioma.py — 補正の方法の比較を「髄膜腫だけ」（v4 の定義、ken 2026-11-24）でやり直す（v4 の Fig 2 の材料、TITAN）。

v3 の Fig 3（髄膜腫 + SFT、EBRAINS 551 / 院内 577 枚、13 組）と同じ比較を、SFT を除いた髄膜腫（519 / 555 枚、両施設にあるサブタイプ
12 組。樹形図に SFT の葉を入れない）で行う。比較のため同じ計算を v3 の定義（13 組）でも行い、v3 の出力と一致するかを確かめる。
条件（CONDITIONS_FIG3 の並び ＋ 参考に相似変換）:
    original     補正なし
    macenko      色正規化（院内の既存の h5 = embedding.patho2.macenko をそのまま読む。作り直さない）
    gan          GAN（院内の既存の h5 = embedding.patho2.gan をそのまま読む）
    combat / centroid / affine_free
                 v4: 髄膜腫だけの補正なしの埋め込みで当てはめて当てる（transfer_correction.fit_source / apply_method、own と同じ。
                     neuroCombat の呼び方・平行移動・相似変換の倍率は combat.py / centroid.py / affine.scale_free と同じ定義）
                 v3: 論文の補正済み h5（embedding.patho2.<variant>、髄膜腫 + SFT で当てはめたもの）をそのまま読む
指標（v3 と同じ関数を呼ぶ）:
    asw_site / ilisi_site  施設のシルエット係数と iLISI（harmonization._reduce → _site_values = method_compare の asw_batch / ilisi と同じ）
    pairs_adjacent         樹形図で同じ組織型の 2 施設の平均が隣り合う組の数（dendrogram.group_means / raw_distance / ward_linkage /
                           pair_stats = method_compare の adjacent_pairs と同じ）、pairs_n = 組の数（両施設にある組織型）
    group_gap_sd / knn_mix 組織型ごとの 2 施設の中心の距離 / 施設内のばらつき、院内の 10 近傍に占める EBRAINS の割合 / 期待値
                           （transfer_correction.site_mixing。組織型の中央値。_min5 は両施設 5 枚以上の組織型だけの中央値）
    nc_histotype_ba_E2P / P2E  組織型の最近傍重心法の balanced accuracy（transfer_correction.nc_ba = nc.py の点推定と同じ。
                           クラスは nc.histotype_set、SFT は入らない）
確認: v3 の行が out/method_compare/summary.csv（asw_batch / ilisi / adjacent_pairs）と out/nc/histotype/bootstrap.csv
    （balanced_accuracy の点推定）に一致するか。v3 の出力は読むだけ。

出力 (output_dir/method_compare_meningioma/):
    summary.csv   definition（meningioma = v4 / paper_v3）× condition の指標
    groups.csv    組織型ごとの group_gap_sd / gap / rms / knn_mix（site_mixing）
    summary.md    v4 と v3 を並べた表と確認

Usage:
    uv run python method_compare_meningioma.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dendrogram import group_means, pair_stats, raw_distance, ward_linkage
from harmonization import _reduce, _site_values
from histotype_labels import apply_set
from transfer_correction import _md, apply_method, fit_source, nc_ba, site_mixing
from utils.display import CONDITION_LABELS_JA, CONDITIONS_FIG3
from utils.loader import load_config, load_data

CONDITIONS = CONDITIONS_FIG3 + ["affine_free"]       # 補正なし → Macenko → GAN → ComBat → 平行移動、参考に相似変換
FIT_HERE = {"combat", "centroid", "affine_free"}     # v4 で髄膜腫だけから当てはめ直す補正
MIN_N = 5


def _variant(cfg: dict, cond: str) -> pd.DataFrame:
    return load_data(cfg, cfg.get("variants", {}).get(cond, cond))


def evaluate(df: pd.DataFrame, cfg: dict, classes: list[str]) -> tuple[dict, list[dict]]:
    ref = cfg["reference"]
    oth = next(s for s in df["source"].unique() if s != ref)
    X = np.stack(df["embedding"].values).astype(np.float64)
    st, lab = df["source"].to_numpy(), df["subtype"].astype(str).to_numpy()
    asw, ilisi = _site_values(_reduce(X, cfg.get("centroid", {}).get("pca_n_components", 50)), st,
                              cfg.get("harmonization", {}).get("lisi_k", 30))
    g = group_means(df, cfg)
    D = raw_distance(g)
    ps = pair_stats(D, ward_linkage(D), g)
    mix, per = site_mixing(X, st, lab, ref, cfg)
    big = [r for r in per if r["n_ebrains"] >= MIN_N and r["n_patho2"] >= MIN_N]
    m = np.isin(lab, classes)
    out = {"n_ebrains": int((st == ref).sum()), "n_patho2": int((st != ref).sum()), "asw_site": asw, "ilisi_site": ilisi,
           "pairs_adjacent": int(ps["adjacent_pairs"]), "pairs_n": int(ps["n_subtypes"]), **mix,
           "group_gap_sd_min5": float(np.median([r["group_gap_sd"] for r in big])),
           "knn_mix_min5": float(np.median([r["knn_mix"] for r in big])), "n_groups_min5": len(big),
           "nc_histotype_ba_E2P": nc_ba(X[m], st[m], lab[m], classes, ref, oth),
           "nc_histotype_ba_P2E": nc_ba(X[m], st[m], lab[m], classes, oth, ref)}
    return out, per


def main() -> None:
    cfg = load_config()
    ref = cfg["reference"]
    out_dir = Path(cfg["output_dir"]) / "method_compare_meningioma"
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = apply_set(cfg, cfg.get("nc", {}).get("histotype_set", "main"))["lp_histotype"]["classes"]
    base = _variant(cfg, "original")
    is_sft = base["subtype"].str.lower().str.contains("sft").to_numpy()
    men = base[~is_sft].reset_index(drop=True)
    print(f"[method_compare_meningioma] v4 meningioma: {men.groupby('source').size().to_dict()}  v3 paper: {base.groupby('source').size().to_dict()}")
    rows, grows, checks = [], [], {}
    for cond in CONDITIONS:
        for definition in ("meningioma", "paper_v3"):
            if definition == "meningioma" and cond in FIT_HERE:   # 髄膜腫だけで当てはめて当てる（own）
                X = np.stack(men["embedding"].values).astype(np.float64)
                st = men["source"].to_numpy()
                fit = fit_source(X, st, ref)
                Y, _ = apply_method(cond, fit, X, st, ref, checks)
                df = men.assign(embedding=list(Y))
            else:
                df = _variant(cfg, cond)
                if definition == "meningioma":
                    df = df[~df["subtype"].str.lower().str.contains("sft")].reset_index(drop=True)
            m, per = evaluate(df, cfg, classes)
            rows.append({"definition": definition, "condition": cond, **m})
            grows += [{"definition": definition, "condition": cond, **r} for r in per]
            print(f"  [{definition} / {cond}] ASW {m['asw_site']:.4f}  pairs {m['pairs_adjacent']}/{m['pairs_n']}  "
                  f"gap/SD {m['group_gap_sd']:.3f}  kNN {m['knn_mix']:.3f}  BA E→P {m['nc_histotype_ba_E2P']:.3f} P→E {m['nc_histotype_ba_P2E']:.3f}")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(grows).to_csv(out_dir / "groups.csv", index=False)

    # 確認: v3 の行 vs v3 の出力（読むだけ）
    out = Path(cfg["output_dir"])
    mc = pd.read_csv(out / "method_compare" / "summary.csv").set_index("variant")
    bs = pd.read_csv(out / "nc" / "histotype" / "bootstrap.csv")
    bs = bs[bs["metric"] == "balanced_accuracy"]
    oth = next(s for s in cfg["embedding"] if s != ref)
    cmp = []
    for r in res[res["definition"] == "paper_v3"].itertuples():
        if r.condition in mc.index:
            cmp += [("asw_site vs method_compare asw_batch", r.condition, r.asw_site, mc.loc[r.condition, "asw_batch"]),
                    ("ilisi_site vs method_compare ilisi", r.condition, r.ilisi_site, mc.loc[r.condition, "ilisi"]),
                    ("pairs_adjacent vs method_compare adjacent_pairs", r.condition, r.pairs_adjacent, mc.loc[r.condition, "adjacent_pairs"])]
        for name, tr in (("E2P", ref), ("P2E", oth)):
            v = bs[(bs["variant"] == r.condition) & (bs["train_source"] == tr)]["value"]
            if len(v):
                cmp.append((f"nc_histotype_ba_{name} vs nc/histotype/bootstrap.csv", r.condition, getattr(r, f"nc_histotype_ba_{name}"), v.iloc[0]))
    chk = pd.DataFrame(cmp, columns=["check", "condition", "got", "expected"])
    chk["diff"] = (chk["got"] - chk["expected"]).abs()
    print(f"  checks vs v3 outputs: {len(chk)} values, max |diff| {chk['diff'].max():.2e}")
    if chk["diff"].max() > 1e-6:
        print(chk[chk["diff"] > 1e-6].to_string(index=False))

    L = ["# 補正の方法の比較: 髄膜腫だけ（v4）と髄膜腫 + SFT（v3）", "",
         "`method_compare_meningioma.py` が作る（TITAN）。定義は docstring。v4 = 髄膜腫だけ（SFT を除く、両施設にある組織型の組）、"
         "ComBat・平行移動・相似変換は髄膜腫だけで当てはめ直し、Macenko・GAN は既存の院内 h5。v3 = 論文の補正済み h5（髄膜腫 + SFT で当てはめ）。",
         "列: 施設のシルエット係数（0 に近いほど施設が混ざる）、隣り合う組 / 組の数、組織型ごとの中心の距離 / 施設内のばらつき（中央値。"
         f"括弧は両施設 {MIN_N} 枚以上の組織型だけ）、近傍の混ざり（中央値、1 = 見分けがつかない）、組織型の最近傍重心法の BA（EBRAINS → 院内 / 院内 → EBRAINS）。", ""]
    tb = []
    for cond in CONDITIONS:
        for definition, tag in (("meningioma", "v4 髄膜腫だけ"), ("paper_v3", "v3 髄膜腫 + SFT")):
            r = res[(res["definition"] == definition) & (res["condition"] == cond)].iloc[0]
            tb.append({"条件": CONDITION_LABELS_JA.get(cond, cond), "定義": tag, "枚数 E/P": f"{r.n_ebrains}/{r.n_patho2}",
                       "ASW（施設）": f"{r.asw_site:.4f}", "隣り合う組": f"{r.pairs_adjacent}/{r.pairs_n}",
                       "中心の距離": f"{r.group_gap_sd:.3f}（{r.group_gap_sd_min5:.3f}）", "近傍の混ざり": f"{r.knn_mix:.3f}（{r.knn_mix_min5:.3f}）",
                       "BA E→P": f"{r.nc_histotype_ba_E2P:.3f}", "BA P→E": f"{r.nc_histotype_ba_P2E:.3f}"})
    L += [_md(pd.DataFrame(tb), index=False), "",
          "## 確認（v3 の行と v3 の出力）", "", f"{len(chk)} 値、最大の差 {chk['diff'].max():.2e}"
          "（out/method_compare/summary.csv の asw_batch / ilisi / adjacent_pairs、out/nc/histotype/bootstrap.csv の点推定）", ""]
    (out_dir / "summary.md").write_text("\n".join(L))
    print(f"  saved: {out_dir}/summary.csv, groups.csv, summary.md")


if __name__ == "__main__":
    main()
