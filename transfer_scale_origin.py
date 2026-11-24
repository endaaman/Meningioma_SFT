"""transfer_scale_origin.py — 診断用: 別の腫瘍から借りた倍率で中心の距離が悪くなるのは「倍率の大きさ」のせいか「広げる中心」のせいか（TITAN）。

仮説: 借りた相似変換 x' = (x − μ_P,src)·s + μ_E,src は推定元の院内の中心 μ_P,src のまわりで広げるので、当て先の群 g の中心には
平行移動の残差 r_tr,g = μ_E,g − μ_P,g − Δ_src（Δ_src = μ_E,src − μ_P,src）に加えて e_g = −(s − 1)(μ_P,g − μ_P,src) が足される:
    r_aff,g = μ_E,g − μ'_P,g = r_tr,g + e_g        （恒等式。‖r_aff‖² − ‖r_tr‖² = ‖e‖² + 2⟨r_tr, e⟩）
ComBat は次元ごとの 1 次式 x' = k⊙x + c（transfer_correction.combat_linear）なので
    r_cb,g = r_tr,g − (k − 1)⊙(μ_P,g − μ_P,src) + [μ_E,src − (k⊙μ_P,src + c)]   （広げる中心の項 ＋ 推定元の中心の合わせ損ね）
切り分けの変種（推定元の Δ_src は同じ。広げる中心を当て先の院内スライド全体の平均 m_P,tgt にする。ラベル不要、EBRAINS の当て先は使わない）:
    translation          x' = x + Δ_src（= s を 1 にした場合）
    affine_borrowed      借りた相似変換（transfer_correction の affine_free）
    combat_borrowed      借りた ComBat（transfer_correction の combat）
    affine_tgt_center    (i)  x' = (x − m_P,tgt)·s_src + m_P,tgt + Δ_src   （倍率は借りる、中心は当て先）
    ownscale_tgt_center  (ii) x' = (x − m_P,tgt)·s_tgt + m_P,tgt + Δ_src   （倍率は当て先自身の s、借りるのは Δ だけ）
    → (i) が平行移動並みなら原因は中心、(i) が悪く (ii) が良ければ原因は倍率の大きさ。
    (i) の余計な項は e'_g = −(s − 1)(μ_P,g − m_P,tgt) で、群が 1 つの当て先（lymphoma / metastasis）では 0（中心は平行移動と同じ）。
当て先: glioma / meningioma（髄膜腫だけ、12 組）/ lymphoma / metastasis。推定元: config の transfer.correction.sources のうち none と
    others_mean（科ごとの当てはめの平均で、広げる中心が 1 つに定まらない）を除き、transfer_correction.source_families で残るもの。
指標は transfer_correction の関数（evaluate: 施設 ASW・residual・pairs_adjacent。site_mixing: 群ごとの group_gap_sd / knn_mix）。
確認: translation / affine_borrowed / combat_borrowed の群の中央値と pairs_adjacent が correction_titan.csv と一致するか、恒等式の残り。
主解析の出力には書かない。

出力 (output_dir/_diagnostics/scale_origin/):
    summary.csv        当て先 × 推定元 × 変種: residual_norm, asw_site, pairs_adjacent, group_gap_sd, knn_mix（群の中央値）, 倍率
    groups.csv         当て先 × 推定元 × 変種 × 群: group_gap_sd / gap / rms / knn_mix
    decomposition.csv  当て先 × 推定元 × 群: ‖r_tr‖, ‖r_aff‖, ‖e‖, 2⟨r_tr, e⟩, ‖e‖² の割合、‖μ_P,g − μ_P,src‖、‖μ_P,g − m_P,tgt‖、
                       (i) と ComBat の同じ分解、恒等式の残り
    summary.md         表

Usage:
    uv run python transfer_scale_origin.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from affine import scale_free
from histotype_labels import apply_set
from transfer_correction import (FAMILIES, TARGET_FAMILIES, _md, apply_method, combat_linear, evaluate, fit_source,
                                 load_embeddings, site_mixing, source_families, target_labels)
from utils.loader import load_config

TARGETS = ["glioma", "meningioma", "lymphoma", "metastasis"]
VARIANTS = ["translation", "affine_borrowed", "combat_borrowed", "affine_tgt_center", "ownscale_tgt_center"]
VARIANT_LABEL = {"translation": "平行移動", "affine_borrowed": "借りた相似変換", "combat_borrowed": "借りた ComBat",
                 "affine_tgt_center": "(i) 借りた s・中心は当て先", "ownscale_tgt_center": "(ii) 自前の s・中心は当て先"}


def main() -> None:
    cfg = load_config()
    ref = cfg["reference"]
    out_dir = Path(cfg["output_dir"]) / "_diagnostics" / "scale_origin"
    out_dir.mkdir(parents=True, exist_ok=True)
    man, X = load_embeddings(cfg, "titan", Path(cfg["output_dir"]) / "transfer", reload=False)
    fam, site = man["family"].to_numpy(), man["site"].to_numpy()
    oth = next(s for s in pd.unique(site) if s != ref)
    ccfg = cfg["transfer"].get("correction", {}) or {}
    sources = [s for s in ccfg.get("sources", []) if s not in ("none", "others_mean")]
    ctx = {"ref": ref, "oth": oth, "min_n": int(ccfg.get("min_n", 5)),
           "pca_n": cfg.get("centroid", {}).get("pca_n_components", 50), "lisi_k": cfg.get("harmonization", {}).get("lisi_k", 30),
           "histotype_classes": apply_set(cfg, cfg.get("nc", {}).get("histotype_set", "main"))["lp_histotype"]["classes"]}
    corr = pd.read_csv(Path(cfg["output_dir"]) / "transfer" / "correction_titan.csv")
    fits, rows, grows, drows, checks, idmax = {}, [], [], [], [], 0.0
    for target in TARGETS:
        tm = np.isin(fam, list(TARGET_FAMILIES[target]))
        Xt, st, lab = X[tm], site[tm], target_labels(man[tm], target)
        e, p = st == ref, st != ref
        m_tgt = Xt[p].mean(0)
        s_tgt = scale_free(Xt, e, p)
        groups = [g for g in pd.unique(lab) if ((lab == g) & e).any() and ((lab == g) & p).any()]
        for source in sources:
            sf = source_families(target, source)
            if sf is None:
                continue
            if sf not in fits:
                sm = np.isin(fam, list(sf))
                fits[sf] = fit_source(X[sm], site[sm], ref)
            fit = fits[sf]
            d_src, s, mu_p_src, mu_e_src = fit["mu_e"] - fit["mu_p"], fit["s"], fit["mu_p"], fit["mu_e"]
            k, c = combat_linear(fit)
            Y = {}
            for v, meth in (("translation", "centroid"), ("affine_borrowed", "affine_free"), ("combat_borrowed", "combat")):
                Y[v], _ = apply_method(meth, fit, Xt, st, ref, {})
            for v, sc in (("affine_tgt_center", s), ("ownscale_tgt_center", s_tgt)):
                Yv = Xt.copy()
                Yv[p] = (Xt[p] - m_tgt) * sc + m_tgt + d_src
                Y[v] = Yv
            for v in VARIANTS:
                m = evaluate(Y[v], st, lab, target, cfg, ctx)
                mix, per = site_mixing(Y[v], st, lab, ref, cfg)
                rows.append({"target": target, "source": source, "variant": v, "s_src": s, "s_tgt": s_tgt,
                             "k_median": float(np.median(k)), "residual_norm": m["residual_norm"], "asw_site": m["asw_site"],
                             "pairs_adjacent": m.get("pairs_adjacent", np.nan), "pairs_n": m.get("pairs_n", np.nan), **mix})
                grows += [{"target": target, "source": source, "variant": v, **r} for r in per]
                cm = {"translation": "centroid", "affine_borrowed": "affine_free", "combat_borrowed": "combat"}.get(v)
                if cm:
                    for metric in ("group_gap_sd", "knn_mix", "pairs_adjacent", "asw_site"):
                        exp = corr[(corr.target == target) & (corr.source == source) & (corr.method == cm) & (corr.metric == metric)]["value"]
                        if len(exp):
                            got = mix[metric] if metric in mix else m[metric]
                            checks.append(abs(float(exp.iloc[0]) - float(got)))
            # 群ごとの分解（恒等式）
            for g in groups:
                gm = lab == g
                mu_e_g, mu_p_g = Xt[gm & e].mean(0), Xt[gm & p].mean(0)
                r_tr = mu_e_g - mu_p_g - d_src
                e_aff = -(s - 1) * (mu_p_g - mu_p_src)
                e_i = -(s - 1) * (mu_p_g - m_tgt)
                e_cb_c = -(k - 1) * (mu_p_g - mu_p_src)
                e_cb_o = mu_e_src - (k * mu_p_src + c)
                obs = {v: mu_e_g - Y[v][gm & p].mean(0) for v in ("affine_borrowed", "affine_tgt_center", "combat_borrowed", "translation")}
                idmax = max(idmax, float(np.abs(obs["translation"] - r_tr).max()), float(np.abs(obs["affine_borrowed"] - (r_tr + e_aff)).max()),
                            float(np.abs(obs["affine_tgt_center"] - (r_tr + e_i)).max()),
                            float(np.abs(obs["combat_borrowed"] - (r_tr + e_cb_c + e_cb_o)).max()))
                n2 = lambda v: float(v @ v)  # noqa: E731
                row = {"target": target, "source": source, "group": g, "n_ebrains": int((gm & e).sum()), "n_patho2": int((gm & p).sum()),
                       "s_src": s, "dist_g_to_src_center": float(np.linalg.norm(mu_p_g - mu_p_src)),
                       "dist_g_to_tgt_center": float(np.linalg.norm(mu_p_g - m_tgt)), "r_tr": np.sqrt(n2(r_tr))}
                for name, ev, robs in (("aff", e_aff, obs["affine_borrowed"]), ("i", e_i, obs["affine_tgt_center"]),
                                       ("cb", e_cb_c + e_cb_o, obs["combat_borrowed"])):
                    diff = n2(robs) - n2(r_tr)
                    row.update({f"r_{name}": np.sqrt(n2(robs)), f"e_{name}": np.sqrt(n2(ev)), f"cross_{name}": 2 * float(r_tr @ ev),
                                f"diff_sq_{name}": diff, f"share_e2_{name}": n2(ev) / diff if abs(diff) > 1e-12 else np.nan})
                row.update({"e_cb_center": float(np.linalg.norm(e_cb_c)), "e_cb_offset": float(np.linalg.norm(e_cb_o))})
                drows.append(row)
        print(f"  [{target}] s_tgt {s_tgt:.3f}, {len(groups)} groups")
    res, gres, dec = pd.DataFrame(rows), pd.DataFrame(grows), pd.DataFrame(drows)
    res.to_csv(out_dir / "summary.csv", index=False)
    gres.to_csv(out_dir / "groups.csv", index=False)
    dec.to_csv(out_dir / "decomposition.csv", index=False)
    print(f"  checks vs correction_titan.csv: {len(checks)} values, max |diff| {max(checks):.2e}; identity max |residual| {idmax:.2e}")

    L = ["# 診断: 借りた倍率で中心の距離が悪くなる理由（TITAN、主解析の出力ではない）", "",
         "`transfer_scale_origin.py` が作る。定義は docstring。値は群（組織型・グリオーマ 3 群）ごとの中心の距離 / 施設内のばらつきの中央値、"
         "近傍の混ざりの中央値、隣り合う組の数。(i) = 倍率は推定元から借り、広げる中心だけ当て先の院内平均に替えたもの。(ii) = 倍率も当て先自身の値。", "",
         f"確認: 平行移動・借りた相似変換・借りた ComBat の値と correction_titan.csv の一致 {len(checks)} 値（最大の差 {max(checks):.1e}）、"
         f"恒等式の残り（最大）{idmax:.1e}。", ""]
    for target in TARGETS:
        d = res[res.target == target]
        tb = []
        for source in pd.unique(d.source):
            row = {"推定元": source, "s（借りた / 当て先）": f"{d[d.source == source].s_src.iloc[0]:.3f} / {d.s_tgt.iloc[0]:.3f}"}
            for v in VARIANTS:
                r = d[(d.source == source) & (d.variant == v)].iloc[0]
                pr = "" if pd.isna(r.pairs_adjacent) else f" / {int(r.pairs_adjacent)}"
                row[VARIANT_LABEL[v]] = f"{r.group_gap_sd:.3f} / {r.knn_mix:.2f}{pr}"
            tb.append(row)
        L += [f"## 当て先 {target}（中心の距離 / 近傍の混ざり / 隣り合う組）", "", _md(pd.DataFrame(tb), index=False), ""]
    dd = dec[dec.source != "own"]
    agg = dd.groupby("target").agg(groups=("group", "size"), diff_sq_aff=("diff_sq_aff", "sum"),
                                   e2_aff=("e_aff", lambda v: float((v ** 2).sum())), cross_aff=("cross_aff", "sum"),
                                   diff_sq_i=("diff_sq_i", "sum"), e2_i=("e_i", lambda v: float((v ** 2).sum())))
    agg["share_e2_aff"] = agg["e2_aff"] / agg["diff_sq_aff"]
    L += ["## 分解（own を除く推定元 × 群の和）", "",
          "‖r_aff‖² − ‖r_tr‖² = ‖e‖² + 2⟨r_tr, e⟩（恒等式）。e = −(s − 1)(μ_P,g − μ_P,src)。(i) の e' = −(s − 1)(μ_P,g − m_P,tgt)。", "",
          _md(agg.round(3), index=True), ""]
    (out_dir / "summary.md").write_text("\n".join(L))
    print(f"  saved: {out_dir}")


if __name__ == "__main__":
    main()
