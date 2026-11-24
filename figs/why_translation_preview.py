"""v4 Fig 5（なぜ平行移動だけか）のプレビュー 2 案（ken の返事待ち。figs/__init__.py の why_translation はスクリプト無しの席）。

案 A（倍率は腫瘍で違う）: _preview_why_translation_A
    a 群ごとの倍率 s（EBRAINS / 院内の施設内の広がりの比、1 = 同じ広がり）と 95% 区間（患者単位のブートストラップ 1,000 回）。
      7 群 ＋ 髄膜腫のサブタイプ 6 つ。題に「7 群で同じ」と「サブタイプで同じ」の検定の p
    b 比べる側: 同じ群の施設のずれと髄膜腫だけのずれの cos と区間（こちらは揃う）
    入力: output_dir/dispersion_ratio/scale_by_group.csv、scale_test.csv（dispersion_ratio.py）
案 B（既存の補正をそのまま借りると中心が崩れる）: _preview_why_translation_B
    a 中心の距離（群の中央値）。横軸 = 平行移動を借りたとき、縦軸 = ComBat（丸）・相似変換（三角）を借りたとき。点 = 当て先 × 推定元
      （当て先 paper / meningioma / glioma / lymphoma / metastasis / all、推定元は own・none 以外の全部）。対角線より上なら平行移動の方が良い
    b 近傍の混ざり（両施設 5 枚以上の群の中央値）。a と同じ形だが、こちらは高いほど良いので対角線より上なら借りた ComBat / 相似変換の方が良い
    c 対照: 借りた ComBat / 中心を当て先の施設平均で取り直した ComBat（ComBat (recentred)）/ 同じ腫瘍の ComBat の近傍の混ざり
    d 機構（out/_diagnostics/scale_origin/）: 平行移動 / 借りた相似変換 / 借りた倍率で広げる中心だけ当て先の院内平均に替えたもの、の中心の距離
    入力: output_dir/transfer/correction_titan.csv・correction_titan_groups.csv（transfer_correction.py）、
          output_dir/_diagnostics/scale_origin/summary.csv（transfer_scale_origin.py）
出力: paper.dir/v4/fig/_preview_why_translation_{A,B}.{png,pdf}、同 _values.csv（paper.version によらず v4 に出す）

Usage:
    uv run python -m figs.why_translation_preview
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figs.common import CONDITION_COLORS, CONDITION_LABELS, FONT, config, out_root, panel, paper_root
from shift_transfer import FAMILY_COLOR
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_LABELS

PAPER_VERSION = "v4"
NAME = "_preview_why_translation"
MIN_N = 5
TARGETS_B = ["paper", "meningioma", "glioma", "lymphoma", "metastasis", "all"]
PAIRS_C = [("glioma", "meningioma"), ("meningioma", "glioma"), ("lymphoma", "glioma"), ("metastasis", "glioma")]
PAIRS_D = PAIRS_C + [("meningioma", "metastasis")]
NAME_T = {"paper": "Meningioma + SFT", "meningioma": "Meningioma", "glioma": "Glioma", "lymphoma": FAMILY_LABELS["lymphoma"],
          "metastasis": "Metastasis", "all": "All 5"}


def _save(fig: plt.Figure, cfg: dict, tag: str) -> None:
    d = paper_root(cfg) / PAPER_VERSION / "fig"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{NAME}_{tag}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {d / f'{NAME}_{tag}'}.png / .pdf")


def plan_a(cfg: dict) -> pd.DataFrame:
    dr = out_root(cfg) / "dispersion_ratio"
    t = pd.read_csv(dr / "scale_by_group.csv")
    tests = pd.read_csv(dr / "scale_test.csv").set_index("test")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(t))[::-1]
    colors = [FAMILY_COLOR.get(f, "gray") for f in t["family"]]
    names = [GLIOMA_GROUP_LABELS.get(g, g) for g in t["group"]]
    for ax, (key, xlab, ref, letter) in zip(axes, (("scale_s", "Scale s (EBRAINS / in-house spread)", 1.0, "a"),
                                                   ("cos_vs_meni", "cos with the meningioma site shift", None, "b"))):
        for yi, r, col in zip(y, t.itertuples(), colors):
            v, lo, hi = getattr(r, key), getattr(r, f"{key}_lo"), getattr(r, f"{key}_hi")
            if not np.isfinite(v):
                continue
            ax.plot([lo, hi], [yi, yi], color=col, lw=1.2)
            ax.scatter([v], [yi], s=18, color=col, zorder=3, marker="o" if r.kind != "subtype" else "s")
        if ref is not None:
            ax.axvline(ref, color="0.6", lw=0.8, ls=":")
        ax.axhline(y[t["kind"].eq("subtype").idxmax()] + 0.5, color="0.85", lw=0.6)
        ax.set_xlabel(xlab)
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, letter, x=-0.05 if letter == "b" else -0.62, y=1.02)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=FONT - 1)
    f7, fs = tests.loc["families_7"], tests.loc["meningioma_subtypes"]
    axes[0].set_title(f"Scale differs between tumours: 7 groups p = {f7.p_value:.3f};\nmeningioma subtypes p = {fs.p_value:.3f}",
                      fontsize=FONT - 1, loc="left")
    axes[1].set_title("…while the shift direction is shared", fontsize=FONT - 1, loc="left")
    axes[1].set_xlim(0.5, 1.0)
    _save(fig, cfg, "A")
    return t


def _median_min_n(g: pd.DataFrame, metric: str) -> pd.Series:
    d = g[(g["metric"] == metric) & (g["n_ebrains"] >= MIN_N) & (g["n_patho2"] >= MIN_N)]
    return d.groupby(["target", "source", "method"])["value"].median()


def plan_b(cfg: dict) -> pd.DataFrame:
    tdir = out_root(cfg) / "transfer"
    corr = pd.read_csv(tdir / "correction_titan.csv")
    grp = pd.read_csv(tdir / "correction_titan_groups.csv")
    gap = corr[corr["metric"] == "group_gap_sd"].set_index(["target", "source", "method"])["value"]
    knn = _median_min_n(grp, "knn_mix")
    keys = sorted({(t, s) for t, s, _ in gap.index if t in TARGETS_B and s not in ("none", "own")})
    rows = [{"target": t, "source": s, **{f"gap_{m}": gap.get((t, s, m), np.nan) for m in ("centroid", "affine_free", "combat")},
             **{f"knn_{m}": knn.get((t, s, m), np.nan) for m in ("centroid", "affine_free", "combat")}} for t, s in keys]
    v = pd.DataFrame(rows)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.32, left=0.09, right=0.98, top=0.95, bottom=0.12)
    for k, (pref, title) in enumerate((("gap", "Site-centre distance (group median; lower is better)"),
                                       ("knn", f"Neighbour mixing (groups ≥ {MIN_N} per site; higher is better)"))):
        ax = fig.add_subplot(gs[0, k])
        for m, mk in (("combat", "o"), ("affine_free", "^")):
            ax.scatter(v[f"{pref}_centroid"], v[f"{pref}_{m}"], s=16, marker=mk, color=CONDITION_COLORS[m], alpha=0.8,
                       label=f"{CONDITION_LABELS[m]} borrowed")
        lo = float(np.nanmin(v[[c for c in v.columns if c.startswith(pref)]].to_numpy())) * 0.9
        hi = float(np.nanmax(v[[c for c in v.columns if c.startswith(pref)]].to_numpy())) * 1.05
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("Translation borrowed")
        ax.set_ylabel("Scale-aware correction borrowed")
        ax.set_title(title, fontsize=FONT, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=FONT - 2, frameon=False, loc="upper left")
        panel(ax, "ab"[k], x=-0.2, y=1.02)
    # c
    ax = fig.add_subplot(gs[1, 0])
    w = 0.26
    for j, (m, src_of, lab, col) in enumerate((("combat", None, "ComBat borrowed", CONDITION_COLORS["combat"]),
                                               ("combat_recentred", None, "ComBat (recentred)", "#9e9ac8"),
                                               ("combat", "own", "ComBat, same tumour", "#1baf7a"))):
        ys = [knn.get((t, src_of or s, m), np.nan) for t, s in PAIRS_C]
        ax.bar(np.arange(len(PAIRS_C)) + (j - 1) * w, ys, width=w * 0.92, color=col, label=lab)
    ax.set_xticks(range(len(PAIRS_C)))
    ax.set_xticklabels([f"{NAME_T[t]}\n← {NAME_T[s]}" for t, s in PAIRS_C], fontsize=FONT - 2)
    ax.set_ylabel(f"Neighbour mixing (≥ {MIN_N} per site)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Control: re-centre the borrowed ComBat", fontsize=FONT, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FONT - 2.5, frameon=False, loc="upper left", ncol=1)
    panel(ax, "c", x=-0.2, y=1.02)
    # d
    so = pd.read_csv(out_root(cfg) / "_diagnostics" / "scale_origin" / "summary.csv").set_index(["target", "source", "variant"])
    ax = fig.add_subplot(gs[1, 1])
    for j, (var, lab, col) in enumerate((("translation", "Translation", CONDITION_COLORS["centroid"]),
                                         ("affine_borrowed", "Similarity borrowed", CONDITION_COLORS["affine_free"]),
                                         ("affine_tgt_center", "Borrowed scale,\ncentre at target mean", "#6a51a3"))):
        ys = [float(so.loc[(t, s, var), "group_gap_sd"]) for t, s in PAIRS_D]
        ax.bar(np.arange(len(PAIRS_D)) + (j - 1) * w, ys, width=w * 0.92, color=col, label=lab)
    ax.set_xticks(range(len(PAIRS_D)))
    ax.set_xticklabels([f"{NAME_T[t]}\n← {NAME_T[s]}" for t, s in PAIRS_D], fontsize=FONT - 2.5)
    ax.set_ylabel("Site-centre distance (group median)")
    ax.set_title("Mechanism: where the scaling is centred", fontsize=FONT, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FONT - 2.5, frameon=False, loc="upper left", ncol=1)
    panel(ax, "d", x=-0.2, y=1.02)
    _save(fig, cfg, "B")
    return v


def main() -> None:
    print(f"[preview] {NAME}")
    cfg = config()
    a = plan_a(cfg)
    b = plan_b(cfg)
    d = paper_root(cfg) / PAPER_VERSION / "fig"
    a.to_csv(d / f"{NAME}_A_values.csv", index=False)
    b.to_csv(d / f"{NAME}_B_values.csv", index=False)
    above = (b["gap_combat"] > b["gap_centroid"]).sum()
    print(f"  B a: ComBat borrowed worse than translation in {above} / {b['gap_combat'].notna().sum()}; "
          f"similarity worse in {(b['gap_affine_free'] > b['gap_centroid']).sum()} / {b['gap_affine_free'].notna().sum()}")
    print(f"  B b: knn ComBat > translation in {(b['knn_combat'] > b['knn_centroid']).sum()} / {b['knn_combat'].notna().sum()}")


if __name__ == "__main__":
    main()
