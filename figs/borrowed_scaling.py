"""v4 Fig 5 倍率を含む補正を借りると何が動くか（観察として示す。補正法の優劣は言わない。ken 2026-11-24）。TITAN。

題・軸・注には量の定義だけを書き、良し悪しの語（better / worse など）は入れない。対角線は「等しい」の意味だけ。

a 一覧（表の形）。当て先 2 つ（グリオーマ ← 髄膜腫、髄膜腫 ← グリオーマ）を左右に。行 = 補正なし / 同じ腫瘍で推定（平行移動・相似変換・
  ComBat）/ 別の腫瘍から借りる（同）。列 = 施設をまたぐ分類（最近傍重心法の balanced accuracy、EBRAINS で作った重心で院内を当てる
  E→P と逆向き P→E。グリオーマは 3 群、髄膜腫は 5 サブタイプ）、2 施設の中心の距離（群の中央値、施設内の広がり = 1）、
  隣り合う組の数、近傍の混ざり（両施設 5 枚以上の群の中央値、1 = 施設を見分けられない）、施設のシルエット係数。
  セルの色は列ごとに「補正なし」からの変化の大きさ（向きは問わない単色。順位づけではない）。
  診断名の検索は相似変換・ComBat の条件が無いので入れない。
b 中心の距離の散布図。横軸 = 平行移動を借りたとき、縦軸 = ComBat（丸）・相似変換（三角）を借りたとき。点 = 当て先 × 推定元の 39 通り
  （当て先 6 つ: 髄膜腫・グリオーマ・リンパ腫・転移・髄膜腫＋SFT・5 つの腫瘍全体、推定元は同じ腫瘍と補正なしを除く全部）。破線 = 等しい
c 近傍の混ざりの散布図（両施設 5 枚以上の群の中央値）。b と同じ形
d 群ごとの倍率 s（EBRAINS / 院内の施設内の広がりの比）と 95% 区間（患者単位のブートストラップ）。7 群 ＋ 髄膜腫のサブタイプ 6 つ。
  点線 = 1。注に「s が群で等しい」の検定の p
e 倍率を掛ける中心を替えた比較（out/_diagnostics/scale_origin/）。当て先 ← 推定元 5 つ、中心の距離、3 本: 平行移動を借りる /
  相似変換を借りる（推定元の院内平均を中心に広げる）/ 借りた倍率で、広げる中心だけ当て先の院内スライド全体の平均にしたもの。
  リンパ腫・転移は群が 1 つなので 3 本目の中心の位置は平行移動と同じで、差は分母（施設内の広がり）による
借りた ComBat の中心を当て先で取り直した対照は Supplementary（figs/combat_recentred.py）。

入力: output_dir/transfer/correction_titan.csv・correction_titan_groups.csv（transfer_correction.py）、
      output_dir/dispersion_ratio/scale_by_group.csv・scale_test.csv（dispersion_ratio.py）、
      output_dir/_diagnostics/scale_origin/summary.csv（transfer_scale_origin.py）
出力: fig/{fig n}_borrowed_scaling.{png,pdf}、同 _values.csv（a〜e の数値）

Usage:
    uv run python -m figs.borrowed_scaling
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from figs import label
from figs.common import CONDITION_COLORS, CONDITION_LABELS, FONT, config, fig_path, out_root, panel, save
from shift_transfer import FAMILY_COLOR
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_LABELS

NAME = "borrowed_scaling"
MIN_N = 5
METHODS = ["centroid", "affine_free", "combat"]
TARGETS_SCATTER = ["paper", "meningioma", "glioma", "lymphoma", "metastasis", "all"]
PAIRS_E = [("glioma", "meningioma"), ("meningioma", "glioma"), ("lymphoma", "glioma"), ("metastasis", "glioma"), ("meningioma", "metastasis")]
NAME_T = {"paper": "Meningioma + SFT", "meningioma": "Meningioma", "glioma": "Glioma", "lymphoma": FAMILY_LABELS["lymphoma"],
          "metastasis": "Metastasis", "all": "All 5"}
# a の当て先: (当て先, 借りる推定元, 題, 分類の課題の列名の接頭辞, 分類の説明)
BLOCKS = [("glioma", "meningioma", "Glioma ← meningioma", "nc_glioma3_ba", "3 groups"),
          ("meningioma", "glioma", "Meningioma ← glioma", "nc_histotype_ba", "5 subtypes")]
ROWS = [("none", "centroid", "No correction"), None,
        ("own", "centroid", "Translation"), ("own", "affine_free", "Similarity"), ("own", "combat", "ComBat"), None,
        ("src", "centroid", "Translation"), ("src", "affine_free", "Similarity"), ("src", "combat", "ComBat")]
HEADERS = ["Estimated on the same tumour", "Borrowed from the other tumour"]
COLS = [("ba_E2P", "Classif.\nBA E→P"), ("ba_P2E", "Classif.\nBA P→E"), ("group_gap_sd", "Centre\ndistance"),
        ("pairs_adjacent", "Cross-site\npairs"), ("knn5", "Neighbour\nmixing"), ("asw_site", "Site\nsilhouette")]
SHADE = "#4c72b0"   # 変化の大きさの単色（白 → この色の 45%）


def load(cfg: dict) -> tuple[pd.Series, pd.Series]:
    tdir = out_root(cfg) / "transfer"
    corr = pd.read_csv(tdir / "correction_titan.csv")
    grp = pd.read_csv(tdir / "correction_titan_groups.csv")
    val = corr.set_index(["target", "source", "method", "metric"])["value"]
    g = grp[(grp["metric"] == "knn_mix") & (grp["n_ebrains"] >= MIN_N) & (grp["n_patho2"] >= MIN_N)]
    knn5 = g.groupby(["target", "source", "method"])["value"].median()
    return val, knn5


def overview_values(val: pd.Series, knn5: pd.Series) -> pd.DataFrame:
    rows = []
    for target, src, title, task, _ in BLOCKS:
        for r in ROWS:
            if r is None:
                continue
            source = src if r[0] == "src" else r[0]
            key = (target, source, r[1])
            rows.append({"target": target, "block": title, "row": ("borrowed" if r[0] == "src" else r[0]), "source": source, "method": r[1],
                         "ba_E2P": val[key + (f"{task}_E2P",)], "ba_P2E": val[key + (f"{task}_P2E",)],
                         "group_gap_sd": val[key + ("group_gap_sd",)], "pairs_adjacent": val[key + ("pairs_adjacent",)],
                         "pairs_n": val[key + ("pairs_n",)], "knn5": knn5[key], "asw_site": val[key + ("asw_site",)]})
    return pd.DataFrame(rows)


def draw_overview(fig: plt.Figure, rect: list[float], ov: pd.DataFrame) -> None:
    """a: 当て先ごとに表を 1 枚。セルの色 = 列ごとの |値 − 補正なし| / 列の最大（単色）。"""
    x0, y0, w, h = rect
    gap = 0.03
    bw = (w - gap) / 2
    base = np.array(to_rgb(SHADE))
    for k, (target, src, title, task, task_label) in enumerate(BLOCKS):
        ax = fig.add_axes([x0 + k * (bw + gap), y0, bw, h])
        d = ov[ov["target"] == target].reset_index(drop=True)
        none = d.iloc[0]
        nrow, ncol = len(ROWS), len(COLS)
        img = np.ones((nrow, ncol, 3))
        it = iter(d.itertuples())
        for i, r in enumerate(ROWS):
            if r is None:
                continue
            row = next(it)
            for j, (c, _) in enumerate(COLS):
                v = getattr(row, c)
                ref = none[c]
                col_vals = d[c].to_numpy(dtype=float)
                span = np.nanmax(np.abs(col_vals - ref))
                t = abs(v - ref) / span if span > 0 else 0.0
                img[i, j] = 1 - 0.45 * t * (1 - base)
                txt = f"{int(v)}/{int(row.pairs_n)}" if c == "pairs_adjacent" else f"{v:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=FONT - 2)
        ax.imshow(img, aspect="auto", interpolation="nearest")
        for i, r in enumerate(ROWS):   # 見出しの行（同じ腫瘍で推定 / 借りる）
            if r is None:
                hdr = HEADERS[0] if i < 3 else HEADERS[1]
                ax.text(-0.45, i, hdr + (f" ({FAMILY_LABELS.get(src, src).lower()})" if i > 3 else ""), ha="left", va="center",
                        fontsize=FONT - 2, fontstyle="italic", color="0.25")
        for i in range(nrow + 1):
            ax.axhline(i - 0.5, color="0.85", lw=0.4)
        ax.set_xticks(range(ncol))
        ax.set_xticklabels([n for _, n in COLS], fontsize=FONT - 2.5)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", length=0, pad=2)
        ax.set_yticks([i for i, r in enumerate(ROWS) if r is not None])
        ax.set_yticklabels([r[2] for r in ROWS if r is not None] if k == 0 else [], fontsize=FONT - 2)
        ax.tick_params(axis="y", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Target: {target}; Δ from {src}", fontsize=FONT - 1, loc="left", pad=22)
        if k == 0:
            panel(ax, "a", x=-0.26, y=1.16)
    fig.text(x0, y0 - 0.012, "Classification: nearest-centroid balanced accuracy across sites (E→P: EBRAINS centroids applied to in-house "
             "slides; P→E: the reverse);\n" + ", ".join(f"{t} {lab}" for t, _, _, _, lab in BLOCKS) + ". Shading: size of the "
             "change from no correction, scaled within each column (either direction).", fontsize=FONT - 2.5, va="top", color="0.25")


def scatter(ax: plt.Axes, v: pd.DataFrame, pref: str, title: str) -> None:
    for m, mk in (("combat", "o"), ("affine_free", "^")):
        ax.scatter(v[f"{pref}_centroid"], v[f"{pref}_{m}"], s=14, marker=mk, color=CONDITION_COLORS[m], alpha=0.8, linewidths=0)
    cols = [f"{pref}_{m}" for m in METHODS]
    lo = float(np.nanmin(v[cols].to_numpy())) * 0.9
    hi = float(np.nanmax(v[cols].to_numpy())) * 1.05
    ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Translation, borrowed", fontsize=FONT - 1)
    ax.set_ylabel("ComBat / similarity, borrowed", fontsize=FONT - 1)
    ax.tick_params(labelsize=FONT - 2)
    ax.set_title(title, fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=CONDITION_COLORS["combat"], markersize=4, label="ComBat, borrowed"),
                       Line2D([], [], marker="^", ls="", color=CONDITION_COLORS["affine_free"], markersize=4, label="Similarity, borrowed"),
                       Line2D([], [], ls="--", color="0.5", lw=0.8, label="equal")],
              fontsize=FONT - 2.5, frameon=False, loc="upper left")


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    val, knn5 = load(cfg)
    ov = overview_values(val, knn5)

    # b・c の 39 通り
    gap = val.xs("group_gap_sd", level="metric")
    keys = sorted({(t, s) for t, s, _ in gap.index if t in TARGETS_SCATTER and s not in ("none", "own")})
    v = pd.DataFrame([{"target": t, "source": s, **{f"gap_{m}": gap.get((t, s, m), np.nan) for m in METHODS},
                       **{f"knn_{m}": knn5.get((t, s, m), np.nan) for m in METHODS}} for t, s in keys])

    fig = plt.figure(figsize=(7.2, 9.4))
    draw_overview(fig, [0.17, 0.765, 0.81, 0.18], ov)
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.98, top=0.66, bottom=0.05, hspace=0.42, wspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    scatter(ax, v, "gap", "Site-centre distance\n(group median; within-site spread = 1)")
    panel(ax, "b", x=-0.22, y=1.04)
    ax = fig.add_subplot(gs[0, 1])
    scatter(ax, v, "knn", f"Neighbour mixing (median of groups with\n≥ {MIN_N} slides per site; 1 = sites indistinguishable)")
    panel(ax, "c", x=-0.22, y=1.04)

    # d 群ごとの倍率
    dr = out_root(cfg) / "dispersion_ratio"
    t = pd.read_csv(dr / "scale_by_group.csv")
    tests = pd.read_csv(dr / "scale_test.csv").set_index("test")
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(t))[::-1]
    for yi, r in zip(y, t.itertuples()):
        col = FAMILY_COLOR.get(r.family, "gray")
        ax.plot([r.scale_s_lo, r.scale_s_hi], [yi, yi], color=col, lw=1.1)
        ax.scatter([r.scale_s], [yi], s=14, color=col, zorder=3, marker="s" if r.kind == "subtype" else "o")
    ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
    ax.axhline(y[t["kind"].eq("subtype").idxmax()] + 0.5, color="0.85", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([GLIOMA_GROUP_LABELS.get(g, g).replace(" meningioma", "") for g in t["group"]], fontsize=FONT - 2.5)
    ax.tick_params(axis="x", labelsize=FONT - 2)
    ax.set_xlabel("Scale s = EBRAINS / in-house within-site spread", fontsize=FONT - 1)
    f7, fs = tests.loc["families_7"], tests.loc["meningioma_subtypes"]
    ax.set_title(f"Scale per group (95% CI, patient bootstrap)\nequal-s test: 7 groups p = {f7.p_value:.3f}; "
                 f"subtypes p = {fs.p_value:.3f}", fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "d", x=-0.55, y=1.04)

    # e 倍率を掛ける中心
    so = pd.read_csv(out_root(cfg) / "_diagnostics" / "scale_origin" / "summary.csv").set_index(["target", "source", "variant"])
    ax = fig.add_subplot(gs[1, 1])
    w = 0.26
    variants = [("translation", "Translation, borrowed", CONDITION_COLORS["centroid"]),
                ("affine_borrowed", "Similarity, borrowed\n(scaled about the source mean)", CONDITION_COLORS["affine_free"]),
                ("affine_tgt_center", "Borrowed scale, scaled about\nthe target in-house mean", "#6a51a3")]
    evals = []
    for j, (var, lab, col) in enumerate(variants):
        ys = [float(so.loc[(tt, s, var), "group_gap_sd"]) for tt, s in PAIRS_E]
        ax.bar(np.arange(len(PAIRS_E)) + (j - 1) * w, ys, width=w * 0.92, color=col, label=lab)
        evals += [{"panel": "e", "target": tt, "source": s, "variant": var, "group_gap_sd": yv} for (tt, s), yv in zip(PAIRS_E, ys)]
    ax.set_xticks(range(len(PAIRS_E)))
    ax.set_xticklabels([f"{NAME_T[tt]}\n← {NAME_T[s].lower() if s != 'lymphoma' else NAME_T[s]}" for tt, s in PAIRS_E],
                       fontsize=FONT - 3)
    ax.set_ylim(0, 1.25)
    ax.tick_params(axis="y", labelsize=FONT - 2)
    ax.set_ylabel("Site-centre distance (within-site spread = 1)", fontsize=FONT - 1.5)
    ax.set_title("Centre of the borrowed scaling", fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FONT - 3, frameon=False, loc="upper left", ncol=1)
    panel(ax, "e", x=-0.22, y=1.04)
    save(fig, cfg, NAME)

    out = pd.concat([ov.assign(panel="a"), v.assign(panel="bc"),
                     t[["group", "family", "kind", "scale_s", "scale_s_lo", "scale_s_hi"]].assign(panel="d"), pd.DataFrame(evals)],
                    ignore_index=True)
    out.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    pd.set_option("display.width", 200)
    print(ov.drop(columns=["target"]).round(3).to_string(index=False))
    print(f"  b: ComBat ≥ translation {(v.gap_combat > v.gap_centroid).sum()}/{v.gap_combat.notna().sum()}, "
          f"similarity {(v.gap_affine_free > v.gap_centroid).sum()}/{v.gap_affine_free.notna().sum()}; "
          f"c: ComBat > translation {(v.knn_combat > v.knn_centroid).sum()}/{v.knn_combat.notna().sum()}, "
          f"similarity {(v.knn_affine_free > v.knn_centroid).sum()}/{v.knn_affine_free.notna().sum()}")


if __name__ == "__main__":
    main()
