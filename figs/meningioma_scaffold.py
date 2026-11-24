"""v4 Fig 2 髄膜腫で足場を作る（GAN → 平行移動）: 髄膜腫では、画像を EBRAINS 側へ寄せる GAN で埋め込みの施設差が縮み、
埋め込みで平均を揃えるだけでそれより先まで行く（TITAN、髄膜腫だけ = EBRAINS 519 / 院内 555 枚、両施設にあるサブタイプ 12 組）。

a パッチの例。行 = 院内 / 院内 → GAN / EBRAINS、列 = 髄膜腫のサブタイプ 6 つ（v3 Fig 2a の figs/tissue.py の描画と候補をそのまま使い、
  SFT の列だけ外す）
b UMAP（補正なし / GAN / 平行移動、髄膜腫だけで条件ごとに当てはめ）。色 = 施設
c 補正法の比較（CONDITIONS_FIG3 の並び: 補正なし → Macenko → GAN → ComBat → 平行移動）。施設の平均どうしの距離、
  施設のシルエット係数、隣り合う組（12 組中）
d 樹形図（補正なし / GAN / 平行移動）。Fig 4 と同じ見せ方（葉 = 施設 × サブタイプの平均、色 = サブタイプ、形 = 施設、
  同じサブタイプが隣り合う組に点線の四角、題に組の数）
サブタイプごとの施設のずれの一様さは Fig 3 に含まれるので入れない。

入力: output_dir/method_compare_meningioma/{summary.csv, umap/{original,gan,centroid}/coords.csv,
      structure/{original,gan,centroid}/}（method_compare_meningioma.py）、paper.dir/v4/patch_candidates/（paper_patches.py の候補）
出力: fig/{fig n}_meningioma_scaffold.{png,pdf}

Usage:
    uv run python -m figs.meningioma_scaffold
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import dendrogram
from figs import label, tissue
from figs.common import CONDITION_COLORS, CONDITION_LABELS, CONDITIONS_FIG3, FONT, config, out_root, panel_fig, save
from figs.subtype_structure import check_frames, legend_handles, marker_specs

NAME = "meningioma_scaffold"
UMAP_CONDS = ["original", "gan", "centroid"]
TREE_CONDS = ["original", "gan", "centroid"]
SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "In-house"}


def _scale_bar(ax: plt.Axes, mpp: float, size_px: int) -> None:
    """figs.tissue._scale_bar の小さい版（パッチが小さいので文字を小さく、線を短い辺の内側に）。"""
    length = tissue.SCALE_UM / mpp
    x0, y0 = size_px * 0.08, size_px * 0.90
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=1.5, solid_capstyle="butt")
    ax.text(x0, y0 - size_px * 0.04, f"{tissue.SCALE_UM} µm", color="white", ha="left", va="bottom", fontsize=FONT - 3, fontweight="bold")


def site_legend(handles: list) -> list:
    """figs.subtype_structure.legend_handles の施設の表記を v4 の表記（In-house）にし、マーカーを小さくする。"""
    for h in handles:
        if isinstance(h, Line2D) and h.get_label() == "patho2":
            h.set_label("In-house")
        if isinstance(h, Line2D) and h.get_marker() not in (None, "None", "none", ""):
            h.set_markersize(5)
    return handles


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    src = out_root(cfg) / "method_compare_meningioma"
    summ = pd.read_csv(src / "summary.csv")
    summ = summ[summ["definition"] == "meningioma"].set_index("condition")
    site_colors = cfg.get("display", {}).get("colors", {}).get("sources", {})

    fig = plt.figure(figsize=(7.2, 8.6))
    top = fig.add_gridspec(3, 6, left=0.08, right=0.60, top=0.965, bottom=0.715, wspace=0.04, hspace=0.04)
    # a: v3 の描画（figs.tissue.draw_patch_grid）を SFT の列を外して使う（実行中だけ列の定義を差し替える。tissue.py は変えない）
    # 小さく描くので縮尺の文字を小さくし、行の名前は v4 の表記（In-house）にする
    saved = (tissue.COLUMNS, tissue.PICK, tissue._scale_bar, tissue._rows)
    cols, pick = tissue.COLUMNS, tissue.PICK
    tissue.COLUMNS = [c for c in cols if c != "SFT"]
    tissue.PICK = {s: [p for c, p in zip(cols, v) if c != "SFT"] for s, v in pick.items()}
    tissue._scale_bar = _scale_bar
    tissue._rows = lambda: [("In-house", "patho2", False), ("In-house\n→ GAN", "patho2", True), ("EBRAINS", "ebrains", False)]
    try:
        tissue.draw_patch_grid(fig, top, cfg)
    finally:
        tissue.COLUMNS, tissue.PICK, tissue._scale_bar, tissue._rows = saved
    panel_fig(fig, 0.005, 0.965, "a")

    # c: 補正法の比較（右上に 3 段）
    cgs = fig.add_gridspec(3, 1, left=0.71, right=0.99, top=0.965, bottom=0.715, hspace=0.25)
    x = range(len(CONDITIONS_FIG3))
    for k, (metric, ylab) in enumerate((("site_shift_norm", "Site-mean\ndistance"), ("asw_site", "Site\nsilhouette"),
                                        ("pairs_adjacent", "Cross-site\npairs (/12)"))):
        ax = fig.add_subplot(cgs[k, 0])
        ys = [float(summ.loc[c, metric]) for c in CONDITIONS_FIG3]
        ax.bar(x, ys, color=[CONDITION_COLORS[c] for c in CONDITIONS_FIG3], width=0.7)
        for xi, y in zip(x, ys):
            ax.text(xi, y, f"{y:.0f}" if metric == "pairs_adjacent" else (f"{y:.2f}" if metric == "site_shift_norm" else f"{y:.3f}"),
                    ha="center", va="bottom", fontsize=FONT - 3)
        ax.set_ylabel(ylab, fontsize=FONT - 2)
        ax.set_ylim(0, max(ys) * 1.3 if metric != "pairs_adjacent" else 12)
        ax.tick_params(axis="y", labelsize=FONT - 3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITIONS_FIG3] if k == 2 else [], rotation=35, ha="right", fontsize=FONT - 2.5)
        ax.spines[["top", "right"]].set_visible(False)
    panel_fig(fig, 0.625, 0.965, "c")

    # b: UMAP
    for k, cond in enumerate(UMAP_CONDS):
        ax = fig.add_axes([0.06 + k * 0.315, 0.405, 0.27, 0.22])
        u = pd.read_csv(src / "umap" / cond / "coords.csv")
        for s in ("patho2", "ebrains"):
            d = u[u["source"] == s]
            ax.scatter(d["umap1"], d["umap2"], s=2.5, color=site_colors.get(s, "gray"), alpha=0.6, linewidths=0, label=SITE_LABEL[s])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{CONDITION_LABELS[cond]}  (site silhouette {summ.loc[cond, 'asw_site']:.3f})", fontsize=FONT - 1, loc="left")
        if k == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=site_colors.get(s, "gray"), label=SITE_LABEL[s], markersize=4)
                               for s in ("ebrains", "patho2")], fontsize=FONT - 2, frameon=False, loc="lower left")
    panel_fig(fig, 0.005, 0.632, "b")

    # d: 樹形図（3 段、縦軸の上限は共通）
    data = {}
    for cond in TREE_CONDS:
        groups, D, Z = dendrogram.load_raw(src / "structure" / cond)
        data[cond] = (groups, pd.DataFrame(D, index=groups["label"], columns=groups["label"]), Z)
    ymax = max(dendrogram.dendrogram_height(mat) for _, mat, _ in data.values()) * 1.05
    for k, cond in enumerate(TREE_CONDS):
        groups, mat, Z = data[cond]
        ax = fig.add_axes([0.08, 0.285 - k * 0.093, 0.90, 0.068])
        n = dendrogram.draw_dendrogram(ax, mat, False, marker_specs(groups, cfg), ymax=ymax, marker_size=22, frame_lw=0.7,
                                       line_lw=0.6, ytick_size=FONT - 3, frame_pad=1.6, band_frac=0.35, bottom_frac=0.12)
        check_frames(ax)
        boxed = len(dendrogram.adjacent_pairs(Z, groups))
        assert n == boxed == int(summ.loc[cond, "pairs_adjacent"]), (cond, n, boxed)
        ax.set_title(f"{CONDITION_LABELS[cond]}   cross-site pairs: {n}/{groups['subtype'].nunique()}", fontsize=FONT - 1, loc="left", pad=2)
    ax_l = fig.add_axes([0.08, 0.015, 0.90, 0.04])
    ax_l.axis("off")
    ax_l.legend(handles=site_legend(legend_handles(data[TREE_CONDS[-1]][0], cfg)), loc="center", ncol=8, frameon=False,
                fontsize=FONT - 2, handlelength=1.2, handletextpad=0.4, columnspacing=0.9)
    panel_fig(fig, 0.005, 0.37, "d")
    save(fig, cfg, NAME)
    print(summ.loc[CONDITIONS_FIG3 + ["affine_free"], ["site_shift_norm", "asw_site", "pairs_adjacent", "group_gap_sd", "knn_mix"]].round(3).to_string())


if __name__ == "__main__":
    main()
