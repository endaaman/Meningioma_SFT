"""サブタイプ間距離のヒートマップ（補正なし / GAN / ComBat / centroid の 4 枚、Supplementary）。

距離は本文 Fig 6（subtype_structure）と同じ euc_mean: 施設 × サブタイプの平均ベクトル間ユークリッド距離を
行列ごとに最大値で割った値（対角 0 なので min-max 正規化と同じ）。カラースケールは 4 枚共通（0〜1）、色バーは 1 本。
並びは 4 枚とも**同じ固定順**（クラスタリングで並べ替えない）: サブタイプ順（utils.display.ordered_subtypes）で、
各サブタイプの EBRAINS と patho2 を隣同士に置く。並びが同じなので 4 枚を並べて比べられる。樹形図は付けない。

入力: output_dir/dendrogram/{original,centroid,gan}/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py）
出力: fig/{fig Sn}_subtype_distance_heatmap.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_distance_heatmap
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dendrogram
from figs import label
from figs.common import CONDITIONS_FULL, FONT, VARIANT_DISPLAY, config, out_root, panel, save, src_markers
from utils.display import ordered_subtypes, shorten, subtype_color_map

NAME = "subtype_distance_heatmap"
VARIANTS = CONDITIONS_FULL
CMAP = "viridis_r"            # 近い = 明るい（本文の旧ヒートマップと同じ向き）


def load(cfg: dict, variant: str) -> tuple[pd.DataFrame, np.ndarray]:
    groups, D, _ = dendrogram.load_raw(out_root(cfg) / "dendrogram" / cfg["variants"].get(variant, variant))
    return groups.reset_index(drop=True), D / D.max()


def fixed_order(groups: pd.DataFrame, cfg: dict) -> list[str]:
    """サブタイプ順 × 施設（EBRAINS, patho2）の label の並び。4 枚で共通に使う。"""
    subs = ordered_subtypes(set(groups["subtype"]), cfg)
    sources = sorted(groups["source"].unique())
    by = {(r.subtype, r.source): r.label for r in groups.itertuples()}
    return [by[(s, src)] for s in subs for src in sources if (s, src) in by]


def draw_heatmap(ax: plt.Axes, groups: pd.DataFrame, D: np.ndarray, order: list[str], cfg: dict,
                 yticks: bool) -> plt.cm.ScalarMappable:
    idx = groups.reset_index().set_index("label").loc[order, "index"].to_numpy()
    im = ax.imshow(D[np.ix_(idx, idx)], cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
    g = groups.set_index("label").loc[order]
    colors = subtype_color_map(list(dict.fromkeys(g["subtype"])), cfg)
    mkr = src_markers(cfg, sorted(groups["source"].unique()))
    n = len(order)
    # サブタイプの区切り（2 施設ずつ）
    for k in range(2, n, 2):
        ax.axhline(k - 0.5, color="white", lw=0.6)
        ax.axvline(k - 0.5, color="white", lw=0.6)
    # 軸: 施設マーカー（サブタイプ色）＋ サブタイプ略称（2 行に 1 つ）
    for i, (sub, src) in enumerate(zip(g["subtype"], g["source"])):
        ax.scatter(i, n - 0.5 + 0.55, marker=mkr[src], color=colors[sub], s=9, clip_on=False)
        if yticks:
            ax.scatter(-0.5 - 0.55, i, marker=mkr[src], color=colors[sub], s=9, clip_on=False)
    centers = np.arange(0.5, n, 2)
    subs = list(g["subtype"])[::2]
    ax.set_xticks(centers)
    ax.set_xticklabels([shorten(s, cfg) for s in subs], rotation=90, fontsize=FONT - 1)
    ax.tick_params(axis="x", length=0, pad=10)
    if yticks:
        ax.set_yticks(centers)
        ax.set_yticklabels([shorten(s, cfg) for s in subs], fontsize=FONT - 1)
        ax.tick_params(axis="y", length=0, pad=10)
    else:
        ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    data = {v: load(cfg, v) for v in VARIANTS}
    order = fixed_order(data["centroid"][0], cfg)
    for v, (groups, _) in data.items():
        assert set(groups["label"]) == set(order), f"{v}: groups differ from the fixed order"
    # 各枚を同じ大きさの正方形 axes に（左端だけ y ラベルの余白を取る）
    # 2 × 2 に並べる（左列だけ y ラベル、下段だけ x ラベルの余白を取る）
    side, gap, left, bottom, top, cb_w, vgap = 3.1, 0.25, 0.75, 0.95, 0.45, 0.12, 0.45
    ncol = 2
    nrow = (len(VARIANTS) + ncol - 1) // ncol
    fig_w = left + ncol * side + (ncol - 1) * gap + 0.25 + cb_w + 1.0
    fig_h = bottom + nrow * side + (nrow - 1) * (vgap + bottom - 0.3) + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, (v, letter) in enumerate(zip(VARIANTS, "abcdefg")):
        r, c = divmod(i, ncol)
        y0 = fig_h - top - (r + 1) * side - r * (vgap + bottom - 0.3)
        ax = fig.add_axes([(left + c * (side + gap)) / fig_w, y0 / fig_h, side / fig_w, side / fig_h])
        groups, D = data[v]
        im = draw_heatmap(ax, groups, D, order, cfg, yticks=(c == 0))
        ax.set_title(VARIANT_DISPLAY.get(v, v), fontsize=FONT + 1, pad=4)
        panel(ax, letter, x=-0.16 if c == 0 else -0.03, y=1.03)
    cax = fig.add_axes([(left + ncol * side + (ncol - 1) * gap + 0.25) / fig_w, bottom / fig_h,
                        cb_w / fig_w, (fig_h - bottom - top) / fig_h])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Normalized distance between mean embeddings\n(euc_mean)", fontsize=FONT)
    cb.ax.tick_params(labelsize=FONT - 1)
    names = {"ebrains": "EBRAINS", "patho2": "patho2"}
    handles = [plt.Line2D([0], [0], marker=m, color="#555555", ls="none", markersize=4, label=names.get(s, s))
               for s, m in src_markers(cfg, sorted(data["centroid"][0]["source"].unique())).items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, -0.01), ncol=2, frameon=False,
               fontsize=FONT)
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
