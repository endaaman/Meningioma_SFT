"""サブタイプ構造: 施設 × サブタイプ平均ベクトルの樹形図（a 補正前 / b centroid 補正後、縦 2 段・同じ縦軸）と
SFT からの距離（c）。

樹形図は既存の `out/dendrogram/{variant}/euc_mean_dendrogram_cross.png` と同じもの:
距離 = 平均ベクトル間のユークリッド距離（euc_mean）を行列ごとに min-max 正規化（対角 0 なので最大値で割るのと同じ）、
Ward 法。描画は dendrogram.draw_dendrogram をそのまま使う（葉 = サブタイプ色 × 施設マーカー、葉の並びで
同じサブタイプの 2 施設が隣り合ったら点線の四角）。各段の「cross-site pairs」はこの四角の数。
c も同じ euc_mean（施設内の SFT と各サブタイプの平均ベクトル間距離）を、b と同じ最大値で割った値。

入力:
    output_dir/dendrogram/{variant}/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py。D = euc_mean の生値）
    output_dir/dendrogram/pairs.csv                                            （dendrogram.py）
    output_dir/sft_distance/sft_distance_raw.csv                               （sft_distance_bar.py）
出力: fig/{fig n}_subtype_structure.{png,pdf}, 同 _pairs.csv, 同 _sft_distance.csv（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure
"""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import dendrogram
import sft_distance_bar
from figs import label
from figs.common import FONT, VARIANT_DISPLAY, config, fig_path, out_root, panel, save, src_markers
from utils.display import ordered_subtypes, shorten, subtype_color_map

NAME = "subtype_structure"
SITE_DISPLAY = {"ebrains": "EBRAINS", "patho2": "patho2"}


def load(cfg: dict, variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(groups, mat): mat は euc_mean の生の距離行列（dendrogram.draw_dendrogram に渡すと元の図と同じ正規化）。"""
    groups, D, _ = dendrogram.load_raw(out_root(cfg) / "dendrogram" / cfg["variants"].get(variant, variant))
    return groups, pd.DataFrame(D, index=groups["label"], columns=groups["label"])


def marker_specs(groups: pd.DataFrame, cfg: dict) -> dict:
    colors = subtype_color_map(list(dict.fromkeys(groups["subtype"])), cfg)
    mkr = src_markers(cfg, sorted(groups["source"].unique()))
    return {r.label: {"color": colors[r.subtype], "marker": mkr[r.source], "subtype": r.subtype,
                      "source": r.source, "short": shorten(r.subtype, cfg)}
            for r in groups.itertuples()}


def draw_trees(fig: plt.Figure, specs: list, variants: list[str], cfg: dict, letters: str) -> list[plt.Axes]:
    """variants の樹形図を specs に縦に積む（縦軸の上限は全段で共通）。"""
    data = {v: load(cfg, v) for v in variants}
    ymax = max(dendrogram.dendrogram_height(mat) for _, mat in data.values()) * 1.05
    axes = []
    for spec, v, letter in zip(specs, variants, letters):
        groups, mat = data[v]
        ax = fig.add_subplot(spec)
        n = dendrogram.draw_dendrogram(ax, mat, False, marker_specs(groups, cfg), ymax=ymax,
                                       marker_size=22, frame_lw=0.9, line_lw=0.8, ytick_size=FONT - 1)
        ax.set_ylabel("Normalized distance\n(Ward)", fontsize=FONT)
        ax.spines["left"].set_visible(True)
        ax.set_title(f"{VARIANT_DISPLAY.get(v, v)}   cross-site pairs: {n}/{groups['subtype'].nunique()}",
                     fontsize=FONT + 1, loc="left", pad=3)
        panel(ax, letter, x=-0.07, y=1.04)
        axes.append(ax)
    return axes


def legend_handles(groups: pd.DataFrame, cfg: dict) -> list:
    """サブタイプ色・施設マーカー・点線の四角の意味を 1 つの凡例にまとめる。"""
    subs = ordered_subtypes(set(groups["subtype"]), cfg)
    colors = subtype_color_map(subs, cfg)
    h = [Patch(color=colors[s], label=shorten(s, cfg)) for s in subs]
    for src, m in src_markers(cfg, sorted(groups["source"].unique())).items():
        h.append(Line2D([0], [0], marker=m, color="#555555", ls="none", markersize=4,
                        label=SITE_DISPLAY.get(src, src)))
    h.append(Rectangle((0, 0), 1, 1, fill=False, linestyle="--", edgecolor="#555555", linewidth=0.9,
                       label="same subtype, both sites adjacent"))
    return h


def sft_table(cfg: dict, variant: str) -> pd.DataFrame:
    """施設内の SFT からの euc_mean を、variant の樹形図と同じ最大値で割った表。"""
    _, mat = load(cfg, variant)
    raw = pd.read_csv(out_root(cfg) / "sft_distance" / "sft_distance_raw.csv")
    raw["distance"] = raw["distance"] / float(np.max(mat.values))
    return raw


def draw(cfg: dict) -> plt.Figure:
    fig = plt.figure(figsize=(7.4, 5.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.34], width_ratios=[3.1, 1],
                  hspace=0.62, wspace=0.42)
    draw_trees(fig, [gs[0, 0], gs[1, 0]], ["original", "centroid"], cfg, "ab")
    ax_c = fig.add_subplot(gs[0:2, 1])
    sft_distance_bar.draw_raw(ax_c, sft_table(cfg, "centroid"), cfg, fontsize=FONT)
    ax_c.get_legend().remove()  # 施設マーカーは共通の凡例
    ax_c.set_xlabel("Normalized distance to SFT\n(euc_mean, within site)", fontsize=FONT)
    ax_c.set_xlim(0, None)
    panel(ax_c, "c", x=-0.55, y=1.01)
    ax_l = fig.add_subplot(gs[2, :]); ax_l.axis("off")
    groups, _ = load(cfg, "centroid")
    ax_l.legend(handles=legend_handles(groups, cfg), loc="center", ncol=9, frameon=False,
                fontsize=FONT - 1, handlelength=1.2, handletextpad=0.4, columnspacing=0.9)
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(draw(cfg), cfg, NAME)
    root = out_root(cfg)
    shutil.copy2(root / "dendrogram" / "pairs.csv", fig_path(cfg, NAME, "_pairs.csv"))
    tab = sft_table(cfg, "centroid")
    tab.to_csv(fig_path(cfg, NAME, "_sft_distance.csv"), index=False)
    order = tab.groupby("subtype")["distance"].mean().sort_values()
    print("  SFT に近い順: " + ", ".join(f"{shorten(k, cfg)} {v:.3f}" for k, v in order.items()))
    print(pd.read_csv(root / "dendrogram" / "pairs.csv").to_string(index=False))


if __name__ == "__main__":
    main()
