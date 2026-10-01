"""サブタイプ構造: 施設 × サブタイプ平均ベクトルの樹形図（a 補正前 / b centroid 補正後 / c GAN 補正後、縦に積む・同じ縦軸）。
GAN 段の有無は INCLUDE_GAN で切り替える（本文は 3 段、ken 決定 2026-10-01）。

樹形図は既存の `out/dendrogram/{variant}/euc_mean_dendrogram_cross.png` と同じもの:
距離 = 平均ベクトル間のユークリッド距離（euc_mean）を行列ごとに min-max 正規化（対角 0 なので最大値で割るのと同じ）、
Ward 法。描画は dendrogram.draw_dendrogram をそのまま使う（葉 = サブタイプ色 × 施設マーカー、葉の並びで
同じサブタイプの 2 施設が隣り合ったら点線の四角）。各段の「cross-site pairs」はこの四角の数。
SFT からの距離は Supplementary（figs/sft_distance.py）へ移した。

各段は元の単体 png（figsize 24x6・既定の余白・draw_dendrogram の既定のマーカー/四角/線）をそのまま再現して
縦に積む（ken: 元と同じバランスで、スタイルは作り込まない）。紙面の幅に縮めても比率は元と同じ。
隣り合う四角が重ならないことを check_frames で確かめる。

入力:
    output_dir/dendrogram/{variant}/euc_mean_raw_groups.csv / euc_mean_raw.npz  （dendrogram.py。D = euc_mean の生値）
    output_dir/dendrogram/pairs.csv                                            （dendrogram.py）
出力: fig/{fig n}_subtype_structure.{png,pdf}, 同 _pairs.csv（番号は figs/__init__.py）

Usage:
    uv run python -m figs.subtype_structure
"""
from __future__ import annotations

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import dendrogram
from figs import label, stem
from figs.common import VARIANT_DISPLAY, config, fig_dir, fig_path, out_root, src_markers
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


INCLUDE_GAN = True                        # GAN を 3 段目に入れる（ken 決定 2026-10-01: 本文は 3 段）

# 各段は元の単体 png（dendrogram.save_dendrogram: figsize 24x6、既定の subplot 余白、draw_dendrogram の既定サイズ）
# をそのまま再現して縦に積む。スタイルは作り込まない（ken: 元と同じバランスで）。
ROW_W, ROW_H = 24.0, 6.0                  # in（元の figsize）
AX_L, AX_R, AX_B, AX_T = 0.125, 0.9, 0.11, 0.88   # 元の subplot 余白（rcParams 既定）
LEGEND_H = 1.4                            # in
DPI = 150                                 # 元の png と同じ
# ken: 元のバランスのまま、葉のマーカーだけ一回り大きく（直径 1.3 倍 = 面積 1.69 倍）。
# 四角はマーカーに合わせて必要最小限だけ広げる（葉区画の端からの余白 2.0 → 1.6、隣の四角との隙間は 3.2 x 単位）。
MARKER_SIZE = 250 * 1.3 ** 2
FRAME_PAD = 1.6


def check_frames(ax: plt.Axes) -> None:
    """点線の四角同士が重ならない（x 範囲が接しも重なりもしない）ことを確かめる。"""
    xs = sorted((p.get_x(), p.get_x() + p.get_width()) for p in ax.patches if isinstance(p, Rectangle))
    for (_, r0), (l1, _) in zip(xs, xs[1:]):
        assert l1 > r0, f"pair frames overlap: {r0} >= {l1}"


def tree_figure(variants: list[str], cfg: dict) -> plt.Figure:
    """variants の樹形図（各段 = 元の単体 png と同じ配置・大きさ）を縦に積み、凡例を下に 1 つ付けた図。
    縦軸の上限は全段で共通。"""
    n = len(variants)
    fig_h = n * ROW_H + LEGEND_H
    fig = plt.figure(figsize=(ROW_W, fig_h))
    data = {v: load(cfg, v) for v in variants}
    ymax = max(dendrogram.dendrogram_height(mat) for _, mat in data.values()) * 1.05
    for i, (v, letter) in enumerate(zip(variants, "abcdefg")):
        groups, mat = data[v]
        row_bottom = fig_h - (i + 1) * ROW_H
        ax = fig.add_axes([AX_L, (row_bottom + AX_B * ROW_H) / fig_h,
                           AX_R - AX_L, (AX_T - AX_B) * ROW_H / fig_h])
        npairs = dendrogram.draw_dendrogram(ax, mat, False, marker_specs(groups, cfg), ymax=ymax,
                                            marker_size=MARKER_SIZE, frame_pad=FRAME_PAD)
        check_frames(ax)
        ax.set_title(f"{VARIANT_DISPLAY.get(v, v)}   cross-site pairs: {npairs}/{groups['subtype'].nunique()}",
                     fontsize=22, loc="left", pad=8)
        ax.text(-0.06, 1.08, letter, transform=ax.transAxes, fontsize=30, fontweight="bold", va="bottom")
    ax_l = fig.add_axes([AX_L, 0.0, AX_R - AX_L, LEGEND_H / fig_h])
    ax_l.axis("off")
    groups, _ = load(cfg, variants[-1])
    ax_l.legend(handles=legend_handles(groups, cfg), loc="center", ncol=9, frameon=False,
                fontsize=18, handlelength=1.6, handletextpad=0.5, columnspacing=1.4, markerscale=1.0)
    return fig


def save_tree_figure(fig: plt.Figure, base) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(f"{base}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {base}.png / .pdf")


def legend_handles(groups: pd.DataFrame, cfg: dict) -> list:
    """サブタイプ色・施設マーカー・点線の四角の意味を 1 つの凡例にまとめる。"""
    subs = ordered_subtypes(set(groups["subtype"]), cfg)
    colors = subtype_color_map(subs, cfg)
    h = [Patch(color=colors[s], label=shorten(s, cfg)) for s in subs]
    for src, m in src_markers(cfg, sorted(groups["source"].unique())).items():
        h.append(Line2D([0], [0], marker=m, color="#555555", ls="none", markersize=18,
                        label=SITE_DISPLAY.get(src, src)))
    h.append(Rectangle((0, 0), 1, 1, fill=False, linestyle="--", edgecolor="#555555", linewidth=2.0,
                       label="same subtype, both sites adjacent"))
    return h


def sft_table(cfg: dict, variant: str) -> pd.DataFrame:
    """施設内の SFT からの euc_mean を、variant の樹形図と同じ最大値で割った表（figs/sft_distance.py が使う）。"""
    _, mat = load(cfg, variant)
    raw = pd.read_csv(out_root(cfg) / "sft_distance" / "sft_distance_raw.csv")
    raw["distance"] = raw["distance"] / float(np.max(mat.values))
    return raw


def variants_for(include_gan: bool) -> list[str]:
    return ["original", "centroid"] + (["gan"] if include_gan else [])


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    fig_dir(cfg).mkdir(parents=True, exist_ok=True)
    save_tree_figure(tree_figure(variants_for(INCLUDE_GAN), cfg), fig_dir(cfg) / stem(NAME))
    root = out_root(cfg)
    shutil.copy2(root / "dendrogram" / "pairs.csv", fig_path(cfg, NAME, "_pairs.csv"))
    print(pd.read_csv(root / "dendrogram" / "pairs.csv").to_string(index=False))


if __name__ == "__main__":
    main()
