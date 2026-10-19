"""施設差と補正: UMAP（補正なし / Macenko / GAN / ComBat / 平行移動）と、補正法の比較（施設差の除去と構造の保存）。

条件は CONDITIONS_FIG3 の 5 条件（本文の 4 条件 ＋ 色正規化 Macenko）。並びは 補正なし → 画像レベル（Macenko → GAN）
→ 埋め込みレベル（ComBat → 平行移動）で、x 軸の下に none / image / embedding の括りを置く（ken 決定 2026-10-20）。

入力:
    output_dir/umap/{variant}/coords.csv          （umap_plot.py）
    output_dir/method_compare/summary.csv         （method_compare.py: 施設指標・隣接ペア・形の指標・施設平均の距離）
    output_dir/harmonization/site_null.csv        （harmonization.py: 施設指標の chance level）
出力: fig/{fig n}_umap_harmonization.{png,pdf}（番号は figs/__init__.py）

パネル:
    a–e  UMAP（パラメータ共通）
    f    施設差をどれだけ落としたか: 施設平均どうしの距離 / ASW 施設（chance の破線）/ iLISI（同）/ 隣接ペア
    g    病変の構造を壊していないか（補正しすぎ）: patho2 内 SFT−髄膜腫 距離（EBRAINS の値の破線）/ ASW 組織型 /
         cLISI 組織型 / 組織型間距離の施設差（%）

Usage:
    uv run python -m figs.umap_harmonization
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figs import label
from figs.common import (CONDITION_COLORS, CONDITION_LABELS, CONDITION_LABELS_SHORT, FONT, config, out_root, panel,
                         save, src_markers)
from utils.display import CONDITIONS_FIG3, ordered_subtypes, shorten, subtype_color_map

NAME = "umap_harmonization"
VARIANTS = CONDITIONS_FIG3
TITLES = {**CONDITION_LABELS, "original": "Uncorrected"}
TIERS = [("none", [0]), ("image", [1, 2]), ("embedding", [3, 4])]   # VARIANTS の添字
NULL_KIND = "within_subtype"  # chance level の基準（サブタイプ内で施設ラベルを並べ替え）


def _draw_umap(ax: plt.Axes, coords: pd.DataFrame, sub_map: dict, src_mkr: dict) -> None:
    for src, mkr in src_mkr.items():
        for sub, color in sub_map.items():
            m = (coords["source"] == src) & (coords["subtype"] == sub)
            if m.any():
                ax.scatter(coords.loc[m, "umap1"], coords.loc[m, "umap2"], c=[color], marker=mkr,
                           s=5, alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=FONT - 1); ax.set_ylabel("UMAP2", fontsize=FONT - 1)
    ax.spines[["top", "right"]].set_visible(False)


def _bar(ax: plt.Axes, s: pd.DataFrame, key: str, title: str, fmt: str, ref: float | None = None,
         ref_label: str = "", zero_line: bool = False, scale: float = 1.0, arrow: str = "",
         ymin: float | None = None, xlabels: bool = True) -> None:
    """条件ごとの棒（値を棒の上に）。ref は基準の破線（chance / EBRAINS の値）、ymin で y 軸の下端を切る。"""
    vals = s.loc[VARIANTS, key].to_numpy(dtype=float) * scale
    x = np.arange(len(VARIANTS))
    base = 0.0 if ymin is None else ymin
    ax.bar(x, vals - base, bottom=base, color=[CONDITION_COLORS[c] for c in VARIANTS], width=0.7, edgecolor="none")
    lo, hi = min(base, vals.min()), max(base, vals.max())
    sub = ""
    if ref is not None:
        ax.axhline(ref, color="#555555", lw=0.8, ls="--")
        lo, hi = min(lo, ref), max(hi, ref)
        sub = f"\n(dashed: {ref_label})"
    if zero_line:
        ax.axhline(0, color="#999999", lw=0.6)
    span = (hi - lo) or 1.0
    for xi, v in zip(x, vals):
        up = v >= base
        ax.text(xi, v + (0.03 * span if up else -0.03 * span), fmt.format(v), ha="center",
                va="bottom" if up else "top", fontsize=FONT - 3)
    ax.set_ylim(lo - 0.12 * span if lo < base else lo, hi + 0.24 * span)
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    ax.set_xticks(x)
    if xlabels:
        ax.set_xticklabels([CONDITION_LABELS_SHORT[c] for c in VARIANTS], fontsize=FONT - 3, rotation=35,
                           ha="right", rotation_mode="anchor")
    else:
        ax.set_xticklabels([])
    ax.set_title(title + (f" {arrow}" if arrow else "") + sub, fontsize=FONT - 1)
    ax.tick_params(axis="y", labelsize=FONT - 3)
    ax.spines[["top", "right"]].set_visible(False)


def _tiers(ax: plt.Axes, y: float = -0.42) -> None:
    """x 軸の下に none / image / embedding の括りを描く。"""
    for name, idx in TIERS:
        x0, x1 = idx[0] - 0.35, idx[-1] + 0.35
        ax.plot([x0, x1], [y, y], color="#333333", lw=0.7, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.05, name, ha="center", va="top", fontsize=FONT - 3,
                transform=ax.get_xaxis_transform(), clip_on=False)


def build(cfg: dict) -> plt.Figure:
    root = out_root(cfg)
    coords = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
              for v in VARIANTS}
    s = pd.read_csv(root / "method_compare" / "summary.csv").set_index("variant")
    null = pd.read_csv(root / "harmonization" / "site_null.csv")
    null = null[null["null"] == NULL_KIND].groupby("metric")["null_mean"].mean()   # 状態間でほぼ同じなので平均
    ref, other = cfg["reference"], next(k for k in cfg["embedding"] if k != cfg["reference"])
    n_sub = int(s["n_subtypes"].iloc[0])
    d_ref = float(s[f"sft_men_dist_{ref}"].iloc[0])
    subs = ordered_subtypes(set(coords["original"]["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))

    fig = plt.figure(figsize=(7.2, 6.7))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.1, 0.16, 0.95, 1.0], hspace=0.62)
    top = gs[0].subgridspec(1, len(VARIANTS), wspace=0.12)
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(top[0, i])
        _draw_umap(ax, coords[v], sub_map, src_mkr)
        ax.set_title(TITLES[v], fontsize=FONT)
        if i:
            ax.set_ylabel("")
        panel(ax, "abcde"[i], x=-0.04)

    # 凡例は UMAP の下に横並び（a–e 共通）
    ax_leg = fig.add_subplot(gs[1]); ax_leg.axis("off")
    handles = [Patch(facecolor=sub_map[t], label=shorten(t, cfg)) for t in subs]
    handles += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=t)
                for t, m in src_mkr.items()]
    ax_leg.legend(handles=handles, loc="center", ncol=10, frameon=False, handlelength=1.0,
                  handleheight=0.9, columnspacing=0.9, handletextpad=0.4, borderaxespad=0, fontsize=FONT - 1)

    # f: 施設差をどれだけ落としたか
    row_f = gs[2].subgridspec(1, 4, wspace=0.55)
    axf = [fig.add_subplot(row_f[0, i]) for i in range(4)]
    _bar(axf[0], s, "site_shift_norm", "Distance between\nsite means", "{:.2f}", arrow="↓", xlabels=False)
    _bar(axf[1], s, "asw_batch", "ASW (site)", "{:.3f}", ref=float(null["asw_batch"]), ref_label="chance", arrow="↓",
         xlabels=False)
    _bar(axf[2], s, "ilisi", "iLISI (site)", "{:.2f}", ref=float(null["ilisi"]), ref_label="chance", arrow="↑",
         xlabels=False)
    _bar(axf[3], s, "adjacent_pairs", f"Adjacent same-histotype\npairs (/{n_sub})", "{:.0f}", arrow="↑", xlabels=False)
    panel(axf[0], "f", x=-0.42, y=1.22)

    # g: 病変の構造を壊していないか（補正しすぎ）
    row_g = gs[3].subgridspec(1, 4, wspace=0.55)
    axg = [fig.add_subplot(row_g[0, i]) for i in range(4)]
    _bar(axg[0], s, f"sft_men_dist_{other}", f"SFT–meningioma dist. ({other})", "{:.2f}",
         ref=d_ref, ref_label=f"{ref.upper()} value", ymin=8.0)
    _bar(axg[1], s, "asw_bio", "ASW (histotype)", "{:+.3f}", zero_line=True, arrow="↑")
    _bar(axg[2], s, "clisi", "cLISI (histotype)", "{:.2f}", arrow="↓")
    _bar(axg[3], s, "subtype_dist_mismatch", "Site mismatch of\nhistotype distances (%)", "{:.1f}", scale=100,
         arrow="↓")
    panel(axg[0], "g", x=-0.42, y=1.22)
    for ax in axg:
        _tiers(ax)
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(build(cfg), cfg, NAME)


if __name__ == "__main__":
    main()
