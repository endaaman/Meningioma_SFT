"""施設差と補正: UMAP（補正なし / GAN / ComBat / centroid）と、施設差・診断クラス分離・組織型分離の指標。

入力:
    output_dir/umap/{original,gan,combat,centroid}/coords.csv  （umap_plot.py）
    output_dir/harmonization/metrics.csv               （harmonization.py）
    output_dir/harmonization/site_null.csv / residual_fraction.csv（同、施設ラベルの並べ替え検定）
出力: fig/{fig n}_umap_harmonization.{png,pdf}（番号は figs/__init__.py）

d パネルの見せ方は D_STYLE の 1 か所で切り替える:
    "plain"    … 指標の棒だけ
    "band"     … 施設 2 指標に偶然のレベルの 95% 範囲（灰色の横帯）
    "line"     … 施設 2 指標に偶然のレベルの平均（破線）
    "residual" … 施設 2 指標を「残った施設差の割合」（補正なし 100% / 偶然 0%）の棒で
偶然のレベルは NULL_KIND（施設ラベルの並べ替えの種類）の帰無分布。

Usage:
    uv run python -m figs.umap_harmonization
    uv run python -m figs.umap_harmonization --preview   # d の 3 案を fig/_preview_fig3d_*.png に出す（figs.all では作らない）
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import harmonization
from figs import label
from figs.common import (CONDITION_COLORS, CONDITION_LABELS, CONDITIONS_FULL, FONT, config, fig_dir, out_root, panel,
                         save, src_markers)
from utils.display import ordered_subtypes, shorten, subtype_color_map

NAME = "umap_harmonization"
VARIANTS = CONDITIONS_FULL  # a–d すべて ComBat を含む 4 条件
TITLES = {**CONDITION_LABELS, "original": "Uncorrected"}
METRIC_KEYS = ["asw_batch", "ilisi", "asw_class", "asw_bio", "clisi"]
# e の見出し（指標名と、良い向き）
METRIC_TITLES = {
    "asw_batch": "ASW (site)\n↓ 0: sites mixed",
    "ilisi":     "iLISI (site)\n↑ sites mixed",
    "asw_class": "ASW (SFT vs Men.)\n↑ separated",
    "asw_bio":   "ASW (subtype)\n↑ separated",
    "clisi":     "cLISI (subtype)\n↓ separated",
}
SITE_KEYS = ("asw_batch", "ilisi")
D_STYLE = "line"             # 本番の d の見せ方（"plain" / "band" / "line" / "residual"）
NULL_KIND = "within_subtype"  # 偶然のレベルの基準（サブタイプ内で施設ラベルを並べ替え）
PREVIEW_STYLES = ("band", "line", "residual")


def _draw_umap(ax: plt.Axes, coords: pd.DataFrame, sub_map: dict, src_mkr: dict) -> None:
    for src, mkr in src_mkr.items():
        for sub, color in sub_map.items():
            m = (coords["source"] == src) & (coords["subtype"] == sub)
            if m.any():
                ax.scatter(coords.loc[m, "umap1"], coords.loc[m, "umap2"], c=[color], marker=mkr,
                           s=6, alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.spines[["top", "right"]].set_visible(False)


def _null_for(null_table: pd.DataFrame | None, key: str) -> dict | None:
    """施設指標の偶然のレベル（状態間でほぼ同じなので平均を取って 1 本にする）。"""
    if null_table is None or key not in SITE_KEYS:
        return None
    t = null_table[(null_table["null"] == NULL_KIND) & (null_table["metric"] == key)
                   & null_table["state"].isin(VARIANTS)]
    return {"mean": t["null_mean"].mean(), "q025": t["null_q025"].mean(), "q975": t["null_q975"].mean()}


def _draw_residual(ax: plt.Axes, resid: pd.DataFrame, key: str) -> None:
    """補正ありの状態の「残った施設差の割合」（%）の棒。"""
    states = [s for s in VARIANTS if s != "original"]
    t = resid[(resid["null"] == NULL_KIND) & (resid["metric"] == key)].set_index("state")
    vals = np.array([t.loc[s, "residual"] * 100 for s in states])
    x = np.arange(len(states))
    ax.bar(x, vals, color=[CONDITION_COLORS[s] for s in states], width=0.55)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([harmonization.STATE_LABELS[s] for s in states], fontsize=FONT - 1)
    ax.set_ylim(0, 105)
    ax.axhline(100, color="#999999", lw=0.8, ls=":")
    ax.text(x[-1] + 0.4, 100, "uncorrected", fontsize=6.5, color="#777777", ha="right", va="bottom")
    name = harmonization.METRIC_INFO[key][0]
    ax.set_title(f"{name}\nresidual (chance = 0%)", fontsize=9)
    ax.set_ylabel("% of uncorrected", fontsize=FONT - 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)


def build(cfg: dict, style: str) -> plt.Figure:
    root = out_root(cfg)
    coords = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
              for v in VARIANTS}
    metrics = pd.read_csv(root / "harmonization" / "metrics.csv")
    null_path = root / "harmonization" / "site_null.csv"
    null_table = pd.read_csv(null_path) if style != "plain" and null_path.exists() else None
    resid = pd.read_csv(root / "harmonization" / "residual_fraction.csv") if style == "residual" else None
    subs = ordered_subtypes(set(coords["original"]["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.15, 0.2, 1], hspace=0.3)
    top = gs[0].subgridspec(1, len(VARIANTS), wspace=0.12)
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(top[0, i])
        _draw_umap(ax, coords[v], sub_map, src_mkr)
        ax.set_title(TITLES[v])
        if i:
            ax.set_ylabel("")
        panel(ax, "abcd"[i], x=-0.04)

    # 凡例は UMAP の下に横並び（a–d 共通）
    ax_leg = fig.add_subplot(gs[1]); ax_leg.axis("off")
    handles = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in subs]
    handles += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=s)
                for s, m in src_mkr.items()]
    if style in ("line", "band"):
        handles.append(Line2D([0], [0], color="#555555", lw=0.9, ls="--", label="chance (e)"))
    ax_leg.legend(handles=handles, loc="center", ncol=10, frameon=False, handlelength=1.0,
                  handleheight=0.9, columnspacing=0.9, handletextpad=0.4, borderaxespad=0,
                  fontsize=FONT - 1)

    bottom = gs[2].subgridspec(1, len(METRIC_KEYS), wspace=0.6)
    for i, key in enumerate(METRIC_KEYS):
        ax = fig.add_subplot(bottom[0, i])
        if style == "residual" and key in SITE_KEYS:
            _draw_residual(ax, resid, key)
        else:
            harmonization.draw_metric(ax, metrics, key, null=_null_for(null_table, key),
                                      null_style="band" if style == "band" else "line", states=VARIANTS,
                                      value_fontsize=6, tick_fontsize=6, title_fontsize=7,
                                      title=METRIC_TITLES[key], null_label="none", value_rotation=90)
        ax.tick_params(axis="x", labelsize=6, rotation=40)
        ax.tick_params(axis="y", labelsize=6)
        for t in ax.get_xticklabels():
            t.set_ha("right")
        if i == 0:
            panel(ax, "e", x=-0.3, y=1.25)
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    if "--preview" in sys.argv[1:]:
        out = fig_dir(cfg)
        out.mkdir(parents=True, exist_ok=True)
        for style in PREVIEW_STYLES:
            fig = build(cfg, style)
            path = out / f"_preview_fig3d_{style}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved: {path}")
        return
    save(build(cfg, D_STYLE), cfg, NAME)


if __name__ == "__main__":
    main()
