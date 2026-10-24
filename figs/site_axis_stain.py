"""施設軸と染色の関係（Supplementary）: 補正なしの TITAN 埋め込みで施設間シフトの単位ベクトル u を取り、各スライドの
施設内で中心化した射影 s = ⟨x − μ_site, u⟩ をスライドの平均色（H / E 強度）と突き合わせる（stain_response.py）。

    a: s vs H（ヘマトキシリン）。点はスライド（施設色）、線は施設内の十分位の bin 平均 ± SE。組織型内の Pearson r を注記
    b: s vs E（エオジン）。同じ
    c: 画像レベルの補正（GAN / Macenko）後に残る施設差 μ_E − μ_P を u に沿う成分と直交成分に分けたもの（TITAN）
    d: u に沿った施設差のうちスライド平均色（H, E, L*, a*, b*）を共変量にして消える割合（TITAN / 平均プーリング）

入力: output_dir/stain_response/{slide_projection,binned,correlations,shift_decomposition,regression}.csv（stain_response.py）
出力: fig/{fig S n}_site_axis_stain.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.site_axis_stain
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figs import label
from figs.common import CONDITION_COLORS, CONDITION_LABELS_SHORT, FONT, SITE_DISPLAY, config, out_root, panel, save

NAME = "site_axis_stain"
AGG = "titan"                                   # a–c は TITAN だけ。d だけ平均プーリングと並べる
SITES = ["ebrains", "patho2"]
FEATURES = [("H", "Hematoxylin intensity, H (slide mean)"), ("E", "Eosin intensity, E (slide mean)")]
X_MAX = {"H": 0.10, "E": None}                 # H は patho2 の外れ値 2 例（H > 0.10）で軸が伸びるので切る（図説に書く）
CONDITIONS = ["original", "gan", "macenko"]    # c: 補正なし（= u の定義）と画像レベルの補正 2 つ
AGG_DISPLAY = {"titan": "TITAN", "mean": "Mean\npooling"}
GREY = "#888888"


def _scatter(ax: plt.Axes, proj: pd.DataFrame, bins: pd.DataFrame, corr: pd.DataFrame, feat: str, xlabel: str,
             colors: dict[str, str]) -> None:
    for site in SITES:
        g = proj[proj["site"] == site]
        ax.scatter(g[feat], g["s"], s=4, color=colors[site], alpha=0.3, linewidths=0, zorder=1)
    for site in SITES:
        b = bins[(bins["site"] == site) & (bins["feature"] == feat)].sort_values("bin")
        ax.errorbar(b["feature_mean"], b["s_mean"], yerr=b["s_se"], color=colors[site], lw=1.3, marker="o", ms=3,
                    markeredgecolor="white", markeredgewidth=0.5, capsize=0, zorder=4,
                    path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    for k, site in enumerate(SITES):
        r = corr[(corr["site"] == site) & (corr["feature"] == feat)]["pearson_r"].iloc[0]
        ax.text(0.03, 0.97 - 0.09 * k, f"{SITE_DISPLAY[site]} r = {r:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=FONT - 1, color=colors[site], zorder=5)
    ax.axhline(0, color="#CCCCCC", lw=0.6, zorder=0)
    ax.set_xlim(0, X_MAX[feat])
    ax.set_xlabel(xlabel, fontsize=FONT)
    ax.set_ylabel("Projection on site-shift axis, s\n(within-site centred)", fontsize=FONT)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT - 1)


def _decomposition(ax: plt.Axes, decomp: pd.DataFrame) -> None:
    w = 0.38
    for i, cond in enumerate(CONDITIONS):
        d = decomp[decomp["condition"] == cond].iloc[0]
        col = CONDITION_COLORS[cond]
        ax.bar(i - w / 2, d["along_u"], width=w, color=col, edgecolor="none", zorder=2)
        ax.bar(i + w / 2, d["orthogonal"], width=w, facecolor="white", edgecolor=col, hatch="////", lw=0.8, zorder=2)
        for x, v in ((i - w / 2, d["along_u"]), (i + w / 2, d["orthogonal"])):
            ax.text(x, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=FONT - 2)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([CONDITION_LABELS_SHORT[c] for c in CONDITIONS], fontsize=FONT - 2)
    ax.set_ylabel("Remaining site shift\n‖μ_E − μ_P‖ (TITAN)", fontsize=FONT)
    ax.set_ylim(0, float(decomp[["along_u", "orthogonal"]].max().max()) * 1.42)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT - 1)
    handles = [Patch(facecolor=GREY, edgecolor="none", label="Along u"),
               Patch(facecolor="white", edgecolor=GREY, hatch="////", label="Orthogonal to u")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=FONT - 2, handlelength=1.4, borderaxespad=0.2)


def _explained(ax: plt.Axes, reg: pd.DataFrame) -> None:
    aggs = [a for a in AGG_DISPLAY if a in set(reg["aggregator"])]
    vals = [float(reg[reg["aggregator"] == a]["site_gap_explained_by_colour"].iloc[0]) * 100 for a in aggs]
    ax.bar(range(len(aggs)), vals, width=0.6, color=GREY, edgecolor="none", zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=FONT - 1)
    ax.set_xticks(range(len(aggs)))
    ax.set_xticklabels([AGG_DISPLAY[a] for a in aggs], fontsize=FONT - 2)
    ax.set_xlim(-0.65, len(aggs) - 0.35)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_ylabel("Site shift along u explained\nby slide mean colour (%)", fontsize=FONT)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FONT - 1)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    root = out_root(cfg) / "stain_response"
    colors = cfg["display"]["colors"]["sources"]
    proj = pd.read_csv(root / "slide_projection.csv")
    proj = proj[(proj["aggregator"] == AGG) & (proj["condition"] == "original")].dropna(subset=["H", "E"])
    bins = pd.read_csv(root / "binned.csv")
    bins = bins[bins["aggregator"] == AGG]
    corr = pd.read_csv(root / "correlations.csv")
    corr = corr[(corr["aggregator"] == AGG) & (corr["adjust"] == "within_subtype")]
    decomp = pd.read_csv(root / "shift_decomposition.csv")
    decomp = decomp[decomp["aggregator"] == AGG]
    reg = pd.read_csv(root / "regression.csv")
    reg = reg[reg["colour_set"] == "H,E,L,a,b"]

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1.25, 1.25, 1.15, 0.7], wspace=0.65,
                  left=0.075, right=0.99, top=0.9, bottom=0.3)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (feat, xlabel) in zip(axes[:2], FEATURES):
        _scatter(ax, proj, bins, corr, feat, xlabel, colors)
    axes[1].set_ylabel("")
    _decomposition(axes[2], decomp)
    _explained(axes[3], reg)
    for ax, letter in zip(axes, "abcd"):
        panel(ax, letter, x=-0.3 if letter in "cd" else -0.22, y=1.0)

    n = {site: int((proj["site"] == site).sum()) for site in SITES}
    handles = [Line2D([0], [0], marker="o", ms=4, lw=0, color=colors[s], label=f"{SITE_DISPLAY[s]} slides (n = {n[s]})")
               for s in SITES]
    handles.append(Line2D([0], [0], marker="o", ms=3, lw=1.3, color="#555555", label="Decile means ± SE (within site)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FONT - 1,
               bbox_to_anchor=(0.36, 0.0), handlelength=1.8, columnspacing=1.6)
    save(fig, cfg, NAME)

    for feat, _ in FEATURES:
        print(f"  {feat}: within-histotype r " + ", ".join(
            f"{SITE_DISPLAY[s]} {corr[(corr['site'] == s) & (corr['feature'] == feat)]['pearson_r'].iloc[0]:+.2f}" for s in SITES))
    for _, d in decomp.iterrows():
        print(f"  {d['condition']}: ‖μ_E − μ_P‖ = {d['shift_norm']:.2f} = along u {d['along_u']:.2f} + orthogonal {d['orthogonal']:.2f}")
    for _, r in reg.iterrows():
        print(f"  {r['aggregator']}: site gap explained by slide colour {r['site_gap_explained_by_colour']:.1%} "
              f"({r['site_coef_unadjusted']:.2f} → {r['site_coef_adjusted']:.2f})")


if __name__ == "__main__":
    main()
