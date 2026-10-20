"""平行移動のシフト推定に要る例数（Supplementary）: 新施設（patho2）から n 例だけ抜いて Δ̂ = mean(EBRAINS) − mean(S) を
推定し、patho2 の全スライドに適用したときの精度を n に対して描く（shift_estimation.py の B）。

    a: cos(Δ̂, Δ*)（Δ* = 組織型を合わせたシフト）
    b: 平行移動後の ASW（施設）。chance level（施設ラベルを組織型内で入れ替えた帰無）を破線、補正なしを点線
    c: 組織型（主要 5 クラス）の最近傍重心法 balanced accuracy（EBRAINS → patho2）。補正なしを点線

線: 無作為（中央値、帯は 2.5–97.5 パーセンタイル）と、単一組織型だけから抜いた場合（patho2 で多い 3 型、中央値のみ、
n はその型の例数まで）。層化と生物学的方向を落とした推定量（*_ortho）は描かない（ken 決定 2026-10-20）。

入力: output_dir/shift_estimation/subsample.csv（shift_estimation.py）、
      output_dir/harmonization/site_null.csv（ASW の chance level、harmonization.py）
出力: fig/{fig S n}_shift_sample_size.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.shift_sample_size
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from figs import label
from figs.common import FONT, SITE_DISPLAY, config, out_root, panel, save
from utils.display import subtype_color_map

NAME = "shift_sample_size"
RANDOM_COLOR = "#AA3377"   # 平行移動の条件色（utils/display.py の centroid）
PANELS = [("cos_vs_star", "cos(estimated shift, Δ*)", None),
          ("asw_site", "ASW (site) after translation", "asw"),
          ("histotype_ba", "Histotype balanced accuracy", "ba")]


def _line(ax, g: pd.DataFrame, key: str, color: str, lw: float, band: bool) -> None:
    g = g.sort_values("n")
    ax.plot(g["n"], g[f"{key}_median"], color=color, lw=lw, marker="o", ms=2.5 if band else 2, zorder=3)
    if band:
        ax.fill_between(g["n"], g[f"{key}_q025"], g[f"{key}_q975"], color=color, alpha=0.18, lw=0, zorder=1)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    root = out_root(cfg)
    df = pd.read_csv(root / "shift_estimation" / "subsample.csv")
    none = df[df["scheme"] == "none"].iloc[0]
    rnd = df[df["scheme"] == "random"]
    single = df[df["scheme"] == "single"]
    subs = [s for s in cfg["display"]["colors"]["subtypes"] if s in set(single["subtype"])]   # config の表示順
    sub_color = subtype_color_map(subs, cfg)
    null = pd.read_csv(root / "harmonization" / "site_null.csv")
    chance = float(null[(null["null"] == "within_subtype") & (null["state"] == "original")
                        & (null["metric"] == "asw_batch")]["null_mean"].iloc[0])
    other = next(k for k in cfg["embedding"] if k != cfg["reference"])
    n_all = int(rnd["n"].max())

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for ax, (key, ylabel, kind) in zip(axes, PANELS):
        _line(ax, rnd, key, RANDOM_COLOR, 1.4, band=True)
        for s in subs:
            _line(ax, single[single["subtype"] == s], key, sub_color[s], 0.9, band=False)
        if kind == "asw":
            ax.axhline(chance, color="#555555", ls="--", lw=0.8, zorder=2)
            ax.axhline(none[f"{key}_median"], color="#555555", ls=":", lw=0.8, zorder=2)
            ax.text(3, chance, "chance", ha="left", va="bottom", fontsize=FONT - 2, color="#555555")
            ax.text(n_all, none[f"{key}_median"], "no correction ", ha="right", va="bottom", fontsize=FONT - 2,
                    color="#555555")
        elif kind == "ba":
            ax.axhline(none[f"{key}_median"], color="#555555", ls=":", lw=0.8, zorder=2)
            ax.text(n_all, none[f"{key}_median"], "no correction ", ha="right", va="top", fontsize=FONT - 2,
                    color="#555555")
        ax.set_xscale("log")
        ax.set_xticks([3, 10, 30, 100, 300], ["3", "10", "30", "100", "300"])
        ax.minorticks_off()
        ax.set_xlabel(f"Slides sampled from {SITE_DISPLAY.get(other, other)} (n)", fontsize=FONT)
        ax.set_ylabel(ylabel, fontsize=FONT)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=FONT - 1)
    axes[0].set_ylim(top=1.0)
    axes[1].set_ylim(bottom=0)
    for ax, letter in zip(axes, "abc"):
        panel(ax, letter, x=-0.22, y=1.0)

    handles = [Line2D([0], [0], color=RANDOM_COLOR, lw=1.4, marker="o", ms=2.5, label="Random (median, 2.5–97.5%)")]
    handles += [Line2D([0], [0], color=sub_color[s], lw=0.9, marker="o", ms=2, label=f"{s.replace(' meningioma', '')} only")
                for s in subs]
    # 凡例は 3 パネルの下に横並び。Δ̂・Δ* の定義は図説（figures.md）に書く
    fig.tight_layout(w_pad=1.2, rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=FONT - 1,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4)
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
