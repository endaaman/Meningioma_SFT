"""v4 Fig 3 腫瘍をまたいでも同じ 1 本: 12 群の施設のずれは、腫瘍によらずほぼ同じ向きと長さの矢印になる（TITAN、補正なし）。

a 矢印図。横軸 = 髄膜腫だけのずれの向き、縦軸 = それに直交する群間の変動の第 1 主成分（24 重心から求めた平面）。
  各群の院内の平均（白抜き）から EBRAINS の平均（塗り）への矢印、色 = 腫瘍。破線 = 髄膜腫だけのずれ。
  右の小さい図は同じ矢印を原点から並べたもの（平行で同じ長さかを見る）
b 群ごとの cos（髄膜腫だけのずれとの。髄膜腫のサブタイプはそのサブタイプを除いた髄膜腫と比べる = leave-group-out）。
  点 = 生の値（塗り）と推定の雑音を差し引いた値（白抜き）、線 = 95% ブートストラップ区間（スライド単位）
腫瘍を食い違わせた対照（旧仕様の c）は Supplementary（figs.shift_mismatch）。

入力: output_dir/shift_transfer/reference_meningioma/arrows.csv（shift_transfer.py --reference meningioma）、
      output_dir/shift_transfer/groups.csv（cos_vs_meni / cos_lo_vs_meni / cos_hi_vs_meni / cos_corrected_vs_meni）
      （区間 lo / hi は生の cos のブートストラップの区間。雑音を差し引いた値 cos_corrected は区間を持たない）
出力: fig/{fig n}_tumour_shift_arrows.{png,pdf}

Usage:
    uv run python -m figs.tumour_shift_arrows
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figs import label
from figs.common import FONT, config, out_root, panel, save
from shift_transfer import FAMILY_COLOR, FAMILY_ORDER, _spread

NAME = "tumour_shift_arrows"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    st = out_root(cfg) / "shift_transfer"
    arr = pd.read_csv(st / "reference_meningioma" / "arrows.csv")
    groups = pd.read_csv(st / "groups.csv")
    grp = arr[arr["role"] == "group"]

    fig = plt.figure(figsize=(7.2, 4.4))
    ax = fig.add_axes([0.07, 0.12, 0.46, 0.80])
    axb = fig.add_axes([0.72, 0.12, 0.26, 0.80])

    # a（右下の小さい図 ax2 は、ラベルの無い左の中段に置く）
    ax2 = ax.inset_axes([0.03, 0.33, 0.30, 0.20])
    xs = np.r_[arr["x_patho2"].values, arr["x_ebrains"].values]
    ys = np.r_[arr["y_patho2"].values, arr["y_ebrains"].values]
    label_x = float(xs.min()) - 0.9
    label_y = _spread(arr["y_patho2"].values, gap=0.62)
    for r, ly in zip(arr.itertuples(index=False), label_y):
        ref = r.role == "reference"
        col = "0.45" if ref else FAMILY_COLOR.get(r.family, "gray")
        style = dict(arrowstyle="-|>", color=col, lw=1.4 if ref else 1.0, ls="--" if ref else "-", shrinkA=0, shrinkB=0, mutation_scale=8)
        ax.annotate("", xy=(r.x_ebrains, r.y_ebrains), xytext=(r.x_patho2, r.y_patho2), arrowprops=style)
        ax.scatter([r.x_patho2], [r.y_patho2], s=12, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3)
        ax.scatter([r.x_ebrains], [r.y_ebrains], s=12, color=col, zorder=3)
        ax.plot([label_x + 0.1, r.x_patho2], [ly, r.y_patho2], color=col, lw=0.4, alpha=0.6)
        name = "Meningioma (all)" if ref else r.group
        ax.text(label_x, ly, name, ha="right", va="center", fontsize=FONT - 2, color=col, fontstyle="italic" if ref else "normal")
        if not ref:
            ax2.annotate("", xy=(r.dx, r.dy), xytext=(0, 0), arrowprops={**style, "lw": 0.8, "mutation_scale": 6})
    ax.set_xlim(label_x - 7.0, float(xs.max()) + 0.6)
    ax.set_ylim(min(float(ys.min()), float(label_y.min())) - 0.8, max(float(ys.max()), float(label_y.max())) + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Meningioma site-shift direction (L2 units)")
    ax.set_ylabel("Between-group PC1 ⊥ shift")
    ax.spines[["top", "right"]].set_visible(False)
    a2 = grp["angle_2d_deg"]
    l2 = grp["length_2d"]
    ax.set_title("Open: in-house, filled: EBRAINS; dashed: meningioma (all)", fontsize=FONT - 1, loc="left")
    panel(ax, "a", x=-0.02, y=1.02)
    ax2.set_xlim(-0.3, float(grp["dx"].max()) + 0.5)
    ax2.set_ylim(float(grp["dy"].min()) - 0.8, float(grp["dy"].max()) + 0.8)
    ax2.set_aspect("equal")
    ax2.axhline(0, color="0.8", lw=0.6)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Common origin", fontsize=FONT - 2, pad=2)
    ax2.text(0.0, -0.08, f"in-plane angle {a2.mean():.1f} ± {a2.std(ddof=1):.1f}°, length {l2.mean():.1f} ± {l2.std(ddof=1):.1f}",
             transform=ax2.transAxes, fontsize=FONT - 2, va="top")

    # b
    g = groups[groups["kind"].isin(["sft", "subtype", "glioma"])].copy()
    g["o"] = g["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    g = g.sort_values(["o"], kind="stable").reset_index(drop=True)
    y = np.arange(len(g))[::-1]
    for yi, r in zip(y, g.itertuples()):
        col = FAMILY_COLOR.get(r.family, "gray")
        axb.plot([r.cos_lo_vs_meni, r.cos_hi_vs_meni], [yi, yi], color=col, lw=1.2)
        axb.scatter([r.cos_vs_meni], [yi], s=18, color=col, zorder=3)
        axb.scatter([r.cos_corrected_vs_meni], [yi], s=18, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
    axb.set_yticks(y)
    axb.set_yticklabels(g["group"], fontsize=FONT - 2)
    axb.set_xlim(0.5, 1.0)
    axb.axvline(1, color="0.8", lw=0.6)
    axb.set_xlabel("cos with the meningioma site shift")
    axb.spines[["top", "right"]].set_visible(False)
    # 区間は生の cos のブートストラップ（施設内のスライドを再抽出）の 95% 区間。雑音を差し引いた値は別の推定量（推定の雑音で
    # 縮んだ長さを補正した cos）で区間を持たないので、区間の外に出ることがある（Microcystic・Angiomatous）。凡例でそう書く
    axb.legend(handles=[Line2D([], [], marker="o", ls="-", lw=1.2, color="0.3", label="raw cos\n(line: 95% bootstrap CI)"),
                        Line2D([], [], marker="o", ls="", markerfacecolor="white", color="0.3",
                               label="noise-corrected cos\n(point estimate, no CI)")],
               fontsize=FONT - 2.5, loc="upper left", frameon=False, labelspacing=0.8)
    panel(axb, "b", x=-0.62, y=1.02)
    save(fig, cfg, NAME)
    print(f"  12 groups: in-plane angle {a2.mean():.2f} ± {a2.std(ddof=1):.2f}, length {l2.mean():.2f} ± {l2.std(ddof=1):.2f}; "
          f"full-space angle (noise-corr.) {grp['angle_corrected_vs_paper_deg'].mean():.1f} ± {grp['angle_corrected_vs_paper_deg'].std(ddof=1):.1f}; "
          f"cos raw {g['cos_vs_meni'].min():.3f}–{g['cos_vs_meni'].max():.3f}")


if __name__ == "__main__":
    main()
