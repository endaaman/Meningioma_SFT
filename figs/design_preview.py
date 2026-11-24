"""v4 Fig 1（研究の設計）の下書き。プレビュー（登録しない。手で描き直すか本番に上げるかは ken）。

a 2 施設 × 5 腫瘍の格子。行 = EBRAINS（基準、●）/ In-house（■）、列 = Meningioma、SFT | Glioma、PCNSL、Metastasis
  （髄膜腫の隣に SFT、グリオーマの隣にリンパ腫と転移）。よくある腫瘍（実線の枠）と鑑別の腫瘍（破線の枠）を枠の線で分け、
  列の上の矢印で「よくある腫瘍で測ったずれを隣の鑑別の腫瘍に借りる」向きを示す。マスの数字 = 枚数（患者数）で、
  output_dir/cohort_table/table1_counts.csv から読む（手で打たない）
b 埋め込みの流れ（箱と矢印）: WSI → パッチ（512 px、約 0.45 µm/px）→ パッチエンコーダ（CONCH v1.5 / Virchow）→ スライド埋め込み
  （TITAN、PRISM = 学習した集約器 / CONCH mean、Virchow mean = パッチの単純平均）
c 「借りる」の模式図（2 次元の概念図）。**点群は説明用の合成データで、実データではない**（乱数 seed 42 の正規分布。図の説明で断る）。
  1 コマ目 = よくある腫瘍（推定元）の 2 施設の点群と平均、その差 Δ。2〜4 コマ目 = 当て先の腫瘍で比べる 3 つの条件
  （No correction / Δ borrowed from another tumour / Δ from the tumour itself）。当て先の院内の点群に足すずれだけが違う
画像の段階の補正（GAN、Macenko）は描かない。色は腫瘍 = shift_transfer.FAMILY_COLOR、施設 = config の display.colors.sources
（Fig 2 の UMAP と同じ）、記号 = ● EBRAINS / ■ In-house。

出力: paper.dir/v4/fig/_preview_design.{png,pdf}（paper.version によらず v4）

Usage:
    uv run python -m figs.design_preview
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figs.cohort_tumours import cell, counts
from figs.common import FONT, config, paper_root
from shift_transfer import FAMILY_COLOR
from utils.display import FAMILY_LABELS

NAME = "_preview_design"
PAPER_VERSION = "v4"
COLS = [("meningioma", "Meningioma", "12 subtypes at both sites", "common"), ("SFT", "SFT", "", "rare"),
        ("glioma", "Glioma", "3 molecular groups", "common"), ("lymphoma", FAMILY_LABELS["lymphoma"], "", "rare"),
        ("metastasis", "Metastasis", "", "rare")]
SITES = [("ebrains", "EBRAINS (reference)", "o"), ("patho2", "In-house", "s")]


def grid(ax: plt.Axes, cfg: dict) -> None:
    t = counts(cfg).set_index(["family", "group"])
    colors = cfg["display"]["colors"]["sources"]
    x0, cw, ch, gap = 1.9, 1.55, 0.36, 0.35   # 列の幅、行の高さ、SFT と Glioma の間の空き
    xs = [x0 + i * cw + (gap if i >= 2 else 0) for i in range(len(COLS))]
    y_rows = [0.85, 0.45]
    for (fam, name, note, kind), x in zip(COLS, xs):
        col = FAMILY_COLOR[fam]
        ls = "-" if kind == "common" else "--"
        ax.add_patch(Rectangle((x + 0.04, y_rows[1] - ch / 2 - 0.04), cw - 0.08, (y_rows[0] - y_rows[1]) + ch + 0.08, fill=False,
                               ec=col, lw=1.6 if kind == "common" else 1.0, ls=ls))
        ax.text(x + cw / 2, y_rows[0] + ch / 2 + 0.17, name, ha="center", va="bottom", fontsize=FONT, color=col, fontweight="bold")
        if note:
            ax.text(x + cw / 2, y_rows[0] + ch / 2 + 0.05, note, ha="center", va="bottom", fontsize=FONT - 2.5, color="0.3")
        r = t.loc[(fam, "(all)")]
        for (site, _, mk), y in zip(SITES, y_rows):
            ax.text(x + cw / 2, y, cell(r, site), ha="center", va="center", fontsize=FONT - 1)
    for (site, lab, mk), y in zip(SITES, y_rows):
        ax.scatter([0.15], [y], marker=mk, s=40, color=colors[site])
        ax.text(0.32, y, lab, ha="left", va="center", fontsize=FONT - 0.5)
    ax.text(0.15, y_rows[1] - ch / 2 - 0.02, "Cells: slides (patients)", fontsize=FONT - 2.5, color="0.35", va="top")
    # 借りる向き。主 = 髄膜腫 ⇄ グリオーマ（両向き、太く濃い弧）、従 = よくある腫瘍 → 隣の鑑別の腫瘍（細く薄い鉤形。グリオーマからは
    # 1 本の幹を PCNSL と転移に枝分かれさせる）
    base = y_rows[0] + ch / 2 + 0.40          # 鉤形の足元（見出しの上）
    stub = base + 0.16                         # 鉤形の横線の高さ
    minor = dict(color="0.6", lw=0.8)
    for src_i, dsts in ((0, [1]), (2, [3, 4])):
        x_s = xs[src_i] + cw / 2 + 0.3
        x_d = [xs[d] + cw / 2 for d in dsts]
        ax.plot([x_s, x_s, max(x_d)], [base, stub, stub], solid_capstyle="round", **minor)   # 幹 → 横線
        for x in x_d:
            ax.annotate("", xy=(x, base), xytext=(x, stub + 0.004), arrowprops=dict(arrowstyle="-|>", mutation_scale=7, **minor))
    ax.text((xs[0] + cw / 2 + 0.3 + xs[1] + cw / 2) / 2, stub + 0.03, "Δ borrowed", ha="center", va="bottom", fontsize=FONT - 3, color="0.5")
    ax.text((xs[2] + cw / 2 + 0.3 + xs[4] + cw / 2) / 2, stub + 0.03, "Δ borrowed", ha="center", va="bottom", fontsize=FONT - 3, color="0.5")
    # 主の矢印は片向き 2 本（ken 2026-11-25）。上 = 髄膜腫 → グリオーマ、下 = グリオーマ → 髄膜腫（どちらもまっすぐ）。
    # どちらも鉤形より上から出す（鉤形と交差しないように。端は見出しの左寄り）。ラベルは各矢印のすぐ上
    xa, xb = xs[0] + cw / 2 - 0.3, xs[2] + cw / 2 - 0.3
    main = dict(mutation_scale=11, lw=1.8, color="0.15", shrinkA=0, shrinkB=0)
    lab = dict(ha="center", va="bottom", fontsize=FONT - 1.5, color="0.15", fontweight="bold")
    y_low = stub + 0.22
    ax.add_patch(FancyArrowPatch((xb, y_low), (xa, y_low), arrowstyle="-|>", **main))
    ax.text((xa + xb) / 2, y_low + 0.03, "glioma Δ → meningioma", **lab)
    y_up = y_low + 0.26   # 上も同じくまっすぐ（弧にすると頂点がラベルに重なる）
    ax.add_patch(FancyArrowPatch((xa, y_up), (xb, y_up), arrowstyle="-|>", **main))
    ax.text((xa + xb) / 2, y_up + 0.03, "meningioma Δ → glioma", **lab)
    top = y_up + 0.12
    ax.legend(handles=[Line2D([], [], color="0.3", lw=1.6, label="Common tumour (shift measured here)"),
                       Line2D([], [], color="0.3", lw=1.0, ls="--", label="Differential tumour (shift borrowed)")],
              loc="upper right", bbox_to_anchor=(1.0, 0.0), fontsize=FONT - 2.5, frameon=False, ncol=2)
    ax.set_xlim(0, xs[-1] + cw + 0.05)
    ax.set_ylim(y_rows[1] - ch / 2 - 0.12, top + 0.2)
    ax.axis("off")


def flow(ax: plt.Axes) -> None:
    boxes = [("Whole-slide image\n(H&E)", ""), ("Patches\n512 px, ~0.45 µm/px", ""),
             ("Patch encoder\nCONCH v1.5 / Virchow", ""),
             ("Slide embedding\nTITAN, PRISM (learned aggregators)\nCONCH mean, Virchow mean", "")]
    w, h, gap = 2.05, 0.8, 0.45
    for i, (txt, _) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#f2f2f2", ec="0.4", lw=0.8))
        ax.text(x + w / 2, h / 2, txt, ha="center", va="center", fontsize=FONT - 1.5 if i < 3 else FONT - 2.2)
        if i:
            ax.annotate("", xy=(x - 0.03, h / 2), xytext=(x - gap + 0.03, h / 2), arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.9))
    ax.set_xlim(-0.05, len(boxes) * (w + gap) - gap + 0.05)
    ax.set_ylim(-0.1, h + 0.1)
    ax.axis("off")


def concept(fig: plt.Figure, rect: list[float], cfg: dict) -> None:
    """合成の 2 次元の点群（説明用）。推定元 = 髄膜腫の色、当て先 = グリオーマの色。"""
    rng = np.random.default_rng(42)
    colors = cfg["display"]["colors"]["sources"]
    d_src = np.array([2.4, 1.0])                 # 推定元のずれ（EBRAINS − 院内）
    d_tgt = np.array([2.5, 0.75])                # 当て先自身のずれ（推定元とほぼ同じ向きと長さ）
    src_e, tgt_e = np.array([1.3, 0.45]), np.array([1.3, 0.35])   # どちらもコマの中央に来るように
    pts = lambda c, n: c + rng.normal(0, 0.35, (n, 2))   # noqa: E731
    S_e, S_p = pts(src_e, 25), pts(src_e - d_src, 25)
    T_e, T_p = pts(tgt_e, 25), pts(tgt_e - d_tgt, 25)
    titles = ["Measure Δ on a common tumour", "No correction", "Δ borrowed from\nanother tumour", "Δ from the tumour itself"]
    x0, y0, w, h = rect
    gapx = 0.02
    pw = (w - 3 * gapx) / 4
    for k, title in enumerate(titles):
        ax = fig.add_axes([x0 + k * (pw + gapx), y0, pw, h])
        if k == 0:
            ax.scatter(*S_e.T, marker="o", s=9, color=FAMILY_COLOR["meningioma"], alpha=0.8, linewidths=0)
            ax.scatter(*S_p.T, marker="s", s=9, facecolors="white", edgecolors=FAMILY_COLOR["meningioma"], linewidths=0.6)
            me, mp = S_e.mean(0), S_p.mean(0)
            ax.annotate("", xy=me, xytext=mp, arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.3))
            ax.text(*(mp + me) / 2 + np.array([0.15, -0.55]), "Δ", fontsize=FONT + 1, fontweight="bold")
        else:
            shift = {1: np.zeros(2), 2: d_src, 3: T_e.mean(0) - T_p.mean(0)}[k]
            ax.scatter(*T_e.T, marker="o", s=9, color=FAMILY_COLOR["glioma"], alpha=0.8, linewidths=0)
            moved = T_p + shift
            ax.scatter(*moved.T, marker="s", s=9, facecolors="white", edgecolors=FAMILY_COLOR["glioma"], linewidths=0.6)
            if k > 1:
                ax.annotate("", xy=T_p.mean(0) + shift, xytext=T_p.mean(0), arrowprops=dict(arrowstyle="-|>", color="0.15", lw=1.1))
                ax.scatter(*T_p.T, marker="s", s=6, facecolors="none", edgecolors="0.75", linewidths=0.4)
        ax.set_xlim(-2.3, 2.4)
        ax.set_ylim(-1.6, 1.7)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.6")
        ax.set_title(title, fontsize=FONT - 1.5, pad=3)
    lax = fig.add_axes([x0, y0 - 0.07, w, 0.04])
    lax.axis("off")
    lax.legend(handles=[Line2D([], [], marker="o", ls="", color=FAMILY_COLOR["meningioma"], label="Common tumour, EBRAINS"),
                        Line2D([], [], marker="s", ls="", markerfacecolor="white", color=FAMILY_COLOR["meningioma"], label="Common tumour, in-house"),
                        Line2D([], [], marker="o", ls="", color=FAMILY_COLOR["glioma"], label="Target tumour, EBRAINS"),
                        Line2D([], [], marker="s", ls="", markerfacecolor="white", color=FAMILY_COLOR["glioma"], label="Target tumour, in-house"),
                        Line2D([], [], marker="s", ls="", markerfacecolor="none", color="0.7", label="before the shift")],
               loc="center", ncol=5, fontsize=FONT - 2.5, frameon=False, handletextpad=0.3, columnspacing=1.0, markerscale=0.8)
    del colors


def main() -> None:
    print(f"[preview] {NAME}")
    cfg = config()
    fig = plt.figure(figsize=(7.2, 5.4))
    ax_a = fig.add_axes([0.02, 0.62, 0.96, 0.36])
    grid(ax_a, cfg)
    fig.text(0.01, 0.975, "a", fontsize=FONT + 4, fontweight="bold", va="top")
    ax_b = fig.add_axes([0.04, 0.42, 0.94, 0.12])
    flow(ax_b)
    fig.text(0.01, 0.56, "b", fontsize=FONT + 4, fontweight="bold", va="top")
    concept(fig, [0.04, 0.11, 0.94, 0.22], cfg)
    fig.text(0.01, 0.385, "c", fontsize=FONT + 4, fontweight="bold", va="top")
    d = paper_root(cfg) / PAPER_VERSION / "fig"
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{NAME}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {d / NAME}.png / .pdf")


if __name__ == "__main__":
    main()
