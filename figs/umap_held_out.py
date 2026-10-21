"""held-out 症例（EBRAINS pat_id 831、ラベル矛盾でコホートから除外した 3 枚）を Fig 3 の UMAP に out-of-sample で写した図（Supplementary）。

3 枚は時間順に ① SFT 初発（27 歳）→ ② Atypical meningioma（同年、別部位）→ ③ SFT 再発（翌年）。重心法（held_out_case.py）では
① が両施設の重心で SFT 側、② と ③ は髄膜腫側（最近傍の組織型は Fibrous）。UMAP 上でどこに落ちるかを、補正なしと平行移動の
2 条件で、3 枚と Fibrous の近傍を拡大して示す。全体の UMAP（SFT の塊を含む）は inset に置き、拡大枠を描く。

    a: 補正なし（original）      b: 平行移動（centroid）
    背景: 施設 = マーカーの形（Fig 3 と同じ）、診断 = 色（髄膜腫 薄灰 / Fibrous 薄青 / SFT 黒）。3 枚 = 黒縁の白いマーカーに番号。
    3 枚は EBRAINS（reference 側）なので埋め込みは条件によらず同じ。条件で変わるのは UMAP の fit（patho2 側の補正後の点群）。

入力: output_dir/umap/{variant}/coords.csv・heldout_coords.csv（umap_plot.py。held-out は fit に使わず transform だけ）
出力: fig/{fig S n}_umap_held_out.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.umap_held_out
"""
from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from figs import label
from figs.common import CONDITION_LABELS, FONT, SITE_DISPLAY, config, out_root, panel, save, src_markers
from utils.display import subtype_color_map

NAME = "umap_held_out"
VARIANTS = ["original", "centroid"]
TITLES = {**CONDITION_LABELS, "original": "Uncorrected"}   # Fig 3 と同じ表記
FIBROUS = "Fibrous meningioma"
# 時間順（番号は図と凡例で共通）。臨床情報は config.yaml の held_out の注記（EBRAINS annotation.csv）から。
ORDER = [("a1965734", "SFT, primary (age 27)"),
         ("a1967e5e", "Atypical meningioma (same year, other site)"),
         ("a19545d6", "SFT, recurrence (next year)")]
MEN_COLOR = "#D0D0D0"          # 髄膜腫（Fibrous 以外）
FIBROUS_ALPHA = 0.55           # Fibrous は subtype_color_map の色を薄くする
ZOOM_W, ZOOM_H = 5.2, 3.4      # 拡大の幅・高さ（UMAP 単位、両パネル共通で尺度を揃える）
# 番号ラベルのオフセット（points）。② と ③ はほぼ重なるので反対側に逃がす
LABEL_OFFSET = {
    "original": {"a1965734": (-14, 14), "a1967e5e": (24, 2), "a19545d6": (-18, 14)},
    "centroid": {"a1965734": (14, 14), "a1967e5e": (-20, -12), "a19545d6": (18, -14)},
}
# inset（全体図）の位置（axes fraction）: 拡大の中で点の無い隅
INSET_RECT = {"original": [0.01, 0.02, 0.34, 0.34], "centroid": [0.65, 0.02, 0.34, 0.34]}
INSET_MARGIN = 0.08


def _colors(coords: pd.DataFrame, cfg: dict) -> dict[str, tuple]:
    """診断 → 色。SFT は Fig 3 と同じ（subtype_color_map）、Fibrous はその色を薄く、他の髄膜腫は薄い灰。"""
    sub_map = subtype_color_map(sorted(set(coords["subtype"])), cfg)
    out = {}
    for sub, c in sub_map.items():
        if "sft" in sub.lower():
            out[sub] = mcolors.to_rgba(c)
        elif sub == FIBROUS:
            out[sub] = mcolors.to_rgba(c, alpha=FIBROUS_ALPHA)
        else:
            out[sub] = mcolors.to_rgba(MEN_COLOR)
    return out


def _scatter(ax: plt.Axes, coords: pd.DataFrame, color: dict, src_mkr: dict, s: float, alpha: float) -> None:
    """描画順: 髄膜腫（灰）→ Fibrous → SFT（黒）の順で上に重ねる。"""
    def tier(sub: str) -> int:
        return 2 if "sft" in sub.lower() else (1 if sub == FIBROUS else 0)
    for t in (0, 1, 2):
        for src, mkr in src_mkr.items():
            m = (coords["source"] == src) & coords["subtype"].map(lambda x: tier(x) == t)
            if m.any():
                ax.scatter(coords.loc[m, "umap1"], coords.loc[m, "umap2"], marker=mkr, s=s, linewidths=0,
                           c=[color[sub] for sub in coords.loc[m, "subtype"]], alpha=alpha, zorder=2 + t)


def _held(ax: plt.Axes, held: pd.DataFrame, mkr: str, s: float, lw: float) -> None:
    ax.scatter(held["umap1"], held["umap2"], marker=mkr, s=s, facecolors="white", edgecolors="black",
               linewidths=lw, zorder=10)


def _zoom_box(held: pd.DataFrame, fibrous: pd.DataFrame) -> tuple[float, float, float, float]:
    """拡大範囲: 3 枚の重心と EBRAINS の Fibrous の中央値の中点を中心に、固定の幅・高さ。3 枚が必ず入るように中心をずらす。"""
    cx = (held["umap1"].mean() + fibrous["umap1"].median()) / 2
    cy = (held["umap2"].mean() + fibrous["umap2"].median()) / 2
    pad = 0.5
    cx = min(max(cx, held["umap1"].max() + pad - ZOOM_W / 2), held["umap1"].min() - pad + ZOOM_W / 2)
    cy = min(max(cy, held["umap2"].max() + pad - ZOOM_H / 2), held["umap2"].min() - pad + ZOOM_H / 2)
    return cx - ZOOM_W / 2, cx + ZOOM_W / 2, cy - ZOOM_H / 2, cy + ZOOM_H / 2


def build(cfg: dict) -> plt.Figure:
    root = out_root(cfg)
    ref = cfg["reference"]
    coords = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
              for v in VARIANTS}
    held = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "heldout_coords.csv", dtype={"case_id": str})
            .set_index("case_id").loc[[c for c, _ in ORDER]].reset_index() for v in VARIANTS}
    color = _colors(coords["original"], cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))
    held_mkr = src_mkr[ref]   # held-out は reference 施設のスライド

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, v, letter in zip(axes, VARIANTS, "ab"):
        c, h = coords[v], held[v]
        x0, x1, y0, y1 = _zoom_box(h, c[(c["source"] == ref) & (c["subtype"] == FIBROUS)])
        _scatter(ax, c, color, src_mkr, s=16, alpha=0.9)
        _held(ax, h, held_mkr, s=70, lw=1.1)
        for i, (case_id, _) in enumerate(ORDER, start=1):
            r = h[h["case_id"] == case_id].iloc[0]
            ax.annotate(str(i), (r["umap1"], r["umap2"]), xytext=LABEL_OFFSET[v][case_id], textcoords="offset points",
                        ha="center", va="center", fontsize=FONT, fontweight="bold", zorder=12,
                        bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec="black", lw=0.7),
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.6, shrinkA=0, shrinkB=4))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP1", fontsize=FONT - 1); ax.set_ylabel("UMAP2", fontsize=FONT - 1)
        ax.set_title(TITLES[v], fontsize=FONT)
        panel(ax, letter, x=-0.04)

        # inset: 全体の UMAP と拡大枠
        ins = ax.inset_axes(INSET_RECT[v])
        _scatter(ins, c, color, src_mkr, s=1.2, alpha=0.9)
        _held(ins, h, held_mkr, s=9, lw=0.6)
        ins.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="black", lw=0.7, zorder=11))
        mx = INSET_MARGIN * (c["umap1"].max() - c["umap1"].min())
        my = INSET_MARGIN * (c["umap2"].max() - c["umap2"].min())
        ins.set_xlim(c["umap1"].min() - mx, c["umap1"].max() + mx)
        ins.set_ylim(c["umap2"].min() - my, c["umap2"].max() + my)
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_linewidth(0.6)

    # 凡例（2 パネル共通、下に横並び）: 診断の色 / 施設の形 / held-out の番号
    handles = [Patch(facecolor=color[next(s for s in color if "sft" in s.lower())], label="SFT"),
               Patch(facecolor=color[FIBROUS], label="Fibrous meningioma"),
               Patch(facecolor=MEN_COLOR, label="Other meningioma")]
    handles += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=SITE_DISPLAY.get(t, t))
                for t, m in src_mkr.items()]
    handles += [Line2D([0], [0], marker=held_mkr, mfc="white", mec="black", mew=1.0, ls="none", markersize=7,
                       label="Held-out slides (one EBRAINS patient)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.085), ncol=6, frameon=False,
               handlelength=1.0, handleheight=0.9, columnspacing=1.0, handletextpad=0.4, fontsize=FONT - 1)
    fig.text(0.5, 0.02, "   ".join(f"{i} {t}" for i, (_, t) in enumerate(ORDER, start=1)),
             ha="center", va="bottom", fontsize=FONT - 1)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.2, wspace=0.12)
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(build(cfg), cfg, NAME)


if __name__ == "__main__":
    main()
