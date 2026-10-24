"""病理医の判断との対応（Fig 7）: 埋め込みの位置が、病理医の診断の揺れ・鑑別の挙がり方と合うかを 2 つの材料で見る。

    a–b  held-out 症例（EBRAINS pat_id 831、SFT 初発 → Atypical → SFT 再発の 3 枚）を Fig 3 の UMAP に写す（旧 Fig S5。
         描画は figs/umap_held_out.py の draw をそのまま使う）。a 補正なし / b 平行移動、Fibrous 近傍の拡大 ＋ 全体の inset。
    c–d  コホート全体の UMAP（Fig 3 と同じ座標）に、STAT6 免疫染色の記載がある patho2 症例を重ねる。c 補正なし / d 平行移動。
         背景は全スライドを薄灰（形は施設、Fig 3 と同じ）。STAT6 の記載がある髄膜腫 48 例（= 病理医が SFT を鑑別に挙げた髄膜腫）は
         組織型の色（Fig 3 の色。Fibrous / Atypical / Angiomatous / Microcystic / NOS、それ以外は濃灰にまとめる）の ■、
         SFT STAT6 核陽性 20 例は黒の ■、STAT6 未施行の SFT 2 例は白い ◆、EBRAINS の SFT 32 枚は黒の○（縁だけ）で SFT の塊の位置を示す。
    e    距離差 d(髄膜腫重心) − d(SFT 重心)（EBRAINS の重心、> 0 で SFT 側）を STAT6 の状態の群ごとに、補正なし / 平行移動で箱ひげ ＋ 点。
         群はレポート本文の読み（stat6_case.py の group_text）: SFT STAT6+ / SFT 未施行 / 髄膜腫 STAT6 記載あり / 記載なし。

入力: output_dir/umap/{original,centroid}/{coords,heldout_coords}.csv（umap_plot.py）、output_dir/stat6/cases.csv（stat6_case.py）、
      config の stat6.status_csv（症例ごとの STAT6 の状態。group_text を使う）
出力: fig/{fig n}_pathology_checks.{png,pdf}（番号は figs/__init__.py）

Usage:
    uv run python -m figs.pathology_checks
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figs import label
from figs.common import CONDITION_COLORS, FONT, SITE_DISPLAY, config, out_root, panel, save, src_markers
from figs.umap_held_out import TITLES, VARIANTS
from figs.umap_held_out import draw as draw_held_out
from utils.display import shorten, subtype_color_map

NAME = "pathology_checks"
# c–d: 組織型の色を付ける STAT6 記載ありの髄膜腫の組織型（この順で凡例。残りは OTHER にまとめる）。例数は stat6_status.csv から
SUBTYPES = ["Fibrous meningioma", "Atypical meningioma", "Angiomatous meningioma", "Microcystic meningioma", "NOS meningioma"]
OTHER_COLOR = "#555555"          # SUBTYPES 以外の STAT6 記載ありの髄膜腫（Metaplastic / Anaplastic / Meningothelial / Chordoid / Transitional / Secretory）
BG_COLOR = "#DCDCDC"             # 背景（全スライド）
# e: 群（group_text）と英語の表示名
GROUPS = [("SFT_STAT6pos", "SFT\nSTAT6+"), ("SFT_untested", "SFT\nnot\ntested"),
          ("Men_STAT6tested", "Meningioma\nSTAT6\nrecorded"), ("Men_untested", "Meningioma\nnot\nrecorded")]
STRIP_MANY = 100                 # これより多い群は点を小さく薄く
UMAP_MARGIN = 0.08               # c–d の軸の余白（SFT の塊が平行移動では右上の隅に来るので広めに）


def _load(cfg: dict) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    root = out_root(cfg)
    status = pd.read_csv(cfg["stat6"]["status_csv"], dtype={"case_id": str})
    coords = {}
    for v in VARIANTS:
        c = pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
        coords[v] = c.merge(status[["case_id", "group_text"]], on="case_id", how="left")
    cases = pd.read_csv(root / "stat6" / "cases.csv", dtype={"case_id": str})
    cases = cases[(cases["centroid_site"] == cfg["reference"]) & cases["condition"].isin(VARIANTS)]
    return coords, status, cases


def _stat6_umap(ax: plt.Axes, c: pd.DataFrame, cfg: dict, site: str, ref: str, src_mkr: dict, sub_color: dict) -> None:
    """c–d の 1 パネル。"""
    for src, mkr in src_mkr.items():
        m = c["source"] == src
        ax.scatter(c.loc[m, "umap1"], c.loc[m, "umap2"], marker=mkr, s=4, c=BG_COLOR, linewidths=0, zorder=1)
    sft_ref = c[(c["source"] == ref) & c["subtype"].str.contains("sft", case=False)]
    ax.scatter(sft_ref["umap1"], sft_ref["umap2"], marker=src_mkr[ref], s=14, facecolors="none", edgecolors="black",
               linewidths=0.5, alpha=0.6, zorder=2)
    mkr = src_mkr[site]
    men = c[(c["source"] == site) & (c["group_text"] == "Men_STAT6tested")]
    ax.scatter(men["umap1"], men["umap2"], marker=mkr, s=18, c=[sub_color.get(s, OTHER_COLOR) for s in men["subtype"]],
               edgecolors="black", linewidths=0.4, zorder=3)
    pos = c[(c["source"] == site) & (c["group_text"] == "SFT_STAT6pos")]
    ax.scatter(pos["umap1"], pos["umap2"], marker=mkr, s=18, c="black", linewidths=0, zorder=4)
    unt = c[(c["source"] == site) & (c["group_text"] == "SFT_untested")]
    ax.scatter(unt["umap1"], unt["umap2"], marker="D", s=22, facecolors="white", edgecolors="black", linewidths=0.8,
               zorder=5)
    mx = UMAP_MARGIN * (c["umap1"].max() - c["umap1"].min())
    my = UMAP_MARGIN * (c["umap2"].max() - c["umap2"].min())
    ax.set_xlim(c["umap1"].min() - mx, c["umap1"].max() + mx)
    ax.set_ylim(c["umap2"].min() - my, c["umap2"].max() + my)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=FONT - 1); ax.set_ylabel("UMAP2", fontsize=FONT - 1)


def _margins(ax: plt.Axes, cases: pd.DataFrame, ref: str) -> list:
    """e: 群 × 条件の箱ひげ ＋ 点。条件の凡例のハンドルを返す。"""
    rng = np.random.default_rng(42)
    groups = [(g, lab) for g, lab in GROUPS if (cases["group_text"] == g).any()]
    off = {"original": -0.2, "centroid": 0.2}
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)
    for v in VARIANTS:
        s = cases[cases["condition"] == v]
        data = [s[s["group_text"] == g]["margin"].to_numpy() for g, _ in groups]
        pos = [k + off[v] for k in range(len(groups))]
        col = CONDITION_COLORS[v]
        ax.boxplot(data, positions=pos, widths=0.3, showfliers=False,
                   medianprops={"color": "#222222", "lw": 1.0}, boxprops={"color": col, "lw": 1.0},
                   whiskerprops={"color": col, "lw": 0.8}, capprops={"color": col, "lw": 0.8}, zorder=2)
        for x, d in zip(pos, data):
            many = len(d) > STRIP_MANY
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(d)), d, s=3 if many else 7, color=col,
                       alpha=0.25 if many else 0.8, lw=0, zorder=1)
    ns = {g: int((cases[cases["condition"] == VARIANTS[0]]["group_text"] == g).sum()) for g, _ in groups}
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lab}\n(n = {ns[g]})" for g, lab in groups], fontsize=FONT - 2)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("d(meningioma) − d(SFT)  (> 0: SFT side)", fontsize=FONT - 2)
    ax.set_title(f"{SITE_DISPLAY.get(ref, ref)} centroids", fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT - 2)
    ax.spines[["top", "right"]].set_visible(False)
    return [Patch(facecolor="white", edgecolor=CONDITION_COLORS[v], lw=1.2, label=TITLES[v]) for v in VARIANTS]


def build(cfg: dict) -> plt.Figure:
    ref, site = cfg["reference"], cfg["stat6"]["site"]
    coords, status, cases = _load(cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))
    sub_color = subtype_color_map(SUBTYPES, cfg)
    men = status[status["group_text"] == "Men_STAT6tested"]
    n_sub = men["subtype"].value_counts()
    n_other = int(len(men) - sum(n_sub.get(s, 0) for s in SUBTYPES))

    fig = plt.figure(figsize=(7.2, 7.9))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[3.3, 0.5, 2.55, 0.9], hspace=0.28)

    # a–b: held-out 症例（umap_held_out.draw）
    top = gs[0].subgridspec(1, 2, wspace=0.12)
    axes_ab = [fig.add_subplot(top[0, i]) for i in range(2)]
    handles_ab, note = draw_held_out(axes_ab, cfg, letters="ab")
    ax_leg = fig.add_subplot(gs[1]); ax_leg.axis("off")
    ax_leg.legend(handles=handles_ab, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False,
                  handlelength=1.0, handleheight=0.9, columnspacing=1.0, handletextpad=0.4, borderaxespad=0,
                  fontsize=FONT - 1)
    ax_leg.text(0.5, -0.1, note, transform=ax_leg.transAxes, ha="center", va="bottom", fontsize=FONT - 1)

    # c–e
    mid = gs[2].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.4], wspace=0.3)
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(mid[0, i])
        _stat6_umap(ax, coords[v], cfg, site, ref, src_mkr, sub_color)
        ax.set_title(TITLES[v], fontsize=FONT)
        if i:
            ax.set_ylabel("")
        panel(ax, "cd"[i], x=-0.04)
    ax_e = fig.add_subplot(mid[0, 2])
    handles_e = _margins(ax_e, cases, ref)
    ax_e.legend(handles=handles_e, loc="upper right", frameon=False, fontsize=FONT - 2, handlelength=1.2,
                handleheight=0.9, borderaxespad=0.2)
    panel(ax_e, "e", x=-0.2)

    # c–d の凡例（下に 2 つ: 髄膜腫の組織型 / SFT と背景）
    bot = gs[3].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.05)
    ax_l1 = fig.add_subplot(bot[0, 0]); ax_l1.axis("off")
    mkr = src_mkr[site]
    h1 = [Line2D([0], [0], marker=mkr, mfc=sub_color[s], mec="black", mew=0.4, ls="none", markersize=5,
                 label=f"{shorten(s, cfg)} ({int(n_sub.get(s, 0))})") for s in SUBTYPES]
    h1 += [Line2D([0], [0], marker=mkr, mfc=OTHER_COLOR, mec="black", mew=0.4, ls="none", markersize=5,
                  label=f"Other ({n_other})")]
    ax_l1.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, 0.55), ncol=3, frameon=False,
                 title=f"Meningioma with STAT6 recorded ({SITE_DISPLAY.get(site, site)}, n = {len(men)}), by subtype",
                 title_fontsize=FONT - 1, handlelength=1.0, columnspacing=1.0, handletextpad=0.4, borderaxespad=0,
                 fontsize=FONT - 1)
    ax_l2 = fig.add_subplot(bot[0, 1]); ax_l2.axis("off")
    n_pos = int((status["group_text"] == "SFT_STAT6pos").sum())
    n_unt = int((status["group_text"] == "SFT_untested").sum())
    n_ref_sft = int(((coords["original"]["source"] == ref)
                     & coords["original"]["subtype"].str.contains("sft", case=False)).sum())
    h2 = [Line2D([0], [0], marker=mkr, mfc="black", mec="black", ls="none", markersize=5,
                 label=f"SFT, STAT6+ ({SITE_DISPLAY.get(site, site)}, n = {n_pos})"),
          Line2D([0], [0], marker="D", mfc="white", mec="black", mew=0.8, ls="none", markersize=5,
                 label=f"SFT, not tested ({SITE_DISPLAY.get(site, site)}, n = {n_unt})"),
          Line2D([0], [0], marker=src_mkr[ref], mfc="none", mec="black", mew=0.5, ls="none", markersize=5,
                 label=f"SFT ({SITE_DISPLAY.get(ref, ref)}, n = {n_ref_sft})"),
          Line2D([0], [0], marker="o", mfc=BG_COLOR, mec="none", ls="none", markersize=4,
                 label="All other slides (● EBRAINS, ■ patho2)")]
    ax_l2.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.5, 0.55), ncol=2, frameon=False,
                 title="SFT and background", title_fontsize=FONT - 1, handlelength=1.0, columnspacing=1.0,
                 handletextpad=0.4, borderaxespad=0, fontsize=FONT - 1)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.02)
    return fig


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    save(build(cfg), cfg, NAME)


if __name__ == "__main__":
    main()
