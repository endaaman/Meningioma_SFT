"""stat6_preview.py — STAT6 の状態別のマージン分布のプレビュー（本番に入れるかは ken 判断待ち。figs/__init__.py には登録しない）。

patho2 の各スライドのマージン d(髄膜腫重心) − d(SFT 重心)（> 0 で SFT）を、STAT6 の状態の群（レポート本文の読み、
stat6_case.py の group_text）ごとに箱ひげ ＋ 点で示す。2 行 × 2 列: 行 = 重心（EBRAINS = 施設間 / patho2 = 施設内 leave-one-out）、
列 = 条件（補正なし / 平行移動）。STAT6 で確認されていない SFT と、EBRAINS → patho2 で見逃した SFT に case_id を付ける。

入力: output_dir/stat6/cases.csv（stat6_case.py）
出力: output_dir/paper/v2/fig/_preview_stat6.png（config の paper.version が v1 のままでも v2 に出す。v1 は触らない約束）

Usage:
    uv run python -m figs.stat6_preview
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from figs.common import CONDITION_LABELS, FONT, SITE_DISPLAY, config, out_root

PAPER_VERSION = "v2"
VARIANTS = ["original", "centroid"]
GROUPS = [("SFT_STAT6pos", "SFT\nSTAT6+"), ("SFT_STAT6neg", "SFT\nSTAT6−"), ("SFT_untested", "SFT\nnot tested"),
          ("Men_STAT6tested", "Meningioma\nSTAT6 tested"), ("Men_untested", "Meningioma\nnot tested")]
LABELLED = ["19-0491", "21-1704", "20-1778", "N22-006", "N23-193"]   # 確認されていない SFT 4 例 ＋ 見逃した SFT
SFT_COLOR, MEN_COLOR = "#000000", "#EE6677"   # SFT 黒 / 髄膜腫は patho2 の施設色（全スライドが patho2）


def main() -> None:
    cfg = config()
    cases = pd.read_csv(out_root(cfg) / "stat6" / "cases.csv")
    sites = [cfg["reference"], cfg["stat6"]["site"]]
    groups = [(g, lab) for g, lab in GROUPS if (cases["group_text"] == g).any()]
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6), sharey=True)
    for i, site in enumerate(sites):
        for j, v in enumerate(VARIANTS):
            ax = axes[i, j]
            s = cases[(cases["centroid_site"] == site) & (cases["condition"] == v)]
            data = [s[s["group_text"] == g]["margin"].to_numpy() for g, _ in groups]
            ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=0)
            ax.boxplot(data, positions=range(len(groups)), widths=0.5, showfliers=False,
                       medianprops={"color": "#333333", "lw": 1.2}, boxprops={"color": "#999999"},
                       whiskerprops={"color": "#999999"}, capprops={"color": "#999999"}, zorder=1)
            for k, ((g, _), d) in enumerate(zip(groups, data)):
                col = SFT_COLOR if g.startswith("SFT") else MEN_COLOR
                big = len(d) > 60
                x = k + rng.uniform(-0.2, 0.2, len(d))
                ax.scatter(x, d, s=6 if big else 14, color=col, alpha=0.25 if big else 0.8, lw=0, zorder=2)
                sub = s[s["group_text"] == g]
                for (_, r), xk in zip(sub.iterrows(), x):
                    if r["case_id"] in LABELLED:
                        ax.annotate(r["case_id"], (xk, r["margin"]), xytext=(4, 0), textcoords="offset points",
                                    fontsize=FONT - 2, va="center", color=col)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([f"{lab}\n(n={len(d)})" for (_, lab), d in zip(groups, data)], fontsize=FONT - 2)
            ax.set_title(f"{CONDITION_LABELS[v]} — centroids: {SITE_DISPLAY[site]}"
                         + (" (leave-one-out)" if site == cfg["stat6"]["site"] else ""), fontsize=FONT)
            if j == 0:
                ax.set_ylabel("d(meningioma) − d(SFT)  (> 0: SFT side)")
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("patho2 slides by STAT6 status (report text reading); TITAN, nearest-centroid margin", fontsize=FONT + 1)
    fig.tight_layout()
    out = out_root(cfg) / "paper" / PAPER_VERSION / "fig" / "_preview_stat6.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
