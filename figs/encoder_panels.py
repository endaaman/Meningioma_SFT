"""v4 Fig 6 エンコーダで共通なこと、違うこと: 同じ腫瘍で推定した平行移動は 4 つのエンコーダで効くが、別の腫瘍から借りて効くのは
TITAN と CONCH の平均。並びは ENCODER_ORDER（CONCH mean → TITAN → Virchow mean → PRISM）、色は ENCODER_COLORS。

a 基礎精度（施設内の 5 分割交差検証の balanced accuracy、患者単位で分割）。横軸 = 課題（SFT 対 髄膜腫、髄膜腫 5 サブタイプ、
  グリオーマ 3 群、5 つの腫瘍）、施設ごとに 1 枚
b ずれの向き。横軸 = エンコーダ、縦軸 = 髄膜腫だけのずれとの cos（生の値）、点 = 12 群（色 = 腫瘍）
c 隣り合う組の数（平行移動）。当て先ごとに 1 枚（髄膜腫 12 組 / グリオーマ 3 組）、3 本（補正なし / 借りる / 同じ腫瘍で推定）
d 施設差の相対的な大きさ（‖Δ_meni‖ ÷ 院内の髄膜腫の施設内の広がり（群内 RMS）。encoder_comparison.py --reference meningioma の値で、
  ずれは髄膜腫だけのもの。Fig 3 の基準と同じ）と、
  借りたときに中心の差が残る割合（‖残差‖ ÷ 補正なしの ‖残差‖、グリオーマ ← 髄膜腫 / 髄膜腫 ← グリオーマ）

入力: output_dir/encoder_baseline/baseline.csv、output_dir/shift_transfer/{,meanpool/,prism/,meanpool_virchow/}groups.csv、
      output_dir/transfer/correction_{encoder}.csv、output_dir/encoder_comparison/reference_meningioma/site_gap.csv
出力: fig/{fig n}_encoder_panels.{png,pdf}、同 _values.csv

Usage:
    uv run python -m figs.encoder_panels
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figs import label
from figs.common import FONT, config, fig_path, out_root, panel, save
from shift_transfer import FAMILY_COLOR
from utils.display import ENCODER_COLORS, ENCODER_LABELS, ENCODER_ORDER, FAMILY_LABELS

NAME = "encoder_panels"
TASKS = [("sft", "SFT vs meningioma"), ("histotype", "Meningioma subtypes"), ("glioma3", "Glioma groups"), ("family5", "Tumour types")]
COND_C = [("none", "No correction", "#a3a29b"), ("borrow", "Δ borrowed", "#2a78d6"), ("own", "Δ from the same tumour", "#1baf7a")]


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    out = out_root(cfg)
    vals = []
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 0.28, 1, 1], hspace=0.62, wspace=0.45, left=0.08, right=0.98, top=0.93, bottom=0.12)
    nenc = len(ENCODER_ORDER)
    w = 0.8 / nenc

    # a
    b = pd.read_csv(out / "encoder_baseline" / "baseline.csv")
    b = b[b["metric"] == "balanced_accuracy"]
    for k, (site, title) in enumerate((("ebrains", "EBRAINS"), ("patho2", "In-house"))):
        ax = fig.add_subplot(gs[0, k])
        for j, enc in enumerate(ENCODER_ORDER):
            ys = [float(b[(b["task"] == t) & (b["site"] == site) & (b["encoder"] == enc)]["value"].iloc[0]) for t, _ in TASKS]
            ax.bar(np.arange(len(TASKS)) + (j - (nenc - 1) / 2) * w, ys, width=w * 0.9, color=ENCODER_COLORS[enc], label=ENCODER_LABELS[enc])
            vals += [{"panel": "a", "site": site, "task": t, "encoder": enc, "value": y} for (t, _), y in zip(TASKS, ys)]
        ax.set_xticks(range(len(TASKS)))
        ax.set_xticklabels([n for _, n in TASKS], fontsize=FONT - 3, rotation=35, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(f"Within-site, {title}", fontsize=FONT - 1, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("Balanced accuracy")
            panel(ax, "a", x=-0.3, y=1.04)
    fig.legend(handles=[Patch(color=ENCODER_COLORS[e], label=ENCODER_LABELS[e]) for e in ENCODER_ORDER],
               loc="upper center", ncol=nenc, frameon=False, fontsize=FONT - 1, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.0, handleheight=0.8, handletextpad=0.5, columnspacing=1.6)

    # b
    ax = fig.add_subplot(gs[0, 3:])
    rng = np.random.default_rng(0)   # 点の横の散らし（表示だけ）
    for i, enc in enumerate(ENCODER_ORDER):
        g = pd.read_csv(out / "shift_transfer" / ("" if enc == "titan" else enc) / "groups.csv")
        g = g[g["kind"].isin(["sft", "subtype", "glioma"])]
        xs = i + rng.uniform(-0.15, 0.15, len(g))
        ax.scatter(xs, g["cos_vs_meni"], s=14, c=[FAMILY_COLOR[f] for f in g["family"]], zorder=3)
        ax.hlines(g["cos_vs_meni"].median(), i - 0.25, i + 0.25, color="0.2", lw=1.2)
        vals += [{"panel": "b", "encoder": enc, "group": r.group, "value": r.cos_vs_meni} for r in g.itertuples()]
    ax.set_xticks(range(nenc))
    ax.set_xticklabels([ENCODER_LABELS[e] for e in ENCODER_ORDER], fontsize=FONT - 2)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("cos with the meningioma site shift")
    ax.set_title("Direction of the site shift, 12 groups (bar = median)", fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fams = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FAMILY_COLOR[f], label=FAMILY_LABELS[f]) for f in fams],
              fontsize=FONT - 3, frameon=False, loc="lower left", ncol=3)
    panel(ax, "b", x=-0.14, y=1.04)

    # c
    for k, (target, other, n, title) in enumerate((("meningioma", "glioma", 12, "Meningioma, 12 pairs"),
                                                   ("glioma", "meningioma", 3, "Glioma, 3 pairs"))):
        ax = fig.add_subplot(gs[1, k])
        for j, enc in enumerate(ENCODER_ORDER):
            c = pd.read_csv(out / "transfer" / f"correction_{enc}.csv")
            c = c[(c["method"] == "centroid") & (c["target"] == target) & (c["metric"] == "pairs_adjacent")]
            ys = [float(c[c["source"] == {"borrow": other}.get(cond, cond)]["value"].iloc[0]) for cond, _, _ in COND_C]
            for m, ((cond, lab, col), y) in enumerate(zip(COND_C, ys)):
                ax.bar(j + (m - 1) * 0.26, y, width=0.24, color=col, label=lab if j == 0 else None)
            vals += [{"panel": "c", "target": target, "encoder": enc, "condition": cond, "value": y} for (cond, _, _), y in zip(COND_C, ys)]
        ax.axhline(n, color="0.5", lw=0.7, ls=":")
        ax.set_ylim(0, n * 1.12)
        ax.set_xticks(range(nenc))
        ax.set_xticklabels([ENCODER_LABELS[e].replace(" ", "\n") for e in ENCODER_ORDER], fontsize=FONT - 3)
        ax.set_title(f"Target: {title}", fontsize=FONT - 1, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.set_ylabel("Cross-site pairs")
            panel(ax, "c", x=-0.3, y=1.04)
            ax.legend(fontsize=FONT - 3, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=3)

    # d
    ax = fig.add_subplot(gs[1, 3:])
    sg = pd.read_csv(out / "encoder_comparison" / "reference_meningioma" / "site_gap.csv").set_index("encoder")
    rel = {e: float(sg.loc[e, "ratio"]) for e in ENCODER_ORDER}   # 髄膜腫だけのずれ ÷ 院内の髄膜腫の群内 RMS
    resid = {}
    for enc in ENCODER_ORDER:
        c = pd.read_csv(out / "transfer" / f"correction_{enc}.csv")
        c = c[(c["method"] == "centroid") & (c["metric"] == "residual_norm")]
        f = lambda t, s: float(c[(c["target"] == t) & (c["source"] == s)]["value"].iloc[0])  # noqa: E731
        resid[enc] = (f("glioma", "meningioma") / f("glioma", "none"), f("meningioma", "glioma") / f("meningioma", "none"))
    x = np.arange(nenc)
    ax.bar(x - 0.27, [rel[e] for e in ENCODER_ORDER], width=0.25, color="#555555", label="‖Δ meningioma‖ / within-site spread")
    ax.bar(x, [resid[e][0] for e in ENCODER_ORDER], width=0.25, color="#9ecae1", label="Residual left, glioma ← meningioma")
    ax.bar(x + 0.27, [resid[e][1] for e in ENCODER_ORDER], width=0.25, color="#3182bd", label="Residual left, meningioma ← glioma")
    for e in ENCODER_ORDER:
        vals += [{"panel": "d", "encoder": e, "quantity": "relative_size", "value": rel[e]},
                 {"panel": "d", "encoder": e, "quantity": "residual_glioma_from_meningioma", "value": resid[e][0]},
                 {"panel": "d", "encoder": e, "quantity": "residual_meningioma_from_glioma", "value": resid[e][1]}]
    ax.set_xticks(x)
    ax.set_xticklabels([ENCODER_LABELS[e] for e in ENCODER_ORDER], fontsize=FONT - 2)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Ratio")
    ax.set_title("Site-shift size and what is left after borrowing", fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=FONT - 3, frameon=False, loc="upper left", ncol=1)
    panel(ax, "d", x=-0.14, y=1.04)
    save(fig, cfg, NAME)
    v = pd.DataFrame(vals)
    v.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    print(v[v.panel.isin(["c", "d"])].to_string(index=False))


if __name__ == "__main__":
    main()
