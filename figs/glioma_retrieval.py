"""v4 Fig 7 出口: グリオーマの参照コホートへの診断名の検索（TITAN）。髄膜腫で測ったずれを足すと、院内のグリオーマを公開コホート
（EBRAINS 全体の 78 診断の重心）に最近傍で当てたときの診断名の一致が上がる。

a 1 位の診断名の一致。横軸 = グリオーマ全体と 3 群、4 本（補正なし / 髄膜腫だけのずれ / 同じ腫瘍のずれ = 上限の参考 /
  EBRAINS の中での値 = 施設差が無い場合の上限、自分を除いた重心）
b 5 位以内の一致。a と同じ形
c 3 群の混同行列（補正なし / 髄膜腫だけのずれ）。行 = 正解の群、列 = 1 位で引けた診断の群（3 群 ＋ ほかのグリオーマ ＋ グリオーマ以外）。
  題に 3 群の balanced accuracy（3 群の recall の平均）

入力: output_dir/atlas/retrieval_titan.csv、glioma_confusion_titan.csv（atlas_retrieval.py、条件 meningioma = 髄膜腫だけで推定した
      ずれ、family = 同じ腫瘍のずれ、ebrains_loo = EBRAINS の中で自分を除いた重心、検索 = nc_euclid）
出力: fig/{fig n}_glioma_retrieval.{png,pdf}、同 _values.csv

Usage:
    uv run python -m figs.glioma_retrieval
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figs import label
from figs.common import FONT, config, fig_path, out_root, panel, save
from utils.display import GLIOMA_GROUP_LABELS

NAME = "glioma_retrieval"
CONDS = [("none", "No correction", "#a3a29b"), ("meningioma", "Δ borrowed from meningioma", "#2a78d6"),
         ("family", "Δ from glioma itself (reference)", "#1baf7a"), ("ebrains_loo", "Within EBRAINS (no site difference)", "#555555")]
GROUPS = [("glioma", "Glioma (all)"), ("ast", "Astrocytoma,\nIDH-mutant"), ("glm", "Glioblastoma,\nIDH-wildtype"), ("odg", "Oligodendro-\nglioma")]
PRED = ["ast", "glm", "odg", "other glioma", "non-glioma"]
PRED_LABEL = {"ast": "Astro.", "glm": "GBM", "odg": "Oligo.", "other glioma": "Other\nglioma", "non-glioma": "Non-\nglioma"}


def balanced_acc(c: pd.DataFrame) -> float:
    t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=["ast", "glm", "odg"]).fillna(0)
    return float(np.mean([t.loc[g, g] / t.loc[g].sum() if g in t.columns and t.loc[g].sum() else 0.0 for g in t.index]))


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    a = out_root(cfg) / "atlas"
    r = pd.read_csv(a / "retrieval_titan.csv")
    r = r[(r["retrieval"] == "nc_euclid") & (r["query_label"] == "(all)")]
    conf = pd.read_csv(a / "glioma_confusion_titan.csv")
    conf = conf[(conf["retrieval"] == "nc_euclid") & (conf["scope"] == "full")]

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28, left=0.08, right=0.98, top=0.93, bottom=0.08)
    vals = []
    w = 0.2
    for k, (metric, title) in enumerate((("top1", "Correct diagnosis ranked first"), ("top5", "Correct diagnosis in the top 5"))):
        ax = fig.add_subplot(gs[0, k])
        for j, (cond, lab, col) in enumerate(CONDS):
            ys = [float(r[(r["metric"] == metric) & (r["condition"] == cond) & (r["query_family"] == g)]["value"].iloc[0]) for g, _ in GROUPS]
            xs = np.arange(len(GROUPS)) + (j - 1.5) * w
            ax.bar(xs, ys, width=w * 0.92, color=col, label=lab)
            vals += [{"panel": "ab", "metric": metric, "condition": cond, "group": g, "value": y} for (g, _), y in zip(GROUPS, ys)]
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([n for _, n in GROUPS], fontsize=FONT - 2)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Fraction of in-house slides")
        ax.set_title(title, fontsize=FONT, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, "ab"[k], x=-0.14, y=1.02)
        if k == 0:
            ax.legend(fontsize=FONT - 2.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    sub = gs[1, :].subgridspec(1, 2, wspace=0.35)
    for k, (cond, lab) in enumerate((("none", "No correction"), ("meningioma", "Δ borrowed from meningioma"))):
        ax = fig.add_subplot(sub[0, k])
        c = conf[conf["condition"] == cond]
        t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=["ast", "glm", "odg"], columns=PRED).fillna(0)
        frac = t.div(t.sum(axis=1), axis=0)
        ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(PRED)):
                v = int(t.iloc[i, j])
                ax.text(j, i, str(v), ha="center", va="center", fontsize=FONT - 1, color="white" if frac.iloc[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(PRED)))
        ax.set_xticklabels([PRED_LABEL[p] for p in PRED], fontsize=FONT - 2)
        ax.set_yticks(range(3))
        ax.set_yticklabels([GLIOMA_GROUP_LABELS[g].replace(", ", ",\n") for g in ["ast", "glm", "odg"]], fontsize=FONT - 2)
        ax.set_xlabel("Group of the first-ranked diagnosis", fontsize=FONT - 1)
        ba = balanced_acc(c)
        ax.set_title(f"{lab}: balanced accuracy {ba:.3f}", fontsize=FONT - 1, loc="left")
        if k == 0:
            panel(ax, "c", x=-0.42, y=1.04)
        vals.append({"panel": "c", "metric": "balanced_accuracy_3", "condition": cond, "group": "glioma", "value": ba})
    save(fig, cfg, NAME)
    pd.DataFrame(vals).to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    v = pd.DataFrame(vals)
    print(v.pivot_table(index=["metric", "condition"], columns="group", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
