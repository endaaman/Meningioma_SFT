"""v4 Fig 7 出口: 参照コホートへの診断名の検索（TITAN）。髄膜腫で測ったずれを足すと、院内のグリオーマ・リンパ腫・転移・SFT を
公開コホート（EBRAINS 全体の 78 診断の重心）に最近傍で当てたときの診断名の一致がどう動くか。

a 1 位の診断名の一致。横軸 = 腫瘍（グリオーマ全体 116 枚、リンパ腫 52、転移 81、SFT 22）、4 本（補正なし / 髄膜腫だけのずれを借りる /
  その腫瘍自身のずれ = 参考 / EBRAINS の中での値 = 施設差が無い場合の目安、同じ患者の全スライドを重心から除いた値）。誤差線 = 患者単位の
  ブートストラップの 95% 区間（1,000 回、ずれは毎回当てはめ直す。EBRAINS の中は問い合わせの患者を再抽出）
b 5 位以内の一致。a と同じ形
c グリオーマの 3 群の内訳（1 位の一致）。a と同じ 4 本と誤差線
d 3 群の混同行列（補正なし / 髄膜腫のずれ）。行 = 正解の群、列 = 1 位で引けた診断の群（3 群 ＋ ほかのグリオーマ ＋ グリオーマ以外）。
  題に 3 群の balanced accuracy

入力: output_dir/atlas/retrieval_titan.csv・glioma_confusion_titan.csv（atlas_retrieval.py、検索 = nc_euclid。条件 meningioma =
      髄膜腫だけのずれ、family = その腫瘍自身のずれ、ebrains_lpo = EBRAINS の中で同じ患者を除いた重心）、
      output_dir/transfer_uncertainty/retrieval_tumours_summary.csv（transfer_uncertainty.py retrieval_tumours、誤差線）
出力: fig/{fig n}_diagnosis_retrieval.{png,pdf}、同 _values.csv

Usage:
    uv run python -m figs.diagnosis_retrieval
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from figs import label
from figs.common import FONT, config, fig_path, out_root, panel, save
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_LABELS

NAME = "diagnosis_retrieval"
CONDS = [("none", "No correction", "#a3a29b"), ("meningioma", "Δ borrowed from meningioma", "#2a78d6"),
         ("family", "Δ from the tumour itself (reference)", "#1baf7a"), ("ebrains_lpo", "Within EBRAINS (patient left out)", "#555555")]
TUMOURS = [("glioma", "Glioma (all)"), ("lymphoma", FAMILY_LABELS["lymphoma"]), ("metastasis", "Metastasis"), ("SFT", "SFT")]
GROUPS = [("ast", "Astrocytoma,\nIDH-mutant"), ("glm", "Glioblastoma,\nIDH-wildtype"), ("odg", "Oligodendro-\nglioma")]
PRED = ["ast", "glm", "odg", "other glioma", "non-glioma"]
PRED_LABEL = {"ast": "Astro.", "glm": "GBM", "odg": "Oligo.", "other glioma": "Other\nglioma", "non-glioma": "Non-\nglioma"}


def balanced_acc(c: pd.DataFrame) -> float:
    t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=["ast", "glm", "odg"]).fillna(0)
    return float(np.mean([t.loc[g, g] / t.loc[g].sum() if g in t.columns and t.loc[g].sum() else 0.0 for g in t.index]))


def bars(ax: plt.Axes, s: pd.DataFrame, keys: list[tuple[str, str]], metric: str, group_col: str, vals: list, panel_tag: str) -> None:
    w = 0.2
    for j, (cond, _, col) in enumerate(CONDS):
        ys, lo, hi = [], [], []
        for k, _ in keys:
            r = s[(s[group_col] == k) & (s["condition"] == cond) & (s["metric"] == metric)].iloc[0]
            ys.append(r["point"])
            lo.append(r["point"] - r["q025"])
            hi.append(r["q975"] - r["point"])
            vals.append({"panel": panel_tag, "metric": metric, "condition": cond, "group": k, "value": r["point"], "q025": r["q025"],
                         "q975": r["q975"], "n": int(r["n"])})
        xs = np.arange(len(keys)) + (j - 1.5) * w
        ax.bar(xs, ys, width=w * 0.92, color=col)
        ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor="0.15", elinewidth=0.6, capsize=1.2)
    ax.set_xticks(range(len(keys)))
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    s = pd.read_csv(out_root(cfg) / "transfer_uncertainty" / "retrieval_tumours_summary.csv")
    ref = pd.read_csv(out_root(cfg) / "atlas" / "retrieval_titan.csv")
    ref = ref[(ref["retrieval"] == "nc_euclid") & (ref["query_label"] == "(all)")].set_index(["condition", "query_family", "metric"])["value"]
    for r in s.itertuples():   # 点推定は atlas_retrieval の出力そのもの
        assert np.isclose(r.point, ref[(r.condition, r.tumour if r.group == "(all)" else r.group, r.metric)], atol=1e-9)
    conf = pd.read_csv(out_root(cfg) / "atlas" / "glioma_confusion_titan.csv")
    conf = conf[(conf["retrieval"] == "nc_euclid") & (conf["scope"] == "full")]
    sa = s[s["group"] == "(all)"]
    sg = s[(s["tumour"] == "glioma") & (s["group"] != "(all)")]
    n_q = {t: int(sa[(sa["tumour"] == t) & (sa["condition"] == "none")]["n"].iloc[0]) for t, _ in TUMOURS}

    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.5, wspace=0.25, left=0.08, right=0.98, top=0.9, bottom=0.08)
    vals: list = []
    for k, (metric, title) in enumerate((("top1", "Correct diagnosis ranked first"), ("top5", "Correct diagnosis in the top 5"))):
        ax = fig.add_subplot(gs[0, k])
        bars(ax, sa, TUMOURS, metric, "tumour", vals, "ab")
        ax.set_xticklabels([f"{n}\n(n = {n_q[t]})" for t, n in TUMOURS], fontsize=FONT - 2)
        ax.set_ylabel("Fraction of query slides\n(n = in-house slides)", fontsize=FONT - 1.5)
        ax.set_title(title, fontsize=FONT, loc="left")
        panel(ax, "ab"[k], x=-0.14, y=1.02)
    fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in CONDS], loc="upper center", ncol=4, frameon=False, fontsize=FONT - 1.5,
               bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=1.2)

    sub = gs[1, :].subgridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.55)
    ax = fig.add_subplot(sub[0, 0])
    bars(ax, sg, GROUPS, "top1", "group", vals, "c")
    ax.set_xticklabels([n for _, n in GROUPS], fontsize=FONT - 2.5)
    ax.set_ylabel("Ranked first, glioma groups", fontsize=FONT - 1)
    ax.set_title("Glioma groups", fontsize=FONT, loc="left")
    panel(ax, "c", x=-0.3, y=1.04)
    for k, (cond, lab) in enumerate((("none", "No correction"), ("meningioma", "Δ borrowed from meningioma"))):
        ax = fig.add_subplot(sub[0, k + 1])
        c = conf[conf["condition"] == cond]
        t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum").reindex(index=["ast", "glm", "odg"], columns=PRED).fillna(0)
        frac = t.div(t.sum(axis=1), axis=0)
        ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(PRED)):
                ax.text(j, i, str(int(t.iloc[i, j])), ha="center", va="center", fontsize=FONT - 1.5,
                        color="white" if frac.iloc[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(PRED)))
        ax.set_xticklabels([PRED_LABEL[p] for p in PRED], fontsize=FONT - 3)
        ax.set_yticks(range(3))
        ax.set_yticklabels([PRED_LABEL[g] for g in ["ast", "glm", "odg"]], fontsize=FONT - 2.5)
        ax.set_xlabel("Group of the first-ranked diagnosis", fontsize=FONT - 2.5)
        ba = balanced_acc(c)
        ax.set_title(f"{lab}\nbalanced accuracy {ba:.3f}", fontsize=FONT - 1.5, loc="left")
        if k == 0:
            panel(ax, "d", x=-0.32, y=1.04)
        vals.append({"panel": "d", "metric": "balanced_accuracy_3", "condition": cond, "group": "glioma", "value": ba})
    save(fig, cfg, NAME)
    v = pd.DataFrame(vals)
    v.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    print(v[v["panel"] == "ab"].pivot_table(index=["metric", "condition"], columns="group", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
