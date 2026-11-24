"""v4 Supplementary の図: ずれの推定に要る枚数（TITAN、平行移動）。語は中立に（どちらが良いとは書かない）。

a 髄膜腫の院内スライドを n 枚だけ使って推定したずれ（EBRAINS は全部）をグリオーマに当てたとき、グリオーマに残る施設差の割合
  （‖残差‖ ÷ 補正なしの ‖残差‖）。横 = n（対数）、線 = 抽出の中央値、帯 = 2.5–97.5% 点。破線 = 院内の全枚数（555）で推定した値、
  点線 = 補正なし（1）
b 枚数の少ない群で、その群自身の院内 n 枚で推定する場合と、よくある腫瘍から借りる場合（借りる元は全枚数）。評価は残りの院内スライド
  （n 枚を除いた分）での 2 施設の中心の距離（施設内の広がり = 1）。6 組: SFT ← 髄膜腫、リンパ腫・転移 ← グリオーマ、グリオーマ 3 群 ←
  髄膜腫。線 = 中央値、帯 = 2.5–97.5% 点（500 回の抽出）。灰 = 補正なし、緑 = 自身の n 枚、青 = 借りる、一点鎖線 = 標本誤差の下限
  （ずれが完全に分かっていても残る距離）。題に 2 本の中央値の線が交わる n（補間）

入力: output_dir/transfer_sample_size/{n_curve_summary.csv, rare_crossover_titan.csv, rare_crossover_nstar.csv}
     （transfer_sample_size.py、transfer_rare_crossover.py）
出力: fig/{fig Sn}_estimation_sample_size.{png,pdf}、同 _values.csv

Usage:
    uv run python -m figs.estimation_sample_size
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figs import label
from figs.common import FONT, config, fig_path, out_root, panel, save
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_LABELS

NAME = "estimation_sample_size"
GROUPS = [("SFT", "meningioma"), ("lym", "glioma"), ("met", "glioma"), ("ast", "meningioma"), ("glm", "meningioma"), ("odg", "meningioma")]
C_NONE, C_OWN, C_BORROW, C_FLOOR = "#a3a29b", "#1baf7a", "#2a78d6", "#333333"


def _ticks(ax: plt.Axes, ns: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=FONT - 2.5)
    ax.minorticks_off()


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    d = out_root(cfg) / "transfer_sample_size"
    vals = []
    fig = plt.figure(figsize=(7.2, 7.4))

    # a
    nc = pd.read_csv(d / "n_curve_summary.csv")
    m = nc[(nc["source"] == "meningioma") & (nc["target"] == "glioma")]
    cur = m[(m["scheme"] == "draw") & (m["variant"] == "patho2")].sort_values("n")
    full = m[m["scheme"] == "full"].iloc[0]
    ax = fig.add_axes([0.09, 0.735, 0.42, 0.22])
    ax.fill_between(cur["n"], cur["residual_rel_q025"], cur["residual_rel_q975"], color=C_BORROW, alpha=0.18, linewidth=0)
    ax.plot(cur["n"], cur["residual_rel_median"], color=C_BORROW, marker="o", ms=3, lw=1.2)
    ax.axhline(full["residual_rel_median"], color=C_BORROW, lw=0.8, ls="--")
    ax.text(cur["n"].max(), full["residual_rel_median"] - 0.02, f"all {int(full['n'])} in-house slides: {full['residual_rel_median']:.2f}",
            ha="right", va="top", fontsize=FONT - 2.5, color=C_BORROW)
    ax.axhline(1.0, color=C_NONE, lw=0.8, ls=":")
    ax.text(cur["n"].min(), 0.985, "no correction", ha="left", va="top", fontsize=FONT - 2.5, color="0.4")
    _ticks(ax, [int(n) for n in cur["n"]])
    ax.set_ylim(0.4, 1.05)
    ax.tick_params(axis="y", labelsize=FONT - 2)
    ax.set_xlabel("In-house meningioma slides used for the shift (n; all EBRAINS slides)", fontsize=FONT - 1.5)
    ax.set_ylabel("Glioma site shift left\n(fraction of no correction)", fontsize=FONT - 1.5)
    ax.set_title("Meningioma shift from n slides, applied to glioma", fontsize=FONT - 1, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "a", x=-0.16, y=1.04)
    vals += [{"panel": "a", "n": int(r.n), "median": r.residual_rel_median, "q025": r.residual_rel_q025, "q975": r.residual_rel_q975}
             for r in cur.itertuples()]
    vals.append({"panel": "a", "n": int(full["n"]), "median": full["residual_rel_median"], "q025": np.nan, "q975": np.nan})

    # b の凡例（右上）
    ax_l = fig.add_axes([0.58, 0.735, 0.40, 0.22])
    ax_l.axis("off")
    ax_l.legend(handles=[Line2D([], [], color=C_NONE, lw=1.4, label="No correction"),
                         Line2D([], [], color=C_OWN, lw=1.4, marker="o", ms=3, label="Shift from the group's own n in-house slides"),
                         Line2D([], [], color=C_BORROW, lw=1.4, marker="o", ms=3, label="Shift borrowed from a common tumour (all slides)"),
                         Line2D([], [], color=C_FLOOR, lw=0.9, ls="-.", label="Sampling floor (shift known exactly)"),
                         Patch(color="0.6", alpha=0.25, label="2.5–97.5% of 500 draws")],
                loc="center left", frameon=False, fontsize=FONT - 2)

    # b
    rc = pd.read_csv(d / "rare_crossover_titan.csv")
    ns = pd.read_csv(d / "rare_crossover_nstar.csv")
    ns = ns[ns["encoder"] == "titan"].set_index(["group", "source"])
    gs = fig.add_gridspec(2, 3, left=0.09, right=0.98, top=0.62, bottom=0.07, hspace=0.62, wspace=0.28)
    for k, (grp, src) in enumerate(GROUPS):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        r = rc[(rc["group"] == grp) & (rc["metric"].isin(["group_gap_sd", "floor_sd"]))]

        def line(cond: str, source: str | None = None) -> pd.DataFrame:
            q = r[r["condition"] == cond]
            if source is not None:
                q = q[q["source"] == source]
            return q.sort_values("n")
        for cond, source, col in (("none", None, C_NONE), ("own", None, C_OWN), ("borrow", src, C_BORROW)):
            q = line(cond, source)
            if cond != "none":
                ax.fill_between(q["n"], q["q025"], q["q975"], color=col, alpha=0.15, linewidth=0)
            ax.plot(q["n"], q["median"], color=col, lw=1.2, marker="o" if cond != "none" else None, ms=2.5)
            vals += [{"panel": "b", "group": grp, "source": source or "", "condition": cond, "n": int(x.n), "median": x.median,
                      "q025": x.q025, "q975": x.q975} for x in q.itertuples()]
        fl = line("floor")
        ax.plot(fl["n"], fl["median"], color=C_FLOOR, lw=0.9, ls="-.")
        vals += [{"panel": "b", "group": grp, "source": "", "condition": "floor", "n": int(x.n), "median": x.median} for x in fl.itertuples()]
        row = ns.loc[(grp, src)]
        nstar = float(row["nstar_obs_interp"])
        n_e, n_p = int(row["n_ebrains"]), int(row["n_patho2"])
        name = {"SFT": "SFT", "lym": FAMILY_LABELS["lymphoma"], "met": FAMILY_LABELS["metastasis"]}.get(
            grp, GLIOMA_GROUP_LABELS.get(grp, grp))
        ax.set_title(f"{name} ← {FAMILY_LABELS[src].lower()}\n(EBRAINS {n_e} / in-house {n_p})\nmedian lines cross at n ≈ {nstar:.1f}",
                     fontsize=FONT - 2.5, loc="left")
        _ticks(ax, sorted(int(n) for n in r["n"].unique()))
        ax.set_ylim(0, max(1.2, float(line("own")["q975"].max()) * 1.05))
        ax.tick_params(axis="y", labelsize=FONT - 2.5)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            panel(ax, "b", x=-0.3, y=1.2)
        if k % 3 == 0:
            ax.set_ylabel("Site-centre distance on the\nremaining in-house slides", fontsize=FONT - 2)
        if k // 3 == 1:
            ax.set_xlabel("n (in-house slides of the group)", fontsize=FONT - 2)
        vals.append({"panel": "b", "group": grp, "source": src, "condition": "n_cross_interp", "median": nstar})
    save(fig, cfg, NAME)
    v = pd.DataFrame(vals)
    v.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    print(v[v["panel"] == "a"].round(3).to_string(index=False))
    print(v[v["condition"] == "n_cross_interp"][["group", "source", "median"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
