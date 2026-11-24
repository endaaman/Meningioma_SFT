"""v4 Supplementary: 腫瘍を食い違わせた対照（Fig 3 の旧 c）。基準は髄膜腫だけのずれ。

5 × 5 の行列（行 = EBRAINS の腫瘍 A、列 = 院内の腫瘍 B）。色 = cos(mean EBRAINS A − mean 院内 B, 髄膜腫だけのずれ)。
対角（同じ腫瘍）は高く、非対角は「腫瘍どうしの差がずれと直交する」幾何の予想（括弧 = ‖Δ‖ / √(‖Δ‖² + ‖b_AB‖²)）どおりに落ちる。
右 = 院内の全スライドに髄膜腫だけのずれを足した後（括弧 = B 自身の残りのずれから予想した値）。

入力: output_dir/shift_transfer/reference_meningioma/mismatch_matrix.csv（shift_transfer.py --reference meningioma）
出力: fig/{fig Sn}_shift_mismatch.{png,pdf}

Usage:
    uv run python -m figs.shift_mismatch
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figs import label
from figs.common import FONT, config, out_root, panel, save
from shift_transfer import FAMILY_ORDER
from utils.display import FAMILY_LABELS

NAME = "shift_mismatch"


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    mm = pd.read_csv(out_root(cfg) / "shift_transfer" / "reference_meningioma" / "mismatch_matrix.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), constrained_layout=True)
    names = [FAMILY_LABELS[f] for f in FAMILY_ORDER]
    for ax, (col, pred, title, letter) in zip(axes, (
            ("cos", "cos_predicted", "Before correction", "a"),
            ("cos_after", "cos_predicted_own_after", "After adding the meningioma shift\nto every in-house slide", "b"))):
        M = mm.pivot(index="ebrains_family", columns="patho2_family", values=col).loc[FAMILY_ORDER, FAMILY_ORDER].to_numpy()
        P = mm.pivot(index="ebrains_family", columns="patho2_family", values=pred).loc[FAMILY_ORDER, FAMILY_ORDER].to_numpy()
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(len(FAMILY_ORDER)):
            for j in range(len(FAMILY_ORDER)):
                txt = ("—" if np.isnan(M[i, j]) else f"{M[i, j]:.2f}") + ("" if np.isnan(P[i, j]) else f"\n({P[i, j]:.2f})")
                ax.text(j, i, txt, ha="center", va="center", fontsize=FONT - 2, color="white" if np.nan_to_num(abs(M[i, j])) > 0.6 else "black")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("In-house tumour")
        ax.set_ylabel("EBRAINS tumour")
        ax.set_title(title, fontsize=FONT, loc="left")
        panel(ax, letter, x=-0.35, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cos with the meningioma site shift")
    save(fig, cfg, NAME)
    off = mm[~mm["matched"]]
    print(f"  diagonal {mm[mm['matched']]['cos'].min():.2f}–{mm[mm['matched']]['cos'].max():.2f}; off-diagonal {off['cos'].min():.2f}–{off['cos'].max():.2f} "
          f"(predicted {off['cos_predicted'].min():.2f}–{off['cos_predicted'].max():.2f}, obs − pred {(off['cos'] - off['cos_predicted']).mean():.2f} ± "
          f"{(off['cos'] - off['cos_predicted']).std(ddof=1):.2f}); after |cos| off-diag max {off['cos_after'].abs().max():.2f}")


if __name__ == "__main__":
    main()
