"""method_comparison_preview.py — 補正法の比較を図にしたプレビュー（本番の図表には入れない、ken 判断待ち）。

Table S3 の数値を、段差（色だけ → GAN → 埋め込みレベル）と ComBat の行き過ぎが見えるように並べる。
条件の並びは 補正なし → 画像レベル（Macenko → GAN、弱い順）→ 埋め込みレベル（ComBat → 平行移動、提案を最後に）。
本番の並び（CONDITIONS_FULL）とは Macenko の位置と、GAN の前に Macenko を置く点が違う。

上段「施設差をどれだけ落としたか」: 施設平均どうしの距離 / ASW 施設 / iLISI / 隣接ペア
下段「病変の構造を壊していないか」: patho2 内 SFT−髄膜腫 距離（EBRAINS の値が基準）/ ASW 組織型 / cLISI 組織型 / 組織型間距離の施設差

入力: output_dir/method_compare/summary.csv（method_compare.py）、output_dir/harmonization/site_null.csv（chance level）
出力: fig/_preview_method_comparison.png / .pdf

Usage:
    uv run python -m figs.method_comparison_preview
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figs.common import CONDITION_COLORS, CONDITION_LABELS_SHORT, FONT, config, fig_dir, out_root

CONDITIONS = ["original", "macenko", "gan", "combat", "centroid"]
TIERS = [("none", [0]), ("image", [1, 2]), ("embedding", [3, 4])]


def _bar(ax, s: pd.DataFrame, key: str, title: str, fmt: str, ref: float | None = None, ref_label: str = "",
         zero_line: bool = False, scale: float = 1.0, arrow: str = "", ymin: float | None = None) -> None:
    vals = s.loc[CONDITIONS, key].to_numpy(dtype=float) * scale
    x = np.arange(len(CONDITIONS))
    base = 0.0 if ymin is None else ymin
    ax.bar(x, vals - base, bottom=base, color=[CONDITION_COLORS[c] for c in CONDITIONS], width=0.7, edgecolor="none")
    lo, hi = min(base, vals.min()), max(base, vals.max())
    sub = ""
    if ref is not None:
        ax.axhline(ref, color="#555555", lw=0.9, ls="--")
        lo, hi = min(lo, ref), max(hi, ref)
        sub = f"\n(dashed = {ref_label})"
    if zero_line:
        ax.axhline(0, color="#999999", lw=0.6)
    span = (hi - lo) or 1.0
    for xi, v in zip(x, vals):
        up = v >= base
        ax.text(xi, v + (0.03 * span if up else -0.03 * span), fmt.format(v), ha="center",
                va="bottom" if up else "top", fontsize=FONT - 2)
    ax.set_ylim(lo - 0.12 * span if lo < base else lo, hi + 0.22 * span)
    ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS_SHORT[c] for c in CONDITIONS], fontsize=FONT - 2, rotation=35, ha="right",
                       rotation_mode="anchor")
    ax.set_title(title + (f"  {arrow}" if arrow else "") + sub, fontsize=FONT)
    ax.tick_params(axis="y", labelsize=FONT - 2)
    ax.spines[["top", "right"]].set_visible(False)


def _tiers(ax) -> None:
    """x 軸の下に none / image-level / embedding-level の括りを描く。"""
    y = -0.5
    for name, idx in TIERS:
        x0, x1 = idx[0] - 0.35, idx[-1] + 0.35
        ax.plot([x0, x1], [y, y], color="#333333", lw=0.8, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.06, name, ha="center", va="top", fontsize=FONT - 2,
                transform=ax.get_xaxis_transform(), clip_on=False)


def main() -> None:
    cfg = config()
    root = out_root(cfg)
    s = pd.read_csv(root / "method_compare" / "summary.csv").set_index("variant")
    null = pd.read_csv(root / "harmonization" / "site_null.csv")
    null = null[null["null"] == "within_subtype"].groupby("metric")["null_mean"].mean()
    ref, other = cfg["reference"], next(k for k in cfg["embedding"] if k != cfg["reference"])
    n_sub = int(s["n_subtypes"].iloc[0])
    d_ref = float(s[f"sft_men_dist_{ref}"].iloc[0])

    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.6))
    top, bottom = axes[0], axes[1]
    _bar(top[0], s, "site_shift_norm", "Distance between site means", "{:.2f}", arrow="↓")
    _bar(top[1], s, "asw_batch", "ASW (site)", "{:.3f}", ref=float(null["asw_batch"]), ref_label="chance", arrow="↓")
    _bar(top[2], s, "ilisi", "iLISI (site)", "{:.2f}", ref=float(null["ilisi"]), ref_label="chance", arrow="↑")
    _bar(top[3], s, "adjacent_pairs", f"Adjacent same-histotype pairs (/{n_sub})", "{:.0f}", arrow="↑")
    _bar(bottom[0], s, f"sft_men_dist_{other}", f"SFT–meningioma distance within {other}", "{:.2f}",
         ref=d_ref, ref_label=f"{ref.upper()} value", ymin=8.0)
    _bar(bottom[1], s, "asw_bio", "ASW (histotype)", "{:+.3f}", zero_line=True, arrow="↑")
    _bar(bottom[2], s, "clisi", "cLISI (histotype)", "{:.2f}", arrow="↓")
    _bar(bottom[3], s, "subtype_dist_mismatch", "Site mismatch of histotype distances (%)", "{:.1f}", scale=100,
         arrow="↓")
    for ax in bottom:
        _tiers(ax)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, hspace=0.9, wspace=0.4)
    for row, text in ((top, "a  How much of the site difference is removed"),
                      (bottom, "b  Is the lesion structure preserved (over-correction)")):
        y = row[0].get_position().y1
        fig.text(0.005, y + 0.065, text, fontsize=FONT + 1, fontweight="bold", va="bottom")
    out = fig_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"_preview_method_comparison.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out / '_preview_method_comparison.png'} / .pdf")


if __name__ == "__main__":
    main()
