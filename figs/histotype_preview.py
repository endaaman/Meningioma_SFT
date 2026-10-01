"""histotype_preview.py — 組織型（主要 7 型）の施設間分類のプレビュー（本番の図表には入れない、ken 判断待ち）。

入力（lp.py --task histotype → lp_bootstrap.py --task histotype の出力、output_dir/lp_histotype/）:
    trained_by_{src}/{variant}/predictions.csv, bootstrap.csv, per_class.csv, mcnemar.csv
出力（番号体系の外。figs.all では作らない）:
    fig/_preview_histotype_cm.png    — 行で正規化した混同行列（2 方向 × 補正なし / GAN / centroid / ComBat）
    fig/_preview_histotype_ci.png    — balanced accuracy・macro F1 の点推定と 95% CI（患者単位 bootstrap）
    _preview_histotype_table.md      — 指標・クラスごとの recall・混同の要点・McNemar

Usage:
    uv run python -m figs.histotype_preview
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from figs.common import CONDITION_COLORS, FONT, SITE_DISPLAY, VARIANT_DISPLAY, config, fig_dir, out_root, paper_dir
from utils.display import order_conditions

VARIANTS = ["original", "gan", "centroid", "combat"]


def _root(cfg: dict):
    return out_root(cfg) / cfg["lp_histotype"].get("output_subdir", "lp_histotype")


def _classes(cfg: dict) -> list[str]:
    return [c.replace(" meningioma", "") for c in cfg["lp_histotype"]["classes"]]


def _directions(cfg: dict) -> list[tuple[str, str]]:
    sources = list(cfg["embedding"].keys())
    return [(s, next(t for t in sources if t != s)) for s in sources]


def _variants(cfg: dict) -> list[str]:
    return [v for v in order_conditions(VARIANTS) if v in cfg["lp_histotype"].get("variants", VARIANTS)]


def confusion(cfg: dict, tr: str, v: str) -> np.ndarray:
    k = len(_classes(cfg))
    df = pd.read_csv(_root(cfg) / f"trained_by_{tr}" / v / "predictions.csv")
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(df["true"], df["pred"]):
        cm[t, p] += 1
    return cm


def cm_figure(cfg: dict) -> plt.Figure:
    classes, variants, directions = _classes(cfg), _variants(cfg), _directions(cfg)
    k = len(classes)
    fig, axes = plt.subplots(len(directions), len(variants), figsize=(2.3 * len(variants), 2.9 * len(directions)))
    for r, (tr, te) in enumerate(directions):
        for c, v in enumerate(variants):
            ax = axes[r, c]
            cm = confusion(cfg, tr, v)
            norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for i in range(k):
                for j in range(k):
                    if cm[i, j]:
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=FONT - 2.5,
                                color="white" if norm[i, j] > 0.5 else "black")
            ax.set_xticks(range(k)); ax.set_yticks(range(k))
            ax.set_xticklabels([s[:4] for s in classes], rotation=90, fontsize=FONT - 2)
            ax.set_yticklabels([s[:4] for s in classes] if c == 0 else [], fontsize=FONT - 2)
            title = VARIANT_DISPLAY[v] + (" (ref.)" if v == "combat" else "")
            ax.set_title(title if r == 0 else "", fontsize=FONT, color=CONDITION_COLORS[v] if v != "original" else "black")
            if c == 0:
                ax.set_ylabel(f"{SITE_DISPLAY[tr]}→{SITE_DISPLAY[te]}\nTrue", fontsize=FONT - 1)
            if r == len(directions) - 1:
                ax.set_xlabel("Predicted", fontsize=FONT - 1)
    fig.suptitle("Histotype (7 classes), row-normalized; numbers = slides", fontsize=FONT + 1)
    fig.tight_layout()
    return fig


def ci_figure(cfg: dict) -> plt.Figure:
    bs = pd.read_csv(_root(cfg) / "bootstrap.csv")
    variants, directions = _variants(cfg), _directions(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, metric, xl in zip(axes, ("balanced_accuracy", "f1_macro"), ("Balanced accuracy", "Macro F1")):
        y, ticks, labels = 0, [], []
        for tr, te in directions:
            for v in variants:
                r = bs[(bs.train_source == tr) & (bs.variant == v) & (bs.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r["value"], -y, xerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                            fmt="o", color=CONDITION_COLORS[v], mec="black", mew=0.5, capsize=2)
                ticks.append(-y); labels.append(f"{SITE_DISPLAY[tr]}→{SITE_DISPLAY[te]}  {VARIANT_DISPLAY[v]}")
                y += 1
            y += 0.6
        ax.axvline(1 / len(_classes(cfg)), color="#888888", ls="--", lw=0.8)
        ax.set_xlabel(xl); ax.set_xlim(0, 1); ax.grid(axis="x", color="#E6E6E6")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].text(1 / len(_classes(cfg)), 0.6, " chance (1/7)", fontsize=FONT - 2, color="#666666", va="bottom")
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels, fontsize=FONT - 1)
    fig.tight_layout()
    return fig


def _confusion_notes(cfg: dict) -> list[str]:
    """ken の観点: Transitional の誤分類先、Angiomatous ↔ Microcystic の混同。"""
    classes = _classes(cfg)
    ix = {c: i for i, c in enumerate(classes)}
    lines = []
    for tr, te in _directions(cfg):
        for v in _variants(cfg):
            cm = confusion(cfg, tr, v)
            t = cm[ix["Transitional"]]
            am, ma = cm[ix["Angiomatous"], ix["Microcystic"]], cm[ix["Microcystic"], ix["Angiomatous"]]
            lines.append(
                f"| {SITE_DISPLAY[tr]}→{SITE_DISPLAY[te]} | {VARIANT_DISPLAY[v]} | "
                f"{t[ix['Transitional']]}/{t.sum()} | {t[ix['Fibrous']]} | {t[ix['Meningothelial']]} | "
                f"{am}/{cm[ix['Angiomatous']].sum()} | {ma}/{cm[ix['Microcystic']].sum()} |")
    return lines


def table_md(cfg: dict) -> str:
    root = _root(cfg)
    bs = pd.read_csv(root / "bootstrap.csv")
    pc = pd.read_csv(root / "per_class.csv")
    mc = pd.read_csv(root / "mcnemar.csv")
    classes, variants = _classes(cfg), _variants(cfg)

    def cell(r: pd.Series) -> str:
        return f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"

    out = ["# 組織型の施設間分類（プレビュー、ken 判断待ち）", "",
           "対象 7 クラス: " + ", ".join(classes) + "。SFT・Atypical / Anaplastic（grade を規定）・希少型・NOS は除外。",
           "点推定 [95% CI]（テスト側施設の患者単位 bootstrap 2000 回）。AUC は one-vs-rest のクラス平均（sklearn "
           "`roc_auc_score(multi_class=\"ovr\", average=\"macro\")`）。偶然の balanced accuracy は 1/7 ≈ 0.143。", "",
           "## 指標", "",
           "| 学習 → 評価 | 補正 | Balanced acc. | Macro F1 | AUC (OvR macro) | Accuracy |",
           "|---|---|---|---|---|---|"]
    for tr, te in _directions(cfg):
        for v in variants:
            sub = bs[(bs.train_source == tr) & (bs.variant == v)].set_index("metric")
            if sub.empty:
                continue
            out.append(f"| {SITE_DISPLAY[tr]} → {SITE_DISPLAY[te]} | {VARIANT_DISPLAY[v]} | "
                       f"{cell(sub.loc['balanced_accuracy'])} | {cell(sub.loc['f1_macro'])} | "
                       f"{cell(sub.loc['auc_ovr_macro'])} | {cell(sub.loc['accuracy'])} |")
    out += ["", "## クラスごとの recall（点推定）", ""]
    for tr, te in _directions(cfg):
        out += [f"### {SITE_DISPLAY[tr]} → {SITE_DISPLAY[te]}", "",
                "| クラス (n) | " + " | ".join(VARIANT_DISPLAY[v] for v in variants) + " |",
                "|---|" + "---|" * len(variants)]
        for c in classes:
            rows = pc[(pc.train_source == tr) & (pc["class"] == c)].set_index("variant")
            n = int(rows["n"].iloc[0])
            out.append(f"| {c} ({n}) | " + " | ".join(f"{rows.loc[v, 'recall']:.2f}" for v in variants) + " |")
        out.append("")
    out += ["## 混同の要点", "",
            "| 学習 → 評価 | 補正 | Transitional 正解 | Tran→Fibrous | Tran→Meningothelial | Angi→Micr | Micr→Angi |",
            "|---|---|---|---|---|---|---|"] + _confusion_notes(cfg)
    out += ["", "## McNemar（正解 / 不正解の対応ありの比較、exact）", "",
            "| 学習 → 評価 | a | b | a のみ正解 | b のみ正解 | p |", "|---|---|---|---|---|---|"]
    for _, r in mc.iterrows():
        out.append(f"| {SITE_DISPLAY[r.train_source]} → {SITE_DISPLAY[r.test_source]} | {VARIANT_DISPLAY[r.variant_a]} | "
                   f"{VARIANT_DISPLAY[r.variant_b]} | {r.a_correct_only} | {r.b_correct_only} | {r.p_exact:.4f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    cfg = config()
    for name, fig in (("cm", cm_figure(cfg)), ("ci", ci_figure(cfg))):
        path = fig_dir(cfg) / f"_preview_histotype_{name}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
        print(f"  saved: {path}")
    md = paper_dir(cfg) / "_preview_histotype_table.md"
    md.write_text(table_md(cfg))
    print(f"  saved: {md}")


if __name__ == "__main__":
    main()
