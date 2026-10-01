"""histotype_compare.py — 組織型の施設間分類を、補正条件 × クラス重みの有無で並べた summary.md を書く。

入力（lp.py --task histotype --output-subdir <root>/{unweighted,weighted} と
      lp_bootstrap.py --task histotype --histotype-subdir 同 の出力）:
    {output_dir}/<root>/{unweighted,weighted}/bootstrap.csv, per_class.csv, mcnemar.csv,
    trained_by_*/<variant>/predictions.csv
出力:
    {output_dir}/<root>/summary.md

Usage:
    uv run python histotype_compare.py --set main --root lp_histotype_affine
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from histotype_labels import add_set_arg, apply_set
from utils.loader import load_config

WEIGHTS = ["unweighted", "weighted"]
METRICS = [("balanced_accuracy", "BA"), ("f1_macro", "macro F1"), ("auc_ovr_macro", "AUC")]
NOTE = {"affine_oracle": "（上限の参考：s を組織型ラベルで推定＝循環）"}


def _fmt(r: pd.Series) -> str:
    return f"{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="lp_histotype_affine")
    parser.add_argument("--variants", nargs="+",
                        default=["original", "gan", "centroid", "combat", "affine_free", "affine_oracle"])
    add_set_arg(parser)
    args = parser.parse_args()
    cfg = apply_set(load_config(), args.set)
    classes = [c.replace(" meningioma", "") for c in cfg["lp_histotype"]["classes"]]
    k = len(classes)
    root = Path(cfg["output_dir"]) / args.root
    sources = list(cfg["embedding"].keys())

    md = [f"# 組織型の施設間分類（{k} クラス: {', '.join(classes)}）— 補正条件 × クラス重み", "",
          "値は点推定 [95% CI]（患者単位 bootstrap）。AUC は one-vs-rest のクラス平均（閾値に依存しない）。",
          "affine_free: patho2 を (x − μ_p)·s + μ_e、s はラベル不要。affine_oracle: s を組織型内 RMS 比で推定"
          "（組織型ラベル＝評価施設のものを含む、を使うので組織型分類の評価としては循環。上限の参考）。", ""]
    scale_path = Path(cfg["output_dir"]) / "affine" / "scale.csv"
    if scale_path.exists():
        sc = pd.read_csv(scale_path)
        md += ["倍率 s: " + "、".join(f"{r.variant} = {r.scale:.4f}" for r in sc.itertuples()), ""]

    bs = {w: pd.read_csv(root / w / "bootstrap.csv") for w in WEIGHTS}
    for train_src in sources:
        test_src = next(s for s in sources if s != train_src)
        md += [f"## {train_src} → {test_src}", "",
               "| 条件 | " + " | ".join(f"{lab}（重みなし）| {lab}（重み付き）" for _, lab in METRICS) + " |",
               "|---|" + "---|" * (2 * len(METRICS))]
        for v in args.variants:
            cells = []
            for m, _ in METRICS:
                for w in WEIGHTS:
                    d = bs[w]
                    r = d[(d.train_source == train_src) & (d.variant == v) & (d.metric == m)]
                    cells.append(_fmt(r.iloc[0]) if len(r) else "—")
            md.append(f"| {v}{NOTE.get(v, '')} | " + " | ".join(cells) + " |")
        md.append("")

        for w in WEIGHTS:
            pc = pd.read_csv(root / w / "per_class.csv")
            pc = pc[pc.train_source == train_src]
            md += [f"**クラスごとの recall（{w}）**", "",
                   "| 条件 | " + " | ".join(f"{c} (n={int(pc[pc['class'] == c].n.iloc[0])})" for c in classes) + " |",
                   "|---|" + "---|" * k]
            for v in args.variants:
                row = pc[pc.variant == v].set_index("class")
                md.append(f"| {v} | " + " | ".join(f"{row.loc[c, 'recall']:.2f}" if c in row.index else "—"
                                                   for c in classes) + " |")
            md.append("")

            md += [f"**混同行列（{w}、行 = 正解、列 = 予測、スライド数）**", ""]
            for v in args.variants:
                path = root / w / f"trained_by_{train_src}" / v / "predictions.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                cm = np.zeros((k, k), int)
                for t, p in zip(df["true"], df["pred"]):
                    cm[int(t), int(p)] += 1
                md += [f"{v}:", "", "| 正解 \\ 予測 | " + " | ".join(classes) + " |", "|---|" + "---|" * k]
                md += [f"| {classes[i]} | " + " | ".join(str(x) for x in cm[i]) + " |" for i in range(k)]
                md.append("")

            mc = pd.read_csv(root / w / "mcnemar.csv")
            mc = mc[(mc.train_source == train_src) & ((mc.variant_a == "centroid") | (mc.variant_b == "centroid"))]
            md += [f"**McNemar（centroid と他条件、{w}）**", "", "| 比較 | 片方だけ正解（a / b） | p |", "|---|---|---|"]
            md += [f"| {r.variant_a} vs {r.variant_b} | {r.a_correct_only} / {r.b_correct_only} | {r.p_exact:.3g} |"
                   for r in mc.itertuples()]
            md.append("")

    (root / "summary.md").write_text("\n".join(md))
    print(f"  saved: {root / 'summary.md'}")


if __name__ == "__main__":
    main()
