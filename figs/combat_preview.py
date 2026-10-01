"""combat_preview.py — ComBat を含めた 4 条件の比較プレビュー（本番の図表には入れない、ken 判断待ち）。

出力（番号体系の外。figs.all では作らない）:
    fig/_preview_combat_umap.png        — UMAP 4 列（補正なし / GAN / centroid / ComBat）
    fig/_preview_combat_dendrogram.png  — 樹形図 4 段（Fig 6 と同じ描き方）
    fig/_preview_combat_lp.png          — LP の balanced accuracy・SFT 感度の点推定と 95% CI（2 方向）
    _preview_combat_table.md            — 施設指標・組織型指標・ペア数・LP の比較表と、組織型の分離の検定
    _preview_combat_subtype_test.csv    — 組織型の分離の検定（下記）の結果

組織型の分離の検定（centroid vs ComBat）:
    1) スライドごとの対応ありの比較: 各スライドの組織型のシルエット値（silhouette_samples）と
       組織型の LISI を両条件で計算し、差（ComBat − centroid）を患者ごとに平均したうえで、
       符号をランダムに反転する並べ替え検定（10,000 回）で平均差の p 値を出す。
    2) 偶然のレベル: 組織型ラベルを施設内で並べ替えたとき（N_NULL 回）の ASW(subtype) と cLISI の平均。
    埋め込みは harmonization.py と同じく PCA 50 次元に落としてから計算する。

Usage:
    uv run python -m figs.combat_preview
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

import harmonization
from figs import subtype_structure
from figs.common import (CONDITION_COLORS, FONT, SITE_DISPLAY, VARIANT_DISPLAY, config, fig_dir, out_root, paper_dir,
                         src_markers)
from figs.umap_harmonization import _draw_umap
from utils.display import ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_data

VARIANTS = ["original", "gan", "centroid", "combat"]
N_FLIP = 10_000
N_NULL = 200
SEED = 42


# ── 組織型の分離の検定 ────────────────────────────────────────────────────────

def _embed(cfg: dict, variant: str) -> pd.DataFrame:
    m = load_data(cfg, cfg["variants"].get(variant, variant))
    return m.sort_values(["source", "case_id"]).reset_index(drop=True)


def _knn(X: np.ndarray, k: int) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]


def _lisi_from_idx(idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    neigh = inv[idx]
    p = np.stack([(neigh == c).mean(axis=1) for c in range(inv.max() + 1)], axis=1)
    return 1.0 / (p ** 2).sum(axis=1)


def subtype_test(cfg: dict) -> pd.DataFrame:
    k = cfg.get("harmonization", {}).get("lisi_k", 30)
    pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
    pat = {}
    for site, lcfg in cfg["label"].items():
        if lcfg.get("patient"):
            pat.update({f"{site}:{c}": f"{site}:{p}" for c, p in
                        pd.read_csv(lcfg["patient"], dtype=str).set_index("case_id")["patient_id"].items()})
    per_slide, null_rows = {}, []
    rng = np.random.default_rng(SEED)
    for v in ("centroid", "combat"):
        m = _embed(cfg, v)
        X = harmonization._reduce(np.stack(m["embedding"].values).astype(np.float64), pre_n)
        sub, src = m["subtype"].values, m["source"].values
        idx = _knn(X, k)
        per_slide[v] = pd.DataFrame({"key": m["source"] + ":" + m["case_id"],
                                     "sil": silhouette_samples(X, sub), "clisi": _lisi_from_idx(idx, sub)})
        a, c = [], []
        for _ in range(N_NULL):
            perm = sub.copy()
            for s in np.unique(src):
                g = np.where(src == s)[0]
                perm[g] = rng.permutation(perm[g])
            a.append(silhouette_score(X, perm)); c.append(_lisi_from_idx(idx, perm).mean())
        null_rows.append({"state": v, "asw_subtype": per_slide[v]["sil"].mean(), "asw_null_mean": np.mean(a),
                          "asw_null_q975": np.quantile(a, 0.975), "clisi": per_slide[v]["clisi"].mean(),
                          "clisi_null_mean": np.mean(c), "clisi_null_q025": np.quantile(c, 0.025)})
    a, b = per_slide["centroid"], per_slide["combat"]
    assert (a["key"].values == b["key"].values).all()
    diff = pd.DataFrame({"patient": a["key"].map(lambda x: pat.get(x, x)),
                         "sil": b["sil"].values - a["sil"].values, "clisi": b["clisi"].values - a["clisi"].values})
    per_pat = diff.groupby("patient")[["sil", "clisi"]].mean()
    rows = []
    for col, name in (("sil", "asw_subtype"), ("clisi", "clisi")):
        d = per_pat[col].values
        obs = d.mean()
        flips = rng.choice([-1.0, 1.0], size=(N_FLIP, len(d))) @ d / len(d)
        p = (1 + (np.abs(flips) >= abs(obs)).sum()) / (N_FLIP + 1)
        rows.append({"metric": name, "diff_combat_minus_centroid": obs, "n_patients": len(d), "p_signflip": p})
    test = pd.DataFrame(rows)
    return test, pd.DataFrame(null_rows)


# ── 図 ────────────────────────────────────────────────────────────────────────

def umap_figure(cfg: dict) -> plt.Figure:
    root = out_root(cfg)
    coords = {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
              for v in VARIANTS}
    subs = ordered_subtypes(set(coords["original"]["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    src_mkr = src_markers(cfg, sorted(coords["original"]["source"].unique()))
    fig = plt.figure(figsize=(9.6, 3.2))
    gs = GridSpec(2, len(VARIANTS), figure=fig, height_ratios=[1, 0.18], wspace=0.12)
    for i, v in enumerate(VARIANTS):
        ax = fig.add_subplot(gs[0, i])
        _draw_umap(ax, coords[v], sub_map, src_mkr)
        ax.set_title(VARIANT_DISPLAY[v])
        if i:
            ax.set_ylabel("")
    ax_l = fig.add_subplot(gs[1, :]); ax_l.axis("off")
    h = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in subs]
    h += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=s) for s, m in src_mkr.items()]
    ax_l.legend(handles=h, loc="center", ncol=10, frameon=False, fontsize=FONT - 1, handlelength=1.0)
    return fig


def lp_figure(cfg: dict) -> plt.Figure:
    bs = pd.read_csv(out_root(cfg) / "lp" / "bootstrap.csv")
    sources = list(cfg["embedding"].keys())
    directions = [(s, next(t for t in sources if t != s)) for s in sources]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, metric, xl in zip(axes, ("balanced_accuracy", "sensitivity"), ("Balanced accuracy", "Sensitivity (SFT)")):
        y, ticks, labels = 0, [], []
        for tr, te in directions:
            for v in VARIANTS:
                r = bs[(bs.train_source == tr) & (bs.test_source == te) & (bs.variant == v) & (bs.metric == metric)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r["value"], -y, xerr=[[r["value"] - r["ci_low"]], [r["ci_high"] - r["value"]]],
                            fmt="o", color=CONDITION_COLORS[v], mec="black", mew=0.5, capsize=2)
                ticks.append(-y); labels.append(f"{SITE_DISPLAY[tr]}→{SITE_DISPLAY[te]}  {VARIANT_DISPLAY[v]}")
                y += 1
            y += 0.6
        ax.set_xlabel(xl); ax.set_xlim(0.45, 1.02); ax.grid(axis="x", color="#E6E6E6")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(ticks); axes[0].set_yticklabels(labels, fontsize=FONT - 1)
    fig.tight_layout()
    return fig


def _save(fig: plt.Figure, name: str, cfg: dict, dpi: int = 300) -> None:
    path = fig_dir(cfg) / f"_preview_combat_{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {path}")


# ── 表 ────────────────────────────────────────────────────────────────────────

def table_md(cfg: dict, test: pd.DataFrame, null: pd.DataFrame) -> str:
    root = out_root(cfg)
    met = pd.read_csv(root / "harmonization" / "metrics.csv").set_index("state")
    pairs = pd.read_csv(root / "dendrogram" / "pairs.csv").set_index("variant")
    bs = pd.read_csv(root / "lp" / "bootstrap.csv")
    sources = list(cfg["embedding"].keys())
    directions = [(s, next(t for t in sources if t != s)) for s in sources]

    def ci(tr, te, v, metric):
        r = bs[(bs.train_source == tr) & (bs.test_source == te) & (bs.variant == v) & (bs.metric == metric)]
        if r.empty:
            return "—"
        r = r.iloc[0]
        return f"{r['value']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"

    head = ["指標"] + [VARIANT_DISPLAY[v] for v in VARIANTS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = [("ASW 施設 ↓", lambda v: f"{met.loc[v, 'asw_batch']:.4f}"),
            ("iLISI 施設 ↑", lambda v: f"{met.loc[v, 'ilisi']:.3f}"),
            ("ASW SFT vs 髄膜腫 ↑", lambda v: f"{met.loc[v, 'asw_class']:.3f}"),
            ("ASW 組織型 ↑", lambda v: f"{met.loc[v, 'asw_bio']:.4f}"),
            ("cLISI 組織型 ↓", lambda v: f"{met.loc[v, 'clisi']:.3f}"),
            ("樹形図 隣接ペア（/13）", lambda v: f"{int(pairs.loc[v, 'adjacent_pairs'])}"),
            ("樹形図 姉妹ペア（/13）", lambda v: f"{int(pairs.loc[v, 'sister_pairs'])}"),
            ("最近傍が同サブタイプ別施設（/26）", lambda v: f"{int(pairs.loc[v, 'nn_any_same_other'])}")]
    for tr, te in directions:
        d = f"{SITE_DISPLAY[tr]}→{SITE_DISPLAY[te]}"
        rows.append((f"LP BA {d}", lambda v, tr=tr, te=te: ci(tr, te, v, "balanced_accuracy")))
        rows.append((f"LP SFT 感度 {d}", lambda v, tr=tr, te=te: ci(tr, te, v, "sensitivity")))
    for name, fn in rows:
        lines.append("| " + " | ".join([name] + [fn(v) for v in VARIANTS]) + " |")

    out = ["# ComBat を含めた比較（プレビュー、本番図表には未反映）", "",
           "ComBat: neuroCombat、共変量なし、ref_batch = EBRAINS（EBRAINS の値は不変）。", "", *lines, "",
           "## 組織型の分離の検定（centroid vs ComBat）", "",
           "スライドごとの組織型シルエット値と組織型 LISI を両条件で計算し、差（ComBat − centroid）を患者ごとに平均、"
           f"符号反転の並べ替え検定（{N_FLIP:,} 回）で平均差を検定（PCA 50 次元、k = {cfg.get('harmonization', {}).get('lisi_k', 30)}）。", "",
           "| 指標 | 平均差 ComBat − centroid | 患者数 | p（符号反転） |", "|---|---|---|---|"]
    for r in test.itertuples():
        out.append(f"| {r.metric} | {r.diff_combat_minus_centroid:+.4f} | {r.n_patients} | {r.p_signflip:.4f} |")
    out += ["", f"偶然のレベル（組織型ラベルを施設内で並べ替え、{N_NULL} 回）:", "",
            "| 条件 | ASW 組織型 | 偶然の平均 / 97.5% 点 | cLISI | 偶然の平均 / 2.5% 点 |", "|---|---|---|---|---|"]
    for r in null.itertuples():
        out.append(f"| {VARIANT_DISPLAY[r.state]} | {r.asw_subtype:.4f} | {r.asw_null_mean:.4f} / {r.asw_null_q975:.4f} "
                   f"| {r.clisi:.3f} | {r.clisi_null_mean:.3f} / {r.clisi_null_q025:.3f} |")
    return "\n".join(out) + "\n"


def main() -> None:
    print("[preview] combat")
    cfg = config()
    fig_dir(cfg).mkdir(parents=True, exist_ok=True)
    _save(umap_figure(cfg), "umap", cfg)
    fig = subtype_structure.tree_figure(VARIANTS, cfg)
    path = fig_dir(cfg) / "_preview_combat_dendrogram.png"
    fig.savefig(path, dpi=subtype_structure.DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {path}")
    _save(lp_figure(cfg), "lp", cfg)
    test, null = subtype_test(cfg)
    pd.concat([test.assign(kind="signflip"), null.assign(kind="null")]).to_csv(
        paper_dir(cfg) / "_preview_combat_subtype_test.csv", index=False)
    md = table_md(cfg, test, null)
    (paper_dir(cfg) / "_preview_combat_table.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
