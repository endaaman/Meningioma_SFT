"""dendrogram.py — TITAN スライド埋め込みの群間比較樹形図（補正前後）。

論文 Fig 5 と同じ形式として、施設 × サブタイプ平均ベクトル間の **生の** ユークリッド距離
（正規化しない）で Ward 法クラスタリングした clustered heatmap と、施設間で同じサブタイプが
対になるかの定量（pairs.csv）も出す。生の距離にするのは:
  - centroid は施設ごとの平行移動なので、施設内の群間ユークリッド距離は補正で変わらない
  - 単位が Fig 3 の |shift|・SFT − 髄膜腫 の大きさと揃う

出力 (output_dir/dendrogram/):
    {variant}/*_dendrogram_*.png            — 既存（正規化した各 metric）
    {variant}/euc_mean_raw_clustered_cross.png — 生のユークリッド距離の clustered heatmap
    pairs.csv                               — variant ごとのペアの定量

Usage:
    uv run python dendrogram.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.gridspec import GridSpec, SubplotSpec
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity as cos_sim, euclidean_distances

from utils.display import make_abbrev, ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data


def save_legend(abbrev: dict[str, str], path: Path, title: str) -> None:
    rows = list(abbrev.items())
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, max(2.0, n * 0.38 + 1.2)))
    ax.axis("off")
    tbl = ax.table(cellText=[[ab, full] for full, ab in rows],
                   colLabels=["Abbrev", "Full name"], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.auto_set_column_width([0, 1])
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e0e0e0")
    ax.set_title(title, fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {path}")


# ── matrix ────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return (2 * euclidean_distances(X, Y).mean()
            - euclidean_distances(X, X).mean()
            - euclidean_distances(Y, Y).mean())


METRICS = [
    ("cos_mean", "Cosine Similarity between Mean Vectors", True,
     lambda X, Y: cos_sim(X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True))[0, 0]),
    ("euc_mean", "Euclidean Distance between Mean Vectors", False,
     lambda X, Y: np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))),
    ("cos_all",  "Average Cosine Similarity of All Pairs", True,
     lambda X, Y: cos_sim(X, Y).mean()),
    ("euc_all",  "Average Euclidean Distance of All Pairs", False,
     lambda X, Y: euclidean_distances(X, Y).mean()),
    ("energy",   "Energy Distance", False, energy_distance),
]


def build_matrix(groups: dict, compute_fn, is_similarity: bool) -> pd.DataFrame:
    labels = list(groups.keys())
    n = len(labels)
    mat = pd.DataFrame(np.full((n, n), 1.0 if is_similarity else 0.0),
                       index=labels, columns=labels)
    for l1, l2 in combinations(labels, 2):
        val = float(compute_fn(groups[l1], groups[l2]))
        mat.loc[l1, l2] = val; mat.loc[l2, l1] = val
    return mat


# ── 生のユークリッド距離（論文 Fig 5） ────────────────────────────────────────

def group_means(merged: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """両施設にあるサブタイプについて、施設 × サブタイプの平均ベクトルと例数を返す。

    行の並びはサブタイプ順（config の表示順）× 施設。例数の少ない群も除外しない。
    """
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]),
        cfg)
    rows = []
    for sub in common:
        for src in sources:
            X = np.stack(merged.loc[(merged["source"] == src) & (merged["subtype"] == sub),
                                    "embedding"].values).astype(np.float64)
            rows.append({"label": f"{src}__{sub}", "source": src, "subtype": sub,
                         "n": len(X), "mean": X.mean(axis=0)})
    return pd.DataFrame(rows)


def raw_distance(groups: pd.DataFrame) -> np.ndarray:
    """平均ベクトル間の生のユークリッド距離行列。"""
    return squareform(pdist(np.stack(groups["mean"].values), metric="euclidean"))


def ward_linkage(D: np.ndarray) -> np.ndarray:
    """既存の樹形図と同じ Ward 法（距離は生のユークリッド距離なので Ward の前提を満たす）。"""
    return linkage(squareform(D, checks=False), method="ward")


def sister_pairs(Z: np.ndarray, groups: pd.DataFrame) -> list[tuple[int, int]]:
    """樹形図で互いに最初に併合される葉の組のうち、同じサブタイプ・別施設のもの。"""
    n = len(groups)
    subs, srcs = groups["subtype"].values, groups["source"].values
    pairs = []
    for a, b, _, _ in Z:
        a, b = int(a), int(b)
        if a < n and b < n and subs[a] == subs[b] and srcs[a] != srcs[b]:
            pairs.append((a, b))
    return pairs


def pair_stats(D: np.ndarray, Z: np.ndarray, groups: pd.DataFrame) -> dict:
    """施設間で同じサブタイプが対になるかの定量。

    sister_pairs        : 樹形図で姉妹（最初に併合）になった同サブタイプの施設ペア数
    nn_other_site_same  : 各群について、もう一方の施設で最も近い群が同じサブタイプである数
    nn_any_same_other   : 各群について、全群（自分以外）で最も近い群が「同じサブタイプ・別施設」である数
    """
    n = len(groups)
    subs, srcs = groups["subtype"].values, groups["source"].values
    nn_other = nn_any = 0
    for i in range(n):
        other = [j for j in range(n) if srcs[j] != srcs[i]]
        j = min(other, key=lambda j: D[i, j])
        nn_other += int(subs[j] == subs[i])
        k = min((j for j in range(n) if j != i), key=lambda j: D[i, j])
        nn_any += int(subs[k] == subs[i] and srcs[k] != srcs[i])
    return {"n_subtypes": groups["subtype"].nunique(), "n_groups": n,
            "sister_pairs": len(sister_pairs(Z, groups)),
            "nn_other_site_same": nn_other, "nn_any_same_other": nn_any}


def draw_clustered(fig: plt.Figure, spec: SubplotSpec, groups: pd.DataFrame, D: np.ndarray,
                   Z: np.ndarray, cfg: dict, fontsize: float = 8) -> plt.Axes:
    """樹形図（上）＋ 葉順に並べた距離ヒートマップ（下）を spec に描き、ヒートマップの ax を返す。

    行ラベルは「略称 (n)」＋ 施設マーカー（色 = サブタイプ）。姉妹になった同サブタイプのペアは
    樹形図の葉の下に同色の帯で示す。
    """
    sub_map = subtype_color_map(ordered_subtypes(set(groups["subtype"]), cfg), cfg)
    src_mkr = cfg.get("display", {}).get("markers", {}).get("sources", {})
    gs = spec.subgridspec(2, 2, height_ratios=[0.2, 1], width_ratios=[1, 0.035],
                          hspace=0.015, wspace=0.03)
    ax_d, ax_h, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    n = len(groups)

    dn = dendrogram(Z, ax=ax_d, no_labels=True, color_threshold=0,
                    above_threshold_color="#555555")
    order = dn["leaves"]
    ax_d.set_xlim(0, 10 * n)
    ax_d.axis("off")
    pos = {leaf: i for i, leaf in enumerate(order)}
    for a, b in sister_pairs(Z, groups):
        i0, i1 = sorted((pos[a], pos[b]))
        ax_d.plot([10 * i0 + 2, 10 * i1 + 8], [0, 0], color=sub_map[groups["subtype"].iloc[a]],
                  lw=3, solid_capstyle="butt", clip_on=False)

    M = D[np.ix_(order, order)]
    im = ax_h.imshow(M, cmap="viridis_r", aspect="auto", extent=(0, n, n, 0),
                     interpolation="nearest")
    ax_h.set_xlim(0, n); ax_h.set_ylim(n, 0)
    ticks = np.arange(n) + 0.5
    names = [f"{shorten(groups['subtype'].iloc[i], cfg)} ({groups['n'].iloc[i]})" for i in order]
    ax_h.set_yticks(ticks); ax_h.set_yticklabels(names, fontsize=fontsize - 1)
    ax_h.set_xticks(ticks); ax_h.set_xticklabels(names, fontsize=fontsize - 1, rotation=90)
    ax_h.tick_params(length=0, pad=9)
    # 施設マーカー（ラベルと軸の間）
    for k, i in enumerate(order):
        mk = src_mkr.get(groups["source"].iloc[i], "o")
        c = sub_map[groups["subtype"].iloc[i]]
        ax_h.plot([-0.012], [1 - (k + 0.5) / n], marker=mk, color=c, markersize=3.6,
                  transform=ax_h.transAxes, clip_on=False, ls="none")
        ax_h.plot([(k + 0.5) / n], [-0.012], marker=mk, color=c, markersize=3.6,
                  transform=ax_h.transAxes, clip_on=False, ls="none")
    for sp in ax_h.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, cax=ax_c)
    cb.set_label("Euclidean distance between mean embeddings", fontsize=fontsize - 1)
    cb.ax.tick_params(labelsize=fontsize - 1)
    return ax_h


def save_clustered(groups: pd.DataFrame, D: np.ndarray, Z: np.ndarray, cfg: dict,
                   path: Path, title: str | None = None) -> None:
    fig = plt.figure(figsize=(6.4, 6.6))
    gs = GridSpec(1, 1, figure=fig)
    ax_h = draw_clustered(fig, gs[0, 0], groups, D, Z, cfg)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    src_mkr = cfg.get("display", {}).get("markers", {}).get("sources", {})
    handles = [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=s)
               for s, m in src_mkr.items()]
    ax_h.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.12, 1.24), frameon=False,
                fontsize=7, title="(n) = slides", title_fontsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def run_raw(merged: pd.DataFrame, out_dir: Path, cfg: dict, variant: str) -> dict:
    """生のユークリッド距離の clustered heatmap を保存し、ペアの定量を返す。"""
    groups = group_means(merged, cfg)
    D = raw_distance(groups)
    Z = ward_linkage(D)
    save_clustered(groups, D, Z, cfg, out_dir / "euc_mean_raw_clustered_cross.png",
                   title=f"{variant}: Euclidean distance between mean embeddings (Ward)")
    stats = {"variant": variant, **pair_stats(D, Z, groups)}
    print("  pairs: " + "  ".join(f"{k}={v}" for k, v in stats.items() if k != "variant"))
    return stats


# ── plot ──────────────────────────────────────────────────────────────────────

def save_dendrogram(mat: pd.DataFrame, title: str, path: Path,
                    is_similarity: bool = False,
                    marker_specs: dict | None = None) -> None:
    """
    marker_specs: {label: {"color": hex, "marker": "o"/"s",
                            "subtype": full_name, "source": src, "short": abbrev}}
    When provided, draws subtype-colored source-shaped markers below each leaf
    and Rectangle frames around adjacent same-subtype pairs.
    """
    norm_v = _normalize(mat.values.astype(float))
    dist_v = (1 - norm_v) if is_similarity else norm_v
    np.fill_diagonal(dist_v, 0)
    Z = linkage(squareform(dist_v, checks=False), method="ward")
    n = len(mat)
    fig, ax = plt.subplots(figsize=(24, 6))
    result = dendrogram(
        Z, labels=mat.index.tolist(), ax=ax,
        no_labels=True, 
        leaf_rotation=0, leaf_font_size=15,
        color_threshold=0, above_threshold_color="gray",
    )
    # ax.set_title(title, fontsize=30)
    # ax.set_xlabel("subtype", fontsize=24)
    # ax.set_ylabel("distance", fontsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="y", labelsize=16)

    if not marker_specs:
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved: {path}")
        return

    ivl = result["ivl"]
    n_leaves = len(ivl)
    # scipy dendrogram places leaf i at x = 5 + 10*i
    leaf_xs = [5 + 10 * i for i in range(n_leaves)]

    _, ymax = ax.get_ylim()
    band     = ymax * 0.12        # マーカー帯の高さ（y単位）
    marker_y = -band * 0.55       # 帯の中央
    pad_x    = 2.0                # x単位（葉間隔 = 10）
    pad_y    = band * 0.12        # y単位

    # ── pair frames ───────────────────────────────────────────────────────────
    for i in range(n_leaves - 1):
        s1 = marker_specs.get(ivl[i], {}).get("subtype")
        s2 = marker_specs.get(ivl[i + 1], {}).get("subtype")
        if s1 and s1 == s2:
            color = marker_specs[ivl[i]]["color"]
            xi, xj = leaf_xs[i], leaf_xs[i + 1]
            x0 = xi - 5 + pad_x
            w  = (xj + 5 - pad_x) - x0
            y0 = marker_y - band * 0.40
            h  = (0.0 - pad_y) - y0
            if w > 0 and h > 0:
                ax.add_patch(Rectangle(
                    (x0, y0), w, h,
                    linewidth=2.0, edgecolor=color, facecolor="none",
                    linestyle="--", clip_on=False, zorder=4,
                ))

    # ── subtype-colored, source-shaped markers ─────────────────────────────────
    for i, lbl in enumerate(ivl):
        spec = marker_specs.get(lbl, {})
        ax.scatter([leaf_xs[i]], [marker_y],
                   color=spec.get("color", "gray"),
                   marker=spec.get("marker", "o"),
                   s=250, zorder=6, clip_on=False)

    ax.set_ylim(marker_y - ymax * 0.05, ymax)

    # ── legend ─────────────────────────────────────────────────────────────────
    # seen_sources: dict[str, str] = {}
    # for spec in marker_specs.values():
    #     src = spec.get("source", "")
    #     if src and src not in seen_sources:
    #         seen_sources[src] = spec.get("marker", "o")
    # handles: list = [
    #     Line2D([0], [0], marker=mk, linestyle="none",
    #            markerfacecolor="dimgray", markeredgecolor="dimgray",
    #            markersize=18, label=src)
    #     for src, mk in seen_sources.items()
    # ]
    # seen_subtypes: dict[str, tuple[str, str]] = {}
    # for lbl in ivl:
    #     spec = marker_specs.get(lbl, {})
    #     sub = spec.get("subtype", "")
    #     if sub and sub not in seen_subtypes:
    #         seen_subtypes[sub] = (spec.get("color", "gray"), spec.get("short", sub[:4]))
    # for _sub, (color, short) in seen_subtypes.items():
    #     handles.append(Line2D([0], [0], marker="o", linestyle="none",
    #                            markerfacecolor=color, markeredgecolor=color,
    #                            markersize=18, label=short))
    # handles.append(Patch(fill=False, linestyle="--", edgecolor="gray",
    #                      linewidth=2.5, label="adj. pair"))
    # ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    #           fontsize=17, title="Legend", title_fontsize=19,
    #           framealpha=0.9, ncol=1,
    #           handletextpad=1.2, labelspacing=1.0, borderpad=1.2)

# ── save ───────────────────────────────────────────────────────────────────────

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ── run ───────────────────────────────────────────────────────────────────────

def _run_metrics(vecs: dict, scope: str, out_dir: Path, cfg: dict,
                 marker_specs: dict | None = None) -> None:
    for slug, title, is_sim, fn in METRICS:
        mat = build_matrix(vecs, fn, is_sim)
        save_dendrogram(mat, f"{title} [{scope}]",
                        out_dir / f"{slug}_dendrogram_{scope}.png",
                        is_sim, marker_specs)
    if cfg.get("comparison", {}).get("mmd", False):
        from hyppo.ksample import MMD
        mmd_kernels = cfg.get("comparison", {}).get("mmd_kernels", [])
        for kernel in mmd_kernels:
            try:
                mat = build_matrix(
                    vecs, lambda X, Y, k=kernel: MMD(compute_kernel=k).statistic(X, Y), False)
                save_dendrogram(mat, f"MMD (kernel={kernel}) [{scope}]",
                                out_dir / f"mmd_{kernel}_dendrogram_{scope}.png",
                                False, marker_specs)
            except Exception as e:
                print(f"  MMD kernel={kernel} skipped: {e}")


def run(merged: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shortened: dict = cfg.get("display", {}).get("shortened", {}).get("subtypes", {})
    subtype_colors: dict = cfg.get("display", {}).get("colors", {}).get("subtypes", {})
    _src_syms = cfg.get("display", {}).get("markers", {}).get("sources", {})
    src_marker = dict(_src_syms)

    # per-source (no cross-site distinction needed)
    for source, grp in merged.groupby("source"):
        subtypes = ordered_subtypes(set(grp["subtype"].unique()), cfg)
        abbrev = make_abbrev(subtypes, cfg)
        save_legend(abbrev, out_dir / f"legend_{source}.png", f"Legend [{source}]")
        vecs = {abbrev[s]: np.stack(grp.loc[grp["subtype"] == s, "embedding"].values)
                for s in subtypes}
        print(f"  [{source}] {len(subtypes)} subtypes")
        _run_metrics(vecs, str(source), out_dir, cfg)

    # cross-source: color = subtype, shape = source
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]),
        cfg)
    if len(common) < 2:
        return
    raw_labels = [f"{src}__{sub}" for sub in common for src in sources]
    abbrev = make_abbrev(raw_labels, cfg)
    save_legend(abbrev, out_dir / "legend_cross.png", "Legend [cross]")
    vecs = {
        abbrev[f"{src}__{sub}"]: np.stack(
            merged.loc[(merged["source"] == src) & (merged["subtype"] == sub), "embedding"].values)
        for sub in common for src in sources
    }
    marker_specs = {
        abbrev[f"{src}__{sub}"]: {
            "color":   subtype_colors.get(sub, "gray"),
            "marker":  src_marker.get(src, "o"),
            "subtype": sub,
            "source":  src,
            "short":   shortened.get(sub, sub[:4].capitalize()),
        }
        for sub in common for src in sources
    }
    print(f"  [cross] {len(vecs)} groups")
    _run_metrics(vecs, "cross", out_dir, cfg, marker_specs)


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["output_dir"]) / "dendrogram"
    variants: dict[str, str] = cfg.get("variants", {"original": "original"})

    pair_rows = []
    for variant, dir_key in variants.items():
        print(f"\n[{variant}]")
        merged = load_data(cfg, dir_key)
        if merged.empty:
            print("  SKIP: no data")
            continue
        print(f"  {len(merged)} cases")
        run(merged, out_root / variant, cfg)
        pair_rows.append(run_raw(merged, out_root / variant, cfg, variant))
    if pair_rows:
        path = out_root / "pairs.csv"
        pd.DataFrame(pair_rows).to_csv(path, index=False)
        print(f"\n  saved: {path}")


if __name__ == "__main__":
    main()
