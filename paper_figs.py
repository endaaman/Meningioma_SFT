"""paper_figs.py — 論文用の図（Fig 2 / 3 / 5）と Table 1 を out/paper/{version}/ に組み立てる。

各パネルは元の解析スクリプトの計算・描画関数を再利用して描き直す（画像の貼り合わせはしない）。
Fig 4 / Table 2（分類性能）は lp 側で作る。

前提（先に流しておくもの）:
    harmonization.py   → out/harmonization/metrics.csv          （Fig 2c）
    shift_direction.py → out/shift_direction/*_original.csv      （Fig 3）
    dataset.py         → out/dataset/table1.{csv,md}             （Table 1）

出力 (output_dir/paper/{paper.version}/):
    fig/fig2.{png,pdf}  — a/b: UMAP（補正前 / centroid 後）, c: 施設差・生物学的分離の指標
    fig/fig3.{png,pdf}  — a: 群ごとのシフトの向き, b: 大きさ, c: シフトと SFT − 髄膜腫 方向
    fig/fig5.{png,pdf}  — a: 施設 × サブタイプ平均の類似度（centroid 後）, b: SFT との近さ
    tables/table1.{csv,md}

Usage:
    uv run python paper_figs.py            # すべて
    uv run python paper_figs.py fig3       # 指定した図だけ
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap as umap_lib
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import confusion_mtx
import harmonization
import sft_distance_bar
import shift_direction
from utils.display import make_abbrev, ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data

FONT = 8
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def _panel(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FONT + 4, fontweight="bold",
            va="bottom", ha="right")


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / name}.png / .pdf")


def _src_markers(cfg: dict, sources: list[str]) -> dict[str, str]:
    syms = cfg.get("display", {}).get("markers", {}).get("sources", {})
    return {s: syms.get(s, "o") for s in sources}


# ── Fig 2 ─────────────────────────────────────────────────────────────────────

def _umap(merged: pd.DataFrame, cfg: dict) -> np.ndarray:
    u = cfg.get("umap", {})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return umap_lib.UMAP(n_components=2, n_neighbors=u.get("n_neighbors", 15),
                             min_dist=u.get("min_dist", 0.01), random_state=u.get("random_state", 42),
                             n_jobs=1).fit_transform(np.stack(merged["embedding"].values))


def _draw_umap(ax: plt.Axes, merged: pd.DataFrame, coords: np.ndarray, cfg: dict,
               sub_map: dict, src_mkr: dict) -> None:
    for src, mkr in src_mkr.items():
        for sub, color in sub_map.items():
            m = ((merged["source"] == src) & (merged["subtype"] == sub)).values
            if m.any():
                ax.scatter(coords[m, 0], coords[m, 1], c=[color], marker=mkr, s=7,
                           alpha=0.75, linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.spines[["top", "right"]].set_visible(False)


def fig2(cfg: dict, out_dir: Path) -> None:
    print("[fig2]")
    orig = load_data(cfg, cfg["variants"].get("original", "original"))
    cent = load_data(cfg, cfg["variants"].get("centroid", "centroid"))
    subs = ordered_subtypes(set(orig["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    src_mkr = _src_markers(cfg, sorted(orig["source"].unique()))
    metrics = pd.read_csv(Path(cfg["output_dir"]) / "harmonization" / "metrics.csv")

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.35, 1], width_ratios=[1, 1, 1, 0.55],
                  hspace=0.42, wspace=0.45)
    # 上段: a / b を 2 等分（凡例は右端の列）
    top = gs[0, 0:3].subgridspec(1, 2, wspace=0.12)
    ax_a, ax_b = fig.add_subplot(top[0, 0]), fig.add_subplot(top[0, 1])
    _draw_umap(ax_a, orig, _umap(orig, cfg), cfg, sub_map, src_mkr)
    _draw_umap(ax_b, cent, _umap(cent, cfg), cfg, sub_map, src_mkr)
    ax_a.set_title("Uncorrected"); ax_b.set_title("Centroid-corrected")
    _panel(ax_a, "a", x=-0.04); _panel(ax_b, "b", x=-0.04)

    ax_leg = fig.add_subplot(gs[0, 3]); ax_leg.axis("off")
    handles = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in subs]
    handles += [Line2D([0], [0], marker=m, color="#666666", ls="none", markersize=5, label=s)
                for s, m in src_mkr.items()]
    ax_leg.legend(handles=handles, loc="center left", frameon=False, handlelength=1.0,
                  handleheight=0.9, borderaxespad=0, fontsize=FONT - 1)

    # 下段: c 指標（施設の混ざり 2 つ + SFT/髄膜腫の分離）
    keys = ["asw_batch", "ilisi", "asw_class"]
    bottom = gs[1, :].subgridspec(1, len(keys), wspace=0.45)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(bottom[0, i])
        harmonization.draw_metric(ax, metrics, key)
        ax.tick_params(axis="x", labelsize=FONT - 1)
        if i == 0:
            _panel(ax, "c", x=-0.22, y=1.22)
    _save(fig, out_dir, "fig2")


# ── Fig 3 ─────────────────────────────────────────────────────────────────────

def fig3(cfg: dict, out_dir: Path) -> None:
    print("[fig3]")
    sd_dir = Path(cfg["output_dir"]) / "shift_direction"
    variant = cfg.get("shift_direction", {}).get("variant", "original")
    table = pd.read_csv(sd_dir / f"shift_direction_{variant}.csv")
    bio = pd.read_csv(sd_dir / f"bio_direction_{variant}.csv")

    fig = plt.figure(figsize=(7.2, 0.24 * len(table) + 1.3))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[3.3, 1.9], wspace=0.12)
    left = outer[0, 0].subgridspec(1, 2, width_ratios=[2.2, 1.2], wspace=0.08)
    ax_a = fig.add_subplot(left[0, 0])
    ax_b = fig.add_subplot(left[0, 1], sharey=ax_a)
    shift_direction.draw_cos(ax_a, table, cfg, fontsize=FONT)
    shift_direction.draw_norm(ax_b, table, cfg, fontsize=FONT)
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_c = fig.add_subplot(outer[0, 1])
    shift_direction.draw_bio(ax_c, bio, cfg, fontsize=FONT)
    _panel(ax_a, "a", x=-0.02, y=1.01); _panel(ax_b, "b", x=0.02, y=1.01)
    # c は等倍軸で縦位置がずれるので、a/b の上端に揃えて図座標で置く
    fig.canvas.draw()
    top = ax_a.get_position().y1
    fig.text(ax_c.get_position().x0, top + 0.01, "c", fontsize=FONT + 4, fontweight="bold", va="bottom")
    _save(fig, out_dir, "fig3")


# ── Fig 5 ─────────────────────────────────────────────────────────────────────

def _cross_matrix(merged: pd.DataFrame, cfg: dict, slug: str) -> tuple[pd.DataFrame, dict, int, list[str]]:
    """confusion_mtx の cross 行列（施設 × 共通サブタイプ）と表示用の正規化値。"""
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]), cfg)
    sub_map = subtype_color_map(common, cfg)
    raw = [f"{src}__{sub}" for sub in common for src in sources]
    abbrev = make_abbrev(raw, cfg)
    vecs = {abbrev[f"{src}__{sub}"]: np.stack(
        merged.loc[(merged["source"] == src) & (merged["subtype"] == sub), "embedding"].values)
        for sub in common for src in sources}
    _, _, is_sim, flip, fn = next(m for m in confusion_mtx.METRICS if m[0] == slug)
    mat = confusion_mtx.build_matrix(vecs, fn, is_sim)
    norm = confusion_mtx._normalize(mat.values.astype(float))
    disp = pd.DataFrame((1 - norm) if flip else norm, index=mat.index, columns=mat.columns)
    src_mkr = _src_markers(cfg, sources)
    specs = {abbrev[f"{src}__{sub}"]: {"color": sub_map[sub], "marker": src_mkr[src]}
             for sub in common for src in sources}
    return disp, specs, len(sources), list(mat.index)


def fig5(cfg: dict, out_dir: Path) -> None:
    print("[fig5]")
    sft_cfg = cfg.get("sft_distance", {})
    variant = sft_cfg.get("variant", "centroid")
    merged = load_data(cfg, cfg["variants"].get(variant, variant))
    disp, specs, gsize, labels = _cross_matrix(merged, cfg, "cos_mean")
    bar_vals, bar_colors = sft_distance_bar.compute_sft_bar_values(merged, cfg, sft_cfg.get("metric", "euc_mean"))

    fig = plt.figure(figsize=(7.2, 4.9))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.35, 1], wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    n = len(disp)
    sns.heatmap(disp, ax=ax_a, cmap="viridis", vmin=0, vmax=1, square=True, annot=False,
                cbar_kws={"shrink": 0.6, "label": "Normalized cosine similarity of mean embeddings",
                          "pad": 0.02})
    confusion_mtx._add_axis_markers(ax_a, labels, specs, n, gsize, fontsize=FONT - 1, markersize=3.5)
    ax_a.tick_params(bottom=False, left=False)
    for k in range(gsize, n, gsize):
        ax_a.axvline(k, color="white", lw=0.8); ax_a.axhline(k, color="white", lw=0.8)
    _panel(ax_a, "a", x=-0.12, y=1.0)

    ax_b = fig.add_subplot(gs[0, 1])
    items = sorted(bar_vals.items(), key=lambda kv: kv[1], reverse=True)
    y = np.arange(len(items))
    ax_b.barh(y, [v for _, v in items], color=[bar_colors[k] for k, _ in items], height=0.62)
    for yi, (_, v) in zip(y, items):
        ax_b.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=FONT - 1)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([shorten(k, cfg) for k, _ in items])
    ax_b.set_ylim(len(items) - 0.5, -0.5)
    ax_b.set_xlim(0, max(v for _, v in items) * 1.25)
    ax_b.set_xlabel("Similarity to SFT\n(1 − normalized distance)")
    ax_b.spines[["top", "right"]].set_visible(False)
    _panel(ax_b, "b", x=-0.25, y=1.0)
    _save(fig, out_dir, "fig5")


# ── Table 1 ───────────────────────────────────────────────────────────────────

def table1(cfg: dict, out_dir: Path) -> None:
    print("[table1]")
    src = Path(cfg["output_dir"]) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("table1.csv", "table1.md"):
        shutil.copy2(src / name, out_dir / name)
        print(f"  copied: {out_dir / name}")


def main() -> None:
    cfg = load_config()
    version = cfg.get("paper", {}).get("version", "v1")
    root = Path(cfg["output_dir"]) / "paper" / version
    jobs = {"fig2": lambda: fig2(cfg, root / "fig"), "fig3": lambda: fig3(cfg, root / "fig"),
            "fig5": lambda: fig5(cfg, root / "fig"), "table1": lambda: table1(cfg, root / "tables")}
    targets = sys.argv[1:] or list(jobs)
    for t in targets:
        jobs[t]()


if __name__ == "__main__":
    main()
