"""v4 Fig 4 構造が移る（主役）: 別の腫瘍で測った施設のずれを足すだけで、同じ腫瘍型が施設をまたいで隣り合う（TITAN、平行移動）。

a 髄膜腫 12 組の樹形図 3 段（補正なし / グリオーマのずれを借りる / 髄膜腫で推定）
b グリオーマ 3 組の樹形図 3 段（補正なし / 髄膜腫のずれを借りる / グリオーマで推定）
c 推定元を横に並べた隣り合う組の数（当て先 髄膜腫 12 組・グリオーマ 3 組。推定元 = 補正なし、SFT、PCNSL、転移、もう一方のよくある
  腫瘍、同じ腫瘍）。SFT から借りると戻らないこともそのまま見せる
d 鑑別の腫瘍（SFT ← 髄膜腫、PCNSL ← グリオーマ、転移 ← グリオーマ）の 2 施設の中心の距離（施設内のばらつき = 1）と
  近傍の混ざり（院内の 10 近傍に占める EBRAINS の割合 / 期待値、1 = 見分けがつかない）。補正なし → 借りる の 2 点。
  SFT は標本誤差の床（2 施設の中心が一致していても残る値、transfer_rare_target.py の gap_noise_sd）を短い線で添える

樹形図の描き方（横幅・余白・マーカー・四角・縦の詰め方）は v3 Fig 5（figs/subtype_structure.py）と同じ部品を使う
（figs/transfer_structure_preview.py を 12 組の定義に直して本番にしたもの）。四角の数は保存した連結 Z の葉の並び
（dendrogram.adjacent_pairs）と correction_titan.csv の pairs_adjacent の両方と一致することを確かめる。

入力: output_dir/transfer/structure/titan/{meningioma,glioma}/{source}__centroid/（transfer_correction.py）、
      output_dir/transfer/correction_titan.csv・correction_titan_groups.csv、rare_target_titan.csv（transfer_rare_target.py）
出力: fig/{fig n}_transfer_structure.{png,pdf}、同 _pairs.csv（段ごとの組の数と、組になった / ならなかったサブタイプ）、
      同 _values.csv（c・d の値）

Usage:
    uv run python -m figs.transfer_structure
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import dendrogram
from figs import label, stem
from figs.common import config, fig_dir, fig_path, out_root, src_markers
from figs.subtype_structure import (AX_B, AX_L, AX_R, FRAME_PAD, LEGEND_H, MARKER_SIZE, ROW_H, ROW_W, TREE_SCALE,
                                    _row_geometry, check_frames, legend_handles, marker_specs)
from utils.display import FAMILY_LABELS, GLIOMA_GROUP_COLORS, GLIOMA_GROUP_LABELS, ordered_subtypes

NAME = "transfer_structure"
ENCODER = "titan"
METHOD = "centroid"
DPI = 150                              # v3 Fig 5 と同じ（図が大きいので 150）
COLUMNS = {                            # 当て先 → [(推定元, 段の題)]
    "meningioma": [("none", "No correction"), ("glioma", "Δ borrowed from glioma"), ("own", "Δ from meningioma itself")],
    "glioma": [("none", "No correction"), ("meningioma", "Δ borrowed from\nmeningioma"), ("own", "Δ from glioma itself")],
}
SOURCES_C = {                          # c の横軸（当て先ごと）
    "meningioma": ["none", "SFT", "lymphoma", "metastasis", "glioma", "own"],
    "glioma": ["none", "SFT", "lymphoma", "metastasis", "meningioma", "own"],
}
TARGET_TITLE = {"meningioma": "Meningioma (12 pairs)", "glioma": "Glioma (3 pairs)"}
COL_GAP = 1.6
TITLE_SIZE = {"meningioma": 22, "glioma": 18}
BOTTOM_H = 6.2                         # in（c・d の帯）
FS = 18                                # c・d の文字


def load_tree(cfg: dict, target: str, source: str):
    d = out_root(cfg) / "transfer" / "structure" / ENCODER / target / f"{source}__{METHOD}"
    groups, D, Z = dendrogram.load_raw(d)
    return groups, pd.DataFrame(D, index=groups["label"], columns=groups["label"]), Z


def glioma_specs(groups: pd.DataFrame, cfg: dict) -> dict:
    mkr = src_markers(cfg, sorted(groups["source"].unique()))
    return {r.label: {"color": GLIOMA_GROUP_COLORS[r.subtype], "marker": mkr[r.source], "subtype": r.subtype,
                      "source": r.source, "short": r.subtype} for r in groups.itertuples()}


def source_label(s: str, target: str) -> str:
    if s == "none":
        return "No correction"
    if s == "own":
        return f"{FAMILY_LABELS[target]}\n(itself)"
    return FAMILY_LABELS.get(s, s)


def ddx_values(cfg: dict) -> pd.DataFrame:
    """d の値: 鑑別の腫瘍の中心の距離・近傍の混ざり（補正なし / 借りる）と、SFT の標本誤差の床。"""
    out = out_root(cfg) / "transfer"
    rt = pd.read_csv(out / f"rare_target_{ENCODER}.csv")
    gr = pd.read_csv(out / f"correction_{ENCODER}_groups.csv")
    rows = []
    for name, target, src, grp, tab in (("SFT ← meningioma", "paper", "meningioma", "SFT", rt),
                                        (f"{FAMILY_LABELS['lymphoma']} ← glioma", "lymphoma", "glioma", "lym", gr),
                                        ("Metastasis ← glioma", "metastasis", "glioma", "met", gr)):
        for cond, s in (("none", "none"), ("borrowed", src)):
            d = tab[(tab["method"] == METHOD) & (tab["source"] == s) & (tab["group"] == grp) & (tab["target"] == target)]
            v = d.set_index("metric")["value"]
            rows.append({"pair": name, "condition": cond, "group_gap_sd": float(v["group_gap_sd"]), "knn_mix": float(v["knn_mix"]),
                         "floor": float(v["gap_noise_sd"]) if "gap_noise_sd" in v else np.nan})
    return pd.DataFrame(rows)


def figure(cfg: dict) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    corr = pd.read_csv(out_root(cfg) / "transfer" / f"correction_{ENCODER}.csv")
    pairs_tab = corr[(corr["method"] == METHOD) & (corr["metric"] == "pairs_adjacent")]
    data = {t: [(s, title, *load_tree(cfg, t, s)) for s, title in rows] for t, rows in COLUMNS.items()}
    n_rows = 3
    row_h, ah, band_frac, bottom_frac = _row_geometry(TREE_SCALE)
    w_left = (AX_R - AX_L) * ROW_W
    leaf_w = w_left / len(data["meningioma"][0][2])
    x_left = AX_L * ROW_W
    x_right = x_left + w_left + COL_GAP
    w_right = leaf_w * len(data["glioma"][0][2])
    fig_w = x_right + w_right + (1 - AX_R) * ROW_W
    fig_h = n_rows * row_h + LEGEND_H + BOTTOM_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    rec = []
    for target, x0, w, letter in (("meningioma", x_left, w_left, "a"), ("glioma", x_right, w_right, "b")):
        rows = data[target]
        ymax = max(dendrogram.dendrogram_height(mat) for _, _, _, mat, _ in rows) * 1.05
        for i, (source, title, groups, mat, Z) in enumerate(rows):
            row_bottom = fig_h - (i + 1) * row_h
            ax = fig.add_axes([x0 / fig_w, (row_bottom + AX_B * ROW_H) / fig_h, w / fig_w, ah / fig_h])
            specs = marker_specs(groups, cfg) if target == "meningioma" else glioma_specs(groups, cfg)
            npairs = dendrogram.draw_dendrogram(ax, mat, False, specs, ymax=ymax, marker_size=MARKER_SIZE, frame_pad=FRAME_PAD,
                                                band_frac=band_frac, bottom_frac=bottom_frac)
            check_frames(ax)
            n = groups["subtype"].nunique()
            boxed = {groups["subtype"].iloc[a] for a, _ in dendrogram.adjacent_pairs(Z, groups)}
            want = int(pairs_tab[(pairs_tab["target"] == target) & (pairs_tab["source"] == source)]["value"].iloc[0])
            assert len(boxed) == npairs == want, (target, source, len(boxed), npairs, want)
            sep = "   " if target == "meningioma" else "\n"
            ax.set_title(f"{title}{sep}cross-site pairs: {npairs}/{n}", fontsize=TITLE_SIZE[target], loc="left", pad=8)
            if i == 0:
                ax.text(-0.06 if target == "meningioma" else -0.35, 1.08, letter, transform=ax.transAxes, fontsize=30,
                        fontweight="bold", va="bottom")
            subs = ordered_subtypes(set(groups["subtype"]), cfg)
            disp = (lambda s: GLIOMA_GROUP_LABELS.get(s, s)) if target == "glioma" else (lambda s: s)
            rec.append({"panel": letter, "target": target, "source": source, "pairs": npairs, "n_groups": n,
                        "paired": "; ".join(disp(s) for s in subs if s in boxed),
                        "not_paired": "; ".join(disp(s) for s in subs if s not in boxed)})
    legend_y = BOTTOM_H
    ax_l = fig.add_axes([x_left / fig_w, legend_y / fig_h, w_left / fig_w, LEGEND_H / fig_h])
    ax_l.axis("off")
    handles = legend_handles(data["meningioma"][-1][2], cfg)
    for h in handles:   # 施設の表記を v4 の表記（In-house）に（figs.subtype_structure は v3 の表記のまま）
        if h.get_label() == "patho2":
            h.set_label("In-house")
    ax_l.legend(handles=handles, loc="center", ncol=9, frameon=False,
                fontsize=18, handlelength=1.6, handletextpad=0.5, columnspacing=1.4)
    ax_r = fig.add_axes([x_right / fig_w, legend_y / fig_h, w_right / fig_w, LEGEND_H / fig_h])
    ax_r.axis("off")
    ax_r.legend(handles=[Patch(color=GLIOMA_GROUP_COLORS[g], label=GLIOMA_GROUP_LABELS[g]) for g in GLIOMA_GROUP_COLORS],
                loc="center left", ncol=1, frameon=False, fontsize=16, handlelength=1.6, handletextpad=0.5)

    # c: 推定元を横に並べた組の数（当て先ごとに 1 枚）
    vals = []
    bot = 0.9 / fig_h
    hgt = (BOTTOM_H - 2.2) / fig_h
    cx = [x_left, x_left + 0.34 * w_left]
    cw = 0.27 * w_left
    for k, target in enumerate(("meningioma", "glioma")):
        ax = fig.add_axes([cx[k] / fig_w, bot, cw / fig_w, hgt])
        srcs = SOURCES_C[target]
        ys = [int(pairs_tab[(pairs_tab["target"] == target) & (pairs_tab["source"] == s)]["value"].iloc[0]) for s in srcs]
        n = rec[0]["n_groups"] if target == "meningioma" else rec[3]["n_groups"]
        colors = ["#a3a29b"] + ["#2a78d6"] * (len(srcs) - 2) + ["#1baf7a"]
        ax.bar(range(len(srcs)), ys, color=colors, width=0.7)
        for x, y in enumerate(ys):
            ax.text(x, y + n * 0.02, str(y), ha="center", va="bottom", fontsize=FS - 2)
        ax.axhline(n, color="#555555", lw=1, ls=":")
        ax.set_ylim(0, n * 1.15)
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels([source_label(s, target) for s in srcs], rotation=35, ha="right", fontsize=FS - 3)
        ax.set_ylabel("Cross-site pairs", fontsize=FS)
        ax.set_title(f"Target: {TARGET_TITLE[target]}\nΔ estimated from", fontsize=FS, loc="left")
        ax.tick_params(axis="y", labelsize=FS - 3)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.text(-0.22, 1.18, "c", transform=ax.transAxes, fontsize=30, fontweight="bold", va="bottom")
        vals += [{"panel": "c", "target": target, "source": s, "pairs_adjacent": y, "n_pairs": n} for s, y in zip(srcs, ys)]

    # d: 鑑別の腫瘍（補正なし → 借りる）
    dv = ddx_values(cfg)
    dx0 = x_left + 0.70 * w_left
    dw = (x_right + w_right - dx0 - 0.6) / 2 - 0.9
    for k, (metric, ylab) in enumerate((("group_gap_sd", "Site-centre distance\n(within-site spread = 1)"),
                                        ("knn_mix", "Neighbour mixing\n(1 = sites indistinguishable)"))):
        ax = fig.add_axes([(dx0 + k * (dw + 1.8)) / fig_w, bot, dw / fig_w, hgt])
        pairs = list(pd.unique(dv["pair"]))
        for x, pname in enumerate(pairs):
            d = dv[dv["pair"] == pname].set_index("condition")
            y0, y1 = d.loc["none", metric], d.loc["borrowed", metric]
            ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=dict(arrowstyle="-|>", color="#2a78d6", lw=2.2, mutation_scale=22,
                                                                               shrinkA=9, shrinkB=9))   # 点（s=160）の縁で止める
            ax.scatter([x], [y0], s=160, color="#a3a29b", zorder=3, label="No correction" if x == 0 else None)
            ax.scatter([x], [y1], s=160, color="#2a78d6", zorder=3, label="Δ borrowed" if x == 0 else None)
            if metric == "group_gap_sd" and np.isfinite(d.loc["none", "floor"]) and pname.startswith("SFT"):
                f = d.loc["none", "floor"]
                ax.hlines(f, x - 0.25, x + 0.25, color="#555555", lw=1.6, ls="--")
                ax.text(x + 0.28, f, "sampling\nfloor", fontsize=FS - 6, va="center", color="#555555")
            vals += [{"panel": "d", "pair": pname, "metric": metric, "none": y0, "borrowed": y1}]
        if metric == "knn_mix":
            ax.axhline(1, color="#555555", lw=1, ls=":")
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels([p.replace(" ← ", "\n← ") for p in pairs], fontsize=FS - 4)
        ax.set_xlim(-0.5, len(pairs) - 0.3)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(ylab, fontsize=FS - 2)
        ax.tick_params(axis="y", labelsize=FS - 3)
        ax.spines[["top", "right"]].set_visible(False)
        if k == 0:
            ax.text(-0.3, 1.18, "d", transform=ax.transAxes, fontsize=30, fontweight="bold", va="bottom")
            ax.legend(fontsize=FS - 5, frameon=False, loc="upper right")
        ax.set_title("Rare tumours, Δ borrowed" if k == 0 else " ", fontsize=FS, loc="left")
    return fig, pd.DataFrame(rec), pd.DataFrame(vals)


def main() -> None:
    print(f"[{label(NAME)}] {NAME}")
    cfg = config()
    fig, rec, vals = figure(cfg)
    d = fig_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{stem(NAME)}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {d / stem(NAME)}.png / .pdf")
    rec.to_csv(fig_path(cfg, NAME, "_pairs.csv"), index=False)
    vals.to_csv(fig_path(cfg, NAME, "_values.csv"), index=False)
    print(rec[["panel", "target", "source", "pairs", "n_groups"]].to_string(index=False))
    print(vals.to_string(index=False))


if __name__ == "__main__":
    main()
