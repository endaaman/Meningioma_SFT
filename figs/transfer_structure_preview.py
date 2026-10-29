"""transfer_structure_preview.py — 転移テストのサブタイプ構造の樹形図のプレビュー（v4 用。figs/__init__.py には登録しない）。

v3 Fig 5（figs/subtype_structure.py）と同じ形の図: 施設 × 群の平均ベクトル間のユークリッド距離を行列ごとに正規化して
Ward 法、葉 = 群の色 × 施設マーカー（丸 EBRAINS / 四角 patho2）、葉の並びで同じ群の 2 施設が隣り合ったら点線の四角、
各段の題に「cross-site pairs: k/n」（四角の数）。段は補正法ではなく、平行移動（centroid）の Δ をどこで推定したか
（transfer_correction.py の推定元）:
    a 補正なし（none）/ b 当て先自身（own = 上限）/ c 別の 1 科 / d 他の科を科ごとに推定して平均（others_mean）/
    e 他の科のスライドをまとめて 1 回で推定（others。科の構成が施設で違う）
左の列 = 当て先 髄膜腫 + SFT（13 群 × 2 施設）、c = グリオーマ。右の列 = 当て先 グリオーマ（ast / glm / odg × 2 施設）、
c = 髄膜腫。TITAN のみ。段の記号は左右で共有（左の列にだけ付ける）。

各段の描画（横幅・余白・マーカー・四角・樹形図部分の縦の詰め方）は figs/subtype_structure.py の定数と関数をそのまま使う
（ken: 元と同じバランスで、スタイルは作り込まない）。右の列は葉 1 枚あたりの横幅を左と同じにする（葉が 6 枚なので細い）。
縦軸の上限は列ごとに 5 段で共通。四角の数は保存した連結 Z の葉の並び（dendrogram.adjacent_pairs）と
correction_titan.csv の pairs_adjacent の両方と一致することを確かめる。

入力: output_dir/transfer/structure/titan/{target}/{source}__centroid/euc_mean_raw_groups.csv / euc_mean_raw.npz
          （transfer_correction.py。D = euc_mean の生値、Z = Ward）
      output_dir/transfer/correction_titan.csv（pairs_adjacent）
出力: paper.dir/v4/fig/_preview_transfer_structure.{png,pdf}, 同 _pairs.csv（段ごとの四角の数と、四角で囲んだ群 / 囲まれなかった群）
      config の paper.version によらず v4 に出す

Usage:
    uv run python -m figs.transfer_structure_preview
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

import dendrogram
from figs.common import config, out_root, paper_root, src_markers
from figs.subtype_structure import (AX_B, AX_L, AX_R, LEGEND_H, MARKER_SIZE, FRAME_PAD, ROW_H, ROW_W, TREE_SCALE,
                                    _row_geometry, check_frames, legend_handles, marker_specs, save_tree_figure)
from utils.display import GLIOMA_GROUP_COLORS, GLIOMA_GROUP_LABELS, ordered_subtypes

PAPER_VERSION = "v4"
NAME = "_preview_transfer_structure"
ENCODER = "titan"
METHOD = "centroid"                    # 平行移動だけ
# 列 = 当て先、段 = 推定元（transfer_correction.py の source）と題。右の列は細いので題を手で折り返し、四角の数は次の行に置く
COLUMNS = {
    "paper": [
        ("none", "No correction"),
        ("own", "Δ from the same tumours (meningioma + SFT; ceiling)"),
        ("glioma", "Δ from glioma"),
        ("others_mean", "Δ from glioma + DLBCL + metastasis, per-family mean"),
        ("others", "Δ from glioma + DLBCL + metastasis, pooled slides (case mix differs between sites)"),
    ],
    "glioma": [
        ("none", "No correction"),
        ("own", "Δ from the same tumours (glioma; ceiling)"),
        ("meningioma", "Δ from meningioma"),
        ("others_mean", "Δ from meningioma + SFT + DLBCL + metastasis,\nper-family mean"),
        ("others", "Δ from meningioma + SFT + DLBCL + metastasis,\npooled slides (case mix similar between sites)"),
    ],
}
COL_GAP = 1.6                          # in。左の列の軸の右端から右の列の軸の左端まで（右の列の縦軸の目盛りが入る）
# 題の文字の大きさ: 左の列は v3 Fig 5 と同じ 22。右の列は 3 行になる（説明 2 行 ＋ 四角の数）ので 18 にして、上の段の葉に
# かからないようにする（段の間は 1.38 in、22 で 4 行だと約 1.6 in になって重なった）。右の余白にははみ出してよい
TITLE_SIZE = {"paper": 22, "glioma": 18}


def load(cfg: dict, target: str, source: str) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """(groups, mat, Z): mat は euc_mean の生の距離行列（draw_dendrogram に渡すと v3 Fig 5 と同じ正規化）。"""
    d = out_root(cfg) / "transfer" / "structure" / ENCODER / target / f"{source}__{METHOD}"
    groups, D, Z = dendrogram.load_raw(d)
    return groups, pd.DataFrame(D, index=groups["label"], columns=groups["label"]), Z


def glioma_specs(groups: pd.DataFrame, cfg: dict) -> dict:
    """グリオーマ 3 群の葉の指定（figs.subtype_structure.marker_specs と同じ形、色は GLIOMA_GROUP_COLORS）。"""
    mkr = src_markers(cfg, sorted(groups["source"].unique()))
    return {r.label: {"color": GLIOMA_GROUP_COLORS[r.subtype], "marker": mkr[r.source], "subtype": r.subtype,
                      "source": r.source, "short": r.subtype}
            for r in groups.itertuples()}


def display_name(target: str, sub: str) -> str:
    return GLIOMA_GROUP_LABELS.get(sub, sub) if target == "glioma" else sub


def figure(cfg: dict) -> tuple[plt.Figure, pd.DataFrame]:
    expected = pd.read_csv(out_root(cfg) / "transfer" / f"correction_{ENCODER}.csv")
    expected = expected[(expected["method"] == METHOD) & (expected["metric"] == "pairs_adjacent")]
    data = {t: [(s, title, *load(cfg, t, s)) for s, title in rows] for t, rows in COLUMNS.items()}
    n_rows = max(len(v) for v in data.values())
    row_h, ah, band_frac, bottom_frac = _row_geometry(TREE_SCALE)
    w_left = (AX_R - AX_L) * ROW_W
    leaf_w = w_left / len(data["paper"][0][2])                     # 葉 1 枚あたりの横幅（in）
    x_left = AX_L * ROW_W
    x_right = x_left + w_left + COL_GAP
    w_right = leaf_w * len(data["glioma"][0][2])
    fig_w = x_right + w_right + (1 - AX_R) * ROW_W
    fig_h = n_rows * row_h + LEGEND_H
    fig = plt.figure(figsize=(fig_w, fig_h))
    out = []
    for target, x0, w in (("paper", x_left, w_left), ("glioma", x_right, w_right)):
        rows = data[target]
        ymax = max(dendrogram.dendrogram_height(mat) for _, _, _, mat, _ in rows) * 1.05
        for i, ((source, title, groups, mat, Z), letter) in enumerate(zip(rows, "abcdefg")):
            row_bottom = fig_h - (i + 1) * row_h
            ax = fig.add_axes([x0 / fig_w, (row_bottom + AX_B * ROW_H) / fig_h, w / fig_w, ah / fig_h])
            specs = marker_specs(groups, cfg) if target == "paper" else glioma_specs(groups, cfg)
            npairs = dendrogram.draw_dendrogram(ax, mat, False, specs, ymax=ymax, marker_size=MARKER_SIZE,
                                                frame_pad=FRAME_PAD, band_frac=band_frac, bottom_frac=bottom_frac)
            check_frames(ax)
            n = groups["subtype"].nunique()
            boxed = {groups["subtype"].iloc[a] for a, _ in dendrogram.adjacent_pairs(Z, groups)}
            want = expected[(expected["target"] == target) & (expected["source"] == source)]["value"]
            assert len(boxed) == npairs == int(want.iloc[0]), (target, source, len(boxed), npairs, want.tolist())
            sep = "   " if target == "paper" else "\n"
            ax.set_title(f"{title}{sep}cross-site pairs: {npairs}/{n}", fontsize=TITLE_SIZE[target], loc="left", pad=8)
            if target == "paper":
                ax.text(-0.06, 1.08, letter, transform=ax.transAxes, fontsize=30, fontweight="bold", va="bottom")
            subs = ordered_subtypes(set(groups["subtype"]), cfg)
            out.append({"target": target, "row": letter, "source": source, "method": METHOD, "encoder": ENCODER,
                        "title": title.replace("\n", " "), "pairs": npairs, "n_groups": n,
                        "paired": "; ".join(display_name(target, s) for s in subs if s in boxed),
                        "not_paired": "; ".join(display_name(target, s) for s in subs if s not in boxed)})
    # 凡例: 左の列の下は v3 Fig 5 と同じ（サブタイプ色・施設マーカー・四角）、右の列の下はグリオーマ 3 群の色
    ax_l = fig.add_axes([x_left / fig_w, 0.0, w_left / fig_w, LEGEND_H / fig_h])
    ax_l.axis("off")
    ax_l.legend(handles=legend_handles(data["paper"][-1][2], cfg), loc="center", ncol=9, frameon=False,
                fontsize=18, handlelength=1.6, handletextpad=0.5, columnspacing=1.4, markerscale=1.0)
    ax_r = fig.add_axes([x_right / fig_w, 0.0, w_right / fig_w, LEGEND_H / fig_h])
    ax_r.axis("off")
    ax_r.legend(handles=[Patch(color=GLIOMA_GROUP_COLORS[g], label=GLIOMA_GROUP_LABELS[g]) for g in GLIOMA_GROUP_COLORS],
                loc="center left", ncol=1, frameon=False, fontsize=18, handlelength=1.6, handletextpad=0.5)
    return fig, pd.DataFrame(out)


def main() -> None:
    print(f"[preview] {NAME}")
    cfg = config()
    d = paper_root(cfg) / PAPER_VERSION / "fig"
    d.mkdir(parents=True, exist_ok=True)
    fig, pairs = figure(cfg)
    save_tree_figure(fig, d / NAME)
    path = d / f"{NAME}_pairs.csv"
    pairs.to_csv(path, index=False)
    print(f"  saved: {path}")
    print(pairs[["target", "row", "source", "pairs", "n_groups"]].to_string(index=False))


if __name__ == "__main__":
    main()
