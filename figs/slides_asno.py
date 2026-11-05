"""slides_asno.py — ASNO（Asian Society for Neuro-Oncology）発表用のスライド図。論文の図表ではない（figs/__init__.py に登録しない）。

ken の先輩が ASNO で発表するための図（16:9 = 13.33 × 7.5 in、1 枚 1 ファイル、英語、白背景、PNG 300 dpi ＋ PDF、文字 14 pt 以上）。
初期の単純な話だけ:
    01 施設差（院内 patho2 と EBRAINS。補正なしの UMAP を施設 / 診断で塗り分け）
    02 GAN（画像レベルの補正）が画像に何をするか（a パッチ一覧 = Fig 2a と同じパッチ、b 色の定量。Macenko は出さない）
    03 補正なし / GAN / 平行移動 の UMAP（施設 / 診断）と施設差の数値（施設平均どうしの距離・ASW 施設）
    04 ペアの回復（施設 × サブタイプ平均の Ward 樹形図、cross-site pairs k/13。Fig 5 と同じ描画）
    05 まとめ（03・04 の数値を 3 条件で並べる）
条件は CONDITIONS_CORE（補正なし / GAN / 平行移動）だけ。ComBat・Macenko・相似変換・腫瘍をまたぐ転移・他のエンコーダ・v4 は出さない。
データは v3 の論文図（Fig 2・3・5）と同じ入力を読むだけで再計算しない（TITAN、髄膜腫 + SFT、v3 のコホート）。
figs.tissue / figs.subtype_structure の定数・ヘルパーを import し、描画は文字を大きくした版をここに写す
（元のモジュールは変えない・回さない。回すと paper/v3 の図を上書きする）。

施設の表示名は SITE_NAMES（凡例・行ラベル）と SITE_SHORT（題・本文）の 1 か所。条件の表示名は utils/display.py の
CONDITION_LABELS。サブタイプは凡例で全名（データのラベル。"… meningioma" は「Meningioma subtype」の見出しの下に語幹だけ）。
数値はすべて CSV から読む（Fig 3f・Fig 5 と同じ値になることを assert で確かめる）。文字がスライドからはみ出したら止まる。

入力（v3 の図と同じ。パッチと色は config の paper.version によらず SOURCE_VERSION = v3 から読む）:
    output_dir/umap/{variant}/coords.csv                                   （umap_plot.py。Fig 3a–e）
    output_dir/method_compare/summary.csv                                  （method_compare.py。Fig 3f）
    output_dir/harmonization/site_null.csv                                 （harmonization.py。ASW 施設の chance）
    output_dir/dendrogram/{variant}/euc_mean_raw_groups.csv / .npz, pairs.csv （dendrogram.py。Fig 5）
    paper.dir/v3/patch_candidates/（candidates.csv・候補の png・patho2_gan_single/ のキャッシュ。Fig 2a）
    paper.dir/v3/color/slide_color.csv                                     （paper_color.py。Fig 2b）
出力: paper.dir/share/asno/{01_domain_shift, 02a_gan_patches, 02b_gan_colour, 03_correction_umap,
      04_pair_recovery, 05_summary}.{png,pdf}

Usage:
    uv run python -m figs.slides_asno
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text
from PIL import Image

import dendrogram
from figs.common import config, out_root, paper_root, src_markers
from figs.subtype_structure import check_frames, load as load_tree, marker_specs
from figs.tissue import COLUMNS as PATCH_COLUMNS, GAN_SITE, PICK, SCALE_UM, _candidate, _mpp, _original
from utils.display import CONDITION_COLORS, CONDITION_LABELS, CONDITIONS_CORE, make_abbrev, ordered_subtypes, \
    subtype_color_map

# ── 施設の表示名（ここだけ書き換えれば全スライドに効く）──────────────────────────────
SITE_NAMES = {"patho2": "In-house (Hokkaido Univ.)", "ebrains": "EBRAINS"}   # 凡例・行ラベル
SITE_SHORT = {"patho2": "in-house", "ebrains": "EBRAINS"}                    # 題・副題の文中
GAN_ARROW = " → GAN"                       # GAN 変換後の院内画像（SITE_NAMES["patho2"] + これ）

VARIANTS = CONDITIONS_CORE                 # 補正なし / GAN / 平行移動
SOURCE_VERSION = "v3"                      # パッチ・色の入力は v3 の図と同じもの
SLIDE_W, SLIDE_H = 40 / 3, 7.5             # in（16:9 = 4000 × 2250 px at 300 dpi）
DPI = 300
SEED = 42                                  # 散布図の描く順・色の図の横ゆらぎ

# 文字の大きさ（pt、13.33 × 7.5 in で）。最小 14
TITLE, SUBTITLE, HEAD, BODY, SMALL = 28, 17, 19, 16, 14
plt.rcParams.update({
    "font.size": BODY, "axes.titlesize": HEAD, "axes.labelsize": BODY,
    "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
POINT_S = 14                               # UMAP の点の面積（pt²）
GREY = "#444444"
MARGIN = 0.45                              # in（題の左端）


# ── 共通 ─────────────────────────────────────────────────────────────────────

def out_dir(cfg: dict) -> Path:
    return paper_root(cfg) / "share" / "asno"


def src_dir(cfg: dict) -> Path:
    return paper_root(cfg) / SOURCE_VERSION


def site_colors(cfg: dict) -> dict[str, str]:
    return cfg["display"]["colors"]["sources"]


def new_slide(title: str, subtitle: str | None = None) -> plt.Figure:
    """題（上端から 0.35 in）と副題（0.95 in、改行で 2 行まで）。副題の下端は上端から約 1.65 in。"""
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor="white")
    fig.text(MARGIN / SLIDE_W, 1 - 0.35 / SLIDE_H, title, fontsize=TITLE, fontweight="bold", ha="left", va="top")
    if subtitle:
        fig.text(MARGIN / SLIDE_W, 1 - 0.98 / SLIDE_H, subtitle, fontsize=SUBTITLE, color=GREY, ha="left", va="top",
                 linespacing=1.3)
    return fig


def footnote(fig: plt.Figure, text: str, x: float = MARGIN, y: float = 0.12) -> None:
    fig.text(x / SLIDE_W, y / SLIDE_H, text, fontsize=SMALL, color=GREY, ha="left", va="bottom", linespacing=1.25)


def axes_in(fig: plt.Figure, x: float, y: float, w: float, h: float) -> plt.Axes:
    """インチ単位（左下が原点）で軸を置く。"""
    return fig.add_axes([x / SLIDE_W, y / SLIDE_H, w / SLIDE_W, h / SLIDE_H])


def check_inside(fig: plt.Figure, name: str) -> None:
    """文字・凡例・軸（目盛りと軸の外に描いたマーカーを含む）がスライドの外にはみ出していないこと
    （はみ出しは目で見落としやすいので機械的に止める）。"""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    items = [(f"axes {i}", ax.get_tightbbox(r)) for i, ax in enumerate(fig.axes)]
    items += [(a.get_text()[:50] if isinstance(a, Text) else "legend", a.get_window_extent(r))
              for a in [*fig.texts, *fig.legends] if a.get_visible()]
    bad = [lab for lab, bb in items if bb is not None and (bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1)]
    assert not bad, f"{name}: outside the slide: {bad}"


def save_slide(fig: plt.Figure, cfg: dict, name: str) -> None:
    check_inside(fig, name)
    d = out_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"{name}.{ext}", dpi=DPI, facecolor="white")   # bbox は切らない（16:9 のまま）
    plt.close(fig)
    print(f"  saved: {d / name}.png / .pdf")


def subtype_name(s: str) -> str:
    """凡例の名前: 髄膜腫のサブタイプは語幹（"Fibrous"）、それ以外（SFT）はそのまま。"""
    return s[: -len(" meningioma")] if s.endswith(" meningioma") else s


def diagnosis_legend(ax: plt.Axes, subs: list[str], sub_map: dict, marker: str | None = "o", **kw) -> Legend:
    """診断の凡例: SFT、見出し「Meningioma subtype」、その下に各サブタイプ（全名の語幹）。marker=None なら四角の色票。"""
    def handle(s: str):
        if marker:
            return Line2D([0], [0], marker=marker, color=sub_map[s], ls="none", markersize=11)
        return Patch(facecolor=sub_map[s])
    other = [s for s in subs if not s.endswith(" meningioma")]
    men = [s for s in subs if s.endswith(" meningioma")]
    handles = [handle(s) for s in other] + [Line2D([], [], ls="none")] + [handle(s) for s in men]
    labels = [subtype_name(s) for s in other] + ["Meningioma subtype"] + [subtype_name(s) for s in men]
    kw = {"frameon": False, "handlelength": 1.0, "handletextpad": 0.5, "labelspacing": 0.28, "fontsize": SMALL,
          "borderaxespad": 0, **kw}
    leg = ax.legend(handles, labels, **kw)
    for t in leg.get_texts():
        if t.get_text() == "Meningioma subtype":
            t.set_fontweight("bold")
    return leg


def site_legend(ax: plt.Axes, cfg: dict, wrap: int | None = None, **kw) -> Legend:
    sc = site_colors(cfg)
    handles = [Line2D([0], [0], marker="o", color=sc[k], ls="none", markersize=11) for k in SITE_NAMES]
    labels = [textwrap.fill(SITE_NAMES[k], wrap) if wrap else SITE_NAMES[k] for k in SITE_NAMES]
    kw = {"frameon": False, "handlelength": 1.0, "handletextpad": 0.5, "labelspacing": 0.3, "fontsize": SMALL,
          "borderaxespad": 0, **kw}
    return ax.legend(handles, labels, **kw)


def load_umap(cfg: dict) -> dict[str, pd.DataFrame]:
    root = out_root(cfg)
    return {v: pd.read_csv(root / "umap" / cfg["variants"].get(v, v) / "coords.csv", dtype={"case_id": str})
            for v in VARIANTS}


def draw_umap(ax: plt.Axes, coords: pd.DataFrame, colors: list) -> None:
    """全点を 1 回で描く（描く順は seed 固定で並べ替え、片方の施設が常に上に乗らないように）。"""
    order = np.random.default_rng(SEED).permutation(len(coords))
    ax.scatter(coords["umap1"].to_numpy()[order], coords["umap2"].to_numpy()[order],
               c=np.asarray(colors, dtype=object)[order].tolist(), s=POINT_S, alpha=0.8, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top", "right"]].set_visible(False)


def site_metrics(cfg: dict) -> tuple[pd.DataFrame, float]:
    """(summary の 3 条件, ASW 施設の chance = 補正なし条件の within-subtype 帰無の平均)。Fig 3f と同じ値。"""
    root = out_root(cfg)
    s = pd.read_csv(root / "method_compare" / "summary.csv").set_index("variant").loc[VARIANTS]
    null = pd.read_csv(root / "harmonization" / "site_null.csv")
    chance = null[(null["null"] == "within_subtype") & (null["state"] == "original") & (null["metric"] == "asw_batch")]
    assert len(chance) == 1
    return s, float(chance["null_mean"].iloc[0])


def site_counts(coords: pd.DataFrame) -> dict[str, int]:
    return coords["source"].value_counts().to_dict()


# ── 01 施設差 ─────────────────────────────────────────────────────────────────

def slide_domain_shift(cfg: dict, umap: dict) -> None:
    c = umap["original"]
    n = site_counts(c)
    subs = ordered_subtypes(set(c["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    ref, other = cfg["reference"], GAN_SITE
    fig = new_slide(f"Site shift: {SITE_SHORT[other]} cohort vs {SITE_SHORT[ref]}",
                    f"Uncorrected TITAN slide embeddings (UMAP; {len(c)} slides, one dot per slide)")
    y0, h, w = 1.3, 4.3, 4.6
    ax_l = axes_in(fig, 0.6, y0, w, h)
    draw_umap(ax_l, c, c["source"].map(site_colors(cfg)).tolist())
    ax_l.set_title("Coloured by site", fontsize=HEAD, pad=10)
    ax_r = axes_in(fig, 0.6 + w + 0.45, y0, w, h)
    draw_umap(ax_r, c, c["subtype"].map(sub_map).tolist())
    ax_r.set_title("Coloured by diagnosis", fontsize=HEAD, pad=10)
    # 施設の凡例（例数つき）は左の図の下、診断の凡例は右端の列
    leg = site_legend(ax_l, cfg, loc="upper left", bbox_to_anchor=(0.0, -0.05), fontsize=BODY)
    for t, k in zip(leg.get_texts(), SITE_NAMES):
        t.set_text(f"{SITE_NAMES[k]} (n = {n[k]})")
    ax_leg = axes_in(fig, 0.6 + 2 * w + 0.75, y0, SLIDE_W - (0.6 + 2 * w + 0.75) - 0.1, h); ax_leg.axis("off")
    diagnosis_legend(ax_leg, subs, sub_map, loc="upper left", bbox_to_anchor=(0, 1.0))
    save_slide(fig, cfg, "01_domain_shift")


# ── 02 GAN が画像に何をするか ────────────────────────────────────────────────────

def _gan_cached(cand_dir: Path, r: pd.Series) -> np.ndarray:
    """GAN 変換後パッチを v3 のキャッシュから読むだけ（figs.tissue._gan は無ければ書き込むので使わない）。"""
    path = cand_dir / f"{GAN_SITE}_gan_single" / f"{r.case_id}_x{r.x}_y{r.y}.png"
    if not path.exists():
        raise FileNotFoundError(f"{path}: v3 の GAN パッチのキャッシュが無い")
    return np.asarray(Image.open(path).convert("RGB"))


def slide_gan_patches(cfg: dict) -> None:
    ref = cfg["reference"]
    cand_dir = src_dir(cfg) / "patch_candidates"
    cand = pd.read_csv(cand_dir / "candidates.csv")
    abbrev = make_abbrev(sorted(cand.subtype.unique()), cfg)
    full = {a: s for s, a in abbrev.items()}
    colors = subtype_color_map([full[a] for a in PATCH_COLUMNS], cfg)
    rows = [(SITE_NAMES[GAN_SITE], GAN_SITE, False),
            (SITE_NAMES[GAN_SITE] + GAN_ARROW, GAN_SITE, True),
            (SITE_NAMES[ref], ref, False)]
    fig = new_slide("What GAN does to the images",
                    "Representative H&E patches per diagnosis\n"
                    f"Middle row: the top-row patches after GAN translation toward the {SITE_SHORT[ref]} appearance")
    cell, gap = 1.42, 0.05
    ncol = len(PATCH_COLUMNS)
    x0 = SLIDE_W - 0.3 - ncol * cell - (ncol - 1) * gap
    top = SLIDE_H - 2.45                                  # 1 行目の上端（題・副題・列見出しの下）
    for i, (row_label, site, gan) in enumerate(rows):
        y = top - (i + 1) * cell - i * gap
        for j, abbr in enumerate(PATCH_COLUMNS):
            r = _candidate(cand, site, full[abbr], PICK[site][j])
            img = _gan_cached(cand_dir, r) if gan else _original(cand_dir, site, abbr, r)
            ax = axes_in(fig, x0 + j * (cell + gap), y, cell, cell)
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.5)
            if i == 0:
                ax.add_patch(Rectangle((0, 1.02), 1, 0.045, transform=ax.transAxes, color=colors[full[abbr]],
                                       clip_on=False))
                ax.text(0.5, 1.10, textwrap.fill(full[abbr], 14), transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=BODY, linespacing=1.1)
            if j == 0:
                ax.text(-0.08, 0.5, textwrap.fill(row_label, 16), transform=ax.transAxes, ha="right", va="center",
                        fontsize=BODY, fontweight="bold", linespacing=1.15)
            if i == 0 and j == 0:
                mpp = _mpp(cfg, site, r.case_id)
                size = img.shape[0]
                length = SCALE_UM / mpp
                xs, ys = size * 0.07, size * 0.91
                ax.plot([xs, xs + length], [ys, ys], color="white", lw=3, solid_capstyle="butt")
                ax.text(xs + length / 2, ys - size * 0.04, f"{SCALE_UM} µm", color="white", ha="center", va="bottom",
                        fontsize=SMALL, fontweight="bold",
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="#333333")])
    footnote(fig, f"Each patch: one of the {cfg['paper_patches']['n_candidates']} nearest to its site × diagnosis centre "
                  "(CONCH v1.5 patch embeddings).\n"
                  f"{SITE_SHORT[ref]} patches come from different cases than the {SITE_SHORT[GAN_SITE]} ones.",
             x=x0 - 2.3, y=0.1)
    save_slide(fig, cfg, "02a_gan_patches")


COLOR_METRICS = [("L", "L* (lightness)"), ("a", "a* (green–red)"), ("b", "b* (blue–yellow)"),
                 ("H", "Hematoxylin"), ("E", "Eosin")]


def colour_table(cfg: dict) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """(スライドごとの平均色, [(群, 表示名, 色)])。群は 院内 / 院内 → GAN / EBRAINS（Macenko は出さない）。
    コホートは figs.tissue.draw_color と同じく現在のラベル CSV に絞る（EBRAINS 554 → 551）。"""
    ref = cfg["reference"]
    df = pd.read_csv(src_dir(cfg) / "color" / "slide_color.csv")
    keep = pd.concat([pd.read_csv(cfg["label"][site]["subtype"], usecols=["case_id"]).assign(site=site)
                      for site in df["site"].unique()])
    df = df.merge(keep, on=["site", "case_id"], how="inner")
    sc = site_colors(cfg)
    groups = [(GAN_SITE, SITE_NAMES[GAN_SITE], sc[GAN_SITE]),
              (f"{GAN_SITE} (GAN)", SITE_NAMES[GAN_SITE] + GAN_ARROW, CONDITION_COLORS["gan"]),
              (ref, SITE_NAMES[ref], sc[ref])]
    return df, groups


def slide_gan_colour(cfg: dict) -> pd.DataFrame:
    ref = cfg["reference"]
    df, groups = colour_table(cfg)
    n = df.groupby("group").size()
    med = df.groupby("group")[[m for m, _ in COLOR_METRICS]].median()
    # 題の主張の確認: GAN 後の中央値は 5 指標とも元の院内より EBRAINS に近い
    for m, _ in COLOR_METRICS:
        assert abs(med.loc[f"{GAN_SITE} (GAN)", m] - med.loc[ref, m]) < abs(med.loc[GAN_SITE, m] - med.loc[ref, m]), m
    n_txt = " / ".join(str(n[g]) for g, _, _ in groups)
    fig = new_slide(f"GAN moves the {SITE_SHORT[GAN_SITE]} colour toward {SITE_SHORT[ref]}",
                    f"Mean colour of tissue pixels per slide ({cfg['paper_color']['n_patches']} random patches each; "
                    "one dot per slide)\n"
                    f"Black bar = median; dashed line = {SITE_SHORT[ref]} median; n = {n_txt} slides")
    rng = np.random.default_rng(SEED)
    w, gap, x0 = 1.92, 0.62, 0.85
    y0, h = 0.4, 4.15
    for j, (m, title) in enumerate(COLOR_METRICS):
        ax = axes_in(fig, x0 + j * (w + gap), y0, w, h)
        for k, (g, _, col) in enumerate(groups):
            v = df.loc[df["group"] == g, m].to_numpy()
            ax.scatter(k + rng.uniform(-0.22, 0.22, len(v)), v, s=7, color=col, alpha=0.45, linewidths=0,
                       rasterized=True)
            ax.hlines(np.median(v), k - 0.34, k + 0.34, color="black", lw=2.5, zorder=5)
        ax.axhline(med.loc[ref, m], color=GREY, lw=1.2, ls="--", zorder=4)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_title(title, fontsize=BODY, pad=8)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], marker="o", color=col, ls="none", markersize=13) for _, _, col in groups]
    fig.legend(handles, [lab for _, lab, _ in groups], loc="upper center", bbox_to_anchor=(0.5, 1 - 1.85 / SLIDE_H),
               ncol=3, frameon=False, fontsize=BODY, handletextpad=0.3, columnspacing=2.0)
    save_slide(fig, cfg, "02b_gan_colour")
    return med


# ── 03 補正なし / GAN / 平行移動 の UMAP ──────────────────────────────────────────

def slide_correction_umap(cfg: dict, umap: dict, s: pd.DataFrame, chance: float) -> None:
    ref = cfg["reference"]
    subs = ordered_subtypes(set(umap["original"]["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    sc = site_colors(cfg)
    fig = new_slide(f"Removing the site shift: {CONDITION_LABELS['gan']} vs {CONDITION_LABELS['centroid']}",
                    f"{CONDITION_LABELS['gan']}: image level ({SITE_SHORT[GAN_SITE]} patches translated toward "
                    f"{SITE_SHORT[ref]} before embedding)\n"
                    f"{CONDITION_LABELS['centroid']}: embedding level ({SITE_SHORT[GAN_SITE]} embeddings shifted by "
                    "the difference of the site means)")
    w, h, gap, x0 = 2.95, 2.02, 0.15, 0.85
    y_bot, vgap = 0.5, 0.12
    y_top = y_bot + h + vgap
    for i, v in enumerate(VARIANTS):
        c = umap[v]
        x = x0 + i * (w + gap)
        ax_t = axes_in(fig, x, y_top, w, h)
        draw_umap(ax_t, c, c["source"].map(sc).tolist())
        ax_b = axes_in(fig, x, y_bot, w, h)
        draw_umap(ax_b, c, c["subtype"].map(sub_map).tolist())
        d, asw = float(s.loc[v, "site_shift_norm"]), float(s.loc[v, "asw_batch"])
        ax_t.text(0.5, 1.36, CONDITION_LABELS[v], transform=ax_t.transAxes, ha="center", va="bottom",
                  fontsize=HEAD, fontweight="bold")
        ax_t.add_patch(Rectangle((0.0, 1.32), 1.0, 0.03, transform=ax_t.transAxes, color=CONDITION_COLORS[v],
                                 clip_on=False))
        ax_t.text(0.5, 1.04, f"Site-mean distance {d:.2f}\nASW (site) {asw:.3f}", transform=ax_t.transAxes,
                  ha="center", va="bottom", fontsize=SMALL, color=GREY, linespacing=1.15)
        if i == 0:
            ax_t.set_ylabel("By site", fontsize=BODY, fontweight="bold")
            ax_b.set_ylabel("By diagnosis", fontsize=BODY, fontweight="bold")
    lx = x0 + 3 * w + 2 * gap + 0.3
    ax_leg = axes_in(fig, lx, y_bot, SLIDE_W - lx - 0.1, 2 * h + vgap + 0.75); ax_leg.axis("off")
    leg1 = site_legend(ax_leg, cfg, loc="upper left", bbox_to_anchor=(0, 1.0))
    ax_leg.add_artist(leg1)
    diagnosis_legend(ax_leg, subs, sub_map, loc="upper left", bbox_to_anchor=(0, 0.87))
    footnote(fig, "Site-mean distance: between site means (TITAN). "
                  f"ASW (site): 0 = mixed; chance {chance:.3f}", x=x0, y=0.1)
    save_slide(fig, cfg, "03_correction_umap")


# ── 04 ペアの回復（樹形図）──────────────────────────────────────────────────────

def _band_fracs(ax_h: float, band_in: float, bottom_in: float) -> tuple[float, float]:
    """draw_dendrogram の band_frac / bottom_frac を、軸の高さ ax_h（in）でマーカー帯が band_in、その下が bottom_in に
    なるように決める（縦軸の範囲 = ymax × (1 + 0.55 band_frac + bottom_frac)）。"""
    tree_in = ax_h - 0.55 * band_in - bottom_in
    return band_in / tree_in, bottom_in / tree_in


def slide_pair_recovery(cfg: dict) -> dict[str, tuple[int, int]]:
    root = out_root(cfg)
    pairs_csv = pd.read_csv(root / "dendrogram" / "pairs.csv").set_index("variant")
    data = {v: load_tree(cfg, v) for v in VARIANTS}
    ymax = max(dendrogram.dendrogram_height(mat) for _, mat in data.values()) * 1.05
    n_sub = {groups["subtype"].nunique() for groups, _ in data.values()}
    assert len(n_sub) == 1
    n_sub = n_sub.pop()
    fig = new_slide("Pair recovery of subtypes across sites",
                    f"Ward clustering of the site × subtype mean embeddings ({n_sub} subtypes present at both sites)")
    x0, w = 0.75, 9.45
    row_h, ax_h, title_h = 1.97, 1.36, 0.36
    band_in, bottom_in = 0.34, 0.17
    band_frac, bottom_frac = _band_fracs(ax_h, band_in, bottom_in)
    top = SLIDE_H - 1.5
    result = {}
    for i, v in enumerate(VARIANTS):
        groups, mat = data[v]
        y = top - i * row_h - title_h - ax_h
        ax = axes_in(fig, x0, y, w, ax_h)
        k = dendrogram.draw_dendrogram(ax, mat, False, marker_specs(groups, cfg), ymax=ymax, marker_size=260,
                                       frame_pad=1.2, ytick_size=SMALL, band_frac=band_frac, bottom_frac=bottom_frac,
                                       line_lw=1.4)
        check_frames(ax)
        assert k == int(pairs_csv.loc[v, "adjacent_pairs"]), (v, k, pairs_csv.loc[v, "adjacent_pairs"])
        assert n_sub == int(pairs_csv.loc[v, "n_subtypes"])
        ax.set_yticks([0, 0.5, 1.0])
        t = ax.text(0.0, 1.06, CONDITION_LABELS[v], transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=HEAD, fontweight="bold")
        ax.annotate(f"   cross-site pairs: {k}/{n_sub}", xy=(1, 0), xycoords=t, ha="left", va="bottom", fontsize=HEAD)
        result[v] = (k, n_sub)
    # 凡例は右端の列: 施設のマーカー・点線の四角の意味、診断（全名）
    groups, _ = data[VARIANTS[-1]]
    subs = ordered_subtypes(set(groups["subtype"]), cfg)
    sub_map = subtype_color_map(subs, cfg)
    lx = x0 + w + 0.3
    ax_leg = axes_in(fig, lx, 0.15, SLIDE_W - lx - 0.05, top - 0.15); ax_leg.axis("off")
    mkr = src_markers(cfg, list(SITE_NAMES))
    handles = [Line2D([0], [0], marker=mkr[k], color="#555555", ls="none", markersize=12) for k in SITE_NAMES]
    handles.append(Rectangle((0, 0), 1, 1, fill=False, linestyle="--", edgecolor="#555555", linewidth=1.8))
    leg1 = ax_leg.legend(handles, [textwrap.fill(SITE_NAMES[k], 16) for k in SITE_NAMES]
                         + ["same subtype,\nboth sites adjacent"], loc="upper left", bbox_to_anchor=(0, 1.0),
                         borderaxespad=0, frameon=False, fontsize=SMALL, handlelength=1.2, handletextpad=0.5,
                         labelspacing=0.35)
    ax_leg.add_artist(leg1)
    diagnosis_legend(ax_leg, subs, sub_map, marker=None, loc="upper left", bbox_to_anchor=(0, 0.69))
    save_slide(fig, cfg, "04_pair_recovery")
    return result


# ── 05 まとめ ────────────────────────────────────────────────────────────────

def _bars(ax: plt.Axes, vals: list[float], fmt, title: str, ref: float | None = None, top: float | None = None) -> None:
    x = np.arange(len(VARIANTS))
    ax.bar(x, vals, color=[CONDITION_COLORS[v] for v in VARIANTS], width=0.68, edgecolor="none")
    top = top if top is not None else max(vals) * 1.22
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025 * top, fmt(v), ha="center", va="bottom", fontsize=BODY, fontweight="bold")
    if ref is not None:
        ax.axhline(ref, color=GREY, lw=1.3, ls="--")
    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(VARIANTS) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([textwrap.fill(CONDITION_LABELS[v], 11, break_long_words=False) for v in VARIANTS],
                       fontsize=SMALL)
    ax.set_title(title, fontsize=HEAD, pad=12)
    ax.spines[["top", "right"]].set_visible(False)


def slide_summary(cfg: dict, s: pd.DataFrame, chance: float, pairs: dict, n: dict[str, int]) -> None:
    ref = cfg["reference"]
    assert sum(n.values()) == int(s["n"].iloc[0])
    n_sub = {k for _, k in pairs.values()}
    assert len(n_sub) == 1
    n_sub = n_sub.pop()
    lab = CONDITION_LABELS
    fig = new_slide(f"Summary: {lab['original']} / {lab['gan']} / {lab['centroid']}",
                    f"{lab['gan']} (image level) reduces the site shift partly; {lab['centroid']} (one shift of the "
                    "embeddings) removes\nthe difference in site means and places more same-subtype pairs from the "
                    "two sites next to each other")
    w, gap, x0 = 3.45, 0.95, 0.95
    y0, h = 1.35, 3.6
    ax = [axes_in(fig, x0 + i * (w + gap), y0, w, h) for i in range(3)]
    _bars(ax[0], [float(s.loc[v, "site_shift_norm"]) for v in VARIANTS], "{:.2f}".format,
          "Distance between\nsite means ↓")
    _bars(ax[1], [float(s.loc[v, "asw_batch"]) for v in VARIANTS], "{:.3f}".format,
          "ASW (site) ↓\n(dashed = chance)", ref=chance)
    _bars(ax[2], [pairs[v][0] for v in VARIANTS], lambda k: f"{k}/{n_sub}",
          f"Cross-site subtype pairs\n(of {n_sub}) ↑", top=n_sub * 1.12)
    footnote(fig, f"{SITE_NAMES[GAN_SITE]} n = {n[GAN_SITE]}, {SITE_NAMES[ref]} n = {n[ref]} slides; "
                  "TITAN slide embeddings. Unpublished data.\n"
                  f"ASW (site): silhouette width by site, 0 = sites mixed (chance {chance:.3f}).",
             x=x0 - 0.5, y=0.1)
    save_slide(fig, cfg, "05_summary")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[slides_asno] ASNO slides (not a paper figure)")
    cfg = config()
    umap = load_umap(cfg)
    s, chance = site_metrics(cfg)
    slide_domain_shift(cfg, umap)
    slide_gan_patches(cfg)
    med = slide_gan_colour(cfg)
    slide_correction_umap(cfg, umap, s, chance)
    pairs = slide_pair_recovery(cfg)
    slide_summary(cfg, s, chance, pairs, site_counts(umap["original"]))
    print("\n  site metrics (= Fig 3f):")
    print(s[["site_shift_norm", "asw_batch", "ilisi", "adjacent_pairs"]].to_string())
    print(f"  ASW (site) chance: {chance:.4f}")
    print("  cross-site pairs (= Fig 5): " + ", ".join(f"{v} {k}/{n}" for v, (k, n) in pairs.items()))
    print("  colour medians:")
    print(med.round(4).to_string())


if __name__ == "__main__":
    main()
