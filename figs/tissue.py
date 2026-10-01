"""figs.tissue — 組織パッチの一覧（a）。施設による色調の違いと GAN 変換を見せる。

列 = サブタイプ（COLUMNS の順）、行 = patho2（元画像）/ patho2 → GAN（同じパッチの変換後）/ EBRAINS（元画像）。
各セルのパッチは paper_patches.py が出した候補（施設 × サブタイプごとに中心に近い順の上位 5 枚）から、
PICK の順位番号で選ぶ。組み替えは PICK の番号を書き換えるだけでよい。

入力:
    patch_candidates/candidates.csv、patch_candidates/{site}/{abbr}/rank{k}_{case}_x{X}_y{Y}.png
    patho2 の original h5 の cache/{gan.patch_size}/gan/patches（GAN 変換後。座標は cache/{size}/coordinates と共有。読んだものは
    patch_candidates/patho2_gan_single/ にキャッシュ）

Usage:
    uv run python -m figs.tissue
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from PIL import Image

from figs.common import FONT, SITE_DISPLAY, config, panel_fig, paper_dir, save
from utils.display import make_abbrev, subtype_color_map

NAME = "tissue"

# 列（左から）。SFT は右端
COLUMNS = ["Meni", "Fibr", "Tran", "Atyp", "Angi", "Anap", "SFT"]
# 候補の順位（1〜5）を列順に。組み替えはここを書き換えるだけ（GAN 行は patho2 と同じパッチ）
PICK = {
    "patho2":  [1, 1, 1, 1, 1, 1, 1],
    "ebrains": [1, 1, 1, 1, 1, 1, 1],
}

GAN_SITE = "patho2"
SCALE_UM = 100  # スケールバーの長さ


def _rows() -> list[tuple[str, str, bool]]:
    """(行ラベル, 施設, GAN 変換後か)。"""
    return [
        (SITE_DISPLAY[GAN_SITE], GAN_SITE, False),
        (f"{SITE_DISPLAY[GAN_SITE]} → GAN", GAN_SITE, True),
        (SITE_DISPLAY["ebrains"], "ebrains", False),
    ]


def _candidate(cand: pd.DataFrame, site: str, subtype: str, rank: int) -> pd.Series:
    sel = cand[(cand.site == site) & (cand.subtype == subtype) & (cand["rank"] == rank)]
    if len(sel) != 1:
        raise KeyError(f"候補が無い: {site} / {subtype} / rank {rank}")
    return sel.iloc[0]


def _original(cand_dir: Path, site: str, abbr: str, r: pd.Series) -> np.ndarray:
    path = cand_dir / site / abbr / f"rank{r['rank']}_{r.case_id}_x{r.x}_y{r.y}.png"
    return np.asarray(Image.open(path).convert("RGB"))


def _gan(cfg: dict, cand_dir: Path, r: pd.Series) -> np.ndarray:
    """patho2 の original h5 から同じ patch_index の GAN 変換後パッチを読む（キャッシュあり）。"""
    cache = cand_dir / f"{GAN_SITE}_gan_single" / f"{r.case_id}_x{r.x}_y{r.y}.png"
    if cache.exists():
        return np.asarray(Image.open(cache).convert("RGB"))
    size = cfg["gan"]["patch_size"]
    h5_path = Path(cfg["embedding"][GAN_SITE]["original"]) / f"{r.case_id}.h5"
    with h5py.File(h5_path, "r") as f:
        # GAN パッチは cache/{size}/patches と同じ並び（座標は cache/{size}/coordinates を共有）
        xy = f[f"cache/{size}/coordinates"][int(r.patch_index)]
        if (int(xy[0]), int(xy[1])) != (int(r.x), int(r.y)):
            raise ValueError(f"{h5_path.name}: GAN パッチの座標 {tuple(xy)} が候補 ({r.x}, {r.y}) と一致しない")
        img = f[f"cache/{size}/gan/patches"][int(r.patch_index)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(cache)
    return img


def _mpp(cfg: dict, site: str, case_id: str) -> float:
    tile_model = cfg["embedding"][site]["tile_model"]
    with h5py.File(Path(cfg["embedding"][site]["original"]) / f"{case_id}.h5", "r") as f:
        return float(f[tile_model].attrs["mpp"])


def _scale_bar(ax: plt.Axes, mpp: float, size_px: int) -> None:
    length = SCALE_UM / mpp
    x0, y0 = size_px * 0.06, size_px * 0.92
    ax.plot([x0, x0 + length], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 - size_px * 0.04, f"{SCALE_UM} µm", color="white",
            ha="center", va="bottom", fontsize=FONT - 1, fontweight="bold")


def draw_patch_grid(fig: plt.Figure, gs, cfg: dict) -> None:
    """gs（行 3 × 列 len(COLUMNS) の GridSpec）にパッチを並べる。"""
    cand_dir = paper_dir(cfg) / "patch_candidates"
    cand = pd.read_csv(cand_dir / "candidates.csv")
    abbrev = make_abbrev(sorted(cand.subtype.unique()), cfg)
    full = {a: s for s, a in abbrev.items()}
    colors = subtype_color_map([full[a] for a in COLUMNS], cfg)

    for i, (row_label, site, gan) in enumerate(_rows()):
        for j, abbr in enumerate(COLUMNS):
            r = _candidate(cand, site, full[abbr], PICK[site][j])
            img = _gan(cfg, cand_dir, r) if gan else _original(cand_dir, site, abbr, r)
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(abbr, fontsize=FONT, pad=6)
                # 列見出しの下にサブタイプ色の細い帯
                ax.add_patch(plt.Rectangle((0, 1.01), 1, 0.035, transform=ax.transAxes,
                                           color=colors[full[abbr]], clip_on=False))
            if j == 0:
                ax.set_ylabel(row_label, fontsize=FONT)
            if i == 0 and j == 0:
                _scale_bar(ax, _mpp(cfg, site, r.case_id), img.shape[0])


def main() -> None:
    cfg = config()
    n_cols = len(COLUMNS)
    cell = 1.05  # inch
    fig = plt.figure(figsize=(cell * n_cols + 0.4, cell * 3 + 0.5))
    # b（色の定量）を下に足すときは、ここの行を増やして別の GridSpec を割り当てる
    gs = GridSpec(3, n_cols, figure=fig, left=0.06, right=0.995, top=0.90, bottom=0.01,
                  wspace=0.03, hspace=0.03)
    draw_patch_grid(fig, gs, cfg)
    panel_fig(fig, 0.0, 0.93, "a")
    save(fig, cfg, NAME)


if __name__ == "__main__":
    main()
