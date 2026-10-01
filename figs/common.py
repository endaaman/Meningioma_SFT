"""common.py — 論文図表の共通処理（config・出力先・スタイル・保存）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from figs import stem
from utils.display import CONDITION_COLORS, CONDITION_ORDER, order_conditions  # noqa: F401
from utils.loader import load_config

FONT = 8
plt.rcParams.update({
    "font.size": FONT, "axes.titlesize": FONT + 1, "axes.labelsize": FONT,
    "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})

SITE_DISPLAY = {"ebrains": "EBRAINS", "patho2": "patho2"}
VARIANT_DISPLAY = {"original": "No correction", "gan": "GAN", "centroid": "Centroid"}
VARIANT_COLOR = CONDITION_COLORS


def config() -> dict:
    return load_config()


def out_root(cfg: dict) -> Path:
    """解析結果の置き場（output_dir）。"""
    return Path(cfg["output_dir"])


def paper_dir(cfg: dict) -> Path:
    return out_root(cfg) / "paper" / cfg.get("paper", {}).get("version", "v1")


def fig_dir(cfg: dict) -> Path:
    return paper_dir(cfg) / "fig"


def table_dir(cfg: dict) -> Path:
    return paper_dir(cfg) / "tables"


def panel(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FONT + 4, fontweight="bold",
            va="bottom", ha="right")


def panel_fig(fig: plt.Figure, x: float, y: float, letter: str) -> None:
    """図座標でパネル記号を置く（等倍軸などで ax 基準だと揃わないとき）。"""
    fig.text(x, y, letter, fontsize=FONT + 4, fontweight="bold", va="bottom")


def save(fig: plt.Figure, cfg: dict, name: str) -> None:
    """図を fig/{番号付きの幹}.png / .pdf に保存する（name は figs/__init__.py に登録した内容名）。"""
    out_dir = fig_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = stem(name)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / base}.png / .pdf")


def fig_path(cfg: dict, name: str, suffix: str) -> Path:
    """図に付随するファイル（CSV 等）のパス。例: fig_path(cfg, "subtype_structure", "_pairs.csv")。"""
    fig_dir(cfg).mkdir(parents=True, exist_ok=True)
    return fig_dir(cfg) / f"{stem(name)}{suffix}"


def table_path(cfg: dict, name: str, ext: str) -> Path:
    table_dir(cfg).mkdir(parents=True, exist_ok=True)
    return table_dir(cfg) / f"{stem(name)}{ext}"


def src_markers(cfg: dict, sources: list[str]) -> dict[str, str]:
    syms = cfg.get("display", {}).get("markers", {}).get("sources", {})
    return {s: syms.get(s, "o") for s in sources}
