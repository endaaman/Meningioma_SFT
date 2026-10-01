"""paper_color.py — 新 Fig 2c 用: スライドごとの組織の平均色（CIELAB と H/E 染色強度）。

スライドごとにパッチを無作為に n_patches 枚（seed 固定）取り、背景（R,G,B すべて > white_thr）を除いた
組織画素の平均を求める。画素は paper_patches.py と同じ読み方（h5 の cache/{size}/patches、無ければ元 WSI）。

群:
    reference 施設（EBRAINS）の元画像 / もう一方（patho2）の元画像 / patho2 の GAN 変換後（cache/{size}/gan/patches）

色空間の変換は skimage と同じ定義を numpy で実装している:
    CIELAB  … sRGB → 線形 RGB → XYZ（D65）→ L*a*b*
    H / E   … Ruifrok & Johnston の色分解（skimage.color.rgb2hed と同じ行列・同じ対数変換）

出力 (output_dir/paper/{version}/color/):
    slide_color.csv   — group, site, case_id, n_patches, n_pixels, L, a, b, H, E
    color_dist.png    — 群ごとの分布（L*, a*, b*, H, E）

Usage:
    uv run python paper_color.py
"""
from __future__ import annotations

import zlib
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_patches import load_subtypes, load_wsi_index, read_patches
from utils.display import CONDITION_COLORS
from utils.loader import load_config


# ── 色空間（skimage と同じ定義）──────────────────────────────────────────────────

_XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
_RGB_FROM_HED = np.array([[0.65, 0.70, 0.29],
                          [0.07, 0.99, 0.11],
                          [0.27, 0.57, 0.78]])
_HED_FROM_RGB = np.linalg.inv(_RGB_FROM_HED)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) in [0, 1] → (N, 3) L*a*b*。"""
    lin = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = lin @ _XYZ_FROM_RGB.T / _WHITE_D65
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def rgb_to_hed(rgb: np.ndarray) -> np.ndarray:
    """rgb: (N, 3) in [0, 1] → (N, 3) H, E, D の染色強度（skimage.color.rgb2hed と同じ）。"""
    log_adjust = np.log(1e-6)
    stains = (np.log(np.maximum(rgb, 1e-6)) / log_adjust) @ _HED_FROM_RGB
    return np.maximum(stains, 0)


# ── 集計 ───────────────────────────────────────────────────────────────────────

def slide_color(patches: np.ndarray, stride: int, white_thr: int) -> dict:
    px = patches[:, ::stride, ::stride].reshape(-1, 3)
    px = px[~(px > white_thr).all(axis=1)].astype(np.float64) / 255
    lab = rgb_to_lab(px).mean(axis=0)
    hed = rgb_to_hed(px).mean(axis=0)
    return {"n_pixels": len(px), "L": lab[0], "a": lab[1], "b": lab[2], "H": hed[0], "E": hed[1]}


def sample_indices(h5_path: Path, tile_model: str, case_id: str, n: int, seed: int) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        total = len(f[f"{tile_model}/coordinates"])
    rng = np.random.default_rng(seed + zlib.crc32(case_id.encode()))
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


def main() -> None:
    cfg = load_config()
    pc = cfg["paper_color"]
    n, stride, thr, seed = pc.get("n_patches", 32), pc.get("stride", 4), pc.get("white_thr", 220), pc.get("seed", 42)
    size = cfg["gan"]["patch_size"]
    out_dir = Path(cfg["output_dir"]) / "paper" / cfg["paper"]["version"] / "color"
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = cfg["reference"]
    sites = list(cfg["embedding"].keys())
    gan_site = next(s for s in sites if s != reference)
    rows = []
    for s in sites:
        emb_dir = Path(cfg["embedding"][s]["original"])
        tile_model = cfg["embedding"][s]["tile_model"]
        wsi_index = load_wsi_index(cfg, s)
        cases = [c for c in load_subtypes(cfg, s).index if (emb_dir / f"{c}.h5").exists()]
        for j, c in enumerate(cases):
            h5_path = emb_dir / f"{c}.h5"
            idx = sample_indices(h5_path, tile_model, c, n, seed)
            groups = [(s, False)] + ([(f"{s} (GAN)", True)] if s == gan_site else [])
            for group, gan in groups:
                patches = read_patches(h5_path, idx, tile_model, size, wsi_index.get(c), gan=gan)
                rows.append({"group": group, "site": s, "case_id": c, "n_patches": len(idx),
                             **slide_color(patches, stride, thr)})
            if (j + 1) % 100 == 0:
                print(f"  {s}: {j + 1}/{len(cases)}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "slide_color.csv", index=False)

    groups = [reference, gan_site, f"{gan_site} (GAN)"]
    src_colors = cfg["display"]["colors"]["sources"]
    colors = [src_colors[reference], src_colors[gan_site], CONDITION_COLORS["gan"]]
    metrics = [("L", "L* (lightness)"), ("a", "a* (green–red)"), ("b", "b* (blue–yellow)"),
               ("H", "Hematoxylin"), ("E", "Eosin")]
    print("\n群ごとの中央値:")
    print(df.groupby("group")[[m for m, _ in metrics]].median().loc[groups].round(3).to_string())

    fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.2))
    rng = np.random.default_rng(seed)
    for ax, (m, label) in zip(axes, metrics):
        for k, (g, col) in enumerate(zip(groups, colors)):
            v = df.loc[df.group == g, m].to_numpy()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=col, alpha=0.45, linewidths=0)
            ax.hlines(np.median(v), k - 0.3, k + 0.3, color="black", lw=1.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(["EBRAINS", "patho2", "patho2\n(GAN)"] if reference == "ebrains" else groups, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "color_dist.png", dpi=200)
    plt.close(fig)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
