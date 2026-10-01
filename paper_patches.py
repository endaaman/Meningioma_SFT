"""paper_patches.py — 新 Fig 2 用の代表パッチ候補を機械的に選ぶ。

施設 × サブタイプごとに、パッチ埋め込み（CONCH, original h5 の {tile_model}/features）の「中心」を
  中心 = スライドごとの平均パッチ埋め込みを、スライド間で平均したもの（大きいスライドに引っ張られない）
として求め、中心にユークリッド距離で近いパッチを、同じ WSI・同じ患者からは 1 枚までの条件で
近い順に n_candidates 枚選ぶ（最も近いパッチの WSI が既出なら、その WSI 以外で次に近いもの）。
白画素率（wsi-toolbox の ptp 法: RGB の max−min < 20 を白画素）が max_white_ratio を超えるパッチは選ばない。
補正はかけず、施設ごとに別々に扱う（施設の色の違いをそのまま見せるため）。

patho2 の候補については、同じ座標の GAN 変換後パッチ（cache/{size}/gan/patches）も並べる。

出力 (output_dir/paper/{version}/patch_candidates/):
    {site}/{subtype}/rank{k}_{case}_x{X}_y{Y}.png
    {site}/{subtype}/contact_sheet.png     — 5 候補（症例 ID・距離つき）
    patho2_gan/{subtype}.png                — 上段: 元画像 / 下段: GAN 変換後
    candidates.csv                          — site, subtype, rank, case_id, patient_id, patch_index, x, y, distance, white_ratio
    slide_means.npz                         — スライド平均埋め込み（再計算を省くキャッシュ）

Usage:
    uv run python paper_patches.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
from PIL import Image

from wsi_toolbox.utils.white import _count_low_range_pixels

from utils.display import make_abbrev
from utils.loader import load_config


WHITE_RGB_RANGE = 20  # wsi-toolbox の is_white_patch_ptp の既定（RGB の max−min がこれ未満を白画素）


# ── ラベル・患者・WSI 索引 ─────────────────────────────────────────────────────

def load_subtypes(cfg: dict, site: str) -> pd.Series:
    df = pd.read_csv(cfg["label"][site]["subtype"])
    col = next(c for c in cfg.get("subtype_cols", ["subtype"]) if c in df.columns)
    return df.set_index("case_id")[col]


def load_patients(cfg: dict, site: str) -> dict[str, str]:
    """case_id → patient_id。patient CSV が無い施設は 1 症例 = 1 患者。"""
    path = cfg["label"][site].get("patient")
    if not path:
        return {}
    with open(path) as f:
        return {r["case_id"]: r["patient_id"] for r in csv.DictReader(f)}


def load_wsi_index(cfg: dict, site: str) -> dict[str, str]:
    """case_id → WSI パス（h5 にパッチ画素が無いときだけ使う）。"""
    idx_cfg = cfg["paper_patches"]["wsi_index"][site]
    df = pd.read_csv(idx_cfg["csv"])
    if "path_col" in idx_cfg:
        return dict(zip(df[idx_cfg["id_col"]], df[idx_cfg["path_col"]]))
    # EBRAINS: {root}/{diagnosis}/{uuid}.ndpi、case_id は uuid の先頭 8 桁
    root = Path(idx_cfg["root"])
    df = df.dropna(subset=["diagnosis"])  # 対照（diagnosis 空欄）は WSI のディレクトリが無い
    return {u[:8]: str(root / d / f"{u}.ndpi") for u, d in zip(df["uuid"], df["diagnosis"])}


# ── パッチ画素 ─────────────────────────────────────────────────────────────────

def read_patches(h5_path: Path, indices: list[int], tile_model: str, size: int,
                 wsi_path: str | None, gan: bool = False) -> np.ndarray:
    """h5 のパッチ画素（cache/{size}/patches）を読む。無ければ元 WSI から座標で読む。"""
    key = f"cache/{size}/gan/patches" if gan else f"cache/{size}/patches"
    with h5py.File(h5_path, "r") as f:
        if key in f:
            cache_coords = f[f"cache/{size}/coordinates"]
            model_coords = f[f"{tile_model}/coordinates"]
            for i in indices:
                if not (cache_coords[i] == model_coords[i]).all():
                    raise ValueError(f"{h5_path.name}: cache と {tile_model} の座標が index {i} で一致しない")
            return np.stack([f[key][i] for i in indices])
        if gan:
            raise KeyError(f"{h5_path.name}: {key} が無い")
        attrs = f[tile_model].attrs
        level, psize = int(attrs["level_used"]), int(attrs["patch_size"])
        coords = f[f"{tile_model}/coordinates"][sorted(indices)]
        coord_of = dict(zip(sorted(indices), coords))
    if wsi_path is None:
        raise FileNotFoundError(f"{h5_path.name}: パッチ画素も WSI パスも無い")
    slide = openslide.OpenSlide(wsi_path)
    ds = slide.level_downsamples[level]
    out = []
    for i in indices:
        x, y = coord_of[i]
        img = slide.read_region((int(x * ds), int(y * ds)), level, (psize, psize)).convert("RGB")
        out.append(np.asarray(img.resize((size, size)) if psize != size else img))
    return np.stack(out)


# ── 選出 ───────────────────────────────────────────────────────────────────────

def slide_means(cfg: dict, site: str, cases: list[str], cache: dict) -> dict[str, np.ndarray]:
    emb_dir = Path(cfg["embedding"][site]["original"])
    key = f"{cfg['embedding'][site]['tile_model']}/{cfg['keys']['tile_features']}"
    out = {}
    for c in cases:
        ck = f"{site}/{c}"
        if ck not in cache:
            with h5py.File(emb_dir / f"{c}.h5", "r") as f:
                cache[ck] = f[key][:].mean(axis=0)
        out[c] = cache[ck]
    return out


def white_ratio(patch: np.ndarray, rgb_range_threshold: int = WHITE_RGB_RANGE) -> float:
    """白画素率。wsi-toolbox の ptp 法と同じ画素判定（RGB の max−min < 20 を白画素）。"""
    return _count_low_range_pixels(patch, rgb_range_threshold) / (patch.shape[0] * patch.shape[1])


def select(cfg: dict, site: str, cases: list[str], patients: dict[str, str], cache: dict, k: int,
           max_white: float, wsi_index: dict[str, str]):
    """中心に近い順に、同一 WSI・同一患者からは 1 枚まで、白画素率 ≤ max_white のパッチを k 枚。

    全スライドの全パッチを中心からの距離で並べ、近い順に画素を読んで白画素率を確かめる
    （白すぎるパッチは飛ばし、その WSI の次に近いパッチが後で候補になる）。k 枚揃ったら止める。
    """
    means = slide_means(cfg, site, cases, cache)
    center = np.mean([means[c] for c in cases], axis=0)
    emb_dir = Path(cfg["embedding"][site]["original"])
    tile_model = cfg["embedding"][site]["tile_model"]
    key = f"{tile_model}/{cfg['keys']['tile_features']}"
    size = cfg["gan"]["patch_size"]
    dists, owner, index = [], [], []
    for j, c in enumerate(cases):
        with h5py.File(emb_dir / f"{c}.h5", "r") as f:
            d = np.linalg.norm(f[key][:] - center, axis=1)
        dists.append(d); owner.append(np.full(len(d), j)); index.append(np.arange(len(d)))
    dists, owner, index = np.concatenate(dists), np.concatenate(owner), np.concatenate(index)
    picked, used_patients, n_read = [], set(), 0
    for o in np.argsort(dists, kind="stable"):
        c = cases[owner[o]]
        pid = patients.get(c, c)
        if pid in used_patients:
            continue
        i = int(index[o])
        img = read_patches(emb_dir / f"{c}.h5", [i], tile_model, size, wsi_index.get(c))[0]
        n_read += 1
        wr = white_ratio(img)
        if wr > max_white:
            continue
        used_patients.add(pid)
        picked.append((float(dists[o]), c, i, wr, img))
        if len(picked) == k:
            break
    return picked, n_read


def contact_sheet(images: list[np.ndarray], titles: list[str], path: Path, rows: int = 1,
                  row_labels: list[str] | None = None) -> None:
    cols = len(images) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.8 * rows), squeeze=False)
    for ax, img, t in zip(axes.flat, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    if row_labels:
        for r, lab in enumerate(row_labels):
            axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    pp = cfg["paper_patches"]
    k, min_n = pp.get("n_candidates", 5), pp.get("min_n", 5)
    max_white = pp.get("max_white_ratio", 0.10)
    size = cfg["gan"]["patch_size"]
    out_dir = Path(cfg["output_dir"]) / "paper" / cfg["paper"]["version"] / "patch_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    sites = list(cfg["embedding"].keys())
    subtypes = {s: load_subtypes(cfg, s) for s in sites}
    for s in sites:  # h5 が存在する症例だけ
        emb_dir = Path(cfg["embedding"][s]["original"])
        subtypes[s] = subtypes[s][[ (emb_dir / f"{c}.h5").exists() for c in subtypes[s].index ]]
    counts = pd.DataFrame({s: subtypes[s].value_counts() for s in sites}).fillna(0)
    targets = [st for st in counts.index if (counts.loc[st] >= min_n).all()]
    abbrev = make_abbrev(targets, cfg)
    print(f"targets ({len(targets)}): {targets}")

    cache_path = out_dir / "slide_means.npz"
    cache = dict(np.load(cache_path)) if cache_path.exists() else {}
    rows = []
    for s in sites:
        patients = load_patients(cfg, s)
        wsi_index = load_wsi_index(cfg, s)
        emb_dir = Path(cfg["embedding"][s]["original"])
        tile_model = cfg["embedding"][s]["tile_model"]
        for st in targets:
            cases = list(subtypes[s][subtypes[s] == st].index)
            picked, n_read = select(cfg, s, cases, patients, cache, k, max_white, wsi_index)
            st_dir = out_dir / s / abbrev[st]
            st_dir.mkdir(parents=True, exist_ok=True)
            imgs, titles = [], []
            for rank, (dist, c, i, wr, img) in enumerate(picked, 1):
                with h5py.File(emb_dir / f"{c}.h5", "r") as f:
                    x, y = (int(v) for v in f[f"{tile_model}/coordinates"][i])
                Image.fromarray(img).save(st_dir / f"rank{rank}_{c}_x{x}_y{y}.png")
                imgs.append(img); titles.append(f"#{rank} {c}\nd={dist:.2f} white={wr:.0%}")
                rows.append({"site": s, "subtype": st, "rank": rank, "case_id": c,
                             "patient_id": patients.get(c, c), "patch_index": i, "x": x, "y": y,
                             "distance": round(dist, 4), "white_ratio": round(wr, 4)})
            contact_sheet(imgs, titles, st_dir / "contact_sheet.png")
            print(f"  {s:8} {abbrev[st]:5} n={len(cases):4}  read={n_read:4}  picked={[p[1] for p in picked]}")
    np.savez(cache_path, **cache)
    cand = pd.DataFrame(rows)
    cand.to_csv(out_dir / "candidates.csv", index=False)

    # patho2 候補の GAN 対（元 / 変換後）
    gan_site = next(s for s in sites if s != cfg.get("reference"))
    gan_dir = out_dir / f"{gan_site}_gan"
    gan_dir.mkdir(exist_ok=True)
    emb_dir = Path(cfg["embedding"][gan_site]["original"])
    tile_model = cfg["embedding"][gan_site]["tile_model"]
    for st in targets:
        sub = cand[(cand.site == gan_site) & (cand.subtype == st)]
        orig, conv, titles = [], [], []
        for r in sub.itertuples():
            orig.append(read_patches(emb_dir / f"{r.case_id}.h5", [r.patch_index], tile_model, size, None)[0])
            conv.append(read_patches(emb_dir / f"{r.case_id}.h5", [r.patch_index], tile_model, size, None, gan=True)[0])
            titles.append(f"#{r.rank} {r.case_id}")
        contact_sheet(orig + conv, titles + [f"{t} (GAN)" for t in titles], gan_dir / f"{abbrev[st]}.png",
                      rows=2, row_labels=["original", "GAN"])
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
