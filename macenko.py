"""macenko.py — 色正規化（Macenko）を patho2 のパッチに適用し、CONCH でエンコードして HDF5 に書き込む。

GAN と同じ向き（対象施設 patho2 → 基準施設 EBRAINS）の画像レベルの補正。EBRAINS は動かさない。

手順:
    fit   基準施設（EBRAINS）の全スライドから無作為にパッチを取り、画素を束ねて Macenko 法で
          染色ベクトル（H・E の光学濃度の向き、3×2）と濃度の 99 パーセンタイル（2）を推定する = 正規化の目標。
          目標を 1 つに固定する（スライドごとに合わせない）。出力 output_dir/macenko/target.npz
    run   対象施設の各パッチについて、同じ Macenko 法でそのパッチの染色ベクトルと濃度の上限を推定し、
          濃度を目標の上限に合わせて目標の染色ベクトルで再合成する（パッチごとの正規化、Macenko 2009）。
          正規化したパッチを CONCH v1.5 でエンコードし {macenko_dir}/{stem}.h5 に書く（gan.py と同じ形式）。
          組織がほとんど無いパッチ（光学濃度が beta 以上の画素が min_pixels 未満）は推定できないので元のまま。
          染色ベクトルと濃度上限の推定は画素を stride で間引いたもので行い、変換は全画素に適用する。
    preview  正規化前後のパッチを並べた PNG（output_dir/macenko/preview.png）。目視確認用。
    color    正規化後の patho2 のスライドごとの平均色（paper_color.py と同じ定義・同じパッチの抽出）を
             output_dir/macenko/slide_color.csv に書き、paper_color.py の出力（EBRAINS / patho2 / patho2 GAN）と
             並べた分布 color_dist.png を作る。Fig 2b の入力（paper/.../color/slide_color.csv）は触らない。
             続けて colordist も実行する。
    colordist 各群の色の分布の reference からの距離（ΔE76・エネルギー距離・Wasserstein・IQR 比）を
             output_dir/macenko/color_distance.csv に書く（CSV だけ読むので軽い）。

定数は torchstain の Macenko 実装と同じ（Io = 240, alpha = 1, beta = 0.15）。

Usage:
    uv run python macenko.py fit
    uv run python macenko.py run
    uv run python macenko.py preview
    uv run python macenko.py all      # fit（target が無ければ）→ preview → run
"""
from __future__ import annotations

import argparse
import gc
import os
import time
import zlib
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

CONFIG_PATH = Path(__file__).parent / "config.yaml"

IO = 240.0      # 透過光の強度（背景の明るさ）
ALPHA = 1.0     # 染色ベクトルの角度のパーセンタイル（alpha, 100 − alpha）
BETA = 0.15     # 光学濃度の下限（これ未満の画素は背景として推定から除く）
MAXC_PCT = 99.0  # 濃度の上限のパーセンタイル


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── Macenko ─────────────────────────────────────────────────────────────────────

def _stain_params_np(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """束ねた画素 (N, 3) uint8 → 染色ベクトル HE (3, 2) と濃度の上限 maxC (2,)。fit 用（numpy）。"""
    od = -np.log((rgb.astype(np.float64) + 1) / IO)
    odhat = od[~np.any(od < BETA, axis=1)]
    _, eigvecs = np.linalg.eigh(np.cov(odhat.T))
    eigvecs = eigvecs[:, 1:3]
    that = odhat @ eigvecs
    phi = np.arctan2(that[:, 1], that[:, 0])
    min_phi, max_phi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
    v_min = eigvecs @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v_max = eigvecs @ np.array([np.cos(max_phi), np.sin(max_phi)])
    he = np.stack([v_min, v_max], axis=1) if v_min[0] > v_max[0] else np.stack([v_max, v_min], axis=1)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.percentile(conc, MAXC_PCT, axis=1)
    return he, max_c


def macenko_normalize(patches: torch.Tensor, he_ref: torch.Tensor, max_c_ref: torch.Tensor,
                      stride: int, min_pixels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """patches (B, H, W, 3) uint8 → 正規化後 (B, H, W, 3) uint8 と、正規化できたかのフラグ (B,) bool。

    パッチごとに染色ベクトルと濃度の上限を推定（画素は stride で間引く）、全画素を目標に合わせて再合成する。
    推定に使える画素が min_pixels 未満のパッチは元のまま返す。
    """
    B, H, W, _ = patches.shape
    od_full = -torch.log((patches.float() + 1) / IO)                      # (B, H, W, 3)
    od_sub = od_full[:, ::stride, ::stride].reshape(B, -1, 3)             # (B, P, 3)
    mask = ~(od_sub < BETA).any(dim=2)                                     # (B, P)
    n = mask.sum(dim=1)                                                    # (B,)
    ok = n >= min_pixels
    w = mask.float()
    n_safe = n.clamp(min=2).float()
    mean = (od_sub * w[..., None]).sum(dim=1) / n_safe[:, None]           # (B, 3)
    centered = (od_sub - mean[:, None, :]) * w[..., None]
    cov = centered.transpose(1, 2) @ centered / (n_safe - 1)[:, None, None]  # (B, 3, 3)
    cov = torch.where(ok[:, None, None], cov, torch.eye(3, device=cov.device).expand_as(cov))
    _, eigvecs = torch.linalg.eigh(cov)
    eigvecs = eigvecs[:, :, 1:3]                                           # (B, 3, 2)
    that = od_sub @ eigvecs                                                # (B, P, 2)
    phi = torch.atan2(that[..., 1], that[..., 0])
    phi = torch.where(mask, phi, torch.full_like(phi, float("nan")))
    q = torch.nanquantile(phi, torch.tensor([ALPHA / 100, 1 - ALPHA / 100], device=phi.device), dim=1)  # (2, B)
    min_phi, max_phi = q[0], q[1]
    v_min = (eigvecs @ torch.stack([torch.cos(min_phi), torch.sin(min_phi)], dim=1)[..., None])[..., 0]  # (B, 3)
    v_max = (eigvecs @ torch.stack([torch.cos(max_phi), torch.sin(max_phi)], dim=1)[..., None])[..., 0]
    swap = (v_min[:, 0] > v_max[:, 0])[:, None]
    h_vec = torch.where(swap, v_min, v_max)
    e_vec = torch.where(swap, v_max, v_min)
    he = torch.stack([h_vec, e_vec], dim=2)                                # (B, 3, 2)
    he = torch.where(ok[:, None, None], he, he_ref.expand_as(he))
    # 濃度（最小二乗、正規方程式）: 上限は間引いた画素で、変換は全画素で
    hth = he.transpose(1, 2) @ he                                          # (B, 2, 2)
    proj = torch.linalg.solve(hth, he.transpose(1, 2))                     # (B, 2, 3)
    conc_sub = proj @ od_sub.transpose(1, 2)                               # (B, 2, P)
    max_c = torch.quantile(conc_sub, MAXC_PCT / 100, dim=2)                # (B, 2)
    max_c = torch.where(ok[:, None], max_c, max_c_ref.expand_as(max_c)).clamp(min=1e-6)
    conc = proj @ od_full.reshape(B, -1, 3).transpose(1, 2)                # (B, 2, H*W)
    conc = conc / max_c[:, :, None] * max_c_ref[None, :, None]
    out = IO * torch.exp(-(he_ref[None] @ conc))                           # (B, 3, H*W)
    out = out.transpose(1, 2).reshape(B, H, W, 3).clamp(0, 255).round().to(torch.uint8)
    out = torch.where(ok[:, None, None, None], out, patches)
    return out, ok


# ── データ ─────────────────────────────────────────────────────────────────────

def _sample_indices(total: int, case_id: str, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + zlib.crc32(case_id.encode()))
    return sorted(rng.choice(total, size=min(n, total), replace=False).tolist())


class _ChunkDataset(Dataset):
    """(h5 パス, start, end) のチャンクを読む。gzip の展開を DataLoader の worker に分散させるため。"""

    def __init__(self, chunks: list[tuple[Path, int, int]], patches_key: str):
        self.chunks, self.patches_key = chunks, patches_key

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, i: int):
        path, start, end = self.chunks[i]
        with h5py.File(path, "r") as f:
            x = f[self.patches_key][start:end]
        return i, torch.from_numpy(x)


# ── fit ────────────────────────────────────────────────────────────────────────

def fit(cfg: dict, out_dir: Path) -> Path:
    mc = cfg["macenko"]
    size = mc.get("patch_size", 512)
    n_patches, stride, seed = mc.get("fit_patches", 16), mc.get("fit_stride", 8), mc.get("seed", 42)
    reference: str = cfg["reference"]
    emb_dir = Path(cfg["embedding"][reference]["original"])
    patches_key = f"cache/{size}/patches"
    files = sorted(emb_dir.glob("*.h5"))
    print(f"[fit] {reference}: {len(files)} slides × {n_patches} patches, stride {stride}")
    pixels, skipped = [], []
    t0 = time.time()
    for j, p in enumerate(files):
        with h5py.File(p, "r") as f:
            if patches_key not in f:      # パッチ画素が h5 に無いスライド（paper_patches.py は元 WSI へ戻るが、ここでは目標の推定から外す）
                skipped.append(p.stem)
                continue
            d = f[patches_key]
            idx = _sample_indices(d.shape[0], p.stem, n_patches, seed)
            x = d[idx][:, ::stride, ::stride].reshape(-1, 3)
        pixels.append(x)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{len(files)}  ({time.time() - t0:.0f}s)")
    rgb = np.concatenate(pixels)
    he, max_c = _stain_params_np(rgb)
    out = out_dir / "target.npz"
    np.savez(out, HE=he, maxC=max_c, n_pixels=len(rgb), n_slides=len(files) - len(skipped), n_patches=n_patches,
             stride=stride, seed=seed, io=IO, alpha=ALPHA, beta=BETA, maxc_pct=MAXC_PCT, reference=reference,
             skipped=np.array(skipped))
    print(f"[fit] slides used {len(files) - len(skipped)}/{len(files)}; skipped (no patch pixels in h5): {skipped}")
    print(f"[fit] HE =\n{he.round(4)}\n[fit] maxC = {max_c.round(4)}  (n_pixels={len(rgb):,})")
    print(f"[fit] saved: {out}")
    return out


def load_target(out_dir: Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    t = np.load(out_dir / "target.npz")
    return (torch.tensor(t["HE"], dtype=torch.float32, device=device),
            torch.tensor(t["maxC"], dtype=torch.float32, device=device))


# ── run ────────────────────────────────────────────────────────────────────────

def _encode(x_uint8: torch.Tensor, model, extract_fn, mean, std, batch_size: int) -> np.ndarray:
    """x_uint8 (N, H, W, 3) uint8 on device → features (N, D) float32（gan.py と同じ前処理）。"""
    device_type = "cuda" if x_uint8.is_cuda else "cpu"
    use_autocast = device_type == "cuda"
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_autocast else torch.float32
    feats = []
    for i in range(0, len(x_uint8), batch_size):
        x = (x_uint8[i:i + batch_size].float() / 255.0).permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        x = (x - mean) / std
        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=dtype, enabled=use_autocast):
            if extract_fn is not None:
                h = extract_fn(model, x)
            else:
                h = model.forward_features(x)[:, 0, ...]
        feats.append(h.float().cpu().numpy())
    return np.concatenate(feats)


def run(cfg: dict, out_dir: Path) -> None:
    from gan import _load_model

    mc = cfg["macenko"]
    size = mc.get("patch_size", 512)
    chunk = mc.get("chunk_size", 256)
    norm_bs = mc.get("norm_batch_size", 32)
    enc_bs = mc.get("encode_batch_size", 256)
    stride, min_pixels = mc.get("stride", 2), mc.get("min_pixels", 1000)
    workers = mc.get("num_workers", 6)
    overwrite = mc.get("overwrite", False)
    device_spec = mc.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_spec == "auto" else device_spec
    print(f"device: {device}")
    he_ref, max_c_ref = load_target(out_dir, device)
    keys = cfg["keys"]
    patches_key = f"cache/{size}/patches"
    reference: str = cfg["reference"]

    for ds_name, ds_cfg in cfg["embedding"].items():
        if ds_name == reference or "macenko" not in ds_cfg:
            continue
        raw_dir, dst_dir = Path(ds_cfg["original"]), Path(ds_cfg["macenko"])
        dst_dir.mkdir(parents=True, exist_ok=True)
        tile_model: str = ds_cfg["tile_model"]
        model, extract_fn, mean, std = _load_model(tile_model, device)
        feature_key, coord_key = f"{tile_model}/{keys['tile_features']}", f"{tile_model}/{keys['tile_coords']}"

        files = sorted(raw_dir.glob("*.h5"))
        todo, chunks, coords_of = [], [], {}
        for p in files:
            if (dst_dir / p.name).exists() and not overwrite:
                continue
            with h5py.File(p, "r") as f:
                n = f[patches_key].shape[0]
                c_cache, c_tile = f[f"cache/{size}/{keys['tile_coords']}"][:], f[coord_key][:]
                assert len(c_tile) == n and np.array_equal(c_cache, c_tile), f"{p.name}: coordinates mismatch"
                coords_of[p] = c_tile
            todo.append(p)
            chunks += [(p, s, min(s + chunk, n)) for s in range(0, n, chunk)]
        print(f"\n[{ds_name}] {len(files)} files, todo {len(todo)} ({len(chunks)} chunks)  ->  {dst_dir}")

        loader = DataLoader(_ChunkDataset(chunks, patches_key), batch_size=None, shuffle=False,
                            num_workers=workers, prefetch_factor=4 if workers else None)
        feats, oks, cur, n_done, t0 = [], [], None, 0, time.time()
        for i, x in loader:
            path, start, end = chunks[i]
            if cur is not None and path != cur:
                raise RuntimeError("chunk order broken")
            cur = path
            x = x.to(device, non_blocking=True)
            normed, ok = [], []
            for j in range(0, len(x), norm_bs):
                y, k = macenko_normalize(x[j:j + norm_bs], he_ref, max_c_ref, stride, min_pixels)
                normed.append(y)
                ok.append(k)
            y = torch.cat(normed)
            feats.append(_encode(y, model, extract_fn, mean, std, enc_bs))
            oks.append(torch.cat(ok).cpu().numpy())
            if end == len(coords_of[path]):   # ファイルの最後のチャンク
                features, okv = np.concatenate(feats), np.concatenate(oks)
                out_path = dst_dir / path.name
                tmp = out_path.with_suffix(".h5.tmp")
                with h5py.File(tmp, "w") as f:
                    grp = f.require_group(tile_model)
                    grp.attrs["patch_size"] = size
                    grp.attrs["preset"] = tile_model
                    grp.attrs["macenko_target"] = str(out_dir / "target.npz")
                    grp.attrs["macenko_fallback"] = int((~okv).sum())
                    f.create_dataset(feature_key, data=features)
                    f.create_dataset(coord_key, data=coords_of[path])
                    f.create_dataset(f"{tile_model}/macenko_ok", data=okv)
                os.replace(tmp, out_path)
                n_done += 1
                el = time.time() - t0
                print(f"  [{n_done}/{len(todo)}] {path.name}: N={len(features)} fallback={int((~okv).sum())}"
                      f"  ({el / 60:.1f} min, eta {el / n_done * (len(todo) - n_done) / 60:.0f} min)", flush=True)
                feats, oks, cur = [], [], None
        del model, mean, std
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


# ── preview ────────────────────────────────────────────────────────────────────

def preview(cfg: dict, out_dir: Path, n_slides: int = 4, n_per: int = 6) -> None:
    import matplotlib.pyplot as plt

    mc = cfg["macenko"]
    size, seed = mc.get("patch_size", 512), mc.get("seed", 42)
    stride, min_pixels = mc.get("stride", 2), mc.get("min_pixels", 1000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    he_ref, max_c_ref = load_target(out_dir, device)
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    patches_key = f"cache/{size}/patches"
    rng = np.random.default_rng(seed)
    rows = []
    for site in (other, reference):
        files = sorted(Path(cfg["embedding"][site]["original"]).glob("*.h5"))
        for p in rng.choice(files, size=n_slides, replace=False):
            with h5py.File(p, "r") as f:
                d = f[patches_key]
                idx = _sample_indices(d.shape[0], p.stem, n_per, seed)
                x = torch.from_numpy(d[idx]).to(device)
            y, ok = macenko_normalize(x, he_ref, max_c_ref, stride, min_pixels)
            rows.append((site, p.stem, x.cpu().numpy(), y.cpu().numpy(), ok.cpu().numpy()))
    fig, axes = plt.subplots(2 * len(rows), n_per, figsize=(1.6 * n_per, 1.7 * 2 * len(rows)))
    for r, (site, stem, x, y, ok) in enumerate(rows):
        for c in range(n_per):
            axes[2 * r, c].imshow(x[c]); axes[2 * r + 1, c].imshow(y[c])
            axes[2 * r + 1, c].set_title("" if ok[c] else "fallback", fontsize=7, color="red")
        axes[2 * r, 0].set_ylabel(f"{site}\n{stem}\noriginal", fontsize=7)
        axes[2 * r + 1, 0].set_ylabel("Macenko", fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Macenko normalization (target: {reference})", fontsize=9)
    fig.tight_layout()
    path = out_dir / "preview.png"
    fig.savefig(path, dpi=110)
    print(f"[preview] saved: {path}")


def color(cfg: dict, out_dir: Path) -> None:
    """正規化後の patho2 の平均色（paper_color.py と同じ定義）。GAN と同じくらい色が揃うかの定量。"""
    import matplotlib.pyplot as plt
    import pandas as pd
    from paper_color import sample_indices, slide_color
    from utils.display import CONDITION_COLORS

    mc, pc = cfg["macenko"], cfg["paper_color"]
    size = mc.get("patch_size", 512)
    stride, min_pixels = mc.get("stride", 2), mc.get("min_pixels", 1000)
    n, px_stride, thr, seed = pc.get("n_patches", 32), pc.get("stride", 4), pc.get("white_thr", 220), pc.get("seed", 42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    he_ref, max_c_ref = load_target(out_dir, device)
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    emb_dir, tile_model = Path(cfg["embedding"][other]["original"]), cfg["embedding"][other]["tile_model"]
    files = sorted(emb_dir.glob("*.h5"))
    rows, t0 = [], time.time()
    for j, p in enumerate(files):
        idx = sample_indices(p, tile_model, p.stem, n, seed)
        with h5py.File(p, "r") as f:
            x = torch.from_numpy(f[f"cache/{size}/patches"][idx]).to(device)
        y, ok = macenko_normalize(x, he_ref, max_c_ref, stride, min_pixels)
        rows.append({"group": f"{other} (Macenko)", "site": other, "case_id": p.stem, "n_patches": len(idx),
                     "n_fallback": int((~ok).sum()), **slide_color(y.cpu().numpy(), px_stride, thr)})
        if (j + 1) % 100 == 0:
            print(f"  {j + 1}/{len(files)}  ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "slide_color.csv", index=False)

    base = Path(cfg["output_dir"]) / "paper" / cfg["paper"]["version"] / "color" / "slide_color.csv"
    metrics = [("L", "L* (lightness)"), ("a", "a* (green–red)"), ("b", "b* (blue–yellow)"), ("H", "Hematoxylin"), ("E", "Eosin")]
    if base.exists():
        df = pd.concat([pd.read_csv(base), df], ignore_index=True)
    groups = [g for g in (reference, other, f"{other} (GAN)", f"{other} (Macenko)") if g in set(df["group"])]
    src_colors = cfg["display"]["colors"]["sources"]
    colors = {reference: src_colors[reference], other: src_colors[other],
              f"{other} (GAN)": CONDITION_COLORS["gan"], f"{other} (Macenko)": CONDITION_COLORS["macenko"]}
    print("\n群ごとの中央値:")
    print(df.groupby("group")[[m for m, _ in metrics]].median().loc[groups].round(3).to_string())
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.8 * len(metrics), 3.2))
    rng = np.random.default_rng(seed)
    for ax, (m, label) in zip(axes, metrics):
        for k, g in enumerate(groups):
            v = df.loc[df.group == g, m].to_numpy()
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=4, color=colors[g], alpha=0.45, linewidths=0)
            ax.hlines(np.median(v), k - 0.3, k + 0.3, color="black", lw=1.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace(" (", "\n(") for g in groups], fontsize=7)
        ax.set_title(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "color_dist.png", dpi=200)
    print(f"[color] saved: {out_dir / 'slide_color.csv'} / color_dist.png")


def color_distance(cfg: dict, out_dir: Path) -> None:
    """各群（patho2 / patho2 GAN / patho2 Macenko）の色の分布が reference（EBRAINS）にどれだけ近いか。

    スライドごとの平均色（paper_color.py / color の出力）から:
        dE76_median   中央値どうしの CIELAB 距離（ΔE*ab、CIE76）
        energy_Lab    L*a*b* 3 次元でのエネルギー距離（分布どうしの距離。位置と広がりの両方を見る）
        W1_<m>        指標ごとの 1 次元 Wasserstein 距離（スライドの分布どうし）
        IQRratio_<m>  指標ごとの四分位範囲の比（群 / reference。1 より小さいほど分布が細い）
    出力 output_dir/macenko/color_distance.csv
    """
    import pandas as pd
    from scipy.spatial.distance import cdist
    from scipy.stats import wasserstein_distance

    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    base = Path(cfg["output_dir"]) / "paper" / cfg["paper"]["version"] / "color" / "slide_color.csv"
    df = pd.concat([pd.read_csv(base), pd.read_csv(out_dir / "slide_color.csv")], ignore_index=True)
    ref = df[df.group == reference]
    cols = ["L", "a", "b", "H", "E"]

    def energy(X: np.ndarray, Y: np.ndarray) -> float:
        return float(np.sqrt(2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()))

    rows = []
    for g in (other, f"{other} (GAN)", f"{other} (Macenko)"):
        d = df[df.group == g]
        if d.empty:
            continue
        med_ref, med = ref[["L", "a", "b"]].median(), d[["L", "a", "b"]].median()
        r = {"group": g, "n": len(d), "dE76_median": float(np.sqrt(((med - med_ref) ** 2).sum())),
             "energy_Lab": energy(ref[["L", "a", "b"]].to_numpy(), d[["L", "a", "b"]].to_numpy())}
        for c in cols:
            r[f"W1_{c}"] = float(wasserstein_distance(ref[c], d[c]))
            iqr = lambda s: s.quantile(0.75) - s.quantile(0.25)
            r[f"IQRratio_{c}"] = float(iqr(d[c]) / iqr(ref[c]))
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "color_distance.csv", index=False)
    print(out.set_index("group").round(3).to_string())
    print(f"[colordist] saved: {out_dir / 'color_distance.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "run", "preview", "color", "colordist", "all"])
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "macenko"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "fit" or (args.cmd == "all" and not (out_dir / "target.npz").exists()):
        fit(cfg, out_dir)
    if args.cmd in ("preview", "all"):
        preview(cfg, out_dir)
    if args.cmd in ("run", "all"):
        run(cfg, out_dir)
    if args.cmd == "color":
        color(cfg, out_dir)
    if args.cmd in ("color", "colordist"):
        color_distance(cfg, out_dir)


if __name__ == "__main__":
    main()
