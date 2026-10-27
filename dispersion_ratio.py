"""dispersion_ratio.py — 施設間の「ばらつきの比」は腫瘍の種類をまたいで共有されるか（Supplementary 用、TITAN。--encoder meanpool で平均プーリング、--encoder prism で PRISM）。

中心のずれ（平行移動 Δ）が腫瘍をまたいでほぼ同じ向き（cos 0.8–0.9、shift_transfer.py）なのに対し、ComBat・相似変換が使う
「施設ごとのばらつきの比」が腫瘍をまたいで同じかを見る。比が腫瘍で違えば、ばらつきの比には施設の性質（色の幅など）と
コホートの中身の多様さ（腫瘍の性質）が混ざっていて、別の腫瘍に移せない（= 転移テストで倍率が移らなかった理由）。

科 f ごとに、次元 d の分散比  r_f,d = log( var_EBRAINS(f, d) / var_patho2(f, d) )  を全次元分（TITAN・meanpool 768、PRISM 1280）出し、
    1. 科どうしの r の相関（Pearson、全次元）  ← 平均のずれの cos（shift_transfer の cos 行列）と並べる
    2. 科の中の分割半分の再現性（施設ごとに無作為に半分に分け、r を 2 つ作って相関。何度か繰り返して中央値）
       ← 科どうしの相関がこれより十分低ければ、違いはノイズではなく腫瘍による
    3. 科ごとの全体の倍率 s_f（affine_free と同じ RMS 比）と、次元ごとの比のばらつき（sd of r）

入力: shift_transfer.py のキャッシュ（out/shift_transfer/{paper,glioma}_slides.csv / _embeddings.npy）。無ければ先に shift_transfer.py を回す。
    --encoder meanpool（CONCH v1.5 のパッチ埋め込みの単純平均、学習なしの集約器の対照）は shift_transfer.py --encoder meanpool の
    キャッシュ（out/shift_transfer/meanpool/）。スライドの集合と並びは TITAN と同じなので、分割半分の乱数の分け方も同じ
    --encoder prism（Virchow v1 → PRISM、1280 次元）は shift_transfer.py --encoder prism のキャッシュ（out/shift_transfer/prism/）。
    スライドの集合と並びは TITAN と同じ
出力 (output_dir/dispersion_ratio/、--encoder meanpool / prism は各ファイル名の末尾に _meanpool / _prism):
    corr_matrix.csv      — 科どうしの r の相関（上三角）と、分割半分の再現性（対角）、平均のずれの cos（下三角、参考）
    corr_matrix_corrected.csv — r の相関を分割半分の再現性で割った値（推定ノイズを差し引いた相関）
    summary.csv          — 科ごとの n、s_f、mean / sd of r、分割半分の再現性
    dispersion_ratio.png — a: 相関行列（r） b: 平均のずれの cos 行列 c: 科ごとの r のヒストグラム

Usage:
    uv run python dispersion_ratio.py [--encoder titan|meanpool|prism]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.loader import load_config

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
SITES = ("ebrains", "patho2")
N_SPLIT = 50
SEED = 42


def load_cache(d: Path) -> pd.DataFrame:
    frames = []
    for name in ("paper", "glioma"):
        df = pd.read_csv(d / f"{name}_slides.csv", dtype=str)
        df["embedding"] = list(np.load(d / f"{name}_embeddings.npy").astype(np.float64))
        if name == "paper":
            df["family"] = np.where(df["subtype"].str.lower().str.contains("sft"), "SFT", "meningioma")
        else:
            df["family"] = df["group"].map({"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"})
        frames.append(df[["source", "family", "embedding"]])
    return pd.concat(frames, ignore_index=True)


def log_var_ratio(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.log(E.var(0, ddof=1)) - np.log(P.var(0, ddof=1))


def rms_scale(E: np.ndarray, P: np.ndarray) -> float:
    rms = lambda Z: np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean())  # noqa: E731
    return float(rms(E) / rms(P))


def split_half_reliability(E: np.ndarray, P: np.ndarray, rng: np.random.Generator, n: int = N_SPLIT) -> float:
    vals = []
    for _ in range(n):
        ie, ip = rng.permutation(len(E)), rng.permutation(len(P))
        r1 = log_var_ratio(E[ie[: len(E) // 2]], P[ip[: len(P) // 2]])
        r2 = log_var_ratio(E[ie[len(E) // 2:]], P[ip[len(P) // 2:]])
        vals.append(np.corrcoef(r1, r2)[0, 1])
    return float(np.median(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["titan", "meanpool", "prism"], default="titan",
                    help="スライド埋め込み。meanpool = CONCH のパッチ平均、prism = Virchow → PRISM（shift_transfer.py --encoder <encoder> のキャッシュ）")
    args = ap.parse_args()
    cfg = load_config()
    cache = Path(cfg["output_dir"]) / "shift_transfer"
    if args.encoder != "titan":
        cache = cache / args.encoder
    sfx = "" if args.encoder == "titan" else f"_{args.encoder}"   # TITAN は前と同じファイル名
    tag = {"titan": "", "meanpool": " [mean-pooled CONCH v1.5]", "prism": " [PRISM]"}[args.encoder]
    out_dir = Path(cfg["output_dir"]) / "dispersion_ratio"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[encoder] {args.encoder}  cache: {cache}")
    df = load_cache(cache)
    rng = np.random.default_rng(SEED)

    X = {f: {s: np.stack(df.loc[(df["family"] == f) & (df["source"] == s), "embedding"].values) for s in SITES} for f in FAMILIES}
    r = {f: log_var_ratio(X[f]["ebrains"], X[f]["patho2"]) for f in FAMILIES}
    delta = {f: X[f]["ebrains"].mean(0) - X[f]["patho2"].mean(0) for f in FAMILIES}
    rows = []
    for f in FAMILIES:
        rows.append(dict(family=f, n_ebrains=len(X[f]["ebrains"]), n_patho2=len(X[f]["patho2"]),
                         scale_s=rms_scale(X[f]["ebrains"], X[f]["patho2"]),
                         mean_log_ratio=float(r[f].mean()), sd_log_ratio=float(r[f].std()),
                         frac_dims_ebrains_wider=float((r[f] > 0).mean()),
                         split_half_r=split_half_reliability(X[f]["ebrains"], X[f]["patho2"], rng)))
    summary = pd.DataFrame(rows)

    n = len(FAMILIES)
    C = np.full((n, n), np.nan)   # 上三角: r の相関、下三角: Δ の cos、対角: 分割半分の再現性
    for i, a in enumerate(FAMILIES):
        for j, b in enumerate(FAMILIES):
            if i < j:
                C[i, j] = np.corrcoef(r[a], r[b])[0, 1]
            elif i > j:
                C[i, j] = float(delta[a] @ delta[b] / np.linalg.norm(delta[a]) / np.linalg.norm(delta[b]))
            else:
                C[i, j] = summary.loc[summary["family"] == a, "split_half_r"].item()
    corr = pd.DataFrame(C, index=FAMILIES, columns=FAMILIES)
    corr.to_csv(out_dir / f"corr_matrix{sfx}.csv")
    # 分割半分の再現性で割って、推定ノイズを差し引いた科どうしの相関（Spearman の減衰補正）。再現性が低い科（SFT など）では当てにならない
    rel = summary.set_index("family")["split_half_r"]
    Cc = np.full((n, n), np.nan)
    for i, a in enumerate(FAMILIES):
        for j, b in enumerate(FAMILIES):
            if i < j and rel[a] > 0 and rel[b] > 0:
                Cc[i, j] = Cc[j, i] = C[i, j] / np.sqrt(rel[a] * rel[b])
    corr_c = pd.DataFrame(Cc, index=FAMILIES, columns=FAMILIES)
    corr_c.to_csv(out_dir / f"corr_matrix_corrected{sfx}.csv")
    summary.to_csv(out_dir / f"summary{sfx}.csv", index=False)

    off = [C[i, j] for i in range(n) for j in range(n) if i < j]
    print("[summary]\n" + summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[matrix] upper = corr of per-dim log variance ratio, lower = cos of mean shift, diag = split-half reliability of the ratio")
    print(corr.round(3).to_string())
    print(f"\ncorr of variance-ratio pattern across families: median {np.median(off):.3f} (range {min(off):.3f}–{max(off):.3f}); "
          f"split-half within family: {summary['split_half_r'].round(3).tolist()}")
    print("\n[corrected for split-half reliability] (only families with reliability > 0.4 are trustworthy)")
    print(corr_c.round(2).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.3]}, constrained_layout=True)
    for ax, title, M in ((axes[0], "per-dimension variance ratio: corr across families" + tag + "\n(diag: split-half reliability within family)", np.triu(C) + np.triu(C, 1).T),
                         (axes[1], "mean shift Δ: cos across families" + tag, np.tril(C, -1) + np.tril(C, -1).T + np.eye(n))):
        im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.set_xticklabels(FAMILIES, rotation=45, ha="right"); ax.set_yticklabels(FAMILIES)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k" if 0.25 < M[i, j] < 0.85 else "w")
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes[:2], fraction=0.02, pad=0.02)
    ax = axes[2]
    for f in FAMILIES:
        ax.hist(r[f], bins=60, histtype="step", lw=1.3, label=f"{f} (s={summary.loc[summary['family'] == f, 'scale_s'].item():.2f})")
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("log( var EBRAINS / var patho2 ) per dimension"); ax.set_ylabel("dimensions")
    ax.set_title("dispersion ratio per dimension, by family" + tag, fontsize=9); ax.legend(fontsize=8)
    fig.savefig(out_dir / f"dispersion_ratio{sfx}.png", dpi=150); plt.close(fig)
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
