"""transfer_color.py — 転移テストの全 2,301 枚の色の統計と、科ごとの「色の幅」「施設軸の H 応答」（TITAN）。

問い（ken の仮説 2026-10-26）: 施設間のばらつきの比（埋め込みの倍率 s = RMS(EBRAINS) / RMS(patho2)）が腫瘍の科で違うのは、
EBRAINS の色の幅が patho2 より広く、それがコホートの中身の多様さと混ざるからではないか。中心のずれ（Δ）は科をまたいで共有される。
→ 科 × 施設ごとに色の統計を取り、(a) 科ごとの s、(b) 施設軸 Δ_paper への射影の色への応答、と突き合わせる。
  stain_response.py は髄膜腫+SFT コホートだけだったので、それを 5 科に広げる。

build（スライドごとの色）:
    out/transfer/manifest.csv の全スライドについて、h5（h5_titan）の conch15_768/coordinates から 32 パッチを無作為に取る
    （paper_color.sample_indices と同じ: seed + crc32(slide_id)。髄膜腫+SFT コホートでは paper_color.py と同じパッチになる）。
    画素は paper_patches.read_patches と同じ読み方（h5 に cache/512/patches があればそれ、無ければ元 WSI（manifest の wsi）から
    level_used・patch_size（≈ 0.45 mpp、512 px）で読む）。背景の規則は paper_color.py と同じ（stride 4 に間引き、R,G,B すべて
    > white_thr の画素を除く）。組織画素が MIN_TISSUE（25%）未満のパッチは捨て、残りの座標から引き直す（別の乱数列、
    読むのは最大 MAX_READ_FACTOR × 32 枚まで）。
    スライドの平均色は paper_color.slide_color（組織画素をまとめた平均: L*a*b* と Ruifrok–Johnston の H / E）、
    スライド内のばらつきは「パッチごとの平均」のパッチ間 SD（sd_L, sd_H, sd_E）。
    out/transfer_color/slide_color.csv に 1 枚ずつ追記する（既にある slide_id は飛ばす = 途中から再開できる）。

analyze:
    埋め込みは transfer_correction.py のキャッシュ（out/transfer/embeddings_titan.npy、並びは embeddings_titan_slides.csv で
    manifest と一致を確認）。Δ_paper = mean(EBRAINS の髄膜腫+SFT コホート) − mean(patho2 の髄膜腫+SFT コホート)（shift_transfer.py と同じ、
    |Δ_paper| = 5.474）、u = Δ_paper / |Δ_paper|。科は manifest の family（meningioma / SFT / glioma / lymphoma / metastasis）。
    髄膜腫+SFT コホート（paper = meningioma + SFT）は参照の行として出す（5 科の相関には入れない）。

    1. 色の幅（family_color.csv）: 科 × 施設ごとの L, a, b, H, E のスライド間の平均と SD、施設平均色の ΔE76（Lab のユークリッド距離）、
       色の幅の比 sd_EBRAINS / sd_patho2（H, E, L と、3 次元の Lab の幅 = 施設平均色からの RMS 距離）、スライド内のばらつきの平均。
       外れ値に強い版として IQR の比（H, E, L）と、Lab の頑健な幅（施設の中央値色からの距離の中央値）の比も出す
       （patho2 の H には少数の極端に濃いスライドがあり、SD 比はそれに引っ張られる）。
       隣に埋め込みの倍率 s（affine.scale_free と同じ RMS 比）、科ごとの平均のずれ |Δ_f| と u 方向の成分・cos(Δ_f, Δ_paper)。
       s_colour_resid = 科 × 施設ごとに埋め込みを色 5 つ（H, E, L, a, b）で回帰した残差の RMS 比（自由度で補正: 残差は
       SSR/(n−6)、全体は SST/(n−1)）。色の幅が s を作っているなら、色を除くと s は 1 に近づくはず。
       比と s にはスライド単位のブートストラップ（施設ごとに復元抽出、N_BOOT 回）の 95% CI を付ける。
       5 科をまたぐ相関（色の幅の比 vs s、Pearson / Spearman）は 5 点しかないので、出すだけで読み込まない。
    2. 施設軸の H 応答（stain_response_by_family.csv）: p = ⟨x − μ_{科, 施設}, u⟩（科 × 施設の中で中心化）と色の相関を
       科 × 施設ごとに。raw と within_group（施設内の群 = manifest の group の平均で両者を残差化。stain_response.residualize、
       例数 MIN_N_GROUP 未満の群は落とす）。site = both は両施設の中心化した値をまとめたもの。軸は u（paper）と、
       参考に科自身のずれの向き u_f（own）。
    3. 色で説明できる施設差（explained_by_family.csv）: 科ごとに stain_response.regression をそのまま当てる
       （s_raw = ⟨x − μ_{科, patho2}, u⟩ ~ site [+ 標準化した H, E, L, a, b]、site_gap_explained_by_colour = 1 − 調整後 / 調整前）。
    受け入れ確認: 髄膜腫+SFT コホートの色が paper_color.py の slide_color.csv と一致するか（パッチを捨てなかったスライド）、
    |Δ_paper| = 5.474、paper の行の相関・説明割合が stain_response の summary（TITAN）と同じ値になるか。

出力 (output_dir/transfer_color/):
    slide_color.csv              — slide_id, site, family, group, n_patches, n_pixels, L, a, b, H, E, sd_L, sd_H, sd_E,
                                   n_rejected（捨てたパッチ数）, pixels（h5_cache / wsi）
    build_errors.csv             — 読めなかったスライド（あれば）
    slide_projection.csv         — スライドごとの色と射影（p_paper, p_own, s_raw_paper, s_raw_own）
    family_color.csv             — 1 の表（科ごと 1 行、両施設の列）
    stain_response_by_family.csv — 2 の相関（縦長）
    explained_by_family.csv      — 3 の回帰
    summary.md                   — 主要な表と読み（日本語）
    transfer_color.png           — a: H・E の分布（科 × 施設）、b: 色の幅の比（Lab の幅・H の IQR）vs s（科ごと、95% CI）、
                                   c: Δ_paper への射影 vs H（科ごと、施設ごとの回帰直線と群内 r）

Usage:
    uv run python transfer_color.py build [--workers 4] [--limit N]
    uv run python transfer_color.py analyze
"""
from __future__ import annotations

import argparse
import csv
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from scipy import stats  # noqa: E402

from affine import scale_free  # noqa: E402
from paper_color import rgb_to_hed, rgb_to_lab, sample_indices, slide_color  # noqa: E402
from paper_patches import read_patches  # noqa: E402
from stain_response import _corr, regression, residualize  # noqa: E402
from utils.loader import load_config, paper_dir  # noqa: E402

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
FAMILY_LABEL = {"meningioma": "Meningioma", "SFT": "SFT", "glioma": "Glioma", "lymphoma": "Lymphoma",
                "metastasis": "Metastasis", "paper": "Meningioma + SFT cohort"}
PAPER = {"meningioma", "SFT"}          # 髄膜腫+SFT コホート（Δ_paper の推定元）
SITES = ("ebrains", "patho2")
SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}
TILE_MODEL = "conch15_768"
PATCH_SIZE = 512                       # h5 の cache/512/patches と同じ（WSI からも patch_size = 512 で読む）
MIN_TISSUE = 0.25                      # 組織画素の割合がこれ未満のパッチは捨てて引き直す
MAX_READ_FACTOR = 3                    # 1 枚あたり読むパッチは最大 n_patches × これ
COLOR5 = ["H", "E", "L", "a", "b"]
FEATURES = ["H", "E", "he_balance", "L", "a", "b"]   # stain_response.COLOR_FEATURES と同じ並び
MIN_N_GROUP = 3                        # within_group の残差化に使う群の最小例数（stain_response の min_n_subtype と同じ）
N_BOOT = 1000
SEED = 42
EXPECTED_SHIFT_NORM = 5.4743           # out/stain_response/summary.md の titan shift_norm
SLIDE_COLS = ["slide_id", "site", "family", "group", "n_patches", "n_pixels", "L", "a", "b", "H", "E",
              "sd_L", "sd_H", "sd_E", "n_rejected", "pixels"]


# ── build ─────────────────────────────────────────────────────────────────────

def _tissue_mask(patches: np.ndarray, stride: int, white_thr: int) -> np.ndarray:
    """(n, h, w, 3) → (n, h', w') の組織画素（paper_color.slide_color と同じ間引きと背景の規則）。"""
    return ~(patches[:, ::stride, ::stride] > white_thr).all(axis=-1)


def _patch_means(patches: np.ndarray, stride: int, white_thr: int) -> np.ndarray:
    """パッチごとの組織画素の平均 L, H, E（(n, 3)）。スライド内のばらつき用。"""
    out = []
    for p in patches:
        px = p[::stride, ::stride].reshape(-1, 3)
        px = px[~(px > white_thr).all(axis=1)].astype(np.float64) / 255
        out.append([rgb_to_lab(px)[:, 0].mean(), *rgb_to_hed(px)[:, :2].mean(axis=0)])
    return np.asarray(out)


def color_one(row: dict, n: int, stride: int, white_thr: int, seed: int) -> dict:
    """1 枚のスライドの色。row は manifest の 1 行（slide_id, site, family, group, wsi, h5_titan）。"""
    h5_path = Path(row["h5_titan"])
    sid = row["slide_id"]
    first = sample_indices(h5_path, TILE_MODEL, sid, n, seed)
    with h5py.File(h5_path, "r") as f:
        total = len(f[f"{TILE_MODEL}/coordinates"])
        pixels = "h5_cache" if f"cache/{PATCH_SIZE}/patches" in f else "wsi"
    # 引き直しの順番: 最初の 32 枚に入らなかった座標を別の乱数列で並べる
    rest = np.setdiff1d(np.arange(total), first)
    rest = np.random.default_rng(seed + zlib.crc32(sid.encode()) + 1).permutation(rest).tolist()
    want, kept, n_read, n_rejected = min(n, total), [], 0, 0
    batch = first
    while batch:
        P = read_patches(h5_path, batch, TILE_MODEL, PATCH_SIZE, row["wsi"])
        n_read += len(batch)
        frac = _tissue_mask(P, stride, white_thr).mean(axis=(1, 2))
        for img, fr in zip(P, frac):
            if fr >= MIN_TISSUE and len(kept) < want:
                kept.append(img)
            elif fr < MIN_TISSUE:
                n_rejected += 1
        need = want - len(kept)
        budget = MAX_READ_FACTOR * n - n_read
        batch = sorted(rest[:min(need, budget)]) if need > 0 and budget > 0 else []
        rest = rest[len(batch):]
    if not kept:
        raise ValueError(f"{sid}: 組織 ≥ {MIN_TISSUE:.0%} のパッチが無い（{n_read} 枚を読んだ）")
    P = np.stack(kept)
    pm = _patch_means(P, stride, white_thr)
    sd = pm.std(axis=0, ddof=1) if len(pm) > 1 else np.full(3, np.nan)
    return {"slide_id": sid, "site": row["site"], "family": row["family"], "group": row["group"],
            "n_patches": len(P), **slide_color(P, stride, white_thr),
            "sd_L": sd[0], "sd_H": sd[1], "sd_E": sd[2], "n_rejected": n_rejected, "pixels": pixels}


def build(cfg: dict, out_dir: Path, workers: int, limit: int | None) -> None:
    pc = cfg.get("paper_color", {})
    n, stride, thr, seed = pc.get("n_patches", 32), pc.get("stride", 4), pc.get("white_thr", 220), pc.get("seed", SEED)
    man = pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"]["manifest"])
    out_csv, err_csv = out_dir / "slide_color.csv", out_dir / "build_errors.csv"
    done = set(pd.read_csv(out_csv, usecols=["slide_id"])["slide_id"].astype(str)) if out_csv.exists() else set()
    todo = man[~man["slide_id"].astype(str).isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"[build] manifest {len(man)} 枚、済み {len(done)}、今回 {len(todo)}（workers={workers}, n={n}, stride={stride}, "
          f"white_thr={thr}, min_tissue={MIN_TISSUE}）", flush=True)
    if todo.empty:
        return
    new_file = not out_csv.exists()
    t0 = time.time()
    n_ok = n_err = 0
    with open(out_csv, "a", newline="") as fo, ProcessPoolExecutor(max_workers=workers) as ex:
        w = csv.DictWriter(fo, fieldnames=SLIDE_COLS)
        if new_file:
            w.writeheader()
        cols = ["slide_id", "site", "family", "group", "wsi", "h5_titan"]
        futs = {ex.submit(color_one, r, n, stride, thr, seed): r["slide_id"] for r in todo[cols].to_dict("records")}
        for k, fut in enumerate(as_completed(futs), 1):
            sid = futs[fut]
            try:
                w.writerow(fut.result())
                fo.flush()
                n_ok += 1
            except Exception as e:  # noqa: BLE001  1 枚の失敗で全体を止めない
                n_err += 1
                with open(err_csv, "a", newline="") as fe:
                    csv.writer(fe).writerow([sid, repr(e)])
                print(f"  ERROR {sid}: {e!r}", flush=True)
            if k % 100 == 0 or k == len(futs):
                el = time.time() - t0
                print(f"  {k}/{len(futs)}  ok={n_ok} err={n_err}  {el / 60:.1f} 分経過、残り約 "
                      f"{el / k * (len(futs) - k) / 60:.0f} 分", flush=True)
    print(f"[build] 完了 ok={n_ok} err={n_err}  {(time.time() - t0) / 60:.1f} 分 → {out_csv}", flush=True)


# ── analyze: 読み込み ─────────────────────────────────────────────────────────

def load_all(cfg: dict, out_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    tdir = Path(cfg["output_dir"]) / "transfer"
    man = pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"]["manifest"], dtype={"slide_id": str})
    order = pd.read_csv(tdir / "embeddings_titan_slides.csv", dtype={"slide_id": str})
    X = np.load(tdir / "embeddings_titan.npy").astype(np.float64)
    if len(order) != len(man) or not (order[["site", "slide_id"]].values == man[["site", "slide_id"]].values).all():
        raise SystemExit("embeddings_titan の並びが manifest と合わない → transfer_correction.py --reload を先に回す")
    if len(X) != len(man):
        raise SystemExit(f"embeddings_titan.npy の行数 {len(X)} ≠ manifest {len(man)}")
    col = pd.read_csv(out_dir / "slide_color.csv", dtype={"slide_id": str}).drop_duplicates("slide_id", keep="last")
    df = man[["slide_id", "site", "family", "group"]].merge(
        col.drop(columns=["site", "family", "group"]), on="slide_id", how="left", validate="one_to_one")
    miss = int(df["H"].isna().sum())
    if miss:
        print(f"  WARN: 色の無いスライド {miss} 枚（解析から除く）")
    df["he_balance"] = np.log(df["H"] / df["E"])
    df["in_paper"] = df["family"].isin(PAPER)
    return df, X


def cohorts(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """科の名前 → manifest 上の bool マスク（paper = 髄膜腫 + SFT を最後に）。"""
    out = {f: (df["family"] == f).to_numpy() for f in FAMILIES}
    out["paper"] = df["in_paper"].to_numpy()
    return out


# ── analyze: 1. 色の幅と倍率 ──────────────────────────────────────────────────

def _lab_spread(Z: np.ndarray) -> float:
    """Lab の 3 次元の幅: 施設平均色からの RMS 距離。"""
    return float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))


def _rms(Z: np.ndarray) -> float:
    """施設平均まわりの RMS 距離（affine.scale_free の中と同じ。ブートストラップで行の添字を渡すために切り出した）。"""
    return float(np.sqrt(((Z - Z.mean(0)) ** 2).sum(1).mean()))


def _lab_spread_robust(Z: np.ndarray) -> float:
    """外れ値に強い Lab の幅: 施設の中央値色（成分ごと）からの距離の中央値。"""
    return float(np.median(np.sqrt(((Z - np.median(Z, axis=0)) ** 2).sum(1))))


def _iqr(v: np.ndarray) -> float:
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def _resid_rms(X: np.ndarray, C: np.ndarray) -> float:
    """埋め込み X を色 C（切片つき）で回帰した残差の RMS（自由度で補正: SSR/(n − p − 1)）。"""
    A = np.column_stack([np.ones(len(C)), (C - C.mean(0)) / C.std(0)])
    beta, *_ = np.linalg.lstsq(A, X, rcond=None)
    R = X - A @ beta
    return float(np.sqrt((R ** 2).sum() / (len(X) - A.shape[1])))


def _tot_rms(X: np.ndarray) -> float:
    return float(np.sqrt(((X - X.mean(0)) ** 2).sum() / (len(X) - 1)))


def family_table(df: pd.DataFrame, X: np.ndarray, delta_paper: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    u = delta_paper / np.linalg.norm(delta_paper)
    ok = df["H"].notna().to_numpy()
    site = df["site"].to_numpy()
    rows = []
    for name, m in cohorts(df).items():
        m = m & ok
        E, P = m & (site == "ebrains"), m & (site == "patho2")
        r: dict = {"family": name, "n_ebrains": int(E.sum()), "n_patho2": int(P.sum())}
        lab = {s_: df.loc[mm, ["L", "a", "b"]].to_numpy() for s_, mm in (("ebrains", E), ("patho2", P))}
        for s_, mm in (("ebrains", E), ("patho2", P)):
            d = df.loc[mm]
            for f in ["L", "a", "b", "H", "E"]:
                r[f"{f}_mean_{s_}"] = d[f].mean()
                r[f"{f}_sd_{s_}"] = d[f].std(ddof=1)
            r[f"lab_spread_{s_}"] = _lab_spread(lab[s_])
            r[f"lab_spread_robust_{s_}"] = _lab_spread_robust(lab[s_])
            for f in ["H", "E", "L"]:
                r[f"{f}_iqr_{s_}"] = _iqr(d[f].to_numpy())
            for f in ["sd_L", "sd_H", "sd_E"]:
                r[f"within_{f}_{s_}"] = d[f].mean()
        for f in ["H", "E", "L"]:
            r[f"sdratio_{f}"] = r[f"{f}_sd_ebrains"] / r[f"{f}_sd_patho2"]
            r[f"iqrratio_{f}"] = r[f"{f}_iqr_ebrains"] / r[f"{f}_iqr_patho2"]
        r["lab_spread_ratio"] = r["lab_spread_ebrains"] / r["lab_spread_patho2"]
        r["lab_spread_robust_ratio"] = r["lab_spread_robust_ebrains"] / r["lab_spread_robust_patho2"]
        r["dE76"] = float(np.linalg.norm(lab["ebrains"].mean(0) - lab["patho2"].mean(0)))
        # 埋め込みの倍率と、色を除いた倍率
        XE, XP = X[E], X[P]
        r["scale_s"] = scale_free(X, E, P)
        CE, CP = df.loc[E, COLOR5].to_numpy(), df.loc[P, COLOR5].to_numpy()
        r["scale_s_dfcorr"] = _tot_rms(XE) / _tot_rms(XP)
        r["scale_s_colour_resid"] = _resid_rms(XE, CE) / _resid_rms(XP, CP)
        r["r2_colour_ebrains"] = 1 - _resid_rms(XE, CE) ** 2 / _tot_rms(XE) ** 2
        r["r2_colour_patho2"] = 1 - _resid_rms(XP, CP) ** 2 / _tot_rms(XP) ** 2
        d_f = XE.mean(0) - XP.mean(0)
        r["shift_norm"] = float(np.linalg.norm(d_f))
        r["shift_along_u"] = float(d_f @ u)
        r["cos_paper"] = float(d_f @ u / np.linalg.norm(d_f))
        # スライド単位のブートストラップ（施設ごとに復元抽出）
        ie, ip = np.flatnonzero(E), np.flatnonzero(P)
        bs = {k: [] for k in ("sdratio_H", "sdratio_E", "sdratio_L", "iqrratio_H", "lab_spread_ratio",
                              "lab_spread_robust_ratio", "scale_s")}
        H, Ec, L = (df[f].to_numpy() for f in ("H", "E", "L"))
        LAB = df[["L", "a", "b"]].to_numpy()
        for _ in range(N_BOOT):
            be, bp = rng.choice(ie, len(ie)), rng.choice(ip, len(ip))
            bs["sdratio_H"].append(H[be].std(ddof=1) / H[bp].std(ddof=1))
            bs["sdratio_E"].append(Ec[be].std(ddof=1) / Ec[bp].std(ddof=1))
            bs["sdratio_L"].append(L[be].std(ddof=1) / L[bp].std(ddof=1))
            bs["iqrratio_H"].append(_iqr(H[be]) / _iqr(H[bp]))
            bs["lab_spread_ratio"].append(_lab_spread(LAB[be]) / _lab_spread(LAB[bp]))
            bs["lab_spread_robust_ratio"].append(_lab_spread_robust(LAB[be]) / _lab_spread_robust(LAB[bp]))
            bs["scale_s"].append(_rms(X[be]) / _rms(X[bp]))
        for k, v in bs.items():
            r[f"{k}_lo"], r[f"{k}_hi"] = np.percentile(v, [2.5, 97.5])
        rows.append(r)
    return pd.DataFrame(rows)


def cross_family_corr(fam: pd.DataFrame) -> pd.DataFrame:
    """5 科をまたぐ相関: 色の幅の比 vs s（と s_colour_resid、|Δ_f|）。5 点なので参考。"""
    f5 = fam[fam["family"].isin(FAMILIES)]
    rows = []
    for x in ["lab_spread_ratio", "lab_spread_robust_ratio", "sdratio_H", "iqrratio_H", "sdratio_E", "sdratio_L", "dE76"]:
        for y in ["scale_s", "scale_s_colour_resid", "shift_norm"]:
            a, b = f5[x].to_numpy(), f5[y].to_numpy()
            rows.append({"x": x, "y": y, "n": len(a), "pearson_r": stats.pearsonr(a, b)[0],
                         "pearson_p": stats.pearsonr(a, b)[1], "spearman_rho": stats.spearmanr(a, b)[0]})
    return pd.DataFrame(rows)


# ── analyze: 2–3. 施設軸への射影と色 ──────────────────────────────────────────

def projections(df: pd.DataFrame, X: np.ndarray, delta_paper: np.ndarray) -> pd.DataFrame:
    """科ごとに p = ⟨x − μ_{科, 施設}, u⟩ と s_raw = ⟨x − μ_{科, patho2}, u⟩。u は Δ_paper（paper）と科自身の Δ_f（own）。"""
    u_paper = delta_paper / np.linalg.norm(delta_paper)
    ok = df["H"].notna().to_numpy()
    site = df["site"].to_numpy()
    frames = []
    for name, m in cohorts(df).items():
        m = m & ok
        E, P = m & (site == "ebrains"), m & (site == "patho2")
        mu = {"ebrains": X[E].mean(0), "patho2": X[P].mean(0)}
        u_own = (mu["ebrains"] - mu["patho2"]) / np.linalg.norm(mu["ebrains"] - mu["patho2"])
        idx = np.flatnonzero(m)
        centre = np.stack([mu[s_] for s_ in site[idx]])
        d = df.iloc[idx][["slide_id", "site", "group", *FEATURES]].copy()
        d.insert(0, "cohort", name)
        for axis, u in (("paper", u_paper), ("own", u_own)):
            d[f"p_{axis}"] = (X[idx] - centre) @ u
            d[f"s_raw_{axis}"] = (X[idx] - mu["patho2"]) @ u
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def response_corr(proj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (coh, axis), _ in [((c, a), None) for c in proj["cohort"].unique() for a in ("paper", "own")]:
        g = proj[proj["cohort"] == coh].rename(columns={"group": "subtype", "slide_id": "case_id", f"p_{axis}": "s"})
        parts = {s_: g[g["site"] == s_] for s_ in SITES}
        res = {s_: residualize(d, ["s", *FEATURES], MIN_N_GROUP) for s_, d in parts.items()}
        # both: p は科 × 施設で中心化済み。色も施設内で中心化してから両施設をまとめる（施設の色の差で薄めないため）
        parts["both"] = pd.concat([d.assign(**{f: d[f] - d[f].mean() for f in FEATURES}) for d in
                                   (parts["ebrains"], parts["patho2"])], ignore_index=True)
        res["both"] = pd.concat([res["ebrains"], res["patho2"]], ignore_index=True)
        for s_ in (*SITES, "both"):
            for feat in FEATURES:
                for adj, d in (("raw", parts[s_]), ("within_group", res[s_])):
                    rows.append({"cohort": coh, "axis": axis, "site": s_, "feature": feat, "adjust": adj, "n": len(d),
                                 **_corr(d[feat].to_numpy(), d["s"].to_numpy())})
    return pd.DataFrame(rows)


def explained(proj: pd.DataFrame) -> pd.DataFrame:
    """科ごとに stain_response.regression（s_raw ~ site [+ 色]）。aggregator 列に科の名前を入れて流用する。"""
    frames = []
    for axis in ("paper", "own"):
        d = proj.rename(columns={f"s_raw_{axis}": "s_raw", f"p_{axis}": "s", "cohort": "aggregator"}).assign(condition="original")
        reg = regression(d, "ebrains").rename(columns={"aggregator": "cohort"})
        reg.insert(1, "axis", axis)
        frames.append(reg)
    return pd.concat(frames, ignore_index=True)


# ── analyze: 受け入れ確認 ─────────────────────────────────────────────────────

def acceptance(cfg: dict, df: pd.DataFrame, delta_norm: float) -> list[str]:
    lines = [f"- |Δ_paper| = {delta_norm:.4f}（stain_response の titan {EXPECTED_SHIFT_NORM}）"
             + (" OK" if abs(delta_norm - EXPECTED_SHIFT_NORM) < 1e-3 else " **NG**")]
    ref = paper_dir(cfg) / "color" / "slide_color.csv"
    if not ref.exists():
        return lines + [f"- paper_color の slide_color.csv が無い（{ref}）"]
    pc = pd.read_csv(ref, dtype={"case_id": str})
    pc = pc[pc["group"] == pc["site"]].rename(columns={"case_id": "slide_id"})
    m = df[df["in_paper"]].merge(pc[["slide_id", "site", "n_pixels", *COLOR5]], on=["slide_id", "site"],
                                 suffixes=("", "_ref"))
    same = m[m["n_rejected"] == 0]
    diff = max(float((same[f] - same[f"{f}_ref"]).abs().max()) for f in COLOR5)
    px = int((same["n_pixels"] != same["n_pixels_ref"]).sum())
    lines.append(f"- 髄膜腫+SFT コホートの色 vs paper_color.py（{ref}）: 突き合わせ {len(m)} 枚、"
                 f"パッチを捨てなかった {len(same)} 枚で max |差| = {diff:.2e}、画素数の不一致 {px} 枚"
                 + (" OK" if diff < 1e-9 and px == 0 else " **NG**")
                 + f"（捨てたパッチがあるのは {int((m['n_rejected'] > 0).sum())} 枚）")
    return lines


# ── analyze: summary / 図 ─────────────────────────────────────────────────────

def _md(df: pd.DataFrame, digits: int = 3) -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines)


def _ci(fam: pd.DataFrame, k: str) -> pd.Series:
    return fam.apply(lambda r: f"{r[k]:.2f} [{r[k + '_lo']:.2f}, {r[k + '_hi']:.2f}]", axis=1)


def write_summary(path: Path, fam: pd.DataFrame, xcorr: pd.DataFrame, corr: pd.DataFrame, expl: pd.DataFrame,
                  acc: list[str], df: pd.DataFrame) -> None:
    order = [*FAMILIES, "paper"]
    fam = fam.set_index("family").loc[order].reset_index()
    t1 = pd.DataFrame({
        "family": fam["family"].map(FAMILY_LABEL), "n_E": fam["n_ebrains"], "n_P": fam["n_patho2"],
        "s（埋め込みの倍率）": _ci(fam, "scale_s"),
        "Lab の幅の比": _ci(fam, "lab_spread_ratio"), "Lab の幅の比（頑健）": _ci(fam, "lab_spread_robust_ratio"),
        "sd 比 H": _ci(fam, "sdratio_H"), "IQR 比 H": _ci(fam, "iqrratio_H"),
        "sd 比 E": _ci(fam, "sdratio_E"), "sd 比 L": _ci(fam, "sdratio_L"),
        "s（色を除く）": fam["scale_s_colour_resid"].round(3), "s（自由度補正のみ）": fam["scale_s_dfcorr"].round(3),
        "|Δ_f|": fam["shift_norm"].round(2), "⟨Δ_f, u⟩": fam["shift_along_u"].round(2), "cos": fam["cos_paper"].round(3),
        "ΔE76": fam["dE76"].round(2),
    })
    t2 = pd.DataFrame({"family": fam["family"].map(FAMILY_LABEL)})
    for f, dg in (("L", 1), ("a", 1), ("b", 1), ("H", 4), ("E", 4)):
        for s_ in SITES:
            t2[f"{f} {SITE_LABEL[s_]}"] = fam.apply(lambda r: f"{r[f'{f}_mean_{s_}']:.{dg}f} ± {r[f'{f}_sd_{s_}']:.{dg}f}", axis=1)
    t3 = pd.DataFrame({"family": fam["family"].map(FAMILY_LABEL)})
    for s_ in SITES:
        t3[f"Lab 幅 {SITE_LABEL[s_]}"] = fam[f"lab_spread_{s_}"].round(2)
    for f, dg in (("sd_H", 4), ("sd_E", 4), ("sd_L", 2)):
        for s_ in SITES:
            t3[f"スライド内 {f} {SITE_LABEL[s_]}"] = fam[f"within_{f}_{s_}"].round(dg)
    for s_ in SITES:
        t3[f"R²(埋め込み ~ 色) {SITE_LABEL[s_]}"] = fam[f"r2_colour_{s_}"].round(3)

    def cpick(coh: str, site: str, feat: str, adj: str = "within_group", axis: str = "paper", col: str = "pearson_r"):
        m = corr[(corr.cohort == coh) & (corr.site == site) & (corr.feature == feat) & (corr.adjust == adj) & (corr.axis == axis)]
        return float(m[col].iloc[0]) if len(m) else np.nan

    t4 = []
    for coh in order:
        r = {"family": FAMILY_LABEL[coh]}
        for s_ in (*SITES, "both"):
            for feat in ("H", "E", "L"):
                r[f"{feat} {SITE_LABEL.get(s_, s_)}"] = (f"{cpick(coh, s_, feat):+.2f} / "
                                                          f"{cpick(coh, s_, feat, col='spearman_rho'):+.2f}")
        t4.append(r)
    t4 = pd.DataFrame(t4)
    t4b = []
    for coh in order:
        r = {"family": FAMILY_LABEL[coh]}
        for s_ in SITES:
            for feat in ("H", "L"):
                r[f"{feat} {SITE_LABEL[s_]} raw"] = cpick(coh, s_, feat, adj="raw")
                r[f"{feat} {SITE_LABEL[s_]} own 軸"] = cpick(coh, s_, feat, axis="own")
        t4b.append(r)
    t4b = pd.DataFrame(t4b)
    ex = expl[expl["colour_set"] == "H,E,L,a,b"]
    t5 = []
    for coh in order:
        r = {"family": FAMILY_LABEL[coh]}
        for axis in ("paper", "own"):
            e = ex[(ex.cohort == coh) & (ex.axis == axis)].iloc[0]
            r[f"{axis}: 施設差 → 色で調整後"] = f"{e['site_coef_unadjusted']:.2f} → {e['site_coef_adjusted']:.2f}"
            r[f"{axis}: 色で説明"] = f"{e['site_gap_explained_by_colour']:.0%}"
            r[f"{axis}: R²(施設内 p ~ 色)"] = f"{e['r2_colour_on_s_within_site']:.3f}"
        eb = expl[(expl.cohort == coh) & (expl.axis == "paper") & (expl.colour_set == "he_balance")].iloc[0]
        r["paper: log(H/E) だけで説明"] = f"{eb['site_gap_explained_by_colour']:.0%}"
        t5.append(r)
    t5 = pd.DataFrame(t5)
    xc = xcorr.copy()

    # ── 読み（数字から組む）──
    f5 = fam[fam["family"].isin(FAMILIES)].set_index("family")
    read = []
    not_wider = [FAMILY_LABEL[f] for f in FAMILIES if f5.loc[f, "lab_spread_ratio"] < 1]
    read.append("- **色の幅（Lab の 3 次元の幅の比 EBRAINS / patho2）**: "
                + ", ".join(f"{FAMILY_LABEL[f]} {f5.loc[f, 'lab_spread_ratio']:.2f}" for f in FAMILIES)
                + f"（髄膜腫+SFT コホート {fam.set_index('family').loc['paper', 'lab_spread_ratio']:.2f}）。H の SD 比は "
                + ", ".join(f"{FAMILY_LABEL[f]} {f5.loc[f, 'sdratio_H']:.2f}" for f in FAMILIES) + "、IQR 比は "
                + ", ".join(f"{FAMILY_LABEL[f]} {f5.loc[f, 'iqrratio_H']:.2f}" for f in FAMILIES)
                + "（patho2 の H には極端に濃い少数のスライドがあり、SD 比はそれに引っ張られて小さく出る）。"
                + (f"EBRAINS の方が狭い科: {', '.join(not_wider)}。" if not_wider else "どの科でも EBRAINS の方が広い。"))
    rr = xc[(xc.x == "lab_spread_ratio") & (xc.y == "scale_s")].iloc[0]
    rh = xc[(xc.x == "sdratio_H") & (xc.y == "scale_s")].iloc[0]
    ri = xc[(xc.x == "iqrratio_H") & (xc.y == "scale_s")].iloc[0]
    rd = xc[(xc.x == "dE76") & (xc.y == "shift_norm")].iloc[0]
    sign = "負（色の幅の比が大きい科ほど s が小さい = 仮説と逆向き）" if ri["pearson_r"] < 0 else "正（仮説と同じ向き）"
    read.append("- **埋め込みの倍率 s との対応（5 点、参考）**: s = "
                + ", ".join(f"{FAMILY_LABEL[f]} {f5.loc[f, 'scale_s']:.2f}" for f in FAMILIES)
                + f"。Lab の幅の比 vs s は Pearson r = {rr['pearson_r']:+.2f}（Spearman {rr['spearman_rho']:+.2f}）、"
                f"H の SD 比 vs s は {rh['pearson_r']:+.2f}（{rh['spearman_rho']:+.2f}）、H の IQR 比 vs s は "
                f"{ri['pearson_r']:+.2f}（{ri['spearman_rho']:+.2f}、p = {ri['pearson_p']:.2f}）。相関の符号は{sign}で、"
                f"色の幅が最も広い SFT（Lab {f5.loc['SFT', 'lab_spread_ratio']:.2f}）の s が最も 1 に近く（{f5.loc['SFT', 'scale_s']:.2f}）、"
                f"s が最も大きい転移（{f5.loc['metastasis', 'scale_s']:.2f}）の色の幅は中くらい（{f5.loc['metastasis', 'lab_spread_ratio']:.2f}）。"
                "5 点で CI も重なるので強くは言えないが、**「EBRAINS の色の幅が広い科ほど s が大きい」は支持されない**。")
    dmax = float((f5["scale_s_colour_resid"] - f5["scale_s_dfcorr"]).abs().max())
    read.append("- **色を除いても倍率はほぼ変わらない**: 科 × 施設ごとに埋め込みを色 5 つで回帰して残差の RMS 比を取ると s は "
                + ", ".join(f"{FAMILY_LABEL[f]} {f5.loc[f, 'scale_s_dfcorr']:.2f} → {f5.loc[f, 'scale_s_colour_resid']:.2f}"
                            for f in FAMILIES)
                + f"（自由度補正済みの s → 色を除いた s、変化は最大 {dmax:.2f}）。埋め込みの全分散のうち色 5 つで説明できる割合は EBRAINS "
                + ", ".join(f"{f5.loc[f, 'r2_colour_ebrains']:.2f}" for f in FAMILIES) + " / patho2 "
                + ", ".join(f"{f5.loc[f, 'r2_colour_patho2']:.2f}" for f in FAMILIES)
                + "（科の順は同じ）と小さい。スライド平均色で見る限り、科による s の違いは色の幅では説明できず、"
                  "コホートの中身（腫瘍の多様さ）の側にあると読むのが素直（スライド平均色はパッチ単位の色・質感を均した粗い量で、"
                  "色の寄与を下に見積もる向きの偏りはある）。")
    read.append("- **施設軸 Δ_paper への射影の H 応答は 5 科すべてにある（群内、Pearson）**: EBRAINS "
                + ", ".join(f"{FAMILY_LABEL[f]} {cpick(f, 'ebrains', 'H'):+.2f}" for f in FAMILIES)
                + " / patho2 "
                + ", ".join(f"{FAMILY_LABEL[f]} {cpick(f, 'patho2', 'H'):+.2f}" for f in FAMILIES)
                + f"（髄膜腫+SFT コホート {cpick('paper', 'ebrains', 'H'):+.2f} / {cpick('paper', 'patho2', 'H'):+.2f}、"
                  "stain_response の TITAN within_subtype は +0.404 / +0.095 で再現）。L* は全科で負"
                  f"（両施設をまとめて {min(cpick(f, 'both', 'L') for f in FAMILIES):+.2f}〜{max(cpick(f, 'both', 'L') for f in FAMILIES):+.2f}）、"
                  "E はほぼ 0（リンパ腫だけ +0.3）。**「H が濃いほど EBRAINS 側に寄る」応答は腫瘍によらず共有される**。"
                  f"想定外: patho2 で応答が弱いのは髄膜腫（{cpick('meningioma', 'patho2', 'H'):+.2f}）だけで、"
                  f"patho2 のグリオーマ・リンパ腫・転移は {cpick('glioma', 'patho2', 'H'):+.2f} / "
                  f"{cpick('lymphoma', 'patho2', 'H'):+.2f} / {cpick('metastasis', 'patho2', 'H'):+.2f}。"
                  "stain_response で「patho2 は応答が弱い」としたのは施設の性質ではなく、髄膜腫+SFT コホートの patho2 側"
                  "（院内の WSI の置き場も Patho2Gliomas と別）に限った話の可能性がある。")
    read.append("- **施設差のうち平均色で説明できる割合（Δ_paper 軸、H,E,L,a,b）**: "
                + ", ".join(f"{FAMILY_LABEL[c]} {ex[(ex.cohort == c) & (ex.axis == 'paper')].iloc[0]['site_gap_explained_by_colour']:.0%}"
                            for c in order)
                + "（stain_response の髄膜腫+SFT コホート TITAN は 14%）。どの科でも施設差の大半は平均色の外にある。"
                + f"平均色の施設差 ΔE76 は科で {f5['dE76'].min():.1f}〜{f5['dE76'].max():.1f} と違うのに、Δ_paper 方向の施設差 ⟨Δ_f, u⟩ は "
                f"{f5['shift_along_u'].min():.1f}〜{f5['shift_along_u'].max():.1f} で、ΔE76 と |Δ_f| の相関は r = {rd['pearson_r']:+.2f}（5 点）。")
    read.append("- 注意: 色はスライドあたり 32 パッチ（h5 の座標 = TITAN が見たパッチから無作為）の組織画素の平均で、"
                "H / E は Ruifrok–Johnston の固定行列。ブートストラップはスライド単位（患者単位ではない）。"
                "SFT（22 / 32 枚）・リンパ腫（59 / 59）・転移（47 / 84）は例数が少なく、比の CI が広い。"
                "グリオーマは EBRAINS 807 / patho2 117 で組成（ast / glm / odg）も施設で違い、科全体の幅には群の違いも入る。")

    body = [
        "# transfer_color — 転移テストの全スライドの色と、科ごとの色の幅・施設軸の H 応答（TITAN）",
        "",
        "定義はスクリプトの docstring。Δ_paper = mean(EBRAINS 髄膜腫+SFT コホート) − mean(patho2 髄膜腫+SFT コホート)、u = Δ_paper/|Δ_paper|。"
        "p = ⟨x − μ_{科, 施設}, u⟩。s = RMS 比（affine.scale_free）。[ ] はスライド単位のブートストラップ 95% CI。",
        "",
        "## 受け入れ確認",
        "",
        *acc,
        f"- 色を取れたスライド {int(df['H'].notna().sum())} / {len(df)} 枚、パッチを捨てたスライド "
        f"{int((df['n_rejected'] > 0).sum())} 枚（捨てたパッチ計 {int(df['n_rejected'].sum())}）、"
        f"32 枚に満たないスライド {int((df['n_patches'] < 32).sum())} 枚。画素の出どころ: "
        + ", ".join(f"{k} {v}" for k, v in df["pixels"].value_counts().items()),
        "",
        "## 色の幅の比と埋め込みの倍率（科ごと）",
        "",
        _md(t1),
        "",
        "## 5 科をまたぐ相関（参考、n = 5）",
        "",
        _md(xc),
        "",
        "## 科 × 施設の平均色（スライド間の平均 ± SD）",
        "",
        _md(t2),
        "",
        "## 科 × 施設の幅・スライド内のばらつき・埋め込みの色依存",
        "",
        "Lab 幅 = 施設平均色からの RMS 距離。スライド内 sd_* = スライドごとのパッチ間 SD の平均。"
        "R²(埋め込み ~ 色) = 768 次元の全分散のうち色 5 つ（H, E, L, a, b）の線形回帰で説明できる割合（自由度補正）。",
        "",
        _md(t3),
        "",
        "## 施設軸 Δ_paper への射影と色の相関（群内、Pearson r / Spearman ρ）",
        "",
        "within_group = 施設内の群（髄膜腫 = サブタイプ、グリオーマ = ast / glm / odg）の平均で p と色を残差化。"
        "both = 両施設の残差をまとめたもの。全部の組み合わせは stain_response_by_family.csv。",
        "",
        _md(t4),
        "",
        "### 同（Pearson）: raw と、科自身の Δ_f の向き（own 軸、群内）",
        "",
        _md(t4b),
        "",
        "## 施設差のうち平均色で説明できる割合（stain_response.regression を科ごとに）",
        "",
        "施設差 = s_raw ~ site の施設係数（paper 軸では ⟨Δ_f, u⟩、own 軸では |Δ_f|）、色で調整後 = s_raw ~ site + 標準化した "
        "H, E, L, a, b の施設係数、色で説明 = 1 − 調整後 / 調整前。",
        "",
        _md(t5),
        "",
        "## 読み",
        "",
        *read,
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def plot(path: Path, df: pd.DataFrame, fam: pd.DataFrame, proj: pd.DataFrame, corr: pd.DataFrame, colors: dict) -> None:
    fig = plt.figure(figsize=(16, 8.4))
    gs = GridSpec(2, 20, figure=fig, height_ratios=[1, 1], hspace=0.55, wspace=1.6)
    rng = np.random.default_rng(SEED)
    d = df[df["H"].notna()]
    # a: H・E の分布
    for j, (feat, label) in enumerate((("H", "Hematoxylin (H)"), ("E", "Eosin (E)"))):
        ax = fig.add_subplot(gs[0, j * 5:(j + 1) * 5])
        for k, f in enumerate(FAMILIES):
            for o, s_ in ((-0.2, "ebrains"), (0.2, "patho2")):
                v = d.loc[(d.family == f) & (d.site == s_), feat].to_numpy()
                ax.scatter(k + o + rng.uniform(-0.12, 0.12, len(v)), v, s=3, color=colors[s_], alpha=0.35, linewidths=0)
                ax.hlines(np.median(v), k + o - 0.15, k + o + 0.15, color="black", lw=1.4)
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels([FAMILY_LABEL[f] for f in FAMILIES], fontsize=8, rotation=30, ha="right")
        top = float(np.quantile(d[feat], 0.998)) * 1.08   # 少数の極端な値で軸がつぶれないよう切る
        n_off = int((d[feat] > top).sum())
        ax.set_ylim(0, top)
        if n_off:
            ax.text(0.99, 0.01, f"{n_off} slides off-scale", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#666666")
        ax.set_title(("a  " if j == 0 else "") + label, fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            for s_ in SITES:
                ax.scatter([], [], s=12, color=colors[s_], label=SITE_LABEL[s_])
            ax.legend(fontsize=8, frameon=False, loc="upper right")
    # b: 色の幅の比 vs s
    f5 = fam.set_index("family").loc[FAMILIES]
    for j, (k, label) in enumerate((("lab_spread_ratio", "Lab spread ratio (EBRAINS / patho2)"),
                                    ("iqrratio_H", "IQR ratio of H (EBRAINS / patho2)"))):
        ax = fig.add_subplot(gs[0, 10 + j * 5:10 + (j + 1) * 5])
        x, y = f5[k].to_numpy(), f5["scale_s"].to_numpy()
        ax.errorbar(x, y, xerr=[x - f5[f"{k}_lo"], f5[f"{k}_hi"] - x],
                    yerr=[y - f5["scale_s_lo"], f5["scale_s_hi"] - y], fmt="o", color="#333333", ms=5,
                    ecolor="#BBBBBB", elinewidth=1, capsize=0)
        for f, xi, yi in zip(FAMILIES, x, y):
            off = (5, -12) if f in ("lymphoma", "SFT") else (5, 4)   # Glioma・Lymphoma・SFT が近いので下に出す
            ax.annotate(FAMILY_LABEL[f], (xi, yi), xytext=off, textcoords="offset points", fontsize=8)
        ax.axhline(1, color="#999999", lw=0.6, ls=":")
        ax.axvline(1, color="#999999", lw=0.6, ls=":")
        r = stats.pearsonr(x, y)[0]
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("embedding scale s (RMS ratio)" if j == 0 else "", fontsize=8)
        ax.set_title(("b  " if j == 0 else "") + f"Pearson r = {r:+.2f} (n = 5 families)", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    # c: Δ_paper への射影 vs H（科ごと、科 × 施設で中心化）
    for k, f in enumerate(FAMILIES):
        ax = fig.add_subplot(gs[1, k * 4:(k + 1) * 4])
        g = proj[proj["cohort"] == f]
        title = []
        for s_ in SITES:
            h = g[g.site == s_]
            ax.scatter(h["H"], h["p_paper"], s=4, color=colors[s_], alpha=0.4, linewidths=0)
            if len(h) >= 3:
                b, a = np.polyfit(h["H"], h["p_paper"], 1)
                xs = np.linspace(h["H"].min(), h["H"].max(), 2)
                ax.plot(xs, a + b * xs, color=colors[s_], lw=1.5)
            m = corr[(corr.cohort == f) & (corr.site == s_) & (corr.feature == "H") & (corr.adjust == "within_group")
                     & (corr.axis == "paper")]
            title.append(f"{SITE_LABEL[s_]} r = {float(m['pearson_r'].iloc[0]):+.2f}")
        ax.set_title(("c  " if k == 0 else "") + f"{FAMILY_LABEL[f]}\n" + "\n".join(title), fontsize=9, loc="left")
        ax.set_xlim(right=float(np.quantile(g["H"], 0.998)) * 1.08)
        ax.set_xlabel("H (slide mean)", fontsize=8)
        if k == 0:
            ax.set_ylabel("projection on Δ_paper\n(centred within family × site)", fontsize=8)
            ax.text(0.0, -0.3, "r: within-group Pearson (groups: meningioma subtypes, ast / glm / odg)",
                    transform=ax.transAxes, fontsize=7, color="#666666")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze(cfg: dict, out_dir: Path) -> None:
    df, X = load_all(cfg, out_dir)
    ok = df["H"].notna().to_numpy()
    site = df["site"].to_numpy()
    paper = df["in_paper"].to_numpy() & ok
    # Δ_paper は色の有無によらず髄膜腫+SFT コホート全体で（shift_transfer.py と同じ定義）
    paper_all = df["in_paper"].to_numpy()
    delta_paper = X[paper_all & (site == "ebrains")].mean(0) - X[paper_all & (site == "patho2")].mean(0)
    dn = float(np.linalg.norm(delta_paper))
    print(f"[analyze] |Δ_paper| = {dn:.4f}  n_paper(色あり) = {int(paper.sum())}")
    acc = acceptance(cfg, df, dn)
    print("\n".join(acc))

    rng = np.random.default_rng(SEED)
    fam = family_table(df, X, delta_paper, rng)
    xcorr = cross_family_corr(fam)
    proj = projections(df, X, delta_paper)
    corr = response_corr(proj)
    expl = explained(proj)

    proj.to_csv(out_dir / "slide_projection.csv", index=False)
    fam.to_csv(out_dir / "family_color.csv", index=False)
    corr.to_csv(out_dir / "stain_response_by_family.csv", index=False)
    expl.to_csv(out_dir / "explained_by_family.csv", index=False)
    write_summary(out_dir / "summary.md", fam, xcorr, corr, expl, acc, df)
    plot(out_dir / "transfer_color.png", df, fam, proj, corr, cfg["display"]["colors"]["sources"])
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(fam[["family", "n_ebrains", "n_patho2", "scale_s", "scale_s_colour_resid", "lab_spread_ratio", "sdratio_H",
                   "sdratio_E", "sdratio_L", "dE76", "shift_norm", "cos_paper"]].round(3).to_string(index=False))
        print(xcorr.round(3).to_string(index=False))
        c = corr[(corr.feature == "H") & (corr.adjust == "within_group") & (corr.axis == "paper")]
        print(c[["cohort", "site", "n", "pearson_r", "spearman_rho"]].round(3).to_string(index=False))
        e = expl[expl.colour_set == "H,E,L,a,b"]
        print(e[["cohort", "axis", "site_coef_unadjusted", "site_coef_adjusted", "site_gap_explained_by_colour",
                 "r2_colour_on_s_within_site"]].round(3).to_string(index=False))
    print(f"saved: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["build", "analyze"])
    ap.add_argument("--workers", type=int, default=4, help="build の並列数（NFS を他のジョブと共有するので 4 以下）")
    ap.add_argument("--limit", type=int, default=None, help="build で今回処理する枚数の上限（試運転用）")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "transfer_color"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.cmd == "build":
        build(cfg, out_dir, args.workers, args.limit)
    else:
        analyze(cfg, out_dir)


if __name__ == "__main__":
    main()
