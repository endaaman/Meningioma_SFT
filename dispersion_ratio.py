"""dispersion_ratio.py — 施設間の「ばらつきの比」は腫瘍の種類をまたいで共有されるか（Supplementary 用、TITAN。--encoder meanpool で平均プーリング、--encoder prism で PRISM、--encoder meanpool_virchow で Virchow のパッチ平均）。

中心のずれ（平行移動 Δ）が腫瘍をまたいでほぼ同じ向き（cos 0.8–0.9、shift_transfer.py）なのに対し、ComBat・相似変換が使う
「施設ごとのばらつきの比」が腫瘍をまたいで同じかを見る。比が腫瘍で違えば、ばらつきの比には施設の性質（色の幅など）と
コホートの中身の多様さ（腫瘍の性質）が混ざっていて、別の腫瘍に移せない（= 転移テストで倍率が移らなかった理由）。

科 f ごとに、次元 d の分散比  r_f,d = log( var_EBRAINS(f, d) / var_patho2(f, d) )  を全次元分（TITAN・meanpool 768、PRISM 1280、meanpool_virchow 2560）出し、
    1. 科どうしの r の相関（Pearson、全次元）  ← 平均のずれの cos（shift_transfer の cos 行列）と並べる
    2. 科の中の分割半分の再現性（施設ごとに無作為に半分に分け、r を 2 つ作って相関。何度か繰り返して中央値）
       ← 科どうしの相関がこれより十分低ければ、違いはノイズではなく腫瘍による
    3. 科ごとの全体の倍率 s_f（affine_free と同じ RMS 比）と、次元ごとの比のばらつき（sd of r）
    4. 倍率は施設の性質ではなく腫瘍ごとに違うか（ken 2026-11-24、論文の要）: 群ごと（髄膜腫・SFT・星細胞腫・膠芽腫・乏突起膠腫・
       リンパ腫・転移と、両施設 shift_transfer.subtype_min_n 枚以上の髄膜腫のサブタイプ）の施設内の全分散の比 tr Σ_E / tr Σ_P
       （ddof = 1）と相似変換の倍率 s（3 と同じ RMS 比）、比較に施設のずれの長さ ‖Δ‖ と髄膜腫のずれとの cos（髄膜腫のサブタイプは
       そのサブタイプを除いた髄膜腫と比べる）。95% 区間は患者単位のブートストラップ（施設 × 科ごとに患者を復元抽出、N_BOOT 回、
       乱数列 default_rng([SEED, 0])）。等しい倍率の検定: 帰無仮説「s は群によらず同じ」の下で、院内の各群を群の平均のまわりで
       s_g / s_0 倍して s をそろえ（s_0 = 群の s の幾何平均）、同じブートストラップで T = sd(log s_g) の帰無分布を作る
       （default_rng([SEED, 1])、p = (1 + #{T* ≥ T_obs}) / (1 + N_BOOT)）。科・群の 7 つと、髄膜腫のサブタイプどうしの 2 本

入力: shift_transfer.py のキャッシュ（out/shift_transfer/{paper,glioma}_slides.csv / _embeddings.npy）。無ければ先に shift_transfer.py を回す。
    --encoder meanpool（CONCH v1.5 のパッチ埋め込みの単純平均、学習なしの集約器の対照）は shift_transfer.py --encoder meanpool の
    キャッシュ（out/shift_transfer/meanpool/）。スライドの集合と並びは TITAN と同じなので、分割半分の乱数の分け方も同じ
    --encoder prism（Virchow v1 → PRISM、1280 次元）は shift_transfer.py --encoder prism のキャッシュ（out/shift_transfer/prism/）。
    スライドの集合と並びは TITAN と同じ
    --encoder meanpool_virchow（Virchow v1 のパッチ特徴 concat 2560 次元の単純平均、PRISM の集約器の対照）は
    shift_transfer.py --encoder meanpool_virchow のキャッシュ（out/shift_transfer/meanpool_virchow/）。スライドの集合と並びは TITAN と同じ
出力 (output_dir/dispersion_ratio/、--encoder meanpool / prism / meanpool_virchow は各ファイル名の末尾に _meanpool / _prism / _meanpool_virchow):
    corr_matrix.csv      — 科どうしの r の相関（上三角）と、分割半分の再現性（対角）、平均のずれの cos（下三角、参考）
    corr_matrix_corrected.csv — r の相関を分割半分の再現性で割った値（推定ノイズを差し引いた相関）
    summary.csv          — 科ごとの n、s_f、mean / sd of r、分割半分の再現性
    dispersion_ratio.png — a: 相関行列（r） b: 平均のずれの cos 行列 c: 科ごとの r のヒストグラム
    scale_by_group.csv   — 4 の群ごとの n・患者数、var_ratio / scale_s / shift_norm / cos_vs_meni と各 _lo / _hi（95% 区間）
    scale_test.csv       — 4 の等しい倍率の検定（T_obs、帰無分布の中央値・95 パーセンタイル、p）

Usage:
    uv run python dispersion_ratio.py [--encoder titan|meanpool|prism|meanpool_virchow]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.cohort import apply_cohort
from utils.display import ordered_subtypes
from utils.loader import load_config

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
SITES = ("ebrains", "patho2")
N_SPLIT = 50
SEED = 42
N_BOOT = 1000
TEST_GROUPS = ["meningioma", "SFT", "ast", "glm", "odg", "lym", "met"]   # 4 の科・群（髄膜腫・SFT は科、グリオーマは 3 群）


def load_cache(d: Path, cfg: dict | None = None) -> pd.DataFrame:
    """shift_transfer.py のキャッシュ（髄膜腫+SFT コホートとグリオーマ・リンパ腫・転移）。グリオーマ側の院内は症例の表
    （utils.cohort、無効ならそのまま）で絞り、群を表の group にする（shift_transfer.load_glioma と同じ）。"""
    frames = []
    for name in ("paper", "glioma"):
        df = pd.read_csv(d / f"{name}_slides.csv", dtype=str)
        df["embedding"] = list(np.load(d / f"{name}_embeddings.npy").astype(np.float64))
        if name == "glioma" and cfg is not None:
            df = apply_cohort(df, cfg, case_col="slide_id", group_cols=["group", "diagnosis"], site_col="source")
        if name == "paper":
            df["family"] = np.where(df["subtype"].str.lower().str.contains("sft"), "SFT", "meningioma")
            df["group"] = np.where(df["family"] == "SFT", "SFT", df["subtype"])
        else:
            df["family"] = df["group"].map({"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"})
        frames.append(df[["source", "family", "group", "patient", "embedding"]])
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
    ap.add_argument("--encoder", choices=["titan", "meanpool", "prism", "meanpool_virchow"], default="titan",
                    help="スライド埋め込み。meanpool = CONCH のパッチ平均、prism = Virchow → PRISM、meanpool_virchow = Virchow のパッチ平均"
                         "（shift_transfer.py --encoder <encoder> のキャッシュ）")
    args = ap.parse_args()
    cfg = load_config()
    cache = Path(cfg["output_dir"]) / "shift_transfer"
    if args.encoder != "titan":
        cache = cache / args.encoder
    sfx = "" if args.encoder == "titan" else f"_{args.encoder}"   # TITAN は前と同じファイル名
    tag = {"titan": "", "meanpool": " [mean-pooled CONCH v1.5]", "prism": " [PRISM]",
           "meanpool_virchow": " [mean-pooled Virchow v1]"}[args.encoder]
    out_dir = Path(cfg["output_dir"]) / "dispersion_ratio"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[encoder] {args.encoder}  cache: {cache}")
    df = load_cache(cache, cfg)
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

    tab, tests = scale_table(df, cfg)
    tab.to_csv(out_dir / f"scale_by_group{sfx}.csv", index=False)
    tests.to_csv(out_dir / f"scale_test{sfx}.csv", index=False)
    print("\n[scale by group] var_ratio = tr Σ_E / tr Σ_P、scale_s、shift_norm、cos_vs_meni（95% 区間は患者単位のブートストラップ）")
    print(tab[["group", "n_ebrains", "n_patho2", "var_ratio", "var_ratio_lo", "var_ratio_hi", "scale_s", "scale_s_lo", "scale_s_hi",
               "shift_norm", "cos_vs_meni"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[equal-scale test]\n" + tests.drop(columns="groups").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved to {out_dir}")


# ── 倍率は腫瘍ごとに違うか（群ごとの全分散の比・相似変換の s と患者単位のブートストラップ、等しい倍率の検定）──────────────

def scale_groups(df: pd.DataFrame, cfg: dict) -> list[tuple[str, str, str, np.ndarray]]:
    """(表示名, 科, 種類, 行のマスク): 科・群の 7 つ（TEST_GROUPS の順）と、髄膜腫のサブタイプ（両施設 shift_transfer.subtype_min_n 枚以上）。"""
    st = cfg.get("shift_transfer", {})
    names, min_n = st.get("group_names", {}), int(st.get("subtype_min_n", 20))
    fam, grp, site = df["family"].to_numpy(), df["group"].to_numpy(), df["source"].to_numpy()
    out = []
    for g in TEST_GROUPS:
        m = fam == g if g in ("meningioma", "SFT") else grp == g
        out.append(({"meningioma": "Meningioma (all)"}.get(g, names.get(g, g)), g if g in ("meningioma", "SFT") else
                    {"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"}[g], "family" if g in
                    ("meningioma", "SFT", "lym", "met") else "glioma", m))
    for sub in ordered_subtypes(set(grp[fam == "meningioma"]), cfg):
        m = (fam == "meningioma") & (grp == sub)
        if min(int((m & (site == s)).sum()) for s in SITES) >= min_n:
            out.append((sub, "meningioma", "subtype", m))
    return out


def _stats(Z: np.ndarray, site: np.ndarray, rows: np.ndarray, groups: list, meni: np.ndarray) -> np.ndarray:
    """rows（重複あり）の上で群ごとの [全分散の比 E/P, s, ‖Δ‖, cos(Δ, 髄膜腫の Δ)]。髄膜腫の Δ は群を除いた髄膜腫（leave-group-out）。"""
    out = np.full((len(groups), 4), np.nan)
    e = site[rows] == "ebrains"
    for k, (_, f, kind, m) in enumerate(groups):
        gm = m[rows]
        E, P = Z[rows[gm & e]], Z[rows[gm & ~e]]
        if len(E) < 2 or len(P) < 2:
            continue
        d = E.mean(0) - P.mean(0)
        out[k, :3] = E.var(0, ddof=1).sum() / P.var(0, ddof=1).sum(), rms_scale(E, P), np.linalg.norm(d)
        if f == "meningioma" and kind == "family":
            continue                                   # 髄膜腫そのもの: 比べる相手が自分
        rm = meni[rows] & ~gm
        dr = Z[rows[rm & e]].mean(0) - Z[rows[rm & ~e]].mean(0)
        out[k, 3] = float(d @ dr / (np.linalg.norm(d) * np.linalg.norm(dr)))
    return out


def _boot_rows(units: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """施設 × 科ごとに患者を復元抽出し、選ばれた患者のスライドを全部入れた行番号（科ごとの患者数は固定）。"""
    return np.concatenate([np.concatenate([u[i] for i in rng.integers(0, len(u), len(u))]) for u in units])


def _units(df: pd.DataFrame) -> list[list[np.ndarray]]:
    """施設 × 科ごとの [患者ごとの行番号の配列]。"""
    out = []
    for s in SITES:
        for f in FAMILIES:
            d = df[(df["source"] == s) & (df["family"] == f)]
            out.append([np.asarray(ix) for ix in d.groupby("patient", sort=True).indices.values()])
            out[-1] = [d.index.to_numpy()[ix] for ix in out[-1]]
    return out


def scale_table(df: pd.DataFrame, cfg: dict, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """群ごとの全分散の比・s・‖Δ‖・cos と患者単位のブートストラップの 95% 区間、等しい倍率の検定（docstring の 4）。"""
    df = df.reset_index(drop=True)
    Z, site = np.stack(df["embedding"].values), df["source"].to_numpy()
    groups = scale_groups(df, cfg)
    meni = (df["family"] == "meningioma").to_numpy()
    units = [u for u in _units(df)]
    obs = _stats(Z, site, np.arange(len(df)), groups, meni)
    rng = np.random.default_rng([SEED, 0])
    B = np.stack([_stats(Z, site, _boot_rows(units, rng), groups, meni) for _ in range(n_boot)])
    rows = []
    for k, (name, f, kind, m) in enumerate(groups):
        r = {"group": name, "family": f, "kind": kind,
             "n_ebrains": int((m & (site == "ebrains")).sum()), "n_patho2": int((m & (site != "ebrains")).sum()),
             "patients_ebrains": int(df.loc[m & (site == "ebrains"), "patient"].nunique()),
             "patients_patho2": int(df.loc[m & (site != "ebrains"), "patient"].nunique())}
        for j, key in enumerate(("var_ratio", "scale_s", "shift_norm", "cos_vs_meni")):
            r[key] = obs[k, j]
            r[f"{key}_lo"], r[f"{key}_hi"] = (np.nanquantile(B[:, k, j], 0.025), np.nanquantile(B[:, k, j], 0.975)) \
                if np.isfinite(B[:, k, j]).any() else (np.nan, np.nan)
        rows.append(r)
    tab = pd.DataFrame(rows)

    # 帰無仮説「倍率 s は群によらず同じ」: 院内の各群を群の平均のまわりで s_g / s_0 倍に伸縮して s をそろえ（s_0 = 群の s の幾何平均）、
    # 同じ患者単位のブートストラップで T = sd(log s_g) の帰無分布を作る。p = (1 + #{T* ≥ T_obs}) / (1 + B)
    tests = []
    for test, sel in (("families_7", [k for k, g in enumerate(groups) if g[2] in ("family", "glioma")]),
                      ("meningioma_subtypes", [k for k, g in enumerate(groups) if g[2] == "subtype"])):
        if len(sel) < 2:
            continue
        ls = np.log(obs[sel, 1])
        t_obs, s0 = float(ls.std()), float(np.exp(ls.mean()))
        Z0 = Z.copy()
        for k in sel:
            pm = groups[k][3] & (site != "ebrains")
            mu = Z[pm].mean(0)
            Z0[pm] = mu + (Z[pm] - mu) * (obs[k, 1] / s0)
        rng0 = np.random.default_rng([SEED, 1])
        T = np.array([np.log(_stats(Z0, site, _boot_rows(units, rng0), [groups[k] for k in sel], meni)[:, 1]).std() for _ in range(n_boot)])
        tests.append({"test": test, "groups": "; ".join(groups[k][0] for k in sel), "n_groups": len(sel), "statistic": "sd(log s)",
                      "t_obs": t_obs, "s_geomean": s0, "null_median": float(np.median(T)), "null_q95": float(np.quantile(T, 0.95)),
                      "p_value": float((1 + (T >= t_obs).sum()) / (1 + n_boot)), "n_boot": n_boot,
                      "s_min": float(np.exp(ls.min())), "s_max": float(np.exp(ls.max()))})
    return tab, pd.DataFrame(tests)


if __name__ == "__main__":
    main()
