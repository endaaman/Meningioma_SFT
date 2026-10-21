"""shift_estimation.py — 平行移動（centroid 補正）のシフト推定、実用上の 2 つの問い。

平行移動は 対象施設（patho2）の平均を reference（EBRAINS）の平均に揃える:
    Δ_raw = mean(EBRAINS) − mean(patho2)
これを実際に使うときの 2 点を、補正前（original）の TITAN 埋め込みで調べる。

A. 症例構成の切り分け（casemix）
    Δ_raw には施設差のほかに「組織型の構成比が違う」ことによる項が混ざる（Fibrous 162 vs 57、
    Transitional 36 vs 68 など）。両施設に min_n 例以上ある組織型 g（SFT も 1 群）について
        Δ* = Σ_g w_g (μ_E(g) − μ_P(g))
    を「組織型を合わせたシフト」とし、Δ_raw と比べる。重み w_g は
        min      … w_g ∝ min(n_E,g, n_P,g)（主）
        sum      … w_g ∝ n_E,g + n_P,g（感度）
        uniform  … 一様（感度）
    報告: |Δ_raw|・|Δ*|・cos(Δ_raw, Δ*)・|Δ_raw − Δ*| / |Δ*| と、Δ_raw の代わりに Δ* で平行移動したときの
    下流指標（ASW(site)・樹形図の隣接ペア数・組織型の最近傍重心法 BA、EBRAINS → patho2）。

B. 推定に要る例数（subsample）
    運用を模す: EBRAINS は全例既知、新施設（patho2）からは n 例の部分集合 S だけが手元にあるとして
        Δ̂ = mean(EBRAINS) − mean(S)
    を推定し、patho2 の全スライドに適用して評価する。抜き方:
        random      … 無作為
        stratified  … patho2 の組織型構成比で層化（最大剰余法で割り当て）
        single      … 1 つの組織型だけから抜く（新施設が 1 型しか出さない最悪ケース。patho2 で多い 3 型、
                      n はその型の例数まで）
        *_ortho     … 同じ標本から、EBRAINS の組織型平均の差 {μ_E(g) − μ_E(all)} が張る部分空間の成分を
                      Δ̂ から落とした推定量 Δ̂⊥（ラベルは reference 側だけ。新施設ではラベル不要）
    抽出は主プロセスで順に行う（seed 固定、設定ごとに独立な乱数列）ので、並列数によらず同じ結果になる。
    1 回ごとの指標: cos(Δ̂, Δ*)・|Δ̂ − Δ*| / |Δ*|（Δ* は A の主重み）・平行移動後の ASW(site)・
    組織型 BA（EBRAINS → patho2、ラベルは評価にだけ使う）・樹形図の隣接ペア数。
    設定（抜き方 × 組織型 × n）ごとに中央値と 2.5 / 97.5 パーセンタイル。

出力 (output_dir/shift_estimation/):
    casemix.csv          — A: 重みごとの比較と下流指標（none / raw の行も）
    casemix_groups.csv   — A: 群ごとの例数・重み・群シフトの大きさと Δ* との cos
    subsample.csv        — B: 設定ごとの要約（n = 0 の補正なし、全例の Δ_raw / Δ* の行も）
    subsample_draws.csv  — B: 1 回ごとの値
    summary.md           — A・B の要点の表と短い読み

Usage:
    uv run python shift_estimation.py [--n-draws N]   # --n-draws は config の値を上書き（試走用）
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, silhouette_score

from dendrogram import group_means, pair_stats, raw_distance, ward_linkage
from harmonization import _reduce, compute_metrics
from histotype_labels import apply_set
from utils.display import ordered_subtypes, shorten
from utils.loader import load_config, load_data

WEIGHTINGS = ["min", "sum", "uniform"]
# summary.md の「読み」。数値を見て書いた文章（実行結果 2026-10-21（n551））。数値や設定を変えたら見直す
READING = """
- A. 症例構成の項は小さい。組織型を合わせた Δ* は Δ_raw と cos 0.988、差は ‖Δ*‖ の 15%（重みの取り方によらず 0.150〜0.155、
  ‖Δ‖ は 5.3〜5.6）。下流では Δ* の方が組織型 BA がわずかに高く（0.636 → 0.652、nc.py の bootstrap CI 0.56〜0.71 の中）、
  ASW(site) はわずかに悪い（0.0094 → 0.0108。偶然のレベル 0.0062 [0.0037, 0.0089]）。隣接ペアは 11 で同じ。
  Δ_raw（施設の平均を揃えるだけ）で結論は変わらない。
- B. 無作為なら 20 例ほどで足りる。n = 20 で cos 0.955 [0.907, 0.978]、ASW 0.019 [0.015, 0.028]、BA 0.636 [0.604, 0.657]
  （全例 0.636）。n = 50 で ASW 0.013、n = 100 で 0.011 と全例（0.0094）にほぼ並ぶ。n = 5 でも ASW は 0.134 → 0.048 まで
  下がるが幅が広く、n = 3 は上側 2.5% で 0.16 と補正なしより悪くなり得る。
- 層化（patho2 の構成比）は小さい n で少し良い（n = 3 で ASW 0.061 vs 0.074、n = 20 で 0.017 vs 0.019）が、n ≥ 50 では
  差が無い。層化には新施設のラベルが要るので、実用上の利点は乏しい。
- 単一組織型だけから抜くと、n を増やしても Δ* に近づかない（バイアス）: Fibrous は cos 0.80・ASW 0.049 で頭打ち、組織型 BA は
  0.51 と補正なし（0.595）より悪い。Meningothelial（cos 0.93、BA 0.633）と Atypical（cos 0.94、BA 0.672）はほぼ全例と同等。
  推定量 mean(E) − mean(S_g) は真の群シフトに「EBRAINS 内の g と全体の平均差」が足し込まれるので、全体から離れた型ほど害が出る
  （‖μ_E(g) − μ_E(all)‖: Fibrous 3.99、Meningothelial 2.30、Atypical 2.09。`out/shift_bio_axes/summary.md`）。ASW(site) は 3 型すべてで改善するので、ASW だけを見ると害に気づけない。
- 生物学的方向の除去（ortho）は使えない。EBRAINS の組織型平均の差が張る 13 次元を落とすと Δ_raw のノルムが 0.72 倍
  （二乗ノルムで半分）になり Δ* との cos は 0.695。ASW は 0.078、隣接ペアは 5 まで戻る。最近傍重心法の BA は補正なしと厳密に同じ 0.595（クラス平均の差が
  落とした部分空間に入るので、割り当てが変わらない）。単一組織型の救済にもならない（Fibrous: cos 0.80 → 0.67）。
  ただしこれは「施設シフトが組織型の方向を向いている」からではない: 組織型平均の差が張る 13 次元は Δ_raw の二乗ノルムの 0.49 を含むが、
  EBRAINS のスライド 13 枚の偏差ベクトルを無作為に取って張った 13 次元も同じ 0.49 を含む（無作為な 13 次元なら 0.017、EBRAINS の上位 5 主成分で 0.60、
  上位 50 で 0.93。この 3 つは n554 での手計算、n551 では再計算していない）。埋め込みの分散が低次元に集中していて、施設シフトもその部分空間にあるため、データから作ったどんな部分空間を落としても
  シフトごと削れてしまう。個々の組織型方向との cos は |cos| ≤ 0.42（最大 Clear cell −0.42、Transitional +0.34、Microcystic −0.33、他は |cos| < 0.26）。
- まとめ: 新施設から組織型を問わず 20〜50 例あれば平均を揃えるだけで全例と同等。1 型に偏った標本は避ける（偏りは例数で
  解消しない）。
"""
MAIN_WEIGHTING = "min"
SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else np.nan


def _rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(a - ref) / np.linalg.norm(ref))


# ── 評価器（平行移動 δ を与えると下流指標を返す）────────────────────────────────

class Evaluator:
    """patho2 を δ だけ平行移動したときの指標。重い前処理（群平均・クラス平均）は 1 回だけ。"""

    def __init__(self, merged: pd.DataFrame, cfg: dict, reference: str, other: str) -> None:
        X = np.stack(merged["embedding"].values).astype(np.float64)
        src = merged["source"].to_numpy()
        sub = merged["subtype"].to_numpy()
        self.is_ref, self.is_oth = src == reference, src == other
        self.E, self.P = X[self.is_ref], X[self.is_oth]
        self.site = np.concatenate([np.full(len(self.E), reference), np.full(len(self.P), other)])
        self.pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
        # 組織型（最近傍重心法。nc.py run_histotype と同じ: EBRAINS のクラス平均に patho2 を割り当てる）
        classes: list[str] = cfg["lp_histotype"]["classes"]
        self.classes = classes
        self.C = np.stack([X[self.is_ref & (sub == c)].mean(0) for c in classes])
        m = self.is_oth & np.isin(sub, classes)
        self.ZP = X[m]
        self.yP = pd.Series(sub[m]).map({c: i for i, c in enumerate(classes)}).to_numpy()
        # 樹形図（dendrogram.py と同じ: 両施設にある組織型の平均ベクトル、Ward）。平行移動は patho2 の平均に足すだけ
        self.groups = group_means(merged, cfg)
        self.group_is_oth = (self.groups["source"] == other).to_numpy()
        self.base_means = np.stack(self.groups["mean"].values)

    def asw_site(self, delta: np.ndarray) -> float:
        """harmonization.compute_metrics の asw_batch と同じ定義（PCA 50 → 施設ラベルのシルエット）。"""
        Xall = np.vstack([self.E, self.P + delta])
        return float(silhouette_score(_reduce(Xall, self.pre_n), self.site))

    def histotype_ba(self, delta: np.ndarray) -> float:
        Z = self.ZP + delta
        D = np.sqrt(((Z[:, None, :] - self.C[None, :, :]) ** 2).sum(-1))
        return float(balanced_accuracy_score(self.yP, D.argmin(1)))

    def adjacent_pairs(self, delta: np.ndarray) -> int:
        means = self.base_means + np.where(self.group_is_oth[:, None], delta[None, :], 0.0)
        g = self.groups.assign(mean=list(means))
        D = raw_distance(g)
        return int(pair_stats(D, ward_linkage(D), g)["adjacent_pairs"])

    def evaluate(self, delta: np.ndarray, delta_star: np.ndarray | None = None) -> dict[str, float]:
        out = {"shift_norm": float(np.linalg.norm(delta)),
               "asw_site": self.asw_site(delta), "histotype_ba": self.histotype_ba(delta),
               "adjacent_pairs": self.adjacent_pairs(delta)}
        if delta_star is not None:
            out = {"cos_vs_star": _cos(delta, delta_star), "rel_err_vs_star": _rel_err(delta, delta_star), **out}
        return out


# ── A. 症例構成の切り分け ─────────────────────────────────────────────────────

def matched_groups(sub: np.ndarray, is_ref: np.ndarray, is_oth: np.ndarray, min_n: int, cfg: dict) -> list[str]:
    """両施設に min_n 例以上ある組織型（SFT も 1 群。shift_direction.py と同じ群）。"""
    return [s for s in ordered_subtypes(set(sub), cfg)
            if ((sub == s) & is_ref).sum() >= min_n and ((sub == s) & is_oth).sum() >= min_n]


def group_weights(n_ref: np.ndarray, n_oth: np.ndarray, weighting: str) -> np.ndarray:
    if weighting == "min":
        w = np.minimum(n_ref, n_oth).astype(np.float64)
    elif weighting == "sum":
        w = (n_ref + n_oth).astype(np.float64)
    elif weighting == "uniform":
        w = np.ones(len(n_ref))
    else:
        raise ValueError(weighting)
    return w / w.sum()


def casemix(X: np.ndarray, sub: np.ndarray, ev: Evaluator, cfg: dict, min_n: int,
            reference: str, other: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    is_ref, is_oth = ev.is_ref, ev.is_oth
    groups = matched_groups(sub, is_ref, is_oth, min_n, cfg)
    n_ref = np.array([((sub == g) & is_ref).sum() for g in groups])
    n_oth = np.array([((sub == g) & is_oth).sum() for g in groups])
    d_g = np.stack([X[(sub == g) & is_ref].mean(0) - X[(sub == g) & is_oth].mean(0) for g in groups])
    d_raw = ev.E.mean(0) - ev.P.mean(0)
    print(f"  Δ* groups ({len(groups)}, min_n={min_n}): " + ", ".join(shorten(g, cfg) for g in groups))

    deltas: dict[str, np.ndarray] = {}
    rows, grp_rows = [], []
    for w_name in WEIGHTINGS:
        w = group_weights(n_ref, n_oth, w_name)
        d_star = (w[:, None] * d_g).sum(0)
        deltas[w_name] = d_star
        rows.append({"weighting": w_name, "n_groups": len(groups),
                     "norm_raw": float(np.linalg.norm(d_raw)), "norm_star": float(np.linalg.norm(d_star)),
                     "cos_raw_star": _cos(d_raw, d_star), "rel_diff_raw_star": _rel_err(d_raw, d_star),
                     **{k: v for k, v in ev.evaluate(d_star).items() if k != "shift_norm"}})
        for g, a, b, wi, dg in zip(groups, n_ref, n_oth, w, d_g):
            grp_rows.append({"weighting": w_name, "group": g, f"n_{reference}": int(a), f"n_{other}": int(b),
                             "weight": float(wi), "group_shift_norm": float(np.linalg.norm(dg)),
                             "cos_group_vs_star": _cos(dg, d_star), "cos_group_vs_raw": _cos(dg, d_raw)})
    # 参照行: 補正なし / Δ_raw（= centroid 補正そのもの）
    ref_rows = [{"weighting": "none", "n_groups": 0, "norm_raw": float(np.linalg.norm(d_raw)), "norm_star": np.nan,
                 "cos_raw_star": np.nan, "rel_diff_raw_star": np.nan,
                 **{k: v for k, v in ev.evaluate(np.zeros_like(d_raw)).items() if k != "shift_norm"}},
                {"weighting": "raw", "n_groups": 0, "norm_raw": float(np.linalg.norm(d_raw)),
                 "norm_star": float(np.linalg.norm(d_raw)), "cos_raw_star": 1.0, "rel_diff_raw_star": 0.0,
                 **{k: v for k, v in ev.evaluate(d_raw).items() if k != "shift_norm"}}]
    table = pd.DataFrame(ref_rows + rows)
    return table, pd.DataFrame(grp_rows), {"raw": d_raw, **deltas}


# ── B. 推定に要る例数 ─────────────────────────────────────────────────────────

def largest_remainder(counts: np.ndarray, n: int) -> np.ndarray:
    """構成比 counts/sum に n を最大剰余法で割り当てる（各群の上限は counts）。"""
    quota = counts / counts.sum() * n
    alloc = np.floor(quota).astype(int)
    rest = n - alloc.sum()
    order = np.argsort(-(quota - alloc), kind="stable")
    for i in order[:rest]:
        alloc[i] += 1
    return np.minimum(alloc, counts)


def bio_basis(X: np.ndarray, sub: np.ndarray, is_ref: np.ndarray, min_n: int, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """EBRAINS の組織型平均の差 {μ_E(g) − μ_E(all)}（EBRAINS に min_n 例以上の群）の正規直交基底（SVD）。"""
    groups = [s for s in ordered_subtypes(set(sub[is_ref]), cfg) if ((sub == s) & is_ref).sum() >= min_n]
    mu_all = X[is_ref].mean(0)
    V = np.stack([X[is_ref & (sub == g)].mean(0) - mu_all for g in groups])
    U, s, _ = np.linalg.svd(V.T, full_matrices=False)
    rank = int((s > s.max() * 1e-8).sum())
    return U[:, :rank], groups


def project_out(delta: np.ndarray, B: np.ndarray) -> np.ndarray:
    return delta - B @ (B.T @ delta)


def _evaluate_many(ev: Evaluator, deltas: np.ndarray, d_star: np.ndarray) -> list[dict[str, float]]:
    return [ev.evaluate(d, d_star) for d in deltas]


def evaluate_parallel(ev: Evaluator, deltas: list[np.ndarray], d_star: np.ndarray, n_jobs: int) -> list[dict[str, float]]:
    """δ のリストを並列に評価（1 評価 ≈ 0.3 s: PCA 50 ＋ シルエット。worker ごとにまとめて渡す）。"""
    if len(deltas) <= 2 or n_jobs == 1:
        return _evaluate_many(ev, np.stack(deltas), d_star)
    chunks = np.array_split(np.stack(deltas), min(n_jobs, len(deltas)))
    out = Parallel(n_jobs=n_jobs)(delayed(_evaluate_many)(ev, c, d_star) for c in chunks)
    return [r for part in out for r in part]


def subsample(X: np.ndarray, sub: np.ndarray, ev: Evaluator, cfg: dict, deltas: dict[str, np.ndarray],
              reference: str, other: str, n_draws: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    se_cfg = cfg.get("shift_estimation", {})
    seed: int = se_cfg.get("seed", 42)
    n_jobs: int = se_cfg.get("n_jobs", 1)
    min_n: int = se_cfg.get("min_n", cfg.get("shift_direction", {}).get("min_n", 5))
    n_list: list[int] = list(se_cfg.get("n_list", [3, 5, 10, 20, 50, 100, 200]))
    singles: list[str] = list(se_cfg.get("single_subtypes", []))
    d_star = deltas[MAIN_WEIGHTING]
    d_raw = deltas["raw"]
    mu_E = ev.E.mean(0)
    idx_P = np.flatnonzero(ev.is_oth)
    n_all = len(idx_P)
    sub_P = sub[idx_P]
    subs_P = ordered_subtypes(set(sub_P), cfg)
    counts_P = np.array([(sub_P == s).sum() for s in subs_P])
    idx_by_sub = {s: idx_P[sub_P == s] for s in subs_P}
    B, basis_groups = bio_basis(X, sub, ev.is_ref, min_n, cfg)
    print(f"  bio basis: rank {B.shape[1]} from {len(basis_groups)} {reference} groups (min_n={min_n})")
    print(f"  Δ_raw: |Δ⊥| / |Δ| = {np.linalg.norm(project_out(d_raw, B)) / np.linalg.norm(d_raw):.3f}, "
          f"Δ*: {np.linalg.norm(project_out(d_star, B)) / np.linalg.norm(d_star):.3f}")

    # 設定の一覧（scheme, subtype, n, 抜き方）。n が群の全例なら決定的なので 1 回
    settings: list[tuple[str, str | None, int]] = []
    for n in sorted(set(n_list) | {n_all}):
        if n <= n_all:
            settings.append(("random", None, n))
    for n in sorted(set(n_list) | {n_all}):
        if n <= n_all:
            settings.append(("stratified", None, n))
    for s in singles:
        n_s = len(idx_by_sub[s])
        for n in sorted({n for n in n_list if n < n_s} | {n_s}):
            settings.append(("single", s, n))

    draw_rows, agg_rows = [], []
    metrics = ["cos_vs_star", "rel_err_vs_star", "shift_norm", "asw_site", "histotype_ba", "adjacent_pairs"]

    def aggregate(scheme: str, subtype: str | None, n: int, rows: list[dict]) -> dict:
        df = pd.DataFrame(rows)
        out = {"scheme": scheme, "subtype": subtype or "", "n": n, "n_draws": len(df)}
        for m in metrics:
            v = df[m].to_numpy(dtype=np.float64)
            out[f"{m}_median"] = float(np.median(v))
            out[f"{m}_q025"] = float(np.quantile(v, 0.025))
            out[f"{m}_q975"] = float(np.quantile(v, 0.975))
        return out

    # 参照行: 補正なし（n = 0）・全例の Δ_raw・Δ*・Δ_raw⊥
    for scheme, d in (("none", np.zeros_like(d_raw)), ("full_raw", d_raw), ("full_matched", d_star),
                      ("full_raw_ortho", project_out(d_raw, B))):
        r = ev.evaluate(d, d_star)
        n = 0 if scheme == "none" else n_all
        draw_rows.append({"scheme": scheme, "subtype": "", "n": n, "draw": 0, **r})
        agg_rows.append(aggregate(scheme, None, n, [r]))
        print(f"  [{scheme}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    t0 = time.time()
    for i, (scheme, subtype, n) in enumerate(settings):
        rng = np.random.default_rng([seed, i])
        pool_n = len(idx_by_sub[subtype]) if scheme == "single" else n_all
        k_draws = 1 if n >= pool_n else n_draws
        est: list[np.ndarray] = []
        for _ in range(k_draws):
            if scheme == "random":
                S = rng.choice(idx_P, n, replace=False)
            elif scheme == "stratified":
                alloc = largest_remainder(counts_P, n)
                S = np.concatenate([rng.choice(idx_by_sub[s], a, replace=False) for s, a in zip(subs_P, alloc) if a > 0])
            else:
                S = rng.choice(idx_by_sub[subtype], n, replace=False)
            est.append(mu_E - X[S].mean(0))
        rows_plain = evaluate_parallel(ev, est, d_star, n_jobs)
        rows_ortho = (evaluate_parallel(ev, [project_out(d, B) for d in est], d_star, n_jobs)
                      if scheme != "stratified" else [])
        for d, r in enumerate(rows_plain):
            draw_rows.append({"scheme": scheme, "subtype": subtype or "", "n": n, "draw": d, **r})
        for d, r in enumerate(rows_ortho):
            draw_rows.append({"scheme": f"{scheme}_ortho", "subtype": subtype or "", "n": n, "draw": d, **r})
        agg_rows.append(aggregate(scheme, subtype, n, rows_plain))
        if rows_ortho:
            agg_rows.append(aggregate(f"{scheme}_ortho", subtype, n, rows_ortho))
        a = agg_rows[-2] if rows_ortho else agg_rows[-1]
        tag = f"{scheme}" + (f"/{shorten(subtype, cfg)}" if subtype else "")
        msg = (f"  [{i + 1}/{len(settings)} {tag} n={n} draws={k_draws}] cos={a['cos_vs_star_median']:.3f} "
               f"[{a['cos_vs_star_q025']:.3f}, {a['cos_vs_star_q975']:.3f}]  ASW={a['asw_site_median']:.4f}  "
               f"BA={a['histotype_ba_median']:.3f}  adj={a['adjacent_pairs_median']:.0f}")
        if rows_ortho:
            o = agg_rows[-1]
            msg += f"  | ortho cos={o['cos_vs_star_median']:.3f} ASW={o['asw_site_median']:.4f} BA={o['histotype_ba_median']:.3f}"
        print(msg + f"  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(agg_rows), pd.DataFrame(draw_rows)


# ── summary.md ────────────────────────────────────────────────────────────────

def _fmt(df: pd.DataFrame, m: str, digits: int = 3) -> pd.Series:
    return df.apply(lambda r: f"{r[f'{m}_median']:.{digits}f} [{r[f'{m}_q025']:.{digits}f}, {r[f'{m}_q975']:.{digits}f}]",
                    axis=1)


def write_summary(path: Path, casemix_table: pd.DataFrame, agg: pd.DataFrame, cfg: dict, n_draws: int,
                  reference: str, other: str, chance: dict | None) -> None:
    L: list[str] = []
    L.append("# shift_estimation — 平行移動のシフト推定（症例構成の切り分け・推定に要る例数）\n")
    L.append(f"入力: TITAN original 埋め込み。向き: {SITE_LABEL.get(other, other)} → {SITE_LABEL.get(reference, reference)}"
             f"（Δ = mean({reference}) − mean({other})）。評価はすべて {other} 全スライドに平行移動を適用した後の値。\n")
    if chance:
        L.append(f"ASW(site) の偶然のレベル（`out/harmonization/site_null.csv`、original / within_subtype）: "
                 f"{chance['null_mean']:.4f} [{chance['null_q025']:.4f}, {chance['null_q975']:.4f}]。\n")

    L.append("## A. 症例構成の切り分け（casemix.csv）\n")
    L.append("Δ* = Σ_g w_g (μ_E(g) − μ_P(g))、両施設 min_n 例以上の組織型（SFT 含む）。"
             "none = 補正なし、raw = Δ_raw（= centroid 補正そのもの）。\n")
    L.append("| 重み | ‖Δ_raw‖ | ‖Δ*‖ | cos(Δ_raw, Δ*) | ‖Δ_raw − Δ*‖ / ‖Δ*‖ | ASW(site) | 隣接ペア | 組織型 BA |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in casemix_table.itertuples():
        f = lambda v, d=3: "—" if pd.isna(v) else f"{v:.{d}f}"
        L.append(f"| {r.weighting} | {f(r.norm_raw, 2)} | {f(r.norm_star, 2)} | {f(r.cos_raw_star, 4)} | "
                 f"{f(r.rel_diff_raw_star)} | {f(r.asw_site, 4)} | {int(r.adjacent_pairs)} | {f(r.histotype_ba)} |")
    L.append("")

    L.append("## B. 推定に要る例数（subsample.csv）\n")
    L.append(f"各設定 {n_draws} 回（n が全例のときは決定的で 1 回）。中央値 [2.5, 97.5 パーセンタイル]。"
             "cos は Δ*（A の主重み）との cos。\n")

    def table(scheme: str, title: str) -> None:
        d = agg[agg["scheme"] == scheme].sort_values("n")
        if d.empty:
            return
        L.append(f"### {title}\n")
        L.append("| n | draws | cos(Δ̂, Δ*) | ‖Δ̂ − Δ*‖ / ‖Δ*‖ | ASW(site) | 組織型 BA | 隣接ペア |")
        L.append("|---|---|---|---|---|---|---|")
        for _, r in d.iterrows():
            L.append(f"| {int(r['n'])} | {int(r['n_draws'])} | {_fmt(d.loc[[r.name]], 'cos_vs_star').iloc[0]} | "
                     f"{_fmt(d.loc[[r.name]], 'rel_err_vs_star', 2).iloc[0]} | {_fmt(d.loc[[r.name]], 'asw_site', 4).iloc[0]} | "
                     f"{_fmt(d.loc[[r.name]], 'histotype_ba').iloc[0]} | {_fmt(d.loc[[r.name]], 'adjacent_pairs', 0).iloc[0]} |")
        L.append("")

    ref = agg[agg["scheme"].isin(["none", "full_raw", "full_matched", "full_raw_ortho"])]
    L.append("### 参照（補正なし / 全例）\n")
    L.append("| 行 | n | cos(Δ, Δ*) | ASW(site) | 組織型 BA | 隣接ペア |")
    L.append("|---|---|---|---|---|---|")
    for _, r in ref.iterrows():
        c = "—" if pd.isna(r["cos_vs_star_median"]) else f"{r['cos_vs_star_median']:.4f}"
        L.append(f"| {r['scheme']} | {int(r['n'])} | {c} | {r['asw_site_median']:.4f} | {r['histotype_ba_median']:.3f} | "
                 f"{int(r['adjacent_pairs_median'])} |")
    L.append("")
    table("random", "無作為（random）")
    table("random_ortho", "無作為 ＋ 生物学的方向の除去（random_ortho）")
    table("stratified", "層化（stratified、patho2 の構成比）")
    for s in sorted(agg.loc[agg["scheme"] == "single", "subtype"].unique(),
                    key=lambda s: -agg.loc[(agg["scheme"] == "single") & (agg["subtype"] == s), "n"].max()):
        table_s = agg[(agg["scheme"] == "single") & (agg["subtype"] == s)]
        L.append(f"### 単一組織型（single）: {s}（{shorten(s, cfg)}）\n")
        L.append("| n | draws | cos(Δ̂, Δ*) | ASW(site) | 組織型 BA | 隣接ペア | ortho: cos | ortho: ASW | ortho: BA |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        o_all = agg[(agg["scheme"] == "single_ortho") & (agg["subtype"] == s)]
        for _, r in table_s.sort_values("n").iterrows():
            o = o_all[o_all["n"] == r["n"]]
            row = table_s.loc[[r.name]]
            cells = [str(int(r["n"])), str(int(r["n_draws"])), _fmt(row, "cos_vs_star").iloc[0],
                     _fmt(row, "asw_site", 4).iloc[0], _fmt(row, "histotype_ba").iloc[0],
                     _fmt(row, "adjacent_pairs", 0).iloc[0]]
            if not o.empty:
                cells += [_fmt(o, "cos_vs_star").iloc[0], _fmt(o, "asw_site", 4).iloc[0], _fmt(o, "histotype_ba").iloc[0]]
            else:
                cells += ["—"] * 3
            L.append("| " + " | ".join(cells) + " |")
        L.append("")

    L.append("## 読み\n")
    L.append(READING.strip() + "\n")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  saved: {path}")


def _chance(cfg: dict) -> dict | None:
    """ASW(site) の偶然のレベル（harmonization.py の施設ラベル並べ替え、original / within_subtype）。"""
    null_path = Path(cfg["output_dir"]) / "harmonization" / "site_null.csv"
    if not null_path.exists():
        return None
    nt = pd.read_csv(null_path)
    r = nt[(nt["state"] == "original") & (nt["null"] == "within_subtype") & (nt["metric"] == "asw_batch")]
    return r.iloc[0].to_dict() if not r.empty else None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-draws", type=int, default=None, help="config の shift_estimation.n_draws を上書き（試走用）")
    ap.add_argument("--summary-only", action="store_true", help="計算せず既存の CSV から summary.md だけ書き直す")
    args = ap.parse_args()

    cfg = load_config()
    cfg_hist_set = cfg.get("nc", {}).get("histotype_set", "main")
    cfg = apply_set(cfg, cfg_hist_set)
    se_cfg = cfg.get("shift_estimation", {})
    n_draws: int = args.n_draws or se_cfg.get("n_draws", 200)
    min_n: int = se_cfg.get("min_n", cfg.get("shift_direction", {}).get("min_n", 5))
    reference: str = cfg["reference"]
    out_dir = Path(cfg["output_dir"]) / "shift_estimation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        agg = pd.read_csv(out_dir / "subsample.csv")
        agg["subtype"] = agg["subtype"].fillna("")
        cm_table = pd.read_csv(out_dir / "casemix.csv")
        n_draws = int(agg["n_draws"].max())
        other = next(s for s in cfg["embedding"] if s != reference)
        write_summary(out_dir / "summary.md", cm_table, agg, cfg, n_draws, reference, other, _chance(cfg))
        return

    print(f"[shift_estimation] variant=original  histotype set={cfg_hist_set}  n_draws={n_draws}  min_n={min_n}")
    merged = load_data(cfg, cfg.get("variants", {}).get("original", "original"))
    if merged.empty:
        print("  SKIP: no data")
        return
    merged = merged.sort_values(["source", "case_id"]).reset_index(drop=True)
    other = next(s for s in merged["source"].unique() if s != reference)
    print(f"  {len(merged)} cases  reference={reference}  other={other}  "
          f"{merged['source'].value_counts().to_dict()}")

    X = np.stack(merged["embedding"].values).astype(np.float64)
    sub = merged["subtype"].to_numpy()
    ev = Evaluator(merged, cfg, reference, other)

    # 整合性: 評価器の ASW(site) は harmonization.compute_metrics の asw_batch と一致するはず
    m0 = compute_metrics(merged, cfg)
    a0 = ev.asw_site(np.zeros(X.shape[1]))
    print(f"  check ASW(site) none: compute_metrics={m0['asw_batch']:.6f}  evaluator={a0:.6f}")

    print("[A] case-mix decomposition")
    cm_table, cm_groups, deltas = casemix(X, sub, ev, cfg, min_n, reference, other)
    cm_table.to_csv(out_dir / "casemix.csv", index=False)
    cm_groups.to_csv(out_dir / "casemix_groups.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(cm_table.round(4).to_string(index=False))
        print(cm_groups[cm_groups["weighting"] == MAIN_WEIGHTING].round(4).to_string(index=False))
    print(f"  saved: {out_dir}/casemix.csv, casemix_groups.csv")

    print("[B] subsample estimation")
    t0 = time.time()
    agg, draws = subsample(X, sub, ev, cfg, deltas, reference, other, n_draws)
    agg.to_csv(out_dir / "subsample.csv", index=False)
    draws.to_csv(out_dir / "subsample_draws.csv", index=False)
    print(f"  saved: {out_dir}/subsample.csv, subsample_draws.csv  ({time.time() - t0:.0f}s, {len(draws)} draws)")

    write_summary(out_dir / "summary.md", cm_table, agg, cfg, n_draws, reference, other, _chance(cfg))


if __name__ == "__main__":
    main()
