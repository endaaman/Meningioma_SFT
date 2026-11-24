"""transfer_rare_crossover.py — 例数の少ない腫瘍の施設のずれは、自前の n 枚で推定するより、よくある腫瘍（髄膜腫・グリオーマ）から
借りた方が良いか。その境目の n（ken の指示 2026-11-24。論文の構図: よくある腫瘍 2 つ ＋ それぞれの鑑別）。

想定: 公開の参照コホート（EBRAINS）は全部使える。新しい施設（院内 = patho2）には当て先の群 T のスライドが n 枚しか無い。
補正は平行移動だけ（倍率つきの法は n が小さいと推定できないので対象外）。院内の T の N 枚から無作為に n 枚 S をとり、残り
R（N − n 枚、推定に使わない）で評価する。条件（同じ抽出で対にして比べる）:
    none                補正なし
    own                 自前（n 枚）: Δ_T(n) = mean(EBRAINS の T) − mean(S)
    borrow (source)     借りる: Δ_B = mean(EBRAINS の借りる元) − mean(院内の借りる元)（両施設の全枚数をまとめた平均の差 =
                        transfer_correction.py の推定元 meningioma / glioma と同じ推定のしかた）。借りる元（source）:
                          meningioma       科 meningioma（SFT を含まない）
                          glioma           科 glioma（ast + glm + odg）
                          *_rest           T が借りる元の科に入るときは T を除いた残り（髄膜腫のサブタイプ ← 残りの髄膜腫、
                                           ast / glm / odg ← 残りのグリオーマ 2 群。参考）
    borrow_thin (source) 借りる（参考、間引き）: 院内の借りる元を THIN_N（20）枚に間引いた Δ（EBRAINS は全部。抽出ごとに引き直す）
群 T と本文の組み合わせ（MAIN_SOURCE）: SFT ← 髄膜腫、lym ← グリオーマ、met ← グリオーマ、ast / glm / odg ← 髄膜腫。もう一方の
    借りる元も出す（SFT ← グリオーマ、lym / met ← 髄膜腫、ast / glm / odg ← 残りのグリオーマ 2 群）。参考に髄膜腫のサブタイプ
    （院内 N_LIST[0] + MIN_REST 枚以上・EBRAINS 2 枚以上。NOS は EBRAINS に無いので外す）← 残りの髄膜腫 / グリオーマ。
n ∈ N_LIST のうち残りが MIN_REST（5）枚以上のもの。各 n で N_DRAWS（500）回。乱数列は default_rng([SEED, crc32(群名), n])（S）と
    [SEED, crc32(群名), n, 1, crc32(借りる元)]（間引き）で、群・n・借りる元ごとに独立（群や借りる元を足しても他は変わらず、
    エンコーダによらず同じ抽出）。

指標（R について。transfer_correction.site_mixing をそのまま呼ぶ: EBRAINS の T と補正後の R を 1 群として渡す）:
    group_gap_sd（主） ‖μ_E,T − (mean(R) + Δ)‖ / RMS、RMS = √((mean_E ‖x − μ_E‖² + mean_R ‖x − mean(R)‖²) / 2)
                       （correction_{encoder}_groups.csv の group_gap_sd と同じ定義。平行移動なので RMS は条件によらない）
    knn_mix            同じ関数の 2 つ目の値（院内 R の 10 近傍に占める EBRAINS の割合 / 期待値）
    hit1（副）         R の各スライド（+ Δ）を EBRAINS 全体の診断ごとの重心に最近傍で当てたとき、T の正解の診断に当たる割合
                       （atlas_retrieval.py の参照: キャッシュ out/atlas/atlas_{encoder}.npy、--min-n 既定 5 のクラス、正解の集合は
                       label_mapping、検索は _dist（euclid）＋ nc_retrieve。titan / prism だけ。キャッシュが無ければ空）
    floor_sd           √tr V / RMS、tr V = Σ_d var_E,d / n_E + var_R,d / (N − n)（shift_transfer.shift_stats）。補正が完全
                       （真の Δ）でも EBRAINS の T と R の平均の標本誤差だけで残る値の期待値 = 全条件に共通の床

理論（平均二乗の近似。norm は高次元で集中するので中央値と比べる）。trS_P / trS_E = 院内 / EBRAINS の T の 1 枚あたりの全分散
    （ddof = 1）、tr V_B = 借りる Δ の標本誤差（shift_stats）、b² = ‖Δ_T − Δ_B‖² からノイズ（tr V_T + tr V_B、shift_stats）を引いたもの:
    own²    ≈ trS_P·(1/n + 1/(N − n))           （EBRAINS の平均は推定と評価で同じものなので打ち消し合う）
    borrow² ≈ b² + trS_E/n_E + tr V_B + trS_P/(N − n)
    floor²  ≈ trS_E/n_E + trS_P/(N − n)
    n*_simple = trS_P / b²（依頼の近似）、n*_metric = trS_P / (b² + trS_E/n_E + tr V_B)（上の 2 式が等しくなる n。この指標での予想）。
    床の確かめ（split_null）: 院内の T を無作為に半分ずつ（⌊N/2⌋ / ⌈N/2⌉）に割り、片方を EBRAINS の代わりにして同じ
    group_gap_sd を出す（施設差が無いときの値、N_DRAWS 回の中央値）と、同じ割り方の √tr V / RMS（理論の床）を並べる。

交点 n*（実測）: 自前の group_gap_sd の中央値が借りる場合の中央値（同じ n、同じ R）を下回る最小の n（格子の上）。それより大きい
    n のすべてでも下回るかを併記。最小の n で既に下回れば「≤ n」、届かなければ「範囲内に交点なし」。log n で線形補間した値も出す。

確認（合わなければ止まる）: T の全枚数の補正なし / 借りる（院内 T の全部に当てた）group_gap_sd が transfer_correction.py の
    correction_{encoder}_groups.csv（当て先 paper / glioma / lymphoma / metastasis の none と推定元 meningioma / glioma、method = centroid。
    SFT ← 髄膜腫は rare_target_{encoder}.csv）と一致するか（*_rest は対応する既存の行が無いので確かめない）。hit1 の補正なし
    （院内 T の全枚数）が atlas の per_slide_{encoder}.csv（nc_euclid, none）とスライドごとに一致するか、atlas の問い合わせの
    埋め込み（manifest_{encoder}.npy）が転移テストのキャッシュと同じか。

出力 (output_dir/transfer_sample_size/。transfer_sample_size.py の n_curve / k_curve / summary とは別名):
    rare_crossover_{encoder}.csv   縦長: encoder, group, family, kind（main / meningioma_subtype）, n_ebrains, n_patho2, n, n_rest,
                                   condition, source, metric, median, q025, q975, mean, n_draws。condition = none / own（source 空）、
                                   borrow / borrow_thin（source = 借りる元）の metric = group_gap_sd / knn_mix / hit1、floor（floor_sd）、
                                   own_vs_borrow / own_vs_borrow_thin（metric = p_own_better = 同じ抽出で自前の方が小さい割合）、
                                   theory_own / theory_none / theory_floor（source 空）・theory_borrow / theory_borrow_thin（median = 理論値）
    rare_crossover_nstar.csv       エンコーダ × 群 × 借りる元: main（本文の組み合わせか）、理論の成分（trS_P, trS_E, n_E, N, rms, ‖Δ_T‖,
                                   ‖Δ_B‖, cos, b², tr V_B）、n*_simple / n*_metric / n*_metric_thin、実測の n*（借りる / 間引き）、
                                   split_null と理論の床
    rare_crossover_summary.md      表（本文の 6 組 → 他の組み合わせ → 髄膜腫のサブタイプ。エンコーダは TITAN が先頭）と確認
    rare_crossover_{encoder}.png   作業用: 6 群の group_gap_sd vs n（中央値と 95% 帯、理論の線、n*）

Usage:
    uv run python transfer_rare_crossover.py [--encoders titan prism meanpool meanpool_virchow] [--n-draws 500] [--n-jobs 12]
"""
from __future__ import annotations

import argparse
import time
import warnings
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402

from atlas_retrieval import ENCODERS as ATLAS_ENCODERS  # noqa: E402
from atlas_retrieval import MIN_N_DEFAULT, _dist, corr_mask, label_mapping, load_atlas, nc_retrieve  # noqa: E402
from shift_transfer import _cos, corrected_cos, shift_stats  # noqa: E402
from transfer_correction import _fmt, _md, load_embeddings, site_mixing  # noqa: E402
from utils.display import ENCODER_LABELS  # noqa: E402
from utils.loader import load_config  # noqa: E402

SEED = 42
N_DRAWS = 500
N_LIST = [2, 3, 5, 8, 10, 15, 20, 30, 50]
MIN_REST = 5                     # 評価に残す院内の T の最小枚数
THIN_N = 20                      # borrow_thin: 院内の借りる元をこの枚数に間引く
N_JOBS = 12
ENCODERS = ["titan", "prism", "meanpool", "meanpool_virchow"]
MAIN_GROUPS = ["SFT", "lym", "met", "ast", "glm", "odg"]          # 表と図の並び（髄膜腫の鑑別 → グリオーマの鑑別 → グリオーマ）
BORROW_FAMILIES = ["meningioma", "glioma"]
MAIN_SOURCE = {"SFT": "meningioma", "lym": "glioma", "met": "glioma", "ast": "meningioma", "glm": "meningioma", "odg": "meningioma"}
SHOW_N = [2, 5, 10, 20, 50]      # summary の一覧に出す自前の n
SOURCE_LABEL = {"meningioma": "髄膜腫", "glioma": "グリオーマ", "meningioma_rest": "残りの髄膜腫", "glioma_rest": "残りのグリオーマ 2 群"}
SOURCE_LABEL_EN = {"meningioma": "meningioma", "glioma": "glioma", "meningioma_rest": "other meningioma", "glioma_rest": "other 2 glioma groups"}
CHECK_TARGET = {"ast": "glioma", "glm": "glioma", "odg": "glioma", "lym": "lymphoma", "met": "metastasis"}   # 他（SFT・髄膜腫）は paper


def _q(v: np.ndarray) -> dict:
    v = v[~np.isnan(v)]
    if not len(v):
        return {"median": np.nan, "q025": np.nan, "q975": np.nan, "mean": np.nan, "n_draws": 0}
    return {"median": float(np.median(v)), "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975)),
            "mean": float(v.mean()), "n_draws": int(len(v))}


def _tr_v(A: np.ndarray, B: np.ndarray) -> float:
    with warnings.catch_warnings():   # 1 枚の群は ddof = 1 の分散が NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return shift_stats({"ebrains": A, "patho2": B})[2]


def _mix(A: np.ndarray, B: np.ndarray, g: str, ref: str, oth: str, cfg: dict) -> dict:
    """site_mixing（transfer_correction）に A = EBRAINS 側、B = 院内側の 1 群として渡した群ごとの値。"""
    Y = np.vstack([A, B])
    st = np.array([ref] * len(A) + [oth] * len(B))
    _, per = site_mixing(Y, st, np.array([g] * len(Y)), ref, cfg)
    return per[0]


# ── reference for hit1 (atlas_retrieval) ─────────────────────────────────────

def load_reference(cfg: dict, enc: str) -> dict | None:
    """atlas_retrieval.py の参照（EBRAINS 全体、--min-n 既定のクラス）の重心と正解の集合。キャッシュが無ければ None（h5 は読まない）。
    クラスの作り方は atlas_retrieval.main と同じ（重心 = 診断ごとのスライド平均、クラスは枚数の多い順）。"""
    adir = Path(cfg["output_dir"]) / "atlas"
    files = [adir / f"atlas_{enc}.npy", adir / f"atlas_{enc}_slides.csv", adir / f"atlas_{enc}_info.json"]
    if enc not in ATLAS_ENCODERS or not all(f.exists() for f in files):
        return None
    atlas, A, _ = load_atlas(cfg, cfg["transfer"][ATLAS_ENCODERS[enc][1]], adir, enc, reload=False)
    counts = atlas["diagnosis"].value_counts()
    classes = sorted(counts[counts >= MIN_N_DEFAULT].index, key=lambda d: (-counts[d], d))
    keep = atlas["diagnosis"].isin(classes).to_numpy()
    cpos = {c: k for k, c in enumerate(classes)}
    y_idx = atlas.loc[keep, "diagnosis"].map(cpos).to_numpy()
    C = np.stack([A[keep][y_idx == k].mean(0) for k in range(len(classes))])
    mapping = label_mapping(cfg, set(atlas["diagnosis"]))
    correct = {lab: [d for d in ds if d in cpos] for lab, ds in mapping.items()}
    return {"C": C, "classes": classes, "correct": correct, "dir": adir, "n_ref": int(keep.sum())}


def hit1(Q: np.ndarray, g: str, ref_info: dict | None) -> np.ndarray | None:
    """Q の各行が T の正解の診断に当たったか（nc_euclid の top-1）。参照が無い・正解の集合が空なら None。"""
    if ref_info is None or not ref_info["correct"].get(g):
        return None
    corr = corr_mask(np.array([g] * len(Q)), ref_info["classes"], ref_info["correct"])
    return nc_retrieve(_dist(Q, ref_info["C"], "euclid"), corr)["hit1"]


# ── simulation ────────────────────────────────────────────────────────────────

def run_cell(g: str, n: int, E_T: np.ndarray, P_T: np.ndarray, borrows: dict, ref_info: dict | None, cfg: dict,
             ref: str, oth: str, n_draws: int) -> list[dict]:
    """群 g・枚数 n の n_draws 回。1 回ごとに S / R と間引きを引き、全条件の指標を R で出す。
    borrows: 借りる元 → (EBRAINS の平均, 院内のスライド, 全枚数の Δ)。"""
    key = zlib.crc32(g.encode())
    rng = np.random.default_rng([SEED, key, n])
    rng_thin = {s: np.random.default_rng([SEED, key, n, 1, zlib.crc32(s.encode())]) for s in borrows}
    mu_e_T = E_T.mean(0)
    rows = []
    for d in range(n_draws):
        perm = rng.permutation(len(P_T))
        S, R = perm[:n], perm[n:]
        deltas = {("none", ""): np.zeros_like(mu_e_T), ("own", ""): mu_e_T - P_T[S].mean(0)}
        for s, (mu_e_B, P_B, delta_B) in borrows.items():
            thin = rng_thin[s].choice(len(P_B), THIN_N, replace=False)
            deltas[("borrow", s)] = delta_B
            deltas[("borrow_thin", s)] = mu_e_B - P_B[thin].mean(0)
        XR = P_T[R]
        tr_v = _tr_v(E_T, XR)
        for (c, s), dl in deltas.items():
            r = _mix(E_T, XR + dl, g, ref, oth, cfg)
            h = hit1(XR + dl, g, ref_info)
            rows.append({"draw": d, "condition": c, "source": s, "group_gap_sd": r["group_gap_sd"], "knn_mix": r["knn_mix"],
                         "hit1": float(h.mean()) if h is not None else np.nan, "floor_sd": float(np.sqrt(tr_v) / r["rms"])})
    return rows


def split_null(g: str, P_T: np.ndarray, cfg: dict, ref: str, oth: str, n_draws: int) -> tuple[float, float]:
    """院内の T を半分ずつに割り、片方を EBRAINS の代わりにした group_gap_sd（施設差なし）と √tr V / RMS の中央値。"""
    rng = np.random.default_rng([SEED, zlib.crc32(g.encode()), 0, 2])
    obs, th = [], []
    h = len(P_T) // 2
    for _ in range(n_draws):
        perm = rng.permutation(len(P_T))
        A, B = P_T[perm[:h]], P_T[perm[h:]]
        r = _mix(A, B, g, ref, oth, cfg)
        obs.append(r["group_gap_sd"])
        th.append(np.sqrt(_tr_v(A, B)) / r["rms"])
    return float(np.median(obs)), float(np.median(th))


def crossover(ns: list[int], own: list[float], other: list[float]) -> tuple[str, float, bool]:
    """自前の中央値が other を下回る最小の n（表示用の文字列）、log n で補間した値、それより大きい n でも下回り続けるか。"""
    below = [o < b for o, b in zip(own, other)]
    if not any(below):
        return "範囲内に交点なし", np.nan, False
    i = below.index(True)
    stays = all(below[i:])
    if i == 0:
        return f"≤ {ns[0]}", np.nan, stays
    d0, d1 = own[i - 1] - other[i - 1], own[i] - other[i]
    x0, x1 = np.log(ns[i - 1]), np.log(ns[i])
    return str(ns[i]), float(np.exp(x0 + (x1 - x0) * d0 / (d0 - d1))), stays


def run_encoder(cfg: dict, enc: str, n_draws: int, n_jobs: int) -> tuple[pd.DataFrame, list[dict], dict]:
    print(f"[transfer_rare_crossover / {enc}]")
    ref = cfg["reference"]
    tdir = Path(cfg["output_dir"]) / "transfer"
    man, X = load_embeddings(cfg, enc, tdir, reload=False)
    fam, grp, site = man["family"].to_numpy(), man["group"].astype(str).to_numpy(), man["site"].to_numpy()
    oth = next(s for s in pd.unique(site) if s != ref)
    e = site == ref
    cnt = man.groupby(["group", "site"]).size().unstack(fill_value=0)
    men_sub = [g for g in cnt.index if g in set(grp[fam == "meningioma"])
               and cnt.loc[g, oth] >= N_LIST[0] + MIN_REST and cnt.loc[g, ref] >= 2]
    groups = [(g, "main") for g in MAIN_GROUPS] + [(g, "meningioma_subtype") for g in sorted(men_sub, key=lambda g: -cnt.loc[g, oth])]
    ref_info = load_reference(cfg, enc)
    checks = {"atlas": None if ref_info is None else {"n_classes": len(ref_info["classes"]), "n_ref": ref_info["n_ref"]}}

    ctx, tasks = {}, []
    for g, kind in groups:
        gm = grp == g
        f = str(fam[gm][0])
        borrows, bsets = {}, {}
        for bf in BORROW_FAMILIES:
            bm = (fam == bf) & ~gm                    # T が借りる元の科に入るときは T を除く（*_rest）
            s = bf + ("_rest" if f == bf else "")
            mu_e_B = X[bm & e].mean(0)
            borrows[s] = (mu_e_B, X[bm & ~e], mu_e_B - X[bm & ~e].mean(0))
            bsets[s] = (X[bm & e], X[bm & ~e])
        main_src = MAIN_SOURCE.get(g, "meningioma_rest")
        ns = [n for n in N_LIST if int((gm & ~e).sum()) - n >= MIN_REST]
        ctx[g] = {"kind": kind, "family": f, "E_T": X[gm & e], "P_T": X[gm & ~e], "borrows": borrows, "bsets": bsets, "ns": ns,
                  "main_source": main_src, "ids_P": man.loc[gm & ~e, "slide_id"].to_numpy()}
        tasks += [(g, n) for n in ns]
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(
        delayed(run_cell)(g, n, ctx[g]["E_T"], ctx[g]["P_T"], ctx[g]["borrows"], ref_info, cfg, ref, oth, n_draws) for g, n in tasks)
    print(f"  {len(tasks)} cells × {n_draws} draws ({time.time() - t0:.0f}s)")
    draws = {(g, n): pd.DataFrame(rows) for (g, n), rows in zip(tasks, out)}

    rows, nstar = [], []
    for g, kind in groups:
        c = ctx[g]
        E_T, P_T, ns = c["E_T"], c["P_T"], c["ns"]
        N, n_e = len(P_T), len(E_T)
        base = {"encoder": enc, "group": g, "family": c["family"], "kind": kind, "n_ebrains": n_e, "n_patho2": N}
        full = _mix(E_T, P_T, g, ref, oth, cfg)
        rms = full["rms"]
        trS_P, trS_E = float(P_T.var(0, ddof=1).sum()), float(E_T.var(0, ddof=1).sum())
        d_T, nd_T, tv_T, ndc_T = shift_stats({"ebrains": E_T, "patho2": P_T})
        sn_obs, sn_th = split_null(g, P_T, cfg, ref, oth, n_draws)
        conds = [("none", ""), ("own", "")] + [(k, s) for s in c["borrows"] for k in ("borrow", "borrow_thin")]
        med = {cs: [] for cs in conds}
        for n in ns:
            dr = draws[(g, n)]
            nb = {**base, "n": n, "n_rest": N - n}
            wide = {(k, s): dr[(dr["condition"] == k) & (dr["source"] == s)].set_index("draw") for k, s in conds}
            for (k, s) in conds:
                for m in ("group_gap_sd", "knn_mix", "hit1"):
                    rows.append({**nb, "condition": k, "source": s, "metric": m, **_q(wide[(k, s)][m].to_numpy(dtype=float))})
                med[(k, s)].append(float(np.median(wide[(k, s)]["group_gap_sd"])))
            rows.append({**nb, "condition": "floor", "source": "", "metric": "floor_sd",
                         **_q(wide[("none", "")]["floor_sd"].to_numpy(dtype=float))})
            for (k, s) in conds[2:]:
                p = float((wide[("own", "")]["group_gap_sd"] < wide[(k, s)]["group_gap_sd"]).mean())
                rows.append({**nb, "condition": f"own_vs_{k}", "source": s, "metric": "p_own_better", "median": p, "q025": np.nan,
                             "q975": np.nan, "mean": p, "n_draws": len(wide[(k, s)])})
            r_ = N - n
            th = {("theory_own", ""): np.sqrt(trS_P * (1 / n + 1 / r_)) / rms,
                  ("theory_none", ""): np.sqrt(max(nd_T ** 2 - tv_T, 0) + trS_E / n_e + trS_P / r_) / rms,
                  ("theory_floor", ""): np.sqrt(trS_E / n_e + trS_P / r_) / rms}
            for s, (E_B, P_B) in c["bsets"].items():
                d_B, _, tv_B, _ = shift_stats({"ebrains": E_B, "patho2": P_B})
                tv_thin = float(E_B.var(0, ddof=1).sum() / len(E_B) + P_B.var(0, ddof=1).sum() / THIN_N)
                b2 = float(np.sum((d_T - d_B) ** 2)) - tv_T - tv_B
                th[("theory_borrow", s)] = np.sqrt(max(b2, 0) + trS_E / n_e + tv_B + trS_P / r_) / rms
                th[("theory_borrow_thin", s)] = np.sqrt(max(b2, 0) + trS_E / n_e + tv_thin + trS_P / r_) / rms
            for (k, s), v in th.items():
                rows.append({**nb, "condition": k, "source": s, "metric": "group_gap_sd", "median": float(v), "q025": np.nan,
                             "q975": np.nan, "mean": float(v), "n_draws": 0})
        for s, (E_B, P_B) in c["bsets"].items():
            d_B, nd_B, tv_B, ndc_B = shift_stats({"ebrains": E_B, "patho2": P_B})
            tv_thin = float(E_B.var(0, ddof=1).sum() / len(E_B) + P_B.var(0, ddof=1).sum() / THIN_N)
            obs2 = float(np.sum((d_T - d_B) ** 2))
            b2 = obs2 - tv_T - tv_B
            den, den_thin = b2 + trS_E / n_e + tv_B, b2 + trS_E / n_e + tv_thin
            x_b, xi_b, st_b = crossover(ns, med[("own", "")], med[("borrow", s)])
            x_t, xi_t, st_t = crossover(ns, med[("own", "")], med[("borrow_thin", s)])
            nstar.append({**base, "source": s, "main": s == c["main_source"], "n_src_ebrains": len(E_B), "n_src_patho2": len(P_B),
                          "rms": rms, "trS_P": trS_P, "trS_E": trS_E, "trS_E_over_nE": trS_E / n_e,
                          "shift_norm_T": nd_T, "shift_norm_T_corrected": ndc_T, "shift_norm_B": nd_B, "noise_trace_T": tv_T,
                          "noise_trace_B": tv_B, "noise_trace_B_thin": tv_thin,
                          "cos_T_B": _cos(d_T, d_B), "cos_T_B_corrected": corrected_cos(d_T, ndc_T, d_B, ndc_B),
                          "diff_norm2_obs": obs2, "b2": b2,
                          "nstar_simple": trS_P / b2 if b2 > 0 else np.inf,
                          "nstar_metric": trS_P / den if den > 0 else np.inf,
                          "nstar_metric_thin": trS_P / den_thin if den_thin > 0 else np.inf,
                          "nstar_obs": x_b, "nstar_obs_interp": xi_b, "nstar_obs_stays": st_b,
                          "nstar_obs_thin": x_t, "nstar_obs_thin_interp": xi_t, "nstar_obs_thin_stays": st_t,
                          "n_max": ns[-1], "split_null_sd": sn_obs, "split_null_sd_theory": sn_th})
            print(f"  {g:>34} ← {s:<16}{' (main)' if s == c['main_source'] else '       '}: n* obs {x_b:>10}"
                  f"  simple {_nstar_txt(nstar[-1]['nstar_simple']):>6}  metric {_nstar_txt(nstar[-1]['nstar_metric']):>6}")
        # 確認用: 全枚数の none / borrow（院内 T の全部に当てる）と hit1 の補正なし
        c["check"] = {("none", ""): full["group_gap_sd"]}
        for s, (_, _, delta_B) in c["borrows"].items():
            c["check"][("borrow", s)] = _mix(E_T, P_T + delta_B, g, ref, oth, cfg)["group_gap_sd"]
        if ref_info is not None:
            h = hit1(P_T, g, ref_info)
            c["check_hit"] = None if h is None else dict(zip(c["ids_P"], h))
    checks.update(verify(cfg, enc, ctx, ref_info, X))
    return pd.DataFrame(rows), nstar, checks


def verify(cfg: dict, enc: str, ctx: dict, ref_info: dict | None, X: np.ndarray) -> dict:
    out_dir = Path(cfg["output_dir"])
    gc = pd.read_csv(out_dir / "transfer" / f"correction_{enc}_groups.csv", float_precision="round_trip")
    gc = gc[(gc["method"] == "centroid") & (gc["metric"] == "group_gap_sd")].set_index(["target", "source", "group"])["value"]
    rt_csv = out_dir / "transfer" / f"rare_target_{enc}.csv"
    rt = pd.read_csv(rt_csv, float_precision="round_trip") if rt_csv.exists() else None
    diffs, done = [], []
    for g, c in ctx.items():
        t = CHECK_TARGET.get(g, "paper")
        for (k, s), got in c["check"].items():
            src = "none" if k == "none" else s
            v = gc.get((t, src, g))
            if v is None and (g, src) == ("SFT", "meningioma") and rt is not None:   # paper ← meningioma は transfer_correction に無い
                r = rt[(rt["method"] == "centroid") & (rt["source"] == "meningioma") & (rt["group"] == g) & (rt["metric"] == "group_gap_sd")]
                v = float(r["value"].iloc[0]) if len(r) else None
            if v is None:
                continue
            diffs.append(abs(float(v) - got))
            done.append(f"{g} ← {src}")
    d = max(diffs) if diffs else np.nan
    if d > 1e-12:
        raise SystemExit(f"[{enc}] 全枚数の group_gap_sd が transfer_correction の出力と合わない（max |差| {d:.2e}）")
    res = {"gap_n": len(diffs), "gap_maxdiff": d}
    if ref_info is not None:
        ps = pd.read_csv(ref_info["dir"] / f"per_slide_{enc}.csv", dtype={"slide_id": str})
        ps = ps[(ps["retrieval"] == "nc_euclid") & (ps["condition"] == "none")].set_index("slide_id")["hit1"]
        n_ok = n_all = 0
        for c in ctx.values():
            for sid, h in (c.get("check_hit") or {}).items():
                n_all += 1
                n_ok += int(bool(ps.loc[sid]) == bool(h))
        if n_ok != n_all:
            raise SystemExit(f"[{enc}] hit1（補正なし）が atlas の per_slide_{enc}.csv と {n_all - n_ok} 枚で合わない")
        mq = np.load(ref_info["dir"] / f"manifest_{enc}.npy").astype(np.float64)
        res.update({"hit_n": n_all, "hit_ok": n_ok, "query_maxdiff": float(np.abs(mq - X).max())})
    print(f"  verify: group_gap_sd {len(diffs)} values max |diff| {d:.1e}" +
          (f"; hit1 {res['hit_ok']} / {res['hit_n']} slides; atlas query embeddings max |diff| {res['query_maxdiff']:.1e}"
           if "hit_n" in res else "; hit1: no atlas cache"))
    return res


# ── output ────────────────────────────────────────────────────────────────────

def _ci(r: pd.Series | None, digits: int = 2) -> str:
    if r is None or pd.isna(r["median"]):
        return ""
    if pd.isna(r["q025"]):
        return _fmt(r["median"], digits)
    return f"{r['median']:.{digits}f} [{r['q025']:.{digits}f}, {r['q975']:.{digits}f}]"


def _nstar_txt(v: float) -> str:
    return "∞" if not np.isfinite(v) else f"{v:.1f}"


def write_summary(path: Path, res: pd.DataFrame, ns: pd.DataFrame, checks: dict, cfg: dict, n_draws: int) -> None:
    names = cfg.get("shift_transfer", {}).get("group_names", {})
    encs = [e for e in ENCODERS if e in set(res["encoder"])]
    S = res.fillna({"source": ""}).set_index(["encoder", "group", "n", "condition", "source", "metric"]).sort_index()

    def get(enc, g, n, cond, src="", metric="group_gap_sd"):
        k = (enc, g, n, cond, src, metric)
        return S.loc[k] if k in S.index else None

    def gname(g):
        return names.get(g, g)

    L = ["# 例数の少ない腫瘍: 自前の n 枚で推定するか、よくある腫瘍（髄膜腫・グリオーマ）から借りるか（平行移動）", "",
         "`transfer_rare_crossover.py` が作る。定義はスクリプトの docstring。EBRAINS は全部使い、院内の群 T の N 枚から n 枚を無作為に"
         f"選んで推定し、残り N − n 枚（推定に使っていない）で評価する（各 n で {n_draws} 回、seed 42）。",
         "主指標 group_gap_sd = ‖μ_E,T − (残りの平均 + Δ)‖ / 施設内のばらつき（transfer_correction.site_mixing、"
         "correction_{encoder}_groups.csv と同じ定義）。値は中央値 [2.5, 97.5 パーセンタイル]。", "",
         "- 借りる = 借りる元（髄膜腫 = SFT を含まない科 meningioma、グリオーマ = ast + glm + odg）の両施設の全枚数の平均の差。"
         "T が借りる元の科に入るときは T を除いた残り（参考）。借りる 20 = 院内の借りる元を 20 枚に間引いた版（参考）",
         "- 床 = 補正が完全（真の Δ）でも、EBRAINS の T と残りの平均の標本誤差だけで残る値の期待値（√tr V / RMS）。全条件に共通",
         "- 自前の値には EBRAINS 側の標本誤差が入らない（EBRAINS の T の平均を推定と評価で同じものを使うので打ち消す）。"
         "借りる・補正なしには入る",
         "- n* 実測 = 自前の中央値が借りる（同じ n・同じ残り）の中央値を下回る最小の n（格子 "
         f"{', '.join(map(str, N_LIST))} のうち残りが {MIN_REST} 枚以上）。括弧は log n で補間した値。"
         "n* 理論 = trS_P / b²（依頼の近似）、n* 指標 = trS_P / (b² + trS_E/n_E + tr V_B)（この指標での予想）。"
         "b² = ‖Δ_T − Δ_借りる元‖² からノイズを引いたもの", ""]

    def overview(sel: pd.DataFrame, title: str) -> list[str]:
        out = [f"## {title}", ""]
        tb = []
        for _, key in sel[["group", "source"]].drop_duplicates().iterrows():
            g, s = key["group"], key["source"]
            for enc in encs:
                r = ns[(ns["encoder"] == enc) & (ns["group"] == g) & (ns["source"] == s)]
                if r.empty:
                    continue
                r = r.iloc[0]
                n0 = min(n for n in N_LIST if (enc, g, n, "none", "", "group_gap_sd") in S.index)
                row = {"群 ← 借りる元": f"{gname(g)} ← {SOURCE_LABEL[s]}", "エンコーダ": ENCODER_LABELS.get(enc, enc),
                       "E/P": f"{r['n_ebrains']}/{r['n_patho2']}",
                       f"補正なし（n={n0}）": _ci(get(enc, g, n0, "none")), f"借りる（n={n0}）": _ci(get(enc, g, n0, "borrow", s)),
                       f"借りる 20（n={n0}）": _ci(get(enc, g, n0, "borrow_thin", s))}
                for n in SHOW_N:
                    row[f"自前 n={n}"] = _ci(get(enc, g, n, "own"))
                row[f"床（n={n0}）"] = _ci(get(enc, g, n0, "floor", "", "floor_sd"))
                interp = "" if np.isnan(r["nstar_obs_interp"]) else f"（{r['nstar_obs_interp']:.1f}）"
                row["n* 実測"] = f"{r['nstar_obs']}{interp}" + ("" if r["nstar_obs_stays"] or r["nstar_obs"] == "範囲内に交点なし" else " ※")
                row["n* 理論"] = _nstar_txt(r["nstar_simple"])
                row["n* 指標"] = _nstar_txt(r["nstar_metric"])
                tb.append(row)
        out += [_md(pd.DataFrame(tb), index=False), "",
                "※ = 交点より大きい n で自前が借りるを上回り直す n がある。空欄 = その n では残りが 5 枚未満。", ""]
        return out

    main = ns[(ns["kind"] == "main") & ns["main"]]
    order = {g: i for i, g in enumerate(MAIN_GROUPS)}
    main = main.sort_values("group", key=lambda x: x.map(order), kind="stable")
    other = ns[(ns["kind"] == "main") & ~ns["main"]].sort_values("group", key=lambda x: x.map(order), kind="stable")
    L += overview(main, "本文の 6 組（SFT ← 髄膜腫、lym / met ← グリオーマ、ast / glm / odg ← 髄膜腫）")
    L += overview(other, "他の組み合わせ（SFT ← グリオーマ、lym / met ← 髄膜腫、ast / glm / odg ← 残りのグリオーマ 2 群）")
    L += overview(ns[ns["kind"] == "meningioma_subtype"], "参考: 髄膜腫のサブタイプ（← 残りの髄膜腫 / ← グリオーマ）")

    L += ["## 理論の成分", "",
          "trS_P / trS_E = 院内 / EBRAINS の T の 1 枚あたりの全分散。b² = ‖Δ_T − Δ_借りる元‖²（観測 diff²）からノイズ tr V_T + tr V_B を"
          "引いたもの。cos = cos(Δ_T, Δ_借りる元)（括弧はノイズを差し引いた長さで割った値）。split null = 院内の T を半分ずつに割って同じ "
          "group_gap_sd を出した中央値（施設差なし）/ 同じ割り方の理論の床 √tr V / RMS。n* 指標 20 = 借りる 20 枚版の理論の交点、"
          "n* 実測 20 = その実測。本文の 6 組 → 他の組み合わせ → 髄膜腫のサブタイプの順。", ""]
    tb = []
    for part in (main, other, ns[ns["kind"] == "meningioma_subtype"]):
        for _, key in part[["group", "source"]].drop_duplicates().iterrows():
            for enc in encs:
                r = ns[(ns["encoder"] == enc) & (ns["group"] == key["group"]) & (ns["source"] == key["source"])]
                if r.empty:
                    continue
                r = r.iloc[0]
                tb.append({"群 ← 借りる元": f"{gname(r['group'])} ← {SOURCE_LABEL[r['source']]}", "エンコーダ": ENCODER_LABELS.get(enc, enc),
                           "RMS": _fmt(r["rms"], 2), "trS_P": _fmt(r["trS_P"], 1), "trS_E/n_E": _fmt(r["trS_E_over_nE"], 2),
                           "‖Δ_T‖": _fmt(r["shift_norm_T"], 2), "‖Δ_借りる元‖": _fmt(r["shift_norm_B"], 2),
                           "cos": f"{r['cos_T_B']:.3f}（{r['cos_T_B_corrected']:.3f}）", "diff²": _fmt(r["diff_norm2_obs"], 2),
                           "b²": _fmt(r["b2"], 2), "tr V_B": _fmt(r["noise_trace_B"], 3),
                           "n* 理論": _nstar_txt(r["nstar_simple"]), "n* 指標": _nstar_txt(r["nstar_metric"]),
                           "n* 指標 20": _nstar_txt(r["nstar_metric_thin"]),
                           "n* 実測 20": r["nstar_obs_thin"] + ("" if np.isnan(r["nstar_obs_thin_interp"]) else f"（{r['nstar_obs_thin_interp']:.1f}）"),
                           "split null": f"{r['split_null_sd']:.3f} / {r['split_null_sd_theory']:.3f}"})
    L += [_md(pd.DataFrame(tb), index=False), ""]

    L += ["## 本文の 6 組の n ごとの値", "",
          "列: 残り = 評価に使う院内の枚数、借りる（他）= もう一方の借りる元、P(自前 < 借りる) = 同じ抽出で自前の方が小さい割合、"
          "理論 = 上の近似式（自前 / 借りる / 床）、hit1 = 残りのスライドが EBRAINS 全体の診断の重心で T の診断に当たる割合"
          "（titan / prism のみ）。", ""]
    for g in MAIN_GROUPS:
        s = MAIN_SOURCE[g]
        for enc in encs:
            sub = ns[(ns["encoder"] == enc) & (ns["group"] == g) & (ns["source"] == s)]
            if sub.empty:
                continue
            so = next(x for x in ns[(ns["encoder"] == enc) & (ns["group"] == g)]["source"] if x != s)
            tb = []
            for n in [n for n in N_LIST if (enc, g, n, "own", "", "group_gap_sd") in S.index]:
                row = {"n": n, "残り": int(get(enc, g, n, "own")["n_rest"]),
                       "補正なし": _ci(get(enc, g, n, "none")), "借りる": _ci(get(enc, g, n, "borrow", s)),
                       "借りる 20": _ci(get(enc, g, n, "borrow_thin", s)), f"借りる（{SOURCE_LABEL[so]}）": _ci(get(enc, g, n, "borrow", so)),
                       "自前": _ci(get(enc, g, n, "own")), "床": _ci(get(enc, g, n, "floor", "", "floor_sd")),
                       "P(自前 < 借りる)": _fmt(get(enc, g, n, "own_vs_borrow", s, "p_own_better")["median"], 3),
                       "理論 自前 / 借りる / 床": " / ".join(f"{get(enc, g, n, k, src)['median']:.2f}" for k, src in
                                                    (("theory_own", ""), ("theory_borrow", s), ("theory_floor", "")))}
                h = [get(enc, g, n, k, src, "hit1") for k, src in (("none", ""), ("borrow", s), ("own", ""))]
                if all(x is not None and not pd.isna(x["median"]) for x in h):
                    row["hit1 なし / 借りる / 自前"] = " / ".join(f"{x['median']:.2f}" for x in h)
                tb.append(row)
            L += [f"### {gname(g)} ← {SOURCE_LABEL[s]}（{ENCODER_LABELS.get(enc, enc)}、EBRAINS {sub.iloc[0]['n_ebrains']} / "
                  f"院内 {sub.iloc[0]['n_patho2']}）", "", _md(pd.DataFrame(tb), index=False), ""]

    L += ["## 確認", "",
          "- 全枚数の院内 T に対する group_gap_sd（補正なし / 借りる）を transfer_correction.py の correction_{encoder}_groups.csv"
          "（当て先 paper / glioma / lymphoma / metastasis の none / meningioma / glioma、centroid）と、SFT ← 髄膜腫を rare_target_{encoder}.csv と"
          "突き合わせた（*_rest は対応する既存の行が無い）。",
          "- hit1 の補正なし（院内 T の全枚数）を atlas_retrieval.py の per_slide_{encoder}.csv（nc_euclid, none）とスライドごとに、"
          "atlas の問い合わせの埋め込みキャッシュを転移テストのキャッシュと突き合わせた。", ""]
    tb = []
    for enc in encs:
        c = checks[enc]
        tb.append({"エンコーダ": ENCODER_LABELS.get(enc, enc), "group_gap_sd 突き合わせ": f"{c['gap_n']} 値、最大の差 {c['gap_maxdiff']:.1e}",
                   "hit1 の参照": "（atlas のキャッシュ無し）" if c["atlas"] is None else f"{c['atlas']['n_classes']} 診断 / {c['atlas']['n_ref']} 枚",
                   "hit1 突き合わせ": f"{c['hit_ok']} / {c['hit_n']} 枚" if "hit_n" in c else "",
                   "問い合わせの埋め込み 最大の差": f"{c['query_maxdiff']:.1e}" if "query_maxdiff" in c else ""})
    L += [_md(pd.DataFrame(tb), index=False), ""]
    path.write_text("\n".join(L))
    print(f"  saved: {path}")


def plot(res: pd.DataFrame, ns: pd.DataFrame, enc: str, path: Path, cfg: dict) -> None:
    names = cfg.get("shift_transfer", {}).get("group_names", {})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, g in zip(axes.flat, MAIN_GROUPS):
        s_main = MAIN_SOURCE[g]
        d = res[(res["encoder"] == enc) & (res["group"] == g)].fillna({"source": ""})
        r = ns[(ns["encoder"] == enc) & (ns["group"] == g) & (ns["source"] == s_main)].iloc[0]
        s_oth = next(x for x in ns[(ns["encoder"] == enc) & (ns["group"] == g)]["source"] if x != s_main)
        style = [("none", "", "#a3a29b", "-", "none"), ("own", "", "#eb6834", "-", "own n slides"),
                 ("borrow", s_main, "#2a78d6", "-", f"borrow: {SOURCE_LABEL_EN[s_main]}"),
                 ("borrow_thin", s_main, "#2a78d6", ":", f"borrow: {SOURCE_LABEL_EN[s_main]} ({THIN_N} in-house)"),
                 ("borrow", s_oth, "#1baf7a", "-", f"borrow: {SOURCE_LABEL_EN[s_oth]}")]
        for cond, src, col, ls, lab in style:
            v = d[(d["condition"] == cond) & (d["source"] == src) & (d["metric"] == "group_gap_sd")].sort_values("n")
            ax.plot(v["n"], v["median"], color=col, ls=ls, marker="o", ms=3, lw=1.5 if ls == "-" else 1.2, label=lab)
            if cond == "own" or (cond == "borrow" and src == s_main):
                ax.fill_between(v["n"], v["q025"], v["q975"], color=col, alpha=0.15, lw=0)
        for cond, src, col in (("theory_own", "", "#eb6834"), ("theory_borrow", s_main, "#2a78d6")):
            v = d[(d["condition"] == cond) & (d["source"] == src)].sort_values("n")
            ax.plot(v["n"], v["median"], color=col, ls="--", lw=0.9, alpha=0.8)
        v = d[(d["condition"] == "floor")].sort_values("n")
        ax.plot(v["n"], v["median"], color="#222222", ls="-.", lw=1.0, label="floor (perfect Δ)")
        if np.isfinite(r["nstar_metric"]):
            ax.axvline(r["nstar_metric"], color="#555555", ls="--", lw=0.8)
        if not np.isnan(r["nstar_obs_interp"]):
            ax.axvline(r["nstar_obs_interp"], color="#eb6834", ls=":", lw=1.2)
        ax.set_xscale("log")
        xs = sorted(d["n"].unique())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(x)) for x in xs], fontsize=8)
        top = float(d[d["metric"].isin(["group_gap_sd", "floor_sd"])]["q975"].max())
        ax.set_ylim(0, top * 1.55)   # 上に凡例の余白
        ax.set_title(f"{names.get(g, g)} ← {SOURCE_LABEL_EN[s_main]}  (EBRAINS {r['n_ebrains']} / in-house {r['n_patho2']})\n"
                     f"n* observed {r['nstar_obs'].replace('範囲内に交点なし', 'none in range')}, theory {_nstar_txt(r['nstar_metric'])}",
                     fontsize=9, loc="left")
        ax.set_xlabel("n (in-house slides of T used for the own estimate)", fontsize=8)
        ax.set_ylabel("group_gap_sd on the held-out rest", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    fig.suptitle(f"Own n slides vs borrowed Δ (translation, {enc}). Dashed = theory; vertical dashed = theoretical n*, "
                 "dotted = observed n* (interpolated)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", choices=ENCODERS, default=ENCODERS)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-jobs", type=int, default=N_JOBS)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "transfer_sample_size"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_res, all_ns, checks = [], [], {}
    for enc in args.encoders:
        res, nstar, checks[enc] = run_encoder(cfg, enc, args.n_draws, args.n_jobs)
        res.to_csv(out_dir / f"rare_crossover_{enc}.csv", index=False)
        print(f"  saved: {out_dir / f'rare_crossover_{enc}.csv'}")
        all_res.append(res)
        all_ns += nstar
    res, ns = pd.concat(all_res, ignore_index=True), pd.DataFrame(all_ns)
    ns.to_csv(out_dir / "rare_crossover_nstar.csv", index=False)
    print(f"  saved: {out_dir / 'rare_crossover_nstar.csv'}")
    write_summary(out_dir / "rare_crossover_summary.md", res, ns, checks, cfg, args.n_draws)
    for enc in args.encoders:
        plot(res, ns, enc, out_dir / f"rare_crossover_{enc}.png", cfg)


if __name__ == "__main__":
    main()
