"""shift_time.py — 院内（patho2）の施設差 Δ は年によってぶれないか（TITAN）。

問い: 平行移動の Δ（EBRAINS 平均 − 院内平均）は、院内のスライドを年ごとに分けて求めても同じ向き・長さか。
    年でぶれる（スキャナや染色の経年変化）なら、1 本の Δ で院内全体を補正する前提が弱くなる。

年: 院内のスライド ID の先頭 2 桁（任意の N / n の後）= 受付年（17-2197 → 2017、N20-097 → 2020、Patho2Gliomas の
    22-2450・N21-134 も同じ）。全院内 ID が `[Nn]?YY-数字` に合うことを確かめ、科ごとの年のヒストグラムを出す。
    照合: (1) 論文コホートの WSI のパス（脳腫瘍レビュー/2017/・/N17/）の年と一致するか、(2) WSI のファイル名の撮像時刻
    （"HE - 2018-05-21 13.16.12"）の年、(3) NDPI の TIFF タグ DateTime（= 撮像日。WSI を読むだけ、NFS には書かない）の年
    との対応表。(3) は out/shift_time/scan_dates.csv にキャッシュする（--rescan で読み直し）。
    主解析は受付年（year_by = id）、撮像年（year_by = scan、DateTime の年。無ければファイル名の時刻）でも同じ表を作る。

科: meningioma（論文コホートの SFT 以外）・glioma（ast / glm / odg）・metastasis（--families で変える）。

Δ の推定量（どちらも院内スライドごとのベクトル v_i の平均 Δ_S = mean_{i∈S} v_i として書ける）:
    raw      v_i = μ_E,科 − x_i（EBRAINS の科の平均 − 院内スライド）。年 y なら Δ_y = μ_E,科 − mean(院内 科 年 y)
    matched  v_i = μ_E,群(i) − x_i（組織型をそろえる: 院内スライドの組織型 = 群の EBRAINS 平均から引く。EBRAINS に
             --min-group-ref 例未満の群（NOS meningioma など）は科の平均にフォールバック）。年ごとの組織型の構成の違いを除く
    Δ_all = 全年の院内の平均（その科の院内全部）。

年ごとの指標（院内がその年に --min-n 枚以上の年）:
    delta_norm   ‖Δ_y‖
    cos_all      cos(Δ_y, Δ_all)（Δ_all は Δ_y のスライドを含む。帰無も同じ作りなので比べられる）
    cos_rest     cos(Δ_y, Δ_rest)（Δ_rest = その年以外の院内の平均で作った Δ）
    cos_paper    cos(Δ_y, Δ_paper)（Δ_paper = 論文コホート（髄膜腫 + SFT）の EBRAINS 平均 − 院内平均）
    proj         Δ_y の Δ_all 方向の成分 Δ_y·Δ_all/‖Δ_all‖（Δ の長さの経年変化）
    orth         Δ_y の Δ_all に直交する成分の長さ
    drift        ‖Δ_y − Δ_all‖ / ‖Δ_all‖（raw ではその年の院内重心と全年の院内重心の距離を施設差で割ったもの）
年の組（a < b）の指標:
    cos          cos(Δ_a, Δ_b)
    dist         ‖Δ_a − Δ_b‖（raw では院内の年 a と年 b の重心どうしの距離）
    dist_ratio   dist / ‖Δ_all‖
    dist_corr    √max(dist² − E_null[dist²], 0)（抽出のノイズを引いた距離）。dist_corr_ratio = dist_corr / ‖Δ_all‖
帰無（--n-null 回、seed 42）:
    null    その科の院内の全年から同じ枚数を無作為に抜いた部分集合（組は互いに素な 2 つの部分集合）
    strat   同じ枚数・同じ組織型の構成（年 y の群ごとの枚数を群の中から無作為に抜く）。年で組織型の構成が違う
            （例: グリオーマの 2019 年は glm 14 / ast 1）ことによる差を帰無に含める
    各帰無の中央値と 2.5–97.5%、片側の p（cos は観測以下の割合、長さ・距離は観測以上の割合、(1 + 数) / (1 + 回数)）。
年の不均一性の並べ替え検定: Q = Σ_y n_y ‖Δ_y − Δ_pool‖²（対象年のスライドだけ）を、年のラベルを並べ替えた分布と比べる
    （null = 全体で並べ替え、strat = 組織型の中で並べ替え）。effect = √(max(Q − E_null[Q], 0) / N) / ‖Δ_all‖。
傾向: 院内スライドごとの Δ_all 方向の成分 v_i·Δ_all/‖Δ_all‖ と年の Spearman 相関（全年のスライド）。

出力 (output_dir/shift_time/):
    shift_by_year.csv   family, estimator, year_by, year, n, metric, obs, null_median, null_lo, null_hi, null_p,
                        strat_median, strat_lo, strat_hi, strat_p
    year_pairs.csv      family, estimator, year_by, year_a, year_b, n_a, n_b, metric, obs, null_* , strat_*
    heterogeneity.csv   family, estimator, year_by, n_years, n, Q, null_p, strat_p, effect_null, effect_strat, spearman_rho, spearman_p
    scan_dates.csv      院内スライドの撮像日（TIFF DateTime とファイル名の時刻）のキャッシュ
    summary.md          年の読み取りの確認・年ごとの表・年の組の要約・不均一性
    shift_time.png      列 = 科、行 = cos(Δ_y, Δ_all) / ‖Δ_y‖（帰無の幅つき）/ 年の組の cos / 院内の年どうしの距離（ノイズを引いた）

Usage:
    uv run python shift_time.py [--min-n 15] [--n-null 2000] [--families meningioma glioma metastasis] [--rescan]
    uv run python shift_time.py --plot-only      # 出力済みの CSV から図だけ
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from transfer_umap import load_titan  # noqa: E402
from utils.loader import load_config  # noqa: E402

REF = "ebrains"
PAPER = {"meningioma", "SFT"}
ID_RE = re.compile(r"[Nn]?(\d{2})-\d+")
TS_RE = re.compile(r"(20\d{2})-\d{2}-\d{2} \d{2}\.\d{2}\.\d{2}")      # ファイル名の撮像時刻
DIR_RE = re.compile(r"/(?:20|N)(\d{2})/")                              # 論文コホートの WSI の年のフォルダ
FAMILY_LABEL = {"meningioma": "Meningioma", "glioma": "Glioma", "metastasis": "Metastasis",
                "lymphoma": "Lymphoma (DLBCL)", "SFT": "SFT"}
YEAR_METRICS = ["delta_norm", "cos_all", "cos_rest", "cos_paper", "proj", "orth", "drift"]
PAIR_METRICS = ["cos", "dist", "dist_ratio"]
LOWER_IS_EXTREME = {"cos_all", "cos_rest", "cos_paper", "cos", "proj"}   # 片側 p の向き（観測以下の割合）


# ── years ─────────────────────────────────────────────────────────────────────

def id_year(slide_id: str) -> int:
    m = ID_RE.fullmatch(slide_id)
    if m is None:
        raise ValueError(f"院内 ID から年が読めない: {slide_id!r}")
    return 2000 + int(m.group(1))


def _tiff_datetime(path: str) -> str:
    import tifffile
    try:
        with tifffile.TiffFile(path) as tf:
            tag = tf.pages[0].tags.get("DateTime")
            return str(tag.value) if tag is not None else ""
    except Exception as e:  # noqa: BLE001 — 読めないスライドは空欄にして続ける
        print(f"  WARNING: {path}: {e}")
        return ""


def scan_dates(p2: pd.DataFrame, cache: Path, rescan: bool) -> pd.DataFrame:
    """院内スライドの撮像日（NDPI の TIFF DateTime と、ファイル名の時刻）。WSI は読むだけ。"""
    if cache.exists() and not rescan:
        c = pd.read_csv(cache, dtype={"slide_id": str, "tiff_datetime": str}, keep_default_na=False)
        if set(c["slide_id"]) >= set(p2["slide_id"]):
            print(f"  cache: {cache}")
            return c
    print(f"  reading TIFF DateTime of {len(p2)} WSI ...")
    with ThreadPoolExecutor(8) as ex:
        dts = list(ex.map(_tiff_datetime, p2["wsi"]))
    c = pd.DataFrame({"slide_id": p2["slide_id"].values, "wsi": p2["wsi"].values, "tiff_datetime": dts,
                      "filename_ts": [m.group(0) if (m := TS_RE.search(w)) else "" for w in p2["wsi"]]})
    c.to_csv(cache, index=False)
    print(f"  saved: {cache}")
    return c


def attach_years(man: pd.DataFrame, cache: Path, rescan: bool) -> tuple[pd.DataFrame, list[str]]:
    """院内に year_id（受付年）・year_scan（撮像年）を付ける。照合の結果を summary 用の行で返す。"""
    p2 = man[man["site"] != REF]
    bad = [s for s in p2["slide_id"] if ID_RE.fullmatch(s) is None]
    if bad:
        raise SystemExit(f"年の読めない院内 ID が {len(bad)} 件: {bad[:20]}")
    man = man.copy()
    man["year_id"] = np.nan
    man.loc[p2.index, "year_id"] = [id_year(s) for s in p2["slide_id"]]
    sd = scan_dates(p2, cache.parent / "scan_dates.csv", rescan).set_index("slide_id").reindex(p2["slide_id"])
    dt_year = pd.to_numeric(sd["tiff_datetime"].str[:4], errors="coerce")
    ts_year = pd.to_numeric(sd["filename_ts"].str[:4], errors="coerce")
    man["year_scan"] = np.nan
    man.loc[p2.index, "year_scan"] = dt_year.fillna(ts_year).to_numpy()
    man["year_scan_src"] = ""
    man.loc[p2.index, "year_scan_src"] = np.where(dt_year.notna(), "tiff", np.where(ts_year.notna(), "filename", "")).tolist()

    lines = []
    p2 = man.loc[p2.index]
    lines += ["## 年の読み取り（院内 837 枚）", "",
              f"- ID `[Nn]?YY-数字` に全 {len(p2)} 件が合う（N / n 付き {int(p2['slide_id'].str.match('[Nn]').sum())} 件）。",
              "- 受付年（ID）の科ごとのヒストグラム:", "", "```",
              pd.crosstab(p2["family"], p2["year_id"].astype(int)).to_string(), "```", ""]
    dir_year = p2["wsi"].str.extract(DIR_RE)[0]
    has = dir_year.notna()
    agree = int((2000 + dir_year[has].astype(int) == p2.loc[has, "year_id"]).sum())
    lines.append(f"- 照合 1（WSI のパスの年のフォルダ /YYYY/・/NYY/。論文コホートの脳腫瘍レビュー/ など）: {int(has.sum())} 件中 {agree} 件一致。")
    def lag(yr: pd.Series, has: np.ndarray) -> str:
        """撮像年 − 受付年 の件数（+0: 何件, +1: 何件 …）。"""
        vc = pd.Series(yr[has].to_numpy() - p2["year_id"].to_numpy()[has]).astype(int).value_counts().sort_index()
        return ", ".join(f"{int(k):+d}: {v}" for k, v in vc.items())

    t = ts_year.notna().to_numpy()
    lines.append(f"- 照合 2（ファイル名の撮像時刻）: {int(t.sum())} 件にあり、撮像年 − 受付年 = {lag(ts_year, t)}。")
    d = dt_year.notna().to_numpy()
    lines.append(f"- 照合 3（NDPI の TIFF DateTime = 撮像日）: {int(d.sum())} / {len(p2)} 件で読めた。"
                 f"撮像年 − 受付年 = {lag(dt_year, d)}。")
    both = d & t
    if both.any():
        same = int((dt_year[both].to_numpy() == ts_year[both].to_numpy()).sum())
        lines.append(f"  ファイル名の時刻もある {int(both.sum())} 件のうち DateTime と年が一致 {same} 件。")
    lines += ["- 撮像年（year_scan）の科ごとのヒストグラム:", "", "```",
              pd.crosstab(p2["family"], p2["year_scan"].fillna(0).astype(int).replace(0, -1)).to_string(), "```",
              "（-1 = 撮像日が読めない）", ""]
    return man, lines


# ── null draws ────────────────────────────────────────────────────────────────

def draw(rng: np.random.Generator, group_idx: list[np.ndarray], counts: list[np.ndarray], n_draw: int) -> list[np.ndarray]:
    """群ごとに counts[j][g] 枚ずつ、互いに素な部分集合 j を n_draw 回抜く。返り値は j ごとの (n_draw, n_j) の添字。"""
    parts: list[list[np.ndarray]] = [[] for _ in counts]
    for g, gi in enumerate(group_idx):
        need = int(sum(c[g] for c in counts))
        if need == 0:
            continue
        sel = gi[np.argsort(rng.random((n_draw, len(gi))), axis=1)[:, :need]]
        o = 0
        for j, c in enumerate(counts):
            parts[j].append(sel[:, o:o + int(c[g])])
            o += int(c[g])
    return [np.concatenate(p, axis=1) for p in parts]


def subset_means(V: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """添字 (n_draw, n) の各行の部分集合の平均（重み行列で計算してメモリを抑える）。"""
    W = np.zeros((idx.shape[0], V.shape[0]))
    np.put_along_axis(W, idx, 1.0 / idx.shape[1], axis=1)
    return W @ V


def _cos_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = np.atleast_2d(A)
    b2 = np.atleast_2d(b)
    return np.sum(A * b2, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b2, axis=1))


def year_stats(D: np.ndarray, n: int, N: int, d_all: np.ndarray, d_paper: np.ndarray) -> dict[str, np.ndarray]:
    """部分集合の Δ（行）から年の指標。Δ_rest は (N·Δ_all − n·Δ_S)/(N − n)。"""
    D = np.atleast_2d(D)
    u = d_all / np.linalg.norm(d_all)
    proj = D @ u
    rest = (N * d_all - n * D) / (N - n)
    return {"delta_norm": np.linalg.norm(D, axis=1), "cos_all": _cos_rows(D, d_all),
            "cos_rest": np.sum(D * rest, 1) / (np.linalg.norm(D, axis=1) * np.linalg.norm(rest, axis=1)),
            "cos_paper": _cos_rows(D, d_paper), "proj": proj,
            "orth": np.linalg.norm(D - proj[:, None] * u, axis=1),
            "drift": np.linalg.norm(D - d_all, axis=1) / np.linalg.norm(d_all)}


def pair_stats(Da: np.ndarray, Db: np.ndarray, d_all: np.ndarray) -> dict[str, np.ndarray]:
    Da, Db = np.atleast_2d(Da), np.atleast_2d(Db)
    dist = np.linalg.norm(Da - Db, axis=1)
    return {"cos": np.sum(Da * Db, 1) / (np.linalg.norm(Da, axis=1) * np.linalg.norm(Db, axis=1)),
            "dist": dist, "dist_ratio": dist / np.linalg.norm(d_all)}


def _summ(obs: float, null: np.ndarray, metric: str, prefix: str) -> dict:
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5])
    k = (null <= obs).sum() if metric in LOWER_IS_EXTREME else (null >= obs).sum()
    return {f"{prefix}_median": med, f"{prefix}_lo": lo, f"{prefix}_hi": hi, f"{prefix}_p": (1 + k) / (1 + len(null))}


# ── analysis ──────────────────────────────────────────────────────────────────

def analyse(fam: str, man: pd.DataFrame, X: np.ndarray, d_paper: np.ndarray, year_by: str, args,
            rng: np.random.Generator) -> tuple[list[dict], list[dict], list[dict]]:
    site = man["site"].to_numpy()
    in_fam = (man["family"] == fam).to_numpy()
    e_idx = np.where(in_fam & (site == REF))[0]
    ycol = f"year_{year_by}"
    p_idx = np.where(in_fam & (site != REF) & man[ycol].notna().to_numpy())[0]
    years_all = man[ycol].to_numpy()[p_idx].astype(int)
    groups = man["group"].astype(str).to_numpy()
    mu_e = X[e_idx].mean(0)
    g_mean = {g: X[e_idx[groups[e_idx] == g]].mean(0) for g in np.unique(groups[e_idx])
              if (groups[e_idx] == g).sum() >= args.min_group_ref}
    xp = X[p_idx]
    gp = groups[p_idx]
    V = {"raw": mu_e[None, :] - xp}
    if len(set(gp)) > 1:                 # 群が 1 つの科（metastasis）では matched = raw なので作らない
        V["matched"] = np.stack([g_mean.get(g, mu_e) for g in gp]) - xp
    N = len(p_idx)
    glist = sorted(set(gp))
    gidx_strat = [np.where(gp == g)[0] for g in glist]
    gidx_all = [np.arange(N)]
    yrs = [y for y in sorted(set(years_all)) if (years_all == y).sum() >= args.min_n]
    fallback = sorted(set(gp) - set(g_mean))
    print(f"  [{fam} / {year_by}] EBRAINS {len(e_idx)}, 院内 {N}, 年 ≥ {args.min_n}: "
          + ", ".join(f"{y} ({(years_all == y).sum()})" for y in yrs)
          + (f"; matched で科の平均にフォールバックする群: {fallback}" if fallback else ""))

    def comp(mask: np.ndarray) -> np.ndarray:
        return np.array([(gp[mask] == g).sum() for g in glist])

    rows_y, rows_p, rows_h = [], [], []
    d_all = {k: v.mean(0) for k, v in V.items()}
    # 年ごと（帰無の抜き取りは 2 つの推定量で共有する）
    for y in yrs:
        m = years_all == y
        n = int(m.sum())
        idx_null = draw(rng, gidx_all, [np.array([n])], args.n_null)[0]
        idx_strat = draw(rng, gidx_strat, [comp(m)], args.n_null)[0]
        for est, Vk in V.items():
            obs = year_stats(Vk[m].mean(0), n, N, d_all[est], d_paper)
            nul = year_stats(subset_means(Vk, idx_null), n, N, d_all[est], d_paper)
            stv = year_stats(subset_means(Vk, idx_strat), n, N, d_all[est], d_paper)
            for met in YEAR_METRICS:
                o = float(obs[met][0])
                rows_y.append({"family": fam, "estimator": est, "year_by": year_by, "year": y, "n": n, "metric": met,
                               "obs": o, **_summ(o, nul[met], met, "null"), **_summ(o, stv[met], met, "strat")})
    # 年の組
    for a, b in combinations(yrs, 2):
        ma, mb = years_all == a, years_all == b
        na, nb = int(ma.sum()), int(mb.sum())
        ia, ib = draw(rng, gidx_all, [np.array([na]), np.array([nb])], args.n_null)
        sa, sb = draw(rng, gidx_strat, [comp(ma), comp(mb)], args.n_null)
        for est, Vk in V.items():
            obs = pair_stats(Vk[ma].mean(0), Vk[mb].mean(0), d_all[est])
            nul = pair_stats(subset_means(Vk, ia), subset_means(Vk, ib), d_all[est])
            stv = pair_stats(subset_means(Vk, sa), subset_means(Vk, sb), d_all[est])
            for met in PAIR_METRICS:
                o = float(obs[met][0])
                row = {"family": fam, "estimator": est, "year_by": year_by, "year_a": a, "year_b": b, "n_a": na, "n_b": nb,
                       "metric": met, "obs": o, **_summ(o, nul[met], met, "null"), **_summ(o, stv[met], met, "strat")}
                rows_p.append(row)
            # ノイズを引いた距離（帰無の dist² の平均を引く）
            for kind, src in (("null", nul), ("strat", stv)):
                dc = np.sqrt(max(float(obs["dist"][0]) ** 2 - float(np.mean(src["dist"] ** 2)), 0.0))
                rows_p.append({"family": fam, "estimator": est, "year_by": year_by, "year_a": a, "year_b": b, "n_a": na,
                               "n_b": nb, "metric": f"dist_corr_{kind}", "obs": dc,
                               **{f"{p}_{s}": np.nan for p in ("null", "strat") for s in ("median", "lo", "hi", "p")}})
                rows_p.append({**rows_p[-1], "metric": f"dist_corr_ratio_{kind}", "obs": dc / np.linalg.norm(d_all[est])})
    # 年の不均一性（並べ替え検定）と、Δ_all 方向の成分の経年傾向
    sel = np.isin(years_all, yrs)
    ys = years_all[sel]
    gsel = gp[sel]
    for est, Vk in V.items():
        Vs = Vk[sel]

        def q_stat(labels: np.ndarray) -> float:
            mu = Vs.mean(0)
            return float(sum((labels == y).sum() * np.sum((Vs[labels == y].mean(0) - mu) ** 2) for y in yrs))

        q = q_stat(ys)
        qn = np.array([q_stat(rng.permutation(ys)) for _ in range(args.n_perm)])
        qs = []
        for _ in range(args.n_perm):
            lab = ys.copy()
            for g in np.unique(gsel):
                w = np.where(gsel == g)[0]
                lab[w] = rng.permutation(lab[w])
            qs.append(q_stat(lab))
        qs = np.array(qs)
        u = d_all[est] / np.linalg.norm(d_all[est])
        rho, pr = spearmanr(Vk @ u, years_all)
        nrm = np.linalg.norm(d_all[est])
        rows_h.append({"family": fam, "estimator": est, "year_by": year_by, "n_years": len(yrs), "n": int(sel.sum()),
                       "Q": q, "null_p": (1 + (qn >= q).sum()) / (1 + len(qn)), "strat_p": (1 + (qs >= q).sum()) / (1 + len(qs)),
                       "effect_null": np.sqrt(max(q - qn.mean(), 0) / sel.sum()) / nrm,
                       "effect_strat": np.sqrt(max(q - qs.mean(), 0) / sel.sum()) / nrm,
                       "delta_all_norm": nrm, "spearman_rho": float(rho), "spearman_p": float(pr), "n_trend": N})
    return rows_y, rows_p, rows_h


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(dy: pd.DataFrame, dp: pd.DataFrame, dh: pd.DataFrame, families: list[str], path: Path) -> None:
    """図は受付年（year_by = id）・raw の Δ。matched（組織型をそろえた Δ）は ◇ で重ねる（群が 1 つの科は無し）。
    年の組の 2 つの行列は科をまたいで色の範囲を共通にする。"""
    est, yb = "raw", "id"
    sel_h = (dh["estimator"] == est) & (dh["year_by"] == yb)
    all_norm = {f: float(dh[sel_h & (dh["family"] == f)]["delta_all_norm"].iloc[0]) for f in families
                if (sel_h & (dh["family"] == f)).any()}
    pp = dp[(dp["estimator"] == est) & (dp["year_by"] == yb)]
    cos_lo = float(np.floor(pp.loc[pp["metric"] == "cos", "obs"].min() * 50) / 50)
    dist_hi = float(np.ceil(pp.loc[pp["metric"] == "dist_corr_ratio_null", "obs"].max() * 20) / 20)
    fig, axes = plt.subplots(4, len(families), figsize=(5.2 * len(families), 17.5), squeeze=False)
    handles = {}
    for j, fam in enumerate(families):
        d = dy[(dy["family"] == fam) & (dy["estimator"] == est) & (dy["year_by"] == yb)]
        if d.empty:
            continue
        yrs = sorted(d["year"].unique())
        x = np.arange(len(yrs))
        nlab = [f"{y}\n(n={int(d[d['year'] == y]['n'].iloc[0])})" for y in yrs]
        for i, (met, ylab) in enumerate([("cos_all", "cos(Δ_year, Δ_all years)"), ("delta_norm", "‖Δ_year‖")]):
            ax = axes[i, j]
            s = d[d["metric"] == met].set_index("year").loc[yrs]
            for off, pre, col, lab in ((-0.18, "null", "#888888", "null: random in-house subset of the same n (median, 2.5–97.5%)"),
                                       (0.18, "strat", "#D9A441", "null: same n and same histotype mix")):
                h = ax.errorbar(x + off, s[f"{pre}_median"], yerr=[s[f"{pre}_median"] - s[f"{pre}_lo"], s[f"{pre}_hi"] - s[f"{pre}_median"]],
                                fmt="_", color=col, ms=10, capsize=3, lw=1.4)
                handles[lab] = h
            handles["observed (raw Δ)"] = ax.plot(x, s["obs"], "o", color="#4477AA", ms=7, zorder=3)[0]
            mt = dy[(dy["family"] == fam) & (dy["estimator"] == "matched") & (dy["year_by"] == yb) & (dy["metric"] == met)]
            if not mt.empty:
                mt = mt.set_index("year").loc[yrs]
                handles["observed, histotype-matched Δ (own null in summary.md)"] = ax.plot(
                    x, mt["obs"], "D", mfc="none", color="#AA3377", ms=6, zorder=3)[0]
            if met == "delta_norm":
                handles["‖Δ_all years‖"] = ax.axhline(all_norm[fam], ls="--", color="k", lw=0.8)
            ax.set_xticks(x); ax.set_xticklabels(nlab, fontsize=8)
            ax.set_ylabel(ylab, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(FAMILY_LABEL.get(fam, fam), fontsize=12)
        # 年の組: cos と、院内の年どうしの距離（ノイズを引いた / ‖Δ_all‖）。下三角だけなので行 = 2 番目以降の年、列 = 最後以外の年
        p = dp[(dp["family"] == fam) & (dp["estimator"] == est) & (dp["year_by"] == yb)]
        rows_y, cols_y = yrs[1:], yrs[:-1]
        for i, (met, title, cmap, vmin, vmax) in enumerate([
                ("cos", "pairwise cos(Δ_a, Δ_b)\n(* below 2.5% of the same-n null; in brackets: null median)", "viridis", cos_lo, 1.0),
                ("dist_corr_ratio_null", "in-house year-to-year centroid distance,\nnoise-corrected, / ‖Δ_all years‖", "magma_r", 0.0, dist_hi)]):
            ax = axes[2 + i, j]
            M = np.full((len(rows_y), len(cols_y)), np.nan)
            Nmed = np.full_like(M, np.nan)
            star = np.zeros(M.shape, bool)
            for r in p[p["metric"] == met].itertuples():
                rb, ca = rows_y.index(r.year_b), cols_y.index(r.year_a)
                M[rb, ca] = r.obs
                Nmed[rb, ca] = r.null_median
                if met == "cos":
                    star[rb, ca] = r.null_p < 0.025
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax)
            mid = (vmin + vmax) / 2
            for rb in range(M.shape[0]):
                for ca in range(M.shape[1]):
                    if np.isnan(M[rb, ca]):
                        continue
                    txt = f"{M[rb, ca]:.2f}" + ("*" if star[rb, ca] else "")
                    if met == "cos":
                        txt += f"\n({Nmed[rb, ca]:.2f})"
                    dark = M[rb, ca] < mid if met == "cos" else M[rb, ca] > mid
                    ax.text(ca, rb, txt, ha="center", va="center", fontsize=7 if len(yrs) > 5 else 8.5,
                            color="w" if dark else "k")
            ax.set_xticks(range(len(cols_y))); ax.set_xticklabels(cols_y, fontsize=8, rotation=90)
            ax.set_yticks(range(len(rows_y))); ax.set_yticklabels(rows_y, fontsize=8)
            ax.set_title(title, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Is the in-house site shift stable over years? (TITAN; year = accession year from the slide ID)", fontsize=12)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3,
               fontsize=8.5, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


# ── summary ───────────────────────────────────────────────────────────────────

def _fmt_null(r, pre: str) -> str:
    return f"{r[f'{pre}_median']:.3f} [{r[f'{pre}_lo']:.3f}, {r[f'{pre}_hi']:.3f}]"


def summary(dy: pd.DataFrame, dp: pd.DataFrame, dh: pd.DataFrame, check_lines: list[str], args, path: Path) -> None:
    L = ["# shift_time — 院内の施設差 Δ は年でぶれないか（TITAN）", "",
         f"年ごとに院内が {args.min_n} 枚以上の年だけ。帰無は {args.n_null} 回（null = 同じ枚数の無作為な部分集合、"
         "strat = 同じ枚数・同じ組織型の構成）、中央値 [2.5%, 97.5%]、p は片側（cos は観測以下、長さ・距離は観測以上）。"
         f"不均一性の並べ替えは {args.n_perm} 回。seed {args.seed}。", ""]
    L += check_lines
    for yb in ("id", "scan"):
        L += [f"## 年ごと（year_by = {yb}{'、受付年' if yb == 'id' else '、撮像年'}）", ""]
        for fam in args.families:
            for est in ("raw", "matched"):
                d = dy[(dy["family"] == fam) & (dy["estimator"] == est) & (dy["year_by"] == yb)]
                if d.empty:
                    continue
                h = dh[(dh["family"] == fam) & (dh["estimator"] == est) & (dh["year_by"] == yb)].iloc[0]
                L += [f"### {fam} / {est}（‖Δ_all‖ = {h['delta_all_norm']:.3f}）", "",
                      "| year | n | ‖Δ_y‖ | null | cos(Δ_y, Δ_all) | null | p | strat | p | cos(Δ_y, Δ_rest) | null | "
                      "cos(Δ_y, Δ_paper) | proj | orth | drift |",
                      "|---|---:|---:|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|"]
                for y in sorted(d["year"].unique()):
                    s = d[d["year"] == y].set_index("metric")
                    L.append(f"| {y} | {int(s.loc['cos_all', 'n'])} | {s.loc['delta_norm', 'obs']:.2f} | "
                             f"{_fmt_null(s.loc['delta_norm'], 'null')} | {s.loc['cos_all', 'obs']:.3f} | "
                             f"{_fmt_null(s.loc['cos_all'], 'null')} | {s.loc['cos_all', 'null_p']:.3f} | "
                             f"{_fmt_null(s.loc['cos_all'], 'strat')} | {s.loc['cos_all', 'strat_p']:.3f} | "
                             f"{s.loc['cos_rest', 'obs']:.3f} | {_fmt_null(s.loc['cos_rest'], 'null')} | "
                             f"{s.loc['cos_paper', 'obs']:.3f} | {s.loc['proj', 'obs']:.2f} | {s.loc['orth', 'obs']:.2f} | "
                             f"{s.loc['drift', 'obs']:.3f} |")
                p = dp[(dp["family"] == fam) & (dp["estimator"] == est) & (dp["year_by"] == yb)]
                pc = p[p["metric"] == "cos"]
                pr = p[p["metric"] == "dist_ratio"]
                pcr = p[p["metric"] == "dist_corr_ratio_null"]
                if not pc.empty:
                    lo = pc.loc[pc["obs"].idxmin()]
                    L += ["", f"- 年の組 {len(pc)} 組: cos {pc['obs'].min():.3f}–{pc['obs'].max():.3f}（中央値 {pc['obs'].median():.3f}）、"
                          f"同じ枚数の帰無の中央値 {pc['null_median'].min():.3f}–{pc['null_median'].max():.3f}。"
                          f"帰無の 2.5% を下回る組 {int((pc['null_p'] < 0.025).sum())}（組織型をそろえた帰無では "
                          f"{int((pc['strat_p'] < 0.025).sum())}）。最小は {int(lo['year_a'])}–{int(lo['year_b'])}"
                          f"（{lo['obs']:.3f}、帰無 {_fmt_null(lo, 'null')}）。",
                          f"- 院内の年どうしの重心の距離 / ‖Δ_all‖: 観測 {pr['obs'].min():.3f}–{pr['obs'].max():.3f}"
                          f"（帰無の中央値 {pr['null_median'].min():.3f}–{pr['null_median'].max():.3f}）、ノイズを引くと "
                          f"{pcr['obs'].min():.3f}–{pcr['obs'].max():.3f}（中央値 {pcr['obs'].median():.3f}）。"]
                L += [f"- 年の不均一性 Q: p = {h['null_p']:.4f}（全体で並べ替え）/ {h['strat_p']:.4f}（組織型の中で並べ替え）、"
                      f"効果量（年の重心のばらつきの帰無超過分 / ‖Δ_all‖）{h['effect_null']:.3f} / {h['effect_strat']:.3f}。"
                      f"Δ_all 方向の成分と年の Spearman ρ = {h['spearman_rho']:+.3f}（p = {h['spearman_p']:.3g}、全 {int(h['n_trend'])} 枚）。", ""]
    path.write_text("\n".join(L) + "\n")
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["meningioma", "glioma", "metastasis"])
    ap.add_argument("--min-n", type=int, default=15, help="院内の年ごとの最小枚数")
    ap.add_argument("--min-group-ref", type=int, default=5, help="matched で群の EBRAINS 平均を使う最小例数（未満は科の平均）")
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rescan", action="store_true", help="WSI の DateTime を読み直す")
    ap.add_argument("--plot-only", action="store_true", help="計算せず、出力済みの CSV から図だけ作り直す")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "shift_time"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        plot(pd.read_csv(out_dir / "shift_by_year.csv"), pd.read_csv(out_dir / "year_pairs.csv"),
             pd.read_csv(out_dir / "heterogeneity.csv"), args.families, out_dir / "shift_time.png")
        return
    print("[load]")
    man, X = load_titan(cfg)
    man, check_lines = attach_years(man, out_dir / "scan_dates.csv", args.rescan)
    for ln in check_lines:
        if ln.startswith("- 照合") or ln.startswith("  "):
            print(" ", ln)
    site, fam = man["site"].to_numpy(), man["family"].to_numpy()
    paper = np.isin(fam, list(PAPER))
    d_paper = X[paper & (site == REF)].mean(0) - X[paper & (site != REF)].mean(0)
    print(f"  ‖Δ_paper‖ = {np.linalg.norm(d_paper):.3f}")

    rng = np.random.default_rng(args.seed)
    ry, rp, rh = [], [], []
    for yb in ("id", "scan"):
        for f in args.families:
            a, b, c = analyse(f, man, X, d_paper, yb, args, rng)
            ry += a; rp += b; rh += c
    dy, dp, dh = pd.DataFrame(ry), pd.DataFrame(rp), pd.DataFrame(rh)
    dy.to_csv(out_dir / "shift_by_year.csv", index=False)
    dp.to_csv(out_dir / "year_pairs.csv", index=False)
    dh.to_csv(out_dir / "heterogeneity.csv", index=False)
    print(f"  saved: {out_dir / 'shift_by_year.csv'}, year_pairs.csv, heterogeneity.csv")
    show = dy[(dy["metric"].isin(["delta_norm", "cos_all"])) & (dy["year_by"] == "id")]
    print(show.pivot_table(index=["family", "estimator", "year", "n"], columns="metric",
                           values=["obs", "null_median", "null_lo", "strat_median"]).round(3).to_string())
    print(dh.round(4).to_string(index=False))
    plot(dy, dp, dh, args.families, out_dir / "shift_time.png")
    summary(dy, dp, dh, check_lines, args, out_dir / "summary.md")


if __name__ == "__main__":
    main()
