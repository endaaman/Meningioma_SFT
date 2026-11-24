"""transfer_uncertainty.py — 転移テストの主要な数字の不確かさと、査読で問われやすい点の確認（足すだけの解析、TITAN、2026-11-24）。

既存の定義・コホート・出力は動かさない（transfer_correction.py・atlas_retrieval.py の関数をそのまま使い、点推定が既存の値と
一致することを最初に確かめる）。出力は output_dir/transfer_uncertainty/ だけ。

サブコマンド（既定は全部）:
  bootstrap  主要な指標の 95% 区間。患者単位のブートストラップ（施設 × 群の中で患者を復元抽出し、その患者のスライドを全部入れる、
             1,000 回）。再抽出のたびに補正を当てはめ直す（当てはめの不確かさを含める）。
             当て先グリオーマ（借りる元 = 髄膜腫）と髄膜腫（借りる元 = グリオーマ）× 補正なし / 同じ腫瘍で推定 / 借りる
             × 平行移動・相似変換・ComBat: 隣り合う組の数（分布）、2 施設の中心の距離（群の中央値）、近傍の混ざり（元のデータで両施設
             5 枚以上の群の中央値）。鑑別の腫瘍（SFT ← 髄膜腫、リンパ腫・転移 ← グリオーマ）の中心の距離と混ざり（補正なし / 借りる × 3 法）。
             近傍の混ざりは、同じスライドの複製を近傍から除く（複製がそのまま近傍に入ると同じ施設の近傍が水増しされるため。
             複製の無い元のデータでは既存の定義と同じ値になる）。
  retrieval  グリオーマの参照コホートへの診断名の検索（atlas_retrieval.py、最近傍重心法 nc_euclid）の 1 位一致の 95% 区間と、
             補正なし 対 髄膜腫から借りる の対応のある差の区間。同じ患者単位のブートストラップで Δ を当てはめ直し、院内の問い合わせも
             再抽出する。参照（EBRAINS の診断名ごとの重心）は固定（参照コホートは与えられたものとして扱う）。
             同じ腫瘍のずれ（family）は atlas_retrieval.py と同じく問い合わせ自身（の複製すべて）を院内側の平均から除く。
  retrieval_tumours  retrieval を 4 つの腫瘍（グリオーマ全体・リンパ腫・転移・SFT）の 1 位・5 位以内に広げたもの（v4 Fig 7 の誤差線）。
             同じ再抽出の流れなのでグリオーマの 1 位は retrieval と同じ。補正なし 対 髄膜腫から借りる の対応のある差、良くなった・悪く
             なった枚数と McNemar の正確検定（点推定のスライドで）。EBRAINS の中（患者単位で除いた値）は問い合わせの患者を再抽出。
  split      「同じ腫瘍で推定」の当てはめと評価の分離。当て先の両施設を、施設 × 群の中で患者単位に半分に分け、片方（A）で補正を
             当てはめ、もう片方（B）で評価する（200 回）。比べる条件はすべて同じ B で評価: 補正なし / A で当てはめ（分離）/
             B 自身で当てはめ（B の中で当てはめと評価が重なる、今の「同じ腫瘍」と同じ形）/ 借りる（借りる元は全スライド）× 3 法。
             近傍の混ざりは元のデータで両施設 5 枚以上の群に限る（B ではその群の枚数がおよそ半分）。
  subsample  SFT から借りて組が戻らないのは枚数のせいか。髄膜腫から EBRAINS 32 枚・院内 22 枚（SFT と同じ枚数）を患者単位で
             無作為に抜いて平行移動のずれを推定し、グリオーマ（3 組）に当てる（500 回）。逆向きにグリオーマから 32 / 22 枚を抜いて
             髄膜腫（12 組）に当てる。SFT から借りた実測値（同じ指標）が分布のどこに来るか、抜いたずれの長さと、全枚数の髄膜腫
             （グリオーマ）のずれ・SFT のずれとの cos も。

出力（output_dir/transfer_uncertainty/）:
    bootstrap_draws.csv / bootstrap_summary.csv / bootstrap_pairs_freq.csv
    retrieval_draws.csv / retrieval_summary.csv
    retrieval_tumours_draws.csv / retrieval_tumours_summary.csv / retrieval_tumours_paired.csv
    split_draws.csv / split_summary.csv
    subsample_draws.csv / subsample_summary.csv
    summary.md

Usage:
    uv run python transfer_uncertainty.py [bootstrap retrieval retrieval_tumours split subsample report] [--n-boot 1000] [--n-split 200] [--n-sub 500]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import binomtest

import atlas_retrieval as ar
import transfer_correction as tc
from utils.loader import load_config
from utils.display import ordered_subtypes

REF = "ebrains"
METHODS = ["centroid", "affine_free", "combat"]
MAIN = {"glioma": "meningioma", "meningioma": "glioma"}           # 当て先 → 借りる元
RARE = {"SFT": "meningioma", "lymphoma": "glioma", "metastasis": "glioma"}
MIN_N = 5
KNN_K = tc.KNN_K
SFT_N = {"ebrains": 32, "patho2": 22}                             # SFT の枚数（subsample で抜く枚数）
GLIOMA = ["ast", "glm", "odg"]


# ── data ──────────────────────────────────────────────────────────────────────

class Data:
    def __init__(self, cfg: dict):
        out = Path(cfg["output_dir"]) / "transfer"
        self.man, self.X = tc.load_embeddings(cfg, "titan", out, False)
        self.fam = self.man["family"].astype(str).to_numpy()
        self.site = self.man["site"].astype(str).to_numpy()
        self.group = self.man["group"].astype(str).to_numpy()
        self.patient = self.man["patient"].astype(str).to_numpy()
        # 施設 × 群 → 患者 → 行
        self.strata: dict[tuple[str, str], list[np.ndarray]] = {}
        for (s, g), d in self.man.groupby(["site", "group"], sort=True):
            self.strata[(s, g)] = [np.asarray(ix) for _, ix in d.groupby("patient", sort=True).groups.items()]
        # 元のデータで両施設 5 枚以上の群（近傍の混ざりの中央値をとる群。当て先ごと）
        self.min5 = {}
        for t in list(MAIN) + list(RARE):
            m = self.fam == t
            g, s = self.group[m], self.site[m]
            self.min5[t] = {x for x in set(g) if ((g == x) & (s == REF)).sum() >= MIN_N and ((g == x) & (s != REF)).sum() >= MIN_N}

    def boot_index(self, rng: np.random.Generator) -> np.ndarray:
        idx = []
        for _, pats in self.strata.items():
            pick = rng.integers(0, len(pats), len(pats))
            idx += [pats[i] for i in pick]
        return np.concatenate(idx)


# ── metrics ───────────────────────────────────────────────────────────────────

def mixing(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, oid: np.ndarray, cfg: dict, min5: set) -> dict:
    """tc.site_mixing と同じ定義。ただし近傍から同じスライド（oid が同じ行 = 自分とその複製）を除く。
    group_gap_sd は両施設にある群すべての中央値、knn_mix は min5 の群の中央値。"""
    e = st == REF
    gaps, knn = [], []
    for g in ordered_subtypes(set(lab[e]) & set(lab[~e]), cfg):
        am, bm = (lab == g) & e, (lab == g) & ~e
        A, B = Y[am], Y[bm]
        n_e, n_p = len(A), len(B)
        mu_e, mu_p = A.mean(0), B.mean(0)
        rms = float(np.sqrt((((A - mu_e) ** 2).sum(1).mean() + ((B - mu_p) ** 2).sum(1).mean()) / 2))
        gaps.append(float(np.linalg.norm(mu_e - mu_p)) / rms if rms > 0 else np.nan)
        if g in min5:
            k = min(KNN_K, n_e + n_p - 1)
            D = cdist(B, np.vstack([A, B]))
            D[oid[bm][:, None] == np.concatenate([oid[am], oid[bm]])[None, :]] = np.inf
            frac = float((np.argsort(D, axis=1, kind="stable")[:, :k] < n_e).mean())
            knn.append(frac / (n_e / (n_e + n_p - 1)))
    return {"group_gap_sd": float(np.nanmedian(gaps)), "knn_mix": float(np.median(knn)) if knn else np.nan}


def evaluate(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, oid: np.ndarray, cfg: dict, min5: set, with_pairs: bool) -> dict:
    out = mixing(Y, st, lab, oid, cfg, min5)
    if with_pairs:
        out.update(tc.pairs(Y, st, lab, cfg))
    return out


def run_target(X: np.ndarray, st: np.ndarray, lab: np.ndarray, oid: np.ndarray, fits: dict, cfg: dict, min5: set,
               with_pairs: bool, methods=METHODS) -> list[dict]:
    """fits = {条件名: 当てはめ}。補正なし ＋ 条件 × 法の評価の行。"""
    rows = [{"condition": "none", "method": "none", **evaluate(X, st, lab, oid, cfg, min5, with_pairs)}]
    checks: dict = {}
    for cond, fit in fits.items():
        for m in methods:
            Y, _ = tc.apply_method(m, fit, X, st, REF, checks)
            rows.append({"condition": cond, "method": m, **evaluate(Y, st, lab, oid, cfg, min5, with_pairs)})
    return rows


def summarise(d: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(keys, sort=False):
        for m in metrics:
            v = g[m].dropna().to_numpy(float)
            if not len(v):
                continue
            rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))), "metric": m, "n_draws": len(v),
                         "median": float(np.median(v)), "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975)),
                         "mean": float(v.mean())})
    return pd.DataFrame(rows)


# ── bootstrap ─────────────────────────────────────────────────────────────────

def one_replicate(D: Data, idx: np.ndarray, cfg: dict) -> list[dict]:
    X, st, fam, grp = D.X[idx], D.site[idx], D.fam[idx], D.group[idx]
    fit = {f: tc.fit_source(X[fam == f], st[fam == f], REF) for f in ("meningioma", "glioma")}
    rows = []
    for t, src in MAIN.items():
        m = fam == t
        for r in run_target(X[m], st[m], grp[m], idx[m], {"own": fit[t], "borrowed": fit[src]}, cfg, D.min5[t], True):
            rows.append({"target": t, "source": src, **r})
    for t, src in RARE.items():
        m = fam == t
        for r in run_target(X[m], st[m], grp[m], idx[m], {"borrowed": fit[src]}, cfg, D.min5[t], False):
            rows.append({"target": t, "source": src, **r})
    return rows


def check_point(rows: list[dict], cfg: dict) -> None:
    """元のデータ（再抽出なし）の値が correction_titan.csv（と _groups.csv の両施設 5 枚以上の中央値）と一致するか。"""
    t = Path(cfg["output_dir"]) / "transfer"
    c = pd.read_csv(t / "correction_titan.csv").set_index(["target", "source", "method", "metric"])["value"]
    g = pd.read_csv(t / "correction_titan_groups.csv")
    g = g[(g["metric"] == "knn_mix") & (g["n_ebrains"] >= MIN_N) & (g["n_patho2"] >= MIN_N)]
    k5 = g.groupby(["target", "source", "method"])["value"].median()
    rt = pd.read_csv(t / "rare_target_titan.csv")   # SFT は correction_titan の当て先に無いので rare_target（当て先 paper の SFT の群）
    rt = rt[(rt["target"] == "paper") & (rt["group"] == "SFT")].set_index(["source", "method", "metric"])["value"]
    bad = 0
    for r in rows:
        src = {"none": "none", "own": "own", "borrowed": r["source"]}[r["condition"]]
        meth = "centroid" if r["method"] == "none" else r["method"]
        key = (r["target"], src, meth)
        if r["target"] == "SFT":
            ref = {"group_gap_sd": rt[(src, meth, "group_gap_sd")], "knn_mix": rt[(src, meth, "knn_mix")]}
        else:
            ref = {"group_gap_sd": c[key + ("group_gap_sd",)], "knn_mix": k5[key]}
        if "pairs_adjacent" in r:
            ref["pairs_adjacent"] = c[key + ("pairs_adjacent",)]
        for k, v in ref.items():
            if not np.isclose(r[k], v, atol=1e-9):
                bad += 1
                print(f"  MISMATCH {key} {k}: {r[k]} vs {v}")
    print(f"  point check vs correction_titan: {len(rows)} rows, mismatches {bad}")
    if bad:
        raise SystemExit("点推定が既存の出力と合わない")


def cmd_bootstrap(D: Data, cfg: dict, out: Path, n_boot: int, seed: int) -> None:
    print(f"[bootstrap] {n_boot} draws (patients within site × group)")
    check_point(one_replicate(D, np.arange(len(D.man)), cfg), cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_boot):
        rows += [{"draw": b, **r} for r in one_replicate(D, D.boot_index(rng), cfg)]
        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{n_boot}")
    d = pd.DataFrame(rows)
    d.to_csv(out / "bootstrap_draws.csv", index=False)
    keys = ["target", "source", "condition", "method"]
    s = summarise(d, keys, ["pairs_adjacent", "group_gap_sd", "knn_mix"])
    s.to_csv(out / "bootstrap_summary.csv", index=False)
    pf = d[d["pairs_adjacent"].notna()].groupby(keys + ["pairs_adjacent"], sort=False).size().rename("count").reset_index()
    pf["fraction"] = pf["count"] / n_boot
    pf.to_csv(out / "bootstrap_pairs_freq.csv", index=False)
    # 対応のある差（借りる − 補正なし、借りた平行移動 − 借りた相似変換 / ComBat）
    w = d.pivot_table(index=["draw", "target"], columns=["condition", "method"], values=["group_gap_sd", "knn_mix"])
    diffs = []
    for metric in ("group_gap_sd", "knn_mix"):
        for t in list(MAIN) + list(RARE):
            ww = w.xs(t, level="target")
            base = ww[(metric, "none", "none")]
            for m in METHODS:
                v = (ww[(metric, "borrowed", m)] - base).dropna().to_numpy()
                diffs.append({"target": t, "metric": metric, "contrast": f"borrowed {m} − none", "median": np.median(v),
                              "q025": np.quantile(v, 0.025), "q975": np.quantile(v, 0.975)})
            for m in ("affine_free", "combat"):
                v = (ww[(metric, "borrowed", m)] - ww[(metric, "borrowed", "centroid")]).dropna().to_numpy()
                diffs.append({"target": t, "metric": metric, "contrast": f"borrowed {m} − borrowed centroid", "median": np.median(v),
                              "q025": np.quantile(v, 0.025), "q975": np.quantile(v, 0.975), "frac_gt0": float((v > 0).mean())})
    pd.DataFrame(diffs).to_csv(out / "bootstrap_contrasts.csv", index=False)
    print(f"  saved: {out}/bootstrap_*.csv")


# ── retrieval ─────────────────────────────────────────────────────────────────

def retrieval_setup(cfg: dict) -> dict:
    """atlas_retrieval.main と同じ参照のクラス・正解の集合（グリオーマ 3 群の問い合わせの分だけ）。"""
    h5_col, key_name = ar.ENCODERS["titan"]
    key = cfg["transfer"][key_name]
    adir = Path(cfg["output_dir"]) / "atlas"
    atlas, A, _ = ar.load_atlas(cfg, key, adir, "titan", False)
    man, M = ar.load_manifest(cfg, h5_col, key, adir, "titan", False)
    counts = atlas["diagnosis"].value_counts()
    classes = sorted(counts[counts >= ar.MIN_N_DEFAULT].index, key=lambda d: (-counts[d], d))
    keep = atlas["diagnosis"].isin(classes).to_numpy()
    R = A[keep]
    cpos = {c: k for k, c in enumerate(classes)}
    y_idx = atlas.loc[keep, "diagnosis"].map(cpos).to_numpy()
    C = np.stack([R[y_idx == k].mean(0) for k in range(len(classes))])
    mapping = ar.label_mapping(cfg, set(atlas["diagnosis"]))
    correct = {lab: [d for d in ds if d in cpos] for lab, ds in mapping.items()}
    return {"man": man, "M": M, "C": C, "classes": classes, "correct": correct}


def retrieval_once(S: dict, idx: np.ndarray, D: Data) -> dict:
    man, M = S["man"], S["M"]
    fam, site, grp = D.fam[idx], D.site[idx], D.group[idx]
    X = M[idx]
    d_men = X[(fam == "meningioma") & (site == REF)].mean(0) - X[(fam == "meningioma") & (site != REF)].mean(0)
    gq = (fam == "glioma") & (site != REF)
    Xq, oq, lq = X[gq], idx[gq], grp[gq]
    mu_e = X[(fam == "glioma") & (site == REF)].mean(0)
    S_p, N_p = Xq.sum(0), len(Xq)
    copies = pd.Series(oq).map(pd.Series(oq).value_counts()).to_numpy()
    fam_shift = mu_e - (S_p - copies[:, None] * Xq) / (N_p - copies)[:, None]   # 問い合わせ自身（の複製すべて）を除く
    corr = ar.corr_mask(lq, S["classes"], S["correct"])
    out = {}
    for cond, Q in (("none", Xq), ("meningioma", Xq + d_men), ("family", Xq + fam_shift)):
        hit = ar.nc_retrieve(ar._dist(Q, S["C"], "euclid"), corr)["hit1"]
        out[(cond, "glioma")] = float(hit.mean())
        for g in GLIOMA:
            out[(cond, g)] = float(hit[lq == g].mean())
    return out


def cmd_retrieval(D: Data, cfg: dict, out: Path, n_boot: int, seed: int) -> None:
    print(f"[retrieval] {n_boot} draws")
    S = retrieval_setup(cfg)
    assert (S["man"][["site", "slide_id"]].to_numpy() == D.man[["site", "slide_id"]].to_numpy()).all()
    p0 = retrieval_once(S, np.arange(len(D.man)), D)
    ref = pd.read_csv(Path(cfg["output_dir"]) / "atlas" / "retrieval_titan.csv")
    ref = ref[(ref["retrieval"] == "nc_euclid") & (ref["metric"] == "top1") & (ref["query_label"] == "(all)")]
    for (cond, g), v in p0.items():
        r = float(ref[(ref["condition"] == cond) & (ref["query_family"] == g)]["value"].iloc[0])
        assert np.isclose(v, r, atol=1e-9), (cond, g, v, r)
    print(f"  point check vs retrieval_titan.csv: ok ({len(p0)} values)")
    rng = np.random.default_rng(seed + 1)
    rows = []
    for b in range(n_boot):
        v = retrieval_once(S, D.boot_index(rng), D)
        rows += [{"draw": b, "condition": c, "group": g, "top1": x} for (c, g), x in v.items()]
    d = pd.DataFrame(rows)
    d.to_csv(out / "retrieval_draws.csv", index=False)
    s = summarise(d, ["condition", "group"], ["top1"])
    s["point"] = [p0[(c, g)] for c, g in zip(s["condition"], s["group"])]
    w = d.pivot_table(index=["draw", "group"], columns="condition", values="top1")
    for contrast, (a, b_) in {"meningioma − none": ("meningioma", "none"), "family − meningioma": ("family", "meningioma")}.items():
        for g in ["glioma"] + GLIOMA:
            v = (w.xs(g, level="group")[a] - w.xs(g, level="group")[b_]).to_numpy()
            s = pd.concat([s, pd.DataFrame([{"condition": contrast, "group": g, "metric": "top1_diff", "n_draws": len(v),
                                             "median": np.median(v), "q025": np.quantile(v, 0.025), "q975": np.quantile(v, 0.975),
                                             "mean": v.mean(), "point": p0[(a, g)] - p0[(b_, g)],
                                             "frac_gt0": float((v > 0).mean())}])], ignore_index=True)
    s.to_csv(out / "retrieval_summary.csv", index=False)
    print(f"  saved: {out}/retrieval_*.csv")


# ── retrieval, 4 tumours ──────────────────────────────────────────────────────

TUMOURS = ["glioma", "lymphoma", "metastasis", "SFT"]
RCONDS = ["none", "meningioma", "family"]


def retrieval_tumours_once(S: dict, idx: np.ndarray, D: Data) -> dict:
    """retrieval_once を 4 つの腫瘍（グリオーマ・リンパ腫・転移・SFT）の院内の問い合わせに広げたもの。1 位と 5 位以内のスライドごとの当たり。"""
    M = S["M"][idx]
    fam, site, grp = D.fam[idx], D.site[idx], D.group[idx]
    d_men = M[(fam == "meningioma") & (site == REF)].mean(0) - M[(fam == "meningioma") & (site != REF)].mean(0)
    out = {}
    for t in TUMOURS:
        qm = (fam == t) & (site != REF)
        Xq, oq, lq = M[qm], idx[qm], grp[qm]
        mu_e = M[(fam == t) & (site == REF)].mean(0)
        copies = pd.Series(oq).map(pd.Series(oq).value_counts()).to_numpy()
        fam_shift = mu_e - (Xq.sum(0) - copies[:, None] * Xq) / (len(Xq) - copies)[:, None]
        corr = ar.corr_mask(lq, S["classes"], S["correct"])
        for cond, Q in (("none", Xq), ("meningioma", Xq + d_men), ("family", Xq + fam_shift)):
            r = ar.nc_retrieve(ar._dist(Q, S["C"], "euclid"), corr)
            out[(t, cond)] = {"hit1": r["hit1"], "hit5": r["hit5"], "oid": oq, "label": lq}
    return out


def cmd_retrieval_tumours(D: Data, cfg: dict, out: Path, n_boot: int, seed: int) -> None:
    """4 つの腫瘍の検索の 1 位・5 位以内の区間（Fig 7）。retrieval と同じ再抽出（同じ seed の流れ）で、グリオーマの 1 位は retrieval と同じ値になる。
    EBRAINS の中（患者単位で除いた値、atlas_retrieval の ebrains_lpo）は、問い合わせの患者を腫瘍の中で再抽出する（参照は固定）。"""
    print(f"[retrieval_tumours] {n_boot} draws")
    S = retrieval_setup(cfg)
    root = Path(cfg["output_dir"]) / "atlas"
    ref = pd.read_csv(root / "retrieval_titan.csv")
    ref = ref[(ref["retrieval"] == "nc_euclid") & (ref["query_label"] == "(all)")].set_index(["condition", "query_family", "metric"])["value"]
    p0 = retrieval_tumours_once(S, np.arange(len(D.man)), D)
    for (t, c), r in p0.items():
        for m, k in (("top1", "hit1"), ("top5", "hit5")):
            assert np.isclose(r[k].mean(), ref[(c, t, m)], atol=1e-9), (t, c, m)
    eb = pd.read_csv(root / "per_slide_titan.csv", dtype={"slide_id": str})
    eb = eb[(eb["retrieval"] == "nc_euclid") & (eb["condition"] == "ebrains_lpo") & eb["family"].isin(TUMOURS) & eb["scorable"]]
    atlas = pd.read_csv(root / "atlas_titan_slides.csv", dtype={"slide_id": str})
    eb = eb.assign(patient=eb["slide_id"].map(dict(zip(atlas["slide_id"], atlas["patient"].astype(str)))))
    assert eb["patient"].notna().all()
    for t, d in eb.groupby("family"):
        for m, k in (("top1", "hit1"), ("top5", "hit5")):
            assert np.isclose(d[k].mean(), ref[("ebrains_lpo", t, m)], atol=1e-9), (t, m)
    print("  point check vs retrieval_titan.csv: ok (none / meningioma / family / ebrains_lpo × 4 tumours × top1 / top5)")
    eb_groups = {t: [np.asarray(v) for v in d.groupby("patient").indices.values()] for t, d in eb.reset_index(drop=True).groupby("family")}
    eb_hits = {t: d.reset_index(drop=True)[["hit1", "hit5"]].to_numpy(float) for t, d in eb.groupby("family")}
    eb_lab = {t: d.reset_index(drop=True)["label"].to_numpy() for t, d in eb.groupby("family")}

    def by_group(t: str, h1: np.ndarray, h5: np.ndarray, lab: np.ndarray) -> list[tuple[str, float, float]]:
        out_ = [("(all)", h1.mean(), h5.mean())]
        if t == "glioma":   # グリオーマの 3 群の内訳（Fig 7c）
            out_ += [(g, h1[lab == g].mean(), h5[lab == g].mean()) for g in GLIOMA]
        return out_
    rng = np.random.default_rng(seed + 1)   # retrieval と同じ流れ（グリオーマの値が retrieval と一致する）
    rng_eb = np.random.default_rng(seed + 4)
    rows = []
    for b in range(n_boot):
        res = retrieval_tumours_once(S, D.boot_index(rng), D)
        for (t, c), r in res.items():
            rows += [{"draw": b, "tumour": t, "group": g, "condition": c, "top1": x1, "top5": x5}
                     for g, x1, x5 in by_group(t, r["hit1"].astype(float), r["hit5"].astype(float), r["label"])]
        for t, pats in eb_groups.items():
            ii = np.concatenate([pats[i] for i in rng_eb.integers(0, len(pats), len(pats))])
            h = eb_hits[t][ii]
            rows += [{"draw": b, "tumour": t, "group": g, "condition": "ebrains_lpo", "top1": x1, "top5": x5}
                     for g, x1, x5 in by_group(t, h[:, 0], h[:, 1], eb_lab[t][ii])]
    d = pd.DataFrame(rows)
    d.to_csv(out / "retrieval_tumours_draws.csv", index=False)
    refg = pd.read_csv(root / "retrieval_titan.csv")
    refg = refg[(refg["retrieval"] == "nc_euclid") & (refg["query_label"] == "(all)")].set_index(["condition", "query_family", "metric"])
    srows = []
    for (t, grp_, c), g in d.groupby(["tumour", "group", "condition"], sort=False):
        key = t if grp_ == "(all)" else grp_
        for m in ("top1", "top5"):
            v = g[m].to_numpy()
            srows.append({"tumour": t, "group": grp_, "condition": c, "metric": m, "n": int(refg.loc[(c, key, m), "n"]),
                          "point": float(refg.loc[(c, key, m), "value"]), "median": np.median(v), "q025": np.quantile(v, 0.025),
                          "q975": np.quantile(v, 0.975)})
    pd.DataFrame(srows).to_csv(out / "retrieval_tumours_summary.csv", index=False)
    w = d[d["group"] == "(all)"].pivot_table(index=["draw", "tumour"], columns="condition", values=["top1", "top5"])
    prows = []
    for t in TUMOURS:
        for m, k in (("top1", "hit1"), ("top5", "hit5")):
            a_, b_ = p0[(t, "none")][k], p0[(t, "meningioma")][k]
            up, down = int((~a_ & b_).sum()), int((a_ & ~b_).sum())
            v = (w[(m, "meningioma")] - w[(m, "none")]).xs(t, level="tumour").to_numpy()
            prows.append({"tumour": t, "metric": m, "n": len(a_), "none": a_.mean(), "borrowed": b_.mean(), "diff": b_.mean() - a_.mean(),
                          "diff_median": np.median(v), "diff_q025": np.quantile(v, 0.025), "diff_q975": np.quantile(v, 0.975),
                          "improved": up, "worsened": down,
                          "mcnemar_exact_p": binomtest(up, up + down, 0.5).pvalue if up + down else np.nan})
    pd.DataFrame(prows).to_csv(out / "retrieval_tumours_paired.csv", index=False)
    print(pd.DataFrame(prows).round(4).to_string(index=False))
    print(f"  saved: {out}/retrieval_tumours_*.csv")


# ── split ─────────────────────────────────────────────────────────────────────

def split_halves(D: Data, t: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    A, B = [], []
    for (s, g), pats in D.strata.items():
        if not ((D.fam == t) & (D.group == g)).any():
            continue
        order = rng.permutation(len(pats))
        h = len(pats) // 2 + (rng.random() < 0.5 if len(pats) % 2 else 0)   # 奇数のときの余りはどちらかに無作為に
        A += [pats[i] for i in order[:h]]
        B += [pats[i] for i in order[h:]]
    return np.concatenate(A) if A else np.array([], int), np.concatenate(B) if B else np.array([], int)


def cmd_split(D: Data, cfg: dict, out: Path, n_split: int, seed: int) -> None:
    print(f"[split] {n_split} patient-level half splits per target")
    full_fit = {f: tc.fit_source(D.X[D.fam == f], D.site[D.fam == f], REF) for f in ("meningioma", "glioma")}
    rng = np.random.default_rng(seed + 2)
    rows = []
    for k in range(n_split):
        for t, src in MAIN.items():
            a, b = split_halves(D, t, rng)
            fa = tc.fit_source(D.X[a], D.site[a], REF)
            fb = tc.fit_source(D.X[b], D.site[b], REF)
            for r in run_target(D.X[b], D.site[b], D.group[b], b, {"own_split": fa, "own_insample_half": fb, "borrowed": full_fit[src]},
                                cfg, D.min5[t], True):
                rows.append({"split": k, "target": t, "source": src, "n_eval_ebrains": int((D.site[b] == REF).sum()),
                             "n_eval_patho2": int((D.site[b] != REF).sum()), **r})
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{n_split}")
    d = pd.DataFrame(rows)
    d.to_csv(out / "split_draws.csv", index=False)
    s = summarise(d, ["target", "source", "condition", "method"], ["pairs_adjacent", "pairs_n", "group_gap_sd", "knn_mix"])
    s.to_csv(out / "split_summary.csv", index=False)
    print(f"  saved: {out}/split_*.csv")


# ── subsample ─────────────────────────────────────────────────────────────────

def draw_slides(D: Data, fam: str, rng: np.random.Generator) -> np.ndarray:
    """科 fam から、施設ごとに SFT と同じ枚数になるまで患者を無作為に足す（足すと超える患者は飛ばす）。"""
    out = []
    for s, n in SFT_N.items():
        m = (D.fam == fam) & (D.site == s)
        pats = pd.Series(np.where(m)[0]).groupby(D.patient[m]).apply(lambda v: v.to_numpy())
        got = []
        for i in rng.permutation(len(pats)):
            if len(got) + len(pats.iloc[i]) <= n:
                got += list(pats.iloc[i])
            if len(got) == n:
                break
        out += got
    return np.asarray(out)


def cmd_subsample(D: Data, cfg: dict, out: Path, n_sub: int, seed: int) -> None:
    print(f"[subsample] {n_sub} draws of {SFT_N} slides")
    delta = {f: D.X[(D.fam == f) & (D.site == REF)].mean(0) - D.X[(D.fam == f) & (D.site != REF)].mean(0)
             for f in ("meningioma", "glioma", "SFT")}
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
    rng = np.random.default_rng(seed + 3)
    rows = []
    for src, t in (("meningioma", "glioma"), ("glioma", "meningioma")):
        m = D.fam == t
        Xt, st, lab, oid = D.X[m], D.site[m], D.group[m], np.where(m)[0]
        # 実測: SFT から借りる / 全枚数の推定元から借りる
        for name, d in (("SFT (observed)", delta["SFT"]), (f"{src}, all slides", delta[src])):
            Y = Xt.copy()
            Y[st != REF] += d
            rows.append({"source_family": src, "target": t, "draw": name, "n_src_ebrains": np.nan, "n_src_patho2": np.nan,
                         "shift_norm": float(np.linalg.norm(d)), "cos_vs_full_source": cos(d, delta[src]),
                         "cos_vs_rest_source": cos(d, delta[src]), "cos_vs_sft": cos(d, delta["SFT"]),
                         **evaluate(Y, st, lab, oid, cfg, D.min5[t], True)})
        for k in range(n_sub):
            ix = draw_slides(D, src, rng)
            d = D.X[ix][D.site[ix] == REF].mean(0) - D.X[ix][D.site[ix] != REF].mean(0)
            rest = (D.fam == src) & ~np.isin(np.arange(len(D.fam)), ix)   # 抜いたスライドを除いた推定元の残り（重なりで cos が上がらないように）
            d_rest = D.X[rest & (D.site == REF)].mean(0) - D.X[rest & (D.site != REF)].mean(0)
            Y = Xt.copy()
            Y[st != REF] += d
            rows.append({"source_family": src, "target": t, "draw": k, "n_src_ebrains": int((D.site[ix] == REF).sum()),
                         "n_src_patho2": int((D.site[ix] != REF).sum()), "shift_norm": float(np.linalg.norm(d)),
                         "cos_vs_full_source": cos(d, delta[src]), "cos_vs_rest_source": cos(d, d_rest), "cos_vs_sft": cos(d, delta["SFT"]),
                         **evaluate(Y, st, lab, oid, cfg, D.min5[t], True)})
    d = pd.DataFrame(rows)
    d.to_csv(out / "subsample_draws.csv", index=False)
    srows = []
    for (src, t), g in d.groupby(["source_family", "target"], sort=False):
        draws = g[g["draw"].map(lambda x: isinstance(x, (int, np.integer)))]
        obs = g[g["draw"] == "SFT (observed)"].iloc[0]
        full = g[g["draw"] == f"{src}, all slides"].iloc[0]
        for metric in ("pairs_adjacent", "group_gap_sd", "knn_mix", "shift_norm", "cos_vs_full_source", "cos_vs_rest_source", "cos_vs_sft"):
            v = draws[metric].to_numpy(float)
            srows.append({"source_family": src, "target": t, "metric": metric, "median": np.median(v), "q025": np.quantile(v, 0.025),
                          "q975": np.quantile(v, 0.975), "sft_observed": obs[metric], "full_source": full[metric],
                          "frac_le_sft": float((v <= obs[metric]).mean()), "frac_ge_sft": float((v >= obs[metric]).mean()),
                          "n_src_ebrains_mean": draws["n_src_ebrains"].mean(), "n_src_patho2_mean": draws["n_src_patho2"].mean()})
    pd.DataFrame(srows).to_csv(out / "subsample_summary.csv", index=False)
    pf = d[d["draw"].map(lambda x: isinstance(x, (int, np.integer)))].groupby(["source_family", "target", "pairs_adjacent"]).size()
    (pf / n_sub).rename("fraction").reset_index().to_csv(out / "subsample_pairs_freq.csv", index=False)
    print(f"  saved: {out}/subsample_*.csv")


def _ci(r: pd.Series, d: int = 3) -> str:
    return f"{r['median']:.{d}f} [{r['q025']:.{d}f}, {r['q975']:.{d}f}]"


def cmd_report(cfg: dict, out: Path) -> None:
    """出力の CSV から summary.md を作る（計算はしない）。"""
    L = ["# 転移テストの不確かさと査読の確認（transfer_uncertainty.py）", ""]
    cond_label = {("none", "none"): "No correction"}
    if (out / "bootstrap_summary.csv").exists():
        s = pd.read_csv(out / "bootstrap_summary.csv")
        pf = pd.read_csv(out / "bootstrap_pairs_freq.csv")
        L += ["## 1. 患者単位のブートストラップ（中央値 [2.5%, 97.5%]、補正は再抽出ごとに当てはめ直す）", "",
              "| 当て先 | 条件 | 法 | 隣り合う組 | 組の数の頻度 | 中心の距離 | 近傍の混ざり（5 枚以上） |", "|---|---|---|---|---|---|---|"]
        for (t, c, m), g in s.groupby(["target", "condition", "method"], sort=False):
            gg = g.set_index("metric")
            pr = _ci(gg.loc["pairs_adjacent"], 0) if "pairs_adjacent" in gg.index else ""
            fr = pf[(pf["target"] == t) & (pf["condition"] == c) & (pf["method"] == m)]
            frs = "; ".join(f"{int(x.pairs_adjacent)}: {x.fraction:.2f}" for x in fr.sort_values("pairs_adjacent").itertuples())
            L.append(f"| {t} | {c} | {m} | {pr} | {frs} | {_ci(gg.loc['group_gap_sd'])} | {_ci(gg.loc['knn_mix'])} |")
        if (out / "bootstrap_contrasts.csv").exists():
            cdf = pd.read_csv(out / "bootstrap_contrasts.csv")
            L += ["", "対応のある差（同じ再抽出の中で）:", "", "| 当て先 | 指標 | 差 | 中央値 [2.5%, 97.5%] | 正の割合 |", "|---|---|---|---|---|"]
            for x in cdf.itertuples():
                L.append(f"| {x.target} | {x.metric} | {x.contrast} | {x.median:.3f} [{x.q025:.3f}, {x.q975:.3f}] | "
                         f"{'' if pd.isna(getattr(x, 'frac_gt0', np.nan)) else f'{x.frac_gt0:.3f}'} |")
        L.append("")
    if (out / "retrieval_summary.csv").exists():
        r = pd.read_csv(out / "retrieval_summary.csv")
        L += ["## 1b. グリオーマの診断名の検索、1 位の一致（参照の重心は固定）", "", "| 条件 | 群 | 点推定 | 中央値 [2.5%, 97.5%] |",
              "|---|---|---|---|"]
        for x in r.itertuples():
            L.append(f"| {x.condition} | {x.group} | {x.point:.3f} | {x.median:.3f} [{x.q025:.3f}, {x.q975:.3f}] |")
        L.append("")
    if (out / "split_summary.csv").exists():
        s = pd.read_csv(out / "split_summary.csv")
        L += ["## 2. 当てはめと評価の分離（患者単位の半分、200 回。評価はすべて同じ半分 B）", "",
              "| 当て先 | 条件 | 法 | 隣り合う組 | 組の総数 | 中心の距離 | 近傍の混ざり |", "|---|---|---|---|---|---|---|"]
        for (t, c, m), g in s.groupby(["target", "condition", "method"], sort=False):
            gg = g.set_index("metric")
            L.append(f"| {t} | {c} | {m} | {_ci(gg.loc['pairs_adjacent'], 0)} | {_ci(gg.loc['pairs_n'], 0)} | "
                     f"{_ci(gg.loc['group_gap_sd'])} | {_ci(gg.loc['knn_mix'])} |")
        L.append("")
    if (out / "subsample_summary.csv").exists():
        s = pd.read_csv(out / "subsample_summary.csv")
        L += ["## 3. SFT と同じ枚数（EBRAINS 32 / 院内 22）で推定したずれ（500 回）", "",
              "| 推定元 → 当て先 | 指標 | 抜いたずれ 中央値 [2.5%, 97.5%] | SFT から借りた実測 | 全枚数から | 実測以下の割合 | 実測以上の割合 |",
              "|---|---|---|---|---|---|---|"]
        for x in s.itertuples():
            L.append(f"| {x.source_family} → {x.target} | {x.metric} | {x.median:.3f} [{x.q025:.3f}, {x.q975:.3f}] | {x.sft_observed:.3f} | "
                     f"{x.full_source:.3f} | {x.frac_le_sft:.3f} | {x.frac_ge_sft:.3f} |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print(f"  saved: {out}/summary.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("commands", nargs="*", choices=["bootstrap", "retrieval", "retrieval_tumours", "split", "subsample", "report"],
                    help="既定（何も書かない）は全部（最後に report）")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-split", type=int, default=200)
    ap.add_argument("--n-sub", type=int, default=500)
    args = ap.parse_args()
    cfg = load_config()
    seed = int(cfg.get("shift_transfer", {}).get("seed", 42))
    out = Path(cfg["output_dir"]) / "transfer_uncertainty"
    out.mkdir(parents=True, exist_ok=True)
    if args.commands == ["report"]:
        cmd_report(cfg, out)
        return
    D = Data(cfg)
    for c in args.commands or ["bootstrap", "retrieval", "retrieval_tumours", "split", "subsample", "report"]:
        if c == "report":
            cmd_report(cfg, out)
            continue
        {"bootstrap": lambda: cmd_bootstrap(D, cfg, out, args.n_boot, seed),
         "retrieval": lambda: cmd_retrieval(D, cfg, out, args.n_boot, seed),
         "retrieval_tumours": lambda: cmd_retrieval_tumours(D, cfg, out, args.n_boot, seed),
         "split": lambda: cmd_split(D, cfg, out, args.n_split, seed),
         "subsample": lambda: cmd_subsample(D, cfg, out, args.n_sub, seed)}[c]()


if __name__ == "__main__":
    main()
