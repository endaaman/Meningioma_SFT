"""transfer_correction.py — 転移テスト: 施設補正 3 法 × 推定元 × 当て先（decisions.md「転移テストの設計」2026-10-26）。

問い: ある腫瘍で推定した施設補正は、別の腫瘍にそのまま使えるか。予想は「平行移動の成分だけが移り、拡大縮小の成分
（相似変換・ComBat）は移らない」。補正は推定元のスライドだけで当てはめ、当て先の patho2 のスライドに適用する
（reference = config の reference = ebrains は常に不変）。

データ: out/transfer/manifest.csv（transfer_manifest.py、2,301 枚）。--encoder titan（既定）は h5_titan の transfer.titan_key、
    --encoder prism は h5_prism の transfer.prism_key（manifest の prism 列が False のスライドがあれば止まる）。
    埋め込みは out/transfer/embeddings_{encoder}.npy ＋ 並び（embeddings_{encoder}_slides.csv）にキャッシュする。
    manifest の並びと合わなければ自動で読み直す（--reload で強制）。
    --encoder meanpool（学習なしの集約器の対照: CONCH v1.5 のパッチ埋め込みの単純平均、meanpool.py と同じ定義）は
    transfer_meanpool.py が作るキャッシュ（out/transfer/embeddings_meanpool.npy ＋ embeddings_meanpool_ids.csv）を読むだけ
    （無ければ止まる。作り直しは transfer_meanpool.py build --reload）。

科（manifest の family）: meningioma / SFT / glioma（ast・glm・odg）/ lymphoma / metastasis。
当て先: paper（髄膜腫 + SFT。論文のベンチが使える）/ glioma / lymphoma / metastasis / all（5 科全部）。
推定元: none（補正なし）/ own（当て先自身 = 上限）/ paper / meningioma / SFT / glioma / lymphoma / metastasis /
        others（当て先の科以外の全部をまとめて 1 回で当てはめ）/
        others_mean（others と同じ科を科ごとに別々に当てはめ、パラメータを科ごとに等しい重みで平均。下の「others_mean」）。
    推定元を飛ばす規則（source_families）:
      - 推定元の科の集合が当て先と同じ → own と同じなので飛ばす（例: paper → paper、glioma → glioma）
      - 推定元が当て先の真部分集合（例: meningioma / SFT → paper）→ 推定に使ったスライドを評価に含む半分だけの
        in-sample になり、転移の問いにならないので飛ばす（部分抽出からの推定は shift_estimation.py が扱う）
      - ただし当て先 all は全部の科を含むので、科ごとの推定元と paper は残す（「1 つの科で推定した補正を全腫瘍の混合に
        当てる」。推定元のスライドは当て先に含まれる = 一部 in-sample）。others は空なので飛ばす
      - others_mean は others と同じ科を使う。当て先 all だけは others が空なので 5 科全部を科ごとに当てはめて平均する
        （= 科の構成をそろえた own。一部 in-sample は all の他の推定元と同じ）
    結果として推定元と当て先は all 以外では素（共通のスライドが無い）。

others_mean（others の構成の問題への対処）: 科の構成が施設で違う（例 glioma: EBRAINS 807 / patho2 117、metastasis 47 / 84）ので、
    まとめて当てはめた others の Δ には科どうしの差（長さ 10–16）が構成の差の重みで混ざる。科ごとに当てはめてから平均すると、
    各科の中では両施設の対応がとれているのでこれが入らない。平均の取り方（各科 f の当てはめ = 科 f の両施設のスライドだけ）:
      centroid          Δ = mean_f(Δ_f)、Δ_f = μ_E,f − μ_P,f
      affine_free       x' = s·x + b と書き、s = mean_f(s_f)、b = mean_f(b_f)、b_f = μ_E,f − s_f·μ_P,f
      combat            科 f の ComBat は次元ごとの 1 次式 x' = k_f·x + c_f（k_f = 1/√δ*_f、
                        c_f = m_f − (m_f + γ*_f·σ_f)·k_f、m_f = stand.mean、σ_f = √var.pooled。共変量なしなので mod.mean = 0）。
                        k = mean_f(k_f)、c = mean_f(c_f) を明示式で当てる。estimates の gamma.star / delta.star / stand.mean /
                        var.pooled をそのまま平均する案は採らない: γ*_f・δ*_f は科ごとの標準化（m_f, σ_f）の座標で表されていて、
                        科をまたいで足すと補正の平均にならない（科の平均 m_f どうしは 10–16 離れている）。データ空間の 1 次式で
                        平均するのは affine_free の規則と同じ。この平均は 1 組の estimates で表せないので neuroCombatFromTraining
                        は通さない（科ごとの 1 次式が FromTraining の出力と一致することは実行時に確かめて表示する）
      combat_recentred  倍率 k = mean_f(k_f) だけを移し、中心は当て先の施設平均（他と同じ）

3 法（論文と同じ定義。推定元の両施設の平均 μ_E, μ_P で当てはめ、当て先の patho2 の x に適用）:
    centroid         x' = x + (μ_E − μ_P)                                         （centroid.py）
    affine_free      x' = (x − μ_P)·s + μ_E、s = 施設平均まわりの RMS 距離の比（ebrains / patho2）（affine.scale_free をそのまま使う）
    combat           neuroCombat(dat=X.T, covars=DataFrame(batch), batch_col="batch", ref_batch=reference)（共変量なし、
                     combat.py と同じ呼び方）を推定元で当てはめ、その estimates を neuroCombatFromTraining で当て先に当てる。
                     FromTraining は estimates["batches"] が整数の符号（np.unique の順: ebrains = 0, patho2 = 1）なので、
                     batch は名前ではなく符号の文字列で渡す。
    combat_recentred （Supplementary の変種）推定元の ComBat の次元ごとの倍率 k_d = 1/√δ*_patho2,d だけを移し、
                     中心は当て先の施設平均で取り直す: x' = (x − μ_P,当て先)·k + μ_E,当て先（当て先の科のラベルが要る版）
    ComBat の確認（標準出力と summary.md）: 推定元 = 当て先（own）で FromTraining が neuroCombat の直接の出力を再現するか
    （max |差|）、reference の施設のスライドが FromTraining で不変か、FromTraining と明示式
    （((x − stand_mean)/√var_pooled − γ*)/√δ*·√var_pooled + stand_mean、共変量なしなので mod_mean = 0）が一致するか。
    一致しなければ明示式を使う（その旨を表示する）。

指標（当て先のスライド、patho2 だけ補正した後）:
    residual_norm  ‖mean(ebrains) − mean(補正後 patho2)‖
    asw_site       施設ラベルのシルエット係数（harmonization.py と同じ: PCA centroid.pca_n_components 次元 → silhouette。
                   harmonization._reduce / _site_values をそのまま使う）。ilisi_site も同じ関数から（k = harmonization.lisi_k）
    distortion     群の平均どうしの距離の比 d_patho2'(a, b) / d_ebrains(a, b) の群の組についての平均（1 = 歪みなし）。
                   群は当て先の両施設で min_n 例以上: paper = 髄膜腫サブタイプ + SFT、glioma = ast / glm / odg、all = 5 科
    scale / shift_norm  推定元で当てはめたパラメータ（affine_free: s、combat / combat_recentred: 次元ごとの倍率 1/√δ* の
                   中央値、centroid: ‖μ_E − μ_P‖。others_mean は平均した s・k・Δ）
    ベンチ:
      paper   pairs_adjacent（施設 × 組織型の平均の Ward 樹形図で同じ組織型の 2 施設が隣り合う数、13 組。dendrogram.py の
              group_means / raw_distance / ward_linkage / pair_stats をそのまま使う）、
              nc_histotype_ba_E2P / P2E（組織型の最近傍重心法、クラスは nc.histotype_set、nc.py と同じ）、
              nc_sft_ba_E2P / P2E（SFT vs 髄膜腫の最近傍重心法）、
              sft_men_dist_patho2 / sft_men_dist_ebrains / subtype_dist_mismatch（「通り越し」と組織型間距離の施設差。
              method_compare.shape_metrics をそのまま使う）
      glioma  nc_glioma3_ba_E2P / P2E（ast / glm / odg の最近傍重心法）、pairs_adjacent（3 組）
      all     nc_family5_ba_E2P / P2E（5 科の最近傍重心法）、pairs_adjacent（5 組）
    最近傍重心法は nc.py の点推定と同じ: 学習施設のクラス平均（nc._dist のユークリッド距離）で最も近いクラスに割り当て、
    balanced accuracy は sklearn の balanced_accuracy_score（lp_bootstrap._metrics と同じ）。E2P = ebrains の平均で
    patho2 を分類、P2E = 補正後の patho2 の平均で ebrains を分類。CI は付けない。

受け入れ確認: source = own・target = paper（と none）で論文の値（out/harmonization/metrics.csv、out/method_compare/summary.csv、
    out/dendrogram/pairs.csv、out/affine/scale.csv、out/nc/*/bootstrap.csv）と、論文の補正済み h5（embedding.patho2.*）を再現するか。
    meanpool は論文の平均プーリングの対照の出力（out/meanpool/summary.csv の none / centroid / combat、out/meanpool/nc/*/bootstrap.csv）と、
    meanpool.py の補正済み h5（{meanpool.root}/{centroid,combat}）と突き合わせる（相似変換は meanpool.py に無いので確認しない）。
    prism は論文の出力が無いので、同じ PRISM を別の経路で読んだ出力との一致を見る（acceptance_prism）: 補正なしの残差と own の
    ‖Δ‖ = shift_transfer.py --encoder prism の groups.csv の shift_norm（paper / glioma / lymphoma / metastasis）、単独の科を
    推定元にした相似変換の s = dispersion_ratio.py --encoder prism の scale_s、own の平行移動の残差 = 0。無いファイルの項目は飛ばす。

出力 (output_dir/transfer/):
    correction_{encoder}.csv          縦長: encoder, method, source, target, metric, value, n_ebrains, n_patho2（当て先）,
                                      n_src_ebrains, n_src_patho2（推定元）。source = none は補正なしで、ピボットしやすいように
                                      各 method に同じ行を置く
    correction_{encoder}_summary.md   受け入れ確認・ComBat の確認・推定元のパラメータ・当て先ごとのピボット（method × source）・ベンチ
    correction_{encoder}.png          a: residual_norm、b: asw_site（当て先ごとのヒートマップ、行 = 推定元、列 = 法）、c: paper のベンチ
    embeddings_{encoder}.npy / embeddings_{encoder}_slides.csv   埋め込みのキャッシュ

Usage:
    uv run python transfer_correction.py [--encoder titan|prism|meanpool] [--reload]
"""
from __future__ import annotations

import argparse
import contextlib
import io
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

np.int = int  # neuroCombat が numpy の旧い別名（np.int）を使うため、import より前に定義する（combat.py と同じ）

from neuroCombat import neuroCombat, neuroCombatFromTraining  # noqa: E402

from affine import scale_free  # noqa: E402
from dendrogram import group_means, pair_stats, raw_distance, ward_linkage  # noqa: E402
from harmonization import _reduce, _site_values  # noqa: E402
from histotype_labels import apply_set  # noqa: E402
from method_compare import shape_metrics  # noqa: E402
from nc import _dist  # noqa: E402
from utils.display import CONDITION_COLORS, CONDITION_LABELS_JA, CONDITION_LABELS_SHORT  # noqa: E402
from utils.loader import load_config, load_data  # noqa: E402

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
TARGET_FAMILIES = {"paper": {"meningioma", "SFT"}, "glioma": {"glioma"}, "lymphoma": {"lymphoma"},
                   "metastasis": {"metastasis"}, "all": set(FAMILIES)}
NAMED_SOURCES = {"paper": {"meningioma", "SFT"}, "meningioma": {"meningioma"}, "SFT": {"SFT"}, "glioma": {"glioma"},
                 "lymphoma": {"lymphoma"}, "metastasis": {"metastasis"}}
METHODS = ["centroid", "affine_free", "combat", "combat_recentred"]
SOURCES = ["none", "own", "paper", "meningioma", "SFT", "glioma", "lymphoma", "metastasis", "others", "others_mean"]
TARGETS = ["paper", "glioma", "lymphoma", "metastasis", "all"]
GLIOMA_GROUPS = ["ast", "glm", "odg"]
# 受け入れ確認で論文の variant と突き合わせる法（none = 補正なし = original）
PAPER_VARIANT = {"none": "original", "centroid": "centroid", "affine_free": "affine_free", "combat": "combat"}


# ── data ──────────────────────────────────────────────────────────────────────

def _read_h5(path: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f[key][:], dtype=np.float32).reshape(-1)


def load_embeddings(cfg: dict, encoder: str, out_dir: Path, reload: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """manifest の全スライドの埋め込み（manifest の並び、float64）。必要なスライドに埋め込みが無ければ止まる。"""
    if encoder == "meanpool":           # transfer_meanpool.py のキャッシュ（並びの確認もそちら）
        from transfer_meanpool import load_cache
        if reload:
            print("  （--reload は meanpool には効かない。作り直すなら transfer_meanpool.py build --reload）")
        man, X, ids = load_cache(cfg)
        print(f"  cache: {out_dir / 'embeddings_meanpool.npy'}  {X.shape}  " + str(ids["source"].value_counts().to_dict()))
        return man, X
    tcfg = cfg["transfer"]
    man = pd.read_csv(Path(cfg["output_dir"]) / tcfg.get("manifest", "transfer/manifest.csv"),
                      dtype={"slide_id": str, "patient": str})
    h5_col, ok_col, key = {"titan": ("h5_titan", "titan", tcfg["titan_key"]),
                           "prism": ("h5_prism", "prism", tcfg["prism_key"])}[encoder]
    ok = man[ok_col].astype(str).str.lower().eq("true")
    if not ok.all():
        miss = man[~ok].groupby(["family", "site"]).size()
        raise SystemExit(f"[{encoder}] {int((~ok).sum())} / {len(man)} 枚に埋め込み（{key}）が無い（manifest の {ok_col} 列が False）。"
                         f"抽出してから transfer_manifest.py を流し直すこと:\n{miss.to_string()}")
    npy, order_csv = out_dir / f"embeddings_{encoder}.npy", out_dir / f"embeddings_{encoder}_slides.csv"
    ident = man[["site", "slide_id"]].reset_index(drop=True)
    if not reload and npy.exists() and order_csv.exists():
        cached = pd.read_csv(order_csv, dtype={"slide_id": str})
        if cached.equals(ident):
            print(f"  cache: {npy}")
            return man, np.load(npy).astype(np.float64)
        print("  cache の並びが manifest と合わないので読み直す")
    print(f"  reading {len(man)} h5 ({key}) ...")
    X = np.stack([_read_h5(p, key) for p in man[h5_col]])
    np.save(npy, X)
    ident.to_csv(order_csv, index=False)
    print(f"  saved: {npy}  {X.shape}")
    return man, X.astype(np.float64)


def target_labels(man: pd.DataFrame, target: str) -> np.ndarray:
    """当て先の群ラベル: paper = 組織型（SFT は 'SFT'）、glioma = ast / glm / odg、all = 科、他 = 群（1 つ）。"""
    return (man["family"] if target == "all" else man["group"]).astype(str).to_numpy()


def source_families(target: str, source: str) -> frozenset | None:
    """推定元の科の集合。飛ばすときは None（規則は docstring の「推定元を飛ばす規則」）。"""
    tf = TARGET_FAMILIES[target]
    if source == "own":
        return frozenset(tf)
    if source == "others":
        rest = set(FAMILIES) - tf
        return frozenset(rest) if rest else None
    if source == "others_mean":          # 科ごとに当てはめて平均する科の集合（all は others が空なので全科）
        rest = set(FAMILIES) - tf
        return frozenset(rest or tf)
    sf = NAMED_SOURCES[source]
    if sf == tf:
        return None                      # own と同じ
    if sf & tf and target != "all":
        return None                      # 当て先の真部分集合（一部 in-sample）
    return frozenset(sf)


# ── fit / apply ───────────────────────────────────────────────────────────────

def fit_source(X: np.ndarray, site: np.ndarray, ref: str) -> dict:
    """推定元のスライドで 3 法を当てはめる（平均・相似変換の s・ComBat の estimates）。"""
    e, p = site == ref, site != ref
    levels = list(np.unique(site))      # neuroCombat の batch の符号（np.unique の順）
    with contextlib.redirect_stdout(io.StringIO()):
        comb = neuroCombat(dat=X.T, covars=pd.DataFrame({"batch": site}), batch_col="batch", ref_batch=ref)
    est = comb["estimates"]
    code_p = levels.index(next(s for s in levels if s != ref))
    return {"mu_e": X[e].mean(0), "mu_p": X[p].mean(0), "s": scale_free(X, e, p), "est": est,
            "codes": {s: str(i) for i, s in enumerate(levels)}, "code_p": code_p,
            "k": 1.0 / np.sqrt(np.asarray(est["delta.star"])[code_p]), "direct": comb["data"].T,
            "n_e": int(e.sum()), "n_p": int(p.sum())}


def combat_linear(fit: dict) -> tuple[np.ndarray, np.ndarray]:
    """推定元の ComBat（patho2 側）を次元ごとの 1 次式 x' = k·x + c に書き直した (k, c)。"""
    est = fit["est"]
    m = np.asarray(est["stand.mean"])[:, 0]
    sd = np.sqrt(np.asarray(est["var.pooled"])[:, 0])
    g = np.asarray(est["gamma.star"])[fit["code_p"]]
    k = fit["k"]
    return k, m - (m + g * sd) * k


def fit_mean(per_family: list[dict]) -> dict:
    """科ごとの当てはめのパラメータを科ごとに等しい重みで平均する（others_mean。docstring の定義）。"""
    lin = [combat_linear(f) for f in per_family]
    return {"kind": "mean",
            "delta": np.mean([f["mu_e"] - f["mu_p"] for f in per_family], axis=0),
            "s": float(np.mean([f["s"] for f in per_family])),
            "b": np.mean([f["mu_e"] - f["s"] * f["mu_p"] for f in per_family], axis=0),
            "k": np.mean([k for k, _ in lin], axis=0), "c": np.mean([c for _, c in lin], axis=0),
            "n_e": int(sum(f["n_e"] for f in per_family)), "n_p": int(sum(f["n_p"] for f in per_family)),
            "parts": per_family}


def combat_explicit(Xb: np.ndarray, est: dict, code: int) -> np.ndarray:
    """ComBat（パラメトリック、共変量なし）の適用を estimates から明示的に書いたもの（FromTraining の確認用）。"""
    m = np.asarray(est["stand.mean"])[:, 0]
    sd = np.sqrt(np.asarray(est["var.pooled"])[:, 0])
    g, d = np.asarray(est["gamma.star"])[code], np.asarray(est["delta.star"])[code]
    return ((Xb - m) / sd - g) / np.sqrt(d) * sd + m


def combat_from_training(Xb: np.ndarray, fit: dict, site_name: str) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        out = neuroCombatFromTraining(dat=Xb.T, batch=np.array([fit["codes"][site_name]] * len(Xb)), estimates=fit["est"])
    return np.asarray(out["data"]).T


def apply_method(method: str, fit: dict, Xt: np.ndarray, st: np.ndarray, ref: str, checks: dict) -> tuple[np.ndarray, dict]:
    """当て先の patho2 だけを補正したコピーと、推定元のパラメータを返す。"""
    Y = Xt.copy()
    p = st != ref
    oth = str(st[p][0])
    if fit.get("kind") == "mean":       # others_mean: 平均したパラメータ（docstring の定義）
        if method == "centroid":
            Y[p] = Xt[p] + fit["delta"]
            return Y, {"shift_norm": float(np.linalg.norm(fit["delta"]))}
        if method == "affine_free":
            Y[p] = fit["s"] * Xt[p] + fit["b"]
            return Y, {"scale": fit["s"]}
        if method == "combat":
            for part in fit["parts"]:  # 科ごとの 1 次式が FromTraining の出力と一致するか
                k, c = combat_linear(part)
                d = float(np.abs(k * Xt[p] + c - combat_from_training(Xt[p], part, oth)).max())
                checks["linear_vs_ft"] = max(checks.get("linear_vs_ft", 0.0), d)
            Y[p] = fit["k"] * Xt[p] + fit["c"]
            return Y, {"scale": float(np.median(fit["k"]))}
    if method == "centroid":
        Y[p] = Xt[p] + (fit["mu_e"] - fit["mu_p"])
        return Y, {"shift_norm": float(np.linalg.norm(fit["mu_e"] - fit["mu_p"]))}
    if method == "affine_free":
        Y[p] = (Xt[p] - fit["mu_p"]) * fit["s"] + fit["mu_e"]
        return Y, {"scale": fit["s"]}
    if method == "combat":
        ft = combat_from_training(Xt[p], fit, oth)
        ex = combat_explicit(Xt[p], fit["est"], fit["code_p"])
        ref_ft = combat_from_training(Xt[~p], fit, ref)
        checks["ft_vs_explicit"] = max(checks.get("ft_vs_explicit", 0.0), float(np.abs(ft - ex).max()))
        checks["ref_unchanged"] = max(checks.get("ref_unchanged", 0.0), float(np.abs(ref_ft - Xt[~p]).max()))
        if checks["ft_vs_explicit"] > 1e-8 and not checks.get("use_explicit"):
            print(f"  WARN neuroCombatFromTraining が明示式と合わない（max |差| {checks['ft_vs_explicit']:.2e}）→ 明示式を使う")
            checks["use_explicit"] = True
        Y[p] = ex if checks.get("use_explicit") else ft
        return Y, {"scale": float(np.median(fit["k"]))}
    if method == "combat_recentred":
        Y[p] = (Xt[p] - Xt[p].mean(0)) * fit["k"] + Xt[~p].mean(0)
        return Y, {"scale": float(np.median(fit["k"]))}
    raise ValueError(f"unknown method: {method}")


# ── metrics ───────────────────────────────────────────────────────────────────

def distortion(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, ref: str, min_n: int) -> tuple[float, int]:
    """群の平均どうしの距離の比 d_patho2'/d_ebrains の組についての平均と、使った群の数。"""
    e, p = st == ref, st != ref
    groups = [g for g in pd.unique(lab) if ((lab == g) & e).sum() >= min_n and ((lab == g) & p).sum() >= min_n]
    if len(groups) < 2:
        return float("nan"), len(groups)
    mu = {(g, w): Y[(lab == g) & m].mean(0) for g in groups for w, m in (("e", e), ("p", p))}
    r = [np.linalg.norm(mu[(a, "p")] - mu[(b, "p")]) / np.linalg.norm(mu[(a, "e")] - mu[(b, "e")])
         for a, b in combinations(groups, 2)]
    return float(np.mean(r)), len(groups)


def nc_ba(Y: np.ndarray, st: np.ndarray, y: np.ndarray, classes: list, tr: str, te: str) -> float:
    """最近傍重心法（nc.py の点推定と同じ）の balanced accuracy。学習施設 tr のクラス平均で te を分類。"""
    C = np.stack([Y[(st == tr) & (y == c)].mean(0) for c in classes])
    tem = st == te
    pred = _dist(Y[tem], C).argmin(1)
    true = np.array([classes.index(v) for v in y[tem]])
    return float(balanced_accuracy_score(true, pred))


def pairs(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, cfg: dict) -> dict:
    df = pd.DataFrame({"source": st, "subtype": lab, "embedding": list(Y)})
    g = group_means(df, cfg)
    D = raw_distance(g)
    ps = pair_stats(D, ward_linkage(D), g)
    return {"pairs_adjacent": ps["adjacent_pairs"], "pairs_n": ps["n_subtypes"]}


def evaluate(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, target: str, cfg: dict, ctx: dict) -> dict:
    ref, oth = ctx["ref"], ctx["oth"]
    e, p = st == ref, st != ref
    asw, ilisi = _site_values(_reduce(Y, ctx["pca_n"]), st, ctx["lisi_k"])
    out = {"residual_norm": float(np.linalg.norm(Y[e].mean(0) - Y[p].mean(0))), "asw_site": asw, "ilisi_site": ilisi}
    if target in ("paper", "glioma", "all"):
        out["distortion"], out["distortion_n_groups"] = distortion(Y, st, lab, ref, ctx["min_n"])
        out.update(pairs(Y, st, lab, cfg))
    if target == "paper":
        out.update(shape_metrics(pd.DataFrame({"source": st, "subtype": lab, "embedding": list(Y)}), cfg))
        cls = ctx["histotype_classes"]
        m = np.isin(lab, cls)
        sft = np.where(lab == "SFT", "SFT", "meningioma")
        for name, (a, b) in (("E2P", (ref, oth)), ("P2E", (oth, ref))):
            out[f"nc_histotype_ba_{name}"] = nc_ba(Y[m], st[m], lab[m], cls, a, b)
            out[f"nc_sft_ba_{name}"] = nc_ba(Y, st, sft, ["meningioma", "SFT"], a, b)
    elif target in ("glioma", "all"):
        cls = GLIOMA_GROUPS if target == "glioma" else FAMILIES
        task = "glioma3" if target == "glioma" else "family5"
        for name, (a, b) in (("E2P", (ref, oth)), ("P2E", (oth, ref))):
            out[f"nc_{task}_ba_{name}"] = nc_ba(Y, st, lab, cls, a, b)
    return out


# ── acceptance ────────────────────────────────────────────────────────────────

def _read(path: Path, index: str) -> pd.DataFrame | None:
    return pd.read_csv(path).set_index(index) if path.exists() else None


def acceptance(res: pd.DataFrame, cfg: dict, own_paper: dict[str, tuple[np.ndarray, np.ndarray]],
               encoder: str = "titan") -> pd.DataFrame:
    """source = own（と none）・target = paper の値を論文の出力と突き合わせる（meanpool は論文の平均プーリングの対照の出力と、
    prism は論文の出力が無いので acceptance_prism）。"""
    if encoder == "prism":
        return acceptance_prism(res, cfg)
    out = Path(cfg["output_dir"])
    if encoder == "meanpool":           # meanpool.py analyze / nc.py --aggregator mean の出力。harmonization・dendrogram・affine は無い
        harm, dg, sc = None, None, None
        mc = _read(out / "meanpool" / "summary.csv", "variant")
        nc_root, mc_name, nc_name, agg = out / "meanpool" / "nc", "meanpool/summary.csv", "meanpool/nc", "mean"
    else:
        harm = _read(out / "harmonization" / "metrics.csv", "state")
        mc = _read(out / "method_compare" / "summary.csv", "variant")
        dg = _read(out / "dendrogram" / "pairs.csv", "variant")
        sc = _read(out / "affine" / "scale.csv", "variant")
        nc_root, mc_name, nc_name, agg = out / "nc", "method_compare/summary.csv", "nc", "titan"
    nc_bs = {t: pd.read_csv(nc_root / t / "bootstrap.csv") for t in ("binary", "histotype")
             if (nc_root / t / "bootstrap.csv").exists()}
    ref = cfg["reference"]
    got = res[(res["target"] == "paper") & res["source"].isin(["own", "none"])]
    rows = []

    def add(method: str, metric: str, expected: float | None, tol: float, src: str) -> None:
        source = "none" if method == "none" else "own"
        m = "centroid" if method == "none" else method
        v = got[(got["source"] == source) & (got["method"] == m) & (got["metric"] == metric)]["value"]
        if expected is None or v.empty:
            return
        g = float(v.iloc[0])
        rows.append({"method": method, "metric": metric, "expected": float(expected), "got": g, "diff": g - float(expected),
                     "ok": abs(g - float(expected)) <= tol, "reference": src})

    for method, variant in PAPER_VARIANT.items():
        if harm is not None and variant in harm.index:
            add(method, "asw_site", harm.loc[variant, "asw_batch"], 1e-4, "harmonization/metrics.csv")
            add(method, "ilisi_site", harm.loc[variant, "ilisi"], 1e-3, "harmonization/metrics.csv")
        elif mc is not None and variant in mc.index:
            add(method, "asw_site", mc.loc[variant, "asw_batch"], 1e-4, mc_name)
            add(method, "ilisi_site", mc.loc[variant, "ilisi"], 1e-3, mc_name)
        if dg is not None and variant in dg.index:
            add(method, "pairs_adjacent", dg.loc[variant, "adjacent_pairs"], 0, "dendrogram/pairs.csv")
        elif encoder == "meanpool" and mc is not None and variant in mc.index:   # meanpool の対の数は summary.csv にある
            add(method, "pairs_adjacent", mc.loc[variant, "adjacent_pairs"], 0, mc_name)
        if mc is not None and variant in mc.index:
            for k in ("sft_men_dist_patho2", "sft_men_dist_ebrains", "subtype_dist_mismatch"):
                add(method, k, mc.loc[variant, k], 1e-3, mc_name)
            add(method, "residual_norm", mc.loc[variant, "site_shift_norm"], 1e-3, mc_name)
        if method == "affine_free" and sc is not None:
            add(method, "scale", sc.loc["affine_free", "scale"], 1e-4, "affine/scale.csv")
        for task, metric in (("histotype", "nc_histotype_ba"), ("binary", "nc_sft_ba")):
            b = nc_bs.get(task)
            if b is None:
                continue
            b = b[(b["variant"] == variant) & (b["metric"] == "balanced_accuracy")]
            for name, tr in (("E2P", ref), ("P2E", next(s for s in cfg["embedding"] if s != ref))):
                v = b[b["train_source"] == tr]["value"]
                if len(v):
                    add(method, f"{metric}_{name}", v.iloc[0], 1e-6, f"{nc_name}/{task}/bootstrap.csv")

    # 論文の補正済み h5（embedding.patho2.<variant>、meanpool は {meanpool.root}/<variant>）と、own で作った patho2 の埋め込みの一致（h5 は float32）
    oth = next(s for s in cfg["embedding"] if s != ref)
    for method, (Y, ids) in own_paper.items():
        variant = PAPER_VARIANT.get(method)
        if variant in (None, "original"):
            continue
        if encoder == "meanpool" and not (Path(cfg["meanpool"]["root"].format(site=oth)) / variant).exists():
            continue                    # meanpool.py が作らない補正（affine_*）
        d = load_data(cfg, variant, aggregator=agg)
        d = d[d["source"] != ref].set_index("case_id")
        common = [i for i in ids if i in d.index]
        if not common:
            continue
        pos = {i: k for k, i in enumerate(ids)}
        H = np.stack(d.loc[common, "embedding"].values).astype(np.float64)
        diff = float(np.abs(Y[[pos[i] for i in common]] - H).max())
        rows.append({"method": method, "metric": f"max|Y − h5|（{len(common)} 枚）", "expected": 0.0, "got": diff, "diff": diff,
                     "ok": diff <= 1e-4, "reference": f"meanpool.root/{variant}" if encoder == "meanpool" else f"embedding.patho2.{variant}"})
    return pd.DataFrame(rows)


def acceptance_prism(res: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """prism の確認: 論文の PRISM の出力は無いので、同じ PRISM（manifest の h5_prism）を別の経路で読んだ出力と突き合わせる。
    shift_transfer.py --encoder prism（out/shift_transfer/prism/groups.csv、論文コホートは load_data の並び）の shift_norm と
    補正なしの残差・own の ‖Δ‖、dispersion_ratio.py --encoder prism（out/dispersion_ratio/summary_prism.csv）の scale_s と
    単独の科を推定元にした相似変換の s。own の平行移動の残差は定義で 0。無いファイルの項目は飛ばす。"""
    out = Path(cfg["output_dir"])
    names = cfg["shift_transfer"].get("group_names", {})
    cols = ["method", "metric", "expected", "got", "diff", "ok", "reference"]
    rows = []

    def add(method: str, source: str, target: str | None, metric: str, expected: float, tol: float, ref: str) -> None:
        d = res[(res["method"] == method) & (res["source"] == source) & (res["metric"] == metric)]
        if target is not None:
            d = d[d["target"] == target]
        if d.empty:
            return
        g = float(d["value"].iloc[0])
        rows.append({"method": "none" if source == "none" else method,
                     "metric": f"{metric}（{source} → {target or d['target'].iloc[0]}）", "expected": float(expected), "got": g,
                     "diff": g - float(expected), "ok": abs(g - float(expected)) <= tol, "reference": ref})

    grp = _read(out / "shift_transfer" / "prism" / "groups.csv", "group")
    if grp is not None:
        for target, gname in (("paper", "Paper cohort"), ("glioma", "Glioma (all)"),
                              ("lymphoma", names.get("lym", "lym")), ("metastasis", names.get("met", "met"))):
            if gname in grp.index:
                add("centroid", "none", target, "residual_norm", grp.loc[gname, "shift_norm"], 1e-4, "shift_transfer/prism/groups.csv")
                add("centroid", "own", target, "shift_norm", grp.loc[gname, "shift_norm"], 1e-4, "shift_transfer/prism/groups.csv")
    disp = _read(out / "dispersion_ratio" / "summary_prism.csv", "family")
    if disp is not None:
        for f in FAMILIES:
            if f in disp.index:   # 推定元のパラメータは当て先によらないので、最初の当て先の行
                add("affine_free", f, None, "scale", disp.loc[f, "scale_s"], 1e-6, "dispersion_ratio/summary_prism.csv")
    for target in TARGETS:
        add("centroid", "own", target, "residual_norm", 0.0, 1e-6, "定義（own の平行移動）")
    return pd.DataFrame(rows, columns=cols)


# ── output ────────────────────────────────────────────────────────────────────

def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "OK" if v else "**NG**"
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer() and abs(v) >= 1):
        return str(int(v))
    return f"{v:.{digits}f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"


def _md(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + [_fmt(v, digits) if not isinstance(v, str) else v for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _label(method: str) -> str:
    return CONDITION_LABELS_JA.get(method, method)


def pivot(res: pd.DataFrame, target: str, metric: str, methods: list[str], sources: list[str]) -> pd.DataFrame:
    d = res[(res["target"] == target) & (res["metric"] == metric)]
    t = d.pivot_table(index="method", columns="source", values="value", aggfunc="first")
    t = t.reindex(index=[m for m in methods if m in t.index], columns=[s for s in sources if s in t.columns])
    t.index = [_label(m) for m in t.index]
    t.index.name = metric
    return t


def bench_table(res: pd.DataFrame, target: str, metrics: list[str], methods: list[str], sources: list[str]) -> pd.DataFrame:
    d = res[(res["target"] == target) & res["metric"].isin(metrics)]
    rows = []
    for s in sources:
        for m in (["centroid"] if s == "none" else methods):
            v = d[(d["source"] == s) & (d["method"] == m)].set_index("metric")["value"]
            if v.empty:
                continue
            rows.append({"source": s, "method": "—" if s == "none" else _label(m), **{k: v.get(k, np.nan) for k in metrics}})
    return pd.DataFrame(rows).set_index("source")


def write_summary(path: Path, res: pd.DataFrame, acc: pd.DataFrame, fits_tbl: pd.DataFrame, checks: dict,
                  encoder: str, methods: list[str], sources: list[str], targets: list[str], groups_used: dict) -> None:
    L = [f"# 転移テスト: 3 法 × 推定元 × 当て先（{encoder}）", "",
         "`transfer_correction.py` が作る。定義はスクリプトの docstring。数値は当て先のスライド（patho2 だけ補正）。",
         "推定元を飛ばす規則: 当て先と同じ科の集合（= own）と、当て先の真部分集合（一部 in-sample）は飛ばす。"
         "当て先 all だけは科ごとの推定元と paper を残す（推定元が当て先に含まれる）。"
         "others_mean = others と同じ科を科ごとに当てはめてパラメータを科ごとに等しい重みで平均（all は 5 科全部）。", ""]
    L += [{"meanpool": "## 受け入れ確認（target = paper、source = own / none と論文の平均プーリングの対照の出力 out/meanpool/）",
           "prism": "## 受け入れ確認（prism: 論文の出力は無いので、shift_transfer.py / dispersion_ratio.py --encoder prism の出力との一致）"}
          .get(encoder, "## 受け入れ確認（target = paper、source = own / none と論文の出力）"), "",
          _md(acc[["method", "metric", "expected", "got", "diff", "ok", "reference"]], digits=4, index=False), ""]
    L += ["## ComBat の転移（neuroCombatFromTraining）の確認", "",
          f"- own で FromTraining が neuroCombat の直接の出力を再現するか（patho2、当て先ごとの max |差| の最大）: "
          f"{checks.get('own_direct', float('nan')):.2e}",
          f"- reference（ebrains）のスライドが FromTraining で不変か（max |差|）: {checks.get('ref_unchanged', float('nan')):.2e}",
          f"- FromTraining と明示式（estimates の gamma.star / delta.star / stand.mean / var.pooled）の一致（全適用の max |差|）: "
          f"{checks.get('ft_vs_explicit', float('nan')):.2e}",
          f"- 使ったもの: {'明示式（FromTraining が合わなかった）' if checks.get('use_explicit') else 'neuroCombatFromTraining（batch は整数の符号の文字列で渡す）'}",
          f"- others_mean: 科ごとの ComBat を 1 次式 k·x + c に書き直したものと FromTraining の出力の一致（max |差|）: "
          f"{checks.get('linear_vs_ft', float('nan')):.2e}。平均した (k, c) は 1 組の estimates で表せないので明示式で当てる",
          ""]
    L += ["## 推定元のパラメータ", "",
          "scale_affine = 相似変換の s（施設平均まわりの RMS 比）、k_combat_* = ComBat の次元ごとの倍率 1/√δ* の分布、"
          "shift_norm = ‖μ_E − μ_P‖。", "", _md(fits_tbl, digits=3), ""]
    if groups_used:
        L += ["distortion の群（当て先の両施設で min_n 例以上）: " +
              "; ".join(f"{t} = {', '.join(g)}" for t, g in groups_used.items()), ""]
    for t in targets:
        n = res[res["target"] == t][["n_ebrains", "n_patho2"]].iloc[0]
        L += [f"## 当て先 {t}（ebrains {int(n['n_ebrains'])} / patho2 {int(n['n_patho2'])}）", ""]
        for metric in ("residual_norm", "asw_site", "distortion"):
            pv = pivot(res, t, metric, methods, sources)
            if pv.size:
                L += [f"### {metric}", "", _md(pv, digits=3), ""]
    L += ["## ベンチ", "", "### paper（髄膜腫 + SFT）", "",
          "pairs = 隣接ペア（/13）、histo = 組織型 5 クラスの最近傍重心法の BA、sft = SFT vs 髄膜腫の BA、"
          "d_sft_P = patho2 内の SFT−髄膜腫の距離（EBRAINS = d_sft_E）、mismatch = 組織型間距離の施設差（中央値）。", ""]
    pb = ["pairs_adjacent", "nc_histotype_ba_E2P", "nc_histotype_ba_P2E", "nc_sft_ba_E2P", "nc_sft_ba_P2E",
          "sft_men_dist_patho2", "sft_men_dist_ebrains", "subtype_dist_mismatch", "distortion"]
    if "paper" in targets:
        tb = bench_table(res, "paper", pb, methods, sources).rename(columns={
            "pairs_adjacent": "pairs", "nc_histotype_ba_E2P": "histo E→P", "nc_histotype_ba_P2E": "histo P→E",
            "nc_sft_ba_E2P": "sft E→P", "nc_sft_ba_P2E": "sft P→E", "sft_men_dist_patho2": "d_sft_P",
            "sft_men_dist_ebrains": "d_sft_E", "subtype_dist_mismatch": "mismatch"})
        L += [_md(tb, digits=3), ""]
    for t, task, n_pairs in (("glioma", "glioma3", 3), ("all", "family5", 5)):
        if t not in targets:
            continue
        mets = [f"nc_{task}_ba_E2P", f"nc_{task}_ba_P2E", "pairs_adjacent", "distortion"]
        tb = bench_table(res, t, mets, methods, sources).rename(columns={
            f"nc_{task}_ba_E2P": "BA E→P", f"nc_{task}_ba_P2E": "BA P→E", "pairs_adjacent": f"pairs (/{n_pairs})"})
        title = "glioma（ast / glm / odg の 3 クラス）" if t == "glioma" else "all（5 科の分類）"
        L += [f"### {title}", "", _md(tb, digits=3), ""]
    path.write_text("\n".join(L))
    print(f"  saved: {path}")


def plot(res: pd.DataFrame, path: Path, methods: list[str], sources: list[str], targets: list[str], encoder: str) -> None:
    fig = plt.figure(figsize=(4.0 * len(targets), 13.5))
    gs = fig.add_gridspec(3, len(targets), height_ratios=[1, 1, 0.85], hspace=0.55, wspace=0.45, top=0.93, bottom=0.08)
    for row, (metric, cmap, digits) in enumerate((("residual_norm", "Blues", 2), ("asw_site", "Purples", 3))):
        vals = res[res["metric"] == metric]["value"]
        vmin, vmax = min(0.0, float(vals.min())), float(vals.max())
        for j, t in enumerate(targets):
            ax = fig.add_subplot(gs[row, j])
            d = res[(res["target"] == t) & (res["metric"] == metric)]
            pv = d.pivot_table(index="source", columns="method", values="value", aggfunc="first")
            pv = pv.reindex(index=[s for s in sources if s in pv.index], columns=[m for m in methods if m in pv.columns])
            ax.imshow(pv.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for (i, k), v in np.ndenumerate(pv.values):
                light = (v - vmin) / (vmax - vmin + 1e-12) > 0.55
                ax.text(k, i, f"{v:.{digits}f}", ha="center", va="center", fontsize=7.5, color="white" if light else "#222222")
            ax.set_xticks(range(pv.shape[1]))
            ax.set_xticklabels([CONDITION_LABELS_SHORT.get(m, m) for m in pv.columns], rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(pv.shape[0]))
            ax.set_yticklabels(pv.index, fontsize=8)
            n = d[["n_ebrains", "n_patho2"]].iloc[0]
            ax.set_title(f"{'ab'[row]}  target: {t} (E {int(n['n_ebrains'])} / P {int(n['n_patho2'])})\n{metric}", fontsize=9, loc="left")
            ax.tick_params(length=0)
    # c: paper のベンチ（推定元ごとに法の棒。none は 1 本）
    bench = [("pairs_adjacent", "adjacent pairs (/13)", None), ("nc_histotype_ba_E2P", "histotype BA  E→P", None),
             ("nc_histotype_ba_P2E", "histotype BA  P→E", None), ("nc_sft_ba_E2P", "SFT vs meningioma BA  E→P", None),
             ("sft_men_dist_patho2", "SFT–meningioma distance in patho2", "sft_men_dist_ebrains")]
    d = res[res["target"] == "paper"]
    srcs = [s for s in sources if s in set(d["source"])]
    w = 0.8 / len(methods)
    for j, (metric, title, line) in enumerate(bench[: len(targets)]):
        ax = fig.add_subplot(gs[2, j])
        for i, s in enumerate(srcs):
            for k, m in enumerate(["centroid"] if s == "none" else methods):
                v = d[(d["source"] == s) & (d["method"] == m) & (d["metric"] == metric)]["value"]
                if v.empty:
                    continue
                x = i if s == "none" else i - 0.4 + w * (k + 0.5)
                ax.bar(x, v.iloc[0], width=w * 0.9, color=CONDITION_COLORS["original"] if s == "none" else CONDITION_COLORS.get(m, "#888888"))
        if line:
            ref_v = d[d["metric"] == line]["value"].iloc[0]
            ax.axhline(ref_v, color="#555555", lw=0.9, ls="--")
            ax.text(-0.45, ref_v, "EBRAINS", fontsize=7, va="bottom", ha="left", color="#555555")
        vv = d[d["metric"] == metric]["value"]
        hi = max(float(vv.max()), ref_v if line else 0.0)
        lo = 0.0 if metric == "pairs_adjacent" else max(0.0, float(vv.min()) - 0.3 * (hi - float(vv.min())) - 0.02)
        ax.set_ylim(lo, hi + (0.6 if metric == "pairs_adjacent" else 0.08 * (hi - lo)))
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels(srcs, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{'c' if j == 0 else ''}  target: paper\n{title}", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONDITION_COLORS["original"])] + \
        [plt.Rectangle((0, 0), 1, 1, color=CONDITION_COLORS.get(m, "#888888")) for m in methods]
    fig.legend(handles, ["None (uncorrected)"] + [CONDITION_LABELS_SHORT.get(m, m) for m in methods], loc="lower center",
               ncol=len(methods) + 1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Transfer of site correction: method × estimation source × target ({encoder}). "
                 "Rows = source of the estimate (none = uncorrected, own = target itself)", fontsize=11, y=0.975)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def _fit_row(fit_rows: list[dict], cohort: str, f: dict) -> None:
    delta = f["delta"] if f.get("kind") == "mean" else f["mu_e"] - f["mu_p"]
    fit_rows.append({"cohort": cohort, "n_ebrains": f["n_e"], "n_patho2": f["n_p"],
                     "shift_norm": float(np.linalg.norm(delta)), "scale_affine": f["s"],
                     "k_combat_median": float(np.median(f["k"])), "k_combat_q05": float(np.quantile(f["k"], 0.05)),
                     "k_combat_q95": float(np.quantile(f["k"], 0.95))})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["titan", "prism", "meanpool"], default="titan",
                    help="スライドエンコーダ（manifest の h5 と key）。meanpool = CONCH のパッチ平均（transfer_meanpool.py のキャッシュ）")
    ap.add_argument("--reload", action="store_true", help="埋め込みのキャッシュを捨てて h5 から読み直す")
    args = ap.parse_args()
    cfg = load_config()
    ccfg = cfg["transfer"].get("correction", {}) or {}
    methods = ccfg.get("methods", METHODS)
    sources = ccfg.get("sources", SOURCES)
    targets = ccfg.get("targets", TARGETS)
    out_dir = Path(cfg["output_dir"]) / "transfer"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = cfg["reference"]

    print(f"[transfer_correction / {args.encoder}]")
    man, X = load_embeddings(cfg, args.encoder, out_dir, args.reload)
    fam, site = man["family"].to_numpy(), man["site"].to_numpy()
    oth = next(s for s in pd.unique(site) if s != ref)
    print(f"  {len(man)} slides: " + ", ".join(f"{f} {int(((fam == f) & (site == ref)).sum())}/{int(((fam == f) & (site == oth)).sum())}"
                                              for f in FAMILIES) + f"  ({ref}/{oth})")
    ctx = {"ref": ref, "oth": oth, "min_n": int(ccfg.get("min_n", 5)),
           "pca_n": cfg.get("centroid", {}).get("pca_n_components", 50),
           "lisi_k": cfg.get("harmonization", {}).get("lisi_k", 30),
           "histotype_classes": apply_set(cfg, cfg.get("nc", {}).get("histotype_set", "main"))["lp_histotype"]["classes"]}

    fits: dict = {}   # 推定元の科の集合（others_mean は ("mean", 科の集合)）→ 当てはめ

    def get_fit(sf: frozenset) -> dict:
        if sf not in fits:
            sm = np.isin(fam, list(sf))
            fits[sf] = fit_source(X[sm], site[sm], ref)
            _fit_row(fit_rows, "+".join(x for x in FAMILIES if x in sf), fits[sf])
        return fits[sf]

    checks: dict = {}
    rows: list[dict] = []
    own_paper: dict[str, tuple[np.ndarray, list[str]]] = {}
    groups_used: dict[str, list[str]] = {}
    fit_rows: list[dict] = []
    for target in targets:
        tm = np.isin(fam, list(TARGET_FAMILIES[target]))
        Xt, st, lab = X[tm], site[tm], target_labels(man[tm], target)
        base = {"encoder": args.encoder, "target": target, "n_ebrains": int((st == ref).sum()), "n_patho2": int((st != ref).sum())}
        m0 = evaluate(Xt, st, lab, target, cfg, ctx)
        if target in ("paper", "glioma", "all"):
            e, p = st == ref, st != ref
            groups_used[target] = [g for g in pd.unique(lab) if ((lab == g) & e).sum() >= ctx["min_n"] and ((lab == g) & p).sum() >= ctx["min_n"]]
        for method in methods:   # none は method によらず同じ（各 method に同じ行を置く）
            rows += [{**base, "method": method, "source": "none", "metric": k, "value": v, "n_src_ebrains": 0, "n_src_patho2": 0}
                     for k, v in m0.items()]
        print(f"  [{target} / none] residual {m0['residual_norm']:.3f}  ASW {m0['asw_site']:.4f}")
        for source in sources:
            if source == "none":
                continue
            sf = source_families(target, source)
            if sf is None:
                continue
            if source == "others_mean":
                key = ("mean", sf)
                if key not in fits:
                    fits[key] = fit_mean([get_fit(frozenset({f})) for f in FAMILIES if f in sf])
                    _fit_row(fit_rows, "mean(" + ", ".join(f for f in FAMILIES if f in sf) + ")", fits[key])
                fit = fits[key]
            else:
                fit = get_fit(sf)
            for method in methods:
                Y, par = apply_method(method, fit, Xt, st, ref, checks)
                if source == "own" and method == "combat":   # 推定元 = 当て先: 直接の出力と比べる（並びは同じ）
                    diff = float(np.abs(Y[st != ref] - fit["direct"][st != ref]).max())
                    ref_diff = float(np.abs(fit["direct"][st == ref] - Xt[st == ref]).max())
                    checks["own_direct"] = max(checks.get("own_direct", 0.0), diff)
                    print(f"  [{target} / own / combat] FromTraining vs neuroCombat 直接: max |差| {diff:.2e}  "
                          f"（直接の出力の reference 不変 {ref_diff:.2e}）")
                if source == "own" and target == "paper":
                    own_paper[method] = (Y[st != ref], man.loc[tm, "slide_id"].to_numpy()[st != ref].tolist())
                m = {**evaluate(Y, st, lab, target, cfg, ctx), **par}
                rows += [{**base, "method": method, "source": source, "metric": k, "value": v,
                          "n_src_ebrains": fit["n_e"], "n_src_patho2": fit["n_p"]} for k, v in m.items()]
                print(f"  [{target} / {source} / {method}] residual {m['residual_norm']:.3f}  ASW {m['asw_site']:.4f}"
                      + (f"  distortion {m['distortion']:.3f}" if "distortion" in m else ""))
    print(f"  ComBat checks: own vs direct {checks.get('own_direct', np.nan):.2e}  reference unchanged {checks.get('ref_unchanged', np.nan):.2e}  "
          f"FromTraining vs explicit {checks.get('ft_vs_explicit', np.nan):.2e}  "
          f"per-family linear form vs FromTraining {checks.get('linear_vs_ft', np.nan):.2e}  -> "
          f"{'explicit' if checks.get('use_explicit') else 'neuroCombatFromTraining'}")

    res = pd.DataFrame(rows)[["encoder", "method", "source", "target", "metric", "value", "n_ebrains", "n_patho2",
                              "n_src_ebrains", "n_src_patho2"]]
    csv = out_dir / f"correction_{args.encoder}.csv"
    res.to_csv(csv, index=False)
    print(f"  saved: {csv}")

    acc = acceptance(res, cfg, own_paper, args.encoder)
    print("\n[acceptance: target = paper, source = own / none vs paper outputs]")
    print(acc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(acc) and not acc["ok"].all():
        print(f"  WARN {int((~acc['ok']).sum())} checks differ")

    fits_tbl = pd.DataFrame(fit_rows).set_index("cohort")
    write_summary(out_dir / f"correction_{args.encoder}_summary.md", res, acc, fits_tbl, checks, args.encoder,
                  methods, sources, targets, groups_used)
    plot(res, out_dir / f"correction_{args.encoder}.png", methods, sources, targets, args.encoder)


if __name__ == "__main__":
    main()
