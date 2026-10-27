"""transfer_sample_size.py — 何枚・何種類の腫瘍があれば、別の腫瘍に移せる Δ を推定できるか（転移テスト、TITAN）。

論文の Fig S4（shift_estimation.py の B: 新施設の n 枚から Δ̂ を推定して全スライドに当てる）を、腫瘍をまたぐ設定で
作り直し、「推定元に使う腫瘍の種類の数」の曲線を足す。補正は平行移動だけ: 推定した Δ̂ を当て先の patho2 のスライドに
足す（reference = config の reference = ebrains は不変）。科・当て先・最近傍重心法の定義は transfer_correction.py と同じ
（import して使う）で、全枚数の行は transfer_correction の method = centroid の行と一致する（受け入れ確認）。

データ: out/transfer/manifest.csv（transfer_manifest.py、2,301 枚）と、transfer_correction.py のキャッシュ
    out/transfer/embeddings_{encoder}.npy（並び = embeddings_{encoder}_slides.csv）。並びを manifest と照合し、合わなければ
    h5 から読み直す（このスクリプトはキャッシュを書かない。out/transfer/ は transfer_correction.py の持ち物）。

A. 推定に要る枚数（n の曲線）
    推定元（既定 paper = 髄膜腫 + SFT。ほかに単独の科 glioma / lymphoma / metastasis）の patho2 から無作為に n 枚 S をとり
        Δ̂ = mean(EBRAINS の推定元) − mean(S)
    n ∈ N_LIST（推定元の patho2 の枚数未満）と全枚数（= transfer_correction の推定元そのもの。決定的なので 1 回 = full）。
    各 n で n_draws 回（既定 200、seed 42、設定ごとに独立な乱数列 default_rng([seed, 推定元の番号, 抜き方の番号, n])。
    推定元を足しても他の推定元の抽出は変わらない）。抜き方:
        patho2 … patho2 だけを n 枚に減らす（EBRAINS は推定元の全部。Fig S4 と同じ運用: 公開側は全部手元にある）
        both   … 両施設から n 枚ずつ（EBRAINS が n 枚に満たなければ全部）
    同じ Δ̂ を、推定元と科が重ならない当て先（paper / glioma / lymphoma / metastasis）すべてに当てる。
    1 回ごとの指標（当て先ごと。Δ_own = μ_E,当て先 − μ_P,当て先 = 当て先自身の Δ = 上限）:
        shift_norm     ‖Δ̂‖
        cos_own        cos(Δ̂, Δ_own)
        residual       ‖μ_E,当て先 − (μ_P,当て先 + Δ̂)‖ = ‖Δ_own − Δ̂‖（transfer_correction の residual_norm）
        residual_rel   residual / ‖Δ_own‖（補正なし = 1、own = 0）
        asw_site       施設のシルエット係数（PCA centroid.pca_n_components 次元 → silhouette。harmonization._reduce と
                       silhouette_score で、transfer_correction が使う harmonization._site_values の 1 つ目と同じ値。iLISI は出さない）
        ba_E2P         最近傍重心法の balanced accuracy（EBRAINS のクラス平均で補正後の patho2 を分類。transfer_correction.nc_ba）:
                       glioma = ast / glm / odg の 3 クラス、paper = 組織型（nc.histotype_set のクラス）。他の当て先は空
        pairs_adjacent 施設 × 群の平均の Ward 樹形図で同じ群の 2 施設が隣り合う数（paper 13 組 / glioma 3 組。dendrogram.py の
                       group_means / raw_distance / ward_linkage / pair_stats。平行移動は patho2 の群平均に足すだけなので群平均は
                       1 回だけ作る = shift_estimation.Evaluator.adjacent_pairs と同じ）
    参照の行（scheme）: none（Δ̂ = 0）/ full（推定元の全スライド）/ own（当て先自身の Δ）。
    「全枚数の転移の値の 10% 以内」に入る n（summary.md）: その n 以上の格子のすべてで中央値が条件を満たす最小の n。条件は 2 通り:
        value … |中央値 − full| ≤ 0.1·|full|（文字どおり。ASW は full が 0 に近いので厳しすぎる）
        gap   … 改善の 90% に届く: (中央値 − full) / (none − full) ≤ 0.1（residual・ASW は小さいほど良い。BA は向きを揃える）
    residual には「上側 2.5% 点も value の条件を満たす」n も出す。

B. 推定元に要る腫瘍の種類（k の曲線）
    当て先ごとに、当て先以外の科（paper → glioma / lymphoma / metastasis の 3、他 → 4）から k 科の部分集合を全部とり
        mean    Δ̂_k = 科ごとの Δ_f = μ_E,f − μ_P,f（科の全スライド）の等しい重みの平均（transfer_correction の others_mean を k 科で）
        pooled  k 科のスライドを全部まとめた μ_E − μ_P（transfer_correction の others を k 科で）
    を当てて A と同じ指標を出す。k = 1 では両者は同じ。部分集合ごとの値（k_curve.csv）と、k ごとの中央値・最小・最大（k_curve_summary.csv）。

受け入れ確認（out/transfer/correction_{encoder}.csv があれば、method = centroid の行と突き合わせる。表は summary.md）:
    A の none / own / full と、B の k = 1（単独の科）・paper（髄膜腫 + SFT の pooled）・k 最大（others_mean / others）の
    residual_norm・asw_site・BA・pairs_adjacent。

出力 (output_dir/transfer_sample_size/。encoder が titan 以外ならその下の <encoder>/):
    n_curve.csv            A の 1 回ごとの値（当て先ごとに 1 行）と参照の行
    n_curve_summary.csv    A の (推定元, 抜き方, 当て先, n) ごとの中央値と 2.5 / 97.5 パーセンタイル（参照の行はその値）
    k_curve.csv            B の部分集合ごとの値（none / own の参照の行も）
    k_curve_summary.csv    B の (当て先, 推定量, k) ごとの中央値・最小・最大
    summary.md             受け入れ確認・n の表・10% に入る n・k の表・読み
    transfer_sample_size.png
        a 残差の比 vs n（推定元 paper、当て先ごと、95% 帯）、b 施設 ASW vs n、c グリオーマ 3 群の BA vs n、
        d 残差の比 vs k（科ごとの平均 vs まとめて）、e・f 単独の科 → paper の残差の比・施設 ASW vs n

Usage:
    uv run python transfer_sample_size.py [--encoder titan|prism] [--n-draws 200] [--asw-draws N] [--n-jobs 12]
                                          [--sources paper glioma lymphoma metastasis] [--summary-only]
    --asw-draws N: ASW（1 回 0.03〜0.3 秒）を各設定の最初の N 回だけ計算する（残りは空。既定は全部）。
    --summary-only: 計算せず既存の CSV から summary.md と png だけ作り直す。
"""
from __future__ import annotations

import argparse
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from dendrogram import group_means, pair_stats, raw_distance, ward_linkage  # noqa: E402
from harmonization import _reduce  # noqa: E402
from histotype_labels import apply_set  # noqa: E402
from transfer_correction import (FAMILIES, GLIOMA_GROUPS, NAMED_SOURCES, TARGET_FAMILIES, nc_ba,  # noqa: E402
                                 target_labels)
from utils.loader import load_config  # noqa: E402

SEED = 42
N_DRAWS = 200
N_LIST = [5, 10, 20, 50, 100, 200]               # 推定元の patho2 の枚数未満だけ使い、全枚数（full）を足す
N_JOBS = 12
SOURCES = ["paper", "glioma", "lymphoma", "metastasis"]          # 既定の n 曲線の推定元（--sources で変える）
SOURCE_INDEX = {s: i for i, s in enumerate(["paper", "meningioma", "SFT", "glioma", "lymphoma", "metastasis"])}  # 乱数列の番号
TARGETS = ["paper", "glioma", "lymphoma", "metastasis"]
VARIANTS = ["patho2", "both"]
METRICS = ["shift_norm", "cos_own", "residual", "residual_rel", "asw_site", "ba_E2P", "pairs_adjacent"]
BA_TASK = {"paper": "histotype", "glioma": "glioma3"}
TC_BA = {"paper": "nc_histotype_ba_E2P", "glioma": "nc_glioma3_ba_E2P"}   # transfer_correction の列名
FAMILY_COLOR = {"meningioma": "#4c72b0", "SFT": "#000000", "glioma": "#dd8452", "lymphoma": "#55a868",
                "metastasis": "#c44e52", "paper": "#4c72b0"}       # shift_transfer.FAMILY_COLOR と同じ（paper は髄膜腫の色）
MARKER = {"paper": "o", "meningioma": "o", "SFT": "v", "glioma": "s", "lymphoma": "^", "metastasis": "D"}
INK, MUTED = "#222222", "#777777"
# summary.md の「読み」。数値を見て書いた文章（2026-10-26 の TITAN の実行、200 回）。数値や設定を変えたら見直す。
# TITAN の数値の文章なので、他の encoder の summary.md には入れない（READING_OTHER）
READING_OTHER = "（{encoder} の読みは未記入。この節の文章は TITAN の数値で書いたもので、titan の summary.md にだけ入れる）"
READING = """
- A. 論文コホート（髄膜腫 + SFT）から別の腫瘍へ（patho2 だけ n 枚、EBRAINS は全部）: 残差の比は n = 20 で全枚数の値の 10% 以内
  （グリオーマ 0.549 [0.501, 0.593] vs 全 577 枚 0.502、DLBCL 0.607 vs 0.556、転移 0.535 vs 0.496）。上側 2.5% 点まで入るのは
  グリオーマで n = 50、DLBCL・転移で n = 100。グリオーマの施設 ASW は n = 20 で 0.017 [0.008, 0.024]（補正なし 0.106、全枚数 0.009、
  own −0.035）で改善の 90% に届く。全枚数の値の 10% 以内（≤ 0.0098）に入るのは n = 200（0.0093）で、0 に近い値の 10% は厳しすぎる。
  グリオーマ 3 群の BA は n = 5 で既に 0.769（補正なし 0.740、全枚数 0.781、own 0.789）。117 枚・3 クラスなので帯は粗い（0.73–0.81）。
- 枚数を増やしても残差の比は 0.50 前後で止まる。足りないのは枚数ではなく向き（全枚数でも cos(Δ̂, Δ_own) = 0.87、腫瘍をまたぐ偏り）で、
  n = 20 のノイズが上乗せするのは残差の比で +0.05 ほど。Fig S4（同じ腫瘍の中、n = 20 で全例並み）と同じ「新施設から 20 枚ほど」が
  腫瘍をまたいでも成り立ち、その先は偏りが支配する。
- 両施設から n 枚ずつ（EBRAINS 側も減らす）では要る枚数が約 2.5 倍: グリオーマの残差の 10% 以内は n = 50（上側 2.5% 点は 200）、
  ASW の 90% も n = 50。公開データを全部使える運用（Fig S4 と同じ）なら新施設側の 20 枚で足りる。
- 単独の科から論文コホートへ（何枚のグリオーマで髄膜腫が直るか）: グリオーマ 20 枚（EBRAINS は 807 枚全部）で残差の比
  0.699 [0.636, 0.818]、施設 ASW 0.074（補正なし 0.134、グリオーマ全 117 枚 0.063、own 0.009）、隣接ペア 8 [5, 8]（補正なし 4、own 11）。
  全枚数の 10% 以内は 50 枚（残差 3.58 vs 3.47、ASW 0.067 vs 0.063）。DLBCL は 20 枚で 0.660（全 59 枚 0.615）。
  転移は少ない枚数で効かない: 5 枚では中央値 1.001（補正なしと同じ）、20 枚で 0.766、全 84 枚で 0.693。転移は原発が雑多で科の中の
  ばらつきが大きく、少数の抽出では平均が定まらない。
- B. 科ごとに Δ_f を求めて平均すると、種類を増やすほど良くなる（残差の比の中央値: paper 0.635 → 0.544 → 0.525、glioma
  0.494 → 0.467 → 0.437 → 0.408、DLBCL 0.624 → 0.586 → 0.560 → 0.535、転移 0.523 → 0.455 → 0.418 → 0.414）。部分集合による
  ばらつきも狭まる（glioma: k = 1 で 0.46–0.76 → k = 3 で 0.37–0.47）。まとめて（pooled）推定すると、構成が施設で偏った
  グリオーマ（EBRAINS 807 / patho2 117）を含む組で崩れる: paper 当て先で k = 3 の 0.974（ASW 0.127、補正なしとほぼ同じ）、
  glioma + 転移の 2 科で 1.041 と補正なしより悪い。glioma が当て先のときは推定元にグリオーマが入らないので pooled も崩れない
  （0.494 → 0.456）。→ 「腫瘍の種類ごとに Δ を求めて平均する」の利点は種類を増やしたときに出る（k = 1 では同じ）。
- 想定外: 等しい重みの平均に SFT（patho2 22 / EBRAINS 32 枚）を入れると悪くなる（glioma 当て先: 髄膜腫だけ 0.501 → + SFT 0.561、
  DLBCL 当て先: 0.549 → 0.647）。SFT 単独の Δ は当て先の Δ と向きが合わない（cos 0.63–0.76、残差の比 0.66–0.89）。n ≈ 20–30 の
  ノイズだけなら A の both の n = 20 程度（残差の比 +0.05–0.10）なので、SFT の Δ そのものが他と違う（shift_transfer の SFT 0.62–0.83 と同じ）。
  最良の部分集合は全部の科とは限らない（glioma 当て先: DLBCL + 転移 0.364 < 全 4 科 0.408）。科の重みの付け方（例数による重み、
  小さい科を外す）は未検討。
- まとめ（控えめに）: 同じ腫瘍が公開側に無くても、別の腫瘍 2–3 種類で各 20〜50 枚あれば、科ごとに Δ を求めて平均した平行移動で
  当て先の施設差の半分前後が取れる（残差の比 0.4–0.55）。腫瘍をまたぐ限り、枚数や種類を増やしても残差の比（中央値・部分集合の値）は 0.36 を下回らなかった。
"""


# ── data ──────────────────────────────────────────────────────────────────────

def load_embeddings(cfg: dict, encoder: str) -> tuple[pd.DataFrame, np.ndarray]:
    """manifest の全スライドの埋め込み（manifest の並び、float64）。transfer_correction のキャッシュを読むだけ（書かない）。"""
    tcfg = cfg["transfer"]
    tdir = Path(cfg["output_dir"]) / "transfer"
    man = pd.read_csv(Path(cfg["output_dir"]) / tcfg.get("manifest", "transfer/manifest.csv"),
                      dtype={"slide_id": str, "patient": str})
    npy, order_csv = tdir / f"embeddings_{encoder}.npy", tdir / f"embeddings_{encoder}_slides.csv"
    ident = man[["site", "slide_id"]].reset_index(drop=True)
    if npy.exists() and order_csv.exists():
        cached = pd.read_csv(order_csv, dtype={"slide_id": str})
        X = np.load(npy)
        if cached.equals(ident) and len(X) == len(man):
            print(f"  cache: {npy}  {X.shape}（並びは manifest と一致）")
            return man, X.astype(np.float64)
        print("  cache の並びが manifest と合わないので h5 から読む（キャッシュは書かない）")
    h5_col, key = {"titan": ("h5_titan", tcfg["titan_key"]), "prism": ("h5_prism", tcfg["prism_key"])}[encoder]
    print(f"  reading {len(man)} h5 ({key}) ...")
    X = []
    for p in man[h5_col]:
        with h5py.File(p, "r") as f:
            X.append(np.asarray(f[key][:], dtype=np.float32).reshape(-1))
    return man, np.stack(X).astype(np.float64)


# ── 当て先（Δ を与えると指標を返す）────────────────────────────────────────────

class Target:
    """当て先 1 つ。patho2 だけを δ だけ平行移動したときの指標。群平均などの前処理は 1 回だけ。"""

    def __init__(self, name: str, X: np.ndarray, man: pd.DataFrame, ref: str, cfg: dict, pca_n: int,
                 histotype_classes: list[str]) -> None:
        tm = np.isin(man["family"].to_numpy(), list(TARGET_FAMILIES[name]))
        self.name, self.ref, self.pca_n = name, ref, pca_n
        self.X, self.st = X[tm], man["site"].to_numpy()[tm]          # transfer_correction と同じ並び（manifest の順）
        self.lab = target_labels(man[tm], name)
        self.oth = next(s for s in pd.unique(self.st) if s != ref)
        self.p = self.st != ref
        self.n_e, self.n_p = int((~self.p).sum()), int(self.p.sum())
        self.d_own = self.X[~self.p].mean(0) - self.X[self.p].mean(0)
        self.d_own_norm = float(np.linalg.norm(self.d_own))
        self.ba_classes = {"glioma": GLIOMA_GROUPS, "paper": histotype_classes}.get(name)
        self.ba_mask = np.isin(self.lab, self.ba_classes) if self.ba_classes else None
        self.groups = None
        if name in ("paper", "glioma"):
            g = group_means(pd.DataFrame({"source": self.st, "subtype": self.lab, "embedding": list(self.X)}), cfg)
            self.groups, self.group_p = g, (g["source"] != ref).to_numpy()
            self.group_mu = np.stack(g["mean"].values)

    def evaluate(self, delta: np.ndarray, asw: bool = True) -> dict[str, float]:
        Y = self.X.copy()
        Y[self.p] += delta
        r = self.d_own - delta
        nd = float(np.linalg.norm(delta))
        out = {"shift_norm": nd,
               "cos_own": float(delta @ self.d_own / (nd * self.d_own_norm)) if nd > 0 else np.nan,
               "residual": float(np.linalg.norm(r)), "residual_rel": float(np.linalg.norm(r)) / self.d_own_norm,
               "asw_site": float(silhouette_score(_reduce(Y, self.pca_n), self.st)) if asw else np.nan,
               "ba_E2P": np.nan, "pairs_adjacent": np.nan}
        if self.ba_classes:
            m = self.ba_mask
            out["ba_E2P"] = nc_ba(Y[m], self.st[m], self.lab[m], self.ba_classes, self.ref, self.oth)
        if self.groups is not None:
            mu = self.group_mu + np.where(self.group_p[:, None], delta[None, :], 0.0)
            g = self.groups.assign(mean=list(mu))
            D = raw_distance(g)
            out["pairs_adjacent"] = int(pair_stats(D, ward_linkage(D), g)["adjacent_pairs"])
        return out


def _eval_chunk(tg: Target, deltas: np.ndarray, asw: np.ndarray) -> list[dict]:
    return [tg.evaluate(d, bool(a)) for d, a in zip(deltas, asw)]


def evaluate_all(tg: Target, deltas: np.ndarray, asw: np.ndarray, n_jobs: int) -> list[dict]:
    """δ の行列を並列に評価（ASW が重い: PCA 50 ＋ シルエット。worker ごとにまとめて渡す）。"""
    if n_jobs == 1 or len(deltas) < 8:
        return _eval_chunk(tg, deltas, asw)
    idx = np.array_split(np.arange(len(deltas)), min(len(deltas), n_jobs * 4))
    out = Parallel(n_jobs=n_jobs)(delayed(_eval_chunk)(tg, deltas[i], asw[i]) for i in idx)
    return [r for part in out for r in part]


def run_entries(entries: list[dict], targets: dict[str, Target], n_jobs: int, label: str) -> pd.DataFrame:
    """entries（meta ＋ delta ＋ 当て先のリスト）を当て先ごとにまとめて評価し、縦長の表にする。"""
    rows = []
    for t, tg in targets.items():
        sel = [e for e in entries if t in e["targets"]]
        if not sel:
            continue
        t0 = time.time()
        deltas = np.stack([e["delta"] if not callable(e["delta"]) else e["delta"](tg) for e in sel])
        res = evaluate_all(tg, deltas, np.array([e.get("asw", True) for e in sel]), n_jobs)
        for e, r in zip(sel, res):
            meta = {k: v for k, v in e.items() if k not in ("delta", "targets", "asw")}
            rows.append({**meta, "target": t, "n_ebrains": tg.n_e, "n_patho2": tg.n_p, "ba_task": BA_TASK.get(t, ""), **r})
        print(f"  [{label} → {t}] {len(sel)} evaluations ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


# ── A. n の曲線 ───────────────────────────────────────────────────────────────

def n_curve_entries(X: np.ndarray, fam: np.ndarray, site: np.ndarray, ref: str, sources: list[str],
                    n_draws: int, asw_draws: int) -> list[dict]:
    entries = []
    for source in sources:
        sf = NAMED_SOURCES[source]
        sm = np.isin(fam, list(sf))
        e_idx, p_idx = np.flatnonzero(sm & (site == ref)), np.flatnonzero(sm & (site != ref))
        mu_e = X[e_idx].mean(0)
        tgts = [t for t in TARGETS if not (TARGET_FAMILIES[t] & sf)]
        base = {"source": source, "n_src_ebrains_all": len(e_idx), "n_src_patho2_all": len(p_idx), "targets": tgts}
        # 参照の行: none / own（当て先ごとに δ が違う）/ full
        entries.append({**base, "variant": "", "scheme": "none", "n": 0, "n_src_ebrains": 0, "draw": 0,
                        "delta": np.zeros(X.shape[1])})
        entries.append({**base, "variant": "", "scheme": "own", "n": np.nan, "n_src_ebrains": np.nan, "draw": 0,
                        "delta": lambda tg: tg.d_own})
        entries.append({**base, "variant": "", "scheme": "full", "n": len(p_idx), "n_src_ebrains": len(e_idx), "draw": 0,
                        "delta": mu_e - X[p_idx].mean(0)})
        for vi, variant in enumerate(VARIANTS):
            for n in [n for n in N_LIST if n < len(p_idx)]:
                rng = np.random.default_rng([SEED, SOURCE_INDEX[source], vi, n])
                for d in range(n_draws):
                    S = rng.choice(p_idx, n, replace=False)
                    if variant == "both" and n < len(e_idx):
                        me, ne = X[rng.choice(e_idx, n, replace=False)].mean(0), n
                    else:
                        me, ne = mu_e, len(e_idx)
                    entries.append({**base, "variant": variant, "scheme": "draw", "n": n, "n_src_ebrains": ne, "draw": d,
                                    "delta": me - X[S].mean(0), "asw": d < asw_draws})
    return entries


def _q(v: np.ndarray, q: float) -> float:
    v = v[~np.isnan(v)]
    return float(np.quantile(v, q)) if len(v) else np.nan


def n_curve_summary(nc: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "variant", "scheme", "target", "n"]
    rows = []
    for k, d in nc.groupby(keys, dropna=False, sort=False):
        out = dict(zip(keys, k))
        out.update({"n_src_ebrains": d["n_src_ebrains"].iloc[0], "n_draws": len(d),
                    "n_draws_asw": int(d["asw_site"].notna().sum())})
        for m in METRICS:
            v = d[m].to_numpy(dtype=np.float64)
            out[f"{m}_median"], out[f"{m}_q025"], out[f"{m}_q975"] = _q(v, 0.5), _q(v, 0.025), _q(v, 0.975)
        rows.append(out)
    return pd.DataFrame(rows)


def curve(ncs: pd.DataFrame, source: str, variant: str, target: str) -> pd.DataFrame:
    """抽出の行（n 昇順）＋ 全枚数（full）の行。"""
    d = ncs[(ncs["source"] == source) & (ncs["variant"] == variant) & (ncs["scheme"] == "draw") & (ncs["target"] == target)]
    f = ncs[(ncs["source"] == source) & (ncs["scheme"] == "full") & (ncs["target"] == target)]
    return pd.concat([d.sort_values("n"), f])


def ref_value(ncs: pd.DataFrame, source: str, target: str, scheme: str, metric: str) -> float:
    r = ncs[(ncs["source"] == source) & (ncs["scheme"] == scheme) & (ncs["target"] == target)]
    return float(r[f"{metric}_median"].iloc[0]) if len(r) else np.nan


def reach_n(c: pd.DataFrame, metric: str, full: float, none: float, rule: str, stat: str = "median",
            higher_better: bool = False) -> str:
    """その n 以上の格子のすべてで条件を満たす最小の n（docstring の value / gap）。"""
    v = c[f"{metric}_{stat}"].to_numpy(dtype=np.float64)
    ns = c["n"].to_numpy()
    if rule == "value":
        ok = np.abs(v - full) <= 0.1 * abs(full)
    else:
        gap = (full - none) if higher_better else (none - full)
        if not np.isfinite(gap) or gap <= 0:
            return "—"
        ok = ((full - v) if higher_better else (v - full)) <= 0.1 * gap
    ok = ok & np.isfinite(v)
    first = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        first = i
    if first is None:
        return "—"
    return f"{int(ns[first])}" + ("（全枚数）" if first == len(ok) - 1 else "")


# ── B. k の曲線 ───────────────────────────────────────────────────────────────

def k_curve_entries(X: np.ndarray, fam: np.ndarray, site: np.ndarray, ref: str) -> list[dict]:
    e_all, p_all = site == ref, site != ref
    d_f = {f: X[(fam == f) & e_all].mean(0) - X[(fam == f) & p_all].mean(0) for f in FAMILIES}
    entries = []
    for t in TARGETS:
        others = [f for f in FAMILIES if f not in TARGET_FAMILIES[t]]
        entries.append({"estimator": "none", "k": 0, "subset": "", "n_src_ebrains": 0, "n_src_patho2": 0,
                        "delta": np.zeros(X.shape[1]), "targets": [t]})
        entries.append({"estimator": "own", "k": np.nan, "subset": "", "n_src_ebrains": np.nan, "n_src_patho2": np.nan,
                        "delta": lambda tg: tg.d_own, "targets": [t]})
        for k in range(1, len(others) + 1):
            for sub in combinations(others, k):
                sm = np.isin(fam, sub)
                n_e, n_p = int((sm & e_all).sum()), int((sm & p_all).sum())
                meta = {"k": k, "subset": "+".join(sub), "n_src_ebrains": n_e, "n_src_patho2": n_p, "targets": [t]}
                entries.append({**meta, "estimator": "mean", "delta": np.mean([d_f[f] for f in sub], axis=0)})
                entries.append({**meta, "estimator": "pooled", "delta": X[sm & e_all].mean(0) - X[sm & p_all].mean(0)})
    return entries


def k_curve_summary(kc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (t, est, k), d in kc[kc["estimator"].isin(["mean", "pooled"])].groupby(["target", "estimator", "k"], sort=False):
        out = {"target": t, "estimator": est, "k": int(k), "n_subsets": len(d)}
        for m in METRICS:
            v = d[m].to_numpy(dtype=np.float64)
            out[f"{m}_median"], out[f"{m}_min"], out[f"{m}_max"] = _q(v, 0.5), _q(v, 0.0), _q(v, 1.0)
        rows.append(out)
    return pd.DataFrame(rows)


# ── 受け入れ確認 ──────────────────────────────────────────────────────────────

def acceptance(nc: pd.DataFrame, kc: pd.DataFrame, tc_path: Path) -> pd.DataFrame | None:
    """全枚数・参照・k の行を transfer_correction（method = centroid）の値と突き合わせる。"""
    if not tc_path.exists():
        return None
    tc = pd.read_csv(tc_path)
    tc = tc[tc["method"] == "centroid"]
    rows = []

    def add(part: str, row: pd.Series, tc_source: str) -> None:
        t = row["target"]
        for mine, theirs in (("residual", "residual_norm"), ("asw_site", "asw_site"), ("ba_E2P", TC_BA.get(t)),
                             ("pairs_adjacent", "pairs_adjacent")):
            if theirs is None:
                continue
            v = tc[(tc["source"] == tc_source) & (tc["target"] == t) & (tc["metric"] == theirs)]["value"]
            if v.empty or pd.isna(row[mine]):
                continue
            exp, got = float(v.iloc[0]), float(row[mine])
            rows.append({"part": part, "target": t, "transfer_correction source": tc_source, "metric": theirs,
                         "expected": exp, "got": got, "diff": got - exp, "ok": abs(got - exp) <= 1e-6})

    refs = nc[nc["scheme"].isin(["none", "own", "full"])]
    refs = pd.concat([refs[refs["scheme"] != "full"].drop_duplicates(["scheme", "target"]),   # none / own は推定元によらない
                      refs[refs["scheme"] == "full"].drop_duplicates(["source", "target"])])
    for _, r in refs.iterrows():
        add(f"A {r['scheme']}" + (f" ({r['source']})" if r["scheme"] == "full" else ""), r,
            r["source"] if r["scheme"] == "full" else r["scheme"])
    kmax = kc.groupby("target")["k"].max()
    for _, r in kc[kc["estimator"].isin(["mean", "pooled"])].iterrows():
        if r["k"] == 1 and r["estimator"] == "mean":
            add("B k = 1", r, r["subset"])
        elif r["estimator"] == "pooled" and r["subset"] == "meningioma+SFT":
            add("B pooled meningioma+SFT", r, "paper")
        elif r["k"] == kmax[r["target"]]:
            add(f"B k = {int(r['k'])} {r['estimator']}", r, "others_mean" if r["estimator"] == "mean" else "others")
    return pd.DataFrame(rows)


# ── summary.md ────────────────────────────────────────────────────────────────

def _f(v: float, digits: int) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def _band(r: pd.Series, m: str, digits: int) -> str:
    med, lo, hi = r[f"{m}_median"], r[f"{m}_q025"], r[f"{m}_q975"]
    if not np.isfinite(med):
        return "—"
    if r["scheme"] != "draw" or np.isclose(lo, hi):
        return _f(med, digits)
    return f"{med:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


METRIC_COLS = [("residual_rel", "残差の比", 3), ("residual", "残差", 2), ("cos_own", "cos(Δ̂, Δ_own)", 3),
               ("asw_site", "施設 ASW", 4), ("ba_E2P", "BA E→P", 3), ("pairs_adjacent", "隣接ペア", 0)]


def n_table(ncs: pd.DataFrame, source: str, variant: str, target: str, with_refs: bool = True) -> list[str]:
    cols = [c for c in METRIC_COLS if not (c[0] == "ba_E2P" and target not in BA_TASK)
            and not (c[0] == "pairs_adjacent" and target not in ("paper", "glioma"))]
    L = ["| n（patho2） | n（EBRAINS） | 回 | " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * (len(cols) + 3)]
    rows = []
    if with_refs:
        rows.append(("補正なし", ncs[(ncs["source"] == source) & (ncs["scheme"] == "none") & (ncs["target"] == target)]))
    c = curve(ncs, source, variant, target)
    for _, r in c.iterrows():
        rows.append((None, r.to_frame().T))
    if with_refs:
        rows.append(("own（当て先自身の Δ）", ncs[(ncs["source"] == source) & (ncs["scheme"] == "own") & (ncs["target"] == target)]))
    for name, d in rows:
        if d.empty:
            continue
        r = d.iloc[0]
        if name is None:
            n = f"{int(r['n'])}" + ("（全枚数）" if r["scheme"] == "full" else "")
            ne, nd = _f(float(r["n_src_ebrains"]), 0), str(int(r["n_draws"]))
            if r["scheme"] == "draw" and r["n_draws_asw"] < r["n_draws"]:
                nd += f"（ASW {int(r['n_draws_asw'])}）"
        else:
            n, ne, nd = name, "", ""
        L.append(f"| {n} | {ne} | {nd} | " + " | ".join(_band(r, m, dg) for m, _, dg in cols) + " |")
    return L + [""]


def write_summary(path: Path, ncs: pd.DataFrame, kc: pd.DataFrame, kcs: pd.DataFrame, acc: pd.DataFrame | None,
                  encoder: str) -> None:
    sources = [s for s in SOURCE_INDEX if s in set(ncs["source"])]          # paper を先に（SOURCE_INDEX の順）
    n_draws = int(ncs.loc[ncs["scheme"] == "draw", "n_draws"].max())
    n_asw = int(ncs.loc[ncs["scheme"] == "draw", "n_draws_asw"].min())
    L = [f"# 何枚・何種類の腫瘍があれば別の腫瘍に移せる Δ を推定できるか（{encoder}）", "",
         "`transfer_sample_size.py` が作る。定義はスクリプトの docstring。補正は平行移動だけ（Δ̂ を当て先の patho2 に足す、"
         "EBRAINS は不変）。残差の比 = ‖Δ_own − Δ̂‖ / ‖Δ_own‖（補正なし 1、当て先自身の Δ で 0）。"
         f"各 n で {n_draws} 回（seed {SEED}）" + (f"、ASW は最初の {n_asw} 回だけ" if n_asw < n_draws else "")
         + "。中央値 [2.5, 97.5 パーセンタイル]。", ""]
    if acc is not None and len(acc):
        L += [f"受け入れ確認: transfer_correction（method = centroid）の全枚数・none・own・k = 1・k 最大の行と {len(acc)} 項目中 "
              f"{int(acc['ok'].sum())} 項目一致（許容 1e-6" + ("" if acc["ok"].all() else "、**一致しない項目がある**") + "。表は末尾）。", ""]

    L += ["## A. 推定に要る枚数（n_curve_summary.csv）", ""]
    for source in sources:
        tg = [t for t in TARGETS if t in set(ncs.loc[ncs["source"] == source, "target"])]
        r0 = ncs[(ncs["source"] == source) & (ncs["scheme"] == "full")].iloc[0]
        L += [f"### 推定元 {source}（EBRAINS {int(r0['n_src_ebrains'])} / patho2 {int(r0['n'])} 枚）", ""]
        for t in tg:
            L += [f"#### {source} → {t}: patho2 だけ n 枚（EBRAINS は全部）", ""] + n_table(ncs, source, "patho2", t)
            L += [f"#### {source} → {t}: 両施設から n 枚ずつ", ""] + n_table(ncs, source, "both", t, with_refs=False)

    L += ["## 全枚数の転移の値の 10% 以内に入る n", "",
          "中央値が、その n 以上の格子のすべてで条件を満たす最小の n。value = |中央値 − full| ≤ 0.1·|full|、"
          "gap = 改善（補正なし → full）の 90% に届く。残差の q975 = 上側 2.5% 点も value の条件を満たす n。"
          "BA は gap だけ（value の 10% は補正なしとの差より広い）。", "",
          "| 推定元 | 抜き方 | 当て先 | full の残差 | 残差 value | 残差 gap | 残差 q975 value | full の ASW | ASW value | ASW gap | full の BA | BA gap |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for source in sources:
        for variant in VARIANTS:
            for t in [t for t in TARGETS if t in set(ncs.loc[ncs["source"] == source, "target"])]:
                c = curve(ncs, source, variant, t)
                fr, nr = (ref_value(ncs, source, t, s, "residual") for s in ("full", "none"))
                fa, na = (ref_value(ncs, source, t, s, "asw_site") for s in ("full", "none"))
                fb, nb = (ref_value(ncs, source, t, s, "ba_E2P") for s in ("full", "none"))
                cells = [source, variant, t, _f(fr, 2), reach_n(c, "residual", fr, nr, "value"),
                         reach_n(c, "residual", fr, nr, "gap"), reach_n(c, "residual", fr, nr, "value", stat="q975"),
                         _f(fa, 4), reach_n(c, "asw_site", fa, na, "value"), reach_n(c, "asw_site", fa, na, "gap")]
                cells += ([_f(fb, 3), reach_n(c, "ba_E2P", fb, nb, "gap", higher_better=True)] if t in BA_TASK else ["", ""])
                L.append("| " + " | ".join(cells) + " |")
    L.append("")

    L += ["## B. 推定元に要る腫瘍の種類（k_curve.csv / k_curve_summary.csv）", "",
          "mean = 科ごとの Δ_f を等しい重みで平均（others_mean を k 科で）、pooled = k 科のスライドをまとめた μ_E − μ_P（others を k 科で）。"
          "k = 1 では同じ。", ""]
    for t in TARGETS:
        d = kc[kc["target"] == t]
        if d.empty:
            continue
        none, own = d[d["estimator"] == "none"].iloc[0], d[d["estimator"] == "own"].iloc[0]
        has_ba = t in BA_TASK              # paper / glioma は BA と隣接ペアがある
        L += [f"### 当て先 {t}（EBRAINS {int(none['n_ebrains'])} / patho2 {int(none['n_patho2'])}）", "",
              f"補正なし: 残差 {none['residual']:.2f}（比 1）、ASW {none['asw_site']:.4f}"
              + (f"、BA {none['ba_E2P']:.3f}、隣接ペア {int(none['pairs_adjacent'])}" if has_ba else "")
              + f"。own: ASW {own['asw_site']:.4f}"
              + (f"、BA {own['ba_E2P']:.3f}、隣接ペア {int(own['pairs_adjacent'])}" if has_ba else "") + "。", "",
              "| k | 部分集合 | EBRAINS / patho2 | 残差の比 mean | pooled | cos mean | pooled | ASW mean | pooled"
              + (" | BA mean | pooled | 隣接ペア mean | pooled" if has_ba else "") + " |", "|" + "---|" * (9 + 4 * has_ba)]
        for (k, sub), g in d[d["estimator"].isin(["mean", "pooled"])].groupby(["k", "subset"], sort=False):
            m, p = g[g["estimator"] == "mean"].iloc[0], g[g["estimator"] == "pooled"].iloc[0]
            cells = [str(int(k)), sub, f"{int(m['n_src_ebrains'])} / {int(m['n_src_patho2'])}",
                     _f(m["residual_rel"], 3), _f(p["residual_rel"], 3), _f(m["cos_own"], 3), _f(p["cos_own"], 3),
                     _f(m["asw_site"], 4), _f(p["asw_site"], 4)]
            if has_ba:
                cells += [_f(m["ba_E2P"], 3), _f(p["ba_E2P"], 3), _f(m["pairs_adjacent"], 0), _f(p["pairs_adjacent"], 0)]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
        s = kcs[kcs["target"] == t]
        L += ["k ごとの中央値 [最小, 最大]（部分集合について）:", "",
              "| k | 部分集合の数 | 残差の比 mean | pooled | ASW mean | pooled" + (" | BA mean | pooled" if has_ba else "") + " |",
              "|" + "---|" * (6 + 2 * has_ba)]
        for k in sorted(s["k"].unique()):
            m, p = s[(s["k"] == k) & (s["estimator"] == "mean")].iloc[0], s[(s["k"] == k) & (s["estimator"] == "pooled")].iloc[0]

            def mm(r: pd.Series, met: str, dg: int) -> str:
                if r["n_subsets"] == 1:
                    return _f(r[f"{met}_median"], dg)
                return f"{r[f'{met}_median']:.{dg}f} [{r[f'{met}_min']:.{dg}f}, {r[f'{met}_max']:.{dg}f}]"
            cells = [str(int(k)), str(int(m["n_subsets"])), mm(m, "residual_rel", 3), mm(p, "residual_rel", 3),
                     mm(m, "asw_site", 4), mm(p, "asw_site", 4)]
            if has_ba:
                cells += [mm(m, "ba_E2P", 3), mm(p, "ba_E2P", 3)]
            L.append("| " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 読み", "", READING.strip() if encoder == "titan" else READING_OTHER.format(encoder=encoder), ""]
    if acc is not None and len(acc):
        L += ["## 受け入れ確認（transfer_correction の method = centroid の行と、許容 1e-6）", "",
              "| part | 当て先 | transfer_correction の source | 指標 | 期待 | 得た値 | 差 | |", "|---|---|---|---|---|---|---|---|"]
        for _, r in acc.iterrows():
            L.append(f"| {r['part']} | {r['target']} | {r['transfer_correction source']} | {r['metric']} | {r['expected']:.4f} | "
                     f"{r['got']:.4f} | {r['diff']:.1e} | {'OK' if r['ok'] else '**NG**'} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"  saved: {path}")


# ── 図 ────────────────────────────────────────────────────────────────────────

X_NONE, X_OWN = 2.4, 1500.0
RES_TEX = r"$\|\Delta_{own} - \hat{\Delta}\| \,/\, \|\Delta_{own}\|$"
N_TICKS = [5, 10, 20, 50, 100, 200, 577]


def _n_axis(ax: plt.Axes, own: bool = True) -> None:
    ax.set_xscale("log")
    ticks = [X_NONE] + N_TICKS + ([X_OWN] if own else [])
    ax.set_xticks(ticks)
    ax.set_xticklabels(["none"] + [str(n) for n in N_TICKS] + (["own"] if own else []), fontsize=8)
    ax.minorticks_off()
    ax.set_xlim(X_NONE / 1.35, (X_OWN if own else 577) * 1.35)
    for x in (3.4, 900.0) if own else (3.4,):
        ax.axvline(x, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlabel(r"n slides of the source used to estimate $\hat{\Delta}$ (patho2; log scale)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)


def _plot_curve(ax: plt.Axes, ncs: pd.DataFrame, source: str, target: str, metric: str, key: str) -> None:
    color, marker = FAMILY_COLOR[key], MARKER[key]
    c = curve(ncs, source, "patho2", target)
    x = c["n"].to_numpy(dtype=float)
    ax.fill_between(x, c[f"{metric}_q025"], c[f"{metric}_q975"], color=color, alpha=0.16, lw=0, zorder=1)
    ax.plot(x, c[f"{metric}_median"], color=color, lw=2, marker=marker, ms=5, zorder=3,
            markeredgecolor="white", markeredgewidth=0.8)
    cb = curve(ncs, source, "both", target)
    ax.plot(cb["n"].to_numpy(dtype=float), cb[f"{metric}_median"], color=color, lw=1.2, ls="--", zorder=2)
    for scheme, xx in (("none", X_NONE), ("own", X_OWN)):
        v = ref_value(ncs, source, target, scheme, metric)
        if np.isfinite(v):
            ax.plot([xx], [v], marker=marker, ms=6, color="white", markeredgecolor=color, markeredgewidth=1.5, zorder=4)


def plot(ncs: pd.DataFrame, kc: pd.DataFrame, kcs: pd.DataFrame, path: Path, encoder: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.0))
    fig.subplots_adjust(hspace=0.42, wspace=0.26, top=0.88, bottom=0.07, left=0.05, right=0.985)
    paper_tg = [t for t in ("glioma", "lymphoma", "metastasis") if t in set(ncs.loc[ncs["source"] == "paper", "target"])]
    panels = [(axes[0, 0], "residual_rel", "a  paper (meningioma + SFT) → other tumours\nresidual site shift  " + RES_TEX),
              (axes[0, 1], "asw_site", "b  paper → other tumours\nsite ASW on the target (PCA-50)"),
              (axes[0, 2], "ba_E2P", "c  paper → glioma\n3-class nearest-centroid BA (ast / glm / odg), EBRAINS → patho2")]
    for ax, metric, title in panels:
        for t in (paper_tg if metric != "ba_E2P" else ["glioma"]):
            _plot_curve(ax, ncs, "paper", t, metric, t)
        _n_axis(ax)
        ax.set_title(title, fontsize=9.5, loc="left")
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("relative residual (none = 1, own = 0)", fontsize=8.5)
    axes[0, 1].axhline(0, color="#999999", lw=0.8)
    axes[0, 1].set_ylabel("site ASW", fontsize=8.5)
    axes[0, 2].set_ylabel("balanced accuracy", fontsize=8.5)
    # 参照線（c: 全枚数の値）
    fb = ref_value(ncs, "paper", "glioma", "full", "ba_E2P")
    axes[0, 2].axhline(fb, color=MUTED, lw=0.8, ls=":")
    axes[0, 2].text(5, fb, f"all 577: {fb:.3f}", fontsize=7.5, color=MUTED, va="bottom")

    # d: k の曲線（科ごとの平均 = 実線・塗り、まとめて = 破線・白抜き、縦線 = 部分集合の最小〜最大）
    ax = axes[1, 0]
    for t in TARGETS:
        s = kcs[kcs["target"] == t]
        if s.empty:
            continue
        for est, ls, off, filled in (("mean", "-", -0.07, True), ("pooled", "--", 0.07, False)):
            d = s[s["estimator"] == est].sort_values("k")
            x = d["k"].to_numpy(dtype=float) + off
            ax.vlines(x, d["residual_rel_min"], d["residual_rel_max"], color=FAMILY_COLOR[t], lw=1.0, alpha=0.6)
            ax.plot(x, d["residual_rel_median"], color=FAMILY_COLOR[t], ls=ls, lw=1.8 if filled else 1.2, marker=MARKER[t], ms=6,
                    markerfacecolor=FAMILY_COLOR[t] if filled else "white", markeredgecolor=FAMILY_COLOR[t], markeredgewidth=1.3)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("k = number of non-target tumour families in the estimate", fontsize=8.5)
    ax.set_ylabel("relative residual (none = 1, own = 0)", fontsize=8.5)
    ax.set_ylim(bottom=0)
    ax.set_title(r"d  $\hat{\Delta}$ from k other tumour families (all subsets; bar = min–max)" "\n"
                 r"solid / filled: mean of per-family $\Delta_f$    dashed / open: slides pooled", fontsize=9.5, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color="#eeeeee", lw=0.8, zorder=0)
    ax.legend(handles=[Line2D([], [], color=FAMILY_COLOR[t], marker=MARKER[t], lw=1.8, label=f"target {t}") for t in TARGETS],
              fontsize=7.5, frameon=False, loc="lower left", ncol=2)

    # e・f: 単独の科 → paper
    singles = [s for s in ("glioma", "lymphoma", "metastasis") if s in set(ncs["source"])]
    for ax, metric, title in ((axes[1, 1], "residual_rel", "e  single other family → paper\nresidual site shift  " + RES_TEX),
                              (axes[1, 2], "asw_site", "f  single other family → paper\nsite ASW on the paper cohort")):
        for s in singles:
            _plot_curve(ax, ncs, s, "paper", metric, s)
        _n_axis(ax)
        ax.set_title(title, fontsize=9.5, loc="left")
        n_full = {s: int(ncs[(ncs["source"] == s) & (ncs["scheme"] == "full")]["n"].iloc[0]) for s in singles}
        ax.legend(handles=[Line2D([], [], color=FAMILY_COLOR[s], marker=MARKER[s], lw=2, label=f"source {s} (all = {n_full[s]})")
                           for s in singles], fontsize=7.5, frameon=False, loc="upper right")
    axes[1, 1].set_ylim(bottom=0)
    axes[1, 1].set_ylabel("relative residual", fontsize=8.5)
    axes[1, 2].axhline(0, color="#999999", lw=0.8)
    axes[1, 2].set_ylabel("site ASW", fontsize=8.5)

    handles = [Line2D([], [], color=FAMILY_COLOR[t], marker=MARKER[t], lw=2, label=f"target {t}") for t in paper_tg]
    handles += [Line2D([], [], color=MUTED, lw=2, label="median, n patho2 slides (EBRAINS all); band = 95% of draws"),
                Line2D([], [], color=MUTED, lw=1.2, ls="--", label="median, n slides from each site"),
                Line2D([], [], color=MUTED, marker="o", ls="", markerfacecolor="white", label=r"none (uncorrected) / own (target's own $\Delta$)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 0.955))
    n_draws = int(ncs.loc[ncs["scheme"] == "draw", "n_draws"].max())
    fig.suptitle(f"How many slides and how many tumour types does a transferable site shift need? ({encoder}, translation only, "
                 f"{n_draws} draws per n, rightmost curve point = all patho2 slides of the source)", fontsize=11.5, y=0.99)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["titan", "prism"], default="titan", help="スライドエンコーダ（transfer_correction と同じ）")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS, help="各 n の抽出回数")
    ap.add_argument("--asw-draws", type=int, default=None, help="ASW を計算する回数（各設定の最初の N 回。既定 = 全部）")
    ap.add_argument("--n-jobs", type=int, default=N_JOBS, help="並列数（ASW の計算）")
    ap.add_argument("--sources", nargs="+", default=SOURCES, choices=list(NAMED_SOURCES), help="n 曲線の推定元")
    ap.add_argument("--summary-only", action="store_true", help="計算せず既存の CSV から summary.md と png を作り直す")
    args = ap.parse_args()
    cfg = load_config()
    ref = cfg["reference"]
    out_dir = Path(cfg["output_dir"]) / "transfer_sample_size"
    if args.encoder != "titan":
        out_dir = out_dir / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    tc_path = Path(cfg["output_dir"]) / "transfer" / f"correction_{args.encoder}.csv"
    print(f"[transfer_sample_size / {args.encoder}] → {out_dir}")

    if not args.summary_only:
        man, X = load_embeddings(cfg, args.encoder)
        fam, site = man["family"].to_numpy(), man["site"].to_numpy()
        pca_n = cfg.get("centroid", {}).get("pca_n_components", 50)
        classes = apply_set(cfg, cfg.get("nc", {}).get("histotype_set", "main"))["lp_histotype"]["classes"]
        targets = {t: Target(t, X, man, ref, cfg, pca_n, classes) for t in TARGETS}
        for t, tg in targets.items():
            print(f"  target {t}: ebrains {tg.n_e} / patho2 {tg.n_p}  ‖Δ_own‖ {tg.d_own_norm:.3f}")
        asw_draws = args.asw_draws if args.asw_draws is not None else args.n_draws
        t0 = time.time()
        print(f"[A] n curve: sources {args.sources}, {args.n_draws} draws (ASW {asw_draws}), n_jobs {args.n_jobs}")
        nc = run_entries(n_curve_entries(X, fam, site, ref, args.sources, args.n_draws, asw_draws), targets, args.n_jobs, "A")
        nc = nc[["source", "variant", "scheme", "n", "n_src_ebrains", "n_src_ebrains_all", "n_src_patho2_all", "draw", "target",
                 "n_ebrains", "n_patho2", "ba_task"] + METRICS]
        nc.to_csv(out_dir / "n_curve.csv", index=False)
        n_curve_summary(nc).to_csv(out_dir / "n_curve_summary.csv", index=False)
        print(f"  saved: {out_dir}/n_curve.csv, n_curve_summary.csv  ({time.time() - t0:.0f}s, {len(nc)} rows)")
        print("[B] k curve")
        kc = run_entries(k_curve_entries(X, fam, site, ref), targets, args.n_jobs, "B")
        kc = kc[["target", "estimator", "k", "subset", "n_src_ebrains", "n_src_patho2", "n_ebrains", "n_patho2", "ba_task"] + METRICS]
        kc.to_csv(out_dir / "k_curve.csv", index=False)
        k_curve_summary(kc).to_csv(out_dir / "k_curve_summary.csv", index=False)
        print(f"  saved: {out_dir}/k_curve.csv, k_curve_summary.csv")

    nc = pd.read_csv(out_dir / "n_curve.csv", keep_default_na=False, na_values=[""])
    nc["variant"] = nc["variant"].fillna("")
    ncs = pd.read_csv(out_dir / "n_curve_summary.csv", keep_default_na=False, na_values=[""])
    ncs["variant"] = ncs["variant"].fillna("")
    kc = pd.read_csv(out_dir / "k_curve.csv", keep_default_na=False, na_values=[""])
    kc["subset"] = kc["subset"].fillna("")
    kcs = pd.read_csv(out_dir / "k_curve_summary.csv")
    acc = acceptance(nc, kc, tc_path)
    if acc is not None:
        print(f"[acceptance vs {tc_path}] {int(acc['ok'].sum())} / {len(acc)} OK")
        if not acc["ok"].all():
            print(acc[~acc["ok"]].to_string(index=False))
    write_summary(out_dir / "summary.md", ncs, kc, kcs, acc, args.encoder)
    plot(ncs, kc, kcs, out_dir / "transfer_sample_size.png", args.encoder)


if __name__ == "__main__":
    main()
