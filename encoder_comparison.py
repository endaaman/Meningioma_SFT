"""encoder_comparison.py — 同じ解析を 4 つのスライド埋め込み（CONCH の平均 / TITAN / Virchow の平均 / PRISM）で回した出力を、
1 つの表と図に並べる（v4 Fig 6「別系統のスライドエンコーダでも同じ」の材料）。

エンコーダ（列の並びは utils/display.py の ENCODER_ORDER）: パッチエンコーダごとに「パッチ埋め込みの単純平均（学習なしの集約器の対照）
→ 学習した集約器」。CONCH v1.5 → meanpool（CONCH の平均）/ titan（TITAN、論文）、Virchow v1（concat 2560 次元）→ meanpool_virchow
（Virchow の平均、transfer_meanpool.py build --source virchow）/ prism（PRISM）。

計算は軽いものだけ（幾何のブロックの先頭の「施設差の大きさ」、下記）。ほかは各解析スクリプトの出力を読んで並べるだけ（入力には書かない）。
出力が無いエンコーダの列は空欄にし、summary.md にどのファイルが無かったかを書く（落ちない）。出力が揃ってから回し直せばその列が埋まる。

読むもの（エンコーダごと。titan は既定の名前、meanpool / meanpool_virchow / prism は各スクリプトの --encoder の出力名）:
    幾何       shift_transfer.py       out/shift_transfer/{,meanpool/,meanpool_virchow/,prism/}{groups,cos_matrix,arrows,mismatch_matrix,
                                       bio_control}.csv と埋め込みのキャッシュ {paper,glioma}_embeddings.npy（次元とノルムの中央値だけ）
    施設差の大きさ transfer_manifest.py ほか  out/transfer/manifest.csv（family / site）と転移テストの埋め込みのキャッシュ
                                       out/transfer/embeddings_{encoder}.npy（manifest の並び。embeddings_{encoder}_slides.csv か _ids.csv の
                                       site / slide_id で manifest と照合し、合わなければその列は空欄）
    転移       transfer_correction.py  out/transfer/correction_{encoder}.csv
    ばらつきの比 dispersion_ratio.py   out/dispersion_ratio/{corr_matrix,corr_matrix_corrected,summary}{,_meanpool,_meanpool_virchow,_prism}.csv
    アトラス   atlas_retrieval.py      out/atlas/retrieval_{encoder}.csv と _summary.md（参照の診断数・枚数）。meanpool / meanpool_virchow は
                                       回していない（空欄）。
                                       PRISM のアトラスは PRISM を抽出した EBRAINS（転移テストの manifest の分）だけなので、参照を
                                       そろえた比較の行も出す: TITAN は retrieval_titan_on_prism.csv（TITAN のアトラスを同じ EBRAINS に絞る）、
                                       PRISM は retrieval_prism.csv のまま
    例数       transfer_sample_size.py out/transfer_sample_size/n_curve_summary.csv（titan）、<encoder>/n_curve_summary.csv
                                       か n_curve_summary_<encoder>.csv（他。meanpool / meanpool_virchow は無い）

尺度: スライド埋め込みのノルムと次元がエンコーダで違う（TITAN・CONCH の平均 768 次元、PRISM 1280 次元、Virchow の平均 2560 次元）ので、
長さ・残差は比で出す（‖Δ_g‖ / ‖Δ_paper‖、残差 / 補正なしの残差、施設差 / 群内 RMS）。絶対値は「参考」の行に残す。

表（行 = 指標、列 = エンコーダ）:
    0 参考          次元、スライド埋め込みのノルムの中央値（転移テストの 2,301 枚 = shift_transfer のキャッシュ）、‖Δ_paper‖ とその比
    1 幾何          施設差の大きさ（ここだけ計算する。転移テストの全スライド = manifest、科は family 列、施設は site 列）:
                      論文コホート = family ∈ {meningioma, SFT}。Δ_S = mean(EBRAINS の S) − mean(patho2 の S)（S = 論文コホート / 科 f）。
                      群内 RMS = patho2 の論文コホートの各スライドとその重心の距離の RMS = sqrt(mean ‖x − x̄‖²)（エンコーダの尺度の物差し）。
                      施設差の相対的な大きさ = ‖Δ_paper‖ / 群内 RMS。Δ_f の分解（5 科）: u = Δ_paper / ‖Δ_paper‖ として
                      平行成分 Δ_f·u と直交成分 sqrt(‖Δ_f‖² − (Δ_f·u)²)、どちらも群内 RMS で割る。ノイズ補正はしない（生の平均の差）。
                      髄膜腫・SFT は Δ_paper に含まれる（in-sample）。
                    群ごとの cos vs Δ_paper（ノイズ補正。括弧は生の cos と 95% ブートストラップ区間）: 髄膜腫サブタイプ（中央値 [範囲]）/
                    SFT / グリオーマ 3 群とまとめ / DLBCL / 転移。群どうしの cos（ノイズ補正）の中央値: 同じ科の組 vs 科をまたぐ組。
                    矢印の図（Δ_paper × 群間変動の PC1 の平面）の 12 群の角度・長さの比（平面内 length_2d / ‖Δ_paper‖）、
                    全次元の ‖Δ_g‖ / ‖Δ_paper‖（ノイズ補正）と角度（ノイズ補正）、髄膜腫–グリオーマの全次元の角度
                    （Glioma (all) の cos_corrected_vs_meni）、食い違い行列の観測 − 予想（非対角 20 マス）、施設内の科の差との |cos| の最大
    2 転移          残差の比（補正後 / 補正なし）と施設 ASW: セル（推定元 → 当て先）× 3 法（平行移動 / 相似変換 / ComBat）。
                    平行移動の残差が 3 法で最小のセルの数（当て先 paper / glioma / lymphoma / metastasis × none・own 以外の推定元。
                    当て先 all は推定元が一部 in-sample なので除く。厳密な最小と、1% 以内の同点を含む数）、相似変換の倍率 s
                    （単独の科の範囲と論文コホート）、others（まとめ）→ others_mean（科ごと平均）の平行移動の残差の比、
                    髄膜腫ベンチ（当て先 paper: 隣接ペア /13、patho2 の SFT−髄膜腫距離 / EBRAINS の同じ距離）、残差の絶対値（参考）
    3 ばらつきの比  科をまたぐ次元ごとの分散比の相関（10 組の中央値 [範囲]）、分割半分の再現性（髄膜腫 / グリオーマ / 転移）、
                    再現性で割った相関（髄膜腫–グリオーマと、3 科の 3 組の範囲）、EBRAINS が広い次元の割合（5 科の範囲）
    4 アトラス      最近傍重心法（nc_euclid）の診断名の top-1、科ごとと全体: 補正なし → Δ_paper → 自分の科の Δ（括弧は EBRAINS LOO）。
                    4a は各エンコーダのアトラス（参照の診断数が違う）、4b は参照をそろえた比較
    5 例数          推定元 論文コホート（patho2 だけ n 枚、EBRAINS は全部）→ 当て先グリオーマ: 残差の中央値がその n 以上の格子の
                    すべてで全枚数の値の 10% 以内に入る最小の n（transfer_sample_size.py の reach_n の value と同じ規則）、
                    残差の比の床（全枚数）、n = 20 の残差の比

出力 (output_dir/encoder_comparison/):
    comparison.csv          1 行 = 表の 1 行。列: block, metric, cell, method, note、エンコーダごとの表示（meanpool / titan /
                            meanpool_virchow / prism）と
                            主の数値（{encoder}_value: 1 つの値はその値、平均 ± SD は平均、中央値 [範囲] は中央値、範囲は最小、
                            「a → b → c」は b、「k/N」は k / N）
    summary.md              入力ファイルの有無と更新時刻、ブロックごとの表（転移はセル × 法をエンコーダの内側に）、平行移動が最小でないセル
    encoder_comparison.png  a 群ごとの cos vs Δ_paper（生の cos と 95% 区間、白抜き = ノイズ補正）/ b 転移の残差の比（セル × 3 法。
                            右下にエンコーダごとの ‖Δ_paper‖ / 群内 RMS）/ c ばらつきの比: 科をまたぐ相関 vs 2 科の再現性の
                            幾何平均（対角線 = 完全に共有）/ d アトラスの top-1（出力のあるエンコーダだけ）

Usage:
    uv run python encoder_comparison.py
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils.display import (CONDITION_LABELS, CONDITION_LABELS_JA, ENCODER_COLORS, ENCODER_LABELS,  # noqa: E402
                           ENCODER_LABELS_JA, ENCODER_MARKERS, ENCODER_ORDER, FAMILY_LABELS, FAMILY_LABELS_JA)
from utils.loader import load_config  # noqa: E402

FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
METHODS = ["centroid", "affine_free", "combat"]
TRANSFER_CELLS = [("paper", "glioma"), ("glioma", "paper"), ("metastasis", "paper"), ("paper", "lymphoma"), ("own", "paper")]
MIN_TARGETS = ["paper", "glioma", "lymphoma", "metastasis"]    # 「平行移動の残差が最小」を数える当て先
TIE_TOL = 0.01                                                 # 「1% 以内の同点を含む」の幅（相対）
BENCH_SOURCES = ["none", "own", "glioma", "metastasis", "others_mean"]
RELIABLE = ["meningioma", "glioma", "metastasis"]              # 分割半分の再現性が 0.4 前後以上の科（dispersion_ratio.py の注記）
ATLAS_ROWS = FAMILIES + ["overall"]
ATLAS_CHAIN = ["none", "paper", "family"]
SS = dict(source="paper", variant="patho2", target="glioma", n_mid=20)
PAPER_FAMILIES = ("meningioma", "SFT")                         # 論文コホート（施設差の大きさの Δ_paper と群内 RMS）
MANIFEST = "transfer/manifest.csv"                              # config の transfer.manifest の既定（transfer_correction.py と同じ）
SRC_JA = {"paper": "論文コホート", "own": "own", "none": "補正なし", "others": "others（まとめ）",
          "others_mean": "others_mean（科ごと平均）", **FAMILY_LABELS_JA}
SRC_EN = {"paper": "Paper cohort", "own": "own", "others": "others", "others_mean": "others (per-family mean)", **FAMILY_LABELS}
ATLAS_JA = {**FAMILY_LABELS_JA, "overall": "全体"}
ATLAS_EN = {**FAMILY_LABELS, "overall": "Overall"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

Val = tuple[float, str]


# ── 入力 ──────────────────────────────────────────────────────────────────────

def input_paths(out: Path, enc: str, manifest: str = MANIFEST) -> dict[str, list[Path]]:
    """名前 → 候補のパス（先に見つかったものを使う）。manifest は output_dir からの相対（config の transfer.manifest）。"""
    st = out / "shift_transfer" if enc == "titan" else out / "shift_transfer" / enc
    sfx = "" if enc == "titan" else f"_{enc}"
    dr, at, ss = out / "dispersion_ratio", out / "atlas", out / "transfer_sample_size"
    same = "retrieval_prism" if enc == "prism" else f"retrieval_{enc}_on_prism"
    return {
        "groups": [st / "groups.csv"], "cos_matrix": [st / "cos_matrix.csv"], "arrows": [st / "arrows.csv"],
        "mismatch": [st / "mismatch_matrix.csv"], "bio": [st / "bio_control.csv"],
        "emb_paper": [st / "paper_embeddings.npy"], "emb_glioma": [st / "glioma_embeddings.npy"],
        "manifest": [out / manifest], "emb_all": [out / "transfer" / f"embeddings_{enc}.npy"],
        "emb_ids": [out / "transfer" / f"embeddings_{enc}_slides.csv", out / "transfer" / f"embeddings_{enc}_ids.csv"],
        "transfer": [out / "transfer" / f"correction_{enc}.csv"],
        "disp_corr": [dr / f"corr_matrix{sfx}.csv"], "disp_corr_c": [dr / f"corr_matrix_corrected{sfx}.csv"],
        "disp_summary": [dr / f"summary{sfx}.csv"],
        "atlas": [at / f"retrieval_{enc}.csv"], "atlas_md": [at / f"retrieval_{enc}_summary.md"],
        "atlas_same": [at / f"{same}.csv"], "atlas_same_md": [at / f"{same}_summary.md"],
        "sample_size": [ss / "n_curve_summary.csv"] if enc == "titan" else
                       [ss / enc / "n_curve_summary.csv", ss / f"n_curve_summary_{enc}.csv"],
    }


def _read(path: Path, name: str):
    if name in ("manifest", "emb_ids"):
        return pd.read_csv(path, dtype={"slide_id": str, "patient": str})   # transfer_correction / transfer_meanpool と同じ読み方
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".md":
        return path.read_text()
    if path.name.startswith("corr_matrix"):
        return pd.read_csv(path, index_col=0)
    if path.name == "n_curve_summary.csv" or path.name.startswith("n_curve_summary_"):
        return pd.read_csv(path, keep_default_na=False, na_values=[""])   # transfer_sample_size.py と同じ読み方
    return pd.read_csv(path)


class Inputs:
    """エンコーダ 1 つの入力。無い・読めないファイルは None（理由は status）。"""

    def __init__(self, out: Path, enc: str, manifest: str = MANIFEST):
        self.enc = enc
        self.data: dict[str, object] = {}
        self.status: dict[str, str] = {}
        for name, cands in input_paths(out, enc, manifest).items():
            self.data[name] = None
            hit = next((p for p in cands if p.exists()), None)
            if hit is None:
                self.status[name] = "無し: " + " / ".join(str(p.relative_to(out)) for p in cands)
                continue
            try:
                self.data[name] = _read(hit, name)
                mtime = datetime.fromtimestamp(hit.stat().st_mtime).strftime("%m-%d %H:%M")
                self.status[name] = f"{hit.relative_to(out)}（{mtime}）"
            except Exception as err:   # 書きかけなど。落とさずに空欄にする
                self.status[name] = f"読めない: {hit.relative_to(out)}（{type(err).__name__}: {err}）"

    @property
    def missing(self) -> list[str]:
        return [n for n, v in self.data.items() if v is None]


# ── 表の値 ────────────────────────────────────────────────────────────────────

def _f(x: float, d: int) -> str:
    t = f"{x:.{d}f}"
    return f"{0:.{d}f}" if float(t) == 0 else t   # -0.000 を 0.000 に


def one(x, d: int = 3) -> Val | None:
    x = float(x)
    return (x, _f(x, d)) if np.isfinite(x) else None


def mean_sd(a, d: int = 2) -> Val:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()), f"{a.mean():.{d}f} ± {a.std(ddof=1):.{d}f}"


def med_range(a, d: int = 3) -> Val:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    m = float(np.median(a))
    return m, f"{m:.{d}f} [{a.min():.{d}f}–{a.max():.{d}f}]"


def span(vals: dict[str, float], d: int = 2, names: dict[str, str] | None = None) -> Val:
    """範囲（最小–最大）。names を渡すと両端の名前も付ける。"""
    s = pd.Series(vals, dtype=float).dropna()
    lo, hi = s.idxmin(), s.idxmax()
    txt = f"{s[lo]:.{d}f}–{s[hi]:.{d}f}"
    if names:
        txt += f"（{names.get(lo, lo)} – {names.get(hi, hi)}）"
    return float(s[lo]), txt


def chain(vals: list[float], d: int = 3, extra: str = "") -> Val:
    return float(vals[1] if len(vals) > 1 else vals[0]), " → ".join(f"{v:.{d}f}" for v in vals) + extra


def count(k: int, n: int) -> Val:
    return k / n, f"{k}/{n}"


def deg(c: float) -> float:
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


class Table:
    def __init__(self, inputs: dict[str, Inputs]):
        self.inputs = inputs
        self.rows: list[dict] = []
        self.warnings: list[str] = []

    def each(self, keys: list[str], fn: Callable) -> dict[str, Val | None]:
        """エンコーダごとに fn(*inputs[keys]) を呼ぶ。入力が無ければ None、値が作れなければ None（警告に残す）。"""
        vals = {}
        for enc in ENCODER_ORDER:
            frames = [self.inputs[enc].data[k] for k in keys]
            if any(f is None for f in frames):
                vals[enc] = None
                continue
            try:
                vals[enc] = fn(*frames)
            except (KeyError, IndexError, ValueError, TypeError, ZeroDivisionError, AttributeError) as err:
                self.warnings.append(f"{enc}: {keys} → {type(err).__name__}: {err}")
                vals[enc] = None
        return vals

    def add(self, block: str, metric: str, vals: dict[str, Val | None], cell: str = "", method: str = "", note: str = "") -> None:
        row = dict(block=block, metric=metric, cell=cell, method=method)
        for enc in ENCODER_ORDER:
            v = vals.get(enc)
            row[enc] = v[1] if v is not None else ""
        for enc in ENCODER_ORDER:
            v = vals.get(enc)
            row[f"{enc}_value"] = v[0] if v is not None else np.nan
        row["note"] = note
        self.rows.append(row)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _group(g: pd.DataFrame, name: str) -> pd.Series:
    r = g[g["group"] == name]
    if r.empty:
        raise KeyError(f"group {name!r} が無い")
    return r.iloc[0]


def _tval(tc: pd.DataFrame, method: str, source: str, target: str, metric: str) -> float:
    r = tc[(tc["method"] == method) & (tc["source"] == source) & (tc["target"] == target) & (tc["metric"] == metric)]
    if r.empty:
        raise KeyError(f"{method} / {source} → {target} / {metric} が無い")
    return float(r["value"].iloc[0])


def _ratio(tc: pd.DataFrame, method: str, source: str, target: str) -> float:
    """残差の比 = 補正後 / 補正なし（none の行は各 method に同じものがある）。"""
    return _tval(tc, method, source, target, "residual_norm") / _tval(tc, "centroid", "none", target, "residual_norm")


def translation_min(tc: pd.DataFrame) -> tuple[int, int, int, list[dict]]:
    """平行移動の残差が 3 法で最小のセル（厳密 / 1% 以内の同点を含む）の数と、厳密に最小でないセル。"""
    r = tc[(tc["metric"] == "residual_norm") & tc["target"].isin(MIN_TARGETS) & ~tc["source"].isin(["none", "own"])
           & tc["method"].isin(METHODS)]
    p = r.pivot_table(index=["target", "source"], columns="method", values="value").dropna()
    if p.empty or not set(METHODS) <= set(p.columns):
        raise ValueError("転移のセルが無い")
    best_other = p[["affine_free", "combat"]].min(axis=1)
    strict = p["centroid"] < best_other
    tie = p["centroid"] <= best_other * (1 + TIE_TOL)
    exc = [dict(target=t, source=s, **{m: float(p.loc[(t, s), m]) for m in METHODS},
                rel=float(p.loc[(t, s), "centroid"] / best_other[(t, s)] - 1))
           for (t, s) in p.index[~strict]]
    return int(strict.sum()), int(tie.sum()), len(p), exc


def _atlas(df: pd.DataFrame, cond: str, fam: str) -> float:
    r = df[(df["retrieval"] == "nc_euclid") & (df["metric"] == "top1") & (df["query_label"] == "(all)")
           & (df["condition"] == cond) & (df["query_family"] == fam)]
    if r.empty:
        raise KeyError(f"atlas {cond} / {fam} が無い")
    return float(r["value"].iloc[0])


def _atlas_ref(md: str) -> Val:
    m = re.search(r"残ったクラス \*\*(\d+) 診断 / (\d+) 枚\*\*", md)
    if not m:
        raise ValueError("summary.md に参照のクラス数が無い")
    return float(m.group(1)), f"{m.group(1)} 診断 / {int(m.group(2)):,} 枚"


def _ss_curve(ncs: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """推定元 SS.source・抜き方 SS.variant・当て先 SS.target の抽出の行（n 昇順）＋全枚数の行、全枚数の行、補正なしの行。"""
    sel = (ncs["source"] == SS["source"]) & (ncs["target"] == SS["target"])
    d = ncs[sel & (ncs["variant"] == SS["variant"]) & (ncs["scheme"] == "draw")].sort_values("n")
    full = ncs[sel & (ncs["scheme"] == "full")]
    none = ncs[sel & (ncs["scheme"] == "none")]
    if d.empty or full.empty or none.empty:
        raise KeyError("n の曲線の行が無い")
    return pd.concat([d, full]), full.iloc[0], none.iloc[0]


def reach_n(c: pd.DataFrame, full: float) -> Val:
    """その n 以上の格子のすべてで |残差の中央値 − full| ≤ 0.1·|full| になる最小の n（transfer_sample_size.reach_n の value）。"""
    v = c["residual_median"].to_numpy(dtype=np.float64)
    ns = c["n"].to_numpy()
    ok = (np.abs(v - full) <= 0.1 * abs(full)) & np.isfinite(v)
    first = None
    for i in range(len(ok) - 1, -1, -1):
        if not ok[i]:
            break
        first = i
    if first is None:
        raise ValueError("10% 以内に入る n が無い")
    n = int(ns[first])
    return float(n), f"{n}" + ("（全枚数）" if first == len(ok) - 1 else "")


def site_gap(X, ids: pd.DataFrame, man: pd.DataFrame, ref: str) -> dict:
    """施設差の大きさ（docstring の 1 幾何の先頭）。X は転移テストの埋め込みのキャッシュ（manifest の並び）。

    返り値: dp_norm = ‖Δ_paper‖、rms = 群内 RMS（施設 ref でない側の論文コホート）、ratio = ‖Δ_paper‖ / rms、
    par / orth = 科ごとの Δ_f の平行成分・直交成分 / rms。並びが manifest と合わなければ ValueError（その列は空欄）。"""
    key = ["site", "slide_id"]
    if len(X) != len(man) or not ids[key].reset_index(drop=True).equals(man[key].reset_index(drop=True)):
        raise ValueError(f"埋め込みのキャッシュ（{len(X)} 行）の並びが manifest（{len(man)} 行）と合わない")
    X = np.asarray(X, dtype=np.float64)
    fam, site = man["family"].astype(str).to_numpy(), man["site"].astype(str).to_numpy()
    others = sorted(set(site) - {ref})
    if len(others) != 1:
        raise ValueError(f"施設が reference {ref!r} ともう 1 つでない: {sorted(set(site))}")
    e, p = site == ref, site == others[0]

    def delta(m: np.ndarray) -> np.ndarray:
        if not (m & e).any() or not (m & p).any():
            raise ValueError("片方の施設にスライドが無い群がある")
        return X[m & e].mean(0) - X[m & p].mean(0)
    paper = np.isin(fam, PAPER_FAMILIES)
    dp = delta(paper)
    P = X[paper & p]
    rms = float(np.sqrt(np.mean(np.sum((P - P.mean(0)) ** 2, axis=1))))
    u = dp / np.linalg.norm(dp)
    par, orth = {}, {}
    for f in FAMILIES:
        d = delta(fam == f)
        a = float(d @ u)
        par[f] = a / rms
        orth[f] = float(np.sqrt(max(float(d @ d) - a * a, 0.0))) / rms
    return dict(dp_norm=float(np.linalg.norm(dp)), rms=rms, ratio=float(np.linalg.norm(dp)) / rms, par=par, orth=orth,
                n_paper_p=int(P.shape[0]))


# ── ブロック ──────────────────────────────────────────────────────────────────

def block_reference(T: Table) -> None:
    b = "0 参考（絶対値）"
    T.add(b, "埋め込みの次元", T.each(["emb_paper"], lambda e: (float(e.shape[1]), f"{e.shape[1]}")))

    def norm_med(p, g):
        X = np.concatenate([np.asarray(p, dtype=np.float64), np.asarray(g, dtype=np.float64)])
        m = float(np.median(np.linalg.norm(X, axis=1)))
        return m, f"{m:.2f}（{len(X):,} 枚）"
    T.add(b, "スライド埋め込みのノルムの中央値", T.each(["emb_paper", "emb_glioma"], norm_med),
          note="shift_transfer のキャッシュ（論文コホート ＋ グリオーマ側 = 転移テストの全スライド）")
    T.add(b, "‖Δ_paper‖", T.each(["groups"], lambda g: one(_group(g, "Paper cohort")["shift_norm"], 3)))

    def rel(g, p, gl):
        X = np.concatenate([np.asarray(p, dtype=np.float64), np.asarray(gl, dtype=np.float64)])
        return one(_group(g, "Paper cohort")["shift_norm"] / np.median(np.linalg.norm(X, axis=1)), 3)
    T.add(b, "‖Δ_paper‖ / ノルムの中央値", T.each(["groups", "emb_paper", "emb_glioma"], rel))


def block_site_gap(T: Table, cfg: dict) -> dict[str, dict | None]:
    """幾何のブロックの先頭: 施設差の相対的な大きさと Δ_f の分解（ここだけ計算する）。"""
    b = "1 幾何"
    gap = T.each(["emb_all", "emb_ids", "manifest"], lambda X, ids, man: site_gap(X, ids, man, cfg["reference"]))
    T.add(b, "施設差の相対的な大きさ ‖Δ_paper‖ / 群内 RMS",
          {e: (g["ratio"], f"{g['ratio']:.3f}（{g['dp_norm']:.3f} / {g['rms']:.3f}）") if g else None for e, g in gap.items()},
          note="括弧は ‖Δ_paper‖ / 群内 RMS。群内 RMS = patho2 の論文コホートの sqrt(mean ‖x − x̄‖²)。"
               "out/transfer/manifest.csv と embeddings_{encoder}.npy から計算")
    for part, label, note in [("par", "Δ_f の Δ_paper 方向の成分 / 群内 RMS", "Δ_f·u（u = Δ_paper / ‖Δ_paper‖）。髄膜腫・SFT は Δ_paper に含まれる"),
                              ("orth", "Δ_f の直交成分 / 群内 RMS", "sqrt(‖Δ_f‖² − (Δ_f·u)²)")]:
        for f in FAMILIES:
            T.add(b, label, {e: one(g[part][f], 3) if g else None for e, g in gap.items()}, cell=FAMILY_LABELS_JA[f], note=note)
    # 確認: manifest から取った ‖Δ_paper‖ が shift_transfer の groups.csv の Paper cohort と同じか（同じスライド・同じ埋め込みのはず）
    for e, g in gap.items():
        grp = T.inputs[e].data["groups"]
        if not g or grp is None:
            continue
        try:
            ref_norm = float(_group(grp, "Paper cohort")["shift_norm"])
        except KeyError as err:
            T.warnings.append(f"{e}: groups.csv → {err}")
            continue
        if not np.isclose(g["dp_norm"], ref_norm, rtol=1e-5, atol=0):
            T.warnings.append(f"{e}: manifest からの ‖Δ_paper‖ {g['dp_norm']:.6f} が groups.csv の {ref_norm:.6f} と合わない")
    return gap


def block_geometry(T: Table, cfg: dict) -> None:
    b = "1 幾何"
    names = cfg["shift_transfer"].get("group_names", {})
    m = "cos vs Δ_paper（ノイズ補正。括弧は生 [95% CI]）"

    def subtypes(g):
        s = g[g["kind"] == "subtype"]
        v, txt = med_range(s["cos_corrected_vs_paper"])
        return v, f"{txt}（{len(s)} 群）"
    T.add(b, m, T.each(["groups"], subtypes), cell="髄膜腫サブタイプ（中央値 [範囲]）")

    def grp(name):
        def f(g):
            r = _group(g, name)
            return (float(r["cos_corrected_vs_paper"]),
                    f"{r['cos_corrected_vs_paper']:.3f}（{r['cos_vs_paper']:.3f} [{r['cos_lo_vs_paper']:.3f}, {r['cos_hi_vs_paper']:.3f}]）")
        return f
    for cell, name in [("SFT", "SFT"), (names.get("ast", "ast"), names.get("ast", "ast")),
                       (names.get("glm", "glm"), names.get("glm", "glm")), (names.get("odg", "odg"), names.get("odg", "odg")),
                       ("グリオーマ（3 群まとめ）", "Glioma (all)"), (names.get("lym", "lym"), names.get("lym", "lym")),
                       (names.get("met", "met"), names.get("met", "met"))]:
        T.add(b, m, T.each(["groups"], grp(name)), cell=cell)

    def pairs(same: bool):
        def f(cm):
            s = cm[cm["same_family"].astype(str) == str(same)]["cos_corrected"]
            v, txt = med_range(s)
            return v, f"{txt}（{len(s)} 組）"
        return f
    T.add(b, "群どうしの cos（ノイズ補正、中央値 [範囲]）", T.each(["cos_matrix"], pairs(True)), cell="同じ科の組")
    T.add(b, "群どうしの cos（ノイズ補正、中央値 [範囲]）", T.each(["cos_matrix"], pairs(False)), cell="科をまたぐ組")

    def arrows(f):
        def g(a):
            ref = a[a["role"] == "reference"].iloc[0]
            return f(a[a["role"] == "group"], ref)
        return g
    T.add(b, "矢印の平面内の角度（度、12 群の平均 ± SD）", T.each(["arrows"], arrows(lambda a, r: mean_sd(a["angle_2d_deg"], 1))))
    T.add(b, "矢印の平面内の長さの比 length_2d / ‖Δ_paper‖（平均 ± SD）",
          T.each(["arrows"], arrows(lambda a, r: mean_sd(a["length_2d"] / r["length_2d"], 2))))
    T.add(b, "‖Δ_g‖ / ‖Δ_paper‖（全次元、ノイズ補正、12 群の平均 ± SD）",
          T.each(["arrows"], arrows(lambda a, r: mean_sd(a["shift_norm_corrected"] / r["shift_norm_corrected"], 2))))
    T.add(b, "Δ_g と Δ_paper の全次元の角度（度、ノイズ補正、12 群の平均 ± SD）",
          T.each(["arrows"], arrows(lambda a, r: mean_sd(a["angle_corrected_vs_paper_deg"], 1))))

    def meni_glioma(g):
        r = _group(g, "Glioma (all)")
        a, a_raw = deg(r["cos_corrected_vs_meni"]), deg(r["cos_vs_meni"])
        return a, f"{a:.1f}（生 {a_raw:.1f}）"
    T.add(b, "全次元の角度 髄膜腫–グリオーマ（度、ノイズ補正。括弧は生）", T.each(["groups"], meni_glioma),
          note="Glioma (all) と Meningioma (all) の Δ の角度（groups.csv の cos_corrected_vs_meni）")

    def mism(f):
        def g(mm):
            off = mm[mm["matched"].astype(str) != "True"]
            return f(off["cos"] - off["cos_predicted"])
        return g
    T.add(b, "食い違い行列 観測 − 予想（非対角 20 マス、平均 ± SD）", T.each(["mismatch"], mism(lambda d: mean_sd(d, 3))))
    T.add(b, "食い違い行列 |観測 − 予想| の最大", T.each(["mismatch"], mism(lambda d: one(np.abs(d).max(), 3))))
    T.add(b, "施設内の科どうしの差と Δ_paper の |cos| の最大", T.each(["bio"], lambda bc: one(bc["cos_vs_shift_paper"].abs().max(), 3)),
          note="bio_control.csv（6 軸 × 2 施設）")


def block_transfer(T: Table) -> dict[str, list[dict]]:
    b = "2 転移"
    for metric, label in [("ratio", "残差の比（補正後 / 補正なし）"), ("asw", "施設 ASW")]:
        for src, tgt in TRANSFER_CELLS:
            cell = f"{SRC_JA[src]} → {SRC_JA[tgt]}" if src != "own" else f"own（{SRC_JA[tgt]}）"
            for meth in METHODS:
                if metric == "ratio":
                    f = (lambda s, t, mm: lambda tc: one(_ratio(tc, mm, s, t), 3))(src, tgt, meth)
                else:
                    f = (lambda s, t, mm: lambda tc: one(_tval(tc, mm, s, t, "asw_site"), 3))(src, tgt, meth)
                T.add(b, label, T.each(["transfer"], f), cell=cell, method=CONDITION_LABELS_JA[meth])
        if metric == "asw":
            for tgt in dict.fromkeys(t for _, t in TRANSFER_CELLS):
                T.add(b, label, T.each(["transfer"], (lambda t: lambda tc: one(_tval(tc, "centroid", "none", t, "asw_site"), 3))(tgt)),
                      cell=f"補正なし（当て先 {SRC_JA[tgt]}）")

    exceptions: dict[str, list[dict]] = {}

    def strict(tc):
        k, _, n, exc = translation_min(tc)
        return count(k, n)

    def tie(tc):
        _, k, n, _ = translation_min(tc)
        return count(k, n)
    T.add(b, "平行移動の残差が 3 法で最小のセル", T.each(["transfer"], strict), cell="厳密に最小",
          note="当て先 paper / glioma / lymphoma / metastasis × 推定元（none・own 以外）")
    T.add(b, "平行移動の残差が 3 法で最小のセル", T.each(["transfer"], tie), cell=f"{TIE_TOL:.0%} 以内の同点を含む")
    for enc in ENCODER_ORDER:
        tc = T.inputs[enc].data["transfer"]
        if tc is not None:
            try:
                exceptions[enc] = translation_min(tc)[3]
            except (KeyError, ValueError) as err:
                T.warnings.append(f"{enc}: translation_min → {err}")

    def scale(tc, source):
        """推定元 source で当てはめた相似変換の s（当て先によらないので最初の行）。"""
        r = tc[(tc["method"] == "affine_free") & (tc["source"] == source) & (tc["metric"] == "scale")]
        if r.empty:
            raise KeyError(f"scale / {source} が無い")
        return float(r["value"].iloc[0])
    T.add(b, "相似変換の倍率 s", T.each(["transfer"], lambda tc: span({f: scale(tc, f) for f in FAMILIES}, 2, FAMILY_LABELS_JA)),
          cell="単独の科（範囲）")
    T.add(b, "相似変換の倍率 s", T.each(["transfer"], lambda tc: one(scale(tc, "paper"), 3)), cell="論文コホート")

    for tgt in MIN_TARGETS:
        f = (lambda t: lambda tc: chain([_ratio(tc, "centroid", "others", t), _ratio(tc, "centroid", "others_mean", t)], 3))(tgt)
        T.add(b, "others → others_mean（平行移動の残差の比）", T.each(["transfer"], f), cell=f"当て先 {SRC_JA[tgt]}")

    for metric, label in [("pairs", "髄膜腫ベンチ: 隣接ペア（/13）"), ("dist", "髄膜腫ベンチ: SFT−髄膜腫距離 patho2 / EBRAINS")]:
        for src in BENCH_SOURCES:
            for meth in (["centroid"] if src == "none" else METHODS):
                if metric == "pairs":
                    f = (lambda s, mm: lambda tc: (lambda v: (v, f"{v:.0f}"))(_tval(tc, mm, s, "paper", "pairs_adjacent")))(src, meth)
                else:
                    f = (lambda s, mm: lambda tc: one(_tval(tc, mm, s, "paper", "sft_men_dist_patho2")
                                                      / _tval(tc, mm, s, "paper", "sft_men_dist_ebrains"), 3))(src, meth)
                T.add(b, label, T.each(["transfer"], f), cell=f"{SRC_JA[src]} → 論文コホート" if src not in ("none", "own") else SRC_JA[src],
                      method="" if src == "none" else CONDITION_LABELS_JA[meth])

    for src, tgt in TRANSFER_CELLS:
        cell = f"{SRC_JA[src]} → {SRC_JA[tgt]}" if src != "own" else f"own（{SRC_JA[tgt]}）"
        for meth in METHODS:
            f = (lambda s, t, mm: lambda tc: one(_tval(tc, mm, s, t, "residual_norm"), 3))(src, tgt, meth)
            T.add(b, "残差 ‖μ_E − μ_P'‖（絶対値・参考）", T.each(["transfer"], f), cell=cell, method=CONDITION_LABELS_JA[meth])
    for tgt in dict.fromkeys(t for _, t in TRANSFER_CELLS):
        T.add(b, "残差 ‖μ_E − μ_P'‖（絶対値・参考）",
              T.each(["transfer"], (lambda t: lambda tc: one(_tval(tc, "centroid", "none", t, "residual_norm"), 3))(tgt)),
              cell=f"補正なし（当て先 {SRC_JA[tgt]}）")
    return exceptions


def block_dispersion(T: Table) -> None:
    b = "3 ばらつきの比"

    def cross(c):
        v = [c.loc[a, f] for i, a in enumerate(FAMILIES) for f in FAMILIES[i + 1:]]
        return med_range(v, 3)
    T.add(b, "科をまたぐ次元ごとの分散比の相関（10 組、中央値 [範囲]）", T.each(["disp_corr"], cross))
    for f in RELIABLE:
        T.add(b, "分割半分の再現性", T.each(["disp_corr"], (lambda fam: lambda c: one(c.loc[fam, fam], 3))(f)), cell=FAMILY_LABELS_JA[f])
    T.add(b, "再現性で割った相関", T.each(["disp_corr_c"], lambda c: one(c.loc["meningioma", "glioma"], 3)), cell="髄膜腫–グリオーマ")

    def trio(c):
        v = {f"{FAMILY_LABELS_JA[a]}–{FAMILY_LABELS_JA[f]}": c.loc[a, f] for i, a in enumerate(RELIABLE) for f in RELIABLE[i + 1:]}
        return span(v, 3)
    T.add(b, "再現性で割った相関", T.each(["disp_corr_c"], trio), cell="髄膜腫・グリオーマ・転移の 3 組（範囲）")
    T.add(b, "EBRAINS が広い次元の割合", T.each(["disp_summary"],
          lambda s: span(s.set_index("family")["frac_dims_ebrains_wider"].to_dict(), 2, FAMILY_LABELS_JA)), cell="5 科（範囲）")


def block_atlas(T: Table) -> None:
    for b, key, md in [("4a アトラス（各エンコーダの参照）", "atlas", "atlas_md"),
                       ("4b アトラス（参照を PRISM のある EBRAINS にそろえる）", "atlas_same", "atlas_same_md")]:
        T.add(b, "参照（診断数 / 枚数）", T.each([md], _atlas_ref))
        for fam in ATLAS_ROWS:
            f = (lambda fm: lambda df: chain([_atlas(df, c, fm) for c in ATLAS_CHAIN], 3, f"（LOO {_atlas(df, 'ebrains_loo', fm):.3f}）"))(fam)
            T.add(b, "top-1（補正なし → Δ_paper → own、括弧は EBRAINS LOO）", T.each([key], f), cell=ATLAS_JA[fam])


def block_sample_size(T: Table) -> None:
    b = "5 例数"
    cell = f"{SRC_JA[SS['source']]} → {SRC_JA[SS['target']]}（patho2 だけ n 枚）"

    def n10(ncs):
        c, full, _ = _ss_curve(ncs)
        return reach_n(c, float(full["residual_median"]))
    T.add(b, "残差が全枚数の値の 10% 以内に入る n", T.each(["sample_size"], n10), cell=cell)
    T.add(b, "残差の比の床（全枚数）", T.each(["sample_size"], lambda ncs: one(_ss_curve(ncs)[1]["residual_rel_median"], 3)), cell=cell)

    def mid(ncs):
        c, _, _ = _ss_curve(ncs)
        r = c[(c["scheme"] == "draw") & (c["n"] == SS["n_mid"])].iloc[0]
        return float(r["residual_rel_median"]), f"{r['residual_rel_median']:.3f} [{r['residual_rel_q025']:.3f}, {r['residual_rel_q975']:.3f}]"
    T.add(b, f"n = {SS['n_mid']} の残差の比（中央値 [95%]）", T.each(["sample_size"], mid), cell=cell)


# ── summary.md ───────────────────────────────────────────────────────────────

def _md(header: list[str], rows: list[list[str]]) -> list[str]:
    L = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    L += ["| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |" for r in rows]
    return L + [""]


def write_summary(path: Path, df: pd.DataFrame, inputs: dict[str, Inputs], exceptions: dict[str, list[dict]],
                  warnings: list[str]) -> None:
    encs = ENCODER_ORDER
    L = [f"# 同じ解析を {len(encs)} つのスライド埋め込みで（{' / '.join(ENCODER_LABELS_JA[e] for e in encs)}）", "",
         "`encoder_comparison.py` が作る。各解析スクリプトの出力を読んで並べるだけ（定義はスクリプトの docstring）。"
         "ただし「1 幾何」の先頭の施設差の大きさ（‖Δ_paper‖ / 群内 RMS と Δ_f の分解）だけは転移テストの埋め込みのキャッシュから計算する。"
         "長さ・残差はエンコーダで尺度が違うので比で出し、絶対値は「参考」の行。空欄 = その出力が無い（下の表）。", "",
         "エンコーダ: CONCH の平均 = CONCH v1.5 のパッチ埋め込みの単純平均（TITAN の集約器の対照）、"
         "Virchow の平均 = Virchow v1 のパッチ特徴（concat 2560 次元）の単純平均（PRISM の集約器の対照）。", ""]
    names = list(input_paths(Path("."), "titan"))
    L += ["## 入力", ""] + _md(["入力"] + [ENCODER_LABELS_JA[e] for e in encs],
                              [[n] + [inputs[e].status[n] for e in encs] for n in names])
    for e in encs:
        miss = inputs[e].missing
        if miss:
            L.append(f"- **{ENCODER_LABELS_JA[e]} で無い・読めない入力**: " + "、".join(miss))
    L.append("")
    if warnings:
        L += ["## 警告（値が作れなかった行）", ""] + [f"- {w}" for w in warnings] + [""]

    for block, d in df.groupby("block", sort=False):
        L += [f"## {block}", ""]
        meth = d[d["method"] != ""]
        if block.startswith("2") and len(meth):
            plain = d[d["method"] == ""]
            for metric, dm in meth.groupby("metric", sort=False):
                L += [f"### {metric}", ""]
                methods = list(dict.fromkeys(dm["method"]))
                header = ["セル"] + [f"{ENCODER_LABELS_JA[e]} {m}" for e in encs for m in methods]
                rows = []
                for cell, dc in dm.groupby("cell", sort=False):
                    r = [cell]
                    for e in encs:
                        for m in methods:
                            v = dc.loc[dc["method"] == m, e]
                            r.append(v.iloc[0] if len(v) else "")
                    rows.append(r)
                # 補正なしの行（method 空）が同じ指標にあれば、表の先頭の平行移動の列に置く
                head = [[pr["cell"]] + [pr[e] if m == methods[0] else "" for e in encs for m in methods]
                        for _, pr in plain[plain["metric"] == metric].iterrows()]
                L += _md(header, head + rows)
            rest = plain[~plain["metric"].isin(set(meth["metric"]))]
            if len(rest):
                L += ["### その他", ""] + _md(["指標", "対象"] + [ENCODER_LABELS_JA[e] for e in encs] + ["注"],
                                            [[r["metric"], r["cell"]] + [r[e] for e in encs] + [r["note"]] for _, r in rest.iterrows()])
        else:
            L += _md(["指標", "対象"] + [ENCODER_LABELS_JA[e] for e in encs] + ["注"],
                     [[r["metric"], r["cell"]] + [r[e] for e in encs] + [r["note"]] for _, r in d.iterrows()])

    L += ["## 平行移動の残差が厳密に最小でないセル", "",
          f"当て先 {' / '.join(MIN_TARGETS)} × 推定元（none・own 以外）。rel = 平行移動 / 他の 2 法の小さい方 − 1"
          f"（{TIE_TOL:.0%} 以内なら「同点を含む」の数に入る）。", ""]
    for e in encs:
        if e not in exceptions:
            L.append(f"- {ENCODER_LABELS_JA[e]}: （入力なし）")
            continue
        exc = exceptions[e]
        if not exc:
            L.append(f"- {ENCODER_LABELS_JA[e]}: なし")
            continue
        L.append(f"- {ENCODER_LABELS_JA[e]}: " + "; ".join(
            f"{SRC_JA.get(x['source'], x['source'])} → {SRC_JA.get(x['target'], x['target'])}"
            f"（{' / '.join(f'{x[m]:.3f}' for m in METHODS)}、rel {x['rel']:+.1%}）" for x in exc))
    L += ["", "（残差の並びは 平行移動 / 相似変換 / ComBat）", "",
          "## 注", "",
          "- 施設差の大きさ: 論文コホート = manifest の family ∈ {meningioma, SFT}。Δ_S = mean(EBRAINS の S) − mean(patho2 の S)。"
          "群内 RMS = patho2 の論文コホートの各スライドとその重心の距離の RMS = sqrt(mean ‖x − x̄‖²)（エンコーダの尺度の物差し）。"
          "Δ_f の分解は u = Δ_paper / ‖Δ_paper‖ への平行成分 Δ_f·u と直交成分 sqrt(‖Δ_f‖² − (Δ_f·u)²) を群内 RMS で割ったもの。"
          "ノイズ補正はしない（生の平均の差。直交成分には群の例数による推定の揺らぎが乗る）。髄膜腫・SFT は Δ_paper に含まれる。",
          "- 幾何の 12 群 = 髄膜腫サブタイプ（両施設で 20 例以上）・SFT・グリオーマ 3 群・DLBCL・転移（素な群）。"
          "cos のノイズ補正は shift_transfer.py の tr V を差し引く版。論文コホートの部分集合は leave-group-out の Δ_paper と比べる。",
          "- 転移の own（論文コホート）は当て先自身で推定した補正（上限）。ComBat は own でも残差が 0 にならない（次元ごとの尺度も合わせるため）。",
          "- アトラスの PRISM は PRISM を抽出した EBRAINS（転移テストの manifest の分）だけが参照なので、全体の EBRAINS を使う TITAN（4a）"
          "とは参照の診断数が違う。エンコーダどうしは 4b（TITAN のアトラスを同じスライドに絞った retrieval_titan_on_prism）で比べる。",
          "- 例数の n は transfer_sample_size.py の summary.md の「残差 value」と同じ規則（その n 以上のすべての格子で中央値が全枚数の値の 10% 以内）。",
          ""]
    path.write_text("\n".join(L))


# ── 図 ────────────────────────────────────────────────────────────────────────

def _style(ax: plt.Axes) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _panel(ax: plt.Axes, letter: str, title: str) -> None:
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="right", va="bottom")


def _offsets(n: int, width: float = 0.56) -> np.ndarray:
    return np.linspace(-width / 2, width / 2, n) if n > 1 else np.zeros(1)


def plot(path: Path, inputs: dict[str, Inputs], gap: dict[str, dict | None]) -> None:
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    have = {e: inputs[e] for e in ENCODER_ORDER}
    off = dict(zip(ENCODER_ORDER, _offsets(len(ENCODER_ORDER))))

    # a: 群ごとの cos vs Δ_paper
    ax = axes[0, 0]
    order: list[str] = []
    fam_of: dict[str, str] = {}
    for e in ENCODER_ORDER:
        g = have[e].data["groups"]
        if g is None:
            continue
        for _, r in g[g["kind"].isin(["subtype", "sft", "glioma", "glioma_all"])].iterrows():
            if r["group"] not in order:
                order.append(r["group"])
                fam_of[r["group"]] = r["family"]
    order = ([x for x in order if fam_of[x] == "meningioma"] + [x for x in order if fam_of[x] == "SFT"]
             + [x for x in order if fam_of[x] not in ("meningioma", "SFT")])
    ypos = {name: i for i, name in enumerate(order)}
    for e in ENCODER_ORDER:
        g = have[e].data["groups"]
        if g is None:
            continue
        g = g[g["group"].isin(order)]
        y = np.array([ypos[n] for n in g["group"]]) + off[e]
        c, mk = ENCODER_COLORS[e], ENCODER_MARKERS[e]
        # 区間は点と別に引く（パーセンタイルのブートストラップは cos を下に偏らせるので、生の cos が上限の外に出る群がある。
        # Virchow の平均の SFT・微小嚢胞性。errorbar は負の幅を受け付けない）
        ax.hlines(y, g["cos_lo_vs_paper"], g["cos_hi_vs_paper"], color=c, lw=1.2, zorder=3)
        ax.plot(g["cos_vs_paper"], y, mk, ms=5, color=c, ls="none", zorder=3)
        ax.scatter(g["cos_corrected_vs_paper"], y, marker=mk, s=26, facecolors="white", edgecolors=c, linewidths=1.2, zorder=4)
    for i in range(1, len(order)):
        if fam_of[order[i]] != fam_of[order[i - 1]]:
            ax.axhline(i - 0.5, color=GRID, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([n.replace(" meningioma", "") for n in order])
    ax.invert_yaxis()
    ax.set_xlabel("cos(Δ_group, Δ_paper)   filled: raw with 95% bootstrap CI   open: noise-corrected")
    _style(ax)
    _panel(ax, "a", "Shift direction of each tumour group vs the paper-cohort shift")

    # b: 転移の残差の比
    ax = axes[0, 1]
    labels, y0 = [], 0
    centers = []
    for src, tgt in TRANSFER_CELLS:
        cell = f"{SRC_EN[src]} → {SRC_EN[tgt]}" if src != "own" else f"own ({SRC_EN[tgt]})"
        centers.append((y0 + 1, cell))
        for meth in METHODS:
            for e in ENCODER_ORDER:
                tc = have[e].data["transfer"]
                if tc is None:
                    continue
                try:
                    v = _ratio(tc, meth, src, tgt)
                except KeyError:
                    continue
                ax.scatter(v, y0 + off[e] * 0.9, marker=ENCODER_MARKERS[e], s=30, color=ENCODER_COLORS[e],
                           edgecolors="white", linewidths=0.6, zorder=3)
            labels.append(CONDITION_LABELS[meth])
            y0 += 1
        y0 += 0.6
    ticks = []
    t = 0
    for _ in TRANSFER_CELLS:
        ticks += [t, t + 1, t + 2]
        t += 3.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    for yc, cell in centers:
        ax.annotate(cell, xy=(0.005, yc - 1.8), xycoords=("axes fraction", "data"), fontsize=7.5, fontweight="bold", color=INK, va="center")
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.text(1.0, -1.6, "no correction", fontsize=7, color=MUTED, ha="center")
    ax.set_ylim(t - 0.6, -2.2)
    ax.set_xlim(-0.03, max(1.15, ax.get_xlim()[1]))
    ax.set_xlabel("residual site gap after correction / before   (‖μ_E − μ_P'‖ / ‖μ_E − μ_P‖)")
    _style(ax)
    _panel(ax, "b", "Transfer: residual ratio (source → target) × method")
    # 施設差の相対的な大きさ（‖Δ_paper‖ / 群内 RMS）を小さく併記（own の行の右の空き）。文字は墨、エンコーダは横の印で
    lines = [(e, g["ratio"]) for e, g in gap.items() if g]
    if lines:
        x0, y0, dy = 0.60, 0.03, 0.03
        for k, (e, r) in enumerate(reversed(lines)):
            yy = y0 + k * dy
            ax.scatter([x0 + 0.008], [yy + 0.007], transform=ax.transAxes, marker=ENCODER_MARKERS[e], s=20,
                       color=ENCODER_COLORS[e], edgecolors="white", linewidths=0.4, zorder=6, clip_on=False)
            ax.text(x0 + 0.025, yy, f"{ENCODER_LABELS[e]}  {r:.2f}", transform=ax.transAxes, fontsize=7, color=INK)
        ax.text(x0, y0 + len(lines) * dy + 0.004, "site gap ‖Δ_paper‖ / within-site RMS\n(RMS over the patho2 paper cohort)",
                transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")

    # c: ばらつきの比
    ax = axes[1, 0]
    for e in ENCODER_ORDER:
        c = have[e].data["disp_corr"]
        if c is None:
            continue
        xs, ys2 = [], []
        for i, a in enumerate(FAMILIES):
            for f in FAMILIES[i + 1:]:
                ra, rf = c.loc[a, a], c.loc[f, f]
                xs.append(np.sqrt(max(ra, 0) * max(rf, 0)))
                ys2.append(c.loc[a, f])
        ax.scatter(xs, ys2, marker=ENCODER_MARKERS[e], s=34, color=ENCODER_COLORS[e], edgecolors="white", linewidths=0.6, zorder=3)
    lim = 0.7
    ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, ls="--")
    ax.text(lim * 0.97, lim * 0.97, "fully shared", fontsize=7, color=MUTED, ha="right", va="bottom", rotation=0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlim(0, lim)
    ax.set_ylim(-0.1, lim)
    ax.set_xlabel("ceiling: √(split-half reliability of family A × family B)")
    ax.set_ylabel("correlation of per-dimension log variance ratio, family A vs B")
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _panel(ax, "c", "Dispersion ratio: shared across tumour families? (10 family pairs)")

    # d: アトラスの top-1
    ax = axes[1, 1]
    series = [("meanpool", "atlas", "-", ENCODER_LABELS["meanpool"]),
              ("titan", "atlas", "-", ENCODER_LABELS["titan"] + " (full EBRAINS atlas)"),
              ("titan", "atlas_same", ":", ENCODER_LABELS["titan"] + " (atlas restricted to PRISM's EBRAINS slides)"),
              ("meanpool_virchow", "atlas", "-", ENCODER_LABELS["meanpool_virchow"]),
              ("prism", "atlas", "-", ENCODER_LABELS["prism"])]
    present = [k for k, (e, key, _, _) in enumerate(series) if have[e].data[key] is not None]   # 出力のある系列だけで横に並べる
    so = dict(zip(present, _offsets(len(present), 0.66)))
    for k, (e, key, ls, _) in enumerate(series):
        df = have[e].data[key]
        if df is None:
            continue
        col, mk = ENCODER_COLORS[e], ENCODER_MARKERS[e]
        for i, fam in enumerate(ATLAS_ROWS):
            try:
                v = [_atlas(df, c, fam) for c in ATLAS_CHAIN]
                loo = _atlas(df, "ebrains_loo", fam)
            except KeyError:
                continue
            x = i + so[k]
            ax.plot([x, x], v[:2], color=col, lw=1.4, ls=ls, zorder=2)
            ax.scatter([x], [v[0]], marker=mk, s=28, facecolors="white", edgecolors=col, linewidths=1.2, zorder=3)
            ax.scatter([x], [v[1]], marker=mk, s=28, color=col, edgecolors="white", linewidths=0.6, zorder=4)
            ax.scatter([x], [v[2]], marker="_", s=90, color=col, linewidths=1.6, zorder=4)
            ax.scatter([x], [loo], marker="x", s=18, color=MUTED, linewidths=1.0, zorder=4)
    ax.set_xticks(range(len(ATLAS_ROWS)))
    ax.set_xticklabels([ATLAS_EN[f] for f in ATLAS_ROWS])
    for i in range(len(ATLAS_ROWS) - 1):
        ax.axvline(i + 0.5, color=GRID, lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1 diagnosis accuracy (nearest EBRAINS diagnosis centroid)")
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _panel(ax, "d", "Reference atlas: in-house slides → EBRAINS diagnoses")
    leg_d = [Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, label="no correction"),
             Line2D([], [], marker="o", ls="none", color=INK, label="+ Δ_paper"),
             Line2D([], [], marker="_", ls="none", color=INK, ms=10, mew=1.6, label="+ own-family Δ"),
             Line2D([], [], marker="x", ls="none", color=MUTED, label="EBRAINS leave-one-out")]
    leg_d += [Line2D([], [], color=ENCODER_COLORS[e], ls=ls, marker=ENCODER_MARKERS[e], label=lab)
              for k, (e, key, ls, lab) in enumerate(series) if k in present]
    absent = list(dict.fromkeys(ENCODER_LABELS[e] for k, (e, _, _, _) in enumerate(series) if k not in present))
    if absent:   # 出力の無いエンコーダは描かない（凡例に 1 行で書く）
        leg_d.append(Line2D([], [], ls="none", label="not run: " + ", ".join(absent)))
    ax.legend(handles=leg_d, fontsize=6.5, loc="lower left", ncol=2, frameon=False)

    handles = [Line2D([], [], marker=ENCODER_MARKERS[e], ls="none", color=ENCODER_COLORS[e], ms=6,
                      label=ENCODER_LABELS[e] + ("" if any(have[e].data[k] is not None for k in ("groups", "transfer", "disp_corr"))
                                                 else " — no output yet")) for e in ENCODER_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.035))
    fig.suptitle("Same analyses with four slide encoders (CONCH v1.5 patches → mean / TITAN, Virchow v1 patches → mean / PRISM)",
                 fontsize=11, y=1.06)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    out = Path(cfg["output_dir"])
    out_dir = out / "encoder_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = cfg.get("transfer", {}).get("manifest", MANIFEST)
    inputs = {e: Inputs(out, e, manifest) for e in ENCODER_ORDER}
    for e in ENCODER_ORDER:
        miss = inputs[e].missing
        print(f"[{e}] 入力 {len(inputs[e].data) - len(miss)}/{len(inputs[e].data)}" + (f"、無い: {', '.join(miss)}" if miss else ""))

    T = Table(inputs)
    block_reference(T)
    gap = block_site_gap(T, cfg)
    block_geometry(T, cfg)
    exceptions = block_transfer(T)
    block_dispersion(T)
    block_atlas(T)
    block_sample_size(T)
    df = T.frame()
    df.to_csv(out_dir / "comparison.csv", index=False)
    write_summary(out_dir / "summary.md", df, inputs, exceptions, T.warnings)
    plot(out_dir / "encoder_comparison.png", inputs, gap)
    for w in T.warnings:
        print("  WARN", w)
    print(f"saved: {out_dir}/comparison.csv ({len(df)} 行), summary.md, encoder_comparison.png")


if __name__ == "__main__":
    main()
