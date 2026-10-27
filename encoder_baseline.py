"""encoder_baseline.py — エンコーダの基礎精度: 4 つのスライド埋め込み（CONCH の平均 / TITAN / Virchow の平均 / PRISM）が、
この論文の分類課題を施設の中だけでどれだけ解けるか（ken 2026-10-28「PRISM はそもそも予測精度は平気なのか」）。
施設差の議論（encoder_comparison.py）の前に置く基礎の表。

評価: 論文の約束どおり訓練不要の最近傍重心法（nc.py と同じ: 学習側のクラス平均 = スライド埋め込みのユークリッド平均に
    最も近いクラス、距離は nc._dist）。施設をまたがない（EBRAINS 内・patho2 内を別々に）。
    患者単位の層化 5 分割交差検証（sklearn の StratifiedGroupKFold、shuffle、seed 42。群 = manifest の patient 列）。
    各 fold で学習側のクラス平均を作り、検証側を分類する。指標は全 fold の検証側の予測をまとめた混同行列から:
    balanced accuracy（主）と macro F1（sklearn の balanced_accuracy_score / f1_score(average="macro")）。
    参考に fold ごとの balanced accuracy の SD も出す（ブートストラップはしない）。
    クラスの患者数が MIN_PATIENTS 未満ならそのクラスを落とし、分割数は min(5, 残ったクラスの患者数の最小) に減らす（注記する）。

課題（クラス定義は transfer_correction.py の論文のベンチと同じ。下の「確認」で施設をまたぐ値を再計算して突き合わせる）:
    sft        SFT vs 髄膜腫: 髄膜腫+SFT コホート（manifest の family ∈ {meningioma, SFT}）の全スライド。group == "SFT" が SFT、
               ほかの髄膜腫サブタイプ（Atypical / Anaplastic / NOS / 希少型も含む）は全部「髄膜腫」（nc_sft と同じ）
    histotype  髄膜腫の主要 5 組織型: config の nc.histotype_set（main = Meningothelial / Fibrous / Psammomatous / Angiomatous /
               Microcystic）のスライドだけ（nc_histotype と同じ）
    glioma3    グリオーマ 3 群（manifest の family == glioma、group = ast / glm / odg。nc_glioma3 と同じ）
    family5    5 科（manifest の family = meningioma / SFT / glioma / lymphoma / metastasis、全 2,301 枚。nc_family5 と同じ）

入力（読むだけ）:
    out/transfer/manifest.csv                     slide_id, site, family, group, patient（transfer_manifest.py）
    out/transfer/embeddings_{encoder}.npy         manifest の並び。並びは embeddings_{encoder}_slides.csv か _ids.csv の
                                                  site / slide_id で manifest と照合する（encoder_comparison.site_gap と同じ形。
                                                  合わなければそのエンコーダは空欄）
    out/transfer/correction_{encoder}.csv         参考の施設をまたぐ値（method centroid、source none / own）:
                                                  sft = nc_sft_ba_*（target paper）、histotype = nc_histotype_ba_*（paper）、
                                                  glioma3 = nc_glioma3_ba_*（glioma）、family5 = nc_family5_ba_*（all）
    out/shift_transfer/paper_slides.csv           確認だけ: 髄膜腫+SFT コホートの subtype が manifest の group と同じか

施設をまたぐ値の並べ方: 行の「施設」は分類される側（評価する施設）。E2P（EBRAINS の重心で patho2 を分類）は patho2 の行、
    P2E（patho2 の重心で EBRAINS を分類）は EBRAINS の行に置く。none = 補正なし、own = 当て先の科（paper / glioma / all）で
    推定した平行移動で patho2 を補正した後（transfer_correction.py の source own、method centroid）。

確認（run.log と summary.md）: 施設をまたぐ none / own を同じクラス定義でこちらでも計算し、correction_{encoder}.csv の値との
    max |差| を出す（定義が同じなら 0）。

出力 (output_dir/encoder_baseline/):
    baseline.csv    縦長: task, site, encoder, metric, value, n_slides, n_patients, n_classes, n_folds。
                    metric = balanced_accuracy / f1_macro / balanced_accuracy_fold_sd（施設内 CV）と
                    balanced_accuracy_cross_none / balanced_accuracy_cross_own（参考、correction_{encoder}.csv から読んだ値。
                    site = 分類される側、n_* はその側のスライド、n_folds は空）
    confusion.csv   施設内 CV の混同行列（全 fold まとめ）: task, site, encoder, true, pred, n
    summary.md      表（行 = 課題 × 施設、列 = 4 エンコーダ、BA（macro F1））、施設をまたぐ参考値との並び、クラス定義と n、
                    落としたクラス・分割数の注記、確認
    baseline.png    課題 × 施設の BA を 4 エンコーダで並べた点（塗り = 施設内 CV、白抜き = 他施設の重心で分類 = 施設をまたぐ none、
                    灰の縦線 = 偶然の水準 1 / クラス数）
    run.log         標準出力の写し

Usage:
    uv run python encoder_baseline.py
"""
from __future__ import annotations

import contextlib
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from histotype_labels import apply_set  # noqa: E402
from nc import _dist  # noqa: E402
from utils.display import (ENCODER_COLORS, ENCODER_LABELS, ENCODER_LABELS_JA, ENCODER_MARKERS,  # noqa: E402
                           ENCODER_ORDER)
from utils.loader import load_config  # noqa: E402

SEED = 42
N_SPLITS = 5
MIN_PATIENTS = 2                        # これ未満の患者数のクラスは落とす（分割数は min(5, 残ったクラスの患者数の最小)）
FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
GLIOMA_GROUPS = ["ast", "glm", "odg"]
SITES = ["ebrains", "patho2"]
SITE_DISPLAY = {"ebrains": "EBRAINS", "patho2": "patho2"}          # figs/common.py の SITE_DISPLAY と同じ
TASKS = ["sft", "histotype", "glioma3", "family5"]
TASK_LABELS = {"sft": "SFT vs meningioma", "histotype": "Meningioma histotype (5 classes)",
               "glioma3": "Glioma: ast / glm / odg", "family5": "Tumour family (5 classes)"}
TASK_LABELS_JA = {"sft": "SFT vs 髄膜腫", "histotype": "髄膜腫の主要 5 組織型", "glioma3": "グリオーマ 3 群（ast / glm / odg）",
                  "family5": "5 科"}
# 施設をまたぐ参考値: 課題 → (correction_{encoder}.csv の target, metric の頭)。target の科は own の平行移動の推定元でもある
CROSS = {"sft": ("paper", "nc_sft_ba"), "histotype": ("paper", "nc_histotype_ba"),
         "glioma3": ("glioma", "nc_glioma3_ba"), "family5": ("all", "nc_family5_ba")}
TARGET_FAMILIES = {"paper": ["meningioma", "SFT"], "glioma": ["glioma"], "all": FAMILIES}
CROSS_DIR = {"E2P": ("ebrains", "patho2"), "P2E": ("patho2", "ebrains")}   # 名前 → (重心を作る施設, 分類される施設)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


# ── 入力 ──────────────────────────────────────────────────────────────────────

def load_manifest(out: Path, cfg: dict) -> pd.DataFrame:
    path = out / cfg.get("transfer", {}).get("manifest", "transfer/manifest.csv")
    return pd.read_csv(path, dtype={"slide_id": str, "patient": str})   # transfer_correction / encoder_comparison と同じ読み方


def load_embedding(out: Path, enc: str, man: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    """転移テストの埋め込みのキャッシュ（manifest の並び、float64）。並びが合わない・無いときは (None, 理由)。"""
    npy = out / "transfer" / f"embeddings_{enc}.npy"
    cands = [out / "transfer" / f"embeddings_{enc}_slides.csv", out / "transfer" / f"embeddings_{enc}_ids.csv"]
    ids_path = next((p for p in cands if p.exists()), None)
    if not npy.exists() or ids_path is None:
        return None, "無し: " + " / ".join(str(p.relative_to(out)) for p in [npy, *cands] if not p.exists())
    X = np.load(npy)
    ids = pd.read_csv(ids_path, dtype={"slide_id": str})
    key = ["site", "slide_id"]
    if len(X) != len(man) or not ids[key].reset_index(drop=True).equals(man[key].reset_index(drop=True)):
        return None, f"並びが manifest と合わない（{npy.name} {len(X)} 行 / {ids_path.name} / manifest {len(man)} 行）"
    return X.astype(np.float64), f"{npy.relative_to(out)} {X.shape} の並び = {ids_path.name} = manifest"


def task_labels(man: pd.DataFrame, task: str, histotype_classes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """課題のスライド（manifest の行のマスク）、ラベル、クラスの並び（docstring の「課題」）。"""
    fam, grp = man["family"].astype(str).to_numpy(), man["group"].astype(str).to_numpy()
    if task == "sft":
        m = np.isin(fam, ["meningioma", "SFT"])
        return m, np.where(grp == "SFT", "SFT", "meningioma"), ["meningioma", "SFT"]
    if task == "histotype":
        return (fam == "meningioma") & np.isin(grp, histotype_classes), grp, list(histotype_classes)
    if task == "glioma3":
        return fam == "glioma", grp, list(GLIOMA_GROUPS)
    if task == "family5":
        return np.ones(len(man), bool), fam, list(FAMILIES)
    raise ValueError(task)


# ── 施設内の交差検証 ──────────────────────────────────────────────────────────

def plan(y: np.ndarray, groups: np.ndarray, classes: list[str]) -> tuple[list[str], list[str], int]:
    """残すクラス、落とすクラス（患者数 < MIN_PATIENTS）、分割数。"""
    pats = {c: len(set(groups[y == c])) for c in classes}
    drop = [c for c in classes if pats[c] < MIN_PATIENTS]
    keep = [c for c in classes if c not in drop]
    n_folds = min(N_SPLITS, min(pats[c] for c in keep)) if len(keep) >= 2 else 0
    return keep, drop, n_folds


def cv_nearest_centroid(X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str], n_folds: int) -> dict:
    """患者単位の層化 k 分割で最近傍重心法。全 fold の検証側の予測（クラス番号）と fold ごとの BA。"""
    yi = np.array([classes.index(v) for v in y])
    pred = np.full(len(yi), -1)
    fold_ba = []
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr, te in cv.split(X, yi, groups):
        absent = [classes[c] for c in range(len(classes)) if not (yi[tr] == c).any()]
        if absent:
            raise RuntimeError(f"学習側にクラスが無い fold: {absent}")
        C = np.stack([X[tr][yi[tr] == c].mean(0) for c in range(len(classes))])
        pred[te] = _dist(X[te], C).argmin(1)
        with warnings.catch_warnings():          # 検証側に無いクラスがある fold（その fold の BA は在るクラスだけの平均）
            warnings.simplefilter("ignore")
            fold_ba.append(float(balanced_accuracy_score(yi[te], pred[te])))
    assert (pred >= 0).all()
    return {"true": yi, "pred": pred, "fold_ba": fold_ba}


# ── 施設をまたぐ値（確認用の再計算） ─────────────────────────────────────────

def cross_ba(X: np.ndarray, site: np.ndarray, y: np.ndarray, classes: list[str], tr: str, te: str) -> float:
    """transfer_correction.nc_ba と同じ: 施設 tr のクラス平均で施設 te を分類した balanced accuracy。"""
    C = np.stack([X[(site == tr) & (y == c)].mean(0) for c in classes])
    tem = site == te
    pred = _dist(X[tem], C).argmin(1)
    return float(balanced_accuracy_score(np.array([classes.index(v) for v in y[tem]]), pred))


def own_shift(X: np.ndarray, site: np.ndarray, fam: np.ndarray, target: str, ref: str) -> np.ndarray:
    """own の平行移動: 当て先の科（target）の Δ = mean(EBRAINS) − mean(patho2) を patho2 に足した埋め込み。"""
    m = np.isin(fam, TARGET_FAMILIES[target])
    Y = X.copy()
    p = site != ref
    Y[p] = X[p] + (X[m & ~p].mean(0) - X[m & p].mean(0))
    return Y


# ── 出力 ──────────────────────────────────────────────────────────────────────

def _f(v, d: int = 3) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def _md(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *["| " + " | ".join(r) + " |" for r in rows], ""]


def write_summary(path: Path, res: pd.DataFrame, counts: pd.DataFrame, notes: list[str], checks: list[str],
                  emb_status: dict[str, str]) -> None:
    encs = [e for e in ENCODER_ORDER]

    def val(task, site, enc, metric):
        r = res[(res.task == task) & (res.site == site) & (res.encoder == enc) & (res.metric == metric)]
        return float(r["value"].iloc[0]) if len(r) else np.nan

    def nrow(task, site):
        r = res[(res.task == task) & (res.site == site) & (res.metric == "balanced_accuracy")]
        return r.iloc[0] if len(r) else None

    L = ["# エンコーダの基礎精度（施設内の最近傍重心法）", "",
         "4 つのスライド埋め込みが、施設をまたがずにこの論文の分類課題をどれだけ解けるか。最近傍重心法（nc.py と同じ、訓練不要）、"
         f"患者単位の層化 {N_SPLITS} 分割交差検証（StratifiedGroupKFold、seed {SEED}）。指標は全 fold の検証側の予測をまとめた"
         "混同行列から。encoder_baseline.py。", "",
         "## 施設内の交差検証: balanced accuracy（macro F1）", ""]
    rows = []
    for task in TASKS:
        for site in SITES:
            r = nrow(task, site)
            if r is None:
                continue
            n = f"{int(r.n_slides)} 枚 / {int(r.n_patients)} 人、{int(r.n_classes)} クラス、{int(r.n_folds)} 分割"
            rows.append([TASK_LABELS_JA[task], SITE_DISPLAY[site], n]
                        + [f"{_f(val(task, site, e, 'balanced_accuracy'))}（{_f(val(task, site, e, 'f1_macro'))}）"
                           if not np.isnan(val(task, site, e, "balanced_accuracy")) else "" for e in encs])
    L += _md(["課題", "施設", "n"] + [ENCODER_LABELS_JA[e] for e in encs], rows)
    sd = res[res.metric == "balanced_accuracy_fold_sd"]["value"]
    if len(sd):
        L += [f"fold ごとの balanced accuracy の SD（参考、baseline.csv の balanced_accuracy_fold_sd）: {sd.min():.3f}–{sd.max():.3f}"
              f"（中央値 {sd.median():.3f}）。偶然の水準は 1 / クラス数（SFT vs 髄膜腫 0.5、グリオーマ 3 群 0.333、5 クラス 0.2）。", ""]

    L += ["## 施設をまたぐ参考値との並び", "",
          "各セル: **施設内 CV** / 他施設の重心で分類（補正なし = none）/ 同（own の平行移動の後）。行の施設は分類される側"
          "（patho2 の行の施設をまたぐ値 = E2P = EBRAINS の重心で patho2 を分類、EBRAINS の行 = P2E）。"
          "施設をまたぐ値は out/transfer/correction_{encoder}.csv の method centroid（読むだけ）。own の平行移動は当て先の科"
          "（SFT vs 髄膜腫・組織型 = 髄膜腫+SFT コホート、グリオーマ 3 群 = グリオーマ、5 科 = 全 5 科）の両施設の平均の差。", ""]
    rows = []
    for task in TASKS:
        for site in SITES:
            if nrow(task, site) is None:
                continue
            cells = []
            for e in encs:
                w, cn, co = (val(task, site, e, m) for m in
                             ("balanced_accuracy", "balanced_accuracy_cross_none", "balanced_accuracy_cross_own"))
                cells.append("" if np.isnan(w) and np.isnan(cn) else f"**{_f(w)}** / {_f(cn)} / {_f(co)}")
            name = "E2P" if site == "patho2" else "P2E"
            rows.append([TASK_LABELS_JA[task], f"{SITE_DISPLAY[site]}（{name}）"] + cells)
    L += _md(["課題", "分類される施設"] + [ENCODER_LABELS_JA[e] for e in encs], rows)

    L += ["## クラス定義と n", "",
          "- SFT vs 髄膜腫: 髄膜腫+SFT コホート（manifest の family ∈ {meningioma, SFT}）の全スライド。group が SFT なら SFT、"
          "ほかの髄膜腫サブタイプ（Atypical / Anaplastic / NOS / 希少型を含む）はすべて髄膜腫（transfer_correction.py の nc_sft と同じ）",
          "- 髄膜腫の主要 5 組織型: config の nc.histotype_set（main）のクラスのスライドだけ（nc_histotype と同じ）",
          "- グリオーマ 3 群: family = glioma の group（ast / glm / odg）",
          "- 5 科: family（meningioma / SFT / glioma / lymphoma / metastasis）",
          "- 患者: manifest の patient 列（EBRAINS は annotation の pat_id、院内は 1 症例 = 1 患者）。EBRAINS には複数のクラスに"
          "スライドを持つ患者がいる（髄膜腫+SFT コホートで 6 人など）。層化は StratifiedGroupKFold がスライドのラベルで行う", ""]
    rows = []
    for (task, site), g in counts.groupby(["task", "site"], sort=False):
        rows.append([TASK_LABELS_JA[task], SITE_DISPLAY[site],
                     "、".join(f"{c.replace(' meningioma', '')} {int(r.n_slides)}（{int(r.n_patients)}）"
                              for c, r in g.set_index("class").iterrows())])
    L += _md(["課題", "施設", "クラスごとのスライド数（患者数）"], rows)
    L += ["## 判断と注記", "", *[f"- {n}" for n in notes], ""]
    L += ["## 確認", "", *[f"- {c}" for c in checks], "", "## 入力", "",
          *[f"- {ENCODER_LABELS_JA[e]}: {s}" for e, s in emb_status.items()], ""]
    path.write_text("\n".join(L))


def plot(path: Path, res: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8})
    rows = [(t, s) for t in TASKS for s in SITES]
    ypos, y, prev = {}, 0.0, None
    for t, s in rows:
        if prev is not None and t != prev:
            y += 0.6
        ypos[(t, s)] = y
        y += 1.0
        prev = t
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    offs = dict(zip(ENCODER_ORDER, np.linspace(-0.27, 0.27, len(ENCODER_ORDER))))

    def val(t, s, e, m):
        r = res[(res.task == t) & (res.site == s) & (res.encoder == e) & (res.metric == m)]
        return float(r["value"].iloc[0]) if len(r) else np.nan

    for t in TASKS:
        k = res[(res.task == t) & (res.metric == "balanced_accuracy")]["n_classes"]
        if not len(k):
            continue
        ys = [ypos[(t, s)] for s in SITES]
        ax.vlines(1.0 / int(k.iloc[0]), min(ys) - 0.45, max(ys) + 0.45, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
    for e in ENCODER_ORDER:
        c, mk = ENCODER_COLORS[e], ENCODER_MARKERS[e]
        for t, s in rows:
            yy = ypos[(t, s)] + offs[e]
            w, cn = val(t, s, e, "balanced_accuracy"), val(t, s, e, "balanced_accuracy_cross_none")
            if not (np.isnan(w) or np.isnan(cn)):
                ax.plot([w, cn], [yy, yy], color=c, lw=0.8, alpha=0.55, zorder=2)
            if not np.isnan(cn):
                ax.scatter(cn, yy, marker=mk, s=30, facecolors="white", edgecolors=c, linewidths=1.2, zorder=3)
            if not np.isnan(w):
                ax.scatter(w, yy, marker=mk, s=40, color=c, edgecolors="white", linewidths=0.6, zorder=4)
    labels = []
    for t, s in rows:
        r = res[(res.task == t) & (res.site == s) & (res.metric == "balanced_accuracy")]
        n = f"  n = {int(r['n_slides'].iloc[0])}" if len(r) else ""
        labels.append(f"{SITE_DISPLAY[s]}{n}")
    ax.set_yticks([ypos[r] for r in rows])
    ax.set_yticklabels(labels)
    for t in TASKS:
        ys = [ypos[(t, s)] for s in SITES]
        ax.text(0.155, min(ys) - 0.62, TASK_LABELS[t], fontsize=8, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.set_ylim(max(ypos.values()) + 0.6, min(ypos.values()) - 0.95)
    ax.set_xlim(0.15, 1.005)
    ax.set_xlabel("Balanced accuracy (nearest centroid)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=ENCODER_MARKERS[e], color=ENCODER_COLORS[e], ls="none", ms=6, label=ENCODER_LABELS[e])
               for e in ENCODER_ORDER]
    handles += [Line2D([], [], marker="o", color=INK, ls="none", ms=6, label=f"within site ({N_SPLITS}-fold CV by patient)"),
                Line2D([], [], marker="o", color=INK, mfc="white", ls="none", ms=6,
                       label="other site's centroids (no correction)"),
                Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="chance (1 / classes)")]
    ax.legend(handles=handles, loc="lower left", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID)
    ax.set_title("Baseline accuracy of each slide encoder", fontsize=9, loc="left", color=INK)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def run(cfg: dict, out_dir: Path) -> None:
    out = Path(cfg["output_dir"])
    ref = cfg["reference"]
    man = load_manifest(out, cfg)
    site, fam = man["site"].astype(str).to_numpy(), man["family"].astype(str).to_numpy()
    patient = man["patient"].astype(str).to_numpy()
    histotype_classes = apply_set(cfg, cfg.get("nc", {}).get("histotype_set", "main"))["lp_histotype"]["classes"]
    print(f"[encoder_baseline] manifest {len(man)} slides; histotype classes: {histotype_classes}")

    checks: list[str] = []
    ps_path = out / "shift_transfer" / "paper_slides.csv"
    if ps_path.exists():   # 髄膜腫+SFT コホートのラベルが shift_transfer の paper_slides.csv と同じか（確認だけ）
        ps = pd.read_csv(ps_path, dtype={"slide_id": str, "patient": str})
        pm = man[np.isin(fam, ["meningioma", "SFT"])].merge(ps, left_on=["site", "slide_id"], right_on=["source", "slide_id"],
                                                            how="outer", suffixes=("", "_ps"), indicator=True)
        both = pm["_merge"] == "both"
        same = int((pm.loc[both, "group"] == pm.loc[both, "subtype"]).sum())
        msg = (f"髄膜腫+SFT コホートの group が shift_transfer/paper_slides.csv の subtype と一致: {same} / {int(both.sum())} 枚"
               f"（片方だけのスライド {int((~both).sum())} 枚）")
        print("  " + msg)
        checks.append(msg)

    rows, cm_rows, count_rows, notes = [], [], [], []
    emb_status: dict[str, str] = {}
    cross_diff: dict[str, float] = {}
    for enc in ENCODER_ORDER:
        X, status = load_embedding(out, enc, man)
        emb_status[enc] = status
        print(f"\n[{enc}] {status}")
        if X is None:
            print(f"  WARN {enc} は空欄")
            notes.append(f"{ENCODER_LABELS_JA[enc]}: 埋め込みが読めないので空欄（{status}）")
            continue
        corr_path = out / "transfer" / f"correction_{enc}.csv"
        corr = pd.read_csv(corr_path) if corr_path.exists() else None
        if corr is None:
            notes.append(f"{ENCODER_LABELS_JA[enc]}: {corr_path.relative_to(out)} が無いので施設をまたぐ参考値は空欄")
        own_cache: dict[str, np.ndarray] = {}
        for task in TASKS:
            m, y, classes = task_labels(man, task, histotype_classes)
            for s in SITES:
                sm = m & (site == s)
                ys, gs = y[sm], patient[sm]
                keep, drop, n_folds = plan(ys, gs, classes)
                if enc == ENCODER_ORDER[0]:
                    for c in classes:
                        count_rows.append({"task": task, "site": s, "class": c, "n_slides": int((ys == c).sum()),
                                           "n_patients": len(set(gs[ys == c]))})
                    if drop:
                        notes.append(f"{TASK_LABELS_JA[task]} / {SITE_DISPLAY[s]}: 患者数 < {MIN_PATIENTS} のクラスを落とした: {drop}")
                    if n_folds < N_SPLITS:
                        notes.append(f"{TASK_LABELS_JA[task]} / {SITE_DISPLAY[s]}: 分割数を {n_folds} に減らした"
                                     f"（クラスの患者数の最小が {N_SPLITS} 未満）")
                if n_folds < 2:
                    continue
                km = np.isin(ys, keep)
                r = cv_nearest_centroid(X[sm][km], ys[km], gs[km], keep, n_folds)
                t, p = r["true"], r["pred"]
                base = {"task": task, "site": s, "encoder": enc, "n_slides": int(km.sum()),
                        "n_patients": len(set(gs[km])), "n_classes": len(keep), "n_folds": n_folds}
                ba = float(balanced_accuracy_score(t, p))
                f1 = float(f1_score(t, p, average="macro", labels=list(range(len(keep)))))
                rows += [{**base, "metric": "balanced_accuracy", "value": ba},
                         {**base, "metric": "f1_macro", "value": f1},
                         {**base, "metric": "balanced_accuracy_fold_sd", "value": float(np.std(r["fold_ba"], ddof=1))}]
                cm = confusion_matrix(t, p, labels=list(range(len(keep))))
                cm_rows += [{"task": task, "site": s, "encoder": enc, "true": keep[i], "pred": keep[j], "n": int(cm[i, j])}
                            for i in range(len(keep)) for j in range(len(keep))]
                print(f"  [{task} / {s}] {base['n_slides']} slides / {base['n_patients']} patients, {len(keep)} classes, "
                      f"{n_folds} folds: BA {ba:.3f}  macro F1 {f1:.3f}  (fold BA {np.mean(r['fold_ba']):.3f} ± "
                      f"{np.std(r['fold_ba'], ddof=1):.3f})")

            # 施設をまたぐ参考値（読むだけ）と、同じクラス定義での再計算（確認）
            target, stem = CROSS[task]
            if target not in own_cache:
                own_cache[target] = own_shift(X, site, fam, target, ref)
            for name, (tr, te) in CROSS_DIR.items():
                te_m = m & (site == te)
                base = {"task": task, "site": te, "encoder": enc, "n_slides": int(te_m.sum()),
                        "n_patients": len(set(patient[te_m])), "n_classes": len(classes), "n_folds": np.nan}
                mine = {"none": cross_ba(X[m], site[m], y[m], classes, tr, te),
                        "own": cross_ba(own_cache[target][m], site[m], y[m], classes, tr, te)}
                for src in ("none", "own"):
                    v = np.nan
                    if corr is not None:
                        q = corr[(corr.method == "centroid") & (corr.source == src) & (corr.target == target)
                                 & (corr.metric == f"{stem}_{name}")]
                        if len(q) == 1:
                            v = float(q["value"].iloc[0])
                            cross_diff[enc] = max(cross_diff.get(enc, 0.0), abs(v - mine[src]))
                    rows.append({**base, "metric": f"balanced_accuracy_cross_{src}", "value": v})
                print(f"  [{task} / {name} cross] none {_f(rows[-2]['value'])} (recomputed {mine['none']:.3f})  "
                      f"own {_f(rows[-1]['value'])} (recomputed {mine['own']:.3f})")

    for enc, d in cross_diff.items():
        ok = d < 1e-9
        msg = (f"{ENCODER_LABELS_JA[enc]}: 施設をまたぐ none / own を同じクラス定義で再計算 → correction_{enc}.csv との max |差| "
               f"{d:.1e}（{'一致' if ok else '不一致: クラス定義か埋め込みが違う'}）")
        print(("  " if ok else "  WARN ") + msg)
        checks.append(msg)

    cols = ["task", "site", "encoder", "metric", "value", "n_slides", "n_patients", "n_classes", "n_folds"]
    res = pd.DataFrame(rows)[cols]
    res["n_folds"] = res["n_folds"].astype("Int64")
    res.to_csv(out_dir / "baseline.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(out_dir / "confusion.csv", index=False)
    counts = pd.DataFrame(count_rows)
    if not notes:
        notes.append(f"落としたクラスは無い（どの課題・施設もクラスの患者数の最小が {N_SPLITS} 以上なので全部 {N_SPLITS} 分割）")
    notes.append("クラスの重心はスライドの平均（nc.py と同じ。患者ごとに平均してからは取らない）。分類はスライド単位")
    cm = pd.DataFrame(cm_rows)
    sft = cm[cm.task == "sft"]
    if len(sft):   # SFT vs 髄膜腫の macro F1 が BA より低い理由（少数クラスの SFT に髄膜腫が割り当てられ、SFT の適合率が下がる）
        g = sft.groupby(["encoder", "site"])
        tp = g.apply(lambda d: d[(d.true == "SFT") & (d.pred == "SFT")]["n"].sum(), include_groups=False)
        rec = tp / g.apply(lambda d: d[d.true == "SFT"]["n"].sum(), include_groups=False)
        prec = tp / g.apply(lambda d: d[d.pred == "SFT"]["n"].sum(), include_groups=False)
        share = g.apply(lambda d: d[d.true == "SFT"]["n"].sum() / d["n"].sum(), include_groups=False)
        notes.append(f"SFT vs 髄膜腫の macro F1 が BA よりかなり低いのは、SFT が少数クラス（{100 * share.min():.1f}–{100 * share.max():.1f}%）"
                     "で最近傍重心法がクラスの事前確率を"
                     "使わないため、髄膜腫の一部が SFT に割り当てられて SFT の適合率が下がるから（SFT の再現率 "
                     f"{rec.min():.2f}–{rec.max():.2f}、適合率 {prec.min():.2f}–{prec.max():.2f}、エンコーダ × 施設の範囲。"
                     "confusion.csv）。主の指標は BA")
    write_summary(out_dir / "summary.md", res, counts, notes, checks, emb_status)
    plot(out_dir / "baseline.png", res)
    print(f"\nsaved: {out_dir}/{{baseline.csv, confusion.csv, summary.md, baseline.png, run.log}}")


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "encoder_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run.log", "w") as log, contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        run(cfg, out_dir)


if __name__ == "__main__":
    main()
