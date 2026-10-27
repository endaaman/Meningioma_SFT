"""atlas_retrieval.py — EBRAINS 全体を参照コホートにした院内スライドの最近傍診断（計画の「参照コホートへの診断名の検索」の図）。

問い: 院内（patho2）のスライドを、公開コホート EBRAINS 全体の診断ごとの重心（約 130 診断のうち --min-n 枚以上のもの）に
最近傍重心法で当てると、どこまで診断名が当たるか。論文の平行移動 Δ（髄膜腫 / SFT だけで推定）を足すと、髄膜腫以外の
腫瘍（グリオーマ・リンパ腫・転移）でもどれだけ良くなるか。

参照（EBRAINS 全体）: ebrains_embeddings.root の全 h5（uuid で重複を除き浅い方を優先。入れ子の「… 1p/19q codeleted」が 7 uuid を
    重複させている）× shift_transfer.ebrains_annotation（annotation.csv）。control = 1 は除き、diagnosis をクラスにする。
    埋め込みのキーが無い h5 は飛ばす。--min-n 枚以上のクラスだけ残す（落ちたクラスは summary に列挙）。
問い合わせ（patho2）: transfer.manifest（transfer_manifest.py）の site = patho2 の全行
    = 論文コホートの patho2（label.patho2.subtype、577 枚。ラベルはこの CSV から取り、manifest と一致することを確かめる）
    ＋ Patho2Gliomas の代表 HE（glioma = ast / glm / odg、lymphoma = lym、metastasis = met、260 枚）。

正解の集合（手で書かない）:
    髄膜腫サブタイプ・SFT: 論文の EBRAINS 側（label.ebrains.subtype、n551）で同じ subtype を持つスライドの annotation.csv の
        diagnosis（uuid 先頭 8 文字 = case_id で結合）。例: SFT → Haemangiopericytoma
    ast / glm / odg / lym / met: shift_transfer.ebrains_groups
    NOS meningioma（patho2 だけ）: 名前に "meningioma" を含む診断ならどれでも正解（髄膜腫とは別枠で報告し、overall に入れない）
    参照に残らなかった診断（--min-n 未満）は正解の集合から落とす。集合が空になったラベル（例: Metaplastic）は
    診断名の指標から外し（scorable = False）、科の指標だけに入れる。
科（粗い指標）: meningioma（名前に "meningioma"）/ SFT（SFT の正解の集合 = HPC）/ glioma（ast・glm・odg の集合の和）/
    lymphoma（lym の集合）/ metastasis（met の集合）/ other（残り全部）。
    副の broad 版は other のうち BROAD_PATTERNS に合う診断を glioma（IDH 野生型の星細胞腫・膠肉腫・巨細胞膠芽腫・びまん性正中
    グリオーマ）と lymphoma（名前に "lymphoma"）に入れる。

条件（patho2 の問い合わせに足すベクトル。推定に使うのは manifest の 2 施設のスライド = transfer_correction.py と同じ集合）:
    none        補正なし
    paper       Δ_paper = mean(EBRAINS 論文コホート n551) − mean(patho2 論文コホート n577)。論文の centroid 補正（髄膜腫 + SFT）
    others      問い合わせの科以外の全部の科で推定（2 施設それぞれの全スライド平均の差。transfer_correction.py の others と同じ）
    others_bal  問い合わせの科以外の科ごとの Δ_f を重み n_E·n_P / (n_E + n_P) で平均したもの。2 施設で科の構成
                （例: EBRAINS はグリオーマが多く patho2 は髄膜腫が多い）が違うと、素朴な others には科どうしの差（長さ 10–16）が
                構成の差の分だけ混ざるので、それを避けた版
    family      問い合わせの科自身で推定（ラベルを使う上限の参考）。問い合わせのスライドは patho2 側の平均から抜く（leave-one-out）
EBRAINS 内の基準（施設差なしの上限）: 問い合わせのラベル（NOS を除く）の正解の集合に入る EBRAINS のスライドを、自分を
    抜いた重心で最近傍重心法（ebrains_loo）。ラベルごとの値を patho2 のラベル構成で重み付けした版（ebrains_loo_mix、
    patho2 の行と構成を揃えた比較用）と、同じ患者の他のスライド（再発標本など）も抜いた版（ebrains_lpo）も出す。

検索と指標:
    nc_euclid（主）: 診断ごとの重心（スライド平均）へのユークリッド距離で全診断を並べる。
    nc_cosine（副）: 同じ重心への cos 距離。
    knn5（副）: 参照のスライドのうちユークリッド距離で最も近い 5 枚の多数決（同数は近い方）。top-5 = その 5 枚の診断。
    top1 / top5    上位 k の中に正解の集合の診断があるか（scorable の行だけ）
    mean_rank      正解の診断の最良順位（1 始まり）の平均（nc_* だけ。knn5 は 5 枚の外の順位が無いので出さない）
    family_top1 / family_top1_broad   top-1 の診断の科が問い合わせの科と一致するか（全行）
    グリオーマの混同: top-1 の診断を ast / glm / odg / other glioma（broad だけに入る）/ non-glioma に分けたもの（full）と、
        ast・glm・odg の集合の診断の重心だけに絞った 3 択（restricted、nc_* だけ）。どちらも balanced accuracy（3 群の recall の平均）。

出力 (output_dir/atlas/):
    retrieval_{enc}.csv          縦長: encoder, retrieval, condition, query_family, query_label, metric, value, n
                                 （query_label = "(all)" が科などの集計、それ以外がラベルごと）
    per_slide_{enc}.csv          スライドごと: retrieval, condition, slide_id, site, family, label, scorable, top1, top1_family,
                                 top5（";" 区切り）, rank_correct, hit1, hit5, family_hit1, family_hit1_broad
    glioma_confusion_{enc}.csv   retrieval, condition, scope（full / restricted）, true, pred, count
    top1_counts_{enc}.csv        retrieval, condition, label, top1, count
    deltas_{enc}.csv             問い合わせの科 × 条件ごとの Δ の長さ、Δ_paper との cos、Δ_family からのずれ
    mapping_{enc}.csv            ラベル → 正解の集合（参照に残ったもの / 落ちたもの）
    retrieval_{enc}_summary.md   参照の構成・ラベルの対応・Δ・ピボット（科 × 条件、EBRAINS の基準の行つき）・ラベルごとの表・
                                 グリオーマの混同・top-1 の診断の内訳
    retrieval_{enc}.png          a: top-1、b: top-5、c: 科の top-1（科 × 条件の棒、EBRAINS LOO を横線）
    atlas_{enc}.npy / atlas_{enc}_slides.csv / atlas_{enc}_info.json、manifest_{enc}.npy / manifest_{enc}_slides.csv
                                 埋め込みのキャッシュ（--reload で読み直す。manifest の並びが変わったら自動で読み直す）
    --min-n が既定（5）以外のときは、キャッシュ以外の出力名の {enc} が {enc}_min{n} になる。
    --atlas-slides <enc2> のときは {enc}_on_{enc2}（と _min{n}）になる。

Usage:
    uv run python atlas_retrieval.py [--reload] [--min-n 5] [--encoder titan] [--atlas-slides prism]
    （--encoder prism は transfer.prism_key を同じ h5 から読む。参照は PRISM のある EBRAINS のスライドだけになる。
    PRISM は転移テストの manifest の 2,301 枚にしか抽出していないので、今は EBRAINS 1,464 枚・診断はその科のものだけ。
    TITAN と同じ参照で比べるには --encoder titan --atlas-slides prism: TITAN の参照を atlas_prism_slides.csv の
    uuid に絞る（先に --encoder prism を回してキャッシュを作る）。問い合わせ・Δ の推定は絞らない（manifest の全部）。）
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.loader import load_config

ENCODERS = {  # encoder → (manifest の h5 の列, config transfer.* のキーの名前)。参照も同じキーを EBRAINS_embeddings の h5 から読む
    "titan": ("h5_titan", "titan_key"),
    "prism": ("h5_prism", "prism_key"),
}
FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
GLIOMA_GROUPS = ["ast", "glm", "odg"]
NOS = "NOS meningioma"
CONDITIONS = ["none", "paper", "others", "others_bal", "family"]
BASELINES = ["ebrains_loo", "ebrains_loo_mix", "ebrains_lpo"]
RETRIEVALS = ["nc_euclid", "nc_cosine", "knn5"]
TOPK = 5
KNN_K = 5
MIN_N_DEFAULT = 5   # これ以外の --min-n では出力名に _min{n} を付ける（感度の確認用。既定の出力を上書きしない）
# 科の broad 版で other から移す診断名のパターン（副の指標だけ。主の科は正解の集合から作る）
BROAD_PATTERNS = {"glioma": re.compile(r"glioblastoma|gliosarcoma|astrocytoma, IDH|oligodendroglioma|diffuse midline glioma", re.I),
                  "lymphoma": re.compile(r"lymphoma", re.I)}
# 集計の単位（retrieval_*.csv の query_family、query_label = "(all)"）
AGG_ORDER = ["meningioma", "meningioma (NOS)", "SFT", "glioma", "ast", "glm", "odg", "lymphoma", "metastasis",
             "glioma+lym+met", "overall"]
COND_LABEL = {"none": "none", "paper": "paper Δ", "others": "others Δ (pooled)", "others_bal": "others Δ (per-family)",
              "family": "own-family Δ (upper ref.)", "ebrains_loo": "EBRAINS LOO", "ebrains_loo_mix": "EBRAINS LOO (patho2 mix)",
              "ebrains_lpo": "EBRAINS leave-patient-out"}
COND_COLOR = {"none": "#a3a29b", "paper": "#2a78d6", "others": "#eda100", "others_bal": "#eb6834", "family": "#1baf7a"}


# ── data ──────────────────────────────────────────────────────────────────────

def _read_h5(path: str | Path, key: str, retries: int = 3) -> np.ndarray | None:
    """h5 から 1 本読む（無ければ None）。prism_extract.py が同じ h5 に書き込み中だと開けないことがあるので、少し待って読み直す。"""
    p = Path(path)
    if not p.exists():
        return None
    for i in range(retries):
        try:
            with h5py.File(p, "r") as f:
                return np.asarray(f[key][:], dtype=np.float32).reshape(-1) if key in f else None
        except OSError as e:
            if i == retries - 1:
                print(f"  WARN cannot open {p}: {e}")
                return None
            time.sleep(2.0)
    return None


def load_atlas(cfg: dict, key: str, out_dir: Path, enc: str, reload: bool) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """EBRAINS の全スライド（control を除く）の埋め込み。列: uuid, slide_id（先頭 8 文字）, patient, diagnosis, h5。"""
    csv, npy, js = out_dir / f"atlas_{enc}_slides.csv", out_dir / f"atlas_{enc}.npy", out_dir / f"atlas_{enc}_info.json"
    if not reload and csv.exists() and npy.exists() and js.exists():
        print(f"  cache: {npy}")
        return (pd.read_csv(csv, dtype={"uuid": str, "slide_id": str, "patient": str}), np.load(npy).astype(np.float64),
                json.loads(js.read_text()))
    root = Path(cfg["ebrains_embeddings"]["root"])
    files = sorted(root.rglob("*_hdf.h5"), key=lambda q: (len(q.parts), str(q)))
    index: dict[str, Path] = {}   # uuid → h5（浅い方を優先）
    for p in files:
        index.setdefault(p.name[: -len("_hdf.h5")], p)
    ann_all = pd.read_csv(cfg["shift_transfer"]["ebrains_annotation"], dtype={"uuid": str, "pat_id": str})
    ann = ann_all[(ann_all["control"] != 1) & ann_all["diagnosis"].notna()]
    print(f"  {len(files)} h5 files / {len(index)} uuids; annotation {len(ann_all)} rows, {len(ann)} non-control")
    rows, embs, no_file, no_key = [], [], [], []
    for r in ann.itertuples(index=False):
        p = index.get(r.uuid)
        if p is None:
            no_file.append(r.uuid)
            continue
        v = _read_h5(p, key)
        if v is None:
            no_key.append(r.uuid)
            continue
        rows.append({"uuid": r.uuid, "slide_id": r.uuid[:8], "patient": str(r.pat_id), "diagnosis": r.diagnosis, "h5": str(p)})
        embs.append(v)
    df = pd.DataFrame(rows)
    X = np.stack(embs).astype(np.float32)
    info = {"n_files": len(files), "n_uuids_with_file": len(index), "n_annotation": len(ann_all),
            "n_control": int((ann_all["control"] == 1).sum()), "n_noncontrol": len(ann),
            "no_file": no_file, "no_key": no_key, "n_loaded": len(df),
            "uuids_without_annotation": sorted(set(index) - set(ann_all["uuid"]))}
    df.to_csv(csv, index=False)
    np.save(npy, X)
    js.write_text(json.dumps(info, indent=1))
    print(f"  saved: {npy} {X.shape}")
    return df, X.astype(np.float64), info


def load_manifest(cfg: dict, h5_col: str, key: str, out_dir: Path, enc: str, reload: bool) -> tuple[pd.DataFrame, np.ndarray]:
    """transfer manifest の全スライド（2 施設）の埋め込み（manifest の並び）。Δ の推定と patho2 の問い合わせに使う。"""
    man = pd.read_csv(Path(cfg["output_dir"]) / cfg["transfer"].get("manifest", "transfer/manifest.csv"),
                      dtype={"slide_id": str, "patient": str})
    ident = man[["site", "slide_id"]].reset_index(drop=True)
    npy, order_csv = out_dir / f"manifest_{enc}.npy", out_dir / f"manifest_{enc}_slides.csv"
    if not reload and npy.exists() and order_csv.exists():
        if pd.read_csv(order_csv, dtype={"slide_id": str}).equals(ident):
            print(f"  cache: {npy}")
            return man, np.load(npy).astype(np.float64)
        print("  cache の並びが manifest と合わないので読み直す")
    print(f"  reading {len(man)} h5 ({key}) ...")
    embs = [_read_h5(p, key) if isinstance(p, str) and p else None for p in man[h5_col]]
    miss = [i for i, v in enumerate(embs) if v is None]
    if miss:
        m = man.iloc[miss].groupby(["family", "site"]).size()
        raise SystemExit(f"[{enc}] {len(miss)} / {len(man)} 枚に {key} が無い（{h5_col}）:\n{m.to_string()}")
    X = np.stack(embs).astype(np.float32)
    np.save(npy, X)
    ident.to_csv(order_csv, index=False)
    print(f"  saved: {npy} {X.shape}")
    return man, X.astype(np.float64)


# ── labels ────────────────────────────────────────────────────────────────────

def label_mapping(cfg: dict, all_diagnoses: set[str]) -> dict[str, list[str]]:
    """patho2 のラベル → 正解になる EBRAINS の diagnosis（参照のクラスで絞る前）。"""
    ann = pd.read_csv(cfg["shift_transfer"]["ebrains_annotation"], dtype={"uuid": str})
    diag_of = dict(zip(ann["uuid"].str[:8], ann["diagnosis"]))
    e = pd.read_csv(cfg["label"]["ebrains"]["subtype"], dtype=str)
    e["diagnosis_ann"] = e["case_id"].map(diag_of)
    if e["diagnosis_ann"].isna().any():
        print(f"  WARN {int(e['diagnosis_ann'].isna().sum())} paper EBRAINS slides not in annotation.csv")
    mapping = {sub: sorted(set(g.dropna())) for sub, g in e.groupby("subtype")["diagnosis_ann"]}
    for g, ds in cfg["shift_transfer"]["ebrains_groups"].items():
        mapping[g] = list(ds)
    mapping[NOS] = sorted(d for d in all_diagnoses if "meningioma" in d.lower())
    return mapping


def diagnosis_family(d: str, mapping: dict[str, list[str]]) -> str:
    if d in mapping["SFT"]:
        return "SFT"
    if "meningioma" in d.lower():
        return "meningioma"
    if any(d in mapping[g] for g in GLIOMA_GROUPS):
        return "glioma"
    if d in mapping["lym"]:
        return "lymphoma"
    if d in mapping["met"]:
        return "metastasis"
    return "other"


def diagnosis_family_broad(d: str, mapping: dict[str, list[str]]) -> str:
    f = diagnosis_family(d, mapping)
    if f != "other":
        return f
    return next((fam for fam, pat in BROAD_PATTERNS.items() if pat.search(d)), "other")


def glioma_group_of(d: str, mapping: dict[str, list[str]]) -> str:
    """top-1 の診断 → ast / glm / odg / other glioma（broad だけ）/ non-glioma（グリオーマの混同の列）。"""
    g = next((g for g in GLIOMA_GROUPS if d in mapping[g]), None)
    if g:
        return g
    return "other glioma" if diagnosis_family_broad(d, mapping) == "glioma" else "non-glioma"


# ── condition shifts ──────────────────────────────────────────────────────────

class Shifts:
    """manifest の 2 施設 × 科の和と枚数から、条件ごとの Δ を作る。"""

    def __init__(self, man: pd.DataFrame, M: np.ndarray):
        fam, site = man["family"].to_numpy(), man["site"].to_numpy()
        self.S = {(s, f): M[(site == s) & (fam == f)].sum(0) for s in ("ebrains", "patho2") for f in FAMILIES}
        self.N = {(s, f): int(((site == s) & (fam == f)).sum()) for s in ("ebrains", "patho2") for f in FAMILIES}

    def mu(self, site: str, fams) -> np.ndarray:
        return sum(self.S[site, f] for f in fams) / sum(self.N[site, f] for f in fams)

    def pooled(self, fams) -> np.ndarray:
        return self.mu("ebrains", fams) - self.mu("patho2", fams)

    def balanced(self, fams) -> np.ndarray:
        w = {f: self.N["ebrains", f] * self.N["patho2", f] / (self.N["ebrains", f] + self.N["patho2", f]) for f in fams}
        return sum(w[f] * self.pooled([f]) for f in fams) / sum(w.values())

    def for_queries(self, cond: str, fam_q: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        out = np.zeros_like(Xq)
        if cond == "none":
            return out
        if cond == "paper":
            return out + self.pooled(["meningioma", "SFT"])
        for f in FAMILIES:
            m = fam_q == f
            if not m.any():
                continue
            rest = [g for g in FAMILIES if g != f]
            if cond == "others":
                out[m] = self.pooled(rest)
            elif cond == "others_bal":
                out[m] = self.balanced(rest)
            elif cond == "family":   # 問い合わせ自身を patho2 側の平均から抜く
                out[m] = self.mu("ebrains", [f]) - (self.S["patho2", f] - Xq[m]) / (self.N["patho2", f] - 1)
            else:
                raise ValueError(cond)
        return out


# ── retrieval ─────────────────────────────────────────────────────────────────

def _dist(Q: np.ndarray, C: np.ndarray, metric: str) -> np.ndarray:
    if metric == "euclid":
        d2 = (Q ** 2).sum(1)[:, None] - 2 * Q @ C.T + (C ** 2).sum(1)[None, :]
        return np.sqrt(np.maximum(d2, 0.0))
    Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    return 1.0 - Qn @ Cn.T


def _from_order(order: np.ndarray, corr: np.ndarray) -> dict:
    """並び（行ごとのクラス番号、-1 埋め）と正解のマスク（行 × クラス）から top / 順位 / 当たり。"""
    valid = order >= 0
    hit = np.where(valid, np.take_along_axis(corr, np.where(valid, order, 0), axis=1), False)
    found = hit.any(1)
    rank = np.where(found, hit.argmax(1) + 1, np.nan)
    return {"top": order[:, :TOPK], "rank": rank, "hit1": hit[:, 0], "hit5": hit[:, :TOPK].any(1)}


def nc_retrieve(D: np.ndarray, corr: np.ndarray) -> dict:
    return _from_order(np.argsort(D, axis=1, kind="stable"), corr)


def knn_retrieve(Dslides: np.ndarray, y_ref_idx: np.ndarray, corr: np.ndarray, n_classes: int) -> dict:
    """最も近い KNN_K 枚の多数決（同数は近い方を先に）。並びは 5 枚に出てきた診断だけ（残りは -1）。"""
    nn = np.argsort(Dslides, axis=1, kind="stable")[:, :KNN_K]
    order = np.full((len(Dslides), TOPK), -1, dtype=int)
    for i, row in enumerate(nn):
        labs = y_ref_idx[row]
        cnt = Counter(labs)
        first = {c: k for k, c in reversed(list(enumerate(labs)))}
        ranked = sorted(cnt, key=lambda c: (-cnt[c], first[c]))[:TOPK]
        order[i, : len(ranked)] = ranked
    res = _from_order(order, corr)
    res["rank"] = np.full(len(order), np.nan)   # 5 枚の外の順位は無いので mean_rank は出さない
    return res


def corr_mask(labels: np.ndarray, classes: list[str], correct: dict[str, list[str]]) -> np.ndarray:
    pos = {c: k for k, c in enumerate(classes)}
    M = np.zeros((len(labels), len(classes)), dtype=bool)
    for i, lab in enumerate(labels):
        for d in correct.get(lab, []):
            M[i, pos[d]] = True
    return M


def loo_distances(R: np.ndarray, y_idx: np.ndarray, patient: np.ndarray, sel: np.ndarray, C: np.ndarray, n_c: np.ndarray,
                  metric: str, by_patient: bool) -> np.ndarray:
    """参照のスライド sel を、自分（by_patient なら同じ患者の全スライド）を抜いた重心に当てたときの距離（sel × クラス）。"""
    S = C * n_c[:, None]
    D = _dist(R[sel], C, metric)
    if not by_patient:
        c = y_idx[sel]
        n_left = n_c[c] - 1
        Cl = np.where(n_left[:, None] > 0, (S[c] - R[sel]) / np.maximum(n_left, 1)[:, None], np.nan)
        dl = _dist_rows(R[sel], Cl, metric)
        D[np.arange(len(sel)), c] = np.where(n_left > 0, dl, np.inf)
        return D
    pos_of = {j: k for k, j in enumerate(sel)}
    for p in np.unique(patient[sel]):
        js = np.where(patient == p)[0]                      # 参照にあるその患者の全スライド
        rows = [pos_of[j] for j in js if j in pos_of]       # そのうち評価するスライド
        for c in np.unique(y_idx[js]):
            jc = js[y_idx[js] == c]
            n_left = n_c[c] - len(jc)
            if n_left <= 0:
                D[rows, c] = np.inf
                continue
            Cl = (S[c] - R[jc].sum(0)) / n_left
            D[rows, c] = _dist(R[sel[rows]], Cl[None, :], metric)[:, 0]
    return D


def _dist_rows(Q: np.ndarray, C: np.ndarray, metric: str) -> np.ndarray:
    """行ごとの対の距離（Q[i] と C[i]）。"""
    if metric == "euclid":
        return np.linalg.norm(Q - C, axis=1)
    return 1.0 - (Q * C).sum(1) / (np.linalg.norm(Q, axis=1) * np.linalg.norm(C, axis=1))


# ── scoring ───────────────────────────────────────────────────────────────────

def per_slide_frame(meta: pd.DataFrame, res: dict, classes: list[str], fam_of_class: np.ndarray, famb_of_class: np.ndarray,
                    retrieval: str, condition: str) -> pd.DataFrame:
    top = res["top"]
    t1 = top[:, 0]
    names = np.array(classes + [""], dtype=object)
    df = meta[["slide_id", "site", "family", "label", "scorable"]].copy()
    df.insert(0, "condition", condition)
    df.insert(0, "retrieval", retrieval)
    df["top1"] = names[t1]
    df["top1_family"] = fam_of_class[t1]
    df["top5"] = [";".join(names[k] for k in row if k >= 0) for row in top]
    df["rank_correct"] = np.where(meta["scorable"].to_numpy(), res["rank"], np.nan)
    df["hit1"] = res["hit1"] & meta["scorable"].to_numpy()
    df["hit5"] = res["hit5"] & meta["scorable"].to_numpy()
    df["family_hit1"] = fam_of_class[t1] == meta["family"].to_numpy()
    df["family_hit1_broad"] = famb_of_class[t1] == meta["family"].to_numpy()
    return df


def _agg_masks(ps: pd.DataFrame) -> dict[str, np.ndarray]:
    fam, lab = ps["family"].to_numpy(), ps["label"].to_numpy()
    return {
        "meningioma": (fam == "meningioma") & (lab != NOS), "meningioma (NOS)": lab == NOS, "SFT": fam == "SFT",
        "glioma": fam == "glioma", "ast": lab == "ast", "glm": lab == "glm", "odg": lab == "odg",
        "lymphoma": fam == "lymphoma", "metastasis": fam == "metastasis",
        "glioma+lym+met": np.isin(fam, ["glioma", "lymphoma", "metastasis"]), "overall": lab != NOS,
    }


def _wmean(v: np.ndarray, w: np.ndarray) -> float:
    ok = ~np.isnan(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")


def aggregate(ps: pd.DataFrame, encoder: str, weights: np.ndarray | None = None) -> list[dict]:
    """1 つの (retrieval, condition) のスライドごとの結果 → 縦長の行（集計とラベルごと）。weights はスライドの重み（mix 用）。"""
    retrieval, condition = ps["retrieval"].iat[0], ps["condition"].iat[0]
    w_all = np.ones(len(ps)) if weights is None else weights
    sc = ps["scorable"].to_numpy()
    rows = []

    def emit(qf: str, ql: str, m: np.ndarray) -> None:
        if not m.any():
            return
        md = m & sc
        vals = {
            "top1": (ps["hit1"].to_numpy(float), md), "top5": (ps["hit5"].to_numpy(float), md),
            "mean_rank": (ps["rank_correct"].to_numpy(float), md),
            "family_top1": (ps["family_hit1"].to_numpy(float), m), "family_top1_broad": (ps["family_hit1_broad"].to_numpy(float), m),
        }
        for metric, (v, mm) in vals.items():
            if retrieval == "knn5" and metric == "mean_rank":
                continue
            if not mm.any():
                continue
            rows.append(dict(encoder=encoder, retrieval=retrieval, condition=condition, query_family=qf, query_label=ql,
                             metric=metric, value=_wmean(v[mm], w_all[mm]), n=int(mm.sum())))

    for qf, m in _agg_masks(ps).items():
        emit(qf, "(all)", m)
    for (fam, lab), idx in ps.groupby(["family", "label"]).indices.items():
        m = np.zeros(len(ps), bool)
        m[idx] = True
        emit(fam, lab, m)
    return rows


def glioma_confusion(ps: pd.DataFrame, D: np.ndarray | None, classes: list[str], mapping: dict, correct: dict) -> list[dict]:
    """グリオーマの 3 群の混同: full（top-1 の診断の群）と restricted（ast・glm・odg の重心だけの 3 択、D があるとき）。"""
    retrieval, condition = ps["retrieval"].iat[0], ps["condition"].iat[0]
    m = ps["label"].isin(GLIOMA_GROUPS).to_numpy()
    if not m.any():
        return []
    true = ps.loc[m, "label"].to_numpy()
    out = []
    pred_full = np.array([glioma_group_of(d, mapping) for d in ps.loc[m, "top1"]])
    preds = {"full": pred_full}
    if D is not None:
        gcls = [k for k, c in enumerate(classes) if any(c in correct[g] for g in GLIOMA_GROUPS)]
        grp = np.array([next(g for g in GLIOMA_GROUPS if classes[k] in correct[g]) for k in gcls])
        preds["restricted"] = grp[np.argmin(D[m][:, gcls], axis=1)]
    for scope, pred in preds.items():
        for t in GLIOMA_GROUPS:
            for p in GLIOMA_GROUPS + ["other glioma", "non-glioma"]:
                if scope == "restricted" and p not in GLIOMA_GROUPS:
                    continue
                out.append(dict(retrieval=retrieval, condition=condition, scope=scope, true=t, pred=p,
                                count=int(((true == t) & (pred == p)).sum())))
    return out


def balanced_acc(conf: pd.DataFrame) -> float:
    rec = [conf.loc[(conf["true"] == t) & (conf["pred"] == t), "count"].sum() / conf.loc[conf["true"] == t, "count"].sum()
           for t in GLIOMA_GROUPS if conf.loc[conf["true"] == t, "count"].sum() > 0]
    return float(np.mean(rec)) if rec else float("nan")


# ── summary ───────────────────────────────────────────────────────────────────

def _fmt(v, digits: int = 3) -> str:
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def _md(df: pd.DataFrame, digits: int = 3, index: bool = True) -> str:
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in d.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in r) + " |")
    return "\n".join(lines)


def pivot(res: pd.DataFrame, retrieval: str, metric: str, rows: list[str], cols: list[str]) -> pd.DataFrame:
    sub = res[(res["retrieval"] == retrieval) & (res["metric"] == metric) & (res["query_label"] == "(all)")]
    p = sub.pivot_table(index="condition", columns="query_family", values="value", aggfunc="first")
    p = p.reindex(index=[r for r in rows if r in p.index], columns=[c for c in cols if c in p.columns])
    p.index = [COND_LABEL.get(i, i) for i in p.index]
    p.index.name = metric
    return p


def n_table(res: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    sub = res[(res["retrieval"] == "nc_euclid") & (res["query_label"] == "(all)")]
    rows = {}
    for cond, metric, name in [("none", "family_top1", "patho2 (all)"), ("none", "top1", "patho2 (scorable)"),
                               ("ebrains_loo", "top1", "EBRAINS LOO")]:
        s = sub[(sub["condition"] == cond) & (sub["metric"] == metric)].set_index("query_family")["n"]
        rows[name] = s.reindex(cols)
    t = pd.DataFrame(rows).T
    t.index.name = "n"
    return t.astype("Int64")


def write_summary(path: Path, enc: str, args, info: dict, classes: list[str], n_c: np.ndarray, dropped: pd.Series,
                  mapping: dict, correct: dict, q_counts: pd.Series, deltas: pd.DataFrame, res: pd.DataFrame,
                  conf: pd.DataFrame, top1c: pd.DataFrame, emb_check: dict) -> None:
    cols = [c for c in AGG_ORDER if c in set(res["query_family"])]
    rows_order = CONDITIONS + BASELINES
    L = [f"# 参照コホートへの診断名の検索: EBRAINS 全体への最近傍診断（{enc}）", "",
         "`atlas_retrieval.py` の出力。主 = nc_euclid（診断ごとの重心へのユークリッド距離）。条件の定義は docstring。", ""]
    if info.get("restricted_to"):
        L += [f"**参照を {info['restricted_to']} の参照のスライド（atlas_{info['restricted_to']}_slides.csv の uuid）に絞った版**"
              f"（{info['n_loaded']} → {info['n_after_restrict']} 枚。--atlas-slides {info['restricted_to']}）。"
              "問い合わせと Δ の推定は絞っていない。下の h5 の本数などは絞る前の値。", ""]
    L += ["## 参照（EBRAINS）", "",
         f"- h5 {info['n_files']} 本 / uuid {info['n_uuids_with_file']}（入れ子の重複を除く）。annotation.csv {info['n_annotation']} 行、"
         f"control {info['n_control']} を除いて {info['n_noncontrol']}。h5 が無い {len(info['no_file'])}、キーが無い {len(info['no_key'])}"
         f"、読めた {info['n_loaded']} 枚（annotation に無い uuid の h5 {len(info['uuids_without_annotation'])} 本）。",
         f"- --min-n {args.min_n}: 残ったクラス **{len(classes)} 診断 / {int(n_c.sum())} 枚**。落ちたクラス {len(dropped)} 診断 / {int(dropped.sum())} 枚:",
         "  " + "; ".join(f"{d} ({n})" for d, n in dropped.items()),
         f"- 論文の EBRAINS 側（original h5）と参照（EBRAINS_embeddings）の同じスライドの埋め込み: {emb_check['n']} 枚、"
         f"最小 cos {emb_check['min_cos']:.6f}、最大 |差| {emb_check['max_abs']:.2e}。", "",
         "## ラベルの対応（正解の集合）", ""]
    mt = pd.DataFrame([{"label": lab, "n_patho2": int(q_counts.get(lab, 0)),
                        "correct (in reference)": "; ".join(f"{d} ({int(n_c[classes.index(d)])})" for d in correct[lab]) or "—",
                        "dropped (< min-n)": "; ".join(d for d in mapping[lab] if d not in correct[lab]) or ""}
                       for lab in q_counts.index])
    L += ["NOS meningioma は名前に meningioma を含む診断の全部。scorable でないラベル（正解の集合が空）は診断名の指標から外す。", "",
          _md(mt, index=False), "",
          "## 条件の Δ（問い合わせの科ごと）", "",
          "‖Δ‖、Δ_paper との cos、Δ_family（科自身・LOO なし）からのずれ ‖Δ − Δ_family‖ / ‖Δ_family‖。", "",
          _md(deltas, index=False), ""]
    for retrieval, title in [("nc_euclid", "主: 最近傍重心法（ユークリッド）"), ("nc_cosine", "副: 最近傍重心法（cos 距離）"),
                             ("knn5", "副: k 近傍（k = 5、スライド単位の多数決）")]:
        L += [f"## {title}", ""]
        metrics = ["top1", "top5", "family_top1", "family_top1_broad"] + (["mean_rank"] if retrieval != "knn5" else [])
        for metric in metrics:
            L += [f"### {metric}", "", _md(pivot(res, retrieval, metric, rows_order, cols)), ""]
        if retrieval == "nc_euclid":
            L += ["### 例数", "", _md(n_table(res, cols)), ""]
    L += ["## ラベルごと（nc_euclid）", "",
          "EBRAINS LOO はそのラベルの正解の集合に入る EBRAINS のスライド。n = patho2 の scorable / EBRAINS LOO の枚数。", ""]
    lab_rows = res[(res["retrieval"] == "nc_euclid") & (res["query_label"] != "(all)")]
    for metric in ("top1", "top5"):
        lt = lab_rows[lab_rows["metric"] == metric].pivot_table(index="query_label", columns="condition", values="value", aggfunc="first")
        nn = lab_rows[lab_rows["metric"] == metric].pivot_table(index="query_label", columns="condition", values="n", aggfunc="first")
        lt = lt.reindex(index=[lab for lab in q_counts.index if lab in lt.index],
                        columns=[c for c in CONDITIONS + ["ebrains_loo"] if c in lt.columns])
        lt.insert(0, "n", [f"{_fmt(nn.at[lab, 'none'], 0) if 'none' in nn.columns else ''} / "
                           f"{_fmt(nn.at[lab, 'ebrains_loo'], 0) if 'ebrains_loo' in nn.columns and lab in nn.index else ''}"
                           for lab in lt.index])
        lt.columns = [COND_LABEL.get(c, c) for c in lt.columns]
        lt.index.name = metric
        L += [_md(lt), ""]
    L += ["## グリオーマの 3 群の混同（nc_euclid）", "",
          "full = 参照全体の top-1 の診断の群（other glioma = broad だけに入る診断、non-glioma = それ以外）。"
          "restricted = ast・glm・odg の集合の診断の重心だけに絞った 3 択。bal_acc = 3 群の recall の平均。", ""]
    for scope in ("full", "restricted"):
        for cond in CONDITIONS + ["ebrains_loo"]:
            c = conf[(conf["retrieval"] == "nc_euclid") & (conf["condition"] == cond) & (conf["scope"] == scope)]
            if c.empty:
                continue
            t = c.pivot_table(index="true", columns="pred", values="count", aggfunc="sum")
            t = t.reindex(index=GLIOMA_GROUPS, columns=[p for p in GLIOMA_GROUPS + ["other glioma", "non-glioma"] if p in t.columns])
            L += [f"**{scope} / {COND_LABEL[cond]}**（bal_acc {balanced_acc(c):.3f}）", "", _md(t, digits=0), ""]
    L += ["## top-1 の診断の内訳（nc_euclid、グリオーマ・リンパ腫・転移・SFT・NOS、上位 4）", ""]
    tc = top1c[(top1c["retrieval"] == "nc_euclid") & top1c["label"].isin(GLIOMA_GROUPS + ["lym", "met", "SFT", NOS])]
    out = []
    for (lab, cond), g in tc.groupby(["label", "condition"], sort=False):
        g = g.sort_values("count", ascending=False)
        tot = g["count"].sum()
        out.append({"label": lab, "condition": cond, "top-1 (count / n)": "; ".join(f"{r.top1} {r.count}" for r in g.head(4).itertuples()) + f" / {tot}"})
    tdf = pd.DataFrame(out)
    tdf["_o"] = tdf["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    tdf = tdf.sort_values(["label", "_o"]).drop(columns="_o")
    L += [_md(tdf, index=False), ""]
    path.write_text("\n".join(L) + "\n")


# ── figure ────────────────────────────────────────────────────────────────────

def plot(res: pd.DataFrame, path: Path, enc: str, conds: list[str], ceiling: str, note: str = "") -> None:
    fams = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
    sub = res[(res["retrieval"] == "nc_euclid") & (res["query_label"] == "(all)")]
    panels = [("top1", "a  diagnosis top-1"), ("top5", "b  diagnosis top-5"), ("family_top1", "c  family top-1")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    width = 0.8 / len(conds)
    x = np.arange(len(fams))
    for ax, (metric, title) in zip(axes, panels):
        s = sub[sub["metric"] == metric]
        for k, cond in enumerate(conds):
            v = [s.loc[(s["condition"] == cond) & (s["query_family"] == f), "value"].squeeze() if
                 ((s["condition"] == cond) & (s["query_family"] == f)).any() else np.nan for f in fams]
            ax.bar(x - 0.4 + width * (k + 0.5), v, width=width * 0.92, color=COND_COLOR[cond], label=COND_LABEL[cond],
                   edgecolor="white", linewidth=0.5)
        cv = s[s["condition"] == ceiling].set_index("query_family")["value"]
        for i, f in enumerate(fams):
            if f in cv.index and np.isfinite(cv[f]):
                ax.hlines(cv[f], i - 0.44, i + 0.44, color="#0b0b0b", lw=2, label=COND_LABEL[ceiling] if i == 0 else None)
        n = sub[(sub["metric"] == ("family_top1" if metric == "family_top1" else "top1")) & (sub["condition"] == "none")].set_index("query_family")["n"]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f}\n(n={int(n[f])})" if f in n.index else f for f in fams], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("accuracy (in-house slides)")
    h, lab = axes[0].get_legend_handles_labels()
    order = [k for k, t in enumerate(lab) if t != COND_LABEL[ceiling]] + [k for k, t in enumerate(lab) if t == COND_LABEL[ceiling]]
    h, lab = [h[k] for k in order], [lab[k] for k in order]
    fig.legend(h, lab, loc="lower center", ncol=len(lab), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Nearest-centroid diagnosis of in-house slides against the EBRAINS reference cohort ({enc.upper()}{note})", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reload", action="store_true", help="埋め込みのキャッシュを捨てて h5 から読み直す")
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT, help=f"参照に残す診断の最小枚数（既定 {MIN_N_DEFAULT}）")
    ap.add_argument("--encoder", choices=sorted(ENCODERS), default="titan")
    ap.add_argument("--atlas-slides", choices=sorted(ENCODERS), default=None,
                    help="参照をこの encoder の参照のキャッシュ（atlas_<enc>_slides.csv）にあるスライドに絞る"
                         "（TITAN と PRISM を同じ参照で比べる用。出力名は <encoder>_on_<atlas-slides>）")
    args = ap.parse_args()
    enc = args.encoder

    cfg = load_config()
    h5_col, key_name = ENCODERS[enc]
    key = cfg["transfer"][key_name]
    out_dir = Path(cfg["output_dir"]) / "atlas"
    out_dir.mkdir(parents=True, exist_ok=True)
    restrict = args.atlas_slides if args.atlas_slides not in (None, enc) else None
    base = enc if restrict is None else f"{enc}_on_{restrict}"
    tag = base if args.min_n == MIN_N_DEFAULT else f"{base}_min{args.min_n}"   # 出力名（キャッシュは encoder ごとに共通）

    print("[load] reference cohort (EBRAINS_embeddings × annotation.csv)")
    atlas, A, info = load_atlas(cfg, key, out_dir, enc, args.reload)
    if restrict is not None:   # 別の encoder の参照と同じスライドに絞る（並びは自分のキャッシュのまま）
        other_csv = out_dir / f"atlas_{restrict}_slides.csv"
        if not other_csv.exists():
            raise SystemExit(f"{other_csv} が無い。先に --encoder {restrict} を回すこと")
        keep_u = set(pd.read_csv(other_csv, dtype={"uuid": str})["uuid"])
        km = atlas["uuid"].isin(keep_u).to_numpy()
        if km.sum() != len(keep_u):
            print(f"  WARN {restrict} の参照 {len(keep_u)} 枚のうち {int(km.sum())} 枚しか {enc} の参照に無い")
        info = {**info, "restricted_to": restrict, "n_after_restrict": int(km.sum())}
        atlas, A = atlas[km].reset_index(drop=True), A[km]
        print(f"  restricted to the {restrict} reference slides: {info['n_loaded']} → {info['n_after_restrict']}")
    print("[load] manifest (Δ の推定と patho2 の問い合わせ)")
    man, M = load_manifest(cfg, h5_col, key, out_dir, enc, args.reload)

    # 参照のクラス
    counts = atlas["diagnosis"].value_counts()
    classes = sorted(counts[counts >= args.min_n].index, key=lambda d: (-counts[d], d))
    dropped = counts[counts < args.min_n].sort_values(ascending=False)
    keep = atlas["diagnosis"].isin(classes).to_numpy()
    ref = atlas[keep].reset_index(drop=True)
    R = A[keep]
    cpos = {c: k for k, c in enumerate(classes)}
    y_idx = ref["diagnosis"].map(cpos).to_numpy()
    C = np.stack([R[y_idx == k].mean(0) for k in range(len(classes))])
    n_c = np.bincount(y_idx, minlength=len(classes)).astype(float)
    print(f"  reference: {len(classes)} diagnoses / {len(ref)} slides (min-n {args.min_n}); dropped {len(dropped)} / {int(dropped.sum())}")

    # ラベルの対応
    mapping = label_mapping(cfg, set(atlas["diagnosis"]))
    correct = {lab: [d for d in ds if d in cpos] for lab, ds in mapping.items()}
    fam_of_class = np.array([diagnosis_family(c, mapping) for c in classes] + [""], dtype=object)
    famb_of_class = np.array([diagnosis_family_broad(c, mapping) for c in classes] + [""], dtype=object)

    # 問い合わせ（patho2）
    qmask = (man["site"] == "patho2").to_numpy()
    q = man[qmask].reset_index(drop=True).copy()
    Xq = M[qmask]
    sub_csv = pd.read_csv(cfg["label"]["patho2"]["subtype"], dtype=str).set_index("case_id")["subtype"]
    paper_q = q["family"].isin(["meningioma", "SFT"])
    if set(q.loc[paper_q, "slide_id"]) != set(sub_csv.index):
        print(f"  WARN manifest の論文コホート（patho2 {int(paper_q.sum())}）と {cfg['label']['patho2']['subtype']}（{len(sub_csv)}）が一致しない")
    q["label"] = np.where(paper_q, q["slide_id"].map(sub_csv).fillna(q["group"]), q["group"])
    if (q.loc[paper_q, "label"] != q.loc[paper_q, "group"]).any():
        print("  WARN manifest の group とラベル CSV の subtype が食い違う行がある（ラベル CSV を使う）")
    for lab in sorted(set(q["label"]) - set(mapping)):
        print(f"  WARN label '{lab}' に対応する EBRAINS の診断が無い（scorable = False）")
        mapping[lab], correct[lab] = [], []
    q["scorable"] = q["label"].map(lambda lab: len(correct[lab]) > 0)
    fam_q = q["family"].to_numpy()
    q_counts = q["label"].value_counts()
    q_counts = q_counts.reindex(sorted(q_counts.index, key=lambda lab: (FAMILIES.index(q.loc[q["label"] == lab, "family"].iat[0]), lab)))
    print(f"  queries: {len(q)} patho2 slides; not scorable: {q.loc[~q['scorable'], 'label'].value_counts().to_dict()}")

    # 論文の EBRAINS 側（original h5）と参照の同じスライドの一致
    eb = man[(man["site"] == "ebrains")].reset_index()
    apos = dict(zip(atlas["slide_id"], range(len(atlas))))
    pairs = [(i, apos[s]) for i, s in zip(eb["index"], eb["slide_id"]) if s in apos]
    Pm, Pa = M[[i for i, _ in pairs]], A[[j for _, j in pairs]]
    emb_check = {"n": len(pairs), "min_cos": float(((Pm * Pa).sum(1) / np.linalg.norm(Pm, axis=1) / np.linalg.norm(Pa, axis=1)).min()),
                 "max_abs": float(np.abs(Pm - Pa).max())}

    # 条件の Δ
    sh = Shifts(man, M)
    d_paper = sh.pooled(["meningioma", "SFT"])
    drows = []
    for f in FAMILIES:
        rest = [g for g in FAMILIES if g != f]
        d_fam = sh.pooled([f])
        for cond, d in [("paper", d_paper), ("others", sh.pooled(rest)), ("others_bal", sh.balanced(rest)), ("family", d_fam)]:
            drows.append(dict(query_family=f, condition=cond, norm=float(np.linalg.norm(d)),
                              cos_vs_paper=float(d @ d_paper / np.linalg.norm(d) / np.linalg.norm(d_paper)),
                              dev_from_family=float(np.linalg.norm(d - d_fam) / np.linalg.norm(d_fam)),
                              n_ebrains=sum(sh.N["ebrains", g] for g in ({"paper": ["meningioma", "SFT"], "family": [f]}.get(cond, rest))),
                              n_patho2=sum(sh.N["patho2", g] for g in ({"paper": ["meningioma", "SFT"], "family": [f]}.get(cond, rest)))))
    deltas = pd.DataFrame(drows)
    print(f"  |Δ_paper| = {np.linalg.norm(d_paper):.3f}")

    corr_q = corr_mask(q["label"].to_numpy(), classes, correct)
    per_slide, long_rows, conf_rows = [], [], []

    # 問い合わせ × 条件
    for cond in CONDITIONS:
        Q = Xq + sh.for_queries(cond, fam_q, Xq)
        Ds = {"nc_euclid": _dist(Q, C, "euclid"), "nc_cosine": _dist(Q, C, "cosine")}
        Dsl = _dist(Q, R, "euclid")
        for retrieval in RETRIEVALS:
            res = knn_retrieve(Dsl, y_idx, corr_q, len(classes)) if retrieval == "knn5" else nc_retrieve(Ds[retrieval], corr_q)
            ps = per_slide_frame(q, res, classes, fam_of_class, famb_of_class, retrieval, cond)
            per_slide.append(ps)
            long_rows += aggregate(ps, enc)
            conf_rows += glioma_confusion(ps, Ds.get(retrieval), classes, mapping, correct)

    # EBRAINS 内の基準: 問い合わせのラベル（NOS を除く）の正解の集合に入るスライド
    label_of_diag = {d: lab for lab in q_counts.index if lab != NOS for d in correct[lab]}
    fam_of_label = q.groupby("label")["family"].first().to_dict()
    sel = np.where(ref["diagnosis"].isin(label_of_diag).to_numpy())[0]
    bmeta = ref.iloc[sel][["slide_id", "diagnosis"]].copy()
    bmeta["site"] = "ebrains"
    bmeta["label"] = bmeta["diagnosis"].map(label_of_diag)
    bmeta["family"] = bmeta["label"].map(fam_of_label)
    bmeta["scorable"] = True
    bmeta = bmeta.reset_index(drop=True)
    corr_b = corr_mask(bmeta["label"].to_numpy(), classes, correct)
    n_lab_e = bmeta["label"].value_counts()
    w_mix = (bmeta["label"].map(q_counts) / bmeta["label"].map(n_lab_e)).to_numpy(float)   # Σ_l n_P,l·v_l / Σ_l n_P,l になる重み
    patient = ref["patient"].to_numpy()
    Dsl_full = _dist(R[sel], R, "euclid")
    for base, by_patient in [("ebrains_loo", False), ("ebrains_lpo", True)]:
        Ds = {"nc_euclid": loo_distances(R, y_idx, patient, sel, C, n_c, "euclid", by_patient),
              "nc_cosine": loo_distances(R, y_idx, patient, sel, C, n_c, "cosine", by_patient)}
        Dsl = Dsl_full.copy()
        if by_patient:
            Dsl[patient[sel][:, None] == patient[None, :]] = np.inf
        else:
            Dsl[np.arange(len(sel)), sel] = np.inf
        for retrieval in RETRIEVALS:
            res = knn_retrieve(Dsl, y_idx, corr_b, len(classes)) if retrieval == "knn5" else nc_retrieve(Ds[retrieval], corr_b)
            ps = per_slide_frame(bmeta, res, classes, fam_of_class, famb_of_class, retrieval, base)
            per_slide.append(ps)
            long_rows += aggregate(ps, enc)
            conf_rows += glioma_confusion(ps, Ds.get(retrieval), classes, mapping, correct)
            if base == "ebrains_loo":
                long_rows += aggregate(ps.assign(condition="ebrains_loo_mix"), enc, weights=w_mix)

    res_df = pd.DataFrame(long_rows)
    ps_df = pd.concat(per_slide, ignore_index=True)
    conf = pd.DataFrame(conf_rows)
    top1c = (ps_df[ps_df["site"] == "patho2"].groupby(["retrieval", "condition", "label", "top1"]).size()
             .rename("count").reset_index())
    mtab = pd.DataFrame([{"label": lab, "family": fam_of_label.get(lab, ""), "n_patho2": int(q_counts.get(lab, 0)),
                          "correct_in_atlas": "; ".join(correct[lab]),
                          "dropped_lt_min_n": "; ".join(d for d in mapping[lab] if d not in correct[lab])} for lab in q_counts.index])

    res_df.to_csv(out_dir / f"retrieval_{tag}.csv", index=False)
    ps_df.to_csv(out_dir / f"per_slide_{tag}.csv", index=False)
    conf.to_csv(out_dir / f"glioma_confusion_{tag}.csv", index=False)
    top1c.to_csv(out_dir / f"top1_counts_{tag}.csv", index=False)
    deltas.to_csv(out_dir / f"deltas_{tag}.csv", index=False)
    mtab.to_csv(out_dir / f"mapping_{tag}.csv", index=False)
    write_summary(out_dir / f"retrieval_{tag}_summary.md", enc, args, info, classes, n_c, dropped, mapping, correct,
                  q_counts, deltas, res_df, conf, top1c, emb_check)
    plot(res_df, out_dir / f"retrieval_{tag}.png", enc, ["none", "paper", "others_bal", "family"], "ebrains_loo",
         note="" if restrict is None else f", reference restricted to the {restrict.upper()} slides")

    pd.set_option("display.width", 220)
    cols = [c for c in AGG_ORDER if c in set(res_df["query_family"])]
    for metric in ("top1", "top5", "family_top1"):
        print(f"\n[nc_euclid] {metric}")
        print(pivot(res_df, "nc_euclid", metric, CONDITIONS + BASELINES, cols).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
