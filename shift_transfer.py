"""shift_transfer.py — 施設間シフト Δ の向きは腫瘍の科（髄膜腫 / グリオーマ / リンパ腫 / 転移）を越えて同じか（予備解析）。

論文のコホート（髄膜腫 + SFT、n551 / n577、TITAN、補正前）で推定した平行移動 Δ_paper と、同じ 2 施設の別の腫瘍
（院内 Patho2Gliomas の代表 HE 1 症例 1 枚 × EBRAINS の対応する診断）のシフト Δ_g の向きと長さを比べる。
    shift = mean(EBRAINS) − mean(patho2)   （patho2 → EBRAINS の向き。shift_direction.py と同じ）

群:
    髄膜腫側: Paper cohort（論文の全例 = centroid.py の Δ）/ Meningioma (all) / SFT / 両施設で subtype_min_n 以上の髄膜腫サブタイプ
    グリオーマ側: ast / glm / odg / Glioma (all = 3 群) / lym / met。院内の群 → EBRAINS の診断名は config の shift_transfer.ebrains_groups

見るもの:
    1. 群ごとの cos(Δ_g, Δ_paper)。論文コホートの部分集合（髄膜腫サブタイプ・SFT）は自分を除いた Δ_paper（leave-group-out）と比べる
    2. 素な群どうしの cos の行列。例数が少ない群ほど推定ノイズで cos が下がるので、ノイズを差し引いた cos も出す:
         E‖Δ̂‖² = ‖Δ‖² + tr V,   tr V = Σ_d ( s²_ebrains,d / n_ebrains + s²_patho2,d / n_patho2 )
         cos_corr(g, h) = Δ̂_g·Δ̂_h / sqrt( (‖Δ̂_g‖² − tr V_g)(‖Δ̂_h‖² − tr V_h) )   （g と h に共通のスライドが無いとき）
    3. 対照: 施設内の科どうしの差（グリオーマ − 髄膜腫 など）と Δ の cos（シフトが生物学的な差と直交しているか）、
       その差の向きが両施設で一致するか
    4. Δ_paper をそのまま当てたときに残る施設差 ‖Δ_g − Δ_paper‖ / ‖Δ_g‖
    5. 平行四辺形の図: 素な 12 群の両施設の重心（24 点）を 2 次元に射影する。x = Δ_paper の単位ベクトル、
       y = 24 重心から Δ_paper 成分を除いた群間変動の第 1 主成分。各群に patho2 → EBRAINS の矢印を引く。
       施設差が腫瘍によらなければ、矢印は全部平行で同じ長さ
    6. 科を食い違わせた差: cos( mean(EBRAINS の A) − mean(patho2 の B), Δ_paper )（5 科 × 5 科、対角 = 揃えた差）。
       mean(E_A) − mean(P_B) = b_AB + Δ_B（b_AB = 施設内の A と B の平均の差）なので、施設差が科の違いと直交し
       Δ_B = Δ_paper なら、A ≠ B の予想値は ‖Δ_paper‖ / sqrt(‖Δ_paper‖² + ‖b_AB‖²)（‖b_AB‖ は 2 施設の平均）。
       patho2 の全スライドに Δ_paper を足した後（補正後）も同じ行列を出す（同じ模型の予想は非対角 0。
       B 自身の残りのシフト r_B = Δ_B − Δ_paper を入れた予想 (r_B·û) / sqrt(‖b_AB‖² + ‖r_B‖²) も出す）
    7. 組成の感度: グリオーマの群の EBRAINS 側の組成（グレードの混じり方、WHO 2016 と CNS5 の違い）を変えたときの
       cos vs Δ_paper。星細胞腫 IDH 変異型をグレード II–III / IV に分ける、膠芽腫 IDH 野生型に IDH 野生型の
       （退形成性）星細胞腫を足す（≈ CNS5 の膠芽腫）、など。院内（patho2）側は変えない

出力 (output_dir/shift_transfer/):
    groups.csv         — 群ごとの n（枚数と患者数）・|shift|（生 / ノイズ補正）・cos vs Δ_paper と vs Δ_meni（生・95% ブートストラップ CI・補正）・残差
    groups_patients.csv — --bootstrap patients のときの groups.csv（CI だけ患者単位の再抽出で取り直す。groups.csv は上書きしない）
    cos_matrix.csv     — 素な群どうしの cos（raw と corrected、縦長）
    bio_control.csv    — 施設内の科どうしの差と Δ の cos
    shift_transfer.png — a: cos 行列（上三角 raw / 下三角 corrected）  b: |shift|
    arrows.csv / arrows.png — 平行四辺形の図（2 次元の座標・矢印の角度と長さ・Δ_paper からの残差。1 行目は Paper cohort 自身）
    mismatch_matrix.csv / mismatch_matrix.png — 科を食い違わせた差の cos（観測と予想、補正前 / 補正後）
    composition.csv    — 組成の感度（変種ごとの cos。CI はスライド単位と患者単位の両方。乱数は compute と別の列）
    paper_slides.csv / paper_embeddings.npy / glioma_slides.csv / glioma_embeddings.npy — 読み込んだ埋め込みのキャッシュ（--reload で読み直す）
    extra_slides.csv / extra_embeddings.npy — 組成の感度に使う EBRAINS の IDH 野生型（退形成性）星細胞腫のキャッシュ

スライド埋め込み（--encoder）:
    titan     （既定）TITAN（{tile_model}/{keys.slide_feature}）。出力は上のとおり output_dir/shift_transfer/
    meanpool  学習なしの集約器の対照: CONCH v1.5 のパッチ埋め込みの単純平均（meanpool.py と同じ定義、768 次元）。
              論文コホートは meanpool.py の h5（load_data(..., aggregator="mean")）、グリオーマ側は transfer_meanpool.py の
              キャッシュ（out/transfer/embeddings_meanpool.npy、先に transfer_meanpool.py build）を (施設, slide_id) で引く。
              キャッシュに無いスライド（組成の感度だけに使う EBRAINS の IDH 野生型星細胞腫）だけ h5 の features から平均を取る。
              出力は同じファイル名で output_dir/shift_transfer/meanpool/ に書く（TITAN の出力には触らない）。
              スライドの集合と並びは TITAN と同じ

Usage:
    uv run python shift_transfer.py [--encoder titan|meanpool] [--reload] [--bootstrap slides|patients]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.display import ordered_subtypes
from utils.loader import load_config, load_data

REF = "Paper cohort"          # 論文の全例（髄膜腫 + SFT）= centroid.py が使う Δ
MENI = "Meningioma (all)"
SFT = "SFT"
GLIOMA_ALL = "Glioma (all)"
SITES = ("ebrains", "patho2")
FAMILY_ORDER = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
FAMILY_COLOR = {"meningioma": "#4c72b0", "SFT": "#000000", "glioma": "#dd8452", "lymphoma": "#55a868", "metastasis": "#c44e52"}
# 組成の感度で glm に足す EBRAINS の診断名（config の ebrains_groups には入れていない = 本解析の glm には入らない）
COMPOSITION_EXTRA = ["Anaplastic astrocytoma, IDH-wildtype", "Diffuse astrocytoma, IDH-wildtype"]
GBM_IDHMUT = "Glioblastoma, IDH-mutant"   # WHO 2016 では膠芽腫、CNS5 では星細胞腫 IDH 変異型 grade 4（config では ast に入れている）
ENCODER_TAG = {"titan": "", "meanpool": "  [mean-pooled CONCH v1.5]"}   # 図の題に付ける（TITAN は付けない = 前の図と同じ）


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── data ──────────────────────────────────────────────────────────────────────

class SlideEmbedder:
    """h5 1 枚 → スライド埋め込み（float32）。
    titan:    {tile_model}/{keys.slide_feature} を読む（前と同じ）。
    meanpool: transfer_meanpool.py のキャッシュに (施設, slide_id) があればそれを引き（h5 は開かない）、無ければ
              {tile_model}/{keys.tile_features} のパッチ方向の平均を取る（meanpool._mean_features、float64 → float32）。"""

    def __init__(self, cfg: dict, encoder: str):
        self.cfg, self.encoder = cfg, encoder
        self.cache: dict[tuple[str, str], np.ndarray] = {}
        self.n_cached = self.n_computed = 0
        if encoder == "meanpool":
            from meanpool import _mean_features
            from transfer_meanpool import load_cache
            self._mean_features = _mean_features
            man, X, _ = load_cache(cfg)
            self.cache = {(s, i): X[k].astype(np.float32) for k, (s, i) in enumerate(zip(man["site"], man["slide_id"]))}

    def key(self, tile_model: str) -> str:
        return f"{tile_model}/{self.cfg['keys']['slide_feature' if self.encoder == 'titan' else 'tile_features']}"

    def __call__(self, path, key: str, site: str, slide_id: str) -> np.ndarray | None:
        """埋め込み。h5 に key が無ければ None。"""
        if self.encoder == "meanpool" and (site, slide_id) in self.cache:
            self.n_cached += 1
            return self.cache[(site, slide_id)]
        with h5py.File(path, "r") as f:
            if key not in f:
                return None
            if self.encoder == "titan":
                return f[key][:]
            self.n_computed += 1
            return self._mean_features(f, key)[0].astype(np.float32)


def _read_cache(csv_path: Path, npy_path: Path) -> pd.DataFrame | None:
    if not (csv_path.exists() and npy_path.exists()):
        return None
    df = pd.read_csv(csv_path, dtype={"patient": str, "slide_id": str})
    df["embedding"] = list(np.load(npy_path))
    return df


def _write_cache(df: pd.DataFrame, csv_path: Path, npy_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns="embedding").to_csv(csv_path, index=False)
    np.save(npy_path, np.stack(df["embedding"].values).astype(np.float32))


def load_paper(cfg: dict, out_dir: Path, reload: bool, encoder: str = "titan") -> pd.DataFrame:
    """論文のコホート（embedding.*.original の TITAN、meanpool は meanpool.py の平均）。列: slide_id, source, subtype, patient, embedding。"""
    csv_path, npy_path = out_dir / "paper_slides.csv", out_dir / "paper_embeddings.npy"
    if not reload and (df := _read_cache(csv_path, npy_path)) is not None:
        return df
    df = load_data(cfg, "original", aggregator="mean" if encoder == "meanpool" else "titan").rename(columns={"case_id": "slide_id"})
    df["patient"] = df["slide_id"]
    for site in SITES:
        pat_csv = cfg["label"].get(site, {}).get("patient")
        if pat_csv and Path(pat_csv).exists():
            pat = pd.read_csv(pat_csv, dtype=str).rename(columns={"case_id": "slide_id", "patient_id": "patient"})
            m = df["source"] == site
            df.loc[m, "patient"] = df.loc[m, "slide_id"].map(pat.set_index("slide_id")["patient"]).fillna(df.loc[m, "slide_id"])
    df = df[["slide_id", "source", "subtype", "patient", "embedding"]]
    _write_cache(df, csv_path, npy_path)
    return df


def load_glioma(cfg: dict, out_dir: Path, reload: bool, embed: SlideEmbedder | None = None) -> pd.DataFrame:
    """Patho2Gliomas の代表 HE（1 症例 1 枚、representatives.csv）と EBRAINS の対応診断の埋め込み（既定 TITAN）。列: slide_id, source, group, patient, diagnosis, embedding。"""
    csv_path, npy_path = out_dir / "glioma_slides.csv", out_dir / "glioma_embeddings.npy"
    if not reload and (df := _read_cache(csv_path, npy_path)) is not None:
        return df
    st = cfg["shift_transfer"]
    pg = cfg["patho2_gliomas"]
    rows: list[dict] = []
    embed = embed or SlideEmbedder(cfg, "titan")

    key = embed.key(pg["tile_model"])
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv")
    reps = pd.read_csv(pg["representatives"])
    reps = reps[reps["note"] != "exclude"]
    merged = reps.merge(slides[["label", "case", "file", "h5"]], on=["case", "file"], how="left")
    for r in merged.itertuples(index=False):
        if not isinstance(r.h5, str) or not Path(r.h5).exists():
            print(f"  WARN [patho2] h5 not found: {r.case} / {r.file}")
            continue
        emb = embed(r.h5, key, "patho2", r.case)
        if emb is None:
            print(f"  WARN [patho2] no '{key}': {r.h5}")
            continue
        rows.append({"slide_id": r.case, "source": "patho2", "group": r.label, "patient": r.case, "diagnosis": r.label, "embedding": emb})

    diag2group = {d: g for g, ds in st["ebrains_groups"].items() for d in ds}
    rows += _ebrains_rows(cfg, diag2group, embed)
    df = pd.DataFrame(rows)
    _write_cache(df, csv_path, npy_path)
    return df


def _ebrains_rows(cfg: dict, diag2group: dict[str, str], embed: SlideEmbedder | None = None) -> list[dict]:
    """EBRAINS の annotation.csv で diagnosis が diag2group にあるスライド（control は除外）の埋め込み（既定 TITAN）。slide_id は uuid の先頭 8 文字。"""
    ann = pd.read_csv(cfg["shift_transfer"]["ebrains_annotation"])
    ann = ann[ann["control"] != 1]
    root = Path(cfg["ebrains_embeddings"]["root"])
    embed = embed or SlideEmbedder(cfg, "titan")
    ekey = embed.key(cfg["ebrains_embeddings"]["tile_model"])
    index: dict[str, Path] = {}   # uuid → h5。「… 1p/19q codeleted」は親 "… and 1p/" と入れ子 "19q codeleted/" に散らばっているので、木全体から引く（浅い方を優先）
    for p in sorted(root.rglob("*_hdf.h5"), key=lambda q: len(q.parts)):
        index.setdefault(p.name[: -len("_hdf.h5")], p)
    rows: list[dict] = []
    for r in ann[ann["diagnosis"].isin(diag2group)].itertuples(index=False):
        p = index.get(r.uuid)
        if p is None:
            print(f"  WARN [ebrains] h5 not found: {r.uuid} ({r.diagnosis})")
            continue
        emb = embed(p, ekey, "ebrains", r.uuid[:8])
        if emb is None:
            print(f"  WARN [ebrains] no '{ekey}': {p}")
            continue
        rows.append({"slide_id": r.uuid[:8], "source": "ebrains", "group": diag2group[r.diagnosis], "patient": str(r.pat_id),
                     "diagnosis": r.diagnosis, "embedding": emb})
    return rows


def load_extra(cfg: dict, out_dir: Path, reload: bool, embed: SlideEmbedder | None = None) -> pd.DataFrame:
    """組成の感度だけに使う EBRAINS の IDH 野生型（退形成性）星細胞腫（COMPOSITION_EXTRA）の埋め込み（既定 TITAN）。列は load_glioma と同じ（group = "glm_extra"）。"""
    csv_path, npy_path = out_dir / "extra_slides.csv", out_dir / "extra_embeddings.npy"
    if not reload and (df := _read_cache(csv_path, npy_path)) is not None:
        return df
    df = pd.DataFrame(_ebrains_rows(cfg, {d: "glm_extra" for d in COMPOSITION_EXTRA}, embed))
    _write_cache(df, csv_path, npy_path)
    return df


def ebrains_grades(cfg: dict) -> pd.Series:
    """uuid の先頭 8 文字（キャッシュの slide_id）→ WHO グレード（annotation.csv の grade）。先頭 8 文字は annotation.csv の中で一意。"""
    ann = pd.read_csv(cfg["shift_transfer"]["ebrains_annotation"], dtype=str)
    key = ann["uuid"].str[:8]
    if not key.is_unique:
        raise ValueError("uuid の先頭 8 文字が annotation.csv の中で一意でない")
    return pd.Series(ann["grade"].values, index=key.values)


# ── groups ────────────────────────────────────────────────────────────────────

def _xs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {s: np.stack(df.loc[df["source"] == s, "embedding"].values).astype(np.float64) for s in SITES}


def _npat(df: pd.DataFrame) -> dict[str, int]:
    return {s: int(df.loc[df["source"] == s, "patient"].nunique()) for s in SITES}


def _ps(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """施設 → 患者 ID（_xs と同じ行の並び）。患者単位のブートストラップ用。"""
    return {s: df.loc[df["source"] == s, "patient"].astype(str).values for s in SITES}


def build_groups(paper: pd.DataFrame, glioma: pd.DataFrame, cfg: dict) -> list[dict]:
    """群ごとに name / family / kind / X（施設 → (n, d)）/ P（施設 → 患者 ID）/ n_pat / mask（論文コホートの部分集合なら paper 上の bool マスク）。"""
    st = cfg["shift_transfer"]
    min_n = int(st.get("subtype_min_n", 20))
    names: dict = st.get("group_names", {})
    fams: dict = st.get("families", {})
    is_sft = paper["subtype"].str.lower().str.contains("sft").values

    def grp(name: str, family: str, kind: str, d: pd.DataFrame, mask: np.ndarray | None) -> dict:
        return dict(name=name, family=family, kind=kind, X=_xs(d), P=_ps(d), n_pat=_npat(d), mask=mask)

    groups = [
        grp(REF, "meningioma", "paper", paper, np.ones(len(paper), bool)),
        grp(MENI, "meningioma", "paper", paper[~is_sft], ~is_sft),
        grp(SFT, "SFT", "sft", paper[is_sft], is_sft),
    ]
    for sub in ordered_subtypes(set(paper.loc[~is_sft, "subtype"]), cfg):
        m = (paper["subtype"] == sub).values
        n = {s: int((m & (paper["source"] == s).values).sum()) for s in SITES}
        if min(n.values()) >= min_n:
            groups.append(grp(sub, "meningioma", "subtype", paper[m], m))
    for g in ("ast", "glm", "odg"):
        d = glioma[glioma["group"] == g]
        groups.append(grp(names.get(g, g), fams.get(g, "glioma"), "glioma", d, None))
    d = glioma[glioma["group"].isin(["ast", "glm", "odg"])]
    groups.append(grp(GLIOMA_ALL, "glioma", "glioma_all", d, None))
    for g in ("lym", "met"):
        d = glioma[glioma["group"] == g]
        if len(d) and all((d["source"] == s).any() for s in SITES):
            groups.append(grp(names.get(g, g), fams.get(g, g), "glioma", d, None))
    return groups


# ── shift statistics ──────────────────────────────────────────────────────────

def shift_stats(X: dict[str, np.ndarray]) -> tuple[np.ndarray, float, float, float]:
    """Δ̂ = mean(EBRAINS) − mean(patho2)、‖Δ̂‖、ノイズ分散の和 tr V、ノイズを差し引いた ‖Δ‖。"""
    e, p = X["ebrains"], X["patho2"]
    delta = e.mean(0) - p.mean(0)
    tr_v = float((e.var(0, ddof=1) / len(e)).sum() + (p.var(0, ddof=1) / len(p)).sum())
    n2 = float(delta @ delta)
    return delta, float(np.sqrt(n2)), tr_v, float(np.sqrt(max(n2 - tr_v, 0.0)))


def corrected_cos(dg: np.ndarray, ng: float, dh: np.ndarray, nh: float) -> float:
    """ノイズを差し引いた長さ ng, nh で正規化した cos（素な群どうし）。長さが 0 に潰れたら NaN。"""
    if ng <= 0 or nh <= 0:
        return float("nan")
    return float(np.clip(dg @ dh / (ng * nh), -1.0, 1.0))


def _patient_sums(X: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """患者ごとのスライドの和と枚数（患者単位の再抽出で、選んだ患者のスライドを全部入れた平均を作るため）。"""
    codes, uniq = pd.factorize(P)
    sums = np.zeros((len(uniq), X.shape[1]))
    np.add.at(sums, codes, X)
    return sums, np.bincount(codes, minlength=len(uniq)).astype(np.float64)


def boot_cos(Xg: dict[str, np.ndarray], Xr: dict[str, np.ndarray], rng: np.random.Generator, n_boot: int,
             Pg: dict[str, np.ndarray] | None = None, Pr: dict[str, np.ndarray] | None = None) -> tuple[float, float]:
    """群と参照の両方を施設内で再抽出したときの cos の 95% 区間。
    Pg / Pr（施設 → 患者 ID）を渡すと患者単位（施設・群ごとに患者を復元抽出し、その患者のスライドを全部入れる）、
    渡さなければスライド単位（既定。groups.csv の数字はこちら。乱数の消費順も前のまま）。"""
    vals = np.empty(n_boot)
    if Pg is None or Pr is None:
        for b in range(n_boot):
            d = []
            for X in (Xg, Xr):
                e, p = X["ebrains"], X["patho2"]
                d.append(e[rng.integers(0, len(e), len(e))].mean(0) - p[rng.integers(0, len(p), len(p))].mean(0))
            vals[b] = _cos(d[0], d[1])
    else:
        tabs = [{s: _patient_sums(X[s], P[s]) for s in SITES} for X, P in ((Xg, Pg), (Xr, Pr))]
        for b in range(n_boot):
            d = []
            for T in tabs:
                m = {}
                for s in SITES:
                    sums, cnt = T[s]
                    idx = rng.integers(0, len(cnt), len(cnt))
                    m[s] = sums[idx].sum(0) / cnt[idx].sum()
                d.append(m["ebrains"] - m["patho2"])
            vals[b] = _cos(d[0], d[1])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def reference_for(group: dict, paper: pd.DataFrame, ref_mask: np.ndarray) -> pd.DataFrame | None:
    """参照（Paper cohort または Meningioma (all)）のスライド。group が論文コホートの部分集合なら、その群のスライドを除く（leave-group-out）。"""
    keep = ref_mask if group["mask"] is None else ref_mask & ~group["mask"]
    if not keep.any():
        return None
    sub = paper[keep]
    if any(not (sub["source"] == s).any() for s in SITES):
        return None
    return sub


def compute(groups: list[dict], paper: pd.DataFrame, cfg: dict, bootstrap: str = "slides") -> tuple[pd.DataFrame, pd.DataFrame]:
    st = cfg["shift_transfer"]
    rng = np.random.default_rng(int(st.get("seed", 42)))
    n_boot = int(st.get("n_boot", 1000))
    ref_defs = {  # 列名の接尾辞 → (参照群の名前, paper 上のマスク)
        "paper": (REF, np.ones(len(paper), bool)),
        "meni": (MENI, ~paper["subtype"].str.lower().str.contains("sft").values),
    }
    for g in groups:
        g["delta"], g["norm"], g["tr_v"], g["norm_corr"] = shift_stats(g["X"])

    rows = []
    for g in groups:
        row = dict(group=g["name"], family=g["family"], kind=g["kind"],
                   n_ebrains=len(g["X"]["ebrains"]), n_patho2=len(g["X"]["patho2"]),
                   n_pat_ebrains=g["n_pat"]["ebrains"], n_pat_patho2=g["n_pat"]["patho2"],
                   shift_norm=g["norm"], shift_norm_corrected=g["norm_corr"], noise_trace=g["tr_v"])
        for suffix, (ref_name, ref_mask) in ref_defs.items():
            if g["kind"] == "paper":   # Paper cohort / Meningioma (all) 自身は参照なので比べない
                for k in ("cos", "cos_lo", "cos_hi", "cos_corrected", "residual"):
                    row[f"{k}_vs_{suffix}"] = np.nan
                row[f"reference_{suffix}"] = ""
                continue
            sub = reference_for(g, paper, ref_mask)
            if sub is None:
                continue
            Xr = _xs(sub)
            dr, nr, _, nr_corr = shift_stats(Xr)
            if bootstrap == "patients":
                lo, hi = boot_cos(g["X"], Xr, rng, n_boot, g["P"], _ps(sub))
            else:
                lo, hi = boot_cos(g["X"], Xr, rng, n_boot)
            row[f"cos_vs_{suffix}"] = _cos(g["delta"], dr)
            row[f"cos_lo_vs_{suffix}"], row[f"cos_hi_vs_{suffix}"] = lo, hi
            row[f"cos_corrected_vs_{suffix}"] = corrected_cos(g["delta"], g["norm_corr"], dr, nr_corr)
            row[f"residual_vs_{suffix}"] = float(np.linalg.norm(g["delta"] - dr) / g["norm"])
            row[f"reference_{suffix}"] = ref_name + (" (leave-group-out)" if g["mask"] is not None else "")
        rows.append(row)
    df_groups = pd.DataFrame(rows)

    disjoint = [g for g in groups if g["kind"] in ("subtype", "sft", "glioma")]
    pairs = []
    for i, a in enumerate(disjoint):
        for b in disjoint[i + 1:]:
            pairs.append(dict(group_a=a["name"], family_a=a["family"], group_b=b["name"], family_b=b["family"],
                              same_family=a["family"] == b["family"],
                              cos=_cos(a["delta"], b["delta"]),
                              cos_corrected=corrected_cos(a["delta"], a["norm_corr"], b["delta"], b["norm_corr"])))
    return df_groups, pd.DataFrame(pairs)


def _family_frames(paper: pd.DataFrame, glioma: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """5 科（FAMILY_ORDER）→ スライド。meningioma = 論文コホートの SFT 以外、glioma = ast + glm + odg。"""
    is_sft = paper["subtype"].str.lower().str.contains("sft").values
    return {
        "meningioma": paper[~is_sft], "SFT": paper[is_sft],
        "glioma": glioma[glioma["group"].isin(["ast", "glm", "odg"])],
        "lymphoma": glioma[glioma["group"] == "lym"], "metastasis": glioma[glioma["group"] == "met"],
    }


def bio_control(paper: pd.DataFrame, glioma: pd.DataFrame, groups: list[dict]) -> pd.DataFrame:
    """施設内の科どうしの差の向きと、施設間シフト（Δ_paper・Δ_glioma）の cos。両施設での向きの一致も。"""
    fam_df = _family_frames(paper, glioma)
    means = {s: {f: np.stack(d.loc[d["source"] == s, "embedding"].values).mean(0) for f, d in fam_df.items() if (d["source"] == s).any()} for s in SITES}
    d_paper = next(g["delta"] for g in groups if g["name"] == REF)
    d_glioma = next(g["delta"] for g in groups if g["name"] == GLIOMA_ALL)
    axes = [("glioma", "meningioma"), ("lymphoma", "meningioma"), ("metastasis", "meningioma"), ("SFT", "meningioma"),
            ("metastasis", "glioma"), ("lymphoma", "glioma")]
    rows = []
    for a, b in axes:
        vec = {s: means[s][a] - means[s][b] for s in SITES if a in means[s] and b in means[s]}
        if len(vec) < 2:
            continue
        for s, v in vec.items():
            rows.append(dict(axis=f"{a} − {b}", site=s, norm=float(np.linalg.norm(v)),
                             cos_vs_shift_paper=_cos(v, d_paper), cos_vs_shift_glioma=_cos(v, d_glioma),
                             cos_between_sites=_cos(vec["ebrains"], vec["patho2"])))
    return pd.DataFrame(rows)


def _deg(c: float) -> float:
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))) if np.isfinite(c) else float("nan")


def arrows(groups: list[dict], paper: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """平行四辺形の図の座標。素な 12 群の両施設の重心（24 点）を、x = Δ_paper の単位ベクトル、
    y = 24 重心（中心化）から Δ_paper 成分を除いた群間変動の第 1 主成分、の平面に射影する。原点は 24 重心の平均。
    残差は ‖Δ_g − Δ_paper‖（Δ_paper は論文の全例）。ノイズ補正の残差は、部分集合なら自分を除いた Δ_paper（leave-group-out）と比べて
    両方の推定ノイズ tr V を差し引く（sqrt(max(‖Δ_g − Δ_lgo‖² − tr V_g − tr V_lgo, 0))）。"""
    ref = next(g for g in groups if g["name"] == REF)
    disjoint = [g for g in groups if g["kind"] in ("subtype", "sft", "glioma")]
    d_paper = ref["delta"]
    ux = d_paper / ref["norm"]
    C = np.stack([g["X"][s].mean(0) for g in disjoint for s in SITES])   # (24, d)。群ごとに ebrains, patho2 の順
    origin = C.mean(0)
    C0 = C - origin
    Cp = C0 - np.outer(C0 @ ux, ux)
    _, sv, vt = np.linalg.svd(Cp, full_matrices=False)
    uy = vt[0]
    fam = np.repeat([g["family"] for g in disjoint], len(SITES))
    if (C0[fam == "glioma"] @ uy).mean() < (C0[fam == "meningioma"] @ uy).mean():   # 符号を固定: グリオーマが上
        uy = -uy
    pc1_frac = float(sv[0] ** 2 / (sv ** 2).sum())

    rows = []
    for g in [ref] + disjoint:
        cE, cP = g["X"]["ebrains"].mean(0) - origin, g["X"]["patho2"].mean(0) - origin
        d = g["delta"]
        dx, dy = float(d @ ux), float(d @ uy)
        res = float(np.linalg.norm(d - d_paper))
        row = dict(group=g["name"], family=g["family"], kind=g["kind"], role="reference" if g is ref else "group",
                   n_ebrains=len(g["X"]["ebrains"]), n_patho2=len(g["X"]["patho2"]),
                   x_patho2=float(cP @ ux), y_patho2=float(cP @ uy), x_ebrains=float(cE @ ux), y_ebrains=float(cE @ uy),
                   dx=dx, dy=dy, angle_2d_deg=float(np.degrees(np.arctan2(dy, dx))), length_2d=float(np.hypot(dx, dy)),
                   in_plane_frac=float((dx ** 2 + dy ** 2) / (d @ d)),
                   shift_norm=g["norm"], shift_norm_corrected=g["norm_corr"], noise_trace=g["tr_v"],
                   angle_vs_paper_deg=_deg(_cos(d, d_paper)),
                   residual_vs_paper=res, residual_rel_vs_paper=res / g["norm"])
        if g is not ref:
            sub = reference_for(g, paper, np.ones(len(paper), bool))
            dr, _, tr_r, nr_corr = shift_stats(_xs(sub))
            r2 = float((d - dr) @ (d - dr))
            row["angle_corrected_vs_paper_deg"] = _deg(corrected_cos(d, g["norm_corr"], dr, nr_corr))
            row["residual_vs_paper_lgo"] = float(np.sqrt(r2))
            row["residual_corrected_vs_paper_lgo"] = float(np.sqrt(max(r2 - g["tr_v"] - tr_r, 0.0)))
        rows.append(row)
    df = pd.DataFrame(rows)
    grp = df[df["role"] == "group"]
    summary = {c: (float(grp[c].mean()), float(grp[c].std(ddof=1)))
               for c in ("angle_2d_deg", "length_2d", "angle_vs_paper_deg", "angle_corrected_vs_paper_deg",
                         "shift_norm", "shift_norm_corrected", "residual_vs_paper", "residual_corrected_vs_paper_lgo")}
    summary["pc1_frac"] = pc1_frac
    summary["paper_norm"] = ref["norm"]
    return df, summary


def mismatch(paper: pd.DataFrame, glioma: pd.DataFrame, d_paper: np.ndarray) -> pd.DataFrame:
    """科を食い違わせた差 v_AB = mean(EBRAINS の A) − mean(patho2 の B) と Δ_paper の cos（5 × 5、A = EBRAINS 側、B = patho2 側）。
    予想（A ≠ B）: v_AB = b_AB + Δ_B（b_AB = 施設内の A − B）。b_AB ⊥ Δ かつ Δ_B = Δ_paper なら cos = ‖Δ_paper‖ / sqrt(‖Δ_paper‖² + ‖b_AB‖²)
    （cos_predicted、‖b_AB‖ は 2 施設の平均）。Δ_B を B 自身のシフトにした版が cos_predicted_own = (Δ_B·û) / sqrt(‖b_AB‖² + ‖Δ_B‖²)。
    補正後（patho2 の全スライドに Δ_paper を足す）: v' = v − Δ_paper。同じ模型の予想は 0、B 自身の残りのシフト r_B = Δ_B − Δ_paper を
    入れた予想が cos_predicted_own_after = (r_B·û) / sqrt(‖b_AB‖² + ‖r_B‖²)。対角（A = B）は予想を出さない。"""
    fam = _family_frames(paper, glioma)
    mean = {s: {f: np.stack(d.loc[d["source"] == s, "embedding"].values).astype(np.float64).mean(0) for f, d in fam.items()} for s in SITES}
    cnt = {s: {f: int((d["source"] == s).sum()) for f, d in fam.items()} for s in SITES}
    n_p = float(np.linalg.norm(d_paper))
    u = d_paper / n_p
    rows = []
    for a in FAMILY_ORDER:          # EBRAINS 側
        for b in FAMILY_ORDER:      # patho2 側
            v = mean["ebrains"][a] - mean["patho2"][b]
            bE = mean["ebrains"][a] - mean["ebrains"][b]
            bP = mean["patho2"][a] - mean["patho2"][b]
            bn = float((np.linalg.norm(bE) + np.linalg.norm(bP)) / 2)
            dB = mean["ebrains"][b] - mean["patho2"][b]
            rB = dB - d_paper
            va = v - d_paper
            same = a == b
            rows.append(dict(
                ebrains_family=a, patho2_family=b, matched=same, n_ebrains=cnt["ebrains"][a], n_patho2=cnt["patho2"][b],
                norm=float(np.linalg.norm(v)), b_norm_ebrains=float(np.linalg.norm(bE)), b_norm_patho2=float(np.linalg.norm(bP)), b_norm=bn,
                cos=_cos(v, d_paper),
                cos_predicted=np.nan if same else n_p / np.sqrt(n_p ** 2 + bn ** 2),
                cos_predicted_own=np.nan if same else float(dB @ u / np.sqrt(bn ** 2 + dB @ dB)),
                norm_after=float(np.linalg.norm(va)), cos_after=_cos(va, d_paper),
                cos_predicted_own_after=np.nan if same else float(rB @ u / np.sqrt(bn ** 2 + rB @ rB)),
            ))
    return pd.DataFrame(rows)


def composition(paper: pd.DataFrame, glioma: pd.DataFrame, extra: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """組成の感度: グリオーマの群の EBRAINS 側の組成を変えた変種ごとの cos vs Δ_paper（生・ノイズ補正）。院内（patho2）側は変えない。
    参照は Paper cohort（論文の全例）と Meningioma (all)。CI は 95% ブートストラップをスライド単位と患者単位の両方で（乱数は compute と別の列なので、
    baseline の CI は groups.csv と小数 2–3 桁目で違う。点推定は同じ）。"""
    st = cfg["shift_transfer"]
    rng = np.random.default_rng(int(st.get("seed", 42)) + 1)
    n_boot = int(st.get("n_boot", 1000))
    grade = ebrains_grades(cfg)
    eb = pd.concat([glioma[glioma["source"] == "ebrains"], extra], ignore_index=True)
    eb["grade"] = eb["slide_id"].map(grade)
    pt = glioma[glioma["source"] == "patho2"]
    gname = cfg["shift_transfer"].get("group_names", {})
    is_sft = paper["subtype"].str.lower().str.contains("sft").values
    refs = {"paper": paper, "meni": paper[~is_sft]}
    ref_stats = {k: (shift_stats(_xs(d)), _xs(d), _ps(d)) for k, d in refs.items()}

    g, dg, grd = eb["group"], eb["diagnosis"], eb["grade"]
    glioma3 = g.isin(["ast", "glm", "odg"])
    variants = [   # (院内の群, 変種, EBRAINS 側の行)
        ("ast", "baseline: grade II–IV (config)", g == "ast"),
        ("ast", "(a) grade II–III only", (g == "ast") & grd.isin(["II", "III"])),
        ("ast", "(b) grade IV only (Glioblastoma, IDH-mutant)", (g == "ast") & (grd == "IV")),
        ("glm", "baseline: Glioblastoma, IDH-wildtype (config)", g == "glm"),
        ("glm", "(c) + IDH-wildtype (anaplastic) astrocytomas (≈ CNS5 glioblastoma)", g.isin(["glm", "glm_extra"])),
        ("glm", "(c') IDH-wildtype (anaplastic) astrocytomas only (grade II–III)", g == "glm_extra"),
        ("glm", "(d) + Glioblastoma, IDH-mutant (WHO 2016 glioblastoma, any IDH)", (g == "glm") | (dg == GBM_IDHMUT)),
        ("odg", "baseline (config)", g == "odg"),
        ("glioma (all)", "baseline: ast + glm + odg (config)", glioma3),
        ("glioma (all)", "(c) + IDH-wildtype (anaplastic) astrocytomas", glioma3 | (g == "glm_extra")),
    ]
    rows = []
    for pg, name, m in variants:
        e = eb[m.values]
        p = pt[pt["group"].isin(["ast", "glm", "odg"])] if pg == "glioma (all)" else pt[pt["group"] == pg]
        d = pd.concat([e, p], ignore_index=True)
        X, P = _xs(d), _ps(d)
        delta, norm, tr_v, norm_corr = shift_stats(X)
        row = dict(patho2_group=pg if pg == "glioma (all)" else gname.get(pg, pg), variant=name,
                   ebrains_diagnoses="; ".join(f"{k} ({v})" for k, v in e["diagnosis"].value_counts().sort_index().items()),
                   ebrains_grades="; ".join(f"{k}: {v}" for k, v in e["grade"].fillna("NA").value_counts().sort_index().items()),
                   n_ebrains=len(e), n_pat_ebrains=int(e["patient"].nunique()), n_patho2=len(p),
                   shift_norm=norm, shift_norm_corrected=norm_corr, noise_trace=tr_v)
        for k, ((dr, nr, _, nr_corr), Xr, Pr) in ref_stats.items():
            row[f"cos_vs_{k}"] = _cos(delta, dr)
            row[f"cos_corrected_vs_{k}"] = corrected_cos(delta, norm_corr, dr, nr_corr)
            if k == "paper":
                row["cos_lo_vs_paper"], row["cos_hi_vs_paper"] = boot_cos(X, Xr, rng, n_boot)
                row["cos_lo_patients_vs_paper"], row["cos_hi_patients_vs_paper"] = boot_cos(X, Xr, rng, n_boot, P, Pr)
                row["residual_vs_paper"] = float(np.linalg.norm(delta - dr) / norm)
        rows.append(row)
    cols = ["patho2_group", "variant", "n_ebrains", "n_pat_ebrains", "n_patho2", "shift_norm", "shift_norm_corrected", "noise_trace",
            "cos_vs_paper", "cos_lo_vs_paper", "cos_hi_vs_paper", "cos_lo_patients_vs_paper", "cos_hi_patients_vs_paper",
            "cos_corrected_vs_paper", "residual_vs_paper", "cos_vs_meni", "cos_corrected_vs_meni", "ebrains_diagnoses", "ebrains_grades"]
    return pd.DataFrame(rows)[cols]


# ── figure ────────────────────────────────────────────────────────────────────

def plot(groups: list[dict], pairs: pd.DataFrame, out_path: Path, tag: str = "") -> None:
    disjoint = [g for g in groups if g["kind"] in ("subtype", "sft", "glioma")]
    disjoint.sort(key=lambda g: FAMILY_ORDER.index(g["family"]) if g["family"] in FAMILY_ORDER else 99)
    names = [g["name"] for g in disjoint]
    n = len(names)
    M = np.full((n, n), np.nan)
    look = {(r.group_a, r.group_b): r for r in pairs.itertuples(index=False)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = look.get((names[i], names[j])) or look.get((names[j], names[i]))
            M[i, j] = r.cos if i < j else r.cos_corrected   # 上三角 raw / 下三角 corrected

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7 + 0.55 * n, 0.5 * n + 3), gridspec_kw={"width_ratios": [n, 6]})
    im = ax.imshow(M, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=9); ax.set_yticklabels(names, fontsize=9)
    for lab, g in zip(ax.get_yticklabels(), disjoint):
        lab.set_color(FAMILY_COLOR.get(g["family"], "k"))
    for lab, g in zip(ax.get_xticklabels(), disjoint):
        lab.set_color(FAMILY_COLOR.get(g["family"], "k"))
    for i in range(n):
        for j in range(n):
            if i != j and np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if 0.25 < M[i, j] < 0.85 else "w")
    fams = [g["family"] for g in disjoint]
    for k in range(1, n):
        if fams[k] != fams[k - 1]:
            ax.axhline(k - 0.5, color="k", lw=1.2); ax.axvline(k - 0.5, color="k", lw=1.2)
    ax.set_title("cos between site-shift vectors" + tag + "\n(upper: raw, lower: noise-corrected)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    allg = [g for g in groups]
    y = np.arange(len(allg))
    ax2.barh(y + 0.2, [g["norm"] for g in allg], height=0.4, color=[FAMILY_COLOR.get(g["family"], "gray") for g in allg], alpha=0.45, label="raw")
    ax2.barh(y - 0.2, [g["norm_corr"] for g in allg], height=0.4, color=[FAMILY_COLOR.get(g["family"], "gray") for g in allg], label="noise-corrected")
    ax2.set_yticks(y); ax2.set_yticklabels([f"{g['name']} ({len(g['X']['ebrains'])}/{len(g['X']['patho2'])})" for g in allg], fontsize=8)
    ax2.invert_yaxis(); ax2.set_xlabel("|shift| (L2)"); ax2.set_title("shift magnitude (n EBRAINS / n patho2)", fontsize=10)
    ax2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _spread(y: np.ndarray, gap: float, n_iter: int = 500) -> np.ndarray:
    """ラベルの縦位置を、元の位置になるべく近いまま gap 以上離す（隣どうしの重なりを半分ずつ押し広げるのを繰り返す）。"""
    order = np.argsort(y)
    p = np.asarray(y, dtype=float)[order].copy()
    for _ in range(n_iter):
        moved = False
        for i in range(1, len(p)):
            over = p[i - 1] + gap - p[i]
            if over > 1e-9:
                p[i - 1] -= over / 2
                p[i] += over / 2
                moved = True
        if not moved:
            break
    out = np.empty_like(p)
    out[order] = p
    return out


def plot_arrows(df: pd.DataFrame, summary: dict, out_path: Path, tag: str = "") -> None:
    """a: 両施設の重心と patho2 → EBRAINS の矢印（平行四辺形）。b: 同じ矢印を原点から並べたもの（平行・同じ長さかを見る）。両方とも縦横同じ縮尺。"""
    from matplotlib.lines import Line2D
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 7.4), gridspec_kw={"width_ratios": [1.0, 1.15]})
    xs = np.r_[df["x_patho2"].values, df["x_ebrains"].values]
    ys = np.r_[df["y_patho2"].values, df["y_ebrains"].values]
    label_x = float(xs.min()) - 0.9                       # ラベルは左の余白に縦に並べ、patho2 の重心へ引き出し線
    label_y = _spread(df["y_patho2"].values, gap=0.62)
    for r, ly in zip(df.itertuples(index=False), label_y):
        is_ref = r.role == "reference"
        col = "0.45" if is_ref else FAMILY_COLOR.get(r.family, "gray")
        style = dict(arrowstyle="-|>", color=col, lw=2.0 if is_ref else 1.5, ls="--" if is_ref else "-", shrinkA=0, shrinkB=0, mutation_scale=12)
        ax.annotate("", xy=(r.x_ebrains, r.y_ebrains), xytext=(r.x_patho2, r.y_patho2), arrowprops=style)
        ax.scatter([r.x_patho2], [r.y_patho2], s=28, facecolors="white", edgecolors=col, linewidths=1.2, zorder=3)
        ax.scatter([r.x_ebrains], [r.y_ebrains], s=28, color=col, zorder=3)
        ax.plot([label_x + 0.1, r.x_patho2], [ly, r.y_patho2], color=col, lw=0.5, alpha=0.6)
        ax.text(label_x, ly, r.group, ha="right", va="center", fontsize=8, color=col, fontstyle="italic" if is_ref else "normal")
        ax2.annotate("", xy=(r.dx, r.dy), xytext=(0, 0), arrowprops=style)
    ax.set_xlim(label_x - 5.6, float(xs.max()) + 0.6)
    ax.set_ylim(min(float(ys.min()), float(label_y.min())) - 0.8, max(float(ys.max()), float(label_y.max())) + 0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δ_paper direction (L2 units)")
    ax.set_ylabel(f"PC1 of between-group variation ⊥ Δ_paper ({summary['pc1_frac']:.0%})")
    ax.set_title("centroids at both sites (open: patho2, filled: EBRAINS)" + tag + "\narrow = site shift of each group, dashed = Paper cohort", fontsize=10)
    ax.grid(alpha=0.25)

    grp = df[df["role"] == "group"]
    ax2.set_xlim(-0.3, float(grp["dx"].max()) + 0.6)
    ax2.set_ylim(float(grp["dy"].min()) - 0.9, float(grp["dy"].max()) + 0.9)
    ax2.set_aspect("equal")
    ax2.axhline(0, color="0.8", lw=0.8)
    ax2.set_xlabel("Δ_paper direction"); ax2.set_ylabel("PC1 ⊥ Δ_paper")
    a2, l2 = summary["angle_2d_deg"], summary["length_2d"]
    af, nf = summary["angle_corrected_vs_paper_deg"], summary["shift_norm_corrected"]
    ax2.set_title(f"same arrows from a common origin\n2-D: angle {a2[0]:.1f} ± {a2[1]:.1f}°, length {l2[0]:.2f} ± {l2[1]:.2f}\n"
                  f"full space (noise-corr.): angle {af[0]:.1f} ± {af[1]:.1f}°, |shift| {nf[0]:.2f} ± {nf[1]:.2f}", fontsize=9)
    ax2.grid(alpha=0.25)
    handles = [Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=f) for f in FAMILY_ORDER if f in set(grp["family"])]
    handles.append(Line2D([], [], color="0.45", lw=2, ls="--", label=REF))
    ax2.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mismatch(mm: pd.DataFrame, out_path: Path, tag: str = "") -> None:
    """科を食い違わせた差の cos の行列。補正前（括弧 = cos_predicted）と補正後（括弧 = cos_predicted_own_after）。"""
    fams = [f for f in FAMILY_ORDER if f in set(mm["ebrains_family"])]
    look = mm.set_index(["ebrains_family", "patho2_family"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    panels = [("cos", "cos_predicted", "before correction" + tag + "\ncos(mean EBRAINS A − mean patho2 B, Δ_paper)\n(predicted ‖Δ‖ / √(‖Δ‖² + ‖b_AB‖²))"),
              ("cos_after", "cos_predicted_own_after", "after adding Δ_paper to every patho2 slide\n(predicted from B's residual shift r_B = Δ_B − Δ_paper;\nzero-order prediction = 0)")]
    for ax, (col, pcol, title) in zip(axes, panels):
        M = np.array([[look.loc[(a, b), col] for b in fams] for a in fams])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for i, a in enumerate(fams):
            for j, b in enumerate(fams):
                pred = look.loc[(a, b), pcol]
                txt = f"{M[i, j]:.2f}" + (f"\n({pred:.2f})" if np.isfinite(pred) else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="w" if abs(M[i, j]) > 0.6 else "k",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(fams))); ax.set_yticks(range(len(fams)))
        ax.set_xticklabels(fams, rotation=30, ha="right"); ax.set_yticklabels(fams)
        for lab, f in zip(ax.get_xticklabels(), fams):
            lab.set_color(FAMILY_COLOR.get(f, "k"))
        for lab, f in zip(ax.get_yticklabels(), fams):
            lab.set_color(FAMILY_COLOR.get(f, "k"))
        ax.set_xlabel("patho2 family B"); ax.set_ylabel("EBRAINS family A")
        ax.set_title(title, fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", choices=["titan", "meanpool"], default="titan",
                    help="スライド埋め込み。meanpool = CONCH のパッチ平均（出力は shift_transfer/meanpool/）")
    ap.add_argument("--reload", action="store_true", help="埋め込みのキャッシュを捨てて h5 から読み直す")
    ap.add_argument("--bootstrap", choices=["slides", "patients"], default="slides",
                    help="groups.csv の CI の再抽出の単位。patients は施設・群ごとに患者を再抽出し groups_patients.csv に書く（groups.csv は上書きしない）")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "shift_transfer"
    if args.encoder != "titan":
        out_dir = out_dir / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ENCODER_TAG[args.encoder]
    embed = SlideEmbedder(cfg, args.encoder)

    print(f"[encoder] {args.encoder} -> {out_dir}")
    print("[load] paper cohort")
    paper = load_paper(cfg, out_dir, args.reload, args.encoder)
    print("  ", paper.groupby("source").size().to_dict())
    print("[load] gliomas (Patho2Gliomas representatives × EBRAINS)")
    glioma = load_glioma(cfg, out_dir, args.reload, embed)
    print(glioma.groupby(["group", "source"]).agg(slides=("slide_id", "size"), patients=("patient", "nunique")).to_string())
    print("[load] EBRAINS IDH-wildtype (anaplastic) astrocytomas（組成の感度だけ）")
    extra = load_extra(cfg, out_dir, args.reload, embed)
    print("  ", extra.groupby("diagnosis").size().to_dict())
    if args.encoder == "meanpool":
        print(f"  meanpool: transfer_meanpool のキャッシュから {embed.n_cached} 枚、h5 から平均を取った {embed.n_computed} 枚")

    groups = build_groups(paper, glioma, cfg)
    df_groups, pairs = compute(groups, paper, cfg, args.bootstrap)
    bio = bio_control(paper, glioma, groups)
    arr, arr_summary = arrows(groups, paper)
    d_paper = next(g["delta"] for g in groups if g["name"] == REF)
    mm = mismatch(paper, glioma, d_paper)
    comp = composition(paper, glioma, extra, cfg)

    # bootstrap に依らない出力はどちらのモードでも同じ内容を書く。groups の CI だけモードで変わるので、患者単位は別名
    df_groups.to_csv(out_dir / ("groups_patients.csv" if args.bootstrap == "patients" else "groups.csv"), index=False)
    pairs.to_csv(out_dir / "cos_matrix.csv", index=False)
    bio.to_csv(out_dir / "bio_control.csv", index=False)
    plot(groups, pairs, out_dir / "shift_transfer.png", tag)
    arr.to_csv(out_dir / "arrows.csv", index=False)
    plot_arrows(arr, arr_summary, out_dir / "arrows.png", tag)
    mm.to_csv(out_dir / "mismatch_matrix.csv", index=False)
    plot_mismatch(mm, out_dir / "mismatch_matrix.png", tag)
    comp.to_csv(out_dir / "composition.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["group", "n_ebrains", "n_patho2", "shift_norm", "shift_norm_corrected", "cos_vs_paper", "cos_lo_vs_paper", "cos_hi_vs_paper",
            "cos_corrected_vs_paper", "residual_vs_paper", "cos_vs_meni", "cos_corrected_vs_meni"]
    print("\n[groups]  shift = mean(EBRAINS) − mean(patho2); cos vs Δ_paper（論文コホートの部分集合は leave-group-out）")
    print(df_groups[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[pairs]  same family vs different family（corrected cos の中央値）")
    print(pairs.groupby("same_family")["cos_corrected"].describe()[["count", "50%", "min", "max"]].to_string())
    print("\n[bio control]")
    print(bio.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    ff = lambda v: f"{v:.3f}"
    print(f"\n[arrows]  x = Δ_paper（|Δ_paper| = {arr_summary['paper_norm']:.3f}）、y = Δ_paper を除いた群間変動の PC1（{arr_summary['pc1_frac']:.1%}）。12 群の mean ± sd:")
    for k, label in [("angle_2d_deg", "2-D angle (deg)"), ("length_2d", "2-D length"), ("angle_vs_paper_deg", "angle vs Δ_paper, full space, raw (deg)"),
                     ("angle_corrected_vs_paper_deg", "angle vs Δ_paper, full space, noise-corr. LGO (deg)"),
                     ("shift_norm", "|shift| raw"), ("shift_norm_corrected", "|shift| noise-corr."),
                     ("residual_vs_paper", "‖Δ_g − Δ_paper‖ raw"), ("residual_corrected_vs_paper_lgo", "‖Δ_g − Δ_paper‖ noise-corr. LGO")]:
        m, s = arr_summary[k]
        print(f"   {label:52s} {m:7.3f} ± {s:.3f}")
    print(arr[["group", "n_ebrains", "n_patho2", "dx", "dy", "angle_2d_deg", "length_2d", "in_plane_frac", "shift_norm", "angle_vs_paper_deg",
               "angle_corrected_vs_paper_deg", "residual_vs_paper", "residual_rel_vs_paper", "residual_corrected_vs_paper_lgo"]]
          .to_string(index=False, float_format=ff))
    print("\n[mismatch]  cos(mean EBRAINS A − mean patho2 B, Δ_paper)。行 = EBRAINS の A、列 = patho2 の B")
    for col in ("cos", "cos_predicted", "cos_after", "cos_predicted_own_after"):
        print(f"  -- {col}")
        print(mm.pivot(index="ebrains_family", columns="patho2_family", values=col).loc[FAMILY_ORDER, FAMILY_ORDER].to_string(float_format=ff))
    print("\n[composition]  グリオーマの群の EBRAINS 側の組成を変えた cos vs Δ_paper（院内側は同じ）")
    print(comp.drop(columns=["ebrains_diagnoses", "ebrains_grades"]).to_string(index=False, float_format=ff))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
