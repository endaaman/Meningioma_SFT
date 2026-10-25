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

出力 (output_dir/shift_transfer/):
    groups.csv         — 群ごとの n（枚数と患者数）・|shift|（生 / ノイズ補正）・cos vs Δ_paper と vs Δ_meni（生・95% ブートストラップ CI・補正）・残差
    cos_matrix.csv     — 素な群どうしの cos（raw と corrected、縦長）
    bio_control.csv    — 施設内の科どうしの差と Δ の cos
    shift_transfer.png — a: cos 行列（上三角 raw / 下三角 corrected）  b: |shift|
    paper_slides.csv / paper_embeddings.npy / glioma_slides.csv / glioma_embeddings.npy — 読み込んだ埋め込みのキャッシュ（--reload で読み直す）

Usage:
    uv run python shift_transfer.py [--reload]
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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── data ──────────────────────────────────────────────────────────────────────

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


def load_paper(cfg: dict, out_dir: Path, reload: bool) -> pd.DataFrame:
    """論文のコホート（embedding.*.original の TITAN）。列: slide_id, source, subtype, patient, embedding。"""
    csv_path, npy_path = out_dir / "paper_slides.csv", out_dir / "paper_embeddings.npy"
    if not reload and (df := _read_cache(csv_path, npy_path)) is not None:
        return df
    df = load_data(cfg, "original").rename(columns={"case_id": "slide_id"})
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


def load_glioma(cfg: dict, out_dir: Path, reload: bool) -> pd.DataFrame:
    """Patho2Gliomas の代表 HE（1 症例 1 枚、representatives.csv）と EBRAINS の対応診断の TITAN。列: slide_id, source, group, patient, diagnosis, embedding。"""
    csv_path, npy_path = out_dir / "glioma_slides.csv", out_dir / "glioma_embeddings.npy"
    if not reload and (df := _read_cache(csv_path, npy_path)) is not None:
        return df
    st = cfg["shift_transfer"]
    pg = cfg["patho2_gliomas"]
    rows: list[dict] = []

    key = f"{pg['tile_model']}/{cfg['keys']['slide_feature']}"
    slides = pd.read_csv(Path(cfg["output_dir"]) / "patho2_gliomas" / "slides.csv")
    reps = pd.read_csv(pg["representatives"])
    reps = reps[reps["note"] != "exclude"]
    merged = reps.merge(slides[["label", "case", "file", "h5"]], on=["case", "file"], how="left")
    for r in merged.itertuples(index=False):
        if not isinstance(r.h5, str) or not Path(r.h5).exists():
            print(f"  WARN [patho2] h5 not found: {r.case} / {r.file}")
            continue
        with h5py.File(r.h5, "r") as f:
            if key not in f:
                print(f"  WARN [patho2] no '{key}': {r.h5}")
                continue
            emb = f[key][:]
        rows.append({"slide_id": r.case, "source": "patho2", "group": r.label, "patient": r.case, "diagnosis": r.label, "embedding": emb})

    ann = pd.read_csv(st["ebrains_annotation"])
    ann = ann[ann["control"] != 1]
    diag2group = {d: g for g, ds in st["ebrains_groups"].items() for d in ds}
    root = Path(cfg["ebrains_embeddings"]["root"])
    ekey = f"{cfg['ebrains_embeddings']['tile_model']}/{cfg['keys']['slide_feature']}"
    index: dict[str, Path] = {}   # uuid → h5。「… 1p/19q codeleted」は親 "… and 1p/" と入れ子 "19q codeleted/" に散らばっているので、木全体から引く（浅い方を優先）
    for p in sorted(root.rglob("*_hdf.h5"), key=lambda q: len(q.parts)):
        index.setdefault(p.name[: -len("_hdf.h5")], p)
    for r in ann[ann["diagnosis"].isin(diag2group)].itertuples(index=False):
        p = index.get(r.uuid)
        if p is None:
            print(f"  WARN [ebrains] h5 not found: {r.uuid} ({r.diagnosis})")
            continue
        with h5py.File(p, "r") as f:
            if ekey not in f:
                print(f"  WARN [ebrains] no '{ekey}': {p}")
                continue
            emb = f[ekey][:]
        rows.append({"slide_id": r.uuid[:8], "source": "ebrains", "group": diag2group[r.diagnosis], "patient": str(r.pat_id),
                     "diagnosis": r.diagnosis, "embedding": emb})
    df = pd.DataFrame(rows)
    _write_cache(df, csv_path, npy_path)
    return df


# ── groups ────────────────────────────────────────────────────────────────────

def _xs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {s: np.stack(df.loc[df["source"] == s, "embedding"].values).astype(np.float64) for s in SITES}


def _npat(df: pd.DataFrame) -> dict[str, int]:
    return {s: int(df.loc[df["source"] == s, "patient"].nunique()) for s in SITES}


def build_groups(paper: pd.DataFrame, glioma: pd.DataFrame, cfg: dict) -> list[dict]:
    """群ごとに name / family / kind / X（施設 → (n, d)）/ n_pat / mask（論文コホートの部分集合なら paper 上の bool マスク）。"""
    st = cfg["shift_transfer"]
    min_n = int(st.get("subtype_min_n", 20))
    names: dict = st.get("group_names", {})
    fams: dict = st.get("families", {})
    is_sft = paper["subtype"].str.lower().str.contains("sft").values
    groups = [
        dict(name=REF, family="meningioma", kind="paper", X=_xs(paper), n_pat=_npat(paper), mask=np.ones(len(paper), bool)),
        dict(name=MENI, family="meningioma", kind="paper", X=_xs(paper[~is_sft]), n_pat=_npat(paper[~is_sft]), mask=~is_sft),
        dict(name=SFT, family="SFT", kind="sft", X=_xs(paper[is_sft]), n_pat=_npat(paper[is_sft]), mask=is_sft),
    ]
    for sub in ordered_subtypes(set(paper.loc[~is_sft, "subtype"]), cfg):
        m = (paper["subtype"] == sub).values
        n = {s: int((m & (paper["source"] == s).values).sum()) for s in SITES}
        if min(n.values()) >= min_n:
            groups.append(dict(name=sub, family="meningioma", kind="subtype", X=_xs(paper[m]), n_pat=_npat(paper[m]), mask=m))
    for g in ("ast", "glm", "odg"):
        d = glioma[glioma["group"] == g]
        groups.append(dict(name=names.get(g, g), family=fams.get(g, "glioma"), kind="glioma", X=_xs(d), n_pat=_npat(d), mask=None))
    d = glioma[glioma["group"].isin(["ast", "glm", "odg"])]
    groups.append(dict(name=GLIOMA_ALL, family="glioma", kind="glioma_all", X=_xs(d), n_pat=_npat(d), mask=None))
    for g in ("lym", "met"):
        d = glioma[glioma["group"] == g]
        if len(d) and all((d["source"] == s).any() for s in SITES):
            groups.append(dict(name=names.get(g, g), family=fams.get(g, g), kind="glioma", X=_xs(d), n_pat=_npat(d), mask=None))
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


def boot_cos(Xg: dict[str, np.ndarray], Xr: dict[str, np.ndarray], rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """群と参照の両方をスライド単位（施設内）で再抽出したときの cos の 95% 区間。"""
    vals = np.empty(n_boot)
    for b in range(n_boot):
        d = []
        for X in (Xg, Xr):
            e, p = X["ebrains"], X["patho2"]
            d.append(e[rng.integers(0, len(e), len(e))].mean(0) - p[rng.integers(0, len(p), len(p))].mean(0))
        vals[b] = _cos(d[0], d[1])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def reference_for(group: dict, paper: pd.DataFrame, ref_mask: np.ndarray) -> dict[str, np.ndarray] | None:
    """参照（Paper cohort または Meningioma (all)）の埋め込み。group が論文コホートの部分集合なら、その群のスライドを除く（leave-group-out）。"""
    keep = ref_mask if group["mask"] is None else ref_mask & ~group["mask"]
    if not keep.any():
        return None
    sub = paper[keep]
    if any(not (sub["source"] == s).any() for s in SITES):
        return None
    return _xs(sub)


def compute(groups: list[dict], paper: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            Xr = reference_for(g, paper, ref_mask)
            if Xr is None:
                continue
            dr, nr, _, nr_corr = shift_stats(Xr)
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


def bio_control(paper: pd.DataFrame, glioma: pd.DataFrame, groups: list[dict]) -> pd.DataFrame:
    """施設内の科どうしの差の向きと、施設間シフト（Δ_paper・Δ_glioma）の cos。両施設での向きの一致も。"""
    is_sft = paper["subtype"].str.lower().str.contains("sft").values
    fam_df = {
        "meningioma": paper[~is_sft], "SFT": paper[is_sft],
        "glioma": glioma[glioma["group"].isin(["ast", "glm", "odg"])],
        "lymphoma": glioma[glioma["group"] == "lym"], "metastasis": glioma[glioma["group"] == "met"],
    }
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


# ── figure ────────────────────────────────────────────────────────────────────

def plot(groups: list[dict], pairs: pd.DataFrame, out_path: Path) -> None:
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
    ax.set_title("cos between site-shift vectors\n(upper: raw, lower: noise-corrected)", fontsize=10)
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reload", action="store_true", help="埋め込みのキャッシュを捨てて h5 から読み直す")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "shift_transfer"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] paper cohort")
    paper = load_paper(cfg, out_dir, args.reload)
    print("  ", paper.groupby("source").size().to_dict())
    print("[load] gliomas (Patho2Gliomas representatives × EBRAINS)")
    glioma = load_glioma(cfg, out_dir, args.reload)
    print(glioma.groupby(["group", "source"]).agg(slides=("slide_id", "size"), patients=("patient", "nunique")).to_string())

    groups = build_groups(paper, glioma, cfg)
    df_groups, pairs = compute(groups, paper, cfg)
    bio = bio_control(paper, glioma, groups)

    df_groups.to_csv(out_dir / "groups.csv", index=False)
    pairs.to_csv(out_dir / "cos_matrix.csv", index=False)
    bio.to_csv(out_dir / "bio_control.csv", index=False)
    plot(groups, pairs, out_dir / "shift_transfer.png")

    pd.set_option("display.width", 200)
    cols = ["group", "n_ebrains", "n_patho2", "shift_norm", "shift_norm_corrected", "cos_vs_paper", "cos_lo_vs_paper", "cos_hi_vs_paper",
            "cos_corrected_vs_paper", "residual_vs_paper", "cos_vs_meni", "cos_corrected_vs_meni"]
    print("\n[groups]  shift = mean(EBRAINS) − mean(patho2); cos vs Δ_paper（論文コホートの部分集合は leave-group-out）")
    print(df_groups[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n[pairs]  same family vs different family（corrected cos の中央値）")
    print(pairs.groupby("same_family")["cos_corrected"].describe()[["count", "50%", "min", "max"]].to_string())
    print("\n[bio control]")
    print(bio.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
