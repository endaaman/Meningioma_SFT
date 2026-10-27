"""transfer_umap.py — 転移テストの 2,301 枚（5 科 × 2 施設）を 1 枚の UMAP に載せ、Δ_paper で院内を動かす前後を見る。

UMAP は補正なしの全 2,301 枚で 1 回だけ当てはめ（パラメータは umap_plot.py と同じ config の umap: n_neighbors / min_dist /
random_state、n_jobs=1）、補正後の院内スライドは同じ reducer の transform で写す（パネルどうしの座標を比べられるように）。
EBRAINS（reference）は補正しないので、どのパネルでも当てはめの座標のまま。

パネル（umap_titan.png）:
    a  補正なし（色 = 科、形 = 施設: EBRAINS ●・院内 ■ に黒縁）
    b  院内を Δ_paper だけ平行移動（Δ_paper = 髄膜腫+SFT コホートの EBRAINS 平均 − 院内平均。centroid.py の Δ と同じ）
    c  院内を自分の科の Δ_f（= EBRAINS 科 f の平均 − 院内 科 f の平均）だけ平行移動（上限。科のラベルが要る）
small multiples（umap_titan_families.png）: 2 行 × 5 列（列 = 科、上 = 補正なし、下 = Δ_paper）。その科の 2 施設を
    施設色（config の display.colors.sources）で、他の科は薄い灰で背景に。

transform の注意: UMAP の transform は当てはめのデータ（補正なしの院内も含む）の近傍から位置を決めるので、動かした院内の
    スライドが補正なしの院内に引き寄せられうる。その確認として、写した院内スライドごとに 768 次元での当てはめデータ中の
    15 近傍（n_neighbors）のうち EBRAINS の割合を出す（knn_ebrains_frac。summary.md に科ごとの平均）。
    あわせて施設の混ざり具合を 768 次元で: 院内スライドごとに「EBRAINS ＋ その条件の院内」の中の 15 近傍（自分を除く）の
    EBRAINS の割合（mix_ebrains_frac）。同じ科の EBRAINS の割合（期待値）と並べる。transform は写す点どうしの近さを
    使わない（当てはめデータの近傍にだけ寄せる）ので、院内が院内どうしで固まっていても UMAP の上では EBRAINS に重なって
    見える。重なりの見た目より mix_ebrains_frac を根拠にすること。

データ: out/transfer/manifest.csv（transfer_manifest.py）と、transfer_correction.py が書く埋め込みのキャッシュ
    out/transfer/embeddings_titan.npy ＋ embeddings_titan_slides.csv（manifest の並び）。キャッシュの並びが manifest と同じか、
    値が h5（manifest の h5_titan の config transfer.titan_key）と一致するかを毎回確かめる（全 2,301 枚、数秒）。
    合わなければ h5 の値を使う（キャッシュは書かない = transfer_correction.py の持ち物）。

出力 (output_dir/transfer_umap/):
    umap_titan.csv            slide_id, site, family, group, condition（uncorrected / delta_paper / delta_own）, umap1, umap2,
                              knn_ebrains_frac（写した院内だけ）, mix_ebrains_frac（院内だけ）
    umap_titan.png            a–c
    umap_titan_families.png   2 × 5 の small multiples
    summary.md                Δ の長さ、transform の近傍の確認、施設の混ざり具合（科 × 条件）

Usage:
    uv run python transfer_umap.py
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from utils.loader import load_config  # noqa: E402

REF = "ebrains"
SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "In-house"}
FAMILIES = ["meningioma", "SFT", "glioma", "lymphoma", "metastasis"]
FAMILY_LABEL = {"meningioma": "Meningioma", "SFT": "SFT", "glioma": "Glioma", "lymphoma": "Lymphoma (DLBCL)",
                "metastasis": "Metastasis"}
# 科の色は shift_transfer.py の FAMILY_COLOR と同じ（並行して編集中のスクリプトを import しないため写した）
FAMILY_COLOR = {"meningioma": "#4c72b0", "SFT": "#000000", "glioma": "#dd8452", "lymphoma": "#55a868", "metastasis": "#c44e52"}
PAPER = {"meningioma", "SFT"}                  # 髄膜腫+SFT コホート（Δ_paper の推定元）
CONDITIONS = ["uncorrected", "delta_paper", "delta_own"]
TITLES = {"uncorrected": "a  Uncorrected",
          "delta_paper": "b  In-house + Δ_paper (meningioma + SFT)",
          "delta_own": "c  In-house + own-family Δ (upper reference)"}


# ── data ──────────────────────────────────────────────────────────────────────

def _read_h5(path: str, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f[key][:], dtype=np.float32).reshape(-1)


def load_titan(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """manifest と TITAN 埋め込み（manifest の並び、float64）。キャッシュは h5 と全数照合する（shift_time.py も使う）。"""
    tcfg = cfg["transfer"]
    out = Path(cfg["output_dir"])
    man = pd.read_csv(out / tcfg.get("manifest", "transfer/manifest.csv"), dtype={"slide_id": str, "patient": str})
    key = tcfg["titan_key"]
    with ThreadPoolExecutor(8) as ex:          # NFS 上の h5 を読む（読むだけ）
        H = np.stack(list(ex.map(lambda p: _read_h5(p, key), man["h5_titan"])))
    npy, order_csv = out / "transfer" / "embeddings_titan.npy", out / "transfer" / "embeddings_titan_slides.csv"
    ident = man[["site", "slide_id"]].reset_index(drop=True)
    if npy.exists() and order_csv.exists():
        same_order = pd.read_csv(order_csv, dtype={"slide_id": str}).equals(ident)
        C = np.load(npy)
        diff = float(np.abs(C - H).max()) if same_order and C.shape == H.shape else float("nan")
        print(f"  cache {npy}: order {'OK' if same_order else 'MISMATCH'}, max |cache − h5| = {diff:.3g}")
        if not (same_order and diff == 0.0):
            print("  WARNING: キャッシュが h5 と合わないので h5 の値を使う")
    else:
        print(f"  cache が無い（{npy}）ので h5 の値を使う")
    print(f"  {len(man)} slides × {H.shape[1]} ({key})")
    return man, H.astype(np.float64)


def deltas(man: pd.DataFrame, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Δ_paper と科ごとの Δ_f（EBRAINS 平均 − 院内平均）。"""
    site, fam = man["site"].to_numpy(), man["family"].to_numpy()
    paper = np.isin(fam, list(PAPER))
    d_paper = X[paper & (site == REF)].mean(0) - X[paper & (site != REF)].mean(0)
    d_own = {f: X[(fam == f) & (site == REF)].mean(0) - X[(fam == f) & (site != REF)].mean(0) for f in FAMILIES}
    return d_paper, d_own


def corrected(man: pd.DataFrame, X: np.ndarray, cond: str, d_paper: np.ndarray, d_own: dict) -> np.ndarray:
    """院内だけを平行移動した埋め込み（EBRAINS は不変）。"""
    p = (man["site"] != REF).to_numpy()
    Y = X.copy()
    if cond == "delta_paper":
        Y[p] += d_paper
    elif cond == "delta_own":
        fam = man["family"].to_numpy()
        for f, d in d_own.items():
            Y[p & (fam == f)] += d
    return Y


def mix_fraction(Y: np.ndarray, is_ref: np.ndarray, k: int) -> np.ndarray:
    """院内スライドごとの、全スライド（EBRAINS ＋ この条件の院内）の中の k 近傍（自分を除く）の EBRAINS の割合。"""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Y)
    idx = nn.kneighbors(Y[~is_ref], return_distance=False)[:, 1:]
    return is_ref[idx].mean(1)


# ── plot ──────────────────────────────────────────────────────────────────────

def _scatter(ax: plt.Axes, xy: np.ndarray, man: pd.DataFrame, cfg: dict) -> None:
    """色 = 科、形 = 施設（EBRAINS ● 塗り / 院内 □ 白抜き）。院内を下に、EBRAINS を上に重ねて、重なっても両方見えるようにする。"""
    mk = cfg.get("display", {}).get("markers", {}).get("sources", {"ebrains": "o", "patho2": "s"})
    site, fam = man["site"].to_numpy(), man["family"].to_numpy()
    for s in ("patho2", REF):
        for f in FAMILIES:
            m = (site == s) & (fam == f)
            if not m.any():
                continue
            if s == REF:
                ax.scatter(xy[m, 0], xy[m, 1], s=6, marker=mk.get(s, "o"), c=FAMILY_COLOR[f], alpha=0.6,
                           linewidths=0, zorder=3)
            else:
                ax.scatter(xy[m, 0], xy[m, 1], s=14, marker=mk.get(s, "s"), facecolors="none",
                           edgecolors=FAMILY_COLOR[f], alpha=0.9, linewidths=0.6, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=9); ax.set_ylabel("UMAP2", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def plot_main(coords: dict[str, np.ndarray], man: pd.DataFrame, cfg: dict, path: Path) -> None:
    mk = cfg.get("display", {}).get("markers", {}).get("sources", {"ebrains": "o", "patho2": "s"})
    allxy = np.concatenate(list(coords.values()))
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.04 * (hi - lo)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    for ax, cond in zip(axes, CONDITIONS):
        _scatter(ax, coords[cond], man, cfg)
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_title(TITLES[cond], loc="left", fontsize=11)
    fam_h = [Line2D([0], [0], marker="o", color=FAMILY_COLOR[f], linestyle="None", markersize=7, label=FAMILY_LABEL[f])
             for f in FAMILIES]
    site_h = [Line2D([0], [0], marker=mk.get(REF, "o"), color="gray", linestyle="None", markersize=7,
                     label=f"{SITE_LABEL[REF]} (n = {int((man['site'] == REF).sum())})"),
              Line2D([0], [0], marker=mk.get("patho2", "s"), color="gray", markerfacecolor="none", linestyle="None",
                     markersize=7, label=f"{SITE_LABEL['patho2']} (n = {int((man['site'] != REF).sum())})")]
    fig.legend(handles=fam_h + site_h, loc="lower center", ncol=7, fontsize=9, frameon=False)
    fig.suptitle("TITAN slide embeddings, 5 tumour families × 2 sites — UMAP fitted once on uncorrected, "
                 "corrected in-house slides projected with transform", fontsize=11)
    plt.tight_layout(rect=(0, 0.06, 1, 0.95))
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


def plot_families(coords: dict[str, np.ndarray], man: pd.DataFrame, cfg: dict, path: Path) -> None:
    src_col = cfg.get("display", {}).get("colors", {}).get("sources", {"ebrains": "#4477AA", "patho2": "#EE6677"})
    site, fam = man["site"].to_numpy(), man["family"].to_numpy()
    rows = ["uncorrected", "delta_paper"]
    row_label = {"uncorrected": "Uncorrected", "delta_paper": "In-house + Δ_paper"}
    allxy = np.concatenate([coords[c] for c in rows])
    lo, hi = allxy.min(0), allxy.max(0)
    pad = 0.04 * (hi - lo)
    fig, axes = plt.subplots(2, len(FAMILIES), figsize=(3.3 * len(FAMILIES), 6.9))
    for i, cond in enumerate(rows):
        xy = coords[cond]
        for j, f in enumerate(FAMILIES):
            ax = axes[i, j]
            bg = fam != f
            ax.scatter(xy[bg, 0], xy[bg, 1], s=3, c="#DDDDDD", linewidths=0, zorder=1)
            for s in (REF, "patho2"):          # 院内は白抜きで下、EBRAINS は塗りで上（重なっても両方見える）
                m = (fam == f) & (site == s)
                if s == REF:
                    ax.scatter(xy[m, 0], xy[m, 1], s=6, marker="o", c=src_col.get(s, "gray"), alpha=0.7,
                               linewidths=0, zorder=3, label=f"{SITE_LABEL[s]} {int(m.sum())}")
                else:
                    ax.scatter(xy[m, 0], xy[m, 1], s=12, marker="s", facecolors="none", edgecolors=src_col.get(s, "gray"),
                               alpha=0.9, linewidths=0.6, zorder=2, label=f"{SITE_LABEL[s]} {int(m.sum())}")
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(FAMILY_LABEL[f], fontsize=11)
                ax.legend(loc="lower right", fontsize=7, frameon=False, handletextpad=0.1, markerscale=1.5)
            if j == 0:
                ax.set_ylabel(row_label[cond], fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "transfer_umap"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[load]")
    man, X = load_titan(cfg)
    site, fam = man["site"].to_numpy(), man["family"].to_numpy()
    is_ref = site == REF
    d_paper, d_own = deltas(man, X)
    print(f"  ‖Δ_paper‖ = {np.linalg.norm(d_paper):.3f}  " +
          "  ".join(f"‖Δ_{f}‖ = {np.linalg.norm(d):.3f}" for f, d in d_own.items()))

    ucfg = cfg.get("umap", {})
    k = int(ucfg.get("n_neighbors", 15))
    print("[umap] fitting on uncorrected 2 sites × 5 families ...")
    import umap as umap_lib  # 読み込みが重いのでここで
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=k, min_dist=ucfg.get("min_dist", 0.01),
                                random_state=ucfg.get("random_state", 42), n_jobs=1)
        fit_xy = reducer.fit_transform(X)

    nn_fit = NearestNeighbors(n_neighbors=k).fit(X)       # transform が見る当てはめデータの近傍（確認用）
    coords, frames, diag = {}, [], []
    for cond in CONDITIONS:
        Y = corrected(man, X, cond, d_paper, d_own)
        xy = fit_xy.copy()
        knn_frac = np.full(len(man), np.nan)
        if cond != "uncorrected":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                xy[~is_ref] = reducer.transform(Y[~is_ref])
            idx = nn_fit.kneighbors(Y[~is_ref], return_distance=False)
            knn_frac[~is_ref] = is_ref[idx].mean(1)
        mix = np.full(len(man), np.nan)
        mix[~is_ref] = mix_fraction(Y, is_ref, k)
        coords[cond] = xy
        frames.append(pd.DataFrame({"slide_id": man["slide_id"], "site": site, "family": fam, "group": man["group"],
                                    "condition": cond, "umap1": xy[:, 0], "umap2": xy[:, 1],
                                    "knn_ebrains_frac": knn_frac, "mix_ebrains_frac": mix}))
        for f in FAMILIES:
            pm = (~is_ref) & (fam == f)
            diag.append(dict(condition=cond, family=f, n_ebrains=int((is_ref & (fam == f)).sum()), n_patho2=int(pm.sum()),
                             knn_ebrains_frac=float(np.nanmean(knn_frac[pm])) if cond != "uncorrected" else np.nan,
                             mix_ebrains_frac=float(mix[pm].mean()),
                             mix_expected=float((is_ref & (fam == f)).sum() / (fam == f).sum())))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "umap_titan.csv", index=False)
    print(f"  saved: {out_dir / 'umap_titan.csv'}")
    dg = pd.DataFrame(diag)
    print(dg.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    plot_main(coords, man, cfg, out_dir / "umap_titan.png")
    plot_families(coords, man, cfg, out_dir / "umap_titan_families.png")

    lines = ["# transfer_umap — 転移テストの 2,301 枚の UMAP（TITAN）", "",
             f"UMAP: n_neighbors {k}, min_dist {ucfg.get('min_dist', 0.01)}, random_state {ucfg.get('random_state', 42)}"
             "（umap_plot.py と同じ config の umap）。補正なしの全スライドで 1 回当てはめ、補正後の院内は transform。", "",
             f"- ‖Δ_paper‖ = {np.linalg.norm(d_paper):.3f}（髄膜腫+SFT コホートの EBRAINS 平均 − 院内平均）",
             "- 科ごとの ‖Δ_f‖ と cos(Δ_f, Δ_paper): " + "、".join(
                 f"{f} {np.linalg.norm(d):.2f} / {d @ d_paper / np.linalg.norm(d) / np.linalg.norm(d_paper):.3f}"
                 for f, d in d_own.items()), "",
             "## 科 × 条件: transform の近傍と施設の混ざり具合（768 次元）", "",
             "- knn_ebrains_frac: 写した院内スライドの、当てはめデータ（補正なしの全 2,301 枚）中の "
             f"{k} 近傍のうち EBRAINS の割合（transform がどこから位置を取るか。補正なしの院内に引かれていれば低い）",
             f"- mix_ebrains_frac: 院内スライドの、「EBRAINS ＋ その条件の院内」の中の {k} 近傍（自分を除く）のうち EBRAINS の割合",
             "- mix_expected: その科の全スライドに占める EBRAINS の割合（科の中で施設が完全に混ざったときの目安）",
             "- transform は写す点どうしの近さを使わないので、院内が院内どうしで固まっていても UMAP では EBRAINS に重なって"
             "見える。重なりの見た目は混ざり具合を過大に見せる（mix_ebrains_frac を見ること）", "",
             "| condition | family | n EBRAINS | n in-house | knn_ebrains_frac | mix_ebrains_frac | mix_expected |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for r in dg.itertuples():
        kf = "—" if np.isnan(r.knn_ebrains_frac) else f"{r.knn_ebrains_frac:.3f}"
        lines.append(f"| {r.condition} | {r.family} | {r.n_ebrains} | {r.n_patho2} | {kf} | {r.mix_ebrains_frac:.3f} | "
                     f"{r.mix_expected:.3f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"  saved: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
