"""held_out_case.py — コホートから除外した held-out スライドの症例検討（最近傍重心法）。

EBRAINS の 1 患者（pat_id 831）は 3 枚のスライドを持ち、診断が矛盾する（SFT 初発 → Atypical meningioma →
SFT 再発）。ラベルが矛盾する患者としてコホート（ラベル CSV n551）から除外した上で、この 3 枚を held-out として
「両施設の SFT / 髄膜腫の重心に当てたとき、どちらに寄るか」を見る。

読み方の前提:
    - held-out のスライドは EBRAINS（reference 側）なので、どの補正条件でも埋め込みは original のまま不変。
      条件によって変わるのは patho2 側の重心だけ（EBRAINS 側の重心はコホートが同じなので条件によらず同じ）。
    - コホートは load_data（ラベル CSV が n551 なので held-out の 3 枚は入らない）。held-out の埋め込みは
      embedding.<reference>.original（--aggregator mean は meanpool.root/original）の {case_id}.h5 から直接読む。
    - 距離・割り当て・マージンの定義は nc.py と同じ: マージン = d(髄膜腫重心) − d(SFT 重心)、> 0 で SFT。
    - 組織型の重心は shift_direction.min_n 例以上ある組織型（SFT を含む）について作り、近い順に 3 つ出す。
    - 物差しとして、3 枚どうしの距離と、EBRAINS コホート内のスライド対距離（SFT−SFT / 髄膜腫−髄膜腫 / SFT−髄膜腫）
      の中央値を出す。

条件: utils.display.CONDITIONS_FIG3（補正なし → Macenko → GAN → ComBat → 平行移動）＋ combat_cov（あれば）。
    無い variant は飛ばす。--aggregator mean は config の meanpool.variants ＋ combat_cov（あれば）。

出力 (output_dir/held_out_case/):
    summary.csv   — 縦長: aggregator, condition, centroid_site, case_id, diagnosis, d_sft, d_men, assigned, margin,
                    nearest_subtypes
    pairwise.csv  — 3 枚どうしの距離（aggregator ごと）
    reference_pairs.csv — EBRAINS コホート内のスライド対距離の中央値（aggregator ごと）
    summary.md    — 人が読む表と、Results に書くための短い所見

Usage:
    uv run python held_out_case.py                      # TITAN と平均プーリングの両方
    uv run python held_out_case.py --aggregator titan   # TITAN だけ
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from utils.display import CONDITION_LABELS_JA, CONDITIONS_FIG3, order_conditions, ordered_subtypes, shorten
from utils.loader import load_config, load_data

SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}
EXTRA_CONDITIONS = ["combat_cov"]   # FIG3 に無いが、生成物があれば参考として含める
N_NEAREST = 3


# ── 読み込み ──────────────────────────────────────────────────────────────────

def _is_sft(s: pd.Series) -> np.ndarray:
    return s.str.contains("sft", case=False).to_numpy()


def held_out_cases(cfg: dict) -> pd.DataFrame:
    """config の held_out.<site> から case_id と元の診断名（held_out.subtype.<site> の CSV から）を引く。"""
    ho = cfg.get("held_out", {})
    rows = []
    for site, cases in ho.items():
        if site == "subtype" or not isinstance(cases, list):
            continue
        csv = ho.get("subtype", {}).get(site)
        df_sub = pd.read_csv(csv).set_index("case_id") if csv else None
        for case_id in cases:
            diag = str(df_sub.loc[case_id, "subtype"]) if df_sub is not None and case_id in df_sub.index else "?"
            rows.append({"site": site, "case_id": case_id, "diagnosis": diag})
    return pd.DataFrame(rows)


def load_held_out(cfg: dict, cases: pd.DataFrame, aggregator: str) -> pd.DataFrame:
    """held-out スライドの埋め込みを original（reference 側は常に original）の h5 から直接読む。"""
    recs = []
    for site, g in cases.groupby("site"):
        ds_cfg = cfg["embedding"][site]
        if aggregator == "mean":
            mp = cfg["meanpool"]
            emb_dir = Path(mp["root"].format(site=site)) / "original"
            h5_key = f"{ds_cfg['tile_model']}/{mp['slide_feature']}"
        else:
            emb_dir = Path(ds_cfg["original"])
            h5_key = f"{ds_cfg['tile_model']}/{cfg['keys']['slide_feature']}"
        for _, r in g.iterrows():
            path = emb_dir / f"{r['case_id']}.h5"
            if not path.exists():
                print(f"  WARN: held-out h5 not found: {path}")
                continue
            with h5py.File(path, "r") as f:
                if h5_key not in f:
                    print(f"  WARN: '{h5_key}' not in {path}")
                    continue
                recs.append({**r.to_dict(), "embedding": f[h5_key][:].astype(np.float64)})
    return pd.DataFrame(recs)


def available_conditions(cfg: dict, aggregator: str) -> list[str]:
    """使う条件（FIG3 ＋ combat_cov）のうち、対象施設（非 reference）の生成物があるもの。"""
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    base = list(cfg["meanpool"]["variants"]) if aggregator == "mean" else list(CONDITIONS_FIG3)
    out = []
    for v in order_conditions(base + [c for c in EXTRA_CONDITIONS if c not in base]):
        dir_key = cfg.get("variants", {}).get(v, v)
        if aggregator == "mean":
            d = Path(cfg["meanpool"]["root"].format(site=other)) / dir_key
        else:
            d = Path(cfg["embedding"][other].get(dir_key, "")) if dir_key in cfg["embedding"][other] else None
        if d is not None and d.exists() and any(d.glob("*.h5")):
            out.append(v)
        else:
            print(f"  SKIP [{aggregator} / {v}]: no data")
    return out


# ── 計算 ──────────────────────────────────────────────────────────────────────

def assign(cohort: pd.DataFrame, held: pd.DataFrame, cfg: dict, condition: str, aggregator: str) -> list[dict]:
    """各施設の重心（SFT / 髄膜腫 / 組織型）に held-out スライドを当てる。"""
    min_n: int = cfg.get("shift_direction", {}).get("min_n", 5)
    X = np.stack(cohort["embedding"].values).astype(np.float64)
    src, sub, sft = cohort["source"].to_numpy(), cohort["subtype"].to_numpy(), _is_sft(cohort["subtype"])
    H = np.stack(held["embedding"].values)
    rows = []
    for site in cfg["embedding"]:
        m = src == site
        if not m.any():
            continue
        C = np.stack([X[m & ~sft].mean(0), X[m & sft].mean(0)])   # 0 = 髄膜腫, 1 = SFT（nc.py と同じ）
        D = cdist(H, C)
        groups = [g for g in ordered_subtypes(set(sub[m]), cfg) if ((sub == g) & m).sum() >= min_n]
        G = np.stack([X[m & (sub == g)].mean(0) for g in groups])
        DG = cdist(H, G)
        for i, (_, h) in enumerate(held.iterrows()):
            order = np.argsort(DG[i])[:N_NEAREST]
            nearest = "; ".join(f"{shorten(groups[j], cfg)} ({DG[i, j]:.2f})" for j in order)
            margin = float(D[i, 0] - D[i, 1])
            rows.append({"aggregator": aggregator, "condition": condition, "centroid_site": site,
                         "n_sft": int((m & sft).sum()), "n_men": int((m & ~sft).sum()),
                         "case_id": h["case_id"], "diagnosis": h["diagnosis"],
                         "d_sft": float(D[i, 1]), "d_men": float(D[i, 0]),
                         "assigned": "SFT" if margin > 0 else "Meningioma", "margin": margin,
                         "nearest_subtypes": nearest})
    return rows


def pairwise(held: pd.DataFrame, aggregator: str) -> pd.DataFrame:
    H = np.stack(held["embedding"].values)
    rows = []
    for i, j in combinations(range(len(held)), 2):
        rows.append({"aggregator": aggregator, "case_a": held.iloc[i]["case_id"], "diagnosis_a": held.iloc[i]["diagnosis"],
                     "case_b": held.iloc[j]["case_id"], "diagnosis_b": held.iloc[j]["diagnosis"],
                     "distance": float(np.linalg.norm(H[i] - H[j]))})
    return pd.DataFrame(rows)


def reference_pairs(cohort: pd.DataFrame, cfg: dict, aggregator: str) -> pd.DataFrame:
    """reference 施設のコホート内のスライド対距離の中央値（SFT−SFT / 髄膜腫−髄膜腫 / SFT−髄膜腫）。"""
    reference: str = cfg["reference"]
    m = cohort["source"].to_numpy() == reference
    X = np.stack(cohort.loc[m, "embedding"].values).astype(np.float64)
    sft = _is_sft(cohort.loc[m, "subtype"])
    rows = [
        {"pair": "SFT-SFT", "n_pairs": int(sft.sum() * (sft.sum() - 1) // 2), "d": pdist(X[sft])},
        {"pair": "Meningioma-Meningioma", "n_pairs": int((~sft).sum() * ((~sft).sum() - 1) // 2), "d": pdist(X[~sft])},
        {"pair": "SFT-Meningioma", "n_pairs": int(sft.sum() * (~sft).sum()), "d": cdist(X[sft], X[~sft]).ravel()},
    ]
    return pd.DataFrame([{"aggregator": aggregator, "site": reference, "pair": r["pair"], "n_pairs": r["n_pairs"],
                          "median": float(np.median(r["d"])), "q25": float(np.percentile(r["d"], 25)),
                          "q75": float(np.percentile(r["d"], 75))} for r in rows])


# ── 書き出し ──────────────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n"
    return head + body


def findings(summary: pd.DataFrame, held: pd.DataFrame, cfg: dict, pw: pd.DataFrame | None = None,
             rp: pd.DataFrame | None = None) -> list[str]:
    """データから機械的に組む所見（断定しない書き方。Results の下書きの材料）。"""
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    out = []
    for agg, sa in summary.groupby("aggregator", sort=False):
        agg_label = "TITAN" if agg == "titan" else "平均プーリング"
        for _, h in held.iterrows():
            s = sa[sa["case_id"] == h["case_id"]]
            ref = s[s["centroid_site"] == reference].iloc[0]
            oth = s[s["centroid_site"] == other]
            n_sft = int((oth["assigned"] == "SFT").sum())
            out.append(f"[{agg_label}] {h['case_id']}（{h['diagnosis']}）: {SITE_LABEL[reference]} の重心では "
                       f"{ref['assigned']}（マージン {ref['margin']:+.2f}）。{SITE_LABEL[other]} の重心では "
                       f"{len(oth)} 条件中 {n_sft} 条件で SFT"
                       + "".join(f"、{CONDITION_LABELS_JA.get(r['condition'], r['condition'])} {r['margin']:+.2f}"
                                 for _, r in oth.iterrows()) + "。")
        # 3 枚が同じ側か（reference 側の重心は条件によらず同じなので、スライドごとに 1 回数える）
        ref_all = sa[sa["centroid_site"] == reference].drop_duplicates("case_id").set_index("case_id")["assigned"]
        sides = ref_all.value_counts()
        if len(sides) == 1:
            out.append(f"[{agg_label}] {SITE_LABEL[reference]} の重心では 3 枚とも {sides.index[0]} 側。")
        else:
            out.append(f"[{agg_label}] {SITE_LABEL[reference]} の重心では "
                       + " / ".join(f"{k} {v} 枚" for k, v in sides.items()) + " に分かれる。")
        agree = (sa.assign(ok=sa["assigned"].eq("SFT") == sa["diagnosis"].str.contains("sft", case=False))
                   .groupby("case_id")["ok"].mean())
        out.append(f"[{agg_label}] 割り当てが元の診断と一致する割合（全条件 × 両施設）: "
                   + ", ".join(f"{c} {v:.0%}" for c, v in agree.items()) + "。")
        if pw is not None and rp is not None:
            p, r = pw[pw["aggregator"] == agg], rp[rp["aggregator"] == agg].set_index("pair")["median"]
            out.append(f"[{agg_label}] 3 枚どうしの距離は {p['distance'].min():.2f}〜{p['distance'].max():.2f}。"
                       f"{SITE_LABEL[reference]} コホートの同クラス対の中央値（SFT−SFT {r['SFT-SFT']:.2f} / "
                       f"髄膜腫−髄膜腫 {r['Meningioma-Meningioma']:.2f}）と比べて"
                       + ("小さく、同一患者の 3 枚は互いに近い。" if p["distance"].max() < min(r["SFT-SFT"], r["Meningioma-Meningioma"])
                          else "同程度かそれ以上で、同一患者でも互いに特に近いとは言えない。")
                       + f" SFT−髄膜腫対の中央値は {r['SFT-Meningioma']:.2f}。")
    return out


def write_md(path: Path, summary: pd.DataFrame, pw: pd.DataFrame, rp: pd.DataFrame, held: pd.DataFrame,
             cfg: dict) -> None:
    reference: str = cfg["reference"]
    other = next(s for s in cfg["embedding"] if s != reference)
    min_n: int = cfg.get("shift_direction", {}).get("min_n", 5)
    L = ["# held-out 症例検討（最近傍重心法）", "",
         "コホート（ラベル CSV の n）から除外した held-out スライドを、両施設の SFT / 髄膜腫の重心に当てたときの距離と割り当て。", "",
         f"- held-out のスライドは {SITE_LABEL[reference]}（reference 側）なので、埋め込みはどの補正条件でも original のまま"
         f"**不変**。条件によって変わるのは **{SITE_LABEL[other]} 側の重心だけ**（{SITE_LABEL[reference]} 側の重心は"
         "コホートが同じなので条件によらず同じ値）。",
         "- held-out の 3 枚はコホートに入っていない（ラベル CSV が除外後のもの）ので、重心には寄与していない。",
         "- マージン = d(髄膜腫重心) − d(SFT 重心)、> 0 で SFT（nc.py と同じ向き）。",
         f"- 組織型の重心はその施設に {min_n} 例以上ある組織型（SFT を含む）。近い順に 3 つ（括弧は距離）。", ""]
    L += ["## held-out スライド", "", _md_table(held, ["site", "case_id", "diagnosis"]), ""]
    fmt = {"d_sft": "{:.2f}", "d_men": "{:.2f}", "margin": "{:+.2f}"}
    for agg, sa in summary.groupby("aggregator", sort=False):
        L += [f"## 集約器: {'TITAN' if agg == 'titan' else '平均プーリング（CONCH パッチ埋め込みの単純平均）'}", ""]
        for cond in order_conditions(sa["condition"].unique()):
            sc = sa[sa["condition"] == cond]
            L += [f"### 条件: {CONDITION_LABELS_JA.get(cond, cond)}（`{cond}`）", ""]
            for site in cfg["embedding"]:
                ss = sc[sc["centroid_site"] == site]
                if ss.empty:
                    continue
                note = "（条件によらず同じ）" if site == reference else ""
                L += [f"重心: {SITE_LABEL[site]}{note}（SFT n={ss['n_sft'].iloc[0]}, 髄膜腫 n={ss['n_men'].iloc[0]}）", "",
                      _md_table(ss, ["case_id", "diagnosis", "d_sft", "d_men", "assigned", "margin", "nearest_subtypes"], fmt),
                      ""]
        L += ["### 物差し", "", "3 枚どうしの距離:", "",
              _md_table(pw[pw["aggregator"] == agg], ["case_a", "diagnosis_a", "case_b", "diagnosis_b", "distance"],
                        {"distance": "{:.2f}"}), "",
              f"{SITE_LABEL[reference]} コホート内のスライド対距離（中央値 [四分位]）:", "",
              _md_table(rp[rp["aggregator"] == agg], ["pair", "n_pairs", "median", "q25", "q75"],
                        {"median": "{:.2f}", "q25": "{:.2f}", "q75": "{:.2f}"}), ""]
    L += ["## 所見（Results の下書きの材料。数字から機械的に組んだもので、解釈は断定しない）", ""]
    L += [f"- {s}" for s in findings(summary, held, cfg, pw, rp)]
    L += ["", "注: 補正条件の patho2 側の重心は、その条件の生成物（centroid / combat など）が作られた時点のコホートに依存する。"
          "生成物を作り直したら再実行すること。"]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregator", choices=["titan", "mean", "both"], default="both",
                    help="スライド埋め込みの集約器（titan / mean = 平均プーリングの対照 / both = 両方、既定）")
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "held_out_case"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = held_out_cases(cfg)
    if cases.empty:
        print("  SKIP: config に held_out が無い")
        return
    print(f"[held_out_case] {len(cases)} held-out slides: " + ", ".join(f"{r.case_id} ({r.diagnosis})" for r in cases.itertuples()))

    aggregators = ["titan", "mean"] if args.aggregator == "both" else [args.aggregator]
    rows, pws, rps, held_any = [], [], [], None
    for agg in aggregators:
        held = load_held_out(cfg, cases, agg)
        if held.empty:
            print(f"  SKIP [{agg}]: held-out embeddings not found")
            continue
        held_any = held if held_any is None else held_any
        for v in available_conditions(cfg, agg):
            cohort = load_data(cfg, cfg.get("variants", {}).get(v, v), aggregator=agg)
            if cohort.empty:
                print(f"  SKIP [{agg} / {v}]: no cohort")
                continue
            leak = set(cohort["case_id"]) & set(held["case_id"])
            if leak:
                raise RuntimeError(f"held-out slides are in the cohort (label CSV not excluded?): {sorted(leak)}")
            if v == "original":
                pws.append(pairwise(held, agg))
                rps.append(reference_pairs(cohort, cfg, agg))
            rows += assign(cohort, held, cfg, v, agg)
            print(f"  [{agg} / {v}] cohort n={len(cohort)} "
                  + "  ".join(f"{r['centroid_site']}:{r['case_id']}={r['assigned'][:3]}({r['margin']:+.2f})"
                              for r in rows if r["aggregator"] == agg and r["condition"] == v))
    if not rows:
        print("  nothing computed")
        return
    summary = pd.DataFrame(rows)
    pw, rp = pd.concat(pws, ignore_index=True), pd.concat(rps, ignore_index=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pw.to_csv(out_dir / "pairwise.csv", index=False)
    rp.to_csv(out_dir / "reference_pairs.csv", index=False)
    write_md(out_dir / "summary.md", summary, pw, rp, held_any[["site", "case_id", "diagnosis"]], cfg)
    print(f"  saved: {out_dir}/{{summary.csv,pairwise.csv,reference_pairs.csv,summary.md}}")


if __name__ == "__main__":
    main()
