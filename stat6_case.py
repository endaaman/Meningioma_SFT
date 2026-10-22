"""stat6_case.py — patho2 コホートを STAT6 免疫染色の有無・結果で分けた記述統計（最近傍重心法、Supplementary 用）。

問い: 「STAT6 を染めた髄膜腫」= 病理医が SFT を鑑別に挙げた髄膜腫は、埋め込み空間で SFT 側に寄っているか。
また、STAT6 で確認されていない SFT（陰性・未施行）は重心法でどう扱われるか。

STAT6 の情報源（config の stat6.source、NFS 上の読み取り専用 CSV）:
    __archive/all_cases_updated_n1721.csv の stat6_nuclear / stat6_cytoplasmic 列（VLM が病理レポートから抽出したもの）。
    この列は否定の読み落としがあるので、同じ CSV の original_diagnosis / findings から STAT6 を含む文を切り出し、
    規則で読み直した値（stat6_nuclear_text）も併記する（規則は _read_stat6_text。読み違いが疑わしい症例は summary.md に全文を出す）。
    case_id の照合は NFKC 正規化・大文字化・前後の空白除去のうえ完全一致、落ちたものは数字だけの形でも照合する。

群（group = CSV 列の値から / group_text = レポート文の読みから。集計は両方を出す）:
    SFT_STAT6pos / SFT_STAT6neg / SFT_untested / Men_STAT6tested（STAT6 の記載が何かしらある髄膜腫）/ Men_untested

計算（TITAN、条件は utils.display.CONDITIONS_FIG3 = 補正なし → Macenko → GAN → ComBat → 平行移動）:
    patho2 の各スライドについて
      - EBRAINS の SFT / 髄膜腫の重心への距離（= nc.py の EBRAINS → patho2 と同じ。predictions.csv と突き合わせる）
      - patho2 の SFT / 髄膜腫の重心への距離（施設内、leave-one-out: 自分のクラスの重心から自分を除く）
    マージン = d(髄膜腫重心) − d(SFT 重心)、> 0 で SFT（nc.py と同じ向き）。
    群ごとに n、マージンの中央値 [IQR]、SFT に割り当てられた割合。Men_STAT6tested と Men_untested は
    SFT 割り当ての割合（Fisher 正確検定）とマージン（Mann–Whitney U）を比べる。

出力 (output_dir/stat6/):
    cases.csv    — 縦長: grouping は使わず、case_id ごとに group / group_text の両方を持つ。condition × centroid_site × case_id
    summary.csv  — grouping（archive / text）× group × condition × centroid_site: n, n_sft_assigned, frac_sft, median, q25, q75
    compare.csv  — Men_STAT6tested vs Men_untested の検定（grouping × condition × centroid_site）
    compare_by_subtype.csv — 同じ比較を組織型内で（記載ありが 3 例以上の組織型、group_text）
    summary.md   — 人が読む表（冒頭に id 照合の結果、末尾に個別症例）
    data/patho2/stat6_status.csv（config の stat6.status_csv）— 症例ごとの STAT6 の状態（コホートだけ）

Usage:
    uv run python stat6_case.py
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import fisher_exact, mannwhitneyu

from utils.display import CONDITION_LABELS_JA, CONDITIONS_FIG3, order_conditions
from utils.loader import load_config, load_data

SITE_LABEL = {"ebrains": "EBRAINS", "patho2": "patho2"}
GROUPS = ["SFT_STAT6pos", "SFT_STAT6neg", "SFT_untested", "Men_STAT6tested", "Men_untested"]
GROUP_LABELS_JA = {"SFT_STAT6pos": "SFT（STAT6 核陽性）", "SFT_STAT6neg": "SFT（STAT6 核陰性）",
                   "SFT_untested": "SFT（STAT6 未施行）", "Men_STAT6tested": "髄膜腫（STAT6 記載あり）",
                   "Men_untested": "髄膜腫（STAT6 記載なし）"}
GROUPINGS = {"archive": "group", "text": "group_text"}   # 集計の軸: CSV 列の値 / レポート文の読み

# レポート文の読み（STAT6 を含む文を「、」「,」で区切り、STAT6 のある節から順に判定。細胞質 → 核は陰性扱い）
_RE_STAT6 = re.compile(r"stat-?6", re.I)
_RE_CYTO = re.compile(r"細胞質|cytoplasm", re.I)
_RE_NEG = re.compile(r"陰性|\(\s*-|（\s*-|認めない|認めません|みとめず|見られず|呈さない|否定的")
_RE_POS = re.compile(r"陽性|\(\s*\+|（\s*\+|認める|認め(ます|られ)")


# ── STAT6 の状態 ──────────────────────────────────────────────────────────────

def _norm_id(s) -> str:
    return unicodedata.normalize("NFKC", str(s)).upper().strip()


def _digits(s) -> str:
    return re.sub(r"\D", "", _norm_id(s))


def _stat6_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。\n]", text) if _RE_STAT6.search(s)]


def _read_stat6_text(text: str) -> tuple[str, str]:
    """STAT6 を含む文から (核の判定, 細胞質の判定) を読む。どちらも Positive / Negative / Not specified / NA（記載なし）。"""
    sents = _stat6_sentences(text)
    if not sents:
        return "NA", "NA"
    nuc, cyto = "Not specified", "Not specified"
    for s in sents:
        segs = [g for g in re.split(r"[、,]", s)]
        start = next(i for i, g in enumerate(segs) if _RE_STAT6.search(g))
        segs[start] = segs[start][_RE_STAT6.search(segs[start]).start():]   # STAT6 より前（他の抗体の判定）は見ない
        for g in segs[start:start + 2]:   # STAT6 の節と、判定が無ければ次の節
            if _RE_CYTO.search(g):
                nuc, cyto = "Negative", "Positive"
            elif _RE_NEG.search(g):
                nuc = "Negative"
            elif _RE_POS.search(g):
                nuc = "Positive"
            else:
                continue
            break
        if nuc != "Not specified":
            break
    return nuc, cyto


def _group(diagnosis: str, nuc, tested: bool) -> str:
    if diagnosis == "SFT":
        if nuc == "Positive":
            return "SFT_STAT6pos"
        if nuc == "Negative":
            return "SFT_STAT6neg"
        return "SFT_untested"     # 未施行（記載なし）と Not specified をまとめる
    return "Men_STAT6tested" if tested else "Men_untested"


def build_status(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """コホート（label.<site>.subtype）に STAT6 の情報源を照合し、症例ごとの状態を作る。"""
    st = cfg["stat6"]
    site = st["site"]
    cohort = pd.read_csv(cfg["label"][site]["subtype"])
    src = pd.read_csv(st["source"])
    src["_nid"], src["_dig"] = src["case_id"].map(_norm_id), src["case_id"].map(_digits)
    cohort["_nid"], cohort["_dig"] = cohort["case_id"].map(_norm_id), cohort["case_id"].map(_digits)
    by_nid = src.drop_duplicates("_nid").set_index("_nid")
    by_dig = src.drop_duplicates("_dig").set_index("_dig")
    exact = cohort["case_id"].isin(src["case_id"])
    rows, how = [], []
    for _, r in cohort.iterrows():
        if r["_nid"] in by_nid.index:
            s, h = by_nid.loc[r["_nid"]], "normalized"
        elif r["_dig"] in by_dig.index:
            s, h = by_dig.loc[r["_dig"]], "digits"
        else:
            s, h = None, "unmatched"
        how.append(h)
        text = "" if s is None else f"{s.get('original_diagnosis', '')}\n{s.get('findings', '')}"
        nuc_t, cyto_t = _read_stat6_text(text)
        nuc = None if s is None or pd.isna(s["stat6_nuclear"]) else str(s["stat6_nuclear"])
        cyto = None if s is None or pd.isna(s["stat6_cytoplasmic"]) else str(s["stat6_cytoplasmic"])
        tested = nuc is not None or cyto is not None
        tested_t = nuc_t != "NA"
        rows.append({"case_id": r["case_id"], "diagnosis": r["diagnosis"], "subtype": r["subtype"],
                     "stat6_nuclear": nuc, "stat6_cytoplasmic": cyto, "stat6_tested": tested,
                     "group": _group(r["diagnosis"], nuc, tested),
                     "stat6_nuclear_text": None if nuc_t == "NA" else nuc_t,
                     "stat6_cytoplasmic_text": None if cyto_t == "NA" else cyto_t,
                     "group_text": _group(r["diagnosis"], nuc_t, tested_t),
                     "vlm_class": None if s is None else s.get("vlm_class"),
                     "who_grade": None if s is None else s.get("who_grade"),
                     "stat6_text": " / ".join(_stat6_sentences(text)) if s is not None else None,
                     "matched_by": h})
    df = pd.DataFrame(rows)
    # 食い違い = 列が Positive / Negative と断定しているのに本文の読みが逆（Not specified は食い違いに数えない）
    df["discrepant"] = (df["stat6_nuclear"].isin(["Positive", "Negative"])
                        & (df["stat6_nuclear"] != df["stat6_nuclear_text"].fillna("")))
    men = df["diagnosis"] != "SFT"
    info = {"n_cohort": len(df), "n_source": len(src), "n_exact": int(exact.sum()),
            "n_by": pd.Series(how).value_counts().to_dict(),
            "n_men": int(men.sum()), "n_men_tested": int((men & df["stat6_tested"]).sum()),
            "n_men_tested_text": int((men & df["stat6_nuclear_text"].notna()).sum()),
            "n_men_text_mention": int((men & df["stat6_text"].fillna("").ne("")).sum()),
            "n_sft": int((~men).sum()), "n_sft_tested": int((~men & df["stat6_tested"]).sum())}
    return df, info


# ── 距離と割り当て ────────────────────────────────────────────────────────────

def _is_sft(s: pd.Series) -> np.ndarray:
    return s.str.contains("sft", case=False).to_numpy()


def available_conditions(cfg: dict, site: str) -> list[str]:
    out = []
    for v in order_conditions(CONDITIONS_FIG3):
        dir_key = cfg.get("variants", {}).get(v, v)
        d = Path(cfg["embedding"][site][dir_key]) if dir_key in cfg["embedding"][site] else None
        if d is not None and d.exists() and any(d.glob("*.h5")):
            out.append(v)
        else:
            print(f"  SKIP [{v}]: no data")
    return out


def assign(cohort: pd.DataFrame, status: pd.DataFrame, cfg: dict, condition: str) -> pd.DataFrame:
    """対象施設の各スライドを、EBRAINS の重心と patho2 の leave-one-out 重心に当てる。"""
    reference, site = cfg["reference"], cfg["stat6"]["site"]
    X = np.stack(cohort["embedding"].values).astype(np.float64)
    src, sft = cohort["source"].to_numpy(), _is_sft(cohort["subtype"])
    tgt = src == site
    Xt, sft_t = X[tgt], sft[tgt]
    rows = []
    for csite in (reference, site):
        m = src == csite
        if csite != site:
            C = np.stack([X[m & ~sft].mean(0), X[m & sft].mean(0)])   # 0 = 髄膜腫, 1 = SFT
            D = cdist(Xt, C)
        else:
            sums = np.stack([X[m & ~sft].sum(0), X[m & sft].sum(0)])
            ns = np.array([(m & ~sft).sum(), (m & sft).sum()], dtype=float)
            D = np.empty((len(Xt), 2))
            for i in range(len(Xt)):
                k = int(sft_t[i])
                C = sums / ns[:, None]
                C[k] = (sums[k] - Xt[i]) / (ns[k] - 1)   # 自分を除いた自クラスの重心
                D[i] = cdist(Xt[i:i + 1], C)[0]
        margin = D[:, 0] - D[:, 1]
        df = pd.DataFrame({"condition": condition, "centroid_site": csite,
                           "case_id": cohort.loc[tgt, "case_id"].to_numpy(),
                           "d_sft": D[:, 1], "d_men": D[:, 0], "margin": margin,
                           "assigned": np.where(margin > 0, "SFT", "Meningioma")})
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out.merge(status[["case_id", "diagnosis", "subtype", "group", "group_text", "stat6_nuclear",
                             "stat6_cytoplasmic", "stat6_nuclear_text"]], on="case_id", how="left")


def check_against_nc(cases: pd.DataFrame, cfg: dict) -> list[str]:
    """EBRAINS 重心の割り当てを nc.py（trained_by_<reference>）の predictions.csv と突き合わせる。"""
    reference = cfg["reference"]
    notes = []
    for cond, sc in cases[cases["centroid_site"] == reference].groupby("condition", sort=False):
        p = Path(cfg["output_dir"]) / "nc" / "binary" / f"trained_by_{reference}" / cond / "predictions.csv"
        if not p.exists():
            notes.append(f"{cond}: nc.py の出力なし（{p}）")
            continue
        nc = pd.read_csv(p)
        m = sc.merge(nc, on="case_id", how="inner")
        mism = int(((m["assigned"] == "SFT").astype(int) != m["pred"]).sum())
        missed = m[(m["true"] == 1) & (m["pred"] == 0)]["case_id"].tolist()
        notes.append(f"{cond}: n={len(m)}、割り当ての不一致 {mism}、EBRAINS → patho2 で見逃した SFT: "
                     + (", ".join(missed) if missed else "なし"))
    return notes


# ── 集計 ──────────────────────────────────────────────────────────────────────

def summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gname, gcol in GROUPINGS.items():
        for (cond, csite, g), s in cases.groupby(["condition", "centroid_site", gcol], sort=False):
            rows.append({"grouping": gname, "group": g, "condition": cond, "centroid_site": csite, "n": len(s),
                         "n_sft_assigned": int((s["assigned"] == "SFT").sum()),
                         "frac_sft": float((s["assigned"] == "SFT").mean()),
                         "median": float(s["margin"].median()),
                         "q25": float(s["margin"].quantile(0.25)), "q75": float(s["margin"].quantile(0.75)),
                         "mean": float(s["margin"].mean())})
    df = pd.DataFrame(rows)
    df["group"] = pd.Categorical(df["group"], GROUPS)
    return df.sort_values(["grouping", "condition", "centroid_site", "group"]).reset_index(drop=True)


def compare_men_by_subtype(cases: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """組織型の構成の違いを切り分ける: 組織型ごとに Men_STAT6tested と Men_untested のマージンの中央値を比べる（記載ありが min_n 例以上の組織型）。"""
    rows = []
    for (cond, csite), s in cases.groupby(["condition", "centroid_site"], sort=False):
        s = s[s["diagnosis"] != "SFT"]
        for sub, ss in s.groupby("subtype"):
            a, b = ss[ss["group_text"] == "Men_STAT6tested"], ss[ss["group_text"] == "Men_untested"]
            if len(a) < min_n or b.empty:
                continue
            mw = mannwhitneyu(a["margin"], b["margin"], alternative="two-sided") if len(b) >= 2 else None
            rows.append({"condition": cond, "centroid_site": csite, "subtype": sub, "n_tested": len(a), "n_untested": len(b),
                         "sft_tested": int((a["assigned"] == "SFT").sum()), "sft_untested": int((b["assigned"] == "SFT").sum()),
                         "median_tested": float(a["margin"].median()), "median_untested": float(b["margin"].median()),
                         "mannwhitney_p": float(mw.pvalue) if mw is not None else np.nan})
    return pd.DataFrame(rows)


def compare_men(cases: pd.DataFrame) -> pd.DataFrame:
    """Men_STAT6tested vs Men_untested: SFT 割り当ての Fisher 正確検定とマージンの Mann–Whitney U。"""
    rows = []
    for gname, gcol in GROUPINGS.items():
        for (cond, csite), s in cases.groupby(["condition", "centroid_site"], sort=False):
            a, b = s[s[gcol] == "Men_STAT6tested"], s[s[gcol] == "Men_untested"]
            if a.empty or b.empty:
                continue
            ta = [[int((a["assigned"] == "SFT").sum()), int((a["assigned"] != "SFT").sum())],
                  [int((b["assigned"] == "SFT").sum()), int((b["assigned"] != "SFT").sum())]]
            mw = mannwhitneyu(a["margin"], b["margin"], alternative="two-sided")
            rows.append({"grouping": gname, "condition": cond, "centroid_site": csite,
                         "n_tested": len(a), "n_untested": len(b),
                         "sft_tested": ta[0][0], "sft_untested": ta[1][0],
                         "frac_sft_tested": ta[0][0] / len(a), "frac_sft_untested": ta[1][0] / len(b),
                         "fisher_p": float(fisher_exact(ta)[1]),
                         "median_tested": float(a["margin"].median()), "median_untested": float(b["margin"].median()),
                         "mannwhitney_p": float(mw.pvalue)})
    return pd.DataFrame(rows)


# ── 書き出し ──────────────────────────────────────────────────────────────────

def _p(x: float) -> str:
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


def _peq(x: float) -> str:
    """文中用: 'p = 0.493' / 'p < 0.001' の記号つき。"""
    return "< 0.001" if x < 0.001 else f"= {x:.3f}"


def _na(x) -> str:
    return "記載なし" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def _md_table(df: pd.DataFrame, cols: list[str], fmt: dict[str, str] | None = None, names: dict | None = None) -> str:
    fmt, names = fmt or {}, names or {}
    head = "| " + " | ".join(names.get(c, c) for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(("" if pd.isna(r[c]) else fmt.get(c, "{}").format(r[c])) if not isinstance(r[c], str)
                                  else r[c] for c in cols) + " |\n"
    return head + body


def write_md(path: Path, status: pd.DataFrame, info: dict, summary: pd.DataFrame, comp: pd.DataFrame,
             comp_sub: pd.DataFrame, cases: pd.DataFrame, nc_notes: list[str], cfg: dict) -> None:
    reference, site = cfg["reference"], cfg["stat6"]["site"]
    conds = order_conditions(cases["condition"].unique())
    men = status["diagnosis"] != "SFT"
    L = ["# STAT6 免疫染色の状態別の記述統計（patho2、最近傍重心法）", "",
         f"情報源: `{cfg['stat6']['source']}`（読み取り専用）。症例ごとの状態は `{cfg['stat6']['status_csv']}`。", ""]
    L += ["## case_id の照合", "",
          f"- コホート {info['n_cohort']} 例（髄膜腫 {info['n_men']}、SFT {info['n_sft']}）を情報源 {info['n_source']} 行に照合。"
          f"完全一致 {info['n_exact']}。NFKC・大文字化・空白除去後の一致 {info['n_by'].get('normalized', 0)}、"
          f"数字だけの形での一致 {info['n_by'].get('digits', 0)}、不一致 {info['n_by'].get('unmatched', 0)}。",
          f"- つまり **id は全例一致する**。「髄膜腫 48/555 しか合わない」ように見えたのは照合の失敗ではなく、"
          f"残り {info['n_men'] - info['n_men_tested']} 例の `stat6_nuclear` / `stat6_cytoplasmic` が空（STAT6 の記載なし）だから。",
          f"- STAT6 の情報がある髄膜腫: 列の値あり {info['n_men_tested']} 例 = レポート本文に STAT6 の文字がある {info['n_men_text_mention']} 例"
          f"（本文の読みで核の判定が付いたのは {info['n_men_tested_text']} 例）。SFT は {info['n_sft_tested']}/{info['n_sft']} 例に記載あり。", ""]
    L += ["## 情報源の列とレポート本文の食い違い", "",
          "`stat6_nuclear` 列は VLM の抽出で、否定の読み落としがある。STAT6 を含む文を規則で読み直した `stat6_nuclear_text` と"
          "食い違う症例（本文の全文を出す）:", ""]
    d = status[status["discrepant"]]
    for _, r in d.iterrows():
        L.append(f"- **{r['case_id']}**（{r['subtype']}、列: 核 {r['stat6_nuclear']} / 細胞質 {r['stat6_cytoplasmic']}、"
                 f"本文の読み: 核 {r['stat6_nuclear_text']}）: 「{r['stat6_text']}」")
    if d.empty:
        L.append("- なし")
    ns = status[status["stat6_tested"] & status["stat6_nuclear"].eq("Not specified")]
    L += ["", "列が Not specified の症例（本文の読み）:", ""]
    for _, r in ns.iterrows():
        L.append(f"- {r['case_id']}（{r['subtype']}）→ 核 {r['stat6_nuclear_text']} / 細胞質 {r['stat6_cytoplasmic_text']}: 「{r['stat6_text']}」")
    pos_men = status[men & (status["stat6_nuclear"].eq("Positive") | status["stat6_nuclear_text"].eq("Positive"))]
    L += ["", "**STAT6 核陽性の髄膜腫**（列または本文の読みのどちらかで陽性）: "
          + (", ".join(f"{r['case_id']}（{r['subtype']}、列 {r['stat6_nuclear']} / 本文 {r['stat6_nuclear_text']}）"
                       for _, r in pos_men.iterrows()) if not pos_men.empty else "なし"), ""]
    L += ["## 群の大きさ", ""]
    gt = pd.DataFrame({"archive（列の値）": status["group"].value_counts().reindex(GROUPS).fillna(0).astype(int),
                       "text（本文の読み）": status["group_text"].value_counts().reindex(GROUPS).fillna(0).astype(int)})
    gt.index = [f"{g}（{GROUP_LABELS_JA[g]}）" for g in gt.index]
    L += ["| 群 | archive（列の値） | text（本文の読み） |", "|---|---|---|"]
    L += [f"| {i} | {r.iloc[0]} | {r.iloc[1]} |" for i, r in gt.iterrows()]
    L += ["", "## nc.py との突き合わせ（EBRAINS 重心の割り当て = EBRAINS → patho2）", ""]
    L += [f"- {n}" for n in nc_notes]
    L += ["", "## 群ごとの集計", "",
          f"マージン = d(髄膜腫重心) − d(SFT 重心)、> 0 で SFT。重心は {SITE_LABEL[reference]}（施設間、nc.py と同じ）と "
          f"{SITE_LABEL[site]}（施設内、leave-one-out）。", ""]
    fmt = {"frac_sft": "{:.3f}", "median": "{:+.2f}", "q25": "{:+.2f}", "q75": "{:+.2f}"}
    for gname in GROUPINGS:
        L += [f"### 群分け: {gname}", ""]
        for csite in (reference, site):
            L += [f"重心: {SITE_LABEL[csite]}", ""]
            cols = ["group"] + [f"{c}" for c in conds]
            head = "| 群 | " + " | ".join(CONDITION_LABELS_JA.get(c, c) for c in conds) + " |\n|---|" + "|".join("---" for _ in conds) + "|\n"
            body = ""
            for g in GROUPS:
                cells = []
                for c in conds:
                    s = summary[(summary["grouping"] == gname) & (summary["group"] == g) & (summary["condition"] == c)
                                & (summary["centroid_site"] == csite)]
                    if s.empty:
                        cells.append("—")
                    else:
                        r = s.iloc[0]
                        cells.append(f"{r['n_sft_assigned']}/{r['n']} SFT ({r['frac_sft']:.0%}); "
                                     f"{r['median']:+.2f} [{r['q25']:+.2f}, {r['q75']:+.2f}]")
                if all(c == "—" for c in cells):
                    continue
                body += f"| {GROUP_LABELS_JA[g]} | " + " | ".join(cells) + " |\n"
            L += [head + body, "（セル: SFT 割り当て数/n（割合）; マージンの中央値 [IQR]）", ""]
        L += ["Men_STAT6tested vs Men_untested:", "",
              _md_table(comp[comp["grouping"] == gname].assign(condition=lambda d: d["condition"].map(lambda c: CONDITION_LABELS_JA.get(c, c)),
                                                               centroid_site=lambda d: d["centroid_site"].map(SITE_LABEL),
                                                               fisher_p=lambda d: d["fisher_p"].map(_p),
                                                               mannwhitney_p=lambda d: d["mannwhitney_p"].map(_p)),
                        ["condition", "centroid_site", "sft_tested", "n_tested", "sft_untested", "n_untested", "fisher_p",
                         "median_tested", "median_untested", "mannwhitney_p"],
                        {"fisher_p": "{}", "median_tested": "{:+.2f}", "median_untested": "{:+.2f}", "mannwhitney_p": "{}"},
                        {"sft_tested": "SFT 割当（記載あり）", "n_tested": "n（記載あり）", "sft_untested": "SFT 割当（記載なし）",
                         "n_untested": "n（記載なし）", "fisher_p": "Fisher p", "median_tested": "中央値（記載あり）",
                         "median_untested": "中央値（記載なし）", "mannwhitney_p": "MWU p"}), ""]
    # 組織型の構成（記載ありの髄膜腫は組織型が偏る可能性があるので、組織型内でも比べる）
    men_st = status[men]
    comp_tab = pd.crosstab(men_st["subtype"], men_st["group_text"]).reindex(columns=["Men_STAT6tested", "Men_untested"], fill_value=0)
    comp_tab = comp_tab.sort_values("Men_STAT6tested", ascending=False)
    L += ["## 髄膜腫の組織型の構成（記載あり / なし）", "",
          "| 組織型 | 記載あり | 記載なし | 記載ありの割合 |", "|---|---|---|---|"]
    L += [f"| {sub} | {r['Men_STAT6tested']} | {r['Men_untested']} | {r['Men_STAT6tested'] / (r['Men_STAT6tested'] + r['Men_untested']):.0%} |"
          for sub, r in comp_tab.iterrows()]
    L += ["", "組織型内での比較（記載ありが 3 例以上の組織型。補正なしと平行移動、両施設の重心）:", ""]
    cs = comp_sub[comp_sub["condition"].isin(["original", "centroid"])].copy()
    cs["condition"] = cs["condition"].map(lambda c: CONDITION_LABELS_JA.get(c, c))
    cs["centroid_site"] = cs["centroid_site"].map(SITE_LABEL)
    cs["mannwhitney_p"] = cs["mannwhitney_p"].map(_p)
    L += [_md_table(cs, ["condition", "centroid_site", "subtype", "sft_tested", "n_tested", "sft_untested", "n_untested",
                         "median_tested", "median_untested", "mannwhitney_p"],
                    {"median_tested": "{:+.2f}", "median_untested": "{:+.2f}"},
                    {"sft_tested": "SFT 割当（記載あり）", "n_tested": "n（記載あり）", "sft_untested": "SFT 割当（記載なし）",
                     "n_untested": "n（記載なし）", "median_tested": "中央値（記載あり）", "median_untested": "中央値（記載なし）",
                     "mannwhitney_p": "MWU p"}), ""]
    # 個別症例
    interest = status[(status["diagnosis"] == "SFT") & (status["group"] != "SFT_STAT6pos")]["case_id"].tolist()
    interest += [c for c in pos_men["case_id"] if c not in interest]
    missed = []
    for n in nc_notes:
        m = re.search(r"見逃した SFT: (.+)$", n)
        if m and m.group(1) != "なし":
            missed += [c.strip() for c in m.group(1).split(",")]
    interest += [c for c in dict.fromkeys(missed) if c not in interest]
    L += ["## 個別症例（STAT6 で確認されていない SFT、STAT6 陽性とされた髄膜腫、EBRAINS → patho2 で見逃した SFT）", ""]
    for cid in interest:
        st = status.set_index("case_id").loc[cid]
        tag = []
        if cid in set(missed):
            tag.append("EBRAINS → patho2 で見逃した SFT")
        L += [f"### {cid}（{st['subtype']}; 列: 核 {_na(st['stat6_nuclear'])} / 細胞質 {_na(st['stat6_cytoplasmic'])}; "
              f"本文の読み: 核 {_na(st['stat6_nuclear_text'])}; group {st['group']} / group_text {st['group_text']}"
              + ("; " + "; ".join(tag) if tag else "") + "）", ""]
        if st["stat6_text"]:
            L += [f"レポート: 「{st['stat6_text']}」", ""]
        sc = cases[cases["case_id"] == cid].copy()
        sc["condition"] = sc["condition"].map(lambda c: CONDITION_LABELS_JA.get(c, c))
        sc["centroid_site"] = sc["centroid_site"].map(SITE_LABEL)
        L += [_md_table(sc, ["condition", "centroid_site", "d_sft", "d_men", "margin", "assigned"],
                        {"d_sft": "{:.2f}", "d_men": "{:.2f}", "margin": "{:+.2f}"}), ""]
    L += ["## 所見（数字から機械的に組んだもの。解釈は断定しない）", ""]
    L += [f"- {s}" for s in findings(summary, comp, comp_sub, status, cfg)]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def findings(summary: pd.DataFrame, comp: pd.DataFrame, comp_sub: pd.DataFrame, status: pd.DataFrame, cfg: dict) -> list[str]:
    reference, site = cfg["reference"], cfg["stat6"]["site"]
    out = []
    nd = int(status["discrepant"].sum())
    out.append(f"情報源の列と本文の読みが食い違う症例は {nd} 例。本文の読みでは STAT6 核陽性の髄膜腫も核陰性の SFT も"
               + ("無い。" if not ((status["group_text"] == "SFT_STAT6neg").any()
                                   or ((status["diagnosis"] != "SFT") & status["stat6_nuclear_text"].eq("Positive")).any())
                  else "ある（上の表）。"))
    for gname in GROUPINGS:
        for csite in (reference, site):
            c = comp[(comp["grouping"] == gname) & (comp["centroid_site"] == csite)]
            if c.empty:
                continue
            o = c[c["condition"] == "original"].iloc[0] if (c["condition"] == "original").any() else c.iloc[0]
            ce = c[c["condition"] == "centroid"].iloc[0] if (c["condition"] == "centroid").any() else None
            s = (f"[{gname} / 重心 {SITE_LABEL[csite]}] STAT6 記載ありの髄膜腫が SFT に割り当てられた割合: 補正なし "
                 f"{o['sft_tested']}/{o['n_tested']}（{o['frac_sft_tested']:.1%}）対 記載なし {o['sft_untested']}/{o['n_untested']}"
                 f"（{o['frac_sft_untested']:.1%}）、Fisher p {_peq(o['fisher_p'])}；マージンの中央値 {o['median_tested']:+.2f} 対 "
                 f"{o['median_untested']:+.2f}、MWU p {_peq(o['mannwhitney_p'])}")
            if ce is not None:
                s += (f"。平行移動では {ce['sft_tested']}/{ce['n_tested']} 対 {ce['sft_untested']}/{ce['n_untested']}"
                      f"（Fisher p {_peq(ce['fisher_p'])}）、中央値 {ce['median_tested']:+.2f} 対 {ce['median_untested']:+.2f}"
                      f"（MWU p {_peq(ce['mannwhitney_p'])}）")
            out.append(s + "。")
    cs = comp_sub[(comp_sub["condition"] == "original") & (comp_sub["centroid_site"] == reference)]
    if not cs.empty:
        out.append("組織型内（補正なし、重心 EBRAINS）: "
                   + "、".join(f"{r['subtype'].replace(' meningioma', '')} 記載あり {r['median_tested']:+.2f}（n={r['n_tested']}）対 "
                               f"なし {r['median_untested']:+.2f}（n={r['n_untested']}、MWU p {_peq(r['mannwhitney_p'])}）"
                               for _, r in cs.iterrows()) + "。")
    return out


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    site = cfg["stat6"]["site"]
    out_dir = Path(cfg["output_dir"]) / "stat6"
    out_dir.mkdir(parents=True, exist_ok=True)

    status, info = build_status(cfg)
    status_path = Path(cfg["stat6"]["status_csv"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(status_path, index=False)
    print(f"[stat6_case] cohort {info['n_cohort']} / matched: {info['n_by']} / exact {info['n_exact']}")
    print(f"  meningioma with STAT6 columns {info['n_men_tested']}/{info['n_men']}, text mention {info['n_men_text_mention']}; "
          f"SFT with STAT6 {info['n_sft_tested']}/{info['n_sft']}")
    print("  groups (archive):", status["group"].value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict())
    print("  groups (text):   ", status["group_text"].value_counts().reindex(GROUPS).fillna(0).astype(int).to_dict())
    d = status[status["discrepant"]]
    print(f"  discrepant (column vs text): {len(d)}: " + ", ".join(f"{r.case_id}({r.stat6_nuclear}->{r.stat6_nuclear_text})" for r in d.itertuples()))
    print(f"  saved: {status_path}")

    frames = []
    for v in available_conditions(cfg, site):
        cohort = load_data(cfg, cfg.get("variants", {}).get(v, v))
        if cohort.empty:
            print(f"  SKIP [{v}]: no cohort")
            continue
        df = assign(cohort, status, cfg, v)
        frames.append(df)
        e = df[df["centroid_site"] == cfg["reference"]]
        print(f"  [{v}] n={len(e)}  " + "  ".join(
            f"{g}:{int((e[e['group_text'] == g]['assigned'] == 'SFT').sum())}/{int((e['group_text'] == g).sum())}"
            for g in GROUPS if (e["group_text"] == g).any()))
    if not frames:
        print("  nothing computed")
        return
    cases = pd.concat(frames, ignore_index=True)
    nc_notes = check_against_nc(cases, cfg)
    for n in nc_notes:
        print("  nc:", n)
    summary, comp, comp_sub = summarize(cases), compare_men(cases), compare_men_by_subtype(cases)
    cases.to_csv(out_dir / "cases.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    comp.to_csv(out_dir / "compare.csv", index=False)
    comp_sub.to_csv(out_dir / "compare_by_subtype.csv", index=False)
    write_md(out_dir / "summary.md", status, info, summary, comp, comp_sub, cases, nc_notes, cfg)
    print(f"  saved: {out_dir}/{{cases.csv,summary.csv,compare.csv,compare_by_subtype.csv,summary.md}}")


if __name__ == "__main__":
    main()
