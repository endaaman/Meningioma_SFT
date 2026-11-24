"""transfer_rare_target.py — 希少な腫瘍への転移: 髄膜腫だけで推定した施設のずれで SFT を補正できるか（SFT を推定に一切使わない）。

transfer_correction.py の当て先 paper（髄膜腫 + SFT）では、推定元 meningioma / SFT を「当て先の真部分集合（一部 in-sample）」として
飛ばしている（source_families）。own（髄膜腫 + SFT コホート全体）は SFT 自身（EBRAINS 32 / 院内 22 枚）を 4〜6% 含む。
ここでは同じ当て先 paper に、推定元 meningioma（SFT を含まない）と SFT（SFT だけ。同じデータで推定して当てるので参考）を当て、
SFT の群の施設の見分けやすさと、樹形図で SFT の 2 施設の平均が隣り合うかを見る。transfer_correction.py の出力には書かない
（既存の出力の行・図を変えないため、別ファイルに出す）。

計算は transfer_correction.py の関数をそのまま呼ぶ（同じ手順・同じ指標定義）:
    埋め込み  load_embeddings（out/transfer/embeddings_{encoder}.npy 等のキャッシュ。読み直さない）
    当てはめ  fit_source（推定元の両施設のスライド）、適用 apply_method（当て先の patho2 だけを補正）
    指標      site_mixing（群ごとの group_gap_sd / gap / rms / knn_mix / knn_frac_ebrains。correction_{encoder}_groups.csv と同じ）、
              pairs（pairs_adjacent。樹形図の元は保存しない）と dendrogram.adjacent_pairs（どの組が隣り合ったか）
    ずれ      shift_transfer.shift_stats（Δ = μ_E − μ_P、‖Δ‖、ノイズ分散の和 tr V、ノイズを差し引いた ‖Δ‖）・_cos・corrected_cos
法: centroid / affine_free / combat（combat_recentred は中心を当て先の施設平均で取り直すので、推定元の違いが中心に出ない。入れない）。
推定元: none / meningioma（新規）/ own（= 髄膜腫 + SFT、既存）/ glioma（既存）/ SFT（新規・参考。平行移動と相似変換は作りから
    SFT の中心の距離が 0、ComBat は経験ベイズの縮小があるので 0 にならない）。
乱数は使わない（fit_source・apply_method・site_mixing・pairs は決定的）。

参考の 2 量（既存の指標ではない。群の枚数が少ないと group_gap_sd が 0 から離れる分の目安）:
    gap_noise_sd      √tr V / RMS_g（tr V = Σ_d var_E,d / n_E + var_P',d / n_P、ddof = 1、x' = 補正後の patho2。shift_transfer.shift_stats）。
                      2 施設の真の中心が一致していても、群の平均の標本誤差だけで group_gap_sd がおよそこの値になる
    gap_corrected_sd  √max(gap² − tr V, 0) / RMS_g。推定元がその群の科を含まない（none、または SFT の群に対する meningioma /
                      glioma）ときだけ。推定に群のスライドを使うと標本誤差が gap と独立でないので NaN。推定元側の誤差は入れていない

確認（合わなければ止まる）: none / own / glioma の行（13 群 × GROUP_METRICS と群の枚数・knn_k）が correction_{encoder}_groups.csv と
    同じ値か、群ごとの隣接の和が correction_{encoder}.csv の pairs_adjacent と同じか。ずれの長さと 髄膜腫だけ − SFT の cos が
    shift_transfer.py の groups.csv（Paper cohort / Meningioma (all) / SFT の shift_norm、SFT 行の cos_vs_meni・cos_corrected_vs_meni）と同じか。

出力 (output_dir/transfer/):
    rare_target_{encoder}.csv   縦長: encoder, method, source, target (= paper), group, n_ebrains, n_patho2, knn_k, metric, value,
                                n_src_ebrains, n_src_patho2。metric = GROUP_METRICS（correction_{encoder}_groups.csv と同じ）＋
                                pair_adjacent（その群の 2 施設の平均が樹形図で隣り合えば 1）＋ gap_noise_sd・gap_corrected_sd（参考）。
                                全 13 群。source = none は各 method に同じ行
    rare_target_shifts.csv      エンコーダ × 推定元: 枚数、shift_norm、shift_norm_corrected、noise_trace、scale_affine（相似変換の s）、
                                k_combat_median（ComBat の次元ごとの倍率 1/√δ* の中央値）
    rare_target_cos.csv         エンコーダ × 推定元の組: cos（推定元が素な組は corrected_cos も）
    rare_target_summary.md      表（TITAN の平行移動が先頭）と確認

Usage:
    uv run python transfer_rare_target.py [--encoders titan prism meanpool meanpool_virchow]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dendrogram import adjacent_pairs, group_means, raw_distance, ward_linkage
from shift_transfer import _cos, corrected_cos, shift_stats
from transfer_correction import (GROUP_METRICS, TARGET_FAMILIES, _fmt, _md, apply_method, fit_source, load_embeddings, pairs,
                                 site_mixing, source_families, target_labels)
from utils.display import CONDITION_LABELS_JA, ENCODER_LABELS
from utils.loader import load_config

ENCODERS = ["titan", "prism", "meanpool", "meanpool_virchow"]
METHODS = ["centroid", "affine_free", "combat"]
TARGET = "paper"
RARE = "SFT"
SOURCES = {"meningioma": frozenset({"meningioma"}), "own": source_families(TARGET, "own"),
           "glioma": source_families(TARGET, "glioma"), "SFT": frozenset({"SFT"})}
ORDER = ["none", "meningioma", "own", "glioma", "SFT"]
EXISTING = ["none", "own", "glioma"]            # correction_{encoder}_groups.csv にある推定元（突き合わせる）
EXTRA_METRICS = ["pair_adjacent", "gap_noise_sd", "gap_corrected_sd"]
COS_PAIRS = [("meningioma", "own"), ("meningioma", "SFT"), ("own", "SFT"), ("glioma", "SFT"), ("meningioma", "glioma")]
SOURCE_LABELS = {"none": "補正なし", "meningioma": "髄膜腫だけ（新規）", "own": "髄膜腫 + SFT（own）",
                 "glioma": "グリオーマ", "SFT": "SFT だけ（参考）"}
SHIFT_TRANSFER_GROUP = {"own": "Paper cohort", "meningioma": "Meningioma (all)", "SFT": "SFT"}   # shift_transfer の groups.csv の group 列


def adjacent_groups(Y: np.ndarray, st: np.ndarray, lab: np.ndarray, cfg: dict) -> list[str]:
    """樹形図の葉の並びで 2 施設の平均が隣り合った群（transfer_correction.pairs と同じ group_means / raw_distance / ward_linkage）。"""
    g = group_means(pd.DataFrame({"source": st, "subtype": lab, "embedding": list(Y)}), cfg)
    subs = g["subtype"].to_numpy()
    return [str(subs[a]) for a, _ in adjacent_pairs(ward_linkage(raw_distance(g)), g)]


def run_encoder(cfg: dict, encoder: str, out_dir: Path) -> tuple[pd.DataFrame, list[dict], list[dict], dict]:
    print(f"[transfer_rare_target / {encoder}]")
    ref = cfg["reference"]
    man, X = load_embeddings(cfg, encoder, out_dir, reload=False)
    fam, site = man["family"].to_numpy(), man["site"].to_numpy()
    tm = np.isin(fam, list(TARGET_FAMILIES[TARGET]))
    Xt, st, lab = X[tm], site[tm], target_labels(man[tm], TARGET)
    e = st == ref
    group_family = dict(zip(lab, fam[tm]))

    fits, shift_rows, deltas = {}, [], {}
    for s, sf in SOURCES.items():
        sm = np.isin(fam, list(sf))
        fits[s] = fit_source(X[sm], site[sm], ref)
        delta, norm, tr_v, norm_corr = shift_stats({"ebrains": X[sm & (site == ref)], "patho2": X[sm & (site != ref)]})
        assert np.array_equal(delta, fits[s]["mu_e"] - fits[s]["mu_p"])
        deltas[s] = (delta, norm_corr)
        shift_rows.append({"encoder": encoder, "source": s, "n_src_ebrains": fits[s]["n_e"], "n_src_patho2": fits[s]["n_p"],
                           "shift_norm": norm, "shift_norm_corrected": norm_corr, "noise_trace": tr_v,
                           "scale_affine": fits[s]["s"], "k_combat_median": float(np.median(fits[s]["k"]))})
    cos_rows = []
    for a, b in COS_PAIRS:
        disjoint = not (SOURCES[a] & SOURCES[b])
        cos_rows.append({"encoder": encoder, "a": a, "b": b, "cos": _cos(deltas[a][0], deltas[b][0]),
                         "cos_corrected": corrected_cos(deltas[a][0], deltas[a][1], deltas[b][0], deltas[b][1]) if disjoint else np.nan,
                         "disjoint": disjoint})

    rows: list[dict] = []
    checks: dict = {}

    def record(method: str, source: str, Y: np.ndarray, n_src: tuple[int, int]) -> None:
        _, per = site_mixing(Y, st, lab, ref, cfg)
        adj = adjacent_groups(Y, st, lab, cfg)
        n_adj = pairs(Y, st, lab, cfg)["pairs_adjacent"]
        if len(adj) != n_adj:
            raise SystemExit(f"[{encoder} / {method} / {source}] adjacent_pairs（{len(adj)}）と pairs_adjacent（{n_adj}）が合わない")
        for r in per:
            g = r["group"]
            with warnings.catch_warnings():   # 片方の施設が 1 枚の群は ddof = 1 の分散が NaN（gap_noise_sd も NaN）
                warnings.simplefilter("ignore", RuntimeWarning)
                _, _, tr_v, _ = shift_stats({"ebrains": Y[(lab == g) & e], "patho2": Y[(lab == g) & ~e]})
            indep = source == "none" or group_family[g] not in SOURCES[source]
            vals = {**{m: r[m] for m in GROUP_METRICS}, "pair_adjacent": int(g in adj),
                    "gap_noise_sd": float(np.sqrt(tr_v) / r["rms"]),
                    "gap_corrected_sd": float(np.sqrt(max(r["gap"] ** 2 - tr_v, 0.0)) / r["rms"]) if indep else np.nan}
            rows.extend({"encoder": encoder, "method": method, "source": source, "target": TARGET, "group": g,
                         "n_ebrains": r["n_ebrains"], "n_patho2": r["n_patho2"], "knn_k": r["knn_k"], "metric": m, "value": v,
                         "n_src_ebrains": n_src[0], "n_src_patho2": n_src[1]} for m, v in vals.items())

    for method in METHODS:   # none は method によらず同じ（transfer_correction.py と同じく各 method に同じ行を置く）
        record(method, "none", Xt, (0, 0))
    for source in ORDER[1:]:
        for method in METHODS:
            Y, _ = apply_method(method, fits[source], Xt, st, ref, checks)
            record(method, source, Y, (fits[source]["n_e"], fits[source]["n_p"]))
    res = pd.DataFrame(rows)
    sft = res[(res["group"] == RARE) & (res["method"] == "centroid")].set_index(["source", "metric"])["value"]
    print("  " + "  ".join(f"{s}: gap/SD {sft[(s, 'group_gap_sd')]:.3f} kNN {sft[(s, 'knn_mix')]:.3f} adj {int(sft[(s, 'pair_adjacent')])}"
                           for s in ORDER) + "  (SFT, centroid)")
    return res, shift_rows, cos_rows, {"checks": checks, **verify(res, shift_rows, cos_rows, cfg, encoder, out_dir)}


def verify(res: pd.DataFrame, shift_rows: list[dict], cos_rows: list[dict], cfg: dict, encoder: str, out_dir: Path) -> dict:
    """none / own / glioma の行を transfer_correction.py の出力と、ずれを shift_transfer.py の出力と突き合わせる。"""
    old = pd.read_csv(out_dir / f"correction_{encoder}_groups.csv", float_precision="round_trip")
    old = old[(old["target"] == TARGET) & old["source"].isin(EXISTING) & old["method"].isin(METHODS)]
    new = res[res["source"].isin(EXISTING) & res["metric"].isin(GROUP_METRICS)]
    key = ["method", "source", "group", "metric"]
    mg = old.merge(new, on=key, how="outer", suffixes=("_old", "_new"), indicator=True)
    if not (mg["_merge"] == "both").all() or len(mg) != len(old) or len(mg) != len(new):
        raise SystemExit(f"[{encoder}] correction_{encoder}_groups.csv と行が対応しない（{len(old)} / {len(new)} / {len(mg)}）")
    for c in ("n_ebrains", "n_patho2", "knn_k"):
        if not (mg[f"{c}_old"] == mg[f"{c}_new"]).all():
            raise SystemExit(f"[{encoder}] {c} が correction_{encoder}_groups.csv と合わない")
    a, b = mg["value_old"].to_numpy(), mg["value_new"].to_numpy()
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise SystemExit(f"[{encoder}] NaN の位置が correction_{encoder}_groups.csv と合わない")
    d_groups = float(np.nanmax(np.abs(a - b)))
    n_exact = int(((a == b) | (np.isnan(a) & np.isnan(b))).sum())

    oc = pd.read_csv(out_dir / f"correction_{encoder}.csv")
    oc = oc[(oc["target"] == TARGET) & (oc["metric"] == "pairs_adjacent") & oc["source"].isin(EXISTING) & oc["method"].isin(METHODS)]
    oc = oc.set_index(["method", "source"])["value"]
    got = res[(res["metric"] == "pair_adjacent") & res["source"].isin(EXISTING)].groupby(["method", "source"])["value"].sum()
    pairs_ok = bool(len(oc) == len(got) == len(METHODS) * len(EXISTING) and (oc.reindex(got.index) == got).all())
    if not pairs_ok or d_groups > 1e-12:
        raise SystemExit(f"[{encoder}] 既存の出力と合わない: groups max |差| {d_groups:.2e}、pairs_adjacent {pairs_ok}")

    st_csv = out_dir.parent / "shift_transfer" / ("" if encoder == "titan" else encoder) / "groups.csv"
    st_diff = {}
    if st_csv.exists():
        g = pd.read_csv(st_csv).set_index("group")
        sh = {r["source"]: r for r in shift_rows}
        for s, gname in SHIFT_TRANSFER_GROUP.items():
            st_diff[f"shift_norm {s}"] = abs(float(g.loc[gname, "shift_norm"]) - sh[s]["shift_norm"])
            st_diff[f"noise_trace {s}"] = abs(float(g.loc[gname, "noise_trace"]) - sh[s]["noise_trace"])
        c = next(r for r in cos_rows if (r["a"], r["b"]) == ("meningioma", "SFT"))
        st_diff["cos meningioma–SFT"] = abs(float(g.loc[RARE, "cos_vs_meni"]) - c["cos"])
        st_diff["cos_corrected meningioma–SFT"] = abs(float(g.loc[RARE, "cos_corrected_vs_meni"]) - c["cos_corrected"])
        worst = max(st_diff.values())
        if worst > 1e-4:
            raise SystemExit(f"[{encoder}] shift_transfer の groups.csv と合わない: {st_diff}")
    print(f"  verify: correction_{encoder}_groups.csv {n_exact} / {len(mg)} values identical (max |diff| {d_groups:.1e}); "
          f"pairs_adjacent {'OK' if pairs_ok else 'NG'}; shift_transfer max |diff| {max(st_diff.values()) if st_diff else float('nan'):.1e}")
    return {"n_values": len(mg), "n_exact": n_exact, "d_groups": d_groups, "pairs_ok": pairs_ok,
            "shift_transfer": st_diff, "shift_transfer_csv": str(st_csv.relative_to(out_dir.parent)) if st_csv.exists() else None}


# ── summary ───────────────────────────────────────────────────────────────────

def _z(v: float) -> float:
    """作りから 0 の値（SFT だけで推定した平行移動・相似変換の gap、丸め誤差 1e-15 程度）を 0 と書く。"""
    return 0.0 if abs(v) < 1e-12 else v


def write_summary(path: Path, res: pd.DataFrame, shifts: pd.DataFrame, cos: pd.DataFrame, info: dict, groups: list[str]) -> None:
    n = res[(res["group"] == RARE)][["n_ebrains", "n_patho2", "knn_k"]].iloc[0]
    n_e, n_p, k = int(n["n_ebrains"]), int(n["n_patho2"]), int(n["knn_k"])
    L = ["# 希少な腫瘍への転移: 髄膜腫だけで推定した施設のずれで SFT を補正できるか", "",
         "`transfer_rare_target.py` が作る。定義はスクリプトの docstring。当て先 = paper（髄膜腫 + SFT、"
         f"{len(groups)} 群: {', '.join(groups)}）の院内だけを補正し、SFT の群（EBRAINS {n_e} / 院内 {n_p} 枚）を見る。"
         "指標は transfer_correction.py と同じ関数（site_mixing・pairs）。",
         f"推定元: none = 補正なし、meningioma = 髄膜腫だけ（SFT を推定に使わない。新規）、own = 髄膜腫 + SFT コホート全体（既存）、"
         "glioma = グリオーマ（既存）、SFT = SFT だけ（同じデータで推定して当てるので平行移動・相似変換は作りから gap = 0。参考）。", "",
         f"- group_gap_sd = ‖μ_E,SFT − μ_P',SFT‖ / RMS_SFT。knn_mix = 院内 SFT の最近傍 {k} 枚（SFT の両施設 {n_e + n_p - 1} 枚の中、"
         f"自分を除く）に占める EBRAINS の割合 / 期待値 {n_e}/{n_e + n_p - 1} = {n_e / (n_e + n_p - 1):.3f}（1 = 見分けがつかない）。gap = 中心の距離（生）",
         "- noise = gap_noise_sd（参考）: 真の中心が一致していても SFT の平均の標本誤差だけで group_gap_sd がおよそこの値になる。"
         "corr = gap_corrected_sd（参考）: ノイズを差し引いた中心の距離 / RMS（推定に SFT を使わない推定元だけ）",
         "- SFT の対 = 当て先 paper の樹形図（施設 × 群の平均、Ward）で SFT の 2 施設の平均が隣り合ったか。対の数 = 隣り合った組の数"
         f"（/{len(groups)}、pairs_adjacent）", ""]

    L += ["## SFT の群（4 エンコーダ × 3 法）", ""]
    tb = []
    for enc in [e for e in ENCODERS if e in set(res["encoder"])]:
        for method in METHODS:
            d = res[(res["encoder"] == enc) & (res["method"] == method)]
            for s in ORDER:
                v = d[(d["source"] == s) & (d["group"] == RARE)].set_index("metric")["value"]
                nsrc = d[d["source"] == s][["n_src_ebrains", "n_src_patho2"]].iloc[0]
                tb.append({"エンコーダ": ENCODER_LABELS.get(enc, enc), "法": CONDITION_LABELS_JA.get(method, method),
                           "推定元": SOURCE_LABELS[s], "推定の枚数 E/P": "—" if s == "none" else f"{int(nsrc.iloc[0])}/{int(nsrc.iloc[1])}",
                           "group_gap_sd": _fmt(_z(v["group_gap_sd"])), "knn_mix": _fmt(v["knn_mix"]), "gap": _fmt(_z(v["gap"])),
                           "noise": _fmt(v["gap_noise_sd"]), "corr": _fmt(v["gap_corrected_sd"]),
                           "SFT の対": "○" if v["pair_adjacent"] == 1 else "×",
                           "対の数": str(int(d[(d["source"] == s) & (d["metric"] == "pair_adjacent")]["value"].sum()))})
    L += [_md(pd.DataFrame(tb), index=False), ""]

    L += ["## 施設のずれ Δ の長さと向き", "",
          "‖Δ‖ = ‖μ_E − μ_P‖（推定元のスライド）、補正 = ノイズを差し引いた長さ（shift_transfer.shift_stats）。s = 相似変換の倍率、"
          "k = ComBat の次元ごとの倍率の中央値。", ""]
    sh = shifts.copy()
    sh["エンコーダ"] = sh["encoder"].map(lambda x: ENCODER_LABELS.get(x, x))
    sh["推定元"] = sh["source"].map(SOURCE_LABELS)
    sh["枚数 E/P"] = sh["n_src_ebrains"].astype(str) + "/" + sh["n_src_patho2"].astype(str)
    L += [_md(sh[["エンコーダ", "推定元", "枚数 E/P", "shift_norm", "shift_norm_corrected", "noise_trace", "scale_affine",
                  "k_combat_median"]].rename(columns={"shift_norm": "‖Δ‖", "shift_norm_corrected": "‖Δ‖ 補正",
                                                      "noise_trace": "tr V", "scale_affine": "s", "k_combat_median": "k 中央値"}),
              index=False), ""]
    L += ["cos(Δ_a, Δ_b)。「補正」= ノイズを差し引いた長さで割った cos（推定元が素な組だけ。own は髄膜腫も SFT も含むので出さない）。", ""]
    ct = cos.copy()
    ct["組"] = ct["a"].map(SOURCE_LABELS) + " – " + ct["b"].map(SOURCE_LABELS)
    ct["v"] = [f"{_fmt(c)}" + (f"（補正 {_fmt(cc)}）" if d else "") for c, cc, d in zip(ct["cos"], ct["cos_corrected"], ct["disjoint"])]
    pv = ct.pivot_table(index="組", columns="encoder", values="v", aggfunc="first", sort=False)
    pv = pv[[e for e in ENCODERS if e in pv.columns]].rename(columns=ENCODER_LABELS)
    pv.index.name = "組"
    L += [_md(pv, index=True), ""]

    L += ["## 確認", "",
          "- none / own / glioma の行（paper の全群 × group_gap_sd / gap / rms / knn_mix / knn_frac_ebrains、群の枚数・knn_k）を "
          "`correction_{encoder}_groups.csv` と、群ごとの隣接の和を `correction_{encoder}.csv` の pairs_adjacent と突き合わせた。",
          "- ずれの長さ・tr V（own / meningioma / SFT）と 髄膜腫だけ – SFT の cos（生・補正）を shift_transfer.py の groups.csv と突き合わせた。", ""]
    ck = []
    for enc, v in info.items():
        st = v["shift_transfer"]
        ck.append({"エンコーダ": ENCODER_LABELS.get(enc, enc), "groups.csv 一致": f"{v['n_exact']} / {v['n_values']}",
                   "最大の差": f"{v['d_groups']:.1e}", "pairs_adjacent": "OK" if v["pairs_ok"] else "**NG**",
                   "shift_transfer 最大の差": f"{max(st.values()):.1e}" if st else "（無い）",
                   "ComBat FromTraining vs 明示式": f"{v['checks'].get('ft_vs_explicit', float('nan')):.1e}",
                   "reference 不変": f"{v['checks'].get('ref_unchanged', float('nan')):.1e}",
                   "使った ComBat": "明示式" if v["checks"].get("use_explicit") else "FromTraining"})
    L += [_md(pd.DataFrame(ck), index=False), ""]
    path.write_text("\n".join(L))
    print(f"  saved: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+", choices=ENCODERS, default=ENCODERS)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "transfer"
    all_res, all_shifts, all_cos, info = [], [], [], {}
    for enc in args.encoders:
        res, shifts, cos, info[enc] = run_encoder(cfg, enc, out_dir)
        csv = out_dir / f"rare_target_{enc}.csv"
        res.to_csv(csv, index=False)
        print(f"  saved: {csv}")
        all_res.append(res)
        all_shifts += shifts
        all_cos += cos
    res = pd.concat(all_res, ignore_index=True)
    shifts, cos = pd.DataFrame(all_shifts), pd.DataFrame(all_cos)
    for name, df in (("rare_target_shifts.csv", shifts), ("rare_target_cos.csv", cos)):
        df.to_csv(out_dir / name, index=False)
        print(f"  saved: {out_dir / name}")
    groups = list(pd.unique(res[(res["encoder"] == args.encoders[0]) & (res["metric"] == "gap")]["group"]))
    write_summary(out_dir / "rare_target_summary.md", res, shifts, cos, info, groups)


if __name__ == "__main__":
    main()
