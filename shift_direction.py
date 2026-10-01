"""shift_direction.py — 施設間シフトの向き（群ごと）の一致度。

群（全体 / 髄膜腫全体 / SFT / 各サブタイプ）ごとに、施設間シフト
    shift = mean(reference) − mean(other)   （other → reference の向き）
を求め、全体シフトとの cos を比べる。cos が群によらず高ければ、施設差は群に依存しない
ほぼ一様な平行移動であり、centroid（重心）補正の前提が成り立つ。

  対照:  施設内の SFT − 髄膜腫 方向（生物学的な差）と全体シフトの cos
  帰無1: 施設ラベルをシャッフルしたときの cos(髄膜腫全体シフト, SFT シフト)
  帰無2: 例数を揃えた帰無 — 各群と同じ (n_reference, n_other) を髄膜腫全体から無作為に抜いて
         作ったシフトと全体シフトの cos。観測 cos がこの分布の中にあれば、群ごとの cos の低さは
         例数の少なさで説明できる（群固有のずれは無い）。

出力 (output_dir/shift_direction/):
    shift_direction_{variant}.csv        — 群ごとの |shift|・cos・帰無2 の分位点
    bio_direction_{variant}.csv          — 施設内 SFT − 髄膜腫 方向との cos（対照）
    null_shuffle_{variant}.csv           — 帰無1 の要約
    shift_direction_{variant}.png

Usage:
    uv run python shift_direction.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.display import ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data

ALL = "All"
MENI = "Meningioma (all)"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _shift(X: np.ndarray, dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    return X[dst].mean(axis=0) - X[src].mean(axis=0)


# ── compute ───────────────────────────────────────────────────────────────────

def compute(merged: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sd_cfg = cfg.get("shift_direction", {})
    min_n: int = sd_cfg.get("min_n", 5)
    n_perm: int = sd_cfg.get("n_perm", 1000)
    rng = np.random.default_rng(sd_cfg.get("seed", 42))

    reference: str = cfg["reference"]
    other = next(s for s in merged["source"].unique() if s != reference)
    print(f"  shift = mean({reference}) − mean({other})")

    X = np.stack(merged["embedding"].values).astype(np.float64)
    is_ref = (merged["source"] == reference).values
    is_oth = ~is_ref
    is_sft = merged["subtype"].str.lower().str.contains("sft").values

    groups: dict[str, np.ndarray] = {
        ALL: np.ones(len(merged), dtype=bool),
        MENI: ~is_sft,
    }
    for sub in ordered_subtypes(set(merged["subtype"]), cfg):
        m = (merged["subtype"] == sub).values
        if (m & is_ref).sum() >= min_n and (m & is_oth).sum() >= min_n:
            groups[sub] = m
        else:
            print(f"  skip {sub}: n={int((m & is_ref).sum())}/{int((m & is_oth).sum())} < {min_n}")

    d_all = _shift(X, is_ref, is_oth)
    d_meni = _shift(X, is_ref & ~is_sft, is_oth & ~is_sft)
    meni_ref = np.flatnonzero(is_ref & ~is_sft)
    meni_oth = np.flatnonzero(is_oth & ~is_sft)

    rows = []
    for name, m in groups.items():
        n_ref, n_oth = int((m & is_ref).sum()), int((m & is_oth).sum())
        d = _shift(X, m & is_ref, m & is_oth)
        obs = _cos(d, d_all)
        # 帰無2: 髄膜腫全体から同じ例数を抜いたシフト
        null = np.array([
            _cos(X[rng.choice(meni_ref, n_ref, replace=False)].mean(axis=0)
                 - X[rng.choice(meni_oth, n_oth, replace=False)].mean(axis=0), d_all)
            for _ in range(n_perm)
        ]) if name not in (ALL, MENI) else np.full(1, np.nan)
        lo, med, hi = (np.nanquantile(null, [0.025, 0.5, 0.975]) if name not in (ALL, MENI)
                       else (np.nan, np.nan, np.nan))
        rows.append({
            "group": name,
            f"n_{reference}": n_ref,
            f"n_{other}": n_oth,
            "shift_norm": float(np.linalg.norm(d)),
            "cos_vs_all": obs,
            "cos_vs_meningioma": _cos(d, d_meni),
            "null_matched_q025": lo,
            "null_matched_q50": med,
            "null_matched_q975": hi,
            # 片側: 髄膜腫から同数抜いたときに観測以下の cos が出る割合（小さいほど群固有のずれ）
            "p_lower": float((null <= obs).mean()) if name not in (ALL, MENI) else np.nan,
        })
    table = pd.DataFrame(rows)

    # 対照: 施設内の SFT − 髄膜腫 方向
    bio_rows = []
    for site, s in ((reference, is_ref), (other, is_oth)):
        bio = _shift(X, s & is_sft, s & ~is_sft)
        bio_rows.append({"site": site, "cos_shift_vs_sft_minus_meningioma": _cos(d_all, bio),
                         "norm_sft_minus_meningioma": float(np.linalg.norm(bio)),
                         "norm_shift_all": float(np.linalg.norm(d_all))})
    bio_table = pd.DataFrame(bio_rows)

    # 帰無1: 施設ラベルのシャッフル
    src = is_ref.copy()
    null1 = []
    for _ in range(n_perm):
        rng.shuffle(src)
        null1.append(_cos(_shift(X, src & ~is_sft, ~src & ~is_sft), _shift(X, src & is_sft, ~src & is_sft)))
    null1 = np.array(null1)
    obs1 = _cos(d_meni, _shift(X, is_ref & is_sft, is_oth & is_sft))
    null_table = pd.DataFrame([{
        "statistic": "cos(meningioma shift, SFT shift)",
        "observed": obs1,
        "null_mean": float(null1.mean()),
        "null_q975": float(np.quantile(null1, 0.975)),
        "null_max": float(null1.max()),
        "p_upper": float((null1 >= obs1).mean()),
        "n_perm": n_perm,
    }])
    return table, bio_table, null_table


# ── plot ──────────────────────────────────────────────────────────────────────

def plot(table: pd.DataFrame, cfg: dict, reference: str, other: str, out_path: Path) -> None:
    subs = [g for g in table["group"] if g not in (ALL, MENI)]
    cmap = subtype_color_map(subs, cfg)
    colors = ["#555555", "#555555"] + [cmap[g] for g in subs]
    labels = [ALL, MENI] + [shorten(g, cfg) for g in subs]
    n_cols = [c for c in table.columns if c.startswith("n_")]
    labels = [f"{lb}  ({r[n_cols[0]]}/{r[n_cols[1]]})" for lb, (_, r) in zip(labels, table.iterrows())]
    y = np.arange(len(table))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.42 * len(table) + 1.6), sharey=True,
                                   gridspec_kw={"width_ratios": [3, 2], "wspace": 0.08})

    # 左: 全体シフトとの cos と、例数を揃えた帰無の 95% 区間
    has_null = table["null_matched_q025"].notna().values
    ax1.hlines(y[has_null], table["null_matched_q025"][has_null], table["null_matched_q975"][has_null],
               color="#c8c8c8", lw=7, zorder=1, label="matched-n null (95%)")
    ax1.scatter(table["null_matched_q50"][has_null], y[has_null], marker="|", s=90, color="#909090", zorder=2)
    ax1.scatter(table["cos_vs_all"], y, c=colors, s=55, zorder=3, edgecolors="white", linewidths=0.8)
    ax1.set_xlim(0.5, 1.02)
    ax1.set_xlabel("cos(group shift, overall shift)")
    ax1.axhline(1.5, color="#dddddd", lw=0.8)
    ax1.legend(loc="lower left", frameon=False, fontsize=9)

    # 右: シフトの大きさ
    ax2.scatter(table["shift_norm"], y, c=colors, s=55, edgecolors="white", linewidths=0.8, zorder=3)
    ax2.axvline(table.loc[table["group"] == ALL, "shift_norm"].iloc[0], color="#bbbbbb", lw=1, ls="--", zorder=1)
    ax2.set_xlim(0, max(table["shift_norm"]) * 1.15)
    ax2.set_xlabel("|shift|")
    ax2.axhline(1.5, color="#dddddd", lw=0.8)

    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"Site shift direction ({other} → {reference}); n = {reference}/{other}", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    variant = cfg.get("shift_direction", {}).get("variant", "original")
    dir_key = cfg.get("variants", {}).get(variant, variant)
    out_dir = Path(cfg["output_dir"]) / "shift_direction"

    print(f"[shift_direction] variant={variant}")
    merged = load_data(cfg, dir_key)
    if merged.empty:
        print("  SKIP: no data")
        return
    print(f"  {len(merged)} cases, sources: {sorted(merged['source'].unique())}")

    table, bio_table, null_table = compute(merged, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("shift_direction", table), ("bio_direction", bio_table), ("null_shuffle", null_table)):
        path = out_dir / f"{name}_{variant}.csv"
        df.to_csv(path, index=False)
        print(f"  saved: {path}")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.round(3).to_string(index=False))
        print(bio_table.round(3).to_string(index=False))
        print(null_table.round(3).to_string(index=False))

    reference = cfg["reference"]
    other = next(s for s in merged["source"].unique() if s != reference)
    plot(table, cfg, reference, other, out_dir / f"shift_direction_{variant}.png")


if __name__ == "__main__":
    main()
