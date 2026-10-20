"""施設差の幾何: 平均プーリング vs TITAN の表（参考用。figs/__init__.py に登録していない）。

2026-10-20 に Supplementary から外した（ken 決定）。表の数値は Results 本文に書く。figs.all では作らないので、
見たいときはこのスクリプトを直接実行する（出力名は番号を持たないので tables/_meanpool_geometry.{csv,md}）。

Fig 4（TITAN）の数値を、同じパッチ埋め込みを学習なしの平均で集約したスライド埋め込み（meanpool.py）と並べる。
集約器によらず「施設差 = 群に依存しない平行移動、施設内の形は保たれる」が成り立つかを見る。

行: 平均プーリング / TITAN
列: 全体シフトの大きさ、群ごとのシフトと全体シフトの cos（min–max）、群ごとの |shift|（min–max）、
    施設内 SFT − 髄膜腫 方向との cos（両施設）、組織型間距離の施設間相関、原点を通る傾き

入力: output_dir/meanpool/{shift_direction,bio_direction,subtype_distance_scaling}_original.csv（meanpool.py）、
      output_dir/shift_direction/{同}_{variant}.csv（shift_direction.py、Fig 4 と同じファイル）
出力: tables/_meanpool_geometry.{csv,md}（番号体系の外）

Usage:
    uv run python -m figs.meanpool_geometry
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import shift_direction
from figs.common import SITE_DISPLAY, config, out_root, table_dir

NAME = "meanpool_geometry"
AGGREGATORS = [("平均プーリング", "mean"), ("TITAN", "titan")]


def _read(root: Path, cfg: dict, agg: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if agg == "mean":
        d, variant = root / "meanpool", "original"
    else:
        d, variant = root / "shift_direction", cfg.get("shift_direction", {}).get("variant", "original")
    return (pd.read_csv(d / f"shift_direction_{variant}.csv"), pd.read_csv(d / f"bio_direction_{variant}.csv"),
            pd.read_csv(d / f"subtype_distance_scaling_{variant}.csv"))


def main() -> None:
    print(f"[unregistered] {NAME}")
    cfg = config()
    root = out_root(cfg)
    ref = cfg["reference"]
    other = next(k for k in cfg["embedding"] if k != ref)
    ref_d, oth_d = SITE_DISPLAY.get(ref, ref), SITE_DISPLAY.get(other, other)
    min_n = cfg.get("shift_direction", {}).get("min_n", 5)

    records, n_groups, n_pairs = [], {}, {}
    for name, agg in AGGREGATORS:
        table, bio, scaling = _read(root, cfg, agg)
        subs = table[~table["group"].isin([shift_direction.ALL, shift_direction.MENI])]
        n_groups[name], n_pairs[name] = len(subs), int(scaling["n_pairs"].iloc[0])
        cos_bio = bio.set_index("site")["cos_shift_vs_sft_minus_meningioma"]
        s = scaling.iloc[0]
        records.append({
            "集約": name,
            "全体シフト |Δ|": f"{table.loc[table['group'] == shift_direction.ALL, 'shift_norm'].iloc[0]:.2f}",
            "群ごとの cos(Δ_g, Δ)": f"{subs['cos_vs_all'].min():.2f}–{subs['cos_vs_all'].max():.2f}",
            "群ごとの |Δ_g|": f"{subs['shift_norm'].min():.2f}–{subs['shift_norm'].max():.2f}",
            f"cos(Δ, SFT−髄膜腫) {ref_d}": f"{cos_bio[ref]:.2f}",
            f"cos(Δ, SFT−髄膜腫) {oth_d}": f"{cos_bio[other]:.2f}",
            "組織型間距離の施設間相関 r": f"{s['pearson_r']:.2f}",
            f"傾き（{oth_d}/{ref_d}）": f"{s['slope_origin']:.2f}",
        })
    table = pd.DataFrame(records)
    table_dir(cfg).mkdir(parents=True, exist_ok=True)
    csv_path = table_dir(cfg) / f"_{NAME}.csv"
    table.to_csv(csv_path, index=False)

    head = list(table.columns)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in head) + " |" for _, r in table.iterrows()]
    n_g = " / ".join(f"{name} {n_groups[name]} 群・{n_pairs[name]} 組" for name, _ in AGGREGATORS)
    lines += ["", f": 施設差の幾何: 平均プーリング vs TITAN。同じパッチ埋め込み（CONCH v1.5）を、学習なしの単純平均（平均プーリング）と"
              f"TITAN の集約器でそれぞれスライド埋め込み（768 次元）にし、補正なしの状態で Fig 4 と同じ量を求めた。"
              f"Δ = 全体の施設間シフト mean({ref_d}) − mean({oth_d})、Δ_g = 群 g（両施設で {min_n} 例以上の組織型と SFT。"
              f"{n_g}）ごとの同じシフト。cos(Δ_g, Δ) と |Δ_g| は群の範囲（min–max）。cos(Δ, SFT−髄膜腫) は各施設内の"
              f"SFT − 髄膜腫 の平均の差との cos（Fig 4c）。相関と傾きは、両施設で {min_n} 例以上の組織型の平均どうしの距離を"
              f"施設間で比べたもの（Fig 4d。傾きは原点を通る回帰、{oth_d} / {ref_d}）。 {{#tbl:meanpool_geometry}}"]
    path = table_dir(cfg) / f"_{NAME}.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {csv_path} / {path.name}")


if __name__ == "__main__":
    main()
