"""cohort.py — 院内のグリオーマ・リンパ腫・転移の症例の定義（症例の表）と、転移テストの埋め込みキャッシュを manifest に揃える小道具。

症例の表（config の patho2_gliomas.cohort、例 data/patho2_gliomas/cohort.csv。transfer_manifest.py --build-cohort が 1 回だけ作る）:
    列 case, folder_group（Patho2Gliomas のフォルダ A / G / O / L / M）, label（フォルダの略号 ast / glm / odg / lym / met）,
    group（解析に使う群。今はフォルダの群のまま）, status（include / exclude）, reason, flag（再確認で疑問が付いた理由。空 = なし）
    patho2_gliomas.cohort_enabled が true で表があるときだけ効く。無効・表が無いときは代表 HE（representatives.csv）の全症例
    （260 例）を今までどおりフォルダの群で使う。patho2_gliomas.exclude_flagged が true なら flag のある症例も外す（感度分析）。

キャッシュの揃え方（align_rows）: 転移テストの埋め込みキャッシュ（out/transfer/embeddings_*.npy と並びの CSV）は manifest の
    並びで作ってある。症例を減らした manifest でも、キャッシュが manifest の (site, slide_id) を全部含んでいればその行を引いて
    使う（h5 を読み直さない・キャッシュは書き換えない）。並びが同じなら恒等。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COHORT_COLS = ["case", "folder_group", "label", "group", "status", "reason", "flag"]


def cohort_table(cfg: dict, verbose: bool = True) -> pd.DataFrame | None:
    """有効な症例の表（case を index）。無効・表が無ければ None。"""
    pg = cfg.get("patho2_gliomas", {}) or {}
    path = pg.get("cohort")
    if not pg.get("cohort_enabled", False) or not path:
        return None
    if not Path(path).exists():
        if verbose:
            print(f"  WARN 症例の表 {path} が無いので、代表 HE の全症例をフォルダの群で使う（transfer_manifest.py --build-cohort で作る）")
        return None
    tbl = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(tbl.columns[: len(COHORT_COLS)]) != COHORT_COLS or tbl["case"].duplicated().any():
        raise SystemExit(f"症例の表 {path} の列が {COHORT_COLS} でないか、case が重複している")
    if not tbl["status"].isin(["include", "exclude"]).all():
        raise SystemExit(f"症例の表 {path} の status は include / exclude だけ")
    tbl = tbl.set_index("case")
    if pg.get("exclude_flagged", False):   # 感度分析: flag のある症例も外す
        tbl.loc[tbl["flag"] != "", "status"] = "exclude"
    return tbl


def apply_cohort(df: pd.DataFrame, cfg: dict, case_col: str, group_cols: list[str], site_col: str | None = None,
                 site: str = "patho2") -> pd.DataFrame:
    """院内の行（site_col が無ければ全行）を症例の表の include に絞り、group_cols を表の group に置き換える（行の並びは保つ）。
    表が無効なら df をそのまま返す。表に無い院内の症例（過去の論文に無い新しい症例）は外す（件数を表示）。"""
    tbl = cohort_table(cfg)
    if tbl is None:
        return df
    m = (df[site_col] == site).to_numpy() if site_col else np.ones(len(df), bool)
    cases = df.loc[m, case_col].astype(str)
    missing = sorted(set(cases) - set(tbl.index))
    if missing:
        print(f"  症例の表に無い院内の症例 {len(missing)} 例は外す: {missing[:10]}")
    keep = ~m.copy()
    keep[m] = cases.map(tbl["status"]).fillna("exclude").to_numpy() == "include"
    out = df[keep].copy()
    mo = (out[site_col] == site).to_numpy() if site_col else np.ones(len(out), bool)
    new = tbl.loc[out.loc[mo, case_col].astype(str), "group"].to_numpy()
    for c in group_cols:
        out.loc[mo, c] = new
    return out


def align_rows(ids: pd.DataFrame, man: pd.DataFrame, keys: tuple[str, ...] = ("site", "slide_id")) -> np.ndarray | None:
    """man の各行が ids（キャッシュの並び）の何行目か。並びが同じなら arange、ids が man を全部含まなければ None。"""
    k = list(keys)
    a, b = ids[k].astype(str).reset_index(drop=True), man[k].astype(str).reset_index(drop=True)
    if a.equals(b):
        return np.arange(len(b))
    if a.duplicated().any():
        return None
    pos = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_frame(a))
    idx = pd.MultiIndex.from_frame(b)
    if not idx.isin(pos.index).all():
        return None
    return pos.loc[idx].to_numpy()
