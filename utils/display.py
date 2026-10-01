"""display.py — 色・順序・略称の表示ヘルパー共通処理。"""
from __future__ import annotations

import matplotlib.pyplot as plt

# 補正条件の色（全図で共通。施設色の青・赤と被らない Tol bright の灰・紫・水色）
CONDITION_COLORS = {"original": "#BBBBBB", "centroid": "#AA3377", "gan": "#66CCEE",
                    "combat": "#CCBB44", "affine_free": "#228833", "affine_oracle": "#88BB88"}
# 補正条件の集合（図表ごとにどちらを使うかを各 figs/*.py で明示する）。
# 並びは全図表で共通: 補正なし → GAN → ComBat → 平行移動（centroid。提案の平行移動を最後に）。
#   FULL: 4 条件。Fig 3・Fig 5・Table 2・Fig 6・Fig S1（本番図表はすべてこれ）
#   CORE: ComBat を除く 3 条件（harmonization.draw_metric の既定など、本番図表では未使用）
CONDITIONS_FULL = ["original", "gan", "combat", "centroid"]
CONDITIONS_CORE = [v for v in CONDITIONS_FULL if v != "combat"]
CONDITION_ORDER = CONDITIONS_FULL  # 並び順の基準（order_conditions が使う）
# Supplementary の表・Fig S3 用: FULL に相似変換を足したもの（本文の図には入れない）
CONDITIONS_SUPP = CONDITIONS_FULL + ["affine_free"]
CONDITIONS_SUPP_REF = CONDITIONS_SUPP + ["affine_oracle"]   # 表のみ。affine_oracle は組織型ラベルを使った参考値
# 補正条件の表示名（図表・表の表示はすべてここから引く。コード内部のキーは変えない）。
#   centroid      = 平行移動（各施設の平均を揃える位置のみの補正）
#   affine_free   = 相似変換（一様なスケール＋平行移動、倍率はラベル不要の推定）
#   affine_oracle = 相似変換（倍率を組織型ラベルから推定した参考値）
#   combat        = ComBat（次元ごとの位置・尺度の補正）
CONDITION_LABELS = {"original": "No correction", "gan": "GAN", "combat": "ComBat", "centroid": "Translation",
                    "affine_free": "Similarity", "affine_oracle": "Similarity (oracle)"}
CONDITION_LABELS_SHORT = {**CONDITION_LABELS, "original": "None"}       # 軸の目盛りなど狭い所
CONDITION_LABELS_JA = {"original": "なし", "gan": "GAN", "combat": "ComBat", "centroid": "平行移動",
                       "affine_free": "相似変換", "affine_oracle": "相似変換（正しい倍率・参考）"}


def order_conditions(variants) -> list[str]:
    """variant のリストを CONDITION_ORDER の順に並べる（未知の variant は後ろに元の順で）。"""
    variants = list(variants)
    known = [v for v in CONDITION_ORDER if v in variants]
    return known + [v for v in variants if v not in CONDITION_ORDER]


def ordered_subtypes(present: set, cfg: dict) -> list[str]:
    config_order = list(cfg.get("display", {}).get("colors", {}).get("subtypes", {}).keys())
    ordered = [s for s in config_order if s in present]
    remaining = sorted(s for s in present if s not in config_order)
    return ordered + remaining


def subtype_color_map(subtypes: list[str], cfg: dict) -> dict[str, str]:
    colors = cfg.get("display", {}).get("colors", {})
    explicit = colors.get("subtypes", {})
    sft_color = colors.get("sft", "#000000")
    tab20 = plt.get_cmap("tab20")
    auto_i, result = 0, {}
    for sub in subtypes:
        if "sft" in sub.lower():
            result[sub] = sft_color
        elif sub in explicit:
            result[sub] = explicit[sub]
        else:
            result[sub] = tab20(auto_i % 20)
            auto_i += 1
    return result


def shorten(name: str, cfg: dict) -> str:
    return cfg.get("display", {}).get("shortened", {}).get("subtypes", {}).get(name, name)


def src_code(src: str) -> str:
    return "".join(p[0].upper() for p in src.replace("-", "_").split("_") if p)[:3]


def make_abbrev(labels: list[str], cfg: dict) -> dict[str, str]:
    shortened = cfg.get("display", {}).get("shortened", {}).get("subtypes", {})
    result = {}
    for label in labels:
        if "__" in label:
            s, sub = label.split("__", 1)
            result[label] = f"{src_code(s)}-{shortened.get(sub, sub[:4].capitalize())}"
        else:
            result[label] = shortened.get(label, label[:4].capitalize())
    return result
