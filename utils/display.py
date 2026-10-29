"""display.py — 色・順序・略称の表示ヘルパー共通処理。"""
from __future__ import annotations

import matplotlib.pyplot as plt

# 補正条件の色（全図で共通。施設色の青・赤と被らない Tol bright の灰・紫・水色）
CONDITION_COLORS = {"original": "#BBBBBB", "centroid": "#AA3377", "gan": "#66CCEE", "macenko": "#009988",
                    "combat": "#CCBB44", "combat_cov": "#E6DC99", "affine_free": "#228833", "affine_oracle": "#88BB88",
                    "combat_recentred": "#997700"}   # combat_recentred は転移テスト（transfer_correction.py）だけ
# 補正条件の集合（図表ごとにどちらを使うかを各 figs/*.py で明示する）。
# 並びは全図表で共通: 補正なし → Macenko → GAN → ComBat → 平行移動（centroid。画像レベルの弱い順 → 埋め込みレベル、
# 提案の平行移動を最後に。Macenko を GAN の前に置くのは ken 決定 2026-10-20）。
#   FULL: 4 条件。Fig 5・Fig 6・Fig S1・S2
#   FIG3: 5 条件 = FULL ＋ Macenko。Fig 3（UMAP と補正法の比較）と Supplementary の Table S1・S2・S4、Fig S3
#   CORE: ComBat を除く 3 条件（harmonization.draw_metric の既定など、本番図表では未使用）
CONDITIONS_FULL = ["original", "gan", "combat", "centroid"]
CONDITIONS_FIG3 = ["original", "macenko", "gan", "combat", "centroid"]
CONDITIONS_CORE = [v for v in CONDITIONS_FULL if v != "combat"]
# 並び順の基準（order_conditions が使う。全条件の順）
CONDITION_ORDER = ["original", "macenko", "gan", "combat", "combat_cov", "centroid", "affine_free", "affine_oracle"]
# Table S3（補正法の比較、全指標）だけ: FIG3 に相似変換を足したもの。affine_oracle（正しい倍率・参考）は図表に出さない
# （ken 決定 2026-10-20。解析の出力 out/method_compare/ 等には残る）
CONDITIONS_SUPP = [v for v in CONDITION_ORDER if v in CONDITIONS_FIG3 or v == "affine_free"]
# 補正条件の表示名（図表・表の表示はすべてここから引く。コード内部のキーは変えない）。
#   centroid      = 平行移動（各施設の平均を揃える位置のみの補正）
#   affine_free   = 相似変換（一様なスケール＋平行移動、倍率はラベル不要の推定）
#   affine_oracle = 相似変換（倍率を組織型ラベルから推定した参考値。論文の図表には出さない）
#   combat        = ComBat（次元ごとの位置・尺度の補正）
#   combat_cov    = ComBat に組織型を共変量として渡したもの（評価施設のラベルを使う査読対応の参考値。
#                   FULL / FIG3 / SUPP のどれにも入れない＝図表には出さない。結果は decisions.md に記録）
#   macenko       = 色正規化（Macenko。画像レベル、GAN と同じ向き patho2 → EBRAINS、目標は EBRAINS 全体から 1 つ）
#   combat_recentred = 転移テスト（transfer_correction.py）だけの変種: ComBat の次元ごとの倍率を推定元から移し、
#                   中心は当て先の施設平均で取り直す（当て先の科のラベルが要る版）。論文の図表の条件集合には入れない
CONDITION_LABELS = {"original": "No correction", "gan": "GAN", "macenko": "Macenko", "combat": "ComBat",
                    "combat_cov": "ComBat (histotype covariate)",
                    "centroid": "Translation", "affine_free": "Similarity", "affine_oracle": "Similarity (oracle)",
                    "combat_recentred": "ComBat (recentred)"}
CONDITION_LABELS_SHORT = {**CONDITION_LABELS, "original": "None"}       # 軸の目盛りなど狭い所
CONDITION_LABELS_JA = {"original": "なし", "gan": "GAN", "macenko": "色正規化（Macenko）", "combat": "ComBat",
                       "combat_cov": "ComBat（組織型を共変量）",
                       "centroid": "平行移動", "affine_free": "相似変換", "affine_oracle": "相似変換（正しい倍率・参考）",
                       "combat_recentred": "ComBat（中心を当て先で取り直し）"}

# スライドエンコーダ（encoder_comparison.py の列。「別系統のスライドエンコーダでも同じ」の比較）。
# 並び: パッチエンコーダごとに「パッチ埋め込みの単純平均（学習なしの集約器の対照）→ 学習した集約器」が隣り合う
# （CONCH v1.5: 平均 = meanpool → TITAN、Virchow v1（concat 2560 次元）: 平均 = meanpool_virchow → PRISM）。
# 平均は 2 つあるので表示名にパッチエンコーダを付ける（CONCH の平均 / Virchow の平均）。
# 色は 4 つで全対の色覚の区別（2 型色覚の模擬で ΔE ≥ 9、通常の色覚で ≥ 24）を満たす組。CONCH の 2 つは寒色、Virchow の 2 つは暖色
# （エンコーダの色。補正条件の色 CONDITION_COLORS とは別の役目なので同じ図で混ぜない）
ENCODER_ORDER = ["meanpool", "titan", "meanpool_virchow", "prism"]
ENCODER_LABELS = {"meanpool": "CONCH mean", "titan": "TITAN", "meanpool_virchow": "Virchow mean", "prism": "PRISM"}
ENCODER_LABELS_JA = {"meanpool": "CONCH の平均", "titan": "TITAN", "meanpool_virchow": "Virchow の平均", "prism": "PRISM"}
ENCODER_COLORS = {"meanpool": "#1baf7a", "titan": "#2a78d6", "meanpool_virchow": "#882255", "prism": "#eb6834"}
ENCODER_MARKERS = {"meanpool": "s", "titan": "o", "meanpool_virchow": "^", "prism": "D"}
# 腫瘍の科（転移テストの 5 科。コード上の名前は shift_transfer / transfer_correction の family のまま）
FAMILY_LABELS = {"meningioma": "Meningioma", "SFT": "SFT", "glioma": "Glioma", "lymphoma": "DLBCL", "metastasis": "Metastasis"}
FAMILY_LABELS_JA = {"meningioma": "髄膜腫", "SFT": "SFT", "glioma": "グリオーマ", "lymphoma": "DLBCL", "metastasis": "転移"}
# 転移テストの当て先 グリオーマの 3 群（manifest の group）。表示名は config の shift_transfer.group_names と同じ。
# 色は髄膜腫サブタイプの色（config の display.colors.subtypes、tab20 から）に無い 3 色（figs/transfer_structure_preview.py）
GLIOMA_GROUP_LABELS = {"ast": "Astrocytoma, IDH-mutant", "glm": "Glioblastoma, IDH-wildtype", "odg": "Oligodendroglioma"}
GLIOMA_GROUP_COLORS = {"ast": "#17becf", "glm": "#bcbd22", "odg": "#7b4173"}


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
