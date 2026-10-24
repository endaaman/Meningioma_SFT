# Meningioma_SFT

MeningiomaとSolitary Fibrous Tumor (SFT) を題材にした埋め込み解析・可視化スクリプト群。

## 構成

```
config.yaml          # 全設定（パス・色・パラメータ）
gan.py               # Cycle-GAN による補正後のバッチをエンコードして HDF5 書き込みするラッパー
macenko.py           # 色正規化（Macenko、patho2 → EBRAINS）: EBRAINS から目標を推定し、patho2 のパッチを正規化して CONCH でエンコード
titan.py             # TITAN による WSI-level aggregate（--variants で対象 variant を指定できる）
centroid.py          # 重心補正（centroid shift）の可視化・評価
combat.py            # ComBat（neuroCombat）による施設補正 → combat/。--covariates subtype で組織型を共変量にした combat_cov/（査読対応の参考）
nc.py                # 訓練不要の施設間分類（最近傍重心法）: SFT vs 髄膜腫と組織型。Fig 5・Table S1・Fig S3・Table S2 の元
held_out_case.py     # コホートから除外した held-out スライド（ラベルが矛盾する患者）の症例検討: 両施設の SFT / 髄膜腫 / 組織型の重心への距離と割り当て（out/held_out_case/）
stat6_case.py        # patho2 を STAT6 免疫染色の状態（レポート由来、config の stat6.source）で分けた記述統計: 両施設の SFT / 髄膜腫の重心への距離と割り当て、STAT6 を染めた髄膜腫 vs 染めていない髄膜腫（out/stat6/、data/patho2/stat6_status.csv）
lp.py                # TITAN 埋め込みの Linear Probe 訓練・評価（参考。論文の評価は nc.py）
lp_bootstrap.py      # lp の予測の患者単位 bootstrap（95% CI）と条件間 McNemar
umap_plot.py         # 補正前後の UMAP プロット（座標 coords.csv も保存）
subtype.py           # サブタイプ別平均ベクトルの UMAP
confusion_mtx.py     # 施設間距離行列（補正前後）
dendrogram.py        # 施設間距離の樹形図（補正前後）。生のユークリッド距離の行列・連結・ペアの定量も保存
dataset.py           # データセット構成の円グラフと Table 1 の元データ（table1.csv）
sft_distance_bar.py  # SFT と各サブタイプ間の距離棒グラフ
shift_direction.py   # 群ごとの施設間シフトの向きの一致度（centroid 補正の前提の検証）
shift_bio_axes.py    # 施設間シフト Δ の生物学的な軸（SFT−髄膜腫 / 組織型の組）に沿った成分 r と共分散を考慮した帰無。Fig 4e の元（out/shift_bio_axes/）
shift_estimation.py  # 平行移動のシフト推定の実用面: 症例構成の切り分け（組織型を合わせたシフト Δ*）と、新施設から何例あれば推定できるか（部分抽出のシミュレーション）。Fig S4 の元
harmonization.py     # 補正の効き具合の指標（ASW・LISI、補正なし / centroid / GAN）
meanpool.py          # 平均プーリングのスライド埋め込み（TITAN の対照）: build で h5 を作り、analyze で Fig 4・Table S3 と同じ指標を出す。Table S4 の元（幾何の数値は Results 本文）
paper_color.py       # スライドごとの組織の平均色（CIELAB と Ruifrok–Johnston の H/E、32 パッチ）。Fig 2c の元（out/paper/{version}/color/slide_color.csv）
stain_response.py    # 埋め込みの施設間シフト方向 u への射影 s と、スライドの H/E 染色バランス（log H/E）の対応。施設内相関・施設差の色による説明割合・応答曲線・補正（GAN/Macenko）による Δs と色変化の対応（out/stain_response/）
figs/                # 論文の図表を out/ の解析結果から組み立てる（重い計算はしない）
  __init__.py        # 図表番号の唯一の定義（MAIN / TABLES / SUPP の順序）
  common.py          # 出力先・共通スタイル・保存
  tissue.py / umap_harmonization.py / shift_geometry.py / classification.py / subtype_structure.py / pathology_checks.py  # 図（内容名）
  umap_held_out.py   # Fig 7a–b（held-out 症例）の描画。pathology_checks.py から呼ぶ（単体では登録しない）
  cohort.py          # 表
  subtype_distance_heatmap.py / sft_distance.py / histotype.py / shift_sample_size.py  # Supplementary の図
  performance.py / histotype_performance.py / method_comparison.py / meanpool_comparison.py / stat6_groups.py  # Supplementary の表
  meanpool_geometry.py  # 参考用（登録していない。figs.all では作らない）
  all.py             # 自分の出力を掃除してから全部作る
utils/
  loader.py          # load_config / load_data（各スクリプトが共通利用）
  display.py         # 色・順序・略称ヘルパー（ordered_subtypes / subtype_color_map / make_abbrev 等）
```

## 環境構築

```bash
# 依存パッケージのインストール（wsi-toolbox を GitHub から取得）
uv sync

# config.yaml は .gitignore 対象。テンプレートをコピーしてパスなどを設定する
cp config.example.yaml config.yaml
```

## フロー

### 実行順

```bash
uv run python gan.py       # GAN 補正パッチを patch-level embedding に変換して gan/ に書き出し
uv run python macenko.py all             # 色正規化: EBRAINS から目標を推定（fit）→ 目視用 preview → patho2 を正規化してエンコード（run）
uv run python titan.py     # patch-level を slide-level に集約（original / gan の 2 variant）
uv run python titan.py --variants macenko  # macenko variant の slide-level 集約
uv run python centroid.py  # slide-level を施設間で重心補正し centroid variant HDF5 を生成
uv run python combat.py    # ComBat（共変量なし）→ combat/
uv run python combat.py --covariates subtype   # 組織型を共変量にした ComBat → combat_cov/（査読対応の参考。combat/ は触らない）

# 以下は順不同（可視化・評価）
uv run python nc.py             # 施設間分類（最近傍重心法、訓練不要）。combat.py / affine.py の後
uv run python nc.py --aggregator mean   # 同じ分類を平均プーリングの埋め込みで（対照、meanpool.py build の後）。出力は out/meanpool/nc/
uv run python held_out_case.py  # コホートから除外した held-out スライド（config の held_out）を両施設の SFT / 髄膜腫の重心に当てる症例検討。出力は out/held_out_case/
uv run python stat6_case.py     # patho2 の STAT6 の状態別の記述統計（Supplementary 用）。出力は out/stat6/ と data/patho2/stat6_status.csv。プレビュー図は uv run python -m figs.stat6_preview
uv run python lp.py             # 参考（線形プローブ）
uv run python lp_bootstrap.py   # lp.py の後
uv run python umap_plot.py
uv run python subtype.py
uv run python confusion_mtx.py
uv run python dendrogram.py
uv run python dendrogram_bootstrap.py   # dendrogram.py と独立。出力は out/dendrogram/pairs_bootstrap.csv
uv run python sft_distance_bar.py
uv run python shift_direction.py
uv run python shift_bio_axes.py     # Δ の生物学的な軸に沿った成分（Fig 4e）。出力は out/shift_bio_axes/
uv run python shift_estimation.py   # 平行移動のシフト推定（症例構成の切り分け・推定に要る例数）。出力は out/shift_estimation/
uv run python harmonization.py
uv run python -u meanpool.py all   # 平均プーリングの対照（build は NFS から全パッチ埋め込みを読むので 10〜20 分）
uv run python stain_response.py    # paper_color.py / macenko.py の slide_color.csv と meanpool.py build の後。TITAN と平均プーリングの両方
uv run python dataset.py   # 他スクリプトと独立。任意のタイミングで実行可

# 論文用の図表（上の出力を使う）。出力は output_dir/paper/{paper.version}/{fig,tables}/、
# ファイル名は番号付き（例 fig3_umap_harmonization.png）。番号は figs/__init__.py だけで決まる
uv run python -m figs.all
uv run python -m figs.umap_harmonization   # 1 つだけ作る場合
```


### 前提

- **施設数は 2 つであること**（`site_a` / `site_b` は `config.example.yaml` のプレースホルダ名。実際の施設名に置き換えて使う）。3 施設以上には対応していない。
- 以下が済んでいること。

```
NDPIファイル
  ├─ [wsi_gan] ─────────────→ HDF5 key: cache/{size}/gan/patches   (GAN補正済みパッチ)
  └─ [wsi-toolbox] ─────────→ HDF5 key: {tile_model}/features      (original patch emb)
```
[wsi_gan]: https://github.com/S-murakami1/wsi_gan
[wsi-toolbox]: https://github.com/technoplasm/wsi-toolbox

- CSV と HDF5 は `config.yaml` で指定したパスに置くこと。

```yaml
label:
  site_a:
    label:   path/to/site_a/case_labels.csv
    subtype: path/to/site_a/case_subtypes.csv
    patient: path/to/site_a/case_patients.csv   # 任意（下記）
  site_b:
    label:   path/to/site_b/case_labels.csv
    subtype: path/to/site_b/case_subtypes.csv
embedding:
  site_a:
    original:   path/to/site_a/embedding   # HDF5 ファイル群のディレクトリ
    centroid:   path/to/centroid/site_a    # centroid.py が書き出す先
    gan:        path/to/gan/site_a         # gan.py が書き出す先
    tile_model: conch15_768                # wsi-toolbox のモデル名（HDF5 キーのプレフィックス）
  site_b:
    original:   path/to/site_b/embedding
    centroid:   path/to/centroid/site_b
    gan:        path/to/gan/site_b
    tile_model: conch15_768
```
- `patient`（`case_id,patient_id`）は任意。`lp.py` の train/val split を患者単位にするのに使い、無い施設は 1 症例 = 1 患者とみなす。EBRAINS は 1 患者に複数スライド（再発標本など）があるため必須で、`annotation.csv` の `uuid` 先頭 8 桁（= case_id）→ `pat_id` で作る。
- HDF5 ファイルは **variant ごとに別ディレクトリ**（`embedding.*.original` / `.gan` / `.centroid`）に配置する。3 variant は同じキー `{tile_model}/aggregates/titan/feature` に slide embedding を書くため、同一 HDF5 に共存させられない。

```
original/{case}.h5     ← 前処理ツール ([wsi_gan] / [wsi-toolbox]) の出力
├── cache/{gan.patch_size}/
│   ├── gan/patches       [wsi_gan]      → gan.py が読む
│   └── gan/coordinates   [wsi_gan]      → gan.py が読む
└── {tile_model}/
    ├── features          [wsi-toolbox]  → titan.py が読む
    ├── coordinates       [wsi-toolbox]  → titan.py が読む
    └── aggregates/titan/feature         ← titan.py が書く      → 解析が読む

gan/{case}.h5          ← gan.py + titan.py の出力
└── {tile_model}/
    ├── features                         ← gan.py が書く        → titan.py が読む
    ├── coordinates                      ← gan.py が書く        → titan.py が読む
    └── aggregates/titan/feature         ← titan.py が書く      → 解析が読む

centroid/{case}.h5     ← centroid.py の出力
└── {tile_model}/
    └── aggregates/titan/feature         ← centroid.py が書く   → 解析が読む
```

- `{gan.patch_size}` / `{tile_model}` は config の値に展開される（例: `cache/512/gan/patches`, `conch15_768/features`）

## 概略図

```
┌──────────────────────────────────┐
│  dataset.py                      │
│  subtype CSV → 円グラフ          │
└──────────────────────────────────┘

HDF5: cache/{size}/gan/patches
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  gan.py  — GAN補正パッチを tile model でエンコード       │
│  → {tile_model}/features  (GAN patch-level embedding)    │
└──────────────────────────────────────────────────────────┘
      │  GAN patch emb            original patch emb
      │                           (wsi-toolbox 済み)
      └───────────────┬───────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  titan.py  — TITAN で slide-level にアグリゲート         │
│  {tile_model}/features → {tile_model}/aggregates/titan   │
│  variants: original / gan                                │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼  slide-level embedding (HDF5)
┌──────────────────────────────────────────────────────────┐
│  centroid.py  — 施設間重心補正                           │
│  → 補正済み slide embedding を centroid variant HDF5 に  │
│    書き出し（配置は embedding.*.centroid で指定）        │
│  → PCA 3パネル / シフトノルム棒グラフ / ASW 評価         │
└──────────────────────────────────────────────────────────┘
                      │  (original / centroid / gan)
                      ▼
┌──────────────────────────────────────────────────────────┐
│  lp.py  — Linear Probe 交差データセット評価              │
│  施設 A→B と B→A の双方向 × 3 variant (original /        │
│  centroid / gan) を比較                                  │
│  → confusion_matrix{,_norm}.png / metrics.csv /          │
│    comparison.csv / checkpoints/ / logs/                 │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼  
──── slide-level embedding を使った各種解析 ────────────────
  umap_plot.py          補正前後の UMAP プロット
  subtype.py            サブタイプ別平均ベクトルの UMAP
  confusion_mtx.py      施設間距離行列（補正前後）
  dendrogram.py         施設間距離の樹形図（補正前後）
  sft_distance_bar.py   SFT と各サブタイプ間の距離棒グラフ
  shift_direction.py    群ごとの施設間シフトの向き（全体シフトとの cos、例数を揃えた帰無と比較）
  shift_estimation.py   平行移動のシフト推定（組織型を合わせたシフト Δ* との比較、新施設の部分抽出で推定したときの精度）
  harmonization.py      施設差の補正の効き具合（ASW・iLISI / cLISI）
```
