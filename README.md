# Meningioma_SFT

MeningiomaとSolitary Fibrous Tumor (SFT) を題材にした埋め込み解析・可視化スクリプト群。

## 構成

```
config.yaml          # 全設定（パス・色・パラメータ）
gan.py               # Cycle-GAN による補正後のバッチをエンコードして HDF5 書き込みするラッパー
macenko.py           # 色正規化（Macenko、patho2 → EBRAINS）: EBRAINS から目標を推定し、patho2 のパッチを正規化して CONCH でエンコード
titan.py             # TITAN による WSI-level aggregate（--variants で対象 variant を指定できる）
titan_batch.py       # 外部の h5 の木に TITAN を書く（EBRAINS 全体の CONCH 特徴 = /mnt/server42/AI/Datasets/EBRAINS_embeddings。config の ebrains_embeddings.priority のグリオーマを先に、残りは名前順。ログは out/ebrains_embeddings/）
patho2_gliomas_extract.py  # 院内グリオーマ（/mnt/server42/AI/Datasets/Patho2Gliomas、旧 BrainTumor）の HE から CONCH v1.5（wsi-toolbox 0.7.0 の preset conch15、cls 1024 と proj 768 を 1 回で）+ TITAN を作り Patho2Gliomas_embeddings に置く（EBRAINS_embeddings と同じ構成）。HE の判定（ファイル名）と全 ndpi の一覧は out/patho2_gliomas/slides.csv
patho2_gliomas_review.py   # 院内グリオーマの 1 症例 1 枚の代表 HE を目で選ぶツール: HE 全部のサムネイル（thumbs）と、症例ごとに並べてクリックで選ぶページ（serve: クリックごとに data/patho2_gliomas/representatives.csv が書かれる。html: サーバーなしの静的版）。HE 1 枚の症例は自動で選択済み、EXCLUDE で症例除外。解析はこの CSV を読む
transfer_manifest.py # 転移テスト（3 法 × 推定元 × 当て先）で使う全スライドの一覧（髄膜腫+SFT コホート ＋ 院内代表 ＋ EBRAINS 対応診断、2,301 枚）と TITAN / PRISM の有無 → out/transfer/manifest.csv。Virchow 抽出の入力にもなる
transfer_correction.py # 転移テスト本体: 施設補正 3 法（平行移動 / 相似変換 / ComBat ＋ 倍率だけ移す ComBat-recentred）を推定元（科・paper・others・科ごとに当てはめて平均する others_mean・own）で当てはめて当て先（paper / glioma / lymphoma / metastasis / all）に当てる。残差・施設 ASW・歪み・論文のベンチ（対・組織型と SFT の最近傍重心法）と、own・paper で論文の値の再現確認（--encoder titan|prism|meanpool|meanpool_virchow、out/transfer/correction_{encoder}.*）
prism_extract.py     # 第 2 のスライドエンコーダ: manifest の全スライドに Virchow（224 px / 0.5 mpp、concat 2560 次元）→ PRISM を足す。書き先は既存の h5（EBRAINS_embeddings / Patho2Gliomas_embeddings / 論文の original）の virchow_2560/…。既存はスキップ、ログ out/transfer/prism_log.csv
atlas_retrieval.py   # 参照コホートへの診断名の検索: EBRAINS 全体（5 枚以上の 78 診断）の診断ごとの重心で院内スライドを最近傍診断。補正なし / 論文の Δ / 他の科の Δ（全平均・科ごと）/ 自分の科の Δ と EBRAINS 内 LOO（out/atlas/）。--encoder prism は参照が PRISM のある EBRAINS 1,464 枚・22 診断に縮む。TITAN を同じ参照で比べるには --atlas-slides prism（出力 *_titan_on_prism*）
dispersion_ratio.py  # Supplementary 用: 施設間の「ばらつきの比」（次元ごとの分散比）は腫瘍をまたいで共有されるか。科どうしの相関 vs 分割半分の再現性、平均のずれの cos と並べる（out/dispersion_ratio/）。shift_transfer.py のキャッシュを読む。--encoder meanpool / prism / meanpool_virchow は *_meanpool / *_prism / *_meanpool_virchow
shift_transfer.py    # 転移テストの予備解析: 髄膜腫+SFT コホートの Δ と、グリオーマ（ast / glm / odg）・DLBCL・転移の施設間シフトの向き（cos、ノイズ補正つき）と長さ。群どうしの cos 行列、施設内の科どうしの差との直交性（out/shift_transfer/）。--encoder meanpool は out/shift_transfer/meanpool/、--encoder prism は out/shift_transfer/prism/（embeddings_prism.npy を引く。組成の感度の IDH 野生型星細胞腫 66 枚は manifest 外で PRISM が無いので NaN）。--encoder meanpool_virchow は out/shift_transfer/meanpool_virchow/（Virchow のパッチ平均、embeddings_meanpool_virchow.npy を引く。同じく 66 枚は NaN）
transfer_meanpool.py # 転移テストの全スライドの平均プーリング埋め込み（CONCH v1.5 パッチ平均、meanpool.py と同じ定義。髄膜腫+SFT コホートは既存の meanpool h5、グリオーマ側は h5 の features から計算）→ out/transfer/embeddings_meanpool.npy。shift_transfer / transfer_correction / dispersion_ratio の --encoder meanpool が読む（集約器の対照）。--source virchow は Virchow のタイル特徴（h5_prism の virchow_2560/features、2560 次元）の平均 → embeddings_meanpool_virchow.npy（PRISM の集約器の対照、--encoder meanpool_virchow が読む。NFS を読むので --workers 4 まで、途中から再開できる、約 1.5 時間）
transfer_umap.py     # 全 2,301 枚（5 科 × 2 施設）の UMAP: 補正なしで 1 回当てはめ、Δ_paper / 自分の科の Δ で補正した院内を transform で写す（補正なし / +Δ_paper / +own と科ごとの小図）。768 次元での院内の 15 近傍に占める EBRAINS の割合を併記（out/transfer_umap/）
transfer_sample_size.py # 別の腫瘍に移せる Δ に要る枚数と種類数（平行移動のみ）: 推定元の院内 n 枚（200 回）→ 他の科の残差・施設 ASW・NC BA・対と、全枚数の 10% 以内に入る n。当て先以外の k 科の科ごと平均 vs まとめ（out/transfer_sample_size/）。Fig S4 の作り直し
transfer_color.py    # 転移テストの全 2,301 枚の色（32 パッチの L*a*b*・H/E とスライド内のばらつき）と、科ごとの色の幅の比 vs 埋め込みの倍率 s（色を除いた s も）、施設軸 Δ_paper への射影の H 応答・平均色で説明できる施設差（out/transfer_color/）。build（WSI を読む、長い）→ analyze
encoder_baseline.py   # エンコーダの基礎精度: 4 つのスライド埋め込みが施設内（EBRAINS 内・patho2 内）で SFT vs 髄膜腫 / 主要 5 組織型 / グリオーマ 3 群 / 5 科をどれだけ解けるか。最近傍重心法・患者単位の層化 5 分割 CV（seed 42）の BA と macro F1 に、施設をまたぐ値（correction_{encoder}.csv の none / own）を参考に並べる（out/encoder_baseline/）。転移テストの埋め込みキャッシュを読むだけ
encoder_comparison.py # 同じ解析を 4 つのスライド埋め込み（CONCH の平均 / TITAN / Virchow の平均 / PRISM。パッチ平均 → 集約器 が隣り合う並び）で並べた表と図（各解析の出力を読むだけ。ただし施設差の相対的な大きさ ‖Δ_paper‖ / 群内 RMS と Δ_f の分解は転移テストの埋め込みキャッシュ out/transfer/embeddings_{encoder}.npy から計算。out/encoder_comparison/）。v4 Fig 6 の材料。出力の無いエンコーダの列は空欄
shift_time.py        # 論文には出さない（記録のみ）: 院内の Δ は年をまたいで安定か。受付年ごとの Δ の cos・長さ、同じ枚数・同じ組成の帰無、並べ替え検定（out/shift_time/）
centroid.py          # 重心補正（centroid shift）の可視化・評価
combat.py            # ComBat（neuroCombat）による施設補正 → combat/。--covariates subtype で組織型を共変量にした combat_cov/（査読対応の参考）
nc.py                # 訓練不要の施設間分類（最近傍重心法）: SFT vs 髄膜腫と組織型。Fig 5・Table S1・Fig S3・Table S2 の元
knn.py               # 施設間分類のスライド単位 k 近傍法（nc.py の対照、平均を使わない。k = 1 / 5 / 15、Table S1 と同じ 5 条件）。Table S6 の元（out/knn/）
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
uv run python knn.py            # 同じ分類をスライド単位の k 近傍法で（最近傍重心法の対照）。出力は out/knn/binary/
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

# グリオーマ（平行移動 Δ の転移テスト用。論文のコホートとは別の h5 の木。どちらも既存はスキップするので途中から再開できる）
uv run python -u titan_batch.py --dry-run                  # EBRAINS_embeddings の対象と順番（グリオーマ → 残り）
uv run python -u titan_batch.py                            # 学生が作った CONCH 特徴（3,121 本）に TITAN を書く（約 2 秒/本）
uv run python -u patho2_gliomas_extract.py --dry-run       # 院内の HE 判定の監査（out/patho2_gliomas/slides.csv）
uv run python -u patho2_gliomas_extract.py                 # 院内 A/G/O/L/M の HE 全部を抽出（約 30–45 秒/枚）。--groups A G O で群を絞れる
uv run python -u patho2_gliomas_review.py thumbs           # HE 全部のサムネイル（out/patho2_gliomas/review/thumbs/、4 分）
uv run python -u patho2_gliomas_review.py serve            # tmux の担当 window で。http://127.0.0.1:8765/ で代表を選ぶ → data/patho2_gliomas/representatives.csv
uv run python transfer_manifest.py                         # 代表を選び終えてから。転移テストの全スライド一覧と埋め込みの揃い具合（out/transfer/manifest.csv）
uv run python transfer_correction.py                       # transfer_manifest.py の後。3 法 × 推定元 × 当て先（--encoder prism は PRISM が揃ってから。埋め込みはキャッシュ、--reload で読み直し）
uv run python -u prism_extract.py --dry-run                # Virchow → PRISM の対象と状態（manifest の h5_prism）
uv run python -u prism_extract.py                          # tmux の担当 window で。EBRAINS → 院内の順に 2,301 枚（1〜3 分/枚、数日）。--sites / --families で絞れる
uv run python atlas_retrieval.py                           # 参照コホートへの診断名の検索（参照 = EBRAINS 全体、out/atlas/）。埋め込みはキャッシュ（--reload で読み直し）。--min-n で参照のクラスの最小枚数（既定以外は出力名に _min{n}）、--encoder prism は抽出後
uv run python dispersion_ratio.py                          # shift_transfer.py の後。ばらつきの比は腫瘍をまたいで共有されるか（out/dispersion_ratio/、Supplementary 用）
uv run python shift_transfer.py                            # 代表を選び終えてから。Δ の向きの群・科による違い（out/shift_transfer/）。埋め込みはキャッシュ（--reload で読み直し）
uv run python -u transfer_meanpool.py build                # transfer_manifest.py の後。平均プーリング埋め込みのキャッシュ（out/transfer/embeddings_meanpool.npy。NFS から features を読む）
uv run python shift_transfer.py --encoder meanpool         # 同様に transfer_correction.py / dispersion_ratio.py --encoder meanpool（集約器の対照。出力は meanpool/ か *_meanpool）
uv run python transfer_umap.py                             # transfer_correction.py（埋め込みのキャッシュ）の後。5 科の UMAP（out/transfer_umap/）
uv run python transfer_sample_size.py                      # 同。枚数 n・種類数 k vs 転移の指標（out/transfer_sample_size/。--summary-only で CSV から図だけ）
uv run python transfer_color.py build --workers 4           # transfer_manifest.py の後。全スライドの色（h5 のパッチ画素か元 WSI。NFS を読むので 4 並列まで、途中から再開できる。約 4 時間）
uv run python transfer_color.py analyze                     # build と transfer_correction.py（埋め込みのキャッシュ）の後（out/transfer_color/）
uv run python shift_time.py                                # 論文には出さない。院内の Δ の経年安定性（out/shift_time/、--plot-only で図だけ）
# PRISM 側（prism_extract.py が全部終わってから。順番どおりに）
uv run python transfer_manifest.py                         # 流し直す（prism 列が True になる）
uv run python shift_transfer.py --encoder prism            # --bootstrap patients も。embeddings_prism.npy のキャッシュを作る（out/shift_transfer/prism/）
uv run python dispersion_ratio.py --encoder prism          # *_prism
uv run python transfer_correction.py --encoder prism       # 上 2 本の後（受け入れ確認がその出力を読む）
uv run python atlas_retrieval.py --encoder prism           # --min-n 20 も。参照は PRISM のある 22 診断に縮む
uv run python atlas_retrieval.py --atlas-slides prism      # TITAN を同じ 22 診断の参照で（*_titan_on_prism*）
uv run python transfer_sample_size.py --encoder prism      # 約 20 分（out/transfer_sample_size/prism/）
uv run python -u transfer_meanpool.py build --source virchow --workers 4   # Virchow のパッチ平均（PRISM の集約器の対照）→ embeddings_meanpool_virchow.npy。prism_extract.py の後。4 並列まで、途中から再開できる、約 1.5 時間
uv run python shift_transfer.py --encoder meanpool_virchow # --bootstrap patients も（out/shift_transfer/meanpool_virchow/）
uv run python dispersion_ratio.py --encoder meanpool_virchow   # *_meanpool_virchow
uv run python transfer_correction.py --encoder meanpool_virchow # 上 2 本の後（受け入れ確認がその出力を読む）
uv run python encoder_baseline.py                          # 各 --encoder の transfer_correction.py の後（埋め込みのキャッシュと correction_{encoder}.csv を読むだけ、数秒）。施設内の基礎精度（out/encoder_baseline/）
uv run python encoder_comparison.py                        # 各 --encoder の解析（shift_transfer / transfer_correction / dispersion_ratio / atlas_retrieval（--atlas-slides prism も）/ transfer_sample_size）の後。4 エンコーダの横並び（out/encoder_comparison/）。atlas_retrieval / transfer_sample_size は CONCH の平均・Virchow の平均では回していないので空欄

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
  shift_transfer.py     Δ の向きは腫瘍の科を越えて同じか（髄膜腫サブタイプ / SFT / グリオーマ 3 群 / DLBCL / 転移のシフトの cos 行列、ノイズ補正つき）
  harmonization.py      施設差の補正の効き具合（ASW・iLISI / cLISI）
```
