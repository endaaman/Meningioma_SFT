# Meningioma_SFT

髄膜腫と孤在性線維性腫瘍（SFT）の WSI を、病理基盤モデル（CONCH v1.5 → TITAN）のスライド埋め込みで
解析し、施設差（院内 `patho2` と公開データ `ebrains`）の形と補正を調べる。学生（山田）の研究を ken が
引き取って論文化している（2026-10〜）。

## 単一情報源

- 構成・実行順: `README.md`
- 原稿の置き場と約束: `out/paper/README.md`
- 図表の構成（Fig / Table ごとの主張・パネル・数値・状態）: `out/paper/v1/figures.md`
- 論文の骨組み（段落ごとの要素）: `out/paper/v1/outline.md`
- 解析の判断の記録（試したこと・数字・決めたこと）: `out/paper/decisions.md`
- 別論文の種: `out/ideas/`

## データ

- `data` と `out` は `~/Sync/Projects/Meningioma_SFT/{data,out}` への symlink（git 管理外、Syncthing 同期）。
- 入力の original h5 は NFS（`/mnt/server42/個人用/山田/...`）への symlink。**NFS 上の既存ファイルは上書き・削除・リネームしない。** 新規作成が要るときは ken に確認。
- 生成物（`centroid` / `gan` / `combat` / `affine_*` の h5、`out/`）は Sync 側の実体。
- `config.yaml` は git 管理外。新しい設定キーは `config.example.yaml` にも足す。

## 解析の約束

- **評価は最近傍重心法（`nc.py`、訓練不要）に統一。** 線形プローブ（`lp.py`）は参考扱い（数字は decisions.md）。
- 補正の表示名とコード上の名前: 平行移動 = `centroid`、相似変換 = `affine_free`（正しい倍率・参考 = `affine_oracle`）、ComBat = `combat`、GAN = `gan`。**表示名は `utils/display.py` の 1 か所**、コード上の名前・出力パスは変えない。
- 条件の集合と並び: `utils/display.py` の `CONDITIONS_FULL`（補正なし → GAN → ComBat → 平行移動）、`CONDITIONS_CORE`、Supplementary 用は `CONDITIONS_SUPP*`。
- 施設 `ebrains` が基準（reference、補正しない側）。GAN は patho2 → EBRAINS の向きだけを使う（元コード: https://github.com/S-murakami1/wsi_gan、その設定のまま）。
- 乱数は固定（seed 42）、学習は決定的。患者単位で分割（EBRAINS は `data/ebrains/patients.csv`、patho2 は 1 症例 = 1 患者）。

## 図表

- 論文用の図表は `figs/` が作る。**ファイル名は内容名**（`figs/umap_harmonization.py` 等）、**番号は `figs/__init__.py` の `MAIN` / `TABLES` / `SUPP` / `SUPP_TABLES` だけで決まる**。`uv run python -m figs.all` で全部作り直す。
- 解析スクリプトが `out/` に CSV 等を出し、`figs/` は読んで組むだけ（重い計算をしない）。
- 本番に入れるか未定のものは**プレビュー**（`_preview_` で始まる名前、`figs/*_preview.py`）。使わないものは ken が `_unused/` に移す。**`_unused` を根拠として参照しない**（参照するのは「使わない」と明示するときだけ）。根拠に使う結果は解析の出力（`out/<解析>/`）に置く。
- 図を変えたら Read で目視し、変えていないはずの図表はハッシュで不変を確認する。

## 書き方

- 原稿は日本語・pandoc markdown。主張は控えめに（「平行移動で十分」「補正どうしの差は決定的でない」）。
- 投稿先の候補から Scientific Reports は外す（ken の希望）。

## git

- このリポジトリは commit / push してよい（ken 了承済み）。`git add` は明示パスのみ。
- 依存の wsi-toolbox は GitHub の commit を `uv.lock` で固定。wsi-toolbox 側のバージョンは上げない（bump は ken がリリース時に行う）。
