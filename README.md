# 医療テキスト仮名化

**Author:** Yuka Otsuki
**Date:** 2026/3/18

---

## 1. 概要 (Overview)

日本語医療テキストを対象に，PHIタグ付与と De-identification を二段階で行う推論パイプラインです．
複数GPUによる並列推論に対応しています．

```
入力CSV
  │
  ▼
[Stage 1] PHIタグ推論 (phi_infer.py)
  │  LoRA適用モデルで個人情報候補に <phi_xxx> タグを付与
  ▼
[Stage 2] De-identification用前処理 (prepare_for_deid.py)
  │  候補氏名・住所リストをランダムサンプリングして付与
  ▼
[Stage 3] De-identification推論 (deid_infer.py)
  │  ベースモデルでPHIタグ箇所を匿名化テキストへ変換
  ▼
出力CSV（匿名化済みテキスト）
```

### PHIタグ一覧

| タグ | 対象 |
|------|------|
| `<phi_age>` | 年齢 |
| `<phi_id>` | 識別番号 |
| `<phi_tel>` | 電話番号 |
| `<phi_job>` | 職業 |
| `<phi_location>` | 住所・地名 |
| `<phi_person>` | 人名 |
| `<phi_hospital>` | 医療機関名 |

---

## 2. 環境構築 (Requirement)

Python 3.13、[uv](https://github.com/astral-sh/uv) を使用します。

```bash
git clone <repository_url>
cd <repository_name>
uv sync
```

### 主な依存ライブラリ

| ライブラリ | 用途 |
|------------|------|
| `transformers` | モデルのロード・推論 |
| `peft` | LoRAアダプタの適用 |
| `bitsandbytes` | 4bit量子化 |
| `torch` | GPU推論 |
| `pandas` | CSV入出力 |
| `pyyaml` | 設定ファイル読み込み |

---

## 3. ディレクトリ構成 (Directory Structure)

```
.
├── README.md
├── pyproject.toml
├── .gitignore
├── configs/
│   └── sample.yaml                 # 設定ファイルのサンプル
├── data/
│   ├── guidelines/                 # De-identification用ガイドラインテキスト
│   ├── guideline_phi_ner.txt       # PHIタグ推論用ガイドライン
│   ├── location*.csv               # 候補住所リスト
│   └── name*.ndjson                # 候補氏名リスト
├── src/
│   └── deid_pipeline/
│       ├── __init__.py
│       ├── cli.py                  # CLIエントリポイント
│       ├── runner.py               # ステージ実行ロジック
│       ├── phi_infer.py            # Stage 1: PHIタグ推論
│       ├── prepare_for_deid.py     # Stage 2: 前処理
│       ├── deid_infer.py           # Stage 3: De-identification推論
│       ├── merge.py                # シャードマージ
│       ├── config.py               # 設定ファイル読み込み
│       └── common.py               # 共通ユーティリティ
└── outputs/                        # 推論結果出力先（Git管理外）
```

---

## 4. 実行手順 (Usage)

### 設定ファイルの準備

`configs/sample.yaml` をコピーして編集してください．

```bash
cp configs/sample.yaml configs/ntcir.yaml
```

主要な設定項目：

| キー | 説明 |
|------|------|
| `hardware.gpus` | 使用するGPUのインデックスリスト |
| `hardware.phi_num_shards` | PHIタグ推論のシャード数（GPU数に合わせる） |
| `hardware.deid_num_shards` | De-identification推論のシャード数（GPU数に合わせる） |
| `paths.raw_input_csv` | 入力CSVのパス |
| `paths.output_root` | 出力ディレクトリのルート |
| `phi_stage.model_id` | PHIタグ推論のベースモデルパス |
| `phi_stage.lora_path` | LoRAアダプタのパス |
| `deid_stage.model_id` | De-identification推論のモデルパス |

> **注意:** `model_id` と `lora_path` はサーバー上の絶対パスを設定してください．モデル重みはGit管理外です．

### 入力CSVフォーマット

| カラム名 | 説明 |
|----------|------|
| `text` | 匿名化対象の医療テキスト（設定の `phi_stage.input_col` で変更可） |

### モデルのダウンロード

PHIタグ推論用のモデルは[ HuggingFace](https://huggingface.co/sociocom/MedPHINER-Llama-3.1-Swallow-8B-Instruct-v0.5) からダウンロードしてください．

```bash
huggingface-cli download sociocom/MedPHINER-Llama-3.1-Swallow-8B-Instruct-v0.5 --local-dir outputs/models/llama-3.1_phi_tag/best
```

> **注意:** ダウンロード先は設定ファイルの `phi_stage.lora_path` に合わせてください。

### 実行コマンド

全ステージ実行：

```bash
uv run deid-pipeline all --config configs/ntcir.yaml
```

ステージ個別実行：

```bash
# Stage 1のみ
uv run deid-pipeline phi --config configs/ntcir.yaml

# Stage 2+3（Stage 1完了後）
uv run deid-pipeline deid --config configs/ntcir.yaml
```

### 出力ファイル

```
outputs/
├── phi/<project_name>/            # Stage 1 シャード出力
├── phi/<project_name>_merged.csv
├── prepare/<project_name>_prepared.csv
├── deid/<project_name>/           # Stage 3 シャード出力
└── deid/<project_name>_merged.csv # 最終出力
```

---

## 5. 実験履歴・メモ (Experiment Log)

| 日付 | 内容 | 結果・メモ |
|------|------|------------|
|      |      |            |

---

## 6. 参考文献 (References)
