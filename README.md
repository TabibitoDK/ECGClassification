# ECGClassification

このプロジェクトでは、機械学習を使用して**心音の二値分類**を行います。

## インストール

### 環境

* Python 3.12.11
* uv

### ローカル環境

1. リポジトリのクローン

   ```bash
   git clone https://github.com/TabibitoDK/ECGClassification.git
   ```

2. uvのインストール

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
   source $HOME/.cargo/env
   uv --version
   ```

3. 仮想環境のセットアップ

   ```bash
   cd ECGClassification
   uv init
   uv venv
   uv add -r requirements.txt
   uv sync
   ```

## データセットの配置場所

以下の2つのCSVファイルが必要です:

- `mitbih_train.csv`
- `mitbih_test.csv`

デフォルトでは、`<repo>/data/` 内に配置されます。次のいずれかの方法で配置場所を変更（上書き）できます:

- `python -m src.train ...` に `--data-dir /path/to/folder` を渡す
- Pythonを実行する前に環境変数 `ECG_DATA_DIR=/path/to/folder` を設定する

### Google Colab

1. Googleドライブのマウント

   ```python
   from google.colab import drive 
   drive.mount('/content/drive')
   ```

2. リポジトリのクローン

   ```bash
   !git clone https://github.com/TabibitoDK/ECGClassification.git
   %cd ECGClassification
   ```

3. Colabランタイム内へのPython依存関係のインストール

   ```bash
   !pip install -r requirements.txt
   ```

4. データセットを学習スクリプトに指定します。CSVファイルをリポジトリ内にコピーするか、ドライブ内に保持したままフォルダを直接参照します:

   ```bash
   # 例: ドライブからリポジトリ内へコピーする場合
   !mkdir -p data
   !cp /content/drive/MyDrive/datasets/mitbih_*.csv data/
   # または、ドライブに置いたまま --data-dir /content/drive/MyDrive/datasets を指定する場合
   ```

5. 学習を開始します。チェックポイントやログをドライブに保存して永続化させます:

   ```bash
   !python -m src.train \
       --save-dir /content/drive/MyDrive/ecg_models \
       --train-name run-001 \
       --data-dir /content/drive/MyDrive/datasets \
       --tensorboard-log-dir /content/drive/MyDrive/ecg_models/run-001/tensorboard
   ```

6. 別のノートブックセルで、TensorBoardを使用してメトリクスをストリーミング表示できます:

   ```python
   %load_ext tensorboard
   %tensorboard --logdir /content/drive/MyDrive/ecg_models/run-001/tensorboard --port 6006
   ```

## 学習CLIオプション

学習のエントリポイントである `python -m src.train` は、ローカルまたはColabのワークフローに合わせて実行内容を調整するためのいくつかのフラグを受け付けます:

| オプション | 説明 |
| --- | --- |
| `--save-dir PATH` | **必須。** チェックポイント、ログ、バックアップ、TensorBoardファイルが保存されるフォルダ。 |
| `--train-name NAME` | **必須。** 実験ごとに隔離して保持するための `--save-dir` 内のサブディレクトリ名。 |
| `--data-dir PATH` | `mitbih_train.csv` と `mitbih_test.csv` を含むディレクトリ。デフォルトは `<repo>/data` または環境変数 `ECG_DATA_DIR`。 |
| `--tensorboard-log-dir PATH` | TensorBoardの出力場所を上書きします（デフォルト: `<save-dir>/<train-name>/tensorboard`）。 |
| `--resume` | `<save-dir>/<train-name>/latest_checkpoint.pth` から学習を再開します。 |
| `--autosave-interval N` | `N` エポックごとに `autosave_checkpoint.pth` ファイルを上書き保存します（デフォルト: 5）。 |
| `--log-interval N` | `N` エポックごとに `metrics.jsonl` へメトリクスを追記します（デフォルト: 5）。 |
| `--backup-interval N` | `N` エポックごとにタイムスタンプ付きのバックアップを保存します（デフォルト: 5）。 |
| `--model {resnet,vgg16}` | モデルのバックボーンを選択します。`resnet`（デフォルト）はより高い精度を提供し、`vgg16` は比較またはアブレーションラン用に使用できます。 |
| `--high-performance` | 大規模アクセラレータ（H100、A100など）向けにチューニングされた設定を有効にします：バッチサイズ1024、より多くのデータローダワーカー、線形学習率のスケーリング、安定性を維持するための線形ウォームアップ。 |

## TensorBoardによる学習のモニタリング

学習スクリプトはデフォルトで、TensorBoardのサマリーを `<save-dir>/<train-name>/tensorboard` に書き込みます。TensorBoardを起動して、損失、精度、学習率、およびエポックごとの分類レポートをモニタリングします:

```bash
tensorboard --logdir /path/to/save-dir/<train-name>/tensorboard --port 6006
```

複数の実行に対して一元管理したい場合は、`python -m src.train ...` 実行時に `--tensorboard-log-dir` を使用してログディレクトリを上書きできます。

## 最新のチェックポイントからの学習の再開

各実行では、`<save-dir>/<train-name>/latest_checkpoint.pth` に最新のチェックポイントがローリング方式で保存されます。`--resume` フラグを追加すると、そのファイルから学習を継続できます。スクリプトはモデルの重みとオプティマイザの状態を復元し、エポックカウンターを引き継ぐため、自動保存やメトリクスの番号付けがそのまま維持されます。

### ローカル

```bash
python -m src.train \
    --save-dir /absolute/path/to/save-dir \
    --train-name run-001 \
    --data-dir /absolute/path/to/datasets \
    --resume
```

`--save-dir` と `--train-name` が、すでに `latest_checkpoint.pth` を含んでいる同じ実行フォルダを指していることを確認してください。チェックポイントが見つからない場合、スクリプトはエラーで停止するため、誤って最初から学習を開始してしまうのを防ぐことができます。

### Google Colab

ドライブがマウントされているため、通常はすでにチェックポイントが保存されている同じドライブフォルダを再利用して再開します:

```bash
!python -m src.train \
    --save-dir /content/drive/MyDrive/ecg_models \
    --train-name run-001 \
    --data-dir /content/drive/MyDrive/datasets \
    --tensorboard-log-dir /content/drive/MyDrive/ecg_models/run-001/tensorboard \
    --resume
```

`--resume` は `/content/drive/MyDrive/ecg_models/run-001/latest_checkpoint.pth` が存在することを想定しています。複数の実行がある場合は、継続したい実行に合わせて `--train-name` を設定してください。再開されたセッション中に保存されたチェックポイントは同じディレクトリに蓄積され続けるため、ロギングを再起動することなく TensorBoard で進捗をモニタリングできます。
