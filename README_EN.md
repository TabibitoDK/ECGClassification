# ECGClassification

This project uses machine learning to perform **binary classification of heart sounds**.

## Installation

### Environment

  * Python 3.12.11
  * uv

### Local Environment

1.  Clone the repository

    ```bash
    git clone https://github.com/Miria089/ECGClassification.git
    ```

2.  Install uv

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
    source $HOME/.cargo/env
    uv --version
    ```

3.  Set up the virtual environment

    ```bash
    cd ECGClassification
    uv init
    uv venv
    uv add -r requirements.txt
    uv sync
    ```

## Dataset Location

Two CSV files are required:

- `mitbih_train.csv`
- `mitbih_test.csv`

By default, they should live under `<repo>/data/`. You can override the location by either:

- Passing `--data-dir /path/to/folder` to `python -m src.train ...`
- Setting the environment variable `ECG_DATA_DIR=/path/to/folder` before running Python

### Google Colab

1.  Mount Google Drive

    ```python
    from google.colab import drive 
    drive.mount('/content/drive')
    ```

2.  Clone the repository

    ```bash
    !git clone https://{PAT}@github.com/Miria089/ECGClassification.git
    %cd ECGClassification
    ```

3.  Install the Python dependencies inside the Colab runtime

    ```bash
    !pip install -r requirements.txt
    ```

4.  Point the training script to your dataset. Either copy the CSV files into the repo or keep them in Drive and reference the folder directly:

    ```bash
    # Example: copy from Drive into the repo
    !mkdir -p data
    !cp /content/drive/MyDrive/datasets/mitbih_*.csv data/
    # or, keep them in Drive and use --data-dir /content/drive/MyDrive/datasets
    ```

5.  Start training. Save checkpoints/logs back to Drive so they persist:

    ```bash
    !python -m src.train \
        --save-dir /content/drive/MyDrive/ecg_models \
        --train-name run-001 \
        --data-dir /content/drive/MyDrive/datasets \
        --tensorboard-log-dir /content/drive/MyDrive/ecg_models/run-001/tensorboard
    ```

6.  In another notebook cell you can stream metrics with TensorBoard:

    ```python
    %load_ext tensorboard
    %tensorboard --logdir /content/drive/MyDrive/ecg_models/run-001/tensorboard --port 6006
    ```

## Training CLI Options

The training entry point `python -m src.train` accepts several flags so you can tailor the run for local or Colab workflows:

| Option | Description |
| --- | --- |
| `--save-dir PATH` | **Required.** Folder where checkpoints, logs, backups, and TensorBoard files are stored. |
| `--train-name NAME` | **Required.** Subdirectory under `--save-dir` to keep each experiment isolated. |
| `--data-dir PATH` | Directory containing `mitbih_train.csv` and `mitbih_test.csv`. Defaults to `<repo>/data` or the `ECG_DATA_DIR` environment variable. |
| `--tensorboard-log-dir PATH` | Override the TensorBoard output location (default: `<save-dir>/<train-name>/tensorboard`). |
| `--resume` | Resume training from `<save-dir>/<train-name>/latest_checkpoint.pth`. |
| `--autosave-interval N` | Overwrite the `autosave_checkpoint.pth` file every `N` epochs (default: 5). |
| `--log-interval N` | Append metrics to `metrics.jsonl` every `N` epochs (default: 5). |
| `--backup-interval N` | Store timestamped backups every `N` epochs (default: 5). |
| `--model {resnet,vgg16}` | Pick the model backbone. `resnet` (default) offers higher accuracy; `vgg16` is available for comparison or ablation runs. |
| `--high-performance` | Enables a configuration tuned for large accelerators (H100 A100, etc.): batch size 1024, more dataloader workers, linear learning-rate scaling, and linear warmup to maintain stability. |

## Monitoring Training with TensorBoard

The training script now writes TensorBoard summaries to `<save-dir>/<train-name>/tensorboard` by default. Launch TensorBoard to monitor losses, accuracies, learning rate, and the per-epoch classification report:

```bash
tensorboard --logdir /path/to/save-dir/<train-name>/tensorboard --port 6006
```

You can override the log directory via `--tensorboard-log-dir` when calling `python -m src.train ...` if you prefer a centralized location for multiple runs.

## Continuing Training from the Latest Checkpoint

Every run stores a rolling checkpoint at `<save-dir>/<train-name>/latest_checkpoint.pth`. Add the `--resume` flag to continue training from that file; the script restores the model weights, optimizer state, and picks up the epoch counter so autosaves/metrics keep the same numbering.

### Local

```bash
python -m src.train \
    --save-dir /absolute/path/to/save-dir \
    --train-name run-001 \
    --data-dir /absolute/path/to/datasets \
    --resume
```

Make sure you point `--save-dir` and `--train-name` to the same run folder that already contains `latest_checkpoint.pth`. The script will error out if the checkpoint is missing, which protects you from accidentally starting over.

### Google Colab

Because Drive is mounted, you typically resume by reusing the same Drive folder that already holds your checkpoints:

```bash
!python -m src.train \
    --save-dir /content/drive/MyDrive/ecg_models \
    --train-name run-001 \
    --data-dir /content/drive/MyDrive/datasets \
    --tensorboard-log-dir /content/drive/MyDrive/ecg_models/run-001/tensorboard \
    --resume
```

`--resume` expects `/content/drive/MyDrive/ecg_models/run-001/latest_checkpoint.pth` to exist. If you have multiple runs, set `--train-name` to the one you want to continue. Checkpoints saved during the resumed session will keep accumulating in the same directory so you can monitor progress with TensorBoard without relaunching logging.
