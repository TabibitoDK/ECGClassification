"""Evaluate a trained ECG model and print precision/recall/F1 table."""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

REPO_ROOT = Path(__file__).resolve().parent.parent

if __package__:
    from .data_loader import ECGDataset
    from .models import Resnet, VGG16
else:  # allow execution via `python src/evaluate_model.py`
    import sys

    SRC_ROOT = Path(__file__).resolve().parent
    if str(SRC_ROOT) not in sys.path:
        sys.path.append(str(SRC_ROOT))
    from data_loader import ECGDataset
    from models import Resnet, VGG16


CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Evaluate model on the ECG dataset.')
    parser.add_argument('--checkpoint', type = Path, required = True, help = 'Checkpoint file to evaluate. Accepts either model.state_dict() or training payloads.')
    parser.add_argument('--model', choices = ['resnet', 'vgg16'], default = 'resnet', help = 'Model architecture that matches the checkpoint (default: resnet).')
    parser.add_argument('--data-dir', type = Path, default = None, help = 'Directory containing mitbih_{train,test}.csv (defaults to <repo>/data or ECG_DATA_DIR).')
    parser.add_argument('--split', choices = ['train', 'test'], default = 'test', help = 'Dataset split to evaluate (default: test).')
    parser.add_argument('--batch-size', type = int, default = 256, help = 'Batch size for evaluation (default: 256).')
    parser.add_argument('--num-workers', type = int, default = 4, help = 'Number of dataloader workers (default: 4).')
    parser.add_argument('--output-csv', type = Path, default = None, help = 'Optional CSV file to save the classification table.')
    return parser.parse_args()


def resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir:
        resolved = data_dir.expanduser()
    elif os.environ.get('ECG_DATA_DIR'):
        resolved = Path(os.environ['ECG_DATA_DIR']).expanduser()
    else:
        resolved = REPO_ROOT / 'data'
    if not resolved.exists():
        raise FileNotFoundError(f'Data directory {resolved} does not exist. Use --data-dir or set ECG_DATA_DIR.')
    os.environ['ECG_DATA_DIR'] = str(resolved)
    return resolved


def build_model(name: str) -> torch.nn.Module:
    if name == 'vgg16':
        return VGG16()
    return Resnet()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path.expanduser(), map_location = device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    if missing:
        print(f'Warning: missing keys in checkpoint: {missing}')
    if unexpected:
        print(f'Warning: unexpected keys in checkpoint: {unexpected}')


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, dim = 1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return average_loss, accuracy, np.array(y_true), np.array(y_pred)


def build_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(
        y_true,
        y_pred,
        labels = labels,
        target_names = CLASS_NAMES,
        zero_division = 0,
        output_dict = True,
    )
    return report


def format_table(report: Dict[str, Dict[str, float]]) -> Tuple[str, List[Sequence[str]]]:
    header = f"{'Category':10} {'Precision':10} {'Recall':10} {'F1-score':10} {'Support':10}"
    separator = '-' * len(header)
    lines = [header, separator]
    csv_rows: List[Sequence[str]] = [['Category', 'Precision', 'Recall', 'F1-score', 'Support']]
    for class_name in CLASS_NAMES:
        metrics = report.get(class_name, {})
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1-score', 0.0)
        support = int(metrics.get('support', 0))
        lines.append(f"{class_name:10} {precision:10.2f} {recall:10.2f} {f1:10.2f} {support:10d}")
        csv_rows.append([class_name, f'{precision:.4f}', f'{recall:.4f}', f'{f1:.4f}', str(support)])
    accuracy = report.get('accuracy', 0.0)
    macro = report.get('macro avg', {})
    weighted = report.get('weighted avg', {})
    lines.append(separator)
    lines.append(f"{'Accuracy':10} {accuracy:10.2f}")
    lines.append(f"{'Macro Avg':10} {macro.get('precision', 0.0):10.2f} {macro.get('recall', 0.0):10.2f} {macro.get('f1-score', 0.0):10.2f}")
    lines.append(f"{'Weighted':10} {weighted.get('precision', 0.0):10.2f} {weighted.get('recall', 0.0):10.2f} {weighted.get('f1-score', 0.0):10.2f}")
    return '\\n'.join(lines), csv_rows


def save_csv(rows: List[Sequence[str]], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open('w', newline = '', encoding = 'utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    print(f'Saved classification table to {path}')


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    print(f'Using dataset directory: {data_dir}')

    dataset = ECGDataset(args.split)
    if len(dataset) == 0:
        raise RuntimeError(f'No samples found in the {args.split} split at {data_dir}.')

    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers, pin_memory = torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model).to(device)
    load_checkpoint(model, args.checkpoint, device)
    print(f'Evaluating checkpoint: {args.checkpoint}')

    loss, accuracy, y_true, y_pred = evaluate(model, dataloader, device)
    report = build_report(y_true, y_pred)
    table_text, csv_rows = format_table(report)

    print('\\nEvaluation metrics:')
    print(f'Loss: {loss:.4f}  Accuracy: {accuracy:.4f}')
    print('\\nClassification table:\\n')
    print(table_text)

    if args.output_csv:
        save_csv(csv_rows, args.output_csv)


if __name__ == '__main__':
    main()
