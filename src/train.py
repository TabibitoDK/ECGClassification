import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from models import VGG16
from models import Resnet
from data_loader import ECGDataset
from sklearn.metrics import classification_report


START_EPOCH = 0
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_BATCH_SIZE = 16
BASE_LEARNING_RATE = 0.001
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_EPOCHS = 100
HIGH_PERFORMANCE_BATCH_SIZE = 1024
HIGH_PERFORMANCE_NUM_WORKERS = 16
HIGH_PERFORMANCE_WARMUP_EPOCHS = 5


def create_tensorboard_writer(log_dir: Path) -> SummaryWriter:
    """
    Create a tensorboard SummaryWriter and provide actionable errors if initialization fails.
    """
    try:
        log_dir.mkdir(parents = True, exist_ok = True)
    except Exception as exc:  # pragma: no cover - defensive filesystem guard
        raise RuntimeError(f'Unable to create TensorBoard log directory at {log_dir}. Check permissions and disk space. Original error: {exc}') from exc

    try:
        writer = SummaryWriter(log_dir = str(log_dir))
    except Exception as exc:  # pragma: no cover - tensorboard-specific error
        raise RuntimeError(f'Failed to initialize TensorBoard SummaryWriter in {log_dir}. Ensure tensorboard and protobuf packages are installed and the directory is writable. Original error: {exc}') from exc

    print(f'TensorBoard logs will be written to {log_dir}')
    return writer


def log_tensorboard_metrics(writer: SummaryWriter, log_dir: Path, epoch: int, report_text: str, *, train_loss: float, test_loss: float, train_accuracy: float, test_accuracy: float, learning_rate: float) -> None:
    """
    Record the metrics for an epoch and provide fast feedback if logging fails, which commonly leaves TensorBoard in the "inactive" state.
    """
    if not writer:
        return

    try:
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
        writer.add_scalar('LearningRate', learning_rate, epoch)
        writer.add_text('ClassificationReport', f'Epoch {epoch}\n{report_text}', epoch)
        writer.flush()  # flush immediately so TensorBoard sees new data without delay
    except Exception as exc:  # pragma: no cover - defensive logging guard
        raise RuntimeError(f'Failed to record TensorBoard metrics for epoch {epoch} in {log_dir}. Ensure the log directory is mounted and writable. Original error: {exc}') from exc


def compute_warmup_lr(epoch_index: int, target_lr: float, warmup_epochs: int) -> float:
    """
    Apply linear warmup for the specified number of epochs.
    """
    if warmup_epochs <= 0:
        return target_lr
    warmup_progress = min(1.0, (epoch_index + 1) / warmup_epochs)
    return target_lr * warmup_progress


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    """
    Update the learning rate for all optimizer parameter groups.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description = 'Train ECG classifier')
    parser.add_argument('--save-dir', required = True, help = 'Base directory to store model checkpoints (e.g., /content/drive/MyDrive/models)')
    parser.add_argument('--autosave-interval', type = int, default = 5, help = 'Save and overwrite autosave checkpoint every N epochs (default: 5)')
    parser.add_argument('--log-interval', type = int, default = 5, help = 'Record metrics to disk every N epochs (default: 5)')
    parser.add_argument('--backup-interval', type = int, default = 5, help = 'Create a backup checkpoint every N epochs (default: 5)')
    parser.add_argument('--train-name', required = True, help = 'Unique identifier for this training run')
    parser.add_argument('--data-dir', default = None, help = 'Directory containing mitbih_train.csv and mitbih_test.csv (defaults to <repo>/data)')
    parser.add_argument('--tensorboard-log-dir', default = None, help = 'Directory for TensorBoard logs (defaults to <save-dir>/<train-name>/tensorboard)')
    parser.add_argument('--resume', action = 'store_true', help = 'Resume training from the latest checkpoint for this run if present')
    parser.add_argument('--high-performance', action = 'store_true', help = 'Enable large batch configuration with aggressive data loading and linear LR scaling')
    parser.add_argument('--model', choices = ['resnet', 'vgg16'], default = 'resnet', help = 'Choose backbone architecture (default: resnet)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    autosave_interval = args.autosave_interval
    log_interval = args.log_interval
    backup_interval = args.backup_interval
    if autosave_interval <= 0:
        raise ValueError('autosave interval must be a positive integer')
    if log_interval <= 0:
        raise ValueError('log interval must be a positive integer')
    if backup_interval <= 0:
        raise ValueError('backup interval must be a positive integer')

    save_dir = Path(args.save_dir).expanduser()
    run_dir = save_dir / args.train_name
    run_dir.mkdir(parents = True, exist_ok = True)

    latest_checkpoint_path = run_dir / 'latest_checkpoint.pth'
    autosave_checkpoint_path = run_dir / 'autosave_checkpoint.pth'
    metrics_log_path = run_dir / 'metrics.jsonl'
    backup_dir = run_dir / 'backups'
    backup_dir.mkdir(exist_ok = True)
    tensorboard_log_dir = Path(args.tensorboard_log_dir).expanduser() if args.tensorboard_log_dir else run_dir / 'tensorboard'
    writer = create_tensorboard_writer(tensorboard_log_dir)

    default_data_dir = REPO_ROOT / 'data'
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f'Data directory {data_dir} does not exist')
    required_files = [data_dir / 'mitbih_train.csv', data_dir / 'mitbih_test.csv']
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f'Missing dataset files in {data_dir}: {", ".join(missing_files)}')
    os.environ['ECG_DATA_DIR'] = str(data_dir)
    print(f'Loading dataset from {data_dir}')

    high_performance = args.high_performance
    batch_size = HIGH_PERFORMANCE_BATCH_SIZE if high_performance else BASE_BATCH_SIZE
    num_workers = HIGH_PERFORMANCE_NUM_WORKERS if high_performance else DEFAULT_NUM_WORKERS
    num_epochs = DEFAULT_NUM_EPOCHS
    warmup_epochs = HIGH_PERFORMANCE_WARMUP_EPOCHS if high_performance else 0
    scaled_learning_rate = BASE_LEARNING_RATE * (batch_size / BASE_BATCH_SIZE)
    if high_performance:
        print('High performance mode enabled: batch_size=1024, linear LR scaling, warmup enabled')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using Device:', device)

    train_dataset = ECGDataset('train')
    test_dataset = ECGDataset('test')
    weights, _ = train_dataset.get_weights()
    sampler = WeightedRandomSampler(weights = weights, num_samples = len(train_dataset), replacement = True)
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, sampler = sampler, shuffle = False, num_workers = num_workers, pin_memory = torch.cuda.is_available())
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = torch.cuda.is_available())

    if args.model == 'resnet':
        model = Resnet().to(device)
    else:
        model = VGG16().to(device)
    print(f'Using model architecture: {args.model}')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = scaled_learning_rate)

    start_epoch = START_EPOCH
    if args.resume:
        if not latest_checkpoint_path.exists():
            raise FileNotFoundError(f'No latest checkpoint found at {latest_checkpoint_path} to resume from')
        checkpoint = torch.load(latest_checkpoint_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', START_EPOCH)
        print(f'Resuming training {args.train_name} from epoch {start_epoch}')
    
    print('---------- Start Training ----------')
        
    for epoch in range(start_epoch, num_epochs):
        current_learning_rate = compute_warmup_lr(epoch, scaled_learning_rate, warmup_epochs)
        set_optimizer_lr(optimizer, current_learning_rate)
        model.train()
        train_loss = 0.0
        train_correct = 0.0

        for i, (train_X, train_y) in tqdm(enumerate(train_loader), desc = f'Epoch {epoch + 1}/{num_epochs} [Train]', ncols = 100, total = len(train_loader)):
            train_X = train_X.to(device)
            train_y = train_y.to(device)

            optimizer.zero_grad()
            out = model(train_X)
            loss = criterion(out, train_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_X.size(0)
            _, predicted = torch.max(out, 1)
            train_correct += (predicted == train_y).sum().item()
        
        train_loss_epoch = train_loss / len(train_dataset)
        train_accuracy = train_correct / len(train_dataset)
        print(f'Train Loss:{train_loss_epoch:.4f} Train Accuracy:{train_accuracy:.4f}')

        model.eval()
        test_loss = 0.0
        test_correct = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for test_X, test_y in tqdm(test_loader, desc = f'Epoch {epoch + 1}/{num_epochs} [Test]', ncols = 100, total = len(test_loader)):
                test_X = test_X.to(device)
                test_y = test_y.to(device)

                out = model(test_X)
                loss = criterion(out, test_y)

                test_loss += loss.item() * test_X.size(0)
                _, predicted = torch.max(out, 1)
                test_correct += (predicted == test_y).sum().item()
                y_true.extend(test_y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        test_loss_epoch = test_loss / len(test_dataset)
        test_accuracy = test_correct / len(test_dataset)
        print(f'Test Loss:{test_loss_epoch:.4f} Test Accuracy:{test_accuracy:.4f}')
        report_text = classification_report(y_true, y_pred, zero_division = 0)
        print('\nClassification Report:')
        print(report_text)

        checkpoint_path = run_dir / f'{args.train_name}-epoch{epoch + 1:03d}-{test_accuracy:.4f}.pth'
        torch.save(model.state_dict(), checkpoint_path)

        latest_payload = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_loss': train_loss_epoch,
            'test_loss': test_loss_epoch,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        torch.save(latest_payload, latest_checkpoint_path)

        if writer:
            log_tensorboard_metrics(
                writer,
                tensorboard_log_dir,
                epoch + 1,
                report_text,
                train_loss = train_loss_epoch,
                test_loss = test_loss_epoch,
                train_accuracy = train_accuracy,
                test_accuracy = test_accuracy,
                learning_rate = optimizer.param_groups[0]['lr'],
            )

        if (epoch + 1) % autosave_interval == 0:
            torch.save(latest_payload, autosave_checkpoint_path)
            print(f'Autosaved checkpoint to {autosave_checkpoint_path}')

        if (epoch + 1) % log_interval == 0:
            metrics_entry = {
                'epoch': epoch + 1,
                'train_loss': train_loss_epoch,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss_epoch,
                'test_accuracy': test_accuracy,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
            with metrics_log_path.open('a', encoding = 'utf-8') as log_file:
                log_file.write(json.dumps(metrics_entry) + '\n')

        if (epoch + 1) % backup_interval == 0:
            backup_path = backup_dir / f'{args.train_name}-backup-epoch{epoch + 1:03d}.pth'
            torch.save(latest_payload, backup_path)
            print(f'Created backup checkpoint at {backup_path}')

    if writer:
        try:
            writer.flush()
            writer.close()
        except Exception as exc:  # pragma: no cover - defensive close guard
            raise RuntimeError(f'Failed to close TensorBoard writer at {tensorboard_log_dir}. Original error: {exc}') from exc

    print('---------- Finish Training ----------')
