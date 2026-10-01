"""Utility for inspecting trained ECG classification models."""

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

if __package__:
    from .data_loader import ECGDataset
    from .models import Resnet, VGG16
else:  # allow execution via `python src/show_layers.py`
    import sys

    SRC_ROOT = Path(__file__).resolve().parent
    if str(SRC_ROOT) not in sys.path:
        sys.path.append(str(SRC_ROOT))
    from data_loader import ECGDataset
    from models import Resnet, VGG16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Show model layers and tensor shapes.')
    parser.add_argument('--checkpoint', type = Path, default = None, help = 'Optional checkpoint to load before printing the layers.')
    parser.add_argument('--model', choices = ['resnet', 'vgg16'], default = 'resnet', help = 'Model architecture to inspect (default: resnet).')
    parser.add_argument('--data-dir', type = Path, default = None, help = 'Directory with mitbih_{train,test}.csv. Used to infer input shapes (defaults to <repo>/data or ECG_DATA_DIR).')
    parser.add_argument('--split', choices = ['train', 'test'], default = 'test', help = 'Dataset split to draw a representative example from (default: test).')
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
    checkpoint = torch.load(checkpoint_path, map_location = device)
    state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict = False)
    if missing:
        print(f'Warning: missing keys in checkpoint: {missing}')
    if unexpected:
        print(f'Warning: unexpected keys in checkpoint: {unexpected}')


def _layer_has_children(module: torch.nn.Module) -> bool:
    return any(True for _ in module.children())


def collect_layer_shapes(model: torch.nn.Module, sample_input: torch.Tensor, device: torch.device) -> List[Tuple[str, str, Sequence[int], Sequence[int], int]]:
    layer_rows: List[Tuple[str, str, Sequence[int], Sequence[int], int]] = []
    hooks = []

    def register_hook(name: str, module: torch.nn.Module) -> None:
        if _layer_has_children(module):
            return

        def hook(_, inputs, output):
            input_shape = tuple(inputs[0].shape) if inputs else 'n/a'
            if isinstance(output, torch.Tensor):
                output_shape: Sequence[int] | str = tuple(output.shape)
            elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                output_shape = tuple(output[0].shape)
            else:
                output_shape = 'n/a'
            num_params = sum(p.numel() for p in module.parameters())
            layer_rows.append((name or '<module>', module.__class__.__name__, input_shape, output_shape, num_params))

        hooks.append(module.register_forward_hook(hook))

    for module_name, module in model.named_modules():
        if module_name == '':
            continue
        register_hook(module_name, module)

    model.eval()
    with torch.no_grad():
        _ = model(sample_input.to(device))

    for handle in hooks:
        handle.remove()

    return layer_rows


def format_table(rows: List[Tuple[str, str, Sequence[int], Sequence[int], int]]) -> str:
    header = f"{'Layer':30} {'Type':15} {'Input Shape':20} {'Output Shape':20} {'Params':>10}"
    lines = [header, '-' * len(header)]
    for layer_name, layer_type, input_shape, output_shape, num_params in rows:
        lines.append(f"{layer_name:30} {layer_type:15} {str(input_shape):20} {str(output_shape):20} {num_params:10d}")
    return '\n'.join(lines)


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    print(f'Using dataset directory: {data_dir}')

    dataset = ECGDataset(args.split)
    if len(dataset) == 0:
        raise RuntimeError(f'No samples found in the {args.split} split at {data_dir}.')
    sample_X, _ = dataset[0]
    sample_batch = sample_X.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model).to(device)
    print(f'Inspecting model: {args.model}')
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint.expanduser(), device)
        print(f'Loaded weights from {args.checkpoint}')

    print('\nModel architecture:\n')
    print(model)

    rows = collect_layer_shapes(model, sample_batch, device)
    table = format_table(rows)
    print(f'\nLayer-by-layer tensor shapes using one {args.split} sample:\n')
    print(table)

    total_params, trainable_params = count_parameters(model)
    print(f'\nTotal parameters: {total_params:,}\nTrainable parameters: {trainable_params:,}')


if __name__ == '__main__':
    main()
