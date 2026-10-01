"""Generate a schematic model diagram similar to thesis figures."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


BLOCK_LAYOUTS: Dict[str, List[Tuple[str, float, float]]] = {
    'resnet': [
        ('Input\nScalogram', 1.5, 3.0),
        ('Stem\n7x7 Conv\n64 ch', 1.3, 3.0),
        ('Residual Block x2\n64 ch', 1.5, 3.0),
        ('Residual Block x2\n128 ch', 1.8, 3.0),
        ('Residual Block x2\n256 ch', 2.1, 3.0),
        ('Residual Block x2\n512 ch', 2.4, 3.0),
        ('Global\nAvgPool', 1.5, 2.0),
        ('FC\n5 classes', 1.0, 1.5),
    ],
    'vgg16': [
        ('Input\nScalogram', 1.5, 3.0),
        ('Conv x2\n64 ch', 1.5, 3.0),
        ('Conv x2\n128 ch', 1.7, 3.0),
        ('Conv x3\n256 ch', 1.9, 3.0),
        ('Conv x3\n512 ch', 2.1, 3.0),
        ('Conv x3\n512 ch', 2.1, 3.0),
        ('Flatten', 1.5, 2.0),
        ('FC x2\n4096 units', 2.4, 2.5),
        ('FC\n5 classes', 1.2, 1.5),
    ],
}

COLOR_PALETTE = ['#f4a261', '#2a9d8f', '#457b9d', '#6d597a', '#b56576', '#eaac8b', '#b5179e', '#4361ee', '#38a3a5']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = 'Render a schematic diagram of the ECG classification model.')
    parser.add_argument('--model', choices = BLOCK_LAYOUTS.keys(), default = 'resnet', help = 'Architecture to visualize (default: resnet).')
    parser.add_argument('--output', type = Path, default = Path('model_diagram.png'), help = 'File path to save the figure (default: model_diagram.png).')
    parser.add_argument('--title', type = str, default = 'ECG Classification Model', help = 'Title shown above the diagram.')
    parser.add_argument('--show', action = 'store_true', help = 'Display the figure window instead of only saving it.')
    return parser.parse_args()


def draw_diagram(model_name: str, title: str) -> plt.Figure:
    blocks = BLOCK_LAYOUTS[model_name]
    fig, ax = plt.subplots(figsize = (14, 4))
    ax.set_axis_off()
    ax.set_title(title, fontsize = 16, pad = 20)

    x = 0.5
    y = 0.5
    gap = 0.35

    for idx, (label, width, height) in enumerate(blocks):
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]
        rect = Rectangle((x, y), width, height, linewidth = 1.5, edgecolor = 'black', facecolor = color, alpha = 0.85)
        ax.add_patch(rect)
        ax.text(x + width / 2, y + height / 2, label, ha = 'center', va = 'center', fontsize = 12, color = 'white', weight = 'bold')

        if idx < len(blocks) - 1:
            next_x = x + width + gap
            arrow = FancyArrowPatch(
                (x + width, y + height / 2),
                (next_x, y + height / 2),
                arrowstyle = '->',
                mutation_scale = 20,
                linewidth = 2,
                color = '#264653',
            )
            ax.add_patch(arrow)
            x = next_x

    total_width = sum(width for _, width, _ in blocks) + gap * (len(blocks) - 1)
    ax.set_xlim(0, total_width + 1)
    ax.set_ylim(0, 4.5)
    return fig


def main() -> None:
    args = parse_args()
    fig = draw_diagram(args.model, args.title)
    output_path = args.output.expanduser()
    fig.savefig(output_path, dpi = 300, bbox_inches = 'tight')
    print(f'Saved diagram to {output_path}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
