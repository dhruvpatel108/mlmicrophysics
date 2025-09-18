#!/usr/bin/env python3
"""
Plot histograms of PREPROCESSED training data (exactly as seen by the model).

This script uses the existing data loading and preprocessing pipeline configured
in the YAML config to ensure the exact same transforms (log, scaling, etc.) are
applied. It then samples a subset of the preprocessed tensors and plots
histograms for all input features and output targets.

Usage:
  python scripts/plot_preprocessed_histograms.py \
    --config configs/deception_quick_test.yml \
    --split train \
    --max-samples 200000 \
    --bins 100 \
    --output-dir <dir>

Notes:
- Works with both loader types: "streaming" and "optimized" (and aliases).
- Inputs are the scaled tensors from the dataset; outputs are the log-transformed
  tendencies produced by the preprocessing.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import logging
import math
import numpy as np
import torch

# Use non-interactive backend for HPC/headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ensure project root (parent of scripts/) is on sys.path for 'models' imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Local imports (module path relative to this file's parent)
from models.streaming_data_loader import create_streaming_data_loaders
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plot_preprocessed_histograms")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def select_loader_factory(loader_type: str):
    lt = (loader_type or '').lower()
    if lt in ("optimized", "optimized_streaming", "optimized-streaming"):
        return create_optimized_streaming_loaders, "optimized"
    # Fallback to the original/streaming implementation
    return create_streaming_data_loaders, "streaming"


def collect_samples_from_loader(loader, input_names, output_names, max_samples: int) -> tuple:
    """Collect up to max_samples from the preprocessed tensors yielded by the loader.

    Returns:
        inputs_dict: {name: np.ndarray}
        outputs_dict: {name: np.ndarray}
        total_collected: int
    """
    inputs_dict = {name: [] for name in input_names}
    outputs_dict = {name: [] for name in output_names}
    total_collected = 0

    # Iterate batches; both loader types yield (x_batch, targets)
    for batch in loader:
        try:
            x_batch, targets = batch
        except Exception:
            # Some collate functions might give dict differently; skip if not tuple
            continue

        # Ensure CPU numpy
        if isinstance(x_batch, torch.Tensor):
            x_np = x_batch.detach().cpu().numpy()
        else:
            x_np = np.asarray(x_batch)

        batch_size = x_np.shape[0]
        if batch_size == 0:
            continue

        # Determine how many samples to take from this batch
        remaining = max_samples - total_collected
        take = min(remaining, batch_size)
        if take <= 0:
            break

        # Uniform subsample indices if batch bigger than needed
        if take < batch_size:
            sel_idx = np.random.choice(batch_size, size=take, replace=False)
            x_sel = x_np[sel_idx]
        else:
            x_sel = x_np

        # Map inputs by column order
        for col_idx, name in enumerate(input_names):
            inputs_dict[name].append(x_sel[:, col_idx])

        # Outputs are stored in targets dict, as tensors shaped (N, 1) or (N,)
        for out_name in output_names:
            if out_name not in targets:
                continue
            t = targets[out_name]
            if isinstance(t, torch.Tensor):
                t_np = t.detach().cpu().view(-1).numpy()
            else:
                t_np = np.asarray(t).reshape(-1)

            if take < batch_size:
                t_np = t_np[sel_idx]
            outputs_dict[out_name].append(t_np)

        total_collected += x_sel.shape[0]
        if total_collected >= max_samples:
            break

    # Concatenate lists into arrays
    for k in inputs_dict:
        if inputs_dict[k]:
            inputs_dict[k] = np.concatenate(inputs_dict[k], axis=0)
        else:
            inputs_dict[k] = np.array([])
    for k in outputs_dict:
        if outputs_dict[k]:
            outputs_dict[k] = np.concatenate(outputs_dict[k], axis=0)
        else:
            outputs_dict[k] = np.array([])

    return inputs_dict, outputs_dict, total_collected


def _plot_hist_grid(data_dict: dict, title: str, bins: int, fig_path: Path):
    names = list(data_dict.keys())
    num_vars = len(names)
    if num_vars == 0:
        logger.warning(f"No variables to plot for {title}")
        return

    # Compute grid
    n_cols = int(math.ceil(math.sqrt(num_vars)))
    n_rows = int(math.ceil(num_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols + 1, 3.2 * n_rows + 1))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes[idx]
        arr = np.asarray(data_dict[name]).astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_title(f"{name} (no data)")
            ax.axis('off')
            continue
        ax.hist(arr, bins=bins, color='steelblue', alpha=0.85, density=False)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)

    # Turn off any extra axes
    for j in range(num_vars, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot histograms of preprocessed training data")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config used for training')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'all'], help='Data split to draw from')
    parser.add_argument('--max-samples', type=int, default=200000, help='Maximum number of samples to include in histograms')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins')
    parser.add_argument('--output-dir', type=str, default='plots/preprocessed_histograms', help='Directory to save plots')
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get('data', {})
    loader_type = data_cfg.get('loader_type', 'streaming')
    input_cols = data_cfg.get('input_cols', [])
    output_cols = data_cfg.get('output_cols', [])
    data_path = data_cfg.get('data_path')
    scaler_cache_dir = data_cfg.get('scaler_cache_dir', './scaler_cache')

    if not input_cols or not output_cols:
        raise ValueError("input_cols and output_cols must be defined in the config")

    if not data_path:
        raise ValueError("data.data_path must be defined in the config")

    loader_factory, resolved = select_loader_factory(loader_type)
    logger.info(f"Using loader type: {resolved}")

    # Build loaders using the exact training configuration
    if resolved == "optimized":
        train_loader, val_loader, _ = loader_factory(
            data_path=data_path,
            config=config,
            train_fraction=data_cfg.get('train_fraction', 0.8),
            batch_size=data_cfg.get('batch_size', 1024),
            scaler_cache_dir=data_cfg.get('scaler_cache_dir', './scaler_cache')
        )
    else:
        train_loader, val_loader, _ = loader_factory(
            data_path=data_path,
            config=config,
            train_fraction=data_cfg.get('train_fraction', 0.8),
            batch_size=data_cfg.get('batch_size', 1024),
            num_workers=data_cfg.get('num_workers', 0),
            scaler_cache_dir=data_cfg.get('scaler_cache_dir', './scaler_cache')
        )

    loader = train_loader if args.split == 'train' else (val_loader if args.split == 'val' else train_loader)

    # Collect preprocessed samples
    inputs_dict, outputs_dict, n = collect_samples_from_loader(
        loader=loader,
        input_names=input_cols,
        output_names=output_cols,
        max_samples=args.max_samples
    )
    logger.info(f"Collected {n} preprocessed samples from split='{args.split}'")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot inputs and outputs
    _plot_hist_grid(inputs_dict, title='Inputs (preprocessed & scaled)', bins=args.bins,
                    fig_path=out_dir / f"inputs_{resolved}_{args.split}.png")
    _plot_hist_grid(outputs_dict, title='Outputs (log-transformed tendencies)', bins=args.bins,
                    fig_path=out_dir / f"outputs_{resolved}_{args.split}.png")

    logger.info("Done.")


if __name__ == '__main__':
    main()


