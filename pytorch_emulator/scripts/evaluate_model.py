#!/usr/bin/env python3
"""
Evaluate Constraint-Aware Microphysics Emulator

Comprehensive evaluation script with beautiful visualizations:
- Scatter plots of true vs predicted values
- R² calculations and metrics
- Physical constraint validation
- Regime analysis plots
- Training curves

Usage:
    python evaluate_model.py --checkpoint outputs/best_checkpoint.pth --config configs/quick_test_5epochs.yml
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List, Optional
import warnings
import logging
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add model and training paths
sys.path.append('models')
sys.path.append('training')

# Import our components
from physics_emulator import ConstraintAwareEmulator
from losses import ConstraintAwareLoss, create_constraint_aware_loss
#from data_loader import create_data_loaders
from streaming_data_loader_v2 import create_optimized_streaming_loaders
from trainer import ConstraintAwareTrainer

# Columns that undergo log-space operations in dataset preprocessing
LOG_TENDENCY_COLUMNS = {"qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"}
LOG_EPSILON = 1e-10

# Set style for beautiful plots
#plt.style.use('seaborn-v0_8')
#sns.set_palette("husl")


def normalize_run_id(run_id: Optional[str]) -> Optional[str]:
    """Ensure run IDs are consistently formatted as run_<id>."""
    if run_id is None:
        return None
    run_id = str(run_id).strip()
    if not run_id:
        return None
    return run_id if run_id.startswith("run_") else f"run_{run_id}"


def infer_run_id_from_path(path_like: str) -> Optional[str]:
    """Extract a run_<id> component from the given path, if present."""
    try:
        path = Path(path_like)
    except Exception:
        return None
    for part in reversed(path.parts):
        if part.startswith("run_"):
            return part
    return None


def load_model_and_config(checkpoint_path: str, config_path: str):
    """Load trained model and configuration."""
    logger.info(f"📁 Loading checkpoint: {checkpoint_path}")
    logger.info(f"📋 Loading config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Create model
    model_config = config['model']
    # Backward-compat: allow either 'head_dims' (preferred) or single 'head_dim'
    head_dims = model_config.get('head_dims')
    if head_dims is None:
        single = model_config.get('head_dim')
        head_dims = [single] if single is not None else [64]

    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dims=head_dims,
        dropout=model_config['dropout']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"   Parameters: {model.get_parameter_count():,}")
    
    return model, config, checkpoint


def make_predictions(model, data_loader, device='cpu', log_frequency=0, max_batches=None):
    """Make predictions on a dataset."""
    model.to(device)
    model.eval()

    all_predictions = {
        'is_active': [],
        'qrtend': [],
        'nctend': [],
        'nrtend': [],
        'qctend': []
    }
    all_targets = {
        'is_active': [],
        'qrtend_TAU': [],
        'nctend_TAU': [],
        'nrtend_TAU': [],
        'qctend_TAU': []
    }
    
    logger.info("🔮 Making predictions...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Model predictions
            predictions = model(inputs)
            
            # Store predictions
            for key in all_predictions:
                if key in predictions:
                    all_predictions[key].append(predictions[key].cpu().numpy())
            
            # Store targets
            for key in all_targets:
                if key in targets:
                    all_targets[key].append(targets[key].cpu().numpy())
            
            if log_frequency and ((batch_idx + 1) % log_frequency == 0):
                total_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
                progress = f"{batch_idx + 1}"
                if total_batches:
                    progress = f"{batch_idx + 1}/{total_batches}"
                logger.info(f"  Processed {progress} batches")
    
    # Concatenate all batches
    for key in all_predictions:
        if all_predictions[key]:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0).flatten()
        else:
            all_predictions[key] = np.array([])
    
    for key in all_targets:
        if all_targets[key]:
            all_targets[key] = np.concatenate(all_targets[key], axis=0).flatten()
        else:
            all_targets[key] = np.array([])
    
    logger.info(f"✅ Predictions completed: {len(all_predictions['qrtend'])} samples")
    
    return all_predictions, all_targets


def restore_outputs_to_eval_space(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    dataset,
    target_space: str = "physical"
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Undo dataset-level transforms (scaling, quantile, log10) and project the
    data into the requested ``target_space``.
    """
    if target_space not in {"physical", "log"}:
        raise ValueError(f"Unsupported target_space '{target_space}'")

    dataset_output_cols = getattr(dataset, "output_cols", None)
    #if not dataset_output_cols:
    #    logger.info("Dataset does not define output columns; skipping inverse transformation.")
    #    return predictions, targets

    usable_cols = [col for col in dataset_output_cols if col in predictions and col in targets]
    if not usable_cols:
        logger.warning("No overlapping output columns found for inverse transformation.")
        return predictions, targets

    scaler = getattr(dataset, "output_scaler", None)
    transformer = getattr(dataset, "output_transformer", None)
    transform_mode = getattr(dataset, "output_transform", "log10")
    usable_log_columns = [col for col in usable_cols if col in LOG_TENDENCY_COLUMNS]

    def _inverse_pipeline(matrix: np.ndarray) -> np.ndarray:
        restored = matrix
        if scaler is not None:
            try:
                restored = scaler.inverse_transform(restored)
            except Exception as exc:
                logger.warning(f"Failed to invert dataset output scaler: {exc}")
        if transform_mode == "quantile" and transformer is not None:
            try:
                restored = transformer.inverse_transform(restored)
            except Exception as exc:
                logger.warning(f"Failed to invert quantile transformer: {exc}")
        return restored

    def _transform_to_physical_space(data_dict: Dict[str, np.ndarray]) -> None:
        stacked = np.stack([data_dict[col] for col in usable_cols], axis=1)
        restored_matrix = _inverse_pipeline(stacked)
        for idx, col in enumerate(usable_cols):
            data_dict[col] = restored_matrix[:, idx]
        if transform_mode == "log10":
            for col in usable_log_columns:
                if col in data_dict:
                    values = data_dict[col]
                    sign = np.sign(values)
                    abs_val = np.power(10.0, np.abs(values)) - LOG_EPSILON
                    data_dict[col] = sign * abs_val

    def _convert_to_log_space(data_dict: Dict[str, np.ndarray]) -> None:
        if target_space != "log":
            return
        for col in usable_log_columns:
            if col in data_dict:
                values = data_dict[col]
                sign = np.sign(values)
                abs_val = np.abs(values) + LOG_EPSILON
                data_dict[col] = sign * np.log10(abs_val)

    # Stage 1: undo dataset transforms back to physical units
    _transform_to_physical_space(predictions)
    _transform_to_physical_space(targets)

    # Stage 2: map to requested evaluation domain
    if target_space == "log":
        _convert_to_log_space(predictions)
        _convert_to_log_space(targets)

    return predictions, targets


def calculate_metrics(y_true, y_pred, name):
    """Calculate regression metrics."""
    if len(y_true) == 0 or len(y_pred) == 0:
        logger.warning(f"No samples available to calculate metrics for {name}")
        return {
            'name': name,
            'r2': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'l2': np.nan,
            'mean_true': np.nan,
            'mean_pred': np.nan,
            'std_true': np.nan,
            'std_pred': np.nan
        }

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    l2_error = np.linalg.norm(y_true - y_pred)
    
    return {
        'name': name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'l2': l2_error,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }


def create_scatter_plots(predictions, targets, output_dir, evaluation_space: str):
    """Create beautiful scatter plots with R² values."""
    logger.info("📊 Creating scatter plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the variables to plot
    plot_vars = [
        ('qrtend', 'qrtend_TAU', 'Rain Tendency (qrtend)'),
        ('nctend', 'nctend_TAU', 'Cloud Number Tendency (nctend)'),
        ('nrtend', 'nrtend_TAU', 'Rain Number Tendency (nrtend)'),
        ('qctend', 'qctend_TAU', 'Cloud Water Tendency (qctend)')
    ]
    
    # Create subplot figure
    fig, axes_grid = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Constraint-Aware Emulator: True vs Predicted', fontsize=16, fontweight='bold')
    
    axes = axes_grid.flatten()
    axes_list = axes.tolist()
    metrics_list = []
    hexbin_handles: List = []
    log_norm = LogNorm(vmin=1)
    max_count = 1.0
    
    for i, (pred_key, target_key, title) in enumerate(plot_vars):
        ax = axes[i]
        logger.info(f"Plotting {pred_key} vs {target_key}")
        #if pred_key == 'qctend':
            # For qctend, we derive it from qrtend in the ground truth
        #    y_true = -targets['qctend_TAU']  # qctend should equal -qrtend
        #    y_pred = predictions['qctend']
        #else:
        #    y_true = targets[target_key]
        #    y_pred = predictions[pred_key]
        y_true = targets[target_key]
        y_pred = predictions[pred_key]

        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, pred_key)
        metrics_list.append(metrics)
        
        hb = ax.hexbin(
            y_true,
            y_pred,
            gridsize=200,
            cmap='viridis',
            mincnt=1,
            norm=log_norm
        )
        hexbin_handles.append(hb)
        counts = hb.get_array()
        if counts.size > 0:
            current_max = counts.max()
            if np.isfinite(current_max):
                max_count = max(max_count, float(current_max))
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        
        # Formatting
        axis_suffix = " (log space)" if evaluation_space == "log" else ""
        ax.set_xlabel(f'True {title}{axis_suffix}', fontsize=12)
        ax.set_ylabel(f'Predicted {title}{axis_suffix}', fontsize=12)
        ax.set_title(f'{title}\nR² = {metrics["r2"]:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
    
    if hexbin_handles:
        log_norm.vmax = max_count
        for hb in hexbin_handles:
            hb.set_clim(log_norm.vmin, log_norm.vmax)
        # Create colorbar that spans the full height of the 2x2 subplot grid
        # Position it on the right side without overlapping
        cbar = fig.colorbar(
            hexbin_handles[0],
            ax=axes_grid,
            pad=0.08,
            aspect=40,
            shrink=1.0
        )
        cbar.set_label('Frequency', fontsize=12)
    
    logger.info(" before tight layout")
    plt.tight_layout(rect=[0, 0, 0.93, 1])  # Leave space on the right for colorbar
    logger.info(" after tight layout")
    plt.savefig(output_dir / 'scatter_plots.png', bbox_inches='tight')
    logger.info("✅ Scatter plots saved to %s", output_dir)
    
    return metrics_list


def create_histogram_plots(predictions, targets, output_dir, evaluation_space: str):
    """Create histograms comparing true and predicted tendencies."""
    logger.info("📈 Creating histogram plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_vars = [
        ('qrtend', 'qrtend_TAU', 'Rain Tendency (qrtend)'),
        ('nctend', 'nctend_TAU', 'Cloud Number Tendency (nctend)'),
        ('nrtend', 'nrtend_TAU', 'Rain Number Tendency (nrtend)'),
        ('qctend', 'qctend_TAU', 'Cloud Water Tendency (qctend)')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('True vs Predicted Distributions', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, (pred_key, target_key, title) in enumerate(plot_vars):
        ax = axes[i]
        y_true = targets[target_key]
        y_pred = predictions[pred_key]

        if len(y_true) == 0 or len(y_pred) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        if evaluation_space == "physical":
            epsilon = 1e-10
            data_true = np.log10(np.abs(y_true) + epsilon)
            data_pred = np.log10(np.abs(y_pred) + epsilon)
            xlabel = 'log₁₀(|Value| + ε)'
            subplot_title = f'{title} (Log Scale)'
        else:
            data_true = y_true
            data_pred = y_pred
            xlabel = 'Log-Space Value'
            subplot_title = f'{title} (Model Log Space)'

        bins = 100
        ax.hist(data_true, bins=bins, alpha=0.6, label='True', color='steelblue', edgecolor='black')
        ax.hist(data_pred, bins=bins, alpha=0.6, label='Predicted', color='darkorange', edgecolor='black')
        ax.set_title(subplot_title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'histograms.png', bbox_inches='tight')
    logger.info(f"✅ Histogram plots saved to {output_dir}")


def log_random_samples(predictions, targets, output_dir, evaluation_space: str, num_samples=1000, seed=42):
    """Log and save random samples of true vs predicted tendencies."""
    logger.info(f"🧪 Sampling {num_samples} random validation entries for inspection...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key_map = [
        ('qrtend', 'qrtend_TAU'),
        ('nctend', 'nctend_TAU'),
        ('nrtend', 'nrtend_TAU'),
        ('qctend', 'qctend_TAU')
    ]

    # Determine available sample count
    sample_counts = [len(predictions[pred_key]) for pred_key, _ in key_map]
    total_samples = min(count for count in sample_counts if count > 0) if sample_counts else 0

    if total_samples == 0:
        logger.warning("No samples available to log random predictions.")
        return

    actual_samples = min(num_samples, total_samples)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_samples, size=actual_samples, replace=False)

    data = {}
    for pred_key, target_key in key_map:
        pred_values = predictions[pred_key][indices]
        true_values = targets[target_key][indices]
        abs_error = np.abs(pred_values - true_values)
        if evaluation_space == "physical":
            denominator = np.abs(true_values)
            epsilon = 1e-14
            safe_denominator = np.where(denominator < epsilon, epsilon, denominator)
            rel_error = abs_error / safe_denominator
        else:
            rel_error = abs_error

        data[f'{pred_key}_pred'] = pred_values
        data[f'{pred_key}_true'] = true_values
        data[f'{pred_key}_abs_error'] = abs_error
        data[f'{pred_key}_rel_error'] = rel_error
        if evaluation_space == "physical":
            data[f'{pred_key}_rel_error_percent'] = rel_error * 100
        

    df_samples = pd.DataFrame(data)
    samples_path = output_dir / 'random_samples.csv'
    df_samples.to_csv(samples_path, index=False)

    logger.info(f"✅ Saved random sample comparison to {samples_path}")
    logger.info("Here are the 10 random samples:")
    logger.info(df_samples.head(10))


def create_constraint_validation_plot(predictions, output_dir, evaluation_space: str):
    """Create constraint validation visualization."""
    logger.info("🔒 Creating constraint validation plots...")
    if evaluation_space == "log":
        logger.warning("Constraint validation is being performed in log space; interpret violations carefully.")
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Physical Constraint Validation', fontsize=16, fontweight='bold')
    
    # 1. qrtend ≥ 0 constraint
    ax1 = axes[0, 0]
    epsilon = 1e-10
    def _maybe_physical(values: np.ndarray) -> np.ndarray:
        if evaluation_space == "physical":
            return values
        sign = np.sign(values)
        abs_val = np.power(10.0, np.abs(values)) - epsilon
        return sign * abs_val

    qrtend_raw = predictions['qrtend']
    qrtend = _maybe_physical(qrtend_raw)
    n_violations = np.sum(qrtend < 0)
    
    ax1.hist(qrtend, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Constraint: ≥0')
    ax1.set_xlabel('qrtend (Rain Tendency)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'qrtend Distribution\nViolations: {n_violations} ({n_violations/len(qrtend)*100:.2f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. nctend ≤ 0 constraint
    ax2 = axes[0, 1]
    nctend_raw = predictions['nctend']
    nctend = _maybe_physical(nctend_raw)
    n_violations = np.sum(nctend > 0)
    
    ax2.hist(nctend, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Constraint: ≤0')
    ax2.set_xlabel('nctend (Cloud Number Tendency)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'nctend Distribution\nViolations: {n_violations} ({n_violations/len(nctend)*100:.2f}%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Mass conservation: qctend = -qrtend
    ax3 = axes[1, 0]
    qctend_raw = predictions['qctend']
    qrtend_raw = predictions['qrtend']
    qctend = _maybe_physical(qctend_raw)
    qrtend = _maybe_physical(qrtend_raw)
    conservation_error = np.abs(qctend + qrtend)
    
    ax3.hist(conservation_error, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('|qctend + qrtend| (Conservation Error)')
    ax3.set_ylabel('Frequency')
    ax3.set_yscale('log')
    ax3.set_title(f'Mass Conservation\nMean Error: {np.mean(conservation_error):.2e}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Classification accuracy
    ax4 = axes[1, 1]
    is_active_pred = predictions['is_active']
    
    ax4.hist(is_active_pred, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Active Probability')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Active/Quiescent Classification')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'constraint_validation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'constraint_validation.pdf', bbox_inches='tight')
    logger.info(f"✅ Constraint validation plots saved to {output_dir}")


def create_training_curves(checkpoint, output_dir):
    """Create training curves if available."""
    logger.info("📈 Creating training curves...")
    
    output_dir = Path(output_dir)
    
    # Try to extract training history
    if 'train_losses' in checkpoint.get('metrics', {}):
        history = checkpoint['metrics']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
        logger.info(f"✅ Training curves saved to {output_dir}")
    else:
        logger.info("⚠️ No training history found in checkpoint")


def print_metrics_summary(metrics_list):
    """print a beautiful metrics summary."""
    logger.info("\n" + "="*70)
    logger.info("📊 EVALUATION METRICS SUMMARY")
    logger.info("="*70)
    
    for metrics in metrics_list:
        logger.info(f"\n🎯 {metrics['name'].upper()}")
        logger.info(f"   R² Score:     {metrics['r2']:.4f}")
        logger.info(f"   RMSE:         {metrics['rmse']:.2e}")
        logger.info(f"   MAE:          {metrics['mae']:.2e}")
        logger.info(f"   L2 Error:     {metrics['l2']:.2e}")
        logger.info(f"   Mean (True):  {metrics['mean_true']:.2e}")
        logger.info(f"   Mean (Pred):  {metrics['mean_pred']:.2e}")
    
    logger.info("\n" + "="*70)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Constraint-Aware Microphysics Emulator")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--num_random_samples',
        type=int,
        default=1000,
        help='Number of random validation samples to log (and save) for inspection'
    )
    parser.add_argument(
        '--max_eval_batches',
        type=int,
        default=None,
        help='Optional cap on the number of validation batches to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--scaler_run_id',
        type=str,
        default=None,
        help='Optional run ID whose scaler artifacts should be reused during evaluation'
    )
    parser.add_argument(
        '--evaluation_space',
        type=str,
        default='physical',
        choices=['physical', 'log'],
        help='Space in which to perform evaluation (physical units or log domain)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("🔍 Starting Model Evaluation")
    logger.info("="*50)
    logger.info(f"📁 Checkpoint: {args.checkpoint}")
    logger.info(f"📋 Config: {args.config}")
    logger.info(f"💾 Output: {args.output_dir}")
    logger.info(f"🖥️  Device: {device}")
    logger.info(f"📐 Evaluation space: {args.evaluation_space}")
    logger.info("="*50)
    
    # Load model and config
    model, config, checkpoint = load_model_and_config(args.checkpoint, args.config)
    
    # Setup data (use validation data for evaluation)e
    data_config = config['data']
    scaler_cache_dir = data_config.get('scaler_cache_dir', './scaler_cache')
    requested_run_id = (
        normalize_run_id(args.scaler_run_id)
        or normalize_run_id(os.environ.get("SCALER_RUN_ID"))
        or infer_run_id_from_path(args.checkpoint)
    )
    if requested_run_id:
        logger.info(f"Re-using scaler artifacts from: {requested_run_id}")

    _train_loader, val_loader, _ = create_optimized_streaming_loaders(
        data_path=data_config['data_path'],
        config=config,
        train_fraction=data_config['train_fraction'],
        batch_size=data_config['batch_size'],
        scaler_cache_dir=scaler_cache_dir,
        run_id_override=requested_run_id
    )
    
    try:
        val_batches = len(val_loader)
    except TypeError:
        val_batches = None
    val_dataset = getattr(val_loader, 'dataset', None)
    val_size = len(val_dataset) if val_dataset is not None and hasattr(val_dataset, '__len__') else None

    if val_batches is not None:
        logger.info(f"📊 Evaluating on {val_batches} validation batches...")
    else:
        logger.info("📊 Evaluating on validation loader (batch count unavailable)...")
    if val_size is not None:
        logger.info(f"   Validation dataset size (samples): {val_size}")

    # Make predictions
    log_frequency = int(config.get('logging', {}).get('log_frequency', 0))
    predictions, targets = make_predictions(
        model,
        val_loader,
        device,
        log_frequency=log_frequency,
        max_batches=args.max_eval_batches
    )

    predictions, targets = restore_outputs_to_eval_space(
        predictions,
        targets,
        val_dataset,
        target_space=args.evaluation_space
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    metrics_list = create_scatter_plots(predictions, targets, output_dir, args.evaluation_space)
    create_histogram_plots(predictions, targets, output_dir, args.evaluation_space)
    create_constraint_validation_plot(predictions, output_dir, args.evaluation_space)
    create_training_curves(checkpoint, output_dir)
    log_random_samples(predictions, targets, output_dir, args.evaluation_space, num_samples=args.num_random_samples)
    
    # print summary
    print_metrics_summary(metrics_list)
    
    logger.info(f"\n🎉 Evaluation completed!")
    logger.info(f"📁 Results saved to: {output_dir.absolute()}")
    logger.info("✅ Check the following files:")
    logger.info(f"   - scatter_plots.png/pdf")
    logger.info(f"   - constraint_validation.png/pdf")
    logger.info(f"   - training_curves.png/pdf")


if __name__ == "__main__":
    main() 