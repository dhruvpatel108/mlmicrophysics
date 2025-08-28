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

import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, List
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

# Set style for beautiful plots
#plt.style.use('seaborn-v0_8')
#sns.set_palette("husl")


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
    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dim=model_config['head_dim'],
        dropout=model_config['dropout']
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✅ Model loaded from epoch {checkpoint['epoch']}")
    logger.info(f"   Parameters: {model.get_parameter_count():,}")
    
    return model, config, checkpoint


def make_predictions(model, data_loader, device='cpu'):
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
            if batch_idx > 1:
                break
            
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Model predictions
            predictions = model(inputs)
            
            # Store predictions
            for key in all_predictions:
                all_predictions[key].append(predictions[key].cpu().numpy())
            
            # Store targets
            for key in all_targets:
                if key in targets:
                    all_targets[key].append(targets[key].cpu().numpy())
            
            log_freq = int(config.get('logging', {}).get('log_frequency', 10))
            if log_freq > 0 and batch_idx % log_freq == 0:
                logger.info(f"  Processed {batch_idx}/{len(data_loader)} batches")
    
    # Concatenate all batches
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0).flatten()
    
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0).flatten()
    
    logger.info(f"✅ Predictions completed: {len(all_predictions['qrtend'])} samples")
    
    return all_predictions, all_targets


def calculate_metrics(y_true, y_pred, name):
    """Calculate regression metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    return {
        'name': name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_true': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_true': np.std(y_true),
        'std_pred': np.std(y_pred)
    }


def create_scatter_plots(predictions, targets, output_dir):
    """Create beautiful scatter plots with R² values."""
    logger.info("📊 Creating scatter plots...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the variables to plot
    plot_vars = [
        ('qrtend', 'qrtend_TAU', 'Rain Tendency (qrtend)', 'log'),
        ('nctend', 'nctend_TAU', 'Cloud Number Tendency (nctend)', 'log'),
        ('nrtend', 'nrtend_TAU', 'Rain Number Tendency (nrtend)', 'log'),
        ('qctend', 'qctend_TAU', 'Cloud Water Tendency (qctend)', 'log')
    ]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Constraint-Aware Emulator: True vs Predicted', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    metrics_list = []
    
    for i, (pred_key, target_key, title, scale) in enumerate(plot_vars):
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
        
        # Create scatter plot
        if scale == 'log':
            # For log scale, handle negative values
            mask_pos_true = y_true > 0
            mask_pos_pred = y_pred > 0
            mask_both_pos = mask_pos_true & mask_pos_pred
            
            if np.sum(mask_both_pos) > 0:
                ax.scatter(
                    y_true[mask_both_pos], 
                    y_pred[mask_both_pos], 
                    alpha=0.5, s=10, color='blue', label='Positive'
                )
            
            # Handle negative values separately
            mask_neg = ~mask_both_pos
            if np.sum(mask_neg) > 0:
                ax.scatter(
                    y_true[mask_neg], 
                    y_pred[mask_neg], 
                    alpha=0.5, s=10, color='red', label='Negative/Zero'
                )
        else:
            ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect')
        
        # Formatting
        ax.set_xlabel(f'True {title}', fontsize=12)
        ax.set_ylabel(f'Predicted {title}', fontsize=12)
        ax.set_title(f'{title}\nR² = {metrics["r2"]:.4f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add constraint info for relevant variables
        if pred_key == 'qrtend':
            constraint_violations = np.sum(y_pred < 0)
            ax.text(0.05, 0.95, f'Constraint: ≥0\nViolations: {constraint_violations}', 
                   transform=ax.transAxes, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        elif pred_key == 'nctend':
            constraint_violations = np.sum(y_pred > 0)
            ax.text(0.05, 0.95, f'Constraint: ≤0\nViolations: {constraint_violations}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    logger.info(f" before tight layout")
    plt.tight_layout()
    logger.info(f" after tight layout")
    plt.savefig(output_dir / 'scatter_plots.png', bbox_inches='tight')
    #plt.savefig(output_dir / 'scatter_plots.pdf', bbox_inches='tight')
    logger.info(f"✅ Scatter plots saved to {output_dir}")
    
    return metrics_list


def create_constraint_validation_plot(predictions, output_dir):
    """Create constraint validation visualization."""
    logger.info("🔒 Creating constraint validation plots...")
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Physical Constraint Validation', fontsize=16, fontweight='bold')
    
    # 1. qrtend ≥ 0 constraint
    ax1 = axes[0, 0]
    qrtend = predictions['qrtend']
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
    nctend = predictions['nctend']
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
    qctend = predictions['qctend']
    qrtend = predictions['qrtend']
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
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for evaluation'
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
    logger.info("="*50)
    
    # Load model and config
    model, config, checkpoint = load_model_and_config(args.checkpoint, args.config)
    
    # Setup data (use validation data for evaluation)e
    data_config = config['data']
    train_loader, val_loader, dataset = create_optimized_streaming_loaders(
        data_path=data_config['data_path'],
        config=config,
        train_fraction=data_config['train_fraction'],
        batch_size=data_config['batch_size'],
        scaler_cache_dir=data_config.get('scaler_cache_dir', './scaler_cache')
    )
    
    logger.info(f"📊 Evaluating on {len(val_loader)} validation batches...")
    logger.info(f"Size of the val_loader: {len(val_loader)} and the number of batches in the val_loader: {len(val_loader.dataset)}")



    # Make predictions
    predictions, targets = make_predictions(model, val_loader, device)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    metrics_list = create_scatter_plots(predictions, targets, output_dir)
    create_constraint_validation_plot(predictions, output_dir)
    create_training_curves(checkpoint, output_dir)
    
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