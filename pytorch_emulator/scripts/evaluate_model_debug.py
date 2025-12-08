#!/usr/bin/env python3
"""
Evaluate Constraint-Aware Microphysics Emulator (Debug Version)

This version prints raw normalized input/output values to inspect what the NN sees.

Usage:
    python evaluate_model_debug.py --checkpoint outputs/best_checkpoint.pth --config configs/quick_test_5epochs.yml
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
from streaming_data_loader_v2 import create_optimized_streaming_loaders
from trainer import ConstraintAwareTrainer

# Columns that undergo log-space operations in dataset preprocessing
LOG_TENDENCY_COLUMNS = {"qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"}
LOG_EPSILON = 1e-10


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


def save_fortran_test_data(inputs_array, preds_concat, targets_concat, indices, 
                           output_dir, input_cols=None, pred_keys=None):
    """
    Save debug data in a format easily readable by Fortran for validation.
    
    Creates files:
    - fortran_test_inputs.txt: Input features (one sample per line)
    - fortran_test_outputs.txt: Model predictions (one sample per line)
    - fortran_test_targets.txt: Ground truth targets (one sample per line)
    - fortran_test_info.txt: Metadata about dimensions and column names
    
    Format is simple space-delimited ASCII that Fortran can read with:
        read(unit, *) array(:)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_samples = len(indices)
    num_features = inputs_array.shape[1]
    
    # Define the output order for tendencies (matching TorchScript export_model.py)
    # Order: qrtend, nctend, nrtend, qctend (NO is_active - it's not in TorchScript model)
    # See export_model.py OUTPUT_TENSOR_ORDER for reference
    output_keys = ['qrtend', 'nctend', 'nrtend', 'qctend']
    target_keys_ordered = ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU']
    
    # Save inputs
    inputs_file = output_dir / 'fortran_test_inputs.txt'
    with open(inputs_file, 'w') as f:
        f.write(f"# {num_samples} {num_features}\n")  # Header: num_samples num_features
        for idx in indices:
            values = inputs_array[idx]
            line = ' '.join(f'{v:.15e}' for v in values)
            f.write(line + '\n')
    logger.info(f"📁 Saved input features to: {inputs_file}")
    
    # Save model predictions
    outputs_file = output_dir / 'fortran_test_outputs.txt'
    num_outputs = len(output_keys)
    with open(outputs_file, 'w') as f:
        f.write(f"# {num_samples} {num_outputs}\n")  # Header: num_samples num_outputs
        for idx in indices:
            values = []
            for key in output_keys:
                if key in preds_concat:
                    val = preds_concat[key][idx]
                    if isinstance(val, np.ndarray):
                        val = val.flatten()[0] if val.size == 1 else float(val)
                    values.append(val)
                else:
                    values.append(0.0)  # Placeholder if key missing
            line = ' '.join(f'{v:.15e}' for v in values)
            f.write(line + '\n')
    logger.info(f"📁 Saved model predictions to: {outputs_file}")
    
    # Save ground truth targets
    targets_file = output_dir / 'fortran_test_targets.txt'
    with open(targets_file, 'w') as f:
        f.write(f"# {num_samples} {num_outputs}\n")
        for idx in indices:
            values = []
            for key in target_keys_ordered:
                if key in targets_concat:
                    val = targets_concat[key][idx]
                    if isinstance(val, np.ndarray):
                        val = val.flatten()[0] if val.size == 1 else float(val)
                    values.append(val)
                else:
                    values.append(0.0)
            line = ' '.join(f'{v:.15e}' for v in values)
            f.write(line + '\n')
    logger.info(f"📁 Saved ground truth targets to: {targets_file}")
    
    # Save metadata/info file
    info_file = output_dir / 'fortran_test_info.txt'
    with open(info_file, 'w') as f:
        f.write("# Fortran Test Data Info\n")
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"num_input_features: {num_features}\n")
        f.write(f"num_outputs: {num_outputs}\n")
        f.write(f"\n# Input feature names (in order):\n")
        if input_cols:
            for i, col in enumerate(input_cols):
                f.write(f"  input[{i}]: {col}\n")
        else:
            for i in range(num_features):
                f.write(f"  input[{i}]: feature_{i}\n")
        f.write(f"\n# Output names (in order):\n")
        for i, key in enumerate(output_keys):
            f.write(f"  output[{i}]: {key}\n")
        f.write(f"\n# File format:\n")
        f.write("# First line is comment with dimensions: # num_samples num_features\n")
        f.write("# Each subsequent line is one sample, space-delimited values\n")
        f.write("# Values are in scientific notation (e.g., 1.234567890123456e-05)\n")
        f.write(f"\n# Sample indices from validation set: {list(indices)}\n")
    logger.info(f"📁 Saved metadata to: {info_file}")
    
    # Also save as binary for more precise Fortran loading (single precision float32)
    binary_inputs_file = output_dir / 'fortran_test_inputs.bin'
    binary_outputs_file = output_dir / 'fortran_test_outputs.bin'
    
    # Prepare arrays for binary output
    selected_inputs = inputs_array[indices].astype(np.float32)
    selected_outputs = np.zeros((num_samples, num_outputs), dtype=np.float32)
    for i, idx in enumerate(indices):
        for j, key in enumerate(output_keys):
            if key in preds_concat:
                val = preds_concat[key][idx]
                if isinstance(val, np.ndarray):
                    val = val.flatten()[0] if val.size == 1 else float(val)
                selected_outputs[i, j] = val
    
    # Save as raw binary (row-major, float32)
    selected_inputs.tofile(binary_inputs_file)
    selected_outputs.tofile(binary_outputs_file)
    logger.info(f"📁 Saved binary inputs to: {binary_inputs_file}")
    logger.info(f"📁 Saved binary outputs to: {binary_outputs_file}")
    
    return inputs_file, outputs_file, targets_file, info_file


def print_raw_datapoints(model, data_loader, device='cpu', num_samples=5, seed=42, 
                         save_dir=None):
    """
    Print random datapoints showing raw normalized inputs and outputs.
    
    This shows the exact values as they are fed to the NN and output by it,
    without any post-processing or inverse transformations.
    
    If save_dir is provided, also saves data for Fortran validation.
    """
    logger.info("\n" + "="*80)
    logger.info("🔬 RAW NORMALIZED DATAPOINTS (as seen by the neural network)")
    logger.info("="*80)
    
    model.to(device)
    model.eval()
    
    # Collect some batches to sample from
    all_inputs = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx >= 10:  # Collect from first 10 batches
                break
            
            inputs = inputs.to(device)
            targets_device = {k: v.to(device) for k, v in targets.items()}
            
            # Get model predictions
            predictions = model(inputs)
            
            all_inputs.append(inputs.cpu().numpy())
            all_targets.append({k: v.cpu().numpy() for k, v in targets_device.items()})
            all_predictions.append({k: v.cpu().numpy() for k, v in predictions.items()})
    
    # Concatenate all collected data
    inputs_array = np.concatenate(all_inputs, axis=0)
    
    # Concatenate targets and predictions
    target_keys = list(all_targets[0].keys())
    pred_keys = list(all_predictions[0].keys())
    
    targets_concat = {k: np.concatenate([t[k] for t in all_targets], axis=0) for k in target_keys}
    preds_concat = {k: np.concatenate([p[k] for p in all_predictions], axis=0) for k in pred_keys}
    
    total_samples = inputs_array.shape[0]
    logger.info(f"Collected {total_samples} samples from first 10 batches")
    
    # Randomly select samples
    rng = np.random.default_rng(seed)
    indices = rng.choice(total_samples, size=min(num_samples, total_samples), replace=False)
    
    # Get input feature names if available from dataset
    dataset = getattr(data_loader, 'dataset', None)
    input_cols = getattr(dataset, 'input_cols', None)
    output_cols = getattr(dataset, 'output_cols', None)
    
    if input_cols:
        logger.info(f"\nInput features ({len(input_cols)} total): {input_cols}")
    if output_cols:
        logger.info(f"Output columns: {output_cols}")
    
    logger.info(f"\nTarget keys in data: {target_keys}")
    logger.info(f"Prediction keys from model: {pred_keys}")
    
    for sample_num, idx in enumerate(indices):
        logger.info("\n" + "-"*80)
        logger.info(f"📌 SAMPLE {sample_num + 1} (index {idx})")
        logger.info("-"*80)
        
        # Print input features
        input_vec = inputs_array[idx]
        logger.info(f"\n🔹 INPUT FEATURES (normalized, shape={input_vec.shape}):")
        
        if input_cols and len(input_cols) == len(input_vec):
            for i, (col_name, val) in enumerate(zip(input_cols, input_vec)):
                logger.info(f"   [{i:2d}] {col_name:20s}: {val:+.8e}")
        else:
            # Print without column names
            for i, val in enumerate(input_vec):
                logger.info(f"   [{i:2d}]: {val:+.8e}")
        
        # Print target values (ground truth)
        logger.info(f"\n🔹 TARGET VALUES (normalized ground truth):")
        for key in target_keys:
            val = targets_concat[key][idx]
            if isinstance(val, np.ndarray):
                val = val.flatten()[0] if val.size == 1 else val
            logger.info(f"   {key:20s}: {val:+.8e}" if np.isscalar(val) or val.size == 1 
                       else f"   {key:20s}: {val}")
        
        # Print model predictions
        logger.info(f"\n🔹 MODEL PREDICTIONS (raw output, no post-processing):")
        for key in pred_keys:
            val = preds_concat[key][idx]
            if isinstance(val, np.ndarray):
                val = val.flatten()[0] if val.size == 1 else val
            logger.info(f"   {key:20s}: {val:+.8e}" if np.isscalar(val) or val.size == 1 
                       else f"   {key:20s}: {val}")
        
        # Compute errors for tendency outputs
        logger.info(f"\n🔹 ERRORS (prediction - target):")
        tendency_keys = ['qrtend', 'nctend', 'nrtend', 'qctend']
        for pred_key in tendency_keys:
            target_key = f"{pred_key}_TAU"
            if pred_key in preds_concat and target_key in targets_concat:
                pred_val = preds_concat[pred_key][idx]
                target_val = targets_concat[target_key][idx]
                if isinstance(pred_val, np.ndarray):
                    pred_val = pred_val.flatten()[0] if pred_val.size == 1 else pred_val
                if isinstance(target_val, np.ndarray):
                    target_val = target_val.flatten()[0] if target_val.size == 1 else target_val
                if np.isscalar(pred_val) and np.isscalar(target_val):
                    error = pred_val - target_val
                    logger.info(f"   {pred_key:20s}: {error:+.8e}")
    
    # Save data for Fortran validation if requested
    if save_dir:
        logger.info("\n" + "="*80)
        logger.info("💾 SAVING DATA FOR FORTRAN VALIDATION")
        logger.info("="*80)
        save_fortran_test_data(
            inputs_array, preds_concat, targets_concat, indices,
            save_dir, input_cols, pred_keys
        )
    
    logger.info("\n" + "="*80)
    logger.info("✅ Raw datapoint inspection complete")
    logger.info("="*80 + "\n")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Constraint-Aware Microphysics Emulator (Debug)")
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
        '--num_samples',
        type=int,
        default=5,
        help='Number of random samples to print (default: 5)'
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
        '--seed',
        type=int,
        default=42,
        help='Random seed for sample selection (default: 42)'
    )
    # Output directory for saving Fortran test data
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save Fortran test data files (if not provided, data is only printed)'
    )
    parser.add_argument(
        '--num_random_samples',
        type=int,
        default=None,
        help='Ignored (for compatibility with evaluate_model.py)'
    )
    parser.add_argument(
        '--max_eval_batches',
        type=int,
        default=None,
        help='Ignored (for compatibility with evaluate_model.py)'
    )
    parser.add_argument(
        '--evaluation_space',
        type=str,
        default='physical',
        choices=['physical', 'log'],
        help='Ignored (for compatibility with evaluate_model.py)'
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info("🔍 Starting Debug Evaluation")
    logger.info("="*50)
    logger.info(f"📁 Checkpoint: {args.checkpoint}")
    logger.info(f"📋 Config: {args.config}")
    logger.info(f"🖥️  Device: {device}")
    logger.info(f"🎲 Random seed: {args.seed}")
    logger.info(f"📊 Samples to print: {args.num_samples}")
    if args.output_dir:
        logger.info(f"💾 Fortran test data will be saved to: {args.output_dir}")
    logger.info("="*50)
    
    # Load model and config
    model, config, checkpoint = load_model_and_config(args.checkpoint, args.config)
    
    # Setup data (use validation data for evaluation)
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
    
    # Print raw datapoints and optionally save for Fortran
    print_raw_datapoints(
        model, 
        val_loader, 
        device=device, 
        num_samples=args.num_samples,
        seed=args.seed,
        save_dir=args.output_dir
    )
    
    logger.info("🎉 Debug evaluation completed!")


if __name__ == "__main__":
    main()

