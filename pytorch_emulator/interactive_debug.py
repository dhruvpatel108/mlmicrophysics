"""
Interactive Debug Script for Loss Analysis

This script runs a few batches interactively to analyze loss components
and understand the behavior of the model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import yaml
import logging
from typing import Dict
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Import our modules
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders
from models.physics_emulator import ConstraintAwareEmulator
from models.losses import ConstraintAwareLoss

def analyze_batch_data(inputs, targets, device='cuda'):
    """Analyze the input and target data to understand distributions."""
    print("\n" + "="*60)
    print("📊 BATCH DATA ANALYSIS")
    print("="*60)
    
    # Input analysis
    print(f"Input shape: {inputs.shape}")
    print(f"Input stats: min={inputs.min():.4f}, max={inputs.max():.4f},
           mean={inputs.mean():.4f}, std={inputs.std():.4f}")
    
    # Check for NaN or infinite values
    if torch.isnan(inputs).any():
        print("⚠️  WARNING: NaN values found in inputs!")
    if torch.isinf(inputs).any():
        print("⚠️  WARNING: Infinite values found in inputs!")
    
    # Target analysis
    print(f"\nTarget keys: {list(targets.keys())}")
    for key, value in targets.items():
        if torch.is_tensor(value):
            print(f"{key}: shape={value.shape}, min={value.min():.4f}, 
                  max={value.max():.4f}, mean={value.mean():.4f}")
            if torch.isnan(value).any():
                print(f"⚠️  WARNING: NaN values found in {key}!")
    
    # Active samples analysis
    active_mask = targets['is_active'].bool().squeeze()
    active_count = active_mask.sum().item()
    total_count = len(active_mask)
    active_fraction = active_count / total_count
    
    print(f"\nActive samples: {active_count}/{total_count} ({active_fraction:.2%})")
    
    if active_count > 0:
        print("\nActive sample target statistics:")
        for key in ['qrtend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qctend_TAU']:
            if key in targets:
                active_values = targets[key][active_mask]
                print(f"{key}: min={active_values.min():.4e}, max={active_values.max():.4e}, 
                      mean={active_values.mean():.4e}")


def analyze_model_outputs(predictions, targets):
    """Analyze model predictions vs targets."""
    print("\n" + "="*60)
    print("🤖 MODEL OUTPUT ANALYSIS")
    print("="*60)
    
    active_mask = targets['is_active'].bool().squeeze()
    active_count = active_mask.sum().item()
    
    # Classification analysis
    is_active_probs = predictions['is_active'].squeeze()
    print(f"Classification probabilities: min={is_active_probs.min():.4f}, 
          max={is_active_probs.max():.4f}, mean={is_active_probs.mean():.4f}")
    
    predicted_active = (is_active_probs > 0.5).sum().item()
    actual_active = active_count
    print(f"Predicted active: {predicted_active}, Actual active: {actual_active}")
    
    # Regression analysis
    print("\nRegression outputs vs targets:")
    for pred_key, target_key in [('qrtend', 'qrtend_TAU'), ('nctend', 'nctend_TAU'), ('nrtend', 'nrtend_TAU')]:
        pred_vals = predictions[pred_key].squeeze()
        target_vals = targets[target_key].squeeze()
        
        print(f"\n{pred_key}:")
        print(f"  Predictions: min={pred_vals.min():.4e}, max={pred_vals.max():.4e}, mean={pred_vals.mean():.4e}")
        print(f"  Targets:     min={target_vals.min():.4e}, max={target_vals.max():.4e}, mean={target_vals.mean():.4e}")
        
        if active_count > 0:
            pred_active = pred_vals[active_mask]
            target_active = target_vals[active_mask]
            mse = ((pred_active - target_active) ** 2).mean()
            print(f"  Active MSE: {mse:.4e}")


def analyze_loss_components(loss_dict, alpha=0.3):
    """Analyze individual loss components."""
    print("\n" + "="*60)
    print("📉 LOSS COMPONENT ANALYSIS")
    print("="*60)
    
    total_loss = loss_dict['total_loss'].item()
    cls_loss = loss_dict['classification_loss'].item()
    reg_loss = loss_dict['regression_loss'].item()
    active_samples = loss_dict['active_samples']
    
    print(f"Total loss: {total_loss:.6f}")
    print(f"Classification loss: {cls_loss:.6f} (weight: {alpha:.2f}, contribution: {alpha * cls_loss:.6f})")
    print(f"Regression loss: {reg_loss:.6f} (weight: {1-alpha:.2f}, contribution: {(1-alpha) * reg_loss:.6f})")
    print(f"Active samples: {active_samples}")
    
    # Check for problematic values
    if cls_loss > 10:
        print("⚠️  WARNING: Classification loss is very high!")
    if reg_loss > 10:
        print("⚠️  WARNING: Regression loss is very high!")
    if cls_loss < 1e-6:
        print("⚠️  WARNING: Classification loss is very low (model may have collapsed)!")


def run_debug_analysis():
    """Run interactive debugging analysis."""
    
    print("🧪 STARTING INTERACTIVE DEBUG ANALYSIS")
    print("="*70)
    
    # Load configuration
    config_path = "configs/debug_overfitting_test.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders (small sample)
    print("\n📊 Creating data loaders...")
    try:
        data_config = config['data']
        train_loader, val_loader, scaler = create_optimized_streaming_loaders(
            data_path=data_config['data_path'],
            config=config,
            train_fraction=data_config.get('train_fraction', 0.8),
            batch_size=data_config.get('batch_size', 64),
            scaler_cache_dir=data_config.get('scaler_cache_dir', './scaler_cache')
        )
        print("✅ Data loaders created successfully")
        print(f"📊 Scaler fitted: {scaler is not None}")
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create model using config parameters
    print("\n🤖 Creating model...")
    model_config = config['model']
    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dim=model_config['head_dim'],
        dropout=model_config['dropout']
    ).to(device)
    
    print(f"Model parameters: {model.get_parameter_count():,}")
    
    # Create loss function using config parameters
    loss_config = config['loss']
    alpha = model_config.get('alpha', 0.3)
    loss_fn = ConstraintAwareLoss(
        alpha=alpha,
        huber_delta=loss_config.get('huber_delta', 1.0),
        conservation_weight=loss_config.get('conservation_weight', 0.1),
        use_masking=model_config.get('use_masking', True)
    ).to(device)
    
    # Analyze first few batches
    model.eval()
    print("\n🔍 Analyzing first 10 batches...")
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= 10:  # Only analyze first 3 batches
                break
                
            print(f"\n{'='*20} BATCH {batch_idx + 1} {'='*20}")
            
            # Move to device
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Analyze input data
            analyze_batch_data(inputs, targets, device)
            
            # Forward pass
            predictions = model(inputs)
            
            # Analyze model outputs
            analyze_model_outputs(predictions, targets)
            
            # Compute loss
            loss_dict = loss_fn(predictions, targets)
            
            # Analyze loss components
            analyze_loss_components(loss_dict, alpha)
            
            print("\n" + "="*50)
    
    print("\n✅ Interactive debug analysis completed!")
    print("\n💡 DIAGNOSTIC QUESTIONS TO ASK:")
    print("1. Are active samples well-distributed or too sparse?")
    print("2. Are target values in reasonable ranges?")
    print("3. Are model outputs in reasonable ranges?")
    print("4. Is one loss component dominating the other?")
    print("5. Are there NaN or infinite values anywhere?")


if __name__ == "__main__":
    try:
        run_debug_analysis()
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()