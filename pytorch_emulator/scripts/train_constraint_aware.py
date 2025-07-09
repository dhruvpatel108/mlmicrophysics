#!/usr/bin/env python3
"""
Train Constraint-Aware Microphysics Emulator

Complete training script for the constraint-aware microphysics emulator.
Loads configuration, sets up model/data/training, and runs the full pipeline.

Usage:
    python train_constraint_aware.py --config configs/constraint_aware_base.yml
"""

import sys
import os
import argparse
import yaml
from pathlib import Path
import torch

# Add model and training paths
sys.path.append('models')
sys.path.append('training')

# Import our components
from physics_emulator import ConstraintAwareEmulator
from losses import ConstraintAwareLoss, create_constraint_aware_loss
from data_loader import create_data_loaders
from trainer import ConstraintAwareTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model(config: dict) -> ConstraintAwareEmulator:
    """Setup the constraint-aware emulator model."""
    model_config = config['model']
    
    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dim=model_config['head_dim'],
        dropout=model_config['dropout']
    )
    
    print(f"✅ Model created with {model.get_parameter_count():,} parameters")
    print(f"   Architecture: {model_config['input_dim']} → {model_config['shared_dims']} → heads")
    
    return model


def setup_loss_function(config: dict) -> ConstraintAwareLoss:
    """Setup the constraint-aware loss function."""
    loss_fn = create_constraint_aware_loss(config)
    
    weights = loss_fn.get_loss_weights()
    print(f"✅ Loss function created")
    print(f"   Classification weight: {weights['classification_weight']:.2f}")
    print(f"   Regression weight: {weights['regression_weight']:.2f}")
    print(f"   Conservation weight: {weights['conservation_weight']:.2f}")
    
    return loss_fn


def setup_data(config: dict):
    """Setup data loaders."""
    data_config = config['data']
    
    print(f"📊 Loading data from: {data_config['data_path']}")
    print(f"   Max files: {data_config['max_files']}")
    print(f"   Subsample: {data_config['subsample']:.1%}")
    print(f"   Batch size: {data_config['batch_size']}")
    
    train_loader, val_loader, dataset = create_data_loaders(
        data_path=data_config['data_path'],
        config=config,
        train_fraction=data_config['train_fraction'],
        batch_size=data_config['batch_size']
    )
    
    # Print data info
    info = dataset.get_data_info()
    print(f"✅ Data loaded successfully")
    print(f"   Total samples: {info['total_samples']:,}")
    print(f"   Active samples: {info['active_samples']:,} ({info['active_fraction']:.1%})")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Constraint-Aware Microphysics Emulator")
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/constraint_aware_base.yml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    print("🚀 Starting Constraint-Aware Microphysics Emulator Training")
    print("=" * 70)
    
    # Load configuration
    config = load_config(args.config)
    print(f"📋 Configuration loaded from: {args.config}")
    
    # Set random seeds for reproducibility
    experiment_config = config.get('experiment', {})
    seed = experiment_config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"🎲 Random seed set to: {seed}")
    
    # Setup components
    print("\n🔧 Setting up model and training components...")
    
    # 1. Model
    model = setup_model(config)
    
    # 2. Loss function
    loss_fn = setup_loss_function(config)
    
    # 3. Data
    train_loader, val_loader, dataset = setup_data(config)
    
    # 4. Trainer
    trainer = ConstraintAwareTrainer(model, loss_fn, config, device=args.device)
    
    print(f"\n🏃‍♂️ Starting training on device: {trainer.device}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Early stopping patience: {config['training']['early_stopping_patience']}")
    print("=" * 70)
    
    # Run training
    try:
        history = trainer.train(train_loader, val_loader)
        
        print("\n" + "=" * 70)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉")
        print(f"   Total epochs: {history['total_epochs']}")
        print(f"   Best validation loss: {history['best_val_loss']:.6f}")
        print(f"   Final training loss: {history['train_losses'][-1]:.6f}")
        print("   Checkpoints saved to:", trainer.output_dir)
        print("=" * 70)
        
        # Test final model constraints
        print("\n🔍 Final model validation...")
        model.eval()
        with torch.no_grad():
            x_test = torch.randn(100, config['model']['input_dim']).to(trainer.device)
            preds = model(x_test)
            
            qr_ok = torch.all(preds['qrtend'] >= 0)
            nc_ok = torch.all(preds['nctend'] <= 0)
            mass_ok = torch.allclose(preds['qctend'], -preds['qrtend'], atol=1e-6)
            
            print(f"✅ Final constraint verification:")
            print(f"   qrtend ≥ 0: {qr_ok}")
            print(f"   nctend ≤ 0: {nc_ok}")
            print(f"   Mass conservation: {mass_ok}")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 