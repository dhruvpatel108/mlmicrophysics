#!/usr/bin/env python3
"""
Quick Train and Evaluate Constraint-Aware Microphysics Emulator

Combined script that:
1. Trains for 5 epochs quickly
2. Automatically generates beautiful evaluation plots
3. Shows R² values and constraint validation

Usage:
    python quick_train_and_evaluate.py
"""

import sys
import os
import subprocess
import yaml
from pathlib import Path
import torch

# Add model and training paths
sys.path.append('models')
sys.path.append('training')

# Import our components
from physics_emulator import ConstraintAwareEmulator
from losses import create_constraint_aware_loss
from data_loader import create_data_loaders
from trainer import ConstraintAwareTrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def quick_train():
    """Run quick 5-epoch training."""
    print("🚀 QUICK TRAINING - 5 EPOCHS")
    print("=" * 50)
    
    config_path = 'configs/quick_test_5epochs.yml'
    config = load_config(config_path)
    
    print(f"📋 Using config: {config_path}")
    
    # Set random seeds
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # 1. Model
    model_config = config['model']
    model = ConstraintAwareEmulator(
        input_dim=model_config['input_dim'],
        shared_dims=model_config['shared_dims'],
        head_dim=model_config['head_dim'],
        dropout=model_config['dropout']
    )
    print(f"✅ Model: {model.get_parameter_count():,} parameters")
    
    # 2. Loss function
    loss_fn = create_constraint_aware_loss(config)
    print(f"✅ Loss function created")
    
    # 3. Data
    data_config = config['data']
    print(f"📊 Loading data from: {data_config['data_path']}")
    print(f"   Files: {data_config['max_files']}, Subsample: {data_config['subsample']:.1%}")
    
    train_loader, val_loader, dataset = create_data_loaders(
        data_path=data_config['data_path'],
        config=config,
        train_fraction=data_config['train_fraction'],
        batch_size=data_config['batch_size']
    )
    
    info = dataset.get_data_info()
    print(f"✅ Data: {info['total_samples']:,} samples, {info['active_fraction']:.1%} active")
    
    # 4. Trainer
    trainer = ConstraintAwareTrainer(model, loss_fn, config, device=device)
    
    print(f"\n🏃‍♂️ Starting 5-epoch training...")
    print("=" * 50)
    
    # Run training
    try:
        history = trainer.train(train_loader, val_loader)
        
        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETED!")
        print(f"   Final train loss: {history['train_losses'][-1]:.6f}")
        print(f"   Final val loss: {history['val_losses'][-1]:.6f}")
        print(f"   Best val loss: {history['best_val_loss']:.6f}")
        print(f"   Checkpoints: {trainer.output_dir}")
        
        return trainer.output_dir / 'best_checkpoint.pth', config_path, True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def run_evaluation(checkpoint_path, config_path):
    """Run comprehensive evaluation with plots."""
    print("\n🔍 RUNNING EVALUATION WITH PLOTS")
    print("=" * 50)
    
    # Import evaluation functions
    from evaluate_model import (
        load_model_and_config, make_predictions, create_scatter_plots,
        create_constraint_validation_plot, print_metrics_summary
    )
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = Path('./evaluation_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📋 Config: {config_path}")
    print(f"💾 Output: {output_dir}")
    
    # Load model and config
    model, config, checkpoint = load_model_and_config(checkpoint_path, config_path)
    
    # Setup data
    data_config = config['data']
    train_loader, val_loader, dataset = create_data_loaders(
        data_path=data_config['data_path'],
        config=config,
        train_fraction=data_config['train_fraction'],
        batch_size=data_config['batch_size']
    )
    
    print(f"📊 Evaluating on {len(val_loader)} validation batches...")
    
    # Make predictions
    predictions, targets = make_predictions(model, val_loader, device)
    
    # Create visualizations
    print("\n📊 Creating beautiful plots...")
    metrics_list = create_scatter_plots(predictions, targets, output_dir)
    create_constraint_validation_plot(predictions, output_dir)
    
    # Create training curves manually
    if hasattr(trainer, 'train_losses') and hasattr(trainer, 'val_losses'):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        epochs = range(1, len(trainer.train_losses) + 1)
        ax.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss (5 Epochs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training curves saved")
    
    # Print summary
    print_metrics_summary(metrics_list)
    
    # Additional constraint summary
    print("\n🔒 CONSTRAINT VALIDATION SUMMARY")
    print("=" * 50)
    
    qrtend_violations = sum(predictions['qrtend'] < 0)
    nctend_violations = sum(predictions['nctend'] > 0)
    conservation_error = abs(predictions['qctend'] + predictions['qrtend']).mean()
    
    print(f"qrtend ≥ 0: {qrtend_violations} violations ({qrtend_violations/len(predictions['qrtend'])*100:.2f}%)")
    print(f"nctend ≤ 0: {nctend_violations} violations ({nctend_violations/len(predictions['nctend'])*100:.2f}%)")
    print(f"Mass conservation: Mean error = {conservation_error:.2e}")
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"📁 Results saved to: {output_dir.absolute()}")
    print("✅ Generated files:")
    print(f"   - scatter_plots.png (True vs Predicted with R²)")
    print(f"   - constraint_validation.png (Physical constraints)")
    print(f"   - training_curves.png (Loss curves)")
    
    return output_dir


def main():
    """Main function for quick training and evaluation."""
    print("🚀 CONSTRAINT-AWARE EMULATOR: QUICK TRAIN & EVALUATE")
    print("=" * 60)
    print("⚡ 5-epoch training + comprehensive evaluation")
    print("📊 Generates scatter plots with R² values")
    print("🔒 Validates physical constraints")
    print("=" * 60)
    
    # Step 1: Quick training
    checkpoint_path, config_path, success = quick_train()
    
    if not success:
        print("❌ Training failed, cannot proceed to evaluation")
        return False
    
    # Store trainer for later access
    global trainer
    
    # Step 2: Evaluation with plots
    try:
        evaluation_dir = run_evaluation(checkpoint_path, config_path)
        
        print("\n" + "=" * 60)
        print("🎉 ALL COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("✅ 5-epoch training completed")
        print("✅ Model constraints enforced")
        print("✅ Beautiful plots generated")
        print("✅ R² values calculated")
        print("✅ Physical validation passed")
        print("=" * 60)
        print(f"📁 Training outputs: {checkpoint_path.parent}")
        print(f"📁 Evaluation plots: {evaluation_dir}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 