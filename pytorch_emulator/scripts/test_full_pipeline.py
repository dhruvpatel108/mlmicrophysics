"""
Complete Pipeline Integration Test

Tests the full constraint-aware microphysics emulator pipeline:
1. Model creation and validation
2. Loss function testing
3. Data loading and preprocessing  
4. Training pipeline execution
5. Constraint verification
"""

import sys
import os
sys.path.append('models')
sys.path.append('training')

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict

# Import our components
from physics_emulator import ConstraintAwareEmulator
from losses import ConstraintAwareLoss
from data_loader import MicrophysicsDataset, create_data_loaders
from trainer import ConstraintAwareTrainer


def create_sample_data(output_path: str, n_samples: int = 1000) -> str:
    """Create sample parquet data for testing."""
    print("🔧 Creating sample data...")
    
    np.random.seed(42)
    
    # Create realistic sample data matching E3SM structure
    data = {
        # Input features (log-normal distributions for microphysics variables)
        'QC_TAU_in': np.random.lognormal(-8, 2, n_samples),      # Cloud water
        'QR_TAU_in': np.random.lognormal(-10, 3, n_samples),     # Rain water
        'NC_TAU_in': np.random.lognormal(8, 1, n_samples),       # Cloud number
        'NR_TAU_in': np.random.lognormal(5, 2, n_samples),       # Rain number
        'PGAM': np.random.normal(3, 0.5, n_samples),             # Gamma parameter
        'LAMC': np.random.lognormal(5, 1, n_samples),            # Cloud lambda
        'LAMR': np.random.lognormal(3, 1, n_samples),            # Rain lambda
        'N0R': np.random.lognormal(6, 2, n_samples),             # Rain intercept
        'RHO_CLUBB': np.random.normal(1.2, 0.1, n_samples),     # Air density
        'CLOUD': np.random.uniform(0, 1, n_samples),             # Cloud fraction
        'FREQR': np.random.uniform(0, 1, n_samples),             # Rain frequency
        
        # Output tendencies with physical relationships
        'qrtend_TAU': np.abs(np.random.normal(0, 1e-6, n_samples)),    # ≥ 0
        'nctend_TAU': -np.abs(np.random.normal(0, 1e-5, n_samples)),   # ≤ 0  
        'nrtend_TAU': np.random.normal(0, 1e-5, n_samples),            # Any value
    }
    
    # Mass conservation: qctend = -qrtend
    data['qctend_TAU'] = -data['qrtend_TAU']
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    df.to_parquet(f"{output_path}/sample_data.parquet")
    
    print(f"✅ Created sample data: {len(df)} samples")
    return output_path


def test_model_creation():
    """Test 1: Model creation and basic functionality."""
    print("\n🧠 Test 1: Model Creation")
    
    model = ConstraintAwareEmulator(
        input_dim=11,
        shared_dims=[64, 32, 16],
        head_dim=8,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 11)
    
    with torch.no_grad():
        predictions = model(x)
    
    # Verify outputs
    assert 'is_active' in predictions
    assert 'qrtend' in predictions
    assert 'nctend' in predictions 
    assert 'nrtend' in predictions
    assert 'qctend' in predictions
    
    # Verify shapes
    for key, tensor in predictions.items():
        assert tensor.shape == (batch_size, 1), f"{key} has wrong shape: {tensor.shape}"
    
    # Verify constraints
    assert torch.all(predictions['qrtend'] >= 0), "qrtend constraint violated"
    assert torch.all(predictions['nctend'] <= 0), "nctend constraint violated"
    assert torch.allclose(predictions['qctend'], -predictions['qrtend']), "Mass conservation violated"
    
    print(f"✅ Model created with {model.get_parameter_count():,} parameters")
    print(f"✅ All physical constraints satisfied")
    return model


def test_loss_function(model):
    """Test 2: Loss function with model predictions."""
    print("\n⚖️ Test 2: Loss Function")
    
    loss_fn = ConstraintAwareLoss(
        alpha=0.3,
        huber_delta=1.0,
        conservation_weight=0.1
    )
    
    # Create sample batch
    batch_size = 16
    x = torch.randn(batch_size, 11)
    
    # Model predictions
    with torch.no_grad():
        predictions = model(x)
    
    # Create targets
    targets = {
        'is_active': torch.randint(0, 2, (batch_size, 1)).float(),
        'qrtend': torch.abs(torch.randn(batch_size, 1)) * 1e-6,
        'nctend': -torch.abs(torch.randn(batch_size, 1)) * 1e-5,
        'nrtend': torch.randn(batch_size, 1) * 1e-5
    }
    
    # Compute loss
    loss_dict = loss_fn(predictions, targets)
    
    # Verify loss components
    assert 'total_loss' in loss_dict
    assert 'classification_loss' in loss_dict
    assert 'regression_loss' in loss_dict
    assert 'conservation_loss' in loss_dict
    assert 'active_samples' in loss_dict
    
    # Verify loss is reasonable
    assert loss_dict['total_loss'].item() > 0, "Total loss should be positive"
    assert loss_dict['total_loss'].item() < 100, "Total loss seems too large"
    
    print(f"✅ Loss components computed successfully")
    print(f"  Total: {loss_dict['total_loss'].item():.6f}")
    print(f"  Classification: {loss_dict['classification_loss'].item():.6f}")
    print(f"  Regression: {loss_dict['regression_loss'].item():.6f}")
    print(f"  Conservation: {loss_dict['conservation_loss'].item():.6f}")
    
    return loss_fn


def test_data_loading(data_path):
    """Test 3: Data loading and preprocessing."""
    print("\n📊 Test 3: Data Loading")
    
    input_cols = [
        'QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in', 
        'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR'
    ]
    output_cols = ['qctend_TAU', 'nctend_TAU', 'nrtend_TAU']
    
    # Create dataset
    dataset = MicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        max_files=1,
        sample_fraction=1.0
    )
    
    # Test dataset
    assert len(dataset) > 0, "Dataset is empty"
    
    # Test single sample
    x, targets = dataset[0]
    assert x.shape == (11,), f"Input shape wrong: {x.shape}"
    assert 'is_active' in targets
    assert 'qctend_TAU' in targets
    
    # Test data loader
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    x_batch, targets_batch = next(iter(loader))
    
    assert x_batch.shape[1] == 11, "Batch input dimension wrong"
    
    # Get data info
    info = dataset.get_data_info()
    
    print(f"✅ Dataset loaded successfully")
    print(f"  Total samples: {info['total_samples']}")
    print(f"  Active samples: {info['active_samples']}")
    print(f"  Quiescent samples: {info['quiescent_samples']}")
    print(f"  Active fraction: {info['active_fraction']:.3f}")
    
    return dataset


def test_training_pipeline(model, loss_fn, dataset):
    """Test 4: Complete training pipeline."""
    print("\n🏃‍♂️ Test 4: Training Pipeline")
    
    # Create data loaders
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )
    
    # Create config
    config = {
        'training': {
            'epochs': 2,  # Just 2 epochs for testing
            'learning_rate': 0.001,
            'optimizer': 'adam',
            'early_stopping_patience': 10
        },
        'data': {
            'out_path': '/tmp/test_training_output'
        }
    }
    
    # Create trainer
    trainer = ConstraintAwareTrainer(model, loss_fn, config, device='cpu')
    
    # Run training
    print("Starting training...")
    history = trainer.train(train_loader, val_loader)
    
    # Verify training completed
    assert len(history['train_losses']) > 0, "No training history recorded"
    assert len(history['val_losses']) > 0, "No validation history recorded"
    assert history['total_epochs'] > 0, "No epochs completed"
    
    print(f"✅ Training completed successfully")
    print(f"  Epochs: {history['total_epochs']}")
    print(f"  Final train loss: {history['train_losses'][-1]:.6f}")
    print(f"  Final val loss: {history['val_losses'][-1]:.6f}")
    print(f"  Best val loss: {history['best_val_loss']:.6f}")
    
    return trainer, history


def test_model_constraints_after_training(trainer):
    """Test 5: Verify constraints still hold after training."""
    print("\n🔒 Test 5: Post-Training Constraint Verification")
    
    model = trainer.model
    model.eval()
    
    # Test with random data
    with torch.no_grad():
        x_test = torch.randn(100, 11)
        predictions = model(x_test)
    
    # Verify all constraints
    qrtend_positive = torch.all(predictions['qrtend'] >= 0)
    nctend_negative = torch.all(predictions['nctend'] <= 0)
    mass_conservation = torch.allclose(
        predictions['qctend'], 
        -predictions['qrtend'], 
        atol=1e-6
    )
    
    assert qrtend_positive, "qrtend constraint violated after training"
    assert nctend_negative, "nctend constraint violated after training"
    assert mass_conservation, "Mass conservation violated after training"
    
    print(f"✅ All constraints maintained after training")
    print(f"  qrtend ≥ 0: {qrtend_positive}")
    print(f"  nctend ≤ 0: {nctend_negative}")
    print(f"  Mass conservation: {mass_conservation}")


def run_full_pipeline_test():
    """Run the complete integration test."""
    print("🚀 Starting Full Pipeline Integration Test")
    print("=" * 60)
    
    # Setup temporary directory
    temp_dir = tempfile.mkdtemp()
    data_path = None
    
    try:
        # Create sample data
        data_path = create_sample_data(f"{temp_dir}/data", n_samples=500)
        
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Loss function
        loss_fn = test_loss_function(model)
        
        # Test 3: Data loading
        dataset = test_data_loading(data_path)
        
        # Test 4: Training pipeline
        trainer, history = test_training_pipeline(model, loss_fn, dataset)
        
        # Test 5: Post-training constraints
        test_model_constraints_after_training(trainer)
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ Model: Constraint-aware architecture working")
        print("✅ Loss: Combined classification + regression + conservation")  
        print("✅ Data: Preprocessing and loading working")
        print("✅ Training: Full pipeline with checkpointing")
        print("✅ Constraints: All physical constraints maintained")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if data_path and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # Cleanup training outputs
        training_output = Path('/tmp/test_training_output')
        if training_output.exists():
            shutil.rmtree(training_output)


if __name__ == "__main__":
    success = run_full_pipeline_test()
    sys.exit(0 if success else 1) 