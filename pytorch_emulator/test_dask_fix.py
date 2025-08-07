#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify Dask loader fixes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from models.dask_data_loader import create_dask_data_loaders
import yaml

def test_dask_loader():
    """Test the fixed Dask loader."""
    
    print("Testing FIXED Dask Loader...")
    
    # Load config
    with open('configs/dask_test.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded. Active threshold: {config['data']['active_threshold']} (type: {type(config['data']['active_threshold'])})")
    
    try:
        # Create data loaders with smaller settings for faster testing
        config['data']['max_files'] = 3  # Use only 3 files for quick test
        config['data']['subsample'] = 0.05  # Use only 5% of data
        
        train_loader, val_loader, scaler = create_dask_data_loaders(
            data_path=config['data']['data_path'],
            config=config,
            n_dask_workers=2,  # Use fewer workers
            scaler_cache_dir="./scaler_cache_dask_test"
        )
        
        print("Data loaders created successfully!")
        print(f"Training dataset size: {len(train_loader.dataset)}")
        print(f"Validation dataset size: {len(val_loader.dataset)}")
        
        # Test a few batches
        print("Testing first few batches...")
        batch_count = 0
        total_samples = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            print(f"  Batch {i+1}: {inputs.shape}")
            batch_count += 1
            total_samples += len(inputs)
            
            if i >= 2:  # Test 3 batches
                break
        
        print(f"Success! Processed {batch_count} batches, {total_samples} samples")
        print("The Dask loader is working!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dask_loader()
    print(f"\n{'SUCCESS' if success else 'FAILED'}") 