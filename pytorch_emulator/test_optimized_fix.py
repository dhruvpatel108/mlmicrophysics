#!/usr/bin/env python3
"""
Quick test to verify optimized streaming loader fixes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from models.streaming_data_loader_v2 import create_optimized_streaming_loaders
import yaml
import torch

def test_optimized_loader():
    """Test the fixed optimized streaming loader."""
    
    print("🧪 Testing FIXED Optimized Streaming Loader...")
    
    # Load config
    with open('configs/optimized_streaming_test.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded. Active threshold: {config['data']['active_threshold']} (type: {type(config['data']['active_threshold'])})")
    
    try:
        # Create data loaders
        train_loader, val_loader, scaler = create_optimized_streaming_loaders(
            data_path=config['data']['data_path'],
            config=config,
            scaler_cache_dir="./scaler_cache_test_fix"
        )
        
        print("✅ Data loaders created successfully!")
        
        # Test a few batches
        print("🔄 Testing first few batches...")
        batch_count = 0
        total_samples = 0
        
        for i, (inputs, targets) in enumerate(train_loader):
            print(f"  Batch {i+1}: {inputs.shape}")
            batch_count += 1
            total_samples += len(inputs)
            
            if i >= 2:  # Test 3 batches
                break
        
        print(f"✅ Success! Processed {batch_count} batches, {total_samples} samples")
        print("🚀 The optimized streaming loader is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_optimized_loader()
    print(f"\n{'🎉 SUCCESS' if success else '💥 FAILED'}") 