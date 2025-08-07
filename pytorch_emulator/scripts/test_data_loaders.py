#!/usr/bin/env python3
"""
Comprehensive Data Loader Performance Testing

This script tests and compares the performance of different data loading approaches:
1. Current Streaming Data Loader (sample-by-sample)
2. Dask-based Data Loader (distributed processing)
3. Optimized Streaming Data Loader (batch-based)

Usage:
    python test_data_loaders.py --config configs/dask_test.yml --loader dask
    python test_data_loaders.py --config configs/optimized_streaming_test.yml --loader optimized
    python test_data_loaders.py --config configs/multi_gpu_test.yml --loader current
"""

import argparse
import time
import yaml
import logging
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import different data loaders
from models.streaming_data_loader import create_streaming_data_loaders
from models.dask_data_loader import create_dask_data_loaders
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_data_loader_performance(
    loader_type: str,
    config: dict,
    max_batches: int = 50,
    max_time: float = 300.0  # 5 minutes max
) -> dict:
    """
    Test data loader performance by measuring:
    - Data loading time
    - Batch iteration speed
    - Memory usage patterns
    - First batch latency
    """
    
    logger.info(f"\n🧪 Testing {loader_type.upper()} Data Loader")
    logger.info("="*60)
    
    results = {
        'loader_type': loader_type,
        'config': config['data'],
        'setup_time': 0.0,
        'first_batch_time': 0.0,
        'avg_batch_time': 0.0,
        'total_batches': 0,
        'total_samples': 0,
        'throughput_samples_per_sec': 0.0,
        'success': False,
        'error': None
    }
    
    try:
        # Measure setup time
        setup_start = time.time()
        
        if loader_type == "current":
            train_loader, val_loader, scaler = create_streaming_data_loaders(
                data_path=config['data']['data_path'],
                config=config,
                batch_size=config['data']['batch_size'],
                scaler_cache_dir="./scaler_cache_current"
            )
        elif loader_type == "dask":
            train_loader, val_loader, scaler = create_dask_data_loaders(
                data_path=config['data']['data_path'],
                config=config,
                batch_size=config['data']['batch_size'],
                scaler_cache_dir="./scaler_cache_dask",
                n_dask_workers=2  # Use 2 Dask workers for testing
            )
        elif loader_type == "optimized":
            train_loader, val_loader, scaler = create_optimized_streaming_loaders(
                data_path=config['data']['data_path'],
                config=config,
                batch_size=config['data']['batch_size'],
                scaler_cache_dir="./scaler_cache_optimized"
            )
        else:
            raise ValueError(f"Unknown loader type: {loader_type}")
        
        setup_time = time.time() - setup_start
        results['setup_time'] = setup_time
        
        logger.info(f"✅ Setup completed in {setup_time:.2f} seconds")
        logger.info(f"📊 Starting batch iteration test...")
        
        # Test batch iteration performance
        batch_times = []
        total_samples = 0
        start_time = time.time()
        first_batch_time = None
        batch_idx = -1  # Initialize to handle case where no batches are processed
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_start = time.time()
            
            # Record first batch time (includes any lazy loading)
            if batch_idx == 0:
                first_batch_time = time.time() - setup_start
                results['first_batch_time'] = first_batch_time
                logger.info(f"🚀 First batch ready in {first_batch_time:.2f} seconds")
            
            # Simulate minimal processing (just move to device)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = {k: v.cuda() for k, v in targets.items()}
            
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            batch_size = len(inputs)
            total_samples += batch_size
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                avg_time = sum(batch_times) / len(batch_times)
                throughput = batch_size / batch_time
                logger.info(f"  Batch {batch_idx:3d}: {batch_size:4d} samples, "
                          f"{batch_time:.4f}s, {throughput:.1f} samples/s")
            
            # Stop if we've reached limits
            if batch_idx >= max_batches:
                logger.info(f"Reached max batches limit ({max_batches})")
                break
            
            if time.time() - start_time > max_time:
                logger.info(f"Reached max time limit ({max_time}s)")
                break
        
        total_time = time.time() - start_time
        
        # Calculate final metrics (handle case where no batches were processed)
        results['total_batches'] = max(0, batch_idx + 1)
        results['total_samples'] = total_samples
        results['avg_batch_time'] = sum(batch_times) / len(batch_times) if batch_times else 0
        results['throughput_samples_per_sec'] = total_samples / total_time if total_time > 0 else 0
        results['success'] = True
        
        # Log final results
        logger.info(f"\n📈 PERFORMANCE RESULTS:")
        logger.info(f"  Setup time:        {results['setup_time']:.2f} seconds")
        logger.info(f"  First batch time:  {results['first_batch_time']:.2f} seconds")
        logger.info(f"  Average batch time: {results['avg_batch_time']:.4f} seconds")
        logger.info(f"  Total batches:     {results['total_batches']}")
        logger.info(f"  Total samples:     {results['total_samples']:,}")
        logger.info(f"  Throughput:        {results['throughput_samples_per_sec']:.1f} samples/sec")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def compare_loaders(results_list):
    """Compare results from multiple loader tests."""
    logger.info(f"\n🏆 PERFORMANCE COMPARISON")
    logger.info("="*60)
    
    # Sort by throughput (best first)
    successful_results = [r for r in results_list if r['success']]
    if not successful_results:
        logger.error("❌ No successful tests to compare!")
        return
    
    successful_results.sort(key=lambda x: x['throughput_samples_per_sec'], reverse=True)
    
    logger.info(f"{'Rank':<4} {'Loader':<12} {'Setup(s)':<10} {'1st Batch(s)':<12} {'Avg Batch(ms)':<14} {'Throughput(samp/s)':<18}")
    logger.info("-" * 80)
    
    for i, result in enumerate(successful_results, 1):
        loader = result['loader_type']
        setup = f"{result['setup_time']:.2f}"
        first_batch = f"{result['first_batch_time']:.2f}"
        avg_batch = f"{result['avg_batch_time']*1000:.2f}"
        throughput = f"{result['throughput_samples_per_sec']:.1f}"
        
        logger.info(f"{i:<4} {loader:<12} {setup:<10} {first_batch:<12} {avg_batch:<14} {throughput:<18}")
    
    # Calculate performance improvements
    if len(successful_results) > 1:
        best = successful_results[0]
        logger.info(f"\n🚀 IMPROVEMENT ANALYSIS:")
        for result in successful_results[1:]:
            speedup = best['throughput_samples_per_sec'] / result['throughput_samples_per_sec']
            logger.info(f"  {best['loader_type']} is {speedup:.2f}x faster than {result['loader_type']}")


def main():
    parser = argparse.ArgumentParser(description="Test data loader performance")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--loader", required=True, 
                       choices=['current', 'dask', 'optimized', 'all'],
                       help="Data loader type to test")
    parser.add_argument("--max-batches", type=int, default=50,
                       help="Maximum number of batches to test")
    parser.add_argument("--max-time", type=float, default=300.0,
                       help="Maximum test time in seconds")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"📋 Loaded config from {args.config}")
    logger.info(f"🗂️  Data path: {config['data']['data_path']}")
    logger.info(f"📁 Max files: {config['data']['max_files']}")
    logger.info(f"📈 Subsample: {config['data']['subsample']}")
    logger.info(f"📊 Batch size: {config['data']['batch_size']}")

    
    # Run tests
    results = []
    
    if args.loader == 'all':
        # Test all loaders
        for loader_type in ['current', 'optimized', 'dask']:
            logger.info(f"\n" + "="*80)
            logger.info(f"🧪 TESTING {loader_type.upper()} DATA LOADER")
            logger.info("="*80)
            
            result = test_data_loader_performance(
                loader_type=loader_type,
                config=config,
                max_batches=args.max_batches,
                max_time=args.max_time
            )
            results.append(result)
            
            # Small delay between tests
            time.sleep(2)
        
        # Compare all results
        compare_loaders(results)
    else:
        # Test single loader
        result = test_data_loader_performance(
            loader_type=args.loader,
            config=config,
            max_batches=args.max_batches,
            max_time=args.max_time
        )
        results.append(result)
    
    # Save results to file
    import json
    results_file = f"data_loader_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to {results_file}")
    logger.info(f"🏁 Testing complete!")


if __name__ == "__main__":
    main() 