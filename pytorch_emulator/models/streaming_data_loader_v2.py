"""
Optimized Streaming Data Loading for Constraint-Aware Microphysics Emulator

This is an improved version of the streaming data loader that yields BATCHES instead of
individual samples, which should be much more efficient than the current approach.

Key Improvements:
- Yields pre-batched data chunks instead of individual samples
- Vectorized preprocessing of entire chunks
- No sample-by-sample iteration bottleneck
- Still memory-efficient for large datasets
- Compatible with PyTorch DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Iterator, Union
import logging
import random
import pickle
from collections import deque
import warnings

# Suppress specific warnings to reduce log noise
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")
warnings.filterwarnings("ignore", message=".*was reported to be.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedStreamingDataset(IterableDataset):
    """
    Optimized streaming dataset that yields batches instead of individual samples.
    
    Much more efficient than the original sample-by-sample approach.
    """
    
    def __init__(
        self,
        data_path: str,
        input_cols: List[str],
        output_cols: List[str],
        batch_size: int = 1024,
        chunk_size: int = 50000,  # Samples per file chunk
        max_files: Optional[int] = None,
        sample_fraction: float = 1.0,
        active_threshold: float = 1.0e-12,
        random_seed: int = 42,
        split: str = "train",  # "train", "val", or "all"
        train_fraction: float = 0.8,
        scaler_path: Optional[str] = None,
        mode: str = "fit_transform",
        disable_length_estimation: bool = False  # Disable length estimation for large datasets
    ):
        """Initialize optimized streaming dataset."""
        super().__init__()
        
        self.data_path = Path(data_path)
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.max_files = max_files
        self.sample_fraction = sample_fraction
        self.active_threshold = float(active_threshold)  
        self.random_seed = random_seed
        self.split = split
        self.train_fraction = train_fraction
        self.scaler_path = scaler_path
        self.mode = mode
        self.disable_length_estimation = disable_length_estimation
        
        # Initialize random state
        self.rng = np.random.RandomState(random_seed)
        random.seed(random_seed)
        
        # Find parquet files
        self.parquet_files = self._find_parquet_files()
        logger.info(f"Found {len(self.parquet_files)} parquet files")
        
        # Initialize scaler
        self.input_scaler = StandardScaler()
        self._load_or_fit_scaler()
        
        # Calculate file splits for train/val
        self._calculate_file_splits()
        
        # Estimate dataset size
        self.estimated_size = self._estimate_dataset_size()
        logger.info(f"Estimated dataset size: {self.estimated_size:,} samples")
    
    def _find_parquet_files(self) -> List[Path]:
        """Find and sort parquet files."""
        parquet_files = list(self.data_path.glob("*.parquet"))
        parquet_files = sorted(parquet_files)
        
        if self.max_files is not None:
            parquet_files = parquet_files[:self.max_files]
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        return parquet_files
    
    def _calculate_file_splits(self):
        """Calculate train/val file splits."""
        if self.split == "all":
            self.active_files = self.parquet_files
        else:
            n_train_files = int(len(self.parquet_files) * self.train_fraction)
            if self.split == "train":
                self.active_files = self.parquet_files[:n_train_files]
            else:  # val
                self.active_files = self.parquet_files[n_train_files:]
        
        logger.info(f"Using {len(self.active_files)} files for {self.split} split")
    
    def _load_or_fit_scaler(self):
        """Load existing scaler or prepare to fit new one."""
        if self.scaler_path and Path(self.scaler_path).exists() and self.mode != "fit_transform":
            logger.info(f"Loading pre-fitted scaler from {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.input_scaler = pickle.load(f)
        else:
            logger.info("Will fit scaler on data")
    
    def _estimate_dataset_size(self) -> int:
        """Estimate total dataset size using parquet metadata (fast and accurate)."""
        if not self.active_files:
            return 0
        
        # For Perlmutter
        #with open("/people/pate014/nersc_mlmicro/mlmicrophysics/data_from_nersc/processed_data/row_dict.pkl", "rb") as f:
        #        row_dict = pickle.load(f)
        #f.close()

        # For Deception
        with open("/rcfs/projects/pioneercloud/dhruv/processed_data/row_dict.pkl", "rb") as f:
            row_dict = pickle.load(f)
        f.close()
        
        total_row = 0
        for file in self.active_files:
            # strip the path from the file name
            file_name = file.name            
            total_row += row_dict[file_name]

        return total_row




    def _estimate_dataset_size_old(self) -> int:
        """Estimate total dataset size using parquet metadata (fast and accurate)."""
        if not self.active_files:
            return 0
        
        try:
            import pyarrow.parquet as pq
            
            # Strategy 1: If we have few files, just read all metadata (fast)
            if len(self.active_files) <= 1000:
                logger.info(f"Deploying strategy 1.. Huha! We have {len(self.active_files)} files")
                total_rows = 0
                for file in self.active_files:
                    try:
                        parquet_file = pq.ParquetFile(file)
                        total_rows += parquet_file.metadata.num_rows
                    except Exception as e:
                        logger.warning(f"Could not read metadata from {file}: {e}")
                        # Fallback: estimate this file as average of others
                        total_rows += 50000  # Conservative estimate
                
                estimated_total = int(total_rows * self.sample_fraction)
                logger.info(f"Exact row count from metadata: {total_rows:,} -> {estimated_total:,} after sampling")
                return estimated_total
            
            # Strategy 2: For many files, sample more intelligently
            else:
                # Sample files from beginning, middle, and end to avoid bias
                n_files = len(self.active_files)
                sample_indices = set()
                
                # Take samples from different parts of the file list
                sample_indices.update(range(min(10, n_files)))  # First 10
                sample_indices.update(range(n_files//2 - 5, n_files//2 + 5))  # Middle 10
                sample_indices.update(range(max(0, n_files - 10), n_files))  # Last 10
                
                sample_files = [self.active_files[i] for i in sorted(sample_indices)]
                total_rows = 0
                
                for file in sample_files:
                    try:
                        parquet_file = pq.ParquetFile(file)
                        total_rows += parquet_file.metadata.num_rows
                    except Exception as e:
                        logger.warning(f"Could not read metadata from {file}: {e}")
                        total_rows += 50000  # Conservative estimate
                
                # Extrapolate to all files
                avg_rows_per_file = total_rows / len(sample_files)
                estimated_total = int(avg_rows_per_file * len(self.active_files) * self.sample_fraction)
                
                logger.info(f"Sampled {len(sample_files)} files, avg {avg_rows_per_file:.0f} rows/file")
                logger.info(f"Estimated total: {estimated_total:,} samples")
                return estimated_total
                
        except ImportError:
            logger.warning("PyArrow not available, falling back to pandas method")
            
        # Fallback to original method if pyarrow not available
        return self._estimate_dataset_size_fallback()
    
    def _estimate_dataset_size_fallback(self) -> int:
        """Fallback estimation method using pandas (slower but works without pyarrow)."""
        if not self.active_files:
            return 0
        
        # Sample more strategically - not just first files
        n_files = len(self.active_files)
        if n_files <= 30:
            sample_files = self.active_files
        else:
            # Sample from beginning, middle, and end
            sample_files = (
                self.active_files[:10] +  # First 10
                self.active_files[n_files//2-5:n_files//2+5] +  # Middle 10  
                self.active_files[-10:]  # Last 10
            )
        
        total_rows = 0
        for file in sample_files:
            try:
                parquet_file = pd.read_parquet(file)
                total_rows += len(parquet_file)
            except Exception as e:
                logger.warning(f"Error reading {file}: {e}")
                total_rows += 50000  # Conservative estimate
        
        # Extrapolate to all files
        avg_rows_per_file = total_rows / len(sample_files)
        estimated_total = int(avg_rows_per_file * len(self.active_files) * self.sample_fraction)
        
        logger.info(f"Fallback: sampled {len(sample_files)} files, avg {avg_rows_per_file:.0f} rows/file")
        logger.info(f"Estimated total: {estimated_total:,} samples")
        return estimated_total
    
    def _preprocess_chunk_vectorized(self, chunk: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess entire chunk using vectorized operations.
        Much more efficient than sample-by-sample processing.
        """
        try:
            chunk = chunk.copy()
            
            # 1. Log transformations (vectorized)
            log_transform_cols = [
                "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
                "LAMC", "LAMR", "N0R"
            ]
            
            for col in log_transform_cols:
                if col in chunk.columns:
                    epsilon = 1e-10
                    chunk[col] = np.log10(np.maximum(chunk[col], epsilon))
            
            # Log transform output tendencies (vectorized)
            output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
            for col in output_log_cols:
                if col in chunk.columns:
                    epsilon = 1e-10
                    sign = np.sign(chunk[col])
                    abs_val = np.abs(chunk[col]) + epsilon
                    chunk[col] = sign * np.log10(abs_val)
            
            # 2. Create active/quiescent labels (vectorized)
            chunk["is_active"] = (np.abs(chunk["qctend_TAU"]) > self.active_threshold).astype(float)
            
            # 3. Apply sampling if needed
            if self.sample_fraction < 1.0:
                n_samples = int(len(chunk) * self.sample_fraction)
                if n_samples > 0:
                    chunk = chunk.sample(n=n_samples, random_state=self.rng.randint(0, 2**31))
            
            # 4. Ensure required columns exist and remove NaN
            required_cols = self.input_cols + self.output_cols + ["is_active"]
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                logger.warning(f"Missing columns: {missing_cols}")
                return None
            
            chunk = chunk[required_cols].dropna()
            
            if len(chunk) == 0:
                return None
            
            return chunk
            
        except Exception as e:
            logger.warning(f"Error preprocessing chunk: {e}")
            return None
    
    def _process_chunk_to_batches(self, chunk: pd.DataFrame) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Convert a preprocessed chunk into batches of tensors.
        This is where the major efficiency gain comes from.
        """
        if len(chunk) == 0:
            return
        
        # Scale all inputs at once (vectorized)
        input_data = self.input_scaler.transform(chunk[self.input_cols].values)
        
        # Create all target arrays at once (vectorized)
        targets_data = {}
        targets_data['is_active'] = chunk['is_active'].values.reshape(-1, 1)
        for col in self.output_cols:
            if col in chunk.columns:
                targets_data[col] = chunk[col].values.reshape(-1, 1)
        
        # Split into batches
        n_samples = len(chunk)
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            
            # Create batch tensors
            batch_inputs = torch.tensor(
                input_data[start_idx:end_idx], 
                dtype=torch.float32
            )
            
            batch_targets = {}
            for key, data in targets_data.items():
                batch_targets[key] = torch.tensor(
                    data[start_idx:end_idx], 
                    dtype=torch.float32
                )
            
            yield batch_inputs, batch_targets
    
    def fit_scaler(self, n_samples_for_fitting: int = 100000):
        """Fit scaler on a sample of the data."""
        if self.mode == "transform":
            return
        
        logger.info(f"Fitting scaler on {n_samples_for_fitting} samples...")
        
        samples_collected = 0
        all_inputs = []
        
        for file_path in self.active_files:
            if samples_collected >= n_samples_for_fitting:
                break
            
            try:
                # Read file in chunks
                try:
                    chunk_iter = pd.read_parquet(file_path, chunksize=self.chunk_size)
                except TypeError:
                    full_data = pd.read_parquet(file_path)
                    chunk_iter = [full_data[i:i+self.chunk_size] for i in range(0, len(full_data), self.chunk_size)]
                
                for chunk in chunk_iter:
                    if samples_collected >= n_samples_for_fitting:
                        break
                    
                    processed_chunk = self._preprocess_chunk_vectorized(chunk)
                    
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        inputs = processed_chunk[self.input_cols].values
                        all_inputs.append(inputs)
                        samples_collected += len(inputs)
                        
                        logger.info(f"Collected {samples_collected}/{n_samples_for_fitting} samples")
            
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        # Fit scaler on collected data
        if all_inputs:
            combined_inputs = np.vstack(all_inputs)
            # Subsample if we have too much data
            if len(combined_inputs) > n_samples_for_fitting:
                indices = np.random.choice(len(combined_inputs), n_samples_for_fitting, replace=False)
                combined_inputs = combined_inputs[indices]
            
            self.input_scaler.fit(combined_inputs)
            logger.info(f"Fitted scaler on {len(combined_inputs)} samples")
            
            # Save scaler
            if self.scaler_path:
                Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.input_scaler, f)
                logger.info(f"Saved scaler to {self.scaler_path}")
    
    def _batch_generator(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Generate batches from files using vectorized processing."""
        
        # Shuffle files
        active_files = self.active_files.copy()
        random.shuffle(active_files)
        
        for file_path in active_files:
            try:
                logger.debug(f"Processing file: {file_path.name}")
                
                # Read file in chunks
                try:
                    chunk_iter = pd.read_parquet(file_path, chunksize=self.chunk_size)
                except TypeError:
                    full_data = pd.read_parquet(file_path)
                    chunk_iter = [full_data[i:i+self.chunk_size] for i in range(0, len(full_data), self.chunk_size)]
                
                for chunk in chunk_iter:
                    # Preprocess entire chunk at once
                    processed_chunk = self._preprocess_chunk_vectorized(chunk)
                    
                    if processed_chunk is None or len(processed_chunk) == 0:
                        continue
                    
                    # Convert chunk to batches and yield
                    yield from self._process_chunk_to_batches(processed_chunk)
            
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Iterate over dataset, yielding batches."""
        return self._batch_generator()
    
    def __len__(self) -> int:
        """Return estimated number of batches using ceiling division."""
        import math
        
        # For very large datasets, disable length estimation to avoid warnings
        # This is controlled by a config parameter
        #if hasattr(self, 'disable_length_estimation') and self.disable_length_estimation:
        #    return 1  # Return minimal value to avoid issues
            
        return max(1, math.ceil(self.estimated_size / self.batch_size))


def create_optimized_streaming_loaders(
    data_path: str,
    config: Dict,
    train_fraction: float = 0.8,
    batch_size: int = 1024,
    scaler_cache_dir: str = "./scaler_cache"
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Create optimized streaming data loaders that yield batches directly.
    
    Note: These return DataLoaders that yield batches, so batch_size in DataLoader
    constructor should be set to None or 1 to avoid double-batching.
    """
    # Extract configuration
    data_config = config.get('data', {})
    input_cols = data_config.get('input_cols', [])
    output_cols = data_config.get('output_cols', [])
    
    # Use config batch size
    actual_batch_size = data_config.get('batch_size', batch_size)
    
    # Check if length estimation should be disabled for large datasets
    disable_length = data_config.get('disable_length_estimation', False)
    if data_config.get('max_files', 0) > 200 or actual_batch_size > 100000:
        disable_length = True
        logger.info("Large dataset detected - disabling length estimation to avoid warnings")
    
    # Create scaler cache directory
    scaler_cache_path = Path(scaler_cache_dir) / "input_scaler_optimized.pkl"
    scaler_cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create training dataset
    logger.info("Creating optimized training dataset...")
    train_dataset = OptimizedStreamingDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        batch_size=actual_batch_size,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=data_config.get('active_threshold', 1e-12),
        random_seed=data_config.get('random_seed', 42),
        split="train",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="fit_transform",
        disable_length_estimation=disable_length
    )
    
    # Fit scaler
    train_dataset.fit_scaler(n_samples_for_fitting=data_config.get('scaler_fit_samples', 100000))
    
    # Create validation dataset
    logger.info("Creating optimized validation dataset...")
    val_dataset = OptimizedStreamingDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        batch_size=actual_batch_size,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=data_config.get('active_threshold', 1e-12),
        random_seed=data_config.get('random_seed', 42),
        split="val",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="transform",
        disable_length_estimation=disable_length
    )
    
    # Create data loaders (batch_size=None since we're yielding pre-batched data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,  # Important: dataset yields batches already
        num_workers=0,    # Keep at 0 for stability
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,  # Important: dataset yields batches already
        num_workers=0,    # Keep at 0 for stability
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created optimized streaming loaders:")
    logger.info(f"  Train: ~{train_dataset.estimated_size:,} samples, ~{len(train_dataset)} batches")
    logger.info(f"  Val: ~{val_dataset.estimated_size:,} samples, ~{len(val_dataset)} batches")
    logger.info(f"  Batch size: {actual_batch_size}")
    
    return train_loader, val_loader, train_dataset.input_scaler


if __name__ == "__main__":
    """Test optimized streaming data loader."""
    print("🧪 Testing Optimized Streaming Data Loader...")
    
    # Disable CUDA for CPU testing
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA for CPU testing
    print("🔧 Running on CPU (CUDA disabled for testing)")
    
    # Create test configuration
    test_config = {
        'data': {
            'input_cols': ['QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in', 
                          'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR'],
            'output_cols': ['qctend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qrtend_TAU'],
            'chunk_size': 10000,
            'max_files': 3,
            'subsample': 0.1,
            'batch_size': 32,  # Smaller batch size for CPU testing
            'random_seed': 42
        }
    }
    
    # Test with real data path
    data_path = "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/"
    
    if Path(data_path).exists():
        try:
            train_loader, val_loader, scaler = create_optimized_streaming_loaders(
                data_path=data_path,
                config=test_config
            )
            
            print("✅ Optimized streaming loaders created successfully!")
            
            # Test training loader
            print("🔄 Testing training loader...")
            batch_count = 0
            for i, (x_batch, targets_batch) in enumerate(train_loader):
                print(f"  Batch {i+1}: input shape {x_batch.shape}")
                for key, tensor in targets_batch.items():
                    print(f"    {key}: {tensor.shape}")
                
                batch_count += 1
                if i >= 2:  # Test first 3 batches
                    break
            
            print(f"✅ Successfully processed {batch_count} batches!")
            print("✅ Optimized streaming loader test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Test data path {data_path} not found. Skipping test.")
        print("✅ Optimized streaming loader implementation completed!") 