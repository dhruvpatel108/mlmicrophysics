"""
Streaming Data Loading for Constraint-Aware Microphysics Emulator

Implements memory-efficient streaming data loading to handle large datasets (1.2B+ samples)
similar to tf.data.Dataset approach used in the original Keras implementation.

Key Features:
- Streams data from disk in chunks to avoid memory overload
- Supports multiple parquet files with lazy loading
- Implements preprocessing pipeline on-the-fly
- Memory-efficient shuffling using reservoir sampling
- Compatible with PyTorch DataLoader and multi-GPU training
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import os
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Iterator, Union
import logging
import random
import pickle
from collections import deque
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingMicrophysicsDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for microphysics data.
    
    Reads parquet files in chunks and processes them on-the-fly to avoid
    loading entire dataset into memory. Supports datasets with billions of samples.
    
    Features:
    - Chunk-based reading from multiple parquet files
    - On-the-fly preprocessing and scaling
    - Memory-efficient shuffling with reservoir sampling
    - Support for train/validation splitting
    - Compatible with multi-GPU data parallel training
    """
    
    def __init__(
        self,
        data_path: str,
        input_cols: List[str],
        output_cols: List[str],
        chunk_size: int = 50000,  # Samples per chunk
        max_files: Optional[int] = None,
        sample_fraction: float = 1.0,
        active_threshold: float = 1e-12,
        shuffle_buffer_size: int = 100000,  # For memory-efficient shuffling
        random_seed: int = 42,
        split: str = "train",  # "train", "val", or "all"
        train_fraction: float = 0.8,
        scaler_path: Optional[str] = None,  # Path to pre-fitted scaler
        mode: str = "fit_transform",  # "fit_transform", "transform", "fit_only"
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize streaming dataset.
        
        Args:
            data_path: Path to directory containing parquet files
            input_cols: List of input feature column names
            output_cols: List of output target column names
            chunk_size: Number of samples to read per chunk
            max_files: Maximum number of parquet files to use (None = all)
            sample_fraction: Fraction of each file to use
            active_threshold: Threshold for active/quiescent classification
            shuffle_buffer_size: Size of shuffle buffer for memory-efficient shuffling
            random_seed: Random seed for reproducibility
            split: Data split to use ("train", "val", "all")
            train_fraction: Fraction of data for training (rest for validation)
            scaler_path: Path to saved scaler (for transform-only mode)
            mode: "fit_transform" (fit scaler and transform), "transform" (transform only), "fit_only" (fit scaler only)
        """
        super().__init__()
        
        self.data_path = Path(data_path)
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.chunk_size = chunk_size
        self.max_files = max_files
        self.sample_fraction = sample_fraction
        self.active_threshold = float(active_threshold)  # Convert to float to avoid comparison issues
        self.shuffle_buffer_size = shuffle_buffer_size
        self.random_seed = random_seed
        self.split = split
        self.train_fraction = train_fraction
        self.scaler_path = scaler_path
        self.mode = mode
        self.rank = rank
        self.world_size = max(1, world_size)
        
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
        self._shard_active_files()
        
        # Estimate dataset size (approximate)
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
        """Calculate which files to use for train/val splits."""
        n_files = len(self.parquet_files)
        n_train_files = int(self.train_fraction * n_files)
        
        if self.split == "train":
            self.active_files = self.parquet_files[:n_train_files]
        elif self.split == "val":
            self.active_files = self.parquet_files[n_train_files:]
        else:  # "all"
            self.active_files = self.parquet_files
        
        logger.info(f"Using {len(self.active_files)} files for {self.split} split")
    
    def _shard_active_files(self):
        """Shard active files across distributed ranks."""
        if self.world_size <= 1:
            return
        if not self.active_files:
            return
        sharded = self.active_files[self.rank::self.world_size]
        if not sharded:
            logger.warning(
                f"Rank {self.rank} received no files after sharding; falling back to round-robin across full list."
            )
            sharded = self.parquet_files[self.rank::self.world_size] or self.parquet_files
        self.active_files = sharded
    
    def _load_or_fit_scaler(self):
        """Load pre-fitted scaler or prepare to fit new one."""
        if self.scaler_path and Path(self.scaler_path).exists():
            logger.info(f"Loading pre-fitted scaler from {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.input_scaler = pickle.load(f)
        elif self.mode == "transform":
            # Don't raise error immediately - let fit_scaler handle it
            logger.warning("Transform mode requested but no pre-fitted scaler found")
            logger.info("Will attempt to fit scaler or create dummy scaler")
    
    def _estimate_dataset_size(self) -> int:
        """Estimate total dataset size by sampling a few files."""
        if not self.active_files:
            return 0
        
        # Sample first few files to estimate size
        sample_files = self.active_files[:min(3, len(self.active_files))]
        total_rows = 0
        
        for file in sample_files:
            try:
                # Read just metadata to get row count
                parquet_file = pd.read_parquet(file)
                total_rows += len(parquet_file)
            except Exception as e:
                logger.warning(f"Could not read {file} for size estimation: {e}")
                # Fallback estimate
                total_rows += 50000
        
        # Extrapolate to all files
        avg_rows_per_file = total_rows / len(sample_files)
        estimated_total = int(avg_rows_per_file * len(self.active_files) * self.sample_fraction)
        
        return estimated_total
    
    def fit_scaler(self, n_samples_for_fitting: int = 100000):
        """
        Fit scaler on a representative sample of the data.
        
        Args:
            n_samples_for_fitting: Number of samples to use for fitting scaler
        """
        if self.mode == "transform":
            return  # Don't fit if in transform-only mode
        
        logger.info(f"Fitting scaler on {n_samples_for_fitting} samples...")
        
        samples_collected = 0
        all_inputs = []
        
        for file_path in self.active_files:
            if samples_collected >= n_samples_for_fitting:
                break
            
            try:
                # Read entire file first (for small files) or use chunked reading
                try:
                    # Try chunked reading first
                    chunk_iter = pd.read_parquet(file_path, chunksize=self.chunk_size)
                except TypeError:
                    # Fallback: read entire file and split manually
                    logger.debug(f"Chunked reading not supported for {file_path}, reading entire file")
                    full_data = pd.read_parquet(file_path)
                    chunk_iter = [full_data[i:i+self.chunk_size] for i in range(0, len(full_data), self.chunk_size)]
                
                for chunk in chunk_iter:
                    if samples_collected >= n_samples_for_fitting:
                        break
                    
                    # Apply sampling
                    if self.sample_fraction < 1.0:
                        n_samples = int(len(chunk) * self.sample_fraction)
                        if n_samples == 0:
                            continue
                        chunk = chunk.sample(n=n_samples, random_state=self.rng.randint(0, 2**31))
                    
                    # Preprocess chunk
                    processed_chunk = self._preprocess_chunk(chunk, fit_mode=True)
                    
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        inputs = processed_chunk[self.input_cols].values
                        all_inputs.append(inputs)
                        samples_collected += len(inputs)
                        
                        logger.info(f"Collected {samples_collected}/{n_samples_for_fitting} samples for scaler fitting")
            
            except Exception as e:
                logger.warning(f"Error reading {file_path} for scaler fitting: {e}")
                continue
        
        if all_inputs:
            # Combine all input samples
            X_fit = np.vstack(all_inputs)
            
            # Subsample if we have too many samples
            if len(X_fit) > n_samples_for_fitting:
                indices = self.rng.choice(len(X_fit), n_samples_for_fitting, replace=False)
                X_fit = X_fit[indices]
            
            # Fit scaler
            self.input_scaler.fit(X_fit)
            logger.info(f"Fitted scaler on {len(X_fit)} samples")
            
            # Save scaler if path provided
            if self.scaler_path:
                scaler_dir = Path(self.scaler_path).parent
                scaler_dir.mkdir(parents=True, exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.input_scaler, f)
                logger.info(f"Saved fitted scaler to {self.scaler_path}")
        else:
            logger.warning("No valid data found for scaler fitting")
            # Create a dummy scaler to prevent transform mode errors
            #logger.info("Creating dummy scaler for transform mode compatibility")
            #dummy_data = np.zeros((100, len(self.input_cols)))
            #self.input_scaler.fit(dummy_data)
    
    def _preprocess_chunk(self, chunk: pd.DataFrame, fit_mode: bool = False) -> Optional[pd.DataFrame]:
        """
        Preprocess a chunk of data.
        
        Args:
            chunk: Raw data chunk
            fit_mode: Whether we're in scaler fitting mode (skip scaling)
            
        Returns:
            Processed chunk or None if invalid
        """
        try:
            chunk = chunk.copy()
            
            # 1. Create active/quiescent labels from the ORIGINAL data first
            chunk["is_active"] = (np.abs(chunk["qctend_TAU"]) > self.active_threshold).astype(float)

            
            # 2. Log transformations for skewed variables
            log_transform_cols = [
                "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
                "LAMC", "LAMR", "N0R"
            ]
            
            for col in log_transform_cols:
                if col in chunk.columns:
                    # Use log(x + epsilon) to handle zeros/negatives
                    epsilon = 1e-10
                    chunk[col] = np.log10(np.maximum(chunk[col], epsilon))
            
            # Also log transform output tendencies
            output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
            for col in output_log_cols:
                if col in chunk.columns:
                    # Handle negative values for tendencies
                    epsilon = 1e-10
                    sign = np.sign(chunk[col])
                    abs_val = np.abs(chunk[col]) + epsilon
                    chunk[col] = sign * np.log10(abs_val)
            

            # 3. Ensure required columns exist
            required_cols = self.input_cols + self.output_cols + ["is_active"]
            missing_cols = [col for col in required_cols if col not in chunk.columns]
            if missing_cols:
                logger.warning(f"Missing columns in chunk: {missing_cols}")
                return None
            
            # 4. Remove rows with NaN values
            initial_len = len(chunk)
            chunk = chunk.dropna(subset=required_cols)
            
            if len(chunk) == 0:
                return None
            
            if len(chunk) < initial_len:
                logger.debug(f"Dropped {initial_len - len(chunk)} rows with NaN values")
            
            return chunk[required_cols]
            
        except Exception as e:
            logger.warning(f"Error preprocessing chunk: {e}")
            return None
    
    def _chunk_generator(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Generate batches from chunks with memory-efficient shuffling."""
        
        # Shuffle files
        active_files = self.active_files.copy()
        random.shuffle(active_files)
        
        # Initialize shuffle buffer
        shuffle_buffer = deque(maxlen=self.shuffle_buffer_size)
        
        for file_path in active_files:
            try:
                logger.debug(f"Processing file: {file_path.name}")
                
                # Read file in chunks with fallback for compatibility
                try:
                    # Try chunked reading first
                    chunk_iter = pd.read_parquet(file_path, chunksize=self.chunk_size)
                except TypeError:
                    # Fallback: read entire file and split manually
                    logger.debug(f"Chunked reading not supported for {file_path}, reading entire file")
                    full_data = pd.read_parquet(file_path)
                    chunk_iter = [full_data[i:i+self.chunk_size] for i in range(0, len(full_data), self.chunk_size)]
                
                for chunk in chunk_iter:
                    # Apply sampling
                    if self.sample_fraction < 1.0:
                        n_samples = int(len(chunk) * self.sample_fraction)
                        if n_samples == 0:
                            continue
                        chunk = chunk.sample(n=n_samples, random_state=self.rng.randint(0, 2**31))
                    
                    # Preprocess chunk
                    processed_chunk = self._preprocess_chunk(chunk)
                    
                    if processed_chunk is None or len(processed_chunk) == 0:
                        continue
                    
                    # Convert to tensors and add to shuffle buffer
                    for idx in range(len(processed_chunk)):
                        try:
                            # Get input features (scaled)
                            if self.mode != "fit_only":
                                x_raw = processed_chunk.iloc[idx][self.input_cols].values.reshape(1, -1)
                                x_scaled = self.input_scaler.transform(x_raw).flatten()
                                x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
                                
                                # Get targets
                                targets = {}
                                targets['is_active'] = torch.tensor(
                                    processed_chunk.iloc[idx]['is_active'], dtype=torch.float32
                                ).reshape(1)
                                
                                for col in self.output_cols:
                                    if col in processed_chunk.columns:
                                        targets[col] = torch.tensor(
                                            processed_chunk.iloc[idx][col], dtype=torch.float32
                                        ).reshape(1)
                                
                                # Add to shuffle buffer
                                shuffle_buffer.append((x_tensor, targets))
                                
                                # Yield samples from shuffle buffer when it's full
                                if len(shuffle_buffer) >= self.shuffle_buffer_size:
                                    # Randomly sample from buffer
                                    idx_to_yield = random.randint(0, len(shuffle_buffer) - 1)
                                    yield shuffle_buffer[idx_to_yield]
                                    # Remove yielded sample
                                    del shuffle_buffer[idx_to_yield]
                        
                        except Exception as e:
                            logger.debug(f"Error processing sample: {e}")
                            continue
            
            except Exception as e:
                logger.warning(f"Error reading file {file_path}: {e}")
                continue
        
        # Yield remaining samples from shuffle buffer
        while shuffle_buffer:
            yield shuffle_buffer.popleft()
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Iterate over dataset."""
        return self._chunk_generator()
    
    def __len__(self) -> int:
        """Return estimated dataset size."""
        return self.estimated_size
    
    def get_data_info(self) -> Dict:
        """Get information about the dataset."""
        return {
            'estimated_size': self.estimated_size,
            'num_files': len(self.active_files),
            'input_features': len(self.input_cols),
            'output_targets': len(self.output_cols),
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'chunk_size': self.chunk_size,
            'split': self.split,
            'sample_fraction': self.sample_fraction
        }


def _resolve_run_directory(preferred_run_id: Optional[str]) -> str:
    """Resolve the run_<id> subdirectory name for scaler caching."""
    if preferred_run_id:
        return preferred_run_id if preferred_run_id.startswith("run_") else f"run_{preferred_run_id}"
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        return f"run_{job_id}" if not str(job_id).startswith("run_") else str(job_id)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"run_{timestamp}"


def create_streaming_data_loaders(
    data_path: str,
    config: Dict,
    train_fraction: float = 0.8,
    batch_size: int = 1024,
    num_workers: int = 0,
    scaler_cache_dir: str = "./scaler_cache",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Create streaming train and validation data loaders.
    
    Args:
        data_path: Path to parquet files
        config: Configuration dictionary
        train_fraction: Fraction of data for training
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes (0 for single-threaded)
        scaler_cache_dir: Directory to cache fitted scalers
        
    Returns:
        Tuple of (train_loader, val_loader, fitted_scaler)
    """
    # Extract configuration
    data_config = config.get('data', {})
    input_cols = data_config.get('input_cols', [])
    output_cols = data_config.get('output_cols', [])
    
    # Create per-run scaler cache directory similar to checkpointing
    preferred_run_id = os.environ.get('SCALER_RUN_ID')
    run_cache_dir = Path(scaler_cache_dir) / _resolve_run_directory(preferred_run_id)
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    scaler_cache_path = run_cache_dir / "input_scaler.pkl"
    
    is_distributed = distributed and world_size > 1

    # Step 1: ensure scaler is fitted (only rank 0 performs fitting to cover full dataset)
    if not is_distributed or rank == 0:
        logger.info("Creating training dataset and fitting scaler...")
        fit_dataset = StreamingMicrophysicsDataset(
            data_path=data_path,
            input_cols=input_cols,
            output_cols=output_cols,
            chunk_size=data_config.get('chunk_size', 50000),
            max_files=data_config.get('max_files'),
            sample_fraction=data_config.get('subsample', 1.0),
            random_seed=data_config.get('random_seed', 42),
            split="train",
            train_fraction=train_fraction,
            scaler_path=str(scaler_cache_path),
            mode="fit_transform",
            rank=0,
            world_size=1
        )
        fit_dataset.fit_scaler(n_samples_for_fitting=data_config.get('scaler_fit_samples', 100000))
        scaler = fit_dataset.input_scaler
        del fit_dataset
    else:
        scaler = StandardScaler()
        logger.info(f"Rank {rank}: waiting for scaler fitted by rank 0...")
    
    # Wait for scaler file if necessary
    if is_distributed and rank != 0:
        timeout_s = data_config.get('scaler_wait_timeout', 600)
        poll_interval = 2
        waited = 0
        while not scaler_cache_path.exists():
            time.sleep(poll_interval)
            waited += poll_interval
            if waited >= timeout_s:
                raise TimeoutError(
                    f"Rank {rank}: timed out waiting for scaler file at {scaler_cache_path}"
                )
        with open(scaler_cache_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Step 2: create sharded datasets for training/validation
    logger.info(f"Creating streaming datasets (distributed={is_distributed}, rank={rank}, world_size={world_size})")
    train_dataset = StreamingMicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        random_seed=data_config.get('random_seed', 42),
        split="train",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="transform",
        rank=rank if is_distributed else 0,
        world_size=world_size if is_distributed else 1
    )
    train_dataset.input_scaler = scaler
    
    logger.info("Creating validation dataset...")
    val_dataset = StreamingMicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        random_seed=data_config.get('random_seed', 42),
        split="val",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="transform",
        rank=rank if is_distributed else 0,
        world_size=world_size if is_distributed else 1
    )
    val_dataset.input_scaler = scaler
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created streaming data loaders:")
    logger.info(f"  Train: {train_dataset.estimated_size:,} samples")
    logger.info(f"  Val: {val_dataset.estimated_size:,} samples")
    
    return train_loader, val_loader, scaler


if __name__ == "__main__":
    """Test streaming data loader."""
    print("🧪 Testing Streaming Data Loader...")
    
    # Create test configuration
    test_config = {
        'data': {
            'input_cols': ['QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in', 
                          'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR'],
            'output_cols': ['qctend_TAU', 'nctend_TAU', 'nrtend_TAU'],
            'chunk_size': 10000,
            'max_files': 3,
            'subsample': 0.1,
            'random_seed': 42
        }
    }
    
    # Test with real data path (replace with your path)
    data_path = "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/"
    
    if Path(data_path).exists():
        try:
            train_loader, val_loader, scaler = create_streaming_data_loaders(
                data_path=data_path,
                config=test_config,
                batch_size=32
            )
            
            print("✅ Streaming data loaders created successfully!")
            
            # Test training loader
            print("🔄 Testing training loader...")
            for i, (x_batch, targets_batch) in enumerate(train_loader):
                print(f"  Batch {i+1}: input shape {x_batch.shape}")
                for key, tensor in targets_batch.items():
                    print(f"    {key}: {tensor.shape}")
                
                if i >= 2:  # Test first 3 batches
                    break
            
            print("✅ Streaming data loader test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Test data path {data_path} not found. Skipping test.")
        print("✅ Streaming data loader implementation completed!") 