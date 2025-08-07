"""
Dask-Based Data Loading for Constraint-Aware Microphysics Emulator

Implements efficient data loading using Dask for distributed processing of large datasets,
similar to the original Keras implementation approach. This should be much faster than
the current sample-by-sample streaming approach.

Key Features:
- Dask-based distributed data processing
- Vectorized chunk processing (not sample-by-sample)
- Memory-efficient lazy loading
- Compatible with PyTorch DataLoader
- Parallel preprocessing across workers
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import dask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable dask warnings
warnings.filterwarnings('ignore', category=UserWarning, module='dask')


class DaskMicrophysicsDataset(Dataset):
    """
    Dask-based PyTorch Dataset for microphysics data with efficient chunk processing.
    
    Uses Dask for distributed data processing, similar to the original Keras implementation.
    Much more efficient than sample-by-sample iteration.
    """
    
    def __init__(
        self,
        data_path: str,
        input_cols: List[str],
        output_cols: List[str],
        max_files: Optional[int] = None,
        sample_fraction: float = 1.0,
        active_threshold: float = 1e-12,
        chunk_size: int = 100000,  # Samples per chunk for processing
        split: str = "train",  # "train", "val", or "all"
        train_fraction: float = 0.8,
        scaler_path: Optional[str] = None,
        mode: str = "fit_transform",  # "fit_transform", "transform", "fit_only"
        n_workers: int = 4,  # Number of Dask workers
        random_seed: int = 42
    ):
        """
        Initialize Dask-based dataset.
        
        Args:
            data_path: Path to directory containing parquet files
            input_cols: List of input feature column names
            output_cols: List of output target column names
            max_files: Maximum number of parquet files to use (None = all)
            sample_fraction: Fraction of data to use
            active_threshold: Threshold for active/quiescent classification
            chunk_size: Samples per chunk for Dask processing
            split: Data split to use ("train", "val", "all")
            train_fraction: Fraction of data for training
            scaler_path: Path to saved scaler
            mode: Processing mode
            n_workers: Number of Dask workers
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.max_files = max_files
        self.sample_fraction = sample_fraction
        self.active_threshold = float(active_threshold)  # Convert to float to avoid comparison issues
        self.chunk_size = chunk_size
        self.split = split
        self.train_fraction = train_fraction
        self.scaler_path = scaler_path
        self.mode = mode
        self.n_workers = n_workers
        self.random_seed = random_seed
        
        # Setup Dask client
        self._setup_dask_client()
        
        # Find and process files
        self.parquet_files = self._find_parquet_files()
        logger.info(f"Found {len(self.parquet_files)} parquet files")
        
        # Setup scaler
        self.input_scaler = StandardScaler()
        self._setup_scaler()
        
        # Load and preprocess data using Dask
        logger.info("Loading and preprocessing data with Dask...")
        self.data_chunks = self._load_and_preprocess_with_dask()
        
        # Convert to in-memory arrays for PyTorch compatibility
        logger.info("Converting to PyTorch tensors...")
        self.input_data, self.target_data = self._convert_to_tensors()
        
        logger.info(f"Dataset ready: {len(self.input_data)} samples")
    
    def _setup_dask_client(self):
        """Setup Dask client for distributed processing."""
        try:
            # Try to connect to existing Dask cluster
            self.client = Client.current()
            logger.info(f"Connected to existing Dask cluster: {self.client}")
        except ValueError:
            # Create local cluster
            logger.info(f"Creating local Dask cluster with {self.n_workers} workers...")
            self.cluster = LocalCluster(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit='4GB',
                silence_logs=False
            )
            self.client = Client(self.cluster)
            logger.info(f"Created Dask cluster: {self.client}")
    
    def _find_parquet_files(self) -> List[Path]:
        """Find and sort parquet files."""
        parquet_files = list(self.data_path.glob("*.parquet"))
        parquet_files = sorted(parquet_files)
        
        if self.max_files is not None:
            parquet_files = parquet_files[:self.max_files]
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        # Split files for train/val
        if self.split != "all":
            n_train_files = int(len(parquet_files) * self.train_fraction)
            if self.split == "train":
                parquet_files = parquet_files[:n_train_files]
            elif self.split == "val":
                parquet_files = parquet_files[n_train_files:]
        
        return parquet_files
    
    def _setup_scaler(self):
        """Setup or load input scaler."""
        if self.scaler_path and Path(self.scaler_path).exists() and self.mode != "fit_transform":
            logger.info(f"Loading pre-fitted scaler from {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.input_scaler = pickle.load(f)
        else:
            logger.info("Will fit scaler on data")
    
    def _preprocess_chunk_dask(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess a chunk of data using vectorized operations.
        Much more efficient than sample-by-sample processing.
        """
        # 1. Log transformations for skewed variables
        log_transform_cols = [
            "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
            "LAMC", "LAMR", "N0R"
        ]
        
        for col in log_transform_cols:
            if col in chunk.columns:
                epsilon = 1e-10
                chunk[col] = np.log10(np.maximum(chunk[col], epsilon))
        
        # Log transform output tendencies
        output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
        for col in output_log_cols:
            if col in chunk.columns:
                epsilon = 1e-10
                sign = np.sign(chunk[col])
                abs_val = np.abs(chunk[col]) + epsilon
                chunk[col] = sign * np.log10(abs_val)
        
        # 2. Create active/quiescent labels
        chunk["is_active"] = (np.abs(chunk["qctend_TAU"]) > self.active_threshold).astype(float)
        
        # 3. Apply sampling if needed
        if self.sample_fraction < 1.0:
            n_samples = int(len(chunk) * self.sample_fraction)
            if n_samples > 0:
                chunk = chunk.sample(n=n_samples, random_state=self.random_seed)
        
        # 4. Ensure required columns exist and remove NaN
        required_cols = self.input_cols + self.output_cols + ["is_active"]
        missing_cols = [col for col in required_cols if col not in chunk.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return pd.DataFrame()  # Return empty DataFrame
        
        chunk = chunk[required_cols].dropna()
        
        return chunk
    
    def _load_and_preprocess_with_dask(self) -> List[pd.DataFrame]:
        """
        Load and preprocess data using Dask for efficient distributed processing.
        """
        # Create Dask DataFrame from parquet files
        logger.info("Creating Dask DataFrame from parquet files...")
        
        # Read all files into a single Dask DataFrame
        ddf = dd.read_parquet(
            [str(f) for f in self.parquet_files],
            chunksize=self.chunk_size
        )
        
        logger.info(f"Loaded Dask DataFrame with {ddf.npartitions} partitions")
        
        # Apply preprocessing to each partition using a static method
        logger.info("Applying preprocessing with Dask...")
        
        # Get the actual configuration values from the instance
        active_threshold = float(self.active_threshold)
        sample_fraction = float(self.sample_fraction)
        input_cols = self.input_cols.copy()
        output_cols = self.output_cols.copy()
        random_seed = self.random_seed
        
        logger.info(f"Using active_threshold: {active_threshold} (type: {type(active_threshold)})")
        logger.info(f"Using sample_fraction: {sample_fraction}")
        logger.info(f"Required input columns: {input_cols}")
        logger.info(f"Required output columns: {output_cols}")
        
        # Create a standalone preprocessing function to avoid serialization issues
        def preprocess_chunk_standalone(chunk):
            """Standalone preprocessing function for Dask."""
            try:
                chunk = chunk.copy()
                
                # 1. Log transformations for skewed variables
                log_transform_cols = [
                    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
                    "LAMC", "LAMR", "N0R"
                ]
                
                for col in log_transform_cols:
                    if col in chunk.columns:
                        epsilon = 1e-10
                        chunk[col] = np.log10(np.maximum(chunk[col], epsilon))
                
                # Log transform output tendencies
                output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
                for col in output_log_cols:
                    if col in chunk.columns:
                        epsilon = 1e-10
                        sign = np.sign(chunk[col])
                        abs_val = np.abs(chunk[col]) + epsilon
                        chunk[col] = sign * np.log10(abs_val)
                
                # 2. Create active/quiescent labels
                if "qctend_TAU" in chunk.columns:
                    chunk["is_active"] = (np.abs(chunk["qctend_TAU"]) > active_threshold).astype(float)
                else:
                    # Skip this chunk if missing required column
                    return pd.DataFrame()
                
                # 3. Apply sampling if needed
                if sample_fraction < 1.0:
                    n_samples = int(len(chunk) * sample_fraction)
                    if n_samples > 0:
                        chunk = chunk.sample(n=n_samples, random_state=random_seed)
                
                # 4. Ensure required columns exist and remove NaN
                required_cols = input_cols + output_cols + ["is_active"]
                missing_cols = [col for col in required_cols if col not in chunk.columns]
                if missing_cols:
                    logger.warning(f"Missing columns in chunk: {missing_cols}")
                    return pd.DataFrame()  # Return empty DataFrame
                
                chunk = chunk[required_cols].dropna()
                
                return chunk
                
            except Exception as e:
                logger.warning(f"Error preprocessing chunk: {e}")
                return pd.DataFrame()  # Return empty DataFrame on error
        
        # Create proper metadata for the result
        result_columns = input_cols + output_cols + ["is_active"]
        meta_dict = {}
        
        # Sample the first partition to get correct dtypes
        sample_chunk = ddf.get_partition(0).head(10)
        for col in result_columns:
            if col == "is_active":
                meta_dict[col] = float  # is_active is always float
            elif col in sample_chunk.columns:
                meta_dict[col] = sample_chunk[col].dtype
            else:
                meta_dict[col] = np.float32  # Default dtype
        
        # Create proper meta DataFrame
        meta_df = pd.DataFrame(columns=result_columns).astype(meta_dict)
        
        processed_ddf = ddf.map_partitions(
            preprocess_chunk_standalone,
            meta=meta_df
        )
        
        # Filter out empty partitions and compute
        logger.info("Computing preprocessed data...")
        with dask.config.set(scheduler='threads'):  # Use threaded scheduler for stability
            processed_chunks = []
            for i in range(processed_ddf.npartitions):
                try:
                    partition = processed_ddf.get_partition(i).compute()
                    if len(partition) > 0:
                        processed_chunks.append(partition)
                        logger.info(f"Processed partition {i}: {len(partition)} samples")
                except Exception as e:
                    logger.warning(f"Error processing partition {i}: {e}")
                    continue
        
        logger.info(f"Processed {len(processed_chunks)} non-empty chunks")
        return processed_chunks
    
    def _fit_scaler_on_chunks(self, chunks: List[pd.DataFrame]):
        """Fit scaler on sample of data from chunks."""
        if self.mode == "transform":
            return
        
        logger.info("Fitting scaler on sample data...")
        
        # Collect sample data for scaler fitting
        sample_data = []
        total_samples = 0
        target_samples = 50000  # Target number of samples for fitting
        
        for chunk in chunks:
            if total_samples >= target_samples:
                break
            
            # Sample from this chunk
            n_from_chunk = min(len(chunk), target_samples - total_samples)
            if n_from_chunk > 0:
                chunk_sample = chunk.sample(n=n_from_chunk, random_state=self.random_seed)
                sample_data.append(chunk_sample[self.input_cols])
                total_samples += n_from_chunk
        
        if sample_data:
            combined_sample = pd.concat(sample_data, ignore_index=True)
            self.input_scaler.fit(combined_sample.values)
            logger.info(f"Fitted scaler on {len(combined_sample)} samples")
            
            # Save scaler if path provided
            if self.scaler_path:
                Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.input_scaler, f)
                logger.info(f"Saved scaler to {self.scaler_path}")
    
    def _convert_to_tensors(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convert processed chunks to PyTorch tensors."""
        
        # Fit scaler if needed
        if self.mode in ["fit_transform", "fit_only"]:
            self._fit_scaler_on_chunks(self.data_chunks)
        
        if self.mode == "fit_only":
            # Return empty tensors, scaler fitting was the goal
            return torch.empty(0, len(self.input_cols)), {}
        
        # Combine all chunks
        logger.info("Combining data chunks...")
        if not self.data_chunks:
            logger.warning("No data chunks to process!")
            return torch.empty(0, len(self.input_cols)), {}
        
        combined_data = pd.concat(self.data_chunks, ignore_index=True)
        logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Scale inputs
        logger.info("Scaling input features...")
        input_data = self.input_scaler.transform(combined_data[self.input_cols].values)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        
        # Prepare targets
        logger.info("Preparing target tensors...")
        target_tensors = {}
        target_tensors['is_active'] = torch.tensor(
            combined_data['is_active'].values, dtype=torch.float32
        ).reshape(-1, 1)
        
        for col in self.output_cols:
            if col in combined_data.columns:
                target_tensors[col] = torch.tensor(
                    combined_data[col].values, dtype=torch.float32
                ).reshape(-1, 1)
        
        return input_tensor, target_tensors
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single sample."""
        inputs = self.input_data[idx]
        targets = {key: tensor[idx] for key, tensor in self.target_data.items()}
        return inputs, targets
    
    def get_data_info(self) -> Dict:
        """Get dataset information."""
        if hasattr(self, 'target_data') and 'is_active' in self.target_data:
            active_count = self.target_data['is_active'].sum().item()
            total_count = len(self.target_data['is_active'])
            active_fraction = active_count / total_count if total_count > 0 else 0.0
        else:
            active_count = 0
            total_count = len(self.input_data)
            active_fraction = 0.0
        
        return {
            'total_samples': total_count,
            'active_samples': int(active_count),
            'quiescent_samples': total_count - int(active_count),
            'active_fraction': active_fraction,
            'input_features': len(self.input_cols),
            'output_targets': len(self.output_cols),
            'input_cols': self.input_cols,
            'output_cols': self.output_cols
        }
    
    def cleanup(self):
        """Cleanup Dask resources."""
        if hasattr(self, 'client'):
            self.client.close()
        if hasattr(self, 'cluster'):
            self.cluster.close()


def create_dask_data_loaders(
    data_path: str,
    config: Dict,
    train_fraction: float = 0.8,
    batch_size: int = 1024,
    num_workers: int = 0,
    scaler_cache_dir: str = "./scaler_cache",
    n_dask_workers: int = 4
) -> Tuple[DataLoader, DataLoader, StandardScaler]:
    """
    Create Dask-based train and validation data loaders.
    
    Args:
        data_path: Path to parquet files
        config: Configuration dictionary
        train_fraction: Fraction of data for training
        batch_size: Batch size for data loaders
        num_workers: Number of DataLoader worker processes
        scaler_cache_dir: Directory to cache fitted scalers
        n_dask_workers: Number of Dask workers for preprocessing
        
    Returns:
        Tuple of (train_loader, val_loader, fitted_scaler)
    """
    # Extract configuration
    data_config = config.get('data', {})
    input_cols = data_config.get('input_cols', [])
    output_cols = data_config.get('output_cols', [])
    
    # Create scaler cache directory
    scaler_cache_path = Path(scaler_cache_dir) / "input_scaler_dask.pkl"
    scaler_cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create training dataset (for fitting scaler)
    logger.info("Creating Dask-based training dataset...")
    train_dataset = DaskMicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=data_config.get('active_threshold', 1e-12),
        chunk_size=data_config.get('chunk_size', 100000),
        split="train",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="fit_transform",
        n_workers=n_dask_workers,
        random_seed=data_config.get('random_seed', 42)
    )
    
    # Create validation dataset (using pre-fitted scaler)
    logger.info("Creating Dask-based validation dataset...")
    val_dataset = DaskMicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=data_config.get('active_threshold', 1e-12),
        chunk_size=data_config.get('chunk_size', 100000),
        split="val",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        mode="transform",
        n_workers=n_dask_workers,
        random_seed=data_config.get('random_seed', 42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
    
    logger.info(f"Created Dask data loaders:")
    logger.info(f"  Train: {len(train_dataset):,} samples")
    logger.info(f"  Val: {len(val_dataset):,} samples")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    
    # Cleanup Dask resources
    train_dataset.cleanup()
    val_dataset.cleanup()
    
    return train_loader, val_loader, train_dataset.input_scaler


if __name__ == "__main__":
    """Test Dask data loader."""
    print("🧪 Testing Dask Data Loader...")
    
    # Create test configuration
    test_config = {
        'data': {
            'input_cols': ['QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in', 
                          'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR'],
            'output_cols': ['qctend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qrtend_TAU'],
            'chunk_size': 50000,
            'max_files': 5,
            'subsample': 0.1,
            'random_seed': 42
        }
    }
    
    # Test with real data path
    data_path = "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/"
    
    if Path(data_path).exists():
        try:
            train_loader, val_loader, scaler = create_dask_data_loaders(
                data_path=data_path,
                config=test_config,
                batch_size=64,
                n_dask_workers=2
            )
            
            print("✅ Dask data loaders created successfully!")
            
            # Test training loader
            print("🔄 Testing training loader...")
            for i, (x_batch, targets_batch) in enumerate(train_loader):
                print(f"  Batch {i+1}: input shape {x_batch.shape}")
                for key, tensor in targets_batch.items():
                    print(f"    {key}: {tensor.shape}")
                
                if i >= 2:  # Test first 3 batches
                    break
            
            print("✅ Dask data loader test completed!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ Test data path {data_path} not found. Skipping test.")
        print("✅ Dask data loader implementation completed!") 