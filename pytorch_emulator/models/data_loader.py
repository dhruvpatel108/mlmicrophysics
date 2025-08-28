"""
Data Loading for Constraint-Aware Microphysics Emulator

Handles loading and preprocessing of E3SM parquet data with log transformations,
standardization, and regime labeling as specified in the EDA analysis.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MicrophysicsDataset(Dataset):
    """
    PyTorch Dataset for microphysics data with constraint-aware preprocessing.
    
    Handles:
    - Log transformations for skewed variables
    - Standardization of input features
    - Active/quiescent regime labeling
    - Physical constraint validation
    """
    
    def __init__(
        self,
        data_path: str,
        input_cols: List[str],
        output_cols: List[str],
        max_files: int = 10,
        sample_fraction: float = 0.1,
        active_threshold: float = 1e-12,
        random_seed: int = 42
    ):
        """
        Initialize dataset from parquet files.
        
        Args:
            data_path: Path to directory containing parquet files
            input_cols: List of input feature column names
            output_cols: List of output target column names  
            max_files: Maximum number of parquet files to load
            sample_fraction: Fraction of data to sample from each file
            active_threshold: Threshold for active/quiescent classification
            random_seed: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.max_files = max_files
        self.sample_fraction = sample_fraction
        self.active_threshold = float(active_threshold)
        self.random_seed = random_seed
        
        # Initialize scalers
        self.input_scaler = StandardScaler()
        self.fitted = False
        
        # Load and preprocess data
        logger.info(f"Loading data from {data_path}...")
        self.data = self._load_and_preprocess_data()
        logger.info(f"Loaded {len(self.data)} samples")
        
    def _load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess data from parquet files."""
        # Find parquet files
        parquet_files = list(self.data_path.glob("*.parquet"))
        parquet_files = sorted(parquet_files)[:self.max_files]
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load data from multiple files
        dfs = []
        for file in parquet_files:
            logger.info(f"Loading {file.name}...")
            df = pd.read_parquet(file)
            
            # Sample data to reduce memory usage
            if self.sample_fraction < 1.0:
                n_samples = int(len(df) * self.sample_fraction)
                df = df.sample(n=n_samples, random_state=self.random_seed)
            
            dfs.append(df)
        
        # Combine all data
        combined_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined data shape: {combined_data.shape}")
        
        # Preprocess
        return self._preprocess_dataframe(combined_data)
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing transformations."""
        df = df.copy()
        
        # 1. Create active/quiescent labels from the ORIGINAL data first
        df["is_active"] = (np.abs(df["qctend_TAU"]) > self.active_threshold).astype(float)


        # 2. Log transformations for skewed variables (from EDA analysis)
        log_transform_cols = [
            "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
            "LAMC", "LAMR", "N0R"
        ]
        
        for col in log_transform_cols:
            if col in df.columns:
                # Use log(x + epsilon) to handle zeros
                epsilon = 1e-10
                df[col] = np.log10(df[col] + epsilon)
        
        # Also log transform output tendencies
        output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
        for col in output_log_cols:
            if col in df.columns:
                # Handle negative values differently for tendencies
                sign = np.sign(df[col])
                abs_val = np.abs(df[col]) + 1e-10
                df[col] = sign * np.log10(abs_val)
        
        
        # 3. Ensure required columns exist
        required_cols = self.input_cols + self.output_cols + ["is_active"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 4. Remove rows with NaN values
        initial_len = len(df)
        df = df.dropna(subset=required_cols)
        final_len = len(df)
        
        if final_len < initial_len:
            logger.warning(f"Dropped {initial_len - final_len} rows with NaN values")
        
        return df[required_cols]
    
    def fit_scalers(self):
        """Fit input scalers on the training data."""
        if not self.fitted:
            X = self.data[self.input_cols].values
            self.input_scaler.fit(X)
            self.fitted = True
            logger.info("Fitted input scaler")
    
    def get_preprocessed_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get preprocessed and scaled data."""
        if not self.fitted:
            self.fit_scalers()
        
        # Scale inputs
        X = self.input_scaler.transform(self.data[self.input_cols].values)
        
        # Prepare targets
        targets = {}
        targets['is_active'] = self.data['is_active'].values.reshape(-1, 1)
        
        for col in self.output_cols:
            if col in self.data.columns:
                targets[col] = self.data[col].values.reshape(-1, 1)
        
        return X, targets
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single sample."""
        if not self.fitted:
            raise RuntimeError("Must call fit_scalers() before accessing data")
        
        # Get input features (scaled)
        x_raw = self.data.iloc[idx][self.input_cols].values.reshape(1, -1)
        x_scaled = self.input_scaler.transform(x_raw).flatten()
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        # Get targets
        targets = {}
        targets['is_active'] = torch.tensor(
            self.data.iloc[idx]['is_active'], dtype=torch.float32
        ).reshape(1)
        
        for col in self.output_cols:
            if col in self.data.columns:
                targets[col] = torch.tensor(
                    self.data.iloc[idx][col], dtype=torch.float32
                ).reshape(1)
        
        return x_tensor, targets
    
    def get_data_info(self) -> Dict:
        """Get information about the loaded data."""
        active_count = (self.data['is_active'] == 1).sum()
        quiescent_count = (self.data['is_active'] == 0).sum()
        
        return {
            'total_samples': len(self.data),
            'active_samples': int(active_count),
            'quiescent_samples': int(quiescent_count),
            'active_fraction': float(active_count / len(self.data)),
            'input_features': len(self.input_cols),
            'output_targets': len(self.output_cols),
            'input_cols': self.input_cols,
            'output_cols': self.output_cols
        }


def create_data_loaders(
    data_path: str,
    config: Dict,
    train_fraction: float = 0.8,
    batch_size: int = 1024
) -> Tuple[DataLoader, DataLoader, MicrophysicsDataset]:
    """
    Create train and validation data loaders.
    
    Args:
        data_path: Path to parquet files
        config: Configuration dictionary
        train_fraction: Fraction of data for training
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, dataset)
    """
    # Extract configuration
    data_config = config.get('data', {})
    input_cols = data_config.get('input_cols', [])
    output_cols = data_config.get('output_cols', [])
    
    # Create dataset
    dataset = MicrophysicsDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        max_files=data_config.get('max_files', 10),
        sample_fraction=data_config.get('subsample', 0.1),
        active_threshold=data_config.get('active_threshold', 1e-12),
        random_seed=data_config.get('random_seed', 42)
    )
    
    # Fit scalers
    dataset.fit_scalers()
    
    # Split data
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(data_config.get('random_seed', 42))
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for NERSC compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader, dataset


# Test the data loader
if __name__ == "__main__":
    print("Testing data loader with sample data...")
    
    # Create sample data for testing
    test_data_path = "/tmp/test_data"
    Path(test_data_path).mkdir(exist_ok=True)
    
    # Create a sample parquet file
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'QC_TAU_in': np.random.lognormal(0, 1, n_samples),
        'QR_TAU_in': np.random.lognormal(0, 1, n_samples),
        'NC_TAU_in': np.random.lognormal(0, 1, n_samples),
        'NR_TAU_in': np.random.lognormal(0, 1, n_samples),
        'PGAM': np.random.normal(0, 1, n_samples),
        'LAMC': np.random.lognormal(0, 1, n_samples),
        'LAMR': np.random.lognormal(0, 1, n_samples),
        'N0R': np.random.lognormal(0, 1, n_samples),
        'RHO_CLUBB': np.random.normal(1.2, 0.1, n_samples),
        'CLOUD': np.random.uniform(0, 1, n_samples),
        'FREQR': np.random.uniform(0, 1, n_samples),
        'qctend_TAU': np.random.normal(0, 0.1, n_samples),
        'nctend_TAU': -np.abs(np.random.normal(0, 0.1, n_samples)),
        'nrtend_TAU': np.random.normal(0, 0.1, n_samples),
        'qrtend_TAU': np.abs(np.random.normal(0, 0.1, n_samples))
    }
    
    df = pd.DataFrame(sample_data)
    df.to_parquet(f"{test_data_path}/test_data.parquet")
    
    # Test dataset
    input_cols = ['QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in', 
                  'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR']
    output_cols = ['qctend_TAU', 'nctend_TAU', 'nrtend_TAU']
    
    dataset = MicrophysicsDataset(
        data_path=test_data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        max_files=1,
        sample_fraction=1.0,
        active_threshold=1e-12
    )
    
    print(f"✅ Dataset created successfully!")
    print(f"Dataset info: {dataset.get_data_info()}")
    
    # Test data loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    x_batch, targets_batch = next(iter(loader))
    
    print(f"✅ Data loader works!")
    print(f"  Batch input shape: {x_batch.shape}")
    print(f"  Batch targets:")
    for key, tensor in targets_batch.items():
        print(f"    {key}: {tensor.shape}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_data_path) 