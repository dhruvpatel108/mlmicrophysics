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
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from typing import Dict, List, Tuple, Optional, Iterator, Union
import time
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
        active_threshold: float = 1.0e-15,
        cloud_threshold: Optional[float] = 0.01,
        mass_input_threshold: Optional[float] = 1.0e-5,
        rho_threshold: Optional[float] = 0.2,
        nrtend_threshold: Optional[float] = 1.0e-10,
        random_seed: int = 42,
        split: str = "train",  # "train", "val", or "all"
        train_fraction: float = 0.8,
        scaler_path: Optional[str] = None,
        output_scaler_path: Optional[str] = None,
        output_transformer_path: Optional[str] = None,
        mode: str = "fit_transform",
        disable_length_estimation: bool = False,  # Disable length estimation for large datasets
        input_transform: str = "log10",
        output_transform: str = "log10",
        input_scaling: str = "standard",
        output_scaling: str = "standard",
        quantile_n_quantiles: int = 1000,
        quantile_subsample: int = 100000,
        rank: int = 0,
        world_size: int = 1,
        nrtend_arcsinh_transform: bool = False,
        nrtend_arcsinh_threshold: float = 1.0e-3,
        nrtend_regime_threshold: Optional[float] = None
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
        self.cloud_threshold = float(cloud_threshold) if cloud_threshold is not None else None
        self.mass_input_threshold = float(mass_input_threshold) if mass_input_threshold is not None else None
        self.rho_threshold = float(rho_threshold) if rho_threshold is not None else None
        self.nrtend_threshold = float(nrtend_threshold) if nrtend_threshold is not None else None
        self.random_seed = random_seed
        self.split = split
        self.train_fraction = train_fraction
        self.scaler_path = scaler_path
        self.output_scaler_path = output_scaler_path
        self.output_transformer_path = output_transformer_path
        self.mode = mode
        self.disable_length_estimation = disable_length_estimation
        self.input_transform = (input_transform or "log10").lower()
        self.output_transform = (output_transform or "log10").lower()
        self.input_scaling = (input_scaling or "standard").lower()
        self.output_scaling = (output_scaling or "standard").lower()
        self.quantile_n_quantiles = int(quantile_n_quantiles)
        self.quantile_subsample = int(quantile_subsample)
        self.rank = rank
        self.world_size = max(1, world_size)
        
        # Inverse hyperbolic sine (arcsinh) transform for nrtend
        self.nrtend_arcsinh_transform = bool(nrtend_arcsinh_transform)
        self.nrtend_arcsinh_threshold = float(nrtend_arcsinh_threshold)

        # MoE regime labels: 0=near-zero, 1=negative (self-collection), 2=positive (rain formation)
        self.nrtend_regime_threshold = float(nrtend_regime_threshold) if nrtend_regime_threshold is not None else None
        if self.nrtend_arcsinh_transform:
            logger.info(
                f"nrtend arcsinh transform ENABLED: y' = arcsinh(y / {self.nrtend_arcsinh_threshold})"
            )
        
        # Initialize random state
        self.rng = np.random.RandomState(random_seed)
        random.seed(random_seed)
        
        # Find parquet files
        self.parquet_files = self._find_parquet_files()
        logger.info(f"Found {len(self.parquet_files)} parquet files")
        
        # Initialize scalers
        # Initialize scalers based on config
        self.input_scaler = RobustScaler() if self.input_scaling == 'robust' else StandardScaler()
        self.output_scaler = RobustScaler() if self.output_scaling == 'robust' else StandardScaler()
        # Initialize transformers
        self.input_transformer: Optional[QuantileTransformer] = None
        self.output_transformer: Optional[QuantileTransformer] = None
        self._load_or_fit_scaler()
        self._load_or_prepare_output_scaler()
        self._load_or_prepare_input_transformer()
        self._load_or_prepare_output_transformer()
        
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

    def _shard_active_files(self):
        """Deprecated: retained for backward compatibility (no-op)."""
        if self.world_size > 1:
            logger.debug("File-level sharding disabled; using batch-level round robin distribution.")
    
    def _load_or_fit_scaler(self):
        """Load existing scaler or prepare to fit new one."""
        if self.scaler_path and Path(self.scaler_path).exists() and self.mode != "fit_transform":
            logger.info(f"Loading pre-fitted scaler from {self.scaler_path}")
            with open(self.scaler_path, 'rb') as f:
                self.input_scaler = pickle.load(f)
        else:
            logger.info("Will fit scaler on data")

    def _load_or_prepare_output_scaler(self):
        """Load existing output scaler or prepare to fit new one (if enabled)."""
        if self.output_scaler_path and Path(self.output_scaler_path).exists() and self.mode != "fit_transform":
            logger.info(f"Loading pre-fitted OUTPUT scaler from {self.output_scaler_path}")
            try:
                with open(self.output_scaler_path, 'rb') as f:
                    self.output_scaler = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load output scaler: {e}. Will refit if possible.")
        else:
            logger.info("Will fit OUTPUT scaler on data")

    def _load_or_prepare_input_transformer(self):
        """Load or prepare input quantile transformer if requested."""
        if self.input_transform != "quantile":
            return
        if self.output_transformer_path:  # Reuse path var for symmetry? Better to have input path
            pass
        # If we introduce input_transformer_path later, this placeholder keeps structure consistent

    def _load_or_prepare_output_transformer(self):
        """Load or prepare output quantile transformer if requested."""
        if self.output_transform != "quantile":
            return
        if self.output_transformer_path and Path(self.output_transformer_path).exists() and self.mode != "fit_transform":
            logger.info(f"Loading pre-fitted OUTPUT QuantileTransformer from {self.output_transformer_path}")
            try:
                with open(self.output_transformer_path, 'rb') as f:
                    self.output_transformer = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load output QuantileTransformer: {e}. Will refit if possible.")
        else:
            logger.info("Will fit OUTPUT QuantileTransformer on data (output_transform=quantile)")
    
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

            # 1. Apply physics-informed filtering before any transforms
            if "qctend_TAU" not in chunk.columns:
                logger.warning("Column 'qctend_TAU' missing from chunk; skipping chunk")
                return None

            # Active filter: always kept as the base mask
            mask = (np.abs(chunk["qctend_TAU"]) > self.active_threshold)

            # --- ACTIVE FILTERS ---
            # Filter 1: CLOUD > cloud_threshold
            if self.cloud_threshold is not None:
                if "CLOUD" in chunk.columns:
                    mask &= chunk["CLOUD"] > self.cloud_threshold
                else:
                    logger.warning("Column 'CLOUD' missing; cloud threshold filter skipped")

            # Filter 2: QC_TAU_in > mass_input_threshold
            if self.mass_input_threshold is not None:
                if "QC_TAU_in" in chunk.columns:
                    mask &= chunk["QC_TAU_in"] > self.mass_input_threshold
                else:
                    logger.warning("Column 'QC_TAU_in' missing; mass input threshold filter skipped")

            # --- DISABLED FILTERS ---
            #if self.rho_threshold is not None:
            #    if "RHO_CLUBB" in chunk.columns:
            #        mask &= chunk["RHO_CLUBB"] > self.rho_threshold
            #    else:
            #        logger.warning("Column 'RHO_CLUBB' missing; rho threshold filter skipped")

            #if self.nrtend_threshold is not None:
            #    if "nrtend_TAU" in chunk.columns:
            #        mask &= np.abs(chunk["nrtend_TAU"]) > self.nrtend_threshold
            #    else:
            #        logger.warning("Column 'nrtend_TAU' missing; nrtend threshold filter skipped")

            #if "QC_TAU_in" in chunk.columns and "QR_TAU_in" in chunk.columns:
            #    mask &= np.isfinite(chunk["QC_TAU_in"]) & np.isfinite(chunk["QR_TAU_in"])
            #if "RHO_CLUBB" in chunk.columns:
            #    mask &= np.isfinite(chunk["RHO_CLUBB"])

            if not np.any(mask):
                return None

            chunk = chunk.loc[mask].copy()

            # 2. Create active/quiescent labels from the ORIGINAL data first (post-filter)
            chunk["is_active"] = (np.abs(chunk["qctend_TAU"]) > self.active_threshold).astype(float)

            # 2b. MoE regime labels from RAW nrtend_TAU (before any transform)
            if self.nrtend_regime_threshold is not None and "nrtend_TAU" in chunk.columns:
                eps = self.nrtend_regime_threshold
                chunk["nrtend_regime"] = 0  # near-zero
                chunk.loc[chunk["nrtend_TAU"] < -eps, "nrtend_regime"] = 1  # self-collection
                chunk.loc[chunk["nrtend_TAU"] > eps, "nrtend_regime"] = 2   # rain formation

            # 3. Input transformation (vectorized)
            if self.input_transform == "log10":
                log_transform_cols = [
                    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
                    "LAMC", "LAMR", "N0R"
                ]
                for col in log_transform_cols:
                    if col in chunk.columns:
                        epsilon = 1e-10
                        chunk[col] = np.log10(np.maximum(chunk[col], epsilon))
            
            # Output transformation: only apply log10 here; quantile done later in transform stage
            if self.output_transform == "log10":
                output_log_cols = ["qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"]
                # If arcsinh is enabled for nrtend, exclude it from log10 transform
                if self.nrtend_arcsinh_transform:
                    output_log_cols = [c for c in output_log_cols if c != "nrtend_TAU"]
                for col in output_log_cols:
                    if col in chunk.columns:
                        epsilon = 1e-10
                        sign = np.sign(chunk[col])
                        abs_val = np.abs(chunk[col]) + epsilon
                        chunk[col] = sign * np.log10(abs_val)
            
            # Apply arcsinh transform to nrtend_TAU if enabled: y' = arcsinh(y / c)
            if self.nrtend_arcsinh_transform:
                if "nrtend_TAU" in chunk.columns:
                    c = self.nrtend_arcsinh_threshold
                    chunk["nrtend_TAU"] = np.arcsinh(chunk["nrtend_TAU"] / c)
            
            
            # 4. Apply sampling if needed
            if self.sample_fraction < 1.0:
                n_samples = int(len(chunk) * self.sample_fraction)
                if n_samples > 0:
                    chunk = chunk.sample(n=n_samples, random_state=self.rng.randint(0, 2**31))
            
            # 5. Ensure required columns exist and remove NaN
            required_cols = self.input_cols + self.output_cols + ["is_active"]
            if self.nrtend_regime_threshold is not None and "nrtend_regime" in chunk.columns:
                required_cols = required_cols + ["nrtend_regime"]
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
        
        # Inputs: transform then scale
        input_matrix = chunk[self.input_cols].values
        if self.input_transform == 'quantile' and self.input_transformer is not None:
            try:
                input_matrix = self.input_transformer.transform(input_matrix)
            except Exception as e:
                logger.debug(f"Input quantile transform failed or not fitted yet: {e}")
        # scale
        input_data = self.input_scaler.transform(input_matrix)
        
        # Create all target arrays at once (vectorized)
        targets_data = {}
        targets_data['is_active'] = chunk['is_active'].values.reshape(-1, 1)

        if "nrtend_regime" in chunk.columns:
            targets_data["nrtend_regime"] = chunk["nrtend_regime"].values.reshape(-1, 1)

        # Outputs matrix in configured column order
        outputs_matrix = None
        if self.output_cols:
            outputs_matrix = chunk[self.output_cols].values
            if outputs_matrix is not None:
                # transform
                if self.output_transform == "quantile" and self.output_transformer is not None:
                    try:
                        outputs_matrix = self.output_transformer.transform(outputs_matrix)
                    except Exception as e:
                        logger.debug(f"Quantile transform failed or not fitted yet: {e}")
                # scale
                try:
                    outputs_matrix = self.output_scaler.transform(outputs_matrix)
                except Exception as e:
                    logger.debug(f"Output scaling transform failed or not fitted yet: {e}")
        # Split back into per-target arrays
        for j, col in enumerate(self.output_cols):
            if col in chunk.columns:
                if outputs_matrix is not None:
                    targets_data[col] = outputs_matrix[:, j].reshape(-1, 1)
                else:
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
    
    def fit_scalers(self, n_samples_for_fitting: int = 100000):
        """Fit input scaler, optional output scaler, and optional quantile transformer on a sample of the data."""
        if self.mode == "transform":
            return
        
        logger.info(f"Preparing to fit transformers/scalers on up to {n_samples_for_fitting} samples...")
        
        samples_collected = 0
        all_inputs = []
        all_outputs = []
        
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
                        if self.output_cols:
                            outs = processed_chunk[self.output_cols].values
                            all_outputs.append(outs)
                        samples_collected += len(inputs)
                        
                        logger.info(f"Collected {samples_collected}/{n_samples_for_fitting} samples")
            
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        # Combine
        if all_inputs:
            combined_inputs = np.vstack(all_inputs)
            # Subsample if we have too much data
            if len(combined_inputs) > n_samples_for_fitting:
                indices = np.random.choice(len(combined_inputs), n_samples_for_fitting, replace=False)
                combined_inputs = combined_inputs[indices]

        if all_outputs:
            combined_outputs = np.vstack(all_outputs)
            if len(combined_outputs) > n_samples_for_fitting:
                indices = np.random.choice(len(combined_outputs), n_samples_for_fitting, replace=False)
                combined_outputs = combined_outputs[indices]

        # Fit transformers first (on combined raw arrays)
        # Inputs quantile
        if self.input_transform == 'quantile' and all_inputs:
            inp_for_quant = combined_inputs
            if len(inp_for_quant) > self.quantile_subsample:
                indices = np.random.choice(len(inp_for_quant), self.quantile_subsample, replace=False)
                inp_for_quant = inp_for_quant[indices]
            self.input_transformer = QuantileTransformer(
                n_quantiles=min(self.quantile_n_quantiles, inp_for_quant.shape[0]),
                output_distribution='normal',
                subsample=self.quantile_subsample,
                copy=True,
                random_state=self.random_seed
            )
            self.input_transformer.fit(inp_for_quant)
            logger.info(f"Fitted INPUT QuantileTransformer on {inp_for_quant.shape[0]} samples")
            # Persist if an input transformer path is provided (not yet wired)

        # Outputs quantile
        if self.output_transform == 'quantile' and all_outputs:
            out_for_quant = combined_outputs
            if len(out_for_quant) > self.quantile_subsample:
                indices = np.random.choice(len(out_for_quant), self.quantile_subsample, replace=False)
                out_for_quant = out_for_quant[indices]
            self.output_transformer = QuantileTransformer(
                n_quantiles=min(self.quantile_n_quantiles, out_for_quant.shape[0]),
                output_distribution='normal',
                subsample=self.quantile_subsample,
                copy=True,
                random_state=self.random_seed
            )
            self.output_transformer.fit(out_for_quant)
            logger.info(f"Fitted OUTPUT QuantileTransformer on {out_for_quant.shape[0]} samples")
            if self.output_transformer_path:
                Path(self.output_transformer_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_transformer_path, 'wb') as f:
                    pickle.dump(self.output_transformer, f)
                logger.info(f"Saved OUTPUT QuantileTransformer to {self.output_transformer_path}")

        # Prepare transformed arrays for scaler fitting (transform-then-scale)
        if all_inputs:
            inputs_for_scaler = combined_inputs
            if self.input_transform == 'quantile' and self.input_transformer is not None:
                inputs_for_scaler = self.input_transformer.transform(inputs_for_scaler)
            # Fit input scaler
            self.input_scaler.fit(inputs_for_scaler)
            logger.info(f"Fitted INPUT {self.input_scaling.title()}Scaler on {inputs_for_scaler.shape[0]} samples")
            if self.scaler_path:
                Path(self.scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.input_scaler, f)
                logger.info(f"Saved input scaler to {self.scaler_path}")

        if all_outputs:
            outputs_for_scaler = combined_outputs
            if self.output_transform == 'quantile' and self.output_transformer is not None:
                outputs_for_scaler = self.output_transformer.transform(outputs_for_scaler)
            # Fit output scaler
            self.output_scaler.fit(outputs_for_scaler)
            logger.info(f"Fitted OUTPUT {self.output_scaling.title()}Scaler on {outputs_for_scaler.shape[0]} samples")
            if self.output_scaler_path:
                Path(self.output_scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_scaler_path, 'wb') as f:
                    pickle.dump(self.output_scaler, f)
                logger.info(f"Saved output scaler to {self.output_scaler_path}")
    
    def _batch_generator(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Generate batches from files ensuring every rank sees the same number of steps."""
        
        # Shuffle files deterministically based on seeded random state
        active_files = self.active_files.copy()
        random.shuffle(active_files)
        
        if self.world_size > 1:
            batch_group_inputs: List[torch.Tensor] = []
            batch_group_targets: List[Dict[str, torch.Tensor]] = []
        
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
                    
                    for batch_inputs, batch_targets in self._process_chunk_to_batches(processed_chunk):
                        if self.world_size <= 1:
                            yield batch_inputs, batch_targets
                            continue
                        
                        batch_group_inputs.append(batch_inputs)
                        batch_group_targets.append(batch_targets)
                        
                        if len(batch_group_inputs) == self.world_size:
                            yield batch_group_inputs[self.rank], batch_group_targets[self.rank]
                            batch_group_inputs.clear()
                            batch_group_targets.clear()
            
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
                continue
        
        if self.world_size > 1 and batch_group_inputs:
            # Pad or drop remainder so that all ranks stay in lock-step
            remainder = len(batch_group_inputs)
            if remainder < self.world_size:
                # Reuse the last available batch to pad the group; safe because training is stochastic
                last_input = batch_group_inputs[-1]
                last_target = batch_group_targets[-1]
                while len(batch_group_inputs) < self.world_size:
                    batch_group_inputs.append(last_input.clone())
                    # Clone target tensors to avoid shared references
                    padded_targets = {k: v.clone() for k, v in last_target.items()}
                    batch_group_targets.append(padded_targets)
            yield batch_group_inputs[self.rank], batch_group_targets[self.rank]
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Iterate over dataset, yielding batches."""
        return self._batch_generator()
    
    def __len__(self) -> int:
        """Return estimated number of batches using ceiling division.

        Data is read in chunk_size-row chunks. _process_chunk_to_batches then
        splits each chunk with batch_size as the step size.  When chunk_size <
        batch_size (the common case) every chunk produces exactly one batch of
        up to chunk_size samples — NOT batch_size samples.  Using batch_size as
        the denominator therefore underestimates the batch count by a factor of
        batch_size / chunk_size.  The correct effective yield granularity is
        min(chunk_size, batch_size).
        """
        import math

        effective_yield_size = min(self.chunk_size, self.batch_size)
        total_batches = max(1, math.ceil(self.estimated_size / effective_yield_size))
        if self.world_size > 1:
            total_batches = max(1, math.floor(total_batches / self.world_size))
        return total_batches


def create_optimized_streaming_loaders(
    data_path: str,
    config: Dict,
    train_fraction: float = 0.8,
    batch_size: int = 1024,
    scaler_cache_dir: str = "./scaler_cache",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    run_id_override: Optional[str] = None
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
    
    # Extract filtering thresholds with sensible defaults (can be overridden via config)
    cloud_threshold = data_config.get('cloud_threshold', 0.01)
    mass_input_threshold = data_config.get('mass_input_threshold', 1e-5)
    rho_threshold = data_config.get('rho_threshold', 0.2)
    nrtend_threshold = data_config.get('nrtend_threshold', 1e-10)
    active_threshold_value = data_config.get('active_threshold', 1e-15)
    if active_threshold_value is None:
        active_threshold_value = 1e-15

    # Arcsinh transform settings for nrtend
    nrtend_arcsinh_transform = bool(data_config.get('nrtend_arcsinh_transform', False))
    nrtend_arcsinh_threshold = float(data_config.get('nrtend_arcsinh_threshold', 1e-3))

    # MoE regime label threshold (None disables regime label computation)
    nrtend_regime_threshold_val = data_config.get('nrtend_regime_threshold')
    nrtend_regime_threshold = float(nrtend_regime_threshold_val) if nrtend_regime_threshold_val is not None else None

    # Create per-run scaler cache directory similar to checkpointing
    def _normalize_run_id(run_id: str) -> str:
        return run_id if run_id.startswith("run_") else f"run_{run_id}"

    job_id = (
        run_id_override
        or os.environ.get("SCALER_RUN_ID")
        or os.environ.get("SLURM_JOB_ID")
    )
    if job_id is None:
        job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized_run_id = _normalize_run_id(job_id)
    run_cache_dir = Path(scaler_cache_dir) / normalized_run_id
    if rank == 0:
        logger.info(f"Using scaler cache directory: {run_cache_dir}")
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    scaler_cache_path = run_cache_dir / "input_scaler_optimized.pkl"
    output_scaler_cache_path = run_cache_dir / "output_scaler_optimized.pkl"
    output_transformer_cache_path = run_cache_dir / "output_quantile_transformer.pkl"
    input_transformer_cache_path = run_cache_dir / "input_quantile_transformer.pkl"
    
    is_distributed = distributed and world_size > 1

    def _wait_for_artifact(path: Path, description: str):
        if not is_distributed or rank == 0:
            return
        timeout_s = data_config.get('scaler_wait_timeout', 600)
        poll_interval = 2
        waited = 0
        while not path.exists():
            time.sleep(poll_interval)
            waited += poll_interval
            if waited >= timeout_s:
                raise TimeoutError(
                    f"Rank {rank}: timed out waiting for {description} at {path}"
                )

    expected_artifacts = [scaler_cache_path]
    if output_cols:
        expected_artifacts.append(output_scaler_cache_path)
    output_transform_mode = str(data_config.get('output_transform', 'log10')).lower()
    if output_transform_mode == 'quantile':
        expected_artifacts.append(output_transformer_cache_path)

    force_refit = bool(data_config.get('force_refit_scalers', False))
    artifacts_exist = all(path.exists() for path in expected_artifacts)
    should_fit_scalers = force_refit or not artifacts_exist

    scaler: Optional[StandardScaler] = None

    if should_fit_scalers:
        if not is_distributed or rank == 0:
            logger.info("Creating optimized training dataset for fitting (artifacts missing or refit forced)...")
            fit_dataset = OptimizedStreamingDataset(
                data_path=data_path,
                input_cols=input_cols,
                output_cols=output_cols,
                batch_size=actual_batch_size,
                chunk_size=data_config.get('chunk_size', 50000),
                max_files=data_config.get('max_files'),
                sample_fraction=data_config.get('subsample', 1.0),
                active_threshold=active_threshold_value,
                cloud_threshold=cloud_threshold,
                mass_input_threshold=mass_input_threshold,
                rho_threshold=rho_threshold,
                nrtend_threshold=nrtend_threshold,
                random_seed=data_config.get('random_seed', 42),
                split="train",
                train_fraction=train_fraction,
                scaler_path=str(scaler_cache_path),
                output_scaler_path=str(output_scaler_cache_path),
                mode="fit_transform",
                disable_length_estimation=disable_length,
                input_transform=str(data_config.get('input_transform', 'log10')).lower(),
                output_transform=output_transform_mode,
                input_scaling=str(data_config.get('input_scaling', 'standard')).lower(),
                output_scaling=str(data_config.get('output_scaling', 'standard')).lower(),
                output_transformer_path=str(output_transformer_cache_path),
                quantile_n_quantiles=int(data_config.get('quantile_n_quantiles', 1000)),
                quantile_subsample=int(data_config.get('quantile_subsample', 100000)),
                rank=0,
                world_size=1,
                nrtend_arcsinh_transform=nrtend_arcsinh_transform,
                nrtend_arcsinh_threshold=nrtend_arcsinh_threshold,
                nrtend_regime_threshold=nrtend_regime_threshold,
            )
            fit_dataset.fit_scalers(n_samples_for_fitting=data_config.get('scaler_fit_samples', 100000))
            scaler = fit_dataset.input_scaler
            del fit_dataset
        else:
            scaler = StandardScaler()
            logger.info(f"Rank {rank}: waiting for scaler artifacts from rank 0...")
    else:
        logger.info("Found existing scaler artifacts; reusing cached transformers/scalers.")
        for artifact in expected_artifacts:
            logger.debug(f"  Using cached artifact: {artifact}")

    # Ensure scaler artifacts exist before proceeding on non-zero ranks
    _wait_for_artifact(scaler_cache_path, "input scaler")
    if output_cols:
        _wait_for_artifact(output_scaler_cache_path, "output scaler")
    if str(data_config.get('output_transform', 'log10')).lower() == 'quantile':
        _wait_for_artifact(output_transformer_cache_path, "output quantile transformer")

    # Stage 2: create sharded datasets that reuse persisted scalers
    logger.info(f"Creating optimized training dataset (distributed={is_distributed}, rank={rank}, world_size={world_size})")
    train_dataset = OptimizedStreamingDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        batch_size=actual_batch_size,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=active_threshold_value,
        cloud_threshold=cloud_threshold,
        mass_input_threshold=mass_input_threshold,
        rho_threshold=rho_threshold,
        nrtend_threshold=nrtend_threshold,
        random_seed=data_config.get('random_seed', 42),
        split="train",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        output_scaler_path=str(output_scaler_cache_path),
        mode="transform",
        disable_length_estimation=disable_length,
        input_transform=str(data_config.get('input_transform', 'log10')).lower(),
        output_transform=str(data_config.get('output_transform', 'log10')).lower(),
        input_scaling=str(data_config.get('input_scaling', 'standard')).lower(),
        output_scaling=str(data_config.get('output_scaling', 'standard')).lower(),
        output_transformer_path=str(output_transformer_cache_path),
        quantile_n_quantiles=int(data_config.get('quantile_n_quantiles', 1000)),
        quantile_subsample=int(data_config.get('quantile_subsample', 100000)),
        rank=rank if is_distributed else 0,
        world_size=world_size if is_distributed else 1,
        nrtend_arcsinh_transform=nrtend_arcsinh_transform,
        nrtend_arcsinh_threshold=nrtend_arcsinh_threshold,
        nrtend_regime_threshold=nrtend_regime_threshold,
    )
    
    logger.info("Creating optimized validation dataset...")
    val_dataset = OptimizedStreamingDataset(
        data_path=data_path,
        input_cols=input_cols,
        output_cols=output_cols,
        batch_size=actual_batch_size,
        chunk_size=data_config.get('chunk_size', 50000),
        max_files=data_config.get('max_files'),
        sample_fraction=data_config.get('subsample', 1.0),
        active_threshold=active_threshold_value,
        cloud_threshold=cloud_threshold,
        mass_input_threshold=mass_input_threshold,
        rho_threshold=rho_threshold,
        nrtend_threshold=nrtend_threshold,
        random_seed=data_config.get('random_seed', 42),
        split="val",
        train_fraction=train_fraction,
        scaler_path=str(scaler_cache_path),
        output_scaler_path=str(output_scaler_cache_path),
        mode="transform",
        disable_length_estimation=disable_length,
        input_transform=str(data_config.get('input_transform', 'log10')).lower(),
        output_transform=str(data_config.get('output_transform', 'log10')).lower(),
        input_scaling=str(data_config.get('input_scaling', 'standard')).lower(),
        output_scaling=str(data_config.get('output_scaling', 'standard')).lower(),
        output_transformer_path=str(output_transformer_cache_path),
        quantile_n_quantiles=int(data_config.get('quantile_n_quantiles', 1000)),
        quantile_subsample=int(data_config.get('quantile_subsample', 100000)),
        rank=rank if is_distributed else 0,
        world_size=world_size if is_distributed else 1,
        nrtend_arcsinh_transform=nrtend_arcsinh_transform,
        nrtend_arcsinh_threshold=nrtend_arcsinh_threshold,
        nrtend_regime_threshold=nrtend_regime_threshold,
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
    
    scaler = train_dataset.input_scaler
    return train_loader, val_loader, scaler


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