#!/usr/bin/env python3
"""
Compute Comprehensive Scaler Statistics for Microphysics Emulator

This script:
1. Computes scaler statistics on a large sample of the training dataset.
2. Calculates statistics at three key preprocessing stages:
   a. Raw data
   b. Log-transformed data
   c. Standardized (scaled) data
3. Validates the fitted scaler on a sample of the validation data.
4. Generates a detailed, human-readable report.

This version uses a streaming approach to handle large datasets without
running out of memory.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
import pickle
import json
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StatsAccumulator:
    """Helper class to compute running statistics in a memory-efficient way."""
    def __init__(self, columns: List[str]):
        self.columns = columns
        self.count = 0
        self.sums = pd.Series(0.0, index=columns, dtype=np.float64)
        self.sum_sqs = pd.Series(0.0, index=columns, dtype=np.float64)
        self.mins = pd.Series(np.inf, index=columns, dtype=np.float64)
        self.maxs = pd.Series(-np.inf, index=columns, dtype=np.float64)

    def update(self, df_chunk: pd.DataFrame):
        """Update running stats with a new chunk of data."""
        if df_chunk.empty:
            return
        chunk = df_chunk[self.columns].astype(np.float64)
        self.count += len(chunk)
        self.sums += chunk.sum()
        self.sum_sqs += (chunk**2).sum()
        self.mins = np.minimum(self.mins, chunk.min())
        self.maxs = np.maximum(self.maxs, chunk.max())

    def finalize(self) -> Dict:
        """Finalize computation and return the statistics dictionary."""
        if self.count == 0:
            zeros = pd.Series(0.0, index=self.columns).to_dict()
            return {'min': zeros, 'max': zeros, 'mean': zeros, 'std': zeros, 'count': 0}
        
        mean = self.sums / self.count
        # Var(X) = E[X^2] - (E[X])^2
        var = (self.sum_sqs / self.count) - (mean**2)
        std = np.sqrt(var.clip(lower=0)) # Clip to avoid negative variance from precision errors
        
        return {
            'min': self.mins.to_dict(),
            'max': self.maxs.to_dict(),
            'mean': mean.to_dict(),
            'std': std.to_dict(),
            'count': self.count
        }

class ScalerStatsComputer:
    """Computes and reports comprehensive scaler statistics using a streaming approach."""
    
    def __init__(
        self,
        data_path: str = "/pscratch/sd/d/dvpatel/mlmicrophysics_project/e3sm/processed_data/",
        output_dir: str = "./scaler_stats",
        train_fraction: float = 0.8,
        train_files_sample_fraction: float = 0.2,
        val_files_sample_count: int = 10,
        chunk_size: int = 50000
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.train_fraction = train_fraction
        self.train_files_sample_fraction = train_files_sample_fraction
        self.val_files_sample_count = val_files_sample_count
        self.chunk_size = chunk_size
        
        self.input_cols = [
            'QC_TAU_in', 'QR_TAU_in', 'NC_TAU_in', 'NR_TAU_in',
            'PGAM', 'LAMC', 'LAMR', 'N0R', 'RHO_CLUBB', 'CLOUD', 'FREQR'
        ]
        self.output_cols = [
            'qctend_TAU', 'nctend_TAU', 'nrtend_TAU', 'qrtend_TAU'
        ]
        self.log_transform_cols = [
            "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", 
            "LAMC", "LAMR", "N0R"
        ]
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.stats = self._get_empty_stats_dict()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_empty_stats_dict(self) -> Dict:
        """Initializes the nested dictionary to store all statistics."""
        return {
            "training": {"raw": {}, "log": {}, "scaled": {}},
            "validation": {"raw": {}, "log": {}, "scaled": {}}
        }

    def find_parquet_files(self) -> Tuple[List[Path], List[Path]]:
        """Find and split parquet files into train/val."""
        parquet_files = sorted(list(self.data_path.glob("*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.data_path}")
        
        n_train_files = int(len(parquet_files) * self.train_fraction)
        train_files = parquet_files[:n_train_files]
        val_files = parquet_files[n_train_files:]
        
        logger.info(f"Found {len(parquet_files)} total files.")
        logger.info(f"Using {len(train_files)} for training, {len(val_files)} for validation.")
        return train_files, val_files
    
    def log_transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Applies log transformation to a chunk."""
        chunk = chunk.copy()
        for col in self.log_transform_cols:
            if col in chunk.columns:
                chunk[col] = np.log10(np.maximum(chunk[col], 1e-10))
        
        for col in self.output_cols:
            if col in chunk.columns:
                sign = np.sign(chunk[col])
                abs_val = np.abs(chunk[col]) + 1e-10
                chunk[col] = sign * np.log10(abs_val)
        return chunk

    def compute_training_stats(self, train_files: List[Path]) -> None:
        """Computes stats for the training set and fits the scaler using a streaming method."""
        logger.info("🔄 Pass 1/2: Computing Raw/Log stats and fitting scaler on training data...")
        
        sample_size = int(len(train_files) * self.train_files_sample_fraction)
        train_files_sample = np.random.choice(train_files, size=max(1, sample_size), replace=False)
        logger.info(f"Using a sample of {len(train_files_sample)} training files for stats computation.")

        raw_accumulator = StatsAccumulator(self.input_cols + self.output_cols)
        log_accumulator = StatsAccumulator(self.input_cols + self.output_cols)

        for i, file_path in enumerate(train_files_sample):
            logger.debug(f"Processing training file {i+1}/{len(train_files_sample)}: {file_path.name}")
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=self.input_cols + self.output_cols):
                raw_chunk = batch.to_pandas()
                raw_accumulator.update(raw_chunk)
                log_chunk = self.log_transform_chunk(raw_chunk)
                log_accumulator.update(log_chunk)
                self.scaler.partial_fit(log_chunk[self.input_cols])

        self.stats['training']['raw'] = raw_accumulator.finalize()
        self.stats['training']['log'] = log_accumulator.finalize()
        
        logger.info("🔄 Pass 2/2: Computing Scaled stats for training data...")
        scaled_accumulator = StatsAccumulator(self.input_cols + self.output_cols)
        for i, file_path in enumerate(train_files_sample):
            logger.debug(f"Processing training file {i+1}/{len(train_files_sample)} for scaling pass.")
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=self.input_cols + self.output_cols):
                raw_chunk = batch.to_pandas()
                log_chunk = self.log_transform_chunk(raw_chunk)
                scaled_inputs = self.scaler.transform(log_chunk[self.input_cols])
                scaled_df = pd.DataFrame(scaled_inputs, columns=self.input_cols)
                for col in self.output_cols:
                    scaled_df[col] = log_chunk[col]
                scaled_accumulator.update(scaled_df)

        self.stats['training']['scaled'] = scaled_accumulator.finalize()
        logger.info("✅ Training statistics computed.")

    def compute_validation_stats(self, val_files: List[Path]) -> None:
        """Computes stats for the validation set using the fitted scaler in a streaming fashion."""
        logger.info(f"🔍 Validating scaler on a sample of {self.val_files_sample_count} validation files...")
        
        val_files_sample = np.random.choice(val_files, size=min(self.val_files_sample_count, len(val_files)), replace=False)
        
        raw_accumulator = StatsAccumulator(self.input_cols + self.output_cols)
        log_accumulator = StatsAccumulator(self.input_cols + self.output_cols)
        scaled_accumulator = StatsAccumulator(self.input_cols + self.output_cols)

        for file_path in val_files_sample:
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size, columns=self.input_cols + self.output_cols):
                raw_chunk = batch.to_pandas()
                # Stage A
                raw_accumulator.update(raw_chunk)
                # Stage B
                log_chunk = self.log_transform_chunk(raw_chunk)
                log_accumulator.update(log_chunk)
                # Stage C
                scaled_inputs = self.scaler.transform(log_chunk[self.input_cols])
                scaled_df = pd.DataFrame(scaled_inputs, columns=self.input_cols)
                for col in self.output_cols:
                    scaled_df[col] = log_chunk[col]
                scaled_accumulator.update(scaled_df)

        self.stats['validation']['raw'] = raw_accumulator.finalize()
        self.stats['validation']['log'] = log_accumulator.finalize()
        self.stats['validation']['scaled'] = scaled_accumulator.finalize()
        logger.info("✅ Validation statistics computed.")
        
    def save_results(self) -> None:
        """Saves scaler and all statistics to files."""
        scaler_path = self.output_dir / f"fitted_scaler_{self.timestamp}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"💾 Scaler saved to {scaler_path}")

        stats_path = self.output_dir / f"full_stats_{self.timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2, cls=NpEncoder)
        logger.info(f"📊 Full statistics saved to {stats_path}")

        self.create_summary_report()

    def create_summary_report(self) -> None:
        """Creates a human-readable summary report."""
        report_path = self.output_dir / f"scaler_report_{self.timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SCALER STATISTICS REPORT (Streaming Method)\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            for split in ["training", "validation"]:
                f.write("#" * 30 + f" {split.upper()} DATA STATISTICS " + "#" * 30 + "\n")
                count = self.stats[split]['raw'].get('count', 0)
                f.write(f"Total samples processed: {count:,}\n\n")
                
                for stage in ["raw", "log", "scaled"]:
                    f.write("-*-*-*-*^^^^^^^^-*-*-*-* STAGE: " + stage.upper() + " -*-*-*-*^^^^^^^^-*-*-*-*\n")
                    
                    if not self.stats[split][stage]:
                        f.write("No stats available for this stage.\n\n")
                        continue

                    f.write("\n--- INPUT FEATURES ---\n")
                    f.write(f"{'Feature':<15} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15}\n")
                    f.write("-" * 75 + "\n")
                    for col in self.input_cols:
                        s = self.stats[split][stage]
                        f.write(f"{col:<15} {s['min'].get(col, 0):<15.4e} {s['max'].get(col, 0):<15.4e} "
                                f"{s['mean'].get(col, 0):<15.4e} {s['std'].get(col, 0):<15.4e}\n")
                    
                    f.write("\n--- OUTPUT FEATURES ---\n")
                    f.write(f"{'Feature':<15} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15}\n")
                    f.write("-" * 75 + "\n")
                    for col in self.output_cols:
                        s = self.stats[split][stage]
                        f.write(f"{col:<15} {s['min'].get(col, 0):<15.4e} {s['max'].get(col, 0):<15.4e} "
                                f"{s['mean'].get(col, 0):<15.4e} {s['std'].get(col, 0):<15.4e}\n")
                    f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("SCALER PARAMETERS (Fitted on Log-Transformed Training Data)\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Feature':<15} {'Scaler Mean':<20} {'Scaler Scale (Std)':<20}\n")
            f.write("-" * 60 + "\n")
            if hasattr(self.scaler, 'mean_'):
                for i, col in enumerate(self.input_cols):
                    f.write(f"{col:<15} {self.scaler.mean_[i]:<20.6f} {self.scaler.scale_[i]:<20.6f}\n")

        logger.info(f"📄 Summary report saved to {report_path}")

    def print_summary(self) -> None:
        """Prints the summary report to the console."""
        report_path = self.output_dir / f"scaler_report_{self.timestamp}.txt"
        if report_path.exists():
            with open(report_path, 'r') as f:
                print(f.read())
        else:
            logger.error("Summary report not found. Please run computation first.")

    def run(self):
        """Execute the full statistics computation workflow."""
        train_files, val_files = self.find_parquet_files()
        self.compute_training_stats(train_files)
        self.compute_validation_stats(val_files)
        self.save_results()
        self.print_summary()

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main():
    """Main function to compute scaler statistics."""
    logger.info("🚀 Starting Scaler Statistics Computation (Memory-Efficient Mode)...")
    stats_computer = ScalerStatsComputer()
    stats_computer.run()
    logger.info("✅ Scaler statistics computation completed!")

if __name__ == "__main__":
    main() 