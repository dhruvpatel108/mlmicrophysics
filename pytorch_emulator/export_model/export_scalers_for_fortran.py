#!/usr/bin/env python3
"""
Export scaler parameters to text files for Fortran integration.

Creates:
- input_scaler_params.txt: mean and scale for input features
- output_scaler_params.txt: mean and scale for output features  
- physical_test_inputs.txt: test inputs in physical (unprocessed) space
- physical_test_outputs.txt: expected outputs in physical space
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Scaler and data paths
SCALER_DIR = Path("/people/pate014/nersc_mlmicro/scaler_cache/deception_distributed_test/run_107666")
TEST_DATA_DIR = Path("/people/pate014/nersc_mlmicro/mlmicrophysics/pytorch_emulator/evaluation_results/run_107666_500samples")
OUTPUT_DIR = Path("/people/pate014/nersc_mlmicro/mlmicrophysics/pytorch_emulator/export_model")

# Feature columns (must match training order)
INPUT_COLS = [
    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
    "PGAM", "LAMC", "LAMR", "N0R", "RHO_CLUBB", "CLOUD", "FREQR"
]
OUTPUT_COLS = ["qrtend_TAU", "nctend_TAU", "nrtend_TAU", "qctend_TAU"]

# Which input columns get log-transformed (indices)
LOG_INPUT_INDICES = [0, 1, 2, 3, 5, 6, 7]  # QC_TAU_in, QR_TAU_in, NC_TAU_in, NR_TAU_in, LAMC, LAMR, N0R

LOG_EPSILON = 1e-10


def load_scalers():
    """Load the sklearn StandardScaler objects."""
    input_scaler_path = SCALER_DIR / "input_scaler_optimized.pkl"
    output_scaler_path = SCALER_DIR / "output_scaler_optimized.pkl"
    
    with open(input_scaler_path, 'rb') as f:
        input_scaler = pickle.load(f)
    
    with open(output_scaler_path, 'rb') as f:
        output_scaler = pickle.load(f)
    
    return input_scaler, output_scaler


def export_scaler_params(input_scaler, output_scaler):
    """Export scaler parameters to text files."""
    # Input scaler
    input_params_file = OUTPUT_DIR / "input_scaler_params.txt"
    with open(input_params_file, 'w') as f:
        f.write(f"# Input scaler parameters for {len(INPUT_COLS)} features\n")
        f.write(f"# Format: mean scale (one line per feature)\n")
        f.write(f"# Log-transformed indices: {LOG_INPUT_INDICES}\n")
        f.write(f"# Feature order: {INPUT_COLS}\n")
        for i, col in enumerate(INPUT_COLS):
            mean = input_scaler.mean_[i]
            scale = input_scaler.scale_[i]
            f.write(f"{mean:.15e} {scale:.15e}\n")
    print(f"Saved input scaler params to: {input_params_file}")
    
    # Output scaler
    output_params_file = OUTPUT_DIR / "output_scaler_params.txt"
    with open(output_params_file, 'w') as f:
        f.write(f"# Output scaler parameters for {len(OUTPUT_COLS)} outputs\n")
        f.write(f"# Format: mean scale (one line per output)\n")
        f.write(f"# Output order: {OUTPUT_COLS}\n")
        for i, col in enumerate(OUTPUT_COLS):
            mean = output_scaler.mean_[i]
            scale = output_scaler.scale_[i]
            f.write(f"{mean:.15e} {scale:.15e}\n")
    print(f"Saved output scaler params to: {output_params_file}")
    
    return input_params_file, output_params_file


def create_physical_test_data(input_scaler, output_scaler):
    """
    Load normalized test data and convert back to physical space.
    This gives us test data that Fortran can use to test full preprocessing pipeline.
    """
    # Load normalized inputs
    inputs_file = TEST_DATA_DIR / "fortran_test_inputs.txt"
    outputs_file = TEST_DATA_DIR / "fortran_test_outputs.txt"
    
    # Read normalized inputs
    with open(inputs_file, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    num_samples, num_features = map(int, header.strip('#').split())
    
    normalized_inputs = np.zeros((num_samples, num_features))
    for i, line in enumerate(lines[1:num_samples+1]):
        normalized_inputs[i] = [float(x) for x in line.split()]
    
    # Read normalized outputs (model predictions)
    with open(outputs_file, 'r') as f:
        lines = f.readlines()
    num_outputs = int(lines[0].strip('#').split()[1])
    
    normalized_outputs = np.zeros((num_samples, num_outputs))
    for i, line in enumerate(lines[1:num_samples+1]):
        normalized_outputs[i] = [float(x) for x in line.split()]
    
    # Inverse transform inputs: inverse scale -> inverse log
    # Step 1: Inverse StandardScaler
    log_space_inputs = input_scaler.inverse_transform(normalized_inputs)
    
    # Step 2: Inverse log transform for specific columns
    physical_inputs = log_space_inputs.copy()
    for idx in LOG_INPUT_INDICES:
        # Inverse of: log10(x + epsilon)  =>  10^x - epsilon
        physical_inputs[:, idx] = np.power(10.0, log_space_inputs[:, idx]) - LOG_EPSILON
    
    # Inverse transform outputs: inverse scale -> inverse log
    # Step 1: Inverse StandardScaler
    log_space_outputs = output_scaler.inverse_transform(normalized_outputs)
    
    # Step 2: Inverse log transform (sign-preserving)
    # Original: sign * log10(|x| + epsilon)
    # Inverse: sign * (10^|y| - epsilon)
    physical_outputs = np.zeros_like(log_space_outputs)
    for j in range(log_space_outputs.shape[1]):
        sign = np.sign(log_space_outputs[:, j])
        abs_val = np.abs(log_space_outputs[:, j])
        physical_outputs[:, j] = sign * (np.power(10.0, abs_val) - LOG_EPSILON)
    
    # Save physical inputs
    physical_inputs_file = OUTPUT_DIR / "physical_test_inputs.txt"
    with open(physical_inputs_file, 'w') as f:
        f.write(f"# {num_samples} {num_features}\n")
        for i in range(num_samples):
            line = ' '.join(f'{v:.15e}' for v in physical_inputs[i])
            f.write(line + '\n')
    print(f"Saved physical test inputs to: {physical_inputs_file}")
    
    # Save physical outputs (expected)
    physical_outputs_file = OUTPUT_DIR / "physical_test_outputs.txt"
    with open(physical_outputs_file, 'w') as f:
        f.write(f"# {num_samples} {num_outputs}\n")
        for i in range(num_samples):
            line = ' '.join(f'{v:.15e}' for v in physical_outputs[i])
            f.write(line + '\n')
    print(f"Saved physical test outputs to: {physical_outputs_file}")
    
    # Also save normalized data for comparison (copy)
    import shutil
    shutil.copy(inputs_file, OUTPUT_DIR / "normalized_test_inputs.txt")
    shutil.copy(outputs_file, OUTPUT_DIR / "normalized_test_outputs.txt")
    print(f"Copied normalized test data to export_model/")
    
    return physical_inputs_file, physical_outputs_file


def main():
    print("="*60)
    print("Exporting Scaler Parameters for Fortran Integration")
    print("="*60)
    
    # Load scalers
    print("\nLoading scalers...")
    input_scaler, output_scaler = load_scalers()
    print(f"  Input scaler: {len(input_scaler.mean_)} features")
    print(f"  Output scaler: {len(output_scaler.mean_)} outputs")
    
    # Export scaler parameters
    print("\nExporting scaler parameters...")
    export_scaler_params(input_scaler, output_scaler)
    
    # Create physical test data
    print("\nCreating physical test data...")
    create_physical_test_data(input_scaler, output_scaler)
    
    print("\n" + "="*60)
    print("Done! Files created in export_model/")
    print("="*60)


if __name__ == "__main__":
    main()


