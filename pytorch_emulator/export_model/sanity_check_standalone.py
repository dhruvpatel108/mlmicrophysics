#!/usr/bin/env python3
"""
Sanity check for standalone emulator.

Takes 100 random samples from test set, runs inference through the standalone
emulator (which has embedded preprocessing/postprocessing), and logs results
to a CSV file.

Also verifies that the embedded input filtering works correctly.

Output CSV columns:
- 11 input feature columns (physical values)
- For each of 4 tendencies: true_value, predicted_value, percent_error
"""

import sys
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SCALER_DIR = Path("/people/pate014/nersc_mlmicro/scaler_cache/deception_distributed_test/run_107666")
TEST_DATA_DIR = THIS_DIR  # Use the existing test data files

# Configuration
NUM_SAMPLES = 100
SEED = 42
LOG_EPSILON = 1e-10
LOG_INPUT_INDICES = [0, 1, 2, 3, 5, 6, 7]

# Filter thresholds (must match export_model_with_preprocessing.py)
QC_TAU_THRESHOLD = 1e-6
CLOUD_THRESHOLD = 0.01

# Column names
INPUT_COLS = [
    "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
    "PGAM", "LAMC", "LAMR", "N0R", "RHO_CLUBB", "CLOUD", "FREQR"
]
OUTPUT_COLS = ["qrtend", "nctend", "nrtend", "qctend"]


def load_scalers():
    """Load sklearn StandardScaler objects."""
    with open(SCALER_DIR / "input_scaler_optimized.pkl", 'rb') as f:
        input_scaler = pickle.load(f)
    with open(SCALER_DIR / "output_scaler_optimized.pkl", 'rb') as f:
        output_scaler = pickle.load(f)
    return input_scaler, output_scaler


def normalized_to_physical_inputs(normalized_inputs, input_scaler):
    """Convert normalized inputs back to physical space."""
    # Inverse StandardScaler
    log_space = input_scaler.inverse_transform(normalized_inputs)
    
    # Inverse log transform for specific columns
    physical = log_space.copy()
    for idx in LOG_INPUT_INDICES:
        physical[:, idx] = np.power(10.0, log_space[:, idx]) - LOG_EPSILON
    
    return physical


def normalized_to_physical_outputs(normalized_outputs, output_scaler):
    """Convert normalized outputs back to physical space."""
    # Inverse StandardScaler
    log_space = output_scaler.inverse_transform(normalized_outputs)
    
    # Inverse log transform (sign-preserving)
    sign = np.sign(log_space)
    abs_val = np.abs(log_space)
    physical = sign * (np.power(10.0, abs_val) - LOG_EPSILON)
    
    return physical


def main():
    print("="*70)
    print("Sanity Check: Standalone Emulator")
    print("="*70)
    
    # Load standalone emulator
    model_path = THIS_DIR / "emulator_standalone.pt"
    print(f"\nLoading standalone emulator: {model_path}")
    model = torch.jit.load(str(model_path), map_location='cpu')
    model.eval()
    print("✓ Model loaded")
    
    # Load scalers (needed to convert test data to physical space)
    print("\nLoading scalers...")
    input_scaler, output_scaler = load_scalers()
    print("✓ Scalers loaded")
    
    # Load normalized test data
    print("\nLoading test data...")
    
    # Try to load from existing files
    normalized_inputs_file = TEST_DATA_DIR / "normalized_test_inputs.txt"
    normalized_outputs_file = TEST_DATA_DIR / "normalized_test_outputs.txt"
    
    if not normalized_inputs_file.exists():
        print(f"ERROR: {normalized_inputs_file} not found. Run export_scalers_for_fortran.py first.")
        sys.exit(1)
    
    # Read normalized inputs
    with open(normalized_inputs_file, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    total_samples, num_features = map(int, header.strip('#').split())
    
    normalized_inputs = np.zeros((total_samples, num_features))
    for i, line in enumerate(lines[1:total_samples+1]):
        normalized_inputs[i] = [float(x) for x in line.split()]
    
    # Read normalized outputs (ground truth)
    with open(normalized_outputs_file, 'r') as f:
        lines = f.readlines()
    num_outputs = int(lines[0].strip('#').split()[1])
    
    normalized_outputs = np.zeros((total_samples, num_outputs))
    for i, line in enumerate(lines[1:total_samples+1]):
        normalized_outputs[i] = [float(x) for x in line.split()]
    
    print(f"✓ Loaded {total_samples} samples")
    
    # Select random samples
    np.random.seed(SEED)
    if total_samples < NUM_SAMPLES:
        print(f"Warning: Only {total_samples} samples available, using all")
        indices = np.arange(total_samples)
    else:
        indices = np.random.choice(total_samples, size=NUM_SAMPLES, replace=False)
    
    print(f"✓ Selected {len(indices)} random samples")
    
    # Convert to physical space
    print("\nConverting to physical space...")
    physical_inputs = normalized_to_physical_inputs(normalized_inputs[indices], input_scaler)
    physical_outputs_true = normalized_to_physical_outputs(normalized_outputs[indices], output_scaler)
    print("✓ Conversion complete")
    
    # Run inference with standalone model (now expects 1-D input per sample)
    print("\nRunning inference with standalone emulator (1-D mode)...")
    physical_outputs_pred = np.zeros((len(indices), 4), dtype=np.float32)
    
    with torch.no_grad():
        for i in range(len(indices)):
            # Input is 1-D: [11]
            input_tensor = torch.tensor(physical_inputs[i], dtype=torch.float32)
            # Output is 1-D: [4]
            output_tensor = model(input_tensor)
            physical_outputs_pred[i] = output_tensor.numpy()
    
    print("✓ Inference complete")
    
    # Calculate percent errors
    print("\nCalculating errors...")
    percent_errors = np.zeros_like(physical_outputs_true)
    for j in range(4):
        true_vals = physical_outputs_true[:, j]
        pred_vals = physical_outputs_pred[:, j]
        
        # Avoid division by zero
        safe_denom = np.where(np.abs(true_vals) < 1e-20, 1e-20, np.abs(true_vals))
        percent_errors[:, j] = np.abs(pred_vals - true_vals) / safe_denom * 100
    
    # Build DataFrame
    print("\nBuilding results DataFrame...")
    
    data = {}
    
    # Input columns
    for i, col in enumerate(INPUT_COLS):
        data[col] = physical_inputs[:, i]
    
    # Output columns: true, pred, %error for each tendency
    for j, col in enumerate(OUTPUT_COLS):
        data[f"{col}_true"] = physical_outputs_true[:, j]
        data[f"{col}_pred"] = physical_outputs_pred[:, j]
        data[f"{col}_pct_error"] = percent_errors[:, j]
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_csv = THIS_DIR / "sanity_check_results.csv"
    df.to_csv(output_csv, index=False, float_format='%.10e')
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    print("\nPercent Error Statistics by Tendency:")
    print("-"*50)
    for col in OUTPUT_COLS:
        errors = df[f"{col}_pct_error"]
        print(f"\n{col}:")
        print(f"  Mean:   {errors.mean():12.4f}%")
        print(f"  Median: {errors.median():12.4f}%")
        print(f"  Max:    {errors.max():12.4f}%")
        print(f"  Min:    {errors.min():12.4f}%")
        print(f"  Std:    {errors.std():12.4f}%")
    
    # Overall statistics
    all_errors = np.concatenate([df[f"{col}_pct_error"].values for col in OUTPUT_COLS])
    print("\n" + "-"*50)
    print("Overall (all tendencies):")
    print(f"  Mean:   {all_errors.mean():12.4f}%")
    print(f"  Median: {np.median(all_errors):12.4f}%")
    print(f"  Max:    {all_errors.max():12.4f}%")
    
    # Sample output
    print("\n" + "="*70)
    print("SAMPLE OUTPUT (first 5 rows)")
    print("="*70)
    print(df.head().to_string())
    
    # Test filtering logic
    print("\n" + "="*70)
    print("FILTER VERIFICATION")
    print("="*70)
    print(f"\nFilter thresholds: QC_TAU_in > {QC_TAU_THRESHOLD}, CLOUD > {CLOUD_THRESHOLD}")
    
    test_cases = [
        # (QC_TAU_in, CLOUD, description, should_pass)
        (1e-5, 0.5, "Both pass", True),
        (1e-7, 0.5, "QC_TAU_in fails", False),
        (1e-5, 0.005, "CLOUD fails", False),
        (1e-7, 0.005, "Both fail", False),
        (1e-6, 0.01, "Both at boundary (fail)", False),
        (1e-5, 0.02, "Both pass (above boundary)", True),
    ]
    
    print("\nTesting filter logic:")
    all_filter_tests_passed = True
    
    with torch.no_grad():
        for qc_val, cloud_val, desc, should_pass in test_cases:
            # Create test input with specific QC_TAU_in and CLOUD values
            test_input = torch.rand(11, dtype=torch.float32)
            test_input[0] = qc_val   # QC_TAU_in
            test_input[1] = 1e-5     # QR_TAU_in
            test_input[2] = 1e8      # NC_TAU_in
            test_input[3] = 1e5      # NR_TAU_in
            test_input[4] = 10.0     # PGAM
            test_input[5] = 1e6      # LAMC
            test_input[6] = 1e4      # LAMR
            test_input[7] = 1e6      # N0R
            test_input[8] = 0.5      # RHO_CLUBB
            test_input[9] = cloud_val  # CLOUD
            test_input[10] = 0.5     # FREQR
            
            output = model(test_input)
            is_zeros = torch.allclose(output, torch.zeros(4), atol=1e-10)
            actual_pass = not is_zeros
            
            status = "✓" if actual_pass == should_pass else "✗"
            output_desc = "DNN output" if actual_pass else "zeros"
            expected_desc = "DNN output" if should_pass else "zeros"
            
            print(f"  {status} {desc}: QC={qc_val:.1e}, CLOUD={cloud_val:.3f}")
            print(f"      Expected: {expected_desc}, Got: {output_desc}")
            
            if actual_pass != should_pass:
                all_filter_tests_passed = False
    
    if all_filter_tests_passed:
        print("\n✓ All filter tests passed!")
    else:
        print("\n✗ Some filter tests failed!")
    
    print("\n" + "="*70)
    print("Sanity check complete!")
    print("="*70)


if __name__ == "__main__":
    main()

