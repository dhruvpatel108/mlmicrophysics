#!/usr/bin/env python3
"""
Export TorchScript model WITH embedded preprocessing and postprocessing.

This creates a self-contained .pt file that:
- Takes raw PHYSICAL inputs as 1-D tensor [11] (no batch dimension!)
- Returns raw PHYSICAL outputs as 1-D tensor [4] (no batch dimension!)

The preprocessing (log transform + StandardScaler) and postprocessing
(inverse StandardScaler + inverse log transform) are embedded in the model.

This is designed for E3SM/FTorch integration where inputs are passed
one sample at a time without a batch dimension.

Usage:
    python export_model_with_preprocessing.py
    
    # Or with custom paths:
    python export_model_with_preprocessing.py \
        --checkpoint /path/to/checkpoint.pth \
        --config /path/to/config.yml \
        --scaler-dir /path/to/scaler_cache/run_id/ \
        --output emulator_standalone.pt
"""

import argparse
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.physics_emulator import ConstraintAwareEmulator
from models.moe_emulator import MoEConstraintAwareEmulator


# Configuration
LOG_EPSILON = 1e-10

# Which input features get log-transformed (0-indexed)
# QC_TAU_in, QR_TAU_in, NC_TAU_in, NR_TAU_in, LAMC, LAMR, N0R
LOG_INPUT_INDICES = [0, 1, 2, 3, 5, 6, 7]

# All outputs get inverse log transform (sign-preserving)
NUM_OUTPUTS = 4

OUTPUT_TENSOR_ORDER = ("qrtend", "nctend", "nrtend", "qctend")

# Physical sign of each output variable in OUTPUT_TENSOR_ORDER.
# +1 = always positive, -1 = always negative, 0 = ambiguous (use sign(y) fallback).
# qrtend >= 0, nctend <= 0, nrtend = ambiguous, qctend <= 0
PHYSICAL_SIGNS = [+1, -1, 0, -1]

# Input filtering thresholds (samples outside training distribution return zeros)
# These match the filters used during training in streaming_data_loader_v2.py
QC_TAU_THRESHOLD = 1e-6   # QC_TAU_in must be > this value
CLOUD_THRESHOLD = 0.01    # CLOUD must be > this value

# Input indices for filtering
QC_TAU_INDEX = 0   # QC_TAU_in
CLOUD_INDEX = 9    # CLOUD


class PreprocessingLayer(nn.Module):
    """
    Preprocessing layer: log transform + StandardScaler normalization.
    
    Converts physical inputs to normalized inputs for the neural network.
    Works with 1-D input (no batch dimension).
    """
    
    def __init__(self, input_mean: torch.Tensor, input_scale: torch.Tensor, 
                 log_indices: List[int], epsilon: float = 1e-10):
        super().__init__()
        # Register as buffers so they're saved with the model and moved to correct device
        self.register_buffer('input_mean', input_mean)
        self.register_buffer('input_scale', input_scale)
        self.register_buffer('log_indices', torch.tensor(log_indices, dtype=torch.long))
        self.epsilon = epsilon
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Physical input tensor [num_features] (1-D, no batch dim!)
        Returns:
            Normalized tensor ready for neural network [num_features]
        """
        # Clone to avoid modifying input
        x = x.clone()
        
        # Step 1: Log transform specific columns
        for idx in self.log_indices:
            x[idx] = torch.log10(x[idx] + self.epsilon)
        
        # Step 2: StandardScaler: (x - mean) / scale
        x = (x - self.input_mean) / self.input_scale
        
        return x


class PostprocessingLayer(nn.Module):
    """
    Postprocessing layer: inverse StandardScaler + inverse log transform.
    
    Converts normalized outputs back to physical tendencies.
    Works with 1-D output (no batch dimension).

    Uses per-variable exact inverse of  f(x) = sign(x) * log10(|x| + eps):
      x > 0:  f(x) = log10(x + eps)       => x = 10^y - eps
      x < 0:  f(x) = -log10(-x + eps)     => x = -(10^{-y} - eps)
      ambiguous: fall back to sign(y) * (10^|y| - eps)
    
    Physical constraints (eps-independent):
      s > 0 (qrtend):  result = raw if raw > 0 else 0  (rain formation ≥ 0)
      s < 0 (nctend, qctend):  result = raw if raw < 0 else 0  (cloud loss ≤ 0)
      s == 0 (nrtend): no clamping (can be positive or negative)
    
    For variables using arcsinh transform: f(x) = arcsinh(x / c):
      inverse is: x = c * sinh(y')
    """
    
    def __init__(self, output_mean: torch.Tensor, output_scale: torch.Tensor,
                 physical_signs: List[int], epsilon: float = 1e-10,
                 arcsinh_indices: Optional[List[int]] = None,
                 arcsinh_threshold: float = 1e-3):
        super().__init__()
        self.register_buffer('output_mean', output_mean)
        self.register_buffer('output_scale', output_scale)
        self.physical_signs = physical_signs
        self.epsilon = epsilon
        self.arcsinh_indices = arcsinh_indices or []
        self.arcsinh_threshold = arcsinh_threshold
    
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: Normalized output tensor [num_outputs] (1-D, no batch dim!)
        Returns:
            Physical output tensor [num_outputs] (tendencies in physical units)
        """
        # Step 1: Inverse StandardScaler: y * scale + mean
        y = y * self.output_scale + self.output_mean

        # Step 2: Per-variable inverse transform + physical constraints.
        result = torch.empty_like(y)
        for i, s in enumerate(self.physical_signs):
            if i in self.arcsinh_indices:
                # Inverse arcsinh: x = c * sinh(y') (nrtend: no sign constraint)
                result[i] = self.arcsinh_threshold * torch.sinh(y[i])
            elif s > 0:
                raw = torch.pow(10.0, y[i]) - self.epsilon
                result[i] = torch.clamp(raw, min=0.0)  # qrtend ≥ 0
            elif s < 0:
                raw = -(torch.pow(10.0, -y[i]) - self.epsilon)
                result[i] = torch.clamp(raw, max=0.0)  # nctend, qctend ≤ 0
            else:
                result[i] = torch.sign(y[i]) * (torch.pow(10.0, torch.abs(y[i])) - self.epsilon)
                # nrtend: no clamping (can be positive or negative)
        
        return result


class StandaloneEmulator(nn.Module):
    """
    Self-contained emulator with embedded preprocessing and postprocessing.
    
    Input: Physical values as 1-D tensor [11] (NO batch dimension!)
    Output: Physical tendencies as 1-D tensor [4] (NO batch dimension!)
    
    This is designed for E3SM/FTorch where data is passed one sample at a time.
    No external scaler files needed!
    """
    
    def __init__(self, 
                 emulator: nn.Module,
                 input_mean: torch.Tensor,
                 input_scale: torch.Tensor,
                 output_mean: torch.Tensor,
                 output_scale: torch.Tensor,
                 log_input_indices: List[int],
                 physical_signs: List[int],
                 epsilon: float = 1e-10,
                 arcsinh_indices: Optional[List[int]] = None,
                 arcsinh_threshold: float = 1e-3):
        super().__init__()
        
        self.preprocessor = PreprocessingLayer(
            input_mean, input_scale, log_input_indices, epsilon
        )
        self.emulator = emulator
        self.postprocessor = PostprocessingLayer(
            output_mean, output_scale, physical_signs, epsilon,
            arcsinh_indices=arcsinh_indices,
            arcsinh_threshold=arcsinh_threshold
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end inference: physical inputs -> physical outputs.
        
        Includes input filtering: if sample is outside training distribution,
        returns zeros instead of running the DNN.
        
        Args:
            x: Physical input tensor [11] (1-D, NO batch dimension!)
               Features: QC_TAU_in, QR_TAU_in, NC_TAU_in, NR_TAU_in, 
                        PGAM, LAMC, LAMR, N0R, RHO_CLUBB, CLOUD, FREQR
        
        Returns:
            Physical output tensor [4] (1-D, NO batch dimension!)
            Elements: qrtend, nctend, nrtend, qctend
            Returns zeros if input fails filtering criteria.
        """
        # Input filtering on RAW physical values (BEFORE any transforms)
        # These thresholds match the training data filters
        qc_tau_in = x[QC_TAU_INDEX]   # QC_TAU_in at index 0
        cloud = x[CLOUD_INDEX]         # CLOUD at index 9
        
        # Check if sample passes filters (TorchScript-compatible)
        passes_filter = (qc_tau_in > QC_TAU_THRESHOLD) & (cloud > CLOUD_THRESHOLD)
        
        # Always run DNN (needed for tracing), then mask output based on filter
        # Preprocess: physical -> normalized (still 1-D)
        x_normalized = self.preprocessor(x)
        
        # Add batch dimension for neural network: [11] -> [1, 11]
        x_batched = x_normalized.unsqueeze(0)
        
        # Run neural network (expects batch dimension)
        outputs = self.emulator(x_batched)
        
        # Stack outputs in correct order: each is [1, 1], result is [1, 4]
        y_normalized_batched = torch.cat([outputs[name] for name in OUTPUT_TENSOR_ORDER], dim=1)
        
        # Remove batch dimension: [1, 4] -> [4]
        y_normalized = y_normalized_batched.squeeze(0)
        
        # Postprocess: normalized -> physical (1-D)
        y_physical = self.postprocessor(y_normalized)
        
        # Apply filter: return DNN output if passes, zeros otherwise
        # Using torch.where for TorchScript compatibility
        zeros = torch.zeros_like(y_physical)
        y_filtered = torch.where(passes_filter, y_physical, zeros)
        
        return y_filtered


def load_scalers(scaler_dir: Path):
    """Load sklearn StandardScaler objects from pickle files."""
    input_scaler_path = scaler_dir / "input_scaler_optimized.pkl"
    output_scaler_path = scaler_dir / "output_scaler_optimized.pkl"
    
    with open(input_scaler_path, 'rb') as f:
        input_scaler = pickle.load(f)
    
    with open(output_scaler_path, 'rb') as f:
        output_scaler = pickle.load(f)
    
    return input_scaler, output_scaler


def load_full_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load full configuration from YAML file."""
    if config_path is None:
        return {}
    import yaml
    with config_path.open("r") as fp:
        return yaml.safe_load(fp)


def instantiate_model(model_cfg: Dict[str, Any]) -> nn.Module:
    """Create model instance from config.
    
    Supports both standard ConstraintAwareEmulator and MoE architecture.
    """
    architecture = model_cfg.get("architecture", "standard")

    if architecture == "moe":
        moe_cfg = model_cfg.get("moe", {})
        return MoEConstraintAwareEmulator(
            input_dim=model_cfg.get("input_dim", 11),
            shared_dims=model_cfg.get("shared_dims", [256, 256, 256, 128, 128]),
            head_dims=model_cfg.get("head_dims", [128, 128, 64, 64, 32]),
            dropout=0.0,
            activation=model_cfg.get("activation", "relu"),
            n_experts=moe_cfg.get("n_experts", 3),
            expert_hidden_dims=moe_cfg.get("expert_hidden_dims"),
            router_hidden_dims=moe_cfg.get("router_hidden_dims"),
            moe_activation=moe_cfg.get("activation", "silu"),
        )

    ctor_kwargs: Dict[str, Any] = {}
    for key in ("input_dim", "shared_dims", "head_dims", "dropout", "activation"):
        if key in model_cfg:
            ctor_kwargs[key] = model_cfg[key]
    ctor_kwargs["dropout"] = 0.0
    return ConstraintAwareEmulator(**ctor_kwargs)


def resolve_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model state dict from checkpoint."""
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def inverse_output_transform(
    y: torch.Tensor,
    physical_signs: List[int],
    epsilon: float = 1e-10,
    arcsinh_indices: Optional[List[int]] = None,
    arcsinh_threshold: float = 1e-3,
) -> torch.Tensor:
    """Apply the per-output inverse transform after inverse scaling.
    Enforces physical constraints: qrtend ≥ 0, nctend ≤ 0, qctend ≤ 0; nrtend unconstrained."""
    result = torch.empty_like(y)
    arcsinh_index_set = set(arcsinh_indices or [])
    for i, s in enumerate(physical_signs):
        if i in arcsinh_index_set:
            result[i] = arcsinh_threshold * torch.sinh(y[i])
        elif s > 0:
            raw = torch.pow(10.0, y[i]) - epsilon
            result[i] = torch.clamp(raw, min=0.0)  # qrtend ≥ 0
        elif s < 0:
            raw = -(torch.pow(10.0, -y[i]) - epsilon)
            result[i] = torch.clamp(raw, max=0.0)  # nctend, qctend ≤ 0
        else:
            result[i] = torch.sign(y[i]) * (torch.pow(10.0, torch.abs(y[i])) - epsilon)
    return result


def verify_standalone_model(standalone_model: nn.Module, 
                           original_model: nn.Module,
                           input_scaler, output_scaler,
                           reorder_indices: List[int],
                           arcsinh_indices: Optional[List[int]],
                           arcsinh_threshold: float,
                           device: torch.device,
                           num_tests: int = 16):
    """
    Verify that standalone model produces same results as original pipeline.
    Tests multiple 1-D inputs individually.
    All test inputs are generated to PASS the input filters.

    ``reorder_indices`` maps OUTPUT_TENSOR_ORDER positions to the scaler's
    fitted column positions so the manual pipeline matches the standalone model.
    """
    print("\n[verify] Testing standalone model against original pipeline...")
    print(f"[verify] Running {num_tests} test samples (1-D each)...")
    print(f"[verify] Filters: QC_TAU_in > {QC_TAU_THRESHOLD}, CLOUD > {CLOUD_THRESHOLD}")
    print(f"[verify] Output scaler reorder indices: {reorder_indices}")
    if arcsinh_indices:
        print(f"[verify] Arcsinh inverse indices: {arcsinh_indices}, threshold={arcsinh_threshold}")
    else:
        print("[verify] Arcsinh inverse indices: none")
    
    max_diff_overall = 0.0
    
    for test_idx in range(num_tests):
        physical_input = torch.rand(11, device=device)
        physical_input[0] = 1e-5 + physical_input[0] * 1e-4   # QC_TAU_in > 1e-6
        physical_input[1] *= 1e-5   # QR_TAU_in  
        physical_input[2] *= 1e8    # NC_TAU_in
        physical_input[3] *= 1e5    # NR_TAU_in
        physical_input[4] *= 20     # PGAM
        physical_input[5] *= 1e6    # LAMC
        physical_input[6] *= 1e4    # LAMR
        physical_input[7] *= 1e6    # N0R
        physical_input[8] = 0.5 + physical_input[8] * 1.0  # RHO_CLUBB
        physical_input[9] = 0.02 + physical_input[9] * 0.98  # CLOUD > 0.01
        physical_input[10] *= 1.0   # FREQR
        
        with torch.no_grad():
            standalone_output = standalone_model(physical_input)
            
            x_manual = physical_input.clone()
            for idx in LOG_INPUT_INDICES:
                x_manual[idx] = torch.log10(x_manual[idx] + LOG_EPSILON)
            
            input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
            input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
            x_manual = (x_manual - input_mean) / input_scale
            
            x_manual_batched = x_manual.unsqueeze(0)
            outputs = original_model(x_manual_batched)
            y_manual_batched = torch.cat([outputs[name] for name in OUTPUT_TENSOR_ORDER], dim=1)
            y_manual = y_manual_batched.squeeze(0)
            
            output_mean = torch.tensor(output_scaler.mean_[reorder_indices], dtype=torch.float32, device=device)
            output_scale = torch.tensor(output_scaler.scale_[reorder_indices], dtype=torch.float32, device=device)
            y_manual = y_manual * output_scale + output_mean
            y_manual = inverse_output_transform(
                y_manual,
                physical_signs=PHYSICAL_SIGNS,
                epsilon=LOG_EPSILON,
                arcsinh_indices=arcsinh_indices,
                arcsinh_threshold=arcsinh_threshold,
            )
        
        max_diff = torch.max(torch.abs(standalone_output - y_manual)).item()
        max_diff_overall = max(max_diff_overall, max_diff)
    
    print(f"[verify] max |Δ| between standalone and manual pipeline: {max_diff_overall:.3e}")
    
    if max_diff_overall > 1e-4:
        print("[warn] Difference is larger than expected. Check implementation.")
    else:
        print("[verify] ✓ Standalone model matches manual pipeline!")
    
    return max_diff_overall


def main():
    parser = argparse.ArgumentParser(
        description="Export standalone TorchScript model with embedded preprocessing"
    )
    
    sample_run_dir = PROJECT_ROOT.parent.parent / "outputs" / "deception_distributed_test" / "run_572567"
    default_checkpoint = sample_run_dir / "latest_checkpoint.pth"
    default_config = sample_run_dir / "config_used.yml"
    default_scaler_dir = Path("/people/pate014/nersc_mlmicro/scaler_cache/deception_distributed_test/run_107666")
    default_output = sample_run_dir / "emulator572567_qcin1e-6_cloud1e-2_nrtend_asinh_clamped.pt"
    
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--scaler-dir", type=Path, default=default_scaler_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-verify", action="store_true")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Exporting Standalone TorchScript Model")
    print("(with embedded preprocessing/postprocessing)")
    print("="*60)
    
    device = torch.device(args.device)
    
    # Load model
    print(f"\n[load] Checkpoint: {args.checkpoint}")
    print(f"[load] Config: {args.config}")
    print(f"[load] Output: {args.output}")
    
    full_config = load_full_config(args.config)
    model_cfg = full_config.get("model", {})
    model = instantiate_model(model_cfg)
    model.to(device)
    model.eval()
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = resolve_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    
    # Load scalers
    print(f"[load] Scaler dir: {args.scaler_dir}")
    if args.scaler_dir.name == "run_107666":
        print("[info] Treating scaler_cache/.../run_107666 as the effective refit scaler directory for run_572567")
    input_scaler, output_scaler = load_scalers(args.scaler_dir)
    
    # Convert scaler parameters to tensors
    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32)
    input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32)

    # Reorder output scaler params to match OUTPUT_TENSOR_ORDER.
    # The scaler was fitted in config output_cols order (e.g. qctend, nctend, nrtend, qrtend)
    # but the model concatenates heads in OUTPUT_TENSOR_ORDER (qrtend, nctend, nrtend, qctend).
    config_output_cols = full_config.get("data", {}).get("output_cols", [])
    if config_output_cols:
        col_to_scaler_idx = {col.replace("_TAU", ""): i for i, col in enumerate(config_output_cols)}
        reorder_indices = [col_to_scaler_idx[name] for name in OUTPUT_TENSOR_ORDER]
        output_mean = torch.tensor(output_scaler.mean_[reorder_indices], dtype=torch.float32)
        output_scale = torch.tensor(output_scaler.scale_[reorder_indices], dtype=torch.float32)
        print(f"[info] Reordered output scaler to match OUTPUT_TENSOR_ORDER: {list(OUTPUT_TENSOR_ORDER)}")
        print(f"[info]   config output_cols order: {config_output_cols}")
        print(f"[info]   reorder indices: {reorder_indices}")
    else:
        reorder_indices = list(range(len(output_scaler.mean_)))
        output_mean = torch.tensor(output_scaler.mean_, dtype=torch.float32)
        output_scale = torch.tensor(output_scaler.scale_, dtype=torch.float32)
        print("[warn] No output_cols in config; using raw scaler order (may be incorrect)")
    
    print(f"[info] Input features: {len(input_mean)}")
    print(f"[info] Output features: {len(output_mean)}")
    print(f"[info] Log-transformed input indices: {LOG_INPUT_INDICES}")
    
    # Check for arcsinh transform configuration
    data_cfg = full_config.get("data", {})
    nrtend_arcsinh_transform = bool(data_cfg.get("nrtend_arcsinh_transform", False))
    nrtend_arcsinh_threshold = float(data_cfg.get("nrtend_arcsinh_threshold", 1e-3))
    print(f"[info] nrtend_arcsinh_transform: {nrtend_arcsinh_transform}")
    
    # Determine which output indices use arcsinh (in OUTPUT_TENSOR_ORDER)
    arcsinh_indices = []
    if nrtend_arcsinh_transform:
        # nrtend is at index 2 in OUTPUT_TENSOR_ORDER = (qrtend, nctend, nrtend, qctend)
        nrtend_idx = list(OUTPUT_TENSOR_ORDER).index("nrtend")
        arcsinh_indices.append(nrtend_idx)
        print(f"[info] arcsinh transform enabled for nrtend (index {nrtend_idx}), threshold={nrtend_arcsinh_threshold}")
        print(f"[info]   Inverse: y = {nrtend_arcsinh_threshold} * sinh(y')")
    
    # Create standalone wrapper
    standalone_model = StandaloneEmulator(
        emulator=model,
        input_mean=input_mean,
        input_scale=input_scale,
        output_mean=output_mean,
        output_scale=output_scale,
        log_input_indices=LOG_INPUT_INDICES,
        physical_signs=PHYSICAL_SIGNS,
        epsilon=LOG_EPSILON,
        arcsinh_indices=arcsinh_indices,
        arcsinh_threshold=nrtend_arcsinh_threshold
    )
    standalone_model.to(device)
    standalone_model.eval()
    
    # Verify before export
    if not args.skip_verify:
        verify_standalone_model(standalone_model, model, input_scaler, output_scaler,
                                reorder_indices, arcsinh_indices, nrtend_arcsinh_threshold, device)
    
    # Trace and export with 1-D input (no batch dimension!)
    print(f"\n[export] Tracing model with 1-D input...")
    print(f"[export] Input filtering embedded: QC_TAU_in > {QC_TAU_THRESHOLD}, CLOUD > {CLOUD_THRESHOLD}")
    dummy_input = torch.rand(11, device=device) * 1e-4  # 1-D physical values
    dummy_input[0] = 1e-5   # QC_TAU_in > 1e-6 (passes filter)
    dummy_input[2] *= 1e12  # NC needs larger values
    dummy_input[3] *= 1e9   # NR needs larger values
    dummy_input[9] = 0.5    # CLOUD > 0.01 (passes filter)
    
    traced = torch.jit.trace(standalone_model, dummy_input, strict=False)
    frozen = torch.jit.freeze(traced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen.save(str(args.output))
    
    print(f"\n[done] Standalone model saved to: {args.output}")
    print("\n" + "="*60)
    print("INPUT/OUTPUT SPECIFICATION (1-D, NO BATCH DIMENSION!)")
    print("="*60)
    
    print("\n** INPUT FILTERING (embedded in model) **")
    print(f"  If QC_TAU_in <= {QC_TAU_THRESHOLD} OR CLOUD <= {CLOUD_THRESHOLD}:")
    print("    -> Returns zeros [0, 0, 0, 0]")
    print("  Otherwise:")
    print("    -> Runs DNN and returns predictions")
    
    print("\nInput tensor [11] - PHYSICAL VALUES (1-D):")
    input_names = ["QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in",
                   "PGAM", "LAMC", "LAMR", "N0R", "RHO_CLUBB", "CLOUD", "FREQR"]
    for i, name in enumerate(input_names):
        log_marker = " (log-transformed internally)" if i in LOG_INPUT_INDICES else ""
        filter_marker = " ** FILTERED **" if i in [QC_TAU_INDEX, CLOUD_INDEX] else ""
        print(f"  [{i:2d}] {name}{log_marker}{filter_marker}")
    
    print("\nOutput tensor [4] - PHYSICAL TENDENCIES (1-D):")
    for i, name in enumerate(OUTPUT_TENSOR_ORDER):
        print(f"  [{i}] {name}")
    
    print("\n" + "="*60)
    print("USAGE IN FORTRAN (1-D arrays, no batch dimension!)")
    print("="*60)
    print("""
! 1-D arrays - NO batch dimension needed!
real(real32), target :: physical_input(11)
real(real32), target :: physical_output(4)

! Set physical input values (raw E3SM variables)
physical_input(1) = qc_value    ! QC_TAU_in
physical_input(2) = qr_value    ! QR_TAU_in
physical_input(3) = nc_value    ! NC_TAU_in
physical_input(4) = nr_value    ! NR_TAU_in
physical_input(5) = pgam_value  ! PGAM
physical_input(6) = lamc_value  ! LAMC
physical_input(7) = lamr_value  ! LAMR
physical_input(8) = n0r_value   ! N0R
physical_input(9) = rho_value   ! RHO_CLUBB
physical_input(10) = cloud_value ! CLOUD
physical_input(11) = freqr_value ! FREQR

! Run inference
call torch_model_forward(model, input_tensors, output_tensors)

! physical_output now contains tendencies in physical units!
! No postprocessing needed!
qrtend = physical_output(1)
nctend = physical_output(2)
nrtend = physical_output(3)
qctend = physical_output(4)
""")
    
    # Verify loaded model with 1-D input
    if not args.skip_verify:
        print("\n[verify] Testing loaded TorchScript model with 1-D input...")
        loaded = torch.jit.load(str(args.output), map_location=device)
        
        # Test with 1-D input (no batch dimension)
        # Ensure test input PASSES filters
        test_input = torch.rand(11, device=device) * 1e-4
        test_input[0] = 1e-5   # QC_TAU_in > 1e-6 (passes filter)
        test_input[2] *= 1e12  # NC needs larger values
        test_input[3] *= 1e9   # NR needs larger values
        test_input[9] = 0.5    # CLOUD > 0.01 (passes filter)
        
        with torch.no_grad():
            original_out = standalone_model(test_input)
            loaded_out = loaded(test_input)
        
        load_diff = torch.max(torch.abs(original_out - loaded_out)).item()
        print(f"[verify] max |Δ| between original and loaded: {load_diff:.3e}")
        print(f"[verify] Input shape: {test_input.shape} (should be [11])")
        print(f"[verify] Output shape: {loaded_out.shape} (should be [4])")
        
        if load_diff < 1e-6:
            print("[verify] ✓ Loaded model matches original!")


if __name__ == "__main__":
    main()
