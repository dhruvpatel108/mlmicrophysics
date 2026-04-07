#!/usr/bin/env python3
"""
Export a standalone TorchScript MoE emulator with:
- embedded input preprocessing
- forced soft routing for nrtend at inference
- embedded physical-space postprocessing
- embedded affine negative-tail nrtend calibration

The exported module is self-contained and designed for E3SM/FTorch-style
single-sample inference:
  input:  1-D physical tensor [11]
  output: 1-D physical tensor [4] in model order:
          [qrtend, nctend, nrtend, qctend]

This exporter targets the validated run_594616 path and intentionally keeps the
implementation clean and narrow rather than extending the older exporter.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.moe_emulator import MoEConstraintAwareEmulator


LOG_EPSILON = 1.0e-10

INPUT_TENSOR_ORDER: Sequence[str] = (
    "QC_TAU_in",
    "QR_TAU_in",
    "NC_TAU_in",
    "NR_TAU_in",
    "PGAM",
    "LAMC",
    "LAMR",
    "N0R",
    "RHO_CLUBB",
    "CLOUD",
    "FREQR",
)

OUTPUT_TENSOR_ORDER: Sequence[str] = (
    "qrtend",
    "nctend",
    "nrtend",
    "qctend",
)

MODEL_OUTPUT_COLS: Sequence[str] = (
    "qrtend_TAU",
    "nctend_TAU",
    "nrtend_TAU",
    "qctend_TAU",
)

# QC_TAU_in, QR_TAU_in, NC_TAU_in, NR_TAU_in, LAMC, LAMR, N0R
LOG_INPUT_INDICES: List[int] = [0, 1, 2, 3, 5, 6, 7]

# qrtend >= 0, nctend <= 0, nrtend unconstrained, qctend <= 0
PHYSICAL_SIGNS: List[int] = [1, -1, 0, -1]

QC_TAU_INDEX = 0
CLOUD_INDEX = 9


def load_full_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Unexpected config format in {config_path}")
    return config


def instantiate_model(model_cfg: Dict[str, Any]) -> nn.Module:
    architecture = str(model_cfg.get("architecture", "")).lower()
    if architecture != "moe":
        raise ValueError(
            f"Expected model.architecture='moe' for this exporter, found '{architecture}'"
        )

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


def resolve_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_scalers(scaler_dir: Path):
    with (scaler_dir / "input_scaler_optimized.pkl").open("rb") as handle:
        input_scaler = pickle.load(handle)
    with (scaler_dir / "output_scaler_optimized.pkl").open("rb") as handle:
        output_scaler = pickle.load(handle)
    return input_scaler, output_scaler


class PreprocessingLayer(nn.Module):
    def __init__(
        self,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        log_indices: List[int],
        epsilon: float = LOG_EPSILON,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_scale", input_scale)
        self.log_indices = log_indices
        self.epsilon = float(epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for idx in self.log_indices:
            x[idx] = torch.log10(x[idx] + self.epsilon)
        return (x - self.input_mean) / self.input_scale


class MoESoftRoutingCore(nn.Module):
    def __init__(self, emulator: nn.Module) -> None:
        super().__init__()
        self.emulator = emulator

    def forward(self, x_batched: torch.Tensor) -> torch.Tensor:
        shared = self.emulator.shared_backbone(x_batched)
        qrtend = self.emulator.qrtend_head(shared)
        nctend = self.emulator.nctend_head(shared)

        router_logits = self.emulator.nrtend_moe.router(shared)
        expert_outputs = torch.cat(
            [expert(shared) for expert in self.emulator.nrtend_moe.experts],
            dim=1,
        )
        gate_probs = torch.softmax(router_logits, dim=1)
        nrtend = (gate_probs * expert_outputs).sum(dim=1, keepdim=True)
        qctend = -qrtend

        return torch.cat([qrtend, nctend, nrtend, qctend], dim=1)


class PostprocessingLayer(nn.Module):
    def __init__(
        self,
        output_mean: torch.Tensor,
        output_scale: torch.Tensor,
        physical_signs: List[int],
        epsilon: float = LOG_EPSILON,
        arcsinh_indices: Optional[List[int]] = None,
        arcsinh_threshold: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("output_scale", output_scale)
        self.physical_signs = physical_signs
        self.epsilon = float(epsilon)
        self.arcsinh_indices = arcsinh_indices or []
        self.arcsinh_threshold = float(arcsinh_threshold)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = y * self.output_scale + self.output_mean
        result = torch.empty_like(y)
        for i, sign in enumerate(self.physical_signs):
            if i in self.arcsinh_indices:
                result[i] = self.arcsinh_threshold * torch.sinh(y[i])
            elif sign > 0:
                raw = torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), y[i]) - self.epsilon
                result[i] = torch.clamp(raw, min=0.0)
            elif sign < 0:
                raw = -(torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), -y[i]) - self.epsilon)
                result[i] = torch.clamp(raw, max=0.0)
            else:
                raw = torch.sign(y[i]) * (
                    torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), torch.abs(y[i])) - self.epsilon
                )
                result[i] = raw
        return result


class NegativeTailAffineCalibration(nn.Module):
    def __init__(
        self,
        thr_1e4: float,
        thr_3e4: float,
        thr_1e5: float,
        slope_1e4_3e4: float,
        intercept_1e4_3e4: float,
        slope_3e4_1e5: float,
        intercept_3e4_1e5: float,
        slope_ge_1e5: float,
        intercept_ge_1e5: float,
    ) -> None:
        super().__init__()
        self.thr_1e4 = float(thr_1e4)
        self.thr_3e4 = float(thr_3e4)
        self.thr_1e5 = float(thr_1e5)
        self.slope_1e4_3e4 = float(slope_1e4_3e4)
        self.intercept_1e4_3e4 = float(intercept_1e4_3e4)
        self.slope_3e4_1e5 = float(slope_3e4_1e5)
        self.intercept_3e4_1e5 = float(intercept_3e4_1e5)
        self.slope_ge_1e5 = float(slope_ge_1e5)
        self.intercept_ge_1e5 = float(intercept_ge_1e5)

    def forward(self, y_physical: torch.Tensor) -> torch.Tensor:
        y = y_physical.clone()
        nrtend = y[2]

        calibrated = torch.where(
            nrtend <= -self.thr_1e5,
            self.slope_ge_1e5 * nrtend + self.intercept_ge_1e5,
            torch.where(
                nrtend <= -self.thr_3e4,
                self.slope_3e4_1e5 * nrtend + self.intercept_3e4_1e5,
                torch.where(
                    nrtend <= -self.thr_1e4,
                    self.slope_1e4_3e4 * nrtend + self.intercept_1e4_3e4,
                    nrtend,
                ),
            ),
        )
        y[2] = calibrated
        return y


class StandaloneMoESoftAffineEmulator(nn.Module):
    def __init__(
        self,
        emulator: nn.Module,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        output_mean: torch.Tensor,
        output_scale: torch.Tensor,
        qc_tau_threshold: float,
        cloud_threshold: float,
        arcsinh_indices: List[int],
        arcsinh_threshold: float,
        calibration_cfg: Dict[str, float],
    ) -> None:
        super().__init__()
        self.preprocessor = PreprocessingLayer(
            input_mean=input_mean,
            input_scale=input_scale,
            log_indices=LOG_INPUT_INDICES,
            epsilon=LOG_EPSILON,
        )
        self.core = MoESoftRoutingCore(emulator)
        self.postprocessor = PostprocessingLayer(
            output_mean=output_mean,
            output_scale=output_scale,
            physical_signs=PHYSICAL_SIGNS,
            epsilon=LOG_EPSILON,
            arcsinh_indices=arcsinh_indices,
            arcsinh_threshold=arcsinh_threshold,
        )
        self.tail_calibration = NegativeTailAffineCalibration(**calibration_cfg)
        self.qc_tau_threshold = float(qc_tau_threshold)
        self.cloud_threshold = float(cloud_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qc_tau_in = x[QC_TAU_INDEX]
        cloud = x[CLOUD_INDEX]
        passes_filter = (qc_tau_in > self.qc_tau_threshold) & (cloud > self.cloud_threshold)

        x_normalized = self.preprocessor(x)
        y_norm = self.core(x_normalized.unsqueeze(0)).squeeze(0)
        y_physical = self.postprocessor(y_norm)
        y_physical = self.tail_calibration(y_physical)

        zeros = torch.zeros_like(y_physical)
        return torch.where(passes_filter, y_physical, zeros)


def build_output_scaler_tensors(
    output_scaler,
    config_output_cols: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[int]]:
    col_to_scaler_idx = {col.replace("_TAU", ""): i for i, col in enumerate(config_output_cols)}
    reorder_indices = [col_to_scaler_idx[name] for name in OUTPUT_TENSOR_ORDER]
    output_mean = torch.tensor(output_scaler.mean_[reorder_indices], dtype=torch.float32, device=device)
    output_scale = torch.tensor(output_scaler.scale_[reorder_indices], dtype=torch.float32, device=device)
    return output_mean, output_scale, reorder_indices


def build_calibration_cfg(calibrator_params: Dict[str, Any]) -> Dict[str, float]:
    if calibrator_params.get("kind") != "piecewise_affine_negative_tail":
        raise ValueError(
            "Expected piecewise_affine_negative_tail calibrator, found "
            f"{calibrator_params.get('kind')}"
        )

    thresholds = calibrator_params["thresholds_abs_pred"]
    coeffs = calibrator_params["coefficients"]
    return {
        "thr_1e4": float(thresholds[0]),
        "thr_3e4": float(thresholds[1]),
        "thr_1e5": float(thresholds[2]),
        "slope_1e4_3e4": float(coeffs["[1.0e+04, 3.0e+04)"]["slope"]),
        "intercept_1e4_3e4": float(coeffs["[1.0e+04, 3.0e+04)"]["intercept"]),
        "slope_3e4_1e5": float(coeffs["[3.0e+04, 1.0e+05)"]["slope"]),
        "intercept_3e4_1e5": float(coeffs["[3.0e+04, 1.0e+05)"]["intercept"]),
        "slope_ge_1e5": float(coeffs["[1.0e+05, inf)"]["slope"]),
        "intercept_ge_1e5": float(coeffs["[1.0e+05, inf)"]["intercept"]),
    }


def inverse_output_transform_manual(
    y_norm_model: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
    arcsinh_indices: List[int],
    arcsinh_threshold: float,
) -> torch.Tensor:
    y = y_norm_model * output_scale + output_mean
    result = torch.empty_like(y)
    for i, sign in enumerate(PHYSICAL_SIGNS):
        if i in arcsinh_indices:
            result[i] = arcsinh_threshold * torch.sinh(y[i])
        elif sign > 0:
            raw = torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), y[i]) - LOG_EPSILON
            result[i] = torch.clamp(raw, min=0.0)
        elif sign < 0:
            raw = -(torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), -y[i]) - LOG_EPSILON)
            result[i] = torch.clamp(raw, max=0.0)
        else:
            result[i] = torch.sign(y[i]) * (
                torch.pow(torch.tensor(10.0, dtype=y.dtype, device=y.device), torch.abs(y[i])) - LOG_EPSILON
            )
    return result


def apply_affine_tail_calibration_manual(y_physical: torch.Tensor, calibration_cfg: Dict[str, float]) -> torch.Tensor:
    y = y_physical.clone()
    nrtend = y[2]
    if nrtend <= -calibration_cfg["thr_1e5"]:
        y[2] = calibration_cfg["slope_ge_1e5"] * nrtend + calibration_cfg["intercept_ge_1e5"]
    elif nrtend <= -calibration_cfg["thr_3e4"]:
        y[2] = calibration_cfg["slope_3e4_1e5"] * nrtend + calibration_cfg["intercept_3e4_1e5"]
    elif nrtend <= -calibration_cfg["thr_1e4"]:
        y[2] = calibration_cfg["slope_1e4_3e4"] * nrtend + calibration_cfg["intercept_1e4_3e4"]
    return y


def verify_standalone_model(
    standalone_model: nn.Module,
    original_model: nn.Module,
    input_mean: torch.Tensor,
    input_scale: torch.Tensor,
    output_mean: torch.Tensor,
    output_scale: torch.Tensor,
    arcsinh_indices: List[int],
    arcsinh_threshold: float,
    calibration_cfg: Dict[str, float],
    qc_tau_threshold: float,
    cloud_threshold: float,
    device: torch.device,
    num_tests: int = 24,
) -> float:
    print("\n[verify] Testing standalone model against manual pipeline...")
    max_diff_overall = 0.0

    for _ in range(num_tests):
        physical_input = torch.rand(11, device=device)
        physical_input[0] = 1.0e-5 + physical_input[0] * 1.0e-4
        physical_input[1] *= 1.0e-5
        physical_input[2] *= 1.0e8
        physical_input[3] *= 1.0e5
        physical_input[4] *= 20.0
        physical_input[5] *= 1.0e6
        physical_input[6] *= 1.0e4
        physical_input[7] *= 1.0e6
        physical_input[8] = 0.5 + physical_input[8]
        physical_input[9] = 0.02 + physical_input[9] * 0.98
        physical_input[10] *= 1.0

        with torch.no_grad():
            standalone_output = standalone_model(physical_input)

            x_manual = physical_input.clone()
            for idx in LOG_INPUT_INDICES:
                x_manual[idx] = torch.log10(x_manual[idx] + LOG_EPSILON)
            x_manual = (x_manual - input_mean) / input_scale

            shared = original_model.shared_backbone(x_manual.unsqueeze(0))
            qrtend = original_model.qrtend_head(shared)
            nctend = original_model.nctend_head(shared)
            router_logits = original_model.nrtend_moe.router(shared)
            expert_outputs = torch.cat(
                [expert(shared) for expert in original_model.nrtend_moe.experts],
                dim=1,
            )
            gate_probs = torch.softmax(router_logits, dim=1)
            nrtend = (gate_probs * expert_outputs).sum(dim=1, keepdim=True)
            qctend = -qrtend
            y_norm = torch.cat([qrtend, nctend, nrtend, qctend], dim=1).squeeze(0)

            manual_output = inverse_output_transform_manual(
                y_norm,
                output_mean=output_mean,
                output_scale=output_scale,
                arcsinh_indices=arcsinh_indices,
                arcsinh_threshold=arcsinh_threshold,
            )
            manual_output = apply_affine_tail_calibration_manual(manual_output, calibration_cfg)

            if not ((physical_input[QC_TAU_INDEX] > qc_tau_threshold) and (physical_input[CLOUD_INDEX] > cloud_threshold)):
                manual_output = torch.zeros_like(manual_output)

        max_diff = torch.max(torch.abs(standalone_output - manual_output)).item()
        max_diff_overall = max(max_diff_overall, max_diff)

    print(f"[verify] max |Δ| between standalone and manual pipeline: {max_diff_overall:.3e}")
    return max_diff_overall


def parse_args() -> argparse.Namespace:
    default_run_dir = PROJECT_ROOT.parent.parent / "outputs" / "moe_nrtend" / "run_594616"
    default_checkpoint = default_run_dir / "best_checkpoint.pth"
    default_config = default_run_dir / "config_used.yml"
    default_scaler_dir = Path("/people/pate014/nersc_mlmicro/scaler_cache/deception_distributed_test/run_107666")
    default_calibrator = (
        PROJECT_ROOT
        / "evaluation_results"
        / "run_594616_soft_tail_calibrated_eval"
        / "calibrator_params.json"
    )
    default_output = default_run_dir / "emulator_moe_594616_soft_affine_validation.pt"

    parser = argparse.ArgumentParser(
        description="Export standalone TorchScript MoE emulator with soft routing and affine tail calibration"
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--scaler-dir", type=Path, default=default_scaler_dir)
    parser.add_argument("--calibrator-json", type=Path, default=default_calibrator)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not args.scaler_dir.exists():
        raise FileNotFoundError(f"Scaler directory not found: {args.scaler_dir}")
    if not args.calibrator_json.exists():
        raise FileNotFoundError(f"Calibrator JSON not found: {args.calibrator_json}")

    full_config = load_full_config(args.config)
    data_cfg = full_config.get("data", {})
    model_cfg = full_config.get("model", {})

    if data_cfg.get("input_transform") != "log10":
        raise ValueError("This exporter currently expects data.input_transform == 'log10'")
    if data_cfg.get("input_scaling") != "standard":
        raise ValueError("This exporter currently expects data.input_scaling == 'standard'")
    if data_cfg.get("output_transform") != "log10":
        raise ValueError("This exporter currently expects data.output_transform == 'log10'")
    if data_cfg.get("output_scaling") != "standard":
        raise ValueError("This exporter currently expects data.output_scaling == 'standard'")

    model = instantiate_model(model_cfg)
    model.to(device)
    model.eval()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(resolve_state_dict(checkpoint))

    input_scaler, output_scaler = load_scalers(args.scaler_dir)
    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
    input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)

    config_output_cols = data_cfg.get("output_cols", [])
    output_mean, output_scale, reorder_indices = build_output_scaler_tensors(
        output_scaler=output_scaler,
        config_output_cols=config_output_cols,
        device=device,
    )

    with args.calibrator_json.open("r") as handle:
        calibrator_params = json.load(handle)
    calibration_cfg = build_calibration_cfg(calibrator_params)

    nrtend_arcsinh_transform = bool(data_cfg.get("nrtend_arcsinh_transform", False))
    nrtend_arcsinh_threshold = float(data_cfg.get("nrtend_arcsinh_threshold", 1.0e-3))
    arcsinh_indices: List[int] = [2] if nrtend_arcsinh_transform else []

    qc_tau_threshold = float(data_cfg.get("mass_input_threshold", 1.0e-6))
    cloud_threshold = float(data_cfg.get("cloud_threshold", 0.01))

    print("=" * 72)
    print("Exporting Standalone MoE Soft-Routing Affine-Calibrated TorchScript Model")
    print("=" * 72)
    print(f"[load] checkpoint:      {args.checkpoint}")
    print(f"[load] config:          {args.config}")
    print(f"[load] scaler dir:      {args.scaler_dir}")
    print(f"[load] calibrator json: {args.calibrator_json}")
    print(f"[load] output:          {args.output}")
    print(f"[info] output scaler reorder indices: {reorder_indices}")
    print(f"[info] QC/CLOUD filter: QC_TAU_in > {qc_tau_threshold}, CLOUD > {cloud_threshold}")
    print(f"[info] arcsinh nrtend: {nrtend_arcsinh_transform} (threshold={nrtend_arcsinh_threshold})")
    print(f"[info] calibration cfg: {calibration_cfg}")

    standalone_model = StandaloneMoESoftAffineEmulator(
        emulator=model,
        input_mean=input_mean,
        input_scale=input_scale,
        output_mean=output_mean,
        output_scale=output_scale,
        qc_tau_threshold=qc_tau_threshold,
        cloud_threshold=cloud_threshold,
        arcsinh_indices=arcsinh_indices,
        arcsinh_threshold=nrtend_arcsinh_threshold,
        calibration_cfg=calibration_cfg,
    )
    standalone_model.to(device)
    standalone_model.eval()

    if not args.skip_verify:
        max_diff = verify_standalone_model(
            standalone_model=standalone_model,
            original_model=model,
            input_mean=input_mean,
            input_scale=input_scale,
            output_mean=output_mean,
            output_scale=output_scale,
            arcsinh_indices=arcsinh_indices,
            arcsinh_threshold=nrtend_arcsinh_threshold,
            calibration_cfg=calibration_cfg,
            qc_tau_threshold=qc_tau_threshold,
            cloud_threshold=cloud_threshold,
            device=device,
        )
        if max_diff > 1.0e-4:
            print("[warn] Verification diff is larger than 1e-4. Inspect before deployment.")
        else:
            print("[verify] ✓ Standalone model matches manual pipeline.")

    dummy_input = torch.rand(11, device=device) * 1.0e-4
    dummy_input[0] = 1.0e-5
    dummy_input[2] *= 1.0e12
    dummy_input[3] *= 1.0e9
    dummy_input[9] = 0.5

    traced = torch.jit.trace(standalone_model, dummy_input, strict=False)
    frozen = torch.jit.freeze(traced)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frozen.save(str(args.output))

    metadata = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "scaler_dir": str(args.scaler_dir),
        "calibrator_json": str(args.calibrator_json),
        "input_order": list(INPUT_TENSOR_ORDER),
        "output_order": list(OUTPUT_TENSOR_ORDER),
        "config_output_cols": list(config_output_cols),
        "model_output_cols": list(MODEL_OUTPUT_COLS),
        "output_scaler_reorder_indices": reorder_indices,
        "input_filter": {
            "qc_tau_threshold": qc_tau_threshold,
            "cloud_threshold": cloud_threshold,
            "failure_behavior": "return zeros",
        },
        "nrtend_postprocess": {
            "soft_routing": True,
            "negative_tail_calibration": calibration_cfg,
            "arcsinh_transform": nrtend_arcsinh_transform,
            "arcsinh_threshold": nrtend_arcsinh_threshold,
        },
    }
    metadata_path = args.output.with_suffix(".json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"\n[done] TorchScript model saved to: {args.output}")
    print(f"[done] Metadata saved to:        {metadata_path}")

    if not args.skip_verify:
        loaded = torch.jit.load(str(args.output), map_location=device)
        with torch.no_grad():
            eager_out = standalone_model(dummy_input)
            scripted_out = loaded(dummy_input)
        load_diff = torch.max(torch.abs(eager_out - scripted_out)).item()
        print(f"[verify] max |Δ| between eager and loaded TorchScript: {load_diff:.3e}")

    print("\nInput tensor [11] physical order:")
    for idx, name in enumerate(INPUT_TENSOR_ORDER):
        print(f"  [{idx:2d}] {name}")

    print("\nOutput tensor [4] physical order:")
    for idx, name in enumerate(OUTPUT_TENSOR_ORDER):
        print(f"  [{idx}] {name}")


if __name__ == "__main__":
    main()
