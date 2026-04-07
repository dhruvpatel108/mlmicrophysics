#!/usr/bin/env python3
"""
Detailed regime-conditional physical-magnitude analysis for nrtend.

This script is meant to complement evaluate_moe_validation.py by focusing only on
physical-space nrtend behavior, broken down by:
- target regime: near_zero / negative / positive
- physical magnitude bins on |true nrtend| using near-equal log-space x3 bins

Key behavior:
- loads the raw checkpoint, not a TorchScript export
- uses the same validation split as the codebase
- applies the target-only rule:
      if |nrtend| < nrtend_regime_threshold, set target nrtend = 0
- computes per-regime and per-magnitude-bin diagnostics in physical space

Outputs:
- nrtend_bin_metrics.csv
- nrtend_regime_summary.csv
- nrtend_bin_counts.png
- nrtend_bin_rmse_mae.png
- nrtend_bin_bias.png
- nrtend_bin_logmag_ratio.png
- nrtend_positive_mag_calibration.png
- nrtend_negative_mag_calibration.png
- nrtend_near_zero_mag_calibration.png
- nrtend_bin_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.moe_emulator import MoEConstraintAwareEmulator
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_OUTPUT_COLS: Tuple[str, ...] = (
    "qrtend_TAU",
    "nctend_TAU",
    "nrtend_TAU",
    "qctend_TAU",
)
MODEL_PRED_KEYS: Tuple[str, ...] = ("qrtend", "nctend", "nrtend", "qctend")
LOG_OUTPUT_COLS = {"qctend_TAU", "nctend_TAU", "nrtend_TAU", "qrtend_TAU"}
LOG_EPSILON = 1.0e-10
REGIME_LABELS = ("near_zero", "negative", "positive")
PHYSICAL_SIGN_BY_COL = {
    "qrtend_TAU": +1,
    "nctend_TAU": -1,
    "nrtend_TAU": 0,
    "qctend_TAU": -1,
}

# Near-equal log-space x3 bins for |true nrtend|
MAG_BIN_EDGES = np.array([
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
    3e-1,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1e3,
    3e3,
    1e4,
    3e4,
    1e5,
    np.inf,
], dtype=np.float64)


def normalize_run_id(run_id: Optional[str]) -> Optional[str]:
    if run_id is None:
        return None
    run_id = str(run_id).strip()
    if not run_id:
        return None
    return run_id if run_id.startswith("run_") else f"run_{run_id}"


def infer_run_id_from_path(path_like: str) -> Optional[str]:
    try:
        path = Path(path_like)
    except Exception:
        return None
    for part in reversed(path.parts):
        if part.startswith("run_"):
            return part
    return None


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_state_dict(checkpoint: Dict) -> Dict:
    for key in ("model_state_dict", "state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def instantiate_moe_model(model_cfg: Dict) -> torch.nn.Module:
    moe_cfg = model_cfg.get("moe", {})
    return MoEConstraintAwareEmulator(
        input_dim=model_cfg.get("input_dim", 11),
        shared_dims=model_cfg.get("shared_dims", [256, 256, 256, 128, 128]),
        head_dims=model_cfg.get("head_dims", [128, 128, 64, 64, 32]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=model_cfg.get("activation", "relu"),
        n_experts=moe_cfg.get("n_experts", 3),
        expert_hidden_dims=moe_cfg.get("expert_hidden_dims"),
        router_hidden_dims=moe_cfg.get("router_hidden_dims"),
        moe_activation=moe_cfg.get("activation", "silu"),
    )


def load_model(checkpoint_path: Path, config: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    architecture = str(model_cfg.get("architecture", "")).lower()
    if architecture != "moe":
        raise ValueError(
            f"analyze_nrtend_magnitude_bins.py expects model.architecture='moe', found '{architecture}'"
        )

    model = instantiate_moe_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(resolve_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def dataset_to_model_indices(dataset_cols: Sequence[str]) -> List[int]:
    return [dataset_cols.index(col) for col in MODEL_OUTPUT_COLS]


def model_to_dataset_matrix(matrix_model: np.ndarray, dataset_cols: Sequence[str]) -> np.ndarray:
    dataset_matrix = np.empty((matrix_model.shape[0], len(dataset_cols)), dtype=np.float64)
    for model_idx, col in enumerate(MODEL_OUTPUT_COLS):
        dataset_idx = dataset_cols.index(col)
        dataset_matrix[:, dataset_idx] = matrix_model[:, model_idx]
    return dataset_matrix


def dataset_to_model_matrix(matrix_dataset: np.ndarray, dataset_cols: Sequence[str]) -> np.ndarray:
    return matrix_dataset[:, dataset_to_model_indices(dataset_cols)]


def inverse_output_pipeline(
    normalized_dataset_matrix: np.ndarray,
    dataset_cols: Sequence[str],
    dataset,
) -> np.ndarray:
    restored = np.asarray(normalized_dataset_matrix, dtype=np.float64).copy()
    scaler = getattr(dataset, "output_scaler", None)
    transformer = getattr(dataset, "output_transformer", None)
    output_transform = getattr(dataset, "output_transform", "log10")
    use_nrtend_arcsinh = bool(getattr(dataset, "nrtend_arcsinh_transform", False))
    arcsinh_threshold = float(getattr(dataset, "nrtend_arcsinh_threshold", 1.0e-3))

    if scaler is not None:
        restored = scaler.inverse_transform(restored)
    if output_transform == "quantile" and transformer is not None:
        restored = transformer.inverse_transform(restored)

    if use_nrtend_arcsinh and "nrtend_TAU" in dataset_cols:
        nrt_idx = dataset_cols.index("nrtend_TAU")
        restored[:, nrt_idx] = arcsinh_threshold * np.sinh(restored[:, nrt_idx])

    if output_transform == "log10":
        for idx, col in enumerate(dataset_cols):
            if use_nrtend_arcsinh and col == "nrtend_TAU":
                continue
            if col not in LOG_OUTPUT_COLS:
                continue
            values = restored[:, idx]
            physical_sign = PHYSICAL_SIGN_BY_COL.get(col, 0)
            if physical_sign > 0:
                restored[:, idx] = np.power(10.0, values) - LOG_EPSILON
            elif physical_sign < 0:
                restored[:, idx] = -(np.power(10.0, -values) - LOG_EPSILON)
            else:
                restored[:, idx] = np.sign(values) * (
                    np.power(10.0, np.abs(values)) - LOG_EPSILON
                )

    return restored


def zero_small_nrtend_targets(
    physical_dataset_matrix: np.ndarray,
    dataset_cols: Sequence[str],
    threshold: Optional[float],
) -> np.ndarray:
    adjusted = np.asarray(physical_dataset_matrix, dtype=np.float64).copy()
    if threshold is None or "nrtend_TAU" not in dataset_cols:
        return adjusted
    nrt_idx = dataset_cols.index("nrtend_TAU")
    adjusted[np.abs(adjusted[:, nrt_idx]) < threshold, nrt_idx] = 0.0
    return adjusted


def batch_predictions_to_model_matrix(predictions: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [predictions[key].detach().cpu().numpy() for key in MODEL_PRED_KEYS],
        axis=1,
    ).astype(np.float64, copy=False)


def batch_targets_to_dataset_matrix(
    targets: Dict[str, torch.Tensor],
    dataset_cols: Sequence[str],
) -> np.ndarray:
    return np.concatenate(
        [targets[col].detach().cpu().numpy() for col in dataset_cols],
        axis=1,
    ).astype(np.float64, copy=False)


def regimes_from_values(values: np.ndarray, threshold: float) -> np.ndarray:
    regimes = np.zeros(values.shape[0], dtype=np.int64)
    regimes[values < -threshold] = 1
    regimes[values > threshold] = 2
    return regimes


def magnitude_bin_index(abs_true: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(MAG_BIN_EDGES, abs_true, side="right") - 1
    return np.clip(idx, 0, len(MAG_BIN_EDGES) - 2)


def make_bin_label(left: float, right: float) -> str:
    if np.isinf(right):
        return f"[{left:.1e}, inf)"
    return f"[{left:.1e}, {right:.1e})"


@dataclass
class BinAccumulator:
    count: int = 0
    sum_true: float = 0.0
    sum_pred: float = 0.0
    sum_abs_true: float = 0.0
    sum_abs_pred: float = 0.0
    sum_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    sign_error_count: int = 0
    pred_near_zero_count: int = 0
    abs_errors: List[np.ndarray] = None
    logmag_ratios: List[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.abs_errors is None:
            self.abs_errors = []
        if self.logmag_ratios is None:
            self.logmag_ratios = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> None:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            return

        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        err = y_pred - y_true
        abs_err = np.abs(err)

        self.count += y_true.size
        self.sum_true += float(y_true.sum())
        self.sum_pred += float(y_pred.sum())
        self.sum_abs_true += float(np.abs(y_true).sum())
        self.sum_abs_pred += float(np.abs(y_pred).sum())
        self.sum_err += float(err.sum())
        self.sum_abs_err += float(abs_err.sum())
        self.sum_sq_err += float(np.dot(err, err))

        true_sign = np.sign(y_true)
        pred_sign = np.sign(y_pred)
        self.sign_error_count += int(np.sum(true_sign != pred_sign))
        self.pred_near_zero_count += int(np.sum(np.abs(y_pred) < threshold))

        self.abs_errors.append(abs_err)
        logmag_ratio = np.log10((np.abs(y_pred) + LOG_EPSILON) / (np.abs(y_true) + LOG_EPSILON))
        self.logmag_ratios.append(logmag_ratio)

    def finalize(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_true": math.nan,
                "mean_pred": math.nan,
                "mean_abs_true": math.nan,
                "mean_abs_pred": math.nan,
                "rmse": math.nan,
                "mae": math.nan,
                "bias": math.nan,
                "median_abs_error": math.nan,
                "p90_abs_error": math.nan,
                "p99_abs_error": math.nan,
                "sign_error_fraction": math.nan,
                "pred_near_zero_fraction": math.nan,
                "median_log10_mag_ratio": math.nan,
                "mean_log10_mag_ratio": math.nan,
            }

        abs_errors = np.concatenate(self.abs_errors) if self.abs_errors else np.array([], dtype=np.float64)
        logmag_ratios = np.concatenate(self.logmag_ratios) if self.logmag_ratios else np.array([], dtype=np.float64)

        return {
            "count": int(self.count),
            "mean_true": self.sum_true / self.count,
            "mean_pred": self.sum_pred / self.count,
            "mean_abs_true": self.sum_abs_true / self.count,
            "mean_abs_pred": self.sum_abs_pred / self.count,
            "rmse": math.sqrt(self.sum_sq_err / self.count),
            "mae": self.sum_abs_err / self.count,
            "bias": self.sum_err / self.count,
            "median_abs_error": float(np.median(abs_errors)) if abs_errors.size else math.nan,
            "p90_abs_error": float(np.percentile(abs_errors, 90)) if abs_errors.size else math.nan,
            "p99_abs_error": float(np.percentile(abs_errors, 99)) if abs_errors.size else math.nan,
            "sign_error_fraction": self.sign_error_count / self.count,
            "pred_near_zero_fraction": self.pred_near_zero_count / self.count,
            "median_log10_mag_ratio": float(np.median(logmag_ratios)) if logmag_ratios.size else math.nan,
            "mean_log10_mag_ratio": float(np.mean(logmag_ratios)) if logmag_ratios.size else math.nan,
        }


def build_regime_bin_accumulators() -> Dict[str, List[BinAccumulator]]:
    n_bins = len(MAG_BIN_EDGES) - 1
    return {
        regime: [BinAccumulator() for _ in range(n_bins)]
        for regime in REGIME_LABELS
    }


def build_regime_accumulators() -> Dict[str, BinAccumulator]:
    return {regime: BinAccumulator() for regime in REGIME_LABELS}


def plot_bin_counts(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for regime in REGIME_LABELS:
        sub = df[df["regime"] == regime]
        ax.plot(sub["bin_center_plot"], sub["count"], marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|true nrtend| bin center")
    ax.set_ylabel("Count")
    ax.set_title("nrtend Bin Counts")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bin_rmse_mae(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for regime in REGIME_LABELS:
        sub = df[df["regime"] == regime]
        axes[0].plot(sub["bin_center_plot"], sub["rmse"], marker="o", label=regime)
        axes[1].plot(sub["bin_center_plot"], sub["mae"], marker="o", label=regime)
    for ax, ylabel in zip(axes, ["RMSE", "MAE"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|true nrtend| bin center")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("nrtend Physical RMSE by Bin")
    axes[1].set_title("nrtend Physical MAE by Bin")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bin_bias(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for regime in REGIME_LABELS:
        sub = df[df["regime"] == regime]
        ax.plot(sub["bin_center_plot"], sub["bias"], marker="o", label=regime)
    ax.set_xscale("log")
    ax.set_xlabel("|true nrtend| bin center")
    ax.set_ylabel("Bias = mean(pred - true)")
    ax.set_title("nrtend Physical Bias by Bin")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_bin_logmag_ratio(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    for regime in REGIME_LABELS:
        sub = df[df["regime"] == regime]
        ax.plot(sub["bin_center_plot"], sub["median_log10_mag_ratio"], marker="o", label=regime)
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("|true nrtend| bin center")
    ax.set_ylabel("median log10((|pred|+eps)/(|true|+eps))")
    ax.set_title("nrtend Magnitude Calibration Error by Bin")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_calibration(df: pd.DataFrame, regime: str, output_path: Path) -> None:
    sub = df[df["regime"] == regime].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(sub["mean_abs_true"], sub["mean_abs_pred"], marker="o")
    finite = np.isfinite(sub["mean_abs_true"]) & np.isfinite(sub["mean_abs_pred"]) & (sub["mean_abs_true"] > 0)
    if np.any(finite):
        lo = min(np.min(sub.loc[finite, "mean_abs_true"]), np.min(sub.loc[finite, "mean_abs_pred"]))
        hi = max(np.max(sub.loc[finite, "mean_abs_true"]), np.max(sub.loc[finite, "mean_abs_pred"]))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("mean |true nrtend|")
    ax.set_ylabel("mean |pred nrtend|")
    ax.set_title(f"{regime} regime magnitude calibration")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detailed physical-space nrtend bin analysis")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_used.yml")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory for analysis artifacts",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device",
    )
    parser.add_argument(
        "--scaler_run_id",
        "--scaler-run-id",
        dest="scaler_run_id",
        type=str,
        default=None,
        help="Run id whose scaler artifacts should be reused",
    )
    parser.add_argument(
        "--max_eval_batches",
        "--max-batches",
        dest="max_eval_batches",
        type=int,
        default=None,
        help="Optional cap on validation batches for smoke tests",
    )
    parser.add_argument(
        "--log_every",
        "--log-every",
        dest="log_every",
        type=int,
        default=10,
        help="Log progress every N batches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    data_cfg = config["data"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    scaler_run_id = (
        normalize_run_id(args.scaler_run_id)
        or normalize_run_id(os.environ.get("SCALER_RUN_ID"))
        or infer_run_id_from_path(str(args.checkpoint))
    )

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Config: %s", args.config)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Device: %s", device)
    logger.info("Scaler run id: %s", scaler_run_id)

    model = load_model(args.checkpoint, config, device)

    _, val_loader, _ = create_optimized_streaming_loaders(
        data_path=data_cfg["data_path"],
        config=config,
        train_fraction=data_cfg["train_fraction"],
        batch_size=data_cfg["batch_size"],
        scaler_cache_dir=data_cfg.get("scaler_cache_dir", "./scaler_cache"),
        run_id_override=scaler_run_id,
    )
    val_dataset = val_loader.dataset
    dataset_cols = list(getattr(val_dataset, "output_cols", []))

    nrtend_threshold = data_cfg.get("nrtend_regime_threshold")
    nrtend_threshold = float(nrtend_threshold) if nrtend_threshold is not None else None
    if nrtend_threshold is None:
        raise ValueError("This analysis expects data.nrtend_regime_threshold to be set.")

    regime_bin_acc = build_regime_bin_accumulators()
    regime_acc = build_regime_accumulators()

    processed_batches = 0
    processed_samples = 0
    start_time = pd.Timestamp.utcnow()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs = inputs.to(device)
            predictions = model(inputs)

            pred_norm_model = batch_predictions_to_model_matrix(predictions)
            target_norm_dataset = batch_targets_to_dataset_matrix(targets, dataset_cols)

            target_phys_dataset = inverse_output_pipeline(target_norm_dataset, dataset_cols, val_dataset)
            target_phys_dataset = zero_small_nrtend_targets(
                target_phys_dataset,
                dataset_cols,
                nrtend_threshold,
            )

            pred_norm_dataset = model_to_dataset_matrix(pred_norm_model, dataset_cols)
            pred_phys_dataset = inverse_output_pipeline(pred_norm_dataset, dataset_cols, val_dataset)

            target_phys_model = dataset_to_model_matrix(target_phys_dataset, dataset_cols)
            pred_phys_model = dataset_to_model_matrix(pred_phys_dataset, dataset_cols)

            y_true = target_phys_model[:, 2]
            y_pred = pred_phys_model[:, 2]

            regimes = regimes_from_values(y_true, nrtend_threshold)
            abs_true = np.abs(y_true)
            bin_idx = magnitude_bin_index(abs_true)

            processed_batches += 1
            processed_samples += y_true.size

            for regime_idx, regime_name in enumerate(REGIME_LABELS):
                regime_mask = regimes == regime_idx
                if not np.any(regime_mask):
                    continue

                yt_reg = y_true[regime_mask]
                yp_reg = y_pred[regime_mask]
                regime_acc[regime_name].update(yt_reg, yp_reg, nrtend_threshold)

                reg_bin_idx = bin_idx[regime_mask]
                for b in np.unique(reg_bin_idx):
                    bin_mask = reg_bin_idx == b
                    regime_bin_acc[regime_name][int(b)].update(
                        yt_reg[bin_mask],
                        yp_reg[bin_mask],
                        nrtend_threshold,
                    )

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    regime_rows = []
    for regime_name, acc in regime_acc.items():
        row = {"regime": regime_name}
        row.update(acc.finalize())
        regime_rows.append(row)
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(args.output_dir / "nrtend_regime_summary.csv", index=False)

    bin_rows = []
    for regime_name in REGIME_LABELS:
        for b, acc in enumerate(regime_bin_acc[regime_name]):
            left = float(MAG_BIN_EDGES[b])
            right = float(MAG_BIN_EDGES[b + 1])
            row = {
                "regime": regime_name,
                "bin_index": b,
                "bin_left": left,
                "bin_right": right,
                "bin_label": make_bin_label(left, right),
                "bin_center_plot": left if left > 0 else MAG_BIN_EDGES[1] / 2.0,
            }
            row.update(acc.finalize())
            bin_rows.append(row)

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(args.output_dir / "nrtend_bin_metrics.csv", index=False)

    plot_bin_counts(bin_df, args.output_dir / "nrtend_bin_counts.png")
    plot_bin_rmse_mae(bin_df, args.output_dir / "nrtend_bin_rmse_mae.png")
    plot_bin_bias(bin_df, args.output_dir / "nrtend_bin_bias.png")
    plot_bin_logmag_ratio(bin_df, args.output_dir / "nrtend_bin_logmag_ratio.png")
    plot_regime_calibration(bin_df, "positive", args.output_dir / "nrtend_positive_mag_calibration.png")
    plot_regime_calibration(bin_df, "negative", args.output_dir / "nrtend_negative_mag_calibration.png")
    plot_regime_calibration(bin_df, "near_zero", args.output_dir / "nrtend_near_zero_mag_calibration.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "processed_batches": processed_batches,
        "processed_samples_after_filtering": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "mag_bin_edges": MAG_BIN_EDGES.tolist(),
    }
    with (args.output_dir / "nrtend_bin_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("nrtend bin analysis complete.")
    logger.info("Processed %d batches and %d filtered validation samples.", processed_batches, processed_samples)
    logger.info("Saved: %s", args.output_dir)


if __name__ == "__main__":
    main()
