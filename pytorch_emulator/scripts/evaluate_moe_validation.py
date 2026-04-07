#!/usr/bin/env python3
"""
Evaluate a MoE microphysics checkpoint on the validation split.

This evaluator is specific to MoE runs such as run_594616:
- it loads the raw checkpoint, not an exported TorchScript artifact
- it evaluates the validation split defined by streaming_data_loader_v2.py
  (the last 20% of sorted parquet files when train_fraction=0.8)
- it applies the target-only nrtend rule:
      if |nrtend| < nrtend_regime_threshold, set target nrtend = 0
- it reports metrics in both:
      1. transformed/scaled space seen by the network
      2. physical space
- it explicitly guards against the output ordering / scaler fitting hazard:
      config output_cols order can differ from model output order

Outputs:
- summary.json
- metrics_summary.csv
- scatter_transformed.png
- scatter_physical.png
- hist_transformed.png
- hist_physical.png
- nrtend_metrics_by_regime.csv
- nrtend_regime_confusion.csv
- nrtend_regime_confusion.png
- validation_files.txt
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
from matplotlib.colors import LogNorm

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
    model = MoEConstraintAwareEmulator(
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
    return model


def load_model(checkpoint_path: Path, config: Dict, device: torch.device) -> torch.nn.Module:
    model_cfg = config["model"]
    architecture = str(model_cfg.get("architecture", "")).lower()
    if architecture != "moe":
        raise ValueError(
            f"evaluate_moe_validation.py expects model.architecture='moe', found '{architecture}'"
        )

    model = instantiate_moe_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(resolve_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


@dataclass
class RunningRegressionMetrics:
    n: int = 0
    sum_true: float = 0.0
    sum_pred: float = 0.0
    sum_true_sq: float = 0.0
    sum_pred_sq: float = 0.0
    sum_true_pred: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    sum_err: float = 0.0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            return

        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]
        err = y_pred - y_true

        self.n += y_true.size
        self.sum_true += float(y_true.sum())
        self.sum_pred += float(y_pred.sum())
        self.sum_true_sq += float(np.dot(y_true, y_true))
        self.sum_pred_sq += float(np.dot(y_pred, y_pred))
        self.sum_true_pred += float(np.dot(y_true, y_pred))
        self.sum_abs_err += float(np.abs(err).sum())
        self.sum_sq_err += float(np.dot(err, err))
        self.sum_err += float(err.sum())

    def finalize(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "count": 0,
                "r2": math.nan,
                "rmse": math.nan,
                "mae": math.nan,
                "bias": math.nan,
                "mean_true": math.nan,
                "mean_pred": math.nan,
                "std_true": math.nan,
                "std_pred": math.nan,
                "pearson_r": math.nan,
            }

        mean_true = self.sum_true / self.n
        mean_pred = self.sum_pred / self.n
        var_true = max(self.sum_true_sq / self.n - mean_true**2, 0.0)
        var_pred = max(self.sum_pred_sq / self.n - mean_pred**2, 0.0)
        sst = self.sum_true_sq - (self.sum_true**2) / self.n
        covariance = self.sum_true_pred - (self.sum_true * self.sum_pred) / self.n
        denom = math.sqrt(
            max(self.sum_true_sq - (self.sum_true**2) / self.n, 0.0)
            * max(self.sum_pred_sq - (self.sum_pred**2) / self.n, 0.0)
        )
        pearson_r = covariance / denom if denom > 0 else math.nan

        return {
            "count": int(self.n),
            "r2": 1.0 - (self.sum_sq_err / sst) if sst > 0 else math.nan,
            "rmse": math.sqrt(self.sum_sq_err / self.n),
            "mae": self.sum_abs_err / self.n,
            "bias": self.sum_err / self.n,
            "mean_true": mean_true,
            "mean_pred": mean_pred,
            "std_true": math.sqrt(var_true),
            "std_pred": math.sqrt(var_pred),
            "pearson_r": pearson_r,
        }


class ApproxPointSampler:
    """Keep a bounded sample for plotting without storing the full validation pass."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.true = np.empty(0, dtype=np.float64)
        self.pred = np.empty(0, dtype=np.float64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        if self.capacity <= 0:
            return

        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            return

        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]

        if self.true.size < self.capacity:
            remaining = self.capacity - self.true.size
            if y_true.size <= remaining:
                keep_true = y_true
                keep_pred = y_pred
            else:
                idx = self.rng.choice(y_true.size, size=remaining, replace=False)
                keep_true = y_true[idx]
                keep_pred = y_pred[idx]
            self.true = np.concatenate([self.true, keep_true])
            self.pred = np.concatenate([self.pred, keep_pred])
            return

        new_take = min(max(self.capacity // 8, 1), y_true.size)
        idx_new = self.rng.choice(y_true.size, size=new_take, replace=False)
        combined_true = np.concatenate([self.true, y_true[idx_new]])
        combined_pred = np.concatenate([self.pred, y_pred[idx_new]])
        keep_idx = self.rng.choice(combined_true.size, size=self.capacity, replace=False)
        self.true = combined_true[keep_idx]
        self.pred = combined_pred[keep_idx]


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


def forward_output_pipeline(
    physical_dataset_matrix: np.ndarray,
    dataset_cols: Sequence[str],
    dataset,
) -> np.ndarray:
    transformed = np.asarray(physical_dataset_matrix, dtype=np.float64).copy()
    scaler = getattr(dataset, "output_scaler", None)
    transformer = getattr(dataset, "output_transformer", None)
    output_transform = getattr(dataset, "output_transform", "log10")
    use_nrtend_arcsinh = bool(getattr(dataset, "nrtend_arcsinh_transform", False))
    arcsinh_threshold = float(getattr(dataset, "nrtend_arcsinh_threshold", 1.0e-3))

    if output_transform == "log10":
        for idx, col in enumerate(dataset_cols):
            if use_nrtend_arcsinh and col == "nrtend_TAU":
                continue
            if col not in LOG_OUTPUT_COLS:
                continue
            values = transformed[:, idx]
            transformed[:, idx] = np.sign(values) * np.log10(np.abs(values) + LOG_EPSILON)

    if use_nrtend_arcsinh and "nrtend_TAU" in dataset_cols:
        nrt_idx = dataset_cols.index("nrtend_TAU")
        transformed[:, nrt_idx] = np.arcsinh(transformed[:, nrt_idx] / arcsinh_threshold)

    if output_transform == "quantile" and transformer is not None:
        transformed = transformer.transform(transformed)
    if scaler is not None:
        transformed = scaler.transform(transformed)

    return transformed


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


def update_confusion(confusion: np.ndarray, true_regimes: np.ndarray, pred_regimes: np.ndarray) -> None:
    np.add.at(confusion, (true_regimes, pred_regimes), 1)


def save_metrics_csv(metrics_by_space: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path) -> None:
    rows = []
    for space, metrics_map in metrics_by_space.items():
        for variable, metrics in metrics_map.items():
            row = {"space": space, "variable": variable}
            row.update(metrics)
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)


def save_validation_files(val_dataset, output_dir: Path) -> Dict[str, object]:
    active_files = list(getattr(val_dataset, "active_files", []))
    file_names = [path.name for path in active_files]
    (output_dir / "validation_files.txt").write_text(
        "\n".join(file_names) + ("\n" if file_names else "")
    )
    return {
        "n_validation_files": len(file_names),
        "first_validation_file": file_names[0] if file_names else None,
        "last_validation_file": file_names[-1] if file_names else None,
    }


def make_scatter_figure(
    samplers: Dict[str, ApproxPointSampler],
    output_path: Path,
    space_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Validation Scatter: {space_label}", fontsize=16, fontweight="bold")
    hexbin_handles = []
    max_count = 1.0
    log_norm = LogNorm(vmin=1)

    for ax, pred_key, col in zip(axes.flatten(), MODEL_PRED_KEYS, MODEL_OUTPUT_COLS):
        sampler = samplers[pred_key]
        if sampler.true.size == 0:
            ax.text(0.5, 0.5, "No sampled points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(pred_key)
            continue

        hb = ax.hexbin(
            sampler.true,
            sampler.pred,
            gridsize=150,
            cmap="viridis",
            mincnt=1,
            norm=log_norm,
        )
        hexbin_handles.append(hb)
        counts = hb.get_array()
        if counts.size > 0:
            max_count = max(max_count, float(np.max(counts)))

        low = float(min(np.min(sampler.true), np.min(sampler.pred)))
        high = float(max(np.max(sampler.true), np.max(sampler.pred)))
        ax.plot([low, high], [low, high], "r--", linewidth=1.5)
        ax.set_xlabel(f"True {pred_key}")
        ax.set_ylabel(f"Predicted {pred_key}")
        ax.set_title(col)
        ax.grid(alpha=0.25)

    if hexbin_handles:
        log_norm.vmax = max_count
        for handle in hexbin_handles:
            handle.set_clim(log_norm.vmin, log_norm.vmax)
        cbar = fig.colorbar(hexbin_handles[0], ax=axes, pad=0.08, aspect=40, shrink=1.0)
        cbar.set_label("Frequency")

    plt.tight_layout(rect=[0, 0, 0.93, 1])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_hist_figure(
    samplers: Dict[str, ApproxPointSampler],
    output_path: Path,
    space_label: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Validation Histograms: {space_label}", fontsize=16, fontweight="bold")

    for ax, pred_key, col in zip(axes.flatten(), MODEL_PRED_KEYS, MODEL_OUTPUT_COLS):
        sampler = samplers[pred_key]
        if sampler.true.size == 0:
            ax.text(0.5, 0.5, "No sampled points", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(pred_key)
            continue

        ax.hist(sampler.true, bins=100, alpha=0.6, label="True", color="steelblue")
        ax.hist(sampler.pred, bins=100, alpha=0.6, label="Pred", color="darkorange")
        ax.set_title(col)
        ax.set_xlabel(pred_key)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_confusion_figure(confusion: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(3), REGIME_LABELS)
    ax.set_yticks(range(3), REGIME_LABELS)
    ax.set_xlabel("Predicted regime from nrtend")
    ax.set_ylabel("Target regime")
    ax.set_title("nrtend Regime Confusion")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(confusion[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MoE microphysics checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_used.yml")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory for evaluation artifacts",
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
        "--plot_sample_size",
        "--plot-sample-size",
        dest="plot_sample_size",
        type=int,
        default=150000,
        help="Approximate per-variable sample size to keep for plots",
    )
    parser.add_argument(
        "--log_every",
        "--log-every",
        dest="log_every",
        type=int,
        default=10,
        help="Log progress every N batches",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for plot sampling")
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
    logger.info("Applying target-only nrtend zeroing threshold: %s", nrtend_threshold)

    validation_summary = save_validation_files(val_dataset, args.output_dir)

    metrics_by_space: Dict[str, Dict[str, RunningRegressionMetrics]] = {
        "transformed": {key: RunningRegressionMetrics() for key in MODEL_PRED_KEYS},
        "physical": {key: RunningRegressionMetrics() for key in MODEL_PRED_KEYS},
    }
    nrtend_regime_metrics = {
        name: RunningRegressionMetrics() for name in REGIME_LABELS
    }
    nrtend_confusion = np.zeros((3, 3), dtype=np.int64)

    samplers = {
        "transformed": {
            key: ApproxPointSampler(args.plot_sample_size, args.seed + 17 * idx)
            for idx, key in enumerate(MODEL_PRED_KEYS)
        },
        "physical": {
            key: ApproxPointSampler(args.plot_sample_size, args.seed + 97 * idx)
            for idx, key in enumerate(MODEL_PRED_KEYS)
        },
    }

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
            target_norm_adjusted_dataset = forward_output_pipeline(
                target_phys_dataset,
                dataset_cols,
                val_dataset,
            )

            pred_norm_dataset = model_to_dataset_matrix(pred_norm_model, dataset_cols)
            pred_phys_dataset = inverse_output_pipeline(pred_norm_dataset, dataset_cols, val_dataset)

            target_norm_model = dataset_to_model_matrix(target_norm_adjusted_dataset, dataset_cols)
            target_phys_model = dataset_to_model_matrix(target_phys_dataset, dataset_cols)
            pred_phys_model = dataset_to_model_matrix(pred_phys_dataset, dataset_cols)

            batch_size = pred_norm_model.shape[0]
            processed_batches += 1
            processed_samples += batch_size

            for idx, key in enumerate(MODEL_PRED_KEYS):
                metrics_by_space["transformed"][key].update(
                    target_norm_model[:, idx], pred_norm_model[:, idx]
                )
                metrics_by_space["physical"][key].update(
                    target_phys_model[:, idx], pred_phys_model[:, idx]
                )
                samplers["transformed"][key].update(
                    target_norm_model[:, idx], pred_norm_model[:, idx]
                )
                samplers["physical"][key].update(
                    target_phys_model[:, idx], pred_phys_model[:, idx]
                )

            if nrtend_threshold is not None:
                nrtend_true = target_phys_model[:, 2]
                nrtend_pred = pred_phys_model[:, 2]
                true_regimes = regimes_from_values(nrtend_true, nrtend_threshold)
                pred_regimes = regimes_from_values(nrtend_pred, nrtend_threshold)
                update_confusion(nrtend_confusion, true_regimes, pred_regimes)

                for regime_idx, regime_name in enumerate(REGIME_LABELS):
                    regime_mask = true_regimes == regime_idx
                    if np.any(regime_mask):
                        nrtend_regime_metrics[regime_name].update(
                            nrtend_true[regime_mask],
                            nrtend_pred[regime_mask],
                        )

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    finalized_metrics: Dict[str, Dict[str, Dict[str, float]]] = {
        space: {key: metrics.finalize() for key, metrics in metrics_map.items()}
        for space, metrics_map in metrics_by_space.items()
    }
    nrtend_regime_final = {
        regime: metrics.finalize() for regime, metrics in nrtend_regime_metrics.items()
    }

    save_metrics_csv(finalized_metrics, args.output_dir)
    pd.DataFrame(
        [{"regime": regime, **values} for regime, values in nrtend_regime_final.items()]
    ).to_csv(args.output_dir / "nrtend_metrics_by_regime.csv", index=False)
    pd.DataFrame(
        nrtend_confusion,
        index=[f"true_{name}" for name in REGIME_LABELS],
        columns=[f"pred_{name}" for name in REGIME_LABELS],
    ).to_csv(args.output_dir / "nrtend_regime_confusion.csv")

    make_scatter_figure(
        samplers["transformed"],
        args.output_dir / "scatter_transformed.png",
        "Transformed / Scaled Space",
    )
    make_scatter_figure(
        samplers["physical"],
        args.output_dir / "scatter_physical.png",
        "Physical Space",
    )
    make_hist_figure(
        samplers["transformed"],
        args.output_dir / "hist_transformed.png",
        "Transformed / Scaled Space",
    )
    make_hist_figure(
        samplers["physical"],
        args.output_dir / "hist_physical.png",
        "Physical Space",
    )
    make_confusion_figure(
        nrtend_confusion,
        args.output_dir / "nrtend_regime_confusion.png",
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "processed_batches": processed_batches,
        "processed_samples_after_filtering": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "validation_split": validation_summary,
        "metrics": finalized_metrics,
        "nrtend_metrics_by_regime": nrtend_regime_final,
        "nrtend_regime_confusion": nrtend_confusion.tolist(),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Evaluation complete.")
    logger.info(
        "Processed %d batches and %d filtered validation samples.",
        processed_batches,
        processed_samples,
    )
    for space, metrics_map in finalized_metrics.items():
        logger.info("Metrics in %s space:", space)
        for key in MODEL_PRED_KEYS:
            metrics = metrics_map[key]
            logger.info(
                "  %-6s R2=% .4f RMSE=% .4e MAE=% .4e Bias=% .4e Pearson=% .4f",
                key,
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
                metrics["bias"],
                metrics["pearson_r"],
            )


if __name__ == "__main__":
    main()
