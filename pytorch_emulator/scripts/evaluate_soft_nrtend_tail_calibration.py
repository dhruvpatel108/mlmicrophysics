#!/usr/bin/env python3
"""
Fit and evaluate a simple deployable nrtend tail calibrator on top of soft routing.

This script is designed as an emulator-only follow-up after the soft-routing
analysis for run_594616. It:

1. Fits a piecewise affine calibrator for physical-space
   soft-routed nrtend using prediction-defined negative-tail bins.
2. Re-evaluates the full validation split with that calibrator applied.
3. Reports whether this lightweight post-processing is enough to make the
   checkpoint integration-ready in physical space.

The calibrator is intentionally simple so it can later be embedded in a
TorchScript export:

    if soft_nrtend <= -1e5:   calibrated = a_ge_1e5   * soft_nrtend + b_ge_1e5
    elif soft_nrtend <= -3e4: calibrated = a_3e4_1e5  * soft_nrtend + b_3e4_1e5
    elif soft_nrtend <= -1e4: calibrated = a_1e4_3e4  * soft_nrtend + b_1e4_3e4
    else: calibrated = soft_nrtend

Important:
- the fit and evaluation are both done on the same validation split, so the
  calibrated metrics are optimistic and should be treated as a decision aid,
  not a final held-out estimate.
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
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in (THIS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_moe_soft_routing as soft_eval
import evaluate_moe_validation as base_eval
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


NEGATIVE_TAIL_THRESHOLDS: Tuple[float, ...] = (1.0e4, 3.0e4, 1.0e5)


@dataclass
class FitAccumulator:
    count: int = 0
    sum_x2: float = 0.0
    sum_xy: float = 0.0
    sum_x: float = 0.0
    sum_true: float = 0.0
    sum_pred: float = 0.0
    ratios: List[np.ndarray] | None = None

    def __post_init__(self) -> None:
        if self.ratios is None:
            self.ratios = []

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(finite_mask):
            return

        y_true = y_true[finite_mask]
        y_pred = y_pred[finite_mask]

        self.count += int(y_true.size)
        self.sum_x2 += float(np.dot(y_pred, y_pred))
        self.sum_xy += float(np.dot(y_pred, y_true))
        self.sum_x += float(y_pred.sum())
        self.sum_true += float(y_true.sum())
        self.sum_pred += float(y_pred.sum())
        ratio = np.divide(
            np.abs(y_true),
            np.abs(y_pred),
            out=np.full_like(y_true, np.nan),
            where=np.abs(y_pred) > 0.0,
        )
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size > 0:
            self.ratios.append(ratio)

    def finalize(self) -> Dict[str, float]:
        if self.count > 0:
            mean_x = self.sum_x / self.count
            mean_y = self.sum_true / self.count
            centered_xx = self.sum_x2 - self.count * mean_x * mean_x
            centered_xy = self.sum_xy - self.count * mean_x * mean_y
            raw_slope = centered_xy / centered_xx if centered_xx > 0 else math.nan
            raw_intercept = mean_y - raw_slope * mean_x if np.isfinite(raw_slope) else math.nan
        else:
            raw_slope = math.nan
            raw_intercept = math.nan

        slope = float(raw_slope) if np.isfinite(raw_slope) else 1.0
        intercept = float(raw_intercept) if np.isfinite(raw_intercept) else 0.0
        ratio_vals = np.concatenate(self.ratios) if self.ratios else np.empty(0, dtype=np.float64)
        return {
            "count": int(self.count),
            "raw_affine_slope": raw_slope,
            "raw_affine_intercept": raw_intercept,
            "affine_slope": slope,
            "affine_intercept": intercept,
            "mean_true": self.sum_true / self.count if self.count > 0 else math.nan,
            "mean_pred": self.sum_pred / self.count if self.count > 0 else math.nan,
            "mean_abs_ratio_true_over_pred": (
                float(np.mean(ratio_vals)) if ratio_vals.size > 0 else math.nan
            ),
            "median_abs_ratio_true_over_pred": (
                float(np.median(ratio_vals)) if ratio_vals.size > 0 else math.nan
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a soft-routing nrtend tail calibrator"
    )
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
        "--log_every",
        "--log-every",
        dest="log_every",
        type=int,
        default=10,
        help="Log progress every N batches",
    )
    parser.add_argument(
        "--integration_r2_threshold",
        "--integration-r2-threshold",
        dest="integration_r2_threshold",
        type=float,
        default=0.97,
        help="Physical-space R2 threshold used for integration readiness checks",
    )
    return parser.parse_args()


def fit_bin_labels() -> Tuple[str, str, str]:
    return (
        "[1.0e+04, 3.0e+04)",
        "[3.0e+04, 1.0e+05)",
        "[1.0e+05, inf)",
    )


def bin_masks_from_prediction(y_soft: np.ndarray) -> Dict[str, np.ndarray]:
    y_soft = np.asarray(y_soft, dtype=np.float64)
    return {
        "[1.0e+04, 3.0e+04)": (y_soft <= -NEGATIVE_TAIL_THRESHOLDS[0]) & (y_soft > -NEGATIVE_TAIL_THRESHOLDS[1]),
        "[3.0e+04, 1.0e+05)": (y_soft <= -NEGATIVE_TAIL_THRESHOLDS[1]) & (y_soft > -NEGATIVE_TAIL_THRESHOLDS[2]),
        "[1.0e+05, inf)": y_soft <= -NEGATIVE_TAIL_THRESHOLDS[2],
    }


def apply_piecewise_calibration(
    y_soft: np.ndarray,
    coefficients: Dict[str, Dict[str, float]],
) -> np.ndarray:
    y_soft = np.asarray(y_soft, dtype=np.float64)
    y_cal = y_soft.copy()
    masks = bin_masks_from_prediction(y_soft)
    for label, mask in masks.items():
        coeff = coefficients.get(label, {})
        slope = float(coeff.get("slope", 1.0))
        intercept = float(coeff.get("intercept", 0.0))
        y_cal[mask] = slope * y_soft[mask] + intercept
    return y_cal


def build_metrics_container() -> Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]:
    return {
        mode: {key: base_eval.RunningRegressionMetrics() for key in base_eval.MODEL_PRED_KEYS}
        for mode in ("soft_baseline", "soft_calibrated")
    }


def build_regime_metrics_container() -> Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]:
    return {
        mode: {name: base_eval.RunningRegressionMetrics() for name in base_eval.REGIME_LABELS}
        for mode in ("soft_baseline", "soft_calibrated")
    }


def build_confusions() -> Dict[str, np.ndarray]:
    return {
        mode: np.zeros((3, 3), dtype=np.int64)
        for mode in ("soft_baseline", "soft_calibrated")
    }


def save_metrics_csv(
    finalized_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for mode, metrics_map in finalized_metrics.items():
        for variable, metrics in metrics_map.items():
            row = {"mode": mode, "space": "physical", "variable": variable}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_summary.csv", index=False)
    return df


def save_regime_metrics_csv(
    finalized_regime_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for mode, metrics_map in finalized_regime_metrics.items():
        for regime, metrics in metrics_map.items():
            row = {"mode": mode, "regime": regime}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "nrtend_regime_metrics.csv", index=False)
    return df


def save_confusions_csv(confusions: Dict[str, np.ndarray], output_dir: Path) -> pd.DataFrame:
    rows = []
    for mode, matrix in confusions.items():
        for true_idx, true_regime in enumerate(base_eval.REGIME_LABELS):
            row = {"mode": mode, "true_regime": true_regime}
            for pred_idx, pred_regime in enumerate(base_eval.REGIME_LABELS):
                row[f"pred_{pred_regime}"] = int(matrix[true_idx, pred_idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "nrtend_regime_confusion.csv", index=False)
    return df


def build_mode_comparison(
    metrics_df: pd.DataFrame,
    integration_r2_threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    pivot = metrics_df.pivot(index="variable", columns="mode")
    rows = []
    for variable in base_eval.MODEL_PRED_KEYS:
        row = {"variable": variable}
        for metric in ("r2", "rmse", "mae", "bias", "pearson_r"):
            baseline_val = pivot[(metric, "soft_baseline")].get(variable, np.nan)
            calibrated_val = pivot[(metric, "soft_calibrated")].get(variable, np.nan)
            row[f"soft_baseline_{metric}"] = baseline_val
            row[f"soft_calibrated_{metric}"] = calibrated_val
            row[f"delta_{metric}_calibrated_minus_baseline"] = calibrated_val - baseline_val
        row["soft_baseline_meets_r2_threshold"] = bool(
            np.isfinite(row["soft_baseline_r2"]) and row["soft_baseline_r2"] >= integration_r2_threshold
        )
        row["soft_calibrated_meets_r2_threshold"] = bool(
            np.isfinite(row["soft_calibrated_r2"]) and row["soft_calibrated_r2"] >= integration_r2_threshold
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "physical_mode_comparison.csv", index=False)
    return df


def plot_physical_metric_by_mode(
    comparison_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comparison_df))
    width = 0.36

    ax.bar(
        x - width / 2,
        comparison_df[f"soft_baseline_{metric}"],
        width,
        label="soft_baseline",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        comparison_df[f"soft_calibrated_{metric}"],
        width,
        label="soft_calibrated",
        color="#54A24B",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["variable"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Physical {metric.upper()} Before vs After Tail Calibration")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_r2_by_mode(regime_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(base_eval.REGIME_LABELS))
    width = 0.36

    baseline_vals = [
        regime_df[(regime_df["mode"] == "soft_baseline") & (regime_df["regime"] == regime)]["r2"].iloc[0]
        for regime in base_eval.REGIME_LABELS
    ]
    calibrated_vals = [
        regime_df[(regime_df["mode"] == "soft_calibrated") & (regime_df["regime"] == regime)]["r2"].iloc[0]
        for regime in base_eval.REGIME_LABELS
    ]

    ax.bar(x - width / 2, baseline_vals, width, label="soft_baseline", color="#4C78A8")
    ax.bar(x + width / 2, calibrated_vals, width, label="soft_calibrated", color="#54A24B")

    ax.set_xticks(x)
    ax.set_xticklabels(base_eval.REGIME_LABELS)
    ax.set_ylabel("R2")
    ax.set_title("nrtend Physical R2 by Regime Before vs After Tail Calibration")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = base_eval.load_config(args.config)
    data_cfg = config["data"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    scaler_run_id = (
        base_eval.normalize_run_id(args.scaler_run_id)
        or base_eval.normalize_run_id(os.environ.get("SCALER_RUN_ID"))
        or base_eval.infer_run_id_from_path(str(args.checkpoint))
    )

    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Config: %s", args.config)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Device: %s", device)
    logger.info("Scaler run id: %s", scaler_run_id)

    model = base_eval.load_model(args.checkpoint, config, device)

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

    validation_summary = base_eval.save_validation_files(val_dataset, args.output_dir)

    fit_accumulators = {label: FitAccumulator() for label in fit_bin_labels()}
    fit_processed_batches = 0
    fit_processed_samples = 0
    fit_flagged_count = 0
    fit_start_time = pd.Timestamp.utcnow()

    logger.info("Starting calibration fit pass over validation split.")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs = inputs.to(device)
            predictions = soft_eval.compute_hard_and_soft_predictions(model, inputs)
            target_norm_dataset = base_eval.batch_targets_to_dataset_matrix(targets, dataset_cols)

            target_phys_dataset = base_eval.inverse_output_pipeline(
                target_norm_dataset,
                dataset_cols,
                val_dataset,
            )
            target_phys_dataset = base_eval.zero_small_nrtend_targets(
                target_phys_dataset,
                dataset_cols,
                nrtend_threshold,
            )
            target_phys_model = base_eval.dataset_to_model_matrix(target_phys_dataset, dataset_cols)

            pred_norm_dataset = base_eval.model_to_dataset_matrix(predictions["soft"], dataset_cols)
            pred_phys_dataset = base_eval.inverse_output_pipeline(
                pred_norm_dataset,
                dataset_cols,
                val_dataset,
            )
            pred_phys_model = base_eval.dataset_to_model_matrix(pred_phys_dataset, dataset_cols)

            y_true = target_phys_model[:, 2]
            y_soft = pred_phys_model[:, 2]
            masks = bin_masks_from_prediction(y_soft)

            for label, mask in masks.items():
                fit_accumulators[label].update(y_true[mask], y_soft[mask])
                fit_flagged_count += int(np.sum(mask))

            fit_processed_batches += 1
            fit_processed_samples += int(y_true.size)

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - fit_start_time).total_seconds()
                logger.info(
                    "Fit pass: processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    fit_processed_samples,
                    elapsed_s,
                )

    calibrator_rows = []
    coefficients: Dict[str, Dict[str, float]] = {}
    for label in fit_bin_labels():
        row = {"bin_label": label}
        row.update(fit_accumulators[label].finalize())
        coefficients[label] = {
            "slope": float(row["affine_slope"]),
            "intercept": float(row["affine_intercept"]),
        }
        calibrator_rows.append(row)
    calibrator_df = pd.DataFrame(calibrator_rows)
    calibrator_df.to_csv(args.output_dir / "calibrator_fit_summary.csv", index=False)

    with (args.output_dir / "calibrator_params.json").open("w") as handle:
        json.dump(
            {
                "kind": "piecewise_affine_negative_tail",
                "same_set_fit_and_eval": True,
                "thresholds_abs_pred": list(NEGATIVE_TAIL_THRESHOLDS),
                "coefficients": coefficients,
            },
            handle,
            indent=2,
        )

    logger.info("Fitted tail calibrator coefficients: %s", coefficients)

    metrics_by_mode = build_metrics_container()
    regime_metrics_by_mode = build_regime_metrics_container()
    confusions = build_confusions()
    threshold_flag_counts = {
        label: 0 for label in fit_bin_labels()
    }

    processed_batches = 0
    processed_samples = 0
    eval_start_time = pd.Timestamp.utcnow()

    logger.info("Starting evaluation pass with calibrated soft routing.")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs = inputs.to(device)
            predictions = soft_eval.compute_hard_and_soft_predictions(model, inputs)
            target_norm_dataset = base_eval.batch_targets_to_dataset_matrix(targets, dataset_cols)

            target_phys_dataset = base_eval.inverse_output_pipeline(
                target_norm_dataset,
                dataset_cols,
                val_dataset,
            )
            target_phys_dataset = base_eval.zero_small_nrtend_targets(
                target_phys_dataset,
                dataset_cols,
                nrtend_threshold,
            )
            target_phys_model = base_eval.dataset_to_model_matrix(target_phys_dataset, dataset_cols)

            pred_norm_dataset = base_eval.model_to_dataset_matrix(predictions["soft"], dataset_cols)
            pred_phys_dataset = base_eval.inverse_output_pipeline(
                pred_norm_dataset,
                dataset_cols,
                val_dataset,
            )
            pred_phys_model = base_eval.dataset_to_model_matrix(pred_phys_dataset, dataset_cols)

            y_true = target_phys_model[:, 2]
            y_soft = pred_phys_model[:, 2]
            y_calibrated = apply_piecewise_calibration(y_soft, coefficients)

            masks = bin_masks_from_prediction(y_soft)
            for label, mask in masks.items():
                threshold_flag_counts[label] += int(np.sum(mask))

            baseline_phys = pred_phys_model.copy()
            calibrated_phys = pred_phys_model.copy()
            calibrated_phys[:, 2] = y_calibrated

            for idx, key in enumerate(base_eval.MODEL_PRED_KEYS):
                metrics_by_mode["soft_baseline"][key].update(
                    target_phys_model[:, idx], baseline_phys[:, idx]
                )
                metrics_by_mode["soft_calibrated"][key].update(
                    target_phys_model[:, idx], calibrated_phys[:, idx]
                )

            if nrtend_threshold is not None:
                true_regimes = base_eval.regimes_from_values(y_true, nrtend_threshold)
                baseline_regimes = base_eval.regimes_from_values(y_soft, nrtend_threshold)
                calibrated_regimes = base_eval.regimes_from_values(y_calibrated, nrtend_threshold)
                base_eval.update_confusion(confusions["soft_baseline"], true_regimes, baseline_regimes)
                base_eval.update_confusion(confusions["soft_calibrated"], true_regimes, calibrated_regimes)

                for regime_idx, regime_name in enumerate(base_eval.REGIME_LABELS):
                    regime_mask = true_regimes == regime_idx
                    if np.any(regime_mask):
                        regime_metrics_by_mode["soft_baseline"][regime_name].update(
                            y_true[regime_mask],
                            y_soft[regime_mask],
                        )
                        regime_metrics_by_mode["soft_calibrated"][regime_name].update(
                            y_true[regime_mask],
                            y_calibrated[regime_mask],
                        )

            processed_batches += 1
            processed_samples += int(target_phys_model.shape[0])

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - eval_start_time).total_seconds()
                logger.info(
                    "Eval pass: processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    finalized_metrics = {
        mode: {key: metric.finalize() for key, metric in metrics_map.items()}
        for mode, metrics_map in metrics_by_mode.items()
    }
    finalized_regime_metrics = {
        mode: {regime: metric.finalize() for regime, metric in metrics_map.items()}
        for mode, metrics_map in regime_metrics_by_mode.items()
    }

    metrics_df = save_metrics_csv(finalized_metrics, args.output_dir)
    regime_df = save_regime_metrics_csv(finalized_regime_metrics, args.output_dir)
    save_confusions_csv(confusions, args.output_dir)
    comparison_df = build_mode_comparison(
        metrics_df=metrics_df,
        integration_r2_threshold=args.integration_r2_threshold,
        output_dir=args.output_dir,
    )

    plot_physical_metric_by_mode(
        comparison_df,
        metric="r2",
        ylabel="R2",
        output_path=args.output_dir / "physical_r2_by_mode.png",
        log_scale=False,
    )
    plot_physical_metric_by_mode(
        comparison_df,
        metric="rmse",
        ylabel="RMSE",
        output_path=args.output_dir / "physical_rmse_by_mode.png",
        log_scale=True,
    )
    plot_regime_r2_by_mode(regime_df, args.output_dir / "nrtend_regime_r2_by_mode.png")

    readiness = {}
    for mode in ("soft_baseline", "soft_calibrated"):
        per_var = {
            var: bool(
                np.isfinite(finalized_metrics[mode][var]["r2"])
                and finalized_metrics[mode][var]["r2"] >= args.integration_r2_threshold
            )
            for var in base_eval.MODEL_PRED_KEYS
        }
        readiness[mode] = {
            "meets_threshold_all_four": bool(all(per_var.values())),
            "variables_meeting_threshold": per_var,
            "n_variables_meeting_threshold": int(sum(per_var.values())),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "same_set_fit_and_eval": True,
        "fit_processed_batches": fit_processed_batches,
        "fit_processed_samples": fit_processed_samples,
        "fit_flagged_prediction_count": int(sum(threshold_flag_counts.values())),
        "processed_batches": processed_batches,
        "processed_samples_after_filtering": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "integration_r2_threshold": args.integration_r2_threshold,
        "validation_split": validation_summary,
        "calibrator": {
            "kind": "piecewise_affine_negative_tail",
            "thresholds_abs_pred": list(NEGATIVE_TAIL_THRESHOLDS),
            "coefficients": coefficients,
            "fit_rows_by_bin": {
                row["bin_label"]: int(row["count"]) for row in calibrator_rows
            },
            "flagged_prediction_rows_by_bin": threshold_flag_counts,
        },
        "readiness": readiness,
        "metrics": finalized_metrics,
        "nrtend_regime_metrics": finalized_regime_metrics,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Soft tail calibration evaluation complete.")
    logger.info(
        "Processed %d batches and %d filtered validation samples in evaluation pass.",
        processed_batches,
        processed_samples,
    )
    for mode in ("soft_baseline", "soft_calibrated"):
        logger.info("Physical metrics for %s:", mode)
        for key in base_eval.MODEL_PRED_KEYS:
            metrics = finalized_metrics[mode][key]
            logger.info(
                "  %-6s R2=% .4f RMSE=% .4e MAE=% .4e Bias=% .4e Pearson=% .4f",
                key,
                metrics["r2"],
                metrics["rmse"],
                metrics["mae"],
                metrics["bias"],
                metrics["pearson_r"],
            )
        logger.info(
            "  readiness: %d/%d variables meet physical R2 >= %.3f",
            readiness[mode]["n_variables_meeting_threshold"],
            len(base_eval.MODEL_PRED_KEYS),
            args.integration_r2_threshold,
        )


if __name__ == "__main__":
    main()
