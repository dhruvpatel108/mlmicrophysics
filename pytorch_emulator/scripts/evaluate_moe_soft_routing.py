#!/usr/bin/env python3
"""
Evaluate a MoE checkpoint with soft routing forced at inference.

This script compares two inference modes on the exact same validation pass:
- hard: current model.eval() behavior using argmax expert selection
- soft: weighted mixture of experts using router probabilities

The main purpose is to answer the E3SM integration question in physical space:
does forcing soft routing at inference materially improve the physical-space
metrics, and is the checkpoint close to an acceptable readiness threshold?

Outputs:
- summary.json
- metrics_summary.csv
- physical_mode_comparison.csv
- nrtend_regime_metrics.csv
- nrtend_regime_confusion.csv
- physical_r2_by_mode.png
- physical_rmse_by_mode.png
- nrtend_regime_r2_by_mode.png
- validation_files.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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

import evaluate_moe_validation as base_eval
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_hard_and_soft_predictions(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> Dict[str, np.ndarray]:
    shared_features = model.shared_backbone(inputs)

    qrtend = model.qrtend_head(shared_features)
    nctend = model.nctend_head(shared_features)

    router_logits = model.nrtend_moe.router(shared_features)
    expert_outputs = torch.cat(
        [expert(shared_features) for expert in model.nrtend_moe.experts],
        dim=-1,
    )  # [B, 3]
    gate_probs = torch.softmax(router_logits, dim=-1)

    hard_expert = torch.argmax(router_logits, dim=-1)
    hard_nrtend = expert_outputs.gather(1, hard_expert.unsqueeze(-1))
    soft_nrtend = (gate_probs * expert_outputs).sum(dim=-1, keepdim=True)
    qctend = -qrtend

    def stack_prediction(nrtend_tensor: torch.Tensor) -> np.ndarray:
        return np.concatenate(
            [
                qrtend.detach().cpu().numpy(),
                nctend.detach().cpu().numpy(),
                nrtend_tensor.detach().cpu().numpy(),
                qctend.detach().cpu().numpy(),
            ],
            axis=1,
        ).astype(np.float64, copy=False)

    return {
        "hard": stack_prediction(hard_nrtend),
        "soft": stack_prediction(soft_nrtend),
        "hard_expert": hard_expert.detach().cpu().numpy().astype(np.int64, copy=False),
        "gate_probs": gate_probs.detach().cpu().numpy().astype(np.float64, copy=False),
    }


def build_mode_metrics() -> Dict[str, Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]]:
    return {
        mode: {
            "transformed": {
                key: base_eval.RunningRegressionMetrics() for key in base_eval.MODEL_PRED_KEYS
            },
            "physical": {
                key: base_eval.RunningRegressionMetrics() for key in base_eval.MODEL_PRED_KEYS
            },
        }
        for mode in ("hard", "soft")
    }


def build_regime_metrics() -> Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]:
    return {
        mode: {name: base_eval.RunningRegressionMetrics() for name in base_eval.REGIME_LABELS}
        for mode in ("hard", "soft")
    }


def build_confusions() -> Dict[str, np.ndarray]:
    return {mode: np.zeros((3, 3), dtype=np.int64) for mode in ("hard", "soft")}


def save_metrics_csv(
    finalized_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for mode, spaces in finalized_metrics.items():
        for space, metrics_map in spaces.items():
            for variable, metrics in metrics_map.items():
                row = {"routing_mode": mode, "space": space, "variable": variable}
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
            row = {"routing_mode": mode, "regime": regime}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "nrtend_regime_metrics.csv", index=False)
    return df


def save_confusions_csv(confusions: Dict[str, np.ndarray], output_dir: Path) -> pd.DataFrame:
    rows = []
    for mode, matrix in confusions.items():
        for true_idx, true_regime in enumerate(base_eval.REGIME_LABELS):
            row = {"routing_mode": mode, "true_regime": true_regime}
            for pred_idx, pred_regime in enumerate(base_eval.REGIME_LABELS):
                row[f"pred_{pred_regime}"] = int(matrix[true_idx, pred_idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "nrtend_regime_confusion.csv", index=False)
    return df


def build_physical_comparison(
    metrics_df: pd.DataFrame,
    integration_r2_threshold: float,
    output_dir: Path,
) -> pd.DataFrame:
    physical = metrics_df[metrics_df["space"] == "physical"].copy()
    pivot = physical.pivot(index="variable", columns="routing_mode")

    rows = []
    for variable in base_eval.MODEL_PRED_KEYS:
        row = {"variable": variable}
        for metric in ("r2", "rmse", "mae", "bias", "pearson_r"):
            hard_val = pivot[(metric, "hard")].get(variable, np.nan)
            soft_val = pivot[(metric, "soft")].get(variable, np.nan)
            row[f"hard_{metric}"] = hard_val
            row[f"soft_{metric}"] = soft_val
            row[f"delta_{metric}_soft_minus_hard"] = soft_val - hard_val
        row["hard_meets_r2_threshold"] = bool(
            np.isfinite(row["hard_r2"]) and row["hard_r2"] >= integration_r2_threshold
        )
        row["soft_meets_r2_threshold"] = bool(
            np.isfinite(row["soft_r2"]) and row["soft_r2"] >= integration_r2_threshold
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

    ax.bar(x - width / 2, comparison_df[f"hard_{metric}"], width, label="hard", color="#4C78A8")
    ax.bar(x + width / 2, comparison_df[f"soft_{metric}"], width, label="soft", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["variable"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"Physical {metric.upper()} by Routing Mode")
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

    hard_vals = [
        regime_df[(regime_df["routing_mode"] == "hard") & (regime_df["regime"] == regime)]["r2"].iloc[0]
        for regime in base_eval.REGIME_LABELS
    ]
    soft_vals = [
        regime_df[(regime_df["routing_mode"] == "soft") & (regime_df["regime"] == regime)]["r2"].iloc[0]
        for regime in base_eval.REGIME_LABELS
    ]

    ax.bar(x - width / 2, hard_vals, width, label="hard", color="#4C78A8")
    ax.bar(x + width / 2, soft_vals, width, label="soft", color="#F58518")

    ax.set_xticks(x)
    ax.set_xticklabels(base_eval.REGIME_LABELS)
    ax.set_ylabel("R2")
    ax.set_title("nrtend Physical R2 by Regime and Routing Mode")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a MoE checkpoint with hard vs forced-soft routing at inference"
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

    metrics_by_mode = build_mode_metrics()
    regime_metrics_by_mode = build_regime_metrics()
    confusions = build_confusions()

    processed_batches = 0
    processed_samples = 0
    start_time = pd.Timestamp.utcnow()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs = inputs.to(device)
            predictions = compute_hard_and_soft_predictions(model, inputs)
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
            target_norm_adjusted_dataset = base_eval.forward_output_pipeline(
                target_phys_dataset,
                dataset_cols,
                val_dataset,
            )

            target_norm_model = base_eval.dataset_to_model_matrix(target_norm_adjusted_dataset, dataset_cols)
            target_phys_model = base_eval.dataset_to_model_matrix(target_phys_dataset, dataset_cols)

            for mode in ("hard", "soft"):
                pred_norm_model = predictions[mode]
                pred_norm_dataset = base_eval.model_to_dataset_matrix(pred_norm_model, dataset_cols)
                pred_phys_dataset = base_eval.inverse_output_pipeline(
                    pred_norm_dataset,
                    dataset_cols,
                    val_dataset,
                )
                pred_phys_model = base_eval.dataset_to_model_matrix(pred_phys_dataset, dataset_cols)

                for idx, key in enumerate(base_eval.MODEL_PRED_KEYS):
                    metrics_by_mode[mode]["transformed"][key].update(
                        target_norm_model[:, idx], pred_norm_model[:, idx]
                    )
                    metrics_by_mode[mode]["physical"][key].update(
                        target_phys_model[:, idx], pred_phys_model[:, idx]
                    )

                if nrtend_threshold is not None:
                    nrt_true = target_phys_model[:, 2]
                    nrt_pred = pred_phys_model[:, 2]
                    true_regimes = base_eval.regimes_from_values(nrt_true, nrtend_threshold)
                    pred_regimes = base_eval.regimes_from_values(nrt_pred, nrtend_threshold)
                    base_eval.update_confusion(confusions[mode], true_regimes, pred_regimes)

                    for regime_idx, regime_name in enumerate(base_eval.REGIME_LABELS):
                        regime_mask = true_regimes == regime_idx
                        if np.any(regime_mask):
                            regime_metrics_by_mode[mode][regime_name].update(
                                nrt_true[regime_mask],
                                nrt_pred[regime_mask],
                            )

            processed_batches += 1
            processed_samples += int(target_norm_model.shape[0])

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    finalized_metrics = {
        mode: {
            space: {key: metric.finalize() for key, metric in metrics_map.items()}
            for space, metrics_map in spaces.items()
        }
        for mode, spaces in metrics_by_mode.items()
    }
    finalized_regime_metrics = {
        mode: {regime: metric.finalize() for regime, metric in metrics_map.items()}
        for mode, metrics_map in regime_metrics_by_mode.items()
    }

    metrics_df = save_metrics_csv(finalized_metrics, args.output_dir)
    regime_df = save_regime_metrics_csv(finalized_regime_metrics, args.output_dir)
    save_confusions_csv(confusions, args.output_dir)
    comparison_df = build_physical_comparison(
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
    for mode in ("hard", "soft"):
        physical_metrics = finalized_metrics[mode]["physical"]
        per_var = {
            var: bool(
                np.isfinite(physical_metrics[var]["r2"])
                and physical_metrics[var]["r2"] >= args.integration_r2_threshold
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
        "processed_batches": processed_batches,
        "processed_samples_after_filtering": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "integration_r2_threshold": args.integration_r2_threshold,
        "validation_split": validation_summary,
        "readiness": readiness,
        "metrics": finalized_metrics,
        "nrtend_regime_metrics": finalized_regime_metrics,
        "nrtend_regime_confusion": {mode: matrix.tolist() for mode, matrix in confusions.items()},
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Soft-routing comparison evaluation complete.")
    logger.info(
        "Processed %d batches and %d filtered validation samples.",
        processed_batches,
        processed_samples,
    )
    for mode in ("hard", "soft"):
        logger.info("Physical metrics for %s routing:", mode)
        for key in base_eval.MODEL_PRED_KEYS:
            metrics = finalized_metrics[mode]["physical"][key]
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
