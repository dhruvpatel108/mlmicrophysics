#!/usr/bin/env python3
"""
Targeted analysis of the residual soft-routing negative nrtend tail.

Purpose
-------
This script answers the follow-up questions after the hard-vs-soft routing
evaluation of run_594616:

1. Is the remaining soft-routing nrtend error localized to a tiny subset?
2. If those samples were filtered or sent to a fallback microphysics path,
   would the overall physical-space nrtend metrics become acceptable?
3. Is there evidence that those samples cluster in metadata or input-feature
   space strongly enough to support a deployable guardrail?

Key outputs
-----------
- counterfactual_metrics.csv
- threshold_overlap_summary.csv
- tail_group_summary.csv
- soft_negative_tail_samples.csv
- tail_file_summary.csv
- tail_metadata_top_values.csv
- tail_exact_cluster_summary.csv
- tail_feature_shifts.csv
- counterfactual_r2.png
- counterfactual_rmse.png
- tail_feature_shift_heatmap.png
- summary.json
- validation_files.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

import analyze_nrtend_problematic_samples as base_problem
import evaluate_moe_validation as base_eval
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


TRUE_NEGATIVE_TAIL_THRESHOLDS: Tuple[float, ...] = (1.0e4, 3.0e4, 1.0e5)
SOFT_PRED_NEGATIVE_THRESHOLDS: Tuple[float, ...] = (1.0e4, 3.0e4, 1.0e5)
TAIL_BIN_LABELS: Tuple[str, ...] = (
    "[1.0e+04, 3.0e+04)",
    "[3.0e+04, 1.0e+05)",
    "[1.0e+05, inf)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze residual soft-routing negative nrtend tails")
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


def scenario_metric_map() -> Dict[str, base_eval.RunningRegressionMetrics]:
    metrics = {"baseline_soft": base_eval.RunningRegressionMetrics()}

    for threshold in TRUE_NEGATIVE_TAIL_THRESHOLDS:
        label = f"{threshold:.0e}".replace("+", "")
        metrics[f"oracle_fallback_true_neg_abs_ge_{label}"] = base_eval.RunningRegressionMetrics()
        metrics[f"oracle_exclude_true_neg_abs_ge_{label}"] = base_eval.RunningRegressionMetrics()

    for threshold in SOFT_PRED_NEGATIVE_THRESHOLDS:
        label = f"{threshold:.0e}".replace("+", "")
        metrics[f"pred_fallback_soft_pred_le_-{label}"] = base_eval.RunningRegressionMetrics()
        metrics[f"pred_clip_soft_pred_floor_-{label}"] = base_eval.RunningRegressionMetrics()

    return metrics


def plot_counterfactual_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    plot_df = df.copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df[metric], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["scenario"], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"nrtend Physical {metric.upper()} Under Counterfactual Tail Handling")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shift_heatmap(feature_df: pd.DataFrame, output_path: Path) -> None:
    if feature_df.empty:
        logger.warning("No feature shift rows available for heatmap.")
        return

    pivot = feature_df.pivot(index="group_label", columns="feature", values="z_shift")
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    matrix = pivot.values.astype(np.float64)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(col_labels)), max(4, 0.65 * len(row_labels))))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Residual Soft Negative-Tail Feature Shift vs Negative Reference")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("z-shift = (mean_tail - mean_reference) / std_reference")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_counterfactual_row(
    scenario: str,
    metrics: Dict[str, float],
    flagged_count: int,
    total_samples: int,
    integration_r2_threshold: float,
) -> Dict[str, object]:
    row = {
        "scenario": scenario,
        "flagged_count": int(flagged_count),
        "flagged_fraction": flagged_count / total_samples if total_samples > 0 else math.nan,
        "meets_nrtend_r2_threshold": bool(
            np.isfinite(metrics["r2"]) and metrics["r2"] >= integration_r2_threshold
        ),
        "estimated_all_four_meet_threshold": bool(
            np.isfinite(metrics["r2"]) and metrics["r2"] >= integration_r2_threshold
        ),
    }
    row.update(metrics)
    return row


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
    validation_summary = base_eval.save_validation_files(val_dataset, args.output_dir)

    dataset_cols = list(getattr(val_dataset, "output_cols", []))
    input_cols = list(getattr(val_dataset, "input_cols", []))
    n_features = len(input_cols)

    nrtend_threshold = data_cfg.get("nrtend_regime_threshold")
    nrtend_threshold = float(nrtend_threshold) if nrtend_threshold is not None else None
    if nrtend_threshold is None:
        raise ValueError("This analysis expects data.nrtend_regime_threshold to be set.")

    integration_r2_threshold = 0.97
    metrics_by_scenario = scenario_metric_map()
    flagged_counts = {name: 0 for name in metrics_by_scenario}
    threshold_overlap_rows: List[Dict[str, object]] = []

    negative_all_stats = base_problem.RunningFeatureStats(n_features)
    negative_ref_stats = base_problem.RunningFeatureStats(n_features)  # negative abs(true) < 1e4
    tail_stats_by_group = {
        "ALL tail [>=1e4]": base_problem.RunningFeatureStats(n_features),
        "[1.0e+04, 3.0e+04)": base_problem.RunningFeatureStats(n_features),
        "[3.0e+04, 1.0e+05)": base_problem.RunningFeatureStats(n_features),
        "[1.0e+05, inf)": base_problem.RunningFeatureStats(n_features),
    }

    tail_rows: List[Dict[str, object]] = []
    processed_batches = 0
    processed_samples = 0
    start_time = pd.Timestamp.utcnow()

    with torch.no_grad():
        for batch_idx, batch in enumerate(base_problem.iter_validation_batches_with_metadata(val_dataset), start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            analysis = base_problem.analyze_nrtend_batch(
                model=model,
                inputs_scaled=batch["inputs_scaled"],
                targets_scaled_dataset=batch["targets_scaled_dataset"],
                dataset_cols=dataset_cols,
                dataset=val_dataset,
                nrtend_threshold=nrtend_threshold,
                device=device,
            )

            y_true = analysis["y_true_phys"]
            y_soft = analysis["y_soft_phys"]
            neg_mask = analysis["true_regime"] == 1
            abs_true = analysis["abs_true"]
            pred_soft = analysis["y_soft_phys"]

            metrics_by_scenario["baseline_soft"].update(y_true, y_soft)
            negative_all_stats.update(batch["inputs_scaled"][neg_mask])

            ref_mask = neg_mask & (abs_true < 1.0e4)
            if np.any(ref_mask):
                negative_ref_stats.update(batch["inputs_scaled"][ref_mask])

            oracle_flags: Dict[float, np.ndarray] = {}
            for threshold in TRUE_NEGATIVE_TAIL_THRESHOLDS:
                label = f"{threshold:.0e}".replace("+", "")
                flag = neg_mask & (abs_true >= threshold)
                oracle_flags[threshold] = flag
                flagged_counts[f"oracle_fallback_true_neg_abs_ge_{label}"] += int(np.sum(flag))
                flagged_counts[f"oracle_exclude_true_neg_abs_ge_{label}"] += int(np.sum(flag))

                y_fallback = y_soft.copy()
                y_fallback[flag] = y_true[flag]
                metrics_by_scenario[f"oracle_fallback_true_neg_abs_ge_{label}"].update(y_true, y_fallback)
                metrics_by_scenario[f"oracle_exclude_true_neg_abs_ge_{label}"].update(
                    y_true[~flag], y_soft[~flag]
                )

            for threshold in SOFT_PRED_NEGATIVE_THRESHOLDS:
                label = f"{threshold:.0e}".replace("+", "")
                flag = pred_soft <= -threshold
                flagged_counts[f"pred_fallback_soft_pred_le_-{label}"] += int(np.sum(flag))
                flagged_counts[f"pred_clip_soft_pred_floor_-{label}"] += int(np.sum(flag))

                y_fallback = y_soft.copy()
                y_fallback[flag] = y_true[flag]
                y_clipped = np.maximum(y_soft, -threshold)

                metrics_by_scenario[f"pred_fallback_soft_pred_le_-{label}"].update(y_true, y_fallback)
                metrics_by_scenario[f"pred_clip_soft_pred_floor_-{label}"].update(y_true, y_clipped)

                oracle_tail = oracle_flags[1.0e4]
                tp = int(np.sum(flag & oracle_tail))
                fp = int(np.sum(flag & ~oracle_tail))
                fn = int(np.sum(~flag & oracle_tail))
                precision = tp / (tp + fp) if (tp + fp) > 0 else math.nan
                recall = tp / (tp + fn) if (tp + fn) > 0 else math.nan
                threshold_overlap_rows.append({
                    "batch_idx": batch_idx,
                    "threshold": threshold,
                    "flagged_count_batch": int(np.sum(flag)),
                    "oracle_tail_count_batch": int(np.sum(oracle_tail)),
                    "tp_batch": tp,
                    "fp_batch": fp,
                    "fn_batch": fn,
                    "precision_batch": precision,
                    "recall_batch": recall,
                })

            tail_mask = oracle_flags[1.0e4]
            if np.any(tail_mask):
                tail_stats_by_group["ALL tail [>=1e4]"].update(batch["inputs_scaled"][tail_mask])

                tail_indices = np.where(tail_mask)[0]
                tail_inputs_scaled = batch["inputs_scaled"][tail_indices]
                tail_inputs_raw = base_problem.inverse_input_pipeline(tail_inputs_scaled, input_cols, val_dataset)
                tail_metadata_frame = (
                    batch["metadata_frame"].iloc[tail_indices].reset_index(drop=True)
                    if batch["metadata_frame"] is not None else None
                )

                for local_idx, idx in enumerate(tail_indices):
                    bin_label = base_problem.make_bin_label(
                        float(base_problem.MAG_BIN_EDGES[int(analysis["bin_idx"][idx])]),
                        float(base_problem.MAG_BIN_EDGES[int(analysis["bin_idx"][idx]) + 1]),
                    )
                    if bin_label in tail_stats_by_group:
                        tail_stats_by_group[bin_label].update(tail_inputs_scaled[local_idx:local_idx + 1])

                    metadata_row = (
                        tail_metadata_frame.iloc[local_idx].to_dict()
                        if tail_metadata_frame is not None else None
                    )
                    row = base_problem.build_sample_row(
                        idx=int(idx),
                        file_path=batch["file_path"],
                        row_in_file=batch["row_in_file"],
                        input_cols=input_cols,
                        input_scaled_row=tail_inputs_scaled[local_idx],
                        input_raw_row=tail_inputs_raw[local_idx],
                        true_regime=analysis["true_regime"],
                        bin_idx=analysis["bin_idx"],
                        y_true_phys=analysis["y_true_phys"],
                        y_pred_phys=analysis["y_pred_phys"],
                        y_soft_phys=analysis["y_soft_phys"],
                        y_true_scaled=analysis["y_true_scaled"],
                        y_pred_scaled=analysis["y_pred_scaled"],
                        y_soft_scaled=analysis["y_soft_scaled"],
                        y_true_postscaler=analysis["y_true_postscaler"],
                        y_pred_postscaler=analysis["y_pred_postscaler"],
                        y_soft_postscaler=analysis["y_soft_postscaler"],
                        abs_err=analysis["abs_err"],
                        soft_abs_err=analysis["soft_abs_err"],
                        err_ratio=analysis["error_ratio"],
                        soft_err_ratio=analysis["soft_error_ratio"],
                        mag_ratio=analysis["magnitude_ratio"],
                        soft_mag_ratio=analysis["soft_magnitude_ratio"],
                        selected_expert=analysis["selected_expert"],
                        router_probs=analysis["router_probs"],
                        metadata_row=metadata_row,
                        group_label="soft_negative_tail",
                    )
                    row["soft_sq_error"] = float(analysis["soft_abs_err"][idx] ** 2)
                    row["oracle_tail_flag"] = True
                    row["tail_bin_label"] = bin_label
                    for threshold in SOFT_PRED_NEGATIVE_THRESHOLDS:
                        label = f"{threshold:.0e}".replace("+", "")
                        row[f"flag_pred_le_-{label}"] = bool(analysis["y_soft_phys"][idx] <= -threshold)
                    tail_rows.append(row)

            processed_batches += 1
            processed_samples += int(y_true.size)

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    counterfactual_rows = []
    for scenario, metric in metrics_by_scenario.items():
        counterfactual_rows.append(
            build_counterfactual_row(
                scenario=scenario,
                metrics=metric.finalize(),
                flagged_count=flagged_counts[scenario],
                total_samples=processed_samples,
                integration_r2_threshold=integration_r2_threshold,
            )
        )
    counterfactual_df = pd.DataFrame(counterfactual_rows).sort_values("r2", ascending=False)
    counterfactual_df.to_csv(args.output_dir / "counterfactual_metrics.csv", index=False)

    overlap_df = pd.DataFrame(threshold_overlap_rows)
    if not overlap_df.empty:
        overlap_summary = (
            overlap_df.groupby("threshold", as_index=False)
            .agg(
                flagged_count=("flagged_count_batch", "sum"),
                oracle_tail_count=("oracle_tail_count_batch", "sum"),
                tp=("tp_batch", "sum"),
                fp=("fp_batch", "sum"),
                fn=("fn_batch", "sum"),
            )
        )
        overlap_summary["precision"] = overlap_summary["tp"] / (overlap_summary["tp"] + overlap_summary["fp"])
        overlap_summary["recall"] = overlap_summary["tp"] / (overlap_summary["tp"] + overlap_summary["fn"])
        overlap_summary.to_csv(args.output_dir / "threshold_overlap_summary.csv", index=False)
    else:
        overlap_summary = pd.DataFrame()
        overlap_summary.to_csv(args.output_dir / "threshold_overlap_summary.csv", index=False)

    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(args.output_dir / "soft_negative_tail_samples.csv", index=False)

    if not tail_df.empty:
        tail_group_summary = (
            tail_df.groupby("tail_bin_label", dropna=False)
            .agg(
                count=("row_in_file", "size"),
                mean_true=("nrtend_true_physical", "mean"),
                mean_soft_pred=("nrtend_soft_pred_physical", "mean"),
                mean_soft_abs_error=("nrtend_soft_abs_error", "mean"),
                rmse_soft=("soft_sq_error", lambda s: math.sqrt(float(np.mean(s)))),
                mean_router_confidence=("router_confidence", "mean"),
            )
            .reset_index()
            .sort_values("count", ascending=False)
        )
        tail_group_summary.to_csv(args.output_dir / "tail_group_summary.csv", index=False)

        tail_file_summary = (
            tail_df.groupby("file_path", dropna=False)
            .agg(
                count=("row_in_file", "size"),
                mean_soft_abs_error=("nrtend_soft_abs_error", "mean"),
                mean_soft_sq_error=("soft_sq_error", "mean"),
                mean_router_confidence=("router_confidence", "mean"),
            )
            .reset_index()
            .sort_values(["count", "mean_soft_sq_error"], ascending=[False, False])
        )
        tail_file_summary["fraction_of_tail"] = tail_file_summary["count"] / len(tail_df)
        tail_file_summary.to_csv(args.output_dir / "tail_file_summary.csv", index=False)

        metadata_cols_present = [col for col in tail_df.columns if col.startswith("metadata__")]
        metadata_rows = []
        for col in metadata_cols_present:
            value_counts = tail_df[col].value_counts(dropna=False).head(50)
            for value, count in value_counts.items():
                metadata_rows.append({
                    "metadata_col": col.replace("metadata__", "", 1),
                    "metadata_value": value,
                    "count": int(count),
                    "fraction_of_tail": count / len(tail_df),
                })
        pd.DataFrame(metadata_rows).to_csv(args.output_dir / "tail_metadata_top_values.csv", index=False)

        exact_group_cols = ["file_path"]
        for candidate in ("metadata__time", "metadata__ncol", "metadata__lev"):
            if candidate in tail_df.columns:
                exact_group_cols.append(candidate)
        tail_exact_cluster = (
            tail_df.groupby(exact_group_cols, dropna=False)
            .agg(
                count=("row_in_file", "size"),
                mean_soft_abs_error=("nrtend_soft_abs_error", "mean"),
                mean_soft_sq_error=("soft_sq_error", "mean"),
                mean_router_confidence=("router_confidence", "mean"),
            )
            .reset_index()
            .sort_values(["count", "mean_soft_sq_error"], ascending=[False, False])
        )
        tail_exact_cluster["fraction_of_tail"] = tail_exact_cluster["count"] / len(tail_df)
        tail_exact_cluster.head(500).to_csv(args.output_dir / "tail_exact_cluster_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "tail_group_summary.csv", index=False)
        pd.DataFrame().to_csv(args.output_dir / "tail_file_summary.csv", index=False)
        pd.DataFrame().to_csv(args.output_dir / "tail_metadata_top_values.csv", index=False)
        pd.DataFrame().to_csv(args.output_dir / "tail_exact_cluster_summary.csv", index=False)

    feature_rows = []
    ref_mean = negative_ref_stats.mean()
    ref_std = negative_ref_stats.std()
    ref_count = negative_ref_stats.count
    all_neg_count = negative_all_stats.count
    for group_label, stats in tail_stats_by_group.items():
        group_mean = stats.mean()
        group_count = stats.count
        for j, feature in enumerate(input_cols):
            denom = ref_std[j] if np.isfinite(ref_std[j]) and ref_std[j] > 0 else np.nan
            z_shift = (group_mean[j] - ref_mean[j]) / denom if np.isfinite(denom) else math.nan
            feature_rows.append({
                "group_label": group_label,
                "feature": feature,
                "reference_group": "negative abs(true) < 1e4",
                "reference_count": ref_count,
                "all_negative_count": all_neg_count,
                "group_count": group_count,
                "reference_mean_scaled": ref_mean[j],
                "reference_std_scaled": ref_std[j],
                "group_mean_scaled": group_mean[j],
                "z_shift": z_shift,
            })
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(args.output_dir / "tail_feature_shifts.csv", index=False)

    plot_counterfactual_metric(
        counterfactual_df,
        metric="r2",
        ylabel="Physical R2",
        output_path=args.output_dir / "counterfactual_r2.png",
        log_scale=False,
    )
    positive_rmse_df = counterfactual_df[counterfactual_df["rmse"] > 0].copy()
    plot_counterfactual_metric(
        positive_rmse_df,
        metric="rmse",
        ylabel="Physical RMSE",
        output_path=args.output_dir / "counterfactual_rmse.png",
        log_scale=True,
    )
    plot_feature_shift_heatmap(feature_df, args.output_dir / "tail_feature_shift_heatmap.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "integration_r2_threshold": integration_r2_threshold,
        "validation_split": validation_summary,
        "true_negative_tail_thresholds": list(TRUE_NEGATIVE_TAIL_THRESHOLDS),
        "soft_prediction_thresholds": list(SOFT_PRED_NEGATIVE_THRESHOLDS),
        "tail_row_count": int(len(tail_df)),
        "tail_fraction": len(tail_df) / processed_samples if processed_samples > 0 else math.nan,
        "best_r2_scenario": (
            counterfactual_df.iloc[counterfactual_df["r2"].idxmax()]["scenario"]
            if not counterfactual_df.empty else None
        ),
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Soft negative-tail analysis complete.")
    logger.info("Processed %d batches and %d validation samples.", processed_batches, processed_samples)
    if not counterfactual_df.empty:
        best_row = counterfactual_df.sort_values("r2", ascending=False).iloc[0]
        logger.info(
            "Best nrtend counterfactual: %s with R2=%.6f and flagged fraction %.6f",
            best_row["scenario"],
            best_row["r2"],
            best_row["flagged_fraction"],
        )


if __name__ == "__main__":
    main()
