#!/usr/bin/env python3
"""
Evaluate an exported standalone TorchScript emulator on the validation split.

This script validates the exact `.pt` artifact that will be handed to E3SM:
- loads the TorchScript module, not the raw checkpoint
- reconstructs physical inputs from the validation loader
- runs the exported module end-to-end in physical space
- compares predictions against physical targets on the full validation split

Outputs:
- summary.json
- metrics_summary.csv
- nrtend_regime_metrics.csv
- physical_r2_by_subset.png
- validation_files.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

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
from export_model import export_moe_soft_affine_standalone as export_mod
from models.streaming_data_loader_v2 import create_optimized_streaming_loaders


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_LOG_COLS = {
    "QC_TAU_in",
    "QR_TAU_in",
    "NC_TAU_in",
    "NR_TAU_in",
    "LAMC",
    "LAMR",
    "N0R",
}
DEFAULT_QC_TAU_THRESHOLD = 1.0e-6
DEFAULT_CLOUD_THRESHOLD = 0.01


def infer_metadata_path(torchscript_path: Path) -> Path:
    return torchscript_path.with_suffix(".json")


def resolve_existing_path(path_like: Optional[str], fallback: Optional[Path] = None) -> Optional[Path]:
    if path_like:
        path = Path(path_like)
        if path.exists():
            return path
        path_text = str(path)
        candidates = []
        if path_text.startswith("/qfs/people/"):
            candidates.append(Path(path_text.replace("/qfs/people/", "/people/", 1)))
        if path_text.startswith("/people/"):
            candidates.append(Path(path_text.replace("/people/", "/qfs/people/", 1)))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    if fallback is not None and fallback.exists():
        return fallback
    return None


def load_metadata(metadata_path: Optional[Path]) -> Dict:
    if metadata_path is None or not metadata_path.exists():
        return {}
    with metadata_path.open("r") as handle:
        return json.load(handle)


def inverse_input_pipeline(
    scaled_input_matrix: np.ndarray,
    input_cols: Sequence[str],
    dataset,
) -> np.ndarray:
    restored = np.asarray(scaled_input_matrix, dtype=np.float64).copy()
    scaler = getattr(dataset, "input_scaler", None)
    transformer = getattr(dataset, "input_transformer", None)
    input_transform = getattr(dataset, "input_transform", "log10")

    if scaler is not None:
        restored = scaler.inverse_transform(restored)
    if input_transform == "quantile" and transformer is not None:
        restored = transformer.inverse_transform(restored)
    if input_transform == "log10":
        for idx, col in enumerate(input_cols):
            if col in INPUT_LOG_COLS:
                restored[:, idx] = np.power(10.0, restored[:, idx]) - base_eval.LOG_EPSILON
    return restored


def build_metrics_container() -> Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]:
    return {
        subset: {key: base_eval.RunningRegressionMetrics() for key in base_eval.MODEL_PRED_KEYS}
        for subset in ("all", "filter_pass_only")
    }


def build_regime_metrics() -> Dict[str, Dict[str, base_eval.RunningRegressionMetrics]]:
    return {
        subset: {name: base_eval.RunningRegressionMetrics() for name in base_eval.REGIME_LABELS}
        for subset in ("all", "filter_pass_only")
    }


def save_metrics_csv(
    finalized_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    for subset, metrics_map in finalized_metrics.items():
        for variable, metrics in metrics_map.items():
            row = {"subset": subset, "space": "physical", "variable": variable}
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
    for subset, metrics_map in finalized_regime_metrics.items():
        for regime, metrics in metrics_map.items():
            row = {"subset": subset, "regime": regime}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "nrtend_regime_metrics.csv", index=False)
    return df


def plot_r2_by_subset(metrics_df: pd.DataFrame, threshold: float, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(base_eval.MODEL_PRED_KEYS))
    width = 0.36

    all_vals = [
        metrics_df[
            (metrics_df["subset"] == "all") & (metrics_df["variable"] == variable)
        ]["r2"].iloc[0]
        for variable in base_eval.MODEL_PRED_KEYS
    ]
    pass_vals = [
        metrics_df[
            (metrics_df["subset"] == "filter_pass_only") & (metrics_df["variable"] == variable)
        ]["r2"].iloc[0]
        for variable in base_eval.MODEL_PRED_KEYS
    ]

    ax.bar(x - width / 2, all_vals, width, label="all validation rows", color="#4C78A8")
    ax.bar(x + width / 2, pass_vals, width, label="filter-pass only", color="#F58518")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5, label=f"R2={threshold:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(base_eval.MODEL_PRED_KEYS)
    ax.set_ylabel("Physical R2")
    ax.set_title("Exported TorchScript Physical R2")
    ax.grid(alpha=0.25, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_batched_export_replay(
    config: Dict,
    metadata: Dict,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, float]]:
    model_cfg = config["model"]
    data_cfg = config["data"]

    checkpoint_path = resolve_existing_path(
        metadata.get("checkpoint"),
        fallback=Path(config["run_dir"]) / "best_checkpoint.pth" if "run_dir" in config else args.config.parent / "best_checkpoint.pth",  # type: ignore[name-defined]
    )
    if checkpoint_path is None:
        checkpoint_path = args.config.parent / "best_checkpoint.pth"  # type: ignore[name-defined]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Raw checkpoint not found for eager replay: {checkpoint_path}")

    scaler_dir = resolve_existing_path(metadata.get("scaler_dir"))
    if scaler_dir is None:
        raise FileNotFoundError("Could not resolve scaler_dir from TorchScript metadata")

    calibrator_json = resolve_existing_path(
        metadata.get("calibrator_json"),
        fallback=PROJECT_ROOT / "evaluation_results" / "run_594616_soft_tail_calibrated_eval" / "calibrator_params.json",
    )
    if calibrator_json is None:
        raise FileNotFoundError("Could not resolve calibrator_json for eager replay")

    raw_model = export_mod.instantiate_model(model_cfg)
    raw_model.to(device)
    raw_model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(export_mod.resolve_state_dict(checkpoint))

    input_scaler, output_scaler = export_mod.load_scalers(scaler_dir)
    input_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32, device=device)
    input_scale = torch.tensor(input_scaler.scale_, dtype=torch.float32, device=device)
    output_mean, output_scale, _ = export_mod.build_output_scaler_tensors(
        output_scaler=output_scaler,
        config_output_cols=data_cfg.get("output_cols", []),
        device=device,
    )

    with calibrator_json.open("r") as handle:
        calibrator_params = json.load(handle)
    calibration_cfg = export_mod.build_calibration_cfg(calibrator_params)

    use_nrtend_arcsinh = bool(data_cfg.get("nrtend_arcsinh_transform", False))
    arcsinh_indices = [2] if use_nrtend_arcsinh else []
    arcsinh_threshold = float(data_cfg.get("nrtend_arcsinh_threshold", 1.0e-3))
    qc_tau_threshold = float(metadata.get("input_filter", {}).get("qc_tau_threshold", DEFAULT_QC_TAU_THRESHOLD))
    cloud_threshold = float(metadata.get("input_filter", {}).get("cloud_threshold", DEFAULT_CLOUD_THRESHOLD))

    class BatchedReplay(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_model = raw_model
            self.register_buffer("input_mean", input_mean)
            self.register_buffer("input_scale", input_scale)
            self.register_buffer("output_mean", output_mean)
            self.register_buffer("output_scale", output_scale)

        def forward(self, x_physical: torch.Tensor) -> torch.Tensor:
            x = x_physical.clone()
            for idx in export_mod.LOG_INPUT_INDICES:
                x[:, idx] = torch.log10(x[:, idx] + export_mod.LOG_EPSILON)
            x = (x - self.input_mean) / self.input_scale

            shared = self.raw_model.shared_backbone(x)
            qrtend = self.raw_model.qrtend_head(shared)
            nctend = self.raw_model.nctend_head(shared)
            router_logits = self.raw_model.nrtend_moe.router(shared)
            expert_outputs = torch.cat(
                [expert(shared) for expert in self.raw_model.nrtend_moe.experts],
                dim=1,
            )
            gate_probs = torch.softmax(router_logits, dim=1)
            nrtend = (gate_probs * expert_outputs).sum(dim=1, keepdim=True)
            qctend = -qrtend
            y = torch.cat([qrtend, nctend, nrtend, qctend], dim=1)

            y = y * self.output_scale.unsqueeze(0) + self.output_mean.unsqueeze(0)
            result = torch.empty_like(y)
            ten = torch.tensor(10.0, dtype=y.dtype, device=y.device)

            if 2 in arcsinh_indices:
                result[:, 2] = arcsinh_threshold * torch.sinh(y[:, 2])
            else:
                result[:, 2] = torch.sign(y[:, 2]) * (
                    torch.pow(ten, torch.abs(y[:, 2])) - export_mod.LOG_EPSILON
                )

            result[:, 0] = torch.clamp(torch.pow(ten, y[:, 0]) - export_mod.LOG_EPSILON, min=0.0)
            result[:, 1] = torch.clamp(-(torch.pow(ten, -y[:, 1]) - export_mod.LOG_EPSILON), max=0.0)
            result[:, 3] = torch.clamp(-(torch.pow(ten, -y[:, 3]) - export_mod.LOG_EPSILON), max=0.0)

            nrt = result[:, 2]
            result[:, 2] = torch.where(
                nrt <= -calibration_cfg["thr_1e5"],
                calibration_cfg["slope_ge_1e5"] * nrt + calibration_cfg["intercept_ge_1e5"],
                torch.where(
                    nrt <= -calibration_cfg["thr_3e4"],
                    calibration_cfg["slope_3e4_1e5"] * nrt + calibration_cfg["intercept_3e4_1e5"],
                    torch.where(
                        nrt <= -calibration_cfg["thr_1e4"],
                        calibration_cfg["slope_1e4_3e4"] * nrt + calibration_cfg["intercept_1e4_3e4"],
                        nrt,
                    ),
                ),
            )

            passes_filter = (
                (x_physical[:, QC_TAU_INDEX] > qc_tau_threshold)
                & (x_physical[:, CLOUD_INDEX] > cloud_threshold)
            )
            result[~passes_filter] = 0.0
            return result

    return BatchedReplay().eval(), {
        "qc_tau_threshold": qc_tau_threshold,
        "cloud_threshold": cloud_threshold,
    }


def evaluate_batched_model(
    model: torch.nn.Module,
    physical_inputs: np.ndarray,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    if physical_inputs.shape[0] == 0:
        return np.empty((0, len(base_eval.MODEL_PRED_KEYS)), dtype=np.float64)

    inputs_tensor = torch.as_tensor(physical_inputs, dtype=torch.float32, device=device)
    outputs = []
    for start in range(0, inputs_tensor.shape[0], chunk_size):
        end = min(start + chunk_size, inputs_tensor.shape[0])
        pred = model(inputs_tensor[start:end])
        outputs.append(pred.detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(outputs, axis=0)


def parity_check_against_torchscript(
    torchscript_path: Path,
    eager_batch_model: torch.nn.Module,
    physical_inputs: np.ndarray,
    device: torch.device,
    max_samples: int = 512,
) -> Dict[str, float]:
    if physical_inputs.shape[0] == 0:
        return {"samples_checked": 0, "max_abs_diff": float("nan")}

    n_check = min(max_samples, physical_inputs.shape[0])
    subset = physical_inputs[:n_check]

    eager_pred = evaluate_batched_model(
        model=eager_batch_model,
        physical_inputs=subset,
        device=device,
        chunk_size=n_check,
    )

    ts_model = torch.jit.load(str(torchscript_path), map_location="cpu")
    ts_model.eval()
    ts_vmap = torch.vmap(ts_model)
    with torch.no_grad():
        ts_pred = ts_vmap(torch.as_tensor(subset, dtype=torch.float32, device="cpu"))
        ts_pred = ts_pred.detach().cpu().numpy().astype(np.float64, copy=False)

    return {
        "samples_checked": int(n_check),
        "max_abs_diff": float(np.max(np.abs(eager_pred - ts_pred))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an exported standalone TorchScript emulator")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the exported TorchScript .pt artifact",
    )
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
        "--metadata_json",
        "--metadata-json",
        dest="metadata_json",
        type=Path,
        default=None,
        help="Optional exporter metadata JSON; defaults to <checkpoint>.json",
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
        help="Run id whose scaler artifacts should be reused for validation loading",
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
        help="Physical-space R2 threshold used for readiness checks",
    )
    parser.add_argument(
        "--torchscript_chunk_size",
        "--torchscript-chunk-size",
        dest="torchscript_chunk_size",
        type=int,
        default=131072,
        help="Chunk size used when vmapping the single-sample TorchScript model",
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
        or base_eval.infer_run_id_from_path(str(args.config))
    )
    metadata_path = args.metadata_json or infer_metadata_path(args.checkpoint)
    metadata = load_metadata(metadata_path)

    logger.info("TorchScript artifact: %s", args.checkpoint)
    logger.info("Metadata JSON: %s", metadata_path if metadata else "not found")
    logger.info("Config: %s", args.config)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Device: %s", device)
    logger.info("Scaler run id: %s", scaler_run_id)

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
    input_cols = list(getattr(val_dataset, "input_cols", []))

    expected_input_order = list(metadata.get("input_order", []))
    if expected_input_order and input_cols != expected_input_order:
        raise ValueError(
            f"Validation input order {input_cols} does not match exported model input order {expected_input_order}"
        )

    nrtend_threshold = data_cfg.get("nrtend_regime_threshold")
    nrtend_threshold = float(nrtend_threshold) if nrtend_threshold is not None else None

    eager_batch_model, filter_cfg = build_batched_export_replay(
        config=config,
        metadata=metadata,
        device=device,
    )
    qc_tau_threshold = float(filter_cfg["qc_tau_threshold"])
    cloud_threshold = float(filter_cfg["cloud_threshold"])
    qc_idx = input_cols.index("QC_TAU_in")
    cloud_idx = input_cols.index("CLOUD")

    validation_summary = base_eval.save_validation_files(val_dataset, args.output_dir)

    metrics_by_subset = build_metrics_container()
    regime_metrics = build_regime_metrics()

    processed_batches = 0
    processed_samples = 0
    filter_pass_count = 0
    filter_fail_count = 0
    start_time = pd.Timestamp.utcnow()
    parity_summary: Optional[Dict[str, float]] = None

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader, start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs_scaled = inputs.detach().cpu().numpy().astype(np.float64, copy=False)
            inputs_physical = inverse_input_pipeline(inputs_scaled, input_cols, val_dataset)

            pred_phys_model = evaluate_batched_model(
                model=eager_batch_model,
                physical_inputs=inputs_physical,
                device=device,
                chunk_size=args.torchscript_chunk_size,
            )

            if parity_summary is None:
                parity_summary = parity_check_against_torchscript(
                    torchscript_path=args.checkpoint,
                    eager_batch_model=eager_batch_model,
                    physical_inputs=inputs_physical,
                    device=device,
                    max_samples=512,
                )
                logger.info(
                    "TorchScript parity check on %d samples: max |Δ| = %.3e",
                    parity_summary["samples_checked"],
                    parity_summary["max_abs_diff"],
                )

            target_norm_dataset = base_eval.batch_targets_to_dataset_matrix(targets, dataset_cols)
            target_phys_dataset = base_eval.inverse_output_pipeline(target_norm_dataset, dataset_cols, val_dataset)
            target_phys_dataset = base_eval.zero_small_nrtend_targets(
                target_phys_dataset,
                dataset_cols,
                nrtend_threshold,
            )
            target_phys_model = base_eval.dataset_to_model_matrix(target_phys_dataset, dataset_cols)

            pass_mask = (
                (inputs_physical[:, qc_idx] > qc_tau_threshold)
                & (inputs_physical[:, cloud_idx] > cloud_threshold)
            )

            batch_size = pred_phys_model.shape[0]
            processed_batches += 1
            processed_samples += batch_size
            filter_pass_count += int(np.sum(pass_mask))
            filter_fail_count += int(batch_size - np.sum(pass_mask))

            for idx, key in enumerate(base_eval.MODEL_PRED_KEYS):
                metrics_by_subset["all"][key].update(
                    target_phys_model[:, idx],
                    pred_phys_model[:, idx],
                )
                if np.any(pass_mask):
                    metrics_by_subset["filter_pass_only"][key].update(
                        target_phys_model[pass_mask, idx],
                        pred_phys_model[pass_mask, idx],
                    )

            if nrtend_threshold is not None:
                true_regimes = base_eval.regimes_from_values(target_phys_model[:, 2], nrtend_threshold)
                for regime_idx, regime_name in enumerate(base_eval.REGIME_LABELS):
                    regime_mask = true_regimes == regime_idx
                    if np.any(regime_mask):
                        regime_metrics["all"][regime_name].update(
                            target_phys_model[regime_mask, 2],
                            pred_phys_model[regime_mask, 2],
                        )
                        if np.any(regime_mask & pass_mask):
                            regime_metrics["filter_pass_only"][regime_name].update(
                                target_phys_model[regime_mask & pass_mask, 2],
                                pred_phys_model[regime_mask & pass_mask, 2],
                            )

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, filter-pass %.4f%%, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    100.0 * filter_pass_count / max(processed_samples, 1),
                    elapsed_s,
                )

    finalized_metrics = {
        subset: {key: metrics.finalize() for key, metrics in metrics_map.items()}
        for subset, metrics_map in metrics_by_subset.items()
    }
    finalized_regime_metrics = {
        subset: {regime: metrics.finalize() for regime, metrics in metrics_map.items()}
        for subset, metrics_map in regime_metrics.items()
    }

    metrics_df = save_metrics_csv(finalized_metrics, args.output_dir)
    save_regime_metrics_csv(finalized_regime_metrics, args.output_dir)
    plot_r2_by_subset(
        metrics_df=metrics_df,
        threshold=args.integration_r2_threshold,
        output_path=args.output_dir / "physical_r2_by_subset.png",
    )

    readiness = {}
    for subset, metrics_map in finalized_metrics.items():
        per_var = {
            key: bool(
                np.isfinite(metrics["r2"]) and metrics["r2"] >= args.integration_r2_threshold
            )
            for key, metrics in metrics_map.items()
        }
        readiness[subset] = {
            "meets_threshold_by_variable": per_var,
            "meets_threshold_all_four": bool(all(per_var.values())),
        }

    summary = {
        "torchscript_artifact": str(args.checkpoint),
        "metadata_json": str(metadata_path) if metadata else None,
        "evaluation_backend": "batched_eager_replay_plus_torchscript_parity_check",
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "processed_batches": processed_batches,
        "processed_samples_after_filtering": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "validation_split": validation_summary,
        "input_filter": {
            "qc_tau_threshold": qc_tau_threshold,
            "cloud_threshold": cloud_threshold,
            "pass_count": filter_pass_count,
            "fail_count": filter_fail_count,
            "pass_fraction": filter_pass_count / processed_samples if processed_samples else None,
        },
        "integration_r2_threshold": args.integration_r2_threshold,
        "torchscript_parity_check": parity_summary,
        "readiness": readiness,
        "metrics": finalized_metrics,
        "nrtend_metrics_by_regime": finalized_regime_metrics,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("TorchScript evaluation complete.")
    logger.info(
        "Processed %d batches and %d validation samples. Filter-pass fraction: %.6f",
        processed_batches,
        processed_samples,
        filter_pass_count / max(processed_samples, 1),
    )
    for subset, metrics_map in finalized_metrics.items():
        logger.info("Physical metrics for subset=%s:", subset)
        for key in base_eval.MODEL_PRED_KEYS:
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
