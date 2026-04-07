#!/usr/bin/env python3
"""
Preflight comparison for a quantile+robust warm-start checkpoint.

This script compares a standard checkpoint such as run_105047 in physical space
under two preprocessing regimes:
1. Original fitted artifacts in the checkpoint's original output-column order
2. Freshly fitted quantile+robust artifacts using model output order

It is designed as a Phase-0 decision gate before launching a new MoE training
run that warm-starts from the standard checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

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

from models.physics_emulator import ConstraintAwareEmulator
from models.streaming_data_loader_v2 import OptimizedStreamingDataset


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
PHYSICAL_SIGN_BY_COL = {
    "qrtend_TAU": +1,
    "nctend_TAU": -1,
    "nrtend_TAU": 0,
    "qctend_TAU": -1,
}
REGIME_LABELS = ("near_zero", "negative", "positive")
LOG_EPSILON = 1.0e-10
DEFAULT_NRTEND_REGIME_THRESHOLD = 1.0e-6


@dataclass
class ArtifactBundle:
    input_scaler: object
    output_scaler: object
    output_transformer: Optional[object]
    output_cols: Tuple[str, ...]
    input_transform: str
    output_transform: str
    nrtend_arcsinh_transform: bool
    nrtend_arcsinh_threshold: float
    artifact_dir: Path


@dataclass
class RunningRegressionMetrics:
    n: int = 0
    sum_true: float = 0.0
    sum_pred: float = 0.0
    sum_true_sq: float = 0.0
    sum_pred_sq: float = 0.0
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

        self.n += int(y_true.size)
        self.sum_true += float(np.sum(y_true))
        self.sum_pred += float(np.sum(y_pred))
        self.sum_true_sq += float(np.dot(y_true, y_true))
        self.sum_pred_sq += float(np.dot(y_pred, y_pred))
        self.sum_abs_err += float(np.sum(np.abs(err)))
        self.sum_sq_err += float(np.dot(err, err))
        self.sum_err += float(np.sum(err))

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
            }

        mean_true = self.sum_true / self.n
        mean_pred = self.sum_pred / self.n
        var_true = max(self.sum_true_sq / self.n - mean_true**2, 0.0)
        var_pred = max(self.sum_pred_sq / self.n - mean_pred**2, 0.0)
        sst = self.sum_true_sq - (self.sum_true**2) / self.n

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


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def instantiate_model(model_cfg: Dict) -> torch.nn.Module:
    return ConstraintAwareEmulator(
        input_dim=model_cfg.get("input_dim", 11),
        shared_dims=model_cfg.get("shared_dims", [256, 256, 256, 128, 128]),
        head_dims=model_cfg.get("head_dims", [128, 128, 64, 64, 32]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=model_cfg.get("activation", "relu"),
    )


def load_model(checkpoint_path: Path, config: Dict, device: torch.device) -> torch.nn.Module:
    model = instantiate_model(config["model"])
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


def batch_predictions_to_model_matrix(predictions: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate(
        [predictions[key].detach().cpu().numpy() for key in MODEL_PRED_KEYS],
        axis=1,
    ).astype(np.float64, copy=False)


def inverse_output_pipeline(
    scaled_dataset_matrix: np.ndarray,
    artifacts: ArtifactBundle,
) -> np.ndarray:
    restored = np.asarray(scaled_dataset_matrix, dtype=np.float64).copy()

    if artifacts.output_scaler is not None:
        restored = artifacts.output_scaler.inverse_transform(restored)
    if artifacts.output_transform == "quantile" and artifacts.output_transformer is not None:
        restored = artifacts.output_transformer.inverse_transform(restored)

    if artifacts.nrtend_arcsinh_transform and "nrtend_TAU" in artifacts.output_cols:
        nrt_idx = artifacts.output_cols.index("nrtend_TAU")
        restored[:, nrt_idx] = artifacts.nrtend_arcsinh_threshold * np.sinh(restored[:, nrt_idx])

    if artifacts.output_transform == "log10":
        for idx, col in enumerate(artifacts.output_cols):
            if artifacts.nrtend_arcsinh_transform and col == "nrtend_TAU":
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
    artifacts: ArtifactBundle,
) -> np.ndarray:
    transformed = np.asarray(physical_dataset_matrix, dtype=np.float64).copy()

    if artifacts.output_transform == "log10":
        for idx, col in enumerate(artifacts.output_cols):
            if artifacts.nrtend_arcsinh_transform and col == "nrtend_TAU":
                continue
            if col not in LOG_OUTPUT_COLS:
                continue
            values = transformed[:, idx]
            transformed[:, idx] = np.sign(values) * np.log10(np.abs(values) + LOG_EPSILON)

    if artifacts.nrtend_arcsinh_transform and "nrtend_TAU" in artifacts.output_cols:
        nrt_idx = artifacts.output_cols.index("nrtend_TAU")
        transformed[:, nrt_idx] = np.arcsinh(
            transformed[:, nrt_idx] / artifacts.nrtend_arcsinh_threshold
        )

    if artifacts.output_transform == "quantile" and artifacts.output_transformer is not None:
        transformed = artifacts.output_transformer.transform(transformed)
    if artifacts.output_scaler is not None:
        transformed = artifacts.output_scaler.transform(transformed)

    return transformed


def apply_target_only_nrtend_rule(
    physical_model_matrix: np.ndarray,
    threshold: float,
) -> np.ndarray:
    adjusted = np.asarray(physical_model_matrix, dtype=np.float64).copy()
    adjusted[np.abs(adjusted[:, 2]) < threshold, 2] = 0.0
    return adjusted


def regimes_from_values(values: np.ndarray, threshold: float) -> np.ndarray:
    regimes = np.zeros(values.shape[0], dtype=np.int64)
    regimes[values < -threshold] = 1
    regimes[values > threshold] = 2
    return regimes


def stable_file_seed(file_path: Path, seed: int) -> int:
    payload = f"{seed}:{file_path.name}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def iter_parquet_chunks(file_path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    try:
        yield from pd.read_parquet(file_path, chunksize=chunk_size)
    except TypeError:
        full_data = pd.read_parquet(file_path)
        for start_idx in range(0, len(full_data), chunk_size):
            yield full_data.iloc[start_idx:start_idx + chunk_size]


def predict_scaled_outputs(
    model: torch.nn.Module,
    input_scaled: np.ndarray,
    device: torch.device,
    inference_batch_size: int,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start_idx in range(0, input_scaled.shape[0], inference_batch_size):
            end_idx = min(start_idx + inference_batch_size, input_scaled.shape[0])
            batch = torch.tensor(input_scaled[start_idx:end_idx], dtype=torch.float32, device=device)
            predictions = model(batch)
            outputs.append(batch_predictions_to_model_matrix(predictions))
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 4), dtype=np.float64)


def compute_split_counts(total_files: int, train_fraction: float) -> Tuple[int, int]:
    n_train = int(total_files * train_fraction)
    return n_train, total_files - n_train


def build_dataset(
    data_config: Dict,
    split: str,
    scaler_path: Path,
    output_scaler_path: Path,
    output_transformer_path: Optional[Path],
    output_cols: Sequence[str],
) -> OptimizedStreamingDataset:
    return OptimizedStreamingDataset(
        data_path=data_config["data_path"],
        input_cols=list(data_config["input_cols"]),
        output_cols=list(output_cols),
        batch_size=int(data_config.get("batch_size", 1024)),
        chunk_size=int(data_config.get("chunk_size", 50000)),
        max_files=data_config.get("max_files"),
        sample_fraction=1.0,
        active_threshold=float(data_config.get("active_threshold", 1.0e-15) or 1.0e-15),
        cloud_threshold=data_config.get("cloud_threshold", 0.01),
        mass_input_threshold=data_config.get("mass_input_threshold", 1.0e-5),
        rho_threshold=data_config.get("rho_threshold", 0.2),
        nrtend_threshold=data_config.get("nrtend_threshold", 1.0e-10),
        random_seed=int(data_config.get("random_seed", 42)),
        split=split,
        train_fraction=float(data_config.get("train_fraction", 0.8)),
        scaler_path=str(scaler_path),
        output_scaler_path=str(output_scaler_path),
        output_transformer_path=str(output_transformer_path) if output_transformer_path is not None else None,
        mode="transform",
        disable_length_estimation=True,
        input_transform=str(data_config.get("input_transform", "log10")).lower(),
        output_transform=str(data_config.get("output_transform", "log10")).lower(),
        input_scaling=str(data_config.get("input_scaling", "standard")).lower(),
        output_scaling=str(data_config.get("output_scaling", "standard")).lower(),
        quantile_n_quantiles=int(data_config.get("quantile_n_quantiles", 1000)),
        quantile_subsample=int(data_config.get("quantile_subsample", 100000)),
        rank=0,
        world_size=1,
        nrtend_arcsinh_transform=bool(data_config.get("nrtend_arcsinh_transform", False)),
        nrtend_arcsinh_threshold=float(data_config.get("nrtend_arcsinh_threshold", 1.0e-3)),
        nrtend_regime_threshold=data_config.get("nrtend_regime_threshold"),
    )


def ensure_quantile_robust_config(config: Dict) -> None:
    data_cfg = config["data"]
    output_transform = str(data_cfg.get("output_transform", "log10")).lower()
    output_scaling = str(data_cfg.get("output_scaling", "standard")).lower()
    if output_transform != "quantile" or output_scaling != "robust":
        raise ValueError(
            "This preflight script is intended for quantile+robust output preprocessing. "
            f"Found output_transform='{output_transform}', output_scaling='{output_scaling}'."
        )


def load_artifact_bundle(
    artifact_dir: Path,
    output_cols: Sequence[str],
    config: Dict,
) -> ArtifactBundle:
    input_scaler_path = artifact_dir / "input_scaler_optimized.pkl"
    output_scaler_path = artifact_dir / "output_scaler_optimized.pkl"
    output_transformer_path = artifact_dir / "output_quantile_transformer.pkl"

    if not input_scaler_path.exists():
        raise FileNotFoundError(f"Input scaler not found: {input_scaler_path}")
    if not output_scaler_path.exists():
        raise FileNotFoundError(f"Output scaler not found: {output_scaler_path}")

    output_transform = str(config["data"].get("output_transform", "log10")).lower()
    output_transformer = None
    if output_transform == "quantile":
        if not output_transformer_path.exists():
            raise FileNotFoundError(f"Output transformer not found: {output_transformer_path}")
        output_transformer = load_pickle(output_transformer_path)

    return ArtifactBundle(
        input_scaler=load_pickle(input_scaler_path),
        output_scaler=load_pickle(output_scaler_path),
        output_transformer=output_transformer,
        output_cols=tuple(output_cols),
        input_transform=str(config["data"].get("input_transform", "log10")).lower(),
        output_transform=output_transform,
        nrtend_arcsinh_transform=bool(config["data"].get("nrtend_arcsinh_transform", False)),
        nrtend_arcsinh_threshold=float(config["data"].get("nrtend_arcsinh_threshold", 1.0e-3)),
        artifact_dir=artifact_dir,
    )


def maybe_fit_fresh_artifacts(
    config: Dict,
    output_dir: Path,
    args: argparse.Namespace,
) -> Tuple[ArtifactBundle, bool, int, int]:
    fresh_dir = output_dir / "fresh_artifacts"
    fresh_dir.mkdir(parents=True, exist_ok=True)

    input_scaler_path = fresh_dir / "input_scaler_optimized.pkl"
    output_scaler_path = fresh_dir / "output_scaler_optimized.pkl"
    output_transformer_path = fresh_dir / "output_quantile_transformer.pkl"

    artifacts_exist = (
        input_scaler_path.exists()
        and output_scaler_path.exists()
        and output_transformer_path.exists()
    )
    reused = artifacts_exist and not args.force_refit_fresh_artifacts

    fresh_config = copy.deepcopy(config)
    fresh_config["data"] = copy.deepcopy(config["data"])
    fresh_config["data"]["output_cols"] = list(MODEL_OUTPUT_COLS)

    requested_fit_samples = (
        int(args.fit_sample_cap)
        if args.fit_sample_cap is not None
        else int(fresh_config["data"].get("scaler_fit_samples", 100000))
    )
    requested_quantile_subsample = (
        int(args.quantile_subsample_cap)
        if args.quantile_subsample_cap is not None
        else int(fresh_config["data"].get("quantile_subsample", 100000))
    )
    requested_quantile_n_quantiles = (
        int(args.quantile_n_quantiles_cap)
        if args.quantile_n_quantiles_cap is not None
        else int(fresh_config["data"].get("quantile_n_quantiles", 1000))
    )

    if not reused:
        logger.info("Fitting fresh model-order artifacts into %s", fresh_dir)
        fit_dataset = OptimizedStreamingDataset(
            data_path=fresh_config["data"]["data_path"],
            input_cols=list(fresh_config["data"]["input_cols"]),
            output_cols=list(MODEL_OUTPUT_COLS),
            batch_size=int(fresh_config["data"].get("batch_size", 1024)),
            chunk_size=int(fresh_config["data"].get("chunk_size", 50000)),
            max_files=fresh_config["data"].get("max_files"),
            sample_fraction=1.0,
            active_threshold=float(fresh_config["data"].get("active_threshold", 1.0e-15) or 1.0e-15),
            cloud_threshold=fresh_config["data"].get("cloud_threshold", 0.01),
            mass_input_threshold=fresh_config["data"].get("mass_input_threshold", 1.0e-5),
            rho_threshold=fresh_config["data"].get("rho_threshold", 0.2),
            nrtend_threshold=fresh_config["data"].get("nrtend_threshold", 1.0e-10),
            random_seed=int(fresh_config["data"].get("random_seed", 42)),
            split="train",
            train_fraction=float(fresh_config["data"].get("train_fraction", 0.8)),
            scaler_path=str(input_scaler_path),
            output_scaler_path=str(output_scaler_path),
            output_transformer_path=str(output_transformer_path),
            mode="fit_transform",
            disable_length_estimation=True,
            input_transform=str(fresh_config["data"].get("input_transform", "log10")).lower(),
            output_transform=str(fresh_config["data"].get("output_transform", "quantile")).lower(),
            input_scaling=str(fresh_config["data"].get("input_scaling", "standard")).lower(),
            output_scaling=str(fresh_config["data"].get("output_scaling", "robust")).lower(),
            quantile_n_quantiles=requested_quantile_n_quantiles,
            quantile_subsample=requested_quantile_subsample,
            rank=0,
            world_size=1,
            nrtend_arcsinh_transform=bool(fresh_config["data"].get("nrtend_arcsinh_transform", False)),
            nrtend_arcsinh_threshold=float(fresh_config["data"].get("nrtend_arcsinh_threshold", 1.0e-3)),
            nrtend_regime_threshold=fresh_config["data"].get("nrtend_regime_threshold"),
        )
        if args.max_train_files is not None:
            fit_dataset.active_files = fit_dataset.active_files[:int(args.max_train_files)]
        fit_dataset.quantile_subsample = requested_quantile_subsample
        fit_dataset.quantile_n_quantiles = requested_quantile_n_quantiles
        fit_dataset.fit_scalers(n_samples_for_fitting=requested_fit_samples)

    fresh_artifacts = load_artifact_bundle(fresh_dir, MODEL_OUTPUT_COLS, fresh_config)
    return fresh_artifacts, reused, requested_fit_samples, requested_quantile_subsample


def plot_metrics_for_space(
    metrics_rows: pd.DataFrame,
    output_path: Path,
    space: str,
    space_label: str,
) -> None:
    plot_df = metrics_rows[
        (metrics_rows["space"] == space) & (metrics_rows["mode"].isin(["original", "fresh"]))
    ].copy()
    variables = [col.replace("_TAU", "") for col in MODEL_OUTPUT_COLS]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("r2", "R2", False),
        ("rmse", "RMSE", True),
        ("mae", "MAE", True),
    ]
    x = np.arange(len(variables))
    width = 0.36

    for ax, (metric_key, title, use_log) in zip(axes, metrics):
        orig_vals = []
        fresh_vals = []
        for variable in MODEL_OUTPUT_COLS:
            sub = plot_df[plot_df["variable"] == variable]
            orig_vals.append(float(sub[sub["mode"] == "original"][metric_key].iloc[0]))
            fresh_vals.append(float(sub[sub["mode"] == "fresh"][metric_key].iloc[0]))

        ax.bar(x - width / 2, orig_vals, width=width, label="original")
        ax.bar(x + width / 2, fresh_vals, width=width, label="fresh")
        ax.set_xticks(x, variables)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    fig.suptitle(f"{space_label} Preflight Metrics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_nrtend_regime_metrics(
    regime_rows: pd.DataFrame,
    output_path: Path,
    space: str,
    space_label: str,
) -> None:
    plot_df = regime_rows[
        (regime_rows["space"] == space) & (regime_rows["mode"].isin(["original", "fresh"]))
    ].copy()
    regimes = list(REGIME_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("r2", "Regime R2", False),
        ("rmse", "Regime RMSE", True),
        ("mae", "Regime MAE", True),
        ("near_zero_leak_fraction", "Near-Zero Leak Fraction", False),
    ]
    x = np.arange(len(regimes))
    width = 0.36

    for ax, (metric_key, title, use_log) in zip(axes.flatten(), panels):
        if metric_key == "near_zero_leak_fraction" and space == "physical":
            leak_df = plot_df[plot_df["regime"] == "near_zero"]
            orig_val = float(leak_df[leak_df["mode"] == "original"][metric_key].iloc[0])
            fresh_val = float(leak_df[leak_df["mode"] == "fresh"][metric_key].iloc[0])
            ax.bar([0 - width / 2], [orig_val], width=width, label="original")
            ax.bar([0 + width / 2], [fresh_val], width=width, label="fresh")
            ax.set_xticks([0], ["near_zero"])
        elif metric_key == "near_zero_leak_fraction":
            ax.text(0.5, 0.5, "Physical-space only metric", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            orig_vals = []
            fresh_vals = []
            for regime in regimes:
                sub = plot_df[plot_df["regime"] == regime]
                orig_vals.append(float(sub[sub["mode"] == "original"][metric_key].iloc[0]))
                fresh_vals.append(float(sub[sub["mode"] == "fresh"][metric_key].iloc[0]))
            ax.bar(x - width / 2, orig_vals, width=width, label="original")
            ax.bar(x + width / 2, fresh_vals, width=width, label="fresh")
            ax.set_xticks(x, regimes)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    fig.suptitle(f"nrtend {space_label} Regime Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_r2_space_gap(space_gap_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    variables = [col.replace("_TAU", "") for col in MODEL_OUTPUT_COLS]
    x = np.arange(len(variables))
    width = 0.36

    for ax, mode in zip(axes, ["original", "fresh"]):
        sub = space_gap_df[space_gap_df["mode"] == mode].copy()
        transformed_vals = []
        physical_vals = []
        for variable in MODEL_OUTPUT_COLS:
            row = sub[sub["variable"] == variable].iloc[0]
            transformed_vals.append(float(row["transformed_r2"]))
            physical_vals.append(float(row["physical_r2"]))

        ax.bar(x - width / 2, transformed_vals, width=width, label="transformed")
        ax.bar(x + width / 2, physical_vals, width=width, label="physical")
        ax.set_xticks(x, variables)
        ax.set_title(mode)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("R2 by Space on the Same Sample", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight quantile+robust warm-start comparison")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to standard checkpoint")
    parser.add_argument("--config", type=Path, required=True, help="Path to config_used.yml")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for preflight artifacts")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device")
    parser.add_argument("--sample_probability", type=float, default=0.005, help="Bernoulli sampling probability per post-filter row")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Base seed for deterministic per-file Bernoulli sampling")
    parser.add_argument("--nrtend_regime_threshold", type=float, default=DEFAULT_NRTEND_REGIME_THRESHOLD, help="Target-only zeroing and regime threshold for nrtend")
    parser.add_argument("--inference_batch_size", type=int, default=262144, help="Max inference batch size per forward pass")
    parser.add_argument("--force_refit_fresh_artifacts", action="store_true", help="Refit fresh model-order artifacts even if cached in output_dir")
    parser.add_argument("--max_train_files", type=int, default=None, help="Optional cap for training files during fresh artifact fitting smoke tests")
    parser.add_argument("--max_val_files", type=int, default=None, help="Optional cap for validation files during smoke tests")
    parser.add_argument("--fit_sample_cap", type=int, default=None, help="Optional override for scaler_fit_samples during smoke tests")
    parser.add_argument("--quantile_subsample_cap", type=int, default=None, help="Optional override for quantile_subsample during smoke tests")
    parser.add_argument("--quantile_n_quantiles_cap", type=int, default=None, help="Optional override for quantile_n_quantiles during smoke tests")
    parser.add_argument("--log_every", type=int, default=10, help="Log progress every N validation files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    ensure_quantile_robust_config(config)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_run_id = normalize_run_id(infer_run_id_from_path(str(args.checkpoint)))
    scaler_cache_dir = Path(config["data"]["scaler_cache_dir"])
    if checkpoint_run_id is None:
        raise ValueError(f"Could not infer run id from checkpoint path: {args.checkpoint}")
    original_artifact_dir = scaler_cache_dir / checkpoint_run_id
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Config: %s", args.config)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Original artifact dir: %s", original_artifact_dir)
    logger.info("Device: %s", device)
    logger.info("Sampling probability: %.6f", args.sample_probability)

    model = load_model(args.checkpoint, config, device)

    original_output_cols = tuple(config["data"]["output_cols"])
    original_artifacts = load_artifact_bundle(original_artifact_dir, original_output_cols, config)
    fresh_artifacts, fresh_reused, requested_fit_samples, requested_quantile_subsample = maybe_fit_fresh_artifacts(
        config,
        args.output_dir,
        args,
    )

    base_dataset = build_dataset(
        data_config=config["data"],
        split="val",
        scaler_path=original_artifact_dir / "input_scaler_optimized.pkl",
        output_scaler_path=original_artifact_dir / "output_scaler_optimized.pkl",
        output_transformer_path=original_artifact_dir / "output_quantile_transformer.pkl",
        output_cols=original_output_cols,
    )

    total_files = len(base_dataset.parquet_files)
    expected_train_files, expected_val_files = compute_split_counts(
        total_files=total_files,
        train_fraction=float(config["data"].get("train_fraction", 0.8)),
    )
    if args.max_val_files is not None:
        base_dataset.active_files = base_dataset.active_files[:int(args.max_val_files)]
    val_files = list(base_dataset.active_files)
    logger.info(
        "Validation split: total=%d, expected_train=%d, expected_val=%d, using_val=%d",
        total_files,
        expected_train_files,
        expected_val_files,
        len(val_files),
    )

    metrics = {
        "physical": {
            "original": {col: RunningRegressionMetrics() for col in MODEL_OUTPUT_COLS},
            "fresh": {col: RunningRegressionMetrics() for col in MODEL_OUTPUT_COLS},
        },
        "transformed": {
            "original": {col: RunningRegressionMetrics() for col in MODEL_OUTPUT_COLS},
            "fresh": {col: RunningRegressionMetrics() for col in MODEL_OUTPUT_COLS},
        },
    }
    regime_metrics = {
        "physical": {
            "original": {regime: RunningRegressionMetrics() for regime in REGIME_LABELS},
            "fresh": {regime: RunningRegressionMetrics() for regime in REGIME_LABELS},
        },
        "transformed": {
            "original": {regime: RunningRegressionMetrics() for regime in REGIME_LABELS},
            "fresh": {regime: RunningRegressionMetrics() for regime in REGIME_LABELS},
        },
    }
    near_zero_counts = {
        "original": {"count": 0, "leak_count": 0},
        "fresh": {"count": 0, "leak_count": 0},
    }

    sampling_rows: List[Dict[str, object]] = []
    total_prefiltered_rows = 0
    total_sampled_rows = 0

    for file_index, file_path in enumerate(val_files, start=1):
        file_rng = np.random.default_rng(stable_file_seed(file_path, args.sampling_seed))
        prefiltered_rows = 0
        sampled_rows = 0

        for chunk in iter_parquet_chunks(file_path, base_dataset.chunk_size):
            processed_chunk = base_dataset._preprocess_chunk_vectorized(chunk)
            if processed_chunk is None or len(processed_chunk) == 0:
                continue

            prefiltered_rows += len(processed_chunk)
            if args.sample_probability >= 1.0:
                sample_mask = np.ones(len(processed_chunk), dtype=bool)
            else:
                sample_mask = file_rng.random(len(processed_chunk)) < args.sample_probability
            if not np.any(sample_mask):
                continue

            sampled_chunk = processed_chunk.loc[sample_mask].copy()
            sampled_rows += len(sampled_chunk)

            input_matrix = sampled_chunk[list(config["data"]["input_cols"])].values.astype(np.float64, copy=False)
            target_phys_model = sampled_chunk[list(MODEL_OUTPUT_COLS)].values.astype(np.float64, copy=False)
            target_phys_model = apply_target_only_nrtend_rule(target_phys_model, args.nrtend_regime_threshold)
            regimes = regimes_from_values(target_phys_model[:, 2], args.nrtend_regime_threshold)
            target_phys_original_order = sampled_chunk[list(original_output_cols)].values.astype(np.float64, copy=False)
            target_phys_original_order = target_phys_original_order.copy()
            nrt_original_idx = original_output_cols.index("nrtend_TAU")
            target_phys_original_order[
                np.abs(target_phys_original_order[:, nrt_original_idx]) < args.nrtend_regime_threshold,
                nrt_original_idx,
            ] = 0.0

            original_input_scaled = original_artifacts.input_scaler.transform(input_matrix).astype(np.float32, copy=False)
            original_pred_scaled_model = predict_scaled_outputs(
                model,
                original_input_scaled,
                device=device,
                inference_batch_size=int(args.inference_batch_size),
            )
            original_target_scaled_dataset = forward_output_pipeline(target_phys_original_order, original_artifacts)
            original_target_scaled_model = dataset_to_model_matrix(
                original_target_scaled_dataset,
                original_artifacts.output_cols,
            )
            original_pred_scaled_dataset = model_to_dataset_matrix(original_pred_scaled_model, original_artifacts.output_cols)
            original_pred_phys_dataset = inverse_output_pipeline(original_pred_scaled_dataset, original_artifacts)
            original_pred_phys_model = dataset_to_model_matrix(original_pred_phys_dataset, original_artifacts.output_cols)

            fresh_input_scaled = fresh_artifacts.input_scaler.transform(input_matrix).astype(np.float32, copy=False)
            fresh_pred_scaled_model = predict_scaled_outputs(
                model,
                fresh_input_scaled,
                device=device,
                inference_batch_size=int(args.inference_batch_size),
            )
            fresh_target_scaled_dataset = forward_output_pipeline(target_phys_model, fresh_artifacts)
            fresh_target_scaled_model = dataset_to_model_matrix(
                fresh_target_scaled_dataset,
                fresh_artifacts.output_cols,
            )
            fresh_pred_scaled_dataset = model_to_dataset_matrix(fresh_pred_scaled_model, fresh_artifacts.output_cols)
            fresh_pred_phys_dataset = inverse_output_pipeline(fresh_pred_scaled_dataset, fresh_artifacts)
            fresh_pred_phys_model = dataset_to_model_matrix(fresh_pred_phys_dataset, fresh_artifacts.output_cols)

            for idx, col in enumerate(MODEL_OUTPUT_COLS):
                metrics["physical"]["original"][col].update(target_phys_model[:, idx], original_pred_phys_model[:, idx])
                metrics["physical"]["fresh"][col].update(target_phys_model[:, idx], fresh_pred_phys_model[:, idx])
                metrics["transformed"]["original"][col].update(
                    original_target_scaled_model[:, idx],
                    original_pred_scaled_model[:, idx],
                )
                metrics["transformed"]["fresh"][col].update(
                    fresh_target_scaled_model[:, idx],
                    fresh_pred_scaled_model[:, idx],
                )

            for regime_idx, regime_name in enumerate(REGIME_LABELS):
                regime_mask = regimes == regime_idx
                if not np.any(regime_mask):
                    continue
                regime_metrics["physical"]["original"][regime_name].update(
                    target_phys_model[regime_mask, 2],
                    original_pred_phys_model[regime_mask, 2],
                )
                regime_metrics["physical"]["fresh"][regime_name].update(
                    target_phys_model[regime_mask, 2],
                    fresh_pred_phys_model[regime_mask, 2],
                )
                regime_metrics["transformed"]["original"][regime_name].update(
                    original_target_scaled_model[regime_mask, 2],
                    original_pred_scaled_model[regime_mask, 2],
                )
                regime_metrics["transformed"]["fresh"][regime_name].update(
                    fresh_target_scaled_model[regime_mask, 2],
                    fresh_pred_scaled_model[regime_mask, 2],
                )
                if regime_name == "near_zero":
                    near_zero_counts["original"]["count"] += int(np.sum(regime_mask))
                    near_zero_counts["fresh"]["count"] += int(np.sum(regime_mask))
                    near_zero_counts["original"]["leak_count"] += int(
                        np.sum(np.abs(original_pred_phys_model[regime_mask, 2]) >= args.nrtend_regime_threshold)
                    )
                    near_zero_counts["fresh"]["leak_count"] += int(
                        np.sum(np.abs(fresh_pred_phys_model[regime_mask, 2]) >= args.nrtend_regime_threshold)
                    )

        realized_fraction = sampled_rows / prefiltered_rows if prefiltered_rows > 0 else math.nan
        sampling_rows.append(
            {
                "file_index": file_index,
                "file_name": file_path.name,
                "file_path": str(file_path),
                "postfilter_rows": int(prefiltered_rows),
                "sampled_rows": int(sampled_rows),
                "realized_fraction": realized_fraction,
                "sampled_any_rows": bool(sampled_rows > 0),
            }
        )
        total_prefiltered_rows += int(prefiltered_rows)
        total_sampled_rows += int(sampled_rows)

        if args.log_every > 0 and file_index % args.log_every == 0:
            logger.info(
                "Processed %d/%d validation files, sampled %d/%d rows so far",
                file_index,
                len(val_files),
                total_sampled_rows,
                total_prefiltered_rows,
            )

    metrics_rows: List[Dict[str, object]] = []
    metrics_final: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        "physical": {"original": {}, "fresh": {}, "delta": {}},
        "transformed": {"original": {}, "fresh": {}, "delta": {}},
    }
    for space in ("physical", "transformed"):
        for variable in MODEL_OUTPUT_COLS:
            orig = metrics[space]["original"][variable].finalize()
            fresh = metrics[space]["fresh"][variable].finalize()
            delta = {
                "count": int(orig["count"]),
                "r2": fresh["r2"] - orig["r2"] if np.isfinite(fresh["r2"]) and np.isfinite(orig["r2"]) else math.nan,
                "rmse": fresh["rmse"] - orig["rmse"] if np.isfinite(fresh["rmse"]) and np.isfinite(orig["rmse"]) else math.nan,
                "mae": fresh["mae"] - orig["mae"] if np.isfinite(fresh["mae"]) and np.isfinite(orig["mae"]) else math.nan,
                "bias": fresh["bias"] - orig["bias"] if np.isfinite(fresh["bias"]) and np.isfinite(orig["bias"]) else math.nan,
                "mean_true": fresh["mean_true"] - orig["mean_true"] if np.isfinite(fresh["mean_true"]) and np.isfinite(orig["mean_true"]) else math.nan,
                "mean_pred": fresh["mean_pred"] - orig["mean_pred"] if np.isfinite(fresh["mean_pred"]) and np.isfinite(orig["mean_pred"]) else math.nan,
                "std_true": fresh["std_true"] - orig["std_true"] if np.isfinite(fresh["std_true"]) and np.isfinite(orig["std_true"]) else math.nan,
                "std_pred": fresh["std_pred"] - orig["std_pred"] if np.isfinite(fresh["std_pred"]) and np.isfinite(orig["std_pred"]) else math.nan,
                "relative_rmse_change": (
                    (fresh["rmse"] - orig["rmse"]) / orig["rmse"]
                    if np.isfinite(fresh["rmse"]) and np.isfinite(orig["rmse"]) and orig["rmse"] != 0
                    else math.nan
                ),
            }
            metrics_final[space]["original"][variable] = orig
            metrics_final[space]["fresh"][variable] = fresh
            metrics_final[space]["delta"][variable] = delta
            metrics_rows.append({"space": space, "mode": "original", "variable": variable, "relative_rmse_change": math.nan, **orig})
            metrics_rows.append({"space": space, "mode": "fresh", "variable": variable, "relative_rmse_change": math.nan, **fresh})
            metrics_rows.append({"space": space, "mode": "delta", "variable": variable, **delta})

    regime_rows: List[Dict[str, object]] = []
    regime_final: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {
        "physical": {"original": {}, "fresh": {}, "delta": {}},
        "transformed": {"original": {}, "fresh": {}, "delta": {}},
    }
    for space in ("physical", "transformed"):
        for regime_name in REGIME_LABELS:
            orig = regime_metrics[space]["original"][regime_name].finalize()
            fresh = regime_metrics[space]["fresh"][regime_name].finalize()
            orig_leak = (
                near_zero_counts["original"]["leak_count"] / near_zero_counts["original"]["count"]
                if space == "physical" and regime_name == "near_zero" and near_zero_counts["original"]["count"] > 0
                else math.nan
            )
            fresh_leak = (
                near_zero_counts["fresh"]["leak_count"] / near_zero_counts["fresh"]["count"]
                if space == "physical" and regime_name == "near_zero" and near_zero_counts["fresh"]["count"] > 0
                else math.nan
            )
            delta = {
                "count": int(orig["count"]),
                "r2": fresh["r2"] - orig["r2"] if np.isfinite(fresh["r2"]) and np.isfinite(orig["r2"]) else math.nan,
                "rmse": fresh["rmse"] - orig["rmse"] if np.isfinite(fresh["rmse"]) and np.isfinite(orig["rmse"]) else math.nan,
                "mae": fresh["mae"] - orig["mae"] if np.isfinite(fresh["mae"]) and np.isfinite(orig["mae"]) else math.nan,
                "bias": fresh["bias"] - orig["bias"] if np.isfinite(fresh["bias"]) and np.isfinite(orig["bias"]) else math.nan,
                "mean_true": fresh["mean_true"] - orig["mean_true"] if np.isfinite(fresh["mean_true"]) and np.isfinite(orig["mean_true"]) else math.nan,
                "mean_pred": fresh["mean_pred"] - orig["mean_pred"] if np.isfinite(fresh["mean_pred"]) and np.isfinite(orig["mean_pred"]) else math.nan,
                "std_true": fresh["std_true"] - orig["std_true"] if np.isfinite(fresh["std_true"]) and np.isfinite(orig["std_true"]) else math.nan,
                "std_pred": fresh["std_pred"] - orig["std_pred"] if np.isfinite(fresh["std_pred"]) and np.isfinite(orig["std_pred"]) else math.nan,
                "near_zero_leak_fraction": (
                    fresh_leak - orig_leak if np.isfinite(fresh_leak) and np.isfinite(orig_leak) else math.nan
                ),
                "relative_rmse_change": (
                    (fresh["rmse"] - orig["rmse"]) / orig["rmse"]
                    if np.isfinite(fresh["rmse"]) and np.isfinite(orig["rmse"]) and orig["rmse"] != 0
                    else math.nan
                ),
            }
            orig_row = dict(orig)
            fresh_row = dict(fresh)
            orig_row["near_zero_leak_fraction"] = orig_leak
            fresh_row["near_zero_leak_fraction"] = fresh_leak
            regime_final[space]["original"][regime_name] = orig_row
            regime_final[space]["fresh"][regime_name] = fresh_row
            regime_final[space]["delta"][regime_name] = delta
            regime_rows.append({"space": space, "mode": "original", "regime": regime_name, "relative_rmse_change": math.nan, **orig_row})
            regime_rows.append({"space": space, "mode": "fresh", "regime": regime_name, "relative_rmse_change": math.nan, **fresh_row})
            regime_rows.append({"space": space, "mode": "delta", "regime": regime_name, **delta})

    metrics_df = pd.DataFrame(metrics_rows)
    regime_df = pd.DataFrame(regime_rows)
    sampling_df = pd.DataFrame(sampling_rows)
    metrics_df.to_csv(args.output_dir / "preflight_metrics_summary.csv", index=False)
    regime_df.to_csv(args.output_dir / "preflight_nrtend_regime_summary.csv", index=False)
    sampling_df.to_csv(args.output_dir / "preflight_validation_sampling.csv", index=False)

    plot_metrics_for_space(
        metrics_df,
        args.output_dir / "preflight_physical_metrics.png",
        space="physical",
        space_label="Physical-Space",
    )
    plot_metrics_for_space(
        metrics_df,
        args.output_dir / "preflight_transformed_metrics.png",
        space="transformed",
        space_label="Transformed-Space",
    )
    plot_nrtend_regime_metrics(
        regime_df,
        args.output_dir / "preflight_nrtend_regime_comparison.png",
        space="physical",
        space_label="Physical",
    )
    plot_nrtend_regime_metrics(
        regime_df,
        args.output_dir / "preflight_nrtend_regime_transformed_comparison.png",
        space="transformed",
        space_label="Transformed",
    )

    space_gap_rows: List[Dict[str, object]] = []
    for mode in ("original", "fresh"):
        for variable in MODEL_OUTPUT_COLS:
            transformed_r2 = metrics_final["transformed"][mode][variable]["r2"]
            physical_r2 = metrics_final["physical"][mode][variable]["r2"]
            space_gap_rows.append(
                {
                    "mode": mode,
                    "variable": variable,
                    "transformed_r2": transformed_r2,
                    "physical_r2": physical_r2,
                    "r2_gap_transformed_minus_physical": (
                        transformed_r2 - physical_r2
                        if np.isfinite(transformed_r2) and np.isfinite(physical_r2)
                        else math.nan
                    ),
                }
            )
    space_gap_df = pd.DataFrame(space_gap_rows)
    space_gap_df.to_csv(args.output_dir / "preflight_space_gap_summary.csv", index=False)
    plot_r2_space_gap(space_gap_df, args.output_dir / "preflight_r2_space_gap.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "original_run_id": checkpoint_run_id,
        "original_artifact_dir": str(original_artifact_dir),
        "fresh_artifact_dir": str(fresh_artifacts.artifact_dir),
        "fresh_artifacts_reused": bool(fresh_reused),
        "total_parquet_files": total_files,
        "expected_train_files": expected_train_files,
        "expected_val_files": expected_val_files,
        "validation_files_used": len(val_files),
        "validation_files_with_samples": int(np.sum(sampling_df["sampled_any_rows"].astype(np.int64))) if not sampling_df.empty else 0,
        "all_validation_files_touched": bool(
            len(val_files) > 0 and int(np.sum(sampling_df["sampled_any_rows"].astype(np.int64))) == len(val_files)
        ) if not sampling_df.empty else False,
        "sample_probability_requested": float(args.sample_probability),
        "sampling_seed": int(args.sampling_seed),
        "nrtend_regime_threshold": float(args.nrtend_regime_threshold),
        "total_postfilter_rows": int(total_prefiltered_rows),
        "total_sampled_rows": int(total_sampled_rows),
        "realized_sampling_fraction": (
            total_sampled_rows / total_prefiltered_rows if total_prefiltered_rows > 0 else math.nan
        ),
        "original_output_cols": list(original_output_cols),
        "fresh_output_cols": list(MODEL_OUTPUT_COLS),
        "fit_budgets": {
            "scaler_fit_samples": requested_fit_samples,
            "quantile_subsample": requested_quantile_subsample,
            "quantile_n_quantiles": (
                int(args.quantile_n_quantiles_cap)
                if args.quantile_n_quantiles_cap is not None
                else int(config["data"].get("quantile_n_quantiles", 1000))
            ),
            "matched_original_run_105047_defaults": bool(
                args.fit_sample_cap is None
                and args.quantile_subsample_cap is None
                and args.quantile_n_quantiles_cap is None
            ),
        },
        "metrics": metrics_final,
        "nrtend_regimes": regime_final,
        "space_gap_summary": {
            mode: {
                rec["variable"]: {
                    "transformed_r2": rec["transformed_r2"],
                    "physical_r2": rec["physical_r2"],
                    "r2_gap_transformed_minus_physical": rec["r2_gap_transformed_minus_physical"],
                }
                for rec in space_gap_df[space_gap_df["mode"] == mode].to_dict(orient="records")
            }
            for mode in ("original", "fresh")
        },
    }
    with (args.output_dir / "preflight_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Preflight comparison complete.")
    logger.info("Total sampled rows: %d / %d", total_sampled_rows, total_prefiltered_rows)
    for variable in MODEL_OUTPUT_COLS:
        orig_phys = metrics_final["physical"]["original"][variable]
        fresh_phys = metrics_final["physical"]["fresh"][variable]
        delta_phys = metrics_final["physical"]["delta"][variable]
        orig_trans = metrics_final["transformed"]["original"][variable]
        fresh_trans = metrics_final["transformed"]["fresh"][variable]
        logger.info(
            "%s: transformed original/fresh R2=% .4f/% .4f | physical original/fresh R2=% .4f/% .4f | physical dRMSE=% .4e (% .2f%%)",
            variable.replace("_TAU", ""),
            orig_trans["r2"],
            fresh_trans["r2"],
            orig_phys["r2"],
            fresh_phys["r2"],
            delta_phys["rmse"],
            100.0 * delta_phys["relative_rmse_change"] if np.isfinite(delta_phys["relative_rmse_change"]) else float("nan"),
        )


if __name__ == "__main__":
    main()
