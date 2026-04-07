#!/usr/bin/env python3
"""
Deep-dive analysis of problematic nrtend samples on the validation split.

Purpose
-------
This script complements the existing MoE evaluation scripts by answering:
1. How many validation samples are truly problematic?
2. Which regime / |true nrtend| bins dominate the physical-space failure?
3. Are the failures sparse outliers or broad systematic issues?
4. Do problematic samples cluster by chosen expert / router confidence?
5. Do problematic samples show a consistent pattern in input feature space?

Definitions
-----------
For non-near-zero target samples, define:
    abs_true_ref = max(|true nrtend|, nrtend_regime_threshold)
    magnitude_ratio = (|pred| + eps) / (abs_true_ref + eps)
    error_ratio     = |pred - true| / (abs_true_ref + eps)

For near-zero target samples (true regime == near_zero), define:
    near_zero_leak_multiplier = |pred| / nrtend_regime_threshold

Composite problematic flags:
    mild    : near_zero leak > 10x threshold OR non-near-zero (magnitude_ratio > 10 OR error_ratio > 10)
    severe  : near_zero leak > 100x threshold OR non-near-zero (magnitude_ratio > 100 OR error_ratio > 100)
    extreme : near_zero leak > 1000x threshold OR non-near-zero (magnitude_ratio > 1000 OR error_ratio > 1000)

Outputs
-------
- problematic_bin_summary.csv
- problematic_focus_groups.csv
- problematic_router_summary.csv
- problematic_feature_shifts.csv
- focus_severe_samples.csv
- focus_severe_file_summary.csv
- focus_severe_metadata_summary.csv
- top_abs_error_samples.csv
- top_error_ratio_samples.csv
- top_magnitude_ratio_samples.csv
- top_near_zero_leak_samples.csv
- problematic_fraction_by_bin.png
- error_contribution_by_bin.png
- soft_vs_hard_gap_by_bin.png
- feature_shift_heatmap.png
- router_problematic_heatmaps.png
- problematic_summary.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
INPUT_LOG_COLS = {"QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", "LAMC", "LAMR", "N0R"}
METADATA_CANDIDATE_COLS: Tuple[str, ...] = ("time", "ncol", "lev", "ilev", "T")

LOG_EPSILON = 1.0e-10
REGIME_LABELS = ("near_zero", "negative", "positive")
EXPERT_LABELS = ("expert_near_zero", "expert_negative", "expert_positive")

PHYSICAL_SIGN_BY_COL = {
    "qrtend_TAU": +1,
    "nctend_TAU": -1,
    "nrtend_TAU": 0,
    "qctend_TAU": -1,
}

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
            f"analyze_nrtend_problematic_samples.py expects model.architecture='moe', "
            f"found '{architecture}'"
        )

    model = instantiate_moe_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(resolve_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


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


def compute_moe_eval_outputs(
    model: torch.nn.Module,
    inputs_tensor: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """Compute hard-route, soft-route, and per-expert nrtend outputs in eval mode."""
    shared_features = model.shared_backbone(inputs_tensor)

    qrtend = model.qrtend_head(shared_features)
    nctend = model.nctend_head(shared_features)

    router_logits_t = model.nrtend_moe.router(shared_features)
    expert_outputs_t = torch.cat(
        [expert(shared_features) for expert in model.nrtend_moe.experts],
        dim=-1,
    )  # [B, 3]
    gate_probs_t = torch.softmax(router_logits_t, dim=-1)
    soft_nrtend_t = (gate_probs_t * expert_outputs_t).sum(dim=-1, keepdim=True)
    selected_expert_t = torch.argmax(router_logits_t, dim=-1)
    hard_nrtend_t = expert_outputs_t.gather(1, selected_expert_t.unsqueeze(-1))
    qctend = -qrtend

    pred_norm_model = np.concatenate(
        [
            qrtend.detach().cpu().numpy(),
            nctend.detach().cpu().numpy(),
            hard_nrtend_t.detach().cpu().numpy(),
            qctend.detach().cpu().numpy(),
        ],
        axis=1,
    ).astype(np.float64, copy=False)

    return {
        "pred_norm_model": pred_norm_model,
        "router_logits": router_logits_t.detach().cpu().numpy().astype(np.float64, copy=False),
        "router_probs": gate_probs_t.detach().cpu().numpy().astype(np.float64, copy=False),
        "selected_expert": selected_expert_t.detach().cpu().numpy().astype(np.int64, copy=False),
        "expert_outputs_scaled": expert_outputs_t.detach().cpu().numpy().astype(np.float64, copy=False),
        "soft_nrtend_scaled": soft_nrtend_t.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False),
        "hard_nrtend_scaled": hard_nrtend_t.detach().cpu().numpy().reshape(-1).astype(np.float64, copy=False),
    }


def inverse_output_scaler_only(
    normalized_dataset_matrix: np.ndarray,
    dataset,
) -> np.ndarray:
    restored = np.asarray(normalized_dataset_matrix, dtype=np.float64).copy()
    scaler = getattr(dataset, "output_scaler", None)
    if scaler is not None:
        restored = scaler.inverse_transform(restored)
    return restored


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
                restored[:, idx] = np.power(10.0, restored[:, idx])
    return restored


def compute_nrtend_variant_details_for_subset(
    base_pred_norm_dataset_subset: np.ndarray,
    soft_scaled_subset: np.ndarray,
    expert_scaled_subset: np.ndarray,
    dataset_cols: Sequence[str],
    dataset,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Reconstruct nrtend variants for a small subset in model/scaler/physical space."""
    nrt_idx = dataset_cols.index("nrtend_TAU")
    variants = {
        "soft": np.asarray(soft_scaled_subset, dtype=np.float64).reshape(-1),
        "expert_near_zero": np.asarray(expert_scaled_subset[:, 0], dtype=np.float64).reshape(-1),
        "expert_negative": np.asarray(expert_scaled_subset[:, 1], dtype=np.float64).reshape(-1),
        "expert_positive": np.asarray(expert_scaled_subset[:, 2], dtype=np.float64).reshape(-1),
    }

    details: Dict[str, Dict[str, np.ndarray]] = {}
    for name, scaled_values in variants.items():
        variant_norm_dataset = np.asarray(base_pred_norm_dataset_subset, dtype=np.float64).copy()
        variant_norm_dataset[:, nrt_idx] = scaled_values
        variant_postscaler = inverse_output_scaler_only(variant_norm_dataset, dataset)[:, nrt_idx]
        variant_phys = inverse_output_pipeline(variant_norm_dataset, dataset_cols, dataset)[:, nrt_idx]
        details[name] = {
            "model_space_scaled": scaled_values,
            "postscaler_space": variant_postscaler,
            "physical": variant_phys,
        }
    return details


def to_python_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def analyze_nrtend_batch(
    model: torch.nn.Module,
    inputs_scaled: np.ndarray,
    targets_scaled_dataset: np.ndarray,
    dataset_cols: Sequence[str],
    dataset,
    nrtend_threshold: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32, device=device)
    moe_outputs = compute_moe_eval_outputs(model, inputs_tensor)

    pred_norm_model = moe_outputs["pred_norm_model"]
    pred_norm_dataset = model_to_dataset_matrix(pred_norm_model, dataset_cols)

    target_postscaler_dataset = inverse_output_scaler_only(targets_scaled_dataset, dataset)
    pred_postscaler_dataset = inverse_output_scaler_only(pred_norm_dataset, dataset)

    target_phys_dataset = inverse_output_pipeline(targets_scaled_dataset, dataset_cols, dataset)
    target_phys_dataset = zero_small_nrtend_targets(
        target_phys_dataset,
        dataset_cols,
        nrtend_threshold,
    )
    pred_phys_dataset = inverse_output_pipeline(pred_norm_dataset, dataset_cols, dataset)

    soft_norm_dataset = np.asarray(pred_norm_dataset, dtype=np.float64).copy()
    nrt_idx_dataset = dataset_cols.index("nrtend_TAU")
    soft_norm_dataset[:, nrt_idx_dataset] = moe_outputs["soft_nrtend_scaled"]
    soft_postscaler_dataset = inverse_output_scaler_only(soft_norm_dataset, dataset)
    soft_phys_dataset = inverse_output_pipeline(soft_norm_dataset, dataset_cols, dataset)

    target_phys_model = dataset_to_model_matrix(target_phys_dataset, dataset_cols)
    pred_phys_model = dataset_to_model_matrix(pred_phys_dataset, dataset_cols)
    soft_phys_model = dataset_to_model_matrix(soft_phys_dataset, dataset_cols)

    target_postscaler_model = dataset_to_model_matrix(target_postscaler_dataset, dataset_cols)
    pred_postscaler_model = dataset_to_model_matrix(pred_postscaler_dataset, dataset_cols)
    soft_postscaler_model = dataset_to_model_matrix(soft_postscaler_dataset, dataset_cols)

    y_true_phys = target_phys_model[:, 2]
    y_pred_phys = pred_phys_model[:, 2]
    y_soft_phys = soft_phys_model[:, 2]

    y_true_scaled = dataset_to_model_matrix(targets_scaled_dataset, dataset_cols)[:, 2]
    y_pred_scaled = moe_outputs["hard_nrtend_scaled"]
    y_soft_scaled = moe_outputs["soft_nrtend_scaled"]

    y_true_postscaler = target_postscaler_model[:, 2]
    y_pred_postscaler = pred_postscaler_model[:, 2]
    y_soft_postscaler = soft_postscaler_model[:, 2]

    abs_true = np.abs(y_true_phys)
    abs_pred = np.abs(y_pred_phys)
    soft_abs_pred = np.abs(y_soft_phys)
    abs_err = np.abs(y_pred_phys - y_true_phys)
    soft_abs_err = np.abs(y_soft_phys - y_true_phys)

    true_regime = regimes_from_values(y_true_phys, nrtend_threshold)
    bin_idx = magnitude_bin_index(abs_true)
    group_idx = true_regime * (len(MAG_BIN_EDGES) - 1) + bin_idx

    abs_true_ref = np.maximum(abs_true, nrtend_threshold)
    magnitude_ratio = (abs_pred + LOG_EPSILON) / (abs_true_ref + LOG_EPSILON)
    error_ratio = abs_err / (abs_true_ref + LOG_EPSILON)
    soft_magnitude_ratio = (soft_abs_pred + LOG_EPSILON) / (abs_true_ref + LOG_EPSILON)
    soft_error_ratio = soft_abs_err / (abs_true_ref + LOG_EPSILON)

    near_zero_mask = true_regime == 0
    sign_mismatch = (true_regime != 0) & (np.sign(y_true_phys) != np.sign(y_pred_phys))
    soft_sign_mismatch = (true_regime != 0) & (np.sign(y_true_phys) != np.sign(y_soft_phys))
    router_mismatch = moe_outputs["selected_expert"] != true_regime

    near_zero_leak_multiplier = np.zeros_like(abs_pred)
    near_zero_leak_multiplier[near_zero_mask] = abs_pred[near_zero_mask] / nrtend_threshold
    soft_near_zero_leak_multiplier = np.zeros_like(soft_abs_pred)
    soft_near_zero_leak_multiplier[near_zero_mask] = soft_abs_pred[near_zero_mask] / nrtend_threshold

    mild_flag = np.where(
        near_zero_mask,
        near_zero_leak_multiplier > 10.0,
        (magnitude_ratio > 10.0) | (error_ratio > 10.0),
    )
    severe_flag = np.where(
        near_zero_mask,
        near_zero_leak_multiplier > 100.0,
        (magnitude_ratio > 100.0) | (error_ratio > 100.0),
    )
    extreme_flag = np.where(
        near_zero_mask,
        near_zero_leak_multiplier > 1000.0,
        (magnitude_ratio > 1000.0) | (error_ratio > 1000.0),
    )

    soft_mild_flag = np.where(
        near_zero_mask,
        soft_near_zero_leak_multiplier > 10.0,
        (soft_magnitude_ratio > 10.0) | (soft_error_ratio > 10.0),
    )
    soft_severe_flag = np.where(
        near_zero_mask,
        soft_near_zero_leak_multiplier > 100.0,
        (soft_magnitude_ratio > 100.0) | (soft_error_ratio > 100.0),
    )
    soft_extreme_flag = np.where(
        near_zero_mask,
        soft_near_zero_leak_multiplier > 1000.0,
        (soft_magnitude_ratio > 1000.0) | (soft_error_ratio > 1000.0),
    )

    return {
        "pred_norm_dataset": pred_norm_dataset,
        "router_probs": moe_outputs["router_probs"],
        "selected_expert": moe_outputs["selected_expert"],
        "router_conf": np.max(moe_outputs["router_probs"], axis=1),
        "expert_scaled": moe_outputs["expert_outputs_scaled"],
        "soft_scaled": y_soft_scaled,
        "y_true_phys": y_true_phys,
        "y_pred_phys": y_pred_phys,
        "y_soft_phys": y_soft_phys,
        "y_true_scaled": y_true_scaled,
        "y_pred_scaled": y_pred_scaled,
        "y_soft_scaled": y_soft_scaled,
        "y_true_postscaler": y_true_postscaler,
        "y_pred_postscaler": y_pred_postscaler,
        "y_soft_postscaler": y_soft_postscaler,
        "abs_true": abs_true,
        "abs_pred": abs_pred,
        "soft_abs_pred": soft_abs_pred,
        "abs_err": abs_err,
        "soft_abs_err": soft_abs_err,
        "true_regime": true_regime,
        "bin_idx": bin_idx,
        "group_idx": group_idx,
        "magnitude_ratio": magnitude_ratio,
        "error_ratio": error_ratio,
        "soft_magnitude_ratio": soft_magnitude_ratio,
        "soft_error_ratio": soft_error_ratio,
        "near_zero_mask": near_zero_mask,
        "sign_mismatch": sign_mismatch,
        "soft_sign_mismatch": soft_sign_mismatch,
        "router_mismatch": router_mismatch,
        "near_zero_leak_multiplier": near_zero_leak_multiplier,
        "soft_near_zero_leak_multiplier": soft_near_zero_leak_multiplier,
        "mild_flag": mild_flag,
        "severe_flag": severe_flag,
        "extreme_flag": extreme_flag,
        "soft_mild_flag": soft_mild_flag,
        "soft_severe_flag": soft_severe_flag,
        "soft_extreme_flag": soft_extreme_flag,
    }


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


def top_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or values.size == 0:
        return np.empty(0, dtype=np.int64)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.empty(0, dtype=np.int64)

    safe = np.where(finite_mask, values, -np.inf)
    k = min(k, int(np.sum(finite_mask)))
    idx = np.argpartition(safe, -k)[-k:]
    idx = idx[np.argsort(safe[idx])[::-1]]
    return idx.astype(np.int64, copy=False)


@dataclass
class RunningFeatureStats:
    n_features: int
    count: int = 0
    sum_x: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    sum_x2: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))

    def __post_init__(self) -> None:
        if self.sum_x.size == 0:
            self.sum_x = np.zeros(self.n_features, dtype=np.float64)
        if self.sum_x2.size == 0:
            self.sum_x2 = np.zeros(self.n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        x = np.asarray(x, dtype=np.float64)
        self.count += x.shape[0]
        self.sum_x += np.sum(x, axis=0)
        self.sum_x2 += np.sum(x * x, axis=0)

    def mean(self) -> np.ndarray:
        if self.count == 0:
            return np.full(self.n_features, np.nan)
        return self.sum_x / self.count

    def std(self) -> np.ndarray:
        if self.count == 0:
            return np.full(self.n_features, np.nan)
        mean = self.mean()
        var = np.maximum(self.sum_x2 / self.count - mean * mean, 0.0)
        return np.sqrt(var)


@dataclass
class ProblemBinAccumulator:
    count: int = 0
    sum_abs_true: float = 0.0
    sum_abs_pred: float = 0.0
    sum_abs_err: float = 0.0
    sum_sq_err: float = 0.0
    sum_soft_abs_pred: float = 0.0
    sum_soft_abs_err: float = 0.0
    sum_soft_sq_err: float = 0.0

    sign_mismatch_count: int = 0
    soft_sign_mismatch_count: int = 0
    router_mismatch_count: int = 0

    mag_ratio_gt10_count: int = 0
    mag_ratio_gt100_count: int = 0
    mag_ratio_gt1000_count: int = 0
    soft_mag_ratio_gt10_count: int = 0
    soft_mag_ratio_gt100_count: int = 0
    soft_mag_ratio_gt1000_count: int = 0

    err_ratio_gt10_count: int = 0
    err_ratio_gt100_count: int = 0
    err_ratio_gt1000_count: int = 0
    soft_err_ratio_gt10_count: int = 0
    soft_err_ratio_gt100_count: int = 0
    soft_err_ratio_gt1000_count: int = 0

    near_zero_leak_gt1x_count: int = 0
    near_zero_leak_gt10x_count: int = 0
    near_zero_leak_gt100x_count: int = 0
    near_zero_leak_gt1000x_count: int = 0

    mild_count: int = 0
    severe_count: int = 0
    extreme_count: int = 0
    soft_mild_count: int = 0
    soft_severe_count: int = 0
    soft_extreme_count: int = 0
    soft_better_count: int = 0
    hard_better_count: int = 0
    severe_soft_better_count: int = 0

    expert_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))
    severe_expert_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.int64))

    router_conf_sum: float = 0.0
    severe_router_conf_sum: float = 0.0

    def update(
        self,
        abs_true: np.ndarray,
        abs_pred: np.ndarray,
        soft_abs_pred: np.ndarray,
        abs_err: np.ndarray,
        soft_abs_err: np.ndarray,
        sign_mismatch: np.ndarray,
        soft_sign_mismatch: np.ndarray,
        router_mismatch: np.ndarray,
        mag_ratio: np.ndarray,
        soft_mag_ratio: np.ndarray,
        err_ratio: np.ndarray,
        soft_err_ratio: np.ndarray,
        near_zero_leak_multiplier: np.ndarray,
        mild_flag: np.ndarray,
        severe_flag: np.ndarray,
        extreme_flag: np.ndarray,
        soft_mild_flag: np.ndarray,
        soft_severe_flag: np.ndarray,
        soft_extreme_flag: np.ndarray,
        selected_expert: np.ndarray,
        router_conf: np.ndarray,
        is_near_zero: np.ndarray,
    ) -> None:
        n = abs_true.size
        if n == 0:
            return

        self.count += int(n)
        self.sum_abs_true += float(np.sum(abs_true))
        self.sum_abs_pred += float(np.sum(abs_pred))
        self.sum_abs_err += float(np.sum(abs_err))
        self.sum_sq_err += float(np.dot(abs_err, abs_err))
        self.sum_soft_abs_pred += float(np.sum(soft_abs_pred))
        self.sum_soft_abs_err += float(np.sum(soft_abs_err))
        self.sum_soft_sq_err += float(np.dot(soft_abs_err, soft_abs_err))

        self.sign_mismatch_count += int(np.sum(sign_mismatch))
        self.soft_sign_mismatch_count += int(np.sum(soft_sign_mismatch))
        self.router_mismatch_count += int(np.sum(router_mismatch))

        self.mag_ratio_gt10_count += int(np.sum(mag_ratio > 10.0))
        self.mag_ratio_gt100_count += int(np.sum(mag_ratio > 100.0))
        self.mag_ratio_gt1000_count += int(np.sum(mag_ratio > 1000.0))
        self.soft_mag_ratio_gt10_count += int(np.sum(soft_mag_ratio > 10.0))
        self.soft_mag_ratio_gt100_count += int(np.sum(soft_mag_ratio > 100.0))
        self.soft_mag_ratio_gt1000_count += int(np.sum(soft_mag_ratio > 1000.0))

        self.err_ratio_gt10_count += int(np.sum(err_ratio > 10.0))
        self.err_ratio_gt100_count += int(np.sum(err_ratio > 100.0))
        self.err_ratio_gt1000_count += int(np.sum(err_ratio > 1000.0))
        self.soft_err_ratio_gt10_count += int(np.sum(soft_err_ratio > 10.0))
        self.soft_err_ratio_gt100_count += int(np.sum(soft_err_ratio > 100.0))
        self.soft_err_ratio_gt1000_count += int(np.sum(soft_err_ratio > 1000.0))

        if np.any(is_near_zero):
            nz_leak = near_zero_leak_multiplier[is_near_zero]
            self.near_zero_leak_gt1x_count += int(np.sum(nz_leak > 1.0))
            self.near_zero_leak_gt10x_count += int(np.sum(nz_leak > 10.0))
            self.near_zero_leak_gt100x_count += int(np.sum(nz_leak > 100.0))
            self.near_zero_leak_gt1000x_count += int(np.sum(nz_leak > 1000.0))

        self.mild_count += int(np.sum(mild_flag))
        self.severe_count += int(np.sum(severe_flag))
        self.extreme_count += int(np.sum(extreme_flag))
        self.soft_mild_count += int(np.sum(soft_mild_flag))
        self.soft_severe_count += int(np.sum(soft_severe_flag))
        self.soft_extreme_count += int(np.sum(soft_extreme_flag))
        self.soft_better_count += int(np.sum(soft_abs_err < abs_err))
        self.hard_better_count += int(np.sum(abs_err < soft_abs_err))
        self.severe_soft_better_count += int(np.sum(severe_flag & (soft_abs_err < abs_err)))

        self.router_conf_sum += float(np.sum(router_conf))
        if np.any(severe_flag):
            self.severe_router_conf_sum += float(np.sum(router_conf[severe_flag]))

        for expert_idx in range(3):
            mask = selected_expert == expert_idx
            self.expert_counts[expert_idx] += int(np.sum(mask))
            if np.any(severe_flag):
                self.severe_expert_counts[expert_idx] += int(np.sum(mask & severe_flag))

    def finalize(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_abs_true": math.nan,
                "mean_abs_pred": math.nan,
                "soft_mean_abs_pred": math.nan,
                "mae": math.nan,
                "rmse": math.nan,
                "soft_mae": math.nan,
                "soft_rmse": math.nan,
                "sign_mismatch_fraction": math.nan,
                "soft_sign_mismatch_fraction": math.nan,
                "router_mismatch_fraction": math.nan,
                "mag_ratio_gt10_fraction": math.nan,
                "mag_ratio_gt100_fraction": math.nan,
                "mag_ratio_gt1000_fraction": math.nan,
                "soft_mag_ratio_gt10_fraction": math.nan,
                "soft_mag_ratio_gt100_fraction": math.nan,
                "soft_mag_ratio_gt1000_fraction": math.nan,
                "err_ratio_gt10_fraction": math.nan,
                "err_ratio_gt100_fraction": math.nan,
                "err_ratio_gt1000_fraction": math.nan,
                "soft_err_ratio_gt10_fraction": math.nan,
                "soft_err_ratio_gt100_fraction": math.nan,
                "soft_err_ratio_gt1000_fraction": math.nan,
                "near_zero_leak_gt1x_fraction": math.nan,
                "near_zero_leak_gt10x_fraction": math.nan,
                "near_zero_leak_gt100x_fraction": math.nan,
                "near_zero_leak_gt1000x_fraction": math.nan,
                "mild_fraction": math.nan,
                "severe_fraction": math.nan,
                "extreme_fraction": math.nan,
                "soft_mild_fraction": math.nan,
                "soft_severe_fraction": math.nan,
                "soft_extreme_fraction": math.nan,
                "fraction_soft_better_abs_error": math.nan,
                "fraction_hard_better_abs_error": math.nan,
                "fraction_soft_better_within_hard_severe": math.nan,
                "mean_abs_error_delta_hard_minus_soft": math.nan,
                "expert_near_zero_fraction": math.nan,
                "expert_negative_fraction": math.nan,
                "expert_positive_fraction": math.nan,
                "severe_expert_near_zero_fraction": math.nan,
                "severe_expert_negative_fraction": math.nan,
                "severe_expert_positive_fraction": math.nan,
                "mean_router_confidence": math.nan,
                "mean_router_confidence_severe": math.nan,
                "sum_abs_error": 0.0,
                "sum_sq_error": 0.0,
                "sum_soft_abs_error": 0.0,
                "sum_soft_sq_error": 0.0,
            }

        severe_denom = self.severe_count if self.severe_count > 0 else np.nan

        return {
            "count": int(self.count),
            "mean_abs_true": self.sum_abs_true / self.count,
            "mean_abs_pred": self.sum_abs_pred / self.count,
            "soft_mean_abs_pred": self.sum_soft_abs_pred / self.count,
            "mae": self.sum_abs_err / self.count,
            "rmse": math.sqrt(self.sum_sq_err / self.count),
            "soft_mae": self.sum_soft_abs_err / self.count,
            "soft_rmse": math.sqrt(self.sum_soft_sq_err / self.count),
            "sign_mismatch_fraction": self.sign_mismatch_count / self.count,
            "soft_sign_mismatch_fraction": self.soft_sign_mismatch_count / self.count,
            "router_mismatch_fraction": self.router_mismatch_count / self.count,
            "mag_ratio_gt10_fraction": self.mag_ratio_gt10_count / self.count,
            "mag_ratio_gt100_fraction": self.mag_ratio_gt100_count / self.count,
            "mag_ratio_gt1000_fraction": self.mag_ratio_gt1000_count / self.count,
            "soft_mag_ratio_gt10_fraction": self.soft_mag_ratio_gt10_count / self.count,
            "soft_mag_ratio_gt100_fraction": self.soft_mag_ratio_gt100_count / self.count,
            "soft_mag_ratio_gt1000_fraction": self.soft_mag_ratio_gt1000_count / self.count,
            "err_ratio_gt10_fraction": self.err_ratio_gt10_count / self.count,
            "err_ratio_gt100_fraction": self.err_ratio_gt100_count / self.count,
            "err_ratio_gt1000_fraction": self.err_ratio_gt1000_count / self.count,
            "soft_err_ratio_gt10_fraction": self.soft_err_ratio_gt10_count / self.count,
            "soft_err_ratio_gt100_fraction": self.soft_err_ratio_gt100_count / self.count,
            "soft_err_ratio_gt1000_fraction": self.soft_err_ratio_gt1000_count / self.count,
            "near_zero_leak_gt1x_fraction": self.near_zero_leak_gt1x_count / self.count,
            "near_zero_leak_gt10x_fraction": self.near_zero_leak_gt10x_count / self.count,
            "near_zero_leak_gt100x_fraction": self.near_zero_leak_gt100x_count / self.count,
            "near_zero_leak_gt1000x_fraction": self.near_zero_leak_gt1000x_count / self.count,
            "mild_fraction": self.mild_count / self.count,
            "severe_fraction": self.severe_count / self.count,
            "extreme_fraction": self.extreme_count / self.count,
            "soft_mild_fraction": self.soft_mild_count / self.count,
            "soft_severe_fraction": self.soft_severe_count / self.count,
            "soft_extreme_fraction": self.soft_extreme_count / self.count,
            "fraction_soft_better_abs_error": self.soft_better_count / self.count,
            "fraction_hard_better_abs_error": self.hard_better_count / self.count,
            "fraction_soft_better_within_hard_severe": (
                self.severe_soft_better_count / severe_denom if not np.isnan(severe_denom) else math.nan
            ),
            "mean_abs_error_delta_hard_minus_soft": (self.sum_abs_err - self.sum_soft_abs_err) / self.count,
            "expert_near_zero_fraction": self.expert_counts[0] / self.count,
            "expert_negative_fraction": self.expert_counts[1] / self.count,
            "expert_positive_fraction": self.expert_counts[2] / self.count,
            "severe_expert_near_zero_fraction": (
                self.severe_expert_counts[0] / severe_denom if not np.isnan(severe_denom) else math.nan
            ),
            "severe_expert_negative_fraction": (
                self.severe_expert_counts[1] / severe_denom if not np.isnan(severe_denom) else math.nan
            ),
            "severe_expert_positive_fraction": (
                self.severe_expert_counts[2] / severe_denom if not np.isnan(severe_denom) else math.nan
            ),
            "mean_router_confidence": self.router_conf_sum / self.count,
            "mean_router_confidence_severe": (
                self.severe_router_conf_sum / severe_denom if not np.isnan(severe_denom) else math.nan
            ),
            "sum_abs_error": self.sum_abs_err,
            "sum_sq_error": self.sum_sq_err,
            "sum_soft_abs_error": self.sum_soft_abs_err,
            "sum_soft_sq_error": self.sum_soft_sq_err,
        }


class TopKTracker:
    def __init__(self, k: int):
        self.k = int(k)
        self.heap: List[Tuple[float, int, Dict[str, object]]] = []
        self.counter = 0

    def add(self, score: float, row: Dict[str, object]) -> None:
        if not np.isfinite(score):
            return
        item = (float(score), self.counter, row)
        self.counter += 1
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
        elif score > self.heap[0][0]:
            heapq.heapreplace(self.heap, item)

    def rows(self) -> List[Dict[str, object]]:
        items = sorted(self.heap, key=lambda x: x[0], reverse=True)
        return [row for _, _, row in items]


def iter_validation_batches_with_metadata(dataset) -> Iterable[Dict[str, object]]:
    """
    Iterate over the validation split while preserving file path and row index.

    This mirrors the data loader pipeline closely but exposes metadata needed
    for exact outlier sample dumps.
    """
    active_files = list(getattr(dataset, "active_files", []))
    for file_path in active_files:
        try:
            try:
                chunk_iter = pd.read_parquet(file_path, chunksize=dataset.chunk_size)
            except TypeError:
                full_data = pd.read_parquet(file_path)
                chunk_iter = [
                    full_data[i:i + dataset.chunk_size]
                    for i in range(0, len(full_data), dataset.chunk_size)
                ]

            for chunk in chunk_iter:
                processed_chunk = dataset._preprocess_chunk_vectorized(chunk)
                if processed_chunk is None or len(processed_chunk) == 0:
                    continue

                metadata_cols = [col for col in METADATA_CANDIDATE_COLS if col in chunk.columns]
                metadata_frame = None
                if metadata_cols:
                    metadata_frame = chunk.loc[processed_chunk.index, metadata_cols].reset_index(drop=True)

                input_matrix = processed_chunk[dataset.input_cols].values
                if dataset.input_transform == "quantile" and dataset.input_transformer is not None:
                    input_matrix = dataset.input_transformer.transform(input_matrix)
                input_scaled = dataset.input_scaler.transform(input_matrix)

                outputs_matrix = processed_chunk[dataset.output_cols].values
                if dataset.output_transform == "quantile" and dataset.output_transformer is not None:
                    outputs_matrix = dataset.output_transformer.transform(outputs_matrix)
                outputs_scaled = dataset.output_scaler.transform(outputs_matrix)

                row_in_file = processed_chunk.index.to_numpy()
                n_samples = len(processed_chunk)

                for start_idx in range(0, n_samples, dataset.batch_size):
                    end_idx = min(start_idx + dataset.batch_size, n_samples)
                    yield {
                        "file_path": str(file_path),
                        "row_in_file": row_in_file[start_idx:end_idx].astype(np.int64, copy=False),
                        "inputs_scaled": input_scaled[start_idx:end_idx].astype(np.float32, copy=False),
                        "targets_scaled_dataset": outputs_scaled[start_idx:end_idx].astype(np.float64, copy=False),
                        "metadata_frame": (
                            metadata_frame.iloc[start_idx:end_idx].reset_index(drop=True)
                            if metadata_frame is not None else None
                        ),
                    }

        except Exception as exc:
            logger.warning("Error processing file %s: %s", file_path, exc)
            continue


def build_sample_row(
    idx: int,
    file_path: str,
    row_in_file: np.ndarray,
    input_cols: Sequence[str],
    input_scaled_row: np.ndarray,
    input_raw_row: np.ndarray,
    true_regime: np.ndarray,
    bin_idx: np.ndarray,
    y_true_phys: np.ndarray,
    y_pred_phys: np.ndarray,
    y_soft_phys: np.ndarray,
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    y_soft_scaled: np.ndarray,
    y_true_postscaler: np.ndarray,
    y_pred_postscaler: np.ndarray,
    y_soft_postscaler: np.ndarray,
    abs_err: np.ndarray,
    soft_abs_err: np.ndarray,
    err_ratio: np.ndarray,
    soft_err_ratio: np.ndarray,
    mag_ratio: np.ndarray,
    soft_mag_ratio: np.ndarray,
    selected_expert: np.ndarray,
    router_probs: np.ndarray,
    candidate_local_idx: Optional[int] = None,
    variant_details: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    metadata_row: Optional[Dict[str, object]] = None,
    group_label: Optional[str] = None,
) -> Dict[str, object]:
    row = {
        "file_path": file_path,
        "row_in_file": int(row_in_file[idx]),
        "true_regime": REGIME_LABELS[int(true_regime[idx])],
        "bin_label": make_bin_label(
            float(MAG_BIN_EDGES[int(bin_idx[idx])]),
            float(MAG_BIN_EDGES[int(bin_idx[idx]) + 1]),
        ),
        "nrtend_true_physical": float(y_true_phys[idx]),
        "nrtend_pred_physical": float(y_pred_phys[idx]),
        "nrtend_soft_pred_physical": float(y_soft_phys[idx]),
        "nrtend_abs_error": float(abs_err[idx]),
        "nrtend_soft_abs_error": float(soft_abs_err[idx]),
        "nrtend_error_ratio": float(err_ratio[idx]),
        "nrtend_soft_error_ratio": float(soft_err_ratio[idx]),
        "nrtend_magnitude_ratio": float(mag_ratio[idx]),
        "nrtend_soft_magnitude_ratio": float(soft_mag_ratio[idx]),
        "nrtend_true_model_space_scaled": float(y_true_scaled[idx]),
        "nrtend_pred_model_space_scaled": float(y_pred_scaled[idx]),
        "nrtend_soft_model_space_scaled": float(y_soft_scaled[idx]),
        "nrtend_true_postscaler_space": float(y_true_postscaler[idx]),
        "nrtend_pred_postscaler_space": float(y_pred_postscaler[idx]),
        "nrtend_soft_postscaler_space": float(y_soft_postscaler[idx]),
        "hard_minus_soft_abs_error": float(abs_err[idx] - soft_abs_err[idx]),
        "soft_better_than_hard": bool(soft_abs_err[idx] < abs_err[idx]),
        "selected_expert": EXPERT_LABELS[int(selected_expert[idx])],
        "router_prob_near_zero": float(router_probs[idx, 0]),
        "router_prob_negative": float(router_probs[idx, 1]),
        "router_prob_positive": float(router_probs[idx, 2]),
        "router_confidence": float(np.max(router_probs[idx])),
    }
    if group_label is not None:
        row["group_label"] = group_label

    for j, col in enumerate(input_cols):
        row[f"input_scaled__{col}"] = float(input_scaled_row[j])
        row[f"input_raw__{col}"] = float(input_raw_row[j])

    if variant_details is not None and candidate_local_idx is not None:
        for name, details in variant_details.items():
            row[f"nrtend_{name}_model_space_scaled"] = float(details["model_space_scaled"][candidate_local_idx])
            row[f"nrtend_{name}_postscaler_space"] = float(details["postscaler_space"][candidate_local_idx])
            row[f"nrtend_{name}_physical"] = float(details["physical"][candidate_local_idx])

    if metadata_row is not None:
        for col, value in metadata_row.items():
            row[f"metadata__{col}"] = to_python_scalar(value)

    return row


def plot_problematic_fraction_by_bin(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    metrics = [
        ("mild_fraction", "Mild Problem Fraction"),
        ("severe_fraction", "Severe Problem Fraction"),
        ("extreme_fraction", "Extreme Problem Fraction"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for regime in REGIME_LABELS:
            sub = df[(df["regime"] == regime) & (df["count"] > 0)]
            if len(sub) == 0:
                continue
            y = np.maximum(sub[metric].values, 1e-12)
            ax.plot(sub["bin_center_plot"], y, marker="o", label=regime)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|true nrtend| bin center")
        ax.set_ylabel("Fraction")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_contribution_by_bin(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
    metrics = [
        ("share_total_abs_error", "Share of Total |Error|"),
        ("share_total_sq_error", "Share of Total Squared Error"),
    ]

    for ax, (metric, title) in zip(axes, metrics):
        for regime in REGIME_LABELS:
            sub = df[(df["regime"] == regime) & (df["count"] > 0)]
            if len(sub) == 0:
                continue
            y = np.maximum(sub[metric].values, 1e-15)
            ax.plot(sub["bin_center_plot"], y, marker="o", label=regime)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|true nrtend| bin center")
        ax.set_ylabel("Fraction")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_soft_vs_hard_gap_by_bin(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    panels = [
        ("rmse_ratio_hard_over_soft", "Hard/Soft RMSE Ratio"),
        ("fraction_soft_better_abs_error", "Fraction Soft Better"),
    ]

    for ax, (metric, title) in zip(axes, panels):
        for regime in REGIME_LABELS:
            sub = df[(df["regime"] == regime) & (df["count"] > 0)].copy()
            if len(sub) == 0:
                continue
            y = sub[metric].values
            finite = np.isfinite(y) & (y > 0)
            if not np.any(finite):
                continue
            ax.plot(sub.loc[finite, "bin_center_plot"], y[finite], marker="o", label=regime)
        ax.set_xscale("log")
        ax.set_xlabel("|true nrtend| bin center")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Ratio")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_shift_heatmap(feature_df: pd.DataFrame, output_path: Path) -> None:
    if feature_df.empty:
        logger.warning("No feature-shift rows available for heatmap.")
        return

    pivot = feature_df.pivot(index="group_label", columns="feature", values="z_shift")
    row_labels = list(pivot.index)
    col_labels = list(pivot.columns)
    matrix = pivot.values.astype(np.float64)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(col_labels)), max(4, 0.55 * len(row_labels))))
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    vmax = max(vmax, 1.0)
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Severe-Sample Feature Shift in Scaled Input Space")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("z-shift = (mean_severe - mean_all) / std_all")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_router_heatmaps(
    router_all_counts: np.ndarray,
    router_severe_counts: np.ndarray,
    output_path: Path,
) -> None:
    def row_normalize(matrix: np.ndarray) -> np.ndarray:
        result = matrix.astype(np.float64).copy()
        row_sums = result.sum(axis=1, keepdims=True)
        nonzero = row_sums[:, 0] > 0
        result[nonzero] = result[nonzero] / row_sums[nonzero]
        result[~nonzero] = np.nan
        return result

    matrices = [
        (row_normalize(router_all_counts), "All Samples"),
        (row_normalize(router_severe_counts), "Severe Samples"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (matrix, title) in zip(axes, matrices):
        im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(3), EXPERT_LABELS)
        ax.set_yticks(range(3), REGIME_LABELS)
        ax.set_xlabel("Selected Expert")
        ax.set_ylabel("True Regime")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                value = matrix[i, j]
                text = "nan" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive analysis of problematic nrtend samples")
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
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=int,
        default=200,
        help="Rows to keep in each top-outlier CSV",
    )
    parser.add_argument(
        "--batch_candidate_topk",
        "--batch-candidate-topk",
        dest="batch_candidate_topk",
        type=int,
        default=24,
        help="Per-batch candidate count for each top-outlier tracker",
    )
    parser.add_argument(
        "--focus_groups",
        "--focus-groups",
        dest="focus_groups",
        type=int,
        default=8,
        help="How many high-impact regime/bin groups to surface in the feature-shift summary",
    )
    parser.add_argument(
        "--min_focus_severe_count",
        "--min-focus-severe-count",
        dest="min_focus_severe_count",
        type=int,
        default=50,
        help="Minimum severe sample count required for a group to appear in the focus summary",
    )
    parser.add_argument(
        "--skip_focus_sample_dump",
        action="store_true",
        help="Skip the second pass that dumps all severe rows for focus groups",
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
    input_cols = list(getattr(val_dataset, "input_cols", []))
    n_features = len(input_cols)

    nrtend_threshold = data_cfg.get("nrtend_regime_threshold")
    nrtend_threshold = float(nrtend_threshold) if nrtend_threshold is not None else None
    if nrtend_threshold is None:
        raise ValueError("This analysis expects data.nrtend_regime_threshold to be set.")

    n_groups = 3 * (len(MAG_BIN_EDGES) - 1)

    group_accumulators = [ProblemBinAccumulator() for _ in range(n_groups)]
    group_all_feature_stats = [RunningFeatureStats(n_features) for _ in range(n_groups)]
    group_severe_feature_stats = [RunningFeatureStats(n_features) for _ in range(n_groups)]

    global_all_feature_stats = RunningFeatureStats(n_features)
    global_severe_feature_stats = RunningFeatureStats(n_features)

    router_all_counts = np.zeros((3, 3), dtype=np.int64)
    router_severe_counts = np.zeros((3, 3), dtype=np.int64)

    top_abs_error = TopKTracker(args.top_k)
    top_error_ratio = TopKTracker(args.top_k)
    top_magnitude_ratio = TopKTracker(args.top_k)
    top_near_zero_leak = TopKTracker(args.top_k)

    processed_batches = 0
    processed_samples = 0
    total_abs_error = 0.0
    total_sq_error = 0.0
    total_soft_abs_error = 0.0
    total_soft_sq_error = 0.0

    start_time = pd.Timestamp.utcnow()

    with torch.no_grad():
        for batch_idx, batch in enumerate(iter_validation_batches_with_metadata(val_dataset), start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs_scaled = batch["inputs_scaled"]
            targets_scaled_dataset = batch["targets_scaled_dataset"]
            file_path = batch["file_path"]
            row_in_file = batch["row_in_file"]

            analysis = analyze_nrtend_batch(
                model=model,
                inputs_scaled=inputs_scaled,
                targets_scaled_dataset=targets_scaled_dataset,
                dataset_cols=dataset_cols,
                dataset=val_dataset,
                nrtend_threshold=nrtend_threshold,
                device=device,
            )

            total_abs_error += float(np.sum(analysis["abs_err"]))
            total_sq_error += float(np.dot(analysis["abs_err"], analysis["abs_err"]))
            total_soft_abs_error += float(np.sum(analysis["soft_abs_err"]))
            total_soft_sq_error += float(np.dot(analysis["soft_abs_err"], analysis["soft_abs_err"]))
            processed_batches += 1
            processed_samples += int(analysis["abs_err"].size)

            global_all_feature_stats.update(inputs_scaled)
            if np.any(analysis["severe_flag"]):
                global_severe_feature_stats.update(inputs_scaled[analysis["severe_flag"]])

            np.add.at(router_all_counts, (analysis["true_regime"], analysis["selected_expert"]), 1)
            if np.any(analysis["severe_flag"]):
                np.add.at(
                    router_severe_counts,
                    (
                        analysis["true_regime"][analysis["severe_flag"]],
                        analysis["selected_expert"][analysis["severe_flag"]],
                    ),
                    1,
                )

            for gid in np.unique(analysis["group_idx"]):
                mask = analysis["group_idx"] == gid
                if not np.any(mask):
                    continue

                group_accumulators[int(gid)].update(
                    abs_true=analysis["abs_true"][mask],
                    abs_pred=analysis["abs_pred"][mask],
                    soft_abs_pred=analysis["soft_abs_pred"][mask],
                    abs_err=analysis["abs_err"][mask],
                    soft_abs_err=analysis["soft_abs_err"][mask],
                    sign_mismatch=analysis["sign_mismatch"][mask],
                    soft_sign_mismatch=analysis["soft_sign_mismatch"][mask],
                    router_mismatch=analysis["router_mismatch"][mask],
                    mag_ratio=analysis["magnitude_ratio"][mask],
                    soft_mag_ratio=analysis["soft_magnitude_ratio"][mask],
                    err_ratio=analysis["error_ratio"][mask],
                    soft_err_ratio=analysis["soft_error_ratio"][mask],
                    near_zero_leak_multiplier=analysis["near_zero_leak_multiplier"][mask],
                    mild_flag=analysis["mild_flag"][mask],
                    severe_flag=analysis["severe_flag"][mask],
                    extreme_flag=analysis["extreme_flag"][mask],
                    soft_mild_flag=analysis["soft_mild_flag"][mask],
                    soft_severe_flag=analysis["soft_severe_flag"][mask],
                    soft_extreme_flag=analysis["soft_extreme_flag"][mask],
                    selected_expert=analysis["selected_expert"][mask],
                    router_conf=analysis["router_conf"][mask],
                    is_near_zero=analysis["near_zero_mask"][mask],
                )
                group_all_feature_stats[int(gid)].update(inputs_scaled[mask])

                severe_mask = mask & analysis["severe_flag"]
                if np.any(severe_mask):
                    group_severe_feature_stats[int(gid)].update(inputs_scaled[severe_mask])

            candidate_maps = {
                "abs_error": top_indices(analysis["abs_err"], args.batch_candidate_topk),
                "error_ratio": top_indices(
                    np.where(~analysis["near_zero_mask"], analysis["error_ratio"], -np.inf),
                    args.batch_candidate_topk,
                ),
                "magnitude_ratio": top_indices(
                    np.where(~analysis["near_zero_mask"], analysis["magnitude_ratio"], -np.inf),
                    args.batch_candidate_topk,
                ),
                "near_zero_leak": top_indices(
                    np.where(analysis["near_zero_mask"], analysis["near_zero_leak_multiplier"], -np.inf),
                    args.batch_candidate_topk,
                ),
            }

            union_candidates = sorted(
                set(int(i) for indices in candidate_maps.values() for i in indices.tolist())
            )
            if union_candidates:
                candidate_inputs_scaled = inputs_scaled[union_candidates]
                candidate_inputs_raw = inverse_input_pipeline(candidate_inputs_scaled, input_cols, val_dataset)
                candidate_metadata_frame = (
                    batch["metadata_frame"].iloc[union_candidates].reset_index(drop=True)
                    if batch["metadata_frame"] is not None else None
                )
                candidate_variant_details = compute_nrtend_variant_details_for_subset(
                    base_pred_norm_dataset_subset=analysis["pred_norm_dataset"][union_candidates],
                    soft_scaled_subset=analysis["soft_scaled"][union_candidates],
                    expert_scaled_subset=analysis["expert_scaled"][union_candidates],
                    dataset_cols=dataset_cols,
                    dataset=val_dataset,
                )
                pos_lookup = {batch_pos: local_pos for local_pos, batch_pos in enumerate(union_candidates)}

                for tracker_name, score_array, tracker in (
                    ("abs_error", analysis["abs_err"], top_abs_error),
                    ("error_ratio", analysis["error_ratio"], top_error_ratio),
                    ("magnitude_ratio", analysis["magnitude_ratio"], top_magnitude_ratio),
                    ("near_zero_leak", analysis["near_zero_leak_multiplier"], top_near_zero_leak),
                ):
                    for idx in candidate_maps[tracker_name]:
                        local = pos_lookup[int(idx)]
                        metadata_row = (
                            candidate_metadata_frame.iloc[local].to_dict()
                            if candidate_metadata_frame is not None else None
                        )
                        row = build_sample_row(
                            idx=int(idx),
                            file_path=file_path,
                            row_in_file=row_in_file,
                            input_cols=input_cols,
                            input_scaled_row=candidate_inputs_scaled[local],
                            input_raw_row=candidate_inputs_raw[local],
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
                            candidate_local_idx=local,
                            variant_details=candidate_variant_details,
                            metadata_row=metadata_row,
                        )
                        tracker.add(score_array[int(idx)], row)

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                elapsed_s = (pd.Timestamp.utcnow() - start_time).total_seconds()
                logger.info(
                    "Processed %d batches, %d samples, elapsed %.1fs",
                    batch_idx,
                    processed_samples,
                    elapsed_s,
                )

    bin_rows = []
    n_bins = len(MAG_BIN_EDGES) - 1
    for regime_idx, regime_name in enumerate(REGIME_LABELS):
        for b in range(n_bins):
            gid = regime_idx * n_bins + b
            acc = group_accumulators[gid].finalize()
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
            row.update(acc)
            row["share_total_abs_error"] = row["sum_abs_error"] / total_abs_error if total_abs_error > 0 else math.nan
            row["share_total_sq_error"] = row["sum_sq_error"] / total_sq_error if total_sq_error > 0 else math.nan
            row["share_total_abs_error_soft"] = (
                row["sum_soft_abs_error"] / total_soft_abs_error if total_soft_abs_error > 0 else math.nan
            )
            row["share_total_sq_error_soft"] = (
                row["sum_soft_sq_error"] / total_soft_sq_error if total_soft_sq_error > 0 else math.nan
            )
            row["rmse_ratio_hard_over_soft"] = (
                row["rmse"] / row["soft_rmse"]
                if np.isfinite(row["rmse"]) and np.isfinite(row["soft_rmse"]) and row["soft_rmse"] > 0
                else math.nan
            )
            row["mae_ratio_hard_over_soft"] = (
                row["mae"] / row["soft_mae"]
                if np.isfinite(row["mae"]) and np.isfinite(row["soft_mae"]) and row["soft_mae"] > 0
                else math.nan
            )
            bin_rows.append(row)

    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(args.output_dir / "problematic_bin_summary.csv", index=False)

    focus_df = (
        bin_df[
            (bin_df["count"] > 0)
            & (bin_df["severe_fraction"].fillna(0.0) > 0.0)
            & ((bin_df["severe_fraction"] * bin_df["count"]) >= args.min_focus_severe_count)
        ]
        .sort_values(["share_total_sq_error", "share_total_abs_error"], ascending=False)
        .head(args.focus_groups)
        .copy()
    )
    focus_df.to_csv(args.output_dir / "problematic_focus_groups.csv", index=False)

    feature_rows = []

    overall_mean = global_all_feature_stats.mean()
    overall_std = global_all_feature_stats.std()
    severe_mean = global_severe_feature_stats.mean()
    severe_fraction_global = (
        global_severe_feature_stats.count / global_all_feature_stats.count
        if global_all_feature_stats.count > 0 else math.nan
    )
    for j, feature in enumerate(input_cols):
        denom = overall_std[j] if np.isfinite(overall_std[j]) and overall_std[j] > 0 else np.nan
        z_shift = (severe_mean[j] - overall_mean[j]) / denom if np.isfinite(denom) else math.nan
        feature_rows.append({
            "group_label": f"ALL severe ({severe_fraction_global:.3%})",
            "regime": "all",
            "bin_label": "all",
            "feature": feature,
            "all_count": global_all_feature_stats.count,
            "severe_count": global_severe_feature_stats.count,
            "severe_fraction": severe_fraction_global,
            "all_mean_scaled": overall_mean[j],
            "severe_mean_scaled": severe_mean[j],
            "all_std_scaled": overall_std[j],
            "z_shift": z_shift,
        })

    for _, focus_row in focus_df.iterrows():
        regime_name = str(focus_row["regime"])
        b = int(focus_row["bin_index"])
        gid = REGIME_LABELS.index(regime_name) * n_bins + b

        all_stats = group_all_feature_stats[gid]
        severe_stats = group_severe_feature_stats[gid]

        all_mean = all_stats.mean()
        all_std = all_stats.std()
        sev_mean = severe_stats.mean()
        sev_fraction = severe_stats.count / all_stats.count if all_stats.count > 0 else math.nan

        group_label = (
            f"{regime_name} {focus_row['bin_label']} "
            f"(sq={focus_row['share_total_sq_error']:.2%}, severe={sev_fraction:.2%})"
        )

        for j, feature in enumerate(input_cols):
            denom = all_std[j] if np.isfinite(all_std[j]) and all_std[j] > 0 else np.nan
            z_shift = (sev_mean[j] - all_mean[j]) / denom if np.isfinite(denom) else math.nan
            feature_rows.append({
                "group_label": group_label,
                "regime": regime_name,
                "bin_label": focus_row["bin_label"],
                "feature": feature,
                "all_count": all_stats.count,
                "severe_count": severe_stats.count,
                "severe_fraction": sev_fraction,
                "all_mean_scaled": all_mean[j],
                "severe_mean_scaled": sev_mean[j],
                "all_std_scaled": all_std[j],
                "z_shift": z_shift,
            })

    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(args.output_dir / "problematic_feature_shifts.csv", index=False)

    router_rows = []
    for subset_name, matrix in (("all", router_all_counts), ("severe", router_severe_counts)):
        for true_idx, true_label in enumerate(REGIME_LABELS):
            row_total = int(np.sum(matrix[true_idx]))
            for expert_idx, expert_label in enumerate(EXPERT_LABELS):
                count = int(matrix[true_idx, expert_idx])
                frac = count / row_total if row_total > 0 else math.nan
                router_rows.append({
                    "subset": subset_name,
                    "true_regime": true_label,
                    "selected_expert": expert_label,
                    "count": count,
                    "row_fraction": frac,
                })

    router_df = pd.DataFrame(router_rows)
    router_df.to_csv(args.output_dir / "problematic_router_summary.csv", index=False)

    focus_rows: List[Dict[str, object]] = []
    if len(focus_df) > 0 and not args.skip_focus_sample_dump:
        focus_gid_to_label: Dict[int, str] = {}
        for _, focus_row in focus_df.iterrows():
            gid = REGIME_LABELS.index(str(focus_row["regime"])) * n_bins + int(focus_row["bin_index"])
            focus_gid_to_label[gid] = (
                f"{focus_row['regime']} {focus_row['bin_label']} "
                f"(sq={focus_row['share_total_sq_error']:.2%}, severe={focus_row['severe_fraction']:.2%})"
            )

        logger.info(
            "Collecting detailed severe rows for %d focus groups in a second pass...",
            len(focus_gid_to_label),
        )
        for batch_idx, batch in enumerate(iter_validation_batches_with_metadata(val_dataset), start=1):
            if args.max_eval_batches is not None and batch_idx > args.max_eval_batches:
                break

            inputs_scaled = batch["inputs_scaled"]
            analysis = analyze_nrtend_batch(
                model=model,
                inputs_scaled=inputs_scaled,
                targets_scaled_dataset=batch["targets_scaled_dataset"],
                dataset_cols=dataset_cols,
                dataset=val_dataset,
                nrtend_threshold=nrtend_threshold,
                device=device,
            )

            focus_mask = analysis["severe_flag"] & np.isin(
                analysis["group_idx"],
                np.array(list(focus_gid_to_label.keys()), dtype=np.int64),
            )
            if not np.any(focus_mask):
                continue

            focus_indices = np.where(focus_mask)[0]
            focus_inputs_scaled = inputs_scaled[focus_indices]
            focus_inputs_raw = inverse_input_pipeline(focus_inputs_scaled, input_cols, val_dataset)
            focus_metadata_frame = (
                batch["metadata_frame"].iloc[focus_indices].reset_index(drop=True)
                if batch["metadata_frame"] is not None else None
            )
            focus_variant_details = compute_nrtend_variant_details_for_subset(
                base_pred_norm_dataset_subset=analysis["pred_norm_dataset"][focus_indices],
                soft_scaled_subset=analysis["soft_scaled"][focus_indices],
                expert_scaled_subset=analysis["expert_scaled"][focus_indices],
                dataset_cols=dataset_cols,
                dataset=val_dataset,
            )

            for local, idx in enumerate(focus_indices):
                metadata_row = (
                    focus_metadata_frame.iloc[local].to_dict()
                    if focus_metadata_frame is not None else None
                )
                gid = int(analysis["group_idx"][idx])
                focus_rows.append(
                    build_sample_row(
                        idx=int(idx),
                        file_path=batch["file_path"],
                        row_in_file=batch["row_in_file"],
                        input_cols=input_cols,
                        input_scaled_row=focus_inputs_scaled[local],
                        input_raw_row=focus_inputs_raw[local],
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
                        candidate_local_idx=local,
                        variant_details=focus_variant_details,
                        metadata_row=metadata_row,
                        group_label=focus_gid_to_label[gid],
                    )
                )

            if args.log_every > 0 and batch_idx % max(args.log_every, 50) == 0:
                logger.info("Second pass processed %d batches for focus-group dumps", batch_idx)

    focus_rows_df = pd.DataFrame(focus_rows)
    focus_rows_df.to_csv(args.output_dir / "focus_severe_samples.csv", index=False)
    if not focus_rows_df.empty:
        focus_file_summary = (
            focus_rows_df.groupby(["group_label", "file_path"], dropna=False)
            .agg(
                count=("row_in_file", "size"),
                min_row_in_file=("row_in_file", "min"),
                max_row_in_file=("row_in_file", "max"),
                mean_router_confidence=("router_confidence", "mean"),
                mean_abs_error=("nrtend_abs_error", "mean"),
                mean_soft_abs_error=("nrtend_soft_abs_error", "mean"),
            )
            .reset_index()
        )
        focus_file_summary["fraction_within_group"] = (
            focus_file_summary["count"] / focus_file_summary.groupby("group_label")["count"].transform("sum")
        )
        focus_file_summary = focus_file_summary.sort_values(
            ["group_label", "count", "mean_abs_error"],
            ascending=[True, False, False],
        )
        focus_file_summary.to_csv(args.output_dir / "focus_severe_file_summary.csv", index=False)

        metadata_cols_present = [col for col in focus_rows_df.columns if col.startswith("metadata__")]
        metadata_rows: List[Dict[str, object]] = []
        for group_label, group_df in focus_rows_df.groupby("group_label", dropna=False):
            group_total = len(group_df)
            for col in metadata_cols_present:
                top_values = group_df[col].value_counts(dropna=False).head(25)
                for value, count in top_values.items():
                    metadata_rows.append({
                        "group_label": group_label,
                        "metadata_col": col.replace("metadata__", "", 1),
                        "metadata_value": value,
                        "count": int(count),
                        "fraction_within_group": count / group_total if group_total > 0 else math.nan,
                    })
        pd.DataFrame(metadata_rows).to_csv(args.output_dir / "focus_severe_metadata_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "focus_severe_file_summary.csv", index=False)
        pd.DataFrame().to_csv(args.output_dir / "focus_severe_metadata_summary.csv", index=False)

    pd.DataFrame(top_abs_error.rows()).to_csv(args.output_dir / "top_abs_error_samples.csv", index=False)
    pd.DataFrame(top_error_ratio.rows()).to_csv(args.output_dir / "top_error_ratio_samples.csv", index=False)
    pd.DataFrame(top_magnitude_ratio.rows()).to_csv(args.output_dir / "top_magnitude_ratio_samples.csv", index=False)
    pd.DataFrame(top_near_zero_leak.rows()).to_csv(args.output_dir / "top_near_zero_leak_samples.csv", index=False)

    plot_problematic_fraction_by_bin(bin_df, args.output_dir / "problematic_fraction_by_bin.png")
    plot_error_contribution_by_bin(bin_df, args.output_dir / "error_contribution_by_bin.png")
    plot_soft_vs_hard_gap_by_bin(bin_df, args.output_dir / "soft_vs_hard_gap_by_bin.png")
    plot_feature_shift_heatmap(feature_df, args.output_dir / "feature_shift_heatmap.png")
    plot_router_heatmaps(router_all_counts, router_severe_counts, args.output_dir / "router_problematic_heatmaps.png")

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "device": str(device),
        "scaler_run_id": scaler_run_id,
        "processed_batches": processed_batches,
        "processed_samples": processed_samples,
        "max_eval_batches": args.max_eval_batches,
        "nrtend_target_zeroing_threshold": nrtend_threshold,
        "mag_bin_edges": MAG_BIN_EDGES.tolist(),
        "global_total_abs_error": total_abs_error,
        "global_total_sq_error": total_sq_error,
        "global_total_abs_error_soft": total_soft_abs_error,
        "global_total_sq_error_soft": total_soft_sq_error,
        "global_severe_count": global_severe_feature_stats.count,
        "global_severe_fraction": (
            global_severe_feature_stats.count / global_all_feature_stats.count
            if global_all_feature_stats.count > 0 else math.nan
        ),
        "focus_groups_count": int(len(focus_df)),
        "focus_severe_rows_dumped": int(len(focus_rows_df)),
    }
    with (args.output_dir / "problematic_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Problematic-sample analysis complete.")
    logger.info("Processed %d batches and %d validation samples.", processed_batches, processed_samples)
    logger.info("Saved artifacts to %s", args.output_dir)


if __name__ == "__main__":
    main()
