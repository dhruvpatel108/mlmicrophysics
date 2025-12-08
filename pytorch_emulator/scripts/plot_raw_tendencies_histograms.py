#!/usr/bin/env python3
"""
Plot histograms of raw tendencies directly from parquet files (no model preprocessing).

Generates three figures (2x2 subplots for the four tendencies):
  1) Raw tendencies (values as stored in parquet)
  2) Log10-transformed tendencies: sign(x) * log10(|x| + eps)
  3) Log10-transformed tendencies for ACTIVE samples only
     (active defined by |qctend_TAU| > active_threshold computed on RAW values)
  4) Quantile-transformed tendencies (Gaussian), using config quantile params

Usage:
  python scripts/plot_raw_tendencies_histograms.py \
    --config mlmicrophysics/pytorch_emulator/configs/deception_quick_test.yml \
    --max-samples -1 \
    --bins 100 \
    --output-dir plots/preprocessed_histograms_quick \
    --active-threshold 1e-12

Notes:
- Reads output_cols from the provided config under data.output_cols.
- Reads data_path from config under data.data_path.
- For active-mask filtering, also reads a cloud fraction column (e.g., 'CLOUD') from input_cols if present.
- Uses headless matplotlib backend for HPC environments.
"""
import argparse
import os
import sys
from pathlib import Path
import math
import logging
import yaml
from sklearn.preprocessing import QuantileTransformer
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("plot_raw_tendencies_histograms")


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_parquet_files(data_path: str, max_files: int = None):
    p = Path(data_path)
    files = sorted(p.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    return files


def quantile_transform_per_feature(data_dict: dict, n_quantiles: int, subsample: int, random_state: int) -> dict:
    out = {}
    for col, arr in data_dict.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            out[col] = arr
            continue
        qt = QuantileTransformer(
            n_quantiles=min(n_quantiles, arr.shape[0]),
            output_distribution='normal',
            subsample=subsample,
            copy=True,
            random_state=random_state
        )
        arr_2d = arr.reshape(-1, 1)
        try:
            tr = qt.fit_transform(arr_2d)
            out[col] = tr.reshape(-1)
        except Exception:
            out[col] = arr
    return out


def collect_tendencies_samples(files, tendency_cols, max_samples: int, extra_cols: list | None = None):
    """Collect up to max_samples for each requested column from parquet files.

    Returns:
        tendencies: dict[str, np.ndarray]
        extras: dict[str, np.ndarray] (empty if extra_cols is None)
    """
    extra_cols = extra_cols or []
    all_cols = list(dict.fromkeys(list(tendency_cols) + list(extra_cols)))
    tend_buffers = {col: [] for col in tendency_cols}
    extra_buffers = {col: [] for col in extra_cols}
    collected = 0
    unlimited = (max_samples is None) or (max_samples <= 0)

    for file_path in files:
        try:
            # Try chunked read if available
            try:
                chunk_iter = pd.read_parquet(file_path, columns=all_cols, chunksize=200_000)
            except TypeError:
                df = pd.read_parquet(file_path, columns=all_cols)
                # split manually to limit memory
                rows = len(df)
                step = 200_000
                chunk_iter = [df.iloc[i:i+step] for i in range(0, rows, step)]

            for chunk in chunk_iter:
                if (not unlimited) and collected >= max_samples:
                    break
                chunk = chunk.dropna(subset=tendency_cols)
                if chunk.empty:
                    continue
                if unlimited:
                    sub = chunk
                    take = len(sub)
                else:
                    remaining = max_samples - collected
                    take = min(remaining, len(chunk))
                    if take < len(chunk):
                        idx = np.random.choice(len(chunk), size=take, replace=False)
                        sub = chunk.iloc[idx]
                    else:
                        sub = chunk
                for col in tendency_cols:
                    tend_buffers[col].append(sub[col].to_numpy())
                for col in extra_cols:
                    if col in sub.columns:
                        extra_buffers[col].append(sub[col].to_numpy())
                collected += len(sub)
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue
        if (not unlimited) and collected >= max_samples:
            break

    tendencies = {}
    for col in tendency_cols:
        if tend_buffers[col]:
            tendencies[col] = np.concatenate(tend_buffers[col], axis=0)
        else:
            tendencies[col] = np.array([])

    extras = {}
    for col in extra_cols:
        if extra_buffers[col]:
            extras[col] = np.concatenate(extra_buffers[col], axis=0)
        else:
            extras[col] = np.array([])

    return tendencies, extras


def plot_hist_grid(data_dict: dict, title: str, bins: int, fig_path: Path):
    names = list(data_dict.keys())
    k = len(names)
    if k == 0:
        logger.warning("No variables to plot")
        return
    n_cols = int(math.ceil(math.sqrt(k)))
    n_rows = int(math.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols + 1, 3.4 * n_rows + 1))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        arr = np.asarray(data_dict[name]).astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            ax.set_title(f"{name} (no data)")
            ax.axis('off')
            continue
        ax.hist(arr, bins=bins, color='steelblue', alpha=0.85, density=False)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    for j in range(k, len(axes)):
        axes[j].axis('off')
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {fig_path}")



def sign_log10(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    x = np.asarray(x)
    sign = np.sign(x)
    #print(f"sign.shape = {sign.shape} | sign[:10] = {sign[:10]} | sign[:-10] = {sign[:-10]}")
    abs_val = np.abs(x) 
    #print(f"abs_val.shape = {abs_val.shape} | abs_val[:10] = {abs_val[:10]} | abs_val[:-10] = {abs_val[:-10]}")
    return np.log10(abs_val + eps) 



def main():
    ap = argparse.ArgumentParser(description="Plot raw/log10/active histograms for tendencies from parquet")
    ap.add_argument('--config', type=str, required=True)
    ap.add_argument('--max-files', type=int, default=None, help='Limit number of parquet files to scan')
    ap.add_argument('--max-samples', type=int, default=-1, help='Number of samples to plot; <=0 means all samples')
    ap.add_argument('--bins', type=int, default=100)
    ap.add_argument('--output-dir', type=str, default='plots/tendencies_raw')
    ap.add_argument('--active-threshold', type=float, default=None, help='Override active threshold; default from config or 1e-12')
    ap.add_argument('--eps', type=float, default=1e-16, help='Epsilon for log10 transformation')
    ap.add_argument('--mass-input-threshold', type=float, default=1e-5, help='Threshold for QC_TAU_in and QR_TAU_in in mass-augmented active mask')
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get('data', {})
    data_path = data_cfg.get('data_path')
    tendency_cols = data_cfg.get('output_cols', [])
    if not data_path or not tendency_cols:
        raise ValueError("config.data.data_path and config.data.output_cols are required")
    active_threshold = args.active_threshold if args.active_threshold is not None else float(data_cfg.get('active_threshold', 1e-12))
    # Cloud fraction and mass input columns
    cloud_col = 'CLOUD'
    qc_col = 'QC_TAU_in'
    qr_col = 'QR_TAU_in'
    input_cols = data_cfg.get('input_cols', [])
    q_nq = int(data_cfg.get('quantile_n_quantiles', 1000))
    q_sub = int(data_cfg.get('quantile_subsample', 100000))
    rnd = int(data_cfg.get('random_seed', 42))

    files = list_parquet_files(data_path, max_files=args.max_files)
    logger.info(f"Found {len(files)} parquet files (using up to {args.max_files or len(files)})")

    # 1) Raw samples (and CLOUD/QC/QR for active masks)
    raw, extras = collect_tendencies_samples(files, tendency_cols, args.max_samples, extra_cols=input_cols)
    plot_hist_grid(raw, title='Raw tendencies (from parquet)', bins=args.bins,
                   fig_path=Path(args.output_dir) / f'raw_tendencies_{args.max_samples}_eps{args.eps}.png')

    # print the max, min, mean, std, zeros_percentage, p25, p50, p75, p95, p99 of the raw tendencies
    for col in tendency_cols:
        print(f"{col}: max={np.max(raw[col])}, min={np.min(raw[col])}, mean={np.mean(raw[col])}, std={np.std(raw[col])}, zeros_percentage={np.sum(raw[col] == 0) / len(raw[col])}, p25={np.percentile(raw[col], 25)}, p50={np.percentile(raw[col], 50)}, p75={np.percentile(raw[col], 75)}, p95={np.percentile(raw[col], 95)}, p99={np.percentile(raw[col], 99)}")

    # 2) Log10-transformed samples: sign*log10(|x|+eps)
    logt = {col: sign_log10(raw[col], eps=args.eps) if raw[col].size else raw[col] for col in tendency_cols}
    #sign_qctend_TAU = np.sign(raw['qctend_TAU'])
    #print(f"sign_qctend_TAU.shape = {sign_qctend_TAU.shape} | sign_qctend_TAU = {sign_qctend_TAU}")
    #abs_qctend_TAU = np.abs(raw['qctend_TAU'])
    #print(f"abs_qctend_TAU.shape = {abs_qctend_TAU.shape} | abs_qctend_TAU = {abs_qctend_TAU}")
    #log10_qctend_TAU =np.log10(abs_qctend_TAU + args.eps)
    #print(f"log10_qctend_TAU.shape = {log10_qctend_TAU.shape} | log10_qctend_TAU = {log10_qctend_TAU}")
    #signed_log10_qctend_TAU = sign_qctend_TAU * log10_qctend_TAU
    #print(f"signed_log10_qctend_TAU.shape = {signed_log10_qctend_TAU.shape} | signed_log10_qctend_TAU = {signed_log10_qctend_TAU}")
    
    
    
    # printing stuff for log10 values of qctend_TAU
    print("--------------------------------")
    # print the shape/size of the logt['qctend_TAU']
    print(f"No. of NaNs = {np.sum(np.isnan(np.asarray(logt['qctend_TAU'])))} | percentage of NaNs = {(np.sum(np.isnan(np.asarray(logt['qctend_TAU']))) / len(logt['qctend_TAU'])) * 100}%")
    print(f"log10(qctend_TAU): shape={logt['qctend_TAU'].shape}, size={len(logt['qctend_TAU'])} | logt['qctend_TAU'][:10]={logt['qctend_TAU'][:10]} | unique={np.unique(np.asarray(logt['qctend_TAU']))}, n_unique={len(np.unique(np.asarray(logt['qctend_TAU'])))}")
    print("--------------------------------")
    print(f"log10(qctend_TAU): max={np.max(logt['qctend_TAU'])}, min={np.min(logt['qctend_TAU'])}, mean={np.mean(logt['qctend_TAU'])}, std={np.std(logt['qctend_TAU'])}, zeros_percentage={np.sum(logt['qctend_TAU'] == 0) / len(logt['qctend_TAU'])}, p25={np.percentile(logt['qctend_TAU'], 25)}, p50={np.percentile(logt['qctend_TAU'], 50)}, p75={np.percentile(logt['qctend_TAU'], 75)}, p95={np.percentile(logt['qctend_TAU'], 95)}, p99={np.percentile(logt['qctend_TAU'], 99)}")
    print("--------------------------------")
    print(f"log10(qrtend_TAU): max={np.max(logt['qrtend_TAU'])}, min={np.min(logt['qrtend_TAU'])}, mean={np.mean(logt['qrtend_TAU'])}, std={np.std(logt['qrtend_TAU'])}, zeros_percentage={np.sum(logt['qrtend_TAU'] == 0) / len(logt['qrtend_TAU'])}, p25={np.percentile(logt['qrtend_TAU'], 25)}, p50={np.percentile(logt['qrtend_TAU'], 50)}, p75={np.percentile(logt['qrtend_TAU'], 75)}, p95={np.percentile(logt['qrtend_TAU'], 95)}, p99={np.percentile(logt['qrtend_TAU'], 99)}")
    print("--------------------------------")
    print(f"log10(nctend_TAU): max={np.max(logt['nctend_TAU'])}, min={np.min(logt['nctend_TAU'])}, mean={np.mean(logt['nctend_TAU'])}, std={np.std(logt['nctend_TAU'])}, zeros_percentage={np.sum(logt['nctend_TAU'] == 0) / len(logt['nctend_TAU'])}, p25={np.percentile(logt['nctend_TAU'], 25)}, p50={np.percentile(logt['nctend_TAU'], 50)}, p75={np.percentile(logt['nctend_TAU'], 75)}, p95={np.percentile(logt['nctend_TAU'], 95)}, p99={np.percentile(logt['nctend_TAU'], 99)}")
    print("--------------------------------")
    print(f"log10(nrtend_TAU): max={np.max(logt['nrtend_TAU'])}, min={np.min(logt['nrtend_TAU'])}, mean={np.mean(logt['nrtend_TAU'])}, std={np.std(logt['nrtend_TAU'])}, zeros_percentage={np.sum(logt['nrtend_TAU'] == 0) / len(logt['nrtend_TAU'])}, p25={np.percentile(logt['nrtend_TAU'], 25)}, p50={np.percentile(logt['nrtend_TAU'], 50)}, p75={np.percentile(logt['nrtend_TAU'], 75)}, p95={np.percentile(logt['nrtend_TAU'], 95)}, p99={np.percentile(logt['nrtend_TAU'], 99)}")
    print("--------------------------------")
    print("================================================")
    print("================================================")
    print("================================================")
    print(f"Total number of samples = {len(logt['qctend_TAU'])}")
    plot_hist_grid(logt, title='Log10-transformed tendencies (sign * log10(|x|+eps))', bins=args.bins,
                   fig_path=Path(args.output_dir) / f'log10_tendencies_{args.max_samples}_eps{args.eps}.png')
    
    # 3) Active-only, defined by |qctend_TAU| > threshold on RAW values
    if 'qctend_TAU' not in tendency_cols:
        logger.warning("qctend_TAU not in output_cols; cannot compute active mask. Skipping active-only histogram.")
    else:
        qct_raw = raw['qctend_TAU']
        if qct_raw.size == 0:
            logger.warning("No qctend_TAU data to build active mask. Skipping active-only histogram.")
        else:
            cloud = extras.get(cloud_col, np.array([]))
            if cloud.size == 0:
                raise ValueError("CLOUD column not found or empty; required for active mask.")
            print(f"qct_raw.shape = {qct_raw.shape}, cloud.shape = {cloud.shape}")
            # Basic active mask
            min_len_basic = min(len(qct_raw), len(cloud))
            qct_arr_basic = qct_raw[:min_len_basic]
            cloud_arr_basic = cloud[:min_len_basic]
            active_mask_basic = (np.abs(qct_arr_basic) > active_threshold) & (cloud_arr_basic > 0.01)
            print(f"active_mask_basic.shape = {active_mask_basic.shape}")

            logt_active = {}
            for col in tendency_cols:
                arr = raw[col]
                if arr.size:
                    # Align to min_len_basic
                    arr_use = arr[:min_len_basic]
                    arr_active = arr_use[active_mask_basic]
                    arr_log = sign_log10(arr_active, eps=args.eps   ) if arr_active.size else arr_active
                    logt_active[col] = arr_log
                else:
                    logt_active[col] = arr
            plot_hist_grid(logt_active, title=f'Log10 tendencies, ACTIVE samples (|qctend_TAU| > {active_threshold:g}) & CLOUD>0.01', bins=args.bins,
                           fig_path=Path(args.output_dir) / f'log10_tendencies_active_{args.max_samples}_eps{args.eps}.png')

            # Mass-augmented mask: add QC_TAU_in and QR_TAU_in thresholds
            qc_vals = extras.get(qc_col, np.array([]))
            qr_vals = extras.get(qr_col, np.array([]))
            if qc_vals.size == 0 or qr_vals.size == 0:
                raise ValueError("QC_TAU_in/QR_TAU_in not found or empty; required for mass-augmented active mask.")
            min_len_mass = min(min_len_basic, len(qc_vals), len(qr_vals))
            qct_arr_mass = qct_raw[:min_len_mass]
            cloud_arr_mass = cloud[:min_len_mass]
            qc_arr = qc_vals[:min_len_mass]
            qr_arr = qr_vals[:min_len_mass]
            mass_thresh = args.mass_input_threshold 
            active_mask_mass = (np.abs(qct_arr_mass) > active_threshold) & (cloud_arr_mass > 0.01) & (qc_arr > mass_thresh) & (qr_arr > mass_thresh)

            logt_active_mass = {}
            for col in tendency_cols:
                arr = raw[col]
                if arr.size:
                    arr_use_m = arr[:min_len_mass]
                    arr_active_m = arr_use_m[active_mask_mass]
                    arr_log_m = sign_log10(arr_active_m, eps=args.eps) if arr_active_m.size else arr_active_m
                    logt_active_mass[col] = arr_log_m
                else:
                    logt_active_mass[col] = arr
            plot_hist_grid(logt_active_mass, title=f'Log10 tendencies, ACTIVE+MASS (CLOUD>0.01, QC/QR>{mass_thresh:g})', bins=args.bins,
                           fig_path=Path(args.output_dir) / f'log10_tendencies_active_mass_{args.max_samples}_eps{args.eps}.png')

    # 3b) Split ACTIVE log10 distributions into 'tower' vs 'notower' per variable
    def split_tower(arr: np.ndarray, bins: int):
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None, None, None
        hist, edges = np.histogram(arr, bins=bins)
        if hist.size == 0:
            return None, None, None
        idx = int(np.argmax(hist))
        if idx == 0:
            boundary = edges[1]
            mask = arr <= boundary
            return mask, 'left', boundary
        elif idx == hist.size - 1:
            boundary = edges[-2]
            mask = arr >= boundary
            return mask, 'right', boundary
        else:
            return None, None, None

    def split_tower_mask_full(arr: np.ndarray, bins: int) -> np.ndarray | None:
        # Returns boolean mask aligned to arr (including non-finite -> False)
        finite = np.isfinite(arr)
        arr_f = arr[finite]
        mask_f, _, _ = split_tower(arr_f, bins)
        if mask_f is None:
            return None
        full = np.zeros_like(arr, dtype=bool)
        full[finite] = mask_f
        return full

    if 'logt_active' in locals() and any(v.size for v in logt_active.values()):
        tower_dict = {}
        notower_dict = {}
        for col, arr in logt_active.items():
            if arr.size == 0:
                tower_dict[col] = arr
                notower_dict[col] = arr
                continue
            mask, orient, boundary = split_tower(arr, bins=args.bins)
            if mask is None:
                # No extreme-dominant bin; place all into notower for this variable
                tower_dict[col] = np.array([])
                notower_dict[col] = arr
            else:
                # Rebuild mask aligned to original (non-finite already dropped earlier)
                fin = np.isfinite(arr)
                # Compose full-length mask where non-finite -> False
                full_mask = np.zeros_like(arr, dtype=bool)
                full_mask[fin] = mask
                tower_dict[col] = arr[full_mask]
                notower_dict[col] = arr[~full_mask]

        # Plot if any tower data present
        if any(v.size for v in tower_dict.values()):
            plot_hist_grid(
                tower_dict,
                title='Log10 tendencies, ACTIVE only - TOWER bins (extreme-dominant) per variable',
                bins=args.bins,
                fig_path=Path(args.output_dir) / f'log10_tendencies_active_tower_{args.max_samples}_eps{args.eps}.png'
            )
        # Plot notower
        if any(v.size for v in notower_dict.values()):
            plot_hist_grid(
                notower_dict,
                title='Log10 tendencies, ACTIVE only - NON-TOWER (remaining) per variable',
                bins=args.bins,
                fig_path=Path(args.output_dir) / f'log10_tendencies_active_notower_{args.max_samples}_eps{args.eps}.png'
            )

    # 3c) Tower split for mass-augmented active mask
    if 'logt_active_mass' in locals() and any(v.size for v in logt_active_mass.values()):
        tower_m = {}
        notower_m = {}
        tower_mask_map = {}
        for col, arr in logt_active_mass.items():
            if arr.size == 0:
                tower_m[col] = arr
                notower_m[col] = arr
                tower_mask_map[col] = np.zeros_like(arr, dtype=bool)
                continue
            mask_full = split_tower_mask_full(arr, bins=args.bins)
            if mask_full is None:
                tower_m[col] = np.array([])
                notower_m[col] = arr
                tower_mask_map[col] = np.zeros_like(arr, dtype=bool)
            else:
                tower_m[col] = arr[mask_full]
                notower_m[col] = arr[~mask_full]
                tower_mask_map[col] = mask_full
        if any(v.size for v in tower_m.values()):
            plot_hist_grid(
                tower_m,
                title='Log10 tendencies, ACTIVE+MASS - TOWER bins (extreme-dominant) per variable',
                bins=args.bins,
                fig_path=Path(args.output_dir) / f'log10_tendencies_active_mass_tower_{args.max_samples}_eps{args.eps}.png'
            )
        if any(v.size for v in notower_m.values()):
            plot_hist_grid(
                notower_m,
                title='Log10 tendencies, ACTIVE+MASS - NON-TOWER (remaining) per variable',
                bins=args.bins,
                fig_path=Path(args.output_dir) / f'log10_tendencies_active_mass_notower_{args.max_samples}_eps{args.eps}.png'
            )

        # 4) Scatter plots: each input vs each output (mass-augmented active samples)
        scatter_dir = Path(args.output_dir) / 'scatter_plots'
        scatter_dir.mkdir(parents=True, exist_ok=True)

        # Build the mass-augmented active boolean index on the raw arrays (aligned by min_len_mass)
        # Recompute to get index-level mask
        qc_vals = extras.get(qc_col, np.array([]))
        qr_vals = extras.get(qr_col, np.array([]))
        cloud = extras.get(cloud_col, np.array([]))
        min_len_mass = min(len(raw[tendency_cols[0]]), len(qc_vals), len(qr_vals), len(cloud), len(raw['qctend_TAU']))
        idx_mask_mass = (
            (np.abs(raw['qctend_TAU'][:min_len_mass]) > active_threshold) &
            (cloud[:min_len_mass] > 0.01) &
            (qc_vals[:min_len_mass] > (args.mass_input_threshold if args.mass_input_threshold is not None else active_threshold)) &
            (qr_vals[:min_len_mass] > (args.mass_input_threshold if args.mass_input_threshold is not None else active_threshold))
        )

        # For each output variable, get tower mask in the active set
        for out_col in tendency_cols:
            # Use logt_active_mass tower mask if available for this output
            arr_log_active = logt_active_mass.get(out_col, np.array([]))
            if arr_log_active.size == 0:
                continue
            tower_mask_active = tower_mask_map.get(out_col, np.zeros_like(arr_log_active, dtype=bool))
            # Now assemble x (inputs) and y (raw outputs) for the same active indices
            # Build map from active compressed back to original indices: we used arr[:min_len_mass][active_mask_mass]
            # Create array of positions of active indices
            active_positions = np.nonzero(idx_mask_mass)[0]
            # y_raw aligned
            y_raw_full = raw[out_col][:min_len_mass]
            y_active = y_raw_full[idx_mask_mass]
            # Sanity: lengths must match
            if len(y_active) != len(arr_log_active):
                # if mismatch due to any filtering, skip this output
                continue
            # For every input feature, scatter
            for in_col in input_cols:
                # All inputs have been sampled into extras; skip if missing
                x_full = extras.get(in_col, None)
                if x_full is None or x_full.size == 0:
                    continue
                x_full = x_full[:min_len_mass]
                x_active = x_full[idx_mask_mass]
                if len(x_active) != len(y_active):
                    continue
                # Split by tower mask (on active-compressed arrays)
                x_tower = x_active[tower_mask_active]
                y_tower = y_active[tower_mask_active]
                x_notower = x_active[~tower_mask_active]
                y_notower = y_active[~tower_mask_active]

                # Plot
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                if x_notower.size:
                    ax.scatter(x_notower, y_notower, s=2, alpha=0.2, color='blue', label='non-tower')
                if x_tower.size:
                    ax.scatter(x_tower, y_tower, s=12, alpha=0.9, color='red', marker='x', label='tower')
                ax.set_xlabel(in_col)
                ax.set_ylabel(out_col)
                ax.set_title(f'{in_col} vs {out_col} (active+mass)')
                ax.grid(True, alpha=0.3)
                if x_tower.size or x_notower.size:
                    ax.legend(loc='best', fontsize=8)
                plt.tight_layout()
                out_name = f'scatter_{in_col}_vs_{out_col}_active_mass_{args.max_samples}_eps{args.eps}.png'
                plt.savefig(scatter_dir / out_name, dpi=130)
                plt.close(fig)

    # 4) Quantile-transformed tendencies using config parameters
    qt_dict = quantile_transform_per_feature(raw, n_quantiles=q_nq, subsample=q_sub, random_state=rnd)
    plot_hist_grid(qt_dict, title=f'Quantile-transformed tendencies (normal), n_quantiles={q_nq}, subsample={q_sub}', bins=args.bins,
                   fig_path=Path(args.output_dir) / f'quantile_tendencies_{args.max_samples}_eps{args.eps}.png')

    logger.info("Done.")


if __name__ == '__main__':
    main()


