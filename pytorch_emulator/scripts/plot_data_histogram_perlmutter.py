"""
Plot histograms of raw tendencies directly from parquet files (no model preprocessing).

This script generates histogram plots for microphysics tendency data in three stages:
  1) Raw tendencies (values as stored in parquet files)
  2) Log10-transformed tendencies
  3) Log10-transformed tendencies for ACTIVE samples only (as defined by qctend_TAU and CLOUD)
  4) Log10-transformed tendencies for ACTIVE samples with MASS thresholds (as defined by qctend_TAU, CLOUD, QC_TAU_in, QR_TAU_in)

Usage:
  python plot_data_histograms.py \
    --config configs/optimized_production_full.yml \
    --max-samples 1000000 

NOTE: Use --max-samples -1 to load ALL samples

Requirements:
- Config file must contain data.data_path, data.output_cols, data.input_cols
- Parquet files must contain tendency columns and CLOUD, QC_TAU_in, QR_TAU_in
"""

import argparse
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for HPC environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION AND FILE HANDLING
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_parquet_files(data_path: str, max_files: int = None) -> list:
    """
    Find all parquet files in the specified directory.
    
    Args:
        data_path: Directory containing parquet files
        max_files: Optional limit on number of files to process
    
    Returns:
        List of Path objects for parquet files
    """
    path = Path(data_path)
    files = sorted(path.glob("*.parquet"))
    
    if max_files is not None:
        files = files[:max_files]
    
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    logger.info(f"Found {len(files)} parquet file(s) in {data_path}")
    return files


# ============================================================================
# DATA LOADING
# ============================================================================

def collect_tendencies_samples(files: list, tendency_cols: list, 
                               max_samples: int, extra_cols: list = None) -> tuple:
    """
    Load tendency data and optional extra columns from parquet files.
    
    Args:
        files: List of parquet file paths
        tendency_cols: Column names for tendency variables
        max_samples: Maximum number of samples to load (<=0 or None means all samples)
        extra_cols: Additional columns to load (e.g., CLOUD, QC_TAU_in)
    
    Returns:
        (tendencies_dict, extras_dict) where each dict maps column name to numpy array
    """
    extra_cols = extra_cols or []
    all_cols = list(set(tendency_cols + extra_cols))
    
    # Initialize buffers for collecting data
    tend_buffers = {col: [] for col in tendency_cols}
    extra_buffers = {col: [] for col in extra_cols}
    
    collected = 0
    # Process all samples if max_samples is None or <= 0
    unlimited = (max_samples is None) or (max_samples <= 0)
    
    if unlimited:
        logger.info(f"Loading ALL samples from {len(files)} file(s)...")
    else:
        logger.info(f"Loading up to {max_samples:,} samples from {len(files)} file(s)...")
    
    for file_path in files:
        try:
            # Read parquet file in chunks to manage memory
            try:
                chunk_iter = pd.read_parquet(file_path, columns=all_cols, chunksize=200_000)
            except TypeError:
                # If chunksize not supported, read full file and split manually
                df = pd.read_parquet(file_path, columns=all_cols)
                chunk_size = 200_000
                chunk_iter = [df.iloc[i:i+chunk_size] 
                             for i in range(0, len(df), chunk_size)]
            
            # Process each chunk
            for chunk in chunk_iter:
                if (not unlimited) and collected >= max_samples:
                    break
                
                # Remove rows with NaN in tendency columns
                chunk = chunk.dropna(subset=tendency_cols)
                if chunk.empty:
                    continue
                
                # Determine how many samples to take from this chunk
                if unlimited:
                    sub = chunk
                else:
                    remaining = max_samples - collected
                    take = min(remaining, len(chunk))
                    if take < len(chunk):
                        # Random sampling if we need fewer than available
                        idx = np.random.choice(len(chunk), size=take, replace=False)
                        sub = chunk.iloc[idx]
                    else:
                        sub = chunk
                
                # Collect tendency columns
                for col in tendency_cols:
                    tend_buffers[col].append(sub[col].to_numpy())
                
                # Collect extra columns (if present)
                for col in extra_cols:
                    if col in sub.columns:
                        extra_buffers[col].append(sub[col].to_numpy())
                
                collected += len(sub)
                
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            continue
        
        if (not unlimited) and collected >= max_samples:
            break
    
    if unlimited:
        logger.info(f"Collected {collected:,} samples (all available)")
    else:
        logger.info(f"Collected {collected:,} samples (requested: {max_samples:,})")
    
    # Concatenate all buffers into single arrays
    tendencies = {
        col: np.concatenate(tend_buffers[col]) if tend_buffers[col] else np.array([])
        for col in tendency_cols
    }
    
    extras = {
        col: np.concatenate(extra_buffers[col]) if extra_buffers[col] else np.array([])
        for col in extra_cols
    }
    
    return tendencies, extras


# ============================================================================
# DATA TRANSFORMATION
# ============================================================================

def sign_log10(x: np.ndarray, eps: float = 1e-16) -> np.ndarray:
    """
    Apply signed log10 transformation: sign(x) * log10(|x| + eps)
    
    This preserves the sign of the data while compressing the range using log scale.
    The epsilon prevents log(0) and handles very small values.
    
    NOTE: This has since been changed to: log10(|x| + eps)
    Args:
        x: Input array
        eps: Small constant added before log to avoid log(0)
    
    Returns:
        Transformed array with same shape as input
    """
    x = np.asarray(x)
    #return np.sign(x) * np.log10(np.abs(x) + eps)
    return np.log10(np.abs(x) + eps)


def create_active_mask(qctend_raw: np.ndarray, cloud: np.ndarray,
                      active_threshold: float, cloud_threshold: float = 0.01,
                      qc_vals: np.ndarray = None, qr_vals: np.ndarray = None,
                      mass_threshold: float = None,
                      rho_vals: np.ndarray = None, rho_threshold: float = None) -> np.ndarray:
    """
    Create boolean mask identifying "active" microphysics samples.
    
    Active samples are defined as those with:
    - Significant tendency: |qctend_TAU| > active_threshold
    - Sufficient cloud: CLOUD > cloud_threshold
    - (Optional) Sufficient mass: QC_TAU_in > mass_threshold AND QR_TAU_in > mass_threshold
    
    Args:
        qctend_raw: Raw qctend_TAU values
        cloud: CLOUD values
        active_threshold: Minimum |qctend_TAU| to be considered active
        (Note: additional hard cap |qctend_TAU| < 1e-5 applied)
        cloud_threshold: Minimum CLOUD value (default 0.01)
        qc_vals: QC_TAU_in values (optional, for mass-augmented mask)
        qr_vals: QR_TAU_in values (optional, for mass-augmented mask)
        mass_threshold: Minimum QC/QR values (optional, for mass-augmented mask)
        rho_vals: RHO values (optional, applied alongside mass filter)
        rho_threshold: Minimum rho value (optional, applied if rho_vals provided)
    
    Returns:
        Boolean array where True = active sample
    """
    # Align all arrays to same length
    arrays = [qctend_raw, cloud]
    if qc_vals is not None and qr_vals is not None and mass_threshold is not None:
        arrays.extend([qc_vals, qr_vals])
    if rho_vals is not None and rho_threshold is not None:
        arrays.append(rho_vals)
    
    min_len = min(len(arr) for arr in arrays)
    qct = qctend_raw[:min_len]
    cld = cloud[:min_len]
    
    # Base mask: significant tendency within bounds AND sufficient cloud
    # Note: chained comparisons with arrays are invalid; perform elementwise comparisons separately
    #mask = (np.abs(qct) > active_threshold) & (np.abs(qct) < 5e-6) & (cld > cloud_threshold)
    mask = (
        (np.abs(qct) > active_threshold) &
        (np.abs(qct) < 1e-5) &
        (cld > cloud_threshold)
    )

    # Add mass thresholds if provided
    if qc_vals is not None and qr_vals is not None and mass_threshold is not None:
        qc = qc_vals[:min_len]
        qr = qr_vals[:min_len]
        mask = mask & (qc > mass_threshold) & (qr > mass_threshold)
    if rho_vals is not None and rho_threshold is not None:
        rho = rho_vals[:min_len]
        mask = mask & (rho > rho_threshold)
    
    n_active = np.sum(mask)
    pct_active = 100 * (n_active / min_len)
    logger.info(f"Active samples: {n_active} / {min_len} ({pct_active:.2f}%)")
    
    return mask


def apply_mask_and_transform(raw_data: dict, tendency_cols: list,
                             mask: np.ndarray, eps: float) -> dict:
    """
    Apply boolean mask to raw data and transform with sign_log10.
    
    Args:
        raw_data: Dictionary of raw tendency arrays
        tendency_cols: List of column names to process
        mask: Boolean mask (True = keep sample)
        eps: Epsilon for log transformation
    
    Returns:
        Dictionary of masked and log-transformed arrays
    """
    mask_len = len(mask)
    transformed = {}
    
    for col in tendency_cols:
        arr = raw_data[col]
        
        if arr.size == 0:
            transformed[col] = arr
            continue
        
        # Align array to mask length and apply mask
        arr_aligned = arr[:mask_len]
        arr_filtered = arr_aligned[mask]
        
        # Apply log transformation
        if arr_filtered.size > 0:
            transformed[col] = sign_log10(arr_filtered, eps=eps)
        else:
            transformed[col] = arr_filtered
    
    return transformed


# ============================================================================
# STATISTICS AND VISUALIZATION
# ============================================================================

def print_statistics(data: np.ndarray, label: str) -> None:
    """
    Print comprehensive statistics for a data array.
    
    Args:
        data: NumPy array to analyze
        label: Descriptive name for the data
    """
    if data.size == 0:
        logger.warning(f"{label}: No data available")
        return
    import scipy.stats as stats
    # Compute statistics
    stats = {
        'max': np.max(data),
        'min': np.min(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data, keepdims=False).mode,
        'std': np.std(data),
        'zeros_pct': 100 * np.sum(data == 0) / len(data),
        'p25': np.percentile(data, 25),
        'p50': np.percentile(data, 50),
        'p75': np.percentile(data, 75),
        'p95': np.percentile(data, 95),
        'p99': np.percentile(data, 99)
    }
    
    # Print in readable format
    print(f"\n{label}:")
    print(f"  Range: [{stats['min']:.3e} to {stats['max']:.3e}]")
    print(f"  Mean: {stats['mean']:.3e}, Std: {stats['std']:.3e}")
    print(f"  Median: {stats['median']:.3e}, Mode: {stats['mode']:.6e}")
    print(f"  Zeros: {stats['zeros_pct']:.2f}%")
    print(f"  Percentiles - 25th: {stats['p25']:.3e}, 50th: {stats['p50']:.3e}, "
          f"75th: {stats['p75']:.3e}, 95th: {stats['p95']:.3e}, 99th: {stats['p99']:.3e}")


def plot_hist_grid(data_dict: dict, title: str, bins: int, fig_path: Path) -> None:
    """
    Create grid of histograms for multiple variables.
    
    Args:
        data_dict: Dictionary mapping variable names to data arrays
        title: Overall figure title
        bins: Number of histogram bins
        fig_path: Path to save figure
    """
    names = list(data_dict.keys())
    n_vars = len(names)
    
    if n_vars == 0:
        logger.warning("No variables to plot")
        return
    
    # Determine grid layout (roughly square)
    n_cols = int(math.ceil(math.sqrt(n_vars)))
    n_rows = int(math.ceil(n_vars / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols + 1, 3.4 * n_rows + 1)
    )
    
    # Ensure axes is always an array
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each variable
    for i, name in enumerate(names):
        ax = axes[i]
        arr = np.asarray(data_dict[name]).astype(float)
        arr = arr[np.isfinite(arr)]  # Remove NaN/Inf
        
        if arr.size == 0:
            ax.set_title(f"{name} (no data)")
            ax.axis('off')
            continue
        
        # Create histogram
        ax.hist(arr, bins=bins, color='steelblue', alpha=0.85, density=False)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    
    # Hide unused subplots
    for j in range(n_vars, len(axes)):
        axes[j].axis('off')
    
    # Format and save
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    
    logger.info(f"Saved plot: {fig_path}")


# ----------------------------------------------------------------------------
# OVERLAID HISTOGRAMS FOR TWO REGIMES
# ----------------------------------------------------------------------------
def plot_overlaid_hist_two(data_a: np.ndarray, data_b: np.ndarray,
                           labels: tuple, title: str, fig_path: Path,
                           bins: int = 100) -> None:
    """
    Plot two overlaid histograms (same bin edges) for easy comparison.

    Args:
        data_a: 1D array for regime A
        data_b: 1D array for regime B
        labels: (label_a, label_b)
        title: Figure title
        fig_path: Path to save the figure
        bins: Number of histogram bins
    """
    arr_a = np.asarray(data_a)
    arr_b = np.asarray(data_b)

    # Remove non-finite values
    arr_a = arr_a[np.isfinite(arr_a)]
    arr_b = arr_b[np.isfinite(arr_b)]

    if arr_a.size == 0 and arr_b.size == 0:
        logger.warning(f"No finite data to plot for: {title}")
        return

    # Use combined data to define shared bin edges
    combined = np.concatenate([arr_a, arr_b]) if arr_a.size and arr_b.size else (arr_a if arr_a.size else arr_b)
    _, edges = np.histogram(combined, bins=bins)

    fig = plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    if arr_a.size:
        ax.hist(arr_a, bins=edges, alpha=0.6, color='tab:blue', label=labels[0], edgecolor='black')
    if arr_b.size:
        ax.hist(arr_b, bins=edges, alpha=0.6, color='tab:orange', label=labels[1], edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved plot: {fig_path}")

# ============================================================================
# SPATIAL HEATMAPS (TAIL EVENTS)
# ============================================================================

def ensure_lon_0_360(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lon = np.where(lon < 0, lon + 360.0, lon)
    lon = np.where(lon >= 360.0, lon - 360.0, lon)
    return lon


def compute_fraction_heatmap_latlon(lat_vals: np.ndarray, lon_vals: np.ndarray,
                                    tail_mask: np.ndarray, base_mask: np.ndarray,
                                    bin_deg: int = 5) -> tuple:
    """
    Compute lat-lon fraction heatmap of tail events relative to base mask counts.

    Returns:
        (fraction, lat_bins, lon_bins)
    """
    lat_vals = np.asarray(lat_vals)[:len(base_mask)][base_mask]
    lon_vals = np.asarray(lon_vals)[:len(base_mask)][base_mask]
    lon_vals = ensure_lon_0_360(lon_vals)

    tail_weights = tail_mask[:len(base_mask)][base_mask].astype(int)

    lat_bins = np.arange(-90, 91, bin_deg)
    lon_bins = np.arange(0, 361, bin_deg)

    # Total samples per cell
    total_counts, _, _ = np.histogram2d(lat_vals, lon_vals, bins=[lat_bins, lon_bins])
    # Tail samples per cell
    tail_counts, _, _ = np.histogram2d(lat_vals, lon_vals, bins=[lat_bins, lon_bins],
                                       weights=tail_weights)

    with np.errstate(invalid='ignore', divide='ignore'):
        fraction = np.divide(tail_counts, total_counts)
    fraction[~np.isfinite(fraction)] = np.nan
    return fraction, lat_bins, lon_bins


def plot_latlon_heatmap(fraction: np.ndarray, lat_bins: np.ndarray, lon_bins: np.ndarray,
                        title: str, fig_path: Path) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    lon_grid, lat_grid = np.meshgrid(lon_bins[:-1], lat_bins[:-1])
    im = ax.pcolormesh(lon_grid, lat_grid, fraction, cmap='inferno', shading='auto')
    plt.colorbar(im, ax=ax, label='Fraction')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap: {fig_path}")


def plot_zonal_mean_heatmap(first_fraction: np.ndarray, last_fraction: np.ndarray,
                            lat_bins: np.ndarray, title: str, fig_path: Path) -> None:
    """
    Create a heatmap with rows = latitude bins, columns = ['first_bin','last_bin'],
    values = zonal mean (mean across longitudes) of fraction.
    """
    # Zonal mean across longitude (axis=1 because lat is rows, lon is cols)
    first_zonal = np.nanmean(first_fraction, axis=1)
    last_zonal = np.nanmean(last_fraction, axis=1)

    data = np.vstack([first_zonal, last_zonal]).T  # shape: (n_lat_bins-1, 2)
    fig = plt.figure(figsize=(6, 8))
    ax = plt.gca()
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Zonal mean fraction')
    ax.set_yticks(np.arange(len(lat_bins)-1))
    # Label y-ticks by bin centers
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    ax.set_yticklabels([f"{lc:.0f}" for lc in lat_centers])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['first_bin', 'last_bin'])
    ax.set_xlabel('Tail bin')
    ax.set_ylabel('Latitude (deg)')
    ax.set_title(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved zonal-mean heatmap: {fig_path}")

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot histograms of raw and transformed tendency data from parquet files"
    )
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Limit number of parquet files to process')
    parser.add_argument('--max-samples', type=int, default=-1,
                       help='Number of samples to load; <=0 means all samples')
    parser.add_argument('--bins', type=int, default=100,
                       help='Number of histogram bins')
    parser.add_argument('--output-dir', type=str, default='plots/tendencies_raw',
                       help='Directory to save output plots')
    parser.add_argument('--active-threshold', type=float, default=1e-15,
                       help='Threshold for |qctend_TAU| to define active samples')
    parser.add_argument('--eps', type=float, default=1e-16,
                       help='Epsilon for log10 transformation to avoid log(0)')
    parser.add_argument('--mass-input-threshold', type=float, default=1e-5,
                       help='Threshold for QC_TAU_in and QR_TAU_in in mass-augmented active mask')
    # Spatial tail analysis
    parser.add_argument('--tail-heatmaps', action='store_true',
                       help='Generate spatial heatmaps (lat-lon fraction and zonal mean) for first and last bins')
    parser.add_argument('--tail-heatmap-var', type=str, default='qctend_TAU',
                       help='Variable (from output_cols) used to define first/last bin events')
    parser.add_argument('--tail-heatmap-bin-deg', type=int, default=5,
                       help='Latitude/Longitude bin size in degrees')
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    cfg = load_config(args.config)
    data_cfg = cfg.get('data', {})
    
    # Extract configuration parameters
    data_path = data_cfg.get('data_path')
    tendency_cols = data_cfg.get('output_cols', [])
    input_cols = data_cfg.get('input_cols', [])
    active_threshold = args.active_threshold
    
    logger.info(f"Tendency columns: {tendency_cols}")
    logger.info(f"Active threshold: {active_threshold}")
    logger.info(f"Log transform epsilon: {args.eps}")
    
    # Find parquet files
    files = list_parquet_files(data_path, max_files=args.max_files)

    # Output directory
    output_dir = f"./plots/histograms_MaxSamples{args.max_samples}_ActiveThresh{active_threshold:g}_MassThresh{args.mass_input_threshold:g}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir
    logger.info(f"Output directory set to: {args.output_dir}")
    
    # ========================================
    # STEP 1: Load raw data
    # ========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading raw tendency data")
    logger.info("=" * 60)
    
    raw, extras = collect_tendencies_samples(
        files, tendency_cols, args.max_samples, extra_cols=input_cols
    )
    
    # Plot raw tendencies
    plot_hist_grid(
        raw,
        title='Raw Tendencies (from parquet files)',
        bins=args.bins,
        fig_path=Path(args.output_dir) / f'raw_tendencies.png'
    )
    
    # Print statistics for raw data
    print("\n" + "=" * 60)
    print("RAW TENDENCY STATISTICS")
    print("=" * 60)
    for col in tendency_cols:
        print_statistics(raw[col], col)
    
    # ========================================
    # STEP 2: Log10-transformed data
    # ========================================
    logger.info("=" * 60)
    logger.info("STEP 2: Applying log10 transformation")
    logger.info("=" * 60)
    
    logt = {
        col: sign_log10(raw[col], eps=args.eps) if raw[col].size else raw[col]
        for col in tendency_cols
    }
    
    # Plot log-transformed tendencies
    plot_hist_grid(
        logt,
        title=f'Log10-Transformed Tendencies (log10(|x| + {args.eps}))',
        bins=args.bins,
        fig_path=Path(args.output_dir) / f'log10_tendencies.png'
    )
    
    # Print statistics for log-transformed data
    print("\n" + "=" * 60)
    print("LOG10-TRANSFORMED TENDENCY STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(logt['qctend_TAU'])}")
    for col in tendency_cols:
        print_statistics(logt[col], f"log10({col})")
    
    # ========================================
    # STEP 3: Active samples ((Filtering based on qctend and CLOUD))
    # ========================================
    logger.info("=" * 60)
    logger.info("STEP 3: Filtering for ACTIVE samples ((Filtering based on qctend and CLOUD))")
    logger.info("=" * 60)
    
    qct_raw = raw['qctend_TAU']
    cloud = extras.get('CLOUD', np.array([]))
    
    # Create basic active mask
    active_mask_basic = create_active_mask(
        qct_raw, cloud, active_threshold, cloud_threshold=0.01
    )
    
    # Apply mask and transform
    logt_active = apply_mask_and_transform(
        raw, tendency_cols, active_mask_basic, args.eps
    )
    
    # Plot active-only tendencies
    plot_hist_grid(
        logt_active,
        title=f'Log10 Tendencies - ACTIVE Samples (|qctend_TAU| > {active_threshold:g}, CLOUD > 0.01)',
        bins=args.bins,
        fig_path=Path(args.output_dir) / f'log10_tendencies_active.png'
    )
    
    # Print statistics for active samples
    print("\n" + "=" * 60)
    print("ACTIVE SAMPLES STATISTICS ((Filtering based on qctend and CLOUD))")
    print("=" * 60)
    for col in tendency_cols:
        print_statistics(logt_active[col], f"log10({col}) [active]")
    
    # ========================================
    # STEP 4: Active samples (mass-augmented mask)
    # ========================================
    logger.info("=" * 60)
    logger.info("STEP 4: Filtering for ACTIVE samples (mass-augmented mask)")
    logger.info("=" * 60)
    
    qc_vals = extras.get('QC_TAU_in', np.array([]))
    qr_vals = extras.get('QR_TAU_in', np.array([]))
    rho_vals = extras.get('RHO_CLUBB', np.array([]))
    
    if qc_vals.size == 0 or qr_vals.size == 0:
        logger.warning("QC_TAU_in or QR_TAU_in not found - skipping mass-augmented mask")
    else:
        # Create mass-augmented active mask
        active_mask_mass = create_active_mask(
            qct_raw, cloud, active_threshold, cloud_threshold=0.01,
            qc_vals=qc_vals, qr_vals=qr_vals, mass_threshold=args.mass_input_threshold,
            rho_vals=rho_vals if rho_vals.size else None,
            rho_threshold=0.2 if rho_vals.size else None
        )

        # Additional filter: |nrtend_TAU| > 1e-10 (if available)
        if 'nrtend_TAU' in raw:
            nrt = raw['nrtend_TAU'][:len(active_mask_mass)]
            nrt_cond = np.abs(nrt) > 1e-10
            active_mask_mass = active_mask_mass & nrt_cond
            logger.info("Applied additional filter: |nrtend_TAU| > 1e-10 to mass-augmented mask")
        else:
            logger.warning("'nrtend_TAU' not found in outputs; skipping |nrtend_TAU|>1e-10 filter")
        
        # Apply mask and transform
        logt_active_mass = apply_mask_and_transform(
            raw, tendency_cols, active_mask_mass, args.eps
        )
        
        # Plot mass-augmented active tendencies
        plot_hist_grid(
            logt_active_mass,
            title=f'Log10 Tendencies - ACTIVE+MASS (|qctend_TAU| > {active_threshold:g}, CLOUD>0.01, \n QC/QR>{args.mass_input_threshold:g}, |nrtend_TAU|>1e-10)',
            bins=args.bins,
            fig_path=Path(args.output_dir) / f'log10_tendencies_active_mass.png'
        )
        
        # Print statistics for mass-augmented active samples
        print("\n" + "=" * 60)
        print("ACTIVE SAMPLES STATISTICS (Mass-Augmented Mask with |nrtend_TAU|>1e-10)")
        print("=" * 60)
        for col in tendency_cols:
            print_statistics(logt_active_mass[col], f"log10({col}) [active+mass]")

        # ======================================================
        # STEP 4b: Input variable histograms by regimes
        # Regime 1:  qctend_TAU > -1e-5
        # Regime 2:  qctend_TAU < -1e-5
        # (Both within ACTIVE+MASS + |nrtend_TAU|>1e-10)
        # ======================================================
        try:
            logger.info("=" * 60)
            logger.info("STEP 4b: Input variable histograms by regimes (ACTIVE+MASS)")
            logger.info("=" * 60)

            # Align qctend values to mask length
            qct_full = qct_raw[:len(active_mask_mass)]
            regime1_mask = active_mask_mass & (qct_full > -1e-5)
            regime2_mask = active_mask_mass & (qct_full < -1e-5)

            out_dir_reg = Path(args.output_dir) / 'inputs_regimes_active_mass'
            out_dir_reg.mkdir(parents=True, exist_ok=True)

            if not input_cols:
                logger.warning("No input_cols provided in config; skipping regime input histograms")
            else:
                for col in input_cols:
                    if col not in extras:
                        logger.warning(f"Input column '{col}' not found in loaded inputs; skipping")
                        continue

                    vals = extras[col][:len(active_mask_mass)]
                    vals_reg1 = vals[regime1_mask]
                    vals_reg2 = vals[regime2_mask]

                    title = f"{col} - ACTIVE+MASS Regimes (threshold -1e-5 on qctend_TAU)"
                    fig_path = out_dir_reg / f"inputs_regimes_{col}.png"
                    plot_overlaid_hist_two(
                        vals_reg1, vals_reg2,
                        labels=("Regime 1: qctend_TAU > -1e-5", "Regime 2: qctend_TAU < -1e-5"),
                        title=title,
                        fig_path=fig_path,
                        bins=int(1.5*args.bins)
                    )
        except Exception as e:
            logger.warning(f"Error while generating regime-based input histograms: {e}")
    
    # ========================================
    # STEP 5: Tail spatial heatmaps (first and last bin) and zonal-mean
    # ========================================
    if args.tail_heatmaps:
        logger.info("=" * 60)
        logger.info("STEP 5: Spatial heatmaps for tail bins")
        logger.info("=" * 60)

        tail_var = args.tail_heatmap_var
        if tail_var not in raw:
            logger.warning(f"tail-heatmap-var '{tail_var}' not found in outputs; skipping heatmaps")
        else:
            # Prefer mass-augmented active mask if built above
            # If not present, fall back to basic active, else all samples
            analysis_mask = None
            try:
                analysis_mask = active_mask_mass
                logger.info("Heatmaps: using ACTIVE+MASS mask")
            except NameError:
                try:
                    analysis_mask = active_mask_basic
                    logger.info("Heatmaps: using ACTIVE mask")
                except NameError:
                    analysis_mask = np.ones_like(raw[tail_var], dtype=bool)
                    logger.info("Heatmaps: using ALL samples")

            # lat/lon must be present in extras
            if 'lat' not in extras or 'lon' not in extras:
                logger.warning("lat/lon not found in loaded inputs; skipping heatmaps")
            else:
                lat_vals = extras['lat']
                lon_vals = extras['lon']

                # Compute histogram for tail_var under analysis mask
                vals = raw[tail_var][:len(analysis_mask)][analysis_mask]
                counts, edges = np.histogram(vals[np.isfinite(vals)], bins=args.bins)
                if len(edges) < 3:
                    logger.warning("Insufficient edges for tail heatmaps; skipping")
                else:
                    first_lo, first_hi = edges[0], edges[1]
                    last_lo, last_hi = edges[-2], edges[-1]

                    full_vals = raw[tail_var][:len(analysis_mask)]
                    first_mask = analysis_mask & (full_vals >= first_lo) & (full_vals < first_hi)
                    last_mask = analysis_mask & (full_vals >= last_lo) & (full_vals <= last_hi)

                    # Compute lat-lon fractions
                    first_frac, lat_bins, lon_bins = compute_fraction_heatmap_latlon(
                        lat_vals, lon_vals, first_mask, analysis_mask, bin_deg=args.tail_heatmap_bin_deg
                    )
                    last_frac, _, _ = compute_fraction_heatmap_latlon(
                        lat_vals, lon_vals, last_mask, analysis_mask, bin_deg=args.tail_heatmap_bin_deg
                    )

                    # Plot maps
                    plot_latlon_heatmap(
                        first_frac, lat_bins, lon_bins,
                        title=f"Fraction of events in FIRST bin of {tail_var}",
                        fig_path=Path(args.output_dir) / f"heatmap_firstbin_{tail_var}.png"
                    )
                    plot_latlon_heatmap(
                        last_frac, lat_bins, lon_bins,
                        title=f"Fraction of events in LAST bin of {tail_var}",
                        fig_path=Path(args.output_dir) / f"heatmap_lastbin_{tail_var}.png"
                    )

                    # Zonal means
                    plot_zonal_mean_heatmap(
                        first_frac, last_frac, lat_bins,
                        title=f"Zonal mean fraction: FIRST vs LAST bin ({tail_var})",
                        fig_path=Path(args.output_dir) / f"zonal_mean_tailbins_{tail_var}.png"
                    )

    # ========================================
    # Done
    # ========================================
    logger.info("=" * 60)
    logger.info("COMPLETE - All plots saved to: " + str(Path(args.output_dir)))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

