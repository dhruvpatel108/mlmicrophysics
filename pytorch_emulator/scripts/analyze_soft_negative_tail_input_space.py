#!/usr/bin/env python3
"""
Map residual soft negative-tail samples into input and metadata space.

This script is meant to complement analyze_soft_negative_tail.py by building
visual overlays between:
- the residual negative-tail samples that dominate the soft-routing error, and
- an ordinary negative-reference set from the same validation split

Outputs:
- input_space_tail_overlay.png
- metadata_tail_overlay.png
- reference_negative_samples.csv
- feature_pair_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


TAIL_BIN_ORDER = ("[1.0e+04, 3.0e+04)", "[3.0e+04, 1.0e+05)", "[1.0e+05, inf)")
TAIL_BIN_COLORS = {
    "[1.0e+04, 3.0e+04)": "#F58518",
    "[3.0e+04, 1.0e+05)": "#54A24B",
    "[1.0e+05, inf)": "#E45756",
}
REFERENCE_COLOR = "#B8B8B8"

INPUT_PAIRS = (
    ("input_raw__NR_TAU_in", "input_raw__N0R"),
    ("input_raw__NR_TAU_in", "input_raw__QR_TAU_in"),
    ("input_raw__NR_TAU_in", "input_raw__LAMC"),
    ("input_raw__NR_TAU_in", "input_raw__CLOUD"),
)

PAIR_AXIS_SCALES = {
    ("input_raw__NR_TAU_in", "input_raw__N0R"): ("log", "log"),
    ("input_raw__NR_TAU_in", "input_raw__QR_TAU_in"): ("log", "log"),
    ("input_raw__NR_TAU_in", "input_raw__LAMC"): ("log", "log"),
    ("input_raw__NR_TAU_in", "input_raw__CLOUD"): ("log", "linear"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map residual soft negative-tail samples in input space")
    parser.add_argument(
        "--tail_samples_csv",
        type=Path,
        required=True,
        help="CSV produced by analyze_soft_negative_tail.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to config_used.yml",
    )
    parser.add_argument(
        "--validation_files_txt",
        type=Path,
        required=True,
        help="Validation file list used by the tail analysis",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for plots and summaries",
    )
    parser.add_argument(
        "--max_reference_samples",
        type=int,
        default=60000,
        help="Approximate total number of negative-reference samples to draw",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip_reference_sampling",
        action="store_true",
        help="Generate tail-only maps without sampling negative-reference rows",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def sample_reference_negatives(
    data_dir: Path,
    validation_files: Sequence[str],
    per_file_quota: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sampled_frames: List[pd.DataFrame] = []
    keep_cols = [
        "QC_TAU_in",
        "QR_TAU_in",
        "NC_TAU_in",
        "NR_TAU_in",
        "PGAM",
        "LAMC",
        "LAMR",
        "N0R",
        "RHO_CLUBB",
        "CLOUD",
        "FREQR",
        "nrtend_TAU",
        "time",
        "ncol",
        "lev",
    ]

    for file_name in validation_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            continue

        available_cols = set(pq.ParquetFile(file_path).schema.names)
        cols = [c for c in keep_cols if c in available_cols]
        if "nrtend_TAU" not in cols:
            continue
        df = pd.read_parquet(
            file_path,
            columns=cols,
            filters=[
                ("QC_TAU_in", ">", 1.0e-6),
                ("CLOUD", ">", 0.01),
                ("nrtend_TAU", "<", 0.0),
                ("nrtend_TAU", ">", -1.0e4),
            ],
        )
        ref = df.copy()
        if len(ref) == 0:
            continue

        n_take = min(per_file_quota, len(ref))
        if len(ref) > n_take:
            idx = rng.choice(len(ref), size=n_take, replace=False)
            ref = ref.iloc[idx].copy()

        ref["group_label"] = "negative_reference"
        sampled_frames.append(ref)

    if not sampled_frames:
        return pd.DataFrame()

    result = pd.concat(sampled_frames, ignore_index=True)
    result = result.rename(columns={col: f"input_raw__{col}" for col in [
        "QC_TAU_in", "QR_TAU_in", "NC_TAU_in", "NR_TAU_in", "PGAM",
        "LAMC", "LAMR", "N0R", "RHO_CLUBB", "CLOUD", "FREQR",
    ]})
    result = result.rename(columns={
        "time": "metadata__time",
        "ncol": "metadata__ncol",
        "lev": "metadata__lev",
        "nrtend_TAU": "nrtend_true_physical",
    })
    return result


def set_axis_scale(ax, axis: str, scale: str) -> None:
    if scale == "log":
        getattr(ax, f"set_{axis}scale")("log")


def make_input_space_overlay(
    ref_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, (x_col, y_col) in zip(axes.flatten(), INPUT_PAIRS):
        ref_sub = ref_df[(ref_df[x_col] > 0) & (ref_df[y_col] > 0)].copy()
        ax.scatter(
            ref_sub[x_col],
            ref_sub[y_col],
            s=6,
            alpha=0.10,
            c=REFERENCE_COLOR,
            linewidths=0,
            label="negative reference (<1e4)",
        )

        for bin_label in TAIL_BIN_ORDER:
            sub = tail_df[tail_df["tail_bin_label"] == bin_label]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=12,
                alpha=0.55,
                c=TAIL_BIN_COLORS[bin_label],
                linewidths=0,
                label=f"tail {bin_label}",
            )

        x_scale, y_scale = PAIR_AXIS_SCALES[(x_col, y_col)]
        set_axis_scale(ax, "x", x_scale)
        set_axis_scale(ax, "y", y_scale)
        ax.set_xlabel(x_col.replace("input_raw__", ""))
        ax.set_ylabel(y_col.replace("input_raw__", ""))
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Residual Negative-Tail Samples vs Negative Reference in Input Space", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_tail_only_input_space(
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for ax, (x_col, y_col) in zip(axes.flatten(), INPUT_PAIRS):
        for bin_label in TAIL_BIN_ORDER:
            sub = tail_df[tail_df["tail_bin_label"] == bin_label]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=10,
                alpha=0.55,
                c=TAIL_BIN_COLORS[bin_label],
                linewidths=0,
                label=f"tail {bin_label}",
            )

        x_scale, y_scale = PAIR_AXIS_SCALES[(x_col, y_col)]
        set_axis_scale(ax, "x", x_scale)
        set_axis_scale(ax, "y", y_scale)
        ax.set_xlabel(x_col.replace("input_raw__", ""))
        ax.set_ylabel(y_col.replace("input_raw__", ""))
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Residual Negative-Tail Samples in Input Space", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_metadata_overlay(
    ref_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ref_meta = ref_df.dropna(subset=["metadata__ncol", "metadata__lev", "metadata__time"])
    axes[0].scatter(
        ref_meta["metadata__ncol"],
        ref_meta["metadata__lev"],
        s=5,
        alpha=0.08,
        c=REFERENCE_COLOR,
        linewidths=0,
        label="negative reference",
    )
    axes[1].scatter(
        ref_meta["metadata__time"],
        ref_meta["metadata__lev"],
        s=5,
        alpha=0.08,
        c=REFERENCE_COLOR,
        linewidths=0,
        label="negative reference",
    )

    for bin_label in TAIL_BIN_ORDER:
        sub = tail_df[tail_df["tail_bin_label"] == bin_label].dropna(
            subset=["metadata__ncol", "metadata__lev", "metadata__time"]
        )
        if len(sub) == 0:
            continue
        axes[0].scatter(
            sub["metadata__ncol"],
            sub["metadata__lev"],
            s=12,
            alpha=0.6,
            c=TAIL_BIN_COLORS[bin_label],
            linewidths=0,
            label=f"tail {bin_label}",
        )
        axes[1].scatter(
            sub["metadata__time"],
            sub["metadata__lev"],
            s=12,
            alpha=0.6,
            c=TAIL_BIN_COLORS[bin_label],
            linewidths=0,
            label=f"tail {bin_label}",
        )

    axes[0].set_xlabel("ncol")
    axes[0].set_ylabel("lev")
    axes[0].set_title("ncol vs lev")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("time")
    axes[1].set_ylabel("lev")
    axes[1].set_title("time vs lev")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Residual Negative-Tail Samples vs Negative Reference in Metadata Space", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_tail_only_metadata(
    tail_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for bin_label in TAIL_BIN_ORDER:
        sub = tail_df[tail_df["tail_bin_label"] == bin_label].dropna(
            subset=["metadata__ncol", "metadata__lev", "metadata__time"]
        )
        if len(sub) == 0:
            continue
        axes[0].scatter(
            sub["metadata__ncol"],
            sub["metadata__lev"],
            s=10,
            alpha=0.55,
            c=TAIL_BIN_COLORS[bin_label],
            linewidths=0,
            label=f"tail {bin_label}",
        )
        axes[1].scatter(
            sub["metadata__time"],
            sub["metadata__lev"],
            s=10,
            alpha=0.55,
            c=TAIL_BIN_COLORS[bin_label],
            linewidths=0,
            label=f"tail {bin_label}",
        )

    axes[0].set_xlabel("ncol")
    axes[0].set_ylabel("lev")
    axes[0].set_title("ncol vs lev")
    axes[0].grid(alpha=0.25)

    axes[1].set_xlabel("time")
    axes[1].set_ylabel("lev")
    axes[1].set_title("time vs lev")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    fig.legend(
        uniq.values(),
        uniq.keys(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("Residual Negative-Tail Samples in Metadata Space", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_feature_pairs(ref_df: pd.DataFrame, tail_df: pd.DataFrame) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "reference_count": int(len(ref_df)),
        "tail_count": int(len(tail_df)),
        "tail_bin_counts": tail_df["tail_bin_label"].value_counts().to_dict(),
        "feature_pairs": {},
    }

    for x_col, y_col in INPUT_PAIRS:
        pair_key = f"{x_col.replace('input_raw__', '')}__vs__{y_col.replace('input_raw__', '')}"
        entry = {}
        ref_sub = ref_df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        entry["reference_p05"] = {
            "x": float(ref_sub[x_col].quantile(0.05)),
            "y": float(ref_sub[y_col].quantile(0.05)),
        }
        entry["reference_p95"] = {
            "x": float(ref_sub[x_col].quantile(0.95)),
            "y": float(ref_sub[y_col].quantile(0.95)),
        }

        bins = {}
        for bin_label in TAIL_BIN_ORDER:
            sub = tail_df[tail_df["tail_bin_label"] == bin_label][[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) == 0:
                continue
            bins[bin_label] = {
                "count": int(len(sub)),
                "median_x": float(sub[x_col].median()),
                "median_y": float(sub[y_col].median()),
                "fraction_outside_reference_p95_box": float(
                    np.mean(
                        (sub[x_col] > entry["reference_p95"]["x"])
                        | (sub[y_col] > entry["reference_p95"]["y"])
                        | (sub[x_col] < entry["reference_p05"]["x"])
                        | (sub[y_col] < entry["reference_p05"]["y"])
                    )
                ),
            }
        entry["tail_bins"] = bins
        summary["feature_pairs"][pair_key] = entry

    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    data_dir = Path(config["data"]["data_path"])
    validation_files = [line.strip() for line in args.validation_files_txt.read_text().splitlines() if line.strip()]
    per_file_quota = max(1, args.max_reference_samples // max(len(validation_files), 1))

    tail_df = pd.read_csv(args.tail_samples_csv)
    tail_df = tail_df[tail_df["oracle_tail_flag"].astype(bool)].copy()
    tail_df = tail_df[tail_df["tail_bin_label"].isin(TAIL_BIN_ORDER)].copy()

    make_tail_only_input_space(tail_df, args.output_dir / "tail_only_input_space.png")
    make_tail_only_metadata(tail_df, args.output_dir / "tail_only_metadata.png")

    if args.skip_reference_sampling:
        return

    ref_df = sample_reference_negatives(
        data_dir=data_dir,
        validation_files=validation_files,
        per_file_quota=per_file_quota,
        seed=args.seed,
    )
    if len(ref_df) == 0:
        raise RuntimeError("No negative reference samples were collected.")

    ref_df.to_csv(args.output_dir / "reference_negative_samples.csv", index=False)
    make_input_space_overlay(ref_df, tail_df, args.output_dir / "input_space_tail_overlay.png")
    make_metadata_overlay(ref_df, tail_df, args.output_dir / "metadata_tail_overlay.png")

    summary = summarize_feature_pairs(ref_df, tail_df)
    with (args.output_dir / "feature_pair_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
