#!/usr/bin/env python3
"""
Describe temperature and pressure(level) variables in parquet dataset.

Computes min, max, mean, and median for specified columns across many files
without loading everything into memory. Columns can be auto-detected via
name patterns or specified explicitly via CLI arguments.

Example:
  python -u mlmicrophysics/pytorch_emulator/scripts/describe_temp_pressure.py \
    --data_path /rcfs/projects/pioneercloud/dhruv/processed_data \
    --temp-cols T T_in temperature \
    --pressure-cols PRES_LVL pressure_lvl

If column names are unknown, omit --temp-cols/--pressure-cols and the script
will try to auto-detect common variants.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


class StreamingStats:
    """Streaming statistics with reservoir sampling for approximate median."""

    def __init__(self, reservoir_size: int = 1_000_000):
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.sum_val: float = 0.0
        self.count: int = 0
        self.reservoir_size = int(reservoir_size)
        self._reservoir: List[float] = []
        self._seen: int = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        # Remove NaNs/Infs
        values = values[np.isfinite(values)]
        if values.size == 0:
            return

        vmin = float(np.min(values))
        vmax = float(np.max(values))
        vsum = float(np.sum(values))
        vcount = int(values.size)

        self.min_val = vmin if self.min_val is None else min(self.min_val, vmin)
        self.max_val = vmax if self.max_val is None else max(self.max_val, vmax)
        self.sum_val += vsum
        self.count += vcount

        # Reservoir sampling (Algorithm R)
        if self.reservoir_size <= 0:
            return
        for val in values:
            self._seen += 1
            if len(self._reservoir) < self.reservoir_size:
                self._reservoir.append(float(val))
            else:
                j = np.random.randint(0, self._seen)
                if j < self.reservoir_size:
                    self._reservoir[j] = float(val)

    def mean(self) -> Optional[float]:
        return None if self.count == 0 else self.sum_val / self.count

    def median(self) -> Optional[float]:
        if not self._reservoir:
            return None
        return float(np.median(np.asarray(self._reservoir, dtype=np.float64)))

    def summary(self) -> Dict[str, Optional[float]]:
        return {
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean(),
            "median": self.median(),
            "count": float(self.count),
        }


def discover_columns(
    data_path: Path,
    preferred_temp: List[str],
    preferred_press: List[str],
) -> Tuple[List[str], List[str]]:
    """Inspect a file to guess temperature and pressure(level) columns.

    Returns two lists (temp_cols, pressure_cols). Lists may be empty if nothing found.
    """
    parquet_files = sorted(data_path.glob("*.parquet"))
    if not parquet_files:
        return [], []

    # Try to get columns via pyarrow schema first (no full read)
    cols: List[str] = []
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(parquet_files[0])
        cols = [n for n in pf.schema.names]
    except Exception:
        # Fallback: read small frame
        try:
            df_head = pd.read_parquet(parquet_files[0])
            cols = list(df_head.columns)
        except Exception:
            cols = []

    lower_cols = {c.lower(): c for c in cols}

    def match_any(candidates: List[str]) -> List[str]:
        found: List[str] = []
        for cand in candidates:
            key = cand.lower()
            if key in lower_cols:
                found.append(lower_cols[key])
        # If strict names not found, try pattern-based search
        if not found:
            patterns = [
                key for key in candidates if len(key) > 2
            ]
            for lc, orig in lower_cols.items():
                if any(p in lc for p in patterns):
                    found.append(orig)
        return sorted(set(found))

    temp_found = match_any(preferred_temp)
    press_found = match_any(preferred_press)
    return temp_found, press_found


def iter_parquet_chunks(
    file_path: Path,
    columns: List[str],
    chunk_size: int,
) -> Iterable[pd.DataFrame]:
    """Yield DataFrame chunks for selected columns, with robust fallbacks."""
    try:
        # pyarrow engine supports chunksize
        it = pd.read_parquet(str(file_path), columns=columns, chunksize=chunk_size)
        for chunk in it:
            yield chunk
        return
    except TypeError:
        # chunksize not supported by current engine: fallback to manual slicing
        df = pd.read_parquet(str(file_path), columns=columns)
        if len(df) == 0:
            return
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i : i + chunk_size]


def compute_stats(
    data_path: Path,
    target_cols: List[str],
    max_files: Optional[int],
    chunk_size: int,
    reservoir_size: int,
    sample_fraction: float,
) -> Dict[str, StreamingStats]:
    stats: Dict[str, StreamingStats] = {c: StreamingStats(reservoir_size) for c in target_cols}
    files = sorted(data_path.glob("*.parquet"))
    if max_files is not None:
        files = files[:max_files]
    for fp in files:
        try:
            for chunk in iter_parquet_chunks(fp, target_cols, chunk_size):
                if sample_fraction < 1.0:
                    n = int(len(chunk) * sample_fraction)
                    if n <= 0:
                        continue
                    chunk = chunk.sample(n=n, random_state=42)
                for col in target_cols:
                    if col in chunk.columns:
                        values = chunk[col].to_numpy()
                        stats[col].update(values)
        except Exception as e:
            print(f"Warning: failed to process {fp.name}: {e}")
            continue
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe temperature and pressure(level) columns.")
    parser.add_argument("--data_path", type=str, required=True, help="Directory with parquet files")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files to scan")
    parser.add_argument("--chunk_size", type=int, default=200_000, help="Rows per chunk read")
    parser.add_argument("--reservoir_size", type=int, default=1_000_000, help="Sample size for median estimate")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help="Row subsample fraction in each chunk (0-1]")
    parser.add_argument("--temp-cols", nargs="*", default=[], help="Explicit temperature column names")
    parser.add_argument("--pressure-cols", nargs="*", default=[], help="Explicit pressure(level) column names")

    # Common variants to try if not specified
    parser.add_argument(
        "--temp-candidates",
        nargs="*",
        default=[
            "T",
            "T_in",
            "TEMP",
            "TEMPERATURE",
            "temperature",
            "temp",
        ],
        help="Candidate names for temperature auto-detection",
    )
    parser.add_argument(
        "--pressure-candidates",
        nargs="*",
        default=[
            "PRES_LVL",
            "P_LVL",
            "pressure_lvl",
            "pressure_level",
            "PRES",
            "PRESSURE",
            "p_in",
            "P",
        ],
        help="Candidate names for pressure(level) auto-detection",
    )

    args = parser.parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Data path not found: {data_path}")

    # Determine columns
    temp_cols: List[str] = list(args.temp_cols)
    press_cols: List[str] = list(args.pressure_cols)
    if not temp_cols or not press_cols:
        auto_temp, auto_press = discover_columns(data_path, args.temp_candidates, args.pressure_candidates)
        if not temp_cols:
            temp_cols = auto_temp
        if not press_cols:
            press_cols = auto_press

    if not temp_cols and not press_cols:
        raise SystemExit("Could not identify temperature or pressure(level) columns. Specify via --temp-cols/--pressure-cols.")

    target_cols = list(dict.fromkeys([*temp_cols, *press_cols]))
    print("Columns to analyze:")
    print(f"  Temperature: {temp_cols or '[]'}")
    print(f"  Pressure(level): {press_cols or '[]'}")

    stats = compute_stats(
        data_path=data_path,
        target_cols=target_cols,
        max_files=args.max_files,
        chunk_size=args.chunk_size,
        reservoir_size=args.reservoir_size,
        sample_fraction=float(args.sample_fraction),
    )

    # Print summary
    def fmt(x: Optional[float]) -> str:
        return "na" if x is None else (f"{x:.6g}")

    print("\nSummary statistics:")
    for col in target_cols:
        s = stats[col].summary()
        print(f"- {col}:")
        print(f"    count:  {int(s['count']) if s['count'] is not None else 0}")
        print(f"    min:    {fmt(s['min'])}")
        print(f"    max:    {fmt(s['max'])}")
        print(f"    mean:   {fmt(s['mean'])}")
        print(f"    median: {fmt(s['median'])}")


if __name__ == "__main__":
    main()




