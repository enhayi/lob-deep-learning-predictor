"""
Synthetic LOB Data Generator and Fast Parquet Benchmark Utility.
Generates realistic Level-2 Limit Order Book datasets with 10 price/volume levels,
realistic spread dynamics, and geometric price walks for testing and offline development.
"""

import logging
import os
import sys
import time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SyntheticLOBGenerator")


def generate_synthetic_lob_dataset(
    num_ticks: int = 15000,
    initial_mid_price: float = 65000.0,
    tick_size: float = 0.1,
    dt_ms: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic Level-2 Order Book snapshots with 10 bid and 10 ask levels.

    :param num_ticks: Total number of LOB updates to synthesize
    :param initial_mid_price: Starting mid-price (e.g. 65000 for BTC/USDT)
    :param tick_size: Minimum price tick size (e.g. 0.1)
    :param dt_ms: Timestamp increment in milliseconds (e.g. 100ms)
    :param seed: Random seed for reproducibility
    :return: Pandas DataFrame formatted to match exchange LOB schema
    """
    np.random.seed(seed)
    logger.info(f"Generating {num_ticks} synthetic LOB snapshots (initial_mid={initial_mid_price})...")

    # 1. Simulate Mid-Price using geometric random walk with mean reversion
    returns = np.random.normal(loc=0.0, scale=0.00015, size=num_ticks)
    # Add occasional microstructural jumps
    jumps = (np.random.rand(num_ticks) < 0.02) * np.random.normal(0, 0.0008, size=num_ticks)
    total_returns = returns + jumps

    mid_prices = initial_mid_price * np.exp(np.cumsum(total_returns))
    # Quantize to tick size
    mid_prices = np.round(mid_prices / tick_size) * tick_size

    # 2. Simulate Timestamps (100ms intervals starting from current epoch ms)
    start_time_ms = int(time.time() * 1000) - (num_ticks * dt_ms)
    timestamps = start_time_ms + np.arange(num_ticks) * dt_ms

    # 3. Simulate Spreads (1 to 4 ticks)
    half_spreads = np.random.choice([0.5, 1.0, 1.5], size=num_ticks, p=[0.7, 0.25, 0.05]) * tick_size

    best_bids = mid_prices - half_spreads
    best_asks = mid_prices + half_spreads

    # Ensure best ask > best bid
    best_asks = np.where(best_asks <= best_bids, best_bids + tick_size, best_asks)

    data_dict: Dict[str, Any] = {"timestamp": timestamps}

    # Generate 10 Asks
    for i in range(1, 11):
        # Price increases further into the book
        level_offset = (i - 1) * tick_size
        noise = np.random.exponential(scale=0.05, size=num_ticks) * tick_size
        ask_p = best_asks + level_offset + np.round(noise / tick_size) * tick_size
        # Volume profile typically follows gamma / lognormal distribution
        ask_v = np.random.gamma(shape=2.0, scale=0.5, size=num_ticks) * (1.0 + 0.1 * i)
        data_dict[f"ask_price_{i}"] = np.round(ask_p, 2)
        data_dict[f"ask_volume_{i}"] = np.round(ask_v, 4)

    # Generate 10 Bids
    for i in range(1, 11):
        # Price decreases further into the book
        level_offset = (i - 1) * tick_size
        noise = np.random.exponential(scale=0.05, size=num_ticks) * tick_size
        bid_p = best_bids - level_offset - np.round(noise / tick_size) * tick_size
        bid_v = np.random.gamma(shape=2.0, scale=0.5, size=num_ticks) * (1.0 + 0.1 * i)
        data_dict[f"bid_price_{i}"] = np.round(bid_p, 2)
        data_dict[f"bid_volume_{i}"] = np.round(bid_v, 4)

    df = pd.DataFrame(data_dict)
    logger.info(f"Generated DataFrame shape: {df.shape}")
    return df


def save_lob_to_parquet(df: pd.DataFrame, output_filepath: str) -> float:
    """
    Save DataFrame to Parquet format using PyArrow engine and Snappy compression.

    :return: Save elapsed time in seconds
    """
    os.makedirs(os.path.dirname(output_filepath) or ".", exist_ok=True)
    t0 = time.time()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_filepath, compression="snappy")
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_filepath) / (1024 * 1024)
    logger.info(
        f"Saved {len(df)} rows to '{output_filepath}' ({file_size_mb:.2f} MB) in {elapsed:.3f}s"
    )
    return elapsed


def verify_parquet_load_speed(filepath: str, max_expected_seconds: float = 2.0) -> Tuple[bool, float, int]:
    """
    Benchmark reading the Parquet file to verify it loads in under 2 seconds.

    :param filepath: Path to the Parquet file
    :param max_expected_seconds: Threshold in seconds (PRD requirement: < 2.0s)
    :return: (passed, elapsed_seconds, row_count)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found at {filepath}")

    t0 = time.time()
    table = pq.read_table(filepath)
    df = table.to_pandas()
    elapsed = time.time() - t0
    num_rows = len(df)

    passed = elapsed < max_expected_seconds
    status_str = "PASSED" if passed else "FAILED"
    logger.info(
        f"[{status_str}] Loaded {num_rows} rows from '{filepath}' in {elapsed:.4f}s "
        f"(Benchmark threshold: < {max_expected_seconds:.1f}s)"
    )
    return passed, elapsed, num_rows


if __name__ == "__main__":
    output_path = os.path.join("data", "raw_lob_data.parquet")
    df = generate_synthetic_lob_dataset(num_ticks=15000)
    save_lob_to_parquet(df, output_path)
    passed, load_time, rows = verify_parquet_load_speed(output_path)
    print(f"Benchmark result: Passed={passed}, LoadTime={load_time:.4f}s, Rows={rows}")
