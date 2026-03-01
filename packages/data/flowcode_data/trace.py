from __future__ import annotations

"""TRACE data loading utilities.

TRACE (Trade Reporting and Compliance Engine) is FINRA's system
for reporting OTC transactions in eligible fixed income securities.

This module provides functions to load and process TRACE data
from parquet files.
"""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_trace(
    path: str | Path,
    columns: list[str] | None = None,
    date_range: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load TRACE trade data from a parquet file.

    Parameters
    ----------
    path : str | Path
        Path to the parquet file containing TRACE data.
    columns : list[str] | None, optional
        Specific columns to load. If None, loads all columns.
    date_range : tuple[str, str] | None, optional
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format.
        If provided, filters data to this date range.

    Returns
    -------
    pd.DataFrame
        TRACE data with columns:
        - date: Trade date (datetime)
        - cusip: Bond identifier (str)
        - price: Execution price (float)
        - volume: Trade volume in thousands (float)
        - side: 'B' for buy, 'S' for sell (str)

        Index is reset (RangeIndex).

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.

    Notes
    -----
    TRACE data should be pre-cleaned to remove:
    - Cancelled trades (asof_cd = 'X')
    - Reversals (asof_cd = 'R')
    - Corrected trades (keep only latest correction)

    Examples
    --------
    >>> df = load_trace("data/trace_2023.parquet")
    >>> df.columns.tolist()
    ['date', 'cusip', 'price', 'volume', 'side']
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TRACE data file not found: {path}")

    # Load parquet
    if columns is not None:
        df = pd.read_parquet(path, columns=columns)
    else:
        df = pd.read_parquet(path)

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Apply date filter if specified
    if date_range is not None and "date" in df.columns:
        start, end = date_range
        mask = (df["date"] >= start) & (df["date"] <= end)
        df = df.loc[mask].copy()
        logger.info(f"Filtered to date range {start} - {end}: {len(df)} trades")

    # Reset index
    df = df.reset_index(drop=True)

    logger.info(f"Loaded {len(df)} TRACE trades from {path}")

    return df


def aggregate_daily_volume(
    trades: pd.DataFrame,
    cusip_col: str = "cusip",
    date_col: str = "date",
    volume_col: str = "volume",
    side_col: str = "side",
) -> pd.DataFrame:
    """
    Aggregate TRACE trades to daily buy/sell volume by CUSIP.

    Parameters
    ----------
    trades : pd.DataFrame
        TRACE trade data.
    cusip_col : str
        Column name for CUSIP identifier.
    date_col : str
        Column name for trade date.
    volume_col : str
        Column name for trade volume.
    side_col : str
        Column name for trade side ('B' or 'S').

    Returns
    -------
    pd.DataFrame
        Daily volume with columns:
        - date: Trade date
        - cusip: Bond identifier
        - buy_volume: Total buy volume
        - sell_volume: Total sell volume
        - total_volume: buy_volume + sell_volume

    Examples
    --------
    >>> daily = aggregate_daily_volume(trades)
    >>> daily.columns.tolist()
    ['date', 'cusip', 'buy_volume', 'sell_volume', 'total_volume']
    """
    # Separate buys and sells
    buys = trades[trades[side_col] == "B"]
    sells = trades[trades[side_col] == "S"]

    # Aggregate by date and cusip
    buy_agg = (
        buys.groupby([date_col, cusip_col])[volume_col]
        .sum()
        .reset_index()
        .rename(columns={volume_col: "buy_volume"})
    )

    sell_agg = (
        sells.groupby([date_col, cusip_col])[volume_col]
        .sum()
        .reset_index()
        .rename(columns={volume_col: "sell_volume"})
    )

    # Merge
    daily = pd.merge(
        buy_agg,
        sell_agg,
        on=[date_col, cusip_col],
        how="outer",
    )

    # Fill NaN with 0
    daily["buy_volume"] = daily["buy_volume"].fillna(0)
    daily["sell_volume"] = daily["sell_volume"].fillna(0)
    daily["total_volume"] = daily["buy_volume"] + daily["sell_volume"]

    return daily
