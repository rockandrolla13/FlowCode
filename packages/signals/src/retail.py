"""Retail order imbalance signal.

This module computes retail order imbalance from TRACE data
using the Quote Midpoint (QMP) classification rule.

Formula (Spec §1.2):
    I_t = (buy_volume - sell_volume) / (buy_volume + sell_volume)

QMP Classification:
    Buy if price > mid + threshold * spread
    Sell otherwise
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def qmp_classify(
    price: float,
    mid: float,
    spread: float,
    threshold: float = 0.1,
) -> Literal["buy", "sell"]:
    """
    Classify trade direction using Quote Midpoint (QMP) rule.

    The QMP rule classifies a trade as a buy if the execution price
    is above the midpoint plus a fraction of the spread.

    Parameters
    ----------
    price : float
        Execution price of the trade.
    mid : float
        Midpoint of bid-ask spread.
    spread : float
        Bid-ask spread (ask - bid).
    threshold : float, default 0.1
        Fraction of spread above midpoint for buy classification.

    Returns
    -------
    Literal["buy", "sell"]
        Trade direction classification.

    Examples
    --------
    >>> qmp_classify(price=100.5, mid=100.0, spread=1.0, threshold=0.1)
    'buy'
    >>> qmp_classify(price=99.5, mid=100.0, spread=1.0, threshold=0.1)
    'sell'

    Notes
    -----
    This implements a simplified QMP rule. The threshold of 0.1 means
    trades are classified as buys if price > mid + 0.1 * spread.
    """
    cutoff = mid + threshold * spread
    return "buy" if price > cutoff else "sell"


def classify_trades_qmp(
    trades: pd.DataFrame,
    price_col: str = "price",
    mid_col: str = "mid",
    spread_col: str = "spread",
    threshold: float = 0.1,
) -> pd.Series:
    """
    Classify all trades in a DataFrame using QMP rule.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with price, mid, and spread columns.
    price_col : str
        Column name for execution price.
    mid_col : str
        Column name for midpoint.
    spread_col : str
        Column name for spread.
    threshold : float
        QMP threshold parameter.

    Returns
    -------
    pd.Series
        Series of 'buy' or 'sell' classifications.
    """
    cutoff = trades[mid_col] + threshold * trades[spread_col]
    return pd.Series(
        np.where(trades[price_col] > cutoff, "buy", "sell"),
        index=trades.index,
    )


def compute_retail_imbalance(
    trades: pd.DataFrame,
    date_col: str = "date",
    cusip_col: str = "cusip",
    volume_col: str = "volume",
    side_col: str = "side",
    buy_value: str = "B",
    sell_value: str = "S",
) -> pd.Series:
    """
    Compute retail order imbalance from trade data.

    Imbalance is defined as:
        I_t = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with date, cusip, volume, and side columns.
    date_col : str
        Column name for trade date.
    cusip_col : str
        Column name for bond identifier.
    volume_col : str
        Column name for trade volume.
    side_col : str
        Column name for trade side (buy/sell indicator).
    buy_value : str
        Value in side_col that indicates a buy trade.
    sell_value : str
        Value in side_col that indicates a sell trade.

    Returns
    -------
    pd.Series
        Imbalance series with MultiIndex (date, cusip).
        Values range from -1 (all sells) to +1 (all buys).
        NaN if total volume is zero.

    Examples
    --------
    >>> imbalance = compute_retail_imbalance(trades)
    >>> imbalance.index.names
    ['date', 'cusip']
    >>> imbalance.loc[('2023-01-15', '037833100')]
    0.25

    Notes
    -----
    Zero total volume results in NaN (not 0) to distinguish from
    balanced flow where buy_volume == sell_volume.
    """
    # Validate required columns
    required_cols = [date_col, cusip_col, volume_col, side_col]
    missing = set(required_cols) - set(trades.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Separate buys and sells
    buys = trades[trades[side_col] == buy_value]
    sells = trades[trades[side_col] == sell_value]

    # Aggregate by date and cusip
    buy_agg = (
        buys.groupby([date_col, cusip_col])[volume_col]
        .sum()
        .rename("buy_volume")
    )
    sell_agg = (
        sells.groupby([date_col, cusip_col])[volume_col]
        .sum()
        .rename("sell_volume")
    )

    # Combine
    combined = pd.concat([buy_agg, sell_agg], axis=1).fillna(0)

    # Compute imbalance
    total = combined["buy_volume"] + combined["sell_volume"]
    diff = combined["buy_volume"] - combined["sell_volume"]

    # Avoid division by zero
    imbalance = diff / total
    imbalance = imbalance.replace([np.inf, -np.inf], np.nan)

    imbalance.name = "retail_imbalance"

    logger.info(
        f"Computed imbalance for {len(imbalance)} (date, cusip) pairs. "
        f"Mean: {imbalance.mean():.4f}, Std: {imbalance.std():.4f}"
    )

    return imbalance


def compute_imbalance_from_volumes(
    buy_volume: pd.Series,
    sell_volume: pd.Series,
) -> pd.Series:
    """
    Compute imbalance from pre-aggregated buy/sell volumes.

    Parameters
    ----------
    buy_volume : pd.Series
        Aggregated buy volumes, indexed by (date, cusip).
    sell_volume : pd.Series
        Aggregated sell volumes, indexed by (date, cusip).

    Returns
    -------
    pd.Series
        Imbalance series, same index as inputs.

    Examples
    --------
    >>> imbalance = compute_imbalance_from_volumes(buys, sells)
    """
    total = buy_volume + sell_volume
    diff = buy_volume - sell_volume

    imbalance = diff / total
    imbalance = imbalance.replace([np.inf, -np.inf], np.nan)
    imbalance.name = "retail_imbalance"

    return imbalance
