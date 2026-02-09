from __future__ import annotations

"""Retail order imbalance signal.

This module computes retail order imbalance from TRACE data
using the Quote Midpoint (QMP) classification rule.

Formula (Spec §1.3):
    I_t = (buy_volume - sell_volume) / (buy_volume + sell_volume)

QMP Classification (Spec §1.5):
    Buy if price > midpoint + (α × spread)
    Sell if price < midpoint - (α × spread)
    Neutral otherwise
    Exclude trades in 40-60% NBBO exclusion zone

Retail Identification (BJZZ subpenny logic):
    A trade is retail if:
    1. Notional < $200,000
    2. mod(Price × 100, 1) > 0 (subpenny pricing)
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants for retail identification
RETAIL_NOTIONAL_THRESHOLD = 200_000  # $200,000
NBBO_EXCLUSION_LOW = 0.40  # 40% of spread
NBBO_EXCLUSION_HIGH = 0.60  # 60% of spread


def is_subpenny(price: float) -> bool:
    """
    Check if price has subpenny pricing.

    Subpenny trades have fractional cents, indicating retail execution.

    Parameters
    ----------
    price : float
        Trade execution price.

    Returns
    -------
    bool
        True if price has subpenny component.

    Examples
    --------
    >>> is_subpenny(100.501)  # Has subpenny
    True
    >>> is_subpenny(100.50)   # No subpenny
    False
    >>> is_subpenny(100.00)   # Round price
    False
    """
    # mod(price * 100, 1) > 0 means there's a fractional cent
    # Use round() to avoid IEEE 754 false positives (e.g. 100.10 * 100 = 10010.000000000002)
    cents = round(price * 100, 6)
    frac = cents % 1
    return frac > 1e-9


def is_retail_trade(
    price: float,
    notional: float,
    notional_threshold: float = RETAIL_NOTIONAL_THRESHOLD,
) -> bool:
    """
    Identify if a trade is likely retail using BJZZ subpenny logic.

    A trade is classified as retail if:
    1. Notional value < $200,000
    2. Price has subpenny component (mod(Price × 100, 1) > 0)

    Parameters
    ----------
    price : float
        Trade execution price.
    notional : float
        Trade notional value in dollars.
    notional_threshold : float, default 200_000
        Maximum notional for retail classification.

    Returns
    -------
    bool
        True if trade is likely retail.

    Examples
    --------
    >>> is_retail_trade(price=100.501, notional=50_000)
    True
    >>> is_retail_trade(price=100.50, notional=50_000)  # No subpenny
    False
    >>> is_retail_trade(price=100.501, notional=500_000)  # Too large
    False

    Notes
    -----
    Based on BJZZ methodology [cite: 41, 73, 1185, 1193].
    Subpenny pricing indicates price improvement from wholesalers,
    which is characteristic of retail order flow.
    """
    return notional < notional_threshold and is_subpenny(price)


def _is_subpenny_mask(prices: pd.Series) -> pd.Series:
    """
    Vectorized subpenny check for a Series of prices.

    Parameters
    ----------
    prices : pd.Series
        Trade execution prices.

    Returns
    -------
    pd.Series
        Boolean series, True where price has subpenny component.
    """
    cents = (prices * 100).round(6)
    frac = cents % 1
    return frac > 1e-9


def classify_retail_trades(
    trades: pd.DataFrame,
    price_col: str = "price",
    notional_col: str = "notional",
    notional_threshold: float = RETAIL_NOTIONAL_THRESHOLD,
) -> pd.Series:
    """
    Classify all trades as retail or institutional.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data.
    price_col : str
        Column name for price.
    notional_col : str
        Column name for notional value.
    notional_threshold : float
        Maximum notional for retail classification.

    Returns
    -------
    pd.Series
        Boolean series, True for retail trades.

    Examples
    --------
    >>> trades = pd.DataFrame({
    ...     'price': [100.501, 100.50, 100.123],
    ...     'notional': [50_000, 50_000, 300_000]
    ... })
    >>> classify_retail_trades(trades)
    0     True
    1    False
    2    False
    dtype: bool
    """
    has_subpenny = _is_subpenny_mask(trades[price_col])
    below_threshold = trades[notional_col] < notional_threshold
    return has_subpenny & below_threshold


def qmp_classify(
    price: float,
    mid: float,
    spread: float,
    threshold: float = 0.1,
) -> Literal["buy", "sell", "neutral"]:
    """
    Classify trade direction using Quote Midpoint (QMP) rule.

    Spec §1.5:
        BUY    if Price > Midpoint + (α × Spread)
        SELL   if Price < Midpoint - (α × Spread)
        NEUTRAL otherwise

    Parameters
    ----------
    price : float
        Execution price of the trade.
    mid : float
        Midpoint of bid-ask spread.
    spread : float
        Bid-ask spread (ask - bid).
    threshold : float, default 0.1
        Fraction of spread for neutral band (α).

    Returns
    -------
    Literal["buy", "sell", "neutral"]
        Trade direction classification.

    Examples
    --------
    >>> qmp_classify(price=100.5, mid=100.0, spread=1.0, threshold=0.1)
    'buy'
    >>> qmp_classify(price=99.5, mid=100.0, spread=1.0, threshold=0.1)
    'sell'
    >>> qmp_classify(price=100.0, mid=100.0, spread=1.0, threshold=0.1)
    'neutral'
    """
    if spread <= 0:
        return "neutral"
    upper = mid + threshold * spread
    lower = mid - threshold * spread
    if price > upper:
        return "buy"
    elif price < lower:
        return "sell"
    else:
        return "neutral"


def qmp_classify_with_exclusion(
    price: float,
    bid: float,
    ask: float,
    exclusion_low: float = NBBO_EXCLUSION_LOW,
    exclusion_high: float = NBBO_EXCLUSION_HIGH,
) -> Literal["buy", "sell", "neutral"]:
    """
    Classify trade direction using Lee-Ready QMP with NBBO exclusion zone.

    Trades within the 40-60% exclusion zone are marked as neutral
    and should be excluded from imbalance calculations.

    Parameters
    ----------
    price : float
        Execution price of the trade.
    bid : float
        Best bid price.
    ask : float
        Best ask price.
    exclusion_low : float, default 0.40
        Lower bound of exclusion zone (fraction of spread from bid).
    exclusion_high : float, default 0.60
        Upper bound of exclusion zone (fraction of spread from bid).

    Returns
    -------
    Literal["buy", "sell", "neutral"]
        Trade direction. "neutral" for trades in exclusion zone.

    Examples
    --------
    >>> qmp_classify_with_exclusion(price=100.8, bid=100.0, ask=101.0)
    'buy'
    >>> qmp_classify_with_exclusion(price=100.2, bid=100.0, ask=101.0)
    'sell'
    >>> qmp_classify_with_exclusion(price=100.5, bid=100.0, ask=101.0)
    'neutral'  # In 40-60% zone

    Notes
    -----
    Based on Lee-Ready QMP with exclusion zone [cite: 1200].
    The exclusion zone filters ambiguous trades near the midpoint.
    """
    spread = ask - bid
    if spread <= 0:
        return "neutral"

    # Position of price within the spread (0 = bid, 1 = ask)
    price_position = (price - bid) / spread

    if price_position > exclusion_high:
        return "buy"
    elif price_position < exclusion_low:
        return "sell"
    else:
        return "neutral"


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
        Series of 'buy', 'sell', or 'neutral' classifications.
    """
    upper = trades[mid_col] + threshold * trades[spread_col]
    lower = trades[mid_col] - threshold * trades[spread_col]
    conditions = [
        trades[price_col] > upper,
        trades[price_col] < lower,
    ]
    choices = ["buy", "sell"]
    return pd.Series(
        np.select(conditions, choices, default="neutral"),
        index=trades.index,
    )


def classify_trades_qmp_with_exclusion(
    trades: pd.DataFrame,
    price_col: str = "price",
    bid_col: str = "bid",
    ask_col: str = "ask",
    exclusion_low: float = NBBO_EXCLUSION_LOW,
    exclusion_high: float = NBBO_EXCLUSION_HIGH,
) -> pd.Series:
    """
    Classify all trades using QMP with NBBO exclusion zone.

    Trades within the 40-60% zone are marked as 'neutral' and
    should be excluded from imbalance calculations.

    Parameters
    ----------
    trades : pd.DataFrame
        Trade data with price, bid, and ask columns.
    price_col : str
        Column name for execution price.
    bid_col : str
        Column name for bid price.
    ask_col : str
        Column name for ask price.
    exclusion_low : float, default 0.40
        Lower bound of exclusion zone.
    exclusion_high : float, default 0.60
        Upper bound of exclusion zone.

    Returns
    -------
    pd.Series
        Series of 'buy', 'sell', or 'neutral' classifications.

    Examples
    --------
    >>> trades = pd.DataFrame({
    ...     'price': [100.8, 100.2, 100.5],
    ...     'bid': [100.0, 100.0, 100.0],
    ...     'ask': [101.0, 101.0, 101.0]
    ... })
    >>> classify_trades_qmp_with_exclusion(trades)
    0       buy
    1      sell
    2    neutral
    dtype: object
    """
    spread = trades[ask_col] - trades[bid_col]
    price_position = (trades[price_col] - trades[bid_col]) / spread

    # Handle zero spread
    price_position = price_position.replace([np.inf, -np.inf], np.nan)

    conditions = [
        price_position > exclusion_high,
        price_position < exclusion_low,
    ]
    choices = ["buy", "sell"]

    return pd.Series(
        np.select(conditions, choices, default="neutral"),
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
