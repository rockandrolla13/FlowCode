"""Portfolio construction and position sizing.

This module provides position sizing functions that convert
signals into portfolio weights/positions.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def signal_to_positions(
    signal: pd.DataFrame,
    method: str = "sign",
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Convert raw signal to position directions.

    Parameters
    ----------
    signal : pd.DataFrame
        Raw signal values.
    method : str, default "sign"
        Conversion method:
        - "sign": Use sign of signal (-1, 0, +1)
        - "raw": Use raw signal values
        - "rank": Use cross-sectional rank
    normalize : bool, default True
        Normalize positions to sum to 1 (absolute).

    Returns
    -------
    pd.DataFrame
        Position directions.
    """
    if method == "sign":
        positions = np.sign(signal)
    elif method == "raw":
        positions = signal.copy()
    elif method == "rank":
        positions = signal.rank(axis=1, pct=True) - 0.5
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        row_sums = positions.abs().sum(axis=1)
        row_sums = row_sums.replace(0, 1)  # Avoid division by zero
        positions = positions.div(row_sums, axis=0)

    return positions


def equal_weight(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    max_positions: int = 50,
    long_only: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Equal-weight position sizing.

    All positions have equal absolute weight.

    Parameters
    ----------
    signal : pd.DataFrame
        Signal values (positive = long, negative = short).
    prices : pd.DataFrame
        Price data (not used for equal weight, included for API consistency).
    max_positions : int, default 50
        Maximum number of positions.
    long_only : bool, default False
        If True, only take long positions.
    **kwargs
        Additional arguments (ignored).

    Returns
    -------
    pd.DataFrame
        Position weights (sum of absolute values = 1).

    Examples
    --------
    >>> positions = equal_weight(signal, prices, max_positions=20)
    >>> positions.abs().sum(axis=1).mean()
    1.0
    """
    # Get position directions
    directions = np.sign(signal)

    if long_only:
        directions = directions.clip(lower=0)

    positions = pd.DataFrame(
        index=signal.index,
        columns=signal.columns,
        dtype=float,
    )

    for date in signal.index:
        row = directions.loc[date]
        nonzero = row[row != 0]

        if len(nonzero) == 0:
            positions.loc[date] = 0
            continue

        # Limit to max positions (top by absolute signal)
        if len(nonzero) > max_positions:
            signal_row = signal.loc[date]
            top_assets = signal_row.abs().nlargest(max_positions).index
            nonzero = nonzero.loc[nonzero.index.isin(top_assets)]

        # Equal weight
        weight = 1.0 / len(nonzero)
        positions.loc[date, nonzero.index] = nonzero * weight

    positions = positions.fillna(0)

    return positions


def risk_parity(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    vol_window: int = 21,
    max_positions: int = 50,
    long_only: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Risk parity (inverse volatility) position sizing.

    Positions are weighted inversely to recent volatility,
    so each position contributes equal risk.

    Parameters
    ----------
    signal : pd.DataFrame
        Signal values.
    prices : pd.DataFrame
        Price data for volatility calculation.
    vol_window : int, default 21
        Window for volatility calculation.
    max_positions : int, default 50
        Maximum number of positions.
    long_only : bool, default False
        If True, only take long positions.
    **kwargs
        Additional arguments (ignored).

    Returns
    -------
    pd.DataFrame
        Position weights.

    Notes
    -----
    Weight_i = (1 / vol_i) / sum(1 / vol_j) for all j in portfolio.
    """
    # Compute rolling volatility
    returns = prices.pct_change()
    volatility = returns.rolling(window=vol_window).std()

    # Get position directions
    directions = np.sign(signal)

    if long_only:
        directions = directions.clip(lower=0)

    positions = pd.DataFrame(
        index=signal.index,
        columns=signal.columns,
        dtype=float,
    )

    for date in signal.index:
        if date not in volatility.index:
            positions.loc[date] = 0
            continue

        row = directions.loc[date]
        vol_row = volatility.loc[date]

        nonzero = row[row != 0]

        if len(nonzero) == 0:
            positions.loc[date] = 0
            continue

        # Limit to max positions
        if len(nonzero) > max_positions:
            signal_row = signal.loc[date]
            top_assets = signal_row.abs().nlargest(max_positions).index
            nonzero = nonzero.loc[nonzero.index.isin(top_assets)]

        # Inverse volatility weights
        inv_vol = 1.0 / vol_row.loc[nonzero.index]
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(0)

        if inv_vol.sum() == 0:
            # Fall back to equal weight
            weight = 1.0 / len(nonzero)
            positions.loc[date, nonzero.index] = nonzero * weight
        else:
            weights = inv_vol / inv_vol.sum()
            positions.loc[date, nonzero.index] = nonzero * weights

    positions = positions.fillna(0)

    return positions


def top_n_positions(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    n_long: int = 10,
    n_short: int = 10,
    **kwargs,
) -> pd.DataFrame:
    """
    Take top N long and bottom N short positions.

    Parameters
    ----------
    signal : pd.DataFrame
        Signal values.
    prices : pd.DataFrame
        Price data (not used, for API consistency).
    n_long : int, default 10
        Number of long positions.
    n_short : int, default 10
        Number of short positions.
    **kwargs
        Additional arguments (ignored).

    Returns
    -------
    pd.DataFrame
        Position weights (+1/n_long for longs, -1/n_short for shorts).
    """
    positions = pd.DataFrame(
        0.0,
        index=signal.index,
        columns=signal.columns,
    )

    for date in signal.index:
        row = signal.loc[date].dropna()

        if len(row) == 0:
            continue

        # Top N for long
        if n_long > 0:
            top = row.nlargest(min(n_long, len(row))).index
            positions.loc[date, top] = 1.0 / n_long

        # Bottom N for short
        if n_short > 0:
            bottom = row.nsmallest(min(n_short, len(row))).index
            positions.loc[date, bottom] = -1.0 / n_short

    return positions
