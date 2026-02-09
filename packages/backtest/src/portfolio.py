from __future__ import annotations

"""Portfolio construction and position sizing.

This module provides position sizing functions that convert
signals into portfolio weights/positions.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    # Rank by absolute signal for max_positions truncation
    ranks = signal.abs().rank(axis=1, ascending=False, method="first")
    mask = (ranks <= max_positions) & (directions != 0)

    # Zero out positions beyond max_positions
    directions = directions.where(mask, 0)

    # Equal weight: divide direction by count of active positions per row
    counts = mask.sum(axis=1).clip(lower=1)
    positions = directions.div(counts, axis=0).fillna(0)

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
    Falls back to equal weight on dates where rolling volatility is
    unavailable (e.g., insufficient history or constant prices).
    """
    # Compute rolling volatility
    returns = prices.pct_change()
    vol = returns.rolling(window=vol_window).std()

    # Get position directions
    directions = np.sign(signal)

    if long_only:
        directions = directions.clip(lower=0)

    # Rank by absolute signal for max_positions truncation
    ranks = signal.abs().rank(axis=1, ascending=False, method="first")
    active_mask = (ranks <= max_positions) & (directions != 0)

    # Align volatility to signal index (dates not in vol get NaN → 0 positions)
    vol_aligned = vol.reindex(signal.index)

    # Inverse volatility weights (masked to active positions only)
    inv_vol = (1.0 / vol_aligned).replace([np.inf, -np.inf], np.nan).fillna(0)
    inv_vol = inv_vol.where(active_mask, 0)

    # Row-wise normalization
    row_sums = inv_vol.sum(axis=1)

    # Where inv_vol sums to 0, fall back to equal weight
    has_vol = row_sums > 0
    n_fallback = int((~has_vol).sum())
    if n_fallback > 0:
        logger.warning(
            "risk_parity: %d of %d dates fell back to equal weight "
            "(zero inverse-volatility sums)", n_fallback, len(row_sums)
        )
    counts = active_mask.sum(axis=1).clip(lower=1)

    # Risk parity weights where volatility available
    rp_weights = inv_vol.div(row_sums.clip(lower=1e-15), axis=0)
    # Equal weights as fallback
    eq_weights = active_mask.astype(float).div(counts, axis=0)

    # Broadcast row-level condition to full DataFrame shape
    has_vol_mask = pd.DataFrame(
        np.tile(has_vol.values[:, np.newaxis], (1, rp_weights.shape[1])),
        index=rp_weights.index,
        columns=rp_weights.columns,
    )
    weights = rp_weights.where(has_vol_mask, eq_weights)
    positions = (directions.where(active_mask, 0) * weights).fillna(0)

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
    valid = signal.notna()

    # Rank descending for longs (highest signal = rank 1)
    ranks_desc = signal.rank(axis=1, ascending=False, method="first", na_option="bottom")
    # Rank ascending for shorts (lowest signal = rank 1)
    ranks_asc = signal.rank(axis=1, ascending=True, method="first", na_option="bottom")

    long_mask = (ranks_desc <= n_long) & valid & (n_long > 0)
    short_mask = (ranks_asc <= n_short) & valid & (n_short > 0)

    positions = (
        long_mask.astype(float) / max(n_long, 1)
        - short_mask.astype(float) / max(n_short, 1)
    )

    return positions
