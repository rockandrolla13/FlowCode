from __future__ import annotations

"""Backtesting engine.

This module provides the main backtesting logic that combines
signals, position sizing, and return calculation.

IMPORTANT: Lookahead Prevention Rules
1. Signal at t can only use data up to t-1
2. Position at t is based on signal at t-1
3. PnL at t is computed using price change from t-1 to t
4. NEVER use .shift(-n) except in PnL calculation
"""

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .results import BacktestResult
from .portfolio import equal_weight

logger = logging.getLogger(__name__)


def compute_returns(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """
    Compute strategy returns from positions and prices.

    Parameters
    ----------
    positions : pd.DataFrame
        Position sizes indexed by date, columns are assets.
    prices : pd.DataFrame
        Price data with same structure as positions.

    Returns
    -------
    pd.Series
        Strategy returns indexed by date.

    Notes
    -----
    Return at t = sum(position_{t-1} * return_t) for all assets.
    This ensures no lookahead bias.
    """
    # Compute asset returns
    asset_returns = prices.pct_change()

    # Align positions and returns
    # Position at t-1 earns return at t
    lagged_positions = positions.shift(1)

    # Strategy return = sum of (position * return) across assets
    strategy_returns = (lagged_positions * asset_returns).sum(axis=1)

    strategy_returns = strategy_returns.fillna(0)

    return strategy_returns


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    """
    Compute performance metrics from returns.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns.

    Returns
    -------
    dict[str, float]
        Dictionary of metrics.

    Notes
    -----
    Formulas match metrics.performance and metrics.risk packages exactly.
    Cross-package imports are not possible due to shared ``src/`` namespace.
    When namespace is fixed, replace inline calculations with direct imports.
    """
    if len(returns) < 2:
        logger.warning("compute_metrics: fewer than 2 return periods, returning empty metrics")
        return {}

    # Sharpe ratio — spec §3.1: μ / σ * √252, ddof=1 (rf=0 assumed)
    std = returns.std(ddof=1)
    sharpe = float((returns.mean() / std) * np.sqrt(252)) if std > 0 else np.nan

    # Max drawdown — spec §4.1: (cum - peak) / peak
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    # Total return
    total_return = float((1 + returns).prod() - 1)

    # Annualized return — (1 + total)^(252/n) - 1
    n_periods = len(returns)
    years = n_periods / 252
    if total_return <= -1.0:
        ann_return = np.nan
    elif years > 0:
        ann_return = float((1 + total_return) ** (1 / years) - 1)
    else:
        ann_return = np.nan

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "mean_return": float(returns.mean()),
        "volatility": float(std * np.sqrt(252)),
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
        "loss_rate": float((returns < 0).mean()),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }


def generate_trades(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate trade log from position changes.

    Parameters
    ----------
    positions : pd.DataFrame
        Position sizes.
    prices : pd.DataFrame
        Price data.

    Returns
    -------
    pd.DataFrame
        Trade log with columns: date, asset, side, size, price.
    """
    position_changes = positions.diff()

    # Stack to long format and filter non-zero changes (vectorized)
    stacked = position_changes.iloc[1:].stack()
    nonzero = stacked[stacked != 0]

    if len(nonzero) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "size", "price"])

    trades_df = nonzero.reset_index()
    trades_df.columns = ["date", "asset", "change"]
    trades_df["side"] = np.where(trades_df["change"] > 0, "buy", "sell")
    trades_df["size"] = trades_df["change"].abs()

    # Merge prices via stack for vectorized lookup
    prices_long = prices.stack().reset_index()
    prices_long.columns = ["date", "asset", "price"]
    trades_df = trades_df.merge(prices_long, on=["date", "asset"], how="left")

    return trades_df[["date", "asset", "side", "size", "price"]]


def run_backtest(
    signal: pd.Series | pd.DataFrame,
    prices: pd.DataFrame,
    position_sizer: Callable | None = None,
    transaction_cost: float = 0.0,
    max_positions: int = 50,
    **kwargs,
) -> BacktestResult:
    """
    Run signal-based backtest.

    Parameters
    ----------
    signal : pd.Series | pd.DataFrame
        Signal values. Positive = long, negative = short.
        If Series, should have MultiIndex (date, asset).
        If DataFrame, index is date, columns are assets.
    prices : pd.DataFrame
        Price data. Index is date, columns are assets.
    position_sizer : Callable | None, optional
        Function(signal, prices, **kwargs) -> positions.
        Default is equal_weight.
    transaction_cost : float, default 0.0
        Cost per unit traded (as fraction of price).
    max_positions : int, default 50
        Maximum number of positions.
    **kwargs
        Additional arguments passed to position_sizer.

    Returns
    -------
    BacktestResult
        Backtest results including returns, positions, trades, metrics.

    Examples
    --------
    >>> result = run_backtest(signal, prices)
    >>> print(result.summary())

    Notes
    -----
    Lookahead Prevention:
    - Signal at t determines position at t+1
    - Position at t earns return from t to t+1
    """
    logger.info("Starting backtest...")

    # Convert signal to DataFrame if Series with MultiIndex
    if isinstance(signal, pd.Series):
        if isinstance(signal.index, pd.MultiIndex):
            signal = signal.unstack()
        else:
            signal = signal.to_frame()

    # Default position sizer
    if position_sizer is None:
        position_sizer = equal_weight

    # Compute positions
    positions = position_sizer(
        signal,
        prices,
        max_positions=max_positions,
        **kwargs,
    )

    # Ensure alignment
    common_dates = positions.index.intersection(prices.index)
    common_assets = positions.columns.intersection(prices.columns)

    positions = positions.loc[common_dates, common_assets]
    prices_aligned = prices.loc[common_dates, common_assets]

    # Compute returns
    returns = compute_returns(positions, prices_aligned)

    # Apply transaction costs
    if transaction_cost > 0:
        turnover = positions.diff().abs().sum(axis=1)
        costs = turnover * transaction_cost
        returns = returns - costs

    # Generate trade log
    trades = generate_trades(positions, prices_aligned)

    # Compute metrics
    metrics = compute_metrics(returns)

    # Store config
    config = {
        "transaction_cost": transaction_cost,
        "max_positions": max_positions,
        "position_sizer": position_sizer.__name__,
    }

    result = BacktestResult(
        returns=returns,
        positions=positions,
        trades=trades,
        metrics=metrics,
        config=config,
    )

    logger.info(f"Backtest complete. Total return: {result.total_return:.2%}")

    return result
