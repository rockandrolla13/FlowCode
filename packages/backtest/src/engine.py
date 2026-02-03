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
    price_col: str = "price",
) -> pd.Series:
    """
    Compute strategy returns from positions and prices.

    Parameters
    ----------
    positions : pd.DataFrame
        Position sizes indexed by date, columns are assets.
    prices : pd.DataFrame
        Price data with same structure as positions.
    price_col : str
        Column name for price if prices is multi-column per asset.

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
    if isinstance(prices, pd.Series):
        asset_returns = prices.pct_change()
    else:
        asset_returns = prices.pct_change()

    # Align positions and returns
    # Position at t-1 earns return at t
    lagged_positions = positions.shift(1)

    # Strategy return = sum of (position * return) across assets
    if isinstance(asset_returns, pd.Series):
        strategy_returns = lagged_positions * asset_returns
    else:
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
    """
    if len(returns) < 2:
        return {}

    # Import here to avoid circular dependency
    # In real implementation, these would be imported at top
    metrics = {}

    # Basic stats
    metrics["total_return"] = float((1 + returns).prod() - 1)
    metrics["mean_return"] = float(returns.mean())
    metrics["volatility"] = float(returns.std() * np.sqrt(252))

    # Sharpe ratio
    if returns.std() > 0:
        metrics["sharpe_ratio"] = float(
            returns.mean() / returns.std() * np.sqrt(252)
        )
    else:
        metrics["sharpe_ratio"] = np.nan

    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    metrics["max_drawdown"] = float(drawdown.min())

    # Win rate
    metrics["win_rate"] = float((returns > 0).mean())
    metrics["loss_rate"] = float((returns < 0).mean())

    # Skewness and kurtosis
    metrics["skewness"] = float(returns.skew())
    metrics["kurtosis"] = float(returns.kurtosis())

    return metrics


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
        Trade log with columns: date, asset, side, size, price, pnl.
    """
    trades = []

    position_changes = positions.diff()

    for date in position_changes.index[1:]:
        for asset in position_changes.columns:
            change = position_changes.loc[date, asset]
            if change != 0:
                trade = {
                    "date": date,
                    "asset": asset,
                    "side": "buy" if change > 0 else "sell",
                    "size": abs(change),
                    "price": prices.loc[date, asset] if asset in prices.columns else np.nan,
                }
                trades.append(trade)

    if not trades:
        return pd.DataFrame(columns=["date", "asset", "side", "size", "price"])

    return pd.DataFrame(trades)


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
        "position_sizer": position_sizer.__name__ if position_sizer else "none",
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
