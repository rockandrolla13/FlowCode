"""
Core performance and risk metrics for credit trading strategies.
"""
from typing import List, Dict
import pandas as pd
import numpy as np

def credit_pnl(
    spread_change: pd.Series,
    pvbp: pd.Series,
    mid_price: pd.Series
) -> pd.Series:
    """
    Calculate spread-based, risk-weighted Profit and Loss (PnL).

    Parameters
    ----------
    spread_change : pd.Series
        Series of spread changes (e.g., 1-week forward).
    pvbp : pd.Series
        Present value of a basis point for risk normalization.
    mid_price : pd.Series
        Mid price of the instrument.

    Returns
    -------
    pd.Series
        The calculated credit PnL. A negative spread change (tightening)
        results in a positive PnL.
    """
    # Formula: -spread_change * (pvbp / mid_price)
    return -spread_change * (pvbp / mid_price)

def sharpe_ratio(
    pnl: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the annualized Sharpe ratio from a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.

    Returns
    -------
    float
        The annualized Sharpe ratio.
    """
    # Assuming daily PnL, annualizing by sqrt(252)
    excess_returns = pnl - risk_free_rate / 252
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

def sortino_ratio(
    pnl: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the annualized Sortino ratio from a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.

    Returns
    -------
    float
        The annualized Sortino ratio.
    """
    excess_returns = pnl - risk_free_rate / 252
    downside_std = excess_returns[excess_returns < 0].std()
    if downside_std == 0:
        return np.inf
    return (excess_returns.mean() / downside_std) * np.sqrt(252)

def max_drawdown(pnl: pd.Series) -> float:
    """
    Calculate the maximum drawdown from a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.

    Returns
    -------
    float
        The maximum drawdown as a negative value.
    """
    cumulative_pnl = pnl.cumsum()
    peak = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - peak
    return drawdown.min()

def max_run_up(pnl: pd.Series) -> float:
    """
    Calculate the maximum run-up from a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.

    Returns
    -------
    float
        The maximum run-up as a positive value.
    """
    cumulative_pnl = pnl.cumsum()
    trough = cumulative_pnl.cummin()
    run_up = cumulative_pnl - trough
    return run_up.max()

def expectancy(pnl: pd.Series) -> float:
    """
    Calculate the expectancy of a strategy from its PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss per trade.

    Returns
    -------
    float
        The expectancy value (average PnL per trade).
    """
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    if len(pnl) == 0:
        return 0.0

    win_rate = len(wins) / len(pnl)
    loss_rate = len(losses) / len(pnl)

    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = losses.mean() if not losses.empty else 0.0

    return (win_rate * avg_win) + (loss_rate * avg_loss)


def average_return(pnl: pd.Series) -> float:
    """
    Calculate the average return from a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.

    Returns
    -------
    float
        The arithmetic mean of the PnL series.
    """
    return pnl.mean()

def return_quantiles(
    pnl: pd.Series,
    quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
) -> Dict[float, float]:
    """
    Calculate the quantiles of a PnL series.

    Parameters
    ----------
    pnl : pd.Series
        Time series of Profit and Loss.
    quantiles : List[float], optional
        List of quantiles to compute, by default [0.05, 0.25, 0.5, 0.75, 0.95].

    Returns
    -------
    Dict[float, float]
        A dictionary mapping each quantile to its corresponding PnL value.
    """
    return pnl.quantile(quantiles).to_dict()

def average_holding_time(trade_log: pd.DataFrame) -> pd.Series:
    """
    Calculate the average holding time per instrument (ISIN).

    This requires a trade log from the backtest engine.

    Parameters
    ----------
    trade_log : pd.DataFrame
        DataFrame with columns ['isin', 'entry_time', 'exit_time'].
        'entry_time' and 'exit_time' should be datetime objects.

    Returns
    -------
    pd.Series
        A Series indexed by 'isin' with the average holding time as a Timedelta.
    """
    if not all(col in trade_log.columns for col in ['isin', 'entry_time', 'exit_time']):
        raise ValueError("trade_log must contain 'isin', 'entry_time', and 'exit_time' columns.")

    trade_log['holding_time'] = trade_log['exit_time'] - trade_log['entry_time']
    return trade_log.groupby('isin')['holding_time'].mean()
