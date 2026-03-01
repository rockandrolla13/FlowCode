from __future__ import annotations

"""Backtest result container.

This module defines the BacktestResult dataclass that holds
all outputs from a backtest run.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class BacktestResult:
    """Container for backtest outputs.

    Attributes
    ----------
    returns : pd.Series
        Strategy returns indexed by date.
    positions : pd.DataFrame
        Position sizes with columns for each asset.
    trades : pd.DataFrame
        Trade log with entry/exit details.
    metrics : dict[str, float]
        Performance metrics (sharpe, max_dd, etc.).
    config : dict[str, Any]
        Configuration used for the backtest.

    Examples
    --------
    >>> result = run_backtest(signal, prices)
    >>> result.metrics["sharpe_ratio"]
    1.5
    >>> result.returns.sum()
    0.25
    """

    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        """Total cumulative return."""
        if len(self.returns) == 0:
            return 0.0
        return float((1 + self.returns).prod() - 1)

    @property
    def n_trades(self) -> int:
        """Number of trades executed."""
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of profitable trades."""
        if len(self.trades) == 0:
            return 0.0
        if "pnl" not in self.trades.columns:
            return 0.0
        wins = (self.trades["pnl"] > 0).sum()
        return float(wins / len(self.trades))

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Backtest Results",
            "=" * 40,
            f"Total Return: {self.total_return:.2%}",
            f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 'N/A')}",
            f"Max Drawdown: {self.metrics.get('max_drawdown', 'N/A')}",
            f"Num Trades: {self.n_trades}",
            f"Win Rate: {self.win_rate:.2%}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "returns": self.returns.to_dict(),
            "positions": self.positions.to_dict(),
            "trades": self.trades.to_dict(),
            "metrics": self.metrics,
            "config": self.config,
        }
