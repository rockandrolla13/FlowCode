"""FlowCode Backtest Package - Strategy harness.

This package orchestrates signals, position sizing, and return
calculation for backtesting credit trading strategies.

Public API:
- run_backtest: Run signal-based backtest
- compute_returns, compute_metrics: Return and metric computation
- equal_weight, risk_parity, top_n_positions: Position sizing
- BacktestResult: Result container
"""

from .engine import run_backtest, compute_returns, compute_metrics
from .portfolio import equal_weight, risk_parity, top_n_positions
from .results import BacktestResult

__all__ = [
    "run_backtest",
    "compute_returns",
    "compute_metrics",
    "equal_weight",
    "risk_parity",
    "top_n_positions",
    "BacktestResult",
]
