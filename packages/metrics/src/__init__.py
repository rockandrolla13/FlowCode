"""FlowCode Metrics Package - Performance and risk metrics.

This package provides pure functions for computing performance
and risk metrics. All functions are stateless with no side effects.

Public API:
- sharpe_ratio: Annualized Sharpe ratio
- sortino_ratio: Annualized Sortino ratio (downside deviation)
- calmar_ratio: Annualized return / max drawdown
- max_drawdown: Maximum peak-to-trough drawdown
- drawdown_series: Running drawdown series
- value_at_risk: Historical VaR
- hit_rate: Signal accuracy
- autocorrelation: Series autocorrelation
"""

from .performance import sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall
from .diagnostics import hit_rate, autocorrelation, signal_decay, information_coefficient

__all__ = [
    # Performance
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "annualized_return",
    # Risk
    "max_drawdown",
    "drawdown_series",
    "value_at_risk",
    "expected_shortfall",
    # Diagnostics
    "hit_rate",
    "autocorrelation",
    "signal_decay",
    "information_coefficient",
]
