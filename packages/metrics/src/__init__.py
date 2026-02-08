"""FlowCode Metrics Package - Performance and risk metrics.

This package provides pure functions for computing performance
and risk metrics. All functions are stateless with no side effects.

Public API:
- sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return, information_ratio
- max_drawdown, drawdown_series, drawdown_duration, max_drawdown_duration
- value_at_risk, expected_shortfall, volatility, downside_volatility
- hit_rate, autocorrelation, autocorrelation_profile, signal_decay
- information_coefficient, turnover, holding_period
"""

from .performance import sharpe_ratio, sortino_ratio, calmar_ratio, annualized_return, information_ratio
from .risk import max_drawdown, drawdown_series, value_at_risk, expected_shortfall, volatility, downside_volatility, drawdown_duration, max_drawdown_duration
from .diagnostics import hit_rate, autocorrelation, autocorrelation_profile, signal_decay, information_coefficient, turnover, holding_period

__all__ = [
    # Performance
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "annualized_return",
    "information_ratio",
    # Risk
    "max_drawdown",
    "drawdown_series",
    "value_at_risk",
    "expected_shortfall",
    "volatility",
    "downside_volatility",
    "drawdown_duration",
    "max_drawdown_duration",
    # Diagnostics
    "hit_rate",
    "autocorrelation",
    "autocorrelation_profile",
    "signal_decay",
    "information_coefficient",
    "turnover",
    "holding_period",
]
