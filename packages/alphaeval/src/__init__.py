"""alphaeval — Corporate Credit Alpha Evaluation Library.

Unified evaluation for corporate bonds, ETFs, and CDX indices in
price space (returns/PnL) and spread space (OAS/Z changes, duration proxies).

Public API — Transforms:
    price_to_returns, equity_curve, delta_spread_bp,
    spread_return_proxy, dv01_pnl, drawdown_series, runup_series

Public API — Metrics:
    profit_factor, win_rate_trades, expectancy, max_runup, cagr,
    tstat_returns, sortino_ratio, var_parametric, vpin,
    ic_star, rank_ic_star, ir_star, r_squared, tstat_ic,
    tracking_error

Public API — Sharpe Inference:
    estimated_sharpe_ratio, ann_estimated_sharpe_ratio,
    estimated_sharpe_ratio_stdev, probabilistic_sharpe_ratio,
    min_track_record_length, num_independent_trials,
    expected_maximum_sr, deflated_sharpe_ratio

Public API — Infrastructure:
    EvalConfig, MetricResult, validate_panel
"""

from .types import EvalConfig, MetricResult
from .panel import validate_panel

__all__ = [
    "EvalConfig",
    "MetricResult",
    "validate_panel",
]
