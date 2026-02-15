"""alphaeval — Corporate Credit Alpha Evaluation Library.

Unified evaluation for corporate bonds, ETFs, and CDX indices in
price space (returns/PnL) and spread space (OAS/Z changes, duration proxies).

Top-level exports: EvalConfig, MetricResult, validate_panel.

Submodules (import directly)::

    src.transforms.returns    — price_to_returns, equity_curve
    src.transforms.spreads    — delta_spread_bp, spread_return_proxy, dv01_pnl
    src.transforms.equity     — drawdown_series, runup_series
    src.metrics.performance   — profit_factor, win_rate_trades, expectancy,
                                max_runup, cagr, tstat_returns, sortino_ratio
    src.metrics.risk          — var_parametric, vpin
    src.metrics.factor        — ic_star, rank_ic_star, ir_star, r_squared, tstat_ic
    src.metrics.relative      — tracking_error
    src.metrics.sharpe_inference — estimated_sharpe_ratio, ann_estimated_sharpe_ratio,
                                  estimated_sharpe_ratio_stdev, probabilistic_sharpe_ratio,
                                  min_track_record_length, num_independent_trials,
                                  expected_maximum_sr, deflated_sharpe_ratio
"""

from .types import EvalConfig, MetricResult
from .panel import validate_panel

__all__ = [
    "EvalConfig",
    "MetricResult",
    "validate_panel",
]
