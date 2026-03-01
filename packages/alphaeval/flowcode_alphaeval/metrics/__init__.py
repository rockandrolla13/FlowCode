"""alphaeval metrics — performance, risk, factor, relative, and Sharpe inference."""

from .performance import (
    cagr,
    expectancy,
    max_runup,
    profit_factor,
    sortino_ratio,
    tstat_returns,
    win_rate_trades,
)
from .risk import var_parametric, vpin
from .factor import ic_star, ir_star, r_squared, rank_ic_star, tstat_ic
from .relative import tracking_error
from .sharpe_inference import (
    ann_estimated_sharpe_ratio,
    deflated_sharpe_ratio,
    estimated_sharpe_ratio,
    estimated_sharpe_ratio_stdev,
    expected_maximum_sr,
    min_track_record_length,
    num_independent_trials,
    probabilistic_sharpe_ratio,
)

__all__ = [
    "profit_factor",
    "win_rate_trades",
    "expectancy",
    "max_runup",
    "cagr",
    "tstat_returns",
    "sortino_ratio",
    "var_parametric",
    "vpin",
    "ic_star",
    "rank_ic_star",
    "ir_star",
    "r_squared",
    "tstat_ic",
    "tracking_error",
    "estimated_sharpe_ratio",
    "ann_estimated_sharpe_ratio",
    "estimated_sharpe_ratio_stdev",
    "probabilistic_sharpe_ratio",
    "min_track_record_length",
    "num_independent_trials",
    "expected_maximum_sr",
    "deflated_sharpe_ratio",
]
