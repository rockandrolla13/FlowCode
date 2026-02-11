"""Core types for alphaeval.

Dataclasses for configuration, metric results, and panel schema contracts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple


@dataclass
class EvalConfig:
    """Evaluation configuration.

    Parameters
    ----------
    periods_per_year : int, default 252
        Trading days per year for annualization.
    ddof : int, default 1
        Degrees of freedom for sample std.
    var_confidence : float, default 0.95
        Confidence level for VaR/ES.
    sharpe_benchmark : float, default 0.0
        SR* benchmark for PSR/DSR.
    psr_confidence : float, default 0.95
        Probability threshold for minTRL.
    risk_free : float, default 0.0
        Risk-free rate per period.
    """

    periods_per_year: int = 252
    ddof: int = 1
    var_confidence: float = 0.95
    sharpe_benchmark: float = 0.0
    psr_confidence: float = 0.95
    risk_free: float = 0.0


class MetricResult(NamedTuple):
    """Single metric output.

    Parameters
    ----------
    name : str
        Metric name (e.g. "sharpe_ratio").
    value : float | None
        Scalar result. None if not computable.
    meta : dict[str, Any] | None
        Extra info (e.g. {"annualized": True, "ddof": 1}).
        None by default to avoid mutable default aliasing.
    """

    name: str
    value: float | None
    meta: dict[str, Any] | None = None


# Required column sets for panel validation
REQUIRED_TIMESERIES = {"date", "instrument", "returns"}
REQUIRED_SPREAD = {"date", "instrument", "spread"}
REQUIRED_FACTOR = {"date", "instrument", "signal", "target"}
REQUIRED_TRADES = {"trade_id", "instrument", "pnl"}
