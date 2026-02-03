"""FlowCode Signals Package - Signal computations.

This package provides signal computations that wrap core functionality.
All signals return pd.Series indexed by (date, cusip).

Public API:
- compute_retail_imbalance: Retail order imbalance signal
- qmp_classify: Quote midpoint classification
- zscore_trigger: Z-score mean reversion trigger
- streak_trigger: Momentum streak trigger
"""

from .retail import compute_retail_imbalance, qmp_classify
from .triggers import zscore_trigger, streak_trigger, compute_zscore, compute_streak

__all__ = [
    "compute_retail_imbalance",
    "qmp_classify",
    "zscore_trigger",
    "streak_trigger",
    "compute_zscore",
    "compute_streak",
]
