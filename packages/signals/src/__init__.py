"""FlowCode Signals Package - Signal computations.

This package provides signal computations that wrap core functionality.
All signals return pd.Series indexed by (date, cusip).

Public API:
- Retail Imbalance: compute_retail_imbalance
- QMP Classification: qmp_classify, qmp_classify_with_exclusion
- Retail Identification: is_retail_trade, is_subpenny, classify_retail_trades
- Triggers: zscore_trigger, streak_trigger
- Credit Signals: credit_pnl, range_position
"""

from .retail import (
    compute_retail_imbalance,
    compute_imbalance_from_volumes,
    qmp_classify,
    qmp_classify_with_exclusion,
    classify_trades_qmp,
    classify_trades_qmp_with_exclusion,
    is_retail_trade,
    is_subpenny,
    classify_retail_trades,
)
from .triggers import zscore_trigger, streak_trigger, compute_zscore, compute_streak
from .credit import credit_pnl, range_position, compute_range_position_rolling

__all__ = [
    # Retail imbalance
    "compute_retail_imbalance",
    "compute_imbalance_from_volumes",
    # QMP classification
    "qmp_classify",
    "qmp_classify_with_exclusion",
    "classify_trades_qmp",
    "classify_trades_qmp_with_exclusion",
    # Retail identification
    "is_retail_trade",
    "is_subpenny",
    "classify_retail_trades",
    # Triggers
    "zscore_trigger",
    "streak_trigger",
    "compute_zscore",
    "compute_streak",
    # Credit signals
    "credit_pnl",
    "range_position",
    "compute_range_position_rolling",
]
