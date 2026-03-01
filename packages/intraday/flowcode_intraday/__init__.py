"""flowcode_intraday — Intraday corporate bond microstructure analytics.

Submodules::

    flowcode_intraday.analytics  — Quote schemas, carry analytics, FV deviation,
                                   cross-source spreads, intraday dynamics,
                                   liquidity, cross-venue, spread profiles
    flowcode_intraday.filters    — Composite quote filters, TRACE transaction/agg
                                   filters, FV-anchored filters, unified interface
"""

from .analytics import (
    CompositeSource,
    PriceSpace,
    ResetPolicy,
    IntradayQuoteSchema,
    IntradayFVSchema,
    JoinedIntradaySchema,
    bid_ask_spread,
    quote_staleness,
    bid_ask_regime,
    carry_per_unit_risk,
    carry_breakeven_spread_move,
    intraday_spread_range,
    cumulative_spread_move,
    cross_venue_price_dislocation,
    spread_time_profile,
)

from .filters import (
    TradeSide,
    TraceTradeSchema,
    TraceAggSchema,
    FullIntradaySchema,
    SignalType,
    detect_directional_pressure,
    filter_significant_quote_moves,
    filter_trace_big_prints,
    aggregate_trace_to_bins,
)

__all__ = [
    # analytics — schemas
    "CompositeSource",
    "PriceSpace",
    "ResetPolicy",
    "IntradayQuoteSchema",
    "IntradayFVSchema",
    "JoinedIntradaySchema",
    # analytics — functions
    "bid_ask_spread",
    "quote_staleness",
    "bid_ask_regime",
    "carry_per_unit_risk",
    "carry_breakeven_spread_move",
    "intraday_spread_range",
    "cumulative_spread_move",
    "cross_venue_price_dislocation",
    "spread_time_profile",
    # filters — schemas
    "TradeSide",
    "TraceTradeSchema",
    "TraceAggSchema",
    "FullIntradaySchema",
    "SignalType",
    # filters — functions
    "detect_directional_pressure",
    "filter_significant_quote_moves",
    "filter_trace_big_prints",
    "aggregate_trace_to_bins",
]
