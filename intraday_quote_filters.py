"""
Credit Bond Quote Movement & Trade Filters
============================================
Adapted from futures tick-data trade filters (filter_big_prints_on_ask/bid).

Three data layers, each with distinct filter types:

  Layer 1 – Composite dealer quotes (MA, TW, TM, CBBT)
  ─────────────────────────────────────────────────────
  Schema: merged_intraday_market_data — one row per (isin, time_bin).
  Columns: bid/mid/offer × {price, benchmark_spread, z_spread} × source.
  No volume, no direction. Filters detect *quote movements* as proxies:
    - Offer tightening → buying pressure
    - Bid widening     → selling pressure
    - Spread compression → liquidity event
  These were the original adaptation when we had no trade data.

  Layer 2 – TRACE transaction data
  ────────────────────────────────
  Schema: trace_intraday_trades — one row per disseminated trade.
  Columns: execution_time, price, yield, volume_par, side, contra_party_type.
  This is what the original futures code actually had: volume + direction.
  Filters are the TRUE analogues of the futures big-print filters:
    - Big prints by volume threshold + direction
    - Block trades above reporting cap
    - Volume surges (clustering in time)
    - Trade-vs-quote: execution price relative to contemporaneous composite quotes

  Layer 3 – TRACE aggregated to time bins
  ────────────────────────────────────────
  Schema: trace_agg — one row per (isin, time_bin), aggregated from raw TRACE.
  Columns: vwap, total_volume, buy_volume, sell_volume, trade_count, etc.
  Joinable with composite quotes. Filters detect flow imbalance and
  VWAP-vs-composite divergence.

  Layer 4 – Cross-venue / unified
  ──────────────────────────────
  Combines signals from all the above into a single directional assessment.
  This is the multi-venue version of the original filter_big_prints interface.

FV-anchored filters (require joined intraday_fv_and_greeks) are included
for model-based rich/cheap detection — the "big print" for carry strategies.

Schema: merged_intraday_market_data + intraday_fv_and_greeks + trace tables
"""
from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Optional, Sequence
from dataclasses import dataclass, field
from enum import Enum

from intraday_microstructure_analytics import (
    CompositeSource,
    PriceSpace,
    IntradayQuoteSchema,
    IntradayFVSchema,
    JoinedIntradaySchema,
    bid_ask_spread,
    quote_staleness,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TRACE Schemas
# ═══════════════════════════════════════════════════════════════════════════

class TradeSide(str, Enum):
    """TRACE trade direction from customer perspective.

    FINRA dissemination reports trades as:
      B  = customer bought from dealer  (dealer sold → hit on the offer)
      S  = customer sold to dealer       (dealer bought → hit on the bid)
      D  = dealer-to-dealer / interdealer (direction ambiguous)
      NA = side could not be determined (e.g., ATS reports)
    """
    CUSTOMER_BUY  = "B"
    CUSTOMER_SELL = "S"
    DEALER        = "D"
    UNKNOWN       = "NA"


@dataclass
class TraceTradeSchema:
    """Column mapping for transaction-level TRACE data.

    One row per disseminated trade. This is the data the original futures
    code always had — volume, direction, execution price — and the reason
    the original filter_big_prints_on_ask/bid worked directly.

    TRACE-specific quirks handled downstream:
      - Capped volumes: trades >$5MM/$1MM report as "5MM+", "1MM+".
        We store the numeric cap and a `volume_capped` flag.
      - Reporting delays: execution_time != report_time.
        Filters can gate on report_delay_s.
      - As-of corrections: as_of_indicator marks amendments.
      - Special price: away-from-market conditions (exercise, dividend).
    """
    # --- Primary Keys ---
    cusip:               str = "cusip"
    isin:                str = "isin"

    # --- Timing ---
    execution_time:      str = "execution_time"
    report_time:         str = "report_time"
    time_bin:            str = "time_bin"         # 5-min bin (derived, for joins)
    trade_date:          str = "trade_date"

    # --- Trade Economics ---
    price:               str = "price"            # clean price per $100 par
    yield_:              str = "yield"            # yield at trade price
    volume_par:          str = "volume_par"       # par amount, USD
    volume_capped:       str = "volume_capped"    # bool: True if reported as "5MM+"

    # --- Direction ---
    side:                str = "side"             # B/S/D/NA (TradeSide values)
    contra_party_type:   str = "rptd_cntpty_type" # C=customer, D=dealer

    # --- Quality Flags ---
    as_of_indicator:     str = "as_of_indicator"  # blank=original, R=reversal, C=correction
    special_price_flag:  str = "spcl_pr_ind"      # blank=normal, else away-from-market
    report_delay_s:      str = "report_delay_s"   # execution_time → report_time, seconds

    # --- Derived Fields (populated by preprocessing) ---
    benchmark_spread:    str = "trace_benchmark_spread"   # G-spread at trade price
    z_spread:            str = "trace_z_spread"           # Z-spread at trade price


@dataclass
class TraceAggSchema:
    """Column mapping for TRACE data aggregated to 5-min time bins.

    One row per (isin, time_bin). Produced by grouping raw TRACE trades
    and computing summary statistics. Joinable with merged_intraday_market_data
    on (isin/cusip, time_bin).

    This gives us the volumetric information at the same granularity as
    composite quotes — enabling direct comparison of volume-weighted
    execution levels vs. composite dealer quotes.
    """
    # --- Keys (shared with IntradayQuoteSchema) ---
    cusip:              str = "cusip"
    isin:               str = "isin"
    time_bin:           str = "time_bin"

    # --- Volume ---
    total_volume:       str = "trace_total_volume"
    buy_volume:         str = "trace_buy_volume"       # customer buys
    sell_volume:        str = "trace_sell_volume"       # customer sells
    dealer_volume:      str = "trace_dealer_volume"     # interdealer
    trade_count:        str = "trace_trade_count"
    buy_count:          str = "trace_buy_count"
    sell_count:         str = "trace_sell_count"

    # --- Price/Spread Aggregates ---
    vwap_price:         str = "trace_vwap_price"
    vwap_spread:        str = "trace_vwap_spread"      # volume-weighted avg benchmark spread
    vwap_z_spread:      str = "trace_vwap_z_spread"    # volume-weighted avg Z-spread
    high_price:         str = "trace_high_price"
    low_price:          str = "trace_low_price"
    last_price:         str = "trace_last_price"

    # --- Flow Metrics ---
    net_imbalance:      str = "trace_net_imbalance"    # buy_volume - sell_volume
    imbalance_ratio:    str = "trace_imbalance_ratio"  # net / total, ∈ [-1, 1]
    max_single_trade:   str = "trace_max_single_trade" # largest trade in bin


@dataclass
class FullIntradaySchema:
    """Combined schema for quotes + FV + TRACE aggregate, all on (isin, time_bin).

    This is the widest join: composite quotes from all 4 venues, Fincad model
    outputs recomputed every 5 mins, and TRACE volume/flow aggregated to the
    same grid. Every filter in this module operates on one or more of these
    sub-schemas.
    """
    quote: IntradayQuoteSchema = field(default_factory=IntradayQuoteSchema)
    fv:    IntradayFVSchema    = field(default_factory=IntradayFVSchema)
    trace: TraceAggSchema      = field(default_factory=TraceAggSchema)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Composite Quote Filters (unchanged from original adaptation)
# ═══════════════════════════════════════════════════════════════════════════
#
# These operate on merged_intraday_market_data (composite dealer quotes).
# No volume, no direction — they detect *quote movements* as flow proxies.
# All 4 sources (MA, TW, TM, CBBT) supported via CompositeSource.
# ═══════════════════════════════════════════════════════════════════════════

def filter_significant_offer_tightening(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    min_move_bps: float = 2.0,
    max_staleness_s: Optional[float] = 600.0,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.DataFrame:
    """Filter for significant offer-side improvements (buying pressure).

    Credit analogue of filter_big_prints_on_ask (TradeType==2, Volume>=threshold).
    A dealer materially tightening their offer is the quote-space signature
    of buying flow. In spread space, tightening = negative Δoffer.

    For the TRUE big-print filter using actual trade volume, use
    filter_trace_big_prints(side='buy') instead.
    """
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        offer_col = schema.spread_col("offer", source)
    elif price_space == PriceSpace.Z_SPREAD:
        offer_col = schema.z_spread_col("offer", source)
    else:
        raise ValueError("Use BENCHMARK_SPREAD or Z_SPREAD for spread-based filters")

    delta_offer = data[offer_col].diff()
    mask = delta_offer <= -min_move_bps

    if max_staleness_s is not None:
        staleness = quote_staleness(data, schema, source)
        stale_col = f"offer_staleness_s_{source.value}"
        if staleness[stale_col].isna().all():
            logger.warning(
                "filter_significant_offer_tightening: all staleness values are NaN "
                "for source '%s'. Staleness filter will exclude all rows.",
                source.value,
            )
        mask &= staleness[stale_col] <= max_staleness_s

    return data[mask].copy()


def filter_significant_bid_widening(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    min_move_bps: float = 2.0,
    max_staleness_s: Optional[float] = 600.0,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.DataFrame:
    """Filter for significant bid-side widening (selling pressure).

    Credit analogue of filter_big_prints_on_bid (TradeType==1, Volume>=threshold).
    Dealer widening bid = selling pressure. In spread space, widening = positive Δbid.

    For the TRUE big-print filter using actual trade volume, use
    filter_trace_big_prints(side='sell') instead.
    """
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        bid_col = schema.spread_col("bid", source)
    elif price_space == PriceSpace.Z_SPREAD:
        bid_col = schema.z_spread_col("bid", source)
    else:
        raise ValueError("Use BENCHMARK_SPREAD or Z_SPREAD for spread-based filters")

    delta_bid = data[bid_col].diff()
    mask = delta_bid >= min_move_bps

    if max_staleness_s is not None:
        staleness = quote_staleness(data, schema, source)
        stale_col = f"bid_staleness_s_{source.value}"
        if staleness[stale_col].isna().all():
            logger.warning(
                "filter_significant_bid_widening: all staleness values are NaN "
                "for source '%s'. Staleness filter will exclude all rows.",
                source.value,
            )
        mask &= staleness[stale_col] <= max_staleness_s

    return data[mask].copy()


def filter_significant_quote_moves(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    min_move_bps: float = 2.0,
    side: str = "both",
    max_staleness_s: Optional[float] = 600.0,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.DataFrame:
    """Filter for significant quote moves on bid, offer, or both sides.

    Unified interface. Adds 'move_direction' column:
    'tightening' (buying), 'widening' (selling), or 'both'.

    Parameters
    ----------
    side : str
        'bid' = bid widening only (selling pressure, in spread space)
        'offer' = offer tightening only (buying pressure, in spread space)
        'both' = either direction
    """
    _VALID_SIDES = {"bid", "offer", "both"}
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be one of {_VALID_SIDES}, got '{side}'")

    if price_space == PriceSpace.BENCHMARK_SPREAD:
        bid_col = schema.spread_col("bid", source)
        offer_col = schema.spread_col("offer", source)
    elif price_space == PriceSpace.Z_SPREAD:
        bid_col = schema.z_spread_col("bid", source)
        offer_col = schema.z_spread_col("offer", source)
    else:
        raise ValueError("Use BENCHMARK_SPREAD or Z_SPREAD")

    delta_bid = data[bid_col].diff()
    delta_offer = data[offer_col].diff()

    bid_widening = delta_bid >= min_move_bps
    offer_tightening = delta_offer <= -min_move_bps

    if max_staleness_s is not None:
        staleness = quote_staleness(data, schema, source)
        bid_widening &= staleness[f"bid_staleness_s_{source.value}"] <= max_staleness_s
        offer_tightening &= staleness[f"offer_staleness_s_{source.value}"] <= max_staleness_s

    if side == "bid":
        mask = bid_widening
    elif side == "offer":
        mask = offer_tightening
    else:
        mask = bid_widening | offer_tightening

    result = data[mask].copy()
    result["move_direction"] = np.where(
        bid_widening[mask].values & ~offer_tightening[mask].values,
        "widening",
        np.where(
            offer_tightening[mask].values & ~bid_widening[mask].values,
            "tightening",
            "both",
        ),
    )
    result["delta_bid_bps"] = delta_bid[mask].values
    result["delta_offer_bps"] = delta_offer[mask].values
    return result


def filter_significant_quote_moves_multi_venue(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    sources: Sequence[CompositeSource] = (
        CompositeSource.MA, CompositeSource.TW,
        CompositeSource.TM, CompositeSource.CBBT,
    ),
    min_move_bps: float = 2.0,
    min_venues_confirming: int = 2,
    max_staleness_s: Optional[float] = 600.0,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.DataFrame:
    """Filter for significant quote moves confirmed across multiple venues.

    Higher conviction filter: a move that only shows on MA but not TW/TM/CBBT
    may be noise or a single dealer. A move confirmed by >= min_venues_confirming
    sources indicates genuine market-wide repricing.

    Parameters
    ----------
    sources : Sequence[CompositeSource]
        Which composite sources to check.
    min_venues_confirming : int
        Minimum number of sources that must independently show a significant
        move (same direction) in the same time_bin.

    Returns
    -------
    pd.DataFrame
        Filtered rows with 'confirming_venues_buy', 'confirming_venues_sell',
        'move_direction', and per-source delta columns.
    """
    n = len(data)
    buy_counts = np.zeros(n, dtype=int)
    sell_counts = np.zeros(n, dtype=int)
    source_deltas = {}

    for src in sources:
        if price_space == PriceSpace.BENCHMARK_SPREAD:
            bid_col = schema.spread_col("bid", src)
            offer_col = schema.spread_col("offer", src)
        else:
            bid_col = schema.z_spread_col("bid", src)
            offer_col = schema.z_spread_col("offer", src)

        # Skip venues whose columns don't exist in this dataset
        if bid_col not in data.columns or offer_col not in data.columns:
            logger.warning(
                "filter_significant_quote_moves_multi_venue: columns '%s'/'%s' "
                "not found, skipping source %s.",
                bid_col, offer_col, src.value,
            )
            continue

        d_bid = data[bid_col].diff()
        d_offer = data[offer_col].diff()

        bid_widen = d_bid >= min_move_bps
        offer_tight = d_offer <= -min_move_bps

        if max_staleness_s is not None:
            try:
                stale = quote_staleness(data, schema, src)
                bid_widen &= stale[f"bid_staleness_s_{src.value}"] <= max_staleness_s
                offer_tight &= stale[f"offer_staleness_s_{src.value}"] <= max_staleness_s
            except KeyError:
                pass  # staleness columns missing for this source

        sell_counts += bid_widen.values.astype(int)
        buy_counts += offer_tight.values.astype(int)
        source_deltas[f"delta_bid_{src.value}"] = d_bid
        source_deltas[f"delta_offer_{src.value}"] = d_offer

    # At least min_venues_confirming must agree on the SAME direction
    buy_confirmed = buy_counts >= min_venues_confirming
    sell_confirmed = sell_counts >= min_venues_confirming
    mask = buy_confirmed | sell_confirmed

    result = data[mask].copy()
    result["confirming_venues_buy"] = buy_counts[mask]
    result["confirming_venues_sell"] = sell_counts[mask]
    result["move_direction"] = np.where(
        buy_confirmed[mask] & ~sell_confirmed[mask],
        "tightening",
        np.where(
            sell_confirmed[mask] & ~buy_confirmed[mask],
            "widening",
            "both",
        ),
    )
    for col_name, series in source_deltas.items():
        result[col_name] = series[mask].values

    return result


def filter_spread_compression_events(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    min_compression_bps: float = 1.0,
    max_staleness_s: Optional[float] = 600.0,
) -> pd.DataFrame:
    """Filter for bid-ask spread compression events.

    Bid-ask narrowing = increased competition / liquidity. The quote-space
    analogue of a surge in volume at a price level.

    Returns
    -------
    pd.DataFrame
        Filtered rows with 'ba_compression_bps' (positive = magnitude of
        compression) and 'ba_after' (spread level after compression).
    """
    ba = bid_ask_spread(data, schema, source, PriceSpace.BENCHMARK_SPREAD)
    delta_ba = ba.diff()
    mask = delta_ba <= -min_compression_bps

    if max_staleness_s is not None:
        staleness = quote_staleness(data, schema, source)
        mask &= staleness[f"bid_staleness_s_{source.value}"] <= max_staleness_s
        mask &= staleness[f"offer_staleness_s_{source.value}"] <= max_staleness_s

    result = data[mask].copy()
    result["ba_compression_bps"] = -delta_ba[mask].values  # positive = compression magnitude
    result["ba_after"] = ba[mask].values
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: TRACE Transaction-Level Filters
# ═══════════════════════════════════════════════════════════════════════════
#
# These operate on trace_intraday_trades — one row per disseminated trade.
# This is where the ORIGINAL futures filter logic lives: we have volume
# and direction, so filter_trace_big_prints is the TRUE analogue of
# filter_big_prints_on_ask/bid. The quote-movement filters above were
# always a proxy for this.
# ═══════════════════════════════════════════════════════════════════════════

def _clean_trace(
    trades: pd.DataFrame,
    schema: TraceTradeSchema = TraceTradeSchema(),
    exclude_special_price: bool = True,
    exclude_corrections: bool = True,
    max_report_delay_s: Optional[float] = None,
) -> pd.DataFrame:
    """Standard TRACE quality filters applied before any analytics.

    Removes:
      - Special-price trades (exercise, dividend, away-from-market)
      - As-of corrections/reversals (keeps only original reports)
      - Excessively late reports (stale prints)
    """
    mask = pd.Series(True, index=trades.index)

    if exclude_special_price and schema.special_price_flag in trades.columns:
        # Special price indicator: blank or NaN = normal
        spcl = trades[schema.special_price_flag].fillna("")
        mask &= (spcl == "") | (spcl == "N")

    if exclude_corrections and schema.as_of_indicator in trades.columns:
        asof = trades[schema.as_of_indicator].fillna("")
        mask &= asof == ""

    if max_report_delay_s is not None and schema.report_delay_s in trades.columns:
        mask &= trades[schema.report_delay_s] <= max_report_delay_s

    return trades[mask]


def filter_trace_big_prints(
    trades: pd.DataFrame,
    schema: TraceTradeSchema = TraceTradeSchema(),
    min_volume: float = 1_000_000,
    side: str = "both",
    include_dealer: bool = False,
    include_capped: bool = True,
    exclude_special_price: bool = True,
    max_report_delay_s: Optional[float] = 3600.0,
) -> pd.DataFrame:
    """The TRUE analogue of filter_big_prints_on_ask / filter_big_prints_on_bid.

    In futures: TradeType == 1 or 2, Volume >= threshold.
    Here:       side == B or S,      volume_par >= threshold.

    This is what we always wanted but couldn't do with composite quotes alone.
    TRACE gives us volume AND direction on actual executions.

    Parameters
    ----------
    min_volume : float
        Minimum par amount in USD. TRACE disseminates trades with capped
        volumes ($5MM+ for IG, $1MM+ for HY), so some qualifying prints
        will report exactly at the cap. Set include_capped=True (default)
        to include these.
    side : str
        'buy'  = customer buys (dealer sold, hit on offer) — buying pressure
        'sell' = customer sells (dealer bought, hit on bid) — selling pressure
        'both' = either direction
    include_dealer : bool
        Include dealer-to-dealer trades (side='D'). Usually excluded since
        D2D direction is ambiguous.
    include_capped : bool
        Include trades where volume_capped=True. These are above the
        dissemination cap — genuinely large prints whose exact size
        is unknown but exceeds the threshold.

    Returns
    -------
    pd.DataFrame
        Filtered trades with added 'trade_direction' column:
        'buy' (customer bought from dealer) or 'sell' (customer sold to dealer).
    """
    df = _clean_trace(trades, schema, exclude_special_price, max_report_delay_s=max_report_delay_s)

    trade_side = df[schema.side].str.upper()

    # Volume filter
    vol = df[schema.volume_par]
    vol_mask = vol >= min_volume
    if include_capped and schema.volume_capped in df.columns:
        vol_mask |= df[schema.volume_capped].astype(bool)

    # Direction filter
    if side == "buy":
        dir_mask = trade_side == TradeSide.CUSTOMER_BUY.value
    elif side == "sell":
        dir_mask = trade_side == TradeSide.CUSTOMER_SELL.value
    elif side == "both":
        dir_mask = trade_side.isin([TradeSide.CUSTOMER_BUY.value, TradeSide.CUSTOMER_SELL.value])
        if include_dealer:
            dir_mask |= trade_side == TradeSide.DEALER.value
    else:
        raise ValueError(f"side must be 'buy', 'sell', or 'both', got: {side}")

    mask = vol_mask & dir_mask
    result = df[mask].copy()

    result["trade_direction"] = np.where(
        result[schema.side].str.upper() == TradeSide.CUSTOMER_BUY.value,
        "buy",
        np.where(
            result[schema.side].str.upper() == TradeSide.CUSTOMER_SELL.value,
            "sell",
            "dealer",
        ),
    )
    return result


def filter_trace_block_trades(
    trades: pd.DataFrame,
    schema: TraceTradeSchema = TraceTradeSchema(),
    min_volume: float = 5_000_000,
    **kwargs,
) -> pd.DataFrame:
    """Filter for block-size TRACE prints.

    Block trades (>=$5MM par for IG, >=$1MM for HY) are the corporate bond
    equivalent of futures block trades. These receive delayed dissemination
    (up to 15 min for IG, end-of-day for HY) but carry strong directional
    information — the institutional flow that moves markets.

    This is filter_trace_big_prints with a higher threshold and defaults
    appropriate for block-trade analysis.
    """
    return filter_trace_big_prints(
        trades, schema,
        min_volume=min_volume,
        include_capped=True,
        **kwargs,
    )


def filter_trace_volume_surge(
    trades: pd.DataFrame,
    schema: TraceTradeSchema = TraceTradeSchema(),
    window_minutes: float = 15.0,
    min_total_volume: float = 5_000_000,
    min_trade_count: int = 3,
    exclude_special_price: bool = True,
) -> pd.DataFrame:
    """Detect clusters of TRACE prints within a rolling time window.

    A single trade may be noise or a crossing. Multiple trades concentrated
    in a short window — especially in the same direction — indicates genuine
    flow. This is the credit analogue of detecting volume bars/surges in
    futures tick data.

    Parameters
    ----------
    window_minutes : float
        Rolling window size. 15 min default aligns with IG block trade
        dissemination delay.
    min_total_volume : float
        Minimum aggregate volume across all trades in the window.
    min_trade_count : int
        Minimum number of distinct trades in the window.

    Returns
    -------
    pd.DataFrame
        Rows where surges were detected, plus 'surge_volume', 'surge_count',
        'surge_buy_volume', 'surge_sell_volume', 'surge_imbalance'.
    """
    df = _clean_trace(trades, schema, exclude_special_price)
    df = df.sort_values(schema.execution_time).reset_index(drop=True)

    exec_time = pd.to_datetime(df[schema.execution_time])
    vol = df[schema.volume_par].values
    side = df[schema.side].str.upper().values
    window_td = np.timedelta64(int(window_minutes * 60), "s")

    n = len(df)
    surge_vol = np.zeros(n, dtype=float)
    surge_count = np.zeros(n, dtype=int)
    surge_buy = np.zeros(n, dtype=float)
    surge_sell = np.zeros(n, dtype=float)

    # Group by bond to avoid cross-contamination
    id_col = schema.cusip if schema.cusip in df.columns else schema.isin
    for _, grp in df.groupby(id_col):
        if len(grp) < min_trade_count:
            continue

        idx = grp.index.values  # integer positions after reset_index
        grp_times = exec_time.values[idx]
        grp_vol = vol[idx]
        grp_side = side[idx]

        left = 0
        for right in range(len(idx)):
            while grp_times[right] - grp_times[left] > window_td:
                left += 1

            win_vol = grp_vol[left:right + 1]
            win_side = grp_side[left:right + 1]

            pos = idx[right]
            surge_vol[pos] = win_vol.sum()
            surge_count[pos] = right - left + 1
            surge_buy[pos] = win_vol[win_side == TradeSide.CUSTOMER_BUY.value].sum()
            surge_sell[pos] = win_vol[win_side == TradeSide.CUSTOMER_SELL.value].sum()

    mask = (surge_vol >= min_total_volume) & (surge_count >= min_trade_count)
    result = df[mask].copy()
    result["surge_volume"] = surge_vol[mask]
    result["surge_count"] = surge_count[mask]
    result["surge_buy_volume"] = surge_buy[mask]
    result["surge_sell_volume"] = surge_sell[mask]
    total_directional = surge_buy[mask] + surge_sell[mask]
    result["surge_imbalance"] = np.where(
        total_directional > 0,
        (surge_buy[mask] - surge_sell[mask]) / total_directional,
        0.0,
    )
    return result


def filter_trace_vs_quotes(
    trades: pd.DataFrame,
    quotes: pd.DataFrame,
    trade_schema: TraceTradeSchema = TraceTradeSchema(),
    quote_schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    price_space: PriceSpace = PriceSpace.CLEAN_PRICE,
    min_through_bps: float = 0.0,
    exclude_special_price: bool = True,
) -> pd.DataFrame:
    """Classify TRACE trades relative to contemporaneous composite quotes.

    This is the MOST direct translation of the original futures filter:
      - Futures: trade executes at the ask → aggressive buyer
      - Credit:  TRACE trade at/through composite offer → aggressive buyer

    Joins each TRACE trade with the composite quote prevailing at that
    time_bin, then classifies:
      'at_offer'    : trade price >= composite offer (aggressive buy)
      'at_bid'      : trade price <= composite bid   (aggressive sell)
      'inside'      : trade price between bid and offer
      'through_offer': trade price > offer by >= min_through_bps (very aggressive buy)
      'through_bid'  : trade price < bid by >= min_through_bps (very aggressive sell)

    In spread space the inequality directions flip because tighter spread =
    higher price (buyer willing to accept less spread = paying more).

    Parameters
    ----------
    trades : pd.DataFrame
        TRACE transaction-level DataFrame.
    quotes : pd.DataFrame
        merged_intraday_market_data DataFrame (composite quotes).
    source : CompositeSource
        Which composite source's bid/offer to compare against.
    price_space : PriceSpace
        CLEAN_PRICE for dollar-price comparison,
        BENCHMARK_SPREAD or Z_SPREAD for spread-space comparison.
    min_through_bps : float
        For 'through' classification, how far past bid/offer the trade must be.

    Returns
    -------
    pd.DataFrame
        Trades with 'trade_vs_quote', 'distance_from_mid_bps',
        'composite_bid', 'composite_offer', 'composite_mid'.
    """
    cleaned = _clean_trace(trades, trade_schema, exclude_special_price)

    # Resolve quote columns based on price_space
    if price_space == PriceSpace.CLEAN_PRICE:
        bid_col = quote_schema.price_col("bid", source)
        offer_col = quote_schema.price_col("offer", source)
        mid_col = quote_schema.price_col("mid", source)
        trade_price_col = trade_schema.price
    elif price_space == PriceSpace.BENCHMARK_SPREAD:
        bid_col = quote_schema.spread_col("bid", source)
        offer_col = quote_schema.spread_col("offer", source)
        mid_col = quote_schema.spread_col("mid", source)
        trade_price_col = trade_schema.benchmark_spread
    elif price_space == PriceSpace.Z_SPREAD:
        bid_col = quote_schema.z_spread_col("bid", source)
        offer_col = quote_schema.z_spread_col("offer", source)
        mid_col = quote_schema.z_spread_col("mid", source)
        trade_price_col = trade_schema.z_spread
    else:
        raise ValueError(f"Use market data price spaces, not FV: {price_space}")

    # Determine join key: prefer cusip, fall back to isin
    join_key = (
        trade_schema.cusip if trade_schema.cusip in cleaned.columns
        else trade_schema.isin
    )
    quote_join_key = (
        quote_schema.cusip if quote_schema.cusip in quotes.columns
        else quote_schema.isin
    )

    # Join trades with prevailing quotes at the same time_bin
    quote_cols = [quote_join_key, quote_schema.time_bin, bid_col, offer_col, mid_col]
    quote_sub = quotes[[c for c in quote_cols if c in quotes.columns]].drop_duplicates(
        subset=[quote_join_key, quote_schema.time_bin]
    )

    result = cleaned.merge(
        quote_sub,
        left_on=[join_key, trade_schema.time_bin],
        right_on=[quote_join_key, quote_schema.time_bin],
        how="inner",
        suffixes=("", "_quote"),
    )

    if len(result) == 0:
        result["trade_vs_quote"] = []
        result["distance_from_mid_bps"] = []
        return result

    tp = result[trade_price_col].values
    bid = result[bid_col].values
    offer = result[offer_col].values
    mid = result[mid_col].values

    # Classification depends on price_space:
    # In price space: higher price = buyer more aggressive
    # In spread space: LOWER spread = buyer more aggressive (tighter spread = higher price)
    if price_space == PriceSpace.CLEAN_PRICE:
        through_offer = tp >= offer + min_through_bps * 0.01  # bps of $100
        at_offer = (tp >= offer) & ~through_offer
        through_bid = tp <= bid - min_through_bps * 0.01
        at_bid = (tp <= bid) & ~through_bid
        dist_from_mid = (tp - mid) * 100  # convert to bps of par
    else:
        # Spread space: lower spread = more aggressive buy
        through_offer = tp <= offer - min_through_bps
        at_offer = (tp <= offer) & ~through_offer
        through_bid = tp >= bid + min_through_bps
        at_bid = (tp >= bid) & ~through_bid
        dist_from_mid = mid - tp  # positive = tighter than mid = buying pressure

    classification = np.full(len(result), "inside", dtype=object)
    classification[through_offer] = "through_offer"
    classification[at_offer & ~through_offer] = "at_offer"
    classification[through_bid] = "through_bid"
    classification[at_bid & ~through_bid] = "at_bid"

    result["trade_vs_quote"] = classification
    result["distance_from_mid_bps"] = dist_from_mid
    result["composite_bid"] = bid
    result["composite_offer"] = offer
    result["composite_mid"] = mid

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: TRACE Aggregated (Time-Bin) Filters
# ═══════════════════════════════════════════════════════════════════════════
#
# These operate on TRACE data aggregated to the same 5-min bins as
# composite quotes — enabling direct join and comparison.
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_trace_to_bins(
    trades: pd.DataFrame,
    schema: TraceTradeSchema = TraceTradeSchema(),
    agg_schema: TraceAggSchema = TraceAggSchema(),
    bin_freq: str = "5min",
    exclude_special_price: bool = True,
) -> pd.DataFrame:
    """Aggregate raw TRACE trades into time bins matching composite quote grid.

    Produces one row per (isin/cusip, time_bin) with summary statistics.
    The output is directly joinable with merged_intraday_market_data.

    Parameters
    ----------
    bin_freq : str
        Pandas frequency string for time binning. Default '5min' matches
        the composite quote grid.

    Returns
    -------
    pd.DataFrame
        Aggregated TRACE data with volume, VWAP, flow metrics per time bin.
    """
    df = _clean_trace(trades, schema, exclude_special_price)
    exec_time = pd.to_datetime(df[schema.execution_time])

    # Assign time bins
    df = df.copy()
    df[agg_schema.time_bin] = exec_time.dt.floor(bin_freq)

    # Determine groupby key
    id_col = schema.cusip if schema.cusip in df.columns else schema.isin
    agg_id_col = agg_schema.cusip if id_col == schema.cusip else agg_schema.isin

    side = df[schema.side].str.upper()
    vol = df[schema.volume_par]
    price = df[schema.price]

    is_buy = side == TradeSide.CUSTOMER_BUY.value
    is_sell = side == TradeSide.CUSTOMER_SELL.value
    is_dealer = side == TradeSide.DEALER.value

    df["_buy_vol"] = vol.where(is_buy, 0.0)
    df["_sell_vol"] = vol.where(is_sell, 0.0)
    df["_dealer_vol"] = vol.where(is_dealer, 0.0)
    df["_is_buy"] = is_buy.astype(int)
    df["_is_sell"] = is_sell.astype(int)
    df["_pv"] = price * vol  # for VWAP

    # Spread-based VWAP if available
    has_spread = schema.benchmark_spread in df.columns
    has_z = schema.z_spread in df.columns
    if has_spread:
        df["_sv"] = df[schema.benchmark_spread] * vol
    if has_z:
        df["_zv"] = df[schema.z_spread] * vol

    group_keys = [id_col, agg_schema.time_bin]
    grouped = df.groupby(group_keys, sort=True)

    # Core aggregation in a single .agg() call for alignment safety
    agg_spec = {
        schema.volume_par: [("_total_vol", "sum"), ("_count", "count"), ("_max_trade", "max")],
        "_buy_vol":        [("_buy_vol_sum", "sum")],
        "_sell_vol":       [("_sell_vol_sum", "sum")],
        "_dealer_vol":     [("_dealer_vol_sum", "sum")],
        "_is_buy":         [("_buy_count", "sum")],
        "_is_sell":        [("_sell_count", "sum")],
        schema.price:      [("_high", "max"), ("_low", "min"), ("_last", "last")],
        "_pv":             [("_pv_sum", "sum")],
    }
    if has_spread:
        agg_spec["_sv"] = [("_sv_sum", "sum")]
    if has_z:
        agg_spec["_zv"] = [("_zv_sum", "sum")]

    raw = grouped.agg(
        **{name: pd.NamedAgg(column=col, aggfunc=func) for col, specs in agg_spec.items() for name, func in specs}
    )

    # Flatten MultiIndex from groupby back to columns
    agg = raw.reset_index()
    rename_map = {
        id_col:        agg_id_col,
        "_total_vol":  agg_schema.total_volume,
        "_buy_vol_sum":  agg_schema.buy_volume,
        "_sell_vol_sum": agg_schema.sell_volume,
        "_dealer_vol_sum": agg_schema.dealer_volume,
        "_count":      agg_schema.trade_count,
        "_buy_count":  agg_schema.buy_count,
        "_sell_count":  agg_schema.sell_count,
        "_high":       agg_schema.high_price,
        "_low":        agg_schema.low_price,
        "_last":       agg_schema.last_price,
        "_max_trade":  agg_schema.max_single_trade,
    }
    agg = agg.rename(columns=rename_map)

    # VWAP
    total_vol = agg[agg_schema.total_volume]
    agg[agg_schema.vwap_price] = np.where(
        total_vol > 0, agg["_pv_sum"] / total_vol, np.nan
    )
    if has_spread:
        agg[agg_schema.vwap_spread] = np.where(
            total_vol > 0, agg["_sv_sum"] / total_vol, np.nan
        )
    if has_z:
        agg[agg_schema.vwap_z_spread] = np.where(
            total_vol > 0, agg["_zv_sum"] / total_vol, np.nan
        )

    # Flow metrics
    net = agg[agg_schema.buy_volume] - agg[agg_schema.sell_volume]
    agg[agg_schema.net_imbalance] = net
    directional = agg[agg_schema.buy_volume] + agg[agg_schema.sell_volume]
    agg[agg_schema.imbalance_ratio] = np.where(directional > 0, net / directional, 0.0)

    # Drop temp columns
    temp_cols = [c for c in agg.columns if c.startswith("_")]
    agg = agg.drop(columns=temp_cols, errors="ignore")

    return agg


def filter_trace_agg_imbalance(
    data: pd.DataFrame,
    schema: TraceAggSchema = TraceAggSchema(),
    min_imbalance_ratio: float = 0.5,
    min_volume: float = 500_000,
    direction: str = "both",
) -> pd.DataFrame:
    """Filter time bins with significant TRACE flow imbalance.

    The credit analogue of order-flow imbalance in futures. A bin where
    80% of volume is customer-buy indicates strong buying pressure.

    Parameters
    ----------
    min_imbalance_ratio : float
        Minimum |imbalance_ratio| to qualify. Range [0, 1].
        0.5 = buy_vol is 3x sell_vol (or vice versa).
    min_volume : float
        Minimum total volume in the bin — ignore thin bins.
    direction : str
        'buy'  = positive imbalance (net buying)
        'sell' = negative imbalance (net selling)
        'both' = either direction

    Returns
    -------
    pd.DataFrame
        Filtered rows with added 'flow_direction' column.
    """
    _VALID_DIRECTIONS = {"buy", "sell", "both"}
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got '{direction}'")

    ratio = data[schema.imbalance_ratio]
    vol_ok = data[schema.total_volume] >= min_volume

    if direction == "buy":
        mask = (ratio >= min_imbalance_ratio) & vol_ok
    elif direction == "sell":
        mask = (ratio <= -min_imbalance_ratio) & vol_ok
    else:
        mask = (ratio.abs() >= min_imbalance_ratio) & vol_ok

    result = data[mask].copy()
    result["flow_direction"] = np.where(
        result[schema.imbalance_ratio] >= 0, "buy", "sell"
    )
    return result


def filter_trace_vwap_vs_composite(
    data: pd.DataFrame,
    schema: FullIntradaySchema = FullIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
    min_deviation_bps: float = 3.0,
    min_volume: float = 500_000,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.DataFrame:
    """Filter for time bins where TRACE VWAP deviates from composite mid.

    When execution VWAP is significantly above/below the composite mid,
    it reveals that actual transactions are clearing away from indicative
    quotes — either the quotes are stale or there's hidden flow.

    Parameters
    ----------
    min_deviation_bps : float
        Minimum |VWAP - composite_mid| to qualify.
    price_space : PriceSpace
        Which price space to compare in.

    Returns
    -------
    pd.DataFrame
        Filtered rows with 'vwap_deviation_bps' and 'vwap_direction'.
    """
    # Composite mid
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        mid_col = schema.quote.spread_col("mid", source)
        vwap_col = schema.trace.vwap_spread
    elif price_space == PriceSpace.Z_SPREAD:
        mid_col = schema.quote.z_spread_col("mid", source)
        vwap_col = schema.trace.vwap_z_spread
    elif price_space == PriceSpace.CLEAN_PRICE:
        mid_col = schema.quote.price_col("mid", source)
        vwap_col = schema.trace.vwap_price
    else:
        raise ValueError(f"Use market data price spaces: {price_space}")

    if mid_col not in data.columns or vwap_col not in data.columns:
        logger.warning(
            "filter_trace_vwap_vs_composite: required columns '%s' or '%s' "
            "not found. Returning empty result.",
            mid_col, vwap_col,
        )
        return data.iloc[:0].copy()

    composite_mid = data[mid_col]
    vwap = data[vwap_col]
    vol_ok = data[schema.trace.total_volume] >= min_volume

    if price_space == PriceSpace.CLEAN_PRICE:
        # Price space: deviation in bps of $100 par
        deviation = (vwap - composite_mid) * 100
    else:
        # Spread space: deviation in bps
        # Negative deviation = VWAP tighter than composite (buying pressure)
        deviation = vwap - composite_mid

    mask = (deviation.abs() >= min_deviation_bps) & vol_ok

    result = data[mask].copy()
    result["vwap_deviation_bps"] = deviation[mask].values
    result["vwap_direction"] = np.where(
        deviation[mask].values >= 0,
        "cheap" if price_space != PriceSpace.CLEAN_PRICE else "above",
        "rich" if price_space != PriceSpace.CLEAN_PRICE else "below",
    )
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: FV-Anchored Filters (model-based rich/cheap)
# ═══════════════════════════════════════════════════════════════════════════
#
# These require the joined data with intraday_fv_and_greeks. The "big
# print" for model-driven / carry strategies.
# ═══════════════════════════════════════════════════════════════════════════

def filter_rich_cheap_extremes(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
    min_deviation_bps: float = 5.0,
    direction: str = "both",
) -> pd.DataFrame:
    """Filter for bonds trading significantly rich or cheap to Fincad FV.

    Instead of "show me big trades," this is "show me big mispricings."

    Parameters
    ----------
    min_deviation_bps : float
        Minimum |venue_z - model_z| to qualify.
    direction : str
        'rich'  = venue tighter than model (negative deviation)
        'cheap' = venue wider than model (positive deviation)
        'both'  = either direction

    Returns
    -------
    pd.DataFrame
        Filtered rows with 'fv_deviation_bps' and 'fv_direction'.
    """
    _VALID_DIRECTIONS = {"rich", "cheap", "both"}
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got '{direction}'")

    deviation = data[schema.fv.venue_z_col(source)] - data[schema.fv.fv_z_spread]

    nan_count = deviation.isna().sum()
    if nan_count > 0:
        logger.warning(
            "filter_rich_cheap_extremes: %d rows have NaN FV deviation "
            "and will be excluded. Check venue_z and fv_z_spread columns.",
            nan_count,
        )

    if direction == "cheap":
        mask = deviation >= min_deviation_bps
    elif direction == "rich":
        mask = deviation <= -min_deviation_bps
    else:
        mask = deviation.abs() >= min_deviation_bps

    result = data[mask].copy()
    result["fv_deviation_bps"] = deviation[mask].values
    result["fv_direction"] = np.where(deviation[mask].values > 0, "cheap", "rich")
    return result


def filter_rich_cheap_carry_scaled(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
    min_carry_days: float = 3.0,
    direction: str = "both",
) -> pd.DataFrame:
    """Filter for bonds whose FV deviation exceeds a carry-days threshold.

    Normalizes deviation by carry rate: a 2bp deviation on a 2yr with
    0.3bp/day carry is ~7 days (actionable). Same 2bp on a 30yr with
    2bp/day carry is 1 day (noise). This is the "big print" for carry strategies.

    For negative-carry bonds, the sign interpretation reverses.

    Parameters
    ----------
    min_carry_days : float
        Minimum |deviation| expressed in days of carry.
    direction : str
        'rich', 'cheap', or 'both'

    Returns
    -------
    pd.DataFrame
        Filtered rows with 'fv_deviation_bps', 'carry_days_deviation', 'fv_direction'.
    """
    _VALID_DIRECTIONS = {"rich", "cheap", "both"}
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got '{direction}'")

    dev_bps = data[schema.fv.venue_z_col(source)] - data[schema.fv.fv_z_spread]
    cr01 = data[schema.fv.unit_cr01]
    daily_carry = data[schema.fv.carry] + data[schema.fv.rolldown]

    small_carry = daily_carry.abs() <= 1e-10
    if small_carry.any():
        logger.warning(
            "filter_rich_cheap_carry_scaled: %d rows have |daily_carry| <= 1e-10 "
            "and will be excluded from carry-days filtering. These bonds may still "
            "have significant FV deviations in bps. Use filter_rich_cheap_extremes() instead.",
            small_carry.sum(),
        )

    dev_usd = dev_bps * cr01
    carry_days = dev_usd / daily_carry.where(~small_carry, np.nan)

    if direction == "cheap":
        mask = carry_days >= min_carry_days
    elif direction == "rich":
        mask = carry_days <= -min_carry_days
    else:
        mask = carry_days.abs() >= min_carry_days

    result = data[mask].copy()
    result["fv_deviation_bps"] = dev_bps[mask].values
    result["carry_days_deviation"] = carry_days[mask].values
    result["fv_direction"] = np.where(carry_days[mask].values > 0, "cheap", "rich")
    return result


def filter_trace_vs_fv(
    trades: pd.DataFrame,
    fv_data: pd.DataFrame,
    trade_schema: TraceTradeSchema = TraceTradeSchema(),
    fv_schema: IntradayFVSchema = IntradayFVSchema(),
    min_deviation_bps: float = 5.0,
    direction: str = "both",
) -> pd.DataFrame:
    """Filter TRACE trades that execute significantly rich or cheap to Fincad FV.

    Combines the directness of actual trade prices (from TRACE) with the
    model-based valuation (from Fincad). A $5MM customer buy at FV-10bps
    is far more interesting than the same trade at FV — this filter finds those.

    Joins on (cusip/isin, time_bin) so each trade is compared against the
    model Z-spread prevailing at that time.

    Parameters
    ----------
    trades : pd.DataFrame
        Raw TRACE transactions.
    fv_data : pd.DataFrame
        intraday_fv_and_greeks table.
    min_deviation_bps : float
        Minimum |trade_z - fv_z| to qualify.
    direction : str
        'rich', 'cheap', or 'both'

    Returns
    -------
    pd.DataFrame
        Filtered trades with 'fv_deviation_bps' and 'fv_direction'.
    """
    _VALID_DIRECTIONS = {"rich", "cheap", "both"}
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}, got '{direction}'")

    cleaned = _clean_trace(trades, trade_schema)

    if trade_schema.z_spread not in cleaned.columns:
        raise ValueError(
            f"TRACE trades must have Z-spread column '{trade_schema.z_spread}'. "
            "Run spread computation preprocessing before calling this filter."
        )

    # Join key
    trade_id = trade_schema.cusip if trade_schema.cusip in cleaned.columns else trade_schema.isin
    fv_id = fv_schema.cusip if fv_schema.cusip in fv_data.columns else fv_schema.isin

    fv_sub = fv_data[[fv_id, fv_schema.timestamp, fv_schema.fv_z_spread]].copy()
    fv_sub["_time_bin"] = pd.to_datetime(fv_sub[fv_schema.timestamp]).dt.floor("5min")
    fv_sub = fv_sub.drop_duplicates(subset=[fv_id, "_time_bin"])

    merged = cleaned.merge(
        fv_sub,
        left_on=[trade_id, trade_schema.time_bin],
        right_on=[fv_id, "_time_bin"],
        how="inner",
        suffixes=("", "_fv"),
    )

    if len(merged) == 0:
        merged["fv_deviation_bps"] = []
        merged["fv_direction"] = []
        return merged

    deviation = merged[trade_schema.z_spread] - merged[fv_schema.fv_z_spread]

    if direction == "cheap":
        mask = deviation >= min_deviation_bps
    elif direction == "rich":
        mask = deviation <= -min_deviation_bps
    else:
        mask = deviation.abs() >= min_deviation_bps

    result = merged[mask].copy()
    result["fv_deviation_bps"] = deviation[mask].values
    result["fv_direction"] = np.where(deviation[mask].values > 0, "cheap", "rich")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: Cross-Venue Unified Interface
# ═══════════════════════════════════════════════════════════════════════════
#
# Combines signals from composite quotes, TRACE, and FV model into a
# single directional pressure assessment. This is the multi-venue
# version of the original filter_big_prints interface.
# ═══════════════════════════════════════════════════════════════════════════

class SignalType(str, Enum):
    """Source of a directional pressure signal."""
    QUOTE_MOVE     = "quote_move"        # composite quote movement
    TRACE_PRINT    = "trace_print"       # individual TRACE trade
    TRACE_FLOW     = "trace_flow"        # aggregated TRACE imbalance
    TRACE_VS_QUOTE = "trace_vs_quote"    # trade execution vs quote level
    FV_DEVIATION   = "fv_deviation"      # model-based rich/cheap


def detect_directional_pressure(
    quotes: pd.DataFrame,
    trace_agg: Optional[pd.DataFrame] = None,
    schema: FullIntradaySchema = FullIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
    min_quote_move_bps: float = 2.0,
    min_trace_imbalance: float = 0.5,
    min_trace_volume: float = 500_000,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
    max_staleness_s: Optional[float] = 600.0,
) -> pd.DataFrame:
    """Unified directional pressure detection across all available data.

    Evaluates each time bin for directional evidence from:
      1. Composite quote movements (all available sources)
      2. TRACE aggregated flow imbalance (if trace_agg provided)

    Returns one row per time bin with:
      - 'pressure_direction': 'buy', 'sell', 'mixed', or 'none'
      - 'pressure_score': in [-1, 1], aggregated across signal types
      - 'signal_count': number of independent signals confirming direction

    The score is a simple signed average of available signals, each
    normalized to [-1, 1]:
      - Quote: sign(delta_mid) x min(|delta_mid| / 5bps, 1)
      - TRACE imbalance_ratio: as-is, already in [-1, 1]

    Parameters
    ----------
    quotes : pd.DataFrame
        merged_intraday_market_data (required).
    trace_agg : pd.DataFrame or None
        TRACE aggregated to time bins (optional, enriches signal).
    """
    n = len(quotes)
    signals = np.zeros(n, dtype=float)
    counts = np.zeros(n, dtype=int)

    # --- Quote-based signal ---
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        mid_col = schema.quote.spread_col("mid", source)
    elif price_space == PriceSpace.Z_SPREAD:
        mid_col = schema.quote.z_spread_col("mid", source)
    else:
        mid_col = schema.quote.price_col("mid", source)

    if mid_col in quotes.columns:
        delta_mid = quotes[mid_col].diff()

        if price_space == PriceSpace.CLEAN_PRICE:
            # Higher price = buying. Normalize: 5bps of par = full signal
            quote_signal = (delta_mid * 100).clip(-5.0, 5.0) / 5.0
        else:
            # Lower spread = buying (tightening). Normalize: 5bps = full signal
            quote_signal = (-delta_mid).clip(-5.0, 5.0) / 5.0

        fresh_mask = np.ones(n, dtype=bool)
        if max_staleness_s is not None:
            try:
                stale = quote_staleness(quotes, schema.quote, source)
                bid_ok = stale[f"bid_staleness_s_{source.value}"] <= max_staleness_s
                off_ok = stale[f"offer_staleness_s_{source.value}"] <= max_staleness_s
                fresh_mask = (bid_ok & off_ok).values
            except KeyError:
                pass

        valid_quote = fresh_mask & (quote_signal.abs().values > 0.01)
        signals[valid_quote] += quote_signal.values[valid_quote]
        counts[valid_quote] += 1

    # --- TRACE flow signal ---
    if trace_agg is not None and schema.trace.imbalance_ratio in trace_agg.columns:
        # Join on time_bin
        join_key = schema.quote.time_bin
        trace_key = schema.trace.time_bin

        trace_sub = trace_agg.set_index(trace_key)
        if join_key in quotes.columns:
            aligned_ratio = quotes[join_key].map(
                trace_sub[schema.trace.imbalance_ratio]
            )
            aligned_vol = quotes[join_key].map(
                trace_sub[schema.trace.total_volume]
            )

            vol_ok = aligned_vol.fillna(0) >= min_trace_volume
            has_imb = aligned_ratio.abs().fillna(0) >= min_trace_imbalance

            trace_valid = vol_ok & has_imb
            signals[trace_valid.values] += aligned_ratio.fillna(0).values[trace_valid.values]
            counts[trace_valid.values] += 1

    # --- Aggregate ---
    total_signals = counts.astype(float)
    avg_signal = np.where(total_signals > 0, signals / total_signals, 0.0)

    result = quotes.copy()
    result["pressure_score"] = avg_signal
    result["signal_count"] = counts
    result["pressure_direction"] = np.where(
        avg_signal > 0.1,
        "buy",
        np.where(
            avg_signal < -0.1,
            "sell",
            np.where(
                counts > 0,
                "mixed",
                "none",
            ),
        ),
    )
    return result
