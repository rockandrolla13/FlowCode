"""
Corporate Bond Intraday Microstructure Analytics
=================================================
Refactored from futures tick-data volume profile toolkit.
Adapted for intraday credit data combining two primary datasets:

  1. merged_intraday_market_data — 5-min composite dealer quotes from 4 sources:
       MA (MarketAxess), TW (TradeWeb), TM (TW/MA combined), CBBT (Bloomberg)
     Contains: bid/mid/offer × {price, benchmark_spread, z_spread} per source,
               liquidity scores, benchmark references, last-change timestamps.

  2. intraday_fv_and_greeks — Fincad model fair values recomputed every 5 mins:
       Same columns as eod_fv_and_greeks (fv_price, fv_z_spread, carry, rolldown,
       unit_cr01, dv01, cr01, csw01, duration, convexity, ytm, etc.)
     Plus venue-specific Z-spreads: ma_composite_mid_z_spread, tw_composite_mid_z_spread,
       tm_mid_z_spread, cbbt_mid_z_spread, composite_z_spread (weighted avg),
       data_source (primary venue for each observation).

Architecture:
  - The two tables join on (isin, time_bin/timestamp) at 5-min grain
  - merged_intraday_market_data provides raw quotes (what the market shows)
  - intraday_fv_and_greeks provides model values (what the model thinks)
  - The delta between them is the core signal: rich/cheap vs model in real time

Key differences from the EOD carry pipeline this was adapted from:
  1. Carry/rolldown/greeks are now live (recomputed every 5 mins), not EOD stale
  2. FV deviation (quote - model) is a real-time signal, not lagged T-1
  3. Multiple redundant price sources enable cross-venue arb detection
  4. No trade flow / RFQ data — this is quote + model data, not transactions
  5. No financing rates (GS_FINANCING_RATES) — omitted from intraday pipeline
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer 1: Schema
# ---------------------------------------------------------------------------

class CompositeSource(str, Enum):
    """The 4 composite quote sources."""
    MA   = "ma"
    TW   = "tw"
    TM   = "tm"
    CBBT = "cbbt"


class PriceSpace(str, Enum):
    """Price/spread representation for analytics axes.

    FV-based spaces come from intraday_fv_and_greeks (model outputs).
    Market-data spaces come from merged_intraday_market_data (raw quotes).

    BENCHMARK_SPREAD: G-spread vs on-the-run treasury (market data, all sources)
    Z_SPREAD:         Z-spread from market data (available for all sources; check data feed)
    FV_Z_SPREAD:      Fincad model Z-spread (from FV table)
    COMPOSITE_Z:      Weighted avg Z-spread across venues (from FV table)
    CLEAN_PRICE:      USD per $100 par (market data, all sources)
    FV_PRICE:         Fincad model clean price (from FV table)
    """
    BENCHMARK_SPREAD = "benchmark_spread"
    Z_SPREAD         = "z_spread"
    FV_Z_SPREAD      = "fv_z_spread"
    COMPOSITE_Z      = "composite_z_spread"
    CLEAN_PRICE      = "price"
    FV_PRICE         = "fv_price"


@dataclass
class IntradayQuoteSchema:
    """Column mapping for merged_intraday_market_data.

    Composite naming: {side}_{metric}_{source}
    e.g. bid_price_ma, mid_benchmark_spread_tw, offer_z_spread_cbbt
    """
    # --- Primary Keys ---
    timestamp:       str = "timestamp"
    time_bin:        str = "time_bin"
    isin:            str = "isin"
    cusip:           str = "cusip"

    # --- Bond Reference ---
    coupon:          str = "coupon"
    maturity:        str = "maturity"
    maturity_bucket: str = "maturity_bucket"
    interest_accrual_date: str = "interest_accrual_date"
    first_coupon_date:     str = "first_coupon_date"
    next_call_date:        str = "next_call_date"
    next_call_price:       str = "next_call_price"

    # --- Composite Timestamps ---
    composite_ts_ma:   str = "composite_timestamp_ma"
    composite_ts_tw:   str = "composite_timestamp_tw"
    composite_ts_tm:   str = "composite_timestamp_tm"
    composite_ts_cbbt: str = "composite_timestamp_cbbt"

    # --- Liquidity Scores ---
    liquidity_score_ma: str = "liquidity_score_ma"
    liquidity_score_tw: str = "liquidity_score_tw"

    # --- Benchmark References ---
    benchmark_isin_ma:   str = "benchmark_isin_ma"
    benchmark_cusip_tw:  str = "benchmark_cusip_tw"
    benchmark_cusip_tm:  str = "benchmark_cusip_tm"
    benchmark_isin_cbbt: str = "benchmark_isin_cbbt"

    # --- Last Change Timestamps ---
    bid_last_chg_ma:     str = "bid_last_chg_time_ma"
    offer_last_chg_ma:   str = "offer_last_chg_time_ma"
    bid_last_chg_tw:     str = "bid_last_chg_time_tw"
    offer_last_chg_tw:   str = "offer_last_chg_time_tw"
    bid_last_chg_tm:     str = "bid_last_chg_time_tm"
    offer_last_chg_tm:   str = "offer_last_chg_time_tm"
    bid_last_chg_cbbt:   str = "bid_last_chg_time_cbbt"
    offer_last_chg_cbbt: str = "offer_last_chg_time_cbbt"

    def col(self, side: str, metric: str, source: CompositeSource) -> str:
        """Build column name: {side}_{metric}_{source.value}.

        Parameters
        ----------
        side : str
            One of 'bid', 'mid', 'offer'.
        metric : str
            Metric name, e.g. 'price', 'benchmark_spread', 'z_spread'.
        source : CompositeSource
            Quote source.

        Returns
        -------
        str
            Formatted column name.
        """
        return f"{side}_{metric}_{source.value}"

    def price_col(self, side: str, source: CompositeSource) -> str:
        """Return price column name for given side and source."""
        return self.col(side, "price", source)

    def spread_col(self, side: str, source: CompositeSource) -> str:
        """Return benchmark spread column name for given side and source."""
        return self.col(side, "benchmark_spread", source)

    def z_spread_col(self, side: str, source: CompositeSource) -> str:
        """Return Z-spread column name for given side and source."""
        return self.col(side, "z_spread", source)


@dataclass
class IntradayFVSchema:
    """Column mapping for intraday_fv_and_greeks.

    Same columns as eod_fv_and_greeks, recomputed every 5 mins,
    plus venue-specific Z-spreads and a composite weighted average.
    """
    # --- Keys (join with quote schema on these) ---
    isin:            str = "isin"
    cusip:           str = "cusip"
    timestamp:       str = "timestamp"
    coupon:          str = "coupon"
    maturity:        str = "maturity"

    # --- Fincad Model Outputs (recomputed intraday) ---
    fv_price:        str = "fv_price"
    fv_price_today:  str = "fv_price_today"
    fv_z_spread:     str = "fv_z_spread"
    carry:           str = "carry"            # USD per $100 par per day
    rolldown:        str = "rolldown"         # USD per $100 par per day
    ytm:             str = "ytm"

    # --- Risk Sensitivities (recomputed intraday) ---
    dv01:            str = "dv01"
    cr01:            str = "cr01"
    unit_cr01:       str = "unit_cr01"        # |dP/dSpread| per $100 par per bp (positive by Fincad convention)
    csw01:           str = "csw01"
    duration:        str = "duration"
    convexity:       str = "convexity"

    # --- Venue-Specific Z-Spreads (inputs Fincad saw) ---
    ma_z_spread:     str = "ma_composite_mid_z_spread"
    tw_z_spread:     str = "tw_composite_mid_z_spread"
    tm_z_spread:     str = "tm_mid_z_spread"
    cbbt_z_spread:   str = "cbbt_mid_z_spread"

    # --- Composite ---
    composite_z:     str = "composite_z_spread"
    data_source:     str = "data_source"

    def venue_z_col(self, source: CompositeSource) -> str:
        """Map a CompositeSource to its venue-specific Z-spread column."""
        mapping = {
            CompositeSource.MA:   self.ma_z_spread,
            CompositeSource.TW:   self.tw_z_spread,
            CompositeSource.TM:   self.tm_z_spread,
            CompositeSource.CBBT: self.cbbt_z_spread,
        }
        return mapping[source]


@dataclass
class JoinedIntradaySchema:
    """Combined schema after joining quotes + FV on (isin, time_bin)."""
    quote: IntradayQuoteSchema = field(default_factory=IntradayQuoteSchema)
    fv:    IntradayFVSchema = field(default_factory=IntradayFVSchema)


# ---------------------------------------------------------------------------
# Layer 2: Session Management
# ---------------------------------------------------------------------------

class ResetPolicy(str, Enum):
    """Session reset policy for accumulator-based analytics.

    DAILY:            Reset at midnight boundary.
    WEEKLY:           Reset at ISO week boundary.
    INTRADAY_SESSION: Reset at midnight AND noon (US credit AM/PM sessions).
    ON_EVENT:         Reset at caller-specified event boundaries (requires event_mask).
    NEVER:            No resets — full-history accumulation.
    """
    DAILY             = "daily"
    WEEKLY            = "weekly"
    INTRADAY_SESSION  = "intraday_session"
    ON_EVENT          = "on_event"
    NEVER             = "never"


def _make_reset_mask(
    time_bins: np.ndarray,
    policy: ResetPolicy,
    event_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Boolean array where True = reset accumulator at this index."""
    n = len(time_bins)
    resets = np.zeros(n, dtype=bool)

    if policy == ResetPolicy.NEVER or n == 0:
        return resets

    ts = pd.to_datetime(time_bins)
    if n > 1 and not ts.is_monotonic_increasing:
        raise ValueError(
            "_make_reset_mask: time_bins must be sorted chronologically. "
            "For multi-bond data, call per-ISIN via groupby."
        )

    if policy == ResetPolicy.DAILY:
        normalized = ts.normalize()
        resets[0] = True
        resets[1:] = normalized[1:] != normalized[:-1]

    elif policy == ResetPolicy.WEEKLY:
        iso = ts.isocalendar()
        year_week = iso.year.values * 100 + iso.week.values
        resets[0] = True
        resets[1:] = year_week[1:] != year_week[:-1]

    elif policy == ResetPolicy.INTRADAY_SESSION:
        normalized = ts.normalize()
        hours = ts.hour.values if hasattr(ts.hour, 'values') else np.array(ts.hour)
        resets[0] = True
        day_change = normalized[1:] != normalized[:-1]
        session_split = (hours[1:] >= 12) & (hours[:-1] < 12)
        resets[1:] = day_change | session_split

    elif policy == ResetPolicy.ON_EVENT:
        if event_mask is None:
            raise ValueError("ON_EVENT policy requires event_mask array")
        resets = event_mask.astype(bool)

    return resets


# ---------------------------------------------------------------------------
# Layer 3a: Carry Analytics (restored at intraday frequency)
# ---------------------------------------------------------------------------

def total_carry(
    data: pd.DataFrame,
    fv: IntradayFVSchema = IntradayFVSchema(),
    horizon_days: int = 1,
) -> pd.Series:
    """Total carry = carry + rolldown, scaled to horizon.

    Now recomputed every 5 mins by Fincad as spreads move, so
    carry reflects current spread levels, not stale EOD values.
    """
    return (data[fv.carry] + data[fv.rolldown]) * horizon_days


def carry_per_unit_risk(
    data: pd.DataFrame,
    fv: IntradayFVSchema = IntradayFVSchema(),
    horizon_days: int = 1,
) -> pd.Series:
    """Carry-to-risk: total carry / unit_cr01.

    (USD/$100/day × days) / (USD/bp) = bp of carry per unit spread risk.
    Updates intraday as spreads move — detects carry compression/expansion
    around new issue pricing, index rebalances, etc.
    """
    cr01 = data[fv.unit_cr01]
    small_cr01 = cr01.abs() <= 1e-10
    if small_cr01.any():
        logger.warning(
            "carry_per_unit_risk: %d rows have |unit_cr01| <= 1e-10, "
            "returning NaN for these. Check Fincad model outputs.",
            small_cr01.sum(),
        )
    tc = total_carry(data, fv, horizon_days)
    return tc / cr01.where(~small_cr01, np.nan)


def carry_breakeven_spread_move(
    data: pd.DataFrame,
    fv: IntradayFVSchema = IntradayFVSchema(),
    horizon_days: int = 1,
) -> pd.Series:
    """Spread widening (bps) that wipes out horizon carry.

    breakeven = total_carry / unit_cr01
    Intraday: updates as Fincad recomputes carry and cr01 at current spread levels.

    Delegates to carry_per_unit_risk — the formulas are identical
    (both express carry in spread-equivalent bps).
    """
    return carry_per_unit_risk(data, fv, horizon_days)


# ---------------------------------------------------------------------------
# Layer 3b: Fair Value Deviation Analytics (the key intraday signal)
# ---------------------------------------------------------------------------

def fv_deviation_price(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
) -> pd.Series:
    """Quote mid price minus Fincad model price (USD per $100 par).

    Positive = market trades rich to model (quote > FV).
    Negative = market trades cheap to model (quote < FV).

    This is the real-time rich/cheap signal that was impossible with
    the EOD pipeline (where FV was computed once, stale by market open).
    """
    quote_mid = data[schema.quote.price_col("mid", source)]
    model_fv = data[schema.fv.fv_price]
    return quote_mid - model_fv


def fv_deviation_z_spread(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
) -> pd.Series:
    """Venue Z-spread minus Fincad model Z-spread (bps).

    Positive = venue sees wider spread than model (market cheap vs model).
    Negative = venue tighter than model (market rich vs model).

    Sign convention is flipped vs price deviation.

    Uses the venue-specific Z-spreads snapshotted in the FV table
    (ma_composite_mid_z_spread etc.) — the exact inputs Fincad saw.
    """
    venue_z = data[schema.fv.venue_z_col(source)]
    model_z = data[schema.fv.fv_z_spread]
    return venue_z - model_z


def fv_deviation_composite(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
) -> pd.Series:
    """Composite (weighted avg) Z-spread minus Fincad model Z-spread.

    The "headline" rich/cheap measure: pipeline's composite_z_spread
    vs model's fv_z_spread. Non-zero values indicate model lag,
    spread curve interpolation error, or genuine mispricing.
    """
    return data[schema.fv.composite_z] - data[schema.fv.fv_z_spread]


def fv_deviation_carry_scaled(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source: CompositeSource = CompositeSource.MA,
) -> pd.Series:
    """FV deviation expressed in days of carry.

    = (venue_z - model_z) × unit_cr01 / total_carry_per_day

    Value of 3.0 means the bond trades 3 days of carry cheap to model.
    Normalizes across the curve: 2bp deviation on a short-dated bond
    with thin carry is more meaningful than 2bp on a 30yr with fat carry.

    For negative-carry bonds, the sign interpretation reverses:
    positive values indicate the deviation direction, not necessarily 'cheap'.
    """
    dev_bps = fv_deviation_z_spread(data, schema, source)
    cr01 = data[schema.fv.unit_cr01]
    daily_carry = data[schema.fv.carry] + data[schema.fv.rolldown]

    small_carry = daily_carry.abs() <= 1e-10
    if small_carry.any():
        logger.warning(
            "fv_deviation_carry_scaled: %d rows have |daily_carry| <= 1e-10, "
            "returning NaN for these. Check carry/rolldown columns.",
            small_carry.sum(),
        )
    dev_usd = dev_bps * cr01
    return dev_usd / daily_carry.where(~small_carry, np.nan)


# ---------------------------------------------------------------------------
# Layer 3c: Cross-Source Spread Analytics
# ---------------------------------------------------------------------------

def bid_ask_spread(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.Series:
    """Bid-ask width for a given source and price space.

    Always computed as offer - bid. In price space, positive = normal market.
    In spread space where bid_spread > offer_spread (common credit convention),
    normal markets produce negative values. Callers should know their
    vendor's bid/offer spread ordering convention.
    """
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        bid_col = schema.spread_col("bid", source)
        offer_col = schema.spread_col("offer", source)
    elif price_space == PriceSpace.Z_SPREAD:
        bid_col = schema.z_spread_col("bid", source)
        offer_col = schema.z_spread_col("offer", source)
    elif price_space == PriceSpace.CLEAN_PRICE:
        bid_col = schema.price_col("bid", source)
        offer_col = schema.price_col("offer", source)
    else:
        raise ValueError(f"Use market data price spaces, not FV: {price_space}")
    return data[offer_col] - data[bid_col]


def cross_source_mid_divergence(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source_a: CompositeSource = CompositeSource.MA,
    source_b: CompositeSource = CompositeSource.TW,
    price_space: PriceSpace = PriceSpace.BENCHMARK_SPREAD,
) -> pd.Series:
    """Mid-to-mid divergence: source_a.mid - source_b.mid."""
    if price_space == PriceSpace.BENCHMARK_SPREAD:
        col_a = schema.spread_col("mid", source_a)
        col_b = schema.spread_col("mid", source_b)
    elif price_space == PriceSpace.Z_SPREAD:
        col_a = schema.z_spread_col("mid", source_a)
        col_b = schema.z_spread_col("mid", source_b)
    elif price_space == PriceSpace.CLEAN_PRICE:
        col_a = schema.price_col("mid", source_a)
        col_b = schema.price_col("mid", source_b)
    else:
        raise ValueError(f"Use market data price spaces: {price_space}")
    return data[col_a] - data[col_b]


def cross_source_vs_composite_fv(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
) -> pd.DataFrame:
    """Each venue's Z-spread deviation from composite_z_spread.

    Isolates which venue is the outlier when there's cross-source disagreement.
    """
    composite = data[schema.fv.composite_z]
    result = pd.DataFrame(index=data.index)

    for source in CompositeSource:
        col = schema.fv.venue_z_col(source)
        if col in data.columns:
            result[f"dev_vs_composite_{source.value}"] = data[col] - composite
        else:
            logger.warning(
                "cross_source_vs_composite_fv: venue column '%s' not found, "
                "skipping %s. Check FV table join.",
                col, source.value,
            )

    result["data_source"] = data[schema.fv.data_source]
    return result


# ---------------------------------------------------------------------------
# Layer 3d: Intraday Quote Dynamics
# ---------------------------------------------------------------------------

def intraday_spread_range(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    price_space: PriceSpace = PriceSpace.FV_Z_SPREAD,
    reset_policy: ResetPolicy = ResetPolicy.DAILY,
    source: Optional[CompositeSource] = None,
    event_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Running session low/high of spread/price.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (session_lows, session_highs) — running arrays per row.

    Default operates on fv_z_spread (model Z-spread) for consistency
    with carry analytics. Can also use any venue's raw quotes.
    """
    mid_col = _resolve_mid_col(price_space, source, schema)

    mids      = np.array(data[mid_col], dtype=np.float64)
    time_bins = np.array(data[schema.quote.time_bin])
    resets    = _make_reset_mask(time_bins, reset_policy, event_mask)
    n         = len(mids)
    if n == 0:
        return np.array([]), np.array([])

    highs = np.full(n, np.nan)
    lows  = np.full(n, np.nan)

    cur_high = cur_low = np.nan
    for i in range(n):
        if resets[i]:
            cur_high = cur_low = mids[i]  # May be NaN if reset lands on missing data; guard at next block handles this
        if not np.isnan(mids[i]):
            # Guard against NaN-poisoned accumulators: assign directly
            if np.isnan(cur_high):
                cur_high = cur_low = mids[i]
            else:
                cur_high = max(cur_high, mids[i])
                cur_low  = min(cur_low, mids[i])
        highs[i] = cur_high
        lows[i]  = cur_low
    return lows, highs


def cumulative_spread_move(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    price_space: PriceSpace = PriceSpace.FV_Z_SPREAD,
    reset_policy: ResetPolicy = ResetPolicy.DAILY,
    source: Optional[CompositeSource] = None,
    event_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Cumulative intraday spread/price move from session open.

    cum_move:     mid(t) - mid(session_open)
    cum_abs_move: Σ|Δmid| (realized spread volatility path)
    """
    mid_col = _resolve_mid_col(price_space, source, schema)

    mids      = np.array(data[mid_col], dtype=np.float64)
    time_bins = np.array(data[schema.quote.time_bin])
    resets    = _make_reset_mask(time_bins, reset_policy, event_mask)
    n         = len(mids)
    if n == 0:
        return pd.DataFrame({"cum_move": [], "cum_abs_move": []})

    cum_move     = np.full(n, np.nan)
    cum_abs_move = np.zeros(n)
    session_open = np.nan
    abs_accum    = 0.0

    for i in range(n):
        if resets[i]:
            session_open = mids[i]
            abs_accum = 0.0
            cum_move[i] = 0.0 if not np.isnan(mids[i]) else np.nan
            cum_abs_move[i] = 0.0
        else:
            if not np.isnan(mids[i]) and not np.isnan(mids[i - 1]):
                abs_accum += abs(mids[i] - mids[i - 1])
            # Guard: if session_open was NaN (started on missing data),
            # use first valid mid as the effective open
            if np.isnan(session_open) and not np.isnan(mids[i]):
                session_open = mids[i]
            valid = not np.isnan(mids[i]) and not np.isnan(session_open)
            cum_move[i] = (mids[i] - session_open) if valid else np.nan
            cum_abs_move[i] = abs_accum

    return pd.DataFrame({
        "cum_move":     cum_move,
        "cum_abs_move": cum_abs_move,
    }, index=data.index)


def quote_staleness(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
) -> pd.DataFrame:
    """Seconds since last bid/offer change for a given source."""
    last_chg_map = {
        CompositeSource.MA:   (schema.bid_last_chg_ma, schema.offer_last_chg_ma),
        CompositeSource.TW:   (schema.bid_last_chg_tw, schema.offer_last_chg_tw),
        CompositeSource.TM:   (schema.bid_last_chg_tm, schema.offer_last_chg_tm),
        CompositeSource.CBBT: (schema.bid_last_chg_cbbt, schema.offer_last_chg_cbbt),
    }
    bid_chg_col, offer_chg_col = last_chg_map[source]
    ts = pd.to_datetime(data[schema.timestamp])
    bid_chg = pd.to_datetime(data[bid_chg_col])
    offer_chg = pd.to_datetime(data[offer_chg_col])

    return pd.DataFrame({
        f"bid_staleness_s_{source.value}":   (ts - bid_chg).dt.total_seconds(),
        f"offer_staleness_s_{source.value}": (ts - offer_chg).dt.total_seconds(),
    })


# ---------------------------------------------------------------------------
# Layer 3e: Liquidity Analytics
# ---------------------------------------------------------------------------

def liquidity_filtered_universe(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    min_liquidity_score_ma: Optional[float] = None,
    min_liquidity_score_tw: Optional[float] = None,
    require_both: bool = False,
    max_bid_ask_bps: Optional[float] = None,
    source_for_ba: CompositeSource = CompositeSource.MA,
) -> pd.DataFrame:
    """Filter to liquid bonds using composite liquidity scores and bid-ask width."""
    mask = pd.Series(True, index=data.index)

    # Validate requested filters have corresponding columns
    if min_liquidity_score_ma is not None and schema.liquidity_score_ma not in data.columns:
        raise KeyError(
            f"liquidity_filtered_universe: min_liquidity_score_ma={min_liquidity_score_ma} "
            f"requested but column '{schema.liquidity_score_ma}' not in DataFrame. "
            f"Set min_liquidity_score_ma=None or check data source."
        )
    if min_liquidity_score_tw is not None and schema.liquidity_score_tw not in data.columns:
        raise KeyError(
            f"liquidity_filtered_universe: min_liquidity_score_tw={min_liquidity_score_tw} "
            f"requested but column '{schema.liquidity_score_tw}' not in DataFrame. "
            f"Set min_liquidity_score_tw=None or check data source."
        )

    has_ma = min_liquidity_score_ma is not None
    has_tw = min_liquidity_score_tw is not None

    if has_ma and has_tw:
        ma_ok = data[schema.liquidity_score_ma] >= min_liquidity_score_ma
        tw_ok = data[schema.liquidity_score_tw] >= min_liquidity_score_tw
        if require_both:
            mask &= ma_ok & tw_ok
        else:
            mask &= ma_ok | tw_ok
    elif has_ma:
        mask &= data[schema.liquidity_score_ma] >= min_liquidity_score_ma
    elif has_tw:
        mask &= data[schema.liquidity_score_tw] >= min_liquidity_score_tw

    if max_bid_ask_bps is not None:
        ba = bid_ask_spread(data, schema, source_for_ba, PriceSpace.BENCHMARK_SPREAD)
        mask &= ba <= max_bid_ask_bps

    return data[mask].copy()


def bid_ask_regime(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source: CompositeSource = CompositeSource.MA,
    tight_threshold_bps: float = 3.0,
    wide_threshold_bps: float = 10.0,
) -> pd.Series:
    """Classify bid-ask regime: tight / normal / wide / crossed.

    NaN bid-ask values are classified as 'unknown'.
    """
    ba = bid_ask_spread(data, schema, source, PriceSpace.BENCHMARK_SPREAD)
    nan_count = ba.isna().sum()
    if nan_count > 0:
        logger.warning(
            "bid_ask_regime: %d rows have NaN bid-ask spread, classified as 'unknown'.",
            nan_count,
        )
    conditions = [ba < 0, ba <= tight_threshold_bps, ba <= wide_threshold_bps]
    choices = ["crossed", "tight", "normal"]
    return pd.Series(
        np.select(conditions, choices, default="wide"),
        index=data.index,
        name=f"ba_regime_{source.value}",
    ).where(~ba.isna(), "unknown")


# ---------------------------------------------------------------------------
# Layer 3f: Cross-Venue Arbitrage Signals
# ---------------------------------------------------------------------------

def cross_venue_price_dislocation(
    data: pd.DataFrame,
    schema: IntradayQuoteSchema = IntradayQuoteSchema(),
    source_a: CompositeSource = CompositeSource.MA,
    source_b: CompositeSource = CompositeSource.TW,
    min_dislocation_usd: float = 0.02,
) -> pd.DataFrame:
    """Detect actionable price dislocations between venues."""
    bid_price_a = data[schema.price_col("bid", source_a)]
    offer_price_a = data[schema.price_col("offer", source_a)]
    bid_price_b = data[schema.price_col("bid", source_b)]
    offer_price_b = data[schema.price_col("offer", source_b)]

    arb_a_over_b = bid_price_a - offer_price_b
    arb_b_over_a = bid_price_b - offer_price_a
    best_arb = np.maximum(arb_a_over_b, arb_b_over_a)
    direction = np.where(arb_a_over_b >= arb_b_over_a, 1, -1)

    return pd.DataFrame({
        "dislocation_usd": best_arb,
        "arb_direction": direction,
        "arb_a_over_b": arb_a_over_b,
        "arb_b_over_a": arb_b_over_a,
        "actionable": best_arb > min_dislocation_usd,
    }, index=data.index)


def cross_venue_dislocation_carry_scaled(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    source_a: CompositeSource = CompositeSource.MA,
    source_b: CompositeSource = CompositeSource.TW,
) -> pd.Series:
    """Cross-venue dislocation in days of carry.

    = dislocation_usd / daily_carry
    Both quantities are in USD/$100 par, so the ratio directly gives days.

    "If I capture this arb, how many days of carry is it worth?"
    """
    dislocations = cross_venue_price_dislocation(data, schema.quote, source_a, source_b)
    daily_carry = data[schema.fv.carry] + data[schema.fv.rolldown]

    small_carry = daily_carry.abs() <= 1e-10
    if small_carry.any():
        logger.warning(
            "cross_venue_dislocation_carry_scaled: %d rows have |daily_carry| <= 1e-10, "
            "returning NaN for these.",
            small_carry.sum(),
        )
    return dislocations["dislocation_usd"] / daily_carry.where(~small_carry, np.nan)


# ---------------------------------------------------------------------------
# Layer 3g: Spread Profile Analytics (time-weighted)
# ---------------------------------------------------------------------------

def spread_time_profile(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    price_space: PriceSpace = PriceSpace.FV_Z_SPREAD,
    reset_policy: ResetPolicy = ResetPolicy.DAILY,
    source: Optional[CompositeSource] = None,
    bucket_width: Optional[float] = None,
    event_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Mode of spread/price during session — time-weighted POC.

    Without trade volume, uses bin-count weighting: the level where
    the mid has spent the most 5-min bins. Default: fv_z_spread.
    """
    if bucket_width is None:
        logger.warning(
            "spread_time_profile: bucket_width is None; every unique float becomes "
            "its own bin. POC results may be meaningless. Consider setting bucket_width."
        )

    mid_col = _resolve_mid_col(price_space, source, schema)

    mids      = np.array(data[mid_col], dtype=np.float64)
    time_bins = np.array(data[schema.quote.time_bin])
    resets    = _make_reset_mask(time_bins, reset_policy, event_mask)
    n         = len(mids)
    if n == 0:
        return np.array([])

    poc       = np.full(n, np.nan)
    last_poc  = np.nan

    profile: dict[float, float] = {}
    for i in range(n):
        if resets[i]:
            profile.clear()
            last_poc = np.nan
        if np.isnan(mids[i]):
            poc[i] = last_poc
            continue
        key = _bucket(mids[i], bucket_width)
        profile[key] = profile.get(key, 0.0) + 1.0
        last_poc = max(profile, key=profile.get)
        poc[i] = last_poc
    return poc


def time_at_spread(
    data: pd.DataFrame,
    schema: JoinedIntradaySchema = JoinedIntradaySchema(),
    price_space: PriceSpace = PriceSpace.FV_Z_SPREAD,
    reset_policy: ResetPolicy = ResetPolicy.DAILY,
    source: Optional[CompositeSource] = None,
    bucket_width: Optional[float] = None,
    event_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Bins spent at current spread level and total session bins."""
    mid_col = _resolve_mid_col(price_space, source, schema)

    mids      = np.array(data[mid_col], dtype=np.float64)
    time_bins = np.array(data[schema.quote.time_bin])
    resets    = _make_reset_mask(time_bins, reset_policy, event_mask)
    n         = len(mids)
    if n == 0:
        return np.array([]), np.array([])

    bins_at_level = np.zeros(n)
    total_bins    = np.zeros(n)
    profile: dict[float, float] = {}
    running_total = 0.0

    for i in range(n):
        if resets[i]:
            profile.clear()
            running_total = 0.0
        if np.isnan(mids[i]):
            bins_at_level[i] = 0.0
            total_bins[i] = running_total
            continue
        key = _bucket(mids[i], bucket_width)
        running_total += 1.0
        profile[key] = profile.get(key, 0.0) + 1.0
        bins_at_level[i] = profile[key]
        total_bins[i] = running_total
    return bins_at_level, total_bins


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mid_col(
    price_space: PriceSpace,
    source: Optional[CompositeSource],
    schema: JoinedIntradaySchema,
) -> str:
    """Map (price_space, source) to column name.

    FV-based spaces come from intraday_fv_and_greeks (ignore source).
    Market-data spaces come from merged_intraday_market_data (require source).
    """
    if price_space == PriceSpace.FV_Z_SPREAD:
        return schema.fv.fv_z_spread
    elif price_space == PriceSpace.COMPOSITE_Z:
        return schema.fv.composite_z
    elif price_space == PriceSpace.FV_PRICE:
        return schema.fv.fv_price
    elif price_space == PriceSpace.BENCHMARK_SPREAD:
        if source is None:
            raise ValueError("Market data price spaces require a CompositeSource")
        return schema.quote.spread_col("mid", source)
    elif price_space == PriceSpace.Z_SPREAD:
        if source is None:
            raise ValueError("Market data price spaces require a CompositeSource")
        return schema.quote.z_spread_col("mid", source)
    elif price_space == PriceSpace.CLEAN_PRICE:
        if source is None:
            raise ValueError("Market data price spaces require a CompositeSource")
        return schema.quote.price_col("mid", source)
    else:
        raise ValueError(f"Unknown price_space: {price_space}")


def _bucket(value: float, width: Optional[float]) -> float:
    """Discretize a spread/price value into buckets."""
    if width is None or width <= 0:
        return value
    return round(value / width) * width
