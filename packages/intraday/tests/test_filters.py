"""Tests for flowcode_intraday.filters — quote/trade filter functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcode_intraday.analytics import (
    CompositeSource,
    PriceSpace,
    IntradayQuoteSchema,
)
from flowcode_intraday.filters import (
    TradeSide,
    TraceTradeSchema,
    TraceAggSchema,
    FullIntradaySchema,
    SignalType,
    filter_significant_quote_moves,
    filter_trace_big_prints,
    aggregate_trace_to_bins,
    detect_directional_pressure,
)


# ── filter_significant_quote_moves ───────────────────────────────────────

class TestFilterSignificantQuoteMoves:
    def _make_quote_data(self, bid_spreads: list, offer_spreads: list) -> pd.DataFrame:
        """Build DataFrame with bid/offer benchmark spread columns for MA."""
        s = IntradayQuoteSchema()
        n = len(bid_spreads)
        return pd.DataFrame({
            s.time_bin: pd.date_range("2025-01-06 09:30", periods=n, freq="5min"),
            s.spread_col("bid", CompositeSource.MA): bid_spreads,
            s.spread_col("offer", CompositeSource.MA): offer_spreads,
        })

    def test_detects_offer_tightening(self) -> None:
        # Offer drops by 3 bps (tightening = buying pressure)
        data = self._make_quote_data([100, 100, 100], [105, 105, 102])
        result = filter_significant_quote_moves(
            data, source=CompositeSource.MA, min_move_bps=2.0,
            side="offer", max_staleness_s=None,
        )
        assert len(result) == 1
        assert result["move_direction"].iloc[0] == "tightening"

    def test_detects_bid_widening(self) -> None:
        # Bid rises by 3 bps (widening = selling pressure)
        data = self._make_quote_data([100, 100, 103], [105, 105, 105])
        result = filter_significant_quote_moves(
            data, source=CompositeSource.MA, min_move_bps=2.0,
            side="bid", max_staleness_s=None,
        )
        assert len(result) == 1
        assert result["move_direction"].iloc[0] == "widening"

    def test_both_sides(self) -> None:
        data = self._make_quote_data([100, 100, 103], [105, 105, 102])
        result = filter_significant_quote_moves(
            data, source=CompositeSource.MA, min_move_bps=2.0,
            side="both", max_staleness_s=None,
        )
        assert len(result) == 1
        assert result["move_direction"].iloc[0] == "both"

    def test_invalid_side_raises(self) -> None:
        data = self._make_quote_data([100], [105])
        with pytest.raises(ValueError):
            filter_significant_quote_moves(
                data, source=CompositeSource.MA, side="invalid",
                max_staleness_s=None,
            )

    def test_below_threshold_excluded(self) -> None:
        # 1 bp move < 2 bp threshold
        data = self._make_quote_data([100, 100, 101], [105, 105, 104])
        result = filter_significant_quote_moves(
            data, source=CompositeSource.MA, min_move_bps=2.0,
            side="both", max_staleness_s=None,
        )
        assert len(result) == 0


# ── filter_trace_big_prints ──────────────────────────────────────────────

class TestFilterTraceBigPrints:
    def _make_trades(self, volumes: list, sides: list) -> pd.DataFrame:
        s = TraceTradeSchema()
        n = len(volumes)
        return pd.DataFrame({
            s.cusip: ["ABC123456"] * n,
            s.execution_time: pd.date_range("2025-01-06 10:00", periods=n, freq="1min"),
            s.report_time: pd.date_range("2025-01-06 10:00", periods=n, freq="1min"),
            s.price: [100.0] * n,
            s.volume_par: volumes,
            s.side: sides,
            s.special_price_flag: [""] * n,
            s.as_of_indicator: [""] * n,
            s.report_delay_s: [0.0] * n,
        })

    def test_filters_by_volume(self) -> None:
        trades = self._make_trades([500_000, 2_000_000, 100_000], ["B", "B", "B"])
        result = filter_trace_big_prints(trades, min_volume=1_000_000,
                                          max_report_delay_s=None)
        assert len(result) == 1
        assert result[TraceTradeSchema().volume_par].iloc[0] == 2_000_000

    def test_filters_by_side_buy(self) -> None:
        trades = self._make_trades([2_000_000, 2_000_000], ["B", "S"])
        result = filter_trace_big_prints(trades, min_volume=1_000_000,
                                          side="buy", max_report_delay_s=None)
        assert len(result) == 1
        assert result["trade_direction"].iloc[0] == "buy"

    def test_filters_by_side_sell(self) -> None:
        trades = self._make_trades([2_000_000, 2_000_000], ["B", "S"])
        result = filter_trace_big_prints(trades, min_volume=1_000_000,
                                          side="sell", max_report_delay_s=None)
        assert len(result) == 1
        assert result["trade_direction"].iloc[0] == "sell"

    def test_invalid_side_raises(self) -> None:
        trades = self._make_trades([2_000_000], ["B"])
        with pytest.raises(ValueError):
            filter_trace_big_prints(trades, side="invalid",
                                     max_report_delay_s=None)


# ── aggregate_trace_to_bins ──────────────────────────────────────────────

class TestAggregateTraceToBins:
    def _make_trades(self, times: list, volumes: list, sides: list,
                     prices: list | None = None) -> pd.DataFrame:
        s = TraceTradeSchema()
        n = len(times)
        if prices is None:
            prices = [100.0] * n
        return pd.DataFrame({
            s.cusip: ["ABC123456"] * n,
            s.isin: ["US1234567890"] * n,
            s.execution_time: pd.to_datetime(times),
            s.report_time: pd.to_datetime(times),
            s.price: prices,
            s.volume_par: volumes,
            s.side: sides,
            s.special_price_flag: [""] * n,
            s.as_of_indicator: [""] * n,
            s.report_delay_s: [0.0] * n,
        })

    def test_basic_aggregation(self) -> None:
        trades = self._make_trades(
            ["2025-01-06 10:01", "2025-01-06 10:02", "2025-01-06 10:03"],
            [1_000_000, 500_000, 500_000],
            ["B", "S", "B"],
            [100, 101, 99],
        )
        result = aggregate_trace_to_bins(trades, bin_freq="5min")
        agg = TraceAggSchema()
        assert len(result) == 1
        assert result[agg.total_volume].iloc[0] == 2_000_000
        assert result[agg.buy_volume].iloc[0] == 1_500_000
        assert result[agg.sell_volume].iloc[0] == 500_000
        assert result[agg.trade_count].iloc[0] == 3

    def test_empty_trades(self) -> None:
        s = TraceTradeSchema()
        trades = pd.DataFrame(columns=[
            s.cusip, s.isin, s.execution_time, s.report_time,
            s.price, s.volume_par, s.side,
            s.special_price_flag, s.as_of_indicator, s.report_delay_s,
        ])
        result = aggregate_trace_to_bins(trades)
        assert len(result) == 0

    def test_imbalance_ratio(self) -> None:
        trades = self._make_trades(
            ["2025-01-06 10:01", "2025-01-06 10:02"],
            [1_000_000, 1_000_000],
            ["B", "B"],
        )
        result = aggregate_trace_to_bins(trades, bin_freq="5min")
        agg = TraceAggSchema()
        assert result[agg.imbalance_ratio].iloc[0] == pytest.approx(1.0)


# ── detect_directional_pressure ──────────────────────────────────────────

class TestDetectDirectionalPressure:
    def _make_quotes(self, mids: list) -> pd.DataFrame:
        """Build minimal quote DataFrame with mid benchmark spread for MA."""
        s = IntradayQuoteSchema()
        n = len(mids)
        times = pd.date_range("2025-01-06 09:30", periods=n, freq="5min")
        return pd.DataFrame({
            s.time_bin: times,
            s.spread_col("mid", CompositeSource.MA): mids,
            s.spread_col("bid", CompositeSource.MA): [m - 1 for m in mids],
            s.spread_col("offer", CompositeSource.MA): [m + 1 for m in mids],
        })

    def test_returns_required_columns(self) -> None:
        quotes = self._make_quotes([100, 100, 100])
        result = detect_directional_pressure(
            quotes, max_staleness_s=None,
            price_space=PriceSpace.BENCHMARK_SPREAD,
        )
        assert "pressure_score" in result.columns
        assert "signal_count" in result.columns
        assert "pressure_direction" in result.columns

    def test_no_movement_gives_none_or_mixed(self) -> None:
        quotes = self._make_quotes([100, 100, 100])
        result = detect_directional_pressure(
            quotes, max_staleness_s=None,
            price_space=PriceSpace.BENCHMARK_SPREAD,
        )
        # Flat spreads → zero delta → "none" (no signal above threshold)
        directions = result["pressure_direction"].unique()
        assert all(d in ("none", "mixed") for d in directions)

    def test_strong_tightening_gives_buy(self) -> None:
        # Large spread decrease → buying pressure (inverted for spreads)
        quotes = self._make_quotes([100, 95, 90])
        result = detect_directional_pressure(
            quotes, max_staleness_s=None,
            min_quote_move_bps=1.0,
            price_space=PriceSpace.BENCHMARK_SPREAD,
        )
        buy_rows = result[result["pressure_direction"] == "buy"]
        assert len(buy_rows) > 0
