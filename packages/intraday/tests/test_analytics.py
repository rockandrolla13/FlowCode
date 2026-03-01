"""Tests for flowcode_intraday.analytics — core intraday microstructure functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flowcode_intraday.analytics import (
    ResetPolicy,
    CompositeSource,
    PriceSpace,
    IntradayQuoteSchema,
    IntradayFVSchema,
    JoinedIntradaySchema,
    _make_reset_mask,
    _bucket,
    bid_ask_spread,
    bid_ask_regime,
    carry_per_unit_risk,
    total_carry,
    intraday_spread_range,
    cumulative_spread_move,
)


# ── _make_reset_mask ─────────────────────────────────────────────────────

class TestMakeResetMask:
    def test_never_policy_returns_all_false(self) -> None:
        ts = pd.date_range("2025-01-06 09:30", periods=5, freq="5min").values
        mask = _make_reset_mask(ts, ResetPolicy.NEVER)
        assert not mask.any()

    def test_empty_array(self) -> None:
        mask = _make_reset_mask(np.array([], dtype="datetime64[ns]"), ResetPolicy.DAILY)
        assert len(mask) == 0

    def test_daily_resets_at_date_boundaries(self) -> None:
        ts = np.array(
            pd.to_datetime(["2025-01-06 10:00", "2025-01-06 14:00",
                            "2025-01-07 09:30", "2025-01-07 14:00"])
        )
        mask = _make_reset_mask(ts, ResetPolicy.DAILY)
        assert list(mask) == [True, False, True, False]

    def test_weekly_resets_at_week_boundaries(self) -> None:
        ts = np.array(
            pd.to_datetime(["2025-01-06 10:00", "2025-01-10 10:00",
                            "2025-01-13 10:00"])
        )
        mask = _make_reset_mask(ts, ResetPolicy.WEEKLY)
        assert list(mask) == [True, False, True]

    def test_intraday_session_resets_at_noon(self) -> None:
        ts = np.array(
            pd.to_datetime(["2025-01-06 09:30", "2025-01-06 11:55",
                            "2025-01-06 12:05", "2025-01-06 15:00"])
        )
        mask = _make_reset_mask(ts, ResetPolicy.INTRADAY_SESSION)
        assert mask[0] == True    # first always resets
        assert mask[1] == False   # same AM session
        assert mask[2] == True    # crosses noon boundary
        assert mask[3] == False   # same PM session

    def test_on_event_requires_mask(self) -> None:
        ts = pd.date_range("2025-01-06", periods=3, freq="5min").values
        with pytest.raises(ValueError, match="event_mask"):
            _make_reset_mask(ts, ResetPolicy.ON_EVENT)

    def test_on_event_uses_provided_mask(self) -> None:
        ts = pd.date_range("2025-01-06", periods=4, freq="5min").values
        event = np.array([False, True, False, True])
        mask = _make_reset_mask(ts, ResetPolicy.ON_EVENT, event_mask=event)
        assert list(mask) == [False, True, False, True]

    def test_unsorted_raises(self) -> None:
        ts = np.array(pd.to_datetime(["2025-01-07", "2025-01-06"]))
        with pytest.raises(ValueError, match="sorted"):
            _make_reset_mask(ts, ResetPolicy.DAILY)


# ── _bucket ──────────────────────────────────────────────────────────────

class TestBucket:
    def test_none_width_returns_value(self) -> None:
        assert _bucket(3.7, None) == 3.7

    def test_zero_width_returns_value(self) -> None:
        assert _bucket(3.7, 0.0) == 3.7

    def test_positive_width_buckets(self) -> None:
        assert _bucket(3.7, 1.0) == 4.0
        assert _bucket(3.2, 1.0) == 3.0
        # Python banker's rounding: round(0.5) = 0
        assert _bucket(2.5, 5.0) == 0.0
        assert _bucket(7.5, 5.0) == 10.0


# ── bid_ask_regime ───────────────────────────────────────────────────────

class TestBidAskRegime:
    def _make_data(self, bid_spreads: list, offer_spreads: list) -> pd.DataFrame:
        """Build minimal DataFrame with bid/offer benchmark spread columns for MA."""
        s = IntradayQuoteSchema()
        n = len(bid_spreads)
        return pd.DataFrame({
            s.time_bin: pd.date_range("2025-01-06 09:30", periods=n, freq="5min"),
            s.spread_col("bid", CompositeSource.MA): bid_spreads,
            s.spread_col("offer", CompositeSource.MA): offer_spreads,
        })

    def test_tight_normal_wide_classification(self) -> None:
        # BA = offer - bid: [2, 5, 15] → tight(≤3), normal(≤10), wide(>10)
        data = self._make_data([100, 100, 100], [102, 105, 115])
        result = bid_ask_regime(data, source=CompositeSource.MA,
                                tight_threshold_bps=3.0, wide_threshold_bps=10.0)
        assert result.iloc[0] == "tight"
        assert result.iloc[1] == "normal"
        assert result.iloc[2] == "wide"

    def test_crossed_market(self) -> None:
        data = self._make_data([105], [100])
        result = bid_ask_regime(data, source=CompositeSource.MA)
        assert result.iloc[0] == "crossed"

    def test_nan_returns_unknown(self) -> None:
        data = self._make_data([np.nan], [np.nan])
        result = bid_ask_regime(data, source=CompositeSource.MA)
        assert result.iloc[0] == "unknown"


# ── carry_per_unit_risk ──────────────────────────────────────────────────

class TestCarryPerUnitRisk:
    def _make_fv_data(self, carry: list, rolldown: list, cr01: list) -> pd.DataFrame:
        fv = IntradayFVSchema()
        return pd.DataFrame({
            fv.carry: carry,
            fv.rolldown: rolldown,
            fv.unit_cr01: cr01,
        })

    def test_normal_computation(self) -> None:
        data = self._make_fv_data([0.01, 0.02], [0.005, 0.01], [0.05, 0.10])
        result = carry_per_unit_risk(data, horizon_days=1)
        assert result.iloc[0] == pytest.approx(0.3)
        assert result.iloc[1] == pytest.approx(0.3)

    def test_tiny_cr01_returns_nan(self) -> None:
        data = self._make_fv_data([0.01], [0.005], [1e-12])
        result = carry_per_unit_risk(data, horizon_days=1)
        assert np.isnan(result.iloc[0])

    def test_zero_cr01_returns_nan(self) -> None:
        data = self._make_fv_data([0.01], [0.005], [0.0])
        result = carry_per_unit_risk(data, horizon_days=1)
        assert np.isnan(result.iloc[0])


# ── intraday_spread_range ────────────────────────────────────────────────

class TestIntradaySpreadRange:
    def _make_joined_data(self, mids: list, times: list | None = None) -> pd.DataFrame:
        schema = JoinedIntradaySchema()
        n = len(mids)
        if times is None:
            times = pd.date_range("2025-01-06 09:30", periods=n, freq="5min")
        return pd.DataFrame({
            schema.fv.fv_z_spread: mids,
            schema.quote.time_bin: times,
        })

    def test_basic_range_tracking(self) -> None:
        data = self._make_joined_data([100, 102, 98, 101])
        lows, highs = intraday_spread_range(data, reset_policy=ResetPolicy.DAILY)
        assert highs[-1] == 102
        assert lows[-1] == 98

    def test_empty_data(self) -> None:
        data = self._make_joined_data([])
        lows, highs = intraday_spread_range(data, reset_policy=ResetPolicy.DAILY)
        assert len(lows) == 0
        assert len(highs) == 0

    def test_nan_at_reset_recovers(self) -> None:
        times = pd.to_datetime(["2025-01-06 09:30", "2025-01-06 09:35",
                                "2025-01-07 09:30", "2025-01-07 09:35"])
        data = self._make_joined_data([100, 102, np.nan, 50], times)
        lows, highs = intraday_spread_range(data, reset_policy=ResetPolicy.DAILY)
        assert lows[1] == 100
        assert highs[1] == 102
        assert np.isnan(lows[2])
        assert lows[3] == 50
        assert highs[3] == 50


# ── cumulative_spread_move ───────────────────────────────────────────────

class TestCumulativeSpreadMove:
    def _make_joined_data(self, mids: list, times: list | None = None) -> pd.DataFrame:
        schema = JoinedIntradaySchema()
        n = len(mids)
        if times is None:
            times = pd.date_range("2025-01-06 09:30", periods=n, freq="5min")
        return pd.DataFrame({
            schema.fv.fv_z_spread: mids,
            schema.quote.time_bin: times,
        })

    def test_basic_cumulative_move(self) -> None:
        # DAILY policy: first row resets, so session_open=100
        data = self._make_joined_data([100, 102, 99])
        result = cumulative_spread_move(data, reset_policy=ResetPolicy.DAILY)
        assert result["cum_move"].iloc[0] == pytest.approx(0.0)
        assert result["cum_move"].iloc[1] == pytest.approx(2.0)
        assert result["cum_move"].iloc[2] == pytest.approx(-1.0)
        assert result["cum_abs_move"].iloc[0] == pytest.approx(0.0)
        assert result["cum_abs_move"].iloc[1] == pytest.approx(2.0)
        assert result["cum_abs_move"].iloc[2] == pytest.approx(5.0)

    def test_empty_data(self) -> None:
        data = self._make_joined_data([])
        result = cumulative_spread_move(data, reset_policy=ResetPolicy.DAILY)
        assert len(result) == 0

    def test_session_open_nan_recovers(self) -> None:
        """If session opens on NaN, first valid mid becomes session_open."""
        times = pd.to_datetime(["2025-01-06 09:30", "2025-01-06 09:35",
                                "2025-01-06 09:40"])
        data = self._make_joined_data([np.nan, 100, 102], times)
        result = cumulative_spread_move(data, reset_policy=ResetPolicy.DAILY)
        assert np.isnan(result["cum_move"].iloc[0])
        assert result["cum_move"].iloc[1] == pytest.approx(0.0)
        assert result["cum_move"].iloc[2] == pytest.approx(2.0)
