"""Tests for diagnostic metrics module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    hit_rate,
    autocorrelation,
    autocorrelation_profile,
    signal_decay,
    information_coefficient,
    turnover,
    holding_period,
)

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "spec" / "fixtures"


class TestHitRate:
    """Tests for hit_rate function."""

    def test_perfect_accuracy(self) -> None:
        """Test hit rate of 1.0 with perfect predictions."""
        signals = pd.Series([1, -1, 1, -1, 1])
        returns = pd.Series([0.01, -0.01, 0.02, -0.01, 0.01])

        result = hit_rate(signals, returns, lag=0)

        assert result == pytest.approx(1.0)

    def test_zero_accuracy(self) -> None:
        """Test hit rate of 0.0 with inverse predictions."""
        signals = pd.Series([1, -1, 1, -1])
        returns = pd.Series([-0.01, 0.01, -0.02, 0.01])

        result = hit_rate(signals, returns, lag=0)

        assert result == pytest.approx(0.0)

    def test_half_accuracy(self) -> None:
        """Test hit rate of 0.5 (random)."""
        signals = pd.Series([1, 1, -1, -1])
        returns = pd.Series([0.01, -0.01, -0.01, 0.01])

        result = hit_rate(signals, returns, lag=0)

        assert result == pytest.approx(0.5)

    def test_with_lag(self) -> None:
        """Test hit rate with lag shifts signals."""
        signals = pd.Series([1, -1, 1, -1, 1])
        returns = pd.Series([0.0, 0.01, -0.01, 0.02, -0.01])

        result = hit_rate(signals, returns, lag=1)

        # After shift: signals[1:] aligned with returns[1:]
        assert not np.isnan(result)

    def test_empty_series(self) -> None:
        """Test empty series returns NaN."""
        result = hit_rate(pd.Series(dtype=float), pd.Series(dtype=float))
        assert np.isnan(result)


class TestAutocorrelation:
    """Tests for autocorrelation function."""

    def test_trending_series(self) -> None:
        """Test trending series has positive autocorrelation."""
        series = pd.Series(range(100), dtype=float)
        result = autocorrelation(series, lag=1)
        assert result > 0.9

    def test_alternating_series(self) -> None:
        """Test alternating series has negative autocorrelation."""
        series = pd.Series([1.0, -1.0] * 50)
        result = autocorrelation(series, lag=1)
        assert result < -0.9

    def test_insufficient_data(self) -> None:
        """Test insufficient data returns NaN."""
        series = pd.Series([1.0])
        result = autocorrelation(series, lag=1)
        assert np.isnan(result)


class TestAutocorrelationProfile:
    """Tests for autocorrelation_profile function."""

    def test_profile_length(self) -> None:
        """Test profile has correct length."""
        series = pd.Series(range(100), dtype=float)
        result = autocorrelation_profile(series, max_lag=10)
        assert len(result) == 10

    def test_profile_index(self) -> None:
        """Test profile index is 1 to max_lag."""
        series = pd.Series(range(100), dtype=float)
        result = autocorrelation_profile(series, max_lag=5)
        assert list(result.index) == [1, 2, 3, 4, 5]


class TestSignalDecay:
    """Tests for signal_decay function."""

    def test_returns_dataframe(self) -> None:
        """Test signal_decay returns a DataFrame."""
        rng = np.random.default_rng(42)
        signal = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100) * 0.01)

        result = signal_decay(signal, returns)

        assert isinstance(result, pd.DataFrame)
        assert "horizon" in result.columns
        assert "correlation" in result.columns
        assert "hit_rate" in result.columns

    def test_default_horizons(self) -> None:
        """Test default horizons are [1, 5, 10, 21]."""
        rng = np.random.default_rng(42)
        signal = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100) * 0.01)

        result = signal_decay(signal, returns)

        assert list(result["horizon"]) == [1, 5, 10, 21]


class TestInformationCoefficient:
    """Tests for information_coefficient function."""

    def test_perfect_signal(self) -> None:
        """Test IC with perfectly correlated signal."""
        signal = pd.Series(range(100), dtype=float)
        returns = pd.Series(range(1, 101), dtype=float)  # shifted by 1

        result = information_coefficient(signal, returns, lag=1)

        assert result > 0.9

    def test_empty_series(self) -> None:
        """Test empty series returns NaN."""
        result = information_coefficient(pd.Series(dtype=float), pd.Series(dtype=float))
        assert np.isnan(result)

    def test_insufficient_data(self) -> None:
        """Test insufficient data returns NaN."""
        result = information_coefficient(
            pd.Series([1, 2, 3]),
            pd.Series([0.01, 0.02, 0.03]),
        )
        assert np.isnan(result)  # < 10 data points after alignment


class TestTurnover:
    """Tests for turnover function."""

    def test_constant_positions(self) -> None:
        """Test zero turnover with constant positions."""
        positions = pd.Series([1.0] * 10)
        result = turnover(positions)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_alternating_positions(self) -> None:
        """Test turnover with alternating positions."""
        positions = pd.Series([1.0, -1.0, 1.0, -1.0])
        result = turnover(positions)
        # Changes: NaN, 2.0, 2.0, 2.0 → mean skips NaN = 2.0
        assert result == pytest.approx(2.0, abs=0.01)

    def test_insufficient_data(self) -> None:
        """Test insufficient data returns NaN."""
        result = turnover(pd.Series([1.0]))
        assert np.isnan(result)


class TestHoldingPeriod:
    """Tests for holding_period function."""

    def test_constant_position(self) -> None:
        """Test holding period with no changes."""
        positions = pd.Series([1.0] * 10)
        result = holding_period(positions)
        assert result == 10.0

    def test_frequent_trades(self) -> None:
        """Test holding period with frequent sign changes."""
        positions = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
        result = holding_period(positions)
        assert result < 3  # frequent changes = short holding

    def test_insufficient_data(self) -> None:
        """Test insufficient data returns NaN."""
        result = holding_period(pd.Series([1.0]))
        assert np.isnan(result)


class TestHitRateFixtures:
    """Fixture-backed tests for hit rate (spec §5.1)."""

    @pytest.fixture
    def hit_rate_cases(self):
        with open(FIXTURES_DIR / "hit_rate_cases.json") as f:
            return json.load(f)

    def test_all_hit_rate_fixture_cases(self, hit_rate_cases) -> None:
        """Test hit_rate against golden fixture cases."""
        for case in hit_rate_cases["cases"]:
            signals_data = case["input"]["signals"]
            returns_data = case["input"]["returns"]
            lag = case["input"].get("lag", 1)

            signals = pd.Series(signals_data, dtype=float)
            returns = pd.Series(returns_data, dtype=float)

            result = hit_rate(signals, returns, lag=lag)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.001)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )


class TestAutocorrFixtures:
    """Fixture-backed tests for autocorrelation (spec §5.2)."""

    @pytest.fixture
    def autocorr_cases(self):
        with open(FIXTURES_DIR / "autocorr_cases.json") as f:
            return json.load(f)

    def test_all_autocorr_fixture_cases(self, autocorr_cases) -> None:
        """Test autocorrelation against golden fixture cases."""
        for case in autocorr_cases["cases"]:
            series = pd.Series(case["input"]["series"], dtype=float)
            lag = case["input"].get("lag", 1)

            result = autocorrelation(series, lag=lag)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.001)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )


class TestIcFixtures:
    """Fixture-backed tests for Information Coefficient (spec §5.3)."""

    @pytest.fixture
    def ic_cases(self):
        with open(FIXTURES_DIR / "ic_cases.json") as f:
            return json.load(f)

    def test_all_ic_fixture_cases(self, ic_cases) -> None:
        """Test information_coefficient against golden fixture cases."""
        for case in ic_cases["cases"]:
            signals = pd.Series(case["input"]["signals"], dtype=float)
            returns = pd.Series(case["input"]["returns"], dtype=float)
            lag = case["input"].get("lag", 1)

            result = information_coefficient(signals, returns, lag=lag)

            if case["expected"] is None:
                assert np.isnan(result), f"Case '{case['name']}': expected NaN"
            else:
                tol = case.get("tolerance", 0.001)
                assert result == pytest.approx(case["expected"], abs=tol), (
                    f"Case '{case['name']}': expected {case['expected']}, got {result}"
                )
