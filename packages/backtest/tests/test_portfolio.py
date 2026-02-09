"""Tests for portfolio construction module."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import equal_weight, risk_parity, top_n_positions


@pytest.fixture
def sample_signal() -> pd.DataFrame:
    """Sample signal DataFrame."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    return pd.DataFrame(
        {
            "AAPL": [1, 1, -1, 1, 0, 1, -1, 1, 1, -1],
            "GOOG": [-1, 1, 1, 0, 1, -1, 1, -1, 0, 1],
            "MSFT": [1, -1, 0, 1, 1, 1, -1, -1, 1, 1],
        },
        index=dates,
        dtype=float,
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Sample price DataFrame."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "AAPL": 100 + rng.standard_normal(10).cumsum(),
            "GOOG": 200 + rng.standard_normal(10).cumsum(),
            "MSFT": 150 + rng.standard_normal(10).cumsum(),
        },
        index=dates,
    )


class TestEqualWeight:
    """Tests for equal_weight function."""

    def test_weights_sum_to_one(self, sample_signal, sample_prices) -> None:
        """Test absolute weights sum to ~1 per row."""
        result = equal_weight(sample_signal, sample_prices)

        for i in range(len(result)):
            abs_sum = result.iloc[i].abs().sum()
            if abs_sum > 0:
                assert abs_sum == pytest.approx(1.0, abs=0.01)

    def test_zero_signal_zero_position(self, sample_signal, sample_prices) -> None:
        """Test zero signals produce zero positions."""
        result = equal_weight(sample_signal, sample_prices)

        # Where signal is 0, position should be 0
        zero_mask = sample_signal == 0
        assert (result[zero_mask].fillna(0) == 0).all().all()

    def test_long_only(self, sample_signal, sample_prices) -> None:
        """Test long_only mode excludes short positions."""
        result = equal_weight(sample_signal, sample_prices, long_only=True)

        assert (result >= 0).all().all()

    def test_max_positions(self, sample_prices) -> None:
        """Test max_positions limits number of active positions."""
        signal = pd.DataFrame(
            np.ones((10, 5)),
            index=sample_prices.index,
            columns=[f"asset_{i}" for i in range(5)],
        )
        prices = pd.DataFrame(
            100 + np.random.default_rng(42).standard_normal((10, 5)).cumsum(axis=0),
            index=sample_prices.index,
            columns=[f"asset_{i}" for i in range(5)],
        )

        result = equal_weight(signal, prices, max_positions=3)

        for i in range(len(result)):
            n_active = (result.iloc[i] != 0).sum()
            assert n_active <= 3


class TestRiskParity:
    """Tests for risk_parity function."""

    def test_returns_dataframe(self, sample_signal, sample_prices) -> None:
        """Test risk_parity returns a DataFrame."""
        result = risk_parity(sample_signal, sample_prices, vol_window=3)
        assert isinstance(result, pd.DataFrame)

    def test_weights_respect_signal_direction(self, sample_signal, sample_prices) -> None:
        """Test weights have same sign as signals."""
        result = risk_parity(sample_signal, sample_prices, vol_window=3)

        # Where signal is nonzero and result is nonzero, signs should match
        for col in result.columns:
            for i in range(len(result)):
                if result.iloc[i][col] != 0 and sample_signal.iloc[i][col] != 0:
                    assert np.sign(result.iloc[i][col]) == np.sign(sample_signal.iloc[i][col])

    def test_long_only(self, sample_signal, sample_prices) -> None:
        """Test long_only mode."""
        result = risk_parity(sample_signal, sample_prices, vol_window=3, long_only=True)
        assert (result >= -1e-10).all().all()


class TestTopNPositions:
    """Tests for top_n_positions function."""

    def test_correct_number_of_positions(self, sample_prices) -> None:
        """Test correct number of long and short positions."""
        rng = np.random.default_rng(42)
        signal = pd.DataFrame(
            rng.standard_normal((10, 5)),
            index=sample_prices.index,
            columns=[f"asset_{i}" for i in range(5)],
        )
        prices = sample_prices.reindex(columns=[f"asset_{i}" for i in range(5)], fill_value=100)

        result = top_n_positions(signal, prices, n_long=2, n_short=2)

        for i in range(len(result)):
            n_long = (result.iloc[i] > 0).sum()
            n_short = (result.iloc[i] < 0).sum()
            assert n_long <= 2
            assert n_short <= 2

    def test_equal_weight_within_side(self, sample_prices) -> None:
        """Test positions are equally weighted within long/short."""
        rng = np.random.default_rng(42)
        signal = pd.DataFrame(
            rng.standard_normal((10, 5)),
            index=sample_prices.index,
            columns=[f"asset_{i}" for i in range(5)],
        )
        prices = sample_prices.reindex(columns=[f"asset_{i}" for i in range(5)], fill_value=100)

        result = top_n_positions(signal, prices, n_long=2, n_short=2)

        for i in range(len(result)):
            longs = result.iloc[i][result.iloc[i] > 0]
            if len(longs) > 0:
                assert longs.nunique() == 1  # all same weight

    def test_no_shorts(self, sample_prices) -> None:
        """Test n_short=0 produces no short positions."""
        rng = np.random.default_rng(42)
        signal = pd.DataFrame(
            rng.standard_normal((10, 3)),
            index=sample_prices.index,
            columns=["A", "B", "C"],
        )
        prices = sample_prices.reindex(columns=["A", "B", "C"], fill_value=100)

        result = top_n_positions(signal, prices, n_long=2, n_short=0)

        assert (result >= 0).all().all()
