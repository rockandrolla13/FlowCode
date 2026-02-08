"""Tests for triggers module."""

import numpy as np
import pandas as pd
import pytest

from src.triggers import (
    compute_zscore,
    zscore_trigger,
    compute_streak,
    streak_trigger,
)


class TestComputeZscore:
    """Tests for compute_zscore function."""

    def test_basic_zscore(self) -> None:
        """Test basic z-score calculation."""
        # Create series with known mean and std
        series = pd.Series([0, 0, 0, 0, 10])  # Last value is extreme

        result = compute_zscore(series, window=4, min_periods=4)

        # Last value should have positive z-score
        assert result.iloc[-1] > 0

    def test_zscore_nan_for_insufficient_history(self) -> None:
        """Test that z-score is NaN when insufficient history."""
        series = pd.Series([1, 2, 3, 4, 5])

        result = compute_zscore(series, window=10, min_periods=10)

        # All values should be NaN (not enough history)
        assert result.isna().all()

    def test_zscore_with_min_periods(self) -> None:
        """Test z-score respects min_periods parameter."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        result = compute_zscore(series, window=5, min_periods=3)

        # First 2 should be NaN, rest should have values
        assert result.iloc[:2].isna().all()
        assert result.iloc[2:].notna().all()

    def test_zscore_constant_series(self) -> None:
        """Test z-score of constant series is NaN (zero std)."""
        series = pd.Series([5, 5, 5, 5, 5])

        result = compute_zscore(series, window=3, min_periods=3)

        # Constant series has zero std, so z-score is NaN
        assert result.iloc[2:].isna().all()


class TestZscoreTrigger:
    """Tests for zscore_trigger function."""

    def test_trigger_on_extreme_value(self) -> None:
        """Test trigger fires on extreme z-score."""
        # Create series where last value is extreme
        series = pd.Series([0.0] * 100 + [100.0])

        result = zscore_trigger(series, window=50, threshold=3.0)

        assert result.iloc[-1] is True or result.iloc[-1] == True

    def test_no_trigger_on_normal_value(self) -> None:
        """Test trigger doesn't fire on normal values."""
        rng = np.random.default_rng(42)
        series = pd.Series(rng.standard_normal(100))

        # With high threshold, should rarely trigger
        result = zscore_trigger(series, window=50, threshold=10.0, min_periods=50)

        # Most values should be False
        assert result.iloc[50:].sum() < 5

    def test_trigger_threshold(self) -> None:
        """Test trigger respects threshold parameter."""
        series = pd.Series([0.0] * 50 + [10.0])

        # Low threshold should trigger
        low_result = zscore_trigger(series, window=25, threshold=1.0)
        # High threshold should not trigger
        high_result = zscore_trigger(series, window=25, threshold=100.0)

        assert low_result.iloc[-1] == True
        assert high_result.iloc[-1] == False


class TestComputeStreak:
    """Tests for compute_streak function."""

    def test_positive_streak(self) -> None:
        """Test counting positive streaks."""
        series = pd.Series([1, 2, 3, 4, 5])  # All positive

        result = compute_streak(series)

        # Should be 1, 2, 3, 4, 5 (counting up)
        assert list(result) == [1, 2, 3, 4, 5]

    def test_negative_streak(self) -> None:
        """Test counting negative streaks."""
        series = pd.Series([-1, -2, -3])  # All negative

        result = compute_streak(series)

        # Should be -1, -2, -3 (negative streak)
        assert list(result) == [-1, -2, -3]

    def test_sign_change_resets_streak(self) -> None:
        """Test streak resets on sign change."""
        series = pd.Series([1, 2, -1, -2, 3])

        result = compute_streak(series)

        # Streaks: 1, 2, -1, -2, 1
        assert list(result) == [1, 2, -1, -2, 1]

    def test_zero_resets_streak(self) -> None:
        """Test zero value resets streak when reset_on_zero=True."""
        series = pd.Series([1, 2, 0, 3, 4])

        result = compute_streak(series, reset_on_zero=True)

        # Zero should reset: 1, 2, 0, 1, 2
        expected = [1, 2, 0, 1, 2]
        assert list(result) == expected


class TestStreakTrigger:
    """Tests for streak_trigger function."""

    def test_trigger_on_long_streak(self) -> None:
        """Test trigger fires on streak >= min_streak."""
        series = pd.Series([1, 2, 3, 4, 5])  # 5-day positive streak

        result = streak_trigger(series, min_streak=3)

        # Positions 3, 4, 5 should trigger (streak >= 3)
        assert result.iloc[2:].all()  # positions 2, 3, 4 (0-indexed)

    def test_no_trigger_on_short_streak(self) -> None:
        """Test trigger doesn't fire on short streaks."""
        series = pd.Series([1, 2, -1, -2, 3])

        result = streak_trigger(series, min_streak=3)

        # No streak reaches 3
        assert result.sum() == 0

    def test_trigger_min_streak_parameter(self) -> None:
        """Test trigger respects min_streak parameter."""
        series = pd.Series([1, 2, 3, 4, 5])

        result_2 = streak_trigger(series, min_streak=2)
        result_4 = streak_trigger(series, min_streak=4)

        assert result_2.sum() > result_4.sum()
