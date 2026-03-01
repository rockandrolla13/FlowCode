"""Tests for risk metrics — var_parametric, vpin."""
import numpy as np
import pandas as pd
import pytest

from flowcode_alphaeval.metrics.risk import var_parametric, vpin


class TestVarParametric:
    def test_basic(self) -> None:
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 500))
        v = var_parametric(returns, confidence=0.95)
        # Verify against manual: losses = -returns, VaR = mu_L + z * sigma_L
        losses = -returns
        from scipy import stats
        expected = losses.mean() + stats.norm.ppf(0.95) * losses.std(ddof=1)
        assert v == pytest.approx(expected)

    def test_vs_manual(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008])
        losses = -returns
        mu = losses.mean()
        sigma = losses.std(ddof=1)
        from scipy import stats
        expected = mu + stats.norm.ppf(0.95) * sigma
        assert var_parametric(returns, 0.95) == pytest.approx(expected)

    def test_single_obs_nan(self) -> None:
        assert np.isnan(var_parametric(pd.Series([0.01])))

    def test_zero_vol(self) -> None:
        returns = pd.Series([0.01, 0.01, 0.01])
        v = var_parametric(returns)
        assert v == pytest.approx(-0.01)  # mu_L = -0.01, sigma=0

    def test_invalid_confidence_raises(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015])
        with pytest.raises(ValueError, match="confidence must be in"):
            var_parametric(returns, confidence=95)
        with pytest.raises(ValueError):
            var_parametric(returns, confidence=0.0)
        with pytest.raises(ValueError):
            var_parametric(returns, confidence=1.0)


class TestVpin:
    def test_basic(self) -> None:
        buy = pd.Series([100, 200, 150, 300, 100])
        sell = pd.Series([50, 100, 200, 100, 200])
        result = vpin(buy, sell, n_buckets=3)
        assert np.isnan(result.iloc[0])  # insufficient window
        assert np.isnan(result.iloc[1])
        assert 0 <= result.iloc[2] <= 1

    def test_balanced_zero_vpin(self) -> None:
        buy = pd.Series([100] * 5)
        sell = pd.Series([100] * 5)
        result = vpin(buy, sell, n_buckets=3)
        assert result.iloc[-1] == pytest.approx(0.0)

    def test_one_sided_max_vpin(self) -> None:
        buy = pd.Series([100] * 5)
        sell = pd.Series([0] * 5)
        result = vpin(buy, sell, n_buckets=3)
        assert result.iloc[-1] == pytest.approx(1.0)

    def test_zero_volume_nan(self) -> None:
        buy = pd.Series([0] * 5)
        sell = pd.Series([0] * 5)
        result = vpin(buy, sell, n_buckets=3)
        assert result.isna().all()

    def test_known_value(self) -> None:
        buy = pd.Series([100, 200, 150, 300, 100])
        sell = pd.Series([50, 100, 200, 100, 200])
        result = vpin(buy, sell, n_buckets=3)
        # Window [0:3]: imbalance = |50|+|100|+|-50| = 200, total = 150+300+350 = 800
        assert result.iloc[2] == pytest.approx(200.0 / 800.0)

    def test_index_mismatch_raises(self) -> None:
        """C4 fix: different indices must raise ValueError."""
        buy = pd.Series([100, 200], index=[0, 1])
        sell = pd.Series([50, 100], index=[0, 2])
        with pytest.raises(ValueError, match="different indices"):
            vpin(buy, sell)

    def test_negative_volumes_clamped(self) -> None:
        """C2 fix: negative volumes clamped to 0, result in [0,1]."""
        buy = pd.Series([100, -50, 200, 150, 100])
        sell = pd.Series([50, 100, 200, 100, 200])
        result = vpin(buy, sell, n_buckets=3)
        # After clamping -50→0, all VPIN values in [0,1]
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()

    def test_negative_volumes_clamped_with_caplog(self, caplog) -> None:
        """C2 fix: negative volumes produce warning."""
        import logging
        buy = pd.Series([100, -50, 200, 150, 100])
        sell = pd.Series([50, 100, 200, 100, 200])
        with caplog.at_level(logging.WARNING, logger="flowcode_alphaeval.metrics.risk"):
            vpin(buy, sell, n_buckets=3)
        assert "negative volume" in caplog.text
