"""Tests for risk metrics — var_parametric, vpin."""
import numpy as np
import pandas as pd
import pytest

from src.metrics.risk import var_parametric, vpin


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
