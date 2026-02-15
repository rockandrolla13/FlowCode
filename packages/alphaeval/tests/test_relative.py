"""Tests for relative metrics — tracking_error."""
import numpy as np
import pandas as pd
import pytest

from src.metrics.relative import tracking_error


class TestTrackingError:
    def test_identical_returns_zero(self) -> None:
        r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        assert tracking_error(r, r) == pytest.approx(0.0)

    def test_positive_te(self) -> None:
        r = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        b = pd.Series([0.005, -0.01, 0.02, 0.005, -0.005])
        te = tracking_error(r, b)
        assert te > 0

    def test_known_value(self) -> None:
        r = pd.Series([0.02, 0.04, 0.06, 0.08])
        b = pd.Series([0.01, 0.02, 0.03, 0.04])
        active = r - b
        expected = np.sqrt(252) * active.std(ddof=1)
        assert tracking_error(r, b) == pytest.approx(expected)

    def test_single_obs_nan(self) -> None:
        assert np.isnan(tracking_error(pd.Series([0.01]), pd.Series([0.02])))

    def test_custom_periods(self) -> None:
        r = pd.Series([0.02, 0.04, 0.06, 0.08])
        b = pd.Series([0.01, 0.02, 0.03, 0.04])
        te_daily = tracking_error(r, b, periods_per_year=252)
        te_monthly = tracking_error(r, b, periods_per_year=12)
        assert te_daily > te_monthly  # sqrt(252) > sqrt(12)
