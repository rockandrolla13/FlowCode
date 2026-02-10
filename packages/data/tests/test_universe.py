"""Tests for universe filtering module."""

import pandas as pd
import pytest

from src.universe import (
    filter_ig,
    filter_hy,
    filter_by_rating,
    filter_by_liquidity,
    RATING_ORDER,
    IG_THRESHOLD,
)


@pytest.fixture
def rated_bonds() -> pd.DataFrame:
    """Sample DataFrame with bonds of various ratings."""
    return pd.DataFrame({
        "cusip": ["A", "B", "C", "D", "E", "F"],
        "rating": ["AAA", "A", "BBB-", "BB+", "B", "CCC"],
        "total_volume": [1000, 500, 200, 800, 50, 10],
        "trade_count": [100, 50, 20, 80, 5, 1],
    })


class TestFilterIG:
    """Tests for filter_ig function."""

    def test_basic_ig_filter(self, rated_bonds) -> None:
        """Test IG filter keeps BBB- and above."""
        result = filter_ig(rated_bonds)

        assert set(result["rating"]) == {"AAA", "A", "BBB-"}
        assert len(result) == 3

    def test_ig_excludes_hy(self, rated_bonds) -> None:
        """Test IG filter excludes HY ratings."""
        result = filter_ig(rated_bonds)

        assert "BB+" not in result["rating"].values
        assert "B" not in result["rating"].values

    def test_missing_column_raises(self) -> None:
        """Test missing rating column raises ValueError."""
        df = pd.DataFrame({"cusip": ["A"]})
        with pytest.raises(ValueError, match="Rating column"):
            filter_ig(df)


class TestFilterHY:
    """Tests for filter_hy function."""

    def test_basic_hy_filter(self, rated_bonds) -> None:
        """Test HY filter keeps BB+ and below."""
        result = filter_hy(rated_bonds)

        assert set(result["rating"]) == {"BB+", "B", "CCC"}
        assert len(result) == 3

    def test_hy_excludes_ig(self, rated_bonds) -> None:
        """Test HY filter excludes IG ratings."""
        result = filter_hy(rated_bonds)

        assert "AAA" not in result["rating"].values
        assert "BBB-" not in result["rating"].values

    def test_ig_and_hy_are_complementary(self, rated_bonds) -> None:
        """Test IG + HY = all bonds with valid ratings."""
        ig = filter_ig(rated_bonds)
        hy = filter_hy(rated_bonds)

        assert len(ig) + len(hy) == len(rated_bonds)


class TestFilterByRating:
    """Tests for filter_by_rating function."""

    def test_min_rating(self, rated_bonds) -> None:
        """Test minimum rating filter."""
        result = filter_by_rating(rated_bonds, min_rating="A")

        # A and above: AAA, A
        assert all(r in {"AAA", "A"} for r in result["rating"])

    def test_max_rating(self, rated_bonds) -> None:
        """Test maximum rating filter (worst allowed)."""
        result = filter_by_rating(rated_bonds, max_rating="BB+")

        # BB+ and worse: BB+, B, CCC
        assert all(r in {"BB+", "B", "CCC"} for r in result["rating"])

    def test_rating_range(self, rated_bonds) -> None:
        """Test filtering by rating range."""
        result = filter_by_rating(rated_bonds, min_rating="BBB-", max_rating="A")

        # Between A and BBB-: A, BBB-
        assert set(result["rating"]) == {"A", "BBB-"}


class TestFilterByLiquidity:
    """Tests for filter_by_liquidity function."""

    def test_min_volume(self, rated_bonds) -> None:
        """Test minimum volume filter."""
        result = filter_by_liquidity(rated_bonds, min_volume=200)

        assert (result["total_volume"] >= 200).all()

    def test_min_trades(self, rated_bonds) -> None:
        """Test minimum trade count filter."""
        result = filter_by_liquidity(rated_bonds, min_trades=50)

        assert (result["trade_count"] >= 50).all()

    def test_combined_filters(self, rated_bonds) -> None:
        """Test combined volume and trade count filter."""
        result = filter_by_liquidity(rated_bonds, min_volume=500, min_trades=50)

        # A (1000, 100), B (500, 50), D (800, 80) all qualify
        assert len(result) == 3


class TestRatingConstants:
    """Tests for rating constants."""

    def test_rating_order_length(self) -> None:
        """Test rating order contains expected number of ratings."""
        assert len(RATING_ORDER) == 22

    def test_ig_threshold_in_order(self) -> None:
        """Test IG threshold is in the rating order."""
        assert IG_THRESHOLD in RATING_ORDER

    def test_aaa_is_first(self) -> None:
        """Test AAA is the highest rating."""
        assert RATING_ORDER[0] == "AAA"

    def test_d_is_last(self) -> None:
        """Test D is the lowest rating."""
        assert RATING_ORDER[-1] == "D"
