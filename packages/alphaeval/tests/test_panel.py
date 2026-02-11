"""Tests for panel validation."""
import numpy as np
import pandas as pd
import pytest

from src.panel import validate_panel


class TestValidatePanel:
    """Tests for validate_panel."""

    def _make_panel(self) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        return pd.DataFrame({
            "date": list(dates) * 2,
            "instrument": ["A"] * 3 + ["B"] * 3,
            "returns": [0.01, -0.02, 0.015, 0.005, -0.01, 0.02],
        })

    def test_columns_to_multiindex(self) -> None:
        df = self._make_panel()
        result = validate_panel(df, required_cols={"date", "instrument", "returns"})
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["date", "instrument"]

    def test_multiindex_passthrough(self) -> None:
        df = self._make_panel().set_index(["date", "instrument"])
        result = validate_panel(df)
        assert isinstance(result.index, pd.MultiIndex)

    def test_missing_column_raises(self) -> None:
        df = self._make_panel()
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_panel(df, required_cols={"date", "instrument", "spread"})

    def test_missing_date_col_raises(self) -> None:
        df = pd.DataFrame({"instrument": ["A"], "returns": [0.01]})
        with pytest.raises(ValueError, match="date"):
            validate_panel(df)

    def test_multiindex_wrong_names_raises(self) -> None:
        df = pd.DataFrame(
            {"returns": [0.01, 0.02]},
            index=pd.MultiIndex.from_tuples(
                [("2024-01-01", "A"), ("2024-01-02", "B")],
                names=["dt", "inst"],
            ),
        )
        with pytest.raises(ValueError, match="MultiIndex must have levels"):
            validate_panel(df)

    def test_string_dates_coerced(self) -> None:
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02"],
            "instrument": ["A", "A"],
            "returns": [0.01, 0.02],
        })
        result = validate_panel(df)
        date_level = result.index.get_level_values("date")
        assert pd.api.types.is_datetime64_any_dtype(date_level)
