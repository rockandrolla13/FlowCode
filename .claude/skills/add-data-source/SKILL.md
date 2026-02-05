# Skill: Add Data Source

## Purpose

Add a new data source to the FlowCode data package.

## Prerequisites

- Understand the data format (CSV, Parquet, etc.)
- Know the schema (columns, types)
- Have sample data for testing

## Steps

### 1. Define Schema

Document the expected schema in the file header:

```python
"""
Schema:
| Column | Type | Constraints |
|--------|------|-------------|
| date | datetime | Not null |
| cusip | str | 9 characters |
| value | float | >= 0 |
"""
```

### 2. Create Loader

Create `packages/data/src/new_source.py`:

```python
"""New data source loader.

Loads data from [describe source].
"""

import logging
from pathlib import Path

import pandas as pd

from .validation import ValidationError

logger = logging.getLogger(__name__)


def load_new_source(
    path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Load new data source.

    Parameters
    ----------
    path : str | Path
        Path to data file (Parquet or CSV).
    start_date : str | None
        Filter start date (inclusive).
    end_date : str | None
        Filter end date (inclusive).

    Returns
    -------
    pd.DataFrame
        Loaded data with columns: date, cusip, value.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValidationError
        If data fails validation.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    # Load based on extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # Validate required columns
    required = {"date", "cusip", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(f"Missing columns: {missing}")

    # Date filtering
    if start_date:
        df = df[df["date"] >= start_date]
    if end_date:
        df = df[df["date"] <= end_date]

    logger.info(f"Loaded {len(df)} rows from {path}")

    return df
```

### 3. Add Validation Rules

Add to `packages/data/src/validation.py`:

```python
def validate_new_source(df: pd.DataFrame) -> list[str]:
    """
    Validate new source data.

    Returns list of error messages (empty if valid).
    """
    errors = []

    # Check required columns
    required = {"date", "cusip", "value"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # Check types
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        errors.append("Column 'date' must be datetime")

    # Check constraints
    if "value" in df.columns and (df["value"] < 0).any():
        errors.append("Column 'value' contains negative values")

    return errors
```

### 4. Add Tests

Create `packages/data/tests/test_new_source.py`:

```python
"""Tests for new source loader."""

import pandas as pd
import pytest
from pathlib import Path

from src.new_source import load_new_source


class TestLoadNewSource:
    def test_load_parquet(self, tmp_path):
        # Create test file
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=3),
            "cusip": ["ABC123456"] * 3,
            "value": [1.0, 2.0, 3.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        result = load_new_source(path)

        assert len(result) == 3
        assert set(result.columns) == {"date", "cusip", "value"}

    def test_date_filtering(self, tmp_path):
        df = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10),
            "cusip": ["ABC123456"] * 10,
            "value": range(10),
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path)

        result = load_new_source(
            path,
            start_date="2023-01-03",
            end_date="2023-01-07",
        )

        assert len(result) == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_new_source("/nonexistent/path.parquet")
```

### 5. Update Package __init__.py

Add export to `packages/data/src/__init__.py`:

```python
from .new_source import load_new_source
```

### 6. Update Configuration

If the source needs config, add to `conf/base.yaml`:

```yaml
data:
  new_source:
    path: /path/to/data
    format: parquet
```

### 7. Verify

```bash
# Run tests
pytest packages/data/tests/test_new_source.py -v

# Type check
mypy packages/data/src/new_source.py
```

## Checklist

- [ ] Schema documented
- [ ] Loader handles Parquet and CSV
- [ ] Date filtering works
- [ ] Validation rules added
- [ ] Tests cover happy path and edge cases
- [ ] Export added to __init__.py
- [ ] Config added (if needed)
- [ ] All tests pass
