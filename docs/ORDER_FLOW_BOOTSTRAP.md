# Order Flow Bootstrap Instructions

Use this prompt to bootstrap a credit/order flow analytics monorepo.

---

## Prompt

I'm building a credit/order flow analytics monorepo called "order_flow". The directory structure is already created.

Create these foundation files in order:

1. **pyproject.toml** (root) - uv workspace config for packages/
2. **.gitignore** - Python defaults
3. **CLAUDE.md** (root) - navigation map, global rules, and Formula Registry table
4. **spec/SPEC.md** - formulas for:
   - §1.1 Credit PnL: `Credit_PnL = -(ΔSpread) × (PVBP / MidPrice)`
   - §1.2 Retail Imbalance: `I_t = (ΣBuy - ΣSell) / (ΣBuy + ΣSell)` with QMP classification
   - §2.1 Z-score trigger: `Z_t = (I_t - μ) / σ`, trigger when `|Z| > threshold`
   - §2.2 Streak trigger: consecutive same-sign days, trigger when `streak ≥ min_streak`
5. **spec/fixtures/imbalance_cases.json** - test cases for imbalance
6. **spec/fixtures/zscore_cases.json** - test cases for z-score
7. **spec/fixtures/streak_cases.json** - test cases for streak
8. **packages/signals/pyproject.toml**
9. **packages/signals/src/__init__.py**
10. **packages/signals/src/params.py** - frozen dataclass parameters (see below)
11. **packages/signals/src/triggers.py** - implement `zscore_trigger()` and `streak_trigger()`
12. **packages/signals/src/imbalance.py** - implement `compute_imbalance()` and `qmp_classify()`
13. **packages/signals/tests/conftest.py** - fixture loading helpers
14. **packages/signals/tests/test_triggers.py** - fixture-driven tests
15. **packages/signals/tests/test_imbalance.py** - fixture-driven tests
16. **tools/verify_spec_parity.py** - checks spec matches code

After creating all files, run: `uv sync && uv run pytest packages/signals/tests/ -v`

---

## QMP Classification Rule

QMP (Quote Midpoint) rule classifies trade direction based on execution price relative to the quote midpoint:

```
BUY    if Price > Midpoint + (α × Spread)
SELL   if Price < Midpoint - (α × Spread)
NEUTRAL otherwise
```

Where:
- `Midpoint = (Bid + Ask) / 2`
- `Spread = Ask - Bid`
- `α = 0.1` (default) - threshold to avoid classifying mid-price trades as directional

Example with α=0.1:
- Bid=99, Ask=101 → Midpoint=100, Spread=2
- Threshold band: [100 - 0.2, 100 + 0.2] = [99.8, 100.2]
- Price=100.5 → BUY (above 100.2)
- Price=99.5 → SELL (below 99.8)
- Price=100.0 → NEUTRAL (within band)

---

## Configuration Pattern

Thresholds MUST be configurable via frozen dataclasses with sensible defaults.

### packages/signals/src/params.py

```python
"""Signal parameters as frozen dataclasses.

Frozen dataclasses provide:
- Immutability (safe for concurrent use)
- Hashable (can be used as dict keys for caching)
- Clear defaults (self-documenting)
- Override flexibility (users can customize)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZScoreParams:
    """Parameters for z-score trigger.

    Attributes
    ----------
    window : int
        Rolling window for mean/std calculation (trading days).
    threshold : float
        Z-score threshold for trigger (absolute value).
    min_periods : int
        Minimum observations before computing z-score.
    """
    window: int = 252
    threshold: float = 7.0
    min_periods: int = 126


@dataclass(frozen=True)
class StreakParams:
    """Parameters for streak trigger.

    Attributes
    ----------
    min_streak : int
        Minimum consecutive same-sign days to trigger.
    """
    min_streak: int = 3


@dataclass(frozen=True)
class QMPParams:
    """Parameters for QMP classification.

    Attributes
    ----------
    alpha : float
        Threshold as fraction of spread. Trades within
        [midpoint - α×spread, midpoint + α×spread] are NEUTRAL.
    """
    alpha: float = 0.1
```

### Function Signatures

Functions accept params with defaults, so tests use defaults but users can override:

```python
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from .params import ZScoreParams, StreakParams, QMPParams


class ZScoreResult(NamedTuple):
    """Result of z-score calculation."""
    zscore: float
    triggered: bool
    direction: int  # +1, -1, or 0


class StreakResult(NamedTuple):
    """Result of streak calculation."""
    streak: int
    triggered: bool
    direction: int  # +1, -1, or 0


def zscore_trigger(
    value: float,
    history: pd.Series,
    params: ZScoreParams = ZScoreParams(),
) -> ZScoreResult:
    """
    Compute z-score and check if trigger threshold exceeded.

    Parameters
    ----------
    value : float
        Current value to evaluate.
    history : pd.Series
        Historical values for mean/std calculation.
    params : ZScoreParams
        Configuration parameters.

    Returns
    -------
    ZScoreResult
        Z-score value, whether triggered, and direction.
    """
    ...


def streak_trigger(
    values: pd.Series,
    params: StreakParams = StreakParams(),
) -> StreakResult:
    """
    Count consecutive same-sign values and check trigger.

    Parameters
    ----------
    values : pd.Series
        Series of values (uses sign for streak counting).
    params : StreakParams
        Configuration parameters.

    Returns
    -------
    StreakResult
        Streak length, whether triggered, and direction.
    """
    ...


def qmp_classify(
    price: float,
    bid: float,
    ask: float,
    params: QMPParams = QMPParams(),
) -> str:
    """
    Classify trade direction using QMP rule.

    Parameters
    ----------
    price : float
        Execution price.
    bid : float
        Best bid price.
    ask : float
        Best ask price.
    params : QMPParams
        Configuration parameters.

    Returns
    -------
    str
        'BUY', 'SELL', or 'NEUTRAL'.
    """
    ...
```

---

## Usage Examples

### Default parameters (tests)
```python
result = zscore_trigger(value=0.5, history=history)
assert result.triggered == True
```

### Custom parameters (production)
```python
conservative = ZScoreParams(window=504, threshold=10.0, min_periods=252)
result = zscore_trigger(value=0.5, history=history, params=conservative)
```

### Override single param
```python
from dataclasses import replace

default = ZScoreParams()
custom = replace(default, threshold=5.0)  # Only change threshold
```

---

## Fixture Specifications

Each fixture file must have this structure:

```json
{
  "description": "What this tests",
  "spec_section": "§X.X",
  "formula": "The formula being tested",
  "cases": [
    {
      "name": "descriptive_snake_case_name",
      "input": { ... },
      "expected": value_or_object,
      "note": "optional explanation"
    }
  ]
}
```

### spec/fixtures/imbalance_cases.json

```json
{
  "description": "Test cases for retail order imbalance calculation",
  "spec_section": "§1.2",
  "formula": "I_t = (ΣBuy - ΣSell) / (ΣBuy + ΣSell)",
  "cases": [
    {
      "name": "all_buys",
      "input": {"buy_volume": 1000.0, "sell_volume": 0.0},
      "expected": 1.0,
      "note": "Maximum bullish imbalance"
    },
    {
      "name": "all_sells",
      "input": {"buy_volume": 0.0, "sell_volume": 1000.0},
      "expected": -1.0,
      "note": "Maximum bearish imbalance"
    },
    {
      "name": "balanced",
      "input": {"buy_volume": 500.0, "sell_volume": 500.0},
      "expected": 0.0,
      "note": "Equal buy/sell volume"
    },
    {
      "name": "slight_buy_bias",
      "input": {"buy_volume": 600.0, "sell_volume": 400.0},
      "expected": 0.2,
      "note": "(600-400)/(600+400) = 200/1000 = 0.2"
    },
    {
      "name": "slight_sell_bias",
      "input": {"buy_volume": 400.0, "sell_volume": 600.0},
      "expected": -0.2
    },
    {
      "name": "zero_volume",
      "input": {"buy_volume": 0.0, "sell_volume": 0.0},
      "expected": null,
      "note": "Division by zero returns NaN"
    },
    {
      "name": "small_volumes",
      "input": {"buy_volume": 1.0, "sell_volume": 3.0},
      "expected": -0.5,
      "note": "(1-3)/(1+3) = -2/4 = -0.5"
    }
  ]
}
```

### spec/fixtures/qmp_cases.json

```json
{
  "description": "Test cases for QMP trade classification",
  "spec_section": "§1.2",
  "formula": "BUY if Price > Mid + α×Spread, SELL if Price < Mid - α×Spread, else NEUTRAL",
  "params": {"alpha": 0.1},
  "cases": [
    {
      "name": "clear_buy",
      "input": {"price": 100.5, "bid": 99.0, "ask": 101.0},
      "expected": "BUY",
      "note": "Mid=100, Spread=2, threshold=0.2, price > 100.2"
    },
    {
      "name": "clear_sell",
      "input": {"price": 99.5, "bid": 99.0, "ask": 101.0},
      "expected": "SELL",
      "note": "price < 99.8"
    },
    {
      "name": "neutral_at_mid",
      "input": {"price": 100.0, "bid": 99.0, "ask": 101.0},
      "expected": "NEUTRAL",
      "note": "Exactly at midpoint"
    },
    {
      "name": "neutral_within_band",
      "input": {"price": 100.1, "bid": 99.0, "ask": 101.0},
      "expected": "NEUTRAL",
      "note": "100.1 is within [99.8, 100.2]"
    },
    {
      "name": "buy_at_boundary",
      "input": {"price": 100.21, "bid": 99.0, "ask": 101.0},
      "expected": "BUY",
      "note": "Just above upper threshold"
    },
    {
      "name": "sell_at_boundary",
      "input": {"price": 99.79, "bid": 99.0, "ask": 101.0},
      "expected": "SELL",
      "note": "Just below lower threshold"
    },
    {
      "name": "tight_spread",
      "input": {"price": 100.05, "bid": 99.9, "ask": 100.1},
      "expected": "BUY",
      "note": "Mid=100, Spread=0.2, threshold=0.02, price > 100.02"
    },
    {
      "name": "wide_spread",
      "input": {"price": 101.0, "bid": 95.0, "ask": 105.0},
      "expected": "NEUTRAL",
      "note": "Mid=100, Spread=10, threshold=1.0, 101 within [99, 101]"
    }
  ]
}
```

### spec/fixtures/zscore_cases.json

```json
{
  "description": "Test cases for z-score trigger",
  "spec_section": "§2.1",
  "formula": "Z_t = (value - μ) / σ, trigger if |Z| > threshold",
  "params": {"window": 5, "threshold": 2.0, "min_periods": 3},
  "cases": [
    {
      "name": "positive_trigger",
      "input": {
        "value": 10.0,
        "history": [1.0, 2.0, 1.5, 1.8, 2.2]
      },
      "expected": {
        "zscore": 17.89,
        "triggered": true,
        "direction": 1
      },
      "tolerance": 0.01,
      "note": "μ=1.7, σ=0.45, Z=(10-1.7)/0.45=18.4"
    },
    {
      "name": "negative_trigger",
      "input": {
        "value": -5.0,
        "history": [1.0, 2.0, 1.5, 1.8, 2.2]
      },
      "expected": {
        "zscore": -14.89,
        "triggered": true,
        "direction": -1
      },
      "tolerance": 0.01
    },
    {
      "name": "no_trigger_within_band",
      "input": {
        "value": 2.0,
        "history": [1.0, 2.0, 1.5, 1.8, 2.2]
      },
      "expected": {
        "zscore": 0.67,
        "triggered": false,
        "direction": 0
      },
      "tolerance": 0.01,
      "note": "Z=0.67 < 2.0 threshold"
    },
    {
      "name": "exactly_at_threshold",
      "input": {
        "value": 2.6,
        "history": [1.0, 1.0, 1.0, 1.0, 1.0]
      },
      "expected": {
        "triggered": false,
        "direction": 0
      },
      "note": "σ=0 when all same, handle gracefully"
    },
    {
      "name": "insufficient_history",
      "input": {
        "value": 5.0,
        "history": [1.0, 2.0]
      },
      "expected": {
        "zscore": null,
        "triggered": false,
        "direction": 0
      },
      "note": "Less than min_periods=3"
    },
    {
      "name": "zero_std",
      "input": {
        "value": 5.0,
        "history": [2.0, 2.0, 2.0, 2.0, 2.0]
      },
      "expected": {
        "zscore": null,
        "triggered": false,
        "direction": 0
      },
      "note": "σ=0, cannot compute z-score"
    }
  ]
}
```

### spec/fixtures/streak_cases.json

```json
{
  "description": "Test cases for streak trigger",
  "spec_section": "§2.2",
  "formula": "Count consecutive same-sign values, trigger if streak >= min_streak",
  "params": {"min_streak": 3},
  "cases": [
    {
      "name": "positive_streak_triggers",
      "input": {
        "values": [0.1, 0.2, 0.15, 0.3]
      },
      "expected": {
        "streak": 4,
        "triggered": true,
        "direction": 1
      },
      "note": "4 consecutive positive values"
    },
    {
      "name": "negative_streak_triggers",
      "input": {
        "values": [-0.1, -0.2, -0.15]
      },
      "expected": {
        "streak": 3,
        "triggered": true,
        "direction": -1
      },
      "note": "Exactly min_streak negative values"
    },
    {
      "name": "streak_too_short",
      "input": {
        "values": [0.1, 0.2]
      },
      "expected": {
        "streak": 2,
        "triggered": false,
        "direction": 0
      },
      "note": "2 < min_streak of 3"
    },
    {
      "name": "streak_broken_by_sign_change",
      "input": {
        "values": [0.1, 0.2, -0.1, 0.3, 0.4]
      },
      "expected": {
        "streak": 2,
        "triggered": false,
        "direction": 0
      },
      "note": "Current streak is 2 (last two positive)"
    },
    {
      "name": "zero_breaks_streak",
      "input": {
        "values": [0.1, 0.2, 0.0, 0.3]
      },
      "expected": {
        "streak": 1,
        "triggered": false,
        "direction": 0
      },
      "note": "Zero has no sign, breaks streak"
    },
    {
      "name": "alternating_signs",
      "input": {
        "values": [0.1, -0.1, 0.1, -0.1, 0.1]
      },
      "expected": {
        "streak": 1,
        "triggered": false,
        "direction": 0
      },
      "note": "Never builds streak > 1"
    },
    {
      "name": "empty_input",
      "input": {
        "values": []
      },
      "expected": {
        "streak": 0,
        "triggered": false,
        "direction": 0
      }
    },
    {
      "name": "single_value",
      "input": {
        "values": [0.5]
      },
      "expected": {
        "streak": 1,
        "triggered": false,
        "direction": 0
      }
    }
  ]
}
```

---

## Notes

- Do NOT create `packages/signals/CLAUDE.md` - Claude Code only reads CLAUDE.md from current directory and parents, not sibling packages
- Root CLAUDE.md should include a Formula Registry table linking spec sections to code
- All tests should load expected values from `spec/fixtures/*.json`, not hardcode them
- Use `tolerance` field for floating-point comparisons
- Use `null` in JSON for values that should be `NaN` or `None` in Python
