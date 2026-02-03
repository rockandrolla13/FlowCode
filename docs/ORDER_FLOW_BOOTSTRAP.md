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

## Notes

- Do NOT create `packages/signals/CLAUDE.md` - Claude Code only reads CLAUDE.md from current directory and parents, not sibling packages
- Root CLAUDE.md should include a Formula Registry table linking spec sections to code
- All tests should load expected values from `spec/fixtures/*.json`, not hardcode them
