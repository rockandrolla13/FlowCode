# Skill: Add Signal

## Purpose

Add a new signal to the FlowCode signals package.

## Prerequisites

- Understand the signal formula and its financial meaning
- Know the input data requirements
- Have test cases ready

## Steps

### 1. Update SPEC.md

Add the new signal formula to `spec/SPEC.md`:

```markdown
### X.X Signal Name

**Formula:**
\`\`\`
formula_here
\`\`\`

**Parameters:**
- param1: description (default: value)

**Code Location:** `packages/signals/src/new_signal.py::function_name()`
**Fixture:** `spec/fixtures/new_signal_cases.json`
```

### 2. Create Fixture

Create `spec/fixtures/new_signal_cases.json`:

```json
{
  "description": "Golden test cases for new signal",
  "spec_section": "§X.X",
  "formula": "formula_here",
  "cases": [
    {
      "name": "basic_case",
      "input": {...},
      "expected": value
    }
  ]
}
```

### 3. Implement Signal

Create `packages/signals/src/new_signal.py`:

```python
"""New signal implementation.

See SPEC.md §X.X for formula definition.
"""

import pandas as pd


def compute_new_signal(
    data: pd.DataFrame,
    param1: int = default_value,
) -> pd.Series:
    """
    Compute new signal.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with required columns.
    param1 : int, default default_value
        Description.

    Returns
    -------
    pd.Series
        Signal values indexed by date.
    """
    # Implementation matching SPEC.md formula
    ...
```

### 4. Add Tests

Create `packages/signals/tests/test_new_signal.py`:

```python
"""Tests for new signal."""

import json
from pathlib import Path
import pytest

from src.new_signal import compute_new_signal

FIXTURES = json.loads(
    (Path(__file__).parent.parent.parent.parent / "spec/fixtures/new_signal_cases.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda c: c["name"])
def test_from_spec(case):
    result = compute_new_signal(**case["input"])
    assert result == pytest.approx(case["expected"], rel=1e-6)
```

### 5. Update Package __init__.py

Add export to `packages/signals/src/__init__.py`:

```python
from .new_signal import compute_new_signal
```

### 6. Update Formula Registry

Add to the Formula Registry table in `spec/SPEC.md`:

```markdown
| New Signal | §X.X | signals.new_signal.compute_new_signal() | new_signal_cases.json |
```

### 7. Update CLAUDE.md (if needed)

If the signal introduces new concepts, add to the Failure Mode Catalog.

### 8. Verify

```bash
# Run tests
pytest packages/signals/tests/test_new_signal.py -v

# Check spec parity
python tools/verify_spec_parity.py

# Type check
mypy packages/signals/src/new_signal.py
```

## Checklist

- [ ] Formula added to SPEC.md
- [ ] Fixture created with edge cases
- [ ] Implementation matches spec exactly
- [ ] Tests load from fixtures
- [ ] Export added to __init__.py
- [ ] Formula Registry updated
- [ ] All tests pass
- [ ] Type hints complete
