# Code Review: intraday_microstructure_analytics.py — Iteration 2

All P0 bugs from iteration 1 are fixed. Second-pass findings:

## P1 — Performance

### 1. `_make_reset_mask` DAILY branch uses Python loop (line 218-220)
The WEEKLY branch is vectorized (line 223-226), but DAILY uses a Python for-loop.
For large datasets (5000 ISINs × 78 bins/day = 390K rows), this is slow.
**Fix**: Vectorize date comparison.

### 2. `_make_reset_mask` INTRADAY_SESSION branch also uses Python loop (line 231-236)
Same perf concern. Vectorizable with `np.diff` on dates and hour boundary mask.

### 3. `_make_reset_mask` empty array crash (line 218)
If `n == 0`, `resets[0] = True` is an IndexError. The callers have empty guards,
but `_make_reset_mask` itself should be robust since it's a reusable utility.

## P1 — Correctness

### 4. `cross_venue_dislocation_carry_scaled` unnecessary intermediate divisions (line 667-674)
The formula:
  dislocation_bps = USD / cr01
  carry_bps = daily_carry / cr01
  result = dislocation_bps / carry_bps = USD / daily_carry
The cr01 cancels out. The double division via cr01 only introduces floating-point
noise and extra NaN-guard checks. Simplify to: `dislocation_usd / daily_carry`.

### 5. `bid_ask_spread` sign convention depends on data vendor (line 399)
In credit: bid_spread > offer_spread (normal market). So `offer - bid < 0`.
The `bid_ask_regime` function treats `ba < 0` as "crossed" — this would flag
every normal observation if the data follows the natural convention.
**Note**: If the data vendor normalizes to offer_spread > bid_spread, this is fine.
Add a docstring note about the expected convention.

## P2 — Readability

### 6. `cumulative_spread_move` line 535 is >100 chars
Split the ternary for readability.
