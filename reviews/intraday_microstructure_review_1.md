# Code Review: intraday_microstructure_analytics.py — Iteration 1

## P0 — Critical Bugs

### 1. NaN poisoning in `intraday_spread_range` (line 474)
When `mids[0]` is NaN (or a reset index has NaN), `cur_high = cur_low = NaN`.
Then `max(NaN, valid_value)` in Python returns NaN (NaN comparisons are always False,
so `max` keeps the first argument). The `if not np.isnan(mids[i])` guard doesn't help
because the *accumulator* is poisoned, not the input.

**Impact**: All highs/lows are NaN from session start until the next reset, even if
there are valid values after the first NaN.

**Fix**: When a non-NaN value arrives and the accumulator is NaN, assign directly.

### 2. NaN poisoning in `cumulative_spread_move` (line 508)
Same pattern: `session_open = mids[0]`. If NaN, `cum_move[i] = mids[i] - NaN = NaN`
for all subsequent non-NaN values until the next reset.

### 3. `spread_time_profile` leaks POC across reset boundaries (line 696)
When `resets[i]` is True AND `mids[i]` is NaN:
- `profile.clear()` runs (correct)
- Then falls into `if np.isnan(mids[i])` → `poc[i] = poc[i-1]` (wrong: grabs pre-reset POC)

**Fix**: Track `last_poc` per session and reset it to NaN on session boundaries.

## P1 — Important Issues

### 4. `carry_breakeven_spread_move` is identical to `carry_per_unit_risk` (lines 262-290)
Both compute `total_carry(data, fv, horizon_days) / unit_cr01.where(...)`.
Identical formula, identical units (bps). One of them is dead code or has the wrong formula.

### 5. `liquidity_filtered_universe` OR logic overwrites prior filters (line 580)
When `require_both=False` and both MA/TW scores are set, the OR branch does:
`mask = (MA_check | TW_check)` — this replaces the entire mask, discarding any
filters that were applied before the MA check. Currently harmless (no prior filters),
but extremely fragile.

### 6. Empty data crashes (lines 474, 508, etc.)
`mids[0]` raises IndexError when DataFrame is empty. Affects:
- `intraday_spread_range`
- `cumulative_spread_move`
- `spread_time_profile`
- `time_at_spread`

### 7. WEEKLY reset doesn't account for year boundaries (line 223)
`ts.isocalendar().week` alone: week 52 of year N+1 and week 52 of year N
would not trigger a reset. Should include year in comparison.

## P2 — Minor

### 8. `JoinedIntradaySchema` type hints are `IntradayQuoteSchema`/`IntradayFVSchema` but defaults are `None`
Should be `Optional[IntradayQuoteSchema]` and `Optional[IntradayFVSchema]`.

### 9. Inconsistent return types across layers
Profile functions return raw `np.ndarray` without index, making them hard
to join back. Consider wrapping in Series/DataFrame with same index as input.
