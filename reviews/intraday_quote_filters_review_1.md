# Code Review: intraday_quote_filters.py — Iteration 1

## P0 — Critical Bugs

### 1. `.diff()` across ISINs produces false signals (lines 87, 133, 180-181, 233)
All filter functions call `.diff()` on the raw data column without grouping by ISIN.
If the DataFrame contains multiple ISINs sorted by (isin, time_bin), the diff at the
boundary between ISIN A's last row and ISIN B's first row compares two different bonds.

Example: ISIN A last offer_spread = 150bps, ISIN B first offer_spread = 80bps.
diff = -70bps → flags as massive offer tightening. **This is a false signal.**

Same bug in `filter_spread_compression_events` (ba.diff()) and both FV-anchored filters
are immune (they compute deviation, not diff).

**Fix**: If data contains multiple ISINs, group by ISIN before diffing.
At minimum, document the single-ISIN precondition. Best: add `isin_col` parameter
and use `groupby().diff()` when present.

## P1 — Important Issues

### 2. `venue_z_map` inline dicts instead of `venue_z_col()` method (lines 275-280, 322-327)
The `filter_rich_cheap_extremes` and `filter_rich_cheap_carry_scaled` functions
create inline venue_z_map dicts instead of using the `IntradayFVSchema.venue_z_col()`
method we added in the analytics module.

### 3. No input validation on `direction` parameter (lines 285-290, 339-344)
Invalid values like "CHEAP" silently fall through to the `else` (abs filter).
Should validate against known values.

### 4. Unused import: `IntradayFVSchema` (line 39)
Imported but never directly referenced. Accessed via `JoinedIntradaySchema.fv`.
Remove to keep imports clean.

## P2 — Minor

### 5. Staleness recomputed per filter call (lines 93-96, 138-141, etc.)
If chaining multiple filters, `quote_staleness()` re-parses timestamps each time.
Not a correctness issue but a perf concern with large DataFrames.

### 6. `filter_rich_cheap_carry_scaled` direction=0 labeled "cheap" (line 349)
`carry_days >= 0` includes zero. Moot in practice (min_carry_days > 0 filters it),
but semantically zero deviation is neither rich nor cheap.
