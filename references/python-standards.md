# Python Standards — Code Review Criteria

## Conciseness & Economy

Look for: redundant variables (inline if used once), repeated expressions (extract if ≥2 uses), verbose conditionals (simplify without nesting), dead code / unreachable branches, unnecessary type conversions, overly defensive checks that duplicate language guarantees.

Avoid recommending: complex ternary chains, implicit boolean coercion tricks, one-liners that sacrifice readability, premature abstraction, changing public interfaces without approval.

## Compatibility & Runtime Safety

**Version Compatibility:** All language/library features must be compatible with the project's declared runtime (e.g., Python 3.9). Replace or guard incompatible features.
- Example: Python 3.10+ `str | None` → `typing.Optional[str]` for 3.9

**Scalar vs Vectorized Type Discipline:** Functions must document and enforce expected input types. Never call scalar-only helpers on Series/arrays. Add type checks at boundaries or provide `*_scalar` / `*_vectorized` variants.
- Example: If `is_close_to_zero(value: float)` is scalar-only, use `(series.abs() < EPSILON)` for vectorized paths.

**Public API Contract:** Public functions must have one canonical calling convention. If both kwargs and config objects are supported, implement a single normalization layer and test both paths.

**Return Type Consistency:** Return types must be stable across implementations. Tests must access results via the declared type. If dict-like access is desired, implement `to_dict()` explicitly and test that adapter.
- Example: GOOD: `result.zscore` / BAD: `result["zscore"]` when return type is a dataclass.

## Semantics & Correctness

**Scalar/Vectorized Parity:** Scalar and vectorized implementations must match semantics exactly — window definition, inclusion/exclusion of current point, boundary behavior. Any intentional divergence must be documented and tested.
- Example: Add parity test: `assert scalar_result.zscore ≈ vectorized_result["zscore"].iloc[-1]`

**No Look-Ahead Bias:** Rolling/window computations must clearly define whether the current observation is included. Default signal generation should avoid look-ahead. Add a parity test for inclusion/exclusion mismatches.

**Edge-Case Alignment (Zero/NaN/Inf):** Handling of zeros, near-zeros, NaNs, and infinities must be consistent across all implementations. If epsilon comparisons are used in one path, the same approach must exist in the other (or the divergence must be spec'd and tested).

**Zero-Variance / Division-by-Zero Safety:**
- When `std==0`: z-score is undefined → return `triggered=False`, `direction=neutral`, `score=NaN`.
- Before applying thresholds, ensure score arrays contain no `inf/-inf`. Require masking/replacement.
- If the agent detects division by `rolling_std/std` without masking `denom==0`, flag as correctness bug.
- If `triggered = score.abs() > threshold` is applied to potentially-inf values, flag it.
- Acceptance: no inf/-inf reaches threshold comparisons; zero variance never triggers; scalar/vectorized match on edge cases.

## Performance & Allocation (Hot Paths)

**No Hidden Copying:** Avoid repeated allocations (Series reassignments, slicing copies, repeated conversions). Prefer single-pass vectorized operations (`np.where`, boolean masks) over chained assignments.
- Example: BAD: `direction = pd.Series(0, ...); direction[mask1] = 1` / GOOD: `np.where(mask1, 1, np.where(mask2, -1, 0))`

**No Python-Level Work in Vectorized Functions:** No per-element string ops, Python loops, or `.apply()` in vectorized paths. Pre-map categories to numeric codes if strings are unavoidable.

**Minimize Pandas ↔ NumPy Conversions:** Choose one representation per function boundary; convert once at ingress/egress. Be suspicious of anything that looks "vectorized" but creates multiple temporaries.

**Visible Allocations:** Call out every new list/Series/DataFrame allocation inside loops or per-tick code. Prefer patterns that reuse objects or operate on preallocated arrays.

**Benchmark as Definition of Done:** Hot-path changes require micro-benchmarks (before/after). Add perf tests or allocation-count checks where feasible.

**Data Structure Selection:** Trailing windows → ring buffer / deque + rolling stats (not slicing). Numeric throughput → numpy arrays (not Python lists or pandas objects).

## Configuration Management

Single factory — centralize config parsing in one module, never parse raw dicts inline. Typed schema (dataclass/pydantic), validate on load. Immutable runtime configs — load once, pass the typed object via DI. No duplicated extraction logic. Cache expensive computed fields. Version your config schema. Provide `make_test_config(overrides={})` for tests.

## Testing

**Goal:** Determinism, strong assertions, real negatives, meaningful perf checks.

**API/Signature Tests:** Include a test validating calling conventions (kwargs vs config). Add a smoke test exercising public entry points as documented.

**Result Shape/Type Tests:** Assert result type and access pattern. Include serialization test if dict form is used.

**Parity Tests (Scalar vs Vectorized):** Parameterized test on same seeded inputs. Adversarial fixtures: exact zeros, near-zeros, NaNs, constant series, alternating signs.

**Coverage Gate:** Every exported function has at least one direct test. Vectorized variants have at least one correctness test and one edge-case test. All `__init__.py` exports appear in the test suite.

**Performance Tests Must Assert:** Include timing/regression assertion (absolute threshold or relative to baseline). No benchmark-only tests that never fail.

**Strong Assertions:** Assert exact expected outputs where feasible. If multiple valid outcomes, assert invariants (monotonicity, shape, sign) plus one concrete expected value for a canonical seed.

**Negative Tests:** Explicit tests for edge/invalid inputs (empty history, non-finite values, all-NaN windows). For vectorized: contiguous sequence of invalids that should produce a defined result.

**Determinism:** Fixed RNG seeds, fixed timestamps, or mocked time sources.

**Style:** Keep tests concise (50–100 lines preferred). Use minimal docstrings. Access data via `.loc[]`, `.iloc[]`, or attribute access — never `df["col"][idx]` chaining.

**PR Review Checklist:**
Config centralized and typed? | Duplicated extraction logic? | Strong assertions for canonical inputs? | Vectorized negative/empty-case tests? | Hot-path microbenchmarks? | RNG/time sources seeded/mocked?
