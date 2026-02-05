# Skill: Verify Change

## Purpose

Verify that a code change is correct, safe, and follows FlowCode standards.

## Prerequisites

- Code change ready for review
- Access to test suite
- Understanding of affected components

## Steps

### 1. Comprehension Gates

Before approving any change, pass these gates:

#### 60-Second Explain

Can you explain the change in plain English in under 60 seconds?

- What problem does it solve?
- How does it solve it?
- What are the edge cases?

If you can't explain it simply, the change is too complex.

#### Change Simulation

Walk through the code path mentally:

1. What functions get called?
2. What data flows through?
3. Where could it fail?

#### 2am Debug Plan

If this code breaks at 2am:

1. What symptom would you see?
2. What log/metric would you check first?
3. How would you roll back?

### 2. Run Tests

```bash
# Run all tests
pytest packages/*/tests/ -v

# Run specific package tests
pytest packages/signals/tests/ -v

# Run with coverage
pytest --cov=packages --cov-report=term-missing
```

### 3. Check Spec Parity

```bash
# Verify formulas match spec
python tools/verify_spec_parity.py
```

### 4. Type Check

```bash
# Run mypy on all packages
mypy packages/*/src/

# Run on specific package
mypy packages/signals/src/
```

### 5. Lint

```bash
# Run ruff
ruff check packages/

# Auto-fix
ruff check packages/ --fix
```

### 6. Review Checklist

#### Code Quality
- [ ] Type hints on all functions
- [ ] Docstrings in NumPy format
- [ ] No hardcoded magic numbers (use conf/)
- [ ] Edge cases handled
- [ ] Errors have clear messages

#### Architecture
- [ ] No circular imports
- [ ] Package boundaries respected
- [ ] Data flows correctly (data → core → signals → backtest)

#### Testing
- [ ] Tests use fixtures from spec/
- [ ] Edge cases tested
- [ ] No flaky tests

#### Documentation
- [ ] SPEC.md updated if formula changed
- [ ] Formula Registry updated
- [ ] CLAUDE.md Failure Mode Catalog updated if new failure modes

### 7. Security Check

- [ ] No hardcoded credentials
- [ ] No SQL injection (if applicable)
- [ ] No path traversal vulnerabilities
- [ ] Input validation present

### 8. Performance Check

For large data operations:

- [ ] Uses vectorized operations (not row-by-row loops)
- [ ] Memory efficient (doesn't duplicate large DataFrames)
- [ ] Appropriate use of .copy() vs view

### 9. Final Verification

```bash
# Full CI check
pytest packages/*/tests/ -v && \
mypy packages/*/src/ && \
ruff check packages/ && \
python tools/verify_spec_parity.py
```

## Common Issues

### Issue: Tests Pass but Code is Wrong

**Cause:** Tests don't use spec fixtures
**Fix:** Update tests to load from `spec/fixtures/`

### Issue: Circular Import

**Cause:** Package A imports from Package B and vice versa
**Fix:** Check dependency graph:
```
data → core → signals → metrics → backtest
```

### Issue: Lookahead Bias

**Cause:** Using future data in signal calculation
**Symptoms:** Unrealistically high Sharpe ratio
**Fix:** Check all `.shift()` calls, ensure positions are lagged

### Issue: NaN Propagation

**Cause:** Missing data not handled
**Fix:** Add explicit NaN handling, check `min_periods`

## Checklist

- [ ] 60-second explain: PASS
- [ ] Change simulation: PASS
- [ ] 2am debug plan: DOCUMENTED
- [ ] All tests pass
- [ ] Spec parity verified
- [ ] Type check clean
- [ ] Lint clean
- [ ] Code review checklist complete
