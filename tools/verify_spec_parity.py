#!/usr/bin/env python3
"""Verify spec-code-test parity.

This tool checks that:
1. Every formula in SPEC.md has a corresponding implementation
2. The named function actually exists in the module (not just the file)
3. Every formula has a corresponding fixture file with valid JSON structure
"""
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
SPEC_PATH = ROOT / "spec" / "SPEC.md"
FIXTURES_DIR = ROOT / "spec" / "fixtures"

# Package name mapping: registry prefix -> directory + module prefix
_PKG_MAP = {
    "signals": ("signals", "flowcode_signals"),
    "metrics": ("metrics", "flowcode_metrics"),
    "backtest": ("backtest", "flowcode_backtest"),
    "data": ("data", "flowcode_data"),
    "alphaeval": ("alphaeval", "flowcode_alphaeval"),
}

REQUIRED_FIXTURE_KEYS = {"description", "cases"}


def parse_formula_registry(spec_content: str) -> list[dict]:
    """Extract formula registry from SPEC.md."""
    registry_match = re.search(
        r"\| Formula \| Spec § \| Code Location \| Fixture \|.*?\n((?:\|.*\n)+)",
        spec_content,
    )

    if not registry_match:
        print("ERROR: Could not find Formula Registry table in SPEC.md")
        return []

    table_rows = registry_match.group(1).strip().split("\n")

    formulas = []
    for row in table_rows:
        if row.startswith("|---") or row.startswith("| ---"):
            continue

        cols = [c.strip() for c in row.split("|")[1:-1]]
        if len(cols) >= 4:
            formulas.append({
                "name": cols[0],
                "spec_section": cols[1],
                "code_location": cols[2],
                "fixture": cols[3],
            })

    return formulas


def check_fixture(fixture_name: str) -> tuple[bool, str]:
    """Check fixture file exists and has valid JSON structure.

    Returns
    -------
    tuple[bool, str]
        (ok, message) — ok is True if fixture is valid.
    """
    fixture_path = FIXTURES_DIR / fixture_name
    if not fixture_path.exists():
        return False, f"file not found: {fixture_path.relative_to(ROOT)}"

    try:
        data = json.loads(fixture_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"invalid JSON: {exc}"

    if not isinstance(data, dict):
        return False, "top-level value must be a JSON object"

    missing = REQUIRED_FIXTURE_KEYS - data.keys()
    if missing:
        return False, f"missing required keys: {sorted(missing)}"

    cases = data["cases"]
    if not isinstance(cases, list) or len(cases) == 0:
        return False, "'cases' must be a non-empty array"

    for i, case in enumerate(cases):
        if "input" not in case and "inputs" not in case:
            return False, f"case[{i}] missing 'input' or 'inputs'"
        has_expected = (
            "expected" in case
            or "expected_low" in case
            or "expected_high" in case
        )
        if not has_expected:
            return False, f"case[{i}] missing 'expected' (or expected_low/expected_high)"

    return True, f"{len(cases)} cases"


def check_code_location(code_location: str) -> tuple[bool, str]:
    """Check that the module file exists AND the function is defined.

    Parameters
    ----------
    code_location : str
        Registry entry like ``signals.credit.credit_pnl()``.

    Returns
    -------
    tuple[bool, str]
        (ok, message).
    """
    parts = code_location.replace("()", "").split(".")

    if len(parts) < 3:
        return False, f"expected package.module.function, got '{code_location}'"

    pkg_key = parts[0]
    module_name = parts[1]
    func_name = parts[2]

    if pkg_key not in _PKG_MAP:
        return False, f"unknown package '{pkg_key}' (known: {sorted(_PKG_MAP)})"

    dir_name, mod_prefix = _PKG_MAP[pkg_key]
    module_path = ROOT / "packages" / dir_name / mod_prefix / f"{module_name}.py"

    if not module_path.exists():
        return False, f"module not found: {module_path.relative_to(ROOT)}"

    # Add package dir to sys.path so importlib can find it
    pkg_dir = str(ROOT / "packages" / dir_name)
    added = False
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
        added = True

    try:
        fq_module = f"{mod_prefix}.{module_name}"
        mod = importlib.import_module(fq_module)
        if not hasattr(mod, func_name):
            available = [a for a in dir(mod) if not a.startswith("_")]
            return False, (
                f"function '{func_name}' not found in {fq_module}; "
                f"available: {available[:10]}"
            )
        return True, f"{fq_module}.{func_name}"
    except Exception as exc:
        return False, f"import error: {exc}"
    finally:
        if added:
            sys.path.remove(pkg_dir)


def main() -> int:
    """Run parity checks."""
    print("FlowCode Spec Parity Checker")
    print("=" * 50)

    if not SPEC_PATH.exists():
        print(f"ERROR: SPEC.md not found at {SPEC_PATH}")
        return 1

    spec_content = SPEC_PATH.read_text()
    formulas = parse_formula_registry(spec_content)

    if not formulas:
        print("ERROR: No formulas found in registry")
        return 1

    print(f"Found {len(formulas)} formulas in registry\n")

    errors: list[str] = []
    warnings: list[str] = []

    for formula in formulas:
        name = formula["name"]
        fixture = formula["fixture"]
        code_loc = formula["code_location"]

        print(f"  {name}")

        # Check fixture
        if fixture and fixture != "N/A":
            ok, msg = check_fixture(fixture)
            if ok:
                print(f"    [OK]   fixture  {fixture} ({msg})")
            else:
                print(f"    [FAIL] fixture  {msg}")
                errors.append(f"{name}: fixture — {msg}")
        else:
            print(f"    [SKIP] fixture  (none specified)")

        # Check code location (file + function)
        ok, msg = check_code_location(code_loc)
        if ok:
            print(f"    [OK]   code     {msg}")
        else:
            print(f"    [FAIL] code     {msg}")
            errors.append(f"{name}: code — {msg}")

    # Summary
    print()
    print("=" * 50)
    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    if warnings:
        print(f"PASSED with {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
        return 0

    print(f"PASSED — all {len(formulas)} formulas verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
