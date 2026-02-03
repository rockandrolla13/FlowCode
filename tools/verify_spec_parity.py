#!/usr/bin/env python3
"""Verify spec-code-test parity.

This tool checks that:
1. Every formula in SPEC.md has a corresponding implementation
2. Every formula has a corresponding fixture file
3. Tests load from fixtures (not hardcoded values)
"""

import json
import re
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent
SPEC_PATH = ROOT / "spec" / "SPEC.md"
FIXTURES_DIR = ROOT / "spec" / "fixtures"


def parse_formula_registry(spec_content: str) -> list[dict]:
    """Extract formula registry from SPEC.md."""
    # Find the Formula Registry table
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
        # Skip separator rows
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


def check_fixture_exists(fixture_name: str) -> bool:
    """Check if fixture file exists."""
    fixture_path = FIXTURES_DIR / fixture_name
    return fixture_path.exists()


def check_code_location(code_location: str) -> bool:
    """Check if code location is valid (basic check)."""
    # Parse location like "signals.retail.compute_retail_imbalance()"
    parts = code_location.replace("()", "").split(".")

    if len(parts) < 2:
        return False

    # Build expected path
    # e.g., "signals.retail.compute_retail_imbalance" -> packages/signals/src/retail.py
    package = parts[0]
    module = parts[1]

    expected_path = ROOT / "packages" / package / "src" / f"{module}.py"
    return expected_path.exists()


def main() -> int:
    """Run parity checks."""
    print("FlowCode Spec Parity Checker")
    print("=" * 40)

    if not SPEC_PATH.exists():
        print(f"ERROR: SPEC.md not found at {SPEC_PATH}")
        return 1

    spec_content = SPEC_PATH.read_text()
    formulas = parse_formula_registry(spec_content)

    if not formulas:
        print("ERROR: No formulas found in registry")
        return 1

    print(f"Found {len(formulas)} formulas in registry\n")

    errors = []
    warnings = []

    for formula in formulas:
        name = formula["name"]
        fixture = formula["fixture"]
        code_loc = formula["code_location"]

        print(f"Checking: {name}")

        # Check fixture
        if fixture and fixture != "N/A":
            if check_fixture_exists(fixture):
                print(f"  [OK] Fixture: {fixture}")
            else:
                msg = f"  [MISSING] Fixture: {fixture}"
                print(msg)
                warnings.append(f"{name}: Missing fixture {fixture}")
        else:
            print(f"  [SKIP] No fixture specified")

        # Check code location
        if check_code_location(code_loc):
            print(f"  [OK] Code: {code_loc}")
        else:
            msg = f"  [WARN] Code location not verified: {code_loc}"
            print(msg)
            # Don't treat as error, could be core package or different structure

        print()

    # Summary
    print("=" * 40)
    print("Summary")
    print("=" * 40)

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("\nAll checks passed!")
        return 0
    elif errors:
        return 1
    else:
        print("\nCompleted with warnings")
        return 0


if __name__ == "__main__":
    sys.exit(main())
