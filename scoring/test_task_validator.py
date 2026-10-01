"""
Temporary test runner for task-specific validators.

Usage:

    python -m scoring.test_task_validator "path/to/trace.json"

Example:

    python -m scoring.test_task_validator "traces/split_a/education_014_..."
"""

import sys
import json
from pathlib import Path

from scoring.task_validators import validate_task


def main():

    # -------------------------------------------------------------
    # Check command-line argument
    # -------------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            '  python -m scoring.test_task_validator '
            '"path/to/trace.json"'
        )

        return


    # -------------------------------------------------------------
    # Resolve path
    # -------------------------------------------------------------

    trace_path = Path(
        sys.argv[1]
    )

    if not trace_path.exists():

        print(
            f"[ERROR] File not found:\n{trace_path}"
        )

        return


    # -------------------------------------------------------------
    # Load trace
    # -------------------------------------------------------------

    try:

        with open(
            trace_path,
            "r",
            encoding="utf-8"
        ) as f:

            trace = json.load(f)

    except Exception as exc:

        print(
            f"[ERROR] Could not load trace: {exc}"
        )

        return


    # -------------------------------------------------------------
    # Run validator
    # -------------------------------------------------------------

    result = validate_task(
        trace
    )


    # -------------------------------------------------------------
    # Print header
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TASK VALIDATOR TEST")
    print("=" * 70)

    print(
        f"Trace : {trace_path.name}"
    )

    print(
        f"Task  : {trace.get('task_id')}"
    )

    print(
        f"Agent : {trace.get('framework')}"
    )

    print(
        f"Rep   : {trace.get('repetition')}"
    )

    print("=" * 70)


    # -------------------------------------------------------------
    # Validator status
    # -------------------------------------------------------------

    print()

    print(
        f"Validator implemented : "
        f"{result.get('implemented')}"
    )

    print(
        f"Overall passed        : "
        f"{result.get('passed')}"
    )

    print(
        f"Validator score       : "
        f"{result.get('score')}"
    )


    # -------------------------------------------------------------
    # Individual checks
    # -------------------------------------------------------------

    print()
    print("CHECKS")
    print("-" * 70)

    checks = result.get(
        "checks",
        {}
    )

    for name, passed in checks.items():

        symbol = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"{symbol:<6} {name}"
        )


    # -------------------------------------------------------------
    # Extracted information
    # -------------------------------------------------------------

    print()
    print("EXTRACTED INFORMATION")
    print("-" * 70)

    extracted = result.get(
        "extracted",
        {}
    )

    print(
        json.dumps(
            extracted,
            indent=2,
            ensure_ascii=False
        )
    )


    # -------------------------------------------------------------
    # Issues
    # -------------------------------------------------------------

    print()
    print("ISSUES")
    print("-" * 70)

    issues = result.get(
        "issues",
        []
    )

    if not issues:

        print(
            "No issues detected."
        )

    else:

        for issue in issues:

            print(
                f"- {issue}"
            )


    # -------------------------------------------------------------
    # Full JSON result
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("RAW VALIDATOR RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()