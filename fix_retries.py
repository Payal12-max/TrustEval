"""
Backfill fix for metrics.retries in already-generated trace files.

Background:
    TraceLogger.calculate_metrics() originally searched for events
    of type "retry", but runner.py actually logs retries with type
    "retry_wait". This meant metrics.retries was always 0 in every
    saved trace, regardless of how many retries actually happened.

    The source has already been fixed in trace_logger.py, so any
    NEW traces will have the correct value. This script retroactively
    fixes metrics.retries in traces generated BEFORE that fix, by
    recounting "retry_wait" events directly from each trace's own
    events list (which was always correct -- only the aggregated
    count was wrong).

What it changes:
    Only trace["metrics"]["retries"]. Every other field -- events,
    final_output, timestamps, all other metrics -- is left exactly
    as-is.

Usage (from project root):
    python fix_retries.py --dry-run          # preview only, no writes
    python fix_retries.py                    # apply fixes
    python fix_retries.py --dir traces/split_a   # target one folder only
"""

import argparse
import json
from pathlib import Path


def fix_file(path: Path, dry_run: bool):

    with open(path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("events", [])

    correct_retries = sum(
        1 for e in events if e.get("type") == "retry_wait"
    )

    current_retries = trace.get("metrics", {}).get("retries", None)

    if current_retries == correct_retries:
        return None  # nothing to fix

    if not dry_run:
        trace.setdefault("metrics", {})["retries"] = correct_retries
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)

    return (path.name, current_retries, correct_retries)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="traces",
        help="Directory to search recursively for trace JSON files "
             "(default: traces/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing any files.",
    )
    args = parser.parse_args()

    traces_dir = Path(args.dir)

    if not traces_dir.exists():
        print(f"ERROR: directory not found: {traces_dir}")
        return

    trace_files = sorted(traces_dir.rglob("*.json"))

    # Skip anything that isn't an individual trace file (e.g. any
    # existing summary.json / trust_scores.json sitting alongside
    # traces, if this is pointed at a results dir by mistake).
    trace_files = [
        p for p in trace_files
        if p.name not in ("summary.json", "trust_scores.json",
                           "task_split.json")
    ]

    print(f"Scanning {len(trace_files)} file(s) under {traces_dir}...")
    if args.dry_run:
        print("(dry run -- no files will be modified)\n")

    changed = []
    skipped_bad_json = []

    for path in trace_files:
        try:
            result = fix_file(path, dry_run=args.dry_run)
        except (json.JSONDecodeError, KeyError) as e:
            skipped_bad_json.append((path.name, str(e)))
            continue

        if result:
            changed.append(result)

    print(f"\n{'Would fix' if args.dry_run else 'Fixed'}: {len(changed)} file(s)")
    for name, old, new in changed:
        print(f"  {name}: retries {old} -> {new}")

    if skipped_bad_json:
        print(f"\nSkipped {len(skipped_bad_json)} file(s) due to read errors:")
        for name, err in skipped_bad_json:
            print(f"  {name}: {err}")

    unchanged = len(trace_files) - len(changed) - len(skipped_bad_json)
    print(f"\nAlready correct / no change needed: {unchanged} file(s)")


if __name__ == "__main__":
    main()