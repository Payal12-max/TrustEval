"""
Trace triage tool.

Scans a folder of trace JSON files and flags ones that likely
need a closer manual look -- truncated responses, tool errors,
suspiciously short output, non-success status, and unusually
long execution times. This does NOT replace your teammate's
process-based evaluation methodology; it's a fast mechanical
pre-filter so you're not reading all 90+ traces by hand just to
find the handful that actually have problems.

Place this at: scoring/triage_traces.py (or wherever your
scoring scripts live)

Run from the project root:

    python scoring/triage_traces.py traces/split_a
    python scoring/triage_traces.py traces/split_b --min-chars 300

Output:
  - A summary printed to the console (counts by flag, by
    framework, by task).
  - A CSV file (triage_report.csv, in the same folder you
    scanned) listing every trace with its flags, so you can
    open it in Excel/Sheets and sort/filter.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

# A final_output shorter than this is flagged as suspiciously
# short -- adjust with --min-chars if your tasks legitimately
# produce short answers.
DEFAULT_MIN_CHARS = 200

# Execution times above this (seconds) are flagged as unusually
# long, often correlating with non-convergence / thrashing.
LONG_EXECUTION_SECONDS = 60

# Heuristic: a final_output that doesn't end with one of these
# characters is likely cut off mid-sentence or mid-table.
# Markdown formatting (trailing "---" dividers, bolded closing
# lines like "**text.**") is common in legitimate, complete
# responses and is stripped before checking, so it doesn't
# trigger false positives.
SENTENCE_END_CHARS = (
    ".", "!", "?", '"', "'", ")", "]", "}", "|", ":",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
)

# Trailing markdown noise to strip before checking the last
# character -- horizontal rules, bold/italic markers, and
# trailing whitespace/newlines.
TRAILING_MARKDOWN_STRIP = ("-", "*", "_", "#", " ", "\n", "\t")


# ============================================================
# Per-trace checks
# ============================================================

def check_trace(trace, filename):
    """
    Run all mechanical checks on a single trace dict.
    Returns a list of flag strings (empty list = looks clean).
    """

    flags = []

    status = trace.get("status", "unknown")
    if status != "success":
        flags.append(f"STATUS_{status.upper()}")

    final_output = trace.get("final_output") or ""

    if not final_output.strip():
        flags.append("EMPTY_OUTPUT")

    elif len(final_output.strip()) < DEFAULT_MIN_CHARS:
        flags.append("SHORT_OUTPUT")

    else:
        # Strip trailing markdown noise (horizontal rules like
        # "---", bold/italic markers, whitespace) before checking
        # the last real character -- a response ending in "---"
        # or "**text.**" is not truncated, it's just formatted.
        trimmed = final_output.strip().rstrip("".join(TRAILING_MARKDOWN_STRIP))

        if trimmed and trimmed[-1] not in SENTENCE_END_CHARS:
            flags.append("POSSIBLY_TRUNCATED")

    execution_time = trace.get("execution_time_seconds", 0)
    if execution_time and execution_time > LONG_EXECUTION_SECONDS:
        flags.append("LONG_EXECUTION")

    metrics = trace.get("metrics", {}) or {}
    if metrics.get("errors", 0) > 0:
        flags.append("HAS_ERRORS")

    events = trace.get("events", []) or []

    tool_error_count = sum(
        1 for event in events if event.get("type") == "tool_error"
    )
    if tool_error_count > 0:
        flags.append(f"TOOL_ERRORS_{tool_error_count}")

    for event in events:
        response_metadata = event.get("response_metadata")
        if response_metadata and response_metadata.get("finish_reason") == "length":
            flags.append("FINISH_REASON_LENGTH")
            break

    retry_count = sum(
        1 for event in events if event.get("type") == "retry_wait"
    )
    if retry_count > 0:
        flags.append(f"RETRIES_{retry_count}")

    return flags


# ============================================================
# Main scan
# ============================================================

def scan_folder(folder_path, min_chars):

    global DEFAULT_MIN_CHARS
    DEFAULT_MIN_CHARS = min_chars

    trace_files = sorted(folder_path.glob("*.json"))

    if not trace_files:
        print(f"No trace files found in {folder_path}")
        sys.exit(1)

    rows = []
    flag_counts = {}
    framework_flag_counts = {}
    clean_count = 0

    for trace_file in trace_files:

        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception as error:
            rows.append({
                "filename": trace_file.name,
                "task_id": "UNREADABLE",
                "framework": "UNREADABLE",
                "repetition": "",
                "status": "UNREADABLE",
                "flags": f"JSON_PARSE_ERROR: {error}",
            })
            continue

        flags = check_trace(trace, trace_file.name)

        if not flags:
            clean_count += 1

        framework = trace.get("framework", "unknown")

        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
            key = (framework, flag)
            framework_flag_counts[key] = framework_flag_counts.get(key, 0) + 1

        rows.append({
            "filename": trace_file.name,
            "task_id": trace.get("task_id", ""),
            "framework": framework,
            "repetition": trace.get("repetition", ""),
            "status": trace.get("status", ""),
            "flags": "; ".join(flags) if flags else "clean",
        })

    return rows, flag_counts, framework_flag_counts, clean_count, len(trace_files)


def print_summary(flag_counts, framework_flag_counts, clean_count, total_count):

    print()
    print("=" * 60)
    print("TRIAGE SUMMARY")
    print("=" * 60)
    print(f"Total traces scanned: {total_count}")
    print(f"Clean (no flags):     {clean_count}")
    print(f"Flagged:              {total_count - clean_count}")

    if flag_counts:
        print("\nFlag breakdown:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {flag:<30} {count}")

    if framework_flag_counts:
        print("\nBy framework:")
        frameworks = sorted(set(fw for fw, _ in framework_flag_counts.keys()))
        for framework in frameworks:
            print(f"\n  {framework}:")
            for (fw, flag), count in sorted(framework_flag_counts.items()):
                if fw == framework:
                    print(f"    {flag:<28} {count}")

    print()


def write_csv(rows, output_path):

    fieldnames = ["filename", "task_id", "framework", "repetition", "status", "flags"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Full report written to: {output_path}")


# ============================================================
# Entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Triage a folder of trace JSON files for common problems."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder of trace JSON files (e.g. traces/split_a).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help=f"Flag final_output shorter than this many characters (default: {DEFAULT_MIN_CHARS}).",
    )
    args = parser.parse_args()

    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"ERROR: folder not found: {folder_path}")
        sys.exit(1)

    rows, flag_counts, framework_flag_counts, clean_count, total_count = scan_folder(
        folder_path, args.min_chars
    )

    print_summary(flag_counts, framework_flag_counts, clean_count, total_count)

    output_path = folder_path / "triage_report.csv"
    write_csv(rows, output_path)


if __name__ == "__main__":
    main()