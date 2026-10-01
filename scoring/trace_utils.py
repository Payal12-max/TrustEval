import re
from pathlib import Path


# Files that should never be treated as individual traces
EXCLUDED_FILES = {
    "summary.json",
    "trust_scores.json",
    "task_split.json",
}


def discover_trace_files(traces_dir: Path) -> list[Path]:
    """
    Recursively discover all trace JSON files under traces_dir.

    Expected structure:

        traces/
            split_a/
                trace1.json
                trace2.json
            split_b/
                trace3.json
                trace4.json

    Only JSON files inside split_a or split_b are considered traces.
    """

    traces_dir = Path(traces_dir)

    if not traces_dir.exists():
        return []

    trace_files = []

    for path in traces_dir.rglob("*.json"):
        # Ignore known non-trace JSON files
        if path.name in EXCLUDED_FILES:
            continue

        # Only accept files inside split_a or split_b
        parent_names = {
            parent.name.lower()
            for parent in path.parents
        }

        if "split_a" not in parent_names and "split_b" not in parent_names:
            continue

        trace_files.append(path)

    return sorted(trace_files)


def get_split(path: Path) -> str:
    """
    Determine whether a trace belongs to split_a or split_b.

    Returns:
        "split_a"
        "split_b"
        "unknown"
    """

    path = Path(path)

    for parent in path.parents:
        name = parent.name.lower()

        if name == "split_a":
            return "split_a"

        if name == "split_b":
            return "split_b"

    return "unknown"


def parse_trace_filename(path: Path) -> dict | None:
    """
    Parse trace filenames.

    Expected format:

        <task_id>_<framework>_rep<repetition>_<unique_id>.json

    Examples:

        productivity_019_crewai_rep5_bbcd6a6.json
        productivity_019_langchain_rep8_6ecf123.json
        productivity_020_autogen_rep1_b2af6a9.json

    Returns:

        {
            "task_id": "productivity_019",
            "framework": "crewai",
            "repetition": 5,
            "unique_id": "bbcd6a6"
        }

    Returns None if the filename does not match.
    """

    path = Path(path)
    filename = path.stem

    pattern = re.compile(
        r"^(?P<task_id>[a-zA-Z]+_\d{3})_"
        r"(?P<framework>[a-zA-Z0-9_-]+?)_"
        r"rep(?P<repetition>\d+)_"
        r"(?P<unique_id>[a-zA-Z0-9-]+)$",
        re.IGNORECASE,
    )

    match = pattern.match(filename)

    if not match:
        return None

    data = match.groupdict()

    return {
        "task_id": data["task_id"],
        "framework": data["framework"],
        "repetition": int(data["repetition"]),
        "unique_id": data["unique_id"],
    }


def resolve_trace_identity(trace: dict, path: Path) -> dict:
    """
    Resolve trace identity using both:

        1. Filename
        2. Metadata stored inside the trace JSON

    Filename is used as the primary source because the filename
    contains task ID, framework, repetition and unique ID.

    If JSON metadata disagrees with the filename, the mismatch
    is recorded instead of silently ignoring it.

    Returns:

        {
            "task_id": ...,
            "framework": ...,
            "repetition": ...,
            "unique_id": ...,
            "split": ...,
            "mismatches": [...]
        }
    """

    path = Path(path)

    filename_identity = parse_trace_filename(path)

    if filename_identity is None:
        filename_identity = {}

    task_id_filename = filename_identity.get("task_id")
    framework_filename = filename_identity.get("framework")
    repetition_filename = filename_identity.get("repetition")
    unique_id_filename = filename_identity.get("unique_id")

    task_id_json = trace.get("task_id")
    framework_json = trace.get("framework")
    repetition_json = trace.get("repetition")
    run_id_json = trace.get("run_id")

    mismatches = []

    # ---------------------------------------------------------
    # Task ID
    # ---------------------------------------------------------

    if (
        task_id_filename is not None
        and task_id_json is not None
        and str(task_id_filename).lower()
        != str(task_id_json).lower()
    ):
        mismatches.append({
            "field": "task_id",
            "filename": task_id_filename,
            "json": task_id_json,
        })

    # ---------------------------------------------------------
    # Framework
    # ---------------------------------------------------------

    if (
        framework_filename is not None
        and framework_json is not None
        and str(framework_filename).lower()
        != str(framework_json).lower()
    ):
        mismatches.append({
            "field": "framework",
            "filename": framework_filename,
            "json": framework_json,
        })

    # ---------------------------------------------------------
    # Repetition
    # ---------------------------------------------------------

    if (
        repetition_filename is not None
        and repetition_json is not None
    ):
        try:
            repetition_json_int = int(repetition_json)

            if repetition_filename != repetition_json_int:
                mismatches.append({
                    "field": "repetition",
                    "filename": repetition_filename,
                    "json": repetition_json,
                })

        except (TypeError, ValueError):
            mismatches.append({
                "field": "repetition",
                "filename": repetition_filename,
                "json": repetition_json,
            })

    # ---------------------------------------------------------
    # Resolve final values
    # ---------------------------------------------------------

    task_id = (
        task_id_filename
        if task_id_filename is not None
        else task_id_json
    )

    framework = (
        framework_filename
        if framework_filename is not None
        else framework_json
    )

    repetition = (
        repetition_filename
        if repetition_filename is not None
        else repetition_json
    )

    # The unique ID is normally only in the filename.
    unique_id = unique_id_filename

    # If there is no filename unique ID, use run_id as a fallback.
    if unique_id is None:
        unique_id = run_id_json

    # Normalize repetition if possible
    if repetition is not None:
        try:
            repetition = int(repetition)
        except (TypeError, ValueError):
            pass

    return {
        "task_id": task_id,
        "framework": framework,
        "repetition": repetition,
        "unique_id": unique_id,
        "split": get_split(path),
        "mismatches": mismatches,
    }


def group_key(trace: dict, path: Path) -> tuple:
    """
    Return the key used to group traces for evaluation.

    Traces are grouped by:

        (task_id, framework)

    Repetition number is NOT part of the grouping key.

    Example:

        productivity_019 + crewai

    will contain:

        rep1
        rep2
        ...
        rep8
    """

    identity = resolve_trace_identity(trace, path)

    return (
        identity.get("task_id"),
        identity.get("framework"),
    )


def group_traces(traces: list) -> dict:
    """
    Group traces by (task_id, framework).

    Expected input:

        [
            {
                "trace": {...},
                "path": Path(...)
            },
            ...
        ]

    Returns:

        {
            ("productivity_019", "crewai"): [
                {...rep1...},
                {...rep2...},
                ...
            ],
            ("productivity_019", "langchain"): [
                {...rep1...},
                ...
            ]
        }

    Each group is sorted by repetition number.
    """

    groups = {}

    for item in traces:
        trace = item["trace"]
        path = item["path"]

        key = group_key(trace, path)

        groups.setdefault(key, []).append(item)

    # Sort every group by repetition number
    for key, items in groups.items():

        items.sort(
            key=lambda item: (
                resolve_trace_identity(
                    item["trace"],
                    item["path"]
                ).get("repetition") or 0
            )
        )

    return groups


def validate_repetition_groups(
    groups: dict,
    expected_repetitions: int = 8
) -> list:
    """
    Validate that each (task_id, framework) group contains
    the expected repetitions.

    Checks:

        - Number of traces
        - Missing repetitions
        - Duplicate repetitions
        - Identity mismatches
        - Invalid repetition numbers

    Returns a list of problems.

    Example problem:

        {
            "task_id": "productivity_019",
            "framework": "crewai",
            "problem": "missing_repetitions",
            "expected": [1,2,3,4,5,6,7,8],
            "found": [1,2,3,4,5,7,8]
        }
    """

    problems = []

    expected = set(range(1, expected_repetitions + 1))

    for (task_id, framework), items in groups.items():

        repetitions = []

        for item in items:
            trace = item["trace"]
            path = item["path"]

            identity = resolve_trace_identity(
                trace,
                path
            )

            repetition = identity.get("repetition")

            # -------------------------------------------------
            # Invalid repetition
            # -------------------------------------------------

            if repetition is None:
                problems.append({
                    "task_id": task_id,
                    "framework": framework,
                    "file": str(path),
                    "problem": "missing_repetition",
                })
                continue

            repetitions.append(repetition)

            # -------------------------------------------------
            # Identity mismatch
            # -------------------------------------------------

            mismatches = identity.get("mismatches", [])

            if mismatches:
                problems.append({
                    "task_id": task_id,
                    "framework": framework,
                    "file": str(path),
                    "problem": "identity_mismatch",
                    "mismatches": mismatches,
                })

        found = set(repetitions)

        # -----------------------------------------------------
        # Wrong number of traces
        # -----------------------------------------------------

        if len(items) != expected_repetitions:
            problems.append({
                "task_id": task_id,
                "framework": framework,
                "problem": "wrong_repetition_count",
                "expected_count": expected_repetitions,
                "found_count": len(items),
            })

        # -----------------------------------------------------
        # Missing repetitions
        # -----------------------------------------------------

        missing = sorted(expected - found)

        if missing:
            problems.append({
                "task_id": task_id,
                "framework": framework,
                "problem": "missing_repetitions",
                "expected": sorted(expected),
                "found": sorted(found),
                "missing": missing,
            })

        # -----------------------------------------------------
        # Duplicate repetitions
        # -----------------------------------------------------

        duplicates = sorted({
            repetition
            for repetition in repetitions
            if repetitions.count(repetition) > 1
        })

        if duplicates:
            problems.append({
                "task_id": task_id,
                "framework": framework,
                "problem": "duplicate_repetitions",
                "duplicates": duplicates,
            })

        # -----------------------------------------------------
        # Unexpected repetition numbers
        # -----------------------------------------------------

        unexpected = sorted(found - expected)

        if unexpected:
            problems.append({
                "task_id": task_id,
                "framework": framework,
                "problem": "unexpected_repetitions",
                "expected": sorted(expected),
                "found": sorted(found),
                "unexpected": unexpected,
            })

    return problems


def print_group_summary(groups: dict) -> None:
    """
    Print a human-readable summary of the discovered groups.

    Useful for checking Step 1 before modifying the scoring logic.
    """

    print("\nTrace groups:")
    print("-" * 70)

    for (task_id, framework), items in sorted(groups.items()):

        repetitions = []

        for item in items:
            identity = resolve_trace_identity(
                item["trace"],
                item["path"]
            )

            repetition = identity.get("repetition")

            if repetition is not None:
                repetitions.append(repetition)

        repetitions = sorted(repetitions)

        print(
            f"{task_id:<20} | "
            f"{framework:<12} | "
            f"{len(items)} traces | "
            f"repetitions: {repetitions}"
        )

    print("-" * 70)
    print(f"Total groups: {len(groups)}")


def print_validation_problems(problems: list) -> None:
    """
    Print repetition/identity validation problems.
    """

    if not problems:
        print("\n✓ No trace grouping problems found.")
        return

    print(
        f"\n⚠ Found {len(problems)} "
        f"trace grouping problem(s):"
    )

    print("-" * 70)

    for problem in problems:

        task_id = problem.get("task_id", "?")
        framework = problem.get("framework", "?")
        problem_type = problem.get("problem", "?")

        print(
            f"{task_id:<20} | "
            f"{framework:<12} | "
            f"{problem_type}"
        )

        if "file" in problem:
            print(
                f"  File: {problem['file']}"
            )

        if "missing" in problem:
            print(
                f"  Missing: {problem['missing']}"
            )

        if "duplicates" in problem:
            print(
                f"  Duplicates: {problem['duplicates']}"
            )

        if "unexpected" in problem:
            print(
                f"  Unexpected: {problem['unexpected']}"
            )

        if "mismatches" in problem:
            print(
                f"  Mismatches: {problem['mismatches']}"
            )

    print("-" * 70)