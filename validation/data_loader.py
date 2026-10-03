"""
TrustEval Member 3
Data loader for Member 2 trust_scores.json

This loader:
- Finds trust_scores.json relative to the project
- Supports the current Member 2 JSON format
- Converts all runs into a pandas DataFrame
- Does NOT require human ratings
"""

import json
from pathlib import Path

import pandas as pd


DIMENSIONS = [
    "safety",
    "robustness",
    "reliability",
    "goal_adherence",
]


def find_trust_scores():
    """
    Automatically locate trust_scores.json.

    Expected project structure:

        TrustEval-main/
        ├── results/
        │   └── trust_scores.json
        ├── scoring/
        ├── validation/
        └── validation_scripts/
    """

    # data_loader.py is:
    # TrustEval-main/validation/data_loader.py
    #
    # parents[1] = TrustEval-main
    project_root = Path(__file__).resolve().parents[1]

    possible_paths = [
        project_root / "results" / "trust_scores.json",
        project_root / "result" / "trust_scores.json",
        project_root / "outputs" / "trust_scores.json",
    ]

    for path in possible_paths:
        if path.exists() and path.is_file():
            return path

    # If it wasn't found, give a useful error message.
    searched = "\n".join(
        f"  - {path}"
        for path in possible_paths
    )

    raise FileNotFoundError(
        "\nCould not find trust_scores.json.\n\n"
        "The program searched these locations:\n"
        f"{searched}\n\n"
        "Expected location:\n"
        f"  {project_root / 'results' / 'trust_scores.json'}\n\n"
        "Please make sure Member 2's trust_scores.json exists "
        "inside the results folder."
    )


def load_trust_scores(path=None):
    """
    Load and flatten Member 2 trust_scores.json.

    Returns a pandas DataFrame with columns:

        trace_id
        task_id
        agent_id
        framework
        domain
        repetition
        status
        execution_time_seconds
        safety
        robustness
        reliability
        goal_adherence
        composite_score
    """

    # ---------------------------------------------------------
    # FIND FILE
    # ---------------------------------------------------------

    if path is None:
        path = find_trust_scores()

    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            "\ntrust_scores.json does not exist at:\n"
            f"{path}\n\n"
            "Check that the file exists in the results folder."
        )

    print(f"\nLoading trust scores from:")
    print(f"  {path}")

    # ---------------------------------------------------------
    # LOAD JSON
    # ---------------------------------------------------------

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    # ---------------------------------------------------------
    # HANDLE DIFFERENT JSON WRAPPERS
    # ---------------------------------------------------------

    if isinstance(raw, dict):

        if "runs" in raw:
            raw = raw["runs"]

        elif "results" in raw:
            raw = raw["results"]

        else:
            # Single run stored as a dictionary
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(
            "Invalid trust_scores.json format.\n"
            "Expected a list of run records."
        )

    # ---------------------------------------------------------
    # FLATTEN RECORDS
    # ---------------------------------------------------------

    rows = []

    for entry in raw:

        if not isinstance(entry, dict):
            continue

        scores = entry.get("scores", {}) or {}

        composite = scores.get(
            "composite",
            {}
        ) or {}

        dimension_scores = composite.get(
            "dimension_scores",
            {}
        ) or {}

        # -----------------------------------------------------
        # BASIC INFORMATION
        # -----------------------------------------------------

        framework = (
            entry.get("framework")
            or entry.get("agent_id")
            or entry.get("agent")
        )

        task_id = entry.get("task_id")

        domain = (
            entry.get("domain")
            or "unknown"
        )

        repetition = entry.get(
            "repetition"
        )

        status = entry.get(
            "status"
        )

        execution_time = entry.get(
            "execution_time_seconds"
        )

        trace_id = (
            entry.get("run_id")
            or entry.get("trace_id")
        )

        row = {
            "trace_id": trace_id,
            "task_id": task_id,
            "agent_id": framework,
            "framework": framework,
            "domain": str(domain).lower(),
            "repetition": repetition,
            "status": status,
            "execution_time_seconds": execution_time,
        }

        # -----------------------------------------------------
        # TRUST DIMENSIONS
        # -----------------------------------------------------

        for dimension in DIMENSIONS:

            value = None

            # Format 1:
            #
            # scores:
            #   safety:
            #       safety_score: 0.8

            dimension_object = scores.get(
                dimension,
                {}
            ) or {}

            if isinstance(
                dimension_object,
                dict
            ):

                value = dimension_object.get(
                    f"{dimension}_score"
                )

                if value is None:
                    value = dimension_object.get(
                        "score"
                    )

            # Format 2:
            #
            # composite:
            #   dimension_scores:
            #       safety: 0.8

            if value is None:

                value = dimension_scores.get(
                    dimension
                )

            # Format 3:
            #
            # direct field:
            #   safety: 0.8

            if value is None:

                value = entry.get(
                    dimension
                )

            row[dimension] = pd.to_numeric(
                value,
                errors="coerce"
            )

        # -----------------------------------------------------
        # COMPOSITE SCORE
        # -----------------------------------------------------

        composite_value = composite.get(
            "composite_trust_score"
        )

        if composite_value is None:

            composite_value = composite.get(
                "score"
            )

        # If composite score isn't explicitly stored,
        # calculate it using the weights saved by Member 2.

        if composite_value is None:

            weights = composite.get(
                "weights_applied",
                {}
            ) or {}

            if (
                weights
                and all(
                    pd.notna(row[d])
                    for d in DIMENSIONS
                )
            ):

                composite_value = sum(
                    row[d]
                    * float(
                        weights.get(d, 0)
                    )
                    for d in DIMENSIONS
                )

        row["composite_score"] = pd.to_numeric(
            composite_value,
            errors="coerce"
        )

        rows.append(row)

    # ---------------------------------------------------------
    # CREATE DATAFRAME
    # ---------------------------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No valid run records were found "
            "in trust_scores.json."
        )

    # ---------------------------------------------------------
    # CONVERT NUMERIC COLUMNS
    # ---------------------------------------------------------

    numeric_columns = (
        DIMENSIONS
        + [
            "composite_score",
            "execution_time_seconds",
            "repetition",
        ]
    )

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # ---------------------------------------------------------
    # BASIC CLEANUP
    # ---------------------------------------------------------

    df["framework"] = df["framework"].astype(str)
    df["agent_id"] = df["agent_id"].astype(str)
    df["domain"] = df["domain"].astype(str)

    return df


if __name__ == "__main__":

    print("=" * 60)
    print("TrustEval — Testing Data Loader")
    print("=" * 60)

    df = load_trust_scores()

    print(
        f"\nLoaded {len(df)} runs."
    )

    print(
        f"Tasks: {df['task_id'].nunique()}"
    )

    print(
        f"Frameworks: {df['framework'].nunique()}"
    )

    print(
        f"Domains: {df['domain'].nunique()}"
    )

    print("\nRuns per framework:")

    print(
        df.groupby("framework").size()
    )

    print("\nFirst 5 rows:")

    print(
        df.head()
    )