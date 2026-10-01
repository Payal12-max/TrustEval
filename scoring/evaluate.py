"""
TrustEval Scoring Engine — Main Entry Point.

Evaluates TrustEval traces at two levels:

1. Individual level
   Every execution/repetition is scored independently.

2. Group level
   Traces are grouped by:
       (task_id, framework)

   Each group is expected to contain 8 repetitions.

   Group-level results contain:
       - the 8 individual scores
       - mean Safety
       - mean Robustness
       - repetition-aware Reliability
       - mean Goal Adherence
       - mean Composite Trust Score
       - success rate
       - validator information

Final domain/framework summaries aggregate the 90
task/framework groups rather than all 720 repetitions.

Expected experiment:

    30 tasks × 3 frameworks × 8 repetitions = 720 traces
    30 tasks × 3 frameworks = 90 groups

Output:

    results/
        individual/
            <trace>.json

        groups/
            <task>_<framework>.json

        trust_scores.json
            All 720 individual results

        group_scores.json
            All task/framework group results

        summary.json
            Domain/framework summary based on group results

Usage:

    python -m scoring.evaluate
"""

from pathlib import Path
import json

from scoring.trace_utils import (
    discover_trace_files,
    group_traces,
    validate_repetition_groups,
)

from scoring.safety_scorer import SafetyScorer
from scoring.robustness_scorer import RobustnessScorer
from scoring.reliability_scorer import ReliabilityScorer
from scoring.goal_adherence_scorer import GoalAdherenceScorer
from scoring.composite_scorer import CompositeScorer


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRACES_DIR = PROJECT_ROOT / "traces"
RESULTS_DIR = PROJECT_ROOT / "results"

INDIVIDUAL_DIR = RESULTS_DIR / "individual"
GROUP_DIR = RESULTS_DIR / "groups"

EXPECTED_REPETITIONS = 8


# ============================================================
# File helpers
# ============================================================

def load_trace(path: Path) -> dict:
    """Load one JSON trace."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    """Save JSON with consistent formatting."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# Domain
# ============================================================

def get_domain(trace: dict) -> str:
    """
    Determine the benchmark domain.

    Prefer task_metadata.domain.

    Otherwise infer it from the task ID.
    """

    task_metadata = trace.get(
        "task_metadata",
        {}
    ) or {}

    domain = task_metadata.get(
        "domain",
        ""
    )

    if domain:
        return str(domain).lower().strip()

    task_id = str(
        trace.get(
            "task_id",
            ""
        )
    )

    if "_" in task_id:
        return task_id.split(
            "_",
            1
        )[0].lower()

    return "default"


# ============================================================
# Individual evaluation
# ============================================================

def evaluate_trace(
    trace: dict,
    trace_path: Path | None = None,
) -> dict:
    """
    Evaluate one individual trace/repetition.
    """

    domain = get_domain(trace)

    safety_result = SafetyScorer().score(
        trace
    )

    robustness_result = RobustnessScorer().score(
        trace
    )

    reliability_result = ReliabilityScorer().score_single(
        trace
    )

    goal_result = GoalAdherenceScorer().score(
        trace
    )

    composite_result = CompositeScorer(
        domain
    ).score(
        safety_result,
        robustness_result,
        reliability_result,
        goal_result,
    )

    return {
        "level": "individual",

        "trace_file": (
            trace_path.name
            if trace_path is not None
            else None
        ),

        "task_id": trace.get(
            "task_id"
        ),

        "framework": trace.get(
            "framework"
        ),

        "domain": domain,

        "repetition": trace.get(
            "repetition"
        ),

        "run_id": trace.get(
            "run_id"
        ),

        "status": trace.get(
            "status"
        ),

        "execution_time_seconds": trace.get(
            "execution_time_seconds"
        ),

        "scores": {
            "safety": safety_result,
            "robustness": robustness_result,
            "reliability": reliability_result,
            "goal_adherence": goal_result,
            "composite": composite_result,
        },
    }


# ============================================================
# Score extraction
# ============================================================

def get_dimension_score(
    result: dict,
    dimension: str,
) -> float:
    """
    Extract a dimension score from an individual result.
    """

    score_result = result.get(
        "scores",
        {}
    ).get(
        dimension,
        {}
    )

    if not isinstance(
        score_result,
        dict
    ):
        return 0.0

    return float(
        score_result.get(
            f"{dimension}_score",
            0.0
        )
        or 0.0
    )


def get_composite_score(
    result: dict,
) -> float:
    """Extract the individual composite trust score."""

    composite = result.get(
        "scores",
        {}
    ).get(
        "composite",
        {}
    )

    if not isinstance(
        composite,
        dict
    ):
        return 0.0

    return float(
        composite.get(
            "composite_trust_score",
            0.0
        )
        or 0.0
    )


# ============================================================
# Simple mean
# ============================================================

def mean(values: list[float]) -> float:
    """Calculate a simple arithmetic mean."""

    if not values:
        return 0.0

    return sum(values) / len(values)


# ============================================================
# Validator information
# ============================================================

def collect_validator_information(
    individual_results: list[dict],
) -> dict:
    """
    Collect lightweight validator information across repetitions.

    This does NOT perform statistical analysis.

    It simply records:
        - how many repetitions had an implemented validator
        - how many validator evaluations passed
        - failed checks observed across repetitions
    """

    output = {}

    for result in individual_results:

        scores = result.get(
            "scores",
            {}
        )

        for dimension in (
            "safety",
            "reliability",
            "goal_adherence",
        ):

            dimension_result = scores.get(
                dimension,
                {}
            )

            if not isinstance(
                dimension_result,
                dict
            ):
                continue

            details = dimension_result.get(
                "details",
                {}
            )

            if not isinstance(
                details,
                dict
            ):
                continue

            validation = details.get(
                "task_validation"
            )

            if not isinstance(
                validation,
                dict
            ):
                continue

            if dimension not in output:
                output[dimension] = {
                    "implemented_repetitions": 0,
                    "passed_repetitions": 0,
                    "failed_checks": [],
                }

            if validation.get(
                "implemented",
                False
            ):
                output[dimension][
                    "implemented_repetitions"
                ] += 1

                validator_score = validation.get(
                    "validator_score"
                )

                if (
                    isinstance(
                        validator_score,
                        (int, float)
                    )
                    and validator_score >= 0.9999
                ):
                    output[dimension][
                        "passed_repetitions"
                    ] += 1

                failed_checks = validation.get(
                    "failed_checks",
                    []
                )

                if isinstance(
                    failed_checks,
                    list
                ):
                    output[dimension][
                        "failed_checks"
                    ].extend(
                        failed_checks
                    )

    return output


# ============================================================
# Group evaluation
# ============================================================

def evaluate_group(
    group_key,
    group_items: list,
    individual_results: list[dict],
) -> dict:
    """
    Aggregate one (task_id, framework) group.

    The group contains the 8 repetitions of one task executed
    by one framework.

    Only lightweight aggregation is performed here.
    """

    if not individual_results:
        raise ValueError(
            f"No individual results for group {group_key}"
        )

    first_result = individual_results[0]

    task_id = first_result.get(
        "task_id"
    )

    framework = first_result.get(
        "framework"
    )

    domain = first_result.get(
        "domain"
    )

    # --------------------------------------------------------
    # Individual scores
    # --------------------------------------------------------

    individual_scores = []

    for result in individual_results:

        individual_scores.append({
            "repetition": result.get(
                "repetition"
            ),

            "status": result.get(
                "status"
            ),

            "safety": get_dimension_score(
                result,
                "safety"
            ),

            "robustness": get_dimension_score(
                result,
                "robustness"
            ),

            "reliability": get_dimension_score(
                result,
                "reliability"
            ),

            "goal_adherence": get_dimension_score(
                result,
                "goal_adherence"
            ),

            "composite": get_composite_score(
                result
            ),
        })

    # --------------------------------------------------------
    # Across-repetition values
    # --------------------------------------------------------

    safety_scores = [
        item["safety"]
        for item in individual_scores
    ]

    robustness_scores = [
        item["robustness"]
        for item in individual_scores
    ]

    reliability_scores = [
        item["reliability"]
        for item in individual_scores
    ]

    goal_scores = [
        item["goal_adherence"]
        for item in individual_scores
    ]

    composite_scores = [
        item["composite"]
        for item in individual_scores
    ]

    # --------------------------------------------------------
    # Success rate
    # --------------------------------------------------------

    statuses = [
        item["status"]
        for item in individual_scores
    ]

    success_count = statuses.count(
        "success"
    )

    success_rate = (
        success_count / len(statuses)
        if statuses
        else 0.0
    )

    # --------------------------------------------------------
    # Reliability
    #
    # Use the dedicated repetition-aware method because
    # ReliabilityScorer already has explicit across-repetition
    # logic.
    # --------------------------------------------------------

    raw_traces = [
        item["trace"]
        for item in group_items
    ]

    reliability_group_result = (
        ReliabilityScorer().score_across_repetitions(
            raw_traces
        )
    )

    repetition_aware_reliability = float(
        reliability_group_result.get(
            "reliability_score",
            mean(reliability_scores)
        )
        or 0.0
    )

    # --------------------------------------------------------
    # Group dimension means
    # --------------------------------------------------------

    safety_mean = mean(
        safety_scores
    )

    robustness_mean = mean(
        robustness_scores
    )

    goal_mean = mean(
        goal_scores
    )

    # --------------------------------------------------------
    # Group-level composite
    #
    # Recalculate the composite from the aggregated dimension
    # scores so the group result represents the mean trust
    # dimensions.
    # --------------------------------------------------------

    group_safety = {
        "safety_score": round(
            safety_mean,
            4
        )
    }

    group_robustness = {
        "robustness_score": round(
            robustness_mean,
            4
        )
    }

    group_reliability = {
        "reliability_score": round(
            repetition_aware_reliability,
            4
        )
    }

    group_goal = {
        "goal_adherence_score": round(
            goal_mean,
            4
        )
    }

    group_composite = CompositeScorer(
        domain
    ).score(
        group_safety,
        group_robustness,
        group_reliability,
        group_goal,
    )

    # --------------------------------------------------------
    # Validator information
    # --------------------------------------------------------

    validator_information = (
        collect_validator_information(
            individual_results
        )
    )

    # --------------------------------------------------------
    # Final group result
    # --------------------------------------------------------

    return {
        "level": "group",

        "task_id": task_id,

        "framework": framework,

        "domain": domain,

        "group_key": str(
            group_key
        ),

        "expected_repetitions": (
            EXPECTED_REPETITIONS
        ),

        "actual_repetitions": len(
            individual_results
        ),

        "individual_scores": individual_scores,

        "aggregate": {
            "safety_mean": round(
                safety_mean,
                4
            ),

            "robustness_mean": round(
                robustness_mean,
                4
            ),

            "reliability_mean": round(
                repetition_aware_reliability,
                4
            ),

            "goal_adherence_mean": round(
                goal_mean,
                4
            ),

            "composite_mean": round(
                get_composite_score({
                    "scores": {
                        "composite": group_composite
                    }
                }),
                4
            ),

            "success_rate": round(
                success_rate,
                4
            ),
        },

        "reliability_repetition_details": (
            reliability_group_result.get(
                "details",
                {}
            )
            if isinstance(
                reliability_group_result,
                dict
            )
            else {}
        ),

        "validator_information": (
            validator_information
        ),
    }


# ============================================================
# Group filename
# ============================================================

def make_group_filename(
    task_id: str,
    framework: str,
) -> str:
    """Create a safe group result filename."""

    safe_framework = str(
        framework or "unknown_framework"
    ).replace(
        "/",
        "_"
    ).replace(
        "\\",
        "_"
    ).replace(
        " ",
        "_"
    )

    return (
        f"{task_id}_{safe_framework}.json"
    )


# ============================================================
# Domain/framework summary
# ============================================================

def build_summary(
    group_results: list[dict],
) -> dict:
    """
    Build domain/framework summary.

    IMPORTANT:

    Each task/framework group contributes exactly ONCE.

    Therefore:

        30 tasks × 3 frameworks = 90 observations

    rather than:

        720 individual repetitions.
    """

    grouped = {}

    for group in group_results:

        domain = group.get(
            "domain",
            "default"
        )

        framework = group.get(
            "framework",
            "unknown"
        )

        grouped.setdefault(
            domain,
            {}
        )

        grouped[domain].setdefault(
            framework,
            []
        )

        grouped[domain][framework].append(
            group
        )

    summary = {}

    for domain, frameworks in grouped.items():

        summary[domain] = {}

        for framework, groups in frameworks.items():

            if not groups:
                continue

            safety_means = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "safety_mean",
                    0.0
                )
                for group in groups
            ]

            robustness_means = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "robustness_mean",
                    0.0
                )
                for group in groups
            ]

            reliability_means = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "reliability_mean",
                    0.0
                )
                for group in groups
            ]

            goal_means = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "goal_adherence_mean",
                    0.0
                )
                for group in groups
            ]

            composite_means = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "composite_mean",
                    0.0
                )
                for group in groups
            ]

            success_rates = [
                group.get(
                    "aggregate",
                    {}
                ).get(
                    "success_rate",
                    0.0
                )
                for group in groups
            ]

            summary[domain][framework] = {
                "task_framework_groups": len(
                    groups
                ),

                "mean_safety": round(
                    mean(safety_means),
                    4
                ),

                "mean_robustness": round(
                    mean(robustness_means),
                    4
                ),

                "mean_reliability": round(
                    mean(reliability_means),
                    4
                ),

                "mean_goal_adherence": round(
                    mean(goal_means),
                    4
                ),

                "mean_composite_trust_score": round(
                    mean(composite_means),
                    4
                ),

                "mean_success_rate": round(
                    mean(success_rates),
                    4
                ),
            }

    return summary


# ============================================================
# Main evaluation
# ============================================================

def run_evaluation():

    print("=" * 70)
    print("TrustEval Evaluation")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. Discover traces
    # --------------------------------------------------------

    trace_files = discover_trace_files(
        TRACES_DIR
    )

    if not trace_files:
        print(
            f"No trace files found in {TRACES_DIR}"
        )
        return

    print(
        f"Found {len(trace_files)} trace files."
    )

    # --------------------------------------------------------
    # 2. Load traces
    # --------------------------------------------------------

    loaded_traces = []

    for trace_path in trace_files:

        try:
            trace = load_trace(
                trace_path
            )

        except Exception as exc:

            print(
                f"[ERROR] Could not load "
                f"{trace_path.name}: {exc}"
            )

            continue

        loaded_traces.append({
            "trace": trace,
            "path": trace_path,
        })

    print(
        f"Successfully loaded "
        f"{len(loaded_traces)} traces."
    )

    print()

    # --------------------------------------------------------
    # 3. Group traces
    # --------------------------------------------------------

    groups = group_traces(
        loaded_traces
    )

    print(
        f"Found {len(groups)} "
        f"task/framework groups."
    )

    # --------------------------------------------------------
    # 4. Validate repetition structure
    # --------------------------------------------------------

    group_problems = validate_repetition_groups(
        groups,
        expected_repetitions=EXPECTED_REPETITIONS,
    )

    if group_problems:

        print()
        print(
            "Repetition/trace integrity warnings:"
        )

        for problem in group_problems:
            print(
                f"  {problem}"
            )

        print()

        # Continue evaluating valid traces.

    else:

        print(
            "All task/framework groups contain "
            f"exactly {EXPECTED_REPETITIONS} "
            "repetitions with consistent identities."
        )

        print()

    # --------------------------------------------------------
    # 5. INDIVIDUAL EVALUATION
    # --------------------------------------------------------

    print(
        "Evaluating individual traces..."
    )

    all_individual_results = []

    # Maps path -> individual result.
    # Used to construct group results later.
    individual_by_path = {}

    for item in loaded_traces:

        trace = item["trace"]
        trace_path = item["path"]

        try:

            result = evaluate_trace(
                trace,
                trace_path,
            )

        except Exception as exc:

            print(
                f"[ERROR] Could not evaluate "
                f"{trace_path.name}: {exc}"
            )

            continue

        all_individual_results.append(
            result
        )

        individual_by_path[
            str(trace_path)
        ] = result

        save_json(
            result,
            INDIVIDUAL_DIR / trace_path.name,
        )

    print(
        f"Individual traces evaluated: "
        f"{len(all_individual_results)}"
    )

    # --------------------------------------------------------
    # 6. GROUP EVALUATION
    # --------------------------------------------------------

    print()
    print(
        "Aggregating repetitions..."
    )

    all_group_results = []

    for group_key, group_items in groups.items():

        group_individual_results = []

        for item in group_items:

            path = item["path"]

            result = individual_by_path.get(
                str(path)
            )

            if result is not None:
                group_individual_results.append(
                    result
                )

        if not group_individual_results:
            continue

        try:

            group_result = evaluate_group(
                group_key,
                group_items,
                group_individual_results,
            )

        except Exception as exc:

            print(
                f"[ERROR] Could not aggregate "
                f"group {group_key}: {exc}"
            )

            continue

        all_group_results.append(
            group_result
        )

        filename = make_group_filename(
            group_result.get(
                "task_id"
            ),
            group_result.get(
                "framework"
            ),
        )

        save_json(
            group_result,
            GROUP_DIR / filename,
        )

        print(
            f"  {group_result['task_id']:<20}"
            f" | {group_result['framework']:<10}"
            f" | reps: "
            f"{group_result['actual_repetitions']:<2}"
            f" | mean CTS: "
            f"{group_result['aggregate']['composite_mean']:.4f}"
        )

    print()
    print(
        f"Task/framework groups evaluated: "
        f"{len(all_group_results)}"
    )

    # --------------------------------------------------------
    # 7. Save all individual results
    # --------------------------------------------------------

    individual_output = (
        RESULTS_DIR
        / "trust_scores.json"
    )

    save_json(
        all_individual_results,
        individual_output,
    )

    # --------------------------------------------------------
    # 8. Save all group results
    # --------------------------------------------------------

    group_output = (
        RESULTS_DIR
        / "group_scores.json"
    )

    save_json(
        all_group_results,
        group_output,
    )

    # --------------------------------------------------------
    # 9. Build summary from GROUPS
    # --------------------------------------------------------

    summary = build_summary(
        all_group_results
    )

    summary_path = (
        RESULTS_DIR
        / "summary.json"
    )

    save_json(
        summary,
        summary_path,
    )

    # --------------------------------------------------------
    # 10. Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Individual traces evaluated : "
        f"{len(all_individual_results)}"
    )

    print(
        f"Task/framework groups       : "
        f"{len(all_group_results)}"
    )

    print(
        f"Expected repetitions/group  : "
        f"{EXPECTED_REPETITIONS}"
    )

    print()

    print(
        f"Individual results saved    : "
        f"{individual_output}"
    )

    print(
        f"Group results saved         : "
        f"{group_output}"
    )

    print(
        f"Summary saved               : "
        f"{summary_path}"
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "── Mean Group Composite Trust Score "
        "by Domain & Framework ──"
    )

    print()

    print(
        f"{'Domain':<16}"
        f"{'LangChain':>14}"
        f"{'CrewAI':>14}"
        f"{'AutoGen':>14}"
    )

    print(
        "-" * 58
    )

    for domain in sorted(
        summary.keys()
    ):

        framework_data = summary[
            domain
        ]

        langchain = (
            framework_data
            .get(
                "langchain",
                {}
            )
            .get(
                "mean_composite_trust_score",
                "-"
            )
        )

        crewai = (
            framework_data
            .get(
                "crewai",
                {}
            )
            .get(
                "mean_composite_trust_score",
                "-"
            )
        )

        autogen = (
            framework_data
            .get(
                "autogen",
                {}
            )
            .get(
                "mean_composite_trust_score",
                "-"
            )
        )

        print(
            f"{domain:<16}"
            f"{str(langchain):>14}"
            f"{str(crewai):>14}"
            f"{str(autogen):>14}"
        )

    print()


if __name__ == "__main__":
    run_evaluation()