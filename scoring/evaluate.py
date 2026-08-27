"""
TrustEval Scoring Engine — Main Entry Point.

Loads all 45 execution traces from traces/prototype_run/
and evaluates each across four trust dimensions.

Saves:
  - results/individual/<trace_name>.json  (one per trace)
  - results/trust_scores.json             (all combined)
  - results/summary.json                  (domain averages)

Usage (from TrustEval-main folder):
    python -m scoring.evaluate
"""

import json
from pathlib import Path

from scoring.safety_scorer import SafetyScorer
from scoring.robustness_scorer import RobustnessScorer
from scoring.reliability_scorer import ReliabilityScorer
from scoring.goal_adherence_scorer import GoalAdherenceScorer
from scoring.composite_scorer import CompositeScorer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR   = PROJECT_ROOT / "traces" / "prototype_run"
RESULTS_DIR  = PROJECT_ROOT / "results"
INDIVIDUAL_DIR = RESULTS_DIR / "individual"


def load_trace(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(result: dict, trace_filename: str):
    """Save individual result with same filename as trace."""
    INDIVIDUAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INDIVIDUAL_DIR / trace_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def get_domain(trace: dict) -> str:
    meta = trace.get("task_metadata", {}) or {}
    domain = meta.get("domain", "")
    if domain:
        return domain.lower().strip()
    task_id = trace.get("task_id", "")
    if "_" in task_id:
        return task_id.split("_")[0].lower()
    return "default"


def evaluate_trace(trace: dict) -> dict:
    domain = get_domain(trace)

    safety_r      = SafetyScorer().score(trace)
    robustness_r  = RobustnessScorer().score(trace)
    reliability_r = ReliabilityScorer().score_single(trace)
    goal_r        = GoalAdherenceScorer().score(trace)
    composite_r   = CompositeScorer(domain).score(
        safety_r, robustness_r, reliability_r, goal_r
    )

    return {
        "task_id":    trace.get("task_id"),
        "framework":  trace.get("framework"),
        "domain":     domain,
        "repetition": trace.get("repetition"),
        "run_id":     trace.get("run_id"),
        "status":     trace.get("status"),
        "execution_time_seconds": trace.get(
            "execution_time_seconds"
        ),
        "scores": {
            "safety":         safety_r,
            "robustness":     robustness_r,
            "reliability":    reliability_r,
            "goal_adherence": goal_r,
            "composite":      composite_r,
        }
    }


def run_evaluation():
    trace_files = sorted(TRACES_DIR.glob("*.json"))

    if not trace_files:
        print(f"No traces found in {TRACES_DIR}")
        return

    print(f"Found {len(trace_files)} traces. Evaluating...\n")

    all_results = []
    domain_summary = {}

    for trace_path in trace_files:
        trace  = load_trace(trace_path)
        result = evaluate_trace(trace)
        all_results.append(result)

        # Save individual result file
        # Same filename as the trace it came from
        save_result(result, trace_path.name)

        cts    = result["scores"]["composite"][
            "composite_trust_score"
        ]
        domain = result["domain"]
        fw     = result["framework"]
        tid    = result["task_id"]

        # Track domain summary
        if domain not in domain_summary:
            domain_summary[domain] = {}
        if fw not in domain_summary[domain]:
            domain_summary[domain][fw] = []
        domain_summary[domain][fw].append(cts)

        print(
            f"  {tid:<20} | {fw:<10} | "
            f"domain: {domain:<12} | CTS: {cts}"
        )

    # Save all results combined
    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "trust_scores.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {}
    for domain, frameworks in domain_summary.items():
        summary[domain] = {}
        for fw, scores in frameworks.items():
            summary[domain][fw] = {
                "mean_cts": round(sum(scores) / len(scores), 4),
                "min_cts":  round(min(scores), 4),
                "max_cts":  round(max(scores), 4),
                "count":    len(scores)
            }

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 65)
    print("EVALUATION COMPLETE")
    print("=" * 65)
    print(f"\nTotal traces evaluated  : {len(all_results)}")
    print(f"Individual results saved: {INDIVIDUAL_DIR}")
    print(f"Combined results saved  : {output_path}")
    print(f"Summary saved           : {summary_path}")

    print(
        "\n── Mean Composite Trust Score by Domain & Framework ──\n"
    )
    print(
        f"{'Domain':<14} {'LangChain':>12} "
        f"{'CrewAI':>12} {'AutoGen':>12}"
    )
    print("-" * 52)

    for domain in sorted(summary.keys()):
        fw_data = summary[domain]
        lc = fw_data.get("langchain", {}).get("mean_cts", "-")
        cr = fw_data.get("crewai",    {}).get("mean_cts", "-")
        ag = fw_data.get("autogen",   {}).get("mean_cts", "-")
        print(
            f"{domain:<14} {str(lc):>12} "
            f"{str(cr):>12} {str(ag):>12}"
        )
    print()


if __name__ == "__main__":
    run_evaluation()