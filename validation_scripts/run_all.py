"""
TrustEval Member 3
Complete Statistical Validation Pipeline

IMPORTANT:
This version does NOT use human ratings.

It automatically finds:

    TrustEval-main/results/trust_scores.json

regardless of the folder from which the command is executed.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

# run_all.py is located at:
#
# TrustEval-main/
#     validation_scripts/
#         run_all.py
#
# Therefore parents[1] = TrustEval-main

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

VALIDATION_DIR = (
    PROJECT_ROOT / "validation"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs"
)

FIGURES_DIR = (
    OUTPUT_DIR / "figures"
)


# ============================================================
# PYTHON IMPORT PATH
# ============================================================

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

sys.path.insert(
    0,
    str(VALIDATION_DIR)
)


# ============================================================
# IMPORT VALIDATION MODULES
# ============================================================

from data_loader import load_trust_scores

from confidence_intervals import (
    per_agent_confidence_intervals
)

from ranking_stability import (
    get_ranking,
    ranking_stability
)

from sensitivity_analysis import (
    sensitivity_analysis
)

from repeatability import (
    repeatability_analysis,
    repeatability_summary
)

from ablation import (
    ablation_study
)

from baseline_comparison import (
    compare_baseline_vs_trusteval
)

from cross_domain_validation import (
    cross_domain_validation
)

from robustness_testing import (
    leave_one_task_out,
    robustness_summary
)

from calibration import (
    calibration_analysis
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRUST SCORE FILE
# ============================================================

TRUST_SCORES_PATH = (
    RESULTS_DIR
    / "trust_scores.json"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_dataframe(
    dataframe,
    filename
):

    output_path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# CHECK PROJECT STRUCTURE
# ============================================================

def check_project_structure():

    print("\nChecking project structure...")

    print(
        f"Project root:\n"
        f"  {PROJECT_ROOT}"
    )

    print(
        f"Results directory:\n"
        f"  {RESULTS_DIR}"
    )

    print(
        f"Trust scores file:\n"
        f"  {TRUST_SCORES_PATH}"
    )

    print(
        f"Output directory:\n"
        f"  {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Check results folder
    # --------------------------------------------------------

    if not RESULTS_DIR.exists():

        raise FileNotFoundError(
            "\nERROR: The results folder does not exist.\n\n"
            f"Expected:\n"
            f"  {RESULTS_DIR}\n\n"
            "Make sure your project has a 'results' folder."
        )

    # --------------------------------------------------------
    # Check trust_scores.json
    # --------------------------------------------------------

    if not TRUST_SCORES_PATH.exists():

        print("\nFiles currently inside results:")

        for file in RESULTS_DIR.iterdir():

            print(
                f"  - {file.name}"
            )

        raise FileNotFoundError(
            "\nERROR: Could not find trust_scores.json.\n\n"
            "The program expected:\n"
            f"  {TRUST_SCORES_PATH}\n\n"
            "If the file has a different name, "
            "rename it to trust_scores.json.\n"
            "If it is somewhere else, move/copy it "
            "into the results folder."
        )

    print(
        "\ntrust_scores.json FOUND."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print(
        "TrustEval — Member 3 Statistical Validation"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # CHECK PATHS
    # --------------------------------------------------------

    check_project_structure()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading Member 2 trust scores...")

    trust = load_trust_scores(
        TRUST_SCORES_PATH
    )

    print(
        f"\nLoaded {len(trust)} execution runs."
    )

    print(
        f"Tasks: "
        f"{trust['task_id'].nunique()}"
    )

    print(
        f"Frameworks: "
        f"{trust['framework'].nunique()}"
    )

    print(
        f"Domains: "
        f"{trust['domain'].nunique()}"
    )

    print(
        f"Repetitions: "
        f"{trust['repetition'].nunique()}"
    )

    print(
        "\nFrameworks:"
    )

    print(
        trust["framework"]
        .dropna()
        .unique()
    )

    # ========================================================
    # 1. CONFIDENCE INTERVALS
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[1/9] Task-level bootstrap confidence intervals"
    )
    print("-" * 75)

    ci = (
        per_agent_confidence_intervals(
            trust
        )
    )

    print(
        ci.to_string(
            index=False
        )
    )

    save_dataframe(
        ci,
        "confidence_intervals.csv"
    )

    # ========================================================
    # 2. RANKING STABILITY
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[2/9] Framework ranking stability"
    )
    print("-" * 75)

    ranking = (
        get_ranking(
            trust
        )
    )

    stability = (
        ranking_stability(
            trust
        )
    )

    print(
        "\nFramework ranking:"
    )

    print(
        ranking
        .sort_values()
        .to_string()
    )

    print(
        "\nRanking stability:"
    )

    print(
        stability
    )

    save_dataframe(
        pd.DataFrame(
            [stability]
        ),
        "ranking_stability.csv"
    )

    # ========================================================
    # 3. SENSITIVITY ANALYSIS
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[3/9] Domain-aware weight sensitivity"
    )
    print("-" * 75)

    sensitivity_results = []

    for perturbation in (
        0.10,
        0.20,
        0.30
    ):

        print(
            f"\nTesting ±{perturbation * 100:.0f}% "
            "weight perturbation..."
        )

        result = sensitivity_analysis(
            trust,
            perturbation=perturbation,
            n_trials=500,
            seed=42
        )

        sensitivity_results.append(
            result
        )

    if sensitivity_results:

        sensitivity = pd.concat(
            sensitivity_results,
            ignore_index=True
        )

    else:

        sensitivity = pd.DataFrame()

    print(
        sensitivity.to_string(
            index=False
        )
    )

    save_dataframe(
        sensitivity,
        "sensitivity_analysis.csv"
    )

    # ========================================================
    # 4. ABLATION
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[4/9] Trust-dimension ablation"
    )
    print("-" * 75)

    ablation = (
        ablation_study(
            trust
        )
    )

    print(
        ablation.to_string(
            index=False
        )
    )

    save_dataframe(
        ablation,
        "ablation_study.csv"
    )

    # ========================================================
    # 5. REPEATABILITY
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[5/9] Repeatability across repeated executions"
    )
    print("-" * 75)

    repeatability = (
        repeatability_analysis(
            trust
        )
    )

    print(
        repeatability
        .head(20)
        .to_string(
            index=False
        )
    )

    repeatability_result = (
        repeatability_summary(
            repeatability
        )
    )

    print(
        "\nRepeatability summary:"
    )

    print(
        repeatability_result
    )

    save_dataframe(
        repeatability,
        "repeatability.csv"
    )

    # ========================================================
    # 6. CROSS DOMAIN
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[6/9] Cross-domain ranking consistency"
    )
    print("-" * 75)

    cross_domain = (
        cross_domain_validation(
            trust
        )
    )

    print(
        cross_domain.to_string(
            index=False
        )
    )

    save_dataframe(
        cross_domain,
        "cross_domain_validation.csv"
    )

    # ========================================================
    # 7. LEAVE-ONE-TASK-OUT
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[7/9] Leave-one-task-out robustness"
    )
    print("-" * 75)

    loto = (
        leave_one_task_out(
            trust
        )
    )

    print(
        loto
        .head(20)
        .to_string(
            index=False
        )
    )

    loto_summary = (
        robustness_summary(
            loto
        )
    )

    print(
        "\nRobustness summary:"
    )

    print(
        loto_summary
    )

    save_dataframe(
        loto,
        "robustness_leave_one_task_out.csv"
    )

    # ========================================================
    # 8. BASELINE COMPARISON
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[8/9] Baseline comparison"
    )
    print("-" * 75)

    baseline = (
        compare_baseline_vs_trusteval(
            trust
        )
    )

    print(
        baseline.to_string(
            index=False
        )
    )

    save_dataframe(
        baseline,
        "baseline_comparison.csv"
    )

    # ========================================================
    # 9. CALIBRATION
    # ========================================================

    print("\n")
    print("-" * 75)
    print(
        "[9/9] Calibration / outcome consistency"
    )
    print("-" * 75)

    calibration = (
        calibration_analysis(
            trust
        )
    )

    print(
        calibration
    )

    save_dataframe(
        pd.DataFrame(
            [calibration]
        ),
        "calibration.csv"
    )

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    summary = pd.DataFrame(
        [{
            "n_runs":
                len(trust),

            "n_tasks":
                trust["task_id"]
                .nunique(),

            "n_frameworks":
                trust["framework"]
                .nunique(),

            "n_domains":
                trust["domain"]
                .nunique(),

            "n_repetitions_max":
                trust
                .groupby(
                    [
                        "task_id",
                        "framework"
                    ]
                )["repetition"]
                .count()
                .max(),

            "mean_composite":
                trust[
                    "composite_score"
                ].mean(),

            "std_composite":
                trust[
                    "composite_score"
                ].std(),
        }]
    )

    save_dataframe(
        summary,
        "validation_summary.csv"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 75)
    print(
        "VALIDATION COMPLETE"
    )
    print("=" * 75)

    print(
        "\nAll validation outputs were saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        "\nHuman ratings were NOT used."
    )

    print(
        "\nThe Member 3 pipeline validated "
        f"{len(trust)} execution runs."
    )

    print("=" * 75)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()