import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, "../validation")
from data_loader import load_trust_scores, DIMENSIONS
from agreement import pairwise_agreement_report
from correlation import correlate_with_humans
from weight_optimization import optimize_weights, MEMBER2_WEIGHTS
from confidence_intervals import per_agent_confidence_intervals
from ranking_stability import ranking_stability, get_ranking
from sensitivity_analysis import sensitivity_analysis
from ablation import ablation_study
from overall_trust_validation import validate_overall_trust
from baseline_comparison import compare_baseline_vs_trusteval
from cross_domain_validation import cross_domain_validation
from repeatability import repeatability_analysis, repeatability_summary
from calibration import calibration_analysis
from robustness_testing import leave_one_rater_out, leave_one_trace_out
from visualizations import plot_correlation_scatter, plot_framework_comparison, plot_dimension_correlation, plot_ablation

os.makedirs("../outputs", exist_ok=True)
os.makedirs("../outputs/figures", exist_ok=True)

print("Loading real trust scores from results/trust_scores.json...")
trust = load_trust_scores("../results/trust_scores.json")
print(f"Loaded {len(trust)} runs across {trust['agent_id'].nunique()} frameworks "
      f"and {trust['domain'].nunique()} domains")

print("Loading human ratings from data/human_ratings.csv...")
human = pd.read_csv("../data/human_ratings.csv")

print("\n1. Inter-rater agreement (ICC)")
agreement = pairwise_agreement_report(human)
print(agreement.to_string(index=False))
agreement.to_csv("../outputs/agreement_report.csv", index=False)

print("\n2. Correlation: TrustEval vs human ratings")
correlation = correlate_with_humans(trust, human)
print(correlation.to_string(index=False))
correlation.to_csv("../outputs/correlation_report.csv", index=False)

print("\n3. Weight optimization")
weights, mse = optimize_weights(trust, human)
print(f"Optimized weights: {weights}  (MSE={mse:.4f})")
print(f"Member 2's fixed weights: {MEMBER2_WEIGHTS}")
pd.DataFrame([weights]).to_csv("../outputs/optimized_weights.csv", index=False)

print("\n4. Bootstrap confidence intervals per framework")
ci_report = per_agent_confidence_intervals(trust)
print(ci_report.to_string(index=False))
ci_report.to_csv("../outputs/confidence_intervals.csv", index=False)

print("\n5. Ranking stability under resampling")
ranking = get_ranking(trust)
stability = ranking_stability(trust)
print(f"Ranking: {dict(ranking.sort_values())}")
print(f"Stability: {stability}")
pd.DataFrame([stability]).to_csv("../outputs/ranking_stability.csv", index=False)

print("\n6. Sensitivity analysis (weight perturbation)")
sensitivity_rows = []
for pert in [0.1, 0.2, 0.3]:
    sensitivity_rows.append(sensitivity_analysis(trust, perturbation=pert))
sensitivity = pd.concat(sensitivity_rows, ignore_index=True)
print(sensitivity.to_string(index=False))
sensitivity.to_csv("../outputs/sensitivity_analysis.csv", index=False)

print("\n7. Ablation study")
ablation = ablation_study(trust, human)
print(ablation.to_string(index=False))
ablation.to_csv("../outputs/ablation_study.csv", index=False)

print("\n8. Overall trustworthiness validation")
overall = validate_overall_trust(trust, human)
print(overall.to_string(index=False))
overall.to_csv("../outputs/overall_trust_validation.csv", index=False)

print("\n9. Baseline comparison (TrustEval vs naive success-only)")
baseline = compare_baseline_vs_trusteval(trust, human)
print(baseline.to_string(index=False))
baseline.to_csv("../outputs/baseline_comparison.csv", index=False)

print("\n10. Cross-domain validation")
cross_domain = cross_domain_validation(trust, human)
print(cross_domain.to_string(index=False))
cross_domain.to_csv("../outputs/cross_domain_validation.csv", index=False)

print("\n11. Repeatability experiments")
repeatability = repeatability_analysis(trust)
print(repeatability.to_string(index=False))
repeatability.to_csv("../outputs/repeatability.csv", index=False)
print("Summary:", repeatability_summary(repeatability))

print("\n12. Statistical calibration")
calibration = calibration_analysis(trust, human)
for k, v in calibration.items():
    print(f"  {k}: {v}")
pd.DataFrame([calibration]).to_csv("../outputs/calibration.csv", index=False)

print("\n13. Robustness testing (leave-one-rater-out)")
loro = leave_one_rater_out(trust, human)
print(loro.to_string(index=False))
loro.to_csv("../outputs/robustness_leave_one_rater_out.csv", index=False)

print("\n14. Robustness testing (leave-one-trace-out)")
loto = leave_one_trace_out(trust, human)
print(loto)
pd.DataFrame([loto]).to_csv("../outputs/robustness_leave_one_trace_out.csv", index=False)

print("\n15. Generating visualizations")
p1 = plot_correlation_scatter(trust, human)
p2 = plot_framework_comparison(ci_report)
p3 = plot_dimension_correlation(correlation)
p4 = plot_ablation(ablation)
print("Saved figures:")
for p in [p1, p2, p3, p4]:
    print(" -", p)

print("\nAll done. Results saved in outputs/, figures saved in outputs/figures/")