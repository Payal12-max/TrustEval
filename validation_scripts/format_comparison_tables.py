"""
Formats key outputs into clean, paper-ready markdown tables (comparison_tables.md)
instead of raw CSVs. Run this AFTER run_all.py has populated outputs/.
"""

import pandas as pd
import os

OUT = "../outputs"


def df_to_md(df: pd.DataFrame, title: str) -> str:
    return f"### {title}\n\n" + df.to_markdown(index=False) + "\n\n"


sections = []

ci = pd.read_csv(f"{OUT}/confidence_intervals.csv")
ci.columns = ["Framework", "Mean Composite Score", "95% CI Lower", "95% CI Upper", "N Traces"]
sections.append(df_to_md(ci, "Table 1: Framework Comparison (Composite Trust Score, 95% Bootstrap CI)"))

corr = pd.read_csv(f"{OUT}/correlation_report.csv")
corr.columns = ["Dimension", "Pearson r", "Pearson p", "Spearman rho", "Spearman p", "N"]
sections.append(df_to_md(corr, "Table 2: Per-Dimension Correlation with Human Judgment"))

opt = pd.read_csv(f"{OUT}/optimized_weights.csv")
fixed = pd.DataFrame([{"safety": 0.15, "robustness": 0.20, "reliability": 0.35, "goal_adherence": 0.30}])
weight_compare = pd.DataFrame({
    "Dimension": ["Safety", "Robustness", "Reliability", "Goal Adherence"],
    "Fixed Weight (Member 2)": [fixed.iloc[0]["safety"], fixed.iloc[0]["robustness"],
                                  fixed.iloc[0]["reliability"], fixed.iloc[0]["goal_adherence"]],
    "Optimized Weight (Member 3)": [opt.iloc[0]["safety"], opt.iloc[0]["robustness"],
                                      opt.iloc[0]["reliability"], opt.iloc[0]["goal_adherence"]],
})
sections.append(df_to_md(weight_compare, "Table 3: Fixed vs. Empirically Optimized Dimension Weights"))

abl = pd.read_csv(f"{OUT}/ablation_study.csv")
abl.columns = ["Removed Dimension", "Pearson r (remaining)", "Drop from Full Model"]
sections.append(df_to_md(abl, "Table 4: Ablation Study — Dimension Importance"))

base = pd.read_csv(f"{OUT}/baseline_comparison.csv")
sections.append(df_to_md(base, "Table 5: TrustEval vs. Naive Success-Only Baseline"))

cross = pd.read_csv(f"{OUT}/cross_domain_validation.csv")
cross.columns = ["Domain", "Pearson r", "N", "Note", "Pearson p"]
sections.append(df_to_md(cross, "Table 6: Cross-Domain Correlation with Human Judgment"))

loro = pd.read_csv(f"{OUT}/robustness_leave_one_rater_out.csv")
loro.columns = ["Excluded Rater", "Pearson r"]
sections.append(df_to_md(loro, "Table 7: Robustness — Leave-One-Rater-Out"))

output = "# Comparison Tables (Paper-Ready)\n\n" + "".join(sections)
with open(f"{OUT}/comparison_tables.md", "w", encoding="utf-8") as f:
    f.write(output)

print(f"Wrote {OUT}/comparison_tables.md with {len(sections)} tables")