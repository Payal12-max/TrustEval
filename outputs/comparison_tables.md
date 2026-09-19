# Comparison Tables (Paper-Ready)

### Table 1: Framework Comparison (Composite Trust Score, 95% Bootstrap CI)

| Framework   |   Mean Composite Score |   95% CI Lower |   95% CI Upper |   N Traces |
|:------------|-----------------------:|---------------:|---------------:|-----------:|
| langchain   |                  0.865 |          0.832 |          0.892 |         15 |
| autogen     |                  0.859 |          0.82  |          0.889 |         15 |
| crewai      |                  0.824 |          0.775 |          0.869 |         15 |

### Table 2: Per-Dimension Correlation with Human Judgment

| Dimension      |   Pearson r |   Pearson p |   Spearman rho |   Spearman p |   N |
|:---------------|------------:|------------:|---------------:|-------------:|----:|
| safety         |      -0.129 |      0.646  |         -0.133 |       0.6374 |  15 |
| robustness     |     nan     |    nan      |        nan     |     nan      |  15 |
| reliability    |       0.149 |      0.5958 |          0.067 |       0.8136 |  15 |
| goal_adherence |       0.587 |      0.0215 |          0.508 |       0.0533 |  15 |

### Table 3: Fixed vs. Empirically Optimized Dimension Weights

| Dimension      |   Fixed Weight (Member 2) |   Optimized Weight (Member 3) |
|:---------------|--------------------------:|------------------------------:|
| Safety         |                      0.15 |                         0.151 |
| Robustness     |                      0.2  |                         0.575 |
| Reliability    |                      0.35 |                         0     |
| Goal Adherence |                      0.3  |                         0.274 |

### Table 4: Ablation Study — Dimension Importance

| Removed Dimension   |   Pearson r (remaining) |   Drop from Full Model |
|:--------------------|------------------------:|-----------------------:|
| goal_adherence      |                   0.419 |                  0.034 |
| safety              |                   0.42  |                  0.033 |
| none (full model)   |                   0.453 |                  0     |
| robustness          |                   0.453 |                  0     |
| reliability         |                   0.505 |                 -0.052 |

### Table 5: TrustEval vs. Naive Success-Only Baseline

| method                      |   pearson_r |   pearson_p |   spearman_rho |   spearman_p |   n | note                            |
|:----------------------------|------------:|------------:|---------------:|-------------:|----:|:--------------------------------|
| TrustEval composite_score   |       0.183 |      0.5147 |          0.235 |        0.399 |  15 | nan                             |
| Naive success-only baseline |     nan     |    nan      |        nan     |      nan     | nan | insufficient variance in sample |

### Table 6: Cross-Domain Correlation with Human Judgment

| Domain       |   Pearson r |   N | Note                                |   Pearson p |
|:-------------|------------:|----:|:------------------------------------|------------:|
| education    |     nan     |   1 | too few rated traces in this domain |    nan      |
| finance      |     nan     |   2 | too few rated traces in this domain |    nan      |
| general      |      -0.933 |   3 | nan                                 |      0.235  |
| lifestyle    |     nan     |   2 | too few rated traces in this domain |    nan      |
| productivity |       0.803 |   3 | nan                                 |      0.4065 |
| shopping     |     nan     |   2 | too few rated traces in this domain |    nan      |
| technical    |     nan     |   1 | too few rated traces in this domain |    nan      |
| travel       |     nan     |   1 | too few rated traces in this domain |    nan      |

### Table 7: Robustness — Leave-One-Rater-Out

| Excluded Rater    |   Pearson r |
|:------------------|------------:|
| none (all raters) |       0.183 |
| rater_1           |       0.093 |
| rater_2           |       0.258 |

