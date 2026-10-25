| Metric | In-distribution (CA test) | Shifted (TX test) | Change [95% CI] | TX-trained reference |
|---|---|---|---|---|
| AUROC (primary, ranking) | 0.907 | 0.891 | -0.016 [-0.020, -0.011] | 0.897 |
| Accuracy at 0.5 (secondary) | 0.826 | 0.802 | -0.024 [-0.030, -0.018] | 0.818 |
| Expected calibration error | 0.009 | 0.074 | +0.065 [0.059, 0.069] | 0.010 |
| Brier score | 0.121 | 0.136 | +0.015 [0.012, 0.018] | 0.125 |
| Mean predicted share > $50k | 0.409 | 0.439 | +0.030 | 0.368 |
| Observed share > $50k | 0.411 | 0.365 | -0.046 | 0.365 |
| False positive rate | 0.143 | 0.215 | +0.073 | 0.135 |
| False negative rate | 0.219 | 0.168 | -0.051 | 0.264 |
