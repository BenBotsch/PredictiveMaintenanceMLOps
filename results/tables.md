Test split: 1,500 rows, 51 failures, threshold 0.781, 1,000 stratified bootstrap resamples.

| Metric | Test | 95 % interval |
| --- | ---: | ---: |
| PR-AUC | 0.882 | 0.803–0.952 |
| ROC-AUC | 0.973 | 0.943–0.994 |
| Precision | 0.949 | 0.868–1.000 |
| Recall | 0.725 | 0.608–0.843 |
| F1 | 0.822 | 0.736–0.896 |

| Failure mode | Detected | Recall |
| --- | ---: | ---: |
| Tool wear (TWF) | 1 of 4 | 25% |
| Heat dissipation (HDF) | 13 of 13 | 100% |
| Power (PWF) | 11 of 18 | 61% |
| Overstrain (OSF) | 16 of 18 | 89% |
| Random (RNF) | 0 of 0 | n/a |
| No mode recorded | 0 of 2 | 0% |

| Input | PR-AUC drop when shuffled |
| --- | ---: |
| `torque_nm` | 0.768 ± 0.014 |
| `rotational_speed_rpm` | 0.495 ± 0.037 |
| `air_temp_k` | 0.479 ± 0.044 |
| `process_temp_k` | 0.347 ± 0.030 |
| `tool_wear_min` | 0.338 ± 0.050 |
| `type` | 0.040 ± 0.017 |

| Batch | Median | p95 |
| ---: | ---: | ---: |
| 1 | 12.2 ms | 14.8 ms |
| 1,000 | 22.6 ms | 32.2 ms |
