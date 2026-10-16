# Pre-registered success criteria

Committed before any real run was started (step 6 of the plan). Do not edit after results come in.

| Criterion | Definition |
|---|---|
| Existence | test error has an interior local maximum over width whose height above the minimum to its left exceeds 2 x the seed-to-seed standard deviation (max of the std at peak and valley) |
| Location | the peak lies within one grid step of the interpolation threshold (smallest width with final train error < 1 %) |
| Direction | label noise, augmentation, more samples and longer training move threshold and peak in the direction the paper predicts |
| Quantitative | peak width, peak test error and final test error at the largest width are compared with the authors' data; differences are reported |

Classification per claim (C1-C4):

- **Fully reproduced**: existence, location and direction hold; peak width differs by at most one grid step and errors by at most 3 percentage points (our own thresholds).
- **Qualitatively reproduced**: existence and direction hold, location or numbers differ clearly.
- **Not reproduced**: existence or direction fails.

Computed automatically in `tables/T1_resnet_summary.csv` and `tables/T2_cnn_summary.csv`
(columns `significant`, `peak_minus_thr_steps`).
