# Does carrying the old state help the same forecast?

Paired frozen-checkpoint evaluation, November 24, 2026. This is an exploratory follow-up on the existing validation cohort, not an independent confirmation experiment.

## Fixed experiment

- Checkpoints: the final 30,000-update cosine continuations for training seeds 182, 183, and 184. Two-layer GDN, width 64, four heads, 110,641 parameters; 12,800 bytes of carried FP32 state plus the latest value.
- Data: the original 512 validation series, seed 920000000. AR(1), coefficient uniform on [-0.95, 0.95], innovation standard deviation 0.5, initial value zero. Use the original saved observations, coefficients, and targets unchanged.
- Origins: 2,048, 4,096, 8,192, and 16,384 observed transitions. Forecast the same next value within each comparison.
- Policies: continuous full history; fresh recurrent and convolution state followed by replay of the last 256 transitions; fresh state followed by replay of the last 2,048 transitions.
- Primary comparison: window-2,048 minus continuous-history reducible error at origin 16,384. Other origins and the 256 window are supporting comparisons.
- No parameter updates, architecture changes, new checkpoint selection, or additional training. Confirmation seed 940000000 remains unopened.

For a window W ending at origin t, the model receives only transition tokens (y[t-W], y[t-W+1]) through (y[t-1], y[t]). Neither y[t+1] nor the hidden coefficient enters the model. This is fresh-window replay at each forecast origin, not periodic resets of an otherwise continuously running model.

## Measurements and interpretation

Primary error is the mean squared discrepancy from the latent next-value mean a*y[t]. Report total expected MSE by adding 0.25, and squared disagreement with the full-history exact Bayesian forecast. Include the full-history Bayes risk and online ridge reference already implemented.

Average losses across the three models, not their predictions. Report every seed separately. Pair series across policies. The 95% pointwise bootstrap intervals use 5,000 resamples of the 512 independent series (bootstrap seed 261124), averaging losses across the three fixed models before resampling. These intervals describe evaluation-series uncertainty conditional on these checkpoints; they are not confidence intervals for a population of training runs. Report training-seed variability separately. Supporting comparisons are exploratory and not corrected for multiple comparisons.

If continuous history wins, the old state provides useful predictive information relative to the tested windows. If window replay wins, it is a useful accuracy intervention for these checkpoints and conditions. Neither result identifies a gate mechanism or establishes a general property of recurrent models. Fresh-window replay also changes initialization and removes the original trajectory's state, so it is a compound deployment intervention.

## Compute and checks

Run locally on CPU, FP32, two CPU threads, batch size 16. Cloud cost is zero. Reuse saved full-history GPU predictions after CPU replay checks on four series per seed at all four origins. Verify that the full-history and 2,048-window predictions agree at origin 2,048 across all 512 series, that whole-window and chunked streaming evaluation agree, that checkpoint weights remain unchanged, and that the model source matches the training snapshot.

The source data were created on a different platform; seed-based float64 regeneration permits absolute rounding differences up to 1e-13 in observations and 1e-15 in coefficients. Inference and scoring use the original saved arrays, not regenerated replacements.

Save checkpoint hashes, the resolved manifest, compact predictions and references, CSV metrics, per-seed contrasts, plots, and a readable report. Plot/table regeneration uses saved compact artifacts without retraining or accessing Modal. No commits or pushes are part of this experiment.
