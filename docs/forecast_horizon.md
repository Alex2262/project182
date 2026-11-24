# Training-history length versus streaming deterioration

**Completed November 23, 2026.** All nine new training runs and six extended evaluations finished. All apps are stopped, all checks passed, and confirmation remains unopened. See the [results and plots](../results/2026-11-23-forecast-horizon/README.md). This protocol tested whether longer training histories restore continued adaptation or move deterioration to a later point. The existing model, optimizer, data family, and state size are fixed.

| Condition | Training | Independent seeds | Work |
|---|---|---|---|
| L64 | Always 64 observations | 182, 183, 184 | Reuse saved 2k/5k/10k checkpoints |
| M256 | Alternate 64 and 256 | 182, 183, 184 | Reuse saved 2k/5k/10k checkpoints |
| M1024 | Alternate 64 and 1,024 | 182, 183, 184 | Three new 10k-update runs |
| M2048 | Alternate 64 and 2,048 | 182, 183, 184 | Three new 10k-update runs |
| M2048 reset256 | Same generated batches and targets as M2048; reset every 256 during long training sequences | 182, 183, 184 | Three new 10k-update runs |

The nine new runs start from scratch. There are six separate evaluation-only jobs for existing checkpoints and one GPU preflight. No Transformer training or confirmation evaluation is included.

## Fixed settings

Gated DeltaNet: two layers, width 64, four heads, 110,641 parameters, 12,800 bytes of persistent FP32 state per series plus the most recent scalar. AR(1) coefficient is uniform on [−0.95,0.95], innovation standard deviation 0.5, and initial value zero. Both lag values are observed input features; the true coefficient is never a model input.

Every new run uses 10,000 updates and 2,048 prediction positions per update: 20,480,000 positions total. Batches are 32×64, 2×1,024, or 1×2,048 as applicable. Adam uses LR 0.0003, betas 0.9/0.999, epsilon 1e-8, no weight decay, gradient clipping at 1, and a constant schedule. FP32, no autocast or TF32. Data seeds are 410000000/410100000/410200000 plus update index. Retain model and optimizer checkpoints at 2k, 5k, and 10k; 10k is the primary endpoint.

The reset control packs each long sequence into independent 256-observation blocks, resetting both recurrent and convolution state. All original targets receive the same loss weight, and each original batch still produces one optimizer update. Inputs, underlying series, targets, and initialization match M2048 seed by seed. Evaluation carries state uninterrupted for every condition. Resetting changes both forward history access and backpropagation span; this control does not separate those mechanisms.

Different maximum training lengths imply different numbers of independent series under the fixed token budget. The M2048 reset control removes that data-composition difference for the paired M2048 comparison.

## Evaluation and decision rule

Use the same 512 validation series (seed 920000000), extended to 16,384 observations. Evaluate lengths 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, and 16,384. The longer generation preserves shorter prefixes on the same platform; cross-platform float64 rounding is checked to a 1e-13 tolerance. The confirmation set (seed 940000000) stays unopened.

Primary error is mean squared error relative to the true conditional next-value mean. Also report disagreement with the exact Bayes forecast and total expected predictive MSE including innovation variance 0.25. Plot error against absolute history and against history divided by the **maximum uninterrupted training history**: 64/256/1024/2048 for the continuous conditions, and 256 for the reset control.

For each training seed, define deterioration as the first tested length where error is at least 1.25 times its earlier minimum and remains above that same threshold at the next tested length. A rise only at 16,384 is unconfirmed because there is no later test point. This is a descriptive, predeclared threshold; its location is not a causal or asymptotic limit. Show the full curves and seed variability alongside it.

If deterioration begins at approximately the same multiple of training history across conditions, that supports a shifted failure point. Sustained improvement well beyond training history supports continued adaptation over the tested range. A flat error floor is distinct from both. A continuous-M2048 advantage over its matched reset control supports the value of longer training history access, while leaving forward-memory and gradient-span effects entangled.

## Checks and resources

CPU checks cover packed-reset versus separate-block outputs and gradients, absence of influence across a reset boundary, unchanged ordinary forward behavior, validation-prefix preservation, the deterioration rule, and a tiny training/checkpoint/16k-evaluation run.

GPU preflight uses ten updates each at 64, 1024, 2048, and 2048 with reset256, plus two-series 16k inference. It checks finite losses, causality, streaming agreement, reset agreement, and timing before the authorized batch proceeds.

Use personal GitHub `Alex2262/project182`, Modal workspace `alexander-tian`, volume `project182-artifacts`, and W&B `alexander-tian-ucb/project182`. Each job uses one A100 40GB, 2 CPU, 8 GiB RAM, no retries, and two-second scale-down. At most eight jobs run concurrently. Training hard timeout is 3,600 seconds with a 3,300-second internal budget; evaluation-only timeout is 900 seconds. The larger timeout is headroom, not a target runtime.

The previous [three-seed results](../results/2026-11-23-forecast-ar1-10k/README.md) and [original experiment plan](forecast_pilot.md) supply the starting point. All new resolved configurations, source hashes, checkpoint fingerprints, and job destinations are recorded in the batch manifest.
