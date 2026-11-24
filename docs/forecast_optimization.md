# Does additional optimization close the long-history forecasting gap?

**Completed November 23, 2026.** All six continuations reached 30k, all apps are stopped, and confirmation remains unopened. See the [results and plots](../results/2026-11-23-forecast-optimization/README.md). Approved November 23, 2026. Six continuations of the three continuous 64/2,048 GDN models from the [training-history sweep](../results/2026-11-23-forecast-horizon/README.md). No new architecture, training-history length, reset control, or confirmation evaluation is included.

| Branch | Seeds | Resume | Endpoint | Learning rate |
|---|---|---:|---:|---|
| Constant | 182, 183, 184 | 10,000 | 30,000 updates | 0.0003 throughout |
| Cosine | 182, 183, 184 | 10,000 | 30,000 updates | 0.0003 to 0.00003 over the next 20,000 updates |

Both branches preserve the parent's model weights and complete Adam state. They see identical subsequent observations and targets seed by seed, using absolute update indices to continue the data stream. The cosine schedule includes both endpoints: update 10,001 uses 0.0003 and update 30,000 uses 0.00003. Resuming an interrupted job does not restart decay.

All other training settings remain fixed: two GDN layers, width 64, four heads, FP32 without TF32 or autocast; alternate 64 and 2,048 observed transitions; 2,048 prediction positions per update; Adam betas 0.9/0.999, epsilon 1e-8, no weight decay, and gradient clipping at 1. Each continuation adds 40,960,000 training positions, for 61,440,000 including the parent run. Each independent series has a hidden AR(1) coefficient uniform on [−0.95,0.95], innovation standard deviation 0.5, and initial value zero.

## Evaluation and interpretation

Save and evaluate 15k, 20k, and 30k checkpoints; **30k is the fixed primary endpoint**. Retain the shared 10k starting checkpoint for direct comparison. Use the existing 512 validation series (seed 920000000) at history lengths 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192, and 16,384. The separate confirmation set stays unopened. Weights are frozen and recurrent state is carried continuously during evaluation.

Report reducible forecast error, squared disagreement with the exact Bayesian predictor, and total expected MSE (including innovation variance 0.25). Show every training seed, seed means and variability, error versus training updates, and full error-versus-history curves. Compare 2,048 (inside the training range) with 4,096–16,384 (beyond it). Do not choose the best checkpoint after seeing validation results.

- Lower errors with a flat history curve mean optimization lowers the floor without restoring continued improvement from additional observations.
- An increasingly declining curve beyond 2,048 supports improved long-history adaptation over the tested range.
- A consistent cosine advantage supports an effect of the learning-rate schedule.
- A plateau in both branches does not establish an architectural limitation or prove convergence under other optimization settings.

The earlier 10k matched reset comparison remains a separate fixed-budget result. It is not extended in this batch.

## Verification and resources

CPU checks verify preserved weights and Adam moments for all six real checkpoints, exact paired training batches, learning-rate endpoints and monotonicity, exact uninterrupted-versus-resumed training, and exact resumption midway through decay. Changes to other hyperparameters are rejected. The model and sampler source files remain unchanged.

Personal resources only: GitHub `Alex2262/project182`, Modal workspace `alexander-tian`, volume `project182-artifacts`, and W&B `alexander-tian-ucb/project182`. Each of the six concurrent jobs uses one A100 40GB, 2 CPU, 8 GiB RAM, no retries, and two-second scale-down. Expected running compute is $9–12, excluding overhead. A 4,800-second hard timeout and 4,500-second internal budget provide runtime headroom; they are not target runtimes. No additional GPU preflight job is launched.

The batch manifest records resolved configurations, parent checkpoint fingerprints, source hashes, personal job destinations, and completion status. Nothing is committed or pushed automatically.

Two runs were resumed twice from verified saved states after local connection errors. The launcher now submits a single call and retries only result retrieval, preserving the original training configuration and endpoint. Full recovery provenance is in the results manifest.
