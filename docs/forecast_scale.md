# AR(1) forecasting: 10k-update, three-seed batch

**Completed November 23, 2026.** All twelve runs reached 10,000 updates, all retained checkpoints passed verification, and all apps are stopped. The confirmation set remains unopened. See the [three-seed results and plots](../results/2026-11-23-forecast-ar1-10k/README.md). This batch extended the first four experiments to assess convergence and repeatability before increasing model capacity or task complexity.

| Architecture | Training histories | Seeds | Endpoint |
|---|---|---|---|
| Gated DeltaNet | 64 | 182, 183, 184 | 10,000 updates each |
| Gated DeltaNet | Alternate 64/256 | 182, 183, 184 | 10,000 updates each |
| Transformer, no positional embeddings | 64 | 182, 183, 184 | 10,000 updates each |
| Transformer, no positional embeddings | Alternate 64/256 | 182, 183, 184 | 10,000 updates each |

The four seed-182 runs continue their existing 2,000-update model and Adam states in new artifact directories. Their original runs and results remain available. Seeds 183 and 184 start from scratch. Total additional work is 112,000 optimizer updates. Each final trajectory uses 20,480,000 training prediction positions.

## Fixed settings

- Model: 2 layers, width 64, 4 heads; same model and data source as the pilot. GDN has 110,641 parameters and 12,800 bytes of carried state per series; Transformer has 99,841 parameters.
- Data: AR(1), coefficient uniform on [−0.95, 0.95], innovation standard deviation 0.5, initial value zero. Both models see observed lag pairs.
- Adam: learning rate 0.0003, betas 0.9/0.999, epsilon 1e-8, constant schedule, no weight decay, gradient clip 1.
- Each update: 2,048 prediction positions; batch 32 at length 64, batch 8 at length 256. FP32; no TF32 or autocast.
- Training data seeds: 410000000, 410100000, 410200000, plus the update index, for training seeds 182, 183, 184 respectively.
- Validation and retained model checkpoints: 2,000, 5,000, and 10,000 updates. All three are reported; the planned primary endpoint is 10,000.
- Validation: same 512 independent series, seed 920000000; histories 16–4,096. Weights frozen. GDN carries state; Transformer receives each full prefix.
- Confirmation seed 940000000 remains unopened.

## What the batch resolves

Does short-trained GDN's long-history degradation reproduce across training seeds? Does it disappear, plateau, or worsen with additional optimization? Does mixed-length training consistently reduce it? The statistical reference remains useful because constant-memory sufficient statistics solve this particular task.

The primary comparison is short versus mixed training at 4,096 observations, alongside full history-length curves. Report variation across training seeds separately from uncertainty across evaluation series. An apparent plateau does not establish an architectural impossibility; this remains a small synthetic control.

## Resources and preservation

Use only GitHub `Alex2262/project182`, Modal workspace `alexander-tian`, volume `project182-artifacts`, and W&B `alexander-tian-ucb/project182`. Each job uses one A100 40GB, 2 CPU, and 8 GiB memory, with a 1,800-second timeout, no retries, and two-second scale-down. At most eight jobs run concurrently. There are exactly twelve authorized jobs; no additional GPU preflight is needed for the tested orchestration-only change.

Estimated additional running compute is $2.43 based on the initial measured timings. Approximately $5 including overhead is a planning estimate, not a billing cap.

Continuation checks require the original checkpoint fingerprint and matching model, data, optimizer, and evaluation settings. Only the endpoint, validation schedule, and runtime bookkeeping may change. Tiny CPU runs verified that continued model and optimizer states exactly match uninterrupted training for both architectures. Actual seed-182 checkpoint imports were also verified locally before launch.

See [the original plan](forecast_pilot.md) and [initial results](../results/2026-11-23-forecast-ar1/README.md). The batch manifest records each resolved configuration, source hashes, parent checkpoint, and run destination.
