# Approved 6×256, 20k-step experiment

Approved on 2026-11-24: eight independent H100 workers in personal Modal
`alexander-tian`. No work compute or work experiment logging is used.

| Architecture | Arm | Learning rate | Seeds | Steps |
|---|---|---:|---|---:|
| HF Llama blocks, NoPE | Main | 1e-4 | 182, 183, 184 | 20,000 |
| FLA Gated DeltaNet | Main | 1e-4 | 182, 183, 184 | 20,000 |
| HF Llama blocks, NoPE | LR sensitivity | 3e-4 | 182 | 20,000 |
| FLA Gated DeltaNet | LR sensitivity | 3e-4 | 182 | 20,000 |

Both architectures use six layers, width 256, four heads, SwiGLU intermediate
width 768 and RMSNorm. Transformer has 5,116,161 parameters; GDN has 6,726,705.
These are equal-depth/width comparisons, not parameter-matched models. Reference
backbone details and deliberate adaptations are in `forecast_reference.md`.

Each run alternates lengths 64 and 2,048, with batches 128 and 4. This is
8,192 prediction positions per step and 163,840,000 per completed run. Data are
fresh AR(1) series, a coefficient drawn independently per series from U[-.95,.95],
innovation SD .5, and initial observation zero. Each token is an observed pair;
its target is the next observation. Corresponding seeds use identical training
data across architecture and LR arms. State starts fresh for each series.
Adam uses constant LR, betas (.9,.999), epsilon 1e-8, no weight decay and gradient
clipping at 1. BF16 backbone operations use FP32 parameters, optimizer state,
regression readout and loss. Triton is pinned to 3.7.1; its FP32 dot default is
IEEE for the supplemental precision checks.

Save checkpoints at 5k, 10k and 20k steps. Evaluate after training, final checkpoint
first, at 2k/4k/8k/16k/32k/64k histories on the same 512 validation series
(seed 920000000). Evaluate full causal prefixes and fresh-state latest-2k windows
at identical origins. Primary metric is squared error against the true conditional
mean a*y[t]. Save per-series predictions, errors relative to full-history Bayes,
and paired window-minus-full differences. The confirmation set is not accessed.

The window Bayes reference conditions on the first visible observation as well
as the visible transitions: that observation's variance depends on the unknown
coefficient and the original stream's fixed-zero start. Compute its posterior
by float64 quadrature and compare 512- and 1,024-node results. At origin 2,048,
the window and full information sets are identical. Per-series uncertainty is
conditional on trained models; three-seed variability must be reported separately.

Primary seed 182 workers additionally compare BF16 and FP32 predictions on four
series at 2k and 16k if time permits. This is a limited numerical diagnostic, not
a guarantee on the 64k error floor. Precision checks cannot alter model selection.
LR sensitivity runs are reported separately from the three-seed primary results.
The 20k step endpoint is a compute budget, not a claim of convergence.

## Execution and limits

Resolved configs: `src/conf/forecast_reference_train.yaml` and the per-batch
`outputs/forecast_reference_train_control/*/resolved_plan.json`.
Launcher: `scripts/modal_forecast_reference_train.py`.
Training and evaluation: `src/forecast_reference_train.py`.

Eight workers maximum, H100, four CPU cores, 16 GiB RAM, timeout 2,040 seconds,
no retries, two-second scaledown. Nominal maximum worker allocation is ~$19.37
at the published rates, with a $20 batch budget; startup and image-build overhead
are separate. Training stops before 1,500 worker seconds and evaluation checks a
1,950-second soft deadline so partial results can be saved. A bounded run is
reported as partial, not treated as a completed 20k experiment.

Artifacts: personal `project182-artifacts` volume under
`forecast_reference_train/<run_name>`. Each includes source snapshots/hashes,
resolved config, backbone config, checkpoints, training curves, compact reference
arrays and per-series predictions. W&B is explicitly personal
`alexander-tian-ucb/project182`, with identity checked before online logging.
Local metrics remain authoritative if logging is unavailable. No commits or
pushes are part of this launch.
