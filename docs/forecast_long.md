# Full history versus recent replay through 64k

Approved on November 24, 2026. Personal Modal workspace `alexander-tian` only.

The question is whether the full-history disadvantage previously observed for
Gated DeltaNet persists at 32,768 and 65,536 observations, and whether a matched
Transformer exhibits the same behavior.

## Registered batch

| Component | Configuration |
| --- | --- |
| GDN | Reuse the three final 30,000-step cosine checkpoints, seeds 182–184 |
| Transformer | Three fresh NoPE models, seeds 182–184; 2 layers, width 64, 4 heads |
| Training | Alternate histories 64 and 2,048; 2,048 predictions/update; 30,000 updates |
| Optimizer | FP32 Adam, LR 0.0003 for first 10k updates, cosine to 0.00003 at 30k; clip norm 1 |
| Data | Existing AR(1) task, parameter uniform in [-0.95, 0.95], innovation SD 0.5 |
| Evaluation | Same 512 validation series, seed 920000000 |
| Origins | 2,048; 4,096; 8,192; 16,384; 32,768; 65,536 |
| Policies | Full causal history; fresh-state replay of latest 2,048 observed transitions |
| Hardware | One A100 40GB, 2 CPU cores and 8 GiB RAM per invocation |

No training, hyperparameter selection, or evaluation uses the reserved confirmation
seed 940000000. This extension remains exploratory. Matched recipes do not establish
that either architecture was individually optimized.

## Correctness and feasibility gate

`forecast_long.transformer_selected` computes exact full causal attention with
bounded blocks of queries. Every query retains all its preceding keys and values.
The final layer computes only requested outputs; earlier layers compute every
required position. This reduces temporary memory, not the history available to
the model. It remains quadratic in sequence length.

CPU checks compare this implementation with a trained ordinary Transformer and
test that future observations cannot change earlier predictions. The GPU preflight
repeats these checks, measures actual 64k inference and training speed, and permits
the batch only if the projected cost with 50% timing headroom is within $15.

GDN evaluations also verify that the regenerated data prefix matches the original
saved dataset and that predictions at old origins reproduce the saved results.
Both models must produce matching full/window predictions at origin 2,048.

## Compute controls and records

The preflight has a 300-second limit. Each Transformer training/evaluation invocation
has a 5,400-second limit, with training stopping safely by approximately 4,200 seconds.
Each GDN evaluation has a 1,800-second limit. Automatic retries are disabled, and
exclusive local submission records prevent duplicate scientific jobs.

At Modal's published A100 40GB/CPU/memory rates on November 24, these invocation
limits reserve approximately $13.73, leaving headroom within the $15 allowance for
startup overhead. This is a compute estimate, not an account-wide billing limit.
Pricing source: <https://modal.com/pricing>.

Resolved configurations, checkpoint hashes, source hashes, logs and submission IDs
are saved in `outputs/forecast_long_control/`. Cloud outputs are under
`/artifacts/forecast_long/` on the personal `project182-artifacts` volume.
Training logs go to `alexander-tian-ucb/project182` in W&B.

Final comparisons should show conditional-mean error, the paired full-minus-window
error difference, individual training seeds, and uncertainty across validation
series. Also report the corresponding total predictive MSE: innovation variance
0.25 dominates the reducible error, so large relative reducible-error differences
can imply small total-MSE improvements.
