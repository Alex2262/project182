# Does training on accumulated state improve GDN's long-stream behavior?

Approved November 24, 2026. Six personal Modal runs: carry and reset conditions,
each initialized from GDN 30k cosine checkpoints with seeds 182, 183 and 184.

The test changes the recurrent state available during training while holding
the scored observations, targets, gradient length and update count fixed.
This is additional offline training. Evaluation never updates model weights.

| Setting | Both conditions |
|---|---|
| Starting point | Same parent weights and Adam optimizer state within each seed |
| New training series | AR(1), a uniform on [-0.95, 0.95], noise SD 0.5, y0=0 |
| Data seeds | 440000000 + 100000*(seed-182) + zero-based additional update |
| Warm-up | 16,384 observed transitions, processed with current weights and no gradients |
| Scored segment | Following 2,048 transitions; loss on all 2,048 next-value predictions |
| Batch | One independent 18,432-transition series per update |
| Updates | 3,000 additional; final total 33,000 |
| Optimizer | Continue parent Adam; constant LR 0.00003; clip norm 1; FP32 |
| Model | Same two-layer width-64 GDN, 4 heads, chunk size 64 |
| Evaluation | Frozen weights; full history and fresh latest-2k replay |
| Origins | 2,048 / 4,096 / 8,192 / 16,384 / 32,768 / 65,536 |
| Evaluation cohort | Same 512 validation series, seed 920000000 |

**Carry:** keep every layer's recurrent and convolution states after warm-up;
detach them before the scored segment. **Reset:** execute the same warm-up,
then discard all recurrent and convolution states before the scored segment.
Both conditions backpropagate only through the scored segment. Prefix states
are recomputed using current weights on each new training series; no stale state
is carried across optimizer updates or series.

## Registered comparisons

Primary: carry-minus-reset error against the true noise-free next-value mean at
65,536 observations, after exactly 3,000 additional updates. Compare three seeds
and paired series uncertainty. Report total expected MSE as well (add 0.25).

Supporting: full versus latest-2k policy within each condition; absolute change
from the frozen 30k parent; the shape of the curve across all origins. A reduced
replay advantage alone is not success if full-history accuracy does not improve.
Improvement only near the warm-up length suggests a shifted generalization limit.
Failure of this intervention cannot establish an inherent architectural limit.

This treatment also changes which old evidence is available during training;
it does not uniquely identify a gate, numerical mechanism, or pure state-age effect.
The reused validation set is exploratory. Confirmation seed 940000000 stays unopened.

## Verification and execution

CPU and GPU checks require carry predictions to match uninterrupted processing,
reset predictions and gradients to match fresh processing, zero input gradients
through the prefix in both conditions, and nonzero suffix gradients. Training
inputs and targets must match exactly between paired configurations. Verify parent
checkpoint hashes and unchanged data/model source hashes before training.

GPU preflight measures both real update paths. Proceed only if the timing estimate
with 50% headroom fits $15 and the per-run timeout allowance. A100 40GB, 2 CPU cores,
8 GiB RAM; preflight timeout 300 seconds, experiment timeout 3,600 seconds. Training
stops safely near 3,000 seconds, reserving time for final evaluation. No automatic
retries; exclusive submission records prevent duplicate launches.

Each run checkpoints every 100 updates and evaluates only the final endpoint.
Personal Modal workspace `alexander-tian`, volume `project182-artifacts`, W&B
`alexander-tian-ucb/project182`. No MatX resources. Local records are under
`outputs/forecast_warmup_control/`; cloud outputs under `/artifacts/forecast_warmup/`.
