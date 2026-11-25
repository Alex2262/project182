# Approved 80k training, long-context adaptation, and layer ablations

Ten jobs on personal Modal (`alexander-tian`), approved November 24, 2026. No precision experiments or automatic retries. Budget: $35. All models remain 4 layers × width 256, using the same reference implementations and synthetic AR(1) task as the completed 40k sweep.

| Jobs | Experiment | Seeds | Steps |
|---|---|---|---|
| 1–2 | Fresh GDN | 185, 186 | 80,000 |
| 3–4 | Fresh Transformer | 185, 186 | 80,000 |
| 5–6 | GDN long-context adaptation | 185, 186 | 5,000 additional |
| 7–8 | GDN matched reset adaptation | 185, 186 | 5,000 additional |
| 9–10 | Frozen GDN layer ablations | 185, 186 | No training |

Fresh runs use peak LR 1e-4, 1k linear warmup, cosine decay to 1e-5, alternating 64/2k histories, and 8,192 prediction positions per optimizer step. Checkpoints: 40k, 60k, 80k. Fixed 2k validation every 10k; all saved checkpoints evaluated through 64k with full-history and fresh latest-2k policies on the existing 512-series validation cohort.

Adaptations restore weights and Adam state from the matching 40k GDN checkpoint at peak LR 1e-4. Warm up from its final LR 1e-5 to 3e-5 over 200 steps, then cosine decay to 3e-6 by adaptation step 5,000. Each optimizer step receives one 65,536-position AR(1) stream. Both arms use exactly the same per-step stream seed. Long adaptation backpropagates through the full stream. Reset adaptation reshapes the same observed transition tokens and targets into 32 chunks of 2,048, processes microbatches of four independent fresh-state chunks, and accumulates loss-weighted gradients before a single clip and optimizer step. Checkpoints: adaptation steps 1k, 2.5k, 5k; fixed 2k validation every 1k. The final lifetime prediction-position budget is 655.36M for both fresh and adapted models.

Layer ablations use the first 64 series from the existing 512-series cohort at query origins 16k and 64k. Each intervention holds the last 2k inputs and query fixed. Conditions: carry all state, zero one recurrent layer (four conditions), zero all recurrent memory, zero convolution caches only, zero all states. Cache contents are cloned before each intervention. The untouched original state is never reused after mutation. Fresh-suffix predictions and state norms are saved alongside all intervention predictions.

Two seeds are exploratory replication. These comparisons test optimization, training-context exposure, and state localization; they do not establish a universal recurrent-model limitation or convergence. Full 64k adaptation versus 2k reset adaptation changes both available history and gradient horizon. The reserved confirmation cohort remains unopened.

The CPU checks verified exact fast/standard AR stream equality, identical paired tokens and targets, loss weighting and accumulated gradients, isolated state edits, and LR schedule endpoints/monotonicity. Parent checkpoint hashes are checked on each worker before loading.

Runtime plan: `scripts/modal_forecast_next.py`; human-readable settings: `src/conf/forecast_next.yaml`. Each launch writes its complete resolved configuration, source hashes, parent hashes, Git revision, call IDs and final records under `outputs/forecast_reference_train_control/`. Source snapshots, model/optimizer checkpoints and predictions persist in the personal `project182-artifacts` volume. Eight training jobs log to personal W&B; evaluation-only ablation records are persisted directly in the volume.

Worker limits: fresh GDN 65 minutes; fresh Transformer 40 minutes; each adaptation 50 minutes; each ablation 15 minutes. Estimated maximum allocated compute: $31.31 before startup/teardown and billing overhead. Training stops before its hard limit to preserve checkpoints and leave evaluation time; incomplete jobs are reported explicitly.
