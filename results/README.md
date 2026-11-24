# Experiment results

- **[November 24: paired full-history versus recent-window forecasts](2026-11-24-forecast-windows/)** — three frozen models on the same targets: full history wins at 4k, but latest-2,048 replay wins at 16k; local CPU evaluation, paired uncertainty, and reproducible artifacts. Latest forecasting results.

- **[November 23: 30k optimization continuations](2026-11-23-forecast-optimization/)** — three paired seeds, constant versus cosine learning rate; large accuracy gains from decay, with long-history deterioration remaining.

- **[November 23: training-history sweep through 16k observations](2026-11-23-forecast-horizon/)** — nine new GDN runs, matched reset control, six extended baseline evaluations, and predeclared deterioration points.

- **[November 23: streaming AR(1), 10k updates × three seeds](2026-11-23-forecast-ar1-10k/)** — all twelve runs complete; training-length intervention, seed variability, checkpoint trajectories, and verified retained weights.

- **[November 23: streaming AR(1) forecasting](2026-11-23-forecast-ar1/)** — GDN and Transformer, short versus mixed training lengths; four completed 2,000-update runs, first history-length curves, and convergence checks. One training seed; exploratory validation.
- **[November 23: forecasting preparation](2026-11-23-forecast-preparation/)** — statistical reference feasibility and CPU verification before the learned-model pilot.
- **[November 20: noisy linear regression, four architectures × three seeds](2026-11-20-linear-d5/)** — 2k/5k/10k checkpoint comparison, confirmation errors, learning curves, order sensitivity, and decay gates.

Each report includes viewable figures and downloadable CSV/JSON tables. Training and evaluation code lives in the repository root. The forecasting reports include compact predictions and references for reproducing their plots and tables. Raw model checkpoints are excluded from Git; their locations are documented in the manifests.
