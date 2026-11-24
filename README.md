# Project182: recurrent in-context forecasting

We study whether a recurrent forecaster keeps benefiting from a longer observation history, and how training history and optimization affect that behavior. The controlled AR(1) task has an exact Bayesian reference using compact running statistics.

- **[Latest result: paired full-history versus recent-window forecasts](results/2026-11-24-forecast-windows/README.md)** — full history wins at 4k observations, but replaying the latest 2,048 wins at 16k in all three tested models.
- **[All experiment reports, plots, and tables](results/README.md)** — includes the forecasting pilots, training-history sweep, optimization continuations, and original regression control.
- **[Forecasting setup and statistical reference](docs/forecast_pilot.md)** and **[latest evaluation protocol](docs/forecast_windows.md)**.

The reports include saved predictions and references so teammates can inspect results and regenerate plots without GPU jobs. Install `requirements-forecast.txt`, then run from this repository's root:

```bash
python scripts/evaluate_forecast_windows.py --repo . \
  --output results/2026-11-24-forecast-windows --summarize-only
```

The latest results remain exploratory: three training seeds on the existing validation cohort, with independent confirmation pending. Raw model checkpoints are excluded from Git; their locations and hashes are recorded in the experiment manifests.

Personal experiment repository: `Alex2262/project182`. Compute runs in the
`alexander-tian` Modal workspace; W&B logs go to `alexander-tian-ucb/project182`.

## Earlier regression pilot

The first pilot compares HF GPT-2, normalized linear attention, Mamba-2, and
Gated DeltaNet on noisy linear regression. It uses 4 layers, width 128, 4 heads,
33 input/label pairs (0–32 observed examples per prediction), batch size 64,
Gaussian weights with identity covariance, uniform inputs in [-1,1]^5, and
noise standard deviation 0.5. No pretrained weights are used.

Training uses FP32 tensors, no autocast, and disables PyTorch TF32. Mamba-2
uses FLA's PyTorch SSD fallback. Gated DeltaNet uses FLA's layer parameters
with an exact PyTorch triangular-solve recurrence, checked against explicit
state updates and their gradients in float64. Linear attention uses its
quadratic causal form. These implementations support architecture-quality
comparisons; their runtimes are **not** optimized-kernel comparisons.

The prior is known. Evaluation measures mean squared disagreement with the
Bayesian posterior mean, divided by mean Bayes risk for noisy labels. The
reference is computed on CPU in float64 at every prefix. Validation and test
sets use distinct fixed seeds; test results do not select learning rates.
All pilot configurations have a single training seed, so results are exploratory.

## Longer-run follow-up

The follow-up retains the original four architectures and learning rate 3e-4.
Seed 182 resumes its existing run IDs to 10,000 steps, retaining 2k, 5k, and
10k checkpoints. After reviewing validation curves, seeds 183 and 184 run
independently at the chosen fixed horizon. Data stream seeds are 1,000,000,
2,000,000, and 3,000,000; models within a seed receive identical training
prompts. A fresh 1,024-prompt confirmation set (seed 300,000,000) is opened
only after all 12 runs have completed. Order tests use 256 prompts and five
permutations at each context length. Reported SD is across training seeds;
three seeds give limited precision even when test-prompt error is small.

```bash
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_followup.py --stage extend --steps 10000
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_followup.py --stage replicate --steps 10000
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_followup.py --stage evaluate
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_followup.py --stage report
```

The single coordinator reserves each worker's complete hard timeout against
an additional $5 allocation before launching up to four GPU workers. Each
worker checkpoints before its time limit. Repeat a training stage to resume
partial runs; completed runs are skipped. If the remaining budget cannot
reserve every pending job at once, only an affordable subset launches.
The ledger, manifest, confirmation predictions, and report are stored under
`followup_v1/` in the existing personal Modal Volume. Original pilot test
scores remain labeled separately from confirmation scores. All cloud
training uses the same unmodified model, sampler, and task implementations
as the initial pilot; this follow-up changes duration and random seeds.

## Run the pilot

Install the local launcher using the personal Modal setup below. Cloud training
dependencies are pinned in `requirements-pilot.txt` and the Modal image; the
legacy `environment.yml` belongs to the original upstream project.

```bash
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action check --max-seconds 300
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action sweep --steps 250 --max-seconds 600
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action selected --steps 1000 --max-seconds 600
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action selected --steps 2000 --max-seconds 600
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action evaluate-selected --max-seconds 300
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_pilot.py --action report
```

The sweep tries 1e-4 and 3e-4 for 250 steps. When the two validation scores
are within 5%, both continue to 1,000 steps; otherwise only the better rate
continues. One rate per architecture then continues to 2,000 steps. In the
first pilot, linear attention remained tied at 1,000 steps, so both of its
rates received 2,000 steps before final selection. A flat
learning curve makes the rate selection inconclusive, even if its validation
score is slightly lower. All stages resume saved checkpoints. The original
training loop's task-sampler and curriculum interfaces remain; `src/config.py`
loads inherited YAML without Quinine's obsolete dependency pins. Each run saves
its resolved YAML, source fingerprint, model and optimizer state, validation
history, and gate snapshots. The checkpoint counter is completed steps, so
resuming does not repeat the previous optimizer update.

The persistent `project182-artifacts` Volume stores `pilot_v1/`. A conservative
launcher ledger reserves each GPU call's full 900-second timeout against 5,400
seconds at $3/hour, leaving $0.50 of the $5 pilot allocation for other overhead.
Completed calls replace their reservation with elapsed time. Provider billing
can lag; the ledger is a conservative compute estimate, not a billing receipt.

---

## Original upstream project

This repository contains the code and models for our paper:

**What Can Transformers Learn In-Context? A Case Study of Simple Function Classes** <br>
*Shivam Garg\*, Dimitris Tsipras\*, Percy Liang, Gregory Valiant* <br>
Paper: http://arxiv.org/abs/2208.01066 <br><br>

![](setting.jpg)

```bibtex
    @InProceedings{garg2022what,
        title={What Can Transformers Learn In-Context? A Case Study of Simple Function Classes},
        author={Shivam Garg and Dimitris Tsipras and Percy Liang and Gregory Valiant},
        year={2022},
        booktitle={arXiv preprint}
    }
```

## Getting started
You can start by cloning our repository and following the steps below.

1. Install the dependencies for our code using Conda. You may need to adjust the environment YAML file depending on your setup.

    ```
    conda env create -f environment.yml
    conda activate in-context-learning
    ```

2. Download [model checkpoints](https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip) and extract them in the current directory.

    ```
    wget https://github.com/dtsip/in-context-learning/releases/download/initial/models.zip
    unzip models.zip
    ```

3. [Optional] If you plan to train, populate `conf/wandb.yaml` with you wandb info.

That's it! You can now explore our pre-trained models or train your own. The key entry points
are as follows (starting from `src`):
- The `eval.ipynb` notebook contains code to load our own pre-trained models, plot the pre-computed metrics, and evaluate them on new data.
- `train.py` takes as argument a configuration yaml from `conf` and trains the corresponding model. You can try `python train.py --config conf/toy.yaml` for a quick training run.

# Maintainers
* [Shivam Garg](https://cs.stanford.edu/~shivamg/)
* [Dimitris Tsipras](https://dtsipras.com/)


## Project182: personal Modal setup

The local Modal launcher uses a separate Python environment:

```bash
uv venv --python 3.12 .venv-modal
uv pip install --python .venv-modal/bin/python -r requirements-modal.txt
source .venv-modal/bin/activate
modal setup
```

Authenticate to the intended personal workspace. For this checkout, use the
`alexander-tian` profile explicitly when launching jobs:

```bash
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/check_modal.py
```

The setup check requests one A100 40 GB GPU with a two-minute execution timeout,
reports its identity, and writes `setup/gpu-check.json` to the persistent
`project182-artifacts` Volume. It does not run a training experiment. Modal
requires a payment method on file for this GPU even when credits are available.

W&B logging uses the personal account `alexander-tian`, workspace
`alexander-tian-ucb`, and project `project182`. These logging defaults are saved
in `src/conf/wandb.yaml`. The Modal Secret `project182-wandb` supplies
`WANDB_API_KEY` only to functions that explicitly request it.

Verify authentication without logging a run:

```bash
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/check_wandb.py
```

Add `--log-test` to create a CPU-only setup run and verify its saved metric.
This does not launch training. The separate GPU check does not require W&B.

Credentials remain in the local Modal configuration or Modal Secrets. Do not
put credentials in YAML configs, source files, or Git. The pilot launcher above
installs its pinned training dependencies in the personal Modal image.
