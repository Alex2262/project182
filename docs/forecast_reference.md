# Reference-backed 6×256 forecasting baselines

The new forecasting backbones live in `src/forecast_reference.py`. They are
trained from scratch on continuous observation pairs, with scalar regression
readouts. They do not download pretrained models. Historical experiments keep
their original `forecast_models.py` implementation and checkpoints.
The existing training runtime selects these backbones when the model config has
`implementation: reference`; old configurations continue to select the legacy
implementation. Reference runs also hash `forecast_reference.py` in their source
provenance. The dedicated benchmark image includes the required dependencies;
historical Modal launchers keep their original lightweight images.

| Setting | Transformer | Gated DeltaNet |
|---|---|---|
| Reference implementation | Hugging Face `LlamaModel`, Transformers 5.6.2 | FLA `GatedDeltaNetModel`, flash-linear-attention/fla-core 0.6.0 |
| Layers / width | 6 / 256 | 6 / 256 |
| Heads | 4 full attention heads | 4 recurrent heads |
| Key / value dimensions per head | 64 / 64 | 64 / 128 |
| Feedforward | SwiGLU, intermediate width 768 | SwiGLU, intermediate width 768 |
| Normalization | RMSNorm, epsilon 1e-6 | RMSNorm, epsilon 1e-6 |
| Positions | Identity rotary adapter: NoPE | Recurrence and short convolution |
| Extra GDN settings | — | Length-4 short convolution, output gate, L2 Q/K normalization, scalar decay/write gates, nonnegative eigenvalue setting |
| Parameters / optimizer state | FP32 | FP32 |
| Precision modes benchmarked | FP32 and BF16 autocast | FP32 and BF16 autocast |
| Regression readout / loss | FP32 | FP32 |

The Transformer retains the upstream attention, MLP, normalization, model loop,
and initialization; the deliberate positional adaptation supplies cos=1/sin=0.
Both models replace vocabulary input/output layers with continuous projections.
The GDN uses upstream fused kernels, without a local recurrence replacement.
Both wrappers retain the `model(tokens, states=None, return_state=False)` API;
the carried state is the reference implementation's cache object.

This is a new architecture condition, not a depth-only ablation of the earlier
LayerNorm/GELU models. Equal depth and width do not imply matched parameter count
or state size. The benchmark records both. NoPE is not a standard positional
Llama model, and these small models are not time-series foundation models.

## Benchmark protocol

The CUDA image installs Triton 3.7.1 after PyTorch. FLA 0.6.0 explicitly rejects
Triton 3.4--3.7.0 for its gated backward kernel on Hopper because of incorrect
gradients. This is a deliberate override of PyTorch 2.10's Triton 3.6.0 package
metadata pin; the benchmark uses eager PyTorch plus FLA JIT kernels, not
`torch.compile`. Kernel correctness checks run before training timings. Compiler
caches live on each container's local disk rather than the shared artifact volume.

**Precision naming:** `float32` in the initial sweep denotes FP32 tensors with
PyTorch TF32 disabled. FLA's Triton kernels have a separate dot-product precision
default, which is TF32 on these NVIDIA GPUs. The `--strict-fp32` supplemental
benchmark sets `TRITON_F32_DEFAULT=ieee` before importing FLA and checks FP32
streaming agreement; explicit precision choices inside upstream kernels remain
unchanged. Do not describe the default FLA tensor-FP32 timing as strict IEEE math.

Resolved specification: `src/conf/forecast_reference_benchmark.yaml`.

- Personal Modal `alexander-tian`, existing `project182-artifacts` volume.
- One A100-40GB and one H100; 4 CPU cores and 16 GiB RAM per worker.
- At most 1,200 seconds per worker, no automatic retries, $5 allocation.
- Adam, learning rate 3e-4, beta=(0.9,0.999), epsilon 1e-8, no weight decay,
  gradient clipping at 1; no distributed training or graph compilation.
- Fresh AR(1) batches, independent coefficient per series, uniform prior
  [-0.95,0.95], innovation SD 0.5, zero initial observation.
- Lengths 64 and 2,048; 2,048 and 8,192 prediction positions per update.
- Four warmup updates and 20 measured updates for each model/precision/batch/length.
- End-to-end timings include sampling, transfer, forward, FP32 MSE, backward,
  gradient clipping, Adam update and GPU synchronization. Sampler/transfer time
  is reported separately. First-call compilation is excluded from steady state.
- Evaluation times full causal forward passes at 2k, 16k and 64k in BF16, batch 1;
  one warmup followed by three measured calls. All output positions are computed.
  These are inference-only timings, excluding data/reference construction.
- No validation or confirmation examples are used. Benchmark updates are not
  evidence of convergence, and multiplying step time by 30k predicts runtime,
  not the number of updates required to converge.

Correctness checks include future-token causality, cached versus full-sequence
predictions, all-parameter gradients on the CPU Transformer adapter, upstream
GDN chunk outputs/state/gradients against its naive recurrence, and FP32 GDN
carried state. BF16-versus-FP32 prediction differences are measured at
initialization only. Precision must be checked again on trained checkpoints
before interpreting very small Bayes excess-error differences.

Results are saved locally under `outputs/forecast_reference_control/` and on the
personal volume under `forecast_reference_benchmark/`, including resolved
configuration, source hashes, dependency versions, timing samples and failures.

```bash
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run scripts/modal_forecast_reference.py --action dry-run
MODAL_PROFILE=alexander-tian .venv-modal/bin/modal run --detach scripts/modal_forecast_reference.py --action benchmark
```

Each benchmark invocation creates two new GPU calls. Do not rerun it merely to
retrieve a result; use the recorded function call IDs or saved artifacts.
