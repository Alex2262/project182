"""End-to-end training timings, with compilation separated from steady state."""
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from forecast_data import sample_streams, transition_tokens
from forecast_reference import ReferenceForecaster


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def save(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def batch(seed, count, length, device='cuda'):
    data = sample_streams(seed, count, length)
    return (torch.as_tensor(transition_tokens(data['values']), dtype=torch.float32, device=device),
            torch.as_tensor(data['values'][:, 2:], dtype=torch.float32, device=device))


def relative_error(actual, expected):
    return float(((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).detach())


def check_delta_kernel():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.ops.gated_delta_rule.naive import naive_recurrent_gated_delta_rule
    torch.manual_seed(935000001)
    q, k = [torch.nn.functional.normalize(torch.randn(1, 128, 4, 64, device='cuda'), dim=-1)
            .to(torch.bfloat16).requires_grad_() for _ in range(2)]
    v = torch.randn(1, 128, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = (-torch.rand(1, 128, 4, device='cuda') * .03).requires_grad_()
    beta = torch.rand(1, 128, 4, device='cuda').requires_grad_()
    h = torch.randn(1, 4, 64, 128, device='cuda') * .1
    actual, state = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=h,
                                         output_final_state=True, use_qk_l2norm_in_kernel=False)
    expected, reference_state = naive_recurrent_gated_delta_rule(q, k, v, beta, g,
                                                                initial_state=h, output_final_state=True)
    probe = torch.randn_like(expected)
    inputs = (q, k, v, g, beta)
    actual_grads = torch.autograd.grad((actual.float() * probe).sum(), inputs)
    reference_grads = torch.autograd.grad((expected * probe).sum(), inputs)
    errors = dict(output_relative_l2=relative_error(actual, expected),
                  state_relative_l2=relative_error(state, reference_state),
                  gradient_relative_l2={name: relative_error(a, b) for name, a, b in
                                        zip(['q', 'k', 'v', 'log_decay', 'beta'], actual_grads, reference_grads)},
                  state_dtype=str(state.dtype))
    assert errors['output_relative_l2'] < .02, errors
    assert errors['state_relative_l2'] < .02, errors
    assert max(errors['gradient_relative_l2'].values()) < .05, errors
    assert state.dtype == torch.float32, errors
    return errors


@torch.no_grad()
def check_model(model):
    model.eval()
    x, _ = batch(935000002, 2, 256)
    full = model(x)
    changed = x.clone(); changed[:, 128:] += 2
    future_error = float((model(changed)[:, :128] - full[:, :128]).abs().max())
    assert future_error < 1e-5, future_error
    first, cache = model(x[:, :128], return_state=True)
    second, cache = model(x[:, 128:], cache, return_state=True)
    streaming_error = float((torch.cat([first, second], dim=1) - full).abs().max())
    # BF16 GEMM rounding changes with sequence shape. Record this diagnostic;
    # use strict FP32 cached/full agreement as the semantic equivalence check.
    result = dict(causality_max_abs=future_error, bf16_streaming_max_abs=streaming_error,
                  bf16_streaming_prediction_mse=float((torch.cat([first, second], 1) - full).square().mean()))
    if model.config['family'] == 'gated_deltanet':
        tensors = [cache[i]['recurrent_state'] for i in range(len(model.backbone.layers))]
        result['recurrent_state_dtypes'] = sorted(set(str(t.dtype) for t in tensors))
        assert result['recurrent_state_dtypes'] == ['torch.float32']
    # This is an initialization-time precision check, NOT proof about trained error floors.
    model.config['precision'] = 'float32'
    try:
        fp32 = model(x)
        result['bf16_vs_fp32_prediction_mse_at_initialization'] = float((full - fp32).square().mean())
        result['bf16_vs_fp32_prediction_max_abs_at_initialization'] = float((full - fp32).abs().max())
        a32, cache32 = model(x[:, :128], return_state=True)
        b32, _ = model(x[:, 128:], cache32, return_state=True)
        result['fp32_streaming_max_abs'] = float((torch.cat([a32, b32], 1) - fp32).abs().max())
        result['fp32_streaming_passed'] = result['fp32_streaming_max_abs'] < 3e-5
    except (RuntimeError, AssertionError) as exc:
        result['fp32_comparison_error'] = str(exc)[:1500]
    finally:
        model.config['precision'] = 'bfloat16'
    return result


def run_benchmark(plan, output):
    started = time.monotonic()
    torch.set_num_threads(plan['modal']['cpu'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    result = dict(status='running', gpu=torch.cuda.get_device_name(), plan=plan,
                  versions={name: importlib.metadata.version(name) for name in plan['dependencies']},
                  checks={}, models={}, training=[], evaluation=[],
                  validation_or_confirmation_data_used=False)
    result['triton_f32_default'] = os.environ.get('TRITON_F32_DEFAULT', 'tf32 (implicit NVIDIA default)')
    def record():
        result['elapsed_seconds'] = time.monotonic() - started
        save(Path(output) / 'benchmark.json', result)
    record()
    emit('benchmark_start', gpu=result['gpu'])
    result['checks']['gdn_kernel'] = check_delta_kernel()
    record(); emit('gdn_kernel_verified', **result['checks']['gdn_kernel'])
    for family in plan['families']:
        cfg = dict(plan['model'], family=family, precision='bfloat16')
        torch.manual_seed(plan['training']['seed'])
        model = ReferenceForecaster(cfg).cuda().float()
        result['models'][family] = model.description()
        result['models'][family]['resolved_backbone_config'] = model.backbone.config.to_dict()
        result['checks'][family] = check_model(model)
        record(); emit('model_checked', family=family, description=model.description(), checks=result['checks'][family])
        del model
        for precision in plan['precisions']:
            for tokens_per_step in plan['training']['tokens_per_step_cases']:
                for length in plan['training']['lengths']:
                    if time.monotonic() - started > 1000:
                        result['status'] = 'partial_time_limit'; record(); return result
                    case = dict(family=family, precision=precision, length=length,
                                tokens_per_step=tokens_per_step, batch_size=tokens_per_step // length)
                    emit('training_case_start', **case)
                    torch.cuda.empty_cache(); torch.manual_seed(plan['training']['seed'])
                    model = ReferenceForecaster(dict(cfg, precision=precision)).cuda().float().train()
                    tr = plan['training']
                    opt = torch.optim.Adam(model.parameters(), lr=tr['learning_rate'],
                                           betas=tuple(tr['betas']), eps=tr['eps'], weight_decay=0.)
                    torch.cuda.reset_peak_memory_stats()
                    timings, sampler, losses = [], [], []
                    try:
                        for step in range(tr['warmup_updates'] + tr['measured_updates']):
                            torch.cuda.synchronize(); before = time.monotonic()
                            x, y = batch(tr['data_seed'] + step, case['batch_size'], length)
                            torch.cuda.synchronize(); sampled = time.monotonic()
                            opt.zero_grad(set_to_none=True)
                            loss = (model(x) - y).square().mean()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), tr['grad_clip'], error_if_nonfinite=True)
                            opt.step(); torch.cuda.synchronize()
                            timings.append(time.monotonic() - before); sampler.append(sampled - before)
                            losses.append(float(loss.detach()))
                        steady = timings[tr['warmup_updates']:]
                        case.update(status='passed', median_seconds_per_step=float(np.median(steady)),
                                    p90_seconds_per_step=float(np.quantile(steady, .9)),
                                    mean_seconds_per_step=float(np.mean(steady)),
                                    sampler_and_transfer_median_seconds=float(np.median(sampler[tr['warmup_updates']:])),
                                    first_step_seconds=timings[0], warmup_seconds=sum(timings[:tr['warmup_updates']]),
                                    tokens_per_second=tokens_per_step / float(np.median(steady)),
                                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                    first_loss=losses[0], last_loss=losses[-1],
                                    measured_step_seconds=steady)
                    except (RuntimeError, AssertionError) as exc:
                        case.update(status='failed', error=str(exc)[:2000])
                    result['training'].append(case); record(); emit('training_case_done', **case)
                    del model, opt
                    torch.cuda.empty_cache()
        # Full forward, all positions, no cached evaluation: conservative versus last-layer selected queries.
        model = ReferenceForecaster(cfg).cuda().float().eval()
        with torch.no_grad():
            for length in plan['evaluation']['lengths']:
                if time.monotonic() - started > 1050:
                    break
                case = dict(family=family, precision='bfloat16', length=length, batch_size=1,
                            method='full causal forward; all output positions')
                try:
                    x, _ = batch(935000010, 1, length)
                    timings = []; torch.cuda.reset_peak_memory_stats()
                    for _ in range(plan['evaluation']['warmup_calls'] + plan['evaluation']['measured_calls']):
                        torch.cuda.synchronize(); before = time.monotonic()
                        pred = model(x); torch.cuda.synchronize()
                        timings.append(time.monotonic() - before)
                        assert torch.isfinite(pred).all()
                    case.update(status='passed', median_seconds_per_series=float(np.median(timings[1:])),
                                first_call_seconds=timings[0], peak_allocated_bytes=torch.cuda.max_memory_allocated())
                except (RuntimeError, AssertionError) as exc:
                    case.update(status='failed', error=str(exc)[:2000])
                result['evaluation'].append(case); record(); emit('evaluation_case_done', **case)
                torch.cuda.empty_cache()
        del model; torch.cuda.empty_cache()
    result['status'] = 'finished' if all(c['status'] == 'passed' for c in result['training'] + result['evaluation']) else 'finished_with_unsupported_cases'
    result['fp32_streaming_checks_passed'] = all(result['checks'][f].get('fp32_streaming_passed', False) for f in plan['families'])
    if not result['fp32_streaming_checks_passed']:
        result['status'] = 'finished_with_failed_checks'
    record()
    return result
