"""Produce a teammate-readable report from finished reference benchmark calls."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('control_directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--a100-directory', type=Path)
    parser.add_argument('--strict-directory', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    directories = dict(a100=args.a100_directory or args.control_directory, h100=args.control_directory)
    results = {gpu: json.loads((directory / f'{gpu}.json').read_text()) for gpu, directory in directories.items()}
    lines = ['# Reference 6×256 benchmark', '',
             'Measured steady-state throughput, not a convergence claim. Fresh synthetic AR(1) batches; '
             'no validation or confirmation examples were used.', '',
             'Transformer: Hugging Face LlamaModel with identity rotary (NoPE). '
             'GDN: FLA GatedDeltaNetModel. Both use 6 layers, width 256, 4 heads, '
             'RMSNorm and SwiGLU with intermediate width 768. GDN key/value head sizes are 64/128.', '',
             'Exact executed source snapshots are under `executed_source/`; their hashes are '
             'checked against each run record. `current_source/` contains the latest harness and report generator.', '',
             'Training includes CPU sampling, host-to-device transfer, forward, FP32 readout/MSE, '
             'backward, gradient clipping and Adam. Four warmup updates precede 20 timed updates. '
             'Thirty-thousand-update estimates alternate lengths 64 and 2,048 equally; '
             'compilation, checkpoint writing and validation are additional. '
             'For GDN, float32 here means FP32 tensors with the reference Triton TF32 dot default; '
             'the separate IEEE-default measurement below addresses this distinction.', '',
             '| GPU | Model | Precision | Positions/update | ms/update, mixed lengths | 10k updates | 30k updates |',
             '|---|---|---|---:|---:|---:|---:|']
    summaries = []
    for gpu, result in results.items():
        if 'training' not in result:
            lines += [f"| {gpu} | Benchmark failed | — | — | — | — | — |"]
            continue
        for family in result['plan']['families']:
            for precision in result['plan']['precisions']:
                for tokens in result['plan']['training']['tokens_per_step_cases']:
                    rows = [c for c in result['training'] if c['family'] == family and c['precision'] == precision
                            and c['tokens_per_step'] == tokens and c['status'] == 'passed']
                    if len(rows) != 2:
                        continue
                    seconds = sum(c['median_seconds_per_step'] for c in rows) / 2
                    summary = dict(gpu=gpu, actual_gpu=result['gpu'], family=family, precision=precision,
                                   tokens_per_step=tokens, mixed_seconds_per_update=seconds,
                                   minutes_10000_updates=10000 * seconds / 60,
                                   minutes_30000_updates=30000 * seconds / 60)
                    summaries.append(summary)
                    lines.append(f"| {gpu.upper()} | {family} | {precision} | {tokens:,} | {1000*seconds:.2f} | "
                                 f"{summary['minutes_10000_updates']:.1f} min | {summary['minutes_30000_updates']:.1f} min |")
    lines += ['', '## Evaluation', '',
              'Batch-one inference only, full forward at every layer and output position, BF16. '
              'The 512-series projection repeats this workload serially; it excludes data/reference '
              'generation and other history lengths. It is not a measured full evaluation run.', '',
              '| GPU | Model | History | Seconds/series | 512 series |', '|---|---|---:|---:|---:|']
    for gpu, result in results.items():
        for row in result.get('evaluation', []):
            if row['status'] == 'passed':
                seconds = row['median_seconds_per_series']
                lines.append(f"| {gpu.upper()} | {row['family']} | {row['length']:,} | {seconds:.4f} | {512*seconds/60:.2f} min |")
    lines += ['', '## Verification and limitations', '',
              '- The GPU kernel check compares FLA chunk outputs, carried state and gradients with its naive recurrence.',
              '- The CPU Transformer adapter check verifies causality, full/cached equality, identity positions and finite gradients.',
              '- BF16 full/cached predictions can differ through rounding. Initialization-time precision comparisons are diagnostic; '
              'they do not establish adequate precision at a trained model\'s error floor.',
              '- Mixed precision retains FP32 model parameters, optimizer state, regression readout/loss and GDN recurrent state.',
              '- Depth, normalization, feedforward structure and GDN value expansion changed relative to previous pilots; '
              'this is a new baseline condition, not a depth-only ablation.',
              '- FLA requires Triton >=3.7.1 for the gated backward kernel on H100. '
              'The image deliberately overrides PyTorch 2.10\'s Triton 3.6.0 package pin; no torch.compile is used.',
              '- An earlier attempt with the bundled compiler was stopped after FLA rejected the H100 backward pass. '
              'Its partial timings are excluded from this report.', '']
    for gpu, result in results.items():
        lines += [f'### {gpu.upper()}', '', f"Actual hardware: `{result.get('gpu', 'unknown')}`. "
                  f"Status: `{result['status']}`. Benchmark invocation: {result.get('elapsed_seconds', 0):.1f} seconds.", '']
        for family, model in result.get('models', {}).items():
            lines.append(f"- {family}: {model['parameters']:,} parameters; "
                         f"{model['persistent_state_elements_per_series']:,} fixed state/cache elements per series "
                         '(Transformer KV cache is separate and grows with history).')
        lines += ['', '```json', json.dumps(result.get('checks', {}), indent=2), '```', '']
        shutil.copyfile(directories[gpu] / f'{gpu}.json', args.output / f'{gpu}.json')
        shutil.copyfile(directories[gpu] / 'resolved_plan.json', args.output / f'{gpu}_resolved_plan.json')
        executed = directories[gpu] / f'{gpu}_executed_source'
        if executed.exists():
            shutil.copytree(executed, args.output / 'executed_source' / gpu, dirs_exist_ok=True)
    if args.strict_directory:
        strict = json.loads((args.strict_directory / 'h100.json').read_text())
        rows = [r for r in strict.get('training', []) if r['status'] == 'passed']
        lines += ['## GDN with IEEE-default FP32 dots on H100', '',
                  'The supplemental run sets TRITON_F32_DEFAULT=ieee before importing FLA. '
                  'Upstream explicit precision choices, such as TF32x3, are retained. '
                  'Both lengths use 8,192 positions/update.', '']
        if len(rows) == 2:
            seconds = sum(r['median_seconds_per_step'] for r in rows) / 2
            lines += [f'Mixed-length updates: **{seconds*1000:.2f} ms**; '
                      f'10k updates: **{seconds*10000/60:.1f} min**; '
                      f'30k updates: **{seconds*30000/60:.1f} min**.', '']
            summaries.append(dict(gpu='h100', family='gated_deltanet', precision='float32_ieee_default',
                                  tokens_per_step=8192, mixed_seconds_per_update=seconds,
                                  minutes_10000_updates=seconds*10000/60, minutes_30000_updates=seconds*30000/60))
        lines += ['```json', json.dumps(strict.get('checks', {}), indent=2), '```', '']
        shutil.copyfile(args.strict_directory / 'h100.json', args.output / 'h100_ieee.json')
        shutil.copyfile(args.strict_directory / 'resolved_plan.json', args.output / 'h100_ieee_resolved_plan.json')
        executed = args.strict_directory / 'h100_executed_source'
        if executed.exists():
            shutil.copytree(executed, args.output / 'executed_source' / 'h100_ieee', dirs_exist_ok=True)
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    shutil.copyfile(args.control_directory / 'resolved_plan.json', args.output / 'resolved_plan.json')
    for relative in ['src/forecast_reference.py', 'src/forecast_reference_benchmark.py', 'src/forecast_runtime.py',
                     'src/conf/forecast_reference_benchmark.yaml', 'scripts/modal_forecast_reference.py',
                     'scripts/check_forecast_reference.py', 'scripts/summarize_forecast_reference.py',
                     'requirements-forecast-reference.txt', 'docs/forecast_reference.md']:
        dest = args.output / 'current_source' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, dest)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
