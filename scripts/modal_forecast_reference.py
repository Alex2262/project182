"""Bounded, personal-workspace reference-backbone benchmarks; no full training."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'alexander-tian'
SOURCES = ['forecast_data.py', 'forecast_reference.py', 'forecast_reference_benchmark.py']


def personal():
    if not modal.is_local():
        return
    from modal import config
    assert os.environ.get('MODAL_PROFILE') == config._profile == PROFILE
    overrides = ['MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET', 'MODAL_OAUTH_REFRESH_TOKEN',
                 'MODAL_OAUTH_CLIENT_ID', 'MODAL_OAUTH_CLIENT_SECRET', 'MODAL_OAUTH_JWT_KEY',
                 'MODAL_CONFIG_PATH', 'MODAL_SERVER_URL']
    assert not any(os.environ.get(k) for k in overrides), 'Unexpected credential override'


personal()
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch==2.10.0', index_url='https://download.pytorch.org/whl/cu128')
         .pip_install('transformers==5.6.2', 'flash-linear-attention==0.6.0', 'fla-core==0.6.0',
                      'numpy==2.2.6', 'scipy==1.16.3', 'PyYAML==6.0.3')
         # FLA rejects Triton 3.4--3.7.0 on Hopper due to incorrect gradients.
         # Intentional compiler override of torch's dependency pin; no torch.compile.
         .pip_install('triton==3.7.1')
         .env({'PYTHONPATH': '/root/project/src', 'OMP_NUM_THREADS': '4',
               'TOKENIZERS_PARALLELISM': 'false', 'FLA_CONV_BACKEND': 'triton',
               'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
               'TRITON_CACHE_DIR': '/tmp/reference-benchmark-triton'}))
for name in SOURCES:
    image = image.add_local_file(ROOT / 'src' / name, remote_path=f'/root/project/src/{name}')
volume = modal.Volume.from_name('project182-artifacts', create_if_missing=False, version=2)
app = modal.App('project182-reference-benchmark')


def execute(plan, run_name):
    import tarfile
    import torch
    # PyTorch's TF32 flag does not control Triton's dot default.
    os.environ['TRITON_F32_DEFAULT'] = plan.get('triton_f32_default', 'tf32')
    from forecast_reference_benchmark import run_benchmark, save
    volume.reload()
    output = Path('/artifacts/forecast_reference_benchmark') / run_name
    output.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        (output / source).write_bytes((Path('/root/project/src') / source).read_bytes())
    gpu_key = torch.cuda.get_device_name().replace(' ', '_')
    archive = Path('/artifacts/reference_compiler_cache') / f'{gpu_key}_triton371.tar.gz'
    if archive.exists():
        with tarfile.open(archive) as handle:
            handle.extractall('/tmp/reference-benchmark-triton', filter='data')
    for name, sha in plan['source_sha256'].items():
        assert hashlib.sha256((Path('/root/project/src') / name).read_bytes()).hexdigest() == sha
    try:
        result = run_benchmark(plan, output)
        result['run_name'] = run_name
        save(output / 'benchmark.json', result)
        return result
    finally:
        archive.parent.mkdir(parents=True, exist_ok=True)
        cache = Path('/tmp/reference-benchmark-triton')
        if cache.exists():
            with tarfile.open(str(archive) + '.tmp', 'w:gz') as handle:
                for child in cache.iterdir():
                    handle.add(child, arcname=child.name)
            Path(str(archive) + '.tmp').replace(archive)
        volume.commit()


options = dict(image=image, cpu=4, memory=16384, timeout=1200, retries=0,
               max_containers=1, scaledown_window=2, volumes={'/artifacts': volume})
bench_a100 = app.function(gpu='A100-40GB', **options)(execute)


@app.function(gpu='H100', **options)
def bench_h100(plan, run_name):
    return execute(plan, run_name)


@app.local_entrypoint()
def main(action: str = 'dry-run', batch_name: str = '', gpu: str = 'both', strict_fp32: bool = False):
    import yaml
    personal()
    plan = yaml.safe_load((ROOT / 'src/conf/forecast_reference_benchmark.yaml').read_text())
    assert gpu in ['a100', 'h100', 'both']
    plan['selected_gpu'] = gpu
    plan['triton_f32_default'] = 'ieee' if strict_fp32 else 'tf32'
    if strict_fp32:
        plan['families'] = ['gated_deltanet']
        plan['precisions'] = ['float32']
        plan['training']['tokens_per_step_cases'] = [8192]
        plan['evaluation']['lengths'] = []
    assert plan['modal']['profile'] == PROFILE
    assert plan['modal']['budget_usd'] == 5
    plan['source_sha256'] = {name: hashlib.sha256((ROOT / 'src' / name).read_bytes()).hexdigest() for name in SOURCES}
    name = batch_name or datetime.now(ZoneInfo('America/Los_Angeles')).strftime('alex_%y%m%d_reference6x256_%H%M%S')
    if strict_fp32:
        name += '_ieee'
    out = ROOT / 'outputs/forecast_reference_control' / name
    if action == 'dry-run':
        print(json.dumps(plan, indent=2)); return
    if action != 'benchmark':
        raise ValueError('Only dry-run and benchmark are supported')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'resolved_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    calls = {}
    for gpu, function in [('a100', bench_a100), ('h100', bench_h100)]:
        if plan['selected_gpu'] not in ['both', gpu]:
            continue
        call = function.spawn(plan, f'{name}_{gpu}')
        calls[gpu] = call
        (out / 'calls.json').write_text(json.dumps({g: c.object_id for g, c in calls.items()}, indent=2) + '\n')
        print(json.dumps(dict(event='submitted', gpu=gpu, call_id=call.object_id, output=str(out))), flush=True)
    for gpu, call in calls.items():
        try:
            result = call.get()
        except Exception as exc:
            result = dict(status='failed', error=repr(exc))
        (out / f'{gpu}.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(event='completed', gpu=gpu, status=result['status'], output=str(out))), flush=True)
