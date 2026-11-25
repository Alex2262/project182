"""Exact full-history and recent-window forecasting through 65,536 observations.

Training/model definitions remain in their existing modules. Transformer
evaluation blocks queries to bound attention memory and computes only requested
positions in the final layer; causal dependencies and full keys are preserved.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from forecast_data import sample_streams, references, score
from forecast_models import Forecaster
from forecast_runtime import make_batch, make_optimizer, predict, setup, synchronize, write_json

ORIGINS = [2048, 4096, 8192, 16384, 32768, 65536]
POLICIES = ['full', 'window2048']


@torch.no_grad()
def transformer_selected(model, tokens, positions, query_block=256):
    """Exact causal Transformer predictions at zero-based positions, bounded Q blocks."""
    if model.config['family'] != 'transformer_nope':
        raise ValueError('Expected Transformer')
    positions = [int(p) for p in positions]
    assert positions == sorted(set(positions)) and 0 <= positions[0] <= positions[-1] < tokens.shape[1]
    # No later token is needed even when the supplied array contains a suffix.
    x = model.input(tokens[:, :positions[-1]+1])
    for layer, block in enumerate(model.blocks):
        batch, length, width = x.shape
        mixer = block.mixer
        qkv = mixer.qkv(block.norm1(x)).reshape(batch, length, 3, mixer.heads, width // mixer.heads)
        q, k, v = (a.transpose(1, 2) for a in qkv.unbind(2))
        wanted = positions if layer == len(model.blocks)-1 else list(range(length))
        chunks = []
        for start in range(0, len(wanted), query_block):
            ids = wanted[start:start+query_block]
            index = torch.tensor(ids, device=x.device)
            end = ids[-1]+1
            mask = torch.arange(end, device=x.device)[None, :] <= index[:, None]
            attended = F.scaled_dot_product_attention(q.index_select(2, index), k[:, :, :end],
                v[:, :, :end], attn_mask=mask, is_causal=False, dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(batch, len(ids), width)
            residual = x.index_select(1, index) + mixer.output(attended)
            chunks.append(residual + block.mlp(block.norm2(residual)))
        x = torch.cat(chunks, dim=1)
    return model.output(model.norm(x)).squeeze(-1)


def load_model(checkpoint, device, expected_sha=None):
    checkpoint = Path(checkpoint)
    if expected_sha is not None:
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Forecaster(saved['config']['model']).to(device).float().eval()
    model.load_state_dict(saved['model'])
    return model, saved


def make_data(count=512, length=65536, seed=920000000):
    data = sample_streams(seed, count, length, .95, .5)
    lengths = [n for n in ORIGINS if n <= length]
    data.update(references(data['values'], lengths, .95, .5))
    data['latent_mean'] = data['theta'][:, None] * data['values'][:, lengths]
    data['target'] = data['values'][:, np.array(lengths)+1]
    data['spec'] = np.array(json.dumps({'seed':seed, 'count':count, 'noise_std':.5,
        'bound':.95, 'initial_value':0.0, 'lengths':lengths}, sort_keys=True))
    return data


@torch.no_grad()
def evaluate_long(plan, checkpoint, output, device='cuda'):
    setup(device)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    spec_path = output/'evaluation_plan.json'
    if spec_path.exists(): assert json.loads(spec_path.read_text()) == plan
    write_json(spec_path, plan)
    started = time.monotonic()
    model, saved = load_model(checkpoint, device, plan.get('checkpoint_sha256'))
    assert saved['completed_steps'] == 30000
    assert saved['config']['model']['family'] == plan['family']
    assert saved['config']['training']['seed'] == plan['seed']
    assert saved['config']['training']['lengths'] == [64, 2048]
    assert saved['config']['training']['schedule'] == 'cosine_decay'
    for name in ['forecast_models.py', 'forecast_data.py']:
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == saved['config']['source_sha256'][name]
    data = make_data()
    reference = {k:v for k,v in data.items() if k != 'values'}
    np.savez_compressed(output/'validation_reference.npz', **reference)
    predictions = np.full((512, len(ORIGINS), 2), np.nan)
    partial = output/'partial_predictions.npz'
    completed = 0
    if partial.exists():
        with np.load(partial) as z:
            completed = int(z['completed_series']); predictions = z['prediction']
    batch = 32 if plan['family'] == 'gated_deltanet' else 1
    assert completed % batch == 0
    old = None
    if plan.get('old_data_directory'):
        olddir = Path(plan['old_data_directory'])
        with np.load(olddir/'data/validation.npz') as z:
            np.testing.assert_array_equal(data['theta'], z['theta'])
            np.testing.assert_array_equal(data['values'][:, :16386], z['values'])
            old_lengths = z['lengths'].tolist()
        with np.load(olddir/'validation_predictions_step_30000.npz') as z:
            old = z['prediction'][:, [old_lengths.index(t) for t in ORIGINS[:4]]]
    synchronize(device)
    inference_start = time.monotonic()
    for first in range(completed, 512, batch):
        last = min(first+batch, 512)
        values = data['values'][first:last]
        if plan['family'] == 'gated_deltanet':
            predictions[first:last, :, 0] = predict(model, {'values':values, 'lengths':np.array(ORIGINS)},
                device, batch_size=batch, streaming_chunk=64)
        else:
            tokens = torch.as_tensor(np.stack((values[:, :-2], values[:, 1:-1]), -1), dtype=torch.float32, device=device)
            predictions[first:last, :, 0] = transformer_selected(model, tokens, [t-1 for t in ORIGINS]).cpu().double().numpy()
        for j,t in enumerate(ORIGINS):
            obs = values[:, t-2048:t+1]
            tokens = torch.as_tensor(np.stack((obs[:, :-1], obs[:, 1:]), -1), dtype=torch.float32, device=device)
            predictions[first:last, j, 1] = model(tokens)[:, -1].cpu().double().numpy()
        if last % 32 == 0 or last == 512:
            temporary = output/'partial.tmp'
            with temporary.open('wb') as f:
                np.savez_compressed(f, prediction=predictions, completed_series=last)
            temporary.replace(partial)
            print(json.dumps({'event':'long_evaluation_progress','seed':plan['seed'],
                'family':plan['family'],'completed_series':last,'elapsed_seconds':time.monotonic()-started}), flush=True)
    synchronize(device)
    assert np.isfinite(predictions).all()
    identity_error = float(np.max(np.abs(predictions[:,0,0]-predictions[:,0,1])))
    np.testing.assert_allclose(predictions[:,0,0], predictions[:,0,1], rtol=0, atol=3e-5)
    verification = {'full_equals_window_at_2048_max_abs_error':identity_error,
                    'confirmation_set_opened':False, 'checkpoint_step':saved['completed_steps']}
    if old is not None:
        np.testing.assert_allclose(predictions[:,:4,0], old, rtol=0, atol=3e-5)
        verification['old_full_history_max_abs_error'] = float(np.max(np.abs(predictions[:,:4,0]-old)))
        verification['original_data_prefix_bit_identical'] = True
    np.savez_compressed(output/'predictions.npz', prediction=predictions, origins=ORIGINS, policies=POLICIES)
    rows = {p:score(predictions[:,:,k], data) for k,p in enumerate(POLICIES)}
    write_json(output/'metrics.json', rows)
    write_json(output/'verification.json', verification)
    result = {'status':'finished','elapsed_seconds':time.monotonic()-started,
        'inference_seconds':time.monotonic()-inference_start, 'family':plan['family'],
        'seed':plan['seed'], 'output':str(output), 'confirmation_set_opened':False,
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}
    write_json(output/'evaluation_result.json', result)
    return result


def preflight(config, parent_paths, output, device='cuda'):
    setup(device)
    started = time.monotonic(); checks = {}; benchmark = {}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(parent_paths['transformer'], device, parent_paths['transformer_sha256'])
    # Check real trained predictions, not just a randomly initialized network.
    d = sample_streams(930000003, 1, 8192)
    with torch.no_grad():
        x = torch.as_tensor(np.stack((d['values'][:,:-2],d['values'][:,1:-1]),-1), dtype=torch.float32, device=device)
        positions = [63,255,2047,4095,8191]
        full = model(x)[:,positions]
        bounded = transformer_selected(model,x,positions)
        err = float((full-bounded).abs().max())
        torch.testing.assert_close(full,bounded,rtol=0,atol=3e-5)
        changed=x.clone();changed[:,4096:]+=3
        causal=float((transformer_selected(model,changed,positions)[:,:4]-bounded[:,:4]).abs().max())
        assert causal < 3e-5
        checks.update(transformer_exact_attention_max_abs_error=err, transformer_causal_error=causal)
    del full,bounded,x,changed,model
    torch.cuda.empty_cache()
    # Benchmark actual training batches, including sampler and transfers.
    torch.manual_seed(config['training']['seed'])
    model=Forecaster(config['model']).to(device).float()
    opt=make_optimizer(model,config)
    for length in [64,2048]:
        timings=[]
        for step in range(16):
            synchronize(device);before=time.monotonic()
            tokens,targets=make_batch(config,step,length,device)
            opt.zero_grad(set_to_none=True)
            loss=(model(tokens)-targets).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            opt.step();synchronize(device)
            assert torch.isfinite(loss)
            timings.append(time.monotonic()-before)
        benchmark[f'train_{length}_seconds_per_step']=float(np.median(timings[4:]))
    del model,opt,tokens,targets,loss
    torch.cuda.empty_cache()
    # Test both complete long-history computation paths before permitting jobs.
    for family,key,count in [('transformer_nope','transformer',2),('gated_deltanet','gdn',32)]:
        model,_=load_model(parent_paths[key],device,parent_paths[key+'_sha256'])
        d=sample_streams(930000005,count,65536)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            synchronize(device);before=time.monotonic()
            if family=='transformer_nope':
                for b in range(count):
                    x=torch.as_tensor(np.stack((d['values'][b:b+1,:-2],d['values'][b:b+1,1:-1]),-1),dtype=torch.float32,device=device)
                    p=transformer_selected(model,x,[t-1 for t in ORIGINS])
                    assert torch.isfinite(p).all()
            else:
                p=predict(model,{'values':d['values'],'lengths':np.array(ORIGINS)},device,batch_size=count)
                assert np.isfinite(p).all()
            synchronize(device)
            benchmark[family+'_full64k_seconds_per_series']=(time.monotonic()-before)/count
            benchmark[family+'_peak_device_bytes']=torch.cuda.max_memory_allocated()
            synchronize(device);before=time.monotonic()
            for t in ORIGINS:
                obs=d['values'][:,t-2048:t+1]
                x=torch.as_tensor(np.stack((obs[:,:-1],obs[:,1:]),-1),dtype=torch.float32,device=device)
                p=model(x)[:,-1]
                assert torch.isfinite(p).all()
            synchronize(device)
            benchmark[family+'_six_windows_seconds_per_series']=(time.monotonic()-before)/count
        del model,d,p,x
        torch.cuda.empty_cache()
    # Guard uses measured timings with 1.5x headroom plus serialization/validation allowance.
    train_estimate=15000*(benchmark['train_64_seconds_per_step']+benchmark['train_2048_seconds_per_step'])
    t_eval=512*(benchmark['transformer_nope_full64k_seconds_per_series']+benchmark['transformer_nope_six_windows_seconds_per_series'])
    g_eval=512*(benchmark['gated_deltanet_full64k_seconds_per_series']+benchmark['gated_deltanet_six_windows_seconds_per_series'])
    padded_t=1.5*(train_estimate+t_eval)+180
    padded_g=1.5*g_eval+60
    elapsed=time.monotonic()-started
    rate=.000583+2*.0000131+8*.00000222
    expected=(3*padded_t+3*padded_g+elapsed)*rate
    result={'status':'passed','checks':checks,'benchmark':benchmark,
        'transformer_train_plus_eval_seconds_with_headroom':padded_t,
        'gdn_eval_seconds_with_headroom':padded_g,
        'projected_running_compute_usd':expected,'rate_usd_per_second':rate,
        'budget_gate_passed':expected<=15 and padded_t<=5100 and padded_g<=1500 and 1.5*train_estimate+180<=4140,
        'gpu':torch.cuda.get_device_name(), 'elapsed_seconds':elapsed,
        'confirmation_set_opened':False}
    write_json(output/'preflight.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result
