"""Reference training and matched adaptation; paired history evaluation, no confirmation set."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

from forecast_data import sample_streams, transition_tokens, references, posterior_moments
from forecast_reference import ReferenceForecaster


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def window_reference(values, origins, window, bound=.95, noise_std=.5, nodes=1024):
    """Bayes conditioned on the visible window AND its first observed value.

    For start>0 that first value has theta-dependent variance, since the original
    stream began at zero. Including this density avoids calling conditional
    transition-only regression the exact window posterior.
    """
    grid, weight = np.polynomial.legendre.leggauss(nodes)
    grid *= bound
    output = np.empty((len(values), len(origins)))
    risk = np.empty_like(output)
    for j, end in enumerate(origins):
        start = max(0, end-window)
        x, y = values[:, start:end], values[:, start+1:end+1]
        sxx, sxy = (x*x).sum(1), (x*y).sum(1)
        if start == 0:
            mean, var = posterior_moments(sxx, sxy, bound, noise_std)
        else:
            # Recenter at the likelihood maximum to avoid large cancellations.
            loc = sxy/sxx
            logp = -.5*sxx[:, None]*(grid[None, :]-loc[:, None])**2/noise_std**2
            start_var = noise_std**2*(1-grid**(2*start))/(1-grid**2)
            logp -= .5*(np.log(start_var)[None, :] + values[:, start, None]**2/start_var[None, :])
            logp += np.log(weight)[None, :]
            p = np.exp(logp-logsumexp(logp, axis=1, keepdims=True))
            mean = p@grid
            var = (p*(grid[None, :]-mean[:, None])**2).sum(1)
        output[:, j] = values[:, end]*mean
        risk[:, j] = values[:, end]**2*var
    return output, risk


def validation_data(config):
    ev, d = config['evaluation'], config['data']
    data = sample_streams(ev['seed'], ev['series'], max(ev['origins']), d['bound'], d['noise_std'])
    vals = data['values']
    data.update(references(vals, ev['origins'], d['bound'], d['noise_std']))
    data['latent_mean'] = data['theta'][:, None]*vals[:, ev['origins']]
    data['target'] = vals[:, np.array(ev['origins'])+1]
    data['window_bayes_mean'], data['window_bayes_epistemic_risk'] = window_reference(vals, ev['origins'], ev['window'])
    coarse, _ = window_reference(vals, ev['origins'], ev['window'], nodes=512)
    data['window_quadrature_max_abs'] = np.array(np.max(np.abs(coarse-data['window_bayes_mean'])))
    if data['window_quadrature_max_abs'] > 1e-7:
        raise AssertionError('Window posterior quadrature has not converged')
    return data


def tokens_at(values, start, end, device):
    return torch.as_tensor(np.stack((values[:, start:end], values[:, start+1:end+1]), -1), dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, data, config, output, step, deadline, persist):
    model.eval()
    ev = config['evaluation']; origins = ev['origins']; n = len(data['values'])
    pred = np.full((n, len(origins), 2), np.nan)
    rows = []
    for j, end in enumerate(origins):
        if time.monotonic() > deadline:
            return {'status':'partial_time_limit', 'completed_origins': origins[:j], 'rows':rows}
        for i in range(n):
            if i % 64 == 0 and time.monotonic() > deadline:
                np.savez_compressed(output/f'partial_predictions_step_{step}.npz', prediction=pred, origins=origins)
                return {'status':'partial_time_limit', 'completed_origins': origins[:j], 'rows':rows}
            vals = data['values'][i:i+1]
            pred[i,j,0] = model(tokens_at(vals,0,end,'cuda'))[0,-1].double().cpu().item()
            if end <= ev['window']:
                # Identical inputs, same numerical execution, no artificial BF16 discrepancy.
                pred[i,j,1] = pred[i,j,0]
            else:
                pred[i,j,1] = model(tokens_at(vals,end-ev['window'],end,'cuda'))[0,-1].double().cpu().item()
        if not np.isfinite(pred[:,j]).all():
            raise FloatingPointError('Nonfinite validation prediction')
        err = (pred[:,j]-data['latent_mean'][:,j,None])**2
        gap = (pred[:,j]-data['bayes_mean'][:,j,None])**2
        diff = err[:,1]-err[:,0]
        row = dict(step=step, origin=end, full_error=float(err[:,0].mean()),
                   window_error=float(err[:,1].mean()), window_minus_full=float(diff.mean()),
                   paired_series_se=float(diff.std(ddof=1)/np.sqrt(n)),
                   full_excess_vs_full_bayes=float(gap[:,0].mean()),
                   window_excess_vs_full_bayes=float(gap[:,1].mean()),
                   full_bayes_error=float(((data['bayes_mean'][:,j]-data['latent_mean'][:,j])**2).mean()),
                   window_bayes_error=float(((data['window_bayes_mean'][:,j]-data['latent_mean'][:,j])**2).mean()))
        rows.append(row)
        np.savez_compressed(output/f'predictions_step_{step}.npz', prediction=pred, origins=origins, policies=ev['policies'])
        save(output/f'metrics_step_{step}.json', {'status':'running', 'rows':rows})
        persist()
        emit('evaluation_origin', run_name=output.name, **row)
    result = {'status':'finished', 'rows':rows, 'full_equals_window_at_2048':bool(np.array_equal(pred[:,0,0],pred[:,0,1]))}
    save(output/f'metrics_step_{step}.json',result)
    return result


@torch.no_grad()
def precision_audit(model, data, config, output, deadline):
    """Limited trained-checkpoint diagnostic, not a bound on 64k precision error."""
    ev=config['evaluation']; result={'scope':'4 series at 2k and 16k; not a 64k precision guarantee', 'rows':[]}
    old = model.config['precision']
    try:
        for end in ev['precision_audit_origins']:
            if time.monotonic() > deadline:
                result['status']='partial_time_limit'; break
            bf=[]; fp=[]
            for i in range(ev['precision_audit_series']):
                x=tokens_at(data['values'][i:i+1],0,end,'cuda')
                model.config['precision']='bfloat16'; bf.append(model(x)[0,-1].float().cpu().item())
                model.config['precision']='float32'; fp.append(model(x)[0,-1].float().cpu().item())
            a,b=np.array(bf),np.array(fp)
            result['rows'].append({'origin':end,'prediction_mse':float(((a-b)**2).mean()),
                                   'max_abs':float(np.abs(a-b).max()),'bf16':a.tolist(),'fp32':b.tolist()})
            save(output/'precision_audit.json',result)
        else: result['status']='finished'
    except (RuntimeError, AssertionError) as exc:
        result.update(status='failed',error=str(exc)[:1500])
    finally: model.config['precision']=old
    save(output/'precision_audit.json',result)
    return result


def scheduled_lr(training, step):
    peak = training['learning_rate']
    if training['schedule'] == 'constant':
        return peak
    if training['schedule'] != 'warmup_cosine':
        raise ValueError('Unsupported training schedule')
    warmup, total = training['warmup_steps'], training['steps']
    minimum = training['min_learning_rate']
    assert 1 < warmup < total-1 and 0 < minimum <= peak
    assert 0 <= step < total
    if step < warmup:
        start=training.get('warmup_start_learning_rate',peak*.1)
        return start+(peak-start)*step/(warmup-1)
    fraction = (step-warmup)/(total-1-warmup)
    return minimum+(peak-minimum)*.5*(1+math.cos(math.pi*fraction))


@torch.no_grad()
def quick_validation(model, data, origin):
    model.eval()
    values, theta = data['values'], data['theta']
    predictions = []
    for i in range(len(values)):
        predictions.append(model(tokens_at(values[i:i+1],0,origin,'cuda'))[0,-1].float().cpu().item())
    predictions = np.asarray(predictions)
    if not np.isfinite(predictions).all():
        raise FloatingPointError('Nonfinite quick-validation prediction')
    error = (predictions-theta*values[:,origin])**2
    model.train()
    return {'origin':origin,'series':len(values),'conditional_mean_mse':float(error.mean()),
            'series_se':float(error.std(ddof=1)/np.sqrt(len(error)))}


def run(config, output, persist=lambda:None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); deadline=started+config['modal']['work_soft_limit_seconds']
    torch.set_num_threads(config['modal']['cpu'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    tr=config['training']; torch.manual_seed(tr['seed'])
    model=ReferenceForecaster(config['model']).cuda().float()
    optimizer=torch.optim.Adam(model.parameters(),lr=tr['learning_rate'],betas=tuple(tr['betas']),eps=tr['eps'],weight_decay=tr['weight_decay'])
    parent_completed=0
    if config.get('resume_from'):
        parent=Path(config['resume_from'])
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==config['resume_sha256']
        saved=torch.load(parent,map_location='cpu',weights_only=False)
        assert saved['completed_steps']==40000 and saved['config']['model']==config['model']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        parent_completed=saved['completed_steps'];del saved
    state=dict(status='starting',run_name=output.name,completed_steps=0,parent_completed_steps=parent_completed,config=config,
               model=model.description(),gpu=torch.cuda.get_device_name(),
               versions={n:importlib.metadata.version(n) for n in ['torch','transformers','fla-core','triton']},
               triton_f32_default=os.environ.get('TRITON_F32_DEFAULT'),confirmation_set_opened=False)
    save(output/'manifest.json',state)
    save(output/'resolved_backbone.json',model.backbone.config.to_dict())
    for filename in config['source_sha256']:
        (output/filename).write_bytes(Path(__file__).with_name(filename).read_bytes())
    persist()
    wb=None
    try:
        import wandb
        # Explicit entity and personal secret; never inherit an ambient entity.
        user=wandb.Api(timeout=15).viewer.username
        if user != config['wandb']['username']:
            raise PermissionError('W&B account does not match the personal workspace')
        wb=wandb.init(entity=config['wandb']['entity'],project=config['wandb']['project'],
                      name=output.name,id=output.name,config=config,dir='/tmp',
                      tags=[f"forecast-reference-{config['model']['layers']}x{config['model']['width']}",config['arm'],config['model']['family']],
                      settings=wandb.Settings(init_timeout=30,disable_git=True,save_code=False))
        state['wandb_url']=wb.url
    except PermissionError: raise
    except Exception as exc:
        state['wandb_warning']=type(exc).__name__+'; local metrics remain authoritative'
        emit('wandb_unavailable',run_name=output.name,error_type=type(exc).__name__)
    quick_data=None
    if config['evaluation'].get('during_training_every'):
        ev=config['evaluation']; quick_data=sample_streams(ev['seed'],ev['series'],2048,config['data']['bound'],config['data']['noise_std'])
    completed=0; history=[]; training_start=time.monotonic()
    def checkpoint(step):
        path=output/f'state_step_{step}.pt'; tmp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_steps=step,total_optimizer_steps=parent_completed+step,config=config),tmp)
        tmp.replace(path)
        state.setdefault('checkpoints',{})[str(step)]=hashlib.sha256(path.read_bytes()).hexdigest()
        save(output/'training.json',history)
        save(output/'manifest.json',state); persist()
    try:
        model.train(); state['status']='training'
        for step in range(tr['steps']):
            if time.monotonic()-started > config['modal']['train_soft_limit_seconds']:
                break
            lr=scheduled_lr(tr,step)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            kind=config.get('kind','fresh')
            if kind in ('long_adaptation','reset_adaptation'):
                from forecast_adaptation import sample_long,adaptation_backward
                length=65536
                data=sample_long(tr['effective_data_seed']+step,length,config['data']['bound'],config['data']['noise_std'])
                loss,x,y=adaptation_backward(model,data,kind,tr['microbatch_chunks'])
            else:
                length=tr['lengths'][step%len(tr['lengths'])]
                data=sample_streams(tr['effective_data_seed']+step,tr['tokens_per_step']//length,length,config['data']['bound'],config['data']['noise_std'])
                x=torch.as_tensor(transition_tokens(data['values']),dtype=torch.float32,device='cuda')
                y=torch.as_tensor(data['values'][:,2:],dtype=torch.float32,device='cuda')
                loss=(model(x)-y).square().mean(); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),tr['grad_clip'],error_if_nonfinite=True)
            optimizer.step(); completed=step+1
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            if completed==1 or completed%tr['log_every']==0:
                row=dict(step=completed,loss=float(loss.detach()),grad_norm=float(grad),length=length,learning_rate=lr,
                         elapsed_seconds=time.monotonic()-training_start)
                history.append(row); state.update(completed_steps=completed,training_elapsed_seconds=row['elapsed_seconds'])
                if wb: wb.log({'train/'+k:v for k,v in row.items()},step=completed)
                if completed==1 or completed%500==0:
                    save(output/'training.json',history); save(output/'manifest.json',state)
                    persist(); emit('training_progress',run_name=output.name,**row)
            if completed in tr['checkpoint_steps']: checkpoint(completed)
            if quick_data is not None and completed%config['evaluation']['during_training_every']==0:
                q=quick_validation(model,quick_data,2048);q['step']=completed;q['learning_rate']=lr
                state.setdefault('quick_validation',[]).append(q)
                save(output/'quick_validation.json',state['quick_validation']);save(output/'manifest.json',state);persist()
                emit('quick_validation',run_name=output.name,**q)
                if wb: wb.log({'validation_2k/'+k:v for k,v in q.items()},step=completed)
        state.update(completed_steps=completed,training_elapsed_seconds=time.monotonic()-training_start)
        if str(completed) not in state.get('checkpoints',{}): checkpoint(completed)
        # Evaluate final checkpoint first so the primary result survives the time cap.
        del optimizer, x, y, loss, grad
        torch.cuda.empty_cache(); state['status']='evaluating'; save(output/'manifest.json',state); persist()
        data=validation_data(config)
        np.savez_compressed(output/'reference.npz',**{k:v for k,v in data.items() if k!='values'})
        state['evaluations']={}
        for step in sorted(map(int,state['checkpoints']),reverse=True):
            if time.monotonic()>deadline: break
            saved=torch.load(output/f'state_step_{step}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']); del saved
            state['evaluations'][str(step)]=evaluate(model,data,config,output,step,deadline,persist)
            save(output/'manifest.json',state);persist()
        state['status']='finished' if completed==tr['steps'] and all(state['evaluations'].get(str(s),{}).get('status')=='finished' for s in tr['checkpoint_steps']) else 'partial_time_limit'
        # Only two primary-seed workers compile the precision diagnostics.
        if config['evaluation'].get('precision_audit_enabled',True) and config['arm']==config['evaluation'].get('precision_audit_arm','main') and tr['seed']==config['evaluation']['precision_audit_seed'] and time.monotonic()<deadline-180:
            saved=torch.load(output/f'state_step_{completed}.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);del saved
            state['precision_audit']=precision_audit(model,data,config,output,deadline)
        return state
    except Exception as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc)[:1800],completed_steps=completed)
        raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started
        save(output/'manifest.json',state); save(output/'training.json',history); persist()
        if wb: wb.finish()
