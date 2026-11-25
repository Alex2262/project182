"""Matched long-context adaptation data and frozen-checkpoint state interventions."""
import hashlib
import time
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
import torch
from forecast_data import sample_streams, transition_tokens
from forecast_reference import ReferenceForecaster
from forecast_reference_train import save, emit, tokens_at


def sample_long(seed, length=65536, bound=.95, noise_std=.5):
    parameter_seed, innovation_seed = np.random.SeedSequence(seed).spawn(2)
    theta = np.random.default_rng(parameter_seed).uniform(-bound,bound,1)
    noise = np.random.default_rng(innovation_seed).normal(scale=noise_std,size=(length+1,1))
    values = np.zeros((1,length+2),dtype=np.float64)
    values[0,1:] = lfilter([1.],[1.,-float(theta[0])],noise[:,0])
    return {'values':values,'theta':theta}


def adaptation_backward(model, data, kind, microbatch=4, device='cuda'):
    x=torch.as_tensor(transition_tokens(data['values']),dtype=torch.float32,device=device)
    y=torch.as_tensor(data['values'][:,2:],dtype=torch.float32,device=device)
    if kind=='long_adaptation':
        loss=(model(x)-y).square().mean();loss.backward()
        return loss.detach(),x,y
    if kind!='reset_adaptation':raise ValueError(kind)
    x=x.reshape(-1,2048,2);y=y.reshape(-1,2048)
    loss=torch.zeros((),device=device)
    for i in range(0,len(x),microbatch):
        part=(model(x[i:i+microbatch])-y[i:i+microbatch]).square().sum()/y.numel()
        part.backward();loss+=part.detach()
    return loss,x,y


def tree_clone(value,zero=False):
    if isinstance(value,torch.Tensor):return torch.zeros_like(value) if zero else value.detach().clone()
    if isinstance(value,dict):return {k:tree_clone(v,zero) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(tree_clone(v,zero) for v in value)
    if isinstance(value,list):return [tree_clone(v,zero) for v in value]
    return value


def intervene_states(states,condition):
    out=tree_clone(states)
    for layer,entry in enumerate(out):
        if condition in ('all_recurrent','all_states') or condition==f'layer_{layer+1}':
            entry['recurrent_state']=tree_clone(entry['recurrent_state'],zero=True)
        if condition in ('conv_only','all_states'):
            entry['conv_state']=tree_clone(entry['conv_state'],zero=True)
    return out


@torch.no_grad()
def run_ablation(config,output,persist=lambda:None):
    from fla.models.cache_utils import Cache
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+config['modal']['work_soft_limit_seconds']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    parent=Path(config['resume_from']);assert hashlib.sha256(parent.read_bytes()).hexdigest()==config['resume_sha256']
    saved=torch.load(parent,map_location='cpu',weights_only=False)
    assert saved['completed_steps']==40000 and saved['config']['model']==config['model']
    model=ReferenceForecaster(config['model']).cuda().float();model.load_state_dict(saved['model']);model.eval();del saved
    for f in config['source_sha256']:(output/f).write_bytes(Path(__file__).with_name(f).read_bytes())
    state={'status':'evaluating','run_name':output.name,'config':config,'completed_steps':0,'model':model.description(),'source_checkpoint_sha256':config['resume_sha256']}
    save(output/'manifest.json',state);persist()
    ev=config['evaluation'];origins=ev['origins'];n=ev['series']
    # Generate the full 512-series cohort before selecting first 64: count affects the RNG layout.
    data=sample_streams(ev['seed'],ev['cohort_size'],max(origins),config['data']['bound'],config['data']['noise_std'])
    vals=data['values'][:n];theta=data['theta'][:n];latent=theta[:,None]*vals[:,origins]
    conditions=['carry']+[f'layer_{i}' for i in range(1,5)]+['all_recurrent','conv_only','all_states']
    pred=np.full((n,len(origins),len(conditions)),np.nan);fresh=np.full((n,len(origins)),np.nan)
    norms=[];rows=[]
    try:
        for j,end in enumerate(origins):
            for i in range(n):
                if time.monotonic()>deadline:raise TimeoutError('Ablation deadline reached')
                v=vals[i:i+1];split=end-2048
                _,cache=model(tokens_at(v,0,split,'cuda'),return_state=True)
                original=tree_clone(cache.to_legacy_cache());seen=cache.get_seq_length();del cache
                norms.append({'series':i,'origin':end,'recurrent_norms':[float(s['recurrent_state'].float().norm()) for s in original]})
                suffix=tokens_at(v,split,end,'cuda')
                for k,condition in enumerate(conditions):
                    modified=Cache.from_legacy_cache(intervene_states(original,condition),seen_tokens=seen)
                    y,_=model(suffix,states=modified,return_state=True)
                    pred[i,j,k]=y[0,-1].double().item();del modified,y
                fresh[i,j]=model(suffix)[0,-1].double().item()
                if (i+1)%16==0:emit('ablation_progress',run_name=output.name,origin=end,completed_series=i+1)
            err=(pred[:,j]-latent[:,j,None])**2
            if not np.isfinite(pred[:,j]).all():raise FloatingPointError('Nonfinite ablation predictions')
            for k,condition in enumerate(conditions):
                delta=err[:,k]-err[:,0]
                row={'origin':end,'condition':condition,'conditional_mean_mse':float(err[:,k].mean()),'delta_vs_carry':float(delta.mean()),'paired_series_se':float(delta.std(ddof=1)/np.sqrt(n))}
                rows.append(row);emit('ablation_metric',run_name=output.name,**row)
            np.savez_compressed(output/'ablation_predictions.npz',prediction=pred,fresh=fresh,latent_mean=latent,theta=theta,origins=origins,conditions=conditions)
            save(output/'ablation_metrics.json',rows);save(output/'state_norms.json',norms);persist()
        state['status']='finished';state['rows']=rows
        state['zero_state_vs_fresh_prediction_mse']=float(np.mean((pred[:,:,-1]-fresh)**2))
        return state
    except Exception as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        state['elapsed_seconds']=time.monotonic()-started;save(output/'manifest.json',state);persist()
