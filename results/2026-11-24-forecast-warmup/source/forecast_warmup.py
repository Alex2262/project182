"""Matched carried-state versus reset continuations; no test-time weight updates."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from forecast_data import sample_streams,transition_tokens
from forecast_long import load_model,evaluate_long
from forecast_runtime import setup,synchronize,make_optimizer,write_json


def batch(config,step,device):
    w=config['warmup_protocol']
    data=sample_streams(w['effective_data_seed']+step,1,w['warmup_length']+w['scored_length'],.95,.5)
    tokens=torch.as_tensor(transition_tokens(data['values']),dtype=torch.float32,device=device)
    targets=torch.as_tensor(data['values'][:,w['warmup_length']+2:],dtype=torch.float32,device=device)
    return tokens,targets


def forward_training(model,tokens,warmup_length,arm):
    assert arm in ('carry','reset')
    # Execute identical prefix computation in both arms, using current weights.
    with torch.no_grad():
        _,states=model(tokens[:,:warmup_length],return_state=True)
    detached=[{k:v.detach() for k,v in layer.items()} for layer in states]
    assert all(not v.requires_grad and v.grad_fn is None for layer in detached for v in layer.values())
    return model(tokens[:,warmup_length:],detached if arm=='carry' else None)


def initialize(config,device):
    w=config['warmup_protocol']
    model,parent=load_model(w['parent_checkpoint'],device,w['parent_checkpoint_sha256'])
    assert parent['completed_steps']==30000
    assert parent['config']['training']['seed']==config['training']['seed']
    assert parent['config']['model']==config['model']
    for name in ['forecast_data.py','forecast_models.py']:
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==parent['config']['source_sha256'][name]
    optimizer=make_optimizer(model,config)
    optimizer.load_state_dict(parent['optimizer'])
    for group in optimizer.param_groups:group['lr']=config['training']['learning_rate']
    return model,optimizer


def correctness(model,device):
    model.train();torch.manual_seed(930000017)
    tokens=torch.randn(1,192,2,device=device,requires_grad=True)
    model.zero_grad(set_to_none=True)
    reset=forward_training(model,tokens,128,'reset')
    reset.square().mean().backward()
    reset_grads=[p.grad.detach().clone() for p in model.parameters()]
    assert torch.count_nonzero(tokens.grad[:,:128])==0
    model.zero_grad(set_to_none=True);tokens.grad=None
    direct=model(tokens[:,128:]);direct.square().mean().backward()
    torch.testing.assert_close(reset,direct,rtol=0,atol=3e-6)
    grad_error=max(float((a-p.grad).abs().max()) for a,p in zip(reset_grads,model.parameters()))
    for a,p in zip(reset_grads,model.parameters()):torch.testing.assert_close(a,p.grad,rtol=0,atol=3e-6)
    model.zero_grad(set_to_none=True);tokens.grad=None
    carry=forward_training(model,tokens,128,'carry')
    with torch.no_grad():continuous=model(tokens)[:,128:]
    torch.testing.assert_close(carry,continuous,rtol=0,atol=3e-6)
    carry.square().mean().backward()
    assert torch.count_nonzero(tokens.grad[:,:128])==0
    assert torch.count_nonzero(tokens.grad[:,128:])>0
    model.zero_grad(set_to_none=True)
    return {'reset_matches_fresh_forward':True,'reset_gradient_max_abs_error':grad_error,
        'carry_matches_continuous_max_abs_error':float((carry-continuous).detach().abs().max()),
        'prefix_input_gradient_zero_both_arms':True,'suffix_input_gradient_nonzero':True}


def preflight(configs,output):
    setup('cuda');started=time.monotonic();result={'checks':{},'seconds_per_update':{}}
    for arm in ['reset','carry']:
        config=next(c for c in configs if c['warmup_protocol']['arm']==arm and c['training']['seed']==182)
        model,opt=initialize(config,'cuda')
        result['checks'][arm]=correctness(model,'cuda');times=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(12):
            synchronize('cuda');before=time.monotonic()
            x,y=batch(config,step,'cuda');model.train();opt.zero_grad(set_to_none=True)
            prediction=forward_training(model,x,16384,arm)
            loss=(prediction-y).square().mean();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            assert torch.isfinite(loss);synchronize('cuda');times.append(time.monotonic()-before)
        result['seconds_per_update'][arm]=float(np.median(times[3:]))
        result.setdefault('peak_device_bytes',{})[arm]=torch.cuda.max_memory_allocated()
        del model,opt,x,y,prediction,loss;torch.cuda.empty_cache()
    rate=.00062696
    projected_per_job={a:1.5*t*3000+180 for a,t in result['seconds_per_update'].items()}
    elapsed=time.monotonic()-started
    result.update(status='passed',elapsed_seconds=elapsed,seconds_per_job_with_headroom=projected_per_job,
        projected_running_compute_usd=(3*sum(projected_per_job.values())+elapsed)*rate,
        rate_usd_per_second=rate,confirmation_set_opened=False)
    result['budget_gate_passed']=result['projected_running_compute_usd']<=15 and max(projected_per_job.values())<2940
    write_json(Path(output)/'preflight.json',result);print(json.dumps(result),flush=True)
    return result


def train_and_evaluate(config,output,max_seconds=3000):
    setup('cuda');output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();w=config['warmup_protocol'];arm=w['arm'];seed=config['training']['seed']
    cp=output/'config.yaml'
    if cp.exists():assert yaml.safe_load(cp.read_text())==config
    cp.write_text(yaml.safe_dump(config,sort_keys=False))
    model,opt=initialize(config,'cuda');completed=0;history=[]
    state_path=output/'state.pt'
    if state_path.exists():
        state=torch.load(state_path,map_location='cuda',weights_only=False)
        assert state['config']==config
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        completed=state['completed_steps']-30000
        history=json.loads((output/'training.json').read_text())
    import wandb
    log_dir=Path('/tmp/project182-warmup-wandb')/output.name;log_dir.mkdir(parents=True,exist_ok=True)
    run=wandb.init(entity='alexander-tian-ucb',project='project182',id=output.name,name=output.name,
        resume='allow',config=config,dir=str(log_dir),tags=['forecast-ar1','warmup',arm],
        settings=wandb.Settings(disable_git=True,save_code=False))
    def save():
        temp=output/'state.tmp'
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'completed_steps':30000+completed,
            'config':config},temp);temp.replace(state_path);write_json(output/'training.json',history)
    try:
        for step in range(completed,3000):
            if time.monotonic()-start>max_seconds-60:break
            x,y=batch(config,step,'cuda');model.train();opt.zero_grad(set_to_none=True)
            pred=forward_training(model,x,16384,arm);loss=(pred-y).square().mean();loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            assert torch.isfinite(loss);opt.step();completed=step+1
            if completed%25==0:
                row={'additional_step':completed,'completed_steps':30000+completed,'loss':float(loss.detach()),
                    'grad_norm_before_clip':float(grad),'elapsed_seconds':time.monotonic()-start}
                history.append(row);run.log(row)
                if completed%250==0:print(json.dumps({'event':'training_progress','arm':arm,'seed':seed,**row}),flush=True)
            if completed%100==0:save()
    finally:
        save();run.finish();wandb.teardown()
    training={'status':'finished' if completed==3000 else 'partial','additional_steps':completed,
        'completed_steps':30000+completed,'elapsed_seconds':time.monotonic()-start}
    write_json(output/'training_result.json',training)
    if completed!=3000:return training
    import shutil
    final=output/'state_step_33000.pt';shutil.copyfile(state_path,final)
    del model,opt,x,y,pred,loss;torch.cuda.empty_cache()
    plan={'family':'gated_deltanet','seed':seed,'arm':arm,'checkpoint_step':33000,
        'training_lengths':[2048],'training_schedule':'constant','checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest(),
        'origins':[2048,4096,8192,16384,32768,65536],'policies':['full','window2048'],
        'series':512,'evaluation_seed':920000000,'confirmation_set_opened':False}
    evaluation=evaluate_long(plan,final,output/'long_evaluation')
    result={'status':'finished','arm':arm,'seed':seed,'training':training,'evaluation':evaluation,
        'total_elapsed_seconds':time.monotonic()-start,'output':str(output)}
    write_json(output/'job_result.json',result);return result
