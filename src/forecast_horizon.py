"""Long-history sweep checks and frozen-checkpoint evaluation; no confirmation data."""
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from forecast_data import sample_streams
from forecast_models import Forecaster
from forecast_runtime import (evaluate, make_batch, make_optimizer, predict, setup,
                              synchronize, training_prediction, write_json)


def source_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['forecast_data.py','forecast_models.py','forecast_runtime.py','forecast_horizon.py']}


def horizon_preflight(config, output, device='cuda'):
    setup(device)
    started=time.monotonic()
    cases=[]
    for length,reset in [(64,0),(1024,0),(2048,0),(2048,256)]:
        torch.manual_seed(config['training']['seed'])
        model=Forecaster(config['model']).to(device).float()
        optimizer=make_optimizer(model,config)
        timings=[]
        for step in range(10):
            tokens,targets=make_batch(config,step,length,device)
            synchronize(device)
            before=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss=(training_prediction(model,tokens,reset)-targets).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            timings.append(time.monotonic()-before)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite preflight loss')
        with torch.no_grad():
            model.eval()
            base=training_prediction(model,tokens[:1],reset)
            changed=tokens[:1].clone()
            changed[:,length//2:]+=10
            causal_error=float((training_prediction(model,changed,reset)[:,:length//2]-base[:,:length//2]).abs().max())
            assert causal_error < 3e-5
            if reset:
                separate=torch.cat([model(tokens[:1,i:i+reset]) for i in range(0,length,reset)],dim=1)
                agreement=float((separate-base).abs().max())
            else:
                state,parts=None,[]
                for offset in range(0,length,37):
                    part,state=model(tokens[:1,offset:offset+37],state,return_state=True)
                    parts.append(part)
                agreement=float((torch.cat(parts,dim=1)-base).abs().max())
            assert agreement < 3e-5
        row={'length':length,'reset_every':reset,'batch_size':len(tokens),'steps':10,
             'median_seconds_per_step_after_warmup':float(np.median(timings[2:])),
             'causality_max_abs_error':causal_error,'streaming_or_reset_agreement_max_abs_error':agreement,
             'final_loss':float(loss.detach())}
        if length==2048 and reset==0:
            data=sample_streams(930000001,2,16384,config['data']['bound'],config['data']['noise_std'])
            data['lengths']=np.array(config['evaluation']['lengths'])
            synchronize(device)
            before=time.monotonic()
            predictions=predict(model,data,device,batch_size=2)
            synchronize(device)
            assert np.isfinite(predictions).all()
            row['two_series_16k_inference_seconds']=time.monotonic()-before
        cases.append(row)
        print(json.dumps({'event':'preflight_case',**row}),flush=True)
    result={'cases':cases,'elapsed_seconds':time.monotonic()-started,'resolved_config':config,
            'source_sha256':source_hashes()}
    write_json(Path(output)/'preflight.json',result)
    return result


def evaluate_saved(plan,parent,output,device='cuda'):
    setup(device)
    parent,output=Path(parent),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'evaluation_plan.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != plan:
        raise ValueError('Evaluation destination already has a different plan')
    write_json(manifest_path,plan)
    parent_config=yaml.safe_load((parent/'config.yaml').read_text())
    assert parent.name==plan['parent_run_name']
    assert parent_config['model']['family']=='gated_deltanet'
    for name in ['forecast_data.py','forecast_models.py']:
        assert parent_config['source_sha256'][name]==source_hashes()[name]
    config=copy.deepcopy(parent_config)
    config['evaluation']['lengths']=plan['evaluation_lengths']
    config['source_sha256']=source_hashes()
    config['evaluation_only']=True
    config['parent_run_name']=parent.name
    (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (output/'parent_config.yaml').write_text(yaml.safe_dump(parent_config,sort_keys=False))
    started=time.monotonic()
    for step in [2000,5000,10000]:
        checkpoint=parent/f'state_step_{step}.pt'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==plan['checkpoint_sha256'][str(step)]
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        assert saved['completed_steps']==step and saved['config']==parent_config
        model=Forecaster(config['model']).to(device).float()
        model.load_state_dict(saved['model'])
        evaluate(model,config,output,device,step)
        print(json.dumps({'event':'evaluation_progress','parent_run_name':parent.name,'step':step}),flush=True)
        del model,saved
    result={'status':'finished','evaluated_steps':[2000,5000,10000],
            'elapsed_seconds':time.monotonic()-started,'output':str(output),
            'parent_run_name':parent.name,'confirmation_set_opened':False}
    write_json(output/'evaluation_result.json',result)
    return result


def deterioration_point(lengths,errors):
    """Predeclared 25% rise, sustained at the next tested length; descriptive only."""
    lengths,errors=list(lengths),np.asarray(errors,dtype=np.float64)
    if len(lengths)!=len(errors) or len(errors)<2 or not np.isfinite(errors).all():
        raise ValueError('Invalid error curve')
    for i in range(1,len(errors)-1):
        threshold=1.25*float(errors[:i].min())
        if errors[i]>=threshold and errors[i+1]>=threshold:
            return {'first_length':int(lengths[i]),'confirmed_at_length':int(lengths[i+1]),
                    'earlier_minimum':float(errors[:i].min()),'threshold':threshold}
    return {'first_length':None,'confirmed_at_length':None,
            'unconfirmed_rise_at_final_length':bool(errors[-1]>=1.25*errors[:-1].min()),
            'tested_through':int(lengths[-1])}
