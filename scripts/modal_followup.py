"""Budgeted follow-up: parallel workers, one serialized budget coordinator."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import modal
volume=modal.Volume.from_name('project182-artifacts',create_if_missing=True,version=2)
image=(modal.Image.debian_slim(python_version='3.12')
       .apt_install('git','build-essential')
       .pip_install('torch==2.10.0',index_url='https://download.pytorch.org/whl/cu128')
       .pip_install('transformers==5.6.2','flash-linear-attention==0.6.0','numpy==2.2.6',
                    'scipy==1.16.3','scikit-learn==1.7.2','xgboost==3.1.1',
                    'wandb==0.31.1','PyYAML==6.0.3','munch==4.0.0',
                    'pandas==2.3.3','matplotlib==3.10.7')
       .env({'PYTHONPATH':'/root/project/src','TOKENIZERS_PARALLELISM':'false',
             'FLA_CONV_BACKEND':'triton','OMP_NUM_THREADS':'2',
             'TRITON_CACHE_DIR':'/artifacts/triton-cache'})
       .add_local_dir(ROOT/'src',remote_path='/root/project/src',ignore=['__pycache__']))
app=modal.App('project182-followup')
FAMILIES=('gpt2','linear_attention','mamba2','gated_deltanet')
BASE=Path('/artifacts/followup_v1')


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=480,
              max_containers=4,scaledown_window=2,retries=0,
              volumes={'/artifacts':volume},
              secrets=[modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])])
def worker(spec:dict):
    import time,traceback
    from pilot_runtime import run_training,compact,json_save
    # A reused container may still own W&B service log descriptors.
    import wandb
    wandb.teardown()
    volume.reload()
    start=time.monotonic()
    record={'spec':spec,'status':'failed'}
    try:
        if spec['action']=='train':
            result=run_training(spec['family'],.0003,spec['steps'],spec['run_id'],440,
                                seed=spec['seed'],followup=True)
            record.update(status='finished',result=compact(result))
        elif spec['action']=='evaluate':
            from followup_runtime import evaluate
            record.update(status='finished',result=evaluate(spec))
        else:raise ValueError(spec['action'])
    except Exception as exc:
        traceback.print_exc();record['error']=type(exc).__name__+': '+str(exc)
    finally:
        record['charged_seconds']=time.monotonic()-start
        json_save(BASE/'jobs'/f"{spec['job_id']}.json",record)
        volume.commit()
    return record


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=120,
              max_containers=4,scaledown_window=2,retries=0,
              volumes={'/artifacts':volume},
              secrets=[modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])])
def evaluation_worker(spec:dict):
    return worker.local(spec)


@app.function(image=image,cpu=1,memory=2048,timeout=1200,max_containers=1,retries=0,
              volumes={'/artifacts':volume})
def batch(stage:str,steps:int=10000,prefix:str=''):
    import time
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from pilot_runtime import json_save
    volume.reload();BASE.mkdir(parents=True,exist_ok=True)
    ledger_file=BASE/'compute_ledger.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    if any(r['status']=='reserved' for r in ledger):
        raise RuntimeError('Another batch is active or has an unresolved timeout reservation')
    manifest_file=BASE/'manifest.json'
    manifest=json.loads(manifest_file.read_text()) if manifest_file.exists() else {
        'phase':'followup-v1','learning_rate':.0003,'seeds':[182,183,184],
        'data_seeds':[1000000,2000000,3000000],
        'confirmation_seed':300000000,'confirmation_prompts':1024,
        'budget_usd':5,'compute_rate_guard_usd_per_hour':3,
        'overhead_reserve_usd':.5,'runs':{},'stages':[]}
    specs=[]
    if stage in ('extend','replicate'):
        if stage=='replicate' and 'chosen_horizon' not in manifest:
            # The operator only launches this stage after reviewing first-seed validation.
            manifest['chosen_horizon']=steps
            manifest['horizon_selection']='First-seed validation only; before confirmation evaluation'
        if stage=='replicate' and steps!=manifest['chosen_horizon']:
            raise ValueError('Replication horizon already fixed')
        date_prefix=prefix or 'alex_'+datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')
        for seed in ([182] if stage=='extend' else [183,184]):
            for family in FAMILIES:
                key=f'{family}:{seed}'
                name=manifest['runs'].get(key)
                if name is None:
                    launch_prefix='alex_261119' if seed==182 else date_prefix
                    name=f'{launch_prefix}_{family}_linear_d5_lr0.0003_s{seed}'
                    manifest['runs'][key]=name
                result_file=Path('/artifacts/pilot_v1/runs')/name/'result.json'
                if result_file.exists() and json.loads(result_file.read_text())['completed_steps']>=steps:
                    continue
                specs.append({'action':'train','family':family,'seed':seed,'steps':steps,'run_id':name})
    elif stage=='evaluate':
        horizon=manifest['chosen_horizon']
        if set(manifest['runs'])!={f'{f}:{s}' for f in FAMILIES for s in (182,183,184)}:
            raise ValueError('All 12 runs must be registered before confirmation evaluation')
        # Freeze every run at the chosen horizon before opening confirmation data.
        for key,name in manifest['runs'].items():
            r=json.loads((Path('/artifacts/pilot_v1/runs')/name/'result.json').read_text())
            if r['completed_steps']!=horizon:raise ValueError(f'Incomplete training: {name}')
        for key,name in manifest['runs'].items():
            family,seed=key.split(':');seed=int(seed)
            result_file=BASE/'evaluations'/f'{name}.json'
            if result_file.exists():continue
            specs.append({'action':'evaluate','family':family,'seed':seed,'steps':horizon,'run_id':name})
        # Identical deterministic confirmation data is precomputed once by CPU coordinator.
        from pilot_eval import fixed_dataset
        fixed_dataset(BASE/'data/confirmation.npz',300000000,1024,33,5,.5)
    else:raise ValueError(stage)
    # One writer reserves the full hard timeout of every worker, including queued work.
    # All workers commit unique run directories; only this coordinator updates the ledger.
    reserve=120 if stage=='evaluate' else 480
    used=sum(r['charged_seconds'] for r in ledger)
    affordable=int((5400-used)//reserve)
    if specs and affordable<1:
        raise RuntimeError(f'Additional $5 cap: {used:.1f}s used; cannot reserve another {reserve}s worker')
    deferred=max(0,len(specs)-affordable)
    specs=specs[:affordable]
    batch_id=f'{stage}_{int(time.time())}'
    for i,spec in enumerate(specs):
        spec['job_id']=f'{batch_id}_{i}'
        ledger.append({'job_id':spec['job_id'],'status':'reserved','charged_seconds':reserve,'spec':spec})
    manifest['stages'].append({'stage':stage,'steps':steps,'batch_id':batch_id,'jobs':len(specs)})
    json_save(manifest_file,manifest);json_save(ledger_file,ledger);volume.commit()
    target=evaluation_worker if stage=='evaluate' else worker
    handles=[target.spawn(spec) for spec in specs]
    completed=[]
    for spec,handle in zip(specs,handles):
        try:result=handle.get()
        except Exception as exc:
            result={'status':'timeout_or_error','charged_seconds':reserve,'error':str(exc),'spec':spec}
        row=next(r for r in ledger if r['job_id']==spec['job_id'])
        row.update({k:result[k] for k in ('status','charged_seconds')})
        if 'error' in result:row['error']=result['error']
        completed.append(result)
        volume.reload();json_save(ledger_file,ledger);volume.commit()
        print(json.dumps(result),flush=True)
    total=sum(r['charged_seconds'] for r in ledger)
    return {'stage':stage,'results':completed,'deferred_jobs':deferred,'total_charged_seconds':total,
            'conservative_compute_usd':total/1200,'with_overhead_reserve_usd':total/1200+.5}


@app.function(image=image,cpu=2,memory=4096,timeout=240,retries=0,volumes={'/artifacts':volume})
def report():
    from followup_report import generate
    volume.reload();result=generate();volume.commit();return result


@app.local_entrypoint()
def main(stage:str='extend',steps:int=10000,prefix:str=''):
    result=report.remote() if stage=='report' else batch.remote(stage,steps,prefix)
    print(json.dumps(result,indent=2))
