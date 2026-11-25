"""Approved ten-job 80k/adaptation/ablation batch; personal workspace only."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import copy
import hashlib
import json
import os
import time
import modal

ROOT=Path(__file__).resolve().parents[1]
PROFILE='alexander-tian'
SOURCES=['forecast_data.py','forecast_reference.py','forecast_reference_train.py','forecast_adaptation.py']

def personal():
    if modal.is_local():
        from modal import config
        assert os.environ.get('MODAL_PROFILE')==config._profile==PROFILE
        assert not any(os.environ.get(k) for k in ['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','MODAL_OAUTH_REFRESH_TOKEN','MODAL_OAUTH_CLIENT_ID','MODAL_OAUTH_CLIENT_SECRET','MODAL_OAUTH_JWT_KEY','MODAL_CONFIG_PATH','MODAL_SERVER_URL'])
personal()
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0',index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('transformers==5.6.2','flash-linear-attention==0.6.0','fla-core==0.6.0','numpy==2.2.6','scipy==1.16.3','PyYAML==6.0.3')
    .pip_install('triton==3.7.1')
    .pip_install('wandb==0.31.1')
    .env({'PYTHONPATH':'/root/project/src','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false',
          'FLA_CONV_BACKEND':'triton','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
          'TRITON_CACHE_DIR':'/tmp/reference-benchmark-triton','TRITON_F32_DEFAULT':'ieee'}))
for name in SOURCES:
    image=image.add_local_file(ROOT/'src'/name,remote_path=f'/root/project/src/{name}')
volume=modal.Volume.from_name('project182-artifacts',create_if_missing=False,version=2)
secret=modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])
app=modal.App('project182-reference-next')

@app.function(image=image,gpu='H100',cpu=4,memory=16384,timeout=3900,retries=0,
    max_containers=10,scaledown_window=2,volumes={'/artifacts':volume},secrets=[secret])
def experiment(config,run_name):
    import tarfile
    import torch
    from forecast_reference_train import run
    from forecast_adaptation import run_ablation
    volume.reload()
    for name,sha in config['source_sha256'].items():
        assert hashlib.sha256((Path('/root/project/src')/name).read_bytes()).hexdigest()==sha
    assert config['modal']['profile']==PROFILE
    key=torch.cuda.get_device_name().replace(' ','_')
    archive=Path('/artifacts/reference_compiler_cache')/f'{key}_triton371.tar.gz'
    if archive.exists():
        with tarfile.open(archive) as handle: handle.extractall('/tmp/reference-benchmark-triton',filter='data')
    runner=run_ablation if config['kind']=='layer_ablation' else run
    return runner(config,Path('/artifacts/forecast_reference_train')/run_name,persist=volume.commit)


def make_plan():
    import yaml
    import subprocess
    base=yaml.safe_load((ROOT/'src/conf/forecast_reference_40k.yaml').read_text())
    base['source_sha256']={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in SOURCES}
    spec=yaml.safe_load((ROOT/'src/conf/forecast_next.yaml').read_text())
    plans=[]
    stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('alex_%y%m%d_next_%H%M%S')
    for kind,family,seconds in [('fresh','gated_deltanet',3900),('fresh','transformer_nope',2400),
                                ('long_adaptation','gated_deltanet',3000),('reset_adaptation','gated_deltanet',3000),
                                ('layer_ablation','gated_deltanet',900)]:
        for seed in spec['seeds']:
            c=copy.deepcopy(base);c['kind']=kind;c['arm']=kind;c['model']['family']=family
            c['modal'].update(timeout_seconds=seconds,train_soft_limit_seconds=seconds-480,
                              work_soft_limit_seconds=seconds-60,budget_usd=35)
            c['evaluation'].update(precision_audit_enabled=False)
            c['training'].update(seed=seed,steps=80000,learning_rate=1e-4,min_learning_rate=1e-5,
                 effective_data_seed=base['training']['data_seed']+100000*(seed-182),
                 total_training_tokens=655360000,checkpoint_steps=[40000,60000,80000])
            if kind!='fresh':
                parent=spec['parent_batch']+f'_gdn_lr1e4_s{seed}'
                m=json.loads((ROOT/'outputs/forecast_reference_train_control'/spec['parent_batch']/'artifacts'/parent/'manifest.json').read_text())
                c['resume_from']=f'/artifacts/forecast_reference_train/{parent}/state_step_40000.pt'
                c['resume_sha256']=m['checkpoints']['40000']
                assert m['config']['model']==c['model']
                c['training'].update(steps=5000,tokens_per_step=65536,lengths=[65536],learning_rate=3e-5,
                    min_learning_rate=3e-6,warmup_steps=200,warmup_start_learning_rate=1e-5,
                    effective_data_seed=630000000+100000*(seed-185),checkpoint_steps=[1000,2500,5000],
                    total_training_tokens=327680000,lifetime_training_tokens=655360000,microbatch_chunks=4)
                c['evaluation']['during_training_every']=1000
            if kind=='layer_ablation':
                c['evaluation'].update(series=64,cohort_size=512,origins=[16384,65536])
                c['training'].update(steps=0,checkpoint_steps=[],total_training_tokens=0)
            short='gdn' if family=='gated_deltanet' else 'tf'
            c['run_name']=f'{stamp}_{short}_{kind}_s{seed}'
            plans.append(c)
    maximum=sum(c['modal']['timeout_seconds']+2 for c in plans)*(.001097+4*.0000131+16*.00000222)
    assert maximum<35
    return {'batch_name':stamp,'budget_usd':35,'configs':plans,'workers':10,'concurrent_workers':10,
            'estimated_max_compute_usd':maximum,'spec':spec,
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@app.local_entrypoint()
def main(action:str='dry-run',plan_file:str=''):
    personal()
    plan=json.loads(Path(plan_file).read_text()) if plan_file else make_plan()
    assert len(plan['configs'])==10 and plan['budget_usd']==35 and plan['estimated_max_compute_usd']<35
    assert plan['launcher_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for c in plan['configs']:
        assert c['model']['layers']==4 and c['model']['width']==256
        assert c['evaluation']['precision_audit_enabled'] is False
        for name,sha in c['source_sha256'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==sha
    if action=='dry-run':
        print(json.dumps(plan,indent=2));return
    if action!='launch': raise ValueError(action)
    out=ROOT/'outputs/forecast_reference_train_control'/plan['batch_name'];out.mkdir(parents=True,exist_ok=False)
    (out/'resolved_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    calls={}
    for c in plan['configs']:
        call=experiment.with_options(timeout=c['modal']['timeout_seconds']).spawn(c,c['run_name']);calls[c['run_name']]=call.object_id
        (out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
        print(json.dumps({'event':'submitted','run':c['run_name'],'call_id':call.object_id}),flush=True)
    print(json.dumps({'event':'all_submitted','control':str(out),'calls':calls}),flush=True)
    pending=dict(calls)
    while pending:
        for name,call_id in list(pending.items()):
            try:
                r=modal.FunctionCall.from_id(call_id).get(timeout=0)
            except (modal.exception.FunctionTimeoutError,modal.exception.OutputExpiredError) as exc:
                r={'status':'failed','error':str(exc)}
            except (TimeoutError,modal.exception.TimeoutError):continue
            except Exception as exc:r={'status':'failed','error':type(exc).__name__+': '+str(exc)[:1000]}
            (out/f'{name}.json').write_text(json.dumps(r,indent=2)+'\n');del pending[name]
            print(json.dumps({'event':'completed','run':name,'status':r['status']}),flush=True)
        if pending: time.sleep(10)
