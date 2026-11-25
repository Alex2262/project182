"""Approved personal 64k evaluation and matched Transformer batch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import modal


def personal():
    if not modal.is_local(): return
    from modal import config
    assert os.environ.get('MODAL_PROFILE') == config._profile == 'alexander-tian'
    overrides=['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','MODAL_OAUTH_REFRESH_TOKEN',
        'MODAL_OAUTH_CLIENT_ID','MODAL_OAUTH_CLIENT_SECRET','MODAL_OAUTH_JWT_KEY',
        'MODAL_CONFIG_PATH','MODAL_SERVER_URL']
    assert not any(os.environ.get(k) for k in overrides), 'Unexpected Modal credential override'


personal()
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0',index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('numpy==2.2.6','scipy==1.16.3','PyYAML==6.0.3','wandb==0.31.1')
    .env({'PYTHONPATH':'/root/project/src','OMP_NUM_THREADS':'2'}))
for name in ['forecast_data.py','forecast_models.py','forecast_runtime.py','forecast_horizon.py','forecast_long.py']:
    image=image.add_local_file(ROOT/'src'/name,remote_path=f'/root/project/src/{name}')
volume=modal.Volume.from_name('project182-artifacts',create_if_missing=False,version=2)
secret=modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])
app=modal.App('project182-forecast-64k')


def logging_identity():
    import wandb
    api=wandb.Api(timeout=30)
    assert api.viewer.username == 'alexander-tian', 'Unexpected W&B user'
    return api.viewer.username


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=300,
    retries=0,max_containers=1,scaledown_window=2,volumes={'/artifacts':volume},secrets=[secret])
def check_gpu(batch):
    from forecast_long import preflight
    from forecast_runtime import write_json
    volume.reload()
    try:
        identity=logging_identity()
        output=f"/artifacts/forecast_long/preflight/{batch['batch_name']}"
        result=preflight(batch['transformer_configs'][0],batch['preflight_parents'],output)
        result['wandb_identity']=identity
        write_json(Path(output)/'preflight.json',result)
        return result
    finally: volume.commit()


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=5400,
    retries=0,max_containers=3,scaledown_window=2,volumes={'/artifacts':volume},secrets=[secret])
def train_evaluate(config,plan,run_name):
    from forecast_long import evaluate_long
    from forecast_runtime import train,write_json
    import wandb
    wandb.teardown();volume.reload()
    assert config['wandb']['entity']=='alexander-tian-ucb' and config['wandb']['project']=='project182'
    identity=logging_identity()
    print(json.dumps({'event':'personal_wandb_verified','username':identity,'run_name':run_name}),flush=True)
    output=Path('/artifacts/forecast_long/runs')/run_name
    started=time.monotonic()
    try:
        result=train(config,output,device='cuda',max_seconds=4200)
        write_json(output/'training_result.json',result)
        if result['status']!='finished': return result
        wandb.teardown()
        ev=evaluate_long(plan,output/'state_step_30000.pt',output/'long_evaluation')
        result={'status':'finished','run_name':run_name,'training':result,'evaluation':ev,
                'total_elapsed_seconds':time.monotonic()-started,'wandb_identity':identity}
        write_json(output/'job_result.json',result)
        return result
    finally: volume.commit()


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=1800,
    retries=0,max_containers=3,scaledown_window=2,volumes={'/artifacts':volume})
def evaluate_gdn(plan,run_name):
    from forecast_long import evaluate_long
    volume.reload()
    try:
        parent=Path(plan['old_data_directory'])
        return evaluate_long(plan,parent/'state_step_30000.pt',f'/artifacts/forecast_long/evaluations/{run_name}')
    finally: volume.commit()


def wait(call):
    from grpclib.const import Status
    from grpclib.exceptions import GRPCError,StreamTerminatedError
    while True:
        try: return call.get(timeout=20)
        except (modal.exception.FunctionTimeoutError,modal.exception.OutputExpiredError): raise
        except (TimeoutError,modal.exception.TimeoutError): continue
        except (ConnectionError,modal.exception.ConnectionError,OSError,StreamTerminatedError) as exc:
            print(json.dumps({'event':'read_retry','call_id':call.object_id,'exception':type(exc).__name__}),flush=True)
            time.sleep(5)
        except GRPCError as exc:
            if exc.status not in (Status.UNAVAILABLE,Status.DEADLINE_EXCEEDED): raise
            time.sleep(5)


def verify_sources(batch):
    for name,expected in batch['source_sha256'].items():
        assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==expected,name


@app.local_entrypoint()
def main(action:str='dry-run',batch_file:str='',result_file:str='',seed:int=182):
    personal()
    batch=json.loads(Path(batch_file).read_text());verify_sources(batch)
    assert batch['budget']['timeout_reservation_usd'] <= 15
    assert action in ['dry-run','preflight','train','evaluate']
    if action=='dry-run':
        print(json.dumps(batch,indent=2));return
    if action=='preflight':call=check_gpu.spawn(batch)
    else:
        assert seed in [182,183,184]
        check=json.loads(Path(batch['preflight_result_file']).read_text())
        assert check['status']=='passed' and check['budget_gate_passed']
        assert check['projected_running_compute_usd']<=15
        # Reserve the entire worst-case invocation time before submitting once.
        # Local launcher is used sequentially for initial submission; O_EXCL
        # prevents accidentally submitting the same scientific job twice.
        name=f"alex_261124_ar1_64k_{'transformer' if action=='train' else 'gdn'}_s{seed}"
        reservation=ROOT/'outputs/forecast_long_control'/f'{name}.submitted'
        reservation.parent.mkdir(parents=True,exist_ok=True)
        fd=os.open(reservation,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        os.close(fd)
        if action=='train':
            cfg=next(c for c in batch['transformer_configs'] if c['training']['seed']==seed)
            plan=next(p for p in batch['transformer_evaluations'] if p['seed']==seed)
            call=train_evaluate.spawn(cfg,plan,name)
        elif action=='evaluate':
            plan=next(p for p in batch['gdn_evaluations'] if p['seed']==seed)
            call=evaluate_gdn.spawn(plan,name)
        else: raise ValueError(action)
        reservation.write_text(json.dumps({'function_call_id':call.object_id,'run_name':name})+'\n')
    print(json.dumps({'event':'submitted','action':action,'seed':seed,'function_call_id':call.object_id}),flush=True)
    result=wait(call)
    Path(result_file).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
