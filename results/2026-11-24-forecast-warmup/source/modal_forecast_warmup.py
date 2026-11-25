"""Personal Modal launch for the approved six matched warm-up continuations."""
import hashlib,json,os,time
from pathlib import Path
import modal

def personal():
    if not modal.is_local():return
    from modal import config
    assert os.environ.get('MODAL_PROFILE')==config._profile=='alexander-tian'
    overrides=['MODAL_TOKEN_ID','MODAL_TOKEN_SECRET','MODAL_OAUTH_REFRESH_TOKEN','MODAL_OAUTH_CLIENT_ID',
        'MODAL_OAUTH_CLIENT_SECRET','MODAL_OAUTH_JWT_KEY','MODAL_CONFIG_PATH','MODAL_SERVER_URL']
    assert not any(os.environ.get(k) for k in overrides)
personal()
ROOT=Path(__file__).resolve().parents[1]
image=(modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.10.0',index_url='https://download.pytorch.org/whl/cu128')
    .pip_install('numpy==2.2.6','scipy==1.16.3','PyYAML==6.0.3','wandb==0.31.1')
    .env({'PYTHONPATH':'/root/project/src','OMP_NUM_THREADS':'2'}))
for name in ['forecast_data.py','forecast_models.py','forecast_runtime.py','forecast_horizon.py','forecast_long.py','forecast_warmup.py']:
    image=image.add_local_file(ROOT/'src'/name,remote_path=f'/root/project/src/{name}')
volume=modal.Volume.from_name('project182-artifacts',create_if_missing=False,version=2)
secret=modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])
app=modal.App('project182-forecast-warmup')

def identity():
    import wandb
    username=wandb.Api(timeout=30).viewer.username
    assert username=='alexander-tian'
    return username

@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=300,retries=0,
    max_containers=1,scaledown_window=2,volumes={'/artifacts':volume},secrets=[secret])
def check_gpu(batch):
    from forecast_warmup import preflight
    from forecast_runtime import write_json
    volume.reload()
    try:
        user=identity();output=Path('/artifacts/forecast_warmup/preflight')/batch['batch_name']
        result=preflight(batch['configs'],output);result['wandb_identity']=user
        write_json(output/'preflight.json',result);return result
    finally:volume.commit()

@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,timeout=3600,retries=0,
    max_containers=6,scaledown_window=2,volumes={'/artifacts':volume},secrets=[secret])
def experiment(config,run_name):
    import wandb
    from forecast_warmup import train_and_evaluate
    wandb.teardown();volume.reload();user=identity()
    print(json.dumps({'event':'personal_wandb_verified','username':user,'run_name':run_name}),flush=True)
    try:return train_and_evaluate(config,Path('/artifacts/forecast_warmup/runs')/run_name)
    finally:volume.commit()

def wait(call):
    from grpclib.const import Status
    from grpclib.exceptions import GRPCError,StreamTerminatedError
    while True:
        try:return call.get(timeout=20)
        except (modal.exception.FunctionTimeoutError,modal.exception.OutputExpiredError):raise
        except (TimeoutError,modal.exception.TimeoutError):continue
        except (ConnectionError,modal.exception.ConnectionError,OSError,StreamTerminatedError):time.sleep(5)
        except GRPCError as exc:
            if exc.status not in (Status.UNAVAILABLE,Status.DEADLINE_EXCEEDED):raise
            time.sleep(5)

@app.local_entrypoint()
def main(action:str='dry-run',batch_file:str='',result_file:str='',seed:int=182,arm:str='carry'):
    personal();batch=json.loads(Path(batch_file).read_text())
    for name,sha in batch['source_sha256'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==sha
    assert action in ['dry-run','preflight','train']
    if action=='dry-run':print(json.dumps(batch,indent=2));return
    control=Path(batch_file).resolve().parent
    name=batch['batch_name']+'_preflight' if action=='preflight' else f'alex_261124_ar1_warmup_{arm}_s{seed}'
    if action=='train':
        assert seed in [182,183,184] and arm in ['carry','reset']
        check=json.loads((control/'preflight_result.json').read_text())
        assert check['status']=='passed' and check['budget_gate_passed'] and check['projected_running_compute_usd']<=15
        config=next(c for c in batch['configs'] if c['training']['seed']==seed and c['warmup_protocol']['arm']==arm)
    reservation=control/f'{name}.submitted'
    fd=os.open(reservation,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
    call=check_gpu.spawn(batch) if action=='preflight' else experiment.spawn(config,name)
    record={'event':'submitted','run_name':name,'action':action,'arm':arm,'seed':seed,'function_call_id':call.object_id}
    reservation.write_text(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
    result=wait(call);Path(result_file).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
