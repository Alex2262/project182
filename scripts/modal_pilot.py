"""Personal Modal launcher. Commands: check, train, evaluate, export.

All calls are explicitly launched in the personal profile. A CPU-built image
contains dependencies; GPU allocation begins only for the actual pilot call.
"""
import json
from pathlib import Path
import modal

ROOT=Path(__file__).resolve().parents[1]
app=modal.App('project182-pilot')
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


@app.function(image=image,gpu='A100-40GB',cpu=2,memory=8192,
              volumes={'/artifacts':volume},
              secrets=[modal.Secret.from_name('project182-wandb',required_keys=['WANDB_API_KEY'])],
              timeout=900,retries=0,max_containers=1,scaledown_window=2)
def execute(action:str,family:str='all',lr:float=.0003,steps:int=250,
            run_id:str='',max_seconds:int=600):
    import sys,time
    sys.path.insert(0,'/root/project/src')
    start=time.monotonic()
    from pilot_runtime import dispatch
    ledger_path=Path('/artifacts/pilot_v1/compute_ledger.json')
    ledger_path.parent.mkdir(parents=True,exist_ok=True)
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    # Reserve the full 900-second timeout, then replace with measured duration.
    # $3/hour is above the requested A100 + 2 CPU + 8 GiB rates; $0.50 is
    # separately reserved for image building, storage, and startup overhead.
    used=sum(item['charged_seconds'] for item in ledger)
    if used+900>5400:
        raise RuntimeError('Pilot budget guard: insufficient remaining $5 allocation')
    record={'action':action,'family':family,'run_id':run_id,'charged_seconds':900,'status':'reserved'}
    ledger.append(record);ledger_path.write_text(json.dumps(ledger,indent=2));volume.commit()
    try:
        result=dispatch(action,family,lr,steps,run_id,max_seconds)
        return result
    finally:
        elapsed=time.monotonic()-start
        record.update({'charged_seconds':elapsed,'status':'finished'})
        ledger_path.write_text(json.dumps(ledger,indent=2))
        print(json.dumps({'call_elapsed_seconds':elapsed,
                          'conservative_compute_cost_usd':elapsed/3600*3}))
        volume.commit()


@app.local_entrypoint()
def main(action:str='check',family:str='all',lr:float=.0003,steps:int=250,
         run_id:str='',max_seconds:int=600):
    if not 1<=max_seconds<=750:
        raise ValueError('max-seconds must be between 1 and 750')
    result=report.remote() if action=='report' else execute.remote(action,family,lr,steps,run_id,max_seconds)
    print(json.dumps(result,indent=2))


@app.function(image=image,cpu=2,memory=4096,volumes={'/artifacts':volume},timeout=180,retries=0)
def report():
    from pilot_report import generate
    result=generate()
    volume.commit()
    return result
