"""Audit optimizer continuity and replay each final forecasting model on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml


def same(left,right):
    if isinstance(left,torch.Tensor):torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for k in left:same(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    else:assert left==right


def verify(raw,repo,manifest_path,output):
    sys.path.insert(0,str(repo/'src'))
    from forecast_models import Forecaster
    from forecast_runtime import predict,setup,learning_rate_at_step
    setup('cpu');manifest=json.loads(manifest_path.read_text());records=[]
    for job in manifest['jobs']:
        directory=raw/job['run_name'];cfg=yaml.safe_load((directory/'config.yaml').read_text())
        for name,expected in cfg['source_sha256'].items():
            assert hashlib.sha256((repo/'src'/name).read_bytes()).hexdigest()==expected
        parent=repo/'outputs/forecast_horizon'/job['parent_run_name']/'state_step_10000.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==cfg['continuation']['checkpoint_sha256']
        parent_state=torch.load(parent,map_location='cpu',weights_only=False)
        hashes={}
        for step in [10000,15000,20000,30000]:
            path=directory/f'state_step_{step}.pt';hashes[str(step)]=hashlib.sha256(path.read_bytes()).hexdigest()
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['completed_steps']==step and saved['config']==cfg
            assert all(int(s['step'])==step for s in saved['optimizer']['state'].values())
            if step==10000:
                same(parent_state['model'],saved['model']);same(parent_state['optimizer'],saved['optimizer'])
            else:
                assert all(g['lr']==learning_rate_at_step(cfg,step-1) for g in saved['optimizer']['param_groups'])
        latest=torch.load(directory/'state.pt',map_location='cpu',weights_only=False)
        same(latest,saved)
        model=Forecaster(cfg['model']).float();model.load_state_dict(saved['model'])
        with np.load(directory/'data/validation.npz') as data:
            sample={'values':data['values'][:2],'lengths':data['lengths']}
        replay=predict(model,sample,'cpu',batch_size=2)
        with np.load(directory/'validation_predictions_step_30000.npz') as data:
            recorded=data['prediction'][:2]
        np.testing.assert_allclose(replay,recorded,rtol=3e-5,atol=3e-5)
        row={'run_name':job['run_name'],'verified_steps':[10000,15000,20000,30000],
             'checkpoint_sha256':hashes,'source_hashes_match':True,
             'parent_model_and_adam_bit_exact':True,'adam_steps_and_learning_rates_verified':True,
             'replay_series':2,'replay_lengths':sample['lengths'].tolist(),
             'cpu_vs_gpu_prediction_max_abs_error':float(np.abs(replay-recorded).max())}
        records.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps({'checks':records,'confirmation_set_opened':False},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.raw,a.repo,a.manifest,a.output)
