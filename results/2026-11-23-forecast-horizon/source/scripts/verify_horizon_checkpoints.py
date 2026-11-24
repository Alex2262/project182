"""Verify new retained states, reused states, and CPU replay at the final endpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml


def verify(raw,repo,manifest_path,output):
    sys.path.insert(0,str(repo/'src'))
    from forecast_models import Forecaster
    from forecast_runtime import predict,setup
    setup('cpu')
    manifest=json.loads(manifest_path.read_text());records=[]
    for job in manifest['jobs']:
        directory=raw/job['run_name']
        cfg=yaml.safe_load((directory/'config.yaml').read_text())
        plan=job.get('resolved_launch_config')
        if plan is None:
            plan=json.loads(Path(job['config_file']).read_text())
        for name,expected in cfg['source_sha256'].items():
            assert hashlib.sha256((repo/'src'/name).read_bytes()).hexdigest()==expected
        hashes={}
        for step in [2000,5000,10000]:
            parent=directory if job['action']=='train' else repo/'outputs/forecast_v1'/job['parent_run_name']
            path=parent/f'state_step_{step}.pt'
            hashes[str(step)]=hashlib.sha256(path.read_bytes()).hexdigest()
            if job['action']=='evaluate':assert hashes[str(step)]==plan['checkpoint_sha256'][str(step)]
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['completed_steps']==step
            if job['action']=='train':assert saved['config']==cfg
        if job['action']=='train':
            latest=torch.load(directory/'state.pt',map_location='cpu',weights_only=False)
            assert latest['completed_steps']==10000 and latest['config']==cfg
            for name,value in saved['model'].items():torch.testing.assert_close(latest['model'][name],value,rtol=0,atol=0)
        model=Forecaster(cfg['model']).float();model.load_state_dict(saved['model'])
        with np.load(directory/'data/validation.npz') as data:
            sample={'values':data['values'][:2],'lengths':data['lengths']}
        replay=predict(model,sample,'cpu',batch_size=2)
        with np.load(directory/'validation_predictions_step_10000.npz') as data:recorded=data['prediction'][:2]
        np.testing.assert_allclose(replay,recorded,rtol=3e-5,atol=3e-5)
        row={'run_name':job['run_name'],'action':job['action'],'verified_steps':[2000,5000,10000],
             'checkpoint_sha256':hashes,'source_hashes_match':True,'replay_series':2,
             'replay_lengths':sample['lengths'].tolist(),
             'cpu_vs_gpu_prediction_max_abs_error':float(np.abs(replay-recorded).max())}
        records.append(row);print(json.dumps(row),flush=True)
    output.write_text(json.dumps({'checks':records,'confirmation_set_opened':False},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.raw,a.repo,a.manifest,a.output)
