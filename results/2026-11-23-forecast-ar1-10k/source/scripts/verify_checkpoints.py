"""Verify retained checkpoints and replay final predictions locally, without GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


def verify(raw, repo, manifest_path, output):
    sys.path.insert(0,str(repo/'src'))
    from forecast_models import Forecaster
    from forecast_runtime import predict, setup
    setup('cpu')
    manifest=json.loads(manifest_path.read_text())
    checks=[]
    for run in manifest['runs']:
        directory=raw/run['run_name']
        config=yaml.safe_load((directory/'config.yaml').read_text())
        for name,expected in config['source_sha256'].items():
            assert hashlib.sha256((repo/'src'/name).read_bytes()).hexdigest()==expected
        for step in [2000,5000,10000]:
            saved=torch.load(directory/f'state_step_{step}.pt',map_location='cpu',weights_only=False)
            assert saved['completed_steps']==step and saved['config']==config
        final=torch.load(directory/'state.pt',map_location='cpu',weights_only=False)
        assert final['completed_steps']==10000 and final['config']==config
        for key,value in saved['model'].items():
            torch.testing.assert_close(final['model'][key],value,rtol=0,atol=0)
        if run['seed']==182:
            parent=repo/'outputs/forecast_v1'/run['parent_run_name']
            imported=torch.load(directory/'state_step_2000.pt',map_location='cpu',weights_only=False)
            original=torch.load(parent/'state.pt',map_location='cpu',weights_only=False)
            for key,value in original['model'].items():
                torch.testing.assert_close(imported['model'][key],value,rtol=0,atol=0)
            for key,state in original['optimizer']['state'].items():
                for field,value in state.items():
                    torch.testing.assert_close(imported['optimizer']['state'][key][field],value,rtol=0,atol=0)
            assert (directory/'validation_predictions_step_2000.npz').read_bytes()==(parent/'validation_predictions_step_2000.npz').read_bytes()
        model=Forecaster(config['model']).float()
        model.load_state_dict(final['model'])
        with np.load(directory/'data/validation.npz') as data:
            sample={'values':data['values'][:2],'lengths':data['lengths']}
        replay=predict(model,sample,'cpu',batch_size=2)
        with np.load(directory/'validation_predictions_step_10000.npz') as predictions:
            recorded=predictions['prediction'][:2]
        np.testing.assert_allclose(replay,recorded,rtol=3e-5,atol=3e-5)
        row={'run_name':run['run_name'],'completed_steps':10000,'retained_checkpoints':[2000,5000,10000],
             'source_hashes_match':True,'checkpoint_configs_match':True,
             'continuation_parent_model_and_optimizer_verified':True if run['seed']==182 else None,
             'replay_series':2,'replay_lengths':sample['lengths'].tolist(),
             'cpu_vs_gpu_prediction_max_abs_error':float(np.abs(replay-recorded).max())}
        checks.append(row)
        print(json.dumps(row),flush=True)
    output.write_text(json.dumps({'checkpoint_checks':checks,'confirmation_set_opened':False},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    verify(a.raw,a.repo,a.manifest,a.output)
