"""Confirmation evaluation after the longer-run horizon has been fixed."""
import json
from pathlib import Path
import numpy as np
import torch
from munch import Munch
from pilot_runtime import ROOT,setup,make_model,json_save
from pilot_eval import fixed_dataset,predictions,metrics,order_sensitivity
BASE=Path('/artifacts/followup_v1')


def evaluate(spec):
    import yaml,wandb
    device=setup();out=ROOT/'runs'/spec['run_id']
    c=Munch.fromDict(yaml.safe_load((out/'config.yaml').read_text()))
    assert c.training.seed==spec['seed']
    model=make_model(c,device)
    confirmation=fixed_dataset(BASE/'data/confirmation.npz',300000000,1024,33,5,.5)
    checkpoints=[2000,5000,spec['steps']]
    entries={}
    for step in dict.fromkeys(checkpoints):
        path=out/'checkpoints'/f'step_{step}.pt'
        state=torch.load(path,map_location=device,weights_only=False)
        assert state['completed_steps']==step
        model.load_state_dict(state['model_state_dict']);del state
        pred=predictions(model,confirmation,device)
        result={'step':step,'test':metrics(pred,confirmation),
                'order_sensitivity':order_sensitivity(model,confirmation,device,n_prompts=256,n_permutations=5)}
        pred_path=BASE/'predictions'/f"{spec['run_id']}_step{step}.npz"
        pred_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(pred_path,pred=pred)
        entries[str(step)]=result
    final=entries[str(spec['steps'])]
    record={'run_id':spec['run_id'],'family':spec['family'],'seed':spec['seed'],
            'horizon':spec['steps'],'confirmation_seed':300000000,
            'n_prompts':1024,'order_prompts':256,'order_permutations':5,
            'checkpoints':entries,'run_url':f"https://forge.coreweave.com/wandb/{c.wandb.entity}/{c.wandb.project}/{spec['run_id']}"}
    api=wandb.Api(timeout=30)
    saved=api.run(f'{c.wandb.entity}/{c.wandb.project}/{spec["run_id"]}')
    saved.summary['confirmation/normalized_excess']=final['test']['normalized_excess']
    saved.summary['confirmation/late_normalized_excess']=final['test']['late_normalized_excess']
    saved.summary['confirmation/order_sensitivity_k32']=final['order_sensitivity']['32']['normalized_change']
    saved.summary['confirmation/step']=spec['steps']
    saved.summary['confirmation/dataset_seed']=300000000
    if (out/'evaluation.json').exists():
        old=json.loads((out/'evaluation.json').read_text())
        saved.summary['test/step']=old['completed_steps']
        saved.summary['test/dataset_seed']=200000000
    saved.summary.update()
    record['wandb_summary_written']=True
    json_save(BASE/'evaluations'/f"{spec['run_id']}.json",record)
    return {'run_id':spec['run_id'],'seed':spec['seed'],'steps':spec['steps'],
            'confirmation_excess':final['test']['normalized_excess'],
            'order_sensitivity_k32':final['order_sensitivity']['32']['normalized_change']}
