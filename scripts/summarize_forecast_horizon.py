"""Recompute the predeclared horizon sweep and keep per-seed failure locations."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=['l64','m256','m1024','m2048','m2048_r256']
LABELS=['Train 64','Train 64/256','Train 64/1024','Train 64/2048','64/2048, reset at 256']
COLORS=['#2166ac','#1b9e77','#e68524','#c7353d','#7b4da5']
STEPS=[2000,5000,10000]


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def summarize(raw,output,manifest_path,repo):
    sys.path.insert(0,str(repo/'src'))
    from forecast_data import score
    from forecast_horizon import deterioration_point
    output.mkdir(parents=True,exist_ok=True)
    for directory in ['configs','predictions']:
        (output/directory).mkdir(exist_ok=True)
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest['jobs'])==15
    common=None; runs=[]; table=[]; checks=[]
    for job in manifest['jobs']:
        assert job['status']=='finished'
        directory=raw/job['run_name']
        cfg=yaml.safe_load((directory/'config.yaml').read_text())
        assert cfg['model']['family']=='gated_deltanet'
        assert cfg['training']['seed']==job['seed']
        assert cfg['training']['steps']==10000
        assert not (directory/'data/confirmation.npz').exists()
        with np.load(directory/'data/validation.npz') as loaded:
            data={k:loaded[k] for k in loaded.files}
        difference=0.0
        if common is None:
            common=data
        else:
            for key in ['spec','lengths']:
                np.testing.assert_array_equal(common[key],data[key])
            for key in ['values','theta','latent_mean','target']:
                np.testing.assert_allclose(common[key],data[key],rtol=1e-13,atol=1e-13)
            for key in ['bayes_mean','bayes_risk','bayes_epistemic_risk']:
                np.testing.assert_allclose(common[key],data[key],rtol=1e-12,atol=1e-12)
            difference=float(np.abs(common['values']-data['values']).max())
        history_path=directory/'training.json' if job['action']=='train' else repo/'outputs/forecast_v1'/job['parent_run_name']/'training.json'
        history=json.loads(history_path.read_text())
        assert [h['step'] for h in history]==list(range(25,10001,25))
        predictions={};results={};prefix_error=0.0
        for step in STEPS:
            pred_path=directory/f'validation_predictions_step_{step}.npz'
            with np.load(pred_path) as loaded:
                predictions[step]=loaded['prediction']
            results[step]=score(predictions[step],common)
            recorded=json.loads((directory/f'validation_step_{step}.json').read_text())
            assert len(recorded)==len(results[step])==11
            for calculated,stored in zip(results[step],recorded):
                for key in ['conditional_mean_error','excess_vs_bayes','expected_predictive_mse','observed_mse']:
                    np.testing.assert_allclose(calculated[key],stored[key],rtol=1e-10,atol=1e-12)
                table.append({'run_name':job['run_name'],'arm':job['arm'],'training_seed':job['seed'],
                              'training_steps':step,'max_training_length':job['max_training_length'],**calculated})
            if job['action']=='evaluate':
                parent=repo/'outputs/forecast_v1'/job['parent_run_name']
                with np.load(parent/f'validation_predictions_step_{step}.npz') as loaded:
                    old=loaded['prediction']
                np.testing.assert_allclose(predictions[step][:,:9],old,rtol=3e-5,atol=3e-5)
                prefix_error=max(prefix_error,float(np.abs(predictions[step][:,:9]-old).max()))
            shutil.copyfile(pred_path,output/'predictions'/f'{job["run_name"]}_step_{step}.npz')
        shutil.copyfile(directory/'config.yaml',output/'configs'/f'{job["run_name"]}.yaml')
        if 'resolved_launch_config' not in job:
            job['resolved_launch_config']=json.loads(Path(job['config_file']).read_text())
        job['config_file']=f'configs/{job["run_name"]}.yaml'
        job.pop('log_file',None);job.pop('launcher_pid',None)
        checks.append({'run_name':job['run_name'],'common_validation_max_abs_difference':difference,
                       'reused_checkpoint_prefix_prediction_max_abs_difference':prefix_error if job['action']=='evaluate' else None,
                       'metrics_recomputed_steps':STEPS})
        runs.append({**job,'config':cfg,'history':history,'predictions':predictions,'results':results})
    lengths=common['lengths']; i64=list(lengths).index(64); ilast=len(lengths)-1
    bayes=common['bayes_epistemic_risk'].mean(0)
    np.savez_compressed(output/'validation_reference.npz',**{k:v for k,v in common.items() if k!='values'})
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    groups=[]; summary=[]; horizons=[]; contrasts=[]
    for arm,label in zip(ARMS,LABELS):
        members=sorted([r for r in runs if r['arm']==arm],key=lambda r:r['seed'])
        assert [r['seed'] for r in members]==[182,183,184]
        groups.append(members)
        for step in STEPS:
            values=np.array([[row['conditional_mean_error'] for row in r['results'][step]] for r in members])
            gap=np.array([[row['excess_vs_bayes'] for row in r['results'][step]] for r in members])
            summary.append({'arm':arm,'label':label,'step':step,'max_training_length':members[0]['max_training_length'],
                            'mean_errors':values.mean(0).tolist(),'seed_sd':values.std(0,ddof=1).tolist(),
                            'mean_bayes_gaps':gap.mean(0).tolist(),'per_seed_errors':values.tolist()})
            for run,error in zip(members,values):
                point=deterioration_point(lengths,error)
                horizons.append({'arm':arm,'seed':run['seed'],'step':step,
                                 'max_training_length':run['max_training_length'],**point,
                                 'normalized_first_length':point['first_length']/run['max_training_length'] if point['first_length'] else None})
    for step in STEPS:
        for length in [4096,16384]:
            index=list(lengths).index(length)
            paired=[]
            for seed in [182,183,184]:
                continuous=next(r for r in runs if r['arm']=='m2048' and r['seed']==seed)
                reset=next(r for r in runs if r['arm']=='m2048_r256' and r['seed']==seed)
                c=(continuous['predictions'][step][:,index]-common['latent_mean'][:,index])**2
                r=(reset['predictions'][step][:,index]-common['latent_mean'][:,index])**2
                difference=c-r
                paired.append({'seed':seed,'continuous_error':float(c.mean()),'reset_error':float(r.mean()),
                               'continuous_minus_reset':float(difference.mean()),
                               'paired_series_se':float(difference.std(ddof=1)/np.sqrt(len(difference)))})
            diffs=[r['continuous_minus_reset'] for r in paired]
            contrasts.append({'step':step,'length':length,'mean_difference':float(np.mean(diffs)),
                              'difference_seed_sd':float(np.std(diffs,ddof=1)),'per_seed':paired})

    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True,constrained_layout=True)
    for members,label,color in zip(groups,LABELS,COLORS):
        values=np.array([[row['conditional_mean_error'] for row in r['results'][10000]] for r in members])
        mean,sd=values.mean(0),values.std(0,ddof=1)
        style='--' if members[0]['arm']=='m2048_r256' else '-'
        for axis,x in zip(axes,[lengths,lengths/members[0]['max_training_length']]):
            axis.plot(x,mean,style,color=color,marker='o',markersize=3,label=label)
            axis.fill_between(x,values.min(0),values.max(0),color=color,alpha=.10)
    axes[0].plot(lengths,bayes,'k:',lw=2,label='Exact Bayes reference')
    axes[1].axvline(1,color='gray',linestyle=':',alpha=.7)
    for axis,title,xlabel in zip(axes,['Absolute history length','History relative to training horizon'],
                                ['Observed transitions','Observed transitions / max uninterrupted training history']):
        axis.set_xscale('log',base=2);axis.set_yscale('log')
        axis.set(title=title,xlabel=xlabel,ylabel='Reducible forecast MSE')
        axis.grid(alpha=.15)
    fig.suptitle('GDN training-history sweep · 10,000 updates · three seeds\nLines: seed means; bands: observed seed range · 512 validation series',fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
    for ext in ['png','svg']:fig.savefig(output/f'horizon_curves.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for members,label,color in zip(groups,LABELS,COLORS):
        style='--' if members[0]['arm']=='m2048_r256' else '-'
        for axis,index in zip(axes,[i64,ilast]):
            values=np.array([[r['results'][step][index]['conditional_mean_error'] for step in STEPS] for r in members])
            mean,sd=values.mean(0),values.std(0,ddof=1)
            for v in values:axis.plot(STEPS,v,style,color=color,alpha=.15,lw=.8)
            axis.plot(STEPS,mean,style,color=color,marker='o',label=label)
            axis.fill_between(STEPS,values.min(0),values.max(0),color=color,alpha=.10)
    for axis,index in zip(axes,[i64,ilast]):
        axis.axhline(bayes[index],color='black',linestyle=':')
        axis.set(yscale='log',xlabel='Training updates',ylabel='Reducible forecast MSE',title=f'After {int(lengths[index]):,} observations')
        axis.set_xticks(STEPS);axis.grid(alpha=.15)
    fig.suptitle('Checkpoint trajectories · fixed 10k primary endpoint\nThick lines: seed means; faint lines: seeds; bands: observed seed range',fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
    for ext in ['png','svg']:fig.savefig(output/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    save(output/'summary.json',{'lengths':lengths.tolist(),'groups':summary,'deterioration_points':horizons,
         'matched_control_contrasts':contrasts,'bayes_epistemic_risk':bayes.tolist(),
         'training_seeds':[182,183,184],'confirmation_set_opened':False,'generated_at_utc':datetime.now(timezone.utc).isoformat()})
    save(output/'metric_verification.json',checks)
    save(output/'training_histories.json',{r['run_name']:r['history'] for r in runs})
    save(output/'manifest.json',manifest)
    return {'final_errors':[{k:g[k] for k in ['arm','mean_errors','seed_sd']} for g in summary if g['step']==10000],
            'final_deterioration_points':[h for h in horizons if h['step']==10000],
            'final_matched_control':[c for c in contrasts if c['step']==10000]}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args();print(json.dumps(summarize(a.raw,a.output,a.manifest,a.repo),indent=2))
