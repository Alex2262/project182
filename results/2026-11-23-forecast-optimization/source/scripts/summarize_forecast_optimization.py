"""Recompute the six paired optimization continuations and their fixed endpoints."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STEPS=[10000,15000,20000,30000]
ARMS=['constant','cosine']
LABELS={'constant':'Constant LR','cosine':'Cosine decay'}
COLORS={'constant':'#2563a6','cosine':'#d95f02'}


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def summarize(raw,output,manifest_path,repo):
    sys.path.insert(0,str(repo/'src'))
    from forecast_data import score
    from forecast_horizon import deterioration_point
    output.mkdir(parents=True,exist_ok=True)
    for d in ['configs','predictions']: (output/d).mkdir(exist_ok=True)
    manifest=json.loads(manifest_path.read_text())
    assert len(manifest['jobs'])==6
    runs=[];table=[];checks=[];common=None
    for job in manifest['jobs']:
        assert job['status']=='finished'
        directory=raw/job['run_name']
        cfg=yaml.safe_load((directory/'config.yaml').read_text())
        assert cfg['training']['steps']==30000 and cfg['training']['lengths']==[64,2048]
        assert cfg['training']['reset_every']==0 and cfg['continuation']['completed_steps']==10000
        assert cfg['training']['validation_steps']==[15000,20000,30000]
        assert not (directory/'data/confirmation.npz').exists()
        with np.load(directory/'data/validation.npz') as loaded:
            data={k:loaded[k] for k in loaded.files}
        reference_difference=0.0
        if common is None: common=data
        else:
            for key in data:
                if key.startswith('bayes_'):
                    reference_difference=max(reference_difference,float(np.max(np.abs(common[key]-data[key]))))
                    np.testing.assert_allclose(common[key],data[key],rtol=0,atol=1e-14)
                else:
                    np.testing.assert_array_equal(common[key],data[key])
        predictions={};results={}
        for step in STEPS:
            pred_path=directory/f'validation_predictions_step_{step}.npz'
            with np.load(pred_path) as loaded: predictions[step]=loaded['prediction']
            results[step]=score(predictions[step],data)
            stored=json.loads((directory/f'validation_step_{step}.json').read_text())
            assert len(stored)==len(results[step])==11
            for actual,recorded in zip(results[step],stored):
                assert actual.keys()==recorded.keys()
                for key in actual:
                    np.testing.assert_allclose(actual[key],recorded[key],rtol=1e-10,atol=1e-12)
                table.append({'run_name':job['run_name'],'arm':job['arm'],'training_seed':job['seed'],
                              'training_steps':step,**actual})
            shutil.copyfile(pred_path,output/'predictions'/f'{job["run_name"]}_step_{step}.npz')
        parent=repo/'outputs/forecast_horizon'/job['parent_run_name']
        with np.load(parent/'validation_predictions_step_10000.npz') as loaded:
            np.testing.assert_array_equal(loaded['prediction'],predictions[10000])
        history=json.loads((directory/'training.json').read_text())
        assert [r['step'] for r in history]==list(range(25,30001,25))
        from forecast_runtime import learning_rate_at_step
        learning_rate_error=0.0
        for row in history:
            if row['step']>10000:
                expected_lr=learning_rate_at_step(cfg,row['step']-1)
                learning_rate_error=max(learning_rate_error,abs(row['learning_rate']-expected_lr))
                # Linux/macOS libm cosine differs by at most a few float64 ulps.
                np.testing.assert_allclose(row['learning_rate'],expected_lr,rtol=0,atol=1e-18)
        shutil.copyfile(directory/'config.yaml',output/'configs'/f'{job["run_name"]}.yaml')
        if 'resolved_launch_config' not in job:
            job['resolved_launch_config']=json.loads(Path(job['config_file']).read_text())
        assert job['resolved_launch_config']==cfg
        job['config_file']=f'configs/{job["run_name"]}.yaml'
        job.pop('log_file',None);job.pop('launcher_pid',None)
        runs.append({**job,'config':cfg,'predictions':predictions,'results':results,'history':history})
        checks.append({'run_name':job['run_name'],'identical_validation_inputs_and_targets':True,
                       'bayes_reference_absolute_tolerance':1e-14,'bayes_reference_max_abs_difference':reference_difference,
                       'parent_10k_predictions_identical':True,'recomputed_steps':STEPS,
                       'all_logged_learning_rates_match':True,'learning_rate_absolute_tolerance':1e-18,
                       'learning_rate_max_abs_error':learning_rate_error})
    lengths=common['lengths'];i2048=list(lengths).index(2048);i16k=list(lengths).index(16384)
    bayes=common['bayes_epistemic_risk'].mean(0)
    groups=[];contrasts=[];horizons=[]
    by_arm={arm:sorted([r for r in runs if r['arm']==arm],key=lambda r:r['seed']) for arm in ARMS}
    for arm,members in by_arm.items():
        assert [r['seed'] for r in members]==[182,183,184]
        for step in STEPS:
            values=np.array([[r['conditional_mean_error'] for r in run['results'][step]] for run in members])
            gaps=np.array([[r['excess_vs_bayes'] for r in run['results'][step]] for run in members])
            groups.append({'arm':arm,'step':step,'mean_errors':values.mean(0).tolist(),
                           'seed_sd':values.std(0,ddof=1).tolist(),'per_seed_errors':values.tolist(),
                           'mean_bayes_gaps':gaps.mean(0).tolist(),
                           'ratio_16k_to_2048':float(values.mean(0)[i16k]/values.mean(0)[i2048]),
                           'per_seed_ratio_16k_to_2048':(values[:,i16k]/values[:,i2048]).tolist()})
            for run,errors in zip(members,values):
                horizons.append({'arm':arm,'step':step,'seed':run['seed'],
                                 **deterioration_point(lengths,errors)})
    for step in STEPS:
        for index in [i2048,i16k]:
            paired=[]
            for constant,cosine in zip(by_arm['constant'],by_arm['cosine']):
                a=(constant['predictions'][step][:,index]-common['latent_mean'][:,index])**2
                b=(cosine['predictions'][step][:,index]-common['latent_mean'][:,index])**2
                paired.append({'seed':constant['seed'],'constant_error':float(a.mean()),'cosine_error':float(b.mean()),
                               'cosine_minus_constant':float((b-a).mean()),
                               'paired_series_se':float((b-a).std(ddof=1)/np.sqrt(len(a)))})
            differences=np.array([p['cosine_minus_constant'] for p in paired])
            contrasts.append({'step':step,'length':int(lengths[index]),'per_seed':paired,
                              'mean_difference':float(differences.mean()),'seed_sd':float(differences.std(ddof=1))})
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    np.savez_compressed(output/'validation_reference.npz',**{k:v for k,v in common.items() if k!='values'})
    summary={'lengths':lengths.tolist(),'groups':groups,'paired_contrasts':contrasts,
             'bayes_epistemic_risk':bayes.tolist(),'deterioration_points':horizons,
             'training_seeds':[182,183,184],'primary_step':30000,'confirmation_set_opened':False,
             'generated_at_utc':datetime.now(timezone.utc).isoformat()}
    save(output/'summary.json',summary);save(output/'metric_verification.json',checks)
    save(output/'training_histories.json',{r['run_name']:r['history'] for r in runs})
    save(output/'manifest.json',manifest)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    initial=next(g for g in groups if g['arm']=='constant' and g['step']==10000)
    for axis,metric,ylabel in zip(axes,['mean_errors','mean_bayes_gaps'],
                                ['Reducible forecast MSE','Squared disagreement with Bayes forecast']):
        axis.plot(lengths,initial[metric],color='#666666',linestyle='--',label='Shared 10k starting models')
        for arm in ARMS:
            g=next(g for g in groups if g['arm']==arm and g['step']==30000)
            axis.plot(lengths,g[metric],color=COLORS[arm],marker='o',label=LABELS[arm]+' · 30k')
            if metric=='mean_errors': values=np.array(g['per_seed_errors'])
            else: values=np.array([[r['excess_vs_bayes'] for r in run['results'][30000]] for run in by_arm[arm]])
            axis.fill_between(lengths,values.min(0),values.max(0),color=COLORS[arm],alpha=.12)
        axis.set_xscale('log',base=2);axis.set_yscale('log')
        axis.set(xlabel='Observed transitions',ylabel=ylabel)
        axis.axvline(2048,color='gray',linestyle=':',alpha=.6)
        axis.grid(alpha=.15)
    axes[0].plot(lengths,bayes,'k:',label='Exact Bayes risk reference')
    fig.suptitle('GDN optimization continuation · three seeds · 512 validation series\nLines: seed means; bands: observed seed ranges; vertical line: maximum training history',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
    for ext in ['png','svg']: fig.savefig(output/f'history_curves.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for axis,index in zip(axes,[i2048,i16k]):
        for arm in ARMS:
            values=np.array([[run['results'][step][index]['conditional_mean_error'] for step in STEPS] for run in by_arm[arm]])
            for v in values:axis.plot(STEPS,v,color=COLORS[arm],alpha=.2,lw=.8)
            axis.plot(STEPS,values.mean(0),color=COLORS[arm],marker='o',label=LABELS[arm])
            axis.fill_between(STEPS,values.min(0),values.max(0),color=COLORS[arm],alpha=.12)
        axis.axhline(bayes[index],color='black',linestyle=':',label='Exact Bayes risk reference')
        axis.set(yscale='log',xlabel='Total training updates',ylabel='Reducible forecast MSE',
                 title=f'After {int(lengths[index]):,} observations')
        axis.set_xticks(STEPS);axis.grid(alpha=.15)
    fig.suptitle('Does more optimization close the gap? · 30k fixed primary endpoint\nThick lines: seed means; faint lines: seeds; bands: observed seed ranges',fontsize=11)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,fontsize=9)
    for ext in ['png','svg']:fig.savefig(output/f'learning_curves.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({'final':[{k:g[k] for k in ['arm','mean_errors','seed_sd','ratio_16k_to_2048']} for g in groups if g['step']==30000]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args();summarize(args.raw,args.output,args.manifest,args.repo)
