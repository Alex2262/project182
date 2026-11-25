"""Verify and summarize paired warm-up continuations against their frozen parents."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast_data import score

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();root=a.repo;out=a.output;out.mkdir(parents=True,exist_ok=True)
raw=root/'outputs/forecast_warmup_artifacts';control=root/'outputs/forecast_warmup_control'
reference=dict(np.load(root/'results/2026-11-24-forecast-64k/reference.npz'))
origins=reference['lengths'];arrays={};checks={};metrics=[];configs={}
for arm in ['carry','reset']:
    values=[]
    for seed in [182,183,184]:
        base=raw/f'{arm}_s{seed}';d=base/'long_evaluation'
        ref=dict(np.load(d/'validation_reference.npz'))
        for k in reference:
            if k in ['bayes_parameter_variance','bayes_epistemic_risk']:np.testing.assert_allclose(reference[k],ref[k],rtol=0,atol=1e-18)
            else:np.testing.assert_array_equal(reference[k],ref[k])
        z=np.load(d/'predictions.npz');pred=z['prediction'];np.testing.assert_array_equal(z['origins'],origins)
        assert pred.shape==(512,6,2) and np.isfinite(pred).all()
        old=json.loads((d/'metrics.json').read_text())
        for j,policy in enumerate(['full','window2048']):
            for row,previous in zip(score(pred[:,:,j],reference),old[policy],strict=True):
                for k,v in row.items():
                    if isinstance(v,(int,float)):np.testing.assert_allclose(v,previous[k],rtol=1e-10,atol=1e-12)
                metrics.append({'arm':arm,'seed':seed,'policy':policy,**row})
        result=json.loads((base/'job_result.json').read_text())
        assert result['status']=='finished' and result['training']['completed_steps']==33000
        sha=hashlib.sha256((base/'state_step_33000.pt').read_bytes()).hexdigest()
        assert sha==result['evaluation']['checkpoint_sha256']
        cfg=yaml.safe_load((base/'config.yaml').read_text());configs[arm,seed]=cfg
        parent=root/f'outputs/forecast_optimization/alex_261123_ar1_opt30k_cosine_s{seed}/state_step_30000.pt'
        assert hashlib.sha256(parent.read_bytes()).hexdigest()==cfg['warmup_protocol']['parent_checkpoint_sha256']
        checks[f'{arm}_{seed}']={'final_and_parent_checkpoint_sha256_verified':True,'all_metrics_rescored':True,'reference_targets_identical':True,**json.loads((d/'verification.json').read_text())}
        dest=out/'artifacts'/f'{arm}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['predictions.npz','metrics.json','verification.json','evaluation_result.json','evaluation_plan.json']:shutil.copyfile(d/name,dest/name)
        for name in ['config.yaml','training.json','training_result.json','job_result.json']:shutil.copyfile(base/name,dest/name)
        values.append(pred)
    arrays[arm]=np.stack(values)
for seed in [182,183,184]:
    x=configs['carry',seed];y=configs['reset',seed]
    assert x['training']==y['training'] and x['model']==y['model'] and x['source_sha256']==y['source_sha256']
    wx=dict(x['warmup_protocol']);wy=dict(y['warmup_protocol']);wx.pop('arm');wy.pop('arm');assert wx==wy
for arm,family in [('parent','gdn'),('transformer','transformer')]:
    arrays[arm]=np.stack([np.load(root/f'results/2026-11-24-forecast-64k/artifacts/{family}_s{s}/predictions.npz')['prediction'] for s in [182,183,184]])
loss={k:(v-reference['latent_mean'][None,:,:,None])**2 for k,v in arrays.items()}
summary={'origins':origins.tolist(),'errors':{k:v.mean((0,1)).tolist() for k,v in loss.items()},'seed_errors':{k:v.mean(1).tolist() for k,v in loss.items()},'confirmation_set_opened':False}
comparisons={
    'carry_minus_reset_full':loss['carry'][:,:,:,0]-loss['reset'][:,:,:,0],
    'carry_minus_parent_full':loss['carry'][:,:,:,0]-loss['parent'][:,:,:,0],
    'carry_full_minus_parent_window':loss['carry'][:,:,:,0]-loss['parent'][:,:,:,1],
    'carry_window_minus_full':loss['carry'][:,:,:,1]-loss['carry'][:,:,:,0],
    'reset_window_minus_full':loss['reset'][:,:,:,1]-loss['reset'][:,:,:,0]}
rng=np.random.default_rng(26112433);indices=rng.integers(0,512,size=(5000,512));contrasts=[];ci={}
for name,delta in comparisons.items():
    paired=delta.mean(0);boot=np.concatenate([paired[i].mean(1) for i in np.array_split(indices,10)])
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0);ci[name]=(lo,hi)
    for j,t in enumerate(origins):contrasts.append({'comparison':name,'origin':int(t),'mean_difference':float(paired[:,j].mean()),'ci_lower':float(lo[j]),'ci_upper':float(hi[j]),'seed_differences':delta.mean(1)[:,j].tolist()})
summary['contrasts']=contrasts
summary['bootstrap']='5000 paired-series resamples, seed 26112433; pointwise intervals conditional on three fixed seed pairs'
seconds=json.loads((control/'preflight_result.json').read_text())['elapsed_seconds']+sum(json.loads((raw/f'{arm}_s{s}/job_result.json').read_text())['total_elapsed_seconds'] for arm in ['carry','reset'] for s in [182,183,184])
summary['instrumented_seconds']=seconds;summary['estimated_running_compute_usd']=seconds*json.loads((control/'preflight_result.json').read_text())['rate_usd_per_second']
for name,value in [('summary.json',summary),('independent_verification.json',checks)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
for name,rows in [('metrics.csv',metrics),('paired_contrasts.csv',contrasts)]:
    with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
np.savez_compressed(out/'reference.npz',**reference)
for name in ['batch.json','launch_manifest.json','preflight_result.json','cpu_checks.json','app_status_after.json']:shutil.copyfile(control/name,out/name)
shutil.copytree(control/'source_snapshot',out/'source',dirs_exist_ok=True);shutil.copyfile(__file__,out/'source/summarize_forecast_warmup.py')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(14,6.7));fig.subplots_adjust(left=.075,right=.985,bottom=.32,top=.82,wspace=.28)
colors={'parent':'#777777','reset':'#c44b3d','carry':'#159477','transformer':'#286ab4'}
labels={'parent':'Original GDN: full history','reset':'Reset-trained GDN: full history','carry':'Carry-trained GDN: full history','transformer':'Transformer: full history'}
for arm in ['parent','reset','carry','transformer']:
    sm=loss[arm][:,:,:,0].mean(1);m=sm.mean(0)
    ax[0].plot(origins,m,color=colors[arm],marker='o',ms=4,ls=':' if arm=='transformer' else '-',label=labels[arm]);ax[0].fill_between(origins,sm.min(0),sm.max(0),color=colors[arm],alpha=.09)
window_seeds=loss['parent'][:,:,:,1].mean(1)
ax[0].plot(origins,window_seeds.mean(0),color='#bd8426',marker='s',ms=4,ls='--',lw=2,label='Original GDN: latest 2k')
ax[0].fill_between(origins,window_seeds.min(0),window_seeds.max(0),color='#bd8426',alpha=.1)
bayes_error=((reference['bayes_mean']-reference['latent_mean'])**2).mean(0)
ax[0].plot(origins,bayes_error,color='#171717',marker='.',ls='-.',label='Exact Bayes: full history')
for arm in ['carry','reset']:
    name=arm+'_window_minus_full';d=comparisons[name].mean((0,1))*1e4;lo,hi=ci[name]
    ax[1].plot(origins,d,color=colors[arm],marker='o',label=labels[arm].replace(': full history',''));ax[1].fill_between(origins,lo*1e4,hi*1e4,color=colors[arm],alpha=.14)
ax[1].axhline(0,color='#555',lw=1);ax[1].text(.02,.95,'Above zero: full history better',transform=ax[1].transAxes,va='top');ax[1].text(.02,.05,'Below zero: latest 2k better',transform=ax[1].transAxes)
ax[0].set(title='Forecast accuracy against previous baselines',ylabel='Reducible forecast error (log scale)',yscale='log')
ax[1].set(title='Does recent-window replay still help?',ylabel='Window error − full-history error (×10⁻⁴)')
for a in ax:
    a.set_xscale('log',base=2);a.set_xticks(origins,[f'{t//1024}k' for t in origins]);a.set_xlabel('Observed history (k = 1,024 transitions)');a.grid(axis='y',alpha=.15);a.legend(fontsize=8,loc='lower right') if a is ax[1] else None
ax[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2,frameon=False)
fig.suptitle('Carry versus reset during training · 3 paired seeds · 512 shared validation series',fontsize=13,y=.95)
fig.text(.5,.035,'Left shading: seed range; Bayes line is empirical error on the same series. Right: paired-series 95% intervals. Weights frozen.',ha='center',fontsize=9)
fig.savefig(out/'carry_vs_reset.png',dpi=180);fig.savefig(out/'carry_vs_reset.pdf');plt.close(fig)
print(json.dumps({'errors':summary['errors'],'64k_contrasts':[c for c in contrasts if c['origin']==65536],'cost':summary['estimated_running_compute_usd']},indent=2))
