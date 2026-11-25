"""Reproduce the completed 64k comparison from downloaded compact artifacts."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from forecast_data import score

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();root=args.repo;out=args.output;out.mkdir(parents=True,exist_ok=True)
raw=root/'outputs/forecast_long_artifacts';control=root/'outputs/forecast_long_control'
predictions={};checks={};ref=None;metrics=[];contrasts=[]
origins=np.array([2048,4096,8192,16384,32768,65536]);policies=['full','window2048']
for family,action in [('GDN','evaluate'),('Transformer','train')]:
    arrays=[]
    for seed in [182,183,184]:
        base=raw/f'{action}_s{seed}';d=base/'long_evaluation' if action=='train' else base
        with np.load(d/'validation_reference.npz') as z:r={k:z[k] for k in z.files}
        if ref is None:ref=r
        reference_errors={}
        for k in ref:
            if k in ['bayes_parameter_variance','bayes_epistemic_risk']:
                np.testing.assert_allclose(ref[k],r[k],rtol=0,atol=1e-18)
                reference_errors[k]=float(np.max(np.abs(ref[k]-r[k])))
            else:np.testing.assert_array_equal(ref[k],r[k])
        with np.load(d/'predictions.npz') as z:
            a=z['prediction'];np.testing.assert_array_equal(z['origins'],origins);np.testing.assert_array_equal(z['policies'],policies)
        assert a.shape==(512,6,2) and np.isfinite(a).all()
        saved=json.loads((d/'metrics.json').read_text())
        for j,policy in enumerate(policies):
            rescored=score(a[:,:,j],ref)
            for row,old in zip(rescored,saved[policy],strict=True):
                for key in row:
                    if isinstance(row[key],(int,float)):np.testing.assert_allclose(row[key],old[key],rtol=1e-10,atol=1e-12)
                metrics.append({'family':family,'seed':seed,'policy':policy,**row})
        checkpoint=base/'state_step_30000.pt' if action=='train' else root/f'outputs/forecast_optimization/alex_261123_ar1_opt30k_cosine_s{seed}/state_step_30000.pt'
        expected=json.loads((d/'evaluation_result.json').read_text())['checkpoint_sha256']
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
        checks[f'{family}_{seed}']={'rescored_all_metrics':True,'targets_parameters_and_reference_means_bit_identical':True,'reference_variance_max_abs_errors':reference_errors,'checkpoint_sha256_verified':True,**json.loads((d/'verification.json').read_text())}
        arrays.append(a)
        dest=out/'artifacts'/f'{family.lower()}_s{seed}';dest.mkdir(parents=True,exist_ok=True)
        for name in ['predictions.npz','metrics.json','verification.json','evaluation_result.json','evaluation_plan.json']:shutil.copyfile(d/name,dest/name)
        if action=='train':
            for name in ['config.yaml','training.json','training_result.json','job_result.json']:shutil.copyfile(base/name,dest/name)
    predictions[family]=np.stack(arrays)
np.savez_compressed(out/'reference.npz',**ref)
rng=np.random.default_rng(26112464);indices=rng.integers(0,512,size=(5000,512))
summary={};bands={};losses={}
for family,a in predictions.items():
    loss=(a-ref['latent_mean'][None,:,:,None])**2;losses[family]=loss
    seedmeans=loss.mean(1);mean=seedmeans.mean(0)
    difference=(loss[:,:,:,1]-loss[:,:,:,0]).mean(0)
    boot=np.concatenate([difference[idx].mean(1) for idx in np.array_split(indices,10)])
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0);bands[family]=(lo,hi)
    summary[family]={'mean_errors':mean.tolist(),'seed_errors':seedmeans.tolist(),'window_minus_full':difference.mean(0).tolist(),'paired_ci_lower':lo.tolist(),'paired_ci_upper':hi.tolist(),'window_error_reduction_fraction':(1-mean[:,1]/mean[:,0]).tolist(),'total_mse_reduction_fraction':((mean[:,0]-mean[:,1])/(.25+mean[:,0])).tolist()}
    for k,t in enumerate(origins):contrasts.append({'family':family,'origin':int(t),'window_minus_full':float(difference[:,k].mean()),'ci_lower':float(lo[k]),'ci_upper':float(hi[k]),'seeds_window_better':int((seedmeans[:,k,1]<seedmeans[:,k,0]).sum())})
bayes=((ref['bayes_mean']-ref['latent_mean'])**2).mean(0)
summary['bayes_empirical_error']=bayes.tolist();summary['origins']=origins.tolist()
summary['confirmation_set_opened']=False;summary['bootstrap']='5000 paired series resamples, seed 26112464; conditions on three fixed models; pointwise intervals'
seconds=json.loads((control/'preflight_result.json').read_text())['elapsed_seconds']
for action in ['train','evaluate']:
    for seed in [182,183,184]:
        r=json.loads((control/f'{action}_s{seed}_result.json').read_text());seconds+=r['total_elapsed_seconds'] if action=='train' else r['elapsed_seconds']
summary['instrumented_seconds']=seconds;summary['estimated_running_compute_usd']=seconds*json.loads((control/'preflight_result.json').read_text())['rate_usd_per_second']
for name,data in [('summary.json',summary),('independent_verification.json',checks)]: (out/name).write_text(json.dumps(data,indent=2)+'\n')
for name,rows in [('metrics.csv',metrics),('paired_contrasts.csv',contrasts)]:
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for name in ['launch_manifest.json','batch.json','preflight_result.json','app_status_after.json']:shutil.copyfile(control/name,out/name)
shutil.copytree(control/'source_snapshot',out/'source',dirs_exist_ok=True)
shutil.copyfile(__file__,out/'source/summarize_forecast_long.py')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(13.5,5.8))
fig.subplots_adjust(left=.075,right=.985,bottom=.2,top=.82,wspace=.27)
colors={'GDN':'#c94a38','Transformer':'#246db5'}
for family,loss in losses.items():
    sm=loss.mean(1);m=sm.mean(0);color=colors[family]
    for j,policy in enumerate(['Full history','Latest 2k']):
        axes[0].plot(origins,m[:,j],color=color,ls='-' if j==0 else '--',marker='o' if j==0 else 's',ms=4,label=f'{family}: {policy}')
        axes[0].fill_between(origins,sm[:,:,j].min(0),sm[:,:,j].max(0),color=color,alpha=.09)
    delta=(loss[:,:,:,1]-loss[:,:,:,0]).mean((0,1))*1e4
    axes[1].plot(origins,delta,color=color,marker='o',label=family)
    lo,hi=bands[family];axes[1].fill_between(origins,lo*1e4,hi*1e4,color=color,alpha=.15)
axes[0].plot(origins,bayes,color='#333333',ls=':',marker='.',label='Exact Bayes: full history')
axes[0].set(yscale='log',ylabel='Error against noise-free next-value mean',title='Full history: Transformer improves, GDN deteriorates')
axes[0].legend(fontsize=9,loc='lower left');axes[0].grid(axis='y',alpha=.18)
axes[1].axhline(0,color='#555555',lw=1)
axes[1].set(ylabel='Window error − full-history error (×10⁻⁴)',title='Same-target comparison of history policies')
axes[1].text(.03,.96,'Above zero: full history better',transform=axes[1].transAxes,va='top',fontsize=10)
axes[1].text(.03,.04,'Below zero: latest 2k better',transform=axes[1].transAxes,fontsize=10)
axes[1].legend(loc='lower right',fontsize=9);axes[1].grid(axis='y',alpha=.18)
for ax in axes:
    ax.set_xscale('log',base=2);ax.set_xticks(origins,[f'{n//1024}k' for n in origins]);ax.set_xlabel('Observed history (k = 1,024 transitions)')
fig.suptitle('30k training updates · 3 seeds · 512 shared validation series',fontsize=14,y=.96)
for ax in axes:ax.title.set_fontsize(11)
fig.text(.5,.035,'Left shading: training-seed range. Right shading: paired-series 95% intervals, conditional on these models. Exploratory validation.',ha='center',fontsize=9)
fig.savefig(out/'comparison_64k.png',dpi=190);fig.savefig(out/'comparison_64k.pdf');plt.close(fig)
print(json.dumps(summary,indent=2))
