"""Rebuild compact figures and paired comparisons from downloaded run artifacts."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',default='alex_261124_ref6x256_192325')
    ap.add_argument('--output',default='results/2026-11-24-reference6x256');args=ap.parse_args()
    control=ROOT/'outputs/forecast_reference_train_control'/args.batch
    plan=json.loads((control/'resolved_plan.json').read_text())
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    configs=plan['configs'];rows=[];contrasts=[];errors={};refs=[];checks=[]
    for c in configs:
        name=c['run_name'];directory=control/'artifacts'/name
        manifest=json.loads((directory/'manifest.json').read_text())
        with np.load(directory/'reference.npz',allow_pickle=False) as z: ref={k:z[k] for k in z.files}
        if refs:
            for key in ['lengths','latent_mean','bayes_mean','window_bayes_mean','theta']:
                if key=='window_bayes_mean':
                    np.testing.assert_allclose(ref[key],refs[0][key],rtol=0,atol=1e-12)
                else:
                    np.testing.assert_array_equal(ref[key],refs[0][key])
        refs.append(ref)
        checks.append({'run':name,'status':manifest['status'],'steps':manifest['completed_steps'],
                       'precision_audit':manifest.get('precision_audit'),
                       'quadrature_max_abs':float(ref['window_quadrature_max_abs']),
                       'cross_worker_window_reference_max_abs':float(np.abs(ref['window_bayes_mean']-refs[0]['window_bayes_mean']).max())})
        for step in c['training']['checkpoint_steps']:
            file=directory/f'predictions_step_{step}.npz'
            if not file.exists():continue
            with np.load(file,allow_pickle=False) as z:
                pred=z['prediction'];origins=z['origins'];policies=z['policies']
            if pred.shape!=(512,6,2) or not np.isfinite(pred).all():continue
            np.testing.assert_array_equal(origins,ref['lengths'])
            np.testing.assert_array_equal(policies,['full','window2048'])
            np.testing.assert_array_equal(pred[:,0,0],pred[:,0,1])
            err=(pred-ref['latent_mean'][:,:,None])**2
            errors[c['model']['family'],c['arm'],c['training']['seed'],step]=err
            for j,n in enumerate(origins):
                for k,policy in enumerate(policies):
                    rows.append(dict(family=c['model']['family'],arm=c['arm'],seed=c['training']['seed'],
                                     learning_rate=c['training']['learning_rate'],step=step,origin=int(n),policy=str(policy),
                                     reducible_mse=float(err[:,j,k].mean()),
                                     bayes_disagreement=float(((pred[:,j,k]-ref['bayes_mean'][:,j])**2).mean())))
        target=out/'raw'/name;target.mkdir(parents=True,exist_ok=True)
        for file in directory.iterdir():
            if file.suffix in ['.json','.npz','.py']:shutil.copyfile(file,target/file.name)
    if not refs:raise ValueError('No reference data')
    ref=refs[0];origins=ref['lengths'];primary={};rng=np.random.default_rng(261125)
    bootstrap=rng.integers(0,512,size=(5000,512))
    for family in ['transformer_nope','gated_deltanet']:
        for step in [5000,10000,20000]:
            keys=[(family,'main',s,step) for s in [182,183,184]]
            if not all(k in errors for k in keys):continue
            error=np.stack([errors[k] for k in keys])
            primary[family,step]=error
            paired=(error[:,:,:,1]-error[:,:,:,0]).mean(0)
            ci=np.quantile(paired[bootstrap].mean(1),[.025,.975],axis=0)
            for j,n in enumerate(origins):
                contrasts.append(dict(family=family,step=step,origin=int(n),
                    full_error=float(error[:,:,j,0].mean()),window_error=float(error[:,:,j,1].mean()),
                    window_minus_full=float(paired[:,j].mean()),conditional_ci_low=float(ci[0,j]),conditional_ci_high=float(ci[1,j]),
                    seed182_delta=float((error[0,:,j,1]-error[0,:,j,0]).mean()),
                    seed183_delta=float((error[1,:,j,1]-error[1,:,j,0]).mean()),
                    seed184_delta=float((error[2,:,j,1]-error[2,:,j,0]).mean())))
    for filename,items in [('metrics.csv',rows),('paired_comparisons.csv',contrasts)]:
        if items:
            with (out/filename).open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    colors={'transformer_nope':'#2764AD','gated_deltanet':'#CC641D'}
    labels={'transformer_nope':'Transformer','gated_deltanet':'GDN'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
    for step in [5000,10000,20000]:
        if not all((f,step) in primary for f in colors):continue
        fig,ax=plt.subplots(figsize=(8,5))
        for family in colors:
            seed_curves=primary[family,step].mean(1)
            for k,policy in enumerate(['full history','latest 2k']):
                avg=seed_curves[:,:,k].mean(0)
                ax.plot(origins,avg,marker='o',color=colors[family],ls='-' if k==0 else '--',label=f'{labels[family]}: {policy}')
                ax.fill_between(origins,seed_curves[:,:,k].min(0),seed_curves[:,:,k].max(0),color=colors[family],alpha=.08)
        ax.plot(origins,((ref['bayes_mean']-ref['latent_mean'])**2).mean(0),color='#333333',ls='-',marker='.',label='Bayes: full history')
        ax.plot(origins,((ref['window_bayes_mean']-ref['latent_mean'])**2).mean(0),color='#777777',ls='--',marker='.',label='Bayes: latest 2k')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(origins,[f'{n//1024}k' for n in origins])
        ax.set_xlabel('Observed history at evaluation');ax.set_ylabel('MSE against true conditional mean')
        ax.set_title(f'6 layers × width 256 · {step//1000}k training steps · LR 1e−4')
        ax.grid(alpha=.18);ax.legend(fontsize=9,ncol=2)
        fig.text(.5,.01,'Mean of 3 training seeds; shaded ranges span seeds. Same 512 validation series.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ['png','pdf']:fig.savefig(out/f'long_history_{step}.{ext}',bbox_inches='tight')
        plt.close(fig)
    if ('gated_deltanet',20000) in primary:
        fig,ax=plt.subplots(figsize=(8,5))
        curves=primary['gated_deltanet',20000].mean(1)
        for i,(seed,color) in enumerate(zip([182,183,184],['#2B7A78','#3E65A8','#A64B86'])):
            for k,policy in enumerate(['full history','latest 2k']):
                ax.plot(origins,curves[i,:,k],color=color,ls='-' if k==0 else '--',marker='o',label=f'Seed {seed}: {policy}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(origins,[f'{n//1024}k' for n in origins])
        ax.set_xlabel('Observed history at evaluation');ax.set_ylabel('MSE against true conditional mean')
        ax.set_title('GDN 6×256: long-history behavior differs across training seeds')
        ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.18)
        fig.text(.5,.01,'20k training steps · LR 1e−4 · identical 512 validation series',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ['png','pdf']:fig.savefig(out/f'gdn_by_seed.{ext}',bbox_inches='tight')
        plt.close(fig)
    legacy_path=ROOT/'results/2026-11-24-forecast-64k'
    if (legacy_path/'metrics.csv').exists():
        with np.load(legacy_path/'reference.npz') as old:
            for key in ['latent_mean','bayes_mean']:
                np.testing.assert_array_equal(old[key],ref[key])
        with (legacy_path/'metrics.csv').open() as f: legacy=list(csv.DictReader(f))
        compare=[]
        for family,oldname in [('transformer_nope','Transformer'),('gated_deltanet','GDN')]:
            for origin in origins:
                for policy in ['full','window2048']:
                    old=[float(r['conditional_mean_error']) for r in legacy if r['family']==oldname and int(r['length'])==origin and r['policy']==policy]
                    new=[r['reducible_mse'] for r in rows if r['family']==family and r['arm']=='main' and r['step']==20000 and r['origin']==origin and r['policy']==policy]
                    if len(old)==3 and len(new)==3:
                        compare.append(dict(family=family,origin=int(origin),policy=policy,legacy_mean=float(np.mean(old)),reference_mean=float(np.mean(new)),reference_over_legacy=float(np.mean(new)/np.mean(old))))
        if compare:
            with (out/'comparison_to_legacy.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(compare[0]));w.writeheader();w.writerows(compare)
    summary={'batch':args.batch,'checks':checks,'primary':contrasts,'rows':rows,
        'uncertainty':'Pointwise paired series bootstrap conditional on the three trained models; seed variability reported separately.',
        'limitations':['20k is a fixed budget, not established convergence.','The reference models also change blocks and precision relative to old 2-layer models.','Primary comparisons use BF16; the limited precision audit does not validate the 64k error floor.','Single-seed LR arms are sensitivity checks, not replicated winners.','Validation is reused; confirmation set unopened.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(control/'resolved_plan.json',out/'resolved_plan.json')
    (out/'README.md').write_text('# Reference 6×256 forecasting experiment\n\n'
        'Eight approved runs: three seeds per architecture at LR 1e-4 and one seed per architecture at LR 3e-4. '
        '20k training steps; 163.84M prediction positions per run. See `summary.json` for completion and numerical-check status.\n\n'
        '![20k-step long-history comparison](long_history_20000.png)\n\n'
        'Full history and fresh-state latest-2k predictions use identical query origins on 512 validation series. '
        'Bayes window inference includes the likelihood of the initial visible observation. Shading spans training seeds. '
        'Paired intervals in `paired_comparisons.csv` resample series conditional on these models.\n\n'
        'Raw source snapshots, manifests, training curves and per-series predictions are under `raw/`. '
        'Checkpoints remain on the personal Modal volume; source hashes and checkpoint hashes are in the manifests. '
        'No confirmation-set access.\n\n'
        'This is a fixed-budget synthetic-task result, not proof of convergence or a claim about foundation-model scale. '
        'Numerical precision checks cover only four series at 2k/16k.\n')
    print(json.dumps({'output':str(out),'complete_primary_groups':[(f,s) for f,s in primary],
                      'final': [r for r in contrasts if r['step']==20000]},indent=2))

if __name__=='__main__':main()
