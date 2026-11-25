"""Rebuild the approved 4x256 warmup/cosine LR sweep from saved predictions."""
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',default='alex_261124_ref4x256_40k_200555');ap.add_argument('--output',default='results/2026-11-24-reference4x256-40k');a=ap.parse_args()
    control=ROOT/'outputs/forecast_reference_train_control'/a.batch
    plan=json.loads((control/'resolved_plan.json').read_text());out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    groups={};rows=[];quick=[];checks=[];reference=None;manifests=[]
    for c in plan['configs']:
        name=c['run_name'];d=control/'artifacts'/name
        if not (d/'reference.npz').exists():continue
        m=json.loads((d/'manifest.json').read_text());manifests.append(m)
        assert c['model']['layers']==4 and c['model']['width']==256
        for f,sha in c['source_sha256'].items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==sha
        with np.load(d/'reference.npz') as z:ref={k:z[k] for k in z.files}
        if reference is None:reference=ref
        for k in ['theta','latent_mean','lengths']:np.testing.assert_array_equal(ref[k],reference[k])
        for k in ['bayes_mean','window_bayes_mean']:np.testing.assert_allclose(ref[k],reference[k],rtol=0,atol=1e-12)
        check={'run':name,'status':m['status'],'steps':m['completed_steps'],'checkpoint_steps':[],
               'precision_audit':m.get('precision_audit'),'source_hashes_verified':True}
        fam=c['model']['family'];lr=c['training']['learning_rate'];seed=c['training']['seed']
        q=json.loads((d/'quick_validation.json').read_text()) if (d/'quick_validation.json').exists() else []
        for x in q:quick.append(dict(family=fam,peak_lr=lr,seed=seed,**x))
        for step in c['training']['checkpoint_steps']:
            p=d/f'predictions_step_{step}.npz'
            if not p.exists():continue
            with np.load(p) as z:pred=z['prediction'];origins=z['origins']
            if not np.isfinite(pred).all():continue
            assert pred.shape==(512,6,2)
            np.testing.assert_array_equal(origins,ref['lengths']);np.testing.assert_array_equal(pred[:,0,0],pred[:,0,1])
            err=(pred-ref['latent_mean'][:,:,None])**2
            saved_metrics=json.loads((d/f'metrics_step_{step}.json').read_text())
            stored=saved_metrics['rows']
            if len(stored)!=len(origins):continue  # A live download can straddle two saved origins.
            for j,origin in enumerate(origins):
                np.testing.assert_allclose(err[:,j].mean(0),[stored[j]['full_error'],stored[j]['window_error']],rtol=0,atol=1e-14)
                for k,pol in enumerate(['full','window2048']):
                    rows.append(dict(family=fam,peak_lr=lr,seed=seed,step=step,origin=int(origin),policy=pol,
                                     reducible_mse=float(err[:,j,k].mean()),bayes_disagreement=float(((pred[:,j,k]-ref['bayes_mean'][:,j])**2).mean())))
            endpoint=next((x for x in q if x['step']==step),None)
            if endpoint:np.testing.assert_allclose(endpoint['conditional_mean_mse'],err[:,0,0].mean(),rtol=0,atol=1e-12)
            groups.setdefault((fam,lr,step),{})[seed]=err
            check['checkpoint_steps'].append(step)
        checks.append(check)
        raw=out/'raw'/name;raw.mkdir(parents=True,exist_ok=True)
        for f in d.iterdir():
            if f.suffix in ['.npz','.json','.py']:shutil.copyfile(f,raw/f.name)
    if reference is None:raise ValueError('No saved reference yet')
    ref=reference;origins=ref['lengths'];aggregates=[];bootstrap=np.random.default_rng(26112440).integers(0,512,(5000,512))
    complete={}
    for key,items in groups.items():
        if set(items)!={185,186,187}:continue
        e=np.stack([items[s] for s in [185,186,187]]);complete[key]=e
        dif=(e[:,:,:,1]-e[:,:,:,0]).mean(0);ci=np.quantile(dif[bootstrap].mean(1),[.025,.975],axis=0)
        for j,origin in enumerate(origins):
            aggregates.append(dict(family=key[0],peak_lr=key[1],step=key[2],origin=int(origin),
                full=float(e[:,:,j,0].mean()),window=float(e[:,:,j,1].mean()),window_minus_full=float(dif[:,j].mean()),
                conditional_series_ci_low=float(ci[0,j]),conditional_series_ci_high=float(ci[1,j]),
                per_seed_delta={str(s):float((e[i,:,j,1]-e[i,:,j,0]).mean()) for i,s in enumerate([185,186,187])}))
    for name,rs in [('metrics.csv',rows),('quick_validation.csv',quick),('paired_comparisons.csv',aggregates)]:
        if not rs:continue
        with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    colors={'transformer_nope':'#2166AC','gated_deltanet':'#CC641D'};labels={'transformer_nope':'Transformer','gated_deltanet':'GDN'}
    bayes=((ref['bayes_mean']-ref['latent_mean'])**2).mean(0);wb=((ref['window_bayes_mean']-ref['latent_mean'])**2).mean(0)
    for lr in [1e-4,3e-4]:
        if not all((f,lr,40000) in complete for f in colors):continue
        fig,ax=plt.subplots(figsize=(8.5,5.4))
        for family,color in colors.items():
            curves=complete[family,lr,40000].mean(1)
            for k,pol in enumerate(['full history','latest 2k']):
                ax.plot(origins,curves[:,:,k].mean(0),color=color,ls='-' if k==0 else '--',marker='o',label=f'{labels[family]}: {pol}')
                ax.fill_between(origins,curves[:,:,k].min(0),curves[:,:,k].max(0),color=color,alpha=.09)
        ax.plot(origins,bayes,color='#333333',marker='.',label='Bayes: full history')
        ax.plot(origins,wb,color='#777777',ls='--',marker='.',label='Bayes: latest 2k')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(origins,[f'{t//1024}k' for t in origins]);ax.grid(alpha=.2)
        ax.set_xlabel('Observed history at evaluation');ax.set_ylabel('MSE against true conditional mean')
        ax.set_title(f'4×256 · 40k steps · peak LR {lr:g} · warmup + cosine decay')
        ax.legend(fontsize=9,ncol=2)
        fig.text(.5,.01,'Mean of 3 seeds; shaded ranges span seeds. Same 512 validation series.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.03,1,1))
        for ext in ['png','pdf']:fig.savefig(out/f'long_history_lr{lr:g}.{ext}',bbox_inches='tight')
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8.5,5.4))
        for i,(seed,col) in enumerate(zip([185,186,187],['#2B7A78','#3E65A8','#A64B86'])):
            curves=complete['gated_deltanet',lr,40000].mean(1)
            for k,pol in enumerate(['full','latest 2k']):ax.plot(origins,curves[i,:,k],color=col,ls='-' if k==0 else '--',marker='o',label=f'Seed {seed}: {pol}')
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(origins,[f'{t//1024}k' for t in origins]);ax.grid(alpha=.2)
        ax.set_xlabel('Observed history at evaluation');ax.set_ylabel('MSE against true conditional mean');ax.set_title(f'GDN by seed · 40k steps · peak LR {lr:g}')
        ax.legend(fontsize=9,ncol=2);fig.tight_layout()
        fig.savefig(out/f'gdn_by_seed_lr{lr:g}.png',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5.2))
    for family,col in colors.items():
        for lr in [1e-4,3e-4]:
            rs=[x for x in quick if x['family']==family and x['peak_lr']==lr]
            steps=sorted(set(x['step'] for x in rs));sx=[];sy=[]
            for step in steps:
                v=[x['conditional_mean_mse'] for x in rs if x['step']==step]
                if len(v)==3:sx.append(step/1000);sy.append(v)
            if not sy:continue
            y=np.array(sy)
            ax.plot(sx,y.mean(1),color=col,ls='-' if lr==1e-4 else '--',marker='o',label=f'{labels[family]}: peak {lr:g}')
            ax.fill_between(sx,y.min(1),y.max(1),color=col,alpha=.09)
    ax.axhline(bayes[0],color='#333333',ls=':',label='Bayes at 2k history')
    ax.set_yscale('log');ax.set_xlabel('Training steps (thousands)');ax.set_ylabel('Validation MSE against underlying signal')
    ax.set_title('4×256 learning curves at a fixed 2k history');ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',bbox_inches='tight');plt.close(fig)
    seconds=sum(m.get('elapsed_seconds',0) for m in manifests)
    summary=dict(batch=a.batch,checks=checks,aggregates=aggregates,rows=rows,quick_validation=quick,
        bayes=bayes.tolist(),window_bayes=wb.tolist(),estimated_worker_compute_usd=seconds*(.001097+4*.0000131+16*.00000222),
        all_complete=len(checks)==12 and all(c['status']=='finished' and c['checkpoint_steps']==[10000,20000,40000] for c in checks),
        uncertainty='Paired series bootstrap conditional on these trained models; seed ranges are not confidence intervals.',
        confirmation_set_opened=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copyfile(control/'resolved_plan.json',out/'resolved_plan.json')
    (out/'README.md').write_text('# 4×256, 40k-step warmup/cosine sweep\n\n12 runs: Transformer and GDN, peak LRs 1e-4 and 3e-4, seeds 185/186/187. Fresh training, 1k linear warmup from 10% of peak, cosine decay to 10% of peak. 327.68M positions/run, alternating 64/2k training histories.\n\n![Learning curves](learning_curves.png)\n\n![Peak LR 1e-4](long_history_lr0.0001.png)\n\n![Peak LR 3e-4](long_history_lr0.0003.png)\n\nMetrics use the same 512 validation series at fixed origins. No confirmation-set access. Mean curves aggregate three seeds; shading spans seeds. All individual predictions, sources and manifests are in `raw/`. `summary.json` records completion and numerical checks. Checkpoints remain on the personal Modal volume.\n\nNeither 40k steps nor a declining loss establishes convergence. Comparison against earlier 2×64 and 6×256 runs changes multiple settings and is not a depth-only ablation.\n')
    print(json.dumps({'output':str(out),'all_complete':summary['all_complete'],'final':[x for x in aggregates if x['step']==40000 and x['origin'] in [2048,65536]]},indent=2))
if __name__=='__main__':main()
