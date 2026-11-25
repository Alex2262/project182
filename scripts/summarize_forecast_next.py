"""Recompute the ten-job 80k/adaptation batch and its paired controls from saved predictions."""
import csv,hashlib,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
BATCH='alex_261124_next_205830'
CONTROL=ROOT/'outputs/forecast_reference_train_control'/BATCH
OUT=ROOT/'results/2026-11-24-long-context-adaptation'
SEEDS=[185,186]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((CONTROL/'resolved_plan.json').read_text())
    ref=None;groups={};rows=[];checks=[];manifests=[];quick=[];ab=[];ab_checks=[]
    for c in plan['configs']:
        name=c['run_name'];d=CONTROL/'artifacts'/name;m=json.loads((d/'manifest.json').read_text());manifests.append(m)
        assert m['status']=='finished'
        for f,sha in c['source_sha256'].items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==sha
        dst=OUT/'raw'/name;dst.mkdir(parents=True,exist_ok=True)
        for f in d.iterdir():
            if f.suffix in ['.json','.npz','.py']:shutil.copyfile(f,dst/f.name)
        seed=c['training']['seed'];kind=c['kind'];family=c['model']['family']
        check={'run':name,'status':m['status'],'steps':m['completed_steps'],'source_hashes_verified':True,'checkpoints':[]}
        checks.append(check)
        if kind=='layer_ablation':
            with np.load(d/'ablation_predictions.npz') as z:
                pred=z['prediction'];latent=z['latent_mean'];fresh=z['fresh'];origins=z['origins'];conditions=z['conditions']
            assert pred.shape==(64,2,8) and np.isfinite(pred).all()
            np.testing.assert_array_equal(pred[:,:,-1],fresh)
            stored=json.loads((d/'ablation_metrics.json').read_text());err=(pred-latent[:,:,None])**2
            for j,origin in enumerate(origins):
                for k,condition in enumerate(conditions):
                    r=next(x for x in stored if x['origin']==origin and x['condition']==condition)
                    np.testing.assert_allclose(err[:,j,k].mean(),r['conditional_mean_mse'],rtol=0,atol=1e-14)
                    ab.append(dict(seed=seed,origin=int(origin),condition=str(condition),mse=float(err[:,j,k].mean()),ratio_vs_carry=float(err[:,j,k].mean()/err[:,j,0].mean())))
            ab_checks.append({'seed':seed,'all_states_equals_fresh':True})
            continue
        with np.load(d/'reference.npz') as z:r={k:z[k] for k in z.files}
        if ref is None:ref=r
        for key in ['theta','latent_mean','lengths']:np.testing.assert_array_equal(r[key],ref[key])
        for key in ['bayes_mean','window_bayes_mean']:np.testing.assert_allclose(r[key],ref[key],rtol=0,atol=1e-12)
        q=json.loads((d/'quick_validation.json').read_text())
        quick.extend(dict(family=family,kind=kind,seed=seed,**x) for x in q)
        for step in c['training']['checkpoint_steps']:
            with np.load(d/f'predictions_step_{step}.npz') as z:pred=z['prediction'];origins=z['origins']
            assert pred.shape==(512,6,2) and np.isfinite(pred).all()
            np.testing.assert_array_equal(origins,ref['lengths']);np.testing.assert_array_equal(pred[:,0,0],pred[:,0,1])
            err=(pred-ref['latent_mean'][:,:,None])**2
            stored=json.loads((d/f'metrics_step_{step}.json').read_text());assert stored['status']=='finished'
            for j,origin in enumerate(origins):
                np.testing.assert_allclose(err[:,j].mean(0),[stored['rows'][j]['full_error'],stored['rows'][j]['window_error']],rtol=0,atol=1e-14)
                for k,policy in enumerate(['full','window2048']):rows.append(dict(family=family,kind=kind,seed=seed,step=step,origin=int(origin),policy=policy,mse=float(err[:,j,k].mean())))
            for x in q:
                if x['step']==step:np.testing.assert_allclose(x['conditional_mean_mse'],err[:,0,0].mean(),rtol=0,atol=1e-12)
            groups.setdefault((family,kind,step),{})[seed]=err
            check['checkpoints'].append(step)
    # Parent reference: restrict to the SAME TWO seeds, not the previous three-seed mean.
    old=ROOT/'results/2026-11-24-reference4x256-40k/raw'
    parent={}
    for seed in SEEDS:
        d=old/f'alex_261124_ref4x256_40k_200555_gdn_lr1e4_s{seed}'
        with np.load(d/'reference.npz') as z:np.testing.assert_array_equal(z['latent_mean'],ref['latent_mean'])
        with np.load(d/'predictions_step_40000.npz') as z:parent[seed]=(z['prediction']-ref['latent_mean'][:,:,None])**2
        dst=OUT/'parents'/d.name;dst.mkdir(parents=True,exist_ok=True)
        for f in ['manifest.json','reference.npz','predictions_step_40000.npz','metrics_step_40000.json']:shutil.copyfile(d/f,dst/f)
    groups['gated_deltanet','parent',40000]=parent
    arrays={k:np.stack([v[s] for s in SEEDS]) for k,v in groups.items()}
    origins=ref['lengths'];bayes=((ref['bayes_mean']-ref['latent_mean'])**2).mean(0);wb=((ref['window_bayes_mean']-ref['latent_mean'])**2).mean(0)
    styles=[(('gated_deltanet','parent',40000),'GDN: original 40k','#888888',':'),
            (('gated_deltanet','fresh',80000),'GDN: fresh 80k','#CC641D','-'),
            (('gated_deltanet','reset_adaptation',5000),'GDN: +5k matched reset training','#AC3973','-'),
            (('transformer_nope','fresh',80000),'Transformer: fresh 80k','#2166AC','-'),
            (('gated_deltanet','long_adaptation',5000),'GDN: +5k long-context training','#16816B','-')]
    plt.rcParams.update({'font.size':11,'figure.dpi':170,'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):
        fig.tight_layout(rect=(0,.045,1,1))
        for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
    def history_axes(ax):
        ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(origins,[f'{int(x)//1024}k' for x in origins]);ax.set_xlabel('Observed history at evaluation');ax.set_ylabel('MSE against true conditional mean');ax.grid(alpha=.2)
    fig,ax=plt.subplots(figsize=(10.5,6.3))
    for key,label,col,ls in styles:
        y=arrays[key].mean(1)[:,:,0]
        ax.plot(origins,y.mean(0),color=col,ls=ls,marker='o',label=label);ax.fill_between(origins,y.min(0),y.max(0),color=col,alpha=.09)
    ax.plot(origins,bayes,color='#222222',marker='.',label='Bayes: full history')
    history_axes(ax);ax.set_title('Long-context adaptation versus matched reset training\n4×256 · all curves use full history at evaluation',fontsize=13);ax.legend(fontsize=9,loc='lower left',ncol=2)
    fig.text(.5,.01,'Means of seeds 185/186; shading spans seeds. Same 512 series.\nAll trained endpoints use 655.36M positions; original 40k uses 327.68M.',ha='center',fontsize=9)
    finish(fig,'long_history_comparison')
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    for seed,col in zip(SEEDS,['#2166AC','#CC641D']):
        for kind,ls,label in [('long_adaptation','-','64k training'),('reset_adaptation','--','matched 2k-reset training')]:
            values=[parent[seed][:,-1,0].mean()]+[groups['gated_deltanet',kind,t][seed][:,-1,0].mean() for t in [1000,2500,5000]]
            ax.plot([0,1000,2500,5000],values,color=col,ls=ls,marker='o',label=f'Seed {seed}: {label}')
    ax.axhline(bayes[-1],color='#333',ls=':',label='Bayes at 64k');ax.set_yscale('log');ax.grid(alpha=.2)
    ax.set_xlabel('Additional adaptation steps');ax.set_ylabel('64k full-history MSE against true conditional mean');ax.set_title('Identical starting checkpoints and adaptation data\nLong history versus resetting every 2k',fontsize=13);ax.legend(fontsize=9,ncol=2,loc='lower left')
    fig.text(.5,.01,'One 64k stream per step in both arms; same targets, optimizer steps and LR schedule.',ha='center',fontsize=9)
    finish(fig,'adaptation_by_seed')
    fig,ax=plt.subplots(figsize=(9.5,5.8))
    for kind,col,label in [('long_adaptation','#16816B','64k adaptation'),('reset_adaptation','#AC3973','Matched reset training')]:
        y=arrays['gated_deltanet',kind,5000].mean(1)
        for k,policy in enumerate(['full history','latest 2k']):
            ax.plot(origins,y[:,:,k].mean(0),color=col,ls='-' if k==0 else '--',marker='o',label=f'{label}: {policy}')
            ax.fill_between(origins,y[:,:,k].min(0),y[:,:,k].max(0),color=col,alpha=.08)
    ax.plot(origins,bayes,color='#222',label='Bayes: full history');ax.plot(origins,wb,color='#777',ls='--',label='Bayes: latest 2k')
    history_axes(ax);ax.set_title('Training intervention versus evaluation-time history restriction',fontsize=13);ax.legend(fontsize=9,ncol=2,loc='lower left')
    fig.text(.5,.01,'Means of seeds 185/186; shading spans seeds. Dashed curves reset only at evaluation.',ha='center',fontsize=9)
    finish(fig,'adaptation_eval_policies')
    fig,ax=plt.subplots(figsize=(10,5.6))
    conditions=['carry','layer_1','layer_2','layer_3','layer_4','all_recurrent','conv_only','all_states']
    labels=['Keep all','Reset L1','Reset L2','Reset L3','Reset L4','All recurrent','Conv only','All state']
    for seed,col in zip(SEEDS,['#2166AC','#CC641D']):
        y=[next(x['ratio_vs_carry'] for x in ab if x['seed']==seed and x['origin']==65536 and x['condition']==k) for k in conditions]
        ax.plot(range(8),y,color=col,marker='o',label=f'Seed {seed}')
    ax.axhline(1,color='#777',ls=':');ax.set_yscale('log');ax.set_xticks(range(8),labels,rotation=20,ha='right');ax.set_ylabel('64k MSE / unchanged-state MSE');ax.set_title('Frozen 40k GDN: selective state resets before the final 2k');ax.legend();ax.grid(alpha=.2)
    fig.text(.5,.01,'Same final 2k inputs and query; first 64 series of the fixed cohort. Below 1 means improvement.',ha='center',fontsize=9)
    finish(fig,'layer_reset_64k')
    rng=np.random.default_rng(26112481);ix=rng.integers(0,512,(5000,512));paired=[]
    for j,origin in enumerate(origins):
        d=(arrays['gated_deltanet','long_adaptation',5000][:,:,j,0]-arrays['gated_deltanet','reset_adaptation',5000][:,:,j,0]).mean(0)
        ci=np.quantile(d[ix].mean(1),[.025,.975])
        paired.append(dict(origin=int(origin),long_minus_reset=float(d.mean()),conditional_series_ci_low=float(ci[0]),conditional_series_ci_high=float(ci[1])))
    finals=[]
    for key,label,col,ls in styles:
        e=arrays[key]
        for j,n in enumerate(origins):
            finals.append(dict(model=label,origin=int(n),full=float(e[:,:,j,0].mean()),window=float(e[:,:,j,1].mean()),per_seed_full={str(seed):float(e[i,:,j,0].mean()) for i,seed in enumerate(SEEDS)}))
    for filename,rs in [('metrics.csv',rows),('final_comparisons.csv',finals),('adaptation_paired.csv',paired),('layer_ablation.csv',ab),('quick_validation.csv',quick)]:
        with (OUT/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
    assert len(manifests)==10 and len(rows)==288 and sum(len(x['checkpoints']) for x in checks)==24
    verification=dict(all_complete=True,jobs=10,training_jobs=8,checkpoint_evaluations=24,ablation_jobs=2,source_hashes_verified=True,saved_metrics_recomputed=True,full_equals_window_at_2k=True,identical_validation_series=True,ablation_checks=ab_checks,checks=checks,containers_remaining=0)
    summary=dict(batch=BATCH,verification=verification,finals=finals,paired=paired,bayes=bayes.tolist(),window_bayes=wb.tolist(),estimated_worker_compute_usd=sum(m['elapsed_seconds'] for m in manifests)*(.001097+4*.0000131+16*.00000222),uncertainty='Two paired seeds. Shading is min/max across seeds; bootstrap resamples series conditional on these trained models.',confirmation_set_opened=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    shutil.copyfile(CONTROL/'resolved_plan.json',OUT/'resolved_plan.json');shutil.copyfile(CONTROL/'cloud_call_verification.json',OUT/'cloud_call_verification.json')
    print(json.dumps({'output':str(OUT),'all_complete':True,'compute_estimate':summary['estimated_worker_compute_usd'],'finals':[x for x in finals if x['origin'] in [2048,65536]]},indent=2))

if __name__=='__main__':main()
