"""Summarize independent training seeds and fresh confirmation evaluations."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from pilot_report import LABELS,COLORS
BASE=Path('/artifacts/followup_v1');RUNS=Path('/artifacts/pilot_v1/runs')


def aggregate(values):
    a=np.asarray(values,dtype=float);mean=float(a.mean());sd=float(a.std(ddof=1))
    half=float(student_t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    return {'values':a.tolist(),'mean':mean,'seed_sd':sd,'min':float(a.min()),'max':float(a.max()),
            'seed_mean_t95_interval':[mean-half,mean+half],'n_seeds':len(a)}


def gate_stats(path):
    g=torch.load(path,map_location='cpu',weights_only=True)
    logs=torch.stack([g[str(i)]['log_alpha'] for i in range(len(g))])
    alpha=logs.exp()
    per_head=logs[:,:,:64].sum(2).exp().mean(1).reshape(-1).tolist()
    result={'mean_alpha':float(alpha.mean()),'fraction_above_099':float((alpha>.99).float().mean()),
            'mean_scalar_decay_product_64_tokens':float(np.mean(per_head)),
            'decay_product_per_layer_head':per_head}
    if 'beta' in g['0']:result['mean_beta']=float(torch.stack([v['beta'] for v in g.values()]).mean())
    return result


def generate():
    out=BASE/'report';out.mkdir(exist_ok=True)
    manifest=json.loads((BASE/'manifest.json').read_text());horizon=manifest['chosen_horizon']
    records=[];gates={};rows=[]
    for family in LABELS:
        gates[family]={}
        for seed in manifest['seeds']:
            name=manifest['runs'][f'{family}:{seed}'];folder=RUNS/name
            ev=json.loads((BASE/'evaluations'/f'{name}.json').read_text())
            r=json.loads((folder/'result.json').read_text())
            assert r['completed_steps']==horizon and ev['horizon']==horizon
            history={v['step']:v for v in json.loads((folder/'validation.json').read_text())}
            compact_history=[{k:v[k] for k in ('step','normalized_excess','late_normalized_excess','mse')} for _,v in sorted(history.items())]
            records.append({'family':family,'seed':seed,'run_id':name,'run_url':ev['run_url'],
                            'parameters':r['parameters'],'state_elements':r['recurrent_state_elements'],
                            'conv_state_elements':r['conv_state_elements'],'history':compact_history,'evaluation':ev})
            for step,v in ev['checkpoints'].items():
                rows.append({'family':family,'seed':seed,'step':int(step),'normalized_excess':v['test']['normalized_excess'],
                             'late_normalized_excess':v['test']['late_normalized_excess'],
                             'order_k32':v['order_sensitivity']['32']['normalized_change'],'run_url':ev['run_url']})
            for step,path in [(0,folder/'gates_initial.pt'),(2000,folder/'checkpoints/2000_gates_latest.pt'),
                              (2000,folder/'gates_step_2000.pt'),(5000,folder/'gates_step_5000.pt'),
                              (horizon,folder/f'gates_step_{horizon}.pt')]:
                if path.exists():gates[family][f'{seed}:{step}']=gate_stats(path)
    summary={}
    for family in LABELS:
        group=[r for r in records if r['family']==family]
        finals=[r['evaluation']['checkpoints'][str(horizon)] for r in group]
        summary[family]={'normalized_excess':aggregate([r['test']['normalized_excess'] for r in finals]),
                         'late_normalized_excess':aggregate([r['test']['late_normalized_excess'] for r in finals]),
                         'order_k32':aggregate([r['order_sensitivity']['32']['normalized_change'] for r in finals])}
    paired={}
    families=list(LABELS)
    for i,a in enumerate(families):
        for b in families[i+1:]:
            av=summary[a]['normalized_excess']['values'];bv=summary[b]['normalized_excess']['values']
            paired[f'{a}_minus_{b}']=aggregate(np.asarray(av)-np.asarray(bv))
    with np.load(BASE/'data/confirmation.npz') as ds:
        risk=ds['risk'];mu=ds['mu'];deff=ds['deff']
        reference={'zero_predictor_excess':float((mu**2).mean()/risk.mean()),'bayes_risk':float(risk.mean()),
                   'final_context_mean_deff':float(deff[:,-1].mean())}
    ledger=json.loads((BASE/'compute_ledger.json').read_text())
    seconds=sum(r['charged_seconds'] for r in ledger)
    report={'manifest':manifest,'summary':summary,'paired_seed_differences':paired,'reference':reference,
            'records':records,'gates':gates,'cost':{'gpu_call_seconds':seconds,'budget_accounted_seconds':seconds,
            'measured_successful_worker_seconds':sum(r['charged_seconds'] for r in ledger if r['status']=='finished'),
            'failed_call_timeout_reservations_seconds':sum(r['charged_seconds'] for r in ledger if r['status']!='finished'),
            'conservative_compute_usd':seconds/1200,
            'with_overhead_reserve_usd':seconds/1200+.5},
            'uncertainty_note':'SD and t intervals use 3 independent training seeds; the same 1024 confirmation prompts are shared. Three seeds give limited precision.'}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    with (out/'results.csv').open('w') as fp:
        w=csv.DictWriter(fp,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,f in zip(axes.flat,LABELS):
        for r in records:
            if r['family']!=f:continue
            h=r['history'];ax.plot([x['step'] for x in h],[x['normalized_excess'] for x in h],label=f"Seed {r['seed']}")
        ax.set(title=LABELS[f],xlabel='Training steps',ylabel='Validation normalized excess',yscale='log')
        ax.axvline(2000,color='gray',ls=':',alpha=.7);ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Longer training and independent seeds · identical architecture and learning rate',fontsize=14)
    fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for i,f in enumerate(LABELS):
        s=summary[f]['normalized_excess'];a=axes[0]
        a.errorbar(i,s['mean'],yerr=s['seed_sd'],fmt='s',color=COLORS[f],capsize=5,markersize=8)
        a.scatter(np.asarray([i-.12,i,i+.12]),s['values'],color=COLORS[f],s=35,alpha=.65)
        group=[r for r in records if r['family']==f]
        steps=sorted(map(int,group[0]['evaluation']['checkpoints']))
        values=np.array([[r['evaluation']['checkpoints'][str(s)]['test']['normalized_excess'] for s in steps] for r in group])
        axes[1].errorbar(steps,values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,label=LABELS[f],color=COLORS[f])
    axes[0].set_xticks(range(4),[LABELS[f] for f in LABELS],rotation=12)
    axes[0].set(ylabel='Confirmation normalized excess',yscale='log',title=f'{horizon:,} steps · mean ± one training-seed SD')
    axes[1].set(xlabel='Training steps',ylabel='Confirmation normalized excess',yscale='log',title='2k → 5k → 10k · mean ± training-seed SD')
    for a in axes:a.grid(alpha=.2)
    axes[1].legend(frameon=False)
    fig.suptitle('Fresh confirmation set · 1,024 prompts · lower is better',fontsize=14)
    fig.savefig(out/'confirmation_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for f in LABELS:
        finals=[r['evaluation']['checkpoints'][str(horizon)] for r in records if r['family']==f]
        vals=np.array([v['test']['pointwise_normalized_excess'] for v in finals]);x=np.arange(vals.shape[1])
        axes[0].plot(x,vals.mean(0),color=COLORS[f],label=LABELS[f])
        axes[0].fill_between(x,np.maximum(0,vals.mean(0)-vals.std(0,ddof=1)),vals.mean(0)+vals.std(0,ddof=1),color=COLORS[f],alpha=.12)
        ks=[4,8,16,32];orders=np.array([[v['order_sensitivity'][str(k)]['normalized_change'] for k in ks] for v in finals])
        axes[1].errorbar(ks,orders.mean(0),yerr=orders.std(0,ddof=1),color=COLORS[f],label=LABELS[f],marker='o',capsize=3)
    axes[0].set(xlabel='Observed context pairs',ylabel='Normalized excess',title='Prediction gap by context · mean ± seed SD')
    axes[1].set(xlabel='Shuffled context pairs',ylabel='Mean squared change / Bayes risk',title='Ordering · 256 prompts × 5 permutations per seed')
    for a in axes:a.grid(alpha=.2);a.legend(frameon=False,fontsize=9)
    fig.savefig(out/'context_and_order.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,f in zip(axes,('mamba2','gated_deltanet')):
        for seed,marker in zip(manifest['seeds'],('o','s','^')):
            series=sorted((int(k.split(':')[1]),v) for k,v in gates[f].items() if k.startswith(f'{seed}:'))
            ax.plot([p for p,_ in series],[v['mean_scalar_decay_product_64_tokens'] for _,v in series],marker=marker,label=f'Seed {seed}')
        ax.set(title=LABELS[f],xlabel='Training steps',ylabel='Mean product of scalar decay gates over 64 tokens',ylim=(-.03,1.03))
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('Scalar decay changes with training · not total information retention',fontsize=14)
    fig.savefig(out/'gate_evolution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,f in zip(axes,('mamba2','gated_deltanet')):
        columns=[];names=[]
        for seed in manifest['seeds']:
            for step in (0,horizon):
                columns.append(gates[f][f'{seed}:{step}']['decay_product_per_layer_head'])
                names.append(f'{seed} / {step//1000}k')
        grid=np.array(columns).T
        im=ax.imshow(grid,aspect='auto',vmin=0,vmax=1,cmap='viridis')
        ax.set_xticks(range(len(names)),names,rotation=30,ha='right')
        ax.set_yticks(range(16),[f'L{i//4+1} H{i%4+1}' for i in range(16)])
        ax.set(title=LABELS[f],xlabel='Training seed / checkpoint',ylabel='Layer and head')
        fig.colorbar(im,ax=ax,label='Mean scalar decay product over 64 tokens')
    fig.suptitle('Head-level decay remains distinct from whole-model information retention',fontsize=14)
    fig.savefig(out/'gate_heads.png',dpi=180);plt.close(fig)
    with zipfile.ZipFile(out/'project182_followup_results.zip','w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in out.iterdir():
            if p.suffix!='.zip':z.write(p,p.name)
        for pattern in ('manifest.json','compute_ledger.json','evaluations/*.json','predictions/*.npz','data/*.npz','jobs/*.json'):
            for p in BASE.glob(pattern):z.write(p,str(p.relative_to(BASE)))
        for r in records:
            folder=RUNS/r['run_id']
            for pattern in ('config.yaml','stage_configs/*.yaml','validation.json','gates_*.pt','checkpoints/*gates*.pt'):
                for p in folder.glob(pattern):z.write(p,'runs/'+r['run_id']+'/'+str(p.relative_to(folder)))
        checks=Path('/artifacts/pilot_v1/checks.json')
        if checks.exists():z.write(checks,'checks/original_pilot_checks.json')
        for p in Path(__file__).parent.glob('*.py'):z.write(p,'source/src/'+p.name)
        for p in (Path(__file__).parent/'conf').rglob('*.yaml'):z.write(p,'source/src/'+str(p.relative_to(Path(__file__).parent)))
    return {'summary':summary,'cost':report['cost'],'files':[p.name for p in out.iterdir()]}
