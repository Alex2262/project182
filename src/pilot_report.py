"""Create portable pilot figures and a machine-readable result bundle on CPU."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'gpt2':'Transformer','linear_attention':'Linear attention','mamba2':'Mamba-2','gated_deltanet':'Gated DeltaNet'}
COLORS={'gpt2':'#3156b8','linear_attention':'#d08b16','mamba2':'#23836b','gated_deltanet':'#a44791'}


def generate(root='/artifacts/pilot_v1'):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    import yaml
    runs=[]
    for path in sorted((root/'runs').glob('*/result.json')):
        r=json.loads(path.read_text());c=yaml.safe_load((path.parent/'config.yaml').read_text())
        r['family']=c['model']['family'];r['lr']=c['training']['learning_rate']
        r['history']=json.loads((path.parent/'validation.json').read_text())
        r['directory']=str(path.parent)
        ev=path.parent/'evaluation.json'
        if ev.exists():r['evaluation']=json.loads(ev.read_text())
        runs.append(r)
    selected={}
    for f in LABELS:
        candidates=[r for r in runs if r['family']==f]
        if candidates:
            max_steps=max(r['completed_steps'] for r in candidates)
            selected[f]=min([r for r in candidates if r['completed_steps']==max_steps],
                            key=lambda r:r['validation']['normalized_excess'])
    # Resample whole prompts, preserving dependence among prefix predictions.
    # These intervals quantify test-prompt uncertainty, not training-seed variance.
    statistics={}
    if (root/'data/test.npz').exists():
        with np.load(root/'data/test.npz') as ds:
            mu=ds['mu']; risk=ds['risk']; ys=ds['ys']
        rng=np.random.default_rng(7182)
        indices=rng.integers(0,len(mu),size=(2000,len(mu)))
        denominator=risk.mean(1)
        def summarize(pred):
            numerator=((pred-mu)**2).mean(1)
            boot=numerator[indices].mean(1)/denominator[indices].mean(1)
            return {'normalized_excess':float(numerator.mean()/denominator.mean()),
                    'prompt_bootstrap_95pct':np.quantile(boot,[.025,.975]).tolist()}
        statistics['zero_predictor']=summarize(np.zeros_like(mu))
        statistics['bayes_risk']=float(risk.mean())
        statistics['empirical_bayes_mse']=float(((ys-mu)**2).mean())
        statistics['interval_scope']='256 held-out prompts; 2000 whole-prompt bootstrap replicates; one training seed'
        for f,r in selected.items():
            path=Path(r['directory'])/'test_predictions.npz'
            if path.exists():
                with np.load(path) as ds:pred=ds['pred']
                statistics[f]=summarize(pred)
    (out/'statistics.json').write_text(json.dumps(statistics,indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for f,r in selected.items():
        # Keep last entry for a step (checkpoint resumes can repeat validation).
        hist={h['step']:h for h in r['history']}
        steps=sorted(hist)
        axes[0].plot(steps,[hist[s]['normalized_excess'] for s in steps],label=LABELS[f],color=COLORS[f])
        if 'evaluation' in r:
            m=r['evaluation']['test']
            axes[1].plot(range(len(m['pointwise_mse'])),m['pointwise_mse'],label=LABELS[f],color=COLORS[f])
    if any('evaluation' in r for r in selected.values()):
        m=next(r['evaluation']['test'] for r in selected.values() if 'evaluation' in r)
        axes[1].plot(range(len(m['pointwise_bayes_risk'])),m['pointwise_bayes_risk'],color='black',ls='--',label='Bayes risk')
    axes[0].set(xlabel='Training steps',ylabel='Validation normalized excess error',title='Learning progress (lower is better)')
    axes[1].set(xlabel='Observed context pairs',ylabel='Test mean squared error',title='Using more examples')
    for ax in axes:ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
    fig.suptitle('Linear regression pilot · d=5 · noise std=0.5 · one training seed',fontsize=14)
    fig.savefig(out/'learning_and_context.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for f,r in selected.items():
        if 'evaluation' not in r:continue
        m=r['evaluation']['test'];o=r['evaluation']['order_sensitivity'];ks=sorted(map(int,o))
        axes[0].plot(range(len(m['pointwise_normalized_excess'])),m['pointwise_normalized_excess'],color=COLORS[f],label=LABELS[f])
        axes[1].plot(ks,[o[str(k)]['normalized_change'] for k in ks],marker='o',color=COLORS[f],label=LABELS[f])
    axes[0].set(xlabel='Observed context pairs',ylabel='Squared disagreement / Bayes risk',title='Distance from the optimal predictor')
    axes[1].set(xlabel='Shuffled context pairs (fixed query)',ylabel='Mean squared change / Bayes risk',title='Order sensitivity · 5 permutations × 64 prompts')
    for ax in axes:ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
    fig.savefig(out/'excess_and_order.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,f in zip(axes.flat,LABELS):
        for r in runs:
            if r['family']!=f:continue
            hist={h['step']:h for h in r['history']};steps=sorted(hist)
            ax.plot(steps,[hist[s]['normalized_excess'] for s in steps],label=f"lr={r['lr']:g}")
        ax.set(title=LABELS[f],xlabel='Training steps',ylabel='Validation normalized excess')
        ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.suptitle('Learning-rate pilot · late learning makes short-run selection uncertain',fontsize=14)
    fig.savefig(out/'learning_rate_comparison.png',dpi=180);plt.close(fig)
    gate_summary={};fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,f in enumerate(('mamba2','gated_deltanet')):
        if f not in selected:continue
        r=selected[f];gate_summary[f]={}
        for col,phase in enumerate(('initial','latest')):
            path=Path(r['directory'])/f'gates_{phase}.pt'
            if not path.exists():continue
            gates=torch.load(path,map_location='cpu',weights_only=True)
            alpha=torch.stack([gates[str(i)]['alpha'] for i in range(len(gates))])
            logs=torch.stack([gates[str(i)]['log_alpha'] for i in range(len(gates))])
            # Axes: layer, prompt, token position, head. Pair retention uses x,y gates.
            pairs=logs.reshape(logs.shape[0],logs.shape[1],-1,2,logs.shape[-1]).sum(3).exp()
            grid=pairs.mean(1).permute(0,2,1).reshape(-1,pairs.shape[2]).numpy()
            im=axes[row,col].imshow(grid,aspect='auto',vmin=0,vmax=1,cmap='viridis')
            axes[row,col].set(title=f'{LABELS[f]} · {phase}',xlabel='Pair position',ylabel='Layer / head')
            fig.colorbar(im,ax=axes[row,col],label='Mean retention across x and y')
            summary={'mean_alpha':float(alpha.mean()),'mean_pair_retention':float(pairs.mean()),
                     'mean_retention_over_32_pairs':float(logs[:,:,:64].sum(2).exp().mean()),
                     'fraction_alpha_above_099':float((alpha>.99).float().mean())}
            if 'beta' in gates['0']:
                summary['mean_beta']=float(torch.stack([v['beta'] for v in gates.values()]).mean())
            gate_summary[f][phase]=summary
    fig.suptitle('Retention gates on fixed validation prompts · not a causal forgetting test',fontsize=14)
    fig.savefig(out/'retention_gates.png',dpi=180);plt.close(fig)
    (out/'gate_summary.json').write_text(json.dumps(gate_summary,indent=2))
    rows=[]
    for r in runs:
        row={'architecture':LABELS[r['family']],'learning_rate':r['lr'],'steps':r['completed_steps'],
             'validation_normalized_excess':r['validation']['normalized_excess'],
             'test_normalized_excess':r.get('evaluation',{}).get('test',{}).get('normalized_excess',''),
             'parameters':r['parameters'],'state_elements':r['recurrent_state_elements'],
             'conv_state_elements':r['conv_state_elements'],
             'seconds_per_step':r['seconds_per_step'],'run_url':r['run_url']}
        rows.append(row)
    with (out/'results.csv').open('w') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    (out/'results.json').write_text(json.dumps({'runs':runs,'gates':gate_summary},indent=2))
    with zipfile.ZipFile(out/'project182_pilot_results.zip','w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for path in out.iterdir():
            if path.suffix!='.zip':z.write(path,path.name)
        for pattern in ('checks.json','compute_ledger.json','data/*.npz','runs/*/config.yaml','runs/*/validation.json','runs/*/evaluation.json','runs/*/test_predictions.npz','runs/*/stage_configs/*.yaml','runs/*/gates_*.pt'):
            for path in root.glob(pattern):z.write(path,str(path.relative_to(root)))
        for path in Path(__file__).parent.glob('*.py'):
            z.write(path,'source/src/'+path.name)
        for path in (Path(__file__).parent/'conf').rglob('*.yaml'):
            z.write(path,'source/src/'+str(path.relative_to(Path(__file__).parent)))
    return {'statistics':statistics,'files':[p.name for p in out.iterdir()],'rows':rows,'gates':gate_summary}
