"""Saved-log comparison of 2x64 and 6x256 runs; no new training/evaluation."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/2026-11-24-learning-curves';OUT.mkdir(parents=True,exist_ok=True)
SIX=ROOT/'results/2026-11-24-reference6x256/raw'
series={};records=[];sources=[]
for depth in [2,6]:
    for family in ['tf','gdn']:
        train={};valid={}
        for seed in [182,183,184]:
            if depth==6:
                directory=SIX/f'alex_261124_ref6x256_192325_{family}_main_s{seed}'
            elif family=='tf':directory=ROOT/f'outputs/forecast_long_artifacts/train_s{seed}'
            else:directory=ROOT/f'outputs/forecast_optimization/alex_261123_ar1_opt30k_cosine_s{seed}'
            path=directory/'training.json';history=json.loads(path.read_text())
            sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            # Common 2k-history logging subset; six-layer logs otherwise sample
            # only even updates, unlike the alternating 25-step legacy logs.
            filtered=[r for r in history if r['length']==2048]
            bins=sorted(set((r['step']-1)//1000 for r in filtered))
            train[seed]=np.array([(1000*b+500,np.mean([r['loss'] for r in filtered if (r['step']-1)//1000==b])) for b in bins])
            valid[seed]={}
            pattern='metrics_step_*.json' if depth==6 else 'validation_step_*.json'
            for path in directory.glob(pattern):
                step=int(path.stem.split('_')[-1]);data=json.loads(path.read_text())
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                rows=data['rows'] if depth==6 else data
                row=next(r for r in rows if r.get('origin',r.get('length'))==2048)
                loss=row['full_error'] if depth==6 else row['conditional_mean_error']
                valid[seed][step]=loss
                records.append({'depth':depth,'width':256 if depth==6 else 64,'family':family,'seed':seed,'step':step,'prediction_positions_seen':step*(8192 if depth==6 else 2048),'validation_2k_conditional_mean_mse':loss})
        steps=sorted(set.intersection(*(set(v) for v in valid.values())))
        series[depth,family]={'training':train,'steps':np.array(steps),'validation':np.array([[valid[s][t] for t in steps] for s in [182,183,184]])}

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
colors={'tf':'#2166AC','gdn':'#CC641D'}
labels={'tf':'Transformer','gdn':'GDN'}
foot='2×64: 2,048 positions/step; LR 3e−4 → 3e−5 after step 10k.\n6×256: 8,192 positions/step; constant LR 1e−4. Mean of 3 seeds; shading spans seeds.'
for axis_kind in ['steps','positions']:
    fig,ax=plt.subplots(figsize=(9,5.6))
    for (depth,family),data in series.items():
        x=data['steps']/1000 if axis_kind=='steps' else data['steps']*(8192 if depth==6 else 2048)/1e6
        y=data['validation']
        ax.plot(x,y.mean(0),color=colors[family],ls='-' if depth==6 else '--',marker='o',lw=2,label=f'{labels[family]} {depth}×{256 if depth==6 else 64}')
        ax.fill_between(x,y.min(0),y.max(0),color=colors[family],alpha=.09)
    with np.load(SIX/'alex_261124_ref6x256_192325_tf_main_s182/reference.npz') as r:
        bayes=float(((r['bayes_mean'][:,0]-r['latent_mean'][:,0])**2).mean())
    ax.axhline(bayes,color='#444444',ls=':',lw=1.8,label='Bayes at 2k history')
    ax.set_yscale('log');ax.grid(alpha=.2)
    ax.set_xlabel('Training steps (thousands)' if axis_kind=='steps' else 'Training prediction positions (millions)')
    ax.set_ylabel('Validation MSE against underlying signal')
    ax.set_title('Learning curves at a fixed 2k evaluation history')
    ax.legend(fontsize=9,ncol=2)
    fig.text(.5,.015,foot,ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,1))
    for ext in ['png','pdf']:fig.savefig(OUT/f'validation_by_{axis_kind}.{ext}',bbox_inches='tight')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5.4))
for (depth,family),data in series.items():
    curves=[data['training'][seed] for seed in [182,183,184]]
    for c in curves[1:]:np.testing.assert_array_equal(c[:,0],curves[0][:,0])
    x=curves[0][:,0]/1000;y=np.array([c[:,1] for c in curves])
    ax.plot(x,y.mean(0),color=colors[family],ls='-' if depth==6 else '--',lw=2,label=f'{labels[family]} {depth}×{256 if depth==6 else 64}')
    ax.fill_between(x,y.min(0),y.max(0),color=colors[family],alpha=.09)
ax.axhline(.25,color='#444444',ls=':',label='Innovation noise variance (0.25)')
ax.set_xlabel('Training steps (thousands)');ax.set_ylabel('Noisy next-observation training MSE')
ax.set_title('Training loss: logged 2k-history batches, averaged in 1k-step bins')
ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
fig.text(.5,.015,'Logged batches only, not all training updates. Mean of 3 seeds; shading spans seeds.\nNoise dominates this loss; validation above resolves much smaller prediction errors.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.09,1,1))
for ext in ['png','pdf']:fig.savefig(OUT/f'training_loss.{ext}',bbox_inches='tight')
plt.close(fig)
with (OUT/'validation_points.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary=[]
for (depth,family),data in series.items():
    for i,s in enumerate(data['steps']):
        summary.append({'depth':depth,'family':family,'step':int(s),'mean_validation_error':float(data['validation'][:,i].mean())})
(OUT/'summary.json').write_text(json.dumps({'sources':sources,'validation':summary,'bayes_at_2k':bayes,'caveats':['Not a depth-only ablation: width, backbone, precision, batch size and LR schedule differ.','Old models: 30k steps / 61.44M positions; new models: 20k steps / 163.84M positions.','Both use the same fixed 512 validation series at 2k history.','Lines connect measured validation checkpoints; no intermediate validation values inferred.','Training curves average logged 2k-history batches in nonoverlapping 1k-step bins.']},indent=2)+'\n')
(OUT/'README.md').write_text('# Saved learning curves: 2×64 versus 6×256\n\n![Validation versus steps](validation_by_steps.png)\n\n![Training loss](training_loss.png)\n\n![Validation versus training positions](validation_by_positions.png)\n\nThese comparisons use the main three seeds, not the single-seed high-LR sensitivity arms. The small models use cosine decay after 10k steps; the larger models use constant LR. Width, implementation, precision and batch size also differ. This is not a depth-only comparison. Training plots use only logged 2k-history batches and 1k-step averaging. Validation plots show the same 512 evaluation series and a fixed 2k history. Source paths/hashes and exact points are in `summary.json` and `validation_points.csv`.\n')
print(json.dumps(summary,indent=2))
