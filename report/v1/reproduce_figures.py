"""Rebuild the report's figures from committed results; no checkpoints or GPU needed."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'figures'
BASE=ROOT/'results/2026-11-24-reference4x256-40k'
ADAPT=ROOT/'results/2026-11-24-long-context-adaptation'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
COL={'transformer_nope':'#2166ac','gated_deltanet':'#d66013'}
NAMES={'transformer_nope':'Transformer','gated_deltanet':'GDN'}
seeds=[185,186,187]
def base_dir(f,s):
 return next((BASE/'raw').glob(f'*_{"tf" if f=="transformer_nope" else "gdn"}_lr1e4_s{s}'))
def errors(d,step):
 with np.load(d/'reference.npz') as r,np.load(d/f'predictions_step_{step}.npz') as p:
  assert p['prediction'].shape==(512,6,2)
  return ((p['prediction']-r['latent_mean'][:,:,None])**2).mean(0)
with np.load(base_dir('gated_deltanet',185)/'reference.npz') as r:
 lengths=r['lengths'];bayes=((r['bayes_mean']-r['latent_mean'])**2).mean(0);window=((r['window_bayes_mean']-r['latent_mean'])**2).mean(0)
def save(fig,name):
 fig.tight_layout()
 for ext in ['pdf','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',dpi=180)
 plt.close(fig)
def history(ax):
 ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xticks(lengths,[f'{n//1024}k' for n in lengths]);ax.set_xlabel('Observed history (transition tokens)');ax.set_ylabel('MSE against oracle conditional mean');ax.grid(alpha=.2)
fig,ax=plt.subplots(figsize=(7.4,4.1))
for fam in COL:
 y=np.stack([errors(base_dir(fam,s),40000) for s in seeds])
 for k,policy in enumerate(['full history','latest 2k']):
  ax.plot(lengths,y[:,:,k].mean(0),color=COL[fam],ls='-' if k==0 else '--',marker='o',ms=4,label=f'{NAMES[fam]}: {policy}')
  if k==0:ax.fill_between(lengths,y[:,:,k].min(0),y[:,:,k].max(0),color=COL[fam],alpha=.10)
ax.plot(lengths,bayes,color='#333333',marker='.',label='Bayes: full history');ax.plot(lengths,window,color='#888888',ls='--',marker='.',label='Bayes: latest 2k')
history(ax);ax.legend(ncol=2,fontsize=9.7,loc='upper left');save(fig,'history_40k')
fig,axs=plt.subplots(1,2,figsize=(7.4,2.8))
q=list(csv.DictReader((BASE/'quick_validation.csv').open()))
for fam in COL:
 curves=[]
 for s in seeds:
  rows=[r for r in json.loads((base_dir(fam,s)/'training.json').read_text()) if r['length']==2048]
  curves.append([np.mean([r['loss'] for r in rows if lo<r['step']<=lo+2000]) for lo in range(0,40000,2000)])
 y=np.array(curves);x=np.arange(1000,40000,2000)
 axs[0].plot(x/1000,y.mean(0),c=COL[fam],label=NAMES[fam]);axs[0].fill_between(x/1000,y.min(0),y.max(0),color=COL[fam],alpha=.12)
 y=np.array([[float(next(r['conditional_mean_mse'] for r in q if r['family']==fam and float(r['peak_lr'])==.0001 and int(r['seed'])==s and int(r['step'])==t)) for t in [10000,20000,30000,40000]] for s in seeds])
 axs[1].plot([10,20,30,40],y.mean(0),color=COL[fam],marker='o',ms=3,label=NAMES[fam]);axs[1].fill_between([10,20,30,40],y.min(0),y.max(0),color=COL[fam],alpha=.12)
axs[0].axhline(.25,c='#888',ls=':',label='Innovation variance');axs[0].set_ylabel('Noisy-target training MSE');axs[0].set_title('Training: logged 2k batches',fontsize=10);axs[0].legend(fontsize=8.5)
axs[1].set_yscale('log');axs[1].set_ylabel('Conditional-mean MSE');axs[1].set_title('Validation: 2k history',fontsize=10)
for ax in axs:ax.set_xlabel('Training steps (thousands)');ax.grid(alpha=.2)
save(fig,'learning_curves')
fig,ax=plt.subplots(figsize=(7.4,3.6))
for s,col in zip([185,186],['#2166ac','#d66013']):
 for kind,ls,label in [('long_adaptation','-','64k training'),('reset_adaptation','--','2k-reset control')]:
  d=next((ADAPT/'raw').glob(f'*_{kind}_s{s}'))
  y=[errors(base_dir('gated_deltanet',s),40000)[-1,0]]+[errors(d,t)[-1,0] for t in [1000,2500,5000]]
  ax.plot([0,1000,2500,5000],y,color=col,ls=ls,marker='o',ms=4,label=f'Seed {s}: {label}')
ax.axhline(bayes[-1],color='#333',ls=':',label='Bayes at 64k');ax.set_yscale('log');ax.set_xlabel('Additional training steps');ax.set_ylabel('64k conditional-mean MSE');ax.grid(alpha=.2);ax.legend(fontsize=9.7,ncol=2,loc='lower left',bbox_to_anchor=(.23,.60));save(fig,'adaptation_paired')
fig,ax=plt.subplots(figsize=(7.4,4.2))
for kind,col,label in [('long_adaptation','#16816b','64k training'),('reset_adaptation','#ac3973','2k-reset control')]:
 y=np.stack([errors(next((ADAPT/'raw').glob(f'*_{kind}_s{s}')),5000) for s in [185,186]])
 for k,policy in enumerate(['full history','latest 2k']):
  ax.plot(lengths,y[:,:,k].mean(0),color=col,ls='-' if k==0 else '--',marker='o',ms=4,label=f'{label}: {policy}')
  if k==0:ax.fill_between(lengths,y[:,:,k].min(0),y[:,:,k].max(0),color=col,alpha=.10)
ax.plot(lengths,bayes,color='#333',label='Bayes: full history');ax.plot(lengths,window,color='#888',ls='--',label='Bayes: latest 2k');history(ax);ax.legend(fontsize=9.7,ncol=2,loc='lower left',bbox_to_anchor=(0,1.01));save(fig,'adaptation_history')
print('Wrote four report figures from committed raw predictions and training logs.')
