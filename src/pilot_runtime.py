"""Pilot orchestration; all outputs are on the persistent project Volume."""
import json,time,random,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import numpy as np
import torch
from munch import Munch
from pilot_models import RegressionModel
from pilot_eval import fixed_dataset,verify_bayes,metrics,predictions,order_sensitivity

ROOT=Path('/artifacts/pilot_v1')
FAMILIES=('gpt2','linear_attention','mamba2','gated_deltanet')


def config(family,lr=.0003,steps=250):
    import hashlib
    source_hash=hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        source_hash.update(source.name.encode());source_hash.update(source.read_bytes())
    return Munch.fromDict({
      'source_sha256':source_hash.hexdigest(),
      'model':{'pilot':True,'family':family,'n_dims':5,'n_positions':129,
               'n_embd':128,'n_layer':4,'n_head':4,'state_size':32},
      'training':{'data':'uniform','task':'noisy_linear_regression','task_kwargs':{'noise_std':.5},
                  'batch_size':64,'learning_rate':lr,'train_steps':steps,'num_tasks':None,
                  'num_training_examples':None,'save_every_steps':100,'keep_every_steps':-1,
                  'seed':182,'data_seed':1000000,'grad_clip':1.,
                  'curriculum':{'dims':{'start':5,'end':5,'inc':1,'interval':2000},
                                'points':{'start':33,'end':33,'inc':2,'interval':2000}}},
      'wandb':{'project':'project182','entity':'alexander-tian-ucb','log_every_steps':25},
      'protocol':{'noise_std':.5,'weight_prior':'N(0,I)','input_distribution':'uniform[-1,1]',
                  'dtype':'float32','matmul_precision':'highest; TF32 disabled',
                  'validation_seed':100000000,'test_seed':200000000,
                  'selection_metric':'validation normalized excess, ratio of means over all positions',
                  'mamba2_backend':'FLA PyTorch fallback','gdn_backend':'PyTorch exact triangular-solve recurrence','pilot_seed_count':1},
      'test_run':False})


def setup():
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False
    return 'cuda'


def make_model(c,device):
    torch.manual_seed(c.training.seed);np.random.seed(c.training.seed);random.seed(c.training.seed)
    return RegressionModel(c.model).to(device).float()


def data(split='validation'):
    seed=100000000 if split=='validation' else 200000000
    return fixed_dataset(ROOT/'data'/f'{split}.npz',seed,256,33,5,.5)


def json_save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def check(family,max_seconds):
    device=setup();start=time.monotonic()
    reference=verify_bayes(); val=data()
    from pilot_models import verify_delta_reference
    reference.update(verify_delta_reference(device))
    results={}
    from train import train_step
    for f in FAMILIES if family=='all' else (family,):
        if time.monotonic()-start>max_seconds-30:
            break
        t0=time.monotonic()
        try:
            c=config(f);model=make_model(c,device)
            x=torch.tensor(val['xs'][:8],device=device,dtype=torch.float32)
            y=torch.tensor(val['ys'][:8],device=device,dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                baseline=model(x,y)
                changed_y=y.clone();changed_y[:,10:]+=3
                changed_x=x.clone();changed_x[:,11:]+=2
                after=model(changed_x,changed_y)
                leakage=(baseline[:,:11]-after[:,:11]).abs().max().item()
                assert leakage<3e-5, f'Causality failed: {leakage}'
            optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
            model.train()
            x=torch.tensor(val['xs'][:64],device=device,dtype=torch.float32)
            y=torch.tensor(val['ys'][:64],device=device,dtype=torch.float32)
            times=[]
            for i in range(15):
                torch.cuda.synchronize();t=time.monotonic()
                loss,_=train_step(model,x,y,optimizer,lambda p,y:(p-y).square().mean(),grad_clip=1.)
                torch.cuda.synchronize()
                assert np.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                times.append(time.monotonic()-t)
            model.eval()
            with torch.no_grad():
                before=model(x[:4],y[:4]).clone()
            checkpoint=ROOT/'checks'/f'{f}.pt';checkpoint.parent.mkdir(exist_ok=True)
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoint)
            restored=make_model(c,device);restored.load_state_dict(torch.load(checkpoint,weights_only=True)['model']);restored.eval()
            with torch.no_grad():
                reload_error=(before-restored(x[:4],y[:4])).abs().max().item()
            assert reload_error<1e-6
            restored_optimizer=torch.optim.Adam(restored.parameters(),lr=.0003)
            restored_optimizer.load_state_dict(torch.load(checkpoint,weights_only=True)['optimizer'])
            model.train();restored.train()
            train_step(model,x,y,optimizer,lambda p,y:(p-y).square().mean(),grad_clip=1.)
            train_step(restored,x,y,restored_optimizer,lambda p,y:(p-y).square().mean(),grad_clip=1.)
            resume_error=max((a-b).abs().max().item() for a,b in zip(model.parameters(),restored.parameters()))
            assert resume_error<2e-6, f'Optimizer resume mismatch: {resume_error}'
            results[f]={'status':'pass','causality_max_change':leakage,'checkpoint_max_change':reload_error,'optimizer_resume_max_change':resume_error,
                        'seconds_per_step_median':float(np.median(times[5:])),
                        'warmup_and_check_seconds':time.monotonic()-t0,
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                        **model.state_info(),'last_smoke_loss':loss}
            del model,restored,optimizer,restored_optimizer
        except Exception as e:
            import traceback;traceback.print_exc()
            results[f]={'status':'failed','error':type(e).__name__+': '+str(e)}
        torch.cuda.empty_cache()
        json_save(ROOT/'checks.json',{'bayes':reference,'models':results})
        print(json.dumps({f:results[f]}),flush=True)
    return {'bayes':reference,'models':results}


def dispatch(action,family,lr,steps,run_id,max_seconds):
    if action=='check':return check(family,max_seconds)
    if action=='train':return run_training(family,lr,steps,run_id,max_seconds)
    if action=='sweep':
        start=time.monotonic();results=[]
        prefix=run_id or 'alex_'+datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')
        for f in FAMILIES if family=='all' else (family,):
            for rate in (.0001,.0003):
                remaining=max_seconds-(time.monotonic()-start)
                if remaining<60:
                    return {'runs':results,'stopped_for_time':True}
                name=f'{prefix}_{f}_linear_d5_lr{rate:g}_s182'
                result=run_training(f,rate,steps,name,int(remaining))
                results.append(compact(result))
                torch.cuda.empty_cache()
        return {'runs':results}
    if action=='selected':
        start=time.monotonic();results=[]
        for f in FAMILIES if family=='all' else (family,):
            candidates=[]
            for result_file in (ROOT/'runs').glob('*/result.json'):
                r=json.loads(result_file.read_text())
                if f'_{f}_linear_' in r['run_id'] and r['completed_steps']>=250:
                    candidates.append(r)
            if not candidates:continue
            advanced=[r for r in candidates if r['completed_steps']>250]
            pool=advanced or candidates
            ranked=sorted(pool,key=lambda r:r['validation']['normalized_excess'])
            chosen=ranked[:1]
            # A flat 250-step curve does not justify eliminating a learning rate.
            # Keep both if their validation scores differ by less than 5%.
            if steps==1000 and not advanced and len(ranked)>1:
                if ranked[1]['validation']['normalized_excess'] <= 1.05*ranked[0]['validation']['normalized_excess']:
                    chosen=ranked[:2]
            for best in chosen:
                remaining=max_seconds-(time.monotonic()-start)
                if remaining<60:return {'runs':results,'stopped_for_time':True}
                import yaml
                cfg=yaml.safe_load((ROOT/'runs'/best['run_id']/'config.yaml').read_text())
                result=run_training(f,cfg['training']['learning_rate'],steps,best['run_id'],int(remaining))
                results.append(compact(result));torch.cuda.empty_cache()
        return {'runs':results}
    if action=='evaluate':return evaluate_run(run_id)
    if action=='evaluate-selected':
        results=[]
        for f in FAMILIES:
            candidates=[json.loads(p.read_text()) for p in (ROOT/'runs').glob('*/result.json')]
            candidates=[r for r in candidates if f'_{f}_linear_' in r['run_id']]
            if not candidates:continue
            max_steps=max(r['completed_steps'] for r in candidates)
            best=min([r for r in candidates if r['completed_steps']==max_steps],key=lambda r:r['validation']['normalized_excess'])
            result=evaluate_run(best['run_id'])
            results.append({'run_id':best['run_id'],'steps':result['completed_steps'],
                            'test_excess':result['test']['normalized_excess'],
                            'late_test_excess':result['test']['late_normalized_excess'],
                            'order_sensitivity':result['order_sensitivity']})
        return {'evaluations':results}
    raise ValueError(action)


def compact(result):
    return {k:result[k] for k in ('run_id','completed_steps','seconds_per_step','elapsed_seconds','run_url')} | {
        'validation_excess':result['validation']['normalized_excess'],
        'late_validation_excess':result['validation']['late_normalized_excess']}


def run_training(family,lr,steps,run_id,max_seconds,seed=182,followup=False):
    from train import train
    import yaml,wandb
    device=setup();c=config(family,lr,steps)
    c.training.seed=seed
    c.training.data_seed=1000000+(seed-182)*1000000
    if followup:
        c.training.validation_every_steps=500
        c.training.checkpoint_steps=[2000,5000,10000,15000,20000]
        c.protocol.phase='followup-v1'
        c.protocol.confirmation_seed=300000000
        c.protocol.intended_training_seeds=[182,183,184]
    if not run_id:
        date=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%y%m%d')
        run_id=f'alex_{date}_{family}_linear_d5_lr{lr:g}_s{seed}'
    out=ROOT/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
    c.out_dir=str(out)
    if (out/'state.pt').exists():
        import shutil
        prior=yaml.safe_load((out/'config.yaml').read_text())
        if prior['model']!=c.model.toDict() or prior['training']['seed']!=seed or prior['training'].get('data_seed')!=c.training.data_seed or prior['training']['learning_rate']!=lr:
            raise ValueError('Resume settings do not match the saved experiment')
        checkpoint=torch.load(out/'state.pt',map_location='cpu',weights_only=False)
        old_step=checkpoint['completed_steps'];del checkpoint
        if followup:
            (out/'checkpoints').mkdir(exist_ok=True)
            snapshot=out/'checkpoints'/f'step_{old_step}.pt'
            if not snapshot.exists():shutil.copy2(out/'state.pt',snapshot)
            for name in ('result.json','config.yaml','gates_latest.pt'):
                source=out/name
                archive=out/'checkpoints'/f'{old_step}_{name}'
                if source.exists() and not archive.exists():shutil.copy2(source,archive)
    (out/'config.yaml').write_text(yaml.safe_dump(c.toDict(),sort_keys=False))
    (out/'stage_configs').mkdir(exist_ok=True)
    (out/'stage_configs'/f'target_{steps}.yaml').write_text(yaml.safe_dump(c.toDict(),sort_keys=False))
    model=make_model(c,device);val=data()
    gates=model.gate_snapshot(torch.tensor(val['xs'][:32],device=device,dtype=torch.float32),
                              torch.tensor(val['ys'][:32],device=device,dtype=torch.float32))
    if gates and not (out/'gates_initial.pt').exists():torch.save(gates,out/'gates_initial.pt')
    log_dir=Path('/tmp/project182-wandb')/run_id if followup else out
    log_dir.mkdir(parents=True,exist_ok=True)
    run=wandb.init(entity=c.wandb.entity,project=c.wandb.project,id=run_id,resume='allow',
                   name=run_id,config=c.toDict(),allow_val_change=True,tags=['followup-v1' if followup else 'pilot-v1','linear-d5',f'seed-{seed}'],
                   settings=wandb.Settings(disable_git=True,save_code=False,init_timeout=60),dir=str(log_dir))
    run.define_metric('iteration')
    run.define_metric('train/*',step_metric='iteration')
    run.define_metric('validation/*',step_metric='iteration')
    run.summary.update({'parameters':sum(p.numel() for p in model.parameters()),**model.state_info()})
    history=[]
    if (out/'validation.json').exists():history=json.loads((out/'validation.json').read_text())
    def validate(step):
        m=metrics(predictions(model,val,device),val)
        row={'step':step,**m};history.append(row);json_save(out/'validation.json',history)
        run.log({'iteration':step,'validation/normalized_excess':m['normalized_excess'],
                 'validation/late_normalized_excess':m['late_normalized_excess'],
                 'validation/mse':m['mse']})
        print(json.dumps({'run':run_id,'step':step,'validation_excess':m['normalized_excess'],
                          'late_excess':m['late_normalized_excess']}),flush=True)
        if followup and step in c.training.checkpoint_steps:
            gate_values=model.gate_snapshot(torch.tensor(val['xs'][:32],device=device,dtype=torch.float32),
                                           torch.tensor(val['ys'][:32],device=device,dtype=torch.float32))
            if gate_values:torch.save(gate_values,out/f'gates_step_{step}.pt')
        model.train()
        return row
    if not (out/'state.pt').exists():validate(0)
    try:
        result=train(model,c,validate=validate,max_seconds=max_seconds)
        validate(result['completed_steps'])
        model.eval()
        gates=model.gate_snapshot(torch.tensor(val['xs'][:32],device=device,dtype=torch.float32),
                                  torch.tensor(val['ys'][:32],device=device,dtype=torch.float32))
        if gates:torch.save(gates,out/'gates_latest.pt')
        result.update({'run_id':run_id,'run_url':run.url,'validation':history[-1],
                       'family':family,'seed':seed,'data_seed':c.training.data_seed,
                       'parameters':sum(p.numel() for p in model.parameters()),**model.state_info()})
        json_save(out/'result.json',result)
        return result
    finally:
        run.finish()
        if followup:wandb.teardown()


def evaluate_run(run_id):
    import yaml
    device=setup();out=ROOT/'runs'/run_id
    c=Munch.fromDict(yaml.safe_load((out/'config.yaml').read_text()))
    model=make_model(c,device);state=torch.load(out/'state.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state_dict']);test=data('test')
    pred=predictions(model,test,device)
    result={'run_id':run_id,'completed_steps':state['completed_steps'],
            'test':metrics(pred,test),'order_sensitivity':order_sensitivity(model,test,device)}
    np.savez_compressed(out/'test_predictions.npz',pred=pred)
    json_save(out/'evaluation.json',result)
    import wandb
    api=wandb.Api(timeout=30)
    saved=api.run(f'{c.wandb.entity}/{c.wandb.project}/{run_id}')
    saved.summary['test/normalized_excess']=result['test']['normalized_excess']
    saved.summary['test/late_normalized_excess']=result['test']['late_normalized_excess']
    saved.summary['test/order_sensitivity_k32']=result['order_sensitivity']['32']['normalized_change']
    saved.summary.update()
    return result
