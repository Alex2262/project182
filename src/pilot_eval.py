"""Fixed Bayesian regression prompts and float64 prefix reference calculations."""
from pathlib import Path
import numpy as np
import torch


def posterior_prefix(xs,ys,noise_std):
    xs=np.asarray(xs,dtype=np.float64); ys=np.asarray(ys,dtype=np.float64)
    b,t,d=xs.shape
    covariance=np.broadcast_to(np.eye(d),(b,d,d)).copy()
    mean=np.zeros((b,d))
    mu=np.empty((b,t)); risk=np.empty_like(mu); deff=np.empty_like(mu)
    for i in range(t):
        x=xs[:,i]; px=np.einsum('bij,bj->bi',covariance,x)
        mu[:,i]=np.einsum('bi,bi->b',x,mean)
        risk[:,i]=noise_std**2+np.einsum('bi,bi->b',x,px)
        deff[:,i]=d-np.trace(covariance,axis1=1,axis2=2)
        mean+=px*((ys[:,i]-mu[:,i])/risk[:,i])[:,None]
        covariance-=np.einsum('bi,bj->bij',px,px)/risk[:,i,None,None]
    return mu,risk,deff


def fixed_dataset(path,seed,n_prompts=256,n_points=33,n_dims=5,noise_std=.5):
    path=Path(path)
    if path.exists():
        with np.load(path) as data:
            result={k:data[k] for k in data.files}
        if int(result['seed'])!=seed or result['xs'].shape!=(n_prompts,n_points,n_dims) or float(result['noise_std'])!=noise_std:
            raise ValueError('Existing fixed dataset does not match the requested specification')
        return result
    rng=np.random.default_rng(seed)
    xs=rng.uniform(-1,1,(n_prompts,n_points,n_dims))
    w=rng.normal(size=(n_prompts,n_dims))
    f=np.einsum('btd,bd->bt',xs,w)
    ys=f+noise_std*rng.normal(size=f.shape)
    mu,risk,deff=posterior_prefix(xs,ys,noise_std)
    result={'xs':xs,'ys':ys,'mu':mu,'risk':risk,'deff':deff,'seed':np.array(seed),
            'noise_std':np.array(noise_std),'latent_y':f}
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**result)
    return result


def verify_bayes():
    rng=np.random.default_rng(916)
    x=rng.normal(size=(3,12,5)); y=rng.normal(size=(3,12)); sigma=.5
    mu,risk,deff=posterior_prefix(x,y,sigma)
    for b in range(3):
        for i in range(12):
            X=x[b,:i]; q=x[b,i]
            if i:
                K=X@X.T; reg=K+sigma**2*np.eye(i); k=X@q
                ref_mu=k@np.linalg.solve(reg,y[b,:i])
                ref_risk=sigma**2+q@q-k@np.linalg.solve(reg,k)
                ref_deff=np.trace(np.linalg.solve(reg,K))
            else:
                ref_mu=0.; ref_risk=sigma**2+q@q; ref_deff=0.
            np.testing.assert_allclose([mu[b,i],risk[b,i],deff[b,i]],
                                       [ref_mu,ref_risk,ref_deff],rtol=1e-10,atol=1e-10)
    shuffled=x.copy(); shuffled_y=y.copy()
    for b in range(3):
        perm=rng.permutation(11)
        shuffled[b,:11]=x[b,perm];shuffled_y[b,:11]=y[b,perm]
    shuffled_mu,shuffled_risk,_=posterior_prefix(shuffled,shuffled_y,sigma)
    np.testing.assert_allclose(shuffled_mu[:,-1],mu[:,-1],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(shuffled_risk[:,-1],risk[:,-1],rtol=1e-10,atol=1e-10)
    return {'prefix_posterior_matches_kernel_solve':True,'bayes_order_invariant':True,'dtype':'float64'}


@torch.no_grad()
def predictions(model,data,device,batch_size=32):
    model.eval()
    out=[]
    for i in range(0,len(data['xs']),batch_size):
        xs=torch.tensor(data['xs'][i:i+batch_size],device=device,dtype=torch.float32)
        ys=torch.tensor(data['ys'][i:i+batch_size],device=device,dtype=torch.float32)
        out.append(model(xs,ys).cpu().double().numpy())
    return np.concatenate(out)


def metrics(pred,data):
    squared=(pred-data['mu'])**2
    mse=(pred-data['ys'])**2
    risk=data['risk']
    return {'normalized_excess':float(squared.mean()/risk.mean()),
            'late_normalized_excess':float(squared[:,16:].mean()/risk[:,16:].mean()),
            'mse':float(mse.mean()),'bayes_risk':float(risk.mean()),
            'pointwise_normalized_excess':(squared.mean(0)/risk.mean(0)).tolist(),
            'pointwise_mse':mse.mean(0).tolist(),'pointwise_bayes_risk':risk.mean(0).tolist(),
            'pointwise_deff':data['deff'].mean(0).tolist()}


@torch.no_grad()
def order_sensitivity(model,data,device,n_prompts=64,n_permutations=5):
    model.eval(); rng=np.random.default_rng(303030)
    result={}
    for k in (4,8,16,32):
        xs=torch.tensor(data['xs'][:n_prompts,:k+1],device=device,dtype=torch.float32)
        ys=torch.tensor(data['ys'][:n_prompts,:k+1],device=device,dtype=torch.float32)
        # Fixed last query, fixed set of k observed pairs, permutation per prompt.
        reference=model(xs,ys)[:,-1]
        changes=[]
        for _ in range(n_permutations):
            perm=torch.tensor(np.stack([rng.permutation(k) for _ in range(n_prompts)]),device=device)
            batch=torch.arange(n_prompts,device=device)[:,None]
            shuffled_x=xs.clone(); shuffled_y=ys.clone()
            shuffled_x[:,:k]=xs[batch,perm]; shuffled_y[:,:k]=ys[batch,perm]
            difference=(model(shuffled_x,shuffled_y)[:,-1]-reference).square()
            changes.append(difference.cpu().numpy())
        value=float(np.mean(changes))
        result[str(k)]={'mean_squared_change':value,
                        'normalized_change':value/float(data['risk'][:n_prompts,k].mean())}
    return result
