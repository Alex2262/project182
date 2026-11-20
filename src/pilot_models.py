"""Small continuous-input backbones for the first regression pilot.

GPT-2 uses HF; GDN and Mamba-2 use flash-linear-attention 0.6.0.
GDN uses an exact PyTorch triangular-solve recurrence for strict FP32.
Mamba-2 deliberately uses its maintained PyTorch SSD fallback (no mamba-ssm
extension installed). LA evaluates the same normalized positive-feature
recurrence through its quadratic, causal attention form at these short lengths.
All trainable tensors and activations are FP32; no autocast is used.
"""
import torch
from torch import nn
from torch.nn import functional as F


class LinearAttention(nn.Module):
    def __init__(self, width, heads, state_size):
        super().__init__()
        self.heads, self.key_dim, self.value_dim = heads, state_size, width // heads
        self.q = nn.Linear(width, heads * state_size, bias=False)
        self.k = nn.Linear(width, heads * state_size, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)

    def forward(self, x):
        b,t,_=x.shape
        q=F.elu(self.q(x).reshape(b,t,self.heads,self.key_dim))+1
        k=F.elu(self.k(x).reshape(b,t,self.heads,self.key_dim))+1
        v=self.v(x).reshape(b,t,self.heads,self.value_dim)
        weights=torch.einsum('bthk,bshk->bhts',q,k)
        mask=torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        weights=weights.masked_fill(~mask,0)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        y=torch.einsum('bhts,bshv->bthv',weights,v)
        return self.out(y.reshape(b,t,-1))


def delta_rule_parallel(q,k,v,log_alpha,beta):
    """Exact finite-sequence form of the scalar-gated delta recurrence.

    q,k,v: B,T,H,K/V; gates: B,T,H. Initial memory is zero.
    Solves for writes u_t=beta_t*(v_t-k_t^T*alpha_t*S_{t-1}), then
    reads S_t=sum_{j<=t} exp(sum_{i=j+1}^t log_alpha_i) k_j u_j^T.
    This short-context implementation is quadratic in T, but represents the
    same fixed-state recurrence; its throughput is not a kernel benchmark.
    """
    t=q.shape[1]
    q,k,v=[x.transpose(1,2) for x in (q,k,v)]
    log_alpha=log_alpha.transpose(1,2);beta=beta.transpose(1,2)
    cumulative=log_alpha.cumsum(-1)
    lower=torch.ones(t,t,device=q.device,dtype=torch.bool).tril()
    log_decay=(cumulative.unsqueeze(-1)-cumulative.unsqueeze(-2)).masked_fill(~lower,-torch.inf)
    decay=log_decay.exp()
    kk=k@k.transpose(-1,-2)
    system=torch.eye(t,device=q.device,dtype=q.dtype)+(beta.unsqueeze(-1)*decay*kk).tril(-1)
    writes=torch.linalg.solve_triangular(system,beta.unsqueeze(-1)*v,upper=False,unitriangular=True)
    output=((q@k.transpose(-1,-2))*decay)@writes
    return (output/(q.shape[-1]**.5)).transpose(1,2)


def gdn_reference(layer,x):
    b,t,_=x.shape
    def project_conv(proj,conv):
        value=proj(x).transpose(1,2)
        value=F.conv1d(value,conv.weight,conv.bias,padding=layer.conv_size-1,groups=value.shape[1])
        return F.silu(value[:,:,:t].transpose(1,2))
    q=project_conv(layer.q_proj,layer.q_conv1d).reshape(b,t,layer.num_heads,layer.head_k_dim)
    k=project_conv(layer.k_proj,layer.k_conv1d).reshape_as(q)
    v=project_conv(layer.v_proj,layer.v_conv1d).reshape(b,t,layer.num_heads,layer.head_v_dim)
    q=q*torch.rsqrt(q.square().sum(-1,keepdim=True)+1e-6)
    k=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    log_alpha=-layer.A_log.exp()*F.softplus(layer.a_proj(x)+layer.dt_bias)
    beta=layer.b_proj(x).sigmoid()
    output=delta_rule_parallel(q,k,v,log_alpha,beta)
    gate=layer.g_proj(x).reshape_as(output)
    # FLA FusedRMSNormGated normalizes before applying its SiLU output gate.
    output=output*torch.rsqrt(output.square().mean(-1,keepdim=True)+layer.o_norm.eps)
    output=output*layer.o_norm.weight*F.silu(gate)
    return layer.o_proj(output.reshape(b,t,-1))


def verify_delta_reference(device):
    torch.manual_seed(71)
    q=torch.randn(2,9,2,4,device=device,dtype=torch.float64,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    k_norm=k*torch.rsqrt(k.square().sum(-1,keepdim=True)+1e-6)
    v=torch.randn(2,9,2,3,device=device,dtype=torch.float64,requires_grad=True)
    g=(-.1*torch.rand(2,9,2,device=device,dtype=torch.float64)).requires_grad_()
    beta=torch.rand_like(g,requires_grad=True)
    parallel=delta_rule_parallel(q,k_norm,v,g,beta)
    state=torch.zeros(2,2,4,3,device=device,dtype=torch.float64);outs=[]
    for i in range(9):
        state=g[:,i].exp()[...,None,None]*state
        residual=v[:,i]-torch.einsum('bhk,bhkv->bhv',k_norm[:,i],state)
        state=state+beta[:,i,...,None,None]*torch.einsum('bhk,bhv->bhkv',k_norm[:,i],residual)
        outs.append(torch.einsum('bhk,bhkv->bhv',q[:,i],state)/2)
    recurrent=torch.stack(outs,1)
    error=(parallel-recurrent).abs().max().item()
    grads1=torch.autograd.grad(parallel.square().sum(),(q,k,v,g,beta),retain_graph=True)
    grads2=torch.autograd.grad(recurrent.square().sum(),(q,k,v,g,beta))
    grad_error=max((a-b).abs().max().item() for a,b in zip(grads1,grads2))
    assert error<1e-10 and grad_error<1e-9,(error,grad_error)
    return {'delta_recurrence_max_error':error,'delta_gradient_max_error':grad_error}


class Block(nn.Module):
    def __init__(self, conf, index):
        super().__init__()
        w,h,s=conf.n_embd,conf.n_head,conf.state_size
        self.family=conf.family
        self.norm1=nn.LayerNorm(w)
        self.norm2=nn.LayerNorm(w)
        self.mlp=nn.Sequential(nn.Linear(w,4*w),nn.GELU(approximate='tanh'),nn.Linear(4*w,w))
        if self.family=='linear_attention':
            self.mixer=LinearAttention(w,h,s)
            self.state_elements=h*(s*(w//h)+s)
            self.conv_state_elements=0
        elif self.family=='gated_deltanet':
            from fla.layers.gated_deltanet import GatedDeltaNet
            # Head key width is state_size; keep total value width fixed at 2*width.
            self.mixer=GatedDeltaNet(hidden_size=w,num_heads=h,head_dim=s,
                                    expand_v=(2*w)/(h*s),use_short_conv=True,
                                    conv_size=4,layer_idx=index,allow_neg_eigval=False)
            self.state_elements=h*s*self.mixer.head_v_dim
            self.conv_state_elements=(2*h*s+2*w)*4
        elif self.family=='mamba2':
            from fla.layers.mamba2 import Mamba2
            self.mixer=Mamba2(hidden_size=w,expand=2,num_heads=h,head_dim=2*w//h,
                              state_size=s,chunk_size=32,layer_idx=index,backend='triton')
            self.state_elements=2*w*s
            self.conv_state_elements=(2*w+2*s)*4
        else:
            raise ValueError(self.family)
        self.capture_gates=False
        self.gates={}

    def forward(self,x):
        z=self.norm1(x)
        if self.capture_gates and self.family in ('mamba2','gated_deltanet'):
            with torch.no_grad():
                m=self.mixer
                if self.family=='mamba2':
                    dt=F.softplus(m.in_proj(z)[...,-m.num_heads:].float()+m.dt_bias)
                    log_alpha=-m.A_log.float().exp()*dt
                    self.gates={'alpha':log_alpha.exp().cpu(),'log_alpha':log_alpha.cpu()}
                else:
                    log_alpha=-m.A_log.float().exp()*F.softplus(m.a_proj(z).float()+m.dt_bias)
                    self.gates={'alpha':log_alpha.exp().cpu(),'log_alpha':log_alpha.cpu(),
                                'beta':m.b_proj(z).sigmoid().cpu()}
        y=gdn_reference(self.mixer,z) if self.family=="gated_deltanet" else self.mixer(z)
        if isinstance(y,tuple):
            y=y[0]
        x=x+y
        return x+self.mlp(self.norm2(x))


class RegressionModel(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.conf=conf
        self.n_dims=conf.n_dims
        self.n_positions=conf.n_positions
        self.name=f'{conf.family}_pilot'
        self.read_in=nn.Linear(conf.n_dims,conf.n_embd)
        self.read_out=nn.Linear(conf.n_embd,1)
        self.family=conf.family
        if conf.family=='gpt2':
            from transformers import GPT2Config,GPT2Model
            cfg=GPT2Config(vocab_size=1,n_positions=2*conf.n_positions,
                           n_embd=conf.n_embd,n_layer=conf.n_layer,n_head=conf.n_head,
                           resid_pdrop=0.,embd_pdrop=0.,attn_pdrop=0.,use_cache=False)
            self.backbone=GPT2Model(cfg)
            # Continuous embeddings bypass the vocabulary table entirely.
            self.backbone.wte.weight.requires_grad_(False)
        else:
            self.blocks=nn.ModuleList([Block(conf,i) for i in range(conf.n_layer)])
            self.norm=nn.LayerNorm(conf.n_embd)

    def forward(self,xs,ys,inds=None):
        b,t,d=xs.shape
        if t>self.n_positions:
            raise ValueError('Prompt exceeds configured position capacity')
        ytokens=F.pad(ys.unsqueeze(-1),(0,d-1))
        tokens=torch.stack((xs,ytokens),dim=2).reshape(b,2*t,d)
        z=self.read_in(tokens)
        if self.family=='gpt2':
            z=self.backbone(inputs_embeds=z,use_cache=False).last_hidden_state
        else:
            for block in self.blocks:
                z=block(z)
            z=self.norm(z)
        pred=self.read_out(z)[:,::2,0]
        return pred if inds is None else pred[:,inds]

    def state_info(self):
        if self.family=='gpt2':
            return {'recurrent_state_elements':0,'conv_state_elements':0,
                    'kv_elements_per_pair':4*self.conf.n_layer*self.conf.n_embd}
        return {'recurrent_state_elements':sum(b.state_elements for b in self.blocks),
                'conv_state_elements':sum(b.conv_state_elements for b in self.blocks)}

    @torch.no_grad()
    def gate_snapshot(self,xs,ys):
        if self.family not in ('mamba2','gated_deltanet'):
            return {}
        for b in self.blocks:
            b.capture_gates=True
        self(xs,ys)
        result={str(i):b.gates for i,b in enumerate(self.blocks)}
        for b in self.blocks:
            b.capture_gates=False
            b.gates={}
        return result
