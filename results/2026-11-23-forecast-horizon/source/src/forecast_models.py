"""Controlled forecasters: GDN recurrence and a causal Transformer without PE.

Both receive the same explicit lag pair, so the AR(1) reference does not require
recovering adjacency from positional embeddings. These are small controlled
models trained from scratch, not pretrained time-series foundation models.
The GDN uses exact PyTorch chunks; its timings are not optimized-kernel claims.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def delta_chunk(q, k, v, log_alpha, beta, state):
    """Exact gated delta recurrence with an incoming state; B,T,H,K/V tensors."""
    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
    log_alpha, beta = log_alpha.transpose(1, 2), beta.transpose(1, 2)
    length = q.shape[-2]
    cumulative = log_alpha.cumsum(-1)
    lower = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
    log_decay = (cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)).masked_fill(~lower, -torch.inf)
    decay = log_decay.exp()
    initial_decay = cumulative.exp()
    gram = k @ k.transpose(-1, -2)
    system = torch.eye(length, device=q.device, dtype=q.dtype) + (
        beta.unsqueeze(-1) * decay * gram).tril(-1)
    rhs = beta.unsqueeze(-1) * (v - initial_decay.unsqueeze(-1) * (k @ state))
    writes = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    output = initial_decay.unsqueeze(-1) * (q @ state) + ((q @ k.transpose(-1, -2)) * decay) @ writes
    final_decay = (cumulative[..., -1:] - cumulative).exp()
    final = initial_decay[..., -1, None, None] * state + (
        k * final_decay.unsqueeze(-1)).transpose(-1, -2) @ writes
    return (output / math.sqrt(q.shape[-1])).transpose(1, 2), final


class DeltaMixer(nn.Module):
    def __init__(self, width, heads, key_dim, chunk_size):
        super().__init__()
        self.width, self.heads, self.key_dim = width, heads, key_dim
        self.value_dim, self.chunk_size = width // heads, chunk_size
        self.qk_width = heads * key_dim
        self.channels = 2 * self.qk_width + width
        self.qkv = nn.Linear(width, self.channels, bias=False)
        self.conv = nn.Conv1d(self.channels, self.channels, 4, groups=self.channels, bias=False)
        self.a_proj = nn.Linear(width, heads, bias=False)
        self.b_proj = nn.Linear(width, heads, bias=False)
        self.output_gate = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)
        self.rms_weight = nn.Parameter(torch.ones(self.value_dim))
        dt = torch.exp(torch.empty(heads).uniform_(math.log(0.001), math.log(0.1)))
        self.dt_bias = nn.Parameter(torch.log(torch.expm1(dt)))
        self.A_log = nn.Parameter(torch.empty(heads).uniform_(1, 16).log())

    def empty_state(self, x):
        return {"memory": x.new_zeros(x.shape[0], self.heads, self.key_dim, self.value_dim),
                "conv": x.new_zeros(x.shape[0], self.channels, 3)}

    def forward(self, x, state=None):
        if state is None:
            state = self.empty_state(x)
        projected = self.qkv(x).transpose(1, 2)
        conv_input = torch.cat((state["conv"], projected), dim=-1)
        convolved = F.silu(self.conv(conv_input)).transpose(1, 2)
        q, k, v = convolved.split((self.qk_width, self.qk_width, self.width), dim=-1)
        q = F.normalize(q.reshape(*x.shape[:2], self.heads, self.key_dim), dim=-1, eps=1e-6)
        k = F.normalize(k.reshape_as(q), dim=-1, eps=1e-6)
        v = v.reshape(*x.shape[:2], self.heads, self.value_dim)
        log_alpha = -self.A_log.exp() * F.softplus(self.a_proj(x) + self.dt_bias)
        beta = self.b_proj(x).sigmoid()
        memory, outputs = state["memory"], []
        for start in range(0, x.shape[1], self.chunk_size):
            sl = slice(start, start + self.chunk_size)
            out, memory = delta_chunk(q[:, sl], k[:, sl], v[:, sl],
                                      log_alpha[:, sl], beta[:, sl], memory)
            outputs.append(out)
        result = torch.cat(outputs, dim=1)
        result = result * torch.rsqrt(result.square().mean(-1, keepdim=True) + 1e-5)
        result = result * self.rms_weight
        result = result.reshape_as(x) * F.silu(self.output_gate(x))
        return self.output(result), {"memory": memory, "conv": conv_input[:, :, -3:].contiguous()}


class AttentionMixer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, width // self.heads)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(2))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        return self.output(out.transpose(1, 2).reshape_as(x))


class ForecastBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config["width"]
        self.family = config["family"]
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        if self.family == "gated_deltanet":
            self.mixer = DeltaMixer(width, config["heads"], config["key_dim"], config["chunk_size"])
        elif self.family == "transformer_nope":
            self.mixer = AttentionMixer(width, config["heads"])
        else:
            raise ValueError(self.family)

    def forward(self, x, state=None):
        if self.family == "gated_deltanet":
            mixed, state = self.mixer(self.norm1(x), state)
        else:
            if state is not None:
                raise ValueError("The Transformer control is evaluated with full prefixes")
            mixed = self.mixer(self.norm1(x))
        x = x + mixed
        return x + self.mlp(self.norm2(x)), state


class Forecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config["width"] % config["heads"] or config["chunk_size"] < 1:
            raise ValueError("Invalid head or chunk dimensions")
        self.input = nn.Linear(2, config["width"])
        self.blocks = nn.ModuleList([ForecastBlock(config) for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(config["width"])
        self.output = nn.Linear(config["width"], 1)

    def forward(self, tokens, states=None, return_state=False):
        if tokens.ndim != 3 or tokens.shape[-1] != 2 or tokens.shape[1] == 0:
            raise ValueError("Expected nonempty B,T,2 observed-transition tokens")
        if states is None:
            states = [None] * len(self.blocks)
        if len(states) != len(self.blocks):
            raise ValueError("State layer count mismatch")
        x, next_states = self.input(tokens), []
        for block, state in zip(self.blocks, states):
            x, state = block(x, state)
            next_states.append(state)
        prediction = self.output(self.norm(x)).squeeze(-1)
        return (prediction, next_states) if return_state else prediction

    def description(self):
        elements = 0
        for block in self.blocks:
            if self.config["family"] == "gated_deltanet":
                m = block.mixer
                elements += m.heads * m.key_dim * m.value_dim + m.channels * 3
        return {"parameters": sum(p.numel() for p in self.parameters()),
                "persistent_state_elements_per_series": elements,
                "persistent_state_bytes_fp32_per_series": 4 * elements,
                "state_excludes": "model weights, temporary chunk activations, last scalar observation",
                "positional_embedding": "none; both models receive adjacent observed-value pairs"}
