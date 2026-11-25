"""Reference backbones for continuous AR(1) forecasting.

HF LlamaModel (identity rotary embedding for the registered NoPE control) and
FLA GatedDeltaNetModel, both from pinned releases. No pretrained weights.
Legacy forecasting models remain available for reproducing old checkpoints.
"""
from contextlib import nullcontext

import torch
from torch import nn


class IdentityRotary(nn.Module):
    """Explicit NoPE adaptation: applying rotary with cos=1, sin=0 is identity."""
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids):
        shape = (*position_ids.shape, self.head_dim)
        return x.new_ones(shape), x.new_zeros(shape)


class ReferenceForecaster(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        width, heads = config['width'], config['heads']
        if width % heads or config['precision'] not in ('float32', 'bfloat16'):
            raise ValueError('Invalid head dimensions or precision')
        common = dict(hidden_size=width, num_hidden_layers=config['layers'],
                      intermediate_size=config['intermediate_size'], vocab_size=1,
                      max_position_embeddings=131072, initializer_range=0.02,
                      use_cache=False, pad_token_id=None, bos_token_id=None, eos_token_id=None)
        if config['family'] == 'transformer_nope':
            from transformers import LlamaConfig, LlamaModel
            cfg = LlamaConfig(**common, num_attention_heads=heads, num_key_value_heads=heads,
                              head_dim=width // heads, rms_norm_eps=1e-6,
                              attention_bias=False, mlp_bias=False, attention_dropout=0.,
                              hidden_act='silu', attn_implementation='sdpa')
            self.backbone = LlamaModel(cfg)
            self.backbone.rotary_emb = IdentityRotary(width // heads)
            # Continuous inputs bypass token embeddings; remove unused parameters.
            self.backbone.embed_tokens = None
        elif config['family'] == 'gated_deltanet':
            from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetModel
            cfg = GatedDeltaNetConfig(**common, num_heads=heads, num_v_heads=heads,
                                     head_dim=config['key_dim'], expand_v=config['expand_v'],
                                     norm_eps=1e-6, hidden_act='swish', attn_mode='chunk',
                                     use_gate=True, use_short_conv=True, conv_size=4,
                                     allow_neg_eigval=False, output_gate_activation='swish',
                                     fuse_norm=True, fuse_swiglu=True, fuse_cross_entropy=False)
            self.backbone = GatedDeltaNetModel(cfg)
            self.backbone.embeddings = None
        else:
            raise ValueError(config['family'])
        self.input = nn.Linear(2, width)
        self.output = nn.Linear(width, 1)
        for layer in (self.input, self.output):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def precision_context(self, device_type):
        if self.config['precision'] == 'bfloat16':
            return torch.autocast(device_type, dtype=torch.bfloat16)
        return nullcontext()

    def forward(self, tokens, states=None, return_state=False):
        if tokens.ndim != 3 or tokens.shape[-1] != 2 or tokens.shape[1] < 1:
            raise ValueError('Expected B,T,2 observed-transition tokens')
        if states is not None and not return_state:
            raise ValueError('Carried state requires return_state=True')
        with self.precision_context(tokens.device.type):
            result = self.backbone(inputs_embeds=self.input(tokens),
                                   past_key_values=states, use_cache=return_state,
                                   return_dict=True)
        # Keep regression readout and MSE in FP32, even with mixed-precision blocks.
        with torch.autocast(tokens.device.type, enabled=False):
            prediction = self.output(result.last_hidden_state.float()).squeeze(-1)
        return (prediction, result.past_key_values) if return_state else prediction

    def description(self):
        recurrent = convolution = 0
        if self.config['family'] == 'gated_deltanet':
            for block in self.backbone.layers:
                m = block.attn
                recurrent += m.num_v_heads * m.head_k_dim * m.head_v_dim
                convolution += (2 * m.key_dim + m.value_dim) * m.conv_size
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    backbone_class=type(self.backbone).__name__,
                    recurrent_state_elements_per_series=recurrent,
                    convolution_cache_elements_per_series=convolution,
                    persistent_state_elements_per_series=recurrent + convolution,
                    positional_embedding='none', precision=self.config['precision'],
                    state_excludes='weights, transient activations; Transformer KV cache grows with history',
                    adaptations=['continuous input/output projections', 'identity rotary for Transformer',
                                 'unused vocabulary embedding removed'])


def build_forecaster(config):
    if config.get('implementation', 'legacy') == 'reference':
        return ReferenceForecaster(config)
    from forecast_models import Forecaster
    return Forecaster(config)
