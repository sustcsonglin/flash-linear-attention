# -*- coding: utf-8 -*-

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174


from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.ops.delta_rule import (fused_chunk_delta_rule,
                                fused_recurrent_linear_attn_delta_rule)

from transformers.utils import logging
import warnings
logger = logging.get_logger(__name__)
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except:
    causal_conv1d_fn = None
    causal_conv1d_update = None
    logger.warning_once("causal_conv1d_fn is not available")
from fla.layers.utils import proj_then_shortconv, make_conv1d_module, shortconv

# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1
class DeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        mode: str = 'fused_chunk',
        chunk_size: int = 16,
        use_gate: bool = True,

        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        *args, **kwargs
    ) -> DeltaNet:
        super().__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        assert mode in ['fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        self.chunk_size = chunk_size
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)

        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.beta_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        
        self.use_short_conv = use_short_conv
        if use_short_conv:
            self.conv_size = conv_size
            self.share_conv_kernel = share_conv_kernel
            assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available, Please install via `pip install causal-conv1d>=1.2.0`."            
            if share_conv_kernel:
                self.h_conv1d = make_conv1d_module(self.hidden_size, conv_size, conv_bias)
            else:
                self.q_conv1d = make_conv1d_module(self.key_dim, conv_size, conv_bias)
                self.k_conv1d = make_conv1d_module(self.key_dim, conv_size, conv_bias)
                self.v_conv1d = make_conv1d_module(self.value_dim, conv_size, conv_bias)

        self.use_gate = use_gate

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.norm = FusedRMSNormSwishGate(self.head_v_dim)
        else:
            self.norm = RMSNorm(self.head_v_dim)
        self.apply(self._initialize_weights)
        print(self)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if self.use_short_conv:
            assert past_key_values is None,  "use_short_conv is not supported yet for inference."
            if self.share_conv_kernel:
                hidden_states = shortconv(hidden_states, self.h_conv1d.weight, self.h_conv1d.bias)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                q = proj_then_shortconv(hidden_states, self.q_proj.weight, 
                self.q_conv1d.weight, self.q_conv1d.bias)
                k = proj_then_shortconv(hidden_states, self.k_proj.weight,
                self.k_conv1d.weight, self.k_conv1d.bias)
                v = proj_then_shortconv(hidden_states, self.v_proj.weight,
                self.v_conv1d.weight, self.v_conv1d.bias)
            
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))
        beta = rearrange(self.beta_proj(hidden_states), 'b l h -> b h l').sigmoid()

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if self.mode == 'fused_recurrent':
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
            o, last_state = fused_recurrent_linear_attn_delta_rule(q, k, v, beta)
        elif self.mode == 'fused_chunk':
            o, last_state = fused_chunk_delta_rule(q, k, v, beta, self.chunk_size)
        else:
            raise NotImplementedError(f"Not supported mode `{self.mode}`.")
        if past_key_values is not None and last_state is not None:
            past_key_values.update(last_state, self.layer_idx)

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.norm(o, g)
        else:
            o = self.norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values


if __name__ == '__main__':
    import torch
    batch = 4
    seq_len = 1024
    hidden_size = 1024
    x = torch.randn(batch, seq_len, hidden_size).to(torch.bfloat16).cuda().requires_grad_(True)
    model = DeltaNet(hidden_size=hidden_size).to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
