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


@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


@torch.jit.script
def l2_norm(x):
    return x / x.norm(p=2, dim=-1, keepdim=True)

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

        self.use_gate = use_gate

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.norm = FusedRMSNormSwishGate(self.head_v_dim)
        else:
            self.norm = RMSNorm(self.head_v_dim)

        self.apply(self._initialize_weights)

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
        q = rearrange(self.q_proj(hidden_states), 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(self.v_proj(hidden_states), 'b l (h d) -> b h l d', h=self.num_heads)
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
