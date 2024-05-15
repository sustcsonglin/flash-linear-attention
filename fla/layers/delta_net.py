# -*- coding: utf-8 -*-

# Sect4.2 of Linear Transformers Are Secretly Fast Weight Programmers https://arxiv.org/abs/2102.11174


from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.modules.rotary import RotaryEmbedding
from fla.ops.delta_rule import (fused_chunk_delta_rule,
                                fused_recurrent_linear_attn_delta_rule)


def simple_norm(x):
    return (F.normalize(x, dim=-1) * x.shape[-1] ** 0.5).to(x)


# @torch.jit.script
def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)


# @torch.jit.script
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


# @torch.jit.script
def elu_norm(x):
    dtype = x.dtype
    x = F.elu(x, 1., False) + 1.
    return (x / x.sum(-1, keepdim=True)).to(dtype)


# https://github.com/IDSIA/recurrent-fwp/blob/master/algorithmic/layers.py#L86C1-L146C1
class DeltaNet(nn.Module):
    def __init__(
        self,
        mode: str = 'fused_chunk',
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        chunk_size: int = 16,
        use_beta: bool = True,
        use_gate: bool = True,
        use_rope: bool = True,
        use_elu: bool = False,
        use_q_swish: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        share_conv_kernel: bool = True,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        **kwargs
    ) -> DeltaNet:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.chunk_size = chunk_size
        self.use_gate = use_gate

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.share_conv_kernel = share_conv_kernel

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.layer_idx = layer_idx

        assert mode in ['fused_chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.use_beta = use_beta
        self.use_elu = use_elu
        self.use_rope = use_rope

        if self.use_beta:
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)

        if use_short_conv:
            self.conv_size = conv_size
            if share_conv_kernel:
                self.h_conv1d = ShortConvolution(hidden_size, conv_size, activation='silu')
            else:
                self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
                self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.norm = FusedRMSNormSwishGate(self.head_v_dim, elementwise_affine, norm_eps)
        else:
            self.norm = RMSNorm(self.head_v_dim, elementwise_affine, norm_eps)

        if use_rope:
            self.rotary = RotaryEmbedding(dim=self.head_qk_dim)

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
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:

        # change to inference mode.
        mode = 'fused_recurrent' if past_key_values is not None else self.mode
        last_state = past_key_values[self.layer_idx] if use_cache else None

        if self.use_short_conv:
            conv_state = last_state[0] if use_cache else None
            if self.share_conv_kernel:
                # conv state is updated inplace
                hidden_states = self.h_conv1d(hidden_states, attention_mask, conv_state)
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
            else:
                conv_state_q = last_state[0] if use_cache else None
                conv_state_k = last_state[1] if use_cache else None
                conv_state_v = last_state[2] if use_cache else None
                q = self.q_proj(hidden_states)
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                q = self.q_conv1d(q, attention_mask, conv_state_q)
                k = self.k_conv1d(k, attention_mask, conv_state_k)
                v = self.v_conv1d(v, attention_mask, conv_state_v)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))

        # There is no relative positional encoding perspective of rope in delta rule.
        # So I intentially disable using rope for delta rule.
        # It will be super weird to use rope in delta rule, but nevertheless you can have a try.
        # if self.use_rope:
        #     seqlen_offset = 0
        #     if past_key_values is not None:
        #         seqlen_offset = past_key_values.get_seq_length()
        #     q, k = map(lambda x: rearrange(x, 'b l (h d) -> b l h d', h=self.num_heads), (q, k))
        #     q, k = self.rotary(q, k, seqlen_offset)
        #     q, k = q.transpose(1, 2), k.transpose(1, 2)
        #     v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        # else:
        q, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (q, k, v))

        if not self.use_elu:
            q = F.normalize(q, p=2, dim=-1).to(q)
            k = F.normalize(k, p=2, dim=-1).to(k)
        else:
            q = elu_norm(q)
            k = elu_norm(k)

        if self.use_beta:
            beta = rearrange(self.b_proj(hidden_states), 'b l h -> b h l').sigmoid()
        else:
            beta = q.new_ones(q.shape[0], q.shape[1], q.shape[2])

        state = past_key_values[self.layer_idx][-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_linear_attn_delta_rule(q, k, v, beta, state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_delta_rule(q, k, v, beta, self.chunk_size, state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            if self.use_short_conv:
                if self.share_conv_kernel:
                    state = (conv_state, recurrent_state)
                else:
                    state = (conv_state_q, conv_state_k, conv_state_v, recurrent_state)
            else:
                state = (recurrent_state,)
            past_key_values.update(state, self.layer_idx)

        o = rearrange(o, 'b h l d -> b l h d')
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.norm(o, g)
        else:
            o = self.norm(o)
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)
        return o, None, past_key_values

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = tuple()
        if self.use_short_conv:
            if self.share_conv_kernel:
                state += (param.new_zeros(batch_size, self.hidden_size, self.conv_size),)
            else:
                # for q/k/v each
                state += (param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.key_dim, self.conv_size),
                          param.new_zeros(batch_size, self.value_dim, self.conv_size))
        state += (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state
