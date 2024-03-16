# -*- coding: utf-8 -*-

# Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules.rotary import RotaryEmbedding
from fla.ops.retention import (chunk_retention, fused_chunk_retention,
                               fused_recurrent_retention, parallel_retention)


class MultiScaleRetention(nn.Module):
    def __init__(
        self,
        d_model: str = 1024,
        expand_k: str = 1,
        expand_v: str = 2,
        num_heads: str = 4,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        mode: str = 'chunk',
        fuse_norm: bool = True,
        layer_idx: int = None,
        *args,
        **kwargs
    ) -> MultiScaleRetention:
        super().__init__()

        self.d_model = d_model
        self.mode = mode
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_chunk', 'parallel', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = ACT2FN[gate_fn]
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        if (gate_fn == 'swish') and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=layernorm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)

        # TODO: fix this issue
        # https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py#L180
        # Ideally, we would want to support arbitrary d_head_qk
        assert self.head_qk_dim <= 256, "head_qk_dim must be less than or equal to 256"
        self.rotary = RotaryEmbedding(dim=self.head_qk_dim, interleaved=False)

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
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode
        q1 = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
        k1 = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)

        seqlen_offset = 0
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length()
        q, k = self.rotary(q1, k1, seqlen_offset)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = rearrange(self.v_proj(hidden_states), 'b n (h d) -> b h n d', h=self.num_heads)

        last_state = past_key_values[self.layer_idx] if use_cache else None
        if mode == 'chunk':
            o, last_state = chunk_retention(q, k, v, initial_state=last_state, output_final_state=use_cache)
        elif mode == 'fused_chunk':
            o, last_state = fused_chunk_retention(q, k, v, initial_state=last_state, output_final_state=use_cache)
        elif mode == 'parallel':
            o, last_state = parallel_retention(q, k, v, initial_state=last_state, output_final_state=use_cache)
        elif mode == 'fused_recurrent':
            o, last_state = fused_recurrent_retention(q, k, v, initial_state=last_state, output_final_state=use_cache)
        else:
            raise NotImplementedError
        if past_key_values is not None and last_state is not None:
            past_key_values.update(last_state, self.layer_idx)

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(hidden_states)
        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')

        else:
            o = self.g_norm(o)
            o = rearrange(o, 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)

        return o, None, past_key_values
