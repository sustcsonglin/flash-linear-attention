# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.cache_utils import Cache

from fla.modules import FusedRMSNormSwishGate, RotaryEmbedding
from fla.ops.abc.chunk import chunk_abc


class ABCAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_slots: Optional[int] = None,
        layernorm_eps: float = 1e-5,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = -32,
        clamp_max: Optional[float] = 32,
        layer_idx: Optional[int] = None,
        **kwargs
    ) -> ABCAttention:
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.key_dim = int(self.hidden_size * self.expand_k)
        self.val_dim = int(self.hidden_size * self.expand_v)
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.val_dim // self.num_heads

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots

        self.layernorm_eps = layernorm_eps
        self.gate_low_rank_dim = gate_low_rank_dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.layer_idx = layer_idx
        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.val_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.val_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.val_dim, bias=False)

        self.g_norm = FusedRMSNormSwishGate(self.head_v_dim)
        self.s_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.num_heads * self.num_slots, bias=True)
        )

        self.rotary_emb = RotaryEmbedding(self.head_k_dim)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # [batch_size, seq_len, n_heads * d_head]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)

        seqlen_offset = 0
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length()
        q, k = self.rotary(q, k, seqlen_offset)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = rearrange(self.v_proj(hidden_states), 'b t (h d) -> b h t d', h=self.num_heads)
        # [batch_size, n_heads, seq_len, num_slots]
        s = rearrange(self.s_proj(hidden_states), 'b t (h m) -> b h t m', h=self.num_heads)
        s = s.clamp_(self.clamp_min, self.clamp_max)

        last_state = past_key_values[self.layer_idx] if use_cache else None
        o, last_state = chunk_abc(q, k, v, s, initial_state=last_state, output_final_state=use_cache)
        if past_key_values is not None and last_state is not None:
            past_key_values.update(last_state, self.layer_idx)

        o = rearrange(o, 'b h t d -> b t h d')
        g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        o = rearrange(self.g_norm(o, g), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values
