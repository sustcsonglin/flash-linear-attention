# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

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
        self.sk_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.num_heads * self.num_slots, bias=True)
        )
        self.sv_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.gate_low_rank_dim, bias=False),
            nn.Linear(self.gate_low_rank_dim, self.num_heads * self.num_slots, bias=True)
        )

        self.rotary_emb = RotaryEmbedding(self.head_k_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.sk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.sk_proj[1].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.sv_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.sv_proj[1].weight, gain=2 ** -2.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        # [batch_size, seq_len, n_heads * d_head]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, -1)
        k = k.view(batch_size, seq_len, self.num_heads, -1)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        kv_seq_len = v.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        q, k = self.rotary_emb(q, k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        if past_key_value is not None:  # reuse k, v, self_attention
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        past_key_value = (k, v) if use_cache else None

        # cast to half precision
        input_dtype = q.dtype
        if input_dtype == torch.float:
            warnings.warn("The input hidden states seems to be silently casted in float32.")
            q = q.to(self.config.torch_dtype)
            k = k.to(self.config.torch_dtype)
            v = v.to(self.config.torch_dtype)

        # [batch_size, n_heads, seq_len, num_slots]
        sk = rearrange(self.sk_proj(hidden_states), 'b t (h m) -> b h t m', h=self.num_heads)
        sv = rearrange(self.sv_proj(hidden_states), 'b t (h m) -> b h t m', h=self.num_heads)

        o = chunk_abc(q, k, v, sk.clamp_(self.clamp_min, self.clamp_max), sv.clamp_(self.clamp_min, self.clamp_max))
        o = rearrange(o, 'b h t d -> b t h d')
        g = rearrange(self.g_proj(hidden_states), 'b t (h d) -> b t h d', h=self.num_heads)
        o = rearrange(self.g_norm(o, g), 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        if not output_attentions:
            p = None

        return o, p, past_key_value
