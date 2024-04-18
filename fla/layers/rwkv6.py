# -*- coding: utf-8 -*-

# "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence"[https://arxiv.org/abs/2404.05892]

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache

from fla.modules import FusedLayerNormSwishGate, LayerNorm
from fla.ops.rwkv6 import chunk_rwkv6, fused_recurrent_rwkv6
from fla.utils import checkpoint


class RWKV6Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        proj_low_rank_dim: int = 32,
        gate_low_rank_dim: int = 64,
        fuse_norm: bool = True,
        layer_idx: int = None,
        eps: float = 1e-5,
        **kwargs
    ) -> RWKV6Attention:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.proj_low_rank_dim = proj_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.layer_idx = layer_idx

        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.eps = eps

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_proj = nn.Sequential(
            LerpLinear(hidden_size, proj_low_rank_dim * 5, bias=False),
            nn.Tanh(),
            nn.Linear(proj_low_rank_dim * 5, hidden_size)
        )
        self.r_proj = LerpLinear(hidden_size, self.key_dim, bias=False)
        self.w_proj = LerpLinear(hidden_size, self.key_dim, low_rank_dim=gate_low_rank_dim, bias=False)
        self.k_proj = LerpLinear(hidden_size, self.key_dim, bias=False)
        self.v_proj = LerpLinear(hidden_size, self.value_dim, bias=False)
        self.g_proj = LerpLinear(hidden_size, self.value_dim, bias=False)
        self.bonus = nn.Parameter(torch.randn(num_heads, self.head_qk_dim))

        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm:
            self.g_norm_swish_gate = FusedLayerNormSwishGate(self.head_v_dim, eps=eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = LayerNorm(self.head_v_dim, eps=eps)
            self.gate_fn = ACT2FN[gate_fn]

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
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
        batch_size, seq_len, hidden_size = hidden_states.size()
        # launching the triton kernel for just one token will actually be slower
        mode = 'fused_recurrent' if hidden_states.shape[1] == 1 else self.mode

        delta = self.time_shift(hidden_states) - hidden_states
        x = self.x_proj[0](hidden_states, delta).view(batch_size, seq_len, -1, self.proj_low_rank_dim)
        r, w, k, v, g = torch.einsum('b l n r, n r d-> b l n d',
                                     self.x_proj[1](x),
                                     self.x_proj[2].weight.view(5, -1, hidden_size)).unbind(-2)
        r = self.r_proj(r, delta)
        w = self.w_proj(w, delta)
        k = self.k_proj(k, delta)
        v = self.v_proj(v, delta)
        g = self.g_proj(g, delta)

        # dealing with left-padding
        if attention_mask is not None:
            v = v.mul_(attention_mask.unsqueeze(-1))
        r, w, k, v = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.num_heads), (r, w, k, v))
        w = -torch.exp(w.float()).type_as(w) + self.eps
        u = self.bonus

        last_state = past_key_values[self.layer_idx] if use_cache else None
        state = last_state[-1] if use_cache else None
        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_rwkv6(r, k, v, w, u, initial_state=state, output_final_state=use_cache)
        elif mode == 'chunk':
            o, recurrent_state = chunk_rwkv6(r, k, v, w, u, initial_state=state, output_final_state=use_cache)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update((recurrent_state,), self.layer_idx, r.shape[2])

        o = rearrange(o, 'b h l d -> b l h d')
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

    def init_state(self, batch_size: int) -> Tuple[torch.Tensor]:
        param = next(self.parameters())
        state = (param.new_zeros(batch_size, self.num_heads, self.head_qk_dim, self.head_v_dim),)
        return state

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim * self.head_v_dim
        return state_size


class LerpLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: Optional[int] = None,
        bias: Optional[bool] = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        if low_rank_dim is None:
            self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            self.linear = LoRA(input_dim, output_dim, low_rank_dim, bias=bias)
        self.mu = nn.Parameter(torch.zeros(input_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}"
        if self.low_rank_dim is not None:
            s += f", low_rank_dim={self.low_rank_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return self.linear(x + delta * self.mu)


class LoRA(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.lora = nn.Sequential(
            nn.Linear(input_dim, low_rank_dim, bias=bias),
            nn.Tanh(),
            nn.Linear(low_rank_dim, output_dim)
        )
        self.lamda = nn.Parameter(torch.zeros(output_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}, low_rank_dim={self.low_rank_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    @checkpoint
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x) + self.lamda


class DDLerp(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        low_rank_dim: int,
        bias: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.low_rank_dim = low_rank_dim
        self.bias = bias

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.lora = LoRA(input_dim, output_dim, low_rank_dim, bias=bias)
        self.mu = nn.Parameter(torch.zeros(1, 1, input_dim))

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.input_dim}, {self.output_dim}, low_rank_dim={self.low_rank_dim}"
        if not self.bias:
            s += f", bias={self.bias}"
        s += ")"
        return s

    @checkpoint
    def forward(self, x: torch.Tensor, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        if delta is None:
            shifted = self.time_shift(x)
            if len(shifted.shape) == 2:
                shifted = shifted.unsqueeze(1)
            delta = shifted - x
        return x + delta * self.lora(x + delta * self.mu)
