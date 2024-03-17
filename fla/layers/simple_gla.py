# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN

from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.ops.simple_gla import chunk_simple_gla


class SimpleGatedLinearAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        mode: str = 'fused_chunk',
        fuse_norm: bool = True,
        *args, **kwargs
    ) -> SimpleGatedLinearAttention:
        super().__init__()
        self.hidden_size = hidden_size

        self.mode = mode
        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        assert mode in ['chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = ACT2FN[gate_fn]

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.gk_proj = nn.Linear(hidden_size, self.num_heads)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        if gate_fn == 'swish' and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(self.head_v_dim, eps=layernorm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)

        self.gate_logit_normalizer = gate_logit_normalizer

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(self, x):
        mode = self.mode
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = rearrange(self.gk_proj(x), 'b n h -> b h n')
        gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)

        if mode == 'chunk':
            o = chunk_simple_gla(q, k, v, gk)
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, 'b h l d -> b l h d')
        g = self.g_proj(x)

        if self.fuse_norm_and_gate:
            g = rearrange(g, 'b l (h d) -> b l h d', h=self.num_heads)
            o = self.g_norm_swish_gate(o, g)
            o = rearrange(o, 'b l h d -> b l (h d)')
        else:
            o = self.g_norm(o)
            o = rearrange(o, 'b l h d -> b l (h d)')
            o = o * self.gate_fn(g)
        o = self.o_proj(o)
        return o


if __name__ == '__main__':
    batch = 4
    seq_len = 1024

    hidden_size = 2048
    x = torch.randn(batch, seq_len, hidden_size).to(torch.bfloat16).cuda().requires_grad_(True)
    model = SimpleGatedLinearAttention(hidden_size=hidden_size, mode='chunk').to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
