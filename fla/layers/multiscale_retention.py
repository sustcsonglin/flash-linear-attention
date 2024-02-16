# -*- coding: utf-8 -*-

# Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.activations import ACT2FN

from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules.rotary import RotaryEmbedding
from fla.ops.retention import (chunk_retention, fused_chunk_retention,
                               fused_recurrent_retention, parallel_retention)


def get_activation_fn(activation):
    if activation == 'swish':
        return F.silu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise NotImplementedError


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
        *args,
        **kwargs
    ) -> MultiScaleRetention:
        super().__init__()

        self.d_model = d_model
        self.mode = mode
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.num_heads = num_heads

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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)

    def forward(self, x):
        mode = self.mode
        q1 = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        k1 = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        q, k = self.rotary(q1, k1)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        if mode == 'chunk':
            o = chunk_retention(q, k, v)
        elif mode == 'fused_chunk':
            o = fused_chunk_retention(q, k, v)
        elif mode == 'parallel':
            o = parallel_retention(q, k, v)
        elif mode == 'fused_recurrent':
            o = fused_recurrent_retention(q, k, v)
        else:
            raise NotImplementedError

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
    import torch
    batch = 4
    seq_len = 1024
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(
        torch.bfloat16).cuda().requires_grad_(True)
    model = MultiScaleRetention().to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
    print(x.grad.shape)
