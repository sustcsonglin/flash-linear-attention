# -*- coding: utf-8 -*-

# Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules.rmsnorm import RMSNorm
from fla.modules.rotary import RotaryEmbedding
from fla.ops.triton.retention import (fused_chunk_retention,
                                      fused_recurrent_retention,
                                      parallel_retention)


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
        *args, **kwargs
    ) -> MultiScaleRetention:
        super().__init__()

        self.d_model = d_model
        self.mode = mode
        self.value_dim = int(d_model * expand_v)
        self.key_dim = int(d_model * expand_k)
        self.num_heads = num_heads
        assert mode in ['fused_chunk', 'chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = get_activation_fn(activation=str(gate_fn))
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)

        self.group_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.rotary = RotaryEmbedding(dim=self.head_qk_dim, interleaved=False)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -2.5)

    def forward(self, x):
        mode = self.mode
        q1 = rearrange(self.q_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        k1 = rearrange(self.k_proj(x), '... (h d) -> ... h d', h=self.num_heads)
        q, k = self.rotary(q1, k1)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        if mode == 'fused_chunk':
            o = fused_chunk_retention(q, k, v)
        elif mode == 'parallel':
            o = parallel_retention(q, k, v)
        elif mode == 'fused_recurrent':
            o = fused_recurrent_retention(q, k, v)
        # TODO: need fix to allow different d_head_qk and d_head_v for "chunk" form
        else:
            raise NotImplementedError
        o = self.group_norm(rearrange(o, 'b h n d -> b n h d'))
        return self.out_proj(rearrange(o, 'b n h d -> b n (h d)') * self.gate_fn(self.g_proj(x)))


if __name__ == '__main__':
    import torch
    batch = 4
    seq_len = 1024
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda().requires_grad_(True)
    model = MultiScaleRetention().to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
    print(x.grad.shape)
