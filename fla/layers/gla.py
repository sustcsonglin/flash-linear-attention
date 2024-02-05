# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


def get_activation_fn(activation):
    if activation == 'swish':
        return F.silu
    elif activation == 'gelu':
        return F.gelu
    else:
        raise NotImplementedError


class GatedLinearAttention(nn.Module):

    def __init__(
        self,
        d_model: int = 1024,
        expand_v: float = 2.0,
        expand_k: float = 1.0,
        num_heads: int = 4,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        mode: str = 'fused_chunk',
        fuse_norm: bool = True,
        *args, **kwargs
    ) -> GatedLinearAttention:
        super().__init__()
        self.d_model = d_model

        self.mode = mode
        self.value_dim = int(d_model * expand_v)
        self.key_dim = int(d_model * expand_k)
        assert mode in ['chunk', 'fused_recurrent',
                        'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"
        self.num_heads = num_heads
        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.g_proj = nn.Linear(d_model, self.value_dim, bias=False)

        self.gk_proj = nn.Sequential(nn.Linear(d_model,  gate_low_rank_dim, bias=False),
                                     nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        if (gate_fn == 'swish') and fuse_norm:
            self.g_norm_swish_gate = FusedRMSNormSwishGate(
                self.head_v_dim, eps=layernorm_eps)
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)

        self.gate_logit_normalizer = gate_logit_normalizer

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)

    def forward(self, x):
        mode = self.mode

        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = rearrange(self.gk_proj(
            x), 'b n (h d) -> b h n d', h=self.num_heads)
        gk = (F.logsigmoid(gk) / self.gate_logit_normalizer)

        if mode == 'fused_recurrent':
            o = fused_recurrent_gla(q, k, v, gk, None)
        elif mode == 'fused_chunk':
            o = fused_chunk_gla(q, k, v, gk)
        elif mode == 'chunk':
            o = chunk_gla(q, k, v, gk)
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

    d_model = 2048
    # x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda().requires_grad_(True)
    # model = GatedLinearAttention(use_gk=True, use_gv=True, mode='fused_chunk').to(torch.bfloat16).cuda()
    # y = model(x)
    # print(y.shape)
    # y.sum().backward()
    # print(x.grad.shape)

    for act in ['swish']:
        org = GatedLinearAttention(
            d_model=d_model, gate_fn=act, fuse_norm=False).to(torch.bfloat16).cuda()
        fused = GatedLinearAttention(
            d_model=d_model, gate_fn=act, fuse_norm=True).to(torch.bfloat16).cuda()
        fused.q_proj.weight.data.copy_(org.q_proj.weight.data)
        fused.k_proj.weight.data.copy_(org.k_proj.weight.data)
        fused.v_proj.weight.data.copy_(org.v_proj.weight.data)
        fused.g_proj.weight.data.copy_(org.g_proj.weight.data)
        fused.o_proj.weight.data.copy_(org.o_proj.weight.data)
        fused.gk_proj[0].weight.data.copy_(org.gk_proj[0].weight.data)
        fused.gk_proj[1].weight.data.copy_(org.gk_proj[1].weight.data)
        fused.gk_proj[1].bias.data.copy_(org.gk_proj[1].bias.data)

        x = torch.randn(batch, seq_len, d_model).to(torch.bfloat16).cuda()
        org_x = x.clone().requires_grad_(True)
        fused_x = x.clone().requires_grad_(True)
        org_o = org(org_x)
        fused_o = fused(fused_x)
        org_o.sum().backward()
        fused_o.sum().backward()
        breakpoint()
        assert org_o.allclose(fused_o, 0, 1e-2), "output not equal"
        assert org_x.grad.allclose(fused_x.grad, 0, 1e-2), "grad not equal"
