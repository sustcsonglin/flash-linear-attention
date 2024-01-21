# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules.rmsnorm import RMSNorm
from fla.ops.triton.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


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
        expand_v: int = 2,
        expand_k: int = 1,
        num_heads: int = 1,
        gate_fn: str = 'swish',
        layernorm_eps: float = 1e-5,
        gate_logit_normalizer: int = 32,
        gate_logit_multiplier: int = 1,
        gate_low_rank_dim: int = 32,
        mode: str = 'fused_chunk',
        chunk_size: int = 64,
        use_gk: bool = True,  # gate associated with key, i.e., $\alpha$ in the paper
        use_gv: bool = False,  # gate associated with value, i.e., $\beta$ in the paper
        *args, **kwargs
    ) -> GatedLinearAttention:
        super().__init__()
        if use_gv is True:
            assert mode in ['chunk', 'fused_recurrent']
        if mode == 'fused_chunk':
            assert use_gk is True
        if mode != 'chunk' and chunk_size != 16:
            warnings.warn(
                f" `chunk_size` is only used for `chunk` mode."
                f" The `{mode}` mode will suppress the passed value of {chunk_size} and always use 16."
            )
        self.use_gk = use_gk
        self.use_gv = use_gv
        self.d_model = d_model
        self.mode = mode
        self.chunk_size = chunk_size
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

        if self.use_gk:
            self.gk_proj = nn.Sequential(nn.Linear(d_model,  gate_low_rank_dim, bias=False),
                                         nn.Linear(gate_low_rank_dim, self.key_dim, bias=True))
        else:
            self.gk_proj = None
        if self.use_gv:
            self.gv_proj = nn.Sequential(nn.Linear(d_model,  gate_low_rank_dim, bias=False),
                                         nn.Linear(gate_low_rank_dim, self.value_dim,
                                                   bias=True))
        else:
            self.gv_proj = None
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.group_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.gate_logit_normalizer = gate_logit_normalizer
        self.gate_logit_multiplier = gate_logit_multiplier

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -2.5)
        if self.gk_proj is not None:
            nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)
        if self.gv_proj is not None:
            nn.init.xavier_uniform_(self.gv_proj[0].weight, gain=2 ** -2.5)
            nn.init.xavier_uniform_(self.gv_proj[1].weight, gain=2 ** -2.5)

    def forward(self, x):
        mode = self.mode
        chunk_size = self.chunk_size

        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)

        if mode == 'chunk' or mode == 'fused_recurrent':
            # for numumerical stable consideration. fused_chunk has better numerical stability
            if self.use_gk:
                gk = self.gk_proj(x).to(torch.float32)
                gk = (F.logsigmoid(gk) / self.gate_logit_normalizer).clamp_min_(-3)
                gk = rearrange(gk, 'b n (h d) -> b h n d', h=self.num_heads)
            else:
                gk = None
            if self.use_gv:
                gv = self.gv_proj(x).to(torch.float32)
                gv = (F.logsigmoid(gv) / self.gate_logit_normalizer).clamp_min_(-3)
                gv = rearrange(gv, 'b n (h d) -> b h n d', h=self.num_heads)
            else:
                gv = None
            if mode == 'fused_recurrent':
                o = fused_recurrent_gla(q, k, v, gk=gk, gv=gv)
            else:
                o = chunk_gla(q, k, v, gk=gk, gv=gv, chunk_size=chunk_size)
        else:
            g = self.gk_proj(x).to(torch.float32)
            g = F.logsigmoid(g * self.gate_logit_multiplier) / self.gate_logit_normalizer
            g = rearrange(g, 'b n (h d) -> b h n d', h=self.num_heads)
            o = fused_chunk_gla(q, k, v, g)

        o = self.group_norm(rearrange(o, 'b h n d -> b n h d'))
        o = self.out_proj(rearrange(o, 'b n h d -> b n (h d)')
                          * self.gate_fn(self.g_proj(x)))
        return o


if __name__ == '__main__':
    batch = 4
    seq_len = 1023
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(
        torch.bfloat16).cuda().requires_grad_(True)
    model = GatedLinearAttention(use_gk=True, use_gv=True, mode='chunk').to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
