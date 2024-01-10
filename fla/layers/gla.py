# -*- coding: utf-8 -*-

# "Gated Linear Attention Transformers with Hardware-Efficient Training"[https://arxiv.org/abs/2312.06635]

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from fla.modules.rmsnorm import RMSNorm
from fla.ops.triton.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class GatedLinearAttention(nn.Module):
    def __init__(self,
                 d_model=1024,
                 expansion_ratio=0.5,
                 num_heads=4,
                 gate_fn="swish",
                 layernorm_eps=1e-5,
                 gate_logit_normalizer=16,
                 gate_low_rank_dim=16,
                 *args,
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.value_dim = d_model
        self.key_dim = int(d_model * expansion_ratio)
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
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.group_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.gate_logit_normalizer = gate_logit_normalizer
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)
        nn.init.xavier_uniform_(self.gk_proj[0].weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.gk_proj[1].weight, gain=2 ** -2.5)

    def forward(self, x, fwd_mode='chunk'):
        assert fwd_mode in ['fused_chunk', 'chunk', 'fused_recurrent']
        assert x.shape[-1] % 16 == 0, "only support dimension divisible by 16 for now"
        assert x.shape[-2] % 16 == 0, "only support input length divisible by 16 for now"
        q = rearrange(self.q_proj(
            x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(
            x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d',
                      h=self.num_heads)
        g = self.gk_proj(x)
        g = F.logsigmoid(g) / self.gate_logit_normalizer
        g = rearrange(g, 'b n (h d) -> b h n d',
                      h=self.num_heads)
        if fwd_mode == 'fused_chunk':
            o = fused_chunk_gla(q, k, v, g)
        elif fwd_mode == 'chunk':
            # for numumerical stable consideration
            g = torch.clamp(g, min=-3)
            o = chunk_gla(q, k, v, g)
        elif fwd_mode == 'fused_recurrent':
            o = fused_recurrent_gla(q, k, v, g)
        else:
            raise NotImplementedError
        o = self.group_norm(rearrange(o, 'b h n d -> b n h d'))
        return self.out_proj(rearrange(o, 'b n h d -> b n (h d)') * self.gate_fn(self.g_proj(x)))


if __name__ == '__main__':
    batch = 4
    seq_len = 1024
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(
        torch.bfloat16).cuda().requires_grad_(True)
    model = GatedLinearAttention().to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
