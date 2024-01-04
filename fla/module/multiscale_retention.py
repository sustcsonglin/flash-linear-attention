# -*- coding: utf-8 -*-

# Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]

from fla.ops.triton.retention import fused_chunk_retention, parallel_retention, fused_recurrent_retention
from fla.module.rmsnorm import RMSNorm
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from fla.module.rotary import RotaryEmbedding


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):
    def __init__(self,
                 embed_dim=1024,
                 expansion_ratio=2,
                 num_heads=4,
                 gate_fn="swish",
                 layernorm_eps=1e-5,
                 *args, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = embed_dim * expansion_ratio
        self.num_heads = num_heads
        self.head_qk_dim = self.embed_dim // num_heads
        self.head_v_dim = self.head_qk_dim * expansion_ratio
        self.scaling = self.head_qk_dim ** -0.5
        self.gate_fn = get_activation_fn(activation=str(gate_fn))
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, self.value_dim, bias=False)
        self.out_proj = nn.Linear(self.value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_v_dim, eps=layernorm_eps)
        self.rotary = RotaryEmbedding(dim=self.head_qk_dim, interleaved=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2 ** -1)

    def forward(self, x, form='fused_chunk'):
        assert form in ['fused_chunk', 'parallel', 'chunk', 'fused_recurrent']
        q1 = rearrange(self.q_proj(
            x), '... (h d) -> ... h d', h=self.num_heads)
        k1 = rearrange(self.k_proj(
            x), '... (h d) -> ... h d', h=self.num_heads)
        q, k = self.rotary(q1, k1)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d',
                      h=self.num_heads).contiguous()
        if form == 'fused_chunk':
            o = fused_chunk_retention(q, k, v)
        elif form == 'parallel':
            o = parallel_retention(q, k, v)
        elif form == 'fused_recurrent':
            o = fused_recurrent_retention(q, k, v)
        # TODO: need fix to allow different d_head_qk and d_head_v for "chunk" form
        else:
            raise NotImplementedError
        o = self.group_norm(rearrange(o, 'b h n d -> b n h d'))
        return self.out_proj(rearrange(o, 'b n h d -> b n (h d)') * self.g_proj(x))


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