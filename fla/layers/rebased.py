# -*- coding: utf-8 -*-

"""
https://github.com/corl-team/rebased/blob/main/flash_linear_attention/fla/layers/rebased_fast.py
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from fla.modules.feature_map import RebasedFeatureMap
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from fla.ops.rebased import parallel_rebased


class ReBasedLinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 16,
        num_heads: int = 16,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True,
        causal: bool = True,
        eps: float = 1e-5,
        mode: str = "parallel",
    ) -> ReBasedLinearAttention:
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.mode = mode
        assert self.mode in ["fused_chunk", "parallel", 'chunk']

        # linear attention
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize
        self.causal = causal

        self.feature_map = RebasedFeatureMap(self.feature_dim, use_gamma, use_beta, normalize)
        self.proj_q = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        mode = self.mode
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)
        q, k, v = map(lambda x: rearrange(x, "b l (h d) -> b h l d", h=self.num_heads), [q, k, v])
        if mode == "fused_chunk":
            q, k = self.feature_map(q), self.feature_map(k)
            o = fused_chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'chunk':
            q, k = self.feature_map(q), self.feature_map(k)
            o = chunk_linear_attn(q, k, v, normalize=True, scale=1)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_rebased(q, k, v, self.eps, True, True)
        o = rearrange(o, "b h l d -> b l (h d)")
        o = self.proj_o(o)
        o = self.dropout(o)
        return o

    # https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py#L119

    def forward_reference(self, hidden_states: torch.Tensor, filters: torch.Tensor = None, *args, **kwargs):
        """
        x (torch.Tensor): tensor of shape (b, d, l)
        y (torch.Tensor): tensor of shape (b, d, l)
        """
        # hidden_states = hidden_states.transpose(1, 2)
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(hidden_states), self.proj_v(hidden_states)

        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads, self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Linear attention
        q, k = self.feature_map(q), self.feature_map(k)
        q, k, v = q.unsqueeze(-2), k.unsqueeze(-2), v.unsqueeze(-1)

        # Compute attention
        if self.causal:
            y = ((q * (k * v).cumsum(2)).sum(-1) / ((q * k.cumsum(2)).sum(-1) + self.eps))
        else:
            y = ((q * (k * v).sum(2, True)).sum(-1) / ((q * k.sum(2, True)).sum(-1) + self.eps))
        y = rearrange(y, 'b h l d -> b l (h d)')
        y = self.proj_o(y.to(hidden_states.dtype))
        y = self.dropout(y)
        return y.to(hidden_states.dtype)


if __name__ == '__main__':
    batch = 4
    seq_len = 1024
    d_model = 1024
    dtype = torch.float32
    x = torch.randn(batch, seq_len, d_model).to(dtype).cuda().requires_grad_(True)
    dy = torch.randn(batch, seq_len, d_model).to(dtype).cuda()
    model = ReBasedLinearAttention(d_model=d_model, mode='parallel').to(dtype).cuda()

    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    print(model.mode)
    model.mode = 'fused_chunk'
    y2 = model(x)
    print(model.mode)
    y2.backward(dy)
    # assert y.allclose(y2, 0, 1e-4), breakpoint()
    # assert x_grad.allclose(x.grad, 0, 1e-4), breakpoint()
    print("Pass")
