# -*- coding: utf-8 -*-

"""
Linear attention in Based.
https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
"""
import math

import opt_einsum as oe
import torch
import torch.nn as nn
from einops import rearrange

from fla.ops.based import fused_chunk_based, parallel_based


def init_feature_map(feature_map: str = 'none', **kwargs: any):
    """
    Initialize query and key mapping for linear attention
    """
    if feature_map in [None, 'none', 'identity']:
        return FeatureMap(**kwargs)
    # Taylor series approximations to exp(x)
    elif feature_map == 'taylor_exp':
        return TaylorExp(**kwargs)
    else:
        raise NotImplementedError(
            f'Sorry "{feature_map}" feature map not implemented.')


class FeatureMap(nn.Module):
    """
    Parent feature map; default is identity function
    """

    def __init__(self,
                 input_dim: int,
                 temp: int = None,
                 head_dim_idx: int = -1,
                 eps: float = 1e-12,
                 **kwargs: any):
        super().__init__()
        self.input_dim = input_dim
        self.head_dim_idx = head_dim_idx
        self.temp = 1. if temp is None else temp
        self.eps = eps

    def forward(self, x: torch.Tensor):
        """
        Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        return x


class TaylorExp(FeatureMap):
    """
    Feature map to compute 2nd-order Taylor approx. of exp(q^T k / sqrt(d))
    """

    def __init__(self, input_dim: int, **kwargs: any):
        super().__init__(input_dim, **kwargs)
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.input_dim)
        self.rrd = math.sqrt(self.rd)
        self.tril_indices = torch.tril_indices(
            self.input_dim, self.input_dim, -1)

    # Running these in parallel
    def forward(self, x: torch.Tensor):
        # Get 2nd-order terms (rearrange(x * x), '... m n -> ... (m n)')
        x2 = (x.unsqueeze(-1) * x.unsqueeze(-2)
              ).flatten(start_dim=-2) / self.r2
        return torch.cat([torch.ones(x[..., :1].shape).to(x.device),
                          x / self.rrd, x2 / self.rd], dim=self.head_dim_idx)

    def forward_mem_save(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x) s.t. f(x)^T f(x') = 1 + x^Tx' + (x^Tx')^2 / 2
        -> Assume x.shape is (batch_size, n_heads, seq_len, head_dim)
        """
        # Slow but memory-saving way to compute 2nd-order terms; how do w/o outer-product first?
        x2 = oe.contract('...m,...n->...mn', x, x) / self.rd
        x2d = torch.diagonal(x2, dim1=-2, dim2=-1) / self.r2
        x2 = x2[..., self.tril_indices[0], self.tril_indices[1]]
        x = torch.cat([torch.ones(x[..., :1].shape).to(x.device),
                       x / self.rrd, x2d, x2], dim=-1)
        return x


class BasedLinearAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        l_max: int = 2048,
        feature_dim: int = 16,
        num_key_value_heads: int = 12,
        num_heads: int = 12,
        feature_name: str = "taylor_exp",
        eps: float = 1e-12,
        causal: bool = True,
        mode: str = "parallel",
    ):
        super().__init__()
        self.d_model = d_model
        self.l_max = l_max
        self.mode = mode
        assert self.mode in ["fused_chunk", "parallel"]

        # linear attention
        self.feature_name = feature_name
        self.feature_dim = feature_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_key_value_heads
        self.causal = causal
        feature_map_kwargs = {
            'input_dim': self.feature_dim,
            'head_dim_idx': -1,
            'temp': 1.,
            'eps': 1e-12
        }
        self.feature_map = init_feature_map(
            feature_map=self.feature_name, **feature_map_kwargs)
        self.proj_q = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_k = nn.Linear(
            self.d_model, self.feature_dim * self.num_heads, bias=False)
        self.proj_v = nn.Linear(
            self.d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.proj_o = nn.Linear(
            self.num_heads * self.head_dim, self.d_model, bias=False)
        self.dropout = nn.Identity()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        mode = self.mode
        b, l, _ = hidden_states.size()
        q, k, v = self.proj_q(hidden_states), self.proj_k(
            hidden_states), self.proj_v(hidden_states)
        q, k, v = map(lambda x: rearrange(
            x, "b l (h d) -> b h l d", h=self.num_heads), [q, k, v])
        if mode == "fused_chunk":
            assert q.shape[-1] <= 16
            o = fused_chunk_based(q, k, v, True, True)
        elif mode == 'parallel':
            assert q.shape[-1] <= 128
            o = parallel_based(q, k, v, True, True)
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
        q, k, v = self.proj_q(hidden_states), self.proj_k(
            hidden_states), self.proj_v(hidden_states)

        q = q.view(b, l, self.num_heads, self.feature_dim).transpose(1, 2)
        k = k.view(b, l, self.num_key_value_heads,
                   self.feature_dim).transpose(1, 2)
        v = v.view(b, l, self.num_key_value_heads,
                   self.head_dim).transpose(1, 2)

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
    x = torch.randn(batch, seq_len, d_model).to(
        dtype).cuda().requires_grad_(True)
    dy = torch.randn(batch, seq_len, d_model).to(
        dtype).cuda()
    model = BasedLinearAttention(d_model=d_model).to(dtype).cuda()
    y = model(x)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = model.forward_reference(x)
    y2.backward(dy)
    assert y.allclose(y2, 0, 1e-4), breakpoint()
    assert x_grad.allclose(x.grad, 0, 1e-4), breakpoint()
