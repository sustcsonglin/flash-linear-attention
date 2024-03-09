# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as ckp

from fla.modules.rmsnorm import layer_norm_fn


def checkpoint(func):
    def wrapper(*args, **kwargs):
        return ckp(func, *args, **kwargs)
    return wrapper

# https://arxiv.org/abs/2402.04347


class HedgehogFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        nn.init.eye_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x)  # shape b, h, l, d
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)


# https://arxiv.org/abs/2103.13076
class T2RFeatureMap(nn.Module):
    def __init__(self, head_dim: int, dot_dim: int = None):
        super().__init__()
        # Trainable map
        if dot_dim is None:
            dot_dim = head_dim
        self.layer = nn.Linear(head_dim, dot_dim)

    def forward(self, x: torch.Tensor):
        return self.layer(x).relu()

# https://arxiv.org/abs/2102.11174


class DPFPFeatureMap(nn.Module):
    def __init__(self, head_dim: int, nu: int = 4):
        super().__init__()
        self.nu = nu

    def forward(self, x: torch.Tensor):
        x = torch.cat([x.relu(), -x.relu()], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


class HadamardFeatureMap(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, head_dim)
        self.layer2 = nn.Linear(head_dim, head_dim)

    def forward(self, x: torch.Tensor):
        return self.layer1(x) * self.layer2(x)


@checkpoint
def flatten_diag_outer_product(x, y):
    z = x.unsqueeze(-1) * y.unsqueeze(-2)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    return z[..., indicies[0], indicies[1]]


@checkpoint
def flatten_diag_outer_product_off1(x, y):
    z = x.unsqueeze(-1) * y.unsqueeze(-2)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]


class LearnableOuterProductFeatureMap(nn.Module):
    def __init__(self, head_dim: int, feature_dim: int):
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, feature_dim, bias=False)
        self.layer2 = nn.Linear(head_dim, feature_dim, bias=False)
        self.normalizer = feature_dim ** -0.5

    def forward(self, x: torch.Tensor):
        return flatten_diag_outer_product(self.layer1(x), self.layer2(x))


class TaylorFeatureMap(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)


class RebasedFeatureMap(TaylorFeatureMap):

    def __init__(
        self,
        head_dim: int,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True
    ) -> RebasedFeatureMap:
        super().__init__(head_dim)

        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize

        self.gamma = None
        self.beta = None
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(head_dim))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(head_dim))

    def forward(self, x: torch.Tensor):
        if self.use_beta and self.use_gamma and self.normalize:
            x = layer_norm_fn(x, self.gamma, self.beta)
        elif self.normalize:
            x = F.layer_norm(x, (self.head_dim,), self.gamma, self.beta)
        elif self.use_gamma and self.use_beta:
            x = torch.addcmul(self.beta, x, self.gamma)
        elif self.use_gamma:
            x = x.mul(self.gamma)
        else:
            raise RuntimeError(f"Not supported combination of `use_gamma`, `use_beta` and `normalize`, "
                               f"which is currentlt set as (`{self.use_gamma}`, `{self.use_beta}`, `{self.normalize}`)")

        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)
