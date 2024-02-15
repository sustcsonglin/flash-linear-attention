# -*- coding: utf-8 -*-

# Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]

from __future__ import annotations

from locale import normalize

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fla.modules import FusedRMSNormSwishGate, RMSNorm
from fla.modules.featue_map import HedgehogFeatureMap, T2RFeatureMap
from fla.modules.rotary import RotaryEmbedding
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn
from torch import normal


class LinearAttention(nn.Module):
    def __init__(
        self,
        d_model: str = 1024,
        expand_k: str = 1.0,
        expand_v: str = 1.0,
        num_heads: str = 8,
        mode: str = 'chunk',
        feature_map: str = 'hedgedog',
        tie_feature_map_qk: bool = False,
        normalize_output: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()

        assert feature_map in ['elu', 'relu', 'hedgehog', 't2r', 'identity'], f"Not supported feature map `{feature_map}`."

        self.d_model = d_model
        self.mode = mode
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.num_heads = num_heads

        assert mode in ['chunk', 'fused_chunk'], f"Not suppoerted mode `{mode}`."
        assert self.key_dim % num_heads == 0, f"key dim must be divisible by num_heads of {num_heads}"
        assert self.value_dim % num_heads == 0, f"value dim must be divisible by num_heads of {num_heads}"

        self.head_qk_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        if feature_map == 'hedgehog':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = HedgehogFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = HedgehogFeatureMap(head_dim=self.head_qk_dim)
        
        elif feature_map == 't2r':
            if tie_feature_map_qk:
                self.feature_map_q = self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)
            else:
                self.feature_map_q = T2RFeatureMap(head_dim=self.head_qk_dim)
                self.feature_map_k = T2RFeatureMap(head_dim=self.head_qk_dim)

        elif feature_map == 'elu':
            def elu(x):
                return F.elu(x) + 1
            self.feature_map_q = elu()
            self.feature_map_k = elu()
        
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
                        
        elif feature_map == 'identity':
            self.feature_map_q = nn.Identity()
            self.feature_map_k = nn.Identity()
        
        else:
            raise NotImplementedError
        
        self.normalize_output = normalize_output
        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, d_model, bias=False)

        
    def forward(self, x):
        mode = self.mode
        q = rearrange(self.q_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.k_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.v_proj(x), 'b n (h d) -> b h n d', h=self.num_heads)
        q = self.feature_map_q(q)
        k = self.feature_map_k(k)
        if mode == 'chunk':
            o = chunk_linear_attn(q, k, v, normalize=self.normalize_output)
        elif mode == 'fused_chunk':
            o = fused_chunk_linear_attn(q, k, v, normalize=self.normalize_output)
        else:
            raise NotImplementedError
        o = rearrange(o, 'b h n d -> b n (h d)')
        o = self.o_proj(o)
        return o


if __name__ == '__main__':
    import torch
    batch = 4
    seq_len = 1024
    d_model = 1024
    x = torch.randn(batch, seq_len, d_model).to(
        torch.bfloat16).cuda().requires_grad_(True)
    model = LinearAttention(d_model=d_model, feature_map='hedgedog').to(torch.bfloat16).cuda()
    y = model(x)
    print(y.shape)
    y.sum().backward()
    print(x.grad.shape)
    print(x.grad.shape)

