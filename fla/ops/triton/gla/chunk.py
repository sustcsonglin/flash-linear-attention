# -*- coding: utf-8 -*-
# Copyright (c) 2023, Songlin Yang
# Gated Linear Attention Transformers with Hardware-Efficient Training: https://arxiv.org/abs/2312.06635
# chunkwise block parallel. Materialize chunkwise hidden states into HBMs. Therefore it is neccessary to have a large chunk size to reduce such materialization overhead.

import torch
import triton
import triton.language as tl
from einops import rearrange
from fla.ops.triton.utils import contiguous
from fla.ops.triton.gla.block_parallel.inter_chunk_contribution.fn import inter_chunk_onc
from fla.ops.triton.gla.block_parallel.intra_chunk_contribution.fn import intra_chunk_onc


def chunk_gla(q, k, v, gk=None, gv=None, chunk_size=128):
    q = rearrange(q, 'b h (n c) d -> b h n c d',
                  c=chunk_size).contiguous() * (q.shape[-1])**-0.5
    k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    v = rearrange(v, 'b h (n c) d -> b h n c d', c=chunk_size).contiguous()
    if gk is not None:
        gk = rearrange(gk, 'b h (n c) d -> b h n c d',
                       c=chunk_size).contiguous()
    if gv is not None:
        gv = rearrange(gv, 'b h (n c) d -> b h n c d',
                       c=chunk_size).contiguous()
    gk, gv, o1 = inter_chunk_onc(q, k, v, gk, gv)
    o2 = intra_chunk_onc(q, k, v, gk, gv)
    return rearrange(o1+o2, 'b h n c d -> b h (n c) d')
